"""Checkpoint-laundering benchmark on the MLP bank (Tracks A + B + D).

Rebuilds the exact 52-pair Table-6 bank (2 refs, 30 related descendants, 22
unrelated), then constructs function-preserving LAUNDERED descendants under five
variants (P, D-mild, D-strong, PD, PDFT) and re-scores every method. The point:
raw weight proximity is decoupled from lineage, so we can see which methods
survive.

Methods scored per (reference, suspect) pair:
    ours (diagonal_dominance)        -- on branch product M = W_out @ W_in
    raw_aligned_frobenius            -- baselines on RAW per-block weights
    raw_singular_value_dist             (this is what an attacker-facing baseline
    raw_weight_cosine                    actually has; M-based baselines borrow M's
                                         invariance, which is OUR contribution)
    rebasin_frobenius                -- Track B: Git-Re-Basin unit alignment
    rebasin_scale_frobenius          -- Track B: alignment + per-unit LS scale
    cka, svcca                       -- activation-space (block inputs)
    ipguard_regr                     -- prediction-space
    aligned_frobenius_M, ...         -- baselines on M (completeness: trivially
    singular_value_dist_M, ...          invariant, shown to make the point)
    weight_cosine_M

Because P/D/PD are function-preserving AND leave M, activations and predictions
exactly unchanged, ours / cka / svcca / ipguard / *_M are identical to their
'none' (unlaundered) values on those variants -- only the RAW baselines and PDFT
move. Every measured number is reported, including cells that contradict the
predictions.

Tracks:
    A  -- the variants x methods AUROC table (this file's core)
    B  -- rebasin_* methods, scored on every variant alongside A
    D  -- controls: (1) launder 4 unrelated with PD, confirm still unrelated;
          (2) ours-invariance dScore/signature-cos on P/D/PD; (3) 'none' variant
          reproduces the Table-6 cells (harness-unchanged check).

CPU, deterministic, seeded. Scoring fans out over a ProcessPoolExecutor operating
on pure-numpy bundles (no torch in workers).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import lineage_baselines as lbase  # noqa: E402
import lineage_detection as ldet  # noqa: E402
import laundering_baselines_raw as lraw  # noqa: E402
import laundering_ops as lops  # noqa: E402
from lineage_benchmark_mlp import collect_activations, predictions  # noqa: E402
from lineage_phase1_mlp import (  # noqa: E402
    branch_products,
    descendant_fine_tune,
    descendant_finetune_new_target,
    descendant_noise,
    descendant_prune,
    descendant_quantize,
    eval_loss,
    fresh_model,
    make_data,
    nondesc_distilled_student,
    train_model,
)

TAU_S = 0.5  # matches lineage_benchmark_mlp.py (Table 6)

# Method registry. Every scorer takes (ref_bundle, sus_bundle) numpy dicts.
PRIMARY_METHODS = [
    "diagonal_dominance",       # ours (on M)
    "raw_aligned_frobenius",    # baselines on raw weights
    "raw_singular_value_dist",
    "raw_weight_cosine",
    "cka",
    "svcca",
    "ipguard_regr",
]
TRACK_B_METHODS = ["rebasin_frobenius", "rebasin_scale_frobenius"]
COMPLETENESS_METHODS = ["aligned_frobenius_M", "singular_value_dist_M", "weight_cosine_M"]
ALL_METHODS = PRIMARY_METHODS + TRACK_B_METHODS + COMPLETENESS_METHODS


# --------------------------------------------------------------- numpy bundles

def bundle(model, Xv) -> dict:
    """Everything the scorers need, as pure numpy (torch model can be dropped)."""
    return {
        "Ms": [np.asarray(M, dtype=np.float64) for M in branch_products(model)],
        "raw": lops.raw_weights(model),
        "acts": collect_activations(model, Xv),
        "preds": predictions(model, Xv),
    }


def score_all(ref: dict, sus: dict, tau_s: float = TAU_S) -> dict:
    """All methods for one (reference, suspect) pair. Pure numpy -> pool-safe."""
    out = {}
    out["diagonal_dominance"], _, _ = ldet.lineage_score(ref["Ms"], sus["Ms"], tau_s)
    out["raw_aligned_frobenius"] = lraw.raw_aligned_frobenius(ref["raw"], sus["raw"])
    out["raw_singular_value_dist"] = lraw.raw_singular_value_dist(ref["raw"], sus["raw"])
    out["raw_weight_cosine"] = lraw.raw_weight_cosine(ref["raw"], sus["raw"])
    out["rebasin_frobenius"] = lraw.rebasin_frobenius(ref["raw"], sus["raw"])
    out["rebasin_scale_frobenius"] = lraw.rebasin_scale_frobenius(ref["raw"], sus["raw"])
    out["cka"] = lbase.cka_lineage_score(ref["acts"], sus["acts"])
    out["svcca"] = lbase.svcca_lineage_score(ref["acts"], sus["acts"])
    out["ipguard_regr"] = lbase.ipguard_match_rate(ref["preds"], sus["preds"])
    out["aligned_frobenius_M"] = lbase.aligned_frobenius(ref["Ms"], sus["Ms"])
    out["singular_value_dist_M"] = lbase.singular_value_distance(ref["Ms"], sus["Ms"])
    out["weight_cosine_M"] = lbase.weight_cosine(ref["Ms"], sus["Ms"])
    return out


# ------------------------------------------------------- worker pool machinery

_REFS: dict = {}


def _init_worker(refs: dict):
    global _REFS
    _REFS = refs


def _score_job(job: tuple):
    variant, pair_id, label, ref_idx, kind, sus = job
    scores = score_all(_REFS[ref_idx], sus)
    return {"variant": variant, "pair_id": pair_id, "label": label,
            "ref": ref_idx, "kind": kind, "scores": scores}


# ---------------------------------------------------------------- build bank

def build_bank(args):
    """Rebuild the exact Table-6 bank. Returns (refs, descendants, unrelated).

    refs[ref_idx]        = {"bundle", "X", "y", "Xv"}   (numpy bundle + FT data)
    descendants[i]       = {"ref", "kind", "model"}     (torch, to be laundered)
    unrelated[j]         = {"ref", "kind", "bundle"}    (numpy, reused unchanged)
    """
    torch.manual_seed(0)
    np.random.seed(0)

    refs, descendants, unrelated = {}, [], []
    for ref_idx in range(args.n_refs):
        target_key = 42 + ref_idx
        X, y = make_data(args.in_dim, n=2000, seed=ref_idx, target_key=target_key)
        Xv, yv = make_data(args.in_dim, n=400, seed=100 + ref_idx, target_key=target_key)

        ref_model = fresh_model(args.depth, args.hidden, args.in_dim, seed=ref_idx)
        train_model(ref_model, X, y, epochs=args.epochs)
        refs[ref_idx] = {"bundle": bundle(ref_model, Xv), "X": X, "y": y,
                         "Xv": Xv, "yv": yv,
                         "eval_loss": eval_loss(ref_model, Xv, yv)}
        print(f"[ref {ref_idx}] trained, eval_loss={refs[ref_idx]['eval_loss']:.4f}",
              flush=True)

        # ----- 30 related descendants (kept as torch models for laundering) -----
        # descendant_noise's head-bias NaN is now fixed at the root (singleton
        # std() guard in lineage_phase1_mlp), so every descendant is finite by
        # construction. Assert it here rather than sanitize after the fact.
        def _add(kind, model):
            bad = [n for n, p in model.named_parameters()
                   if not torch.isfinite(p).all()]
            assert not bad, f"non-finite parameters in {kind} descendant: {bad}"
            descendants.append({"ref": ref_idx, "kind": kind, "model": model})

        for k in range(args.n_per_descendant_type):
            _add("fine_tune",
                 descendant_fine_tune(ref_model, X, y, epochs=args.ft_epochs))
            X2, y2 = make_data(args.in_dim, n=2000, seed=200 + ref_idx * 10 + k,
                               target_key=target_key + 100 + k)
            _add("fine_tune_new_target",
                 descendant_finetune_new_target(ref_model, X2, y2, epochs=args.ft_epochs))
            _add("noise",
                 descendant_noise(ref_model, sigma_rel=0.02, seed=ref_idx * 100 + k))
            _add("prune",
                 descendant_prune(ref_model, sparsity=0.3, seed=ref_idx * 100 + k))
            _add("quantize", descendant_quantize(ref_model, levels=256))

        # ----- 22 unrelated (diff-seed + distilled), numpy bundles, unchanged -----
        for k in range(args.n_same_arch_diff_seed):
            seed = 1000 + ref_idx * 100 + k
            other = fresh_model(args.depth, args.hidden, args.in_dim, seed=seed)
            train_model(other, X, y, epochs=args.epochs)
            unrelated.append({"ref": ref_idx, "kind": "diff_seed_same_task",
                              "seed": seed,
                              "bundle": bundle(other, refs[ref_idx]["Xv"])})
        for k in range(args.n_distilled):
            stu = nondesc_distilled_student(ref_model, X, epochs=args.epochs // 2,
                                            seed=9000 + ref_idx * 100 + k,
                                            depth=args.depth, hidden=args.hidden,
                                            in_dim=args.in_dim)
            unrelated.append({"ref": ref_idx, "kind": "distilled",
                              "bundle": bundle(stu, refs[ref_idx]["Xv"])})
        print(f"[ref {ref_idx}] bank done ({len(descendants)} desc, "
              f"{len(unrelated)} unrelated so far)", flush=True)
    return refs, descendants, unrelated


# ---------------------------------------------------------- laundering + gate

def launder_descendants(descendants, refs, variant, seed_base, probes):
    """Launder all 30 descendants under `variant`; run the gate; return numpy
    bundles + gate log. Also keeps unlaundered Ms alongside for the invariance
    check. `variant='none'` returns the unlaundered descendants."""
    laundered, gate_log = [], []
    for i, d in enumerate(descendants):
        ref_idx = d["ref"]
        Xv = refs[ref_idx]["Xv"]
        orig = d["model"]
        if variant == "none":
            model, dev = orig, 0.0
        else:
            ft_data = (refs[ref_idx]["X"], refs[ref_idx]["y"])
            model, pre = lops.launder(orig, variant, seed=seed_base + i,
                                      ft_data=ft_data, ft_epochs=args_ft_epochs())
            dev = lops.function_deviation(pre, orig, probes)  # gate on pre-FT stage
            if not lops.gate_ok(dev):
                raise AssertionError(
                    f"GATE FAIL variant={variant} desc#{i} kind={d['kind']} "
                    f"max_dev={dev:.3e} >= {lops.GATE_THRESHOLD}")
        rec = {"ref": ref_idx, "kind": d["kind"], "bundle": bundle(model, Xv)}
        if variant == "PDFT":
            Xv_r, yv_r = refs[ref_idx]["Xv"], refs[ref_idx]["yv"]
            rec["utility_laundered"] = eval_loss(model, Xv_r, yv_r)
            rec["utility_unlaundered"] = eval_loss(orig, Xv_r, yv_r)
        laundered.append(rec)
        gate_log.append({"desc": i, "kind": d["kind"], "ref": ref_idx,
                         "max_deviation": dev})
    return laundered, gate_log


# small global so the laundering helper can read the PDFT epoch count
_ARGS = None
def args_ft_epochs():
    return _ARGS.pdft_epochs


# ------------------------------------------------------------- AUROC assembly

def aggregate(related_scores, unrelated_scores):
    """related_scores: list of {'kind','scores'}; unrelated_scores: same.
    Returns per-method {AUROC, mean_related, min_related, mean_unrelated,
    max_unrelated, n_pairs}."""
    labels = np.array([1] * len(related_scores) + [0] * len(unrelated_scores))
    per_method = {}
    for m in ALL_METHODS:
        rel = np.array([r["scores"][m] for r in related_scores], dtype=float)
        unr = np.array([u["scores"][m] for u in unrelated_scores], dtype=float)
        scores = np.concatenate([rel, unr])
        try:
            au = float(roc_auc_score(labels, scores))
        except Exception:
            au = float("nan")
        per_method[m] = {
            "AUROC": au,
            "mean_related": float(rel.mean()),
            "min_related": float(rel.min()),
            "mean_unrelated": float(unr.mean()),
            "max_unrelated": float(unr.max()),
            "n_pairs": int(len(scores)),
        }
    return per_method


# ------------------------------------------------------------------------ main

def main():
    global _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, default=2)
    ap.add_argument("--n-per-descendant-type", type=int, default=3)
    ap.add_argument("--n-same-arch-diff-seed", type=int, default=8)
    ap.add_argument("--n-distilled", type=int, default=3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--in-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=30,
                    help="epochs used to build the original descendants (Table 6)")
    ap.add_argument("--pdft-epochs", type=int, default=5,
                    help="light fine-tuning epochs for the PDFT variant")
    ap.add_argument("--variants", default="none,P,D-mild,D-strong,PD,PDFT")
    ap.add_argument("--seed-base", type=int, default=7000,
                    help="base seed for laundering randomness")
    ap.add_argument("--workers", type=int, default=0,
                    help="process pool size for scoring (0 => cpu_count-1)")
    ap.add_argument("--out", default="results/laundering/")
    args = ap.parse_args()
    _ARGS = args

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    outdir = Path(args.out)
    (outdir / "by_cell").mkdir(parents=True, exist_ok=True)
    n_workers = args.workers or max(1, (torch.get_num_threads() or 2))
    # cap workers to something sane on this box
    import os
    n_workers = min(n_workers, os.cpu_count() or 4)

    t0 = time.time()
    print(f"Building bank (seed 0)... variants={variants} workers={n_workers}",
          flush=True)
    refs, descendants, unrelated = build_bank(args)
    assert len(descendants) == args.n_refs * args.n_per_descendant_type * 5, \
        f"expected {args.n_refs*args.n_per_descendant_type*5} descendants, got {len(descendants)}"
    print(f"Bank: {len(descendants)} related, {len(unrelated)} unrelated "
          f"({time.time()-t0:.1f}s)", flush=True)

    probes = lops.make_probes(args.in_dim, n=lops.N_PROBES, seed=12345)

    # ---- launder per variant + run the HARD gate (single process, torch) ----
    laundered_by_variant, gate_by_variant, pdft_utility = {}, {}, []
    for v in variants:
        laund, glog = launder_descendants(descendants, refs, v, args.seed_base, probes)
        laundered_by_variant[v] = laund
        gate_by_variant[v] = glog
        maxdev = max(g["max_deviation"] for g in glog) if glog else 0.0
        print(f"[launder {v:8s}] gate max_dev={maxdev:.3e} "
              f"({'PASS' if maxdev < lops.GATE_THRESHOLD else 'FAIL'})", flush=True)
        if v == "PDFT":
            for r in laund:
                pdft_utility.append({"ref": r["ref"], "kind": r["kind"],
                                     "utility_laundered": r["utility_laundered"],
                                     "utility_unlaundered": r["utility_unlaundered"]})

    # ---- build scoring jobs (unrelated scored ONCE; variant-independent) ----
    ref_bundles = {ri: refs[ri]["bundle"] for ri in refs}
    jobs = []
    for j, u in enumerate(unrelated):
        jobs.append(("_unrelated", j, 0, u["ref"], u["kind"], u["bundle"]))
    for v in variants:
        for i, r in enumerate(laundered_by_variant[v]):
            jobs.append((v, i, 1, r["ref"], r["kind"], r["bundle"]))

    print(f"Scoring {len(jobs)} pair-jobs across {n_workers} workers...", flush=True)
    if n_workers <= 1:
        _init_worker(ref_bundles)
        results = [_score_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker,
                                 initargs=(ref_bundles,)) as ex:
            results = list(ex.map(_score_job, jobs, chunksize=4))

    unrelated_scores = [r for r in results if r["variant"] == "_unrelated"]
    related_by_variant = {v: [r for r in results if r["variant"] == v] for v in variants}

    # ---- aggregate per variant ----
    summary = {}  # variant -> method -> stats
    for v in variants:
        summary[v] = aggregate(related_by_variant[v], unrelated_scores)

    # ---- Track D(2): ours-invariance on P/D/PD (dScore + signature cosine) ----
    invariance = {}
    if "none" in variants:
        base = {(r["ref"], r["pair_id"]): r for r in related_by_variant["none"]}
        base_bundle = {i: laundered_by_variant["none"][i]["bundle"]
                       for i in range(len(laundered_by_variant["none"]))}
        for v in ["P", "D-mild", "D-strong", "PD"]:
            if v not in variants:
                continue
            dmax, cos_min = 0.0, 1.0
            for i, r in enumerate(related_by_variant[v]):
                b = base[(r["ref"], r["pair_id"])]
                dmax = max(dmax, abs(r["scores"]["diagonal_dominance"]
                                     - b["scores"]["diagonal_dominance"]))
                Ms_v = laundered_by_variant[v][i]["bundle"]["Ms"]
                Ms_b = base_bundle[i]["Ms"]
                for Mv, Mb in zip(Ms_v, Ms_b):
                    c = float(ldet.residual_signature(Mv) @ ldet.residual_signature(Mb))
                    cos_min = min(cos_min, c)
            invariance[v] = {"max_dScore_vs_none": dmax, "min_signature_cosine": cos_min}
            print(f"[invariance {v:8s}] max|dL|={dmax:.2e}  "
                  f"min sig-cos={cos_min:.10f}", flush=True)

    # ---- Track D(1): launder 4 diff-seed unrelated with PD, confirm unrelated ----
    track_d1 = []
    diff_seed_idx = [j for j, u in enumerate(unrelated)
                     if u["kind"] == "diff_seed_same_task"][:4]
    # Need torch models to launder unrelated; rebuild those 4 from their stored
    # construction seed (identical to build_bank), then PD-launder.
    for rank, j in enumerate(diff_seed_idx):
        u = unrelated[j]
        ref_idx = u["ref"]
        other = fresh_model(args.depth, args.hidden, args.in_dim, seed=u["seed"])
        train_model(other, refs[ref_idx]["X"], refs[ref_idx]["y"], epochs=args.epochs)
        laund_u, pre_u = lops.launder(other, "PD", seed=args.seed_base + 5000 + rank)
        dev = lops.function_deviation(pre_u, other, probes)
        sus = bundle(laund_u, refs[ref_idx]["Xv"])
        sc_l = score_all(refs[ref_idx]["bundle"], sus)
        sc_o = score_all(refs[ref_idx]["bundle"], u["bundle"])
        track_d1.append({"unrelated_idx": j, "ref": ref_idx, "gate_dev": dev,
                         "scores_PD_laundered": sc_l, "scores_unlaundered": sc_o})
    print(f"[Track D1] laundered {len(track_d1)} unrelated with PD "
          f"(max gate_dev={max(t['gate_dev'] for t in track_d1):.2e})", flush=True)

    # ---- write per-(variant, method) JSONs ----
    for v in variants:
        for m in ALL_METHODS:
            rel = related_by_variant[v]
            cell = {
                "variant": v, "method": m,
                **summary[v][m],
                "related_scores": [{"ref": r["ref"], "kind": r["kind"],
                                    "score": r["scores"][m]} for r in rel],
                "unrelated_scores": [{"ref": u["ref"], "kind": u["kind"],
                                      "score": u["scores"][m]} for u in unrelated_scores],
            }
            (outdir / "by_cell" / f"{v}__{m}.json").write_text(json.dumps(cell, indent=2))

    # ---- aggregate summary.csv ----
    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "AUROC", "mean_related", "min_related",
                    "mean_unrelated", "max_unrelated", "n_pairs"])
        for m in ALL_METHODS:
            for v in variants:
                s = summary[v][m]
                w.writerow([m, v, f"{s['AUROC']:.6f}", f"{s['mean_related']:.6f}",
                            f"{s['min_related']:.6f}", f"{s['mean_unrelated']:.6f}",
                            f"{s['max_unrelated']:.6f}", s["n_pairs"]])

    # ---- full JSON dump ----
    full = {
        "config": vars(args),
        "seeds": {"bank": 0, "laundering_base": args.seed_base, "probes": 12345,
                  "pdft_ft_offset": 777, "track_d1_offset": 5000},
        "variants": variants,
        "methods": {"primary": PRIMARY_METHODS, "track_b": TRACK_B_METHODS,
                    "completeness_on_M": COMPLETENESS_METHODS},
        "gate_threshold": lops.GATE_THRESHOLD,
        "n_probes": lops.N_PROBES,
        "summary": summary,
        "gate_deviations": {v: gate_by_variant[v] for v in variants},
        "gate_max_deviation": {v: (max(g["max_deviation"] for g in gate_by_variant[v])
                                   if gate_by_variant[v] else 0.0) for v in variants},
        "ours_invariance": invariance,
        "pdft_utility": pdft_utility,
        "track_d1_unrelated_pd": track_d1,
        "ref_eval_loss": {ri: refs[ri]["eval_loss"] for ri in refs},
        "wall_seconds": time.time() - t0,
    }
    (outdir / "laundering_full.json").write_text(json.dumps(full, indent=2))

    # ---- console table ----
    print(f"\nWrote {outdir}/ ({time.time()-t0:.1f}s)\n")
    print("AUROC table (rows = methods, cols = variants):")
    header = "  " + f"{'method':26s}" + "".join(f"{v:>10s}" for v in variants)
    print(header)
    for m in ALL_METHODS:
        row = f"  {m:26s}" + "".join(f"{summary[v][m]['AUROC']:10.3f}" for v in variants)
        print(row)
    print("\nGate max deviation per variant:")
    for v in variants:
        gm = full["gate_max_deviation"][v]
        print(f"  {v:10s} {gm:.3e}  {'PASS' if gm < lops.GATE_THRESHOLD else 'FAIL'}")


if __name__ == "__main__":
    main()
