"""E1 (GPT-2 half): global residual-stream permutation attack + recovery defense.

Self-contained. Runs on the already-trained paper benchmark artifacts (no
retraining): phase1_roots.pkl / phase2_descendants.pkl (cached branch products
Ms), checkpoints/root_i/epoch_N.pt (root weights) and models/<id>.pt (suspect
weights, needed for the raw-weight / Re-Basin baselines).

Builds the 45 test-split pairs exactly as compute_gpt2_baselines_auroc.py does
(21 descendant positives + 3 distilled + 21 cross-root negatives), then for each
of --seeds permutation seeds evaluates three conditions on five methods:

    Clean              -- unlaundered reproduction
    Global-P           -- one global residual permutation P per attacked suspect
                          (descendant positives only; negatives stay clean),
                          realized on cached Ms as M -> P M P^T and on raw weights
                          by reindexing the d_model residual dimension
    Global-P+recovery  -- recover P via row/col-norm descriptors + Hungarian and
                          undo it before scoring (applied to ALL pairs)

Methods (paper Table 5 `tab:laundering`: Ours | Re-Basin | Al.Frob | SVD | W.Cos):
    ours                    = lineage_detection.lineage_score (gated Hungarian, on M)
    rebasin_scale           = gpt2_laundering_baselines.rebasin_scale_frobenius_gpt2 (raw)
    raw_aligned_frobenius   = gpt2_laundering_baselines.raw_aligned_frobenius_gpt2   (raw)
    raw_singular_value_dist = gpt2_laundering_baselines.singular_value_distance_gpt2 (raw)
    raw_weight_cosine       = gpt2_laundering_baselines.raw_weight_cosine_gpt2       (raw)

The function-preservation gate (validate_function_preservation, max|Δlogit|<1e-4,
top-1 == 1.0) is certified on real attacked descendant models (GPU forward passes;
CPU fallback works but is slower). For a permutation the LayerNorm transform is
EXACT, so the gate passes at fp32 roundoff (vs ~0.55 for the dense-Q attempt).

Run on the SageMaker box (has the 17 GB artifacts + a GPU):
    cd ~/I-dropped-a-neural-net/experiments/scripts
    python e1_global_perm_gpt2.py \
        --benchmark-dir results/lineage_benchmark_gpt2_paper_v2 \
        --epochs 3 --seeds 5
Expected runtime: a few minutes (no training). Writes results/laundering/e1_global_perm/gpt2/.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import attack_global_perm as agp  # noqa: E402
import lineage_detection as ldet  # noqa: E402
import gpt2_laundering_baselines as glb  # noqa: E402

CONDITIONS = ["clean", "global_p", "global_p_recovery"]
METHODS = [
    "ours",
    "rebasin_scale",
    "raw_aligned_frobenius",
    "raw_singular_value_dist",
    "raw_weight_cosine",
]


# ---------------------------------------------------------------- scorers

def _score(method: str, ref, sus, tau_s: float) -> float:
    """ref/sus are {'Ms': [np.ndarray], 'raw': {'Wins','Wouts'} or None}."""
    if method == "ours":
        return ldet.lineage_score(ref["Ms"], sus["Ms"], tau_s)[0]
    if ref["raw"] is None or sus["raw"] is None:
        return float("nan")  # raw weights unavailable (e.g. --save-models not used)
    if method == "rebasin_scale":
        return glb.rebasin_scale_frobenius_gpt2(ref["raw"], sus["raw"])
    if method == "raw_aligned_frobenius":
        return glb.raw_aligned_frobenius_gpt2(ref["raw"], sus["raw"])
    if method == "raw_singular_value_dist":
        return glb.singular_value_distance_gpt2(ref["raw"], sus["raw"])
    if method == "raw_weight_cosine":
        return glb.raw_weight_cosine_gpt2(ref["raw"], sus["raw"])
    raise ValueError(method)


def _score_all(ref, sus, tau_s) -> dict:
    return {m: _score(m, ref, sus, tau_s) for m in METHODS}


def _attack_pack(pack, P, perm) -> dict:
    """Apply the global permutation to a clean pack: M -> P M P^T, raw reindexed."""
    Ms_att = [P @ np.asarray(M, dtype=np.float64) @ P.T for M in pack["Ms"]]
    raw_att = agp.permute_residual_raw(pack["raw"], perm) if pack["raw"] is not None else None
    return {"Ms": Ms_att, "raw": raw_att}


def _recover_pack(ref, sus) -> dict:
    """Blind recovery: recover P from branch products, undo on M and raw."""
    col_ind = agp.recover_permutation(ref["Ms"], sus["Ms"])
    Ms_al = agp.align_Ms(sus["Ms"], col_ind)
    raw_al = agp.permute_residual_raw(sus["raw"], col_ind) if sus["raw"] is not None else None
    return {"Ms": Ms_al, "raw": raw_al}, col_ind


# ---------------------------------------------------------------- aggregate

def _aggregate(related, unrelated) -> dict:
    out = {}
    for m in METHODS:
        rel = np.array([r[m] for r in related], dtype=float)
        unr = np.array([u[m] for u in unrelated], dtype=float)
        labels = np.array([1] * len(rel) + [0] * len(unr))
        scores = np.concatenate([rel, unr])
        finite = np.isfinite(scores)
        try:
            if finite.any() and len(set(labels[finite].tolist())) == 2:
                au = float(roc_auc_score(labels[finite], scores[finite]))
            else:
                au = float("nan")
        except Exception:
            au = float("nan")
        out[m] = {
            "AUROC": au,
            "mean_related": float(np.nanmean(rel)) if rel.size else float("nan"),
            "min_related": float(np.nanmin(rel)) if rel.size else float("nan"),
            "mean_unrelated": float(np.nanmean(unr)) if unr.size else float("nan"),
            "max_unrelated": float(np.nanmax(unr)) if unr.size else float("nan"),
            "n_pairs": int(len(scores)),
        }
    return out


# ---------------------------------------------------------------- artifact loading

def _load_pickles(bdir: Path):
    p1 = bdir / "phase1_roots.pkl"
    p2 = bdir / "phase2_descendants.pkl"
    for p in (p1, p2):
        if not p.exists():
            print(f"\nERROR: missing {p}\nThe GPT-2 benchmark artifacts are absent. "
                  f"Train once (GPU, ~6-8 hr) with:\n"
                  f"    python -m gpt2_lineage_benchmark.run_benchmark --preset paper "
                  f"--save-models --device cuda\n", file=sys.stderr)
            sys.exit(2)
    with open(p1, "rb") as f:
        roots_pkl = pickle.load(f)
    with open(p2, "rb") as f:
        desc_pkl = pickle.load(f)
    print(f"[schema] phase1 keys: {list(roots_pkl.keys())}")
    print(f"[schema] phase2 keys: {list(desc_pkl.keys())}")
    return roots_pkl, desc_pkl


def _resolve_split(roots_info, bdir: Path):
    # Prefer explicit splits in roots_info; else benchmark_results.json; else [5,6,7].
    test = [info["root_idx"] for info in roots_info if info.get("split") == "test"]
    if not test:
        br = bdir / "benchmark_results.json"
        if br.exists():
            with open(br) as f:
                bench = json.load(f)
            test = [r["root_idx"] for r in bench.get("roots", []) if r.get("split") == "test"]
    if not test:
        test = [5, 6, 7]
    return sorted(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", default="results/lineage_benchmark_gpt2_paper_v2")
    ap.add_argument("--epochs", type=int, default=3,
                    help="root checkpoint epoch to load (checkpoints/root_i/epoch_N.pt)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--gate-per-seed", type=int, default=3,
                    help="# attacked descendants to certify with the function gate "
                         "per seed (spread across test roots; 0 = all). The permutation "
                         "is exactly function-preserving by construction, so a sample is "
                         "a representative certificate.")
    ap.add_argument("--no-gate", action="store_true", help="skip the forward-pass gate")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/laundering/e1_global_perm/gpt2"))
    args = ap.parse_args()

    import torch  # noqa: E402
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    bdir = Path(args.benchmark_dir)
    if not bdir.is_absolute():
        bdir = SCRIPT_DIR / args.benchmark_dir
    outdir = Path(args.out)
    (outdir / "by_cell").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"benchmark_dir={bdir}\ndevice={device}\n")
    roots_pkl, desc_pkl = _load_pickles(bdir)

    roots_info = roots_pkl["roots_info"]
    root_sigs = roots_pkl["root_signatures"]
    tau_s = float(roots_pkl.get("tau_s", 0.5))
    d_model = int(np.asarray(root_sigs[0][0]).shape[0])
    print(f"tau_s={tau_s:.4f}  d_model={d_model}  L={len(root_sigs[0])}")

    roots = {info["root_idx"]: {"Ms": [np.asarray(M, np.float64) for M in Ms]}
             for info, Ms in zip(roots_info, root_sigs)}
    descendants = {d["id"]: {"Ms": [np.asarray(M, np.float64) for M in d["Ms"]],
                             "root_idx": d["root_idx"], "type": d["type"]}
                   for d in desc_pkl["descendants"]}
    students = {s["id"]: {"Ms": [np.asarray(M, np.float64) for M in s["Ms"]],
                          "root_idx": s.get("root_idx", s.get("teacher_root_idx"))}
                for s in desc_pkl["students"]}
    test_roots = _resolve_split(roots_info, bdir)
    print(f"test_roots={test_roots}  "
          f"({len(descendants)} descendants, {len(students)} students total)")

    # ---- raw-weight loader (roots from checkpoints, suspects from models/) ----
    from gpt2_lineage_benchmark.config import ModelConfig
    from gpt2_lineage_benchmark.model import load_checkpoint
    mcfg = ModelConfig()
    raw_cache: dict = {}

    def _load_model(key: str):
        if key.startswith("root_"):
            i = int(key.split("_")[1])
            path = bdir / "checkpoints" / f"root_{i}" / f"epoch_{args.epochs}.pt"
        else:
            path = bdir / "models" / f"{key}.pt"
        if not path.exists():
            return None
        model, *_ = load_checkpoint(path, config=mcfg, device=device)
        return model

    def _raw_for(key: str):
        if key in raw_cache:
            return raw_cache[key]
        model = _load_model(key)
        if model is None:
            print(f"  [warn] weights missing for {key}; raw baselines -> NaN for its pairs")
            raw_cache[key] = None
            return None
        from laundering_gpt2_ops import raw_weights_gpt2
        raw_cache[key] = raw_weights_gpt2(model)
        del model
        return raw_cache[key]

    # ---- build the 45 pairs ----
    # positive: descendant (test root); negative: distilled (test root) + cross-root.
    pairs = []  # {kind, label, ref_key, sus_key, ref_Ms, sus_Ms, attacked(bool)}
    for did, d in descendants.items():
        if d["root_idx"] in test_roots and d["root_idx"] in roots:
            pairs.append({"kind": d["type"], "label": 1, "ref_key": f"root_{d['root_idx']}",
                          "sus_key": did, "ref_Ms": roots[d["root_idx"]]["Ms"],
                          "sus_Ms": d["Ms"], "attacked": True})
    for sid, s in students.items():
        if s["root_idx"] in test_roots and s["root_idx"] in roots:
            pairs.append({"kind": "distilled", "label": 0, "ref_key": f"root_{s['root_idx']}",
                          "sus_key": sid, "ref_Ms": roots[s["root_idx"]]["Ms"],
                          "sus_Ms": s["Ms"], "attacked": False})
    for i in test_roots:
        for j in roots:
            if i == j:
                continue
            pairs.append({"kind": "cross_root", "label": 0, "ref_key": f"root_{i}",
                          "sus_key": f"root_{j}", "ref_Ms": roots[i]["Ms"],
                          "sus_Ms": roots[j]["Ms"], "attacked": False})
    n_pos = sum(p["label"] for p in pairs)
    print(f"pairs={len(pairs)}  positives={n_pos}  negatives={len(pairs)-n_pos}\n")

    # ---- load raw weights for every entity referenced (once) ----
    for p in pairs:
        p["ref_raw"] = _raw_for(p["ref_key"])
        p["sus_raw"] = _raw_for(p["sus_key"])

    def ref_pack(p):
        return {"Ms": p["ref_Ms"], "raw": p["ref_raw"]}

    def clean_sus_pack(p):
        return {"Ms": p["sus_Ms"], "raw": p["sus_raw"]}

    # ---- gate helper (loads real models, applies P, validates) ----
    tokenizer = valbatch = None
    if not args.no_gate:
        from transformers import GPT2Tokenizer
        from laundering_gpt2_ops import make_validation_batch
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        valbatch = make_validation_batch(tokenizer)

    def run_gate(sus_key, seed):
        from laundering_gpt2_ops import validate_function_preservation
        model = _load_model(sus_key)
        if model is None:
            return None
        perm_model, _ = agp.apply_global_permutation_gpt2(model, seed, device=device)
        res = validate_function_preservation(model, perm_model, valbatch, device=device)
        del model, perm_model
        return res

    # gate targets: spread across test roots
    pos_pairs = [p for p in pairs if p["attacked"]]
    if args.gate_per_seed and args.gate_per_seed < len(pos_pairs):
        by_root, targets = {}, []
        for p in pos_pairs:
            by_root.setdefault(p["ref_key"], []).append(p)
        while len(targets) < args.gate_per_seed:
            for k in list(by_root):
                if by_root[k]:
                    targets.append(by_root[k].pop(0))
                    if len(targets) >= args.gate_per_seed:
                        break
    else:
        targets = pos_pairs

    # ---- Clean condition (seed-independent) ----
    clean_scores = [(_score_all(ref_pack(p), clean_sus_pack(p), tau_s), p) for p in pairs]
    clean_rel = [s for s, p in clean_scores if p["label"] == 1]
    clean_unr = [s for s, p in clean_scores if p["label"] == 0]
    clean_agg = _aggregate(clean_rel, clean_unr)

    # ---- per-seed sweep ----
    per_seed = {c: {m: [] for m in METHODS} for c in CONDITIONS}
    for m in METHODS:
        per_seed["clean"][m] = [clean_agg[m]["AUROC"]] * args.seeds
    gate_logs = []
    recovery_exact = []           # per (seed, positive)
    n_desc_exact = []             # per seed
    cells = {"clean": {"related": clean_rel, "unrelated": clean_unr}}

    for s in range(args.seeds):
        # per-attacked-suspect permutation (indexed by position among positives)
        perms = {}
        for i, p in enumerate(pos_pairs):
            P, perm = agp.make_global_permutation(d_model, args.seed_base + 1000 * s + i)
            perms[p["sus_key"]] = (P, perm)

        gp_rel, gp_unr, gpr_rel, gpr_unr = [], [], [], []
        exact_this = []
        for p in pairs:
            ref = ref_pack(p)
            clean_sus = clean_sus_pack(p)
            if p["attacked"]:
                P, perm = perms[p["sus_key"]]
                gp_sus = _attack_pack(clean_sus, P, perm)
            else:
                gp_sus = clean_sus
            gp = _score_all(ref, gp_sus, tau_s)
            rec_sus, col_ind = _recover_pack(ref, gp_sus)
            gpr = _score_all(ref, rec_sus, tau_s)
            if p["label"] == 1:
                gp_rel.append(gp); gpr_rel.append(gpr)
                if p["attacked"]:
                    frac = agp.exact_recovery_fraction(col_ind, perms[p["sus_key"]][1])
                    exact_this.append(frac); recovery_exact.append(frac)
            else:
                gp_unr.append(gp); gpr_unr.append(gpr)

        gp_agg = _aggregate(gp_rel, gp_unr)
        gpr_agg = _aggregate(gpr_rel, gpr_unr)
        for m in METHODS:
            per_seed["global_p"][m].append(gp_agg[m]["AUROC"])
            per_seed["global_p_recovery"][m].append(gpr_agg[m]["AUROC"])
        n_desc_exact.append(int(sum(1 for f in exact_this if f == 1.0)))

        # gate this seed
        seed_gate = []
        if not args.no_gate:
            for p in targets:
                res = run_gate(p["sus_key"], args.seed_base + 1000 * s
                               + pos_pairs.index(p))
                if res is not None:
                    seed_gate.append({"seed": s, "sus": p["sus_key"], **res})
        gate_logs.extend(seed_gate)
        gmax = max((g["max_logit_diff"] for g in seed_gate), default=float("nan"))
        print(f"[seed {s}] ours GP={gp_agg['ours']['AUROC']:.3f} "
              f"GP+rec={gpr_agg['ours']['AUROC']:.3f}  "
              f"exact_desc={n_desc_exact[-1]}/{len(pos_pairs)}  "
              f"gate_max_logit={gmax:.2e}", flush=True)
        if s == args.seeds - 1:
            cells["global_p"] = {"related": gp_rel, "unrelated": gp_unr}
            cells["global_p_recovery"] = {"related": gpr_rel, "unrelated": gpr_unr}

    # ---- summarize ----
    def mean_std(v):
        a = np.array(v, float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    summary = {c: {m: dict(zip(("mean", "std"), mean_std(per_seed[c][m])))
                   for m in METHODS} for c in CONDITIONS}
    gate_max = max((g["max_logit_diff"] for g in gate_logs), default=float("nan"))
    gate_top1_min = min((g["top1_agreement"] for g in gate_logs), default=float("nan"))
    gate_pass_all = all(g["gate_pass"] for g in gate_logs) if gate_logs else None

    # ---- by_cell ----
    kinds = {"related": [p["kind"] for p in pairs if p["label"] == 1],
             "unrelated": [p["kind"] for p in pairs if p["label"] == 0]}
    refs = {"related": [p["ref_key"] for p in pairs if p["label"] == 1],
            "unrelated": [p["ref_key"] for p in pairs if p["label"] == 0]}
    for c in CONDITIONS:
        rel, unr = cells[c]["related"], cells[c]["unrelated"]
        agg = _aggregate(rel, unr)
        for m in METHODS:
            cell = {"variant": c, "method": m, **agg[m],
                    "AUROC_per_seed": per_seed[c][m],
                    "AUROC_mean": summary[c][m]["mean"], "AUROC_std": summary[c][m]["std"],
                    "related_scores": [{"ref": refs["related"][k], "kind": kinds["related"][k],
                                        "score": rel[k][m]} for k in range(len(rel))],
                    "unrelated_scores": [{"ref": refs["unrelated"][k], "kind": kinds["unrelated"][k],
                                          "score": unr[k][m]} for k in range(len(unr))]}
            (outdir / "by_cell" / f"{c}__{m}.json").write_text(json.dumps(cell, indent=2))

    # ---- summary.csv ----
    with (outdir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "seed", "AUROC"])
        for m in METHODS:
            for c in CONDITIONS:
                for s in range(args.seeds):
                    w.writerow([m, c, s, f"{per_seed[c][m][s]:.6f}"])
                w.writerow([m, c, "mean", f"{summary[c][m]['mean']:.6f}"])
                w.writerow([m, c, "std", f"{summary[c][m]['std']:.6f}"])

    # ---- full JSON ----
    full = {
        "config": vars(args), "benchmark": f"GPT-2 45-pair (d_model={d_model}, L={len(root_sigs[0])})",
        "device": device, "tau_s": tau_s, "test_roots": test_roots,
        "seeds": {"perm_base": args.seed_base, "perm_formula": "seed_base + 1000*s + pos_idx"},
        "conditions": CONDITIONS, "methods": METHODS,
        "auroc_per_seed": per_seed, "auroc_mean_std": summary,
        "gate": {"threshold_logit": 1e-4, "max_logit_diff": gate_max,
                 "min_top1_agreement": gate_top1_min, "all_pass": gate_pass_all,
                 "n_certified": len(gate_logs), "logs": gate_logs},
        "recovery": {
            "exact_coord_fraction_mean": float(np.mean(recovery_exact)) if recovery_exact else None,
            "exact_coord_fraction_min": float(np.min(recovery_exact)) if recovery_exact else None,
            "n_positives": n_pos, "n_desc_exactly_recovered_per_seed": n_desc_exact,
        },
        "raw_available": {k: (v is not None) for k, v in raw_cache.items()},
        "wall_seconds": time.time() - t0,
    }
    (outdir / "e1_full.json").write_text(json.dumps(full, indent=2))

    # ---- console ----
    print(f"\nWrote {outdir}/ ({time.time()-t0:.1f}s)\n")
    print("AUROC (mean over seeds; rows=methods, cols=conditions):")
    print("  " + f"{'method':26s}" + "".join(f"{c:>20s}" for c in CONDITIONS))
    for m in METHODS:
        row = f"  {m:26s}"
        for c in CONDITIONS:
            row += f"{summary[c][m]['mean']:12.3f}±{summary[c][m]['std']:.3f}   "
        print(row)
    print(f"\nGate: max|Δlogit|={gate_max:.3e}  min top1={gate_top1_min:.4f}  "
          f"all_pass={gate_pass_all}  (n={len(gate_logs)})")
    if recovery_exact:
        print(f"Recovery: exact-coord fraction mean={np.mean(recovery_exact):.4f}, "
              f"descendants exactly recovered per seed={n_desc_exact}/{n_pos}")


if __name__ == "__main__":
    main()
