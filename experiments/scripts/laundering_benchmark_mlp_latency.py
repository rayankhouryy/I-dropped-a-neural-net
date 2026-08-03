"""Checkpoint-laundering benchmark on the MLP bank with per-method latency measurement.

Extends laundering_benchmark_mlp.py with timing instrumentation for each scoring method.
Outputs latency statistics alongside AUROC results.

Usage:
    python laundering_benchmark_mlp_latency.py --out results/laundering_latency/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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

TAU_S = 0.5

# Methods to benchmark (weight-space only for the paper table)
WEIGHT_SPACE_METHODS = [
    "diagonal_dominance",       # ours
    "raw_aligned_frobenius",
    "raw_singular_value_dist",
    "raw_weight_cosine",
    "rebasin_scale_frobenius",  # Re-Basin + scale
]


def bundle(model, Xv) -> dict:
    """Everything the scorers need, as pure numpy."""
    return {
        "Ms": [np.asarray(M, dtype=np.float64) for M in branch_products(model)],
        "raw": lops.raw_weights(model),
    }


def score_all_with_latency(ref: dict, sus: dict, tau_s: float = TAU_S) -> tuple[dict, dict]:
    """Score all methods and return (scores, latencies_ms)."""
    scores = {}
    latencies = {}

    # Ours (diagonal dominance / centered residual signature)
    t0 = time.perf_counter()
    scores["diagonal_dominance"], _, _ = ldet.lineage_score(ref["Ms"], sus["Ms"], tau_s)
    latencies["diagonal_dominance"] = (time.perf_counter() - t0) * 1000

    # Raw baselines
    t0 = time.perf_counter()
    scores["raw_aligned_frobenius"] = lraw.raw_aligned_frobenius(ref["raw"], sus["raw"])
    latencies["raw_aligned_frobenius"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores["raw_singular_value_dist"] = lraw.raw_singular_value_dist(ref["raw"], sus["raw"])
    latencies["raw_singular_value_dist"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores["raw_weight_cosine"] = lraw.raw_weight_cosine(ref["raw"], sus["raw"])
    latencies["raw_weight_cosine"] = (time.perf_counter() - t0) * 1000

    # Re-Basin + scale (solves L Hungarian assignments)
    t0 = time.perf_counter()
    scores["rebasin_scale_frobenius"] = lraw.rebasin_scale_frobenius(ref["raw"], sus["raw"])
    latencies["rebasin_scale_frobenius"] = (time.perf_counter() - t0) * 1000

    return scores, latencies


_REFS: dict = {}


def _init_worker(refs: dict):
    global _REFS
    _REFS = refs


def _score_job(job: tuple):
    variant, pair_id, label, ref_idx, kind, sus = job
    scores, latencies = score_all_with_latency(_REFS[ref_idx], sus)
    return {"variant": variant, "pair_id": pair_id, "label": label,
            "ref": ref_idx, "kind": kind, "scores": scores, "latencies": latencies}


def build_bank(args):
    """Rebuild the 52-pair bank."""
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


_ARGS = None
def args_ft_epochs():
    return _ARGS.pdft_epochs


def launder_descendants(descendants, refs, variant, seed_base, probes):
    """Launder all descendants under `variant`."""
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
            dev = lops.function_deviation(pre, orig, probes)
            if not lops.gate_ok(dev):
                raise AssertionError(
                    f"GATE FAIL variant={variant} desc#{i} kind={d['kind']} "
                    f"max_dev={dev:.3e} >= {lops.GATE_THRESHOLD}")
        rec = {"ref": ref_idx, "kind": d["kind"], "bundle": bundle(model, Xv)}
        laundered.append(rec)
        gate_log.append({"desc": i, "kind": d["kind"], "ref": ref_idx,
                         "max_deviation": dev})
    return laundered, gate_log


def aggregate(related_scores, unrelated_scores):
    """Compute AUROC and score statistics per method."""
    labels = np.array([1] * len(related_scores) + [0] * len(unrelated_scores))
    per_method = {}
    for m in WEIGHT_SPACE_METHODS:
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


def aggregate_latencies(results):
    """Aggregate latency statistics across all scoring calls."""
    latencies_by_method = defaultdict(list)
    for r in results:
        if "latencies" in r:
            for m, lat in r["latencies"].items():
                latencies_by_method[m].append(lat)

    stats = {}
    for m in WEIGHT_SPACE_METHODS:
        lats = latencies_by_method.get(m, [])
        if lats:
            stats[m] = {
                "mean_ms": float(np.mean(lats)),
                "std_ms": float(np.std(lats)),
                "min_ms": float(np.min(lats)),
                "max_ms": float(np.max(lats)),
                "n_calls": len(lats),
            }
    return stats


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
    ap.add_argument("--ft-epochs", type=int, default=30)
    ap.add_argument("--pdft-epochs", type=int, default=5)
    ap.add_argument("--variants", default="none,P,D-mild,D-strong,PD,PDFT")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=1,
                    help="Use 1 worker for accurate latency measurement")
    ap.add_argument("--out", default="results/laundering_latency/")
    args = ap.parse_args()
    _ARGS = args

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    n_workers = args.workers

    t0 = time.time()
    print(f"Building bank (seed 0)... variants={variants}", flush=True)
    refs, descendants, unrelated = build_bank(args)
    print(f"Bank: {len(descendants)} related, {len(unrelated)} unrelated "
          f"({time.time()-t0:.1f}s)", flush=True)

    probes = lops.make_probes(args.in_dim, n=lops.N_PROBES, seed=12345)

    # Launder per variant
    laundered_by_variant, gate_by_variant = {}, {}
    for v in variants:
        laund, glog = launder_descendants(descendants, refs, v, args.seed_base, probes)
        laundered_by_variant[v] = laund
        gate_by_variant[v] = glog
        maxdev = max(g["max_deviation"] for g in glog) if glog else 0.0
        print(f"[launder {v:8s}] gate max_dev={maxdev:.3e}", flush=True)

    # Build scoring jobs
    ref_bundles = {ri: refs[ri]["bundle"] for ri in refs}
    jobs = []
    for j, u in enumerate(unrelated):
        jobs.append(("_unrelated", j, 0, u["ref"], u["kind"], u["bundle"]))
    for v in variants:
        for i, r in enumerate(laundered_by_variant[v]):
            jobs.append((v, i, 1, r["ref"], r["kind"], r["bundle"]))

    print(f"Scoring {len(jobs)} pair-jobs (single-threaded for latency)...", flush=True)
    _init_worker(ref_bundles)
    results = [_score_job(job) for job in jobs]

    unrelated_scores = [r for r in results if r["variant"] == "_unrelated"]
    related_by_variant = {v: [r for r in results if r["variant"] == v] for v in variants}

    # Aggregate AUROC per variant
    summary = {}
    for v in variants:
        summary[v] = aggregate(related_by_variant[v], unrelated_scores)

    # Aggregate latencies (from 'none' variant - representative)
    all_scored = unrelated_scores + related_by_variant.get("none", [])
    latency_stats = aggregate_latencies(all_scored)

    # Write summary CSV with latency
    csv_path = outdir / "summary_with_latency.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "AUROC", "mean_related", "min_related",
                    "mean_unrelated", "max_unrelated", "latency_ms"])
        for m in WEIGHT_SPACE_METHODS:
            lat = latency_stats.get(m, {}).get("mean_ms", float("nan"))
            for v in variants:
                s = summary[v][m]
                w.writerow([m, v, f"{s['AUROC']:.4f}", f"{s['mean_related']:.6f}",
                            f"{s['min_related']:.6f}", f"{s['mean_unrelated']:.6f}",
                            f"{s['max_unrelated']:.6f}", f"{lat:.3f}"])

    # Write latency-only CSV
    lat_csv = outdir / "latency_stats.csv"
    with lat_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_ms", "std_ms", "min_ms", "max_ms", "n_calls"])
        for m in WEIGHT_SPACE_METHODS:
            s = latency_stats.get(m, {})
            w.writerow([m, f"{s.get('mean_ms', 0):.3f}", f"{s.get('std_ms', 0):.3f}",
                        f"{s.get('min_ms', 0):.3f}", f"{s.get('max_ms', 0):.3f}",
                        s.get('n_calls', 0)])

    # Full JSON dump
    full = {
        "config": vars(args),
        "variants": variants,
        "methods": WEIGHT_SPACE_METHODS,
        "summary": summary,
        "latency_stats": latency_stats,
        "gate_max_deviation": {v: (max(g["max_deviation"] for g in gate_by_variant[v])
                                   if gate_by_variant[v] else 0.0) for v in variants},
        "wall_seconds": time.time() - t0,
    }
    (outdir / "results_with_latency.json").write_text(json.dumps(full, indent=2))

    # Console output
    print(f"\n{'='*70}")
    print("AUROC + Latency (weight-space methods only)")
    print(f"{'='*70}")
    header = f"{'Method':28s}" + "".join(f"{v:>8s}" for v in variants) + f"{'Lat(ms)':>10s}"
    print(header)
    print("-" * len(header))
    for m in WEIGHT_SPACE_METHODS:
        lat = latency_stats.get(m, {}).get("mean_ms", float("nan"))
        row = f"{m:28s}" + "".join(f"{summary[v][m]['AUROC']:8.2f}" for v in variants)
        row += f"{lat:10.2f}"
        print(row)

    print(f"\nWrote {outdir}/ ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
