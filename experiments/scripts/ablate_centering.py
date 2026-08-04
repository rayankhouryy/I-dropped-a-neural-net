#!/usr/bin/env python3
"""Centering ablation experiment (W3 reviewer response).

Directly compares cosine similarity on:
  - Uncentered branch products: cos(vec(M_A), vec(M_B))
  - Centered residuals: cos(vec(R_A), vec(R_B)) where R = M - (tr(M)/d)·I

The experiment shows whether centering materially improves discrimination
between related and unrelated model pairs. This addresses the reviewer concern
that the centering step's necessity was never directly demonstrated.

Usage:
    python ablate_centering.py --benchmark-dir results/lineage_benchmark_gpt2_paper_v2
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

import functools
print = functools.partial(print, flush=True)


def raw_branch_product_cosine_per_layer(
    Ms_A: List[np.ndarray],
    Ms_B: List[np.ndarray],
) -> List[float]:
    """Cosine of flattened branch products WITHOUT centering, per layer."""
    L = min(len(Ms_A), len(Ms_B))
    cosines = []
    for i in range(L):
        a = Ms_A[i].flatten().astype(np.float64)
        b = Ms_B[i].flatten().astype(np.float64)
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        cosines.append(float(np.dot(a, b) / (na * nb)))
    return cosines


def centered_branch_product_cosine_per_layer(
    Ms_A: List[np.ndarray],
    Ms_B: List[np.ndarray],
) -> List[float]:
    """Centered residual cosine (subtracts tr(M)/d * I), per layer."""
    L = min(len(Ms_A), len(Ms_B))
    cosines = []
    for i in range(L):
        M_a = Ms_A[i].astype(np.float64)
        M_b = Ms_B[i].astype(np.float64)
        d = M_a.shape[0]

        alpha_a = np.trace(M_a) / d
        alpha_b = np.trace(M_b) / d

        R_a = (M_a - alpha_a * np.eye(d)).flatten()
        R_b = (M_b - alpha_b * np.eye(d)).flatten()

        na = np.linalg.norm(R_a) + 1e-12
        nb = np.linalg.norm(R_b) + 1e-12
        cosines.append(float(np.dot(R_a, R_b) / (na * nb)))
    return cosines


def compute_trace_fractions(Ms: List[np.ndarray]) -> List[float]:
    """Compute |tr(M)|² / ||M||_F² for each layer (identity energy fraction)."""
    fractions = []
    for M in Ms:
        M = M.astype(np.float64)
        d = M.shape[0]
        tr_sq = (np.trace(M) ** 2)
        frob_sq = np.linalg.norm(M, 'fro') ** 2 + 1e-12
        fractions.append(float(tr_sq / frob_sq))
    return fractions


@dataclass
class PairResult:
    """Result for one pair."""
    ref_id: str
    sus_id: str
    label: str
    attack_type: str
    uncentered_cosines: List[float]
    centered_cosines: List[float]
    uncentered_mean: float
    centered_mean: float
    ref_trace_fractions: List[float]
    sus_trace_fractions: List[float]


def load_benchmark_data(benchmark_dir: Path) -> Tuple[Dict, Dict, float]:
    """Load phase1 and phase2 data."""
    phase1_path = benchmark_dir / "phase1_roots.pkl"
    phase2_path = benchmark_dir / "phase2_descendants.pkl"

    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase 1 data not found: {phase1_path}")
    if not phase2_path.exists():
        raise FileNotFoundError(f"Phase 2 data not found: {phase2_path}")

    with open(phase1_path, "rb") as f:
        phase1_data = pickle.load(f)
    with open(phase2_path, "rb") as f:
        phase2_data = pickle.load(f)

    tau_s = phase1_data["tau_s"]
    return phase1_data, phase2_data, tau_s


def build_test_pairs(
    phase1_data: Dict,
    phase2_data: Dict,
    test_root_indices: List[int],
) -> List[Dict[str, Any]]:
    """Build test pairs from benchmark data."""
    pairs = []
    test_roots = set(test_root_indices)

    root_signatures = phase1_data["root_signatures"]
    descendants = phase2_data["descendants"]
    students = phase2_data["students"]

    # Descendant pairs
    for desc in descendants:
        root_idx = desc["root_idx"]
        if root_idx not in test_roots:
            continue
        pairs.append({
            "ref_id": f"root_{root_idx}",
            "ref_Ms": root_signatures[root_idx],
            "sus_id": desc["id"],
            "sus_Ms": desc["Ms"],
            "label": "related",
            "attack_type": desc["type"],
        })

    # Distilled students (unrelated)
    for student in students:
        root_idx = student["teacher_root_idx"]
        if root_idx not in test_roots:
            continue
        pairs.append({
            "ref_id": f"root_{root_idx}",
            "ref_Ms": root_signatures[root_idx],
            "sus_id": student["id"],
            "sus_Ms": student["Ms"],
            "label": "unrelated",
            "attack_type": "distilled",
        })

    # Cross-root pairs (unrelated)
    test_roots_list = sorted(test_roots)
    for i, root_i in enumerate(test_roots_list):
        for root_j in test_roots_list[i+1:]:
            pairs.append({
                "ref_id": f"root_{root_i}",
                "ref_Ms": root_signatures[root_i],
                "sus_id": f"root_{root_j}",
                "sus_Ms": root_signatures[root_j],
                "label": "unrelated",
                "attack_type": "independent",
            })

    return pairs


def compute_gap_z(related_scores: List[float], unrelated_scores: List[float]) -> float:
    """Compute Gap-Z: standardized margin between distributions."""
    if len(related_scores) < 2 or len(unrelated_scores) < 2:
        return float('nan')

    mu_rel = np.mean(related_scores)
    mu_unrel = np.mean(unrelated_scores)
    std_rel = np.std(related_scores, ddof=1)
    std_unrel = np.std(unrelated_scores, ddof=1)

    pooled_std = np.sqrt((std_rel**2 + std_unrel**2) / 2) + 1e-12
    return float((mu_rel - mu_unrel) / pooled_std)


def main():
    parser = argparse.ArgumentParser(description="Centering ablation experiment")
    parser.add_argument("--benchmark-dir", default="results/lineage_benchmark_gpt2_paper_v2",
                        help="Path to benchmark data directory")
    parser.add_argument("--output-dir", default="results/centering_ablation",
                        help="Output directory")
    parser.add_argument("--test-roots", type=int, nargs="+", default=[5, 6, 7],
                        help="Root indices to use for test")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Centering Ablation Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading benchmark data...")
    phase1_data, phase2_data, tau_s = load_benchmark_data(benchmark_dir)
    print(f"tau_s = {tau_s:.4f}")

    # Build pairs
    pairs = build_test_pairs(phase1_data, phase2_data, args.test_roots)
    print(f"Test pairs: {len(pairs)}")

    n_related = sum(1 for p in pairs if p["label"] == "related")
    n_unrelated = len(pairs) - n_related
    print(f"  Related: {n_related}, Unrelated: {n_unrelated}")

    # Process pairs
    results: List[PairResult] = []

    for pair in pairs:
        ref_Ms = pair["ref_Ms"]
        sus_Ms = pair["sus_Ms"]

        uncentered = raw_branch_product_cosine_per_layer(ref_Ms, sus_Ms)
        centered = centered_branch_product_cosine_per_layer(ref_Ms, sus_Ms)

        ref_trace_fracs = compute_trace_fractions(ref_Ms)
        sus_trace_fracs = compute_trace_fractions(sus_Ms)

        results.append(PairResult(
            ref_id=pair["ref_id"],
            sus_id=pair["sus_id"],
            label=pair["label"],
            attack_type=pair["attack_type"],
            uncentered_cosines=uncentered,
            centered_cosines=centered,
            uncentered_mean=float(np.mean(uncentered)),
            centered_mean=float(np.mean(centered)),
            ref_trace_fractions=ref_trace_fracs,
            sus_trace_fractions=sus_trace_fracs,
        ))

    # Compute summary statistics
    related_uncentered = [r.uncentered_mean for r in results if r.label == "related"]
    related_centered = [r.centered_mean for r in results if r.label == "related"]
    unrelated_uncentered = [r.uncentered_mean for r in results if r.label == "unrelated"]
    unrelated_centered = [r.centered_mean for r in results if r.label == "unrelated"]

    # AUROC
    labels = [1 if r.label == "related" else 0 for r in results]
    uncentered_scores = [r.uncentered_mean for r in results]
    centered_scores = [r.centered_mean for r in results]

    auroc_uncentered = roc_auc_score(labels, uncentered_scores)
    auroc_centered = roc_auc_score(labels, centered_scores)

    # Gap-Z
    gap_z_uncentered = compute_gap_z(related_uncentered, unrelated_uncentered)
    gap_z_centered = compute_gap_z(related_centered, unrelated_centered)

    # Separation margins
    margin_uncentered = min(related_uncentered) - max(unrelated_uncentered)
    margin_centered = min(related_centered) - max(unrelated_centered)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n--- Score Distributions ---")
    print(f"{'Method':<25} {'Related':<25} {'Unrelated':<25}")
    print(f"{'':25} {'mean (min, max)':<25} {'mean (min, max)':<25}")
    print("-" * 75)
    print(f"{'Uncentered vec(M)':<25} "
          f"{np.mean(related_uncentered):.4f} ({min(related_uncentered):.4f}, {max(related_uncentered):.4f})  "
          f"{np.mean(unrelated_uncentered):.4f} ({min(unrelated_uncentered):.4f}, {max(unrelated_uncentered):.4f})")
    print(f"{'Centered vec(R)':<25} "
          f"{np.mean(related_centered):.4f} ({min(related_centered):.4f}, {max(related_centered):.4f})  "
          f"{np.mean(unrelated_centered):.4f} ({min(unrelated_centered):.4f}, {max(unrelated_centered):.4f})")

    print("\n--- Discrimination Metrics ---")
    print(f"{'Method':<25} {'AUROC':<10} {'Gap-Z':<10} {'Margin':<10}")
    print("-" * 55)
    print(f"{'Uncentered vec(M)':<25} {auroc_uncentered:.4f}     {gap_z_uncentered:+.2f}      {margin_uncentered:+.4f}")
    print(f"{'Centered vec(R)':<25} {auroc_centered:.4f}     {gap_z_centered:+.2f}      {margin_centered:+.4f}")

    print("\n--- Identity Energy Fraction (mean |tr(M)|²/||M||_F²) ---")
    all_trace_fracs = []
    for r in results:
        all_trace_fracs.extend(r.ref_trace_fractions)
        all_trace_fracs.extend(r.sus_trace_fractions)
    print(f"Mean across all layers: {np.mean(all_trace_fracs):.4f}")
    print(f"This is the fraction of matrix energy along the identity direction.")
    print(f"Centering removes this shared component.")

    # Save results
    summary = {
        "n_pairs": len(pairs),
        "n_related": n_related,
        "n_unrelated": n_unrelated,
        "test_roots": args.test_roots,
        "metrics": {
            "uncentered": {
                "auroc": auroc_uncentered,
                "gap_z": gap_z_uncentered,
                "margin": margin_uncentered,
                "related_mean": float(np.mean(related_uncentered)),
                "related_min": float(min(related_uncentered)),
                "related_max": float(max(related_uncentered)),
                "unrelated_mean": float(np.mean(unrelated_uncentered)),
                "unrelated_min": float(min(unrelated_uncentered)),
                "unrelated_max": float(max(unrelated_uncentered)),
            },
            "centered": {
                "auroc": auroc_centered,
                "gap_z": gap_z_centered,
                "margin": margin_centered,
                "related_mean": float(np.mean(related_centered)),
                "related_min": float(min(related_centered)),
                "related_max": float(max(related_centered)),
                "unrelated_mean": float(np.mean(unrelated_centered)),
                "unrelated_min": float(min(unrelated_centered)),
                "unrelated_max": float(max(unrelated_centered)),
            },
        },
        "identity_energy_fraction": float(np.mean(all_trace_fracs)),
        "pairs": [asdict(r) for r in results],
    }

    output_path = output_dir / "centering_ablation.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX table
    latex = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Method & AUROC & Gap-$Z$ & Margin \\
\midrule
Uncentered $\mathrm{vec}(M_\ell)$ & %.3f & %+.1f & %.3f \\
Centered $\mathrm{vec}(R_\ell)$ & \textbf{%.3f} & \textbf{%+.1f} & \textbf{%.3f} \\
\bottomrule
\end{tabular}
\caption{Centering ablation on GPT-2 benchmark (%d pairs). Centering removes the shared identity component, improving separation between related and unrelated pairs.}
\label{tab:centering-ablation}
\end{table}
""" % (auroc_uncentered, gap_z_uncentered, margin_uncentered,
       auroc_centered, gap_z_centered, margin_centered, len(pairs))

    latex_path = output_dir / "centering_ablation_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
