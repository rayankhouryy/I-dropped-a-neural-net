#!/usr/bin/env python3
"""Compute AUROC and Gap-Z for baseline methods on GPT-2 benchmark.

Produces results comparable to Table 1 (MLP baselines) for GPT-2.

Uses pre-computed branch products (Ms) from the benchmark pickle files.

Baselines computed (all from branch products, CPU only):
    1. Centered Residual Signature (ours)
    2. Weight Cosine (on branch products)
    3. Aligned Frobenius (on branch products)
    4. Singular Value Distance (on branch products)

Note: CKA, SVCCA, and IPGuard require forward passes through the actual
models, which are not saved in the benchmark. Those baselines are omitted.

Usage:
    cd experiments/scripts
    python compute_gpt2_baselines_auroc.py

Output:
    results/lineage_benchmark_gpt2_paper/baseline_auroc.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

script_dir = Path(__file__).parent.resolve()


def centered_residual_signature(
    Ms_A: List[np.ndarray], Ms_B: List[np.ndarray]
) -> float:
    """Our method: centered branch product cosine similarity."""
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
    return float(np.mean(cosines))


def weight_cosine(Ms_A: List[np.ndarray], Ms_B: List[np.ndarray]) -> float:
    """Cosine similarity of flattened branch products with Hungarian alignment."""
    L = len(Ms_A)
    fA = [M.astype(np.float64).ravel() for M in Ms_A]
    fB = [M.astype(np.float64).ravel() for M in Ms_B]
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            na = np.linalg.norm(fA[i]) + 1e-12
            nb = np.linalg.norm(fB[j]) + 1e-12
            D[i, j] = -float(fA[i] @ fB[j]) / (na * nb)
    row, col = linear_sum_assignment(D)
    return float(-D[row, col].mean())


def aligned_frobenius(Ms_A: List[np.ndarray], Ms_B: List[np.ndarray]) -> float:
    """Frobenius distance with Hungarian alignment (returned as similarity)."""
    L = len(Ms_A)
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            diff = Ms_A[i].astype(np.float64) - Ms_B[j].astype(np.float64)
            D[i, j] = np.linalg.norm(diff, ord='fro')
    row, col = linear_sum_assignment(D)
    mean_dist = float(D[row, col].mean())
    norms = [np.linalg.norm(M.astype(np.float64), ord='fro') for M in Ms_A]
    mean_norm = float(np.mean(norms))
    return -mean_dist / (mean_norm + 1e-12)


def singular_value_distance(
    Ms_A: List[np.ndarray], Ms_B: List[np.ndarray]
) -> float:
    """Wasserstein-1 between singular-value spectra (returned as similarity)."""
    L = len(Ms_A)
    sA = []
    for M in Ms_A:
        sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
        sA.append(np.sort(sv)[::-1])
    sB = []
    for M in Ms_B:
        sv = np.linalg.svd(M.astype(np.float64), compute_uv=False)
        sB.append(np.sort(sv)[::-1])

    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            k = min(len(sA[i]), len(sB[j]))
            D[i, j] = float(np.mean(np.abs(sA[i][:k] - sB[j][:k])))
    row, col = linear_sum_assignment(D)
    mean_dist = float(D[row, col].mean())
    mean_scale = float(np.mean([s[0] for s in sA]) + 1e-12)
    return -mean_dist / mean_scale


def compute_gap_z(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Compute Gap-Z: (mean_pos - mean_neg) / pooled_std."""
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    mean_pos, mean_neg = pos.mean(), neg.mean()
    std_pos, std_neg = pos.std(), neg.std()
    n_pos, n_neg = len(pos), len(neg)
    pooled_var = ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2)
    pooled_std = np.sqrt(pooled_var / (n_pos + n_neg - 2))
    return float((mean_pos - mean_neg) / (pooled_std + 1e-12))


def main():
    parser = argparse.ArgumentParser(
        description="Compute AUROC/Gap-Z for GPT-2 baselines"
    )
    parser.add_argument(
        "--results-dir",
        default="results/lineage_benchmark_gpt2_paper",
        help="Path to benchmark results directory"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / args.results_dir

    print(f"Results dir: {results_dir}")

    # Load benchmark results for split info
    with open(results_dir / "benchmark_results.json") as f:
        benchmark = json.load(f)

    # Load pre-computed data from pickle files
    print("\nLoading data from pickle files...")
    roots_path = results_dir / "phase1_roots.pkl"
    descendants_path = results_dir / "phase2_descendants.pkl"

    with open(roots_path, "rb") as f:
        roots_pkl = pickle.load(f)
    with open(descendants_path, "rb") as f:
        desc_pkl = pickle.load(f)

    # Parse roots: root_signatures is list of list of Ms per root
    roots_info = roots_pkl["roots_info"]
    root_signatures = roots_pkl["root_signatures"]

    roots = {}
    for info, Ms in zip(roots_info, root_signatures):
        idx = info["root_idx"]
        roots[idx] = {
            "Ms": Ms,
            "split": info.get("split", "unknown"),
        }
    print(f"Loaded {len(roots)} roots")

    # Parse descendants
    descendants = {}
    for d in desc_pkl["descendants"]:
        desc_id = d["id"]
        descendants[desc_id] = {
            "Ms": d["Ms"],
            "root_idx": d["root_idx"],
            "type": d["type"],
        }

    # Parse students (distilled)
    students = {}
    for s in desc_pkl["students"]:
        student_id = s["id"]
        students[student_id] = {
            "Ms": s["Ms"],
            "root_idx": s["root_idx"],
            "type": "distilled_student",
        }
    print(f"Loaded {len(descendants)} descendants, {len(students)} students")

    # Identify test-split roots
    test_roots = [r["root_idx"] for r in benchmark["roots"]
                  if r.get("split") == "test"]
    if not test_roots:
        test_roots = [5, 6, 7]
    print(f"\nTest roots: {test_roots}")

    # Define methods
    methods = {
        "Centered Res. Sig. (ours)": centered_residual_signature,
        "Weight Cosine": weight_cosine,
        "Aligned Frobenius": aligned_frobenius,
        "Singular Value Distance": singular_value_distance,
    }

    print("\n" + "=" * 70)
    print("Computing baseline scores...")
    print("=" * 70)

    all_scores = {m: {"positive": [], "negative": []} for m in methods}

    # Descendant pairs (positive)
    for desc_id, desc_data in descendants.items():
        root_idx = desc_data["root_idx"]
        if root_idx not in test_roots or root_idx not in roots:
            continue

        root_Ms = roots[root_idx]["Ms"]
        desc_Ms = desc_data["Ms"]
        print(f"  {desc_id} (descendant)...", end=" ", flush=True)

        for method_name, scorer in methods.items():
            score = scorer(root_Ms, desc_Ms)
            all_scores[method_name]["positive"].append(score)
        print("done")

    # Student pairs (negative - distilled)
    for student_id, student_data in students.items():
        root_idx = student_data["root_idx"]
        if root_idx not in test_roots or root_idx not in roots:
            continue

        root_Ms = roots[root_idx]["Ms"]
        student_Ms = student_data["Ms"]
        print(f"  {student_id} (distilled)...", end=" ", flush=True)

        for method_name, scorer in methods.items():
            score = scorer(root_Ms, student_Ms)
            all_scores[method_name]["negative"].append(score)
        print("done")

    # Independent pairs (negative - cross-root)
    for i in test_roots:
        for j in range(len(roots)):
            if i == j or i not in roots or j not in roots:
                continue

            print(f"  root_{i} vs root_{j} (independent)...", end=" ", flush=True)

            for method_name, scorer in methods.items():
                score = scorer(roots[i]["Ms"], roots[j]["Ms"])
                all_scores[method_name]["negative"].append(score)
            print("done")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}
    for method_name in methods:
        pos = all_scores[method_name]["positive"]
        neg = all_scores[method_name]["negative"]

        if not pos or not neg:
            print(f"  {method_name}: insufficient data")
            continue

        y_true = [1] * len(pos) + [0] * len(neg)
        y_score = pos + neg

        auroc = roc_auc_score(y_true, y_score)
        gap_z = compute_gap_z(pos, neg)

        results[method_name] = {
            "auroc": auroc,
            "gap_z": gap_z,
            "n_positive": len(pos),
            "n_negative": len(neg),
            "pos_mean": float(np.mean(pos)),
            "pos_std": float(np.std(pos)),
            "neg_mean": float(np.mean(neg)),
            "neg_std": float(np.std(neg)),
        }

        print(f"\n{method_name}:")
        print(f"  AUROC = {auroc:.3f}")
        print(f"  Gap-Z = {gap_z:+.1f}")
        print(f"  Positive: n={len(pos)}, mean={np.mean(pos):.4f}")
        print(f"  Negative: n={len(neg)}, mean={np.mean(neg):.4f}")

    print("\n" + "=" * 70)
    print("TABLE FOR PAPER (GPT-2 Baseline Comparison)")
    print("=" * 70)
    print(f"{'Method':<30} {'AUROC':>8} {'Gap-Z':>10}")
    print("-" * 50)
    for method_name, r in results.items():
        auroc_str = f"{r['auroc']:.3f}"
        gap_z_str = f"{r['gap_z']:+.1f}"
        print(f"{method_name:<30} {auroc_str:>8} {gap_z_str:>10}")

    output_path = results_dir / "baseline_auroc.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
