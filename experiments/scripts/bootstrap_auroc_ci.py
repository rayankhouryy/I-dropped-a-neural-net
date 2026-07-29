"""Compute bootstrap 95% CIs for AUROC values across benchmarks.

Loads per-pair scores from existing result files and outputs CIs in a format
ready for paper updates. Uses 10,000 bootstrap resamples by default.

Output format: AUROC (lower–upper), e.g., 1.000 (0.998–1.000)
"""
import csv
import json
from pathlib import Path

import numpy as np


def auroc(pos, neg):
    """Compute AUROC via Mann-Whitney U statistic."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += (p > neg).sum() + 0.5 * (p == neg).sum()
    return wins / (len(pos) * len(neg))


def bootstrap_auroc_ci(pos, neg, n_bootstrap=10000, seed=42):
    """Compute AUROC and 95% CI via bootstrap resampling."""
    np.random.seed(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)

    emp_auroc = auroc(pos, neg)

    boot_aurocs = np.array([
        auroc(
            np.random.choice(pos, size=len(pos), replace=True),
            np.random.choice(neg, size=len(neg), replace=True),
        )
        for _ in range(n_bootstrap)
    ])

    ci_lower = float(np.percentile(boot_aurocs, 2.5))
    ci_upper = float(np.percentile(boot_aurocs, 97.5))

    return {
        "auroc": float(emp_auroc),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_bootstrap": n_bootstrap,
    }


def format_ci(result, decimals=3):
    """Format AUROC with CI for paper: 1.000 (0.998–1.000)"""
    fmt = f"{{:.{decimals}f}}"
    auroc_str = fmt.format(result["auroc"])
    lower_str = fmt.format(result["ci_lower"])
    upper_str = fmt.format(result["ci_upper"])
    return f"{auroc_str} ({lower_str}–{upper_str})"


def load_52pair_mlp(path):
    """Load 52-pair MLP baseline benchmark (lineage_baselines_mlp.json)."""
    with open(path) as f:
        data = json.load(f)

    results = {}
    methods = [
        "diagonal_dominance", "aligned_frobenius", "singular_value_dist",
        "weight_cosine", "cka", "svcca", "ipguard_regr"
    ]

    for method in methods:
        pos, neg = [], []
        for pair in data["pairs"]:
            score = pair["scores"].get(method)
            if score is None:
                continue
            if pair["label"] == 1:
                pos.append(score)
            else:
                neg.append(score)
        if pos and neg:
            results[method] = bootstrap_auroc_ci(pos, neg)

    return results


def load_159pair_mlp(path):
    """Load 159-pair MLP benchmark (lineage_phase1_mlp.json)."""
    with open(path) as f:
        data = json.load(f)

    pos, neg = [], []
    for pair in data["pairs"]:
        score = pair.get("lineage")
        if score is None:
            continue
        if pair["label"] == "descendant":
            pos.append(score)
        else:
            neg.append(score)

    return {"lineage": bootstrap_auroc_ci(pos, neg)}


def load_resnet18_cifar(path):
    """Load ResNet-18 CIFAR benchmark (lineage_phase2_resnet18_cifar.json)."""
    with open(path) as f:
        data = json.load(f)

    pos, neg = [], []
    for record in data["records"]:
        score = record.get("score")
        if score is None:
            continue
        if record["label"] == "descendant":
            pos.append(score)
        else:
            neg.append(score)

    return {"lineage": bootstrap_auroc_ci(pos, neg)}


def load_harder_bench(path):
    """Load harder benchmark with graft/merge (lineage_harder_bench.json)."""
    with open(path) as f:
        data = json.load(f)

    methods = [
        "diagonal_dominance", "aligned_frobenius", "singular_value_dist",
        "weight_cosine", "cka", "svcca", "ipguard_regr"
    ]

    results = {"graft": {}, "merge": {}, "all": {}}

    for regime in ["graft", "merge"]:
        for method in methods:
            pos, neg = [], []
            for pair in data["pairs"]:
                if pair["regime"] != regime:
                    continue
                score = pair["scores"].get(method)
                if score is None:
                    continue
                if pair["label"] == 1:
                    pos.append(score)
                else:
                    neg.append(score)
            if pos and neg:
                results[regime][method] = bootstrap_auroc_ci(pos, neg)

    for method in methods:
        pos, neg = [], []
        for pair in data["pairs"]:
            score = pair["scores"].get(method)
            if score is None:
                continue
            if pair["label"] == 1:
                pos.append(score)
            else:
                neg.append(score)
        if pos and neg:
            results["all"][method] = bootstrap_auroc_ci(pos, neg)

    return results


def main():
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results"

    print("=" * 72)
    print("Bootstrap 95% CIs for AUROC (10,000 resamples)")
    print("=" * 72)
    print()

    # 52-pair MLP benchmark
    print("52-pair MLP Benchmark (Table 5 baseline)")
    print("-" * 40)
    mlp52 = load_52pair_mlp(results_dir / "lineage_baselines_mlp.json")
    for method, result in mlp52.items():
        print(f"  {method:25s}: {format_ci(result)} (n={result['n_pos']}+/{result['n_neg']}-)")
    print()

    # 159-pair MLP benchmark
    print("159-pair MLP Benchmark (Table 4)")
    print("-" * 40)
    mlp159 = load_159pair_mlp(results_dir / "lineage_phase1_mlp.json")
    for method, result in mlp159.items():
        print(f"  {method:25s}: {format_ci(result)} (n={result['n_pos']}+/{result['n_neg']}-)")
    print()

    # ResNet-18 CIFAR benchmark
    print("ResNet-18 CIFAR-10 Benchmark")
    print("-" * 40)
    resnet = load_resnet18_cifar(results_dir / "lineage_phase2_resnet18_cifar.json")
    for method, result in resnet.items():
        print(f"  {method:25s}: {format_ci(result)} (n={result['n_pos']}+/{result['n_neg']}-)")
    print()

    # Harder benchmark (graft/merge)
    print("Harder Benchmark (Graft/Merge)")
    print("-" * 40)
    harder = load_harder_bench(results_dir / "lineage_harder_bench.json")

    print("  Graft regime:")
    for method, result in harder["graft"].items():
        print(f"    {method:23s}: {format_ci(result)}")
    print()
    print("  Merge regime:")
    for method, result in harder["merge"].items():
        print(f"    {method:23s}: {format_ci(result)}")
    print()
    print("  All (combined):")
    for method, result in harder["all"].items():
        print(f"    {method:23s}: {format_ci(result)}")
    print()

    # Summary for paper
    print("=" * 72)
    print("SUMMARY FOR PAPER (copy-paste ready)")
    print("=" * 72)
    print()
    print("Main text AUROC mentions:")
    print(f"  159-pair MLP (lineage):    {format_ci(mlp159['lineage'])}")
    print(f"  ResNet-18 CIFAR (lineage): {format_ci(resnet['lineage'])}")
    print()
    print("52-pair MLP (Table 5 'none' column, our method = diagonal_dominance):")
    print(f"  diagonal_dominance:        {format_ci(mlp52['diagonal_dominance'])}")
    print()
    print("Harder bench (Appendix, diagonal_dominance):")
    print(f"  Graft:                     {format_ci(harder['graft']['diagonal_dominance'])}")
    print(f"  Merge:                     {format_ci(harder['merge']['diagonal_dominance'])}")


if __name__ == "__main__":
    main()
