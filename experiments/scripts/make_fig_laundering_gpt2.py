#!/usr/bin/env python3
"""Generate figures and LaTeX tables for GPT-2 laundering experiment.

Outputs:
    1. AUROC bar chart by condition and method
    2. Invariance check visualization
    3. Score distributions
    4. LaTeX tables for paper

Usage:
    python make_fig_laundering_gpt2.py --results-dir results/laundering_gpt2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Try to import matplotlib; skip if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping figure generation")


# ---------------------------------------------------------------- color schemes

COLORS = {
    "centered_residual_signature": "#2ecc71",  # green - our method
    "raw_weight_cosine": "#e74c3c",            # red - collapses
    "raw_aligned_frobenius": "#e67e22",        # orange - collapses
    "singular_value_distance": "#9b59b6",      # purple - may be invariant
    "rebasin_frobenius": "#3498db",            # blue - recovery
    "rebasin_scale_frobenius": "#1abc9c",      # teal - recovery
}

METHOD_LABELS = {
    "centered_residual_signature": "Ours (Centered Residual)",
    "raw_weight_cosine": "Raw Weight Cosine",
    "raw_aligned_frobenius": "Aligned Frobenius",
    "singular_value_distance": "SVD Distance",
    "rebasin_frobenius": "Re-Basin",
    "rebasin_scale_frobenius": "Re-Basin + Scale",
}

CONDITION_ORDER = ["NONE", "P-SUSPECT", "P-BOTH", "P+FT"]


# ---------------------------------------------------------------- figure generation

def plot_auroc_by_condition(
    summary: Dict[str, Any],
    output_path: Path,
):
    """Bar chart of AUROC by condition and method."""
    if not HAS_MATPLOTLIB:
        return

    methods = [
        "centered_residual_signature",
        "raw_weight_cosine",
        "raw_aligned_frobenius",
        "singular_value_distance",
        "rebasin_frobenius",
        "rebasin_scale_frobenius",
    ]

    conditions = [c for c in CONDITION_ORDER if c in summary]
    n_methods = len(methods)
    n_conditions = len(conditions)

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.12
    x = np.arange(n_conditions)

    for i, method in enumerate(methods):
        aurocs = []
        for cond in conditions:
            if method in summary[cond]:
                aurocs.append(summary[cond][method]["auroc"])
            else:
                aurocs.append(0)

        offset = (i - n_methods/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, aurocs, bar_width,
                      label=METHOD_LABELS.get(method, method),
                      color=COLORS.get(method, "#95a5a6"))

    ax.set_xlabel("Laundering Condition", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Lineage Detection Under Hidden-Unit Permutation", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_invariance_check(
    invariance_checks: List[Dict],
    output_path: Path,
):
    """Visualize branch product invariance (cosine should be 1.0)."""
    if not HAS_MATPLOTLIB:
        return

    if not invariance_checks:
        print("No invariance checks to plot")
        return

    # Extract per-block cosines
    all_cosines = []
    for check in invariance_checks:
        if "per_block_cosine" in check:
            all_cosines.extend(check["per_block_cosine"])

    if not all_cosines:
        print("No cosine data in invariance checks")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(all_cosines, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect invariance')

    ax.set_xlabel("Cosine Similarity (M vs M')", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Branch Product Invariance Under Permutation", fontsize=14)
    ax.legend()

    # Add text annotation
    min_cos = min(all_cosines)
    max_cos = max(all_cosines)
    ax.text(0.02, 0.98, f"Min: {min_cos:.10f}\nMax: {max_cos:.10f}",
            transform=ax.transAxes, verticalalignment='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_distributions(
    results_dir: Path,
    output_path: Path,
):
    """Plot score distributions for descendants vs non-descendants."""
    if not HAS_MATPLOTLIB:
        return

    import csv

    scores_path = results_dir / "scores_by_pair.csv"
    if not scores_path.exists():
        print(f"Scores file not found: {scores_path}")
        return

    # Read scores
    with open(scores_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No score data")
        return

    # Get conditions
    conditions = sorted(set(r["condition"] for r in rows))
    conditions = [c for c in CONDITION_ORDER if c in conditions]

    fig, axes = plt.subplots(1, len(conditions), figsize=(4*len(conditions), 4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    method = "centered_residual_signature"

    for ax, cond in zip(axes, conditions):
        cond_rows = [r for r in rows if r["condition"] == cond]

        desc_scores = [float(r[method]) for r in cond_rows if r["label"] == "descendant"]
        nondesc_scores = [float(r[method]) for r in cond_rows if r["label"] == "non_descendant"]

        if desc_scores:
            ax.hist(desc_scores, bins=20, alpha=0.6, label="Descendant", color="#2ecc71")
        if nondesc_scores:
            ax.hist(nondesc_scores, bins=20, alpha=0.6, label="Non-descendant", color="#e74c3c")

        ax.set_xlabel("Lineage Score", fontsize=10)
        ax.set_title(cond, fontsize=12)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle("Score Distributions: Centered Residual Signature", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------- LaTeX tables

def generate_main_table(summary: Dict[str, Any], output_path: Path):
    """Generate LaTeX table for main laundering results."""
    methods = [
        ("centered_residual_signature", "Centered Residual (Ours)"),
        ("raw_weight_cosine", "Raw Weight Cosine"),
        ("raw_aligned_frobenius", "Aligned Frobenius"),
        ("singular_value_distance", "SVD Distance"),
        ("rebasin_frobenius", "Re-Basin"),
        ("rebasin_scale_frobenius", "Re-Basin + Scale"),
    ]

    conditions = [c for c in CONDITION_ORDER if c in summary]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Lineage detection AUROC under hidden-unit permutation. Our method remains stable while raw-weight baselines collapse.}",
        r"\label{tab:laundering}",
        r"\begin{tabular}{l" + "c" * len(conditions) + "}",
        r"\toprule",
        r"Method & " + " & ".join(conditions) + r" \\",
        r"\midrule",
    ]

    for method_key, method_name in methods:
        row = method_name
        for cond in conditions:
            if cond in summary and method_key in summary[cond]:
                auroc = summary[cond][method_key]["auroc"]
                if np.isnan(auroc):
                    row += " & --"
                else:
                    # Bold our method
                    if method_key == "centered_residual_signature":
                        row += f" & \\textbf{{{auroc:.3f}}}"
                    else:
                        row += f" & {auroc:.3f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def generate_ablation_table(summary: Dict[str, Any], output_path: Path):
    """Generate LaTeX table for ablation study."""
    if "NONE" not in summary:
        print("No NONE condition for ablation table")
        return

    methods = [
        ("raw_weight_cosine", "A. Raw Weight Cosine"),
        ("raw_branch_product_cosine", "B. Raw Branch Product"),
        ("centered_branch_product_cosine", "C. Centered Branch Product"),
        ("centered_with_gating", "D. + Trace Gating"),
        ("centered_residual_signature", "E. + Block Alignment (Full)"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study showing the contribution of each component to lineage discrimination (NONE condition).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & AUROC & $\Delta$ vs A \\",
        r"\midrule",
    ]

    base_auroc = None
    for method_key, method_name in methods:
        if method_key in summary["NONE"]:
            auroc = summary["NONE"][method_key]["auroc"]
            if base_auroc is None:
                base_auroc = auroc
                delta = "--"
            else:
                delta = f"+{auroc - base_auroc:.3f}" if auroc > base_auroc else f"{auroc - base_auroc:.3f}"

            if np.isnan(auroc):
                lines.append(f"{method_name} & -- & -- \\\\")
            else:
                lines.append(f"{method_name} & {auroc:.3f} & {delta} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Generate figures for GPT-2 laundering experiment")
    parser.add_argument("--results-dir", default="results/laundering_gpt2",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for figures (default: results-dir/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Load invariance checks
    invariance_path = results_dir / "invariance_checks.json"
    invariance_checks = []
    if invariance_path.exists():
        with open(invariance_path, "r") as f:
            invariance_checks = json.load(f)

    # Generate figures
    if HAS_MATPLOTLIB:
        plot_auroc_by_condition(summary, output_dir / "auroc_by_condition.pdf")
        plot_auroc_by_condition(summary, output_dir / "auroc_by_condition.png")
        plot_invariance_check(invariance_checks, output_dir / "invariance_check.pdf")
        plot_invariance_check(invariance_checks, output_dir / "invariance_check.png")
        plot_score_distributions(results_dir, output_dir / "score_distributions.pdf")
        plot_score_distributions(results_dir, output_dir / "score_distributions.png")

    # Generate LaTeX tables
    generate_main_table(summary, output_dir / "laundering_table.tex")
    generate_ablation_table(summary, output_dir / "ablation_table.tex")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
