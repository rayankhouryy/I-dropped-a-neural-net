"""Generate figures for Section 6: Applications.

Creates publication-quality figures for the paper:
  fig_training_qa_summary.pdf - Training QA early warning results
  fig_compression_audit_summary.pdf - Compression audit E correlation
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def fig_training_qa_summary(output_path):
    """Create summary figure for training QA case study."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    pathologies = [
        "Healthy\nBaseline",
        "LR High\n(1e-2)",
        "LR Low\n(1e-6)",
        "No Skip\n(Plain)",
        "High WD\n(λ=1)",
        "Small Init\n(σ=.02)",
    ]

    final_acc = [1.0, 1.0, 0.06, 0.03, 0.24, 0.22]
    ep10_acc = [0.08, 1.0, 0.06, 0.06, 0.03, 0.32]
    neg_trace = [1.0, 1.0, 0.44, 0.47, 0.0, 0.31]

    colors = ['#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c']

    x = np.arange(len(pathologies))
    width = 0.7

    ax1 = axes[0]
    bars1 = ax1.bar(x, final_acc, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_ylabel("Final Pair Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pathologies, fontsize=7)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("(a) Final Pair Accuracy", fontweight='bold')
    ax1.text(5.5, 0.92, '90%', fontsize=7, color='gray')

    ax2 = axes[1]
    bars2 = ax2.bar(x, ep10_acc, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel("Pair Accuracy at Epoch 10")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pathologies, fontsize=7)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("(b) Early Warning (Epoch 10)", fontweight='bold')
    ax2.text(5.5, 0.52, '50%', fontsize=7, color='red')

    ax3 = axes[2]
    bars3 = ax3.bar(x, neg_trace, width, color=colors, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_ylabel("Fraction Negative Trace")
    ax3.set_xticks(x)
    ax3.set_xticklabels(pathologies, fontsize=7)
    ax3.set_ylim(0, 1.15)
    ax3.set_title("(c) Dynamical Isometry", fontweight='bold')
    ax3.text(5.5, 0.52, '50%', fontsize=7, color='gray')

    pass_patch = mpatches.Patch(color='#2ecc71', label='Healthy')
    fail_patch = mpatches.Patch(color='#e74c3c', label='Pathological')
    fig.legend(handles=[pass_patch, fail_patch], loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=300)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_compression_audit_summary(output_path):
    """Create summary figure for compression audit case study."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1 = axes[0]

    methods = ['FP16', 'INT8', 'INT4', 'Prune\n30%', 'Prune\n50%', 'Prune\n70%', 'FT\n50ep']
    corrs = [1.000, 1.000, 0.975, 0.982, 0.912, 0.752, 0.999]

    x = np.arange(len(methods))
    colors = ['#2ecc71' if c > 0.9 else '#27ae60' if c > 0.7 else '#f39c12' for c in corrs]

    bars = ax1.bar(x, corrs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.90, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axhline(y=0.70, color='#27ae60', linestyle=':', linewidth=1, alpha=0.7)

    ax1.set_ylabel("E Matrix Correlation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("(a) Compression Preserves Fingerprint", fontweight='bold')

    for i, (bar, corr) in enumerate(zip(bars, corrs)):
        ax1.text(bar.get_x() + bar.get_width()/2, corr + 0.02, f"{corr:.2f}",
                ha='center', va='bottom', fontsize=7)

    ax2 = axes[1]

    categories = ['Quantization\n(FP16-INT4)', 'Pruning\n(30-70%)', 'Fine-tune\n(50 ep)',
                  'Distillation', 'Independent\nTraining']
    means = [0.992, 0.882, 0.999, -0.007, 0.010]
    colors2 = ['#2ecc71', '#27ae60', '#2ecc71', '#e74c3c', '#e74c3c']

    x2 = np.arange(len(categories))
    bars2 = ax2.bar(x2, means, color=colors2, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0.30, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax2.set_ylabel("Mean E Correlation")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=7)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title("(b) Derivation Detection", fontweight='bold')

    ax2.text(4.5, 0.32, 'Threshold', fontsize=7, color='gray')

    for i, (bar, mean) in enumerate(zip(bars2, means)):
        y_pos = max(mean + 0.03, 0.05)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f"{mean:.2f}",
                ha='center', va='bottom', fontsize=7)

    derived_patch = mpatches.Patch(color='#2ecc71', label='Derived (>0.70)')
    not_patch = mpatches.Patch(color='#e74c3c', label='Not derived (<0.30)')
    ax2.legend(handles=[derived_patch, not_patch], loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=300)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_zkp_protocol(output_path):
    """Create protocol diagram for ZKP case study."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.add_patch(mpatches.FancyBboxPatch((0.3, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                                          facecolor='#3498db', edgecolor='black', linewidth=1.5))
    ax.text(1.3, 4.25, "PROVER\n(Owner)", ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

    ax.add_patch(mpatches.FancyBboxPatch((7.7, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                                          facecolor='#e74c3c', edgecolor='black', linewidth=1.5))
    ax.text(8.7, 4.25, "VERIFIER\n(Challenger)", ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

    ax.add_patch(mpatches.FancyBboxPatch((4, 4.5), 2, 1, boxstyle="round,pad=0.1",
                                          facecolor='#2ecc71', edgecolor='black', linewidth=1.5))
    ax.text(5, 5, "REGISTRY", ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

    ax.annotate("", xy=(4, 4.75), xytext=(2.3, 4.25),
                arrowprops=dict(arrowstyle="->", color='#3498db', lw=1.5))
    ax.text(3.1, 4.8, "1. REGISTER", ha='center', fontsize=8, fontweight='bold', color='#3498db')
    ax.text(3.1, 4.4, "Hash(E, r)", ha='center', fontsize=7, style='italic')

    ax.annotate("", xy=(2.3, 3.2), xytext=(7.7, 3.2),
                arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=1.5))
    ax.text(5, 3.5, "2. CHALLENGE", ha='center', fontsize=8, fontweight='bold', color='#e74c3c')
    ax.text(5, 2.9, '"Reveal blocks [k₁, k₂, ...]"', ha='center', fontsize=7, style='italic')

    ax.annotate("", xy=(7.7, 2.2), xytext=(2.3, 2.2),
                arrowprops=dict(arrowstyle="->", color='#3498db', lw=1.5))
    ax.text(5, 2.5, "3. RESPONSE", ha='center', fontsize=8, fontweight='bold', color='#3498db')
    ax.text(5, 1.9, "W_in[k], W_out[k], r[k]", ha='center', fontsize=7, style='italic')

    ax.add_patch(mpatches.FancyBboxPatch((6.5, 0.3), 3.2, 1.2, boxstyle="round,pad=0.1",
                                          facecolor='#f39c12', edgecolor='black', linewidth=1.5))
    ax.text(8.1, 0.9, "4. VERIFY\nE_hash? eps? DD>0.5?",
            ha='center', va='center', fontsize=7, fontweight='bold')

    ax.text(0.5, 0.9, "Commitment:\nE = M + εI\nHash(E || r)",
            fontsize=7, va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=300)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def main():
    figures_dir = Path(__file__).parent / "../paper/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating application figures for paper...")
    fig_training_qa_summary(figures_dir / "fig_training_qa_summary")
    fig_compression_audit_summary(figures_dir / "fig_compression_audit_summary")
    fig_zkp_protocol(figures_dir / "fig_zkp_protocol")

    print("\nDone!")


if __name__ == "__main__":
    main()
