"""Generate figures for Case Study 1: Training Quality Assurance.

Creates:
  fig_dd_emergence_trajectories.pdf - DD emergence over epochs for each pathology
  fig_dd_heatmaps_comparison.pdf - DD heatmaps at key epochs
  fig_early_warning_correlation.pdf - Early DD vs final quality correlation
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Matplotlib style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

PATHOLOGY_LABELS = {
    "healthy_baseline": "Healthy Baseline",
    "lr_too_low": "LR Too Low (1e-6)",
    "lr_too_high": "LR Too High (1e-2)",
    "no_skip": "No Skip (PlainNet)",
    "high_weight_decay": "High Weight Decay",
    "small_init": "Small Init (σ=0.02)",
}

PATHOLOGY_COLORS = {
    "healthy_baseline": "#2ecc71",  # green
    "lr_too_low": "#e74c3c",        # red
    "lr_too_high": "#3498db",       # blue
    "no_skip": "#9b59b6",           # purple
    "high_weight_decay": "#f39c12", # orange
    "small_init": "#1abc9c",        # teal
}


def load_data(case_study_dir):
    """Load experiment results."""
    csv_path = case_study_dir / "training_qa_results.csv"
    json_path = case_study_dir / "training_qa_summary.json"

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        summary = json.load(f)

    return df, summary


def fig_emergence_trajectories(df, output_path):
    """Create 2x3 grid showing DD emergence for each pathology."""
    pathologies = list(PATHOLOGY_LABELS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, pathology in enumerate(pathologies):
        ax = axes[idx]
        pdf = df[df["pathology"] == pathology]

        # Plot each seed
        for seed in pdf["seed"].unique():
            sdf = pdf[pdf["seed"] == seed].sort_values("epoch")
            ax.plot(sdf["epoch"], sdf["pair_acc"],
                    color=PATHOLOGY_COLORS[pathology], alpha=0.6, linewidth=1.5)

        # Mean line
        mean_df = pdf.groupby("epoch")["pair_acc"].mean().reset_index()
        ax.plot(mean_df["epoch"], mean_df["pair_acc"],
                color=PATHOLOGY_COLORS[pathology], linewidth=2.5, label="Mean")

        # Reference lines
        ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axvline(x=5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=10, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(PATHOLOGY_LABELS[pathology])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 200)

        if idx >= 3:
            ax.set_xlabel("Epoch")
        if idx % 3 == 0:
            ax.set_ylabel("Pair Accuracy")

    # Add annotations
    axes[0].annotate("90% threshold", xy=(150, 0.9), fontsize=8, color="gray")
    axes[0].annotate("50% threshold", xy=(150, 0.5), fontsize=8, color="gray")
    axes[0].annotate("Early warning\ncheckpoints", xy=(7, 0.1), fontsize=7, color="red", alpha=0.7)

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_early_warning_correlation(df, output_path):
    """Scatter plot: pair_acc at epoch 10 vs final eval_loss."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Plot 1: Epoch 10 pair_acc vs Final pair_acc
    ax1 = axes[0]
    for pathology in PATHOLOGY_LABELS:
        pdf = df[df["pathology"] == pathology]
        ep10 = pdf[pdf["epoch"] == 10]
        final = pdf[pdf["epoch"] == 200]

        if len(ep10) == 0 or len(final) == 0:
            continue

        # Match by seed
        for seed in ep10["seed"].unique():
            e10_row = ep10[ep10["seed"] == seed]
            f_row = final[final["seed"] == seed]
            if len(e10_row) == 0 or len(f_row) == 0:
                continue

            ax1.scatter(e10_row["pair_acc"].values[0], f_row["pair_acc"].values[0],
                       c=PATHOLOGY_COLORS[pathology], s=80, alpha=0.8,
                       label=PATHOLOGY_LABELS[pathology] if seed == 0 else "")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax1.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Pair Accuracy at Epoch 10")
    ax1.set_ylabel("Final Pair Accuracy (Epoch 200)")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Early Warning: Epoch 10 → Final")
    ax1.annotate("Warning\nthreshold", xy=(0.52, 0.1), fontsize=8, color="red", alpha=0.7)
    ax1.legend(loc="lower right", fontsize=8)

    # Plot 2: Epoch 5 pair_acc vs Final pair_acc
    ax2 = axes[1]
    for pathology in PATHOLOGY_LABELS:
        pdf = df[df["pathology"] == pathology]
        ep5 = pdf[pdf["epoch"] == 5]
        final = pdf[pdf["epoch"] == 200]

        if len(ep5) == 0 or len(final) == 0:
            continue

        for seed in ep5["seed"].unique():
            e5_row = ep5[ep5["seed"] == seed]
            f_row = final[final["seed"] == seed]
            if len(e5_row) == 0 or len(f_row) == 0:
                continue

            ax2.scatter(e5_row["pair_acc"].values[0], f_row["pair_acc"].values[0],
                       c=PATHOLOGY_COLORS[pathology], s=80, alpha=0.8)

    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Pair Accuracy at Epoch 5")
    ax2.set_ylabel("Final Pair Accuracy (Epoch 200)")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Early Warning: Epoch 5 → Final")

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_pathology_summary(df, summary, output_path):
    """Bar chart comparing final metrics across pathologies."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    pathologies = list(PATHOLOGY_LABELS.keys())
    x = np.arange(len(pathologies))
    width = 0.6

    # Plot 1: Final pair accuracy
    ax1 = axes[0]
    final_accs = []
    final_stds = []
    for p in pathologies:
        stats = summary["pathologies"][p]["final"]
        final_accs.append(stats["mean_pair_acc"])
        final_stds.append(stats["std_pair_acc"])

    bars = ax1.bar(x, final_accs, width, yerr=final_stds, capsize=3,
                   color=[PATHOLOGY_COLORS[p] for p in pathologies], alpha=0.8)
    ax1.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax1.set_ylabel("Final Pair Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels([PATHOLOGY_LABELS[p].replace(" ", "\n") for p in pathologies],
                        fontsize=8, rotation=0)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Final Pair Accuracy (Epoch 200)")
    ax1.annotate("90% threshold", xy=(4.5, 0.92), fontsize=8, color="red")

    # Plot 2: Epoch 10 pair accuracy (early warning)
    ax2 = axes[1]
    ep10_accs = []
    for p in pathologies:
        pdf = df[(df["pathology"] == p) & (df["epoch"] == 10)]
        ep10_accs.append(pdf["pair_acc"].mean() if len(pdf) > 0 else 0)

    bars = ax2.bar(x, ep10_accs, width,
                   color=[PATHOLOGY_COLORS[p] for p in pathologies], alpha=0.8)
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax2.set_ylabel("Pair Accuracy at Epoch 10")
    ax2.set_xticks(x)
    ax2.set_xticklabels([PATHOLOGY_LABELS[p].replace(" ", "\n") for p in pathologies],
                        fontsize=8, rotation=0)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Early Warning Signal (Epoch 10)")
    ax2.annotate("50% warning", xy=(4.5, 0.52), fontsize=8, color="red")

    # Plot 3: Emergence epoch
    ax3 = axes[2]
    emerge_epochs = []
    for p in pathologies:
        me = summary["pathologies"][p]["mean_emergence_epoch"]
        emerge_epochs.append(me if me is not None else 250)  # 250 = never

    bars = ax3.bar(x, emerge_epochs, width,
                   color=[PATHOLOGY_COLORS[p] for p in pathologies], alpha=0.8)
    ax3.axhline(y=10, color="green", linestyle="--", alpha=0.7, linewidth=1)
    ax3.set_ylabel("Emergence Epoch")
    ax3.set_xticks(x)
    ax3.set_xticklabels([PATHOLOGY_LABELS[p].replace(" ", "\n") for p in pathologies],
                        fontsize=8, rotation=0)
    ax3.set_ylim(0, 260)
    ax3.set_title("Epoch of DD Emergence (pair_acc > 90%)")
    ax3.annotate("Target: <10", xy=(4.5, 15), fontsize=8, color="green")

    # Mark "never" bars
    for i, e in enumerate(emerge_epochs):
        if e >= 250:
            ax3.annotate("Never", xy=(i, 255), ha="center", fontsize=7, color="red")

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_training_dynamics(df, output_path):
    """Show training loss and DD metrics over time for healthy vs pathological."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    pathologies_to_plot = ["healthy_baseline", "lr_too_low", "no_skip", "small_init"]

    # Top left: Training loss
    ax1 = axes[0, 0]
    for p in pathologies_to_plot:
        pdf = df[df["pathology"] == p]
        mean_df = pdf.groupby("epoch").agg({"train_loss": "mean"}).reset_index()
        ax1.semilogy(mean_df["epoch"], mean_df["train_loss"],
                     label=PATHOLOGY_LABELS[p], color=PATHOLOGY_COLORS[p], linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (log)")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 200)

    # Top right: Pair accuracy
    ax2 = axes[0, 1]
    for p in pathologies_to_plot:
        pdf = df[df["pathology"] == p]
        mean_df = pdf.groupby("epoch").agg({"pair_acc": "mean"}).reset_index()
        ax2.plot(mean_df["epoch"], mean_df["pair_acc"],
                 label=PATHOLOGY_LABELS[p], color=PATHOLOGY_COLORS[p], linewidth=2)
    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pair Accuracy")
    ax2.set_title("DD Fingerprint Emergence")
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-0.05, 1.05)

    # Bottom left: Pair separation
    ax3 = axes[1, 0]
    for p in pathologies_to_plot:
        pdf = df[df["pathology"] == p]
        mean_df = pdf.groupby("epoch").agg({"pair_sep": "mean"}).reset_index()
        ax3.plot(mean_df["epoch"], mean_df["pair_sep"],
                 label=PATHOLOGY_LABELS[p], color=PATHOLOGY_COLORS[p], linewidth=2)
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Pair Separation")
    ax3.set_title("Pair Separation (min correct - max incorrect)")
    ax3.set_xlim(0, 200)

    # Bottom right: Negative trace percentage
    ax4 = axes[1, 1]
    for p in pathologies_to_plot:
        pdf = df[df["pathology"] == p]
        mean_df = pdf.groupby("epoch").agg({"pct_negative_trace": "mean"}).reset_index()
        ax4.plot(mean_df["epoch"], mean_df["pct_negative_trace"],
                 label=PATHOLOGY_LABELS[p], color=PATHOLOGY_COLORS[p], linewidth=2)
    ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Fraction Negative Trace")
    ax4.set_title("Negative Trace (Dynamical Isometry)")
    ax4.set_xlim(0, 200)
    ax4.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def main():
    case_study_dir = Path(__file__).parent / "../case_studies/case_study_1"
    figures_dir = case_study_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, summary = load_data(case_study_dir)

    print("\nGenerating figures...")
    fig_emergence_trajectories(df, figures_dir / "fig_dd_emergence_trajectories")
    fig_early_warning_correlation(df, figures_dir / "fig_early_warning_correlation")
    fig_pathology_summary(df, summary, figures_dir / "fig_pathology_summary")
    fig_training_dynamics(df, figures_dir / "fig_training_dynamics")

    print("\nDone!")


if __name__ == "__main__":
    main()
