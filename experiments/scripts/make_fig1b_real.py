"""Generate Figure 1b from real experimental data.

Reads: results/fig1b_score_matrices.json
Outputs: figures/fig1b_real.{pdf,png}
         paper/AAAI/AnonymousSubmission/LaTeX/fig1b_real.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures"
PAPER_DIR = Path(__file__).parent.parent.parent / "paper" / "AAAI" / "AnonymousSubmission" / "LaTeX"

def main():
    # Load real data
    with open(RESULTS_DIR / "fig1b_score_matrices.json") as f:
        data = json.load(f)

    epochs = data["config"]["checkpoint_epochs"]
    depth = data["config"]["depth"]

    # Create figure: 2 rows (ResNet, PlainNet) x 5 columns (epochs)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))

    # Color normalization - use max from final trained ResNet
    vmax = max(
        np.max(data["resnet"]["epoch_200"]["score_matrix"]),
        np.max(data["plainnet"]["epoch_200"]["score_matrix"])
    )
    vmin = 0

    # Plot each epoch
    for col, ep in enumerate(epochs):
        # ResNet row
        S_res = np.array(data["resnet"][f"epoch_{ep}"]["score_matrix"])
        acc_res = data["resnet"][f"epoch_{ep}"]["pair_accuracy"]

        im = axes[0, col].imshow(S_res, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel("ResNet", fontsize=9, fontweight='bold')

        # PlainNet row
        S_plain = np.array(data["plainnet"][f"epoch_{ep}"]["score_matrix"])
        acc_plain = data["plainnet"][f"epoch_{ep}"]["pair_accuracy"]

        axes[1, col].imshow(S_plain, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel("PlainNet", fontsize=9, fontweight='bold')

        # Epoch label and accuracy
        axes[0, col].set_title(f"Epoch {ep}", fontsize=9)
        axes[1, col].set_xlabel(f"{acc_plain:.0%}", fontsize=8, color='#C62828' if acc_plain < 0.5 else '#2E7D32')

        # Add accuracy below ResNet
        axes[0, col].set_xlabel(f"{acc_res:.0%}", fontsize=8, color='#2E7D32' if acc_res > 0.5 else '#C62828')

    # Add final accuracy annotations
    axes[0, -1].annotate("100%", xy=(1.05, 0.5), xycoords='axes fraction',
                         fontsize=10, fontweight='bold', color='#2E7D32', va='center')
    axes[1, -1].annotate("0%", xy=(1.05, 0.5), xycoords='axes fraction',
                         fontsize=10, fontweight='bold', color='#C62828', va='center')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$s(i,j)$', fontsize=9)

    # Title
    fig.suptitle("Score Matrix $s(i,j)$ During Training: Diagonal Emerges Only with Skip Connections",
                 fontsize=10, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    for ext in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f"fig1b_real.{ext}")
        fig.savefig(PAPER_DIR / f"fig1b_real.{ext}")

    print(f"Saved to {FIG_DIR}/fig1b_real.{{pdf,png}}")
    print(f"Saved to {PAPER_DIR}/fig1b_real.{{pdf,png}}")
    plt.close()


if __name__ == "__main__":
    main()
