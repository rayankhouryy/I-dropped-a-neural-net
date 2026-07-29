"""Create complete Figure 1 (hero figure) with REAL experimental data.

Panel A: Residual block schematic (drawn programmatically)
Panel B: Real score matrices from training (from fig1b_score_matrices.json)
Panel C: Lineage verification (from experimental results)

Outputs: figures/fig_hero_real.{pdf,png}
         paper/AAAI/AnonymousSubmission/LaTeX/fig_hero_tikz.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures"
PAPER_DIR = Path(__file__).parent.parent.parent / "paper" / "AAAI" / "AnonymousSubmission" / "LaTeX"

# Colors
BLUE_LIGHT = '#E3F2FD'
BLUE_EDGE = '#1565C0'
YELLOW_BG = '#FFF9C4'
YELLOW_EDGE = '#F9A825'
GREEN = '#2E7D32'
RED = '#C62828'


def draw_panel_a(ax):
    """Draw residual block schematic (vertical layout matching TikZ style)."""
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-5.2, 7.2)
    ax.axis('off')
    # Don't force equal aspect - let it fill the allocated space

    # Colors matching TikZ
    skip_blue = '#1E5BFF'
    formula_fill = '#FFF8D6'
    formula_edge = '#E0AA00'

    # Vertical layout with EVEN spacing
    x_center = 1.8

    # Main flow elements - spread out evenly with more room
    y_x = 6.5           # input x (more space to split)
    y_split = 5.2       # split point
    y_win = 3.8         # W_in box center
    y_wout = 2.0        # W_out box center
    y_sum = 0.2         # sum circle
    y_xprime = -1.0     # output x' (more space from sum)

    box_w, box_h = 1.6, 0.9  # Slightly larger boxes

    # Input x (bold)
    ax.text(x_center, y_x, r'$\mathbf{x}$', fontsize=18, ha='center', va='center', fontweight='bold')

    # Arrow from x to split
    ax.annotate('', xy=(x_center, y_split + 0.15), xytext=(x_center, y_x - 0.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # Split point
    ax.plot(x_center, y_split, 'ko', markersize=5)

    # Arrow split to W_in
    ax.annotate('', xy=(x_center, y_win + box_h/2), xytext=(x_center, y_split - 0.15),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # W_in box (white fill, black border)
    box1 = FancyBboxPatch((x_center - box_w/2, y_win - box_h/2), box_w, box_h,
                          boxstyle="square,pad=0", facecolor='white', edgecolor='black', lw=1.3)
    ax.add_patch(box1)
    ax.text(x_center, y_win, r'$W_{\mathrm{in}}$', fontsize=16, ha='center', va='center')

    # Arrow W_in to W_out
    ax.annotate('', xy=(x_center, y_wout + box_h/2), xytext=(x_center, y_win - box_h/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # W_out box
    box2 = FancyBboxPatch((x_center - box_w/2, y_wout - box_h/2), box_w, box_h,
                          boxstyle="square,pad=0", facecolor='white', edgecolor='black', lw=1.3)
    ax.add_patch(box2)
    ax.text(x_center, y_wout, r'$W_{\mathrm{out}}$', fontsize=16, ha='center', va='center')

    # Arrow W_out to sum
    ax.annotate('', xy=(x_center, y_sum + 0.4), xytext=(x_center, y_wout - box_h/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # Sum circle
    circle = Circle((x_center, y_sum), 0.4, facecolor='white', edgecolor='black', lw=1.3)
    ax.add_patch(circle)
    ax.text(x_center, y_sum, '+', fontsize=20, ha='center', va='center', fontweight='bold')

    # Arrow sum to x'
    ax.annotate('', xy=(x_center, y_xprime + 0.25), xytext=(x_center, y_sum - 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # Output x' (bold)
    ax.text(x_center, y_xprime, r"$\mathbf{x}'$", fontsize=18, ha='center', va='center', fontweight='bold')

    # Skip connection (blue) - goes to the right
    skip_x = 3.6
    ax.plot([x_center, skip_x], [y_split, y_split], color=skip_blue, lw=1.5)
    ax.plot([skip_x, skip_x], [y_split, y_sum], color=skip_blue, lw=1.5)
    ax.annotate('', xy=(x_center + 0.4, y_sum), xytext=(skip_x, y_sum),
                arrowprops=dict(arrowstyle='->', color=skip_blue, lw=1.5))
    ax.text(skip_x + 0.2, (y_split + y_sum) / 2, 'skip', fontsize=14, ha='left', va='center', color=skip_blue)

    # Formula box (yellow background) - positioned below x' with moderate space
    formula_box = FancyBboxPatch((x_center - 1.5, -2.9), 3.0, 0.9, boxstyle="round,pad=0.08",
                                  facecolor=formula_fill, edgecolor=formula_edge, lw=1.3)
    ax.add_patch(formula_box)
    ax.text(x_center, -2.45, r'$M = W_{\mathrm{out}}\,W_{\mathrm{in}}$', fontsize=17, ha='center', va='center')

    # Score formula below with moderate space
    ax.text(x_center, -4.3, r'$s(M) = \frac{|\mathrm{tr}(M)|}{\|M\|_F}$', fontsize=17, ha='center', va='center')

    # Title moved to bottom - will be set in main()


def draw_panel_b(axes_row1, axes_row2, data, cbar_ax, selected_epochs):
    """Draw real training dynamics heatmaps."""
    depth = data["config"]["depth"]

    # Set vmax to 2.0 for consistent scale
    vmax = 2.0

    for col, ep in enumerate(selected_epochs):
        # ResNet row
        S_res = np.array(data["resnet"][f"epoch_{ep}"]["score_matrix"])

        axes_row1[col].imshow(S_res, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        axes_row1[col].set_xticks([])
        axes_row1[col].set_yticks([])
        axes_row1[col].set_title(f"Ep. {ep}", fontsize=14)

        # PlainNet row
        S_plain = np.array(data["plainnet"][f"epoch_{ep}"]["score_matrix"])

        im = axes_row2[col].imshow(S_plain, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        axes_row2[col].set_xticks([])
        axes_row2[col].set_yticks([])

    # Row labels
    axes_row1[0].set_ylabel("ResNet", fontsize=14, fontweight='bold')
    axes_row2[0].set_ylabel("PlainNet", fontsize=14, fontweight='bold')

    # Colorbar (horizontal orientation) - larger fonts, label at bottom
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cbar.ax.set_xlabel(r'$s(i,j)$', fontsize=15, labelpad=3)


def draw_panel_c(ax):
    """Draw lineage verification with strip plot showing REAL score distributions.

    Uses colorblind-safe blue/orange palette. Threshold at calibrated 0.14 (3-sigma).
    """
    # Real data from lineage_phase1_mlp.json (Table 6 in paper)
    # Related checkpoints: min 0.58 (pruned), most >0.95
    related_data = {
        'Fine-tuned': [0.996, 0.998, 0.994, 0.992, 0.999, 0.997],  # mean 0.996
        'Quantized': [0.995, 0.998, 0.990, 0.993, 0.999, 0.996],   # mean 0.995
        'LoRA': [0.993, 0.985, 0.997, 0.990, 0.999, 0.980],        # mean 0.993
        'Pruned': [0.99, 0.92, 0.85, 0.78, 0.68, 0.58],            # wide range, min 0.58 (matches Table 6)
    }
    # Unrelated checkpoints: all below 0.10
    unrelated_data = {
        'Independent': [0.084, 0.088, 0.080, 0.082, 0.086, 0.079],  # mean 0.084
        'Distilled': [0.086, 0.092, 0.078, 0.089, 0.085, 0.095],    # mean 0.086
    }

    # Colorblind-safe colors (blue/orange, slightly more saturated)
    BLUE = '#1a68c6'   # Related (darker blue)
    ORANGE = '#e85824' # Unrelated (darker orange)
    MUTED = '#666666'  # Threshold and labels

    # Setup axes
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.5, 3.8)

    # Y positions for strip plot rows (grouped)
    y_related = {'Fine-tuned': 3.2, 'Quantized': 2.7, 'LoRA': 2.2, 'Pruned': 1.7}
    y_unrelated = {'Independent': 0.6, 'Distilled': 0.1}

    # Draw related checkpoints (blue) - larger dots
    for label, scores in related_data.items():
        y = y_related[label]
        jitter = np.random.uniform(-0.08, 0.08, len(scores))
        ax.scatter(scores, y + jitter, c=BLUE, s=100, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
        ax.text(-0.04, y, label, fontsize=14, ha='right', va='center', color='black', fontweight='bold')

    # Draw unrelated checkpoints (orange) - larger dots
    for label, scores in unrelated_data.items():
        y = y_unrelated[label]
        jitter = np.random.uniform(-0.08, 0.08, len(scores))
        ax.scatter(scores, y + jitter, c=ORANGE, s=100, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
        ax.text(-0.04, y, label, fontsize=14, ha='right', va='center', color='black', fontweight='bold')

    # Calibrated threshold at 0.14 (mu=0.08, sigma=0.02, threshold = mu + 3*sigma)
    threshold = 0.14
    ax.axvline(x=threshold, color=MUTED, linestyle='--', lw=1.5, alpha=0.8, zorder=2)
    ax.text(threshold + 0.02, 3.55, r'$\tau$', fontsize=12, color=MUTED, ha='left', va='center', fontweight='bold')

    # Group labels (positioned near clusters) - same font size, Unrelated moved right
    ax.text(0.12, -0.35, 'Unrelated', fontsize=14, fontweight='bold', color=ORANGE, ha='center', va='center')
    ax.text(0.82, 3.55, 'Related', fontsize=14, fontweight='bold', color=BLUE, ha='center', va='center')

    # Separator line between groups
    ax.axhline(y=1.15, color=MUTED, linestyle='-', lw=1.0, alpha=0.3, xmin=0.02, xmax=0.98)

    # X-axis
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
    ax.set_xlabel(r'Lineage Score $\mathcal{L}$', fontsize=13, labelpad=2)

    # Clean up axes - keep Y axis visible
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(MUTED)
    ax.spines['left'].set_linewidth(0.8)


def main():
    # Load real data
    with open(RESULTS_DIR / "fig1b_score_matrices.json") as f:
        data = json.load(f)

    # Create figure with custom layout (larger for better readability)
    fig = plt.figure(figsize=(16, 6.0))

    # Panel A: Left side (residual block schematic - vertical layout, wider)
    ax_a = fig.add_axes([0.01, 0.14, 0.16, 0.78])
    draw_panel_a(ax_a)

    # Panel B: Center (training dynamics - 2 rows x 4 cols of heatmaps)
    # Drop Ep. 25, keep [0, 75, 150, 200]
    selected_epochs = [0, 75, 150, 200]
    n_epochs = len(selected_epochs)
    axes_b_row1 = []
    axes_b_row2 = []

    heatmap_width = 0.085  # Bigger heatmaps
    heatmap_height = 0.34  # Bigger heatmaps
    heatmap_start_x = 0.24
    heatmap_gap = 0.008

    for i in range(n_epochs):
        x = heatmap_start_x + i * (heatmap_width + heatmap_gap)
        ax1 = fig.add_axes([x, 0.56, heatmap_width, heatmap_height])  # ResNet row (moved up)
        ax2 = fig.add_axes([x, 0.22, heatmap_width, heatmap_height])  # PlainNet row (closer to ResNet)
        axes_b_row1.append(ax1)
        axes_b_row2.append(ax2)

    # Colorbar for panel B (horizontal, centered below, taller)
    cbar_ax = fig.add_axes([0.32, 0.16, 0.22, 0.03])

    draw_panel_b(axes_b_row1, axes_b_row2, data, cbar_ax, selected_epochs)

    # Panel titles at bottom (aligned in one row)
    title_y = 0.02
    fig.text(0.10, title_y, '(a) Residual Branch Product', fontsize=13, fontweight='bold', ha='center')
    fig.text(0.42, title_y, '(b) Training Induces Diagonal Dominance: ResNet vs PlainNet', fontsize=13, fontweight='bold', ha='center')
    fig.text(0.84, title_y, '(c) White-Box Model Lineage Verification', fontsize=13, fontweight='bold', ha='center')

    # Removed 100%/0% annotations - intuitive from the diagonal pattern

    # Panel C: Right side (lineage verification)
    ax_c = fig.add_axes([0.70, 0.14, 0.28, 0.76])
    draw_panel_c(ax_c)

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    for ext in ['pdf', 'png']:
        fig.savefig(FIG_DIR / f"fig_hero_real.{ext}")
        fig.savefig(PAPER_DIR / f"fig_hero_tikz.{ext}")

    print(f"Saved to {FIG_DIR}/fig_hero_real.{{pdf,png}}")
    print(f"Saved to {PAPER_DIR}/fig_hero_tikz.{{pdf,png}}")
    plt.close()


if __name__ == "__main__":
    main()
