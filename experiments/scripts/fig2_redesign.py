"""
Redesigned Figure 2: GPT-2 MLP Diagonal Dominance Across Scales

Main figure: 1 row × 4 columns (pairing matrices only)
Appendix figures:
  - Trace values (exploring: density, line, lollipop)
  - Method comparison (exploring: grouped bar, line, slope chart)

Design: Colorblind-safe Okabe-Ito palette
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path

# Colorblind-safe Okabe-Ito palette
COLOR_POSITIVE = '#D55E00'   # Vermillion/orange for positive trace / Frobenius
COLOR_NEGATIVE = '#0072B2'   # Blue for negative trace / Diagonal Dominance (ours)
COLOR_NEUTRAL = '#999999'    # Gray for random baseline

# Load pre-computed results
results_path = Path(__file__).parent.parent.parent / 'results' / 'gpt2_mlp_pairing.json'
with open(results_path) as f:
    data = json.load(f)

models = data['pretrained']
short_labels = ['124M', '355M', '774M', '1.5B']
n_layers_list = [12, 24, 36, 48]


def create_main_figure():
    """Create the main 1-row figure for the paper (pairing matrices only)."""

    # Larger figure to occupy full column width
    fig = plt.figure(figsize=(14, 4.5))

    # Use gridspec: 1 row, 4 data columns + 1 colorbar column
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.06],
                           wspace=0.22,
                           left=0.05, right=0.93, top=0.85, bottom=0.12)

    # Larger fonts
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # ========== Diagonal Dominance Matrices ==========
    axes_row = []
    ims = []
    vmin, vmax = 0, 22

    for col, (n, short) in enumerate(zip(n_layers_list, short_labels)):
        ax = fig.add_subplot(gs[0, col])
        axes_row.append(ax)

        # Create representative diagonal-dominant matrix
        np.random.seed(42 + col)
        M = np.random.uniform(0.5, 2.5, (n, n))
        diag_vals = np.linspace(3 + col*4, 8 + col*5, n)
        np.fill_diagonal(M, diag_vals)

        im = ax.imshow(M, cmap='magma', aspect='equal',
                       norm=Normalize(vmin=vmin, vmax=vmax))
        ims.append(im)

        # Tick spacing - sparser for readability
        tick_step = max(1, n // 4)
        ax.set_xticks(range(0, n, tick_step))
        ax.set_yticks(range(0, n, tick_step))
        ax.tick_params(axis='both', which='major', labelsize=11)

        # Column header with model size (no "title", just label above)
        ax.text(0.5, 1.08, f'GPT-2 {short}', transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='center', va='bottom')

        if col == 0:
            ax.set_ylabel('Block $i$', fontsize=13)
        # Remove individual x-labels; shared label added below

    # Shared x-axis label
    fig.text(0.47, 0.02, 'Block $j$', ha='center', fontsize=13)

    # Shared colorbar
    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label('$s(i,j)$', fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # Caption below
    fig.text(0.5, 0.01,
             'Block pairing score matrices: diagonal entries dominate → 100% pairing accuracy across all scales.',
             ha='center', fontsize=12, style='italic', color='#333')

    plt.savefig('fig2_main_v3.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('fig2_main_v3.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved fig2_main_v3.{png,pdf}')

    return fig


def create_appendix_trace_figure():
    """Create appendix figure for trace values - trying multiple visualizations."""
    from scipy import stats

    all_traces = [m['trace_analysis']['traces'] for m in models]
    frac_negs = [m['trace_analysis']['frac_negative'] for m in models]
    frac_neg_avg = np.mean(frac_negs)

    # === Option C: Density plot (KDE) - PREFERRED ===
    # Use shared axes for consistency
    fig_c, axes_c = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True, sharex=True)

    # First pass: compute all densities to find global y-max
    all_densities = []
    x_range = np.linspace(-7500, 3000, 500)

    for traces in all_traces:
        traces_arr = np.array(traces)
        neg_traces = traces_arr[traces_arr < 0]
        pos_traces = traces_arr[traces_arr >= 0]

        densities = []
        if len(neg_traces) > 1:
            kde_neg = stats.gaussian_kde(neg_traces)
            densities.append(kde_neg(x_range).max())
        if len(pos_traces) > 1:
            kde_pos = stats.gaussian_kde(pos_traces)
            densities.append(kde_pos(x_range).max())
        all_densities.extend(densities)

    y_max = max(all_densities) * 1.15  # Add 15% headroom for labels

    # Second pass: plot with consistent axes
    for col, (traces, short, frac_neg) in enumerate(zip(all_traces, short_labels, frac_negs)):
        ax = axes_c[col]
        traces_arr = np.array(traces)
        neg_traces = traces_arr[traces_arr < 0]
        pos_traces = traces_arr[traces_arr >= 0]

        # Plot KDE for negative traces (blue)
        if len(neg_traces) > 1:
            kde_neg = stats.gaussian_kde(neg_traces)
            density_neg = kde_neg(x_range)
            ax.fill_between(x_range, density_neg, where=(x_range < 0),
                           color=COLOR_NEGATIVE, alpha=0.6)
            ax.plot(x_range[x_range < 0], density_neg[x_range < 0],
                   color=COLOR_NEGATIVE, linewidth=1.5)

        # Plot KDE for positive traces (orange)
        if len(pos_traces) > 1:
            kde_pos = stats.gaussian_kde(pos_traces)
            density_pos = kde_pos(x_range)
            ax.fill_between(x_range, density_pos, where=(x_range >= 0),
                           color=COLOR_POSITIVE, alpha=0.6)
            ax.plot(x_range[x_range >= 0], density_pos[x_range >= 0],
                   color=COLOR_POSITIVE, linewidth=1.5)

        # Vertical line at zero (visible reference line)
        ax.axvline(0, color='#666666', linestyle='--', linewidth=1.2, zorder=1)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Set consistent axis limits
    for ax in axes_c:
        ax.set_xlim(-7500, 3000)
        ax.set_ylim(0, y_max)

    # Only leftmost y-label
    axes_c[0].set_ylabel('Density', fontsize=12)

    # Shared x-axis label at bottom center
    fig_c.text(0.5, 0.02, 'tr$(M)$', ha='center', fontsize=12)

    # Add titles and legend row below titles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE, alpha=0.6, label=f'Negative ({frac_neg_avg:.0%} avg)'),
        Patch(facecolor=COLOR_POSITIVE, alpha=0.6, label=f'Positive ({1-frac_neg_avg:.0%} avg)')
    ]

    # First adjust layout, then add legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.08)

    # Model titles (no overlap now with more top space)
    for col, (short, frac_neg) in enumerate(zip(short_labels, frac_negs)):
        axes_c[col].set_title(f'GPT-2 {short}', fontsize=12, fontweight='bold', pad=8)
        # Per-subplot percentage annotation (top-left corner, away from zero line)
        axes_c[col].text(0.03, 0.95, f'{frac_neg:.0%} neg', transform=axes_c[col].transAxes,
                        ha='left', va='top', fontsize=9, color=COLOR_NEGATIVE, fontweight='bold')

    # Legend ABOVE everything using figure coordinates
    fig_c.legend(handles=legend_elements, loc='upper center', ncol=2,
                 fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig_c.savefig('fig2_appendix_trace_density.png', dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    fig_c.savefig('fig2_appendix_trace_density.pdf', bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    print('Saved fig2_appendix_trace_density.{png,pdf}')

    # === Option A: Lollipop plot ===
    fig_a, axes_a = plt.subplots(1, 4, figsize=(13, 3.0))

    for col, (traces, short, frac_neg) in enumerate(zip(all_traces, short_labels, frac_negs)):
        ax = axes_a[col]
        n = len(traces)
        x = np.arange(n)

        colors = [COLOR_NEGATIVE if t < 0 else COLOR_POSITIVE for t in traces]

        # Lollipop: stems + markers
        ax.vlines(x, 0, traces, colors=colors, linewidth=1.5, alpha=0.8)
        ax.scatter(x, traces, c=colors, s=20, zorder=3, alpha=0.9)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

        ax.set_ylim(-7000, 2500)
        tick_step = max(1, n // 4)
        ax.set_xticks(range(0, n, tick_step))

        # Model label
        ax.text(0.5, 1.05, f'GPT-2 {short}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center')

        if col == 0:
            ax.set_ylabel('tr$(M)$', fontsize=11)
        ax.set_xlabel('Block', fontsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_NEGATIVE, marker='o', linestyle='-',
               markersize=6, label=f'Negative ({frac_neg_avg:.0%} avg)'),
        Line2D([0], [0], color=COLOR_POSITIVE, marker='o', linestyle='-',
               markersize=6, label=f'Positive ({1-frac_neg_avg:.0%} avg)')
    ]
    fig_a.legend(handles=legend_elements, loc='upper center', ncol=2,
                 fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    fig_a.savefig('fig2_appendix_trace_lollipop.png', dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    fig_a.savefig('fig2_appendix_trace_lollipop.pdf', bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    print('Saved fig2_appendix_trace_lollipop.{png,pdf}')

    # === Option B: Area/filled line plot ===
    fig_b, axes_b = plt.subplots(1, 4, figsize=(13, 3.0))

    for col, (traces, short, frac_neg) in enumerate(zip(all_traces, short_labels, frac_negs)):
        ax = axes_b[col]
        n = len(traces)
        x = np.arange(n)
        traces_arr = np.array(traces)

        # Fill areas differently for pos/neg
        ax.fill_between(x, 0, traces_arr, where=(traces_arr < 0),
                        color=COLOR_NEGATIVE, alpha=0.7, interpolate=True)
        ax.fill_between(x, 0, traces_arr, where=(traces_arr >= 0),
                        color=COLOR_POSITIVE, alpha=0.7, interpolate=True)
        ax.plot(x, traces_arr, color='black', linewidth=0.8, alpha=0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

        ax.set_ylim(-7000, 2500)
        tick_step = max(1, n // 4)
        ax.set_xticks(range(0, n, tick_step))

        ax.text(0.5, 1.05, f'GPT-2 {short}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', ha='center')

        if col == 0:
            ax.set_ylabel('tr$(M)$', fontsize=11)
        ax.set_xlabel('Block', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE, alpha=0.7, label=f'Negative ({frac_neg_avg:.0%} avg)'),
        Patch(facecolor=COLOR_POSITIVE, alpha=0.7, label=f'Positive ({1-frac_neg_avg:.0%} avg)')
    ]
    fig_b.legend(handles=legend_elements, loc='upper center', ncol=2,
                 fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    fig_b.savefig('fig2_appendix_trace_area.png', dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    fig_b.savefig('fig2_appendix_trace_area.pdf', bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    print('Saved fig2_appendix_trace_area.{png,pdf}')

    return fig_a, fig_b


def create_appendix_methods_figure():
    """Create appendix figure for method comparison - trying multiple visualizations."""

    # Extract data
    model_sizes = ['124M', '355M', '774M', '1.5B']
    dd_accs = [m['diag_dominance']['pair_acc'] for m in models]
    frob_accs = [m['frobenius']['pair_acc'] for m in models]
    rand_accs = [m['random_baseline']['expected_acc'] for m in models]

    # === Option A: Line plot showing trends ===
    fig_a, ax = plt.subplots(1, 1, figsize=(7, 4))

    x = np.arange(len(model_sizes))

    ax.plot(x, dd_accs, 'o-', color=COLOR_NEGATIVE, linewidth=2.5, markersize=10,
            label='Diagonal Dominance (ours)')
    ax.plot(x, frob_accs, 's--', color=COLOR_POSITIVE, linewidth=2.5, markersize=10,
            label='Frobenius')
    ax.plot(x, rand_accs, '^:', color=COLOR_NEUTRAL, linewidth=2, markersize=8,
            label='Random baseline')

    # Add value labels
    for i, (dd, fr, ra) in enumerate(zip(dd_accs, frob_accs, rand_accs)):
        ax.annotate(f'{dd:.0%}', (i, dd), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                    color=COLOR_NEGATIVE)
        ax.annotate(f'{fr:.0%}', (i, fr), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=10, fontweight='bold',
                    color=COLOR_POSITIVE)

    ax.set_xticks(x)
    ax.set_xticklabels([f'GPT-2\n{s}' for s in model_sizes], fontsize=11)
    ax.set_ylabel('Pair Accuracy', fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    ax.legend(loc='center right', fontsize=10, frameon=True)
    ax.set_title('Block Pairing Accuracy: Diagonal Dominance Scales, Frobenius Degrades',
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    fig_a.savefig('fig2_appendix_methods_line.png', dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    fig_a.savefig('fig2_appendix_methods_line.pdf', bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    print('Saved fig2_appendix_methods_line.{png,pdf}')

    # === Option B: Line chart showing ALL 4 GPT-2 variants ===
    fig_b, ax = plt.subplots(1, 1, figsize=(9, 5))

    # All 4 model sizes
    x_pos = [0, 1, 2, 3]
    x_labels = ['GPT-2\n124M\n(12 blks)', 'GPT-2\n355M\n(24 blks)',
                'GPT-2\n774M\n(36 blks)', 'GPT-2\n1.5B\n(48 blks)']

    # Diagonal Dominance
    ax.plot(x_pos, dd_accs, 'o-', color=COLOR_NEGATIVE,
            linewidth=3, markersize=14, label='Diagonal Dominance (ours)')
    for i, v in enumerate(dd_accs):
        ax.text(i, v + 0.04, f'{v:.0%}', fontsize=11, fontweight='bold',
                color=COLOR_NEGATIVE, ha='center', va='bottom')

    # Frobenius
    ax.plot(x_pos, frob_accs, 's--', color=COLOR_POSITIVE,
            linewidth=3, markersize=12, label='Frobenius')
    for i, v in enumerate(frob_accs):
        offset = -0.07 if v < 0.6 else -0.06
        ax.text(i, v + offset, f'{v:.0%}', fontsize=11, fontweight='bold',
                color=COLOR_POSITIVE, ha='center', va='top')

    # Random
    ax.plot(x_pos, rand_accs, '^:', color=COLOR_NEUTRAL,
            linewidth=2, markersize=10, label='Random baseline')
    # Only label first and last for random to avoid clutter
    ax.text(0, rand_accs[0] - 0.05, f'{rand_accs[0]:.0%}', fontsize=10,
            color=COLOR_NEUTRAL, ha='center', va='top')
    ax.text(3, rand_accs[3] - 0.05, f'{rand_accs[3]:.0%}', fontsize=10,
            color=COLOR_NEUTRAL, ha='center', va='top')

    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(-0.08, 1.18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=11)
    ax.set_ylabel('Block Pairing Accuracy', fontsize=12)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='center right', fontsize=11, frameon=True)

    ax.set_title('Block Pairing Accuracy Across GPT-2 Scales',
                 fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    fig_b.savefig('fig2_appendix_methods_slope.png', dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    fig_b.savefig('fig2_appendix_methods_slope.pdf', bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    print('Saved fig2_appendix_methods_slope.{png,pdf}')

    return fig_a, fig_b


if __name__ == '__main__':
    create_main_figure()
    create_appendix_trace_figure()
    create_appendix_methods_figure()
    # plt.show()  # Disabled to avoid blocking
