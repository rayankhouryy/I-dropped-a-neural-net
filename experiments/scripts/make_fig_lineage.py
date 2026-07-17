"""Three-panel figure for Issue #30 model-level lineage detection.

Per @singh96aman's spec, produces the canonical figure:

  Panel A: Score distributions   -- descendant vs non-descendant histograms
  Panel B: ROC curves            -- proposed score vs baselines
  Panel C: Utility-fingerprint   -- lineage vs accuracy/utility scatter
                                    (shows fingerprint only collapses when
                                     model utility also collapses)

Reads results/lineage_phase1_mlp.json. Writes:
  figures/fig_lineage_score_distributions.{png,pdf}
  figures/fig_lineage_roc.{png,pdf}
  figures/fig_lineage_utility_tradeoff.{png,pdf}
  paper/figures/fig_lineage_three_panel.{png,pdf}   (combined version)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Colorblind-safe Okabe-Ito palette
COLORS = {
    'descendant':              '#0072B2',   # blue (was green)
    'non_descendant':          '#D55E00',   # orange (was red)
    'finetune_same':           '#0072B2',   # blue
    'finetune_diff':           '#56B4E9',   # sky blue
    'noise':                   '#009E73',   # teal
    'prune':                   '#F0E442',   # yellow
    'quantize':                '#CC79A7',   # pink
    'independent_same_task':   '#D55E00',   # orange
    'independent_diff_task':   '#E69F00',   # amber
    'distilled_student':       '#D55E00',   # orange
    'random_init':             '#999999',   # gray
}

# Distinct markers for panel (c) - improves accessibility
MARKERS = {
    'finetune_same':           'o',
    'finetune_diff':           's',
    'noise':                   '^',
    'prune':                   'D',
    'quantize':                'v',
    'independent_same_task':   'X',
}

ATTACK_LABEL = {
    'finetune_same':         'Fine-tune (same task)',
    'finetune_diff':         'Fine-tune (different task)',
    'noise':                 'Gaussian weight noise',
    'prune':                 'Magnitude pruning',
    'quantize':              'Quantization',
    'independent_same_task': 'Independent (same task)',
    'independent_diff_task': 'Independent (different task)',
    'distilled_student':     'Distilled student',
    'random_init':           'Random init',
}


def panel_a_distributions(ax, data):
    """Histogram of lineage scores, descendant vs non-descendant, by category."""
    pairs = data['pairs']
    descendants = [p['lineage'] for p in pairs if p['label'] == 'descendant']
    nondescendants = [p['lineage'] for p in pairs if p['label'] == 'non_descendant']

    bins = np.linspace(min(min(descendants), min(nondescendants)) - 0.05,
                       max(max(descendants), max(nondescendants)) + 0.05,
                       40)
    ax.hist(nondescendants, bins=bins, color=COLORS['non_descendant'],
            alpha=0.6, edgecolor='black', linewidth=0.4,
            label=f'Non-descendants (n={len(nondescendants)})')
    ax.hist(descendants, bins=bins, color=COLORS['descendant'],
            alpha=0.6, edgecolor='black', linewidth=0.4,
            label=f'Descendants (n={len(descendants)})')
    ax.set_xlabel(r'Lineage score $\mathcal{L}(A, B)$')
    ax.set_ylabel('Count')
    ax.set_title('(a) Score distributions')
    ax.legend(loc='upper center', fontsize=8)
    ax.grid(True, alpha=0.3)


def _roc_curve(scores, labels):
    """Compute ROC curve for sklearn-free environments."""
    order = np.argsort(-np.asarray(scores))
    sorted_labels = np.asarray(labels)[order]
    n_pos = int(sorted_labels.sum())
    n_neg = len(sorted_labels) - n_pos
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = np.concatenate([[0], tp / max(n_pos, 1), [1]])
    fpr = np.concatenate([[0], fp / max(n_neg, 1), [1]])
    # AUROC
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    auroc = float(trapz(tpr, fpr))
    return fpr, tpr, auroc


def panel_b_roc(ax, data):
    """ROC: proposed lineage vs diag-only baseline.

    Key insight: diagonal-dominance alone fails (AUROC=0.417) because it can't
    distinguish two trained residual models. The residual-signature score succeeds.
    Other methods (raw cosine, Frobenius) also reach AUROC=1.0 on this benchmark,
    showing the task is well-separated; the contribution is interpretability, not AUROC.
    """
    pairs = data['pairs']
    labels = np.array([1 if p['label'] == 'descendant' else 0 for p in pairs])
    # Focus on key comparison: proposed vs baseline that should fail
    score_metrics = [
        ('lineage',   r'Residual-signature $\mathcal{L}$ (ours)', '#0072B2', '-',  2.5),
        ('diag_only', r'Diagonal-dominance only (expected to fail)', '#D55E00', '--', 2.0),
    ]
    for key, label, color, ls, lw in score_metrics:
        scores = [p.get(key, 0.0) for p in pairs]
        fpr, tpr, auroc = _roc_curve(scores, labels)
        ax.plot(fpr, tpr, label=f'{label}  (AUROC={auroc:.3f})',
                color=color, linestyle=ls, linewidth=lw)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('False positive rate', fontsize=10)
    ax.set_ylabel('True positive rate', fontsize=10)
    ax.set_title('(b) ROC: lineage detection', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def panel_c_utility(ax, data):
    """Utility (eval loss) vs lineage score, colored by attack type.

    The story: descendants stay high on lineage even when their utility
    degrades from the original; the fingerprint only collapses when the
    model is essentially destroyed.
    """
    pairs = data['pairs']
    # only show descendants here (the controlled-degradation rows)
    attacks_to_show = ['noise', 'prune', 'quantize',
                        'finetune_same', 'finetune_diff']
    for atk in attacks_to_show:
        rows = [p for p in pairs
                if p['attack_type'] == atk and 'utility' in p
                and p['utility'] is not None]
        if not rows:
            continue
        xs = [p['utility'] for p in rows]
        ys = [p['lineage'] for p in rows]
        marker = MARKERS.get(atk, 'o')
        ax.scatter(xs, ys, color=COLORS.get(atk, 'gray'), s=55, alpha=0.85,
                    edgecolor='black', linewidth=0.5, marker=marker,
                    label=ATTACK_LABEL.get(atk, atk))
    # Independent controls for reference
    rows = [p for p in pairs
            if p['attack_type'] == 'independent_same_task'
            and 'utility' in p and p['utility'] is not None]
    if rows:
        xs = [p['utility'] for p in rows]
        ys = [p['lineage'] for p in rows]
        ax.scatter(xs, ys, color=COLORS['independent_same_task'], s=55,
                    marker='X', linewidth=1.2,
                    label=ATTACK_LABEL['independent_same_task'])

    ax.set_xlabel('Eval loss (log scale, lower = better utility)', fontsize=10)
    ax.set_ylabel(r'Lineage score $\mathcal{L}(\text{ref}, B)$', fontsize=10)
    ax.set_title('(c) Lineage vs utility tradeoff', fontsize=11)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')


def per_attack_bar(ax, data):
    """Mean lineage score per attack type with error bars."""
    per_attack = data.get('per_attack_summary', {})
    if not per_attack:
        return
    order = sorted(per_attack.keys(),
                    key=lambda k: -per_attack[k]['mean_lineage'])
    means = [per_attack[k]['mean_lineage'] for k in order]
    stds  = [per_attack[k]['std_lineage']  for k in order]
    labels = [ATTACK_LABEL.get(k, k) for k in order]
    # Colorblind-safe: blue for descendants, orange for non-descendants
    colors = [COLORS['descendant'] if per_attack[k]['label'] == 'descendant'
              else COLORS['non_descendant'] for k in order]
    ypos = np.arange(len(order))
    ax.barh(ypos, means, xerr=stds, color=colors, alpha=0.75,
            edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r'Mean lineage score $\overline{\mathcal{L}}$', fontsize=10)
    ax.set_title('(d) Mean lineage by checkpoint type', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    # Add visual separator between descendants and non-descendants
    # Find the boundary
    desc_count = sum(1 for k in order if per_attack[k]['label'] == 'descendant')
    if 0 < desc_count < len(order):
        ax.axhline(desc_count - 0.5, color='#666666', linestyle='--', linewidth=1, alpha=0.7)


def make_panel_b_only_pdf(data, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    panel_b_roc(ax, data)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_panel_a_only_pdf(data, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    panel_a_distributions(ax, data)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_panel_c_only_pdf(data, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    panel_c_utility(ax, data)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_combined(data, out_path):
    """Create 1x2 panel figure: ROC (a) and Utility tradeoff (b) only.

    Panels removed as redundant with Figure 1(c).
    """
    # Large figure for full-width display, extra width for external legend
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Increase font sizes globally for this figure
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    })

    panel_b_roc_large(axes[0], data)
    panel_c_utility_large(axes[1], data)

    fig.tight_layout()
    # Make room for external legend on the right
    fig.subplots_adjust(wspace=0.3, right=0.85)
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def panel_b_roc_large(ax, data):
    """ROC panel with larger text and markers for full-width figure."""
    pairs = data['pairs']
    labels = np.array([1 if p['label'] == 'descendant' else 0 for p in pairs])
    score_metrics = [
        ('lineage',   r'Residual-signature (ours)', '#0072B2', '-',  3.0),
        ('diag_only', r'Diagonal-only (ablation)', '#D55E00', '--', 2.5),
    ]
    for key, label, color, ls, lw in score_metrics:
        scores = [p.get(key, 0.0) for p in pairs]
        fpr, tpr, auroc = _roc_curve(scores, labels)
        ax.plot(fpr, tpr, label=f'{label}  (AUROC={auroc:.3f})',
                color=color, linestyle=ls, linewidth=lw)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False positive rate', fontsize=13)
    ax.set_ylabel('True positive rate', fontsize=13)
    ax.set_title('(a) ROC: lineage detection', fontsize=14, fontweight='bold')
    # Legend in upper-left (dead space in ROC plots)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis='both', labelsize=11)


def panel_c_utility_large(ax, data):
    """Horizontal range plot showing score ranges per transformation type.

    Clear visual: most transformations preserve fingerprint near 1.0,
    only pruning shows wide range, independent baseline is clearly separated.
    """
    pairs = data['pairs']

    # Collect score ranges for each transformation type
    transform_data = {}
    categories = [
        ('independent_same_task', 'Independent\n(unrelated)', '#D55E00'),  # Orange - baseline
        ('prune',                 'Pruning',                   '#F0E442'),  # Yellow - wide range
        ('quantize',              'Quantization',              '#CC79A7'),  # Pink - preserved
        ('noise',                 'Noise',                     '#009E73'),  # Teal - preserved
        ('finetune_same',         'Fine-tune',                 '#0072B2'),  # Blue - preserved
    ]

    for atk, label, color in categories:
        rows = [p for p in pairs if p['attack_type'] == atk]
        if rows:
            scores = [p['lineage'] for p in rows]
            transform_data[label] = {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'scores': scores,
                'color': color
            }

    # Plot horizontal ranges
    y_positions = np.arange(len(transform_data))
    labels = list(transform_data.keys())

    for i, (label, d) in enumerate(transform_data.items()):
        # Solid line connecting min to max (the range) - solid for data, dashed reserved for thresholds
        ax.plot([d['min'], d['max']], [i, i], color=d['color'],
                linestyle='-', linewidth=3, alpha=0.7, zorder=1)
        # Endpoint markers (min and max) - small circles
        ax.scatter([d['min'], d['max']], [i, i], color=d['color'], s=60,
                   edgecolor='black', linewidth=1, marker='o', zorder=3)
        # Mean marker (diamond)
        ax.scatter([d['mean']], [i], color=d['color'], s=140, zorder=4,
                   edgecolor='black', linewidth=1.5, marker='D')
        # Individual points (jittered) - shows distribution shape
        jitter = np.random.uniform(-0.12, 0.12, len(d['scores']))
        ax.scatter(d['scores'], i + jitter, color=d['color'], s=18, alpha=0.35, zorder=2)

    # Reference line: decision threshold only (dashed = reference, solid = data)
    ax.axvline(0.5, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r'Lineage score $\mathcal{L}$', fontsize=13)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.5, len(transform_data) - 0.5)
    ax.set_title('(b) Score ranges by transformation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='both', labelsize=11)

    # Add annotation explaining pruning (pointing to the pruning row)
    ax.annotate('Wide range:\ndegrades with\nutility loss',
                xy=(0.6, 1), xytext=(0.3, 2.5),
                fontsize=9, color='#666666', ha='center',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1))


def make_branching_heatmap(data, out_path):
    """Lineage-matrix heatmap for the Phase 1b branching tree."""
    L_mat = np.asarray(data['lineage_matrix'], dtype=np.float64)
    all_ids = data['all_ids']

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(L_mat, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(all_ids)))
    ax.set_yticks(np.arange(len(all_ids)))
    ax.set_xticklabels(all_ids, rotation=0)
    ax.set_yticklabels(all_ids)
    ax.set_xlabel('Suspect')
    ax.set_ylabel('Reference')
    ax.set_title(r'Phase 1b: lineage matrix $\mathcal{L}(A, B)$ over branching tree')
    # Annotate each cell
    for i in range(len(all_ids)):
        for j in range(len(all_ids)):
            v = L_mat[i, j]
            txt_color = 'white' if v < 0.30 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_ancestry_chains(data, out_path):
    """For each leaf in {C4, C5}, plot the lineage scores in ranked order."""
    rq1 = data['rq1_ancestry_tracing']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, leaf in zip(axes, ['C4', 'C5']):
        ranking = rq1[leaf]
        names = [t[0] for t in ranking]
        scores = [t[1] for t in ranking]
        rels = [t[2] for t in ranking]
        colors_rel = {
            'ancestor_dist1': '#1B5E20',
            'ancestor_dist2': '#388E3C',
            'sibling':        '#66BB6A',
            'uncle':          '#9CCC65',
            'cousin':         '#C0CA33',
            'independent':    '#C62828',
        }
        bar_colors = [colors_rel.get(r, 'gray') for r in rels]
        ypos = np.arange(len(names))
        ax.barh(ypos, scores, color=bar_colors, alpha=0.85,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f'{n} ({rels[i]})' for i, n in enumerate(names)],
                           fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(r'Lineage score $\mathcal{L}(\,$' + leaf + r'$\,, \cdot)$')
        ax.set_title(f'Ancestry ranking for {leaf}')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1.0)
        for i, s in enumerate(scores):
            ax.text(s + 0.01, i, f'{s:+.3f}', va='center', fontsize=8)

    fig.suptitle('Phase 1b RQ1: lineage score ranks ancestors before collaterals before independents',
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_figures = Path('paper/figures')
    paper_figures.mkdir(parents=True, exist_ok=True)

    # Phase 1 figures (main run)
    phase1_path = Path('results/lineage_phase1_mlp.json')
    if phase1_path.exists():
        with open(phase1_path) as f:
            data = json.load(f)
        make_panel_a_only_pdf(data, figures_dir / 'fig_lineage_score_distributions.pdf')
        make_panel_b_only_pdf(data, figures_dir / 'fig_lineage_roc.pdf')
        make_panel_c_only_pdf(data, figures_dir / 'fig_lineage_utility_tradeoff.pdf')
        make_combined(data, paper_figures / 'fig_lineage_three_panel.pdf')
        print('Phase 1 figures written.')
    else:
        print(f'Skipping Phase 1 figures: {phase1_path} not found yet')

    # Phase 1b figures (branching tree)
    phase1b_path = Path('results/lineage_phase1b_branching.json')
    if phase1b_path.exists():
        with open(phase1b_path) as f:
            data = json.load(f)
        make_branching_heatmap(data, figures_dir / 'fig_lineage_branching_heatmap.pdf')
        make_ancestry_chains(data, figures_dir / 'fig_lineage_ancestry_chains.pdf')
        make_branching_heatmap(data, paper_figures / 'fig_lineage_branching_heatmap.pdf')
        make_ancestry_chains(data, paper_figures / 'fig_lineage_ancestry_chains.pdf')
        print('Phase 1b figures written.')
    else:
        print(f'Skipping Phase 1b figures: {phase1b_path} not found yet')


if __name__ == '__main__':
    main()
