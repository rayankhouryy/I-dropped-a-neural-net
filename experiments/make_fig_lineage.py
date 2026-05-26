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


COLORS = {
    'descendant':              '#2E7D32',   # green
    'non_descendant':          '#C62828',   # red
    'finetune_same':           '#1565C0',
    'finetune_diff':           '#0277BD',
    'noise':                   '#00838F',
    'prune':                   '#558B2F',
    'quantize':                '#9E9D24',
    'independent_same_task':   '#D84315',
    'independent_diff_task':   '#E65100',
    'distilled_student':       '#AD1457',
    'random_init':             '#6A1B9A',
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
    """ROC: proposed lineage vs diag-only, raw-cos, frob baselines."""
    pairs = data['pairs']
    labels = np.array([1 if p['label'] == 'descendant' else 0 for p in pairs])
    score_metrics = [
        ('lineage',   r'Residual-signature $\mathcal{L}$ (proposed)', '#2E7D32', '-',  2.5),
        ('raw_cos',   r'Raw branch-product cosine',                   '#1565C0', '--', 1.5),
        ('frob_dist', r'$-$Frobenius distance',                       '#6A1B9A', ':',  1.5),
        ('diag_only', r'Diagonal-dominance only (baseline that should fail)',
                                                                      '#C62828', '-.', 1.5),
    ]
    for key, label, color, ls, lw in score_metrics:
        scores = [p.get(key, 0.0) for p in pairs]
        fpr, tpr, auroc = _roc_curve(scores, labels)
        ax.plot(fpr, tpr, label=f'{label}  (AUROC={auroc:.3f})',
                color=color, linestyle=ls, linewidth=lw)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('(b) ROC: descendant vs non-descendant')
    ax.legend(loc='lower right', fontsize=7)
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
        ax.scatter(xs, ys, color=COLORS.get(atk, 'gray'), s=45, alpha=0.85,
                    edgecolor='black', linewidth=0.5,
                    label=ATTACK_LABEL.get(atk, atk))
    # Independent controls for reference
    rows = [p for p in pairs
            if p['attack_type'] == 'independent_same_task'
            and 'utility' in p and p['utility'] is not None]
    if rows:
        xs = [p['utility'] for p in rows]
        ys = [p['lineage'] for p in rows]
        ax.scatter(xs, ys, color=COLORS['independent_same_task'], s=45,
                    marker='x', linewidth=1.2,
                    label=ATTACK_LABEL['independent_same_task'])

    ax.set_xlabel('Eval loss (lower = better preserved utility)')
    ax.set_ylabel(r'Lineage score $\mathcal{L}(\text{ref}, B)$')
    ax.set_title('(c) Lineage vs utility tradeoff')
    ax.legend(loc='best', fontsize=7)
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
    colors = [COLORS['descendant'] if per_attack[k]['label'] == 'descendant'
              else COLORS['non_descendant'] for k in order]
    ypos = np.arange(len(order))
    ax.barh(ypos, means, xerr=stds, color=colors, alpha=0.75,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r'Mean lineage score $\overline{\mathcal{L}}$')
    ax.set_title('(d) Mean lineage by attack/control type')
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')


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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_a_distributions(axes[0, 0], data)
    panel_b_roc(axes[0, 1], data)
    panel_c_utility(axes[1, 0], data)
    per_attack_bar(axes[1, 1], data)
    fig.suptitle('Model-level lineage detection from diagonal-dominance fingerprints (Issue #30)',
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


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
