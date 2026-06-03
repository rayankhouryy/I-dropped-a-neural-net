"""Figure for AAAI Section 6 adaptive evasion experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = Path('results/lineage_section6_attacks.json')
    if not path.exists():
        print(f"missing {path}")
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel A: invariance summary ---
    ax = axes[0]
    bars = []
    labels = []
    colors = []
    bars.append(1.0); labels.append('self-similarity\n$\\mathcal{L}(A,A)$'); colors.append('#2E7D32')
    e1 = data['exp1_orthogonal_rotation']
    bars.append(e1['mean_lineage']); labels.append('Orth. rotation\n(weight-product invariant)');  colors.append('#1565C0')
    e2 = data['exp2_hidden_permutation']
    bars.append(e2['mean_lineage']); labels.append('Hidden permutation\n(function-preserving)');   colors.append('#0277BD')
    # Best utility-preserving attack from exp3
    exp3 = data['exp3_gradient_attack']['runs']
    util_pres = [r for r in exp3 if r['utility_drift'] < 1e-3]
    best_util = min((r['final_lineage'] for r in util_pres), default=None)
    if best_util is not None:
        bars.append(best_util); labels.append('Gradient attack\n(utility-preserving)'); colors.append('#9E9D24')
    bars.append(data['baseline_independent_lineage']); labels.append('Independent\n(non-descendant baseline)'); colors.append('#C62828')

    xpos = np.arange(len(bars))
    ax.bar(xpos, bars, color=colors, alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'Lineage score $\mathcal{L}(A, A_{\mathrm{attack}})$')
    ax.set_title('(a) Attack-vs-baseline lineage scores')
    ax.set_ylim(-0.05, 1.10)
    for i, v in enumerate(bars):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel B: gradient attack Pareto curve ---
    ax = axes[1]
    exp3_runs = data['exp3_gradient_attack']['runs']
    util_drifts = [r['utility_drift'] for r in exp3_runs]
    L_finals    = [r['final_lineage']  for r in exp3_runs]
    lambdas     = [r['lambda_utility'] for r in exp3_runs]
    ax.scatter(util_drifts, L_finals, c=range(len(exp3_runs)),
                cmap='viridis', s=80, edgecolor='black', linewidth=0.5,
                zorder=3)
    for i, (x, y, lam) in enumerate(zip(util_drifts, L_finals, lambdas)):
        ax.annotate(f'$\\lambda$={lam:g}', (x, y),
                     xytext=(8, 4), textcoords='offset points',
                     fontsize=8)

    # Decision regions
    ax.axhline(data['baseline_independent_lineage'], color='#C62828',
                linestyle='--', linewidth=1, alpha=0.7,
                label=f"Independent baseline ($\\mathcal{{L}}\\approx{data['baseline_independent_lineage']:.3f}$)")
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                label='Notional descendant threshold ($\\mathcal{L} = 0.5$)')
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_xlabel('Utility drift  $\\|f(A) - f(A + \\Delta)\\|^2$')
    ax.set_ylabel(r'Lineage score $\mathcal{L}(A, A + \Delta)$')
    ax.set_title('(b) Gradient-attack Pareto curve')
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Adaptive evasion: function-preserving reparameterizations leave the fingerprint exactly invariant;\n'
        'gradient attacks must destroy utility to suppress lineage.',
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_pdf = Path('figures/fig_lineage_attacks.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.with_suffix('.png'), dpi=200, bbox_inches='tight')
    Path('paper/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(Path('paper/figures/fig_lineage_attacks.pdf'),
                 bbox_inches='tight')
    fig.savefig(Path('paper/figures/fig_lineage_attacks.png'),
                 dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_pdf} + paper/figures/")


if __name__ == '__main__':
    main()
