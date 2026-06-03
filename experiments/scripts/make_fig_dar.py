"""Build the DAR fingerprint visualization for #40 and the paper.

4-panel figure:
  (a) Negative-trace fraction (the headline ablation): standard 0.99 vs DAR 0.02
  (b) Fingerprint magnitude vs separation: bars showing diag-s correct/off for both archs
  (c) Lineage score distributions: histograms of descendant vs independent L,
      overlaid for both architectures
  (d) Per-attack descendant scores: grouped bars by attack family, min line, indep ceiling

Reads results/dar_fingerprint_v2.json, writes
  figures/fig_dar_fingerprint.pdf
  figures/fig_dar_fingerprint.png
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'results' / 'dar_fingerprint_v2.json') as f:
    R = json.load(f)

STD = R['results']['standard']
DAR = R['results']['dar']

# colour scheme: standard residual = blue, DAR = orange
COL_STD = '#2C7FB8'
COL_DAR = '#E66101'
COL_DESC = '#1A9850'
COL_INDEP = '#D73027'

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))


# ----- (a) Negative-trace fraction -----
ax = axes[0]
neg_std = STD['fingerprint']['neg_trace_frac']
neg_dar = DAR['fingerprint']['neg_trace_frac']
bars = ax.bar(['Standard\nresidual', 'DAR\nrouting'], [neg_std, neg_dar],
              color=[COL_STD, COL_DAR], width=0.55, edgecolor='black',
              linewidth=0.5)
for b, v in zip(bars, [neg_std, neg_dar]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f'{v:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_ylabel(r'Fraction of blocks with $\mathrm{tr}(W_\mathrm{out} W_\mathrm{in}) < 0$')
ax.set_title('(a) Negative-identity ablation')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, zorder=0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis='y', alpha=0.3)


# ----- (b) Fingerprint magnitude / separation -----
ax = axes[1]
labels = ['correct\npairs', 'off-diagonal\npairs']
std_vals = [STD['fingerprint']['mean_diag_s_correct'],
            STD['fingerprint']['mean_diag_s_off']]
dar_vals = [DAR['fingerprint']['mean_diag_s_correct'],
            DAR['fingerprint']['mean_diag_s_off']]
x = np.arange(len(labels))
w = 0.36
b1 = ax.bar(x - w/2, std_vals, w, color=COL_STD, label='Standard',
            edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, dar_vals, w, color=COL_DAR, label='DAR',
            edgecolor='black', linewidth=0.5)
for bars, vals in [(b1, std_vals), (b2, dar_vals)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f'{v:.2f}',
                ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel(r'Diagonal-dominance score $s(M) = |\mathrm{tr}(M)| / \|M\|_F$')
ax.set_title('(b) Trace concentration')
ax.legend(loc='upper right', framealpha=0.95)
ax.set_ylim(0, max(max(std_vals), max(dar_vals)) * 1.25)
sep_std = STD['fingerprint']['separation']
sep_dar = DAR['fingerprint']['separation']
ax.text(0.5, 0.97, f'Separation: std={sep_std:.2f}  DAR={sep_dar:.2f}',
        transform=ax.transAxes, ha='center', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='gray', linewidth=0.5))
ax.grid(axis='y', alpha=0.3)


# ----- (c) Lineage score distributions -----
ax = axes[2]
bins = np.linspace(-0.05, 1.05, 32)

std_desc = [d['score'] for d in STD['lineage']['desc_records']]
std_ind  = [d['score'] for d in STD['lineage']['indep_records']]
dar_desc = [d['score'] for d in DAR['lineage']['desc_records']]
dar_ind  = [d['score'] for d in DAR['lineage']['indep_records']]

ax.hist(std_ind,  bins=bins, color=COL_STD, alpha=0.55, label='Standard indep.',
         edgecolor='black', linewidth=0.4)
ax.hist(dar_ind,  bins=bins, color=COL_DAR, alpha=0.55, label='DAR indep.',
         edgecolor='black', linewidth=0.4)
ax.hist(std_desc, bins=bins, color=COL_STD, alpha=0.55, label='Standard desc.',
         hatch='///', edgecolor='black', linewidth=0.4)
ax.hist(dar_desc, bins=bins, color=COL_DAR, alpha=0.55, label='DAR desc.',
         hatch='///', edgecolor='black', linewidth=0.4)
ax.axvline(STD['tau_s'], color=COL_STD, linestyle='--', linewidth=1.0,
            label=fr"$\tau_s$ std={STD['tau_s']:.3f}")
ax.axvline(DAR['tau_s'], color=COL_DAR, linestyle='--', linewidth=1.0,
            label=fr"$\tau_s$ DAR={DAR['tau_s']:.3f}")
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel(r'Lineage score $\mathcal{L}(A, B)$')
ax.set_ylabel('Pair count')
ax.set_title('(c) Lineage score distributions')
ax.legend(loc='upper center', fontsize=7, ncol=2, framealpha=0.95)
ax.grid(axis='y', alpha=0.3)


# ----- (d) Per-attack descendant scores -----
ax = axes[3]
attacks = ['ft_same', 'ft_diff', 'noise', 'prune', 'quant']
display_names = ['ft_same', 'ft_diff', 'noise', 'prune', 'quant']

def attack_mean_min(records, atk):
    vals = [d['score'] for d in records if d['attack'] == atk]
    return (float(np.mean(vals)), float(np.min(vals))) if vals else (0.0, 0.0)

std_means = [attack_mean_min(STD['lineage']['desc_records'], a)[0] for a in attacks]
std_mins  = [attack_mean_min(STD['lineage']['desc_records'], a)[1] for a in attacks]
dar_means = [attack_mean_min(DAR['lineage']['desc_records'], a)[0] for a in attacks]
dar_mins  = [attack_mean_min(DAR['lineage']['desc_records'], a)[1] for a in attacks]

x = np.arange(len(attacks))
w = 0.36
ax.bar(x - w/2, std_means, w, color=COL_STD, edgecolor='black',
        linewidth=0.5, label='Standard mean')
ax.bar(x + w/2, dar_means, w, color=COL_DAR, edgecolor='black',
        linewidth=0.5, label='DAR mean')
# overlay min as crosses
ax.plot(x - w/2, std_mins, marker='_', color='black', markersize=14,
         linestyle='None', markeredgewidth=2, label='min')
ax.plot(x + w/2, dar_mins, marker='_', color='black', markersize=14,
         linestyle='None', markeredgewidth=2)
# indep ceilings as horizontal dashed lines
ax.axhline(STD['lineage']['indep_max'], color=COL_STD, linestyle=':',
            linewidth=1.0, label=f"std indep ceiling = {STD['lineage']['indep_max']:.2f}")
ax.axhline(DAR['lineage']['indep_max'], color=COL_DAR, linestyle=':',
            linewidth=1.0, label=f"DAR indep ceiling = {DAR['lineage']['indep_max']:.2f}")

ax.set_xticks(x)
ax.set_xticklabels(display_names, rotation=0)
ax.set_ylabel(r'Descendant lineage score $\mathcal{L}$')
ax.set_title('(d) Per-attack survival')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right', fontsize=7, ncol=1, framealpha=0.95)
ax.grid(axis='y', alpha=0.3)


fig.tight_layout()
out_dir = ROOT / 'figures'
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / 'fig_dar_fingerprint.pdf', bbox_inches='tight')
fig.savefig(out_dir / 'fig_dar_fingerprint.png', bbox_inches='tight', dpi=180)
# also copy into paper/figures for easy includegraphics
paper_fig_dir = ROOT / 'paper' / 'figures'
paper_fig_dir.mkdir(exist_ok=True)
fig.savefig(paper_fig_dir / 'fig_dar_fingerprint.pdf', bbox_inches='tight')
print(f"Wrote {out_dir / 'fig_dar_fingerprint.pdf'}")
print(f"Wrote {out_dir / 'fig_dar_fingerprint.png'}")
print(f"Wrote {paper_fig_dir / 'fig_dar_fingerprint.pdf'}")
