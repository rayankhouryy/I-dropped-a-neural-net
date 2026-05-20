"""
Task 2 + Task 3: wrong-vs-correct extraction figure and clean JSON.

Combines the data from
  results/resnet_extraction_ablation.json   (trained, pair + triple)
  results/resnet_triple_random_baseline.json (random init, pair + triple)
into:
  figures/fig_resnet_wrong_vs_correct_factorization.{png,pdf}
  results/resnet_factorization_ablation.json
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

abl  = json.load(open('results/resnet_extraction_ablation.json'))
rand = json.load(open('results/resnet_triple_random_baseline.json'))

# ---------------- assemble clean rows ----------------
# We focus on the Bottleneck models where the W3W1 -> W3W2W1 distinction matters.
TARGETS = [
    ('resnet50',  'layer3'),
    ('resnet101', 'layer3'),
    ('resnet152', 'layer2'),
    ('resnet152', 'layer3'),
]

rand_index = {(r['model'], r['stage'], r['mode']): r for r in rand}

clean = []
for r in abl:
    if r['block_type'] != 'bottleneck':
        continue
    for stage_name, modes in r['stages'].items():
        if (r['model'], stage_name) not in TARGETS:
            continue
        for mode, scores in modes.items():
            ep = scores.get('endpoints')
            tr = scores.get('triple')
            if ep is None or tr is None:
                continue
            rb = rand_index.get((r['model'], stage_name, mode))
            clean.append({
                'model':       r['model'],
                'stage':       stage_name,
                'block_type':  r['block_type'],
                'mode':        mode,
                'n':           ep['n'],
                'chance':      ep['chance'],
                'wrong_endpoint': {
                    'product':  'W3W1',
                    'pair_acc': ep['pair_acc'],
                    'auc':      ep['auc'],
                    'sep':      ep['pair_sep'],
                },
                'correct_triple': {
                    'product':  'W3W2W1',
                    'pair_acc': tr['pair_acc'],
                    'auc':      tr['auc'],
                    'sep':      tr['pair_sep'],
                },
                'random_init_triple': None if rb is None else {
                    'product':  'W3W2W1 (random-init)',
                    'pair_acc_mean': rb['triple_mean_acc'],
                    'pair_acc_std':  rb['triple_std_acc'],
                    'auc':           rb['triple_mean_auc'],
                    'n_seeds':       rb['n_seeds'],
                },
            })

with open('results/resnet_factorization_ablation.json', 'w') as f:
    json.dump(clean, f, indent=2)
print(f"Saved {len(clean)} rows to results/resnet_factorization_ablation.json")

# ---------------- figure ----------------
# Cleaner layout: collapse the 2 modes into channel_sum only (center_tap gives
# essentially the same result), 4 groups (one per model/stage) with 3 bars
# each (endpoint trained, triple trained, triple random-init).  Stack the two
# panels vertically so each has horizontal room and labels stay short.
rows_csm = [r for r in clean if r['mode'] == 'channel_sum']

labels = [f"{r['model']} / {r['stage']}\n({r['n']} blocks)" for r in rows_csm]
wrong_accs    = [r['wrong_endpoint']['pair_acc']             for r in rows_csm]
correct_accs  = [r['correct_triple']['pair_acc']             for r in rows_csm]
random_accs   = [r['random_init_triple']['pair_acc_mean']    for r in rows_csm]
random_stds   = [r['random_init_triple']['pair_acc_std']     for r in rows_csm]
wrong_aucs    = [r['wrong_endpoint']['auc']                  for r in rows_csm]
correct_aucs  = [r['correct_triple']['auc']                  for r in rows_csm]
random_aucs   = [r['random_init_triple']['auc']              for r in rows_csm]
chances       = [r['chance']                                 for r in rows_csm]

x = np.arange(len(rows_csm))
width = 0.26

# Colors with clear semantics
C_WRONG   = '#d62728'  # red    - the wrong extraction
C_CORRECT = '#2ca02c'  # green  - the correct extraction
C_RANDOM  = '#7f7f7f'  # gray   - random-init control

fig, axes = plt.subplots(2, 1, figsize=(11, 8.2),
                          gridspec_kw={'hspace': 0.42})

# Build shared legend handles
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=C_WRONG,   edgecolor='black',
          label=r'Wrong endpoint $W_3 W_1$   (trained)'),
    Patch(facecolor=C_CORRECT, edgecolor='black',
          label=r'Correct triple $W_3 W_2 W_1$   (trained)'),
    Patch(facecolor=C_RANDOM, edgecolor='black', alpha=0.85,
          label=r'Correct triple $W_3 W_2 W_1$   (random init, 3 seeds)'),
]

# ===== Panel (a): pair accuracy =====
ax1 = axes[0]
ax1.bar(x - width, wrong_accs,   width, color=C_WRONG,   edgecolor='black', linewidth=0.5)
ax1.bar(x,          correct_accs, width, color=C_CORRECT, edgecolor='black', linewidth=0.5)
ax1.bar(x + width,  random_accs,  width, color=C_RANDOM, alpha=0.85,
        yerr=random_stds, capsize=4, edgecolor='black', linewidth=0.5)

# Annotate each bar with its value
for xi, v in zip(x - width, wrong_accs):
    ax1.text(xi, v + 0.025, f'{v:.0%}', ha='center', fontsize=10, color=C_WRONG)
for xi, v in zip(x, correct_accs):
    ax1.text(xi, v + 0.025, f'{v:.0%}', ha='center', fontsize=11,
             color=C_CORRECT, fontweight='bold')
for xi, v, s in zip(x + width, random_accs, random_stds):
    ax1.text(xi, v + s + 0.025, f'{v:.0%}', ha='center', fontsize=10, color=C_RANDOM)

# Dotted chance line per group with a single shared label
for xi, c in zip(x, chances):
    ax1.hlines(c, xi - 1.6*width, xi + 1.6*width,
               color='black', linewidth=1.4, linestyle=':')
    ax1.text(xi, c - 0.04, f'chance {c:.0%}', fontsize=8.5,
             color='black', va='top', ha='center')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel('pair accuracy', fontsize=12)
ax1.set_ylim(0, 1.20)
ax1.set_title('(a) Pair accuracy', fontsize=12, loc='left')
ax1.grid(True, axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# ===== Panel (b): AUC =====
ax2 = axes[1]
ax2.bar(x - width, wrong_aucs,   width, color=C_WRONG,   edgecolor='black', linewidth=0.5)
ax2.bar(x,          correct_aucs, width, color=C_CORRECT, edgecolor='black', linewidth=0.5)
ax2.bar(x + width,  random_aucs,  width, color=C_RANDOM, alpha=0.85,
        edgecolor='black', linewidth=0.5)

for xi, v in zip(x - width, wrong_aucs):
    ax2.text(xi, v + 0.025, f'{v:.2f}', ha='center', fontsize=10, color=C_WRONG)
for xi, v in zip(x, correct_aucs):
    ax2.text(xi, v + 0.025, f'{v:.2f}', ha='center', fontsize=11,
             color=C_CORRECT, fontweight='bold')
for xi, v in zip(x + width, random_aucs):
    ax2.text(xi, v + 0.025, f'{v:.2f}', ha='center', fontsize=10, color=C_RANDOM)

ax2.axhline(0.5, color='black', linewidth=1.4, linestyle=':')
ax2.text(len(x) - 0.5, 0.52, 'AUC = 0.5  (random discrimination)',
         fontsize=9, color='black', ha='right', va='bottom')

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_ylabel('AUC (correct vs incorrect)', fontsize=12)
ax2.set_ylim(0, 1.18)
ax2.set_title('(b) Signal-to-noise (AUC of correct- vs incorrect-pair score distributions)',
              fontsize=12, loc='left')
ax2.grid(True, axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

# Single shared legend at the top
fig.legend(handles=legend_handles,
           loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=3, fontsize=10, framealpha=0.92,
           handlelength=1.6, columnspacing=2.0, borderaxespad=0.0)

fig.suptitle('Architecture-aware factorization rescues Bottleneck ResNet pairing',
             fontsize=13, y=1.045)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('figures/fig_resnet_wrong_vs_correct_factorization.png',
            dpi=160, bbox_inches='tight')
fig.savefig('figures/fig_resnet_wrong_vs_correct_factorization.pdf',
            bbox_inches='tight')
print('Saved figures/fig_resnet_wrong_vs_correct_factorization.{png,pdf}')

# ---------------- summary ----------------
print()
print('='*78)
print('SUMMARY TABLE (channel_sum mode)')
print('='*78)
print(f"  {'model':10s} {'stage':7s} {'n':>3s} {'chance':>7s} "
      f"{'W3W1 (trained)':>16s} {'W3W2W1 (trained)':>18s} {'W3W2W1 (random)':>18s}")
for row in clean:
    if row['mode'] != 'channel_sum':
        continue
    ri = row['random_init_triple']
    rand_str = (f"{ri['pair_acc_mean']:.0%} +/- {ri['pair_acc_std']:.0%}"
                if ri else 'n/a')
    print(f"  {row['model']:10s} {row['stage']:7s} {row['n']:>3d} "
          f"{row['chance']:>7.1%} "
          f"{row['wrong_endpoint']['pair_acc']:>16.0%} "
          f"{row['correct_triple']['pair_acc']:>18.0%} "
          f"{rand_str:>18s}")
