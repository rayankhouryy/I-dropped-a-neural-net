"""
Re-render the Bottleneck factor ablation figure with the correct
structure: cross-factor pairing tests (the meaningful ones) shown
prominently with both trained and random-init bars; self-similarity
controls shown separately and labeled as identity controls (since
random init recovers them at 100% too).

Uses:
  results/deepdive_resnet_factors.json          (trained)
  results/deepdive_resnet_factors_random.json   (random init, 5 seeds)

Outputs:
  figures/fig_deepdive_resnet_factors.{png,pdf}  (overwrites the old one)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Path("figures").mkdir(exist_ok=True)

trained_data = json.load(open('results/deepdive_resnet_factors.json'))
random_data  = json.load(open('results/deepdive_resnet_factors_random.json'))

n = trained_data['n']
chance = 1.0 / n

# Build a lookup: trained results by subset name
trained_by_name = {r['subset']: r for r in trained_data['results']}
random_by_name  = random_data['aggregate']

# Manually pair trained name -> random name (they differ slightly)
PAIR_MAPPING = [
    # display_label, trained_key, random_key, category
    ('$W_3 W_2 W_1$\n(full triple)',
     'W_3 W_2 W_1 (full triple)',
     'W_3 W_2 W_1 (full triple)',
     'cross'),
    ('$W_3 W_1$\n(endpoint, no $W_2$)',
     'W_3 W_1 (endpoint, no W_2)',
     'W_3 W_1 (endpoint)',
     'cross'),
    ('$W_1$\nself-sim',
     'W_1 alone (self-similarity)',
     'W_1 self-sim',
     'self_sim'),
    ('$W_2$\nself-sim',
     'W_2 alone (self-similarity)',
     'W_2 self-sim',
     'self_sim'),
    ('$W_3$\nself-sim',
     'W_3 alone (self-similarity)',
     'W_3 self-sim',
     'self_sim'),
    ('$W_2 W_1$\nself-sim (no $W_3$)',
     'W_2 W_1 self-sim (no W_3)',
     'W_2 W_1 self-sim',
     'self_sim'),
    ('$W_3 W_2$\nself-sim (no $W_1$)',
     'W_3 W_2 self-sim (no W_1)',
     'W_3 W_2 self-sim',
     'self_sim'),
]

# build paired list
rows = []
for lbl, tk, rk, cat in PAIR_MAPPING:
    t = trained_by_name[tk]
    r = random_by_name[rk]
    rows.append({
        'label':           lbl,
        'category':        cat,
        'trained_acc':     t['pair_acc'],
        'trained_auc':     t['auc'],
        'random_acc_mean': r['mean_pair_acc'],
        'random_acc_std':  r['std_pair_acc'],
        'random_auc_mean': r['mean_auc'],
    })

# Sort: cross-factor first (the meaningful tests), then self-sim controls
rows.sort(key=lambda r: (0 if r['category'] == 'cross' else 1))

n_cross = sum(1 for r in rows if r['category'] == 'cross')
n_self  = len(rows) - n_cross

# ----------------- figure -----------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

C_TRAINED = '#2ca02c'
C_RANDOM  = '#7f7f7f'

x = np.arange(len(rows))
width = 0.36

# Panel (a): pair accuracy
ax1 = axes[0]
ax1.bar(x - width/2, [r['trained_acc'] for r in rows], width,
        color=C_TRAINED, edgecolor='black', linewidth=0.5,
        label='trained')
ax1.bar(x + width/2, [r['random_acc_mean'] for r in rows], width,
        color=C_RANDOM, edgecolor='black', linewidth=0.5, alpha=0.85,
        yerr=[r['random_acc_std'] for r in rows], capsize=3,
        label='random init (5 seeds)')

# chance line
ax1.axhline(chance, color='k', ls=':', lw=1.2)
ax1.text(len(x) - 0.5, chance + 0.025, f'chance = 1/{n} = {chance:.1%}',
         fontsize=9, ha='right', color='k')

# Numbers on top
for xi, r in zip(x - width/2, rows):
    ax1.text(xi, r['trained_acc'] + 0.025,
             f"{r['trained_acc']:.0%}", ha='center', fontsize=8,
             color=C_TRAINED, fontweight='bold')
for xi, r in zip(x + width/2, rows):
    ax1.text(xi, r['random_acc_mean'] + r['random_acc_std'] + 0.025,
             f"{r['random_acc_mean']:.0%}", ha='center', fontsize=8,
             color=C_RANDOM)

# Separating divider between categories
ax1.axvline(n_cross - 0.5, color='black', ls='--', lw=0.8, alpha=0.4)

ax1.set_xticks(x)
ax1.set_xticklabels([r['label'] for r in rows], fontsize=8.5)
ax1.set_ylabel('Hungarian pair accuracy', fontsize=11)
ax1.set_ylim(0, 1.18)
ax1.set_title(f'(a) Pair accuracy on ResNet-101 layer3 (n={n})',
              fontsize=11)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax1.grid(True, axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Category banners
ax1.text(0.5 * (n_cross - 1), 1.13, 'cross-factor pairing\n(meaningful)',
         ha='center', fontsize=9, color='black',
         bbox=dict(boxstyle='round,pad=0.25', fc='#e6ffe6',
                   ec='black', alpha=0.85, linewidth=0.5))
ax1.text(n_cross + 0.5 * (n_self - 1), 1.13,
         'self-similarity\n(identity control - works at init too)',
         ha='center', fontsize=9, color='black',
         bbox=dict(boxstyle='round,pad=0.25', fc='#f0f0f0',
                   ec='black', alpha=0.85, linewidth=0.5))

# Panel (b): AUC
ax2 = axes[1]
ax2.bar(x - width/2, [r['trained_auc'] for r in rows], width,
        color=C_TRAINED, edgecolor='black', linewidth=0.5,
        label='trained')
ax2.bar(x + width/2, [r['random_auc_mean'] for r in rows], width,
        color=C_RANDOM, edgecolor='black', linewidth=0.5, alpha=0.85,
        label='random init (5 seeds)')

ax2.axhline(0.5, color='k', ls=':', lw=1.2)
ax2.text(len(x) - 0.5, 0.46, 'AUC = 0.5  (random)', fontsize=9,
         ha='right', color='k')

for xi, r in zip(x - width/2, rows):
    ax2.text(xi, r['trained_auc'] + 0.02,
             f"{r['trained_auc']:.2f}", ha='center', fontsize=8,
             color=C_TRAINED, fontweight='bold')
for xi, r in zip(x + width/2, rows):
    ax2.text(xi, r['random_auc_mean'] + 0.02,
             f"{r['random_auc_mean']:.2f}", ha='center', fontsize=8,
             color=C_RANDOM)

ax2.axvline(n_cross - 0.5, color='black', ls='--', lw=0.8, alpha=0.4)

ax2.set_xticks(x)
ax2.set_xticklabels([r['label'] for r in rows], fontsize=8.5)
ax2.set_ylabel('AUC (correct vs incorrect)', fontsize=11)
ax2.set_ylim(0, 1.18)
ax2.set_title('(b) Signal-to-noise per subset',
              fontsize=11)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax2.grid(True, axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(0.5 * (n_cross - 1), 1.13, 'cross-factor pairing',
         ha='center', fontsize=9, color='black',
         bbox=dict(boxstyle='round,pad=0.25', fc='#e6ffe6',
                   ec='black', alpha=0.85, linewidth=0.5))
ax2.text(n_cross + 0.5 * (n_self - 1), 1.13, 'self-similarity (identity control)',
         ha='center', fontsize=9, color='black',
         bbox=dict(boxstyle='round,pad=0.25', fc='#f0f0f0',
                   ec='black', alpha=0.85, linewidth=0.5))

fig.suptitle('Bottleneck factorization: endpoint $W_3 W_1$ fails, full branch product $W_3 W_2 W_1$ succeeds',
             fontsize=12.5, y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig('figures/fig_deepdive_resnet_factors.png', dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_deepdive_resnet_factors.pdf',           bbox_inches='tight')
import shutil
shutil.copy('figures/fig_deepdive_resnet_factors.png',
            'paper/figures/fig_deepdive_resnet_factors.png')
shutil.copy('figures/fig_deepdive_resnet_factors.pdf',
            'paper/figures/fig_deepdive_resnet_factors.pdf')
print('Saved figures/fig_deepdive_resnet_factors.{png,pdf} (and synced into paper/figures/)')

# print the key headline
print()
print("=" * 78)
print("HEADLINE (for the paper):")
print("=" * 78)
print(f"  Cross-factor pairing on n={n} Bottleneck blocks (ResNet-101 layer3):")
for r in rows:
    if r['category'] == 'cross':
        print(f"    {r['label']:<50s}")
        print(f"      trained:    pair_acc = {r['trained_acc']:.0%},  AUC = {r['trained_auc']:.3f}")
        print(f"      random init: pair_acc = {r['random_acc_mean']:.0%} +/- {r['random_acc_std']:.0%},  "
              f"AUC = {r['random_auc_mean']:.3f}")
print()
print("  Self-similarity controls (random init also at 100%, so these are identity controls):")
for r in rows:
    if r['category'] == 'self_sim':
        print(f"    {r['label'].replace(chr(10), ' '):<60s}: "
              f"trained {r['trained_acc']:.0%}, random {r['random_acc_mean']:.0%}+-{r['random_acc_std']:.0%}")
