"""
Clean replacement for fig_torchvision_resnet_pairing.{png,pdf}.

Shows the d(i,j) diagonal-dominance matrix for the best (largest n)
stage of each ImageNet ResNet variant under the architecture-aware
extraction:

  ResNet34 layer3 (5 BasicBlocks):     d(i,j) = |tr(W_2[j] W_1[i])| / ||.||_F
  ResNet50 layer3 (5 Bottlenecks):     d(i,j) = |tr(W_3[j] W_2[j] W_1[i])| / ||.||_F
  ResNet101 layer3 (22 Bottlenecks):   triple as above
  ResNet152 layer3 (35 Bottlenecks):   triple as above

Channel-sum spatial collapse, BatchNorm folded into each conv.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, 'experiments')
from resnet_extraction_ablation import (
    fold_bn_scale,
    conv_to_matrix_channel_sum,
    score_pair_matrix,
    score_triple_endpoint,
    hungarian_accuracy,
    auc_correct_vs_incorrect,
)

torch.set_grad_enabled(False)
Path('figures').mkdir(exist_ok=True)


def extract_bottleneck_stage_triple(stage):
    """Return per-block (W1, W2, W3) numpy channel matrices, dropping
    blocks that change the channel dim (the 'downsample' first block)."""
    W1s, W2s, W3s = [], [], []
    for block in stage:
        if getattr(block, 'downsample', None) is not None:
            continue
        w1 = fold_bn_scale(block.conv1.weight, block.bn1)
        w2 = fold_bn_scale(block.conv2.weight, block.bn2)
        w3 = fold_bn_scale(block.conv3.weight, block.bn3)
        W1s.append(conv_to_matrix_channel_sum(w1))
        W2s.append(conv_to_matrix_channel_sum(w2))
        W3s.append(conv_to_matrix_channel_sum(w3))
    return W1s, W2s, W3s


def extract_basicblock_stage_pair(stage):
    W1s, W2s = [], []
    for block in stage:
        if getattr(block, 'downsample', None) is not None:
            continue
        w1 = fold_bn_scale(block.conv1.weight, block.bn1)
        w2 = fold_bn_scale(block.conv2.weight, block.bn2)
        W1s.append(conv_to_matrix_channel_sum(w1))
        W2s.append(conv_to_matrix_channel_sum(w2))
    return W1s, W2s


import torchvision.models as MM

SPECS = [
    # (display_name, builder, weights, stage_attr, block_type)
    ('ResNet-34',  MM.resnet34,  'IMAGENET1K_V1', 'layer3', 'basicblock'),
    ('ResNet-50',  MM.resnet50,  'IMAGENET1K_V2', 'layer3', 'bottleneck'),
    ('ResNet-101', MM.resnet101, 'IMAGENET1K_V2', 'layer3', 'bottleneck'),
    ('ResNet-152', MM.resnet152, 'IMAGENET1K_V2', 'layer3', 'bottleneck'),
]


panels = []
for name, builder, weights, stage_attr, block_type in SPECS:
    print(f"Loading {name} / {stage_attr} ({block_type}) ...")
    model = builder(weights=weights)
    model.eval()
    stage = getattr(model, stage_attr)

    if block_type == 'bottleneck':
        W1s, W2s, W3s = extract_bottleneck_stage_triple(stage)
        M = score_triple_endpoint(W1s, W2s, W3s)
        product_label = r'$d(i,j) = |\mathrm{tr}(W_3^{(j)} W_2^{(j)} W_1^{(i)})| / \|\cdot\|_F$'
    else:
        W1s, W2s = extract_basicblock_stage_pair(stage)
        M = score_pair_matrix(W1s, W2s)
        product_label = r'$d(i,j) = |\mathrm{tr}(W_2^{(j)} W_1^{(i)})| / \|\cdot\|_F$'

    n = M.shape[0]
    pair_acc, pair_sep = hungarian_accuracy(M)
    auc = auc_correct_vs_incorrect(M)
    chance = 1.0 / n
    panels.append({
        'name': name, 'stage': stage_attr, 'block_type': block_type,
        'n': n, 'M': M, 'pair_acc': pair_acc, 'pair_sep': pair_sep,
        'auc': auc, 'chance': chance, 'product_label': product_label,
    })
    del model


# ---------------------- figure ----------------------
fig = plt.figure(figsize=(17, 5.1))
gs = GridSpec(1, 4, figure=fig, wspace=0.5)

vmax_global = max(p['M'].max() for p in panels)

for k, p in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(p['M'], cmap='magma', aspect='equal', vmin=0)
    bt = ('Bottleneck: $M = W_3 W_2 W_1$' if p['block_type'] == 'bottleneck'
          else 'BasicBlock: $M = W_2 W_1$')
    # 2-line title above the heatmap
    ax.set_title(
        f"{p['name']} / {p['stage']}\n{bt}",
        fontsize=11, pad=6,
    )
    ticks = list(range(0, p['n'], max(1, p['n'] // 6)))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'output-side block idx $j$', fontsize=9)
    if k == 0:
        ax.set_ylabel(r'input-side block idx $i$', fontsize=9)

    # Stats BELOW the heatmap, broken into 2 short lines so each fits
    # under its own panel without bleeding into the neighbouring one.
    stats_text = (
        f"$n = {p['n']}$,  pair acc = {p['pair_acc']:.0%}\n"
        f"chance {p['chance']:.0%},  AUC = {p['auc']:.2f}"
    )
    ax.text(
        0.5, -0.32, stats_text,
        transform=ax.transAxes, fontsize=9.5,
        ha='center', va='top',
    )

    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(r'$d(i, j)$', fontsize=8.5)

fig.suptitle(
    'Architecture-aware diagonal-dominance on ImageNet ResNets (largest stage per variant)',
    fontsize=12.5, y=1.02,
)
plt.savefig('figures/fig_torchvision_resnet_pairing.png',
            dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_torchvision_resnet_pairing.pdf',
            bbox_inches='tight')
print('Saved figures/fig_torchvision_resnet_pairing.{png,pdf}')

# also sync into paper/figures
import shutil
shutil.copy('figures/fig_torchvision_resnet_pairing.png',
            'paper/figures/fig_torchvision_resnet_pairing.png')
shutil.copy('figures/fig_torchvision_resnet_pairing.pdf',
            'paper/figures/fig_torchvision_resnet_pairing.pdf')
print('Synced into paper/figures/.')

print()
print('Summary:')
for p in panels:
    print(f"  {p['name']:11s} {p['stage']:7s}  n={p['n']:2d}  "
          f"pair_acc={p['pair_acc']:.0%}  AUC={p['auc']:.3f}  "
          f"sep={p['pair_sep']:+.3f}")
