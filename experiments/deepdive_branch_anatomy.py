"""
Deep dive: cross-architecture margin scaling, ViT per-head attention,
and ResNet Bottleneck factor ablation.

Three analyses combined in one script so they all share the same model
loads and pairing utilities:

  A) Margin scaling: for every architecture/path we've measured, compute
     - mean d(i,i)
     - sqrt(d_model) (Prop-1 theoretical upper bound)
     - mean d(i,i) / sqrt(d) (fraction of bound)
     - implied ||E||_F / (epsilon * sqrt(d)) (off-diagonal-to-diagonal
       energy ratio derived from Prop 1)

  B) ViT-B per-head: split W_V and W_O into 12 head-slabs each, compute
     per-layer per-head d(i, i; h), and look at the layer x head heatmap
     of within-block scores plus the cross-block signal carried by each
     head.

  C) ResNet Bottleneck factor ablation: for ResNet-101 layer3 (22 blocks),
     score every subset of (W_1, W_2, W_3):
        W_1 alone, W_2 alone, W_3 alone,
        W_2 W_1, W_3 W_2, W_3 W_1, W_3 W_2 W_1.
     Show only the full triple carries the signal.

Outputs:
  figures/fig_deepdive_margin_scaling.{png,pdf}
  figures/fig_deepdive_vit_perhead.{png,pdf}
  figures/fig_deepdive_resnet_factors.{png,pdf}
  results/deepdive_margin_scaling.json
  results/deepdive_vit_perhead.json
  results/deepdive_resnet_factors.json
"""
import json, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
torch.set_grad_enabled(False)


# ============================================================================
# PART A: cross-architecture margin scaling
# ============================================================================
# Pull mean d(i,i) and the relevant d_model from the JSONs we already have.
# For each entry, compute the implied ||E||_F / (epsilon * sqrt(d)) using
# Prop 1: d(i,i) = sqrt(d) / sqrt(1 + (||E||_F / (epsilon sqrt(d)))^2).
# So  alpha := ||E||_F / (epsilon * sqrt(d)) = sqrt(d / d(i,i)^2 - 1).

def alpha_from_d(d_iidot, d_model):
    """Compute the off-to-diagonal-energy ratio implied by d(i,i)."""
    bound = np.sqrt(d_model)
    if d_iidot >= bound:  # numerical safety
        return 0.0
    return float(np.sqrt((d_model / d_iidot**2) - 1))


def gather_margin_rows():
    rows = []

    # Park's puzzle: d=48, mean d(i,i) ~ 2.785 (we know from README + paper)
    rows.append({
        'family': 'Park puzzle',
        'arch':   'Residual MLP',
        'd_model': 48,
        'n':       48,
        'mean_d':  2.785,
    })

    # GPT-2 MLP family
    mlp_json = json.load(open('results/gpt2_mlp_pairing.json'))
    for r in mlp_json['pretrained']:
        rows.append({
            'family': r['model'],
            'arch':   'Transformer MLP',
            'd_model': r['d_model'],
            'n':       r['n_layers'],
            'mean_d':  r['diag_dominance']['mean_correct'],
        })

    # GPT-2 attention
    attn_json = json.load(open('results/gpt2_attention_pairing.json'))
    for r in attn_json['pretrained']:
        d = r['d_model']
        rows.append({
            'family': r['model'],
            'arch':   'Attn V<->O',
            'd_model': d,
            'n':       r['n_layers'],
            'mean_d':  r['VO_full']['mean_correct'],
        })
        rows.append({
            'family': r['model'],
            'arch':   'Attn Q<->K',
            'd_model': d,
            'n':       r['n_layers'],
            'mean_d':  r['QK']['mean_correct'],
        })

    # ResNet Bottleneck (channel_sum mode)
    res_json = json.load(open('results/resnet_factorization_ablation.json'))
    for r in res_json:
        if r['mode'] != 'channel_sum':
            continue
        if r['block_type'] != 'bottleneck':
            continue
        # For Bottleneck, d_model = planes (residual stream channels).
        # Look up from the ablation JSON.
        ablation = json.load(open('results/resnet_extraction_ablation.json'))
        for ab in ablation:
            if ab['model'] != r['model']:
                continue
            for s_name, s_modes in ab['stages'].items():
                if s_name != r['stage']:
                    continue
                if 'channel_sum' not in s_modes:
                    continue
                # We need mean_correct from the triple score
                tr_entry = s_modes['channel_sum'].get('triple')
                if tr_entry is None:
                    continue
                # Use the planes dim as d_model
                ab_block = ab['stages'][s_name]['channel_sum']['triple']
                # ab_block has n=22 etc but we need planes; pull from the
                # surrounding entry
                # The triple score uses M = W_3 W_2 W_1 in R^{planes x planes}
                # Get planes from ablation result
                planes = None
                # Crude: fall back to known per-stage planes counts
                if r['model'] == 'resnet50':
                    planes = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}[r['stage']]
                elif r['model'] == 'resnet101':
                    planes = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}[r['stage']]
                elif r['model'] == 'resnet152':
                    planes = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}[r['stage']]
                rows.append({
                    'family': f"{r['model']}/{r['stage']}",
                    'arch':   'ResNet Bottleneck',
                    'd_model': planes,
                    'n':       r['n'],
                    'mean_d':  ab_block['mean_correct'] if 'mean_correct' in ab_block else None,
                })

    # We need to enrich: mean_correct on the triple isn't always saved
    # in resnet_factorization_ablation.json. Let's read from the original
    # extraction ablation directly for ResNet Bottleneck triple.
    rows = [r for r in rows if r['mean_d'] is not None]

    # ConvNeXt-T + ViT-B
    mv = json.load(open('results/modern_vision_pairing.json'))
    cnx = mv['convnext_tiny']
    for sname, sdata in cnx['stages'].items():
        rows.append({
            'family': f"convnext_tiny/{sname}",
            'arch':   'ConvNeXt MLP',
            'd_model': sdata['d_model'],
            'n':       sdata['mlp']['n'],
            'mean_d':  sdata['mlp']['mean_correct'],
        })

    vit = mv['vit_b_16']
    for k, label in [('mlp', 'ViT MLP'), ('vo', 'ViT V<->O'), ('qk', 'ViT Q<->K')]:
        rows.append({
            'family': 'vit_b_16',
            'arch':   label,
            'd_model': vit['d_model'],
            'n':       vit['n_layers'],
            'mean_d':  vit[k]['mean_correct'],
        })

    # Compute fraction-of-bound and implied alpha for every row
    for r in rows:
        sd = np.sqrt(r['d_model'])
        r['bound'] = sd
        r['frac_of_bound'] = r['mean_d'] / sd
        r['alpha'] = alpha_from_d(r['mean_d'], r['d_model'])

    return rows


def part_A_run():
    print('\n' + '=' * 78)
    print('PART A: cross-architecture margin scaling')
    print('=' * 78)
    rows = gather_margin_rows()

    print(f"\n{'family':<22s} {'arch':<22s} {'d':>5s} {'sqrt(d)':>7s} "
          f"{'mean_d':>7s} {'frac_bnd':>8s} {'alpha':>7s}")
    print('-' * 84)
    for r in rows:
        print(f"  {r['family']:<20s} {r['arch']:<22s} {r['d_model']:>5d} "
              f"{r['bound']:>7.2f} {r['mean_d']:>7.3f} "
              f"{r['frac_of_bound']:>7.1%} {r['alpha']:>7.3f}")

    json.dump(rows, open('results/deepdive_margin_scaling.json', 'w'), indent=2)
    print('\nSaved results/deepdive_margin_scaling.json')

    # Plot: bar chart of fraction-of-bound per (family, arch)
    families_of_interest = [
        ('Park puzzle',        'Residual MLP',       'Park puzzle MLP'),
        ('gpt2',               'Transformer MLP',    'GPT-2 MLP'),
        ('gpt2-medium',        'Transformer MLP',    'GPT-2-medium MLP'),
        ('gpt2-large',         'Transformer MLP',    'GPT-2-large MLP'),
        ('gpt2-xl',            'Transformer MLP',    'GPT-2-xl MLP'),
        ('gpt2',               'Attn V<->O',         'GPT-2 V/O'),
        ('gpt2-xl',            'Attn V<->O',         'GPT-2-xl V/O'),
        ('gpt2',               'Attn Q<->K',         'GPT-2 Q/K'),
        ('gpt2-xl',            'Attn Q<->K',         'GPT-2-xl Q/K'),
        ('convnext_tiny/stage3', 'ConvNeXt MLP',     'ConvNeXt-T stage3'),
        ('vit_b_16',           'ViT MLP',            'ViT-B MLP'),
        ('vit_b_16',           'ViT V<->O',          'ViT-B V/O'),
        ('vit_b_16',           'ViT Q<->K',          'ViT-B Q/K'),
    ]
    plot_rows = []
    for fam, arch, lbl in families_of_interest:
        for r in rows:
            if r['family'] == fam and r['arch'] == arch:
                plot_rows.append({**r, 'label': lbl})
                break

    # Color by branch family
    palette = {
        'Residual MLP':     '#1f77b4',
        'Transformer MLP':  '#ff7f0e',
        'Attn V<->O':       '#2ca02c',
        'Attn Q<->K':       '#d62728',
        'ConvNeXt MLP':     '#9467bd',
        'ViT MLP':          '#8c564b',
        'ViT V<->O':        '#17becf',
        'ViT Q<->K':        '#bcbd22',
    }
    colors = [palette.get(r['arch'], 'gray') for r in plot_rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    x = np.arange(len(plot_rows))
    fracs = [r['frac_of_bound'] for r in plot_rows]
    axes[0].bar(x, fracs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].axhline(1.0, color='k', lw=1.0, ls='--', alpha=0.5)
    axes[0].text(len(x) - 0.5, 1.02, r'$\sqrt{d}$ bound', fontsize=9, ha='right')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r['label'] for r in plot_rows], rotation=40,
                              ha='right', fontsize=9)
    axes[0].set_ylabel(r'$\overline{d(i,i)}\,/\,\sqrt{d}$', fontsize=11)
    axes[0].set_title('(a) Fraction of the $\\sqrt{d}$ bound achieved', fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, axis='y', alpha=0.3)
    for xi, v in zip(x, fracs):
        axes[0].text(xi, v + 0.02, f'{v:.0%}', ha='center', fontsize=8)

    # Panel B: implied alpha
    alphas = [r['alpha'] for r in plot_rows]
    axes[1].bar(x, alphas, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r['label'] for r in plot_rows], rotation=40,
                              ha='right', fontsize=9)
    axes[1].set_ylabel(r'implied $\alpha = \|E\|_F\,/\,(\varepsilon\sqrt{d})$',
                       fontsize=11)
    axes[1].set_title('(b) Off-diagonal energy relative to scaled identity (lower = closer to $-\\varepsilon I$)',
                      fontsize=10.5)
    axes[1].grid(True, axis='y', alpha=0.3)
    for xi, v in zip(x, alphas):
        axes[1].text(xi, v + 0.1, f'{v:.1f}', ha='center', fontsize=8)

    fig.suptitle('Cross-architecture margin scaling: how close is each branch to $-\\varepsilon I$?',
                 fontsize=12.5, y=1.03)
    plt.tight_layout()
    plt.savefig('figures/fig_deepdive_margin_scaling.png', dpi=160, bbox_inches='tight')
    plt.savefig('figures/fig_deepdive_margin_scaling.pdf', bbox_inches='tight')
    print('Saved figures/fig_deepdive_margin_scaling.{png,pdf}')


# ============================================================================
# PART B: ViT-B per-head attention
# ============================================================================
def part_B_run():
    print('\n' + '=' * 78)
    print('PART B: ViT-B per-head attention')
    print('=' * 78)
    import torchvision.models as M
    vit = M.vit_b_16(weights='IMAGENET1K_V1')
    vit.eval()
    n_layers = len(vit.encoder.layers)
    d        = vit.encoder.layers[0].self_attention.embed_dim
    nh       = vit.encoder.layers[0].self_attention.num_heads
    d_head   = d // nh
    print(f"  encoder layers: {n_layers}, d_model: {d}, n_heads: {nh}, d_head: {d_head}")

    # Extract W_V and W_O per layer
    W_V_layers = []   # each (d, d) = stacked (nh head-slabs of (d_head, d))
    W_O_layers = []
    for lay in vit.encoder.layers:
        in_proj = lay.self_attention.in_proj_weight.detach().float().cpu().numpy()
        # in_proj is (3d, d); split into Q, K, V (each d x d)
        _, _, W_V = np.split(in_proj, 3, axis=0)
        W_O = lay.self_attention.out_proj.weight.detach().float().cpu().numpy()
        W_V_layers.append(W_V)
        W_O_layers.append(W_O)
    del vit; gc.collect()

    # Per-head extraction.
    # PyTorch's MultiheadAttention: V is reshaped as (B, S, nh, d_head) before
    # attention; W_V's output axis is head-major-then-d_head.
    # So row 0..d_head-1 of W_V correspond to head 0; rows d_head..2*d_head-1
    # are head 1; etc.
    # For W_O, the INPUT axis is head-major: cols 0..d_head-1 read from head 0.
    def head_slab_V(W_V, h):
        return W_V[h*d_head:(h+1)*d_head, :]   # (d_head, d)
    def head_slab_O(W_O, h):
        return W_O[:, h*d_head:(h+1)*d_head]   # (d, d_head)

    # For each (layer i, head h), the head's "branch" matrix is
    #    M_i^h := head_slab_O(W_O_i, h) @ head_slab_V(W_V_i, h)   in R^{d x d}
    # of rank at most d_head. We use the same diagonal-dominance ratio.

    # 1) Per-layer per-head SELF score d(i,i; h) -- how much each head
    #    inside its own block contributes to the diagonal.
    self_score = np.zeros((n_layers, nh))
    for i in range(n_layers):
        for h in range(nh):
            P = head_slab_O(W_O_layers[i], h) @ head_slab_V(W_V_layers[i], h)
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            self_score[i, h] = tr / fr

    # 2) Per-head Hungarian: for each head h, build a (n_layers x n_layers)
    #    d(i, j; h) matrix and measure pair_acc, AUC, mean correct.
    per_head_results = []
    for h in range(nh):
        M = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(n_layers):
                P = head_slab_O(W_O_layers[j], h) @ head_slab_V(W_V_layers[i], h)
                tr = abs(np.trace(P))
                fr = np.linalg.norm(P, 'fro') + 1e-12
                M[i, j] = tr / fr
        _, col = linear_sum_assignment(-M)
        pair_acc = float((col == np.arange(n_layers)).mean())
        diag = np.diag(M)
        off  = M[~np.eye(n_layers, dtype=bool)]
        pos = diag[:, None]; neg = off[None, :]
        auc = float(((pos > neg).sum() + 0.5 * (pos == neg).sum())
                    / (pos.size * neg.size))
        per_head_results.append({
            'head':           h,
            'pair_acc':       pair_acc,
            'auc':            auc,
            'mean_correct':   float(diag.mean()),
            'mean_incorrect': float(off.mean()),
        })

    print(f"\n  Per-head Hungarian pairing (each head's own d(i,j) matrix):")
    print(f"  {'head':>4s} {'pair_acc':>9s} {'AUC':>6s} {'mean_corr':>10s} {'mean_inc':>9s}")
    for r in per_head_results:
        print(f"  {r['head']:>4d} {r['pair_acc']:>8.0%}  {r['auc']:>6.3f} "
              f"{r['mean_correct']:>10.3f} {r['mean_incorrect']:>9.3f}")

    json.dump({
        'n_layers':         n_layers,
        'n_heads':          nh,
        'd_head':           d_head,
        'self_score_layer_x_head': self_score.tolist(),
        'per_head_results': per_head_results,
    }, open('results/deepdive_vit_perhead.json', 'w'), indent=2)
    print('\nSaved results/deepdive_vit_perhead.json')

    # Plot: heatmap of self_score (layer x head) + bar of per-head pair_acc + AUC
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(self_score, cmap='magma', aspect='auto', vmin=0)
    ax1.set_xlabel('head index')
    ax1.set_ylabel('layer index')
    ax1.set_xticks(range(nh))
    ax1.set_yticks(range(n_layers))
    ax1.set_title(r'(a) ViT-B/16 within-block per-head $d(i,i;h)$',
                  fontsize=11)
    plt.colorbar(im, ax=ax1, shrink=0.9, label='self score')

    ax2 = fig.add_subplot(gs[1])
    accs = [r['pair_acc'] for r in per_head_results]
    bar_colors = ['#2ca02c' if a > 0.5 else '#d62728' for a in accs]
    ax2.bar(range(nh), accs, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(1.0/n_layers, color='k', ls=':', lw=1.0, alpha=0.6)
    ax2.text(nh - 0.5, 1.0/n_layers + 0.02, f'chance 1/{n_layers}',
             fontsize=8, color='k', ha='right')
    ax2.set_xlabel('head index')
    ax2.set_ylabel('pair_acc using only this head')
    ax2.set_xticks(range(nh))
    ax2.set_title('(b) Hungarian pair_acc when restricted\nto one head only',
                  fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis='y', alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    aucs = [r['auc'] for r in per_head_results]
    bar_colors_auc = ['#2ca02c' if a > 0.7 else ('#ff7f0e' if a > 0.55 else '#d62728')
                       for a in aucs]
    ax3.bar(range(nh), aucs, color=bar_colors_auc, edgecolor='black', linewidth=0.5)
    ax3.axhline(0.5, color='k', ls=':', lw=1.0, alpha=0.6)
    ax3.set_xlabel('head index')
    ax3.set_ylabel('AUC (correct vs incorrect)')
    ax3.set_xticks(range(nh))
    ax3.set_title('(c) Per-head AUC: signal carried by each head',
                  fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, axis='y', alpha=0.3)

    fig.suptitle('ViT-B/16 attention: which heads carry the pairing fingerprint?',
                 fontsize=12.5, y=1.03)
    plt.tight_layout()
    plt.savefig('figures/fig_deepdive_vit_perhead.png', dpi=160, bbox_inches='tight')
    plt.savefig('figures/fig_deepdive_vit_perhead.pdf', bbox_inches='tight')
    print('Saved figures/fig_deepdive_vit_perhead.{png,pdf}')

    # quick summary
    high_acc = [r for r in per_head_results if r['pair_acc'] > 0.5]
    print(f"\n  Heads with pair_acc > 50%: {len(high_acc)} / {nh}")
    print(f"  Head pair_acc range: [{min(accs):.0%}, {max(accs):.0%}]")
    print(f"  Head AUC range:      [{min(aucs):.3f}, {max(aucs):.3f}]")


# ============================================================================
# PART C: ResNet Bottleneck factor ablation
# ============================================================================
def part_C_run():
    print('\n' + '=' * 78)
    print('PART C: ResNet-101 layer3 Bottleneck factor ablation (22 blocks)')
    print('=' * 78)
    import torchvision.models as M

    # We compute every subset of (W_1, W_2, W_3) for each block,
    # then score blocks against each other using diagonal-dominance
    # on each subset product.
    # Import via direct file path so it works whether cwd is repo root
    # or a subdirectory.
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        'resnet_extraction_ablation',
        os.path.join('experiments', 'resnet_extraction_ablation.py'),
    )
    _rea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_rea)
    fold_bn_scale = _rea.fold_bn_scale
    conv_to_matrix_channel_sum = _rea.conv_to_matrix_channel_sum

    model = M.resnet101(weights='IMAGENET1K_V2')
    model.eval()
    stage = model.layer3
    blocks = []
    for blk in stage:
        if getattr(blk, 'downsample', None) is not None:
            continue
        w1 = conv_to_matrix_channel_sum(fold_bn_scale(blk.conv1.weight, blk.bn1))
        w2 = conv_to_matrix_channel_sum(fold_bn_scale(blk.conv2.weight, blk.bn2))
        w3 = conv_to_matrix_channel_sum(fold_bn_scale(blk.conv3.weight, blk.bn3))
        blocks.append((w1, w2, w3))
    del model; gc.collect()
    n = len(blocks)
    print(f"  n_blocks = {n}, planes = {blocks[0][0].shape[1]}, mid = {blocks[0][0].shape[0]}")

    # Compute the "self ratio" |tr(P)| / ||P||_F for a single matrix P
    def self_ratio(P):
        return abs(np.trace(P)) / (np.linalg.norm(P, 'fro') + 1e-12)

    def score_pair(A_list, B_list):
        """Compute d(i,j) = |tr(B[j] A[i])| / ||...||_F across blocks."""
        N = len(A_list)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                P = B_list[j] @ A_list[i]
                tr_ = abs(np.trace(P))
                fr_ = np.linalg.norm(P, 'fro') + 1e-12
                M[i, j] = tr_ / fr_
        return M

    def score_self(matrices):
        """Single-matrix score across blocks: M[i,j] = self_ratio of mixed matrices?
        For a single matrix per block (e.g. W_1 alone) there's no cross-block
        product, but we can still ask 'is block i's matrix more similar to
        itself than to block j?' This needs a definition. We measure
        |tr(M_j @ M_i.T)| / ||...||_F as a similarity-style score.
        """
        N = len(matrices)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                P = matrices[j] @ matrices[i].T
                M[i, j] = self_ratio(P)
        return M

    def hungarian(M):
        N = M.shape[0]
        _, col = linear_sum_assignment(-M)
        return float((col == np.arange(N)).mean())

    def auc_corr_vs_inc(M):
        N = M.shape[0]
        diag = np.diag(M); off = M[~np.eye(N, dtype=bool)]
        pos = diag[:, None]; neg = off[None, :]
        return float(((pos > neg).sum() + 0.5 * (pos == neg).sum())
                      / (pos.size * neg.size))

    # Build per-factor lists
    W_1s = [b[0] for b in blocks]  # (mid, planes)
    W_2s = [b[1] for b in blocks]  # (mid, mid)
    W_3s = [b[2] for b in blocks]  # (planes, mid)

    # Pre-compute compositions
    W_2_W_1   = [W_2s[i] @ W_1s[i] for i in range(n)]
    W_3_W_2   = [W_3s[i] @ W_2s[i] for i in range(n)]
    W_3_W_1   = [W_3s[i] @ W_1s[i] if False else None for i in range(n)]
    # Wait, W_3 @ W_1 is shape (planes, planes) only if W_3 is (planes, mid)
    # and W_1 is (mid, planes). Check: W_1 = (mid, planes), W_3 = (planes, mid)
    # So W_3 @ W_1 is (planes, planes). 
    W_3_W_1   = [W_3s[i] @ W_1s[i] for i in range(n)]
    W_3_W_2_W_1 = [W_3s[i] @ W_2s[i] @ W_1s[i] for i in range(n)]

    # Scoring strategies:
    # - Single factor: similarity via M_i @ M_j^T
    # - Pair of factors that compose to a (k x k) square matrix: use self_ratio
    #   on the composition itself, but it's per-block, no cross-block product.
    #
    # For a clean "diagonal-dominance pair score" we need a way to compute
    # d(i,j) by mixing block i's piece with block j's piece, getting a square
    # matrix, taking tr and Frobenius. For each subset we define what's
    # "block i's input piece" and "block j's output piece":
    #
    #   Subset W_3 W_2 W_1: block i contributes W_1, block j contributes W_3 W_2.
    #     d(i,j) = |tr(W_3^j W_2^j W_1^i)| / ||W_3^j W_2^j W_1^i||_F   (the paper's score)
    #
    #   Subset W_3 W_1: block i contributes W_1, block j contributes W_3.
    #     d(i,j) = |tr(W_3^j W_1^i)| / ||...||_F   (the endpoint we know fails)
    #
    #   Subset W_2 W_1: block i contributes W_1, block j contributes W_2.
    #     d(i,j) = |tr(W_2^j W_1^i)| / ||W_2^j W_1^i||_F      (W_2^j W_1^i is (mid, planes))
    #     This isn't square; need to be careful. Let's use the square version:
    #     define M = (W_2^j W_1^i)(W_2^j W_1^i)^T which is (mid, mid) and use
    #     self_ratio. Actually, let's drop nonsquare cases and only score
    #     square cross-block products.
    #
    # For each subset, we describe what i-piece and j-piece we use.
    #
    #   subset     | i-piece     | j-piece     | composed matrix shape
    #   -----------|-------------|-------------|----------------------
    #   W_1 only   | W_1^i       | (W_1^j)^T   | (mid,planes)(planes,mid) = (mid, mid)
    #   W_2 only   | W_2^i       | (W_2^j)^T   | (mid,mid)(mid,mid)       = (mid, mid)
    #   W_3 only   | W_3^i       | (W_3^j)^T   | (planes,mid)(mid,planes) = (planes, planes)
    #   W_3 W_1    | W_1^i       | W_3^j       | (planes,mid)(mid,planes) = (planes, planes)
    #   W_2 W_1    | W_2^i W_1^i | (W_2^j W_1^j)^T | square in (mid)
    #              | i pre-applied, j pre-applied, take outer product
    #              Hmm actually we want d(i,j) on a SINGLE composition.
    #              Let's use the joint composition: M_ij = (W_2^j W_1^j)(W_2^i W_1^i)^T
    #   W_3 W_2    | W_3^i W_2^i | (W_3^j W_2^j)^T | square in (planes)
    #   W_3 W_2 W_1| W_1^i       | W_3^j W_2^j | (planes,planes)
    #
    # Simpler: every score is M_ij = OUT_j @ IN_i where IN_i and OUT_j are
    # rectangular and OUT_j @ IN_i is square.

    subsets = []

    # "Full triple" (what works)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = (W_3s[j] @ W_2s[j]) @ W_1s[i]
            M[i, j] = self_ratio(P)
    subsets.append(('W_3 W_2 W_1 (full triple)', M))

    # Endpoint (what we showed fails)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = W_3s[j] @ W_1s[i]
            M[i, j] = self_ratio(P)
    subsets.append(('W_3 W_1 (endpoint, no W_2)', M))

    # W_2 . W_1 only: produces (mid, planes), need square. Use
    # P = W_2^j W_1^i (mid, planes) and score with self_ratio? not square.
    # Define M = (W_2^j W_1^i) (W_2^i W_1^i)^T -- no, that mixes too much.
    # Cleanest: pair i's W_1 with j's W_2 (j-side), via
    # P = (W_2^j W_1^i)(W_2^j W_1^i).T which is (mid, mid). This is the
    # 'two-factor i->j stream' truncated before W_3.
    # But that's quadratic in i's contribution. Easier: skip the W_2 W_1
    # double and just look at single-factor self-similarity tests for W_1
    # and W_2 alone, which directly answer 'is the signal in just one factor?'

    # W_1 alone: M_ij = W_1^j (W_1^i)^T  (mid, mid)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = W_1s[j] @ W_1s[i].T
            M[i, j] = self_ratio(P)
    subsets.append(('W_1 alone (self-similarity)', M))

    # W_2 alone (mid, mid): M_ij = W_2^j (W_2^i)^T (mid, mid)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = W_2s[j] @ W_2s[i].T
            M[i, j] = self_ratio(P)
    subsets.append(('W_2 alone (self-similarity)', M))

    # W_3 alone (planes, mid): M_ij = W_3^j (W_3^i)^T (planes, planes)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = W_3s[j] @ W_3s[i].T
            M[i, j] = self_ratio(P)
    subsets.append(('W_3 alone (self-similarity)', M))

    # "Bundled W_2 W_1 (with j's W_2 + i's W_1)": M_ij = (W_2^j W_1^i) @ (W_2^j W_1^i)^T
    # actually a better, symmetric formulation: P = (W_2^j W_1^i)
    # We want a square matrix; outer product would conflate.
    # Best: use M_ij = trace-similarity between full intermediate products.
    # That is: define h_i := W_2^i W_1^i (mid, planes), and score
    #   d(i,j) = |tr(h_j h_i^T)| / ||h_j h_i^T||_F
    M = np.zeros((n, n))
    h = [W_2s[i] @ W_1s[i] for i in range(n)]
    for i in range(n):
        for j in range(n):
            P = h[j] @ h[i].T
            M[i, j] = self_ratio(P)
    subsets.append(('W_2 W_1 self-sim (no W_3)', M))

    # W_3 W_2 self-sim (no W_1): k_i := W_3^i W_2^i (planes, mid)
    M = np.zeros((n, n))
    k = [W_3s[i] @ W_2s[i] for i in range(n)]
    for i in range(n):
        for j in range(n):
            P = k[j] @ k[i].T
            M[i, j] = self_ratio(P)
    subsets.append(('W_3 W_2 self-sim (no W_1)', M))

    # Score each
    print(f"\n  {'subset':<32s}  {'pair_acc':>9s}  {'AUC':>6s}  "
          f"{'mean_corr':>10s}  {'mean_inc':>9s}")
    results = []
    for name, M in subsets:
        diag = np.diag(M); off = M[~np.eye(n, dtype=bool)]
        pa = hungarian(M); auc = auc_corr_vs_inc(M)
        mc, mi = float(diag.mean()), float(off.mean())
        results.append({
            'subset': name,
            'pair_acc': pa, 'auc': auc,
            'mean_correct': mc, 'mean_incorrect': mi,
        })
        print(f"  {name:<32s}  {pa:>8.0%}  {auc:>6.3f}  {mc:>10.3f}  {mi:>9.3f}")

    json.dump({'n': n, 'results': results},
              open('results/deepdive_resnet_factors.json', 'w'), indent=2)
    print('\nSaved results/deepdive_resnet_factors.json')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels = [r['subset'] for r in results]
    chance = 1.0 / n
    accs = [r['pair_acc'] for r in results]
    aucs = [r['auc']      for r in results]

    # Highlight the full triple
    bar_colors_acc = ['#2ca02c' if 'full triple' in lbl else
                      ('#d62728' if 'endpoint' in lbl else '#7f7f7f')
                      for lbl in labels]

    axes[0].bar(range(len(labels)), accs, color=bar_colors_acc,
                edgecolor='black', linewidth=0.5)
    axes[0].axhline(chance, color='k', ls=':', lw=1.2)
    axes[0].text(len(labels) - 0.5, chance + 0.02,
                 f'chance = {chance:.1%}', fontsize=9, ha='right')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('pair accuracy')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title(f'(a) Hungarian pair_acc per subset (n={n})',
                      fontsize=11)
    axes[0].grid(True, axis='y', alpha=0.3)
    for xi, v in zip(range(len(labels)), accs):
        axes[0].text(xi, v + 0.02, f'{v:.0%}', ha='center', fontsize=8)

    axes[1].bar(range(len(labels)), aucs, color=bar_colors_acc,
                edgecolor='black', linewidth=0.5)
    axes[1].axhline(0.5, color='k', ls=':', lw=1.2)
    axes[1].text(len(labels) - 0.5, 0.46, 'AUC = 0.5  (random)',
                 fontsize=8.5, ha='right', color='k')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('AUC (correct vs incorrect)')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('(b) Signal-to-noise per subset',
                      fontsize=11)
    axes[1].grid(True, axis='y', alpha=0.3)
    for xi, v in zip(range(len(labels)), aucs):
        axes[1].text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

    fig.suptitle('ResNet-101 layer3 Bottleneck factor ablation: only the full triple carries the signal',
                 fontsize=12.5, y=1.03)
    plt.tight_layout()
    plt.savefig('figures/fig_deepdive_resnet_factors.png', dpi=160, bbox_inches='tight')
    plt.savefig('figures/fig_deepdive_resnet_factors.pdf', bbox_inches='tight')
    print('Saved figures/fig_deepdive_resnet_factors.{png,pdf}')


# ============================================================================
# main
# ============================================================================
if __name__ == '__main__':
    part_A_run()
    part_C_run()
    part_B_run()
