"""
Controlled mechanism study for torchvision ResNet pairing.

Issue #5 follow-up: separate the confounds in the initial torchvision
result. The initial run used a 2-matrix extraction (W_in_eff, W_out_eff)
on Bottleneck blocks that actually have THREE convs per residual branch
(conv1, conv2, conv3). This script implements the correct
architecture-aware extractions, then sweeps multiple extraction
methods for each.

Hypotheses being separated:

  H_factorization : Bottleneck blocks need a TRIPLE-product
                    M = W_3 . W_2 . W_1, not a pairwise (W_in, W_out)
                    M = W_3 . W_1. If the triple version recovers high
                    pair_acc, the initial 'BN kills the signal' finding
                    was a factorization artifact.

  H_extraction    : The channel-sum spatial collapse is too lossy.
                    Center-tap or full spatial-aware extractions might
                    recover the signal.

  H_BN            : BatchNorm itself disrupts the signal (the original
                    claim; tested in a separate MLP-norm-ablation script).

  H_bottleneck    : Bottleneck residual flow does not factor as
                    -epsilon * I + E in the same way two-conv residual
                    blocks do.

This script tests H_factorization and H_extraction. The MLP norm
ablation lives in mlp_norm_ablation.py and isolates H_BN.

Outputs:
  figures/fig_resnet_extraction_ablation.{png,pdf}
  results/resnet_extraction_ablation.json
"""

import json
import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

torch.set_grad_enabled(False)


# ----------------------------------------------------------------- BN folding
def fold_bn_scale(conv_weight: torch.Tensor, bn: torch.nn.BatchNorm2d) -> torch.Tensor:
    """Multiplicative BN factor folded into the conv's output channels.

    BN(z)_c = gamma_c * (z_c - mu_c) / sqrt(var_c + eps) + beta_c
    Only the linear factor gamma/sqrt(var+eps) matters for d(i,j)
    because trace and Frobenius norm are both linear in W.
    """
    eps = bn.eps
    scale = bn.weight / torch.sqrt(bn.running_var + eps)
    return conv_weight * scale.view(-1, 1, 1, 1)


# ----------------------------------------------------------------- extractions
def conv_to_matrix_channel_sum(W: torch.Tensor) -> np.ndarray:
    """Sum the conv kernel over all spatial taps -> channel matrix."""
    return W.sum(dim=(2, 3)).cpu().numpy().astype(np.float32)


def conv_to_matrix_center_tap(W: torch.Tensor) -> np.ndarray:
    """Take only the center spatial entry of the conv kernel."""
    kH, kW = W.shape[2], W.shape[3]
    cH, cW = kH // 2, kW // 2
    return W[:, :, cH, cW].cpu().numpy().astype(np.float32)


def conv_compose_spatial(W2: torch.Tensor, W1: torch.Tensor) -> torch.Tensor:
    """Spatial conv composition: K_out[u,v] = sum_{a,b} K2[u-a, v-b] @ K1[a, b].

    Returns a tensor of shape (out_channels_2, in_channels_1, kH2+kH1-1, kW2+kW1-1).
    Implemented via PyTorch's conv_transpose / explicit kernel arithmetic;
    here we just use direct enumeration since kernels are small (1x1 or 3x3).
    """
    O2, M, k2H, k2W = W2.shape
    Mp, I, k1H, k1W = W1.shape
    assert M == Mp, "channel mismatch in composition"
    new_kH = k1H + k2H - 1
    new_kW = k1W + k2W - 1
    K = torch.zeros(O2, I, new_kH, new_kW, dtype=W1.dtype, device=W1.device)
    # K[u,v] = sum_{a,b: 0<=u-a<k2H, 0<=v-b<k2W} W2[u-a, v-b] @ W1[a, b]
    for a in range(k1H):
        for b in range(k1W):
            for du in range(k2H):
                for dv in range(k2W):
                    u, v = a + du, b + dv
                    K[:, :, u, v] += W2[:, :, du, dv] @ W1[:, :, a, b]
    return K


def conv_compose_triple_spatial(W3, W2, W1):
    """Compose three conv kernels spatially: K = K3 * K2 * K1."""
    K12 = conv_compose_spatial(W2, W1)
    K123 = conv_compose_spatial(W3, K12)
    return K123


def composed_kernel_to_diag_dom(K: torch.Tensor) -> float:
    """For a composed conv kernel K of shape (O, O, k, k) where O=O_in=O_out,
    compute the spatial-aware ratio:
       d = |sum_{u,v} tr(K[u,v])| / sqrt(sum_{u,v} ||K[u,v]||_F^2)
    """
    K = K.cpu().numpy().astype(np.float32)
    # K has shape (out, in, kH, kW). We need square channel matrices.
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"non-square channel dims: {K.shape[:2]}")
    kH, kW = K.shape[2], K.shape[3]
    tr_sum = 0.0
    fr_sq = 0.0
    for u in range(kH):
        for v in range(kW):
            block = K[:, :, u, v]  # (O, O)
            tr_sum += np.trace(block)
            fr_sq += np.sum(block ** 2)
    return abs(tr_sum) / (np.sqrt(fr_sq) + 1e-12)


# ----------------------------------------------------------------- pairing utils
def hungarian_accuracy(M: np.ndarray) -> tuple[float, float]:
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    return pair_acc, pair_sep


def auc_correct_vs_incorrect(M: np.ndarray) -> float:
    """Threshold-free measure: AUC of correct vs incorrect d-scores."""
    n = M.shape[0]
    diag = np.diag(M)
    off  = M[~np.eye(n, dtype=bool)]
    # Mann-Whitney U / AUC
    pos = diag[:, None]   # (n, 1)
    neg = off[None, :]    # (1, n*(n-1))
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    total = pos.size * neg.size
    return float(wins / total)


# ----------------------------------------------------------------- block extraction
def extract_bottleneck_block_matrices(block, mode='channel_sum', fold_bn=True):
    """For a Bottleneck block, return:
       M_eff = W_3_eff . W_2_eff . W_1_eff  (planes x planes channel matrix)
    using one of several extraction modes.

    mode == 'channel_sum'   : sum each conv over spatial, multiply channel matrices
    mode == 'center_tap'    : center tap of each conv (1x1 are unchanged, 3x3 -> center)
    mode == 'spatial_compose': fully compose convs spatially, then score with the
                              spatial-aware ratio (see composed_kernel_to_diag_dom)
    """
    if fold_bn:
        w1 = fold_bn_scale(block.conv1.weight, block.bn1)
        w2 = fold_bn_scale(block.conv2.weight, block.bn2)
        w3 = fold_bn_scale(block.conv3.weight, block.bn3)
    else:
        w1 = block.conv1.weight
        w2 = block.conv2.weight
        w3 = block.conv3.weight

    if mode in ('channel_sum', 'center_tap'):
        to_mat = (conv_to_matrix_channel_sum if mode == 'channel_sum'
                  else conv_to_matrix_center_tap)
        W1 = to_mat(w1)
        W2 = to_mat(w2)
        W3 = to_mat(w3)
        return W1, W2, W3, None  # None = no composed kernel needed
    elif mode == 'spatial_compose':
        # Return the composed triple kernel
        K123 = conv_compose_triple_spatial(w3, w2, w1)  # (planes, planes, kH, kW)
        return None, None, None, K123
    else:
        raise ValueError(mode)


def extract_basicblock_matrices(block, mode='channel_sum', fold_bn=True):
    """For a BasicBlock, M_eff = W_2 . W_1, both 3x3 channel matrices."""
    if fold_bn:
        w1 = fold_bn_scale(block.conv1.weight, block.bn1)
        w2 = fold_bn_scale(block.conv2.weight, block.bn2)
    else:
        w1 = block.conv1.weight
        w2 = block.conv2.weight

    if mode in ('channel_sum', 'center_tap'):
        to_mat = (conv_to_matrix_channel_sum if mode == 'channel_sum'
                  else conv_to_matrix_center_tap)
        return to_mat(w1), to_mat(w2), None
    elif mode == 'spatial_compose':
        K12 = conv_compose_spatial(w2, w1)
        return None, None, K12
    else:
        raise ValueError(mode)


# ----------------------------------------------------------------- score matrices
def score_pair_matrix(W_ins, W_outs):
    """Two-matrix pairing: d(i, j) = |tr(W_outs[j] . W_ins[i])| / ||.||_F"""
    n = len(W_ins)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = W_outs[j] @ W_ins[i]
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def score_triple_endpoint(W_1s, W_2s, W_3s):
    """For Bottleneck: d(i, j) = |tr(W_3[j] . W_2[j] . W_1[i])| / ||.||_F.
    Tests whether block i's W_1 pairs with block j's (W_2, W_3).
    Same matching task as the original pair score; W_2 of the j-side is fixed.
    """
    n = len(W_1s)
    M = np.zeros((n, n))
    # Pre-compute W_3[j] @ W_2[j]
    W32 = [W_3s[j] @ W_2s[j] for j in range(n)]
    for i in range(n):
        for j in range(n):
            P = W32[j] @ W_1s[i]
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def score_full_triple(W_1s, W_2s, W_3s):
    """All three matrices come from block i for the diagonal entry; cross-block
    products use j's (W_2, W_3) with i's W_1. M_i = W_3[i] W_2[i] W_1[i].
    This is what most directly tests 'does block i's full residual branch
    diagonalize?' For the off-diagonal we mix one piece of i in.
    Identical to score_triple_endpoint with W_3, W_2 paired together.
    """
    return score_triple_endpoint(W_1s, W_2s, W_3s)


def score_self_only(W_1s, W_2s, W_3s):
    """Diagonal-only sanity check: just M_i = W_3 W_2 W_1 for each i, no pairing.
    Returns the diagonal-dominance scores themselves (no matching applied)."""
    n = len(W_1s)
    diag = np.zeros(n)
    for i in range(n):
        P = W_3s[i] @ W_2s[i] @ W_1s[i]
        diag[i] = abs(np.trace(P)) / (np.linalg.norm(P, 'fro') + 1e-12)
    return diag


# ----------------------------------------------------------------- stages
def stage_matrices(stage, block_type, mode, fold_bn):
    """Build the list of per-block matrix tuples for a given stage."""
    out = []
    skipped = 0
    for block in stage:
        if getattr(block, 'downsample', None) is not None:
            skipped += 1
            continue
        if block_type == 'bottleneck':
            tup = extract_bottleneck_block_matrices(block, mode=mode, fold_bn=fold_bn)
        else:
            tup = extract_basicblock_matrices(block, mode=mode, fold_bn=fold_bn)
        out.append(tup)
    return out, skipped


# ----------------------------------------------------------------- main per stage
def score_stage(blocks, block_type, mode):
    """Compute the d(i, j) score matrix for a stage's blocks.
    Returns (M_score, label) where label describes the extraction.
    Returns None if the extraction does not apply (e.g. spatial_compose on
    blocks where the composed kernel has non-square channel dims).
    """
    n = len(blocks)
    if mode == 'spatial_compose':
        # Score from the composed kernels; can only handle equal-channel products
        if block_type == 'bottleneck':
            # blocks[i] = (None, None, None, K_i)
            # We need the composed triple kernel for each block, then compare.
            # For pairing across blocks we'd need a single triple kernel from
            # each (already what each entry is). Pair d(i, j): how
            # 'identity-like' is block i's kernel when sitting in block j's slot?
            # Simplest cross score: use block i's composed triple kernel as the
            # diagonal (correct) and block j's as off-diagonal proxies.
            # But that's not really a pairing problem -- it's self-scoring.
            # Skip cross-block spatial-compose for triples; return self-diag.
            return None, 'spatial_compose_triple_skipped'
        else:
            # BasicBlock: (None, None, K_i). Same issue -- composed kernel is
            # already the whole branch; no cross-block product to take.
            return None, 'spatial_compose_pair_skipped'

    if block_type == 'bottleneck':
        W_1s = [t[0] for t in blocks]
        W_2s = [t[1] for t in blocks]
        W_3s = [t[2] for t in blocks]
        # Two scores: TRIPLE (W_3 W_2 W_1 with j-side W_3,W_2 tied) and
        # PAIRWISE-ENDPOINTS (W_3 . W_1, what we tried originally)
        M_triple = score_triple_endpoint(W_1s, W_2s, W_3s)
        M_pair_endpoints = score_pair_matrix(W_1s, W_3s)
        return {'triple': M_triple, 'endpoints': M_pair_endpoints}, mode
    else:
        W_1s = [t[0] for t in blocks]
        W_2s = [t[1] for t in blocks]
        M_pair = score_pair_matrix(W_1s, W_2s)
        return {'pair': M_pair}, mode


# ----------------------------------------------------------------- runner
def run_model(name, model, block_type):
    print(f"\n{'='*78}\n{name}  (block: {block_type})\n{'='*78}")
    results = {'model': name, 'block_type': block_type, 'stages': {}}

    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        stage = getattr(model, stage_name)
        results['stages'][stage_name] = {}
        for mode in ('channel_sum', 'center_tap'):
            blocks, skipped = stage_matrices(stage, block_type, mode, fold_bn=True)
            n = len(blocks)
            if n < 3:
                continue
            scored, _ = score_stage(blocks, block_type, mode)
            if scored is None:
                continue
            entry = {}
            for score_name, M in scored.items():
                pair_acc, pair_sep = hungarian_accuracy(M)
                auc = auc_correct_vs_incorrect(M)
                chance = 1.0 / n
                entry[score_name] = {
                    'n': n,
                    'pair_acc': pair_acc,
                    'pair_sep': pair_sep,
                    'auc': auc,
                    'chance': chance,
                    'acc_over_chance': pair_acc / chance,
                }
                print(f"  {stage_name:7s} mode={mode:11s} score={score_name:10s}  "
                      f"n={n:2d}  pair_acc={pair_acc:.0%} ({pair_acc/chance:4.1f}x chance)  "
                      f"sep={pair_sep:+.3f}  AUC={auc:.3f}")
            results['stages'][stage_name][mode] = entry
        print()
    return results


# ----------------------------------------------------------------- figure
def make_figure(all_results):
    rows = []
    for r in all_results:
        for stage_name, modes in r['stages'].items():
            for mode, scores in modes.items():
                for score_name, m in scores.items():
                    rows.append({
                        'model':       r['model'],
                        'stage':       stage_name,
                        'mode':        mode,
                        'score':       score_name,
                        'n':           m['n'],
                        'pair_acc':    m['pair_acc'],
                        'auc':         m['auc'],
                        'sep':         m['pair_sep'],
                        'chance':      m['chance'],
                        'acc_over_ch': m['acc_over_chance'],
                    })
    if not rows:
        print("No rows to plot")
        return

    # Bar chart: accuracy / chance for each (model, stage, mode, score)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Build sortable labels and group by (model, stage)
    labels = [f"{r['model']}\n{r['stage']}\nn={r['n']}" for r in rows]
    methods = [f"{r['mode']}/{r['score']}" for r in rows]
    acc_ratio = [r['acc_over_ch'] for r in rows]
    auc_vals = [r['auc'] for r in rows]

    method_palette = {}
    palette_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for k, m in enumerate(sorted(set(methods))):
        method_palette[m] = palette_colors[k % len(palette_colors)]
    bar_colors = [method_palette[m] for m in methods]

    x = range(len(rows))
    axes[0].bar(x, acc_ratio, color=bar_colors, alpha=0.85)
    axes[0].axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.5, label='chance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=6, rotation=90)
    axes[0].set_ylabel('pair_acc / chance', fontsize=9)
    axes[0].set_title('Acc/chance per (model, stage, extraction, score)', fontsize=10)
    axes[0].grid(True, axis='y', alpha=0.3)
    # Legend (single point per method)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=m) for m, c in method_palette.items()]
    axes[0].legend(handles=legend_handles, fontsize=8, loc='upper right')

    axes[1].bar(x, auc_vals, color=bar_colors, alpha=0.85)
    axes[1].axhline(0.5, color='k', lw=0.7, ls='--', alpha=0.5, label='AUC=0.5 (random)')
    axes[1].axhline(1.0, color='C2', lw=0.7, ls=':', alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6, rotation=90)
    axes[1].set_ylabel('AUC (correct vs incorrect)', fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig('figures/fig_resnet_extraction_ablation.png',
                dpi=140, bbox_inches='tight')
    fig.savefig('figures/fig_resnet_extraction_ablation.pdf',
                bbox_inches='tight')
    print("\nSaved figures/fig_resnet_extraction_ablation.{png,pdf}")


# ----------------------------------------------------------------- main
if __name__ == '__main__':
    import torchvision.models as M

    MODEL_SPECS = [
        ('resnet18',  M.resnet18,  'IMAGENET1K_V1', 'basicblock'),
        ('resnet34',  M.resnet34,  'IMAGENET1K_V1', 'basicblock'),
        ('resnet50',  M.resnet50,  'IMAGENET1K_V2', 'bottleneck'),
        ('resnet101', M.resnet101, 'IMAGENET1K_V2', 'bottleneck'),
        ('resnet152', M.resnet152, 'IMAGENET1K_V2', 'bottleneck'),
    ]

    all_results = []
    for name, builder, weights_str, block_type in MODEL_SPECS:
        print(f"\nLoading {name} ({weights_str})...")
        model = builder(weights=weights_str)
        model.eval()
        r = run_model(name, model, block_type)
        all_results.append(r)

        with open('results/resnet_extraction_ablation.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        del model; gc.collect()

    make_figure(all_results)

    print('\n' + '='*78)
    print('SUMMARY: where does the diagonal-dominance signal survive?')
    print('='*78)
    for r in all_results:
        print(f"\n{r['model']} ({r['block_type']}):")
        for stage_name, modes in r['stages'].items():
            for mode, scores in modes.items():
                for score_name, m in scores.items():
                    flag = ' <==' if m['acc_over_chance'] >= 2.0 else ''
                    print(f"  {stage_name:7s} mode={mode:11s} score={score_name:10s}  "
                          f"n={m['n']:2d}  acc={m['pair_acc']:.0%} "
                          f"({m['acc_over_chance']:4.1f}x chance)  AUC={m['auc']:.3f}{flag}")
