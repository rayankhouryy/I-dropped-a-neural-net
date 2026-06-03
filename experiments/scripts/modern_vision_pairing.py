"""
Issue #14 closure: ConvNeXt-T and ViT-B pairing.

Tests whether the architecture-aware diagonal-dominance fingerprint
extends to modern vision architectures with LayerNorm-style
normalization (ConvNeXt-T) and a vision transformer (ViT-B-16).

Pipeline per model:

  ConvNeXt-T CNBlock (depthwise 7x7 conv, LayerNorm, Linear, GELU, Linear):
    The Linear-GELU-Linear path is structurally identical to a
    transformer MLP. We pair on M = W_2 . W_1 where
      W_1: (4d, d) is mlp.block[3].weight
      W_2: (d, 4d) is mlp.block[5].weight
    The depthwise conv is per-channel (groups=dim) so it does not mix
    channels; we ignore it for the channel-mixing matrix.

    Tested stages (CNBlocks per stage in features[]):
      features[1] -- 3 blocks, dim=96
      features[3] -- 3 blocks, dim=192
      features[5] -- 9 blocks, dim=384  <- largest, primary test
      features[7] -- 3 blocks, dim=768

  ViT-B-16:
    12 standard transformer encoder blocks. Three pairing paths:
      MLP   : M = W_2 . W_1   (768x768)
      V<->O : M = W_O . W_V   (768x768)
      Q<->K : M = W_Q . W_K^T (768x768, control)

    in_proj_weight packs (Q | K | V) on the output axis.

Random-init baselines for both (3 seeds each).

Outputs:
  figures/fig_modern_vision_pairing.{png,pdf}
  results/modern_vision_pairing.json
"""

import json, gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
torch.set_grad_enabled(False)

N_SEEDS = 3


# ----------------------------------------------------------------- math
def diag_dominance_matrix(A_list, B_list):
    """d(i, j) = |tr(B[j] @ A[i])| / ||B[j] @ A[i]||_F"""
    n = len(A_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = B_list[j] @ A_list[i]
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def evaluate(M):
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off  = M[~np.eye(n, dtype=bool)]
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    return {
        'n': n,
        'chance': 1.0 / n,
        'pair_acc':       pair_acc,
        'acc_over_chance': pair_acc * n,
        'pair_sep':       pair_sep,
        'mean_correct':   float(diag.mean()),
        'mean_incorrect': float(off.mean()),
    }


def auc_correct_vs_incorrect(M):
    n = M.shape[0]
    diag = np.diag(M)
    off  = M[~np.eye(n, dtype=bool)]
    pos = diag[:, None]
    neg = off[None, :]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    total = pos.size * neg.size
    return float(wins / total)


def trace_signs(A_list, B_list):
    traces = [float(np.trace(B_list[i] @ A_list[i])) for i in range(len(A_list))]
    traces = np.array(traces)
    return {
        'mean_trace': float(traces.mean()),
        'frac_negative': float((traces < 0).mean()),
        'traces': traces.tolist(),
    }


# ----------------------------------------------------------------- ConvNeXt
def convnext_extract_stage(stage):
    """Return W_1s, W_2s as numpy channel-mixing matrices for every CNBlock
    in a ConvNeXt stage (Sequential of CNBlocks)."""
    W1s, W2s = [], []
    for blk in stage:
        # blk.block is Sequential: Conv2d, Permute, LayerNorm, Linear, GELU, Linear, Permute
        w1 = blk.block[3].weight.detach().float().cpu().numpy()   # (4d, d)
        w2 = blk.block[5].weight.detach().float().cpu().numpy()   # (d, 4d)
        W1s.append(w1)
        W2s.append(w2)
    return W1s, W2s


def run_convnext(model_name, builder, weights, label):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    model = builder(weights=weights)
    model.eval()

    out = {'model': model_name, 'label': label, 'arch_family': 'convnext',
           'stages': {}}

    # Find CNBlock stages: features[1], features[3], features[5], features[7]
    stages = {
        'stage1': model.features[1],
        'stage2': model.features[3],
        'stage3': model.features[5],   # largest
        'stage4': model.features[7],
    }
    for sname, stage in stages.items():
        W1s, W2s = convnext_extract_stage(stage)
        n = len(W1s)
        d = W1s[0].shape[1]
        if n < 3:
            continue
        M_mlp = diag_dominance_matrix(W1s, W2s)
        res = evaluate(M_mlp)
        res['auc'] = auc_correct_vs_incorrect(M_mlp)
        tr = trace_signs(W1s, W2s)
        out['stages'][sname] = {
            'd_model': int(d),
            'mlp':     res,
            'trace':   tr,
            '_M': M_mlp,
        }
        print(f"  {sname:7s} (d={d:4d}, n={n:2d}): MLP pair_acc={res['pair_acc']:.0%} "
              f"({res['acc_over_chance']:.1f}x)  sep={res['pair_sep']:+.3f}  "
              f"AUC={res['auc']:.3f}  neg_tr={tr['frac_negative']:.0%}")

    del model; gc.collect()
    return out


# ----------------------------------------------------------------- ViT
def vit_extract_qkv(encoder_layer):
    """Split in_proj_weight into W_Q, W_K, W_V each shape (d, d)."""
    in_proj = encoder_layer.self_attention.in_proj_weight.detach().float().cpu().numpy()
    d = in_proj.shape[1]
    W_q, W_k, W_v = np.split(in_proj, 3, axis=0)  # each (d, d)
    W_o = encoder_layer.self_attention.out_proj.weight.detach().float().cpu().numpy()
    return W_q, W_k, W_v, W_o


def vit_extract_mlp(encoder_layer):
    w1 = encoder_layer.mlp[0].weight.detach().float().cpu().numpy()   # (mlp_dim, d)
    w2 = encoder_layer.mlp[3].weight.detach().float().cpu().numpy()   # (d, mlp_dim)
    return w1, w2


def run_vit(model_name, builder, weights, label):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    model = builder(weights=weights)
    model.eval()

    layers = list(model.encoder.layers)
    n = len(layers)
    d = layers[0].self_attention.embed_dim
    print(f"  encoder layers: {n}, d_model: {d}")

    W_Q, W_K, W_V, W_O, W_mlp1, W_mlp2 = [], [], [], [], [], []
    for lay in layers:
        wq, wk, wv, wo = vit_extract_qkv(lay)
        w1, w2 = vit_extract_mlp(lay)
        W_Q.append(wq); W_K.append(wk); W_V.append(wv); W_O.append(wo)
        W_mlp1.append(w1); W_mlp2.append(w2)

    # MLP pairing
    M_mlp = diag_dominance_matrix(W_mlp1, W_mlp2)
    r_mlp = evaluate(M_mlp);  r_mlp['auc']  = auc_correct_vs_incorrect(M_mlp)
    tr_mlp = trace_signs(W_mlp1, W_mlp2)

    # V <-> O pairing
    M_vo = diag_dominance_matrix(W_V, W_O)
    r_vo = evaluate(M_vo);    r_vo['auc']  = auc_correct_vs_incorrect(M_vo)
    tr_vo = trace_signs(W_V, W_O)

    # Q <-> K pairing (transposed)
    W_K_T = [w.T for w in W_K]
    M_qk = diag_dominance_matrix(W_Q, W_K_T)
    r_qk = evaluate(M_qk);    r_qk['auc']  = auc_correct_vs_incorrect(M_qk)
    tr_qk = trace_signs(W_Q, W_K_T)

    print(f"  MLP   : {r_mlp['pair_acc']:.0%} ({r_mlp['acc_over_chance']:.1f}x)  "
          f"sep={r_mlp['pair_sep']:+.3f}  AUC={r_mlp['auc']:.3f}  "
          f"neg_tr={tr_mlp['frac_negative']:.0%}")
    print(f"  V<->O : {r_vo['pair_acc']:.0%} ({r_vo['acc_over_chance']:.1f}x)  "
          f"sep={r_vo['pair_sep']:+.3f}  AUC={r_vo['auc']:.3f}  "
          f"neg_tr={tr_vo['frac_negative']:.0%}")
    print(f"  Q<->K : {r_qk['pair_acc']:.0%} ({r_qk['acc_over_chance']:.1f}x)  "
          f"sep={r_qk['pair_sep']:+.3f}  AUC={r_qk['auc']:.3f}  "
          f"neg_tr={tr_qk['frac_negative']:.0%}")

    out = {
        'model': model_name, 'label': label, 'arch_family': 'vit',
        'n_layers': n, 'd_model': d,
        'mlp':   {**r_mlp,  'trace': tr_mlp,  '_M': M_mlp},
        'vo':    {**r_vo,   'trace': tr_vo,   '_M': M_vo},
        'qk':    {**r_qk,   'trace': tr_qk,   '_M': M_qk},
    }
    del model; gc.collect()
    return out


# ----------------------------------------------------------------- random baselines
def random_baseline_convnext(builder, seeds=N_SEEDS):
    print(f"\n--- random-init ConvNeXt-T baseline ({seeds} seeds) ---")
    accs = []
    aucs = []
    seps = []
    for s in range(seeds):
        torch.manual_seed(s)
        model = builder(weights=None)
        model.eval()
        W1s, W2s = convnext_extract_stage(model.features[5])  # 9-block stage
        M = diag_dominance_matrix(W1s, W2s)
        r = evaluate(M); a = auc_correct_vs_incorrect(M)
        accs.append(r['pair_acc']); aucs.append(a); seps.append(r['pair_sep'])
        print(f"  seed {s}: stage3 acc={r['pair_acc']:.0%} ({r['acc_over_chance']:.1f}x chance)  "
              f"sep={r['pair_sep']:+.3f}  AUC={a:.3f}")
        del model; gc.collect()
    return {
        'n_blocks': len(W1s),
        'chance': 1.0 / len(W1s),
        'mean_acc': float(np.mean(accs)),
        'std_acc':  float(np.std(accs)),
        'mean_auc': float(np.mean(aucs)),
        'mean_sep': float(np.mean(seps)),
    }


def random_baseline_vit(builder, seeds=N_SEEDS):
    print(f"\n--- random-init ViT-B-16 baseline ({seeds} seeds) ---")
    out = {'mlp': [], 'vo': [], 'qk': []}
    for s in range(seeds):
        torch.manual_seed(s)
        model = builder(weights=None)
        model.eval()
        layers = list(model.encoder.layers)
        W_Q, W_K, W_V, W_O, W_mlp1, W_mlp2 = [], [], [], [], [], []
        for lay in layers:
            wq, wk, wv, wo = vit_extract_qkv(lay)
            w1, w2 = vit_extract_mlp(lay)
            W_Q.append(wq); W_K.append(wk); W_V.append(wv); W_O.append(wo)
            W_mlp1.append(w1); W_mlp2.append(w2)
        Mmlp = diag_dominance_matrix(W_mlp1, W_mlp2); rmlp = evaluate(Mmlp); amlp = auc_correct_vs_incorrect(Mmlp)
        Mvo  = diag_dominance_matrix(W_V, W_O);       rvo  = evaluate(Mvo);  avo  = auc_correct_vs_incorrect(Mvo)
        W_K_T = [w.T for w in W_K]
        Mqk  = diag_dominance_matrix(W_Q, W_K_T);     rqk  = evaluate(Mqk);  aqk  = auc_correct_vs_incorrect(Mqk)
        out['mlp'].append({'pair_acc': rmlp['pair_acc'], 'auc': amlp, 'sep': rmlp['pair_sep']})
        out['vo'].append( {'pair_acc': rvo['pair_acc'],  'auc': avo,  'sep': rvo['pair_sep']})
        out['qk'].append( {'pair_acc': rqk['pair_acc'],  'auc': aqk,  'sep': rqk['pair_sep']})
        print(f"  seed {s}: MLP={rmlp['pair_acc']:.0%}  V<->O={rvo['pair_acc']:.0%}  "
              f"Q<->K={rqk['pair_acc']:.0%}  (chance {1/len(layers):.1%})")
        del model; gc.collect()
    agg = {}
    for k in ['mlp', 'vo', 'qk']:
        vals = out[k]
        agg[k] = {
            'mean_acc': float(np.mean([v['pair_acc'] for v in vals])),
            'std_acc':  float(np.std( [v['pair_acc'] for v in vals])),
            'mean_auc': float(np.mean([v['auc']      for v in vals])),
            'mean_sep': float(np.mean([v['sep']      for v in vals])),
        }
    agg['chance'] = 1.0 / len(layers)
    agg['n_layers'] = len(layers)
    return agg


# ----------------------------------------------------------------- figure
def make_figure(convnext_out, vit_out):
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.1))

    # Panel 1: ConvNeXt-T stage3 MLP
    s3 = convnext_out['stages']['stage3']
    M = s3['_M']
    im = axes[0].imshow(M, cmap='magma', aspect='equal', vmin=0)
    axes[0].set_title(
        f"ConvNeXt-T / stage3\n"
        f"CNBlock MLP: $M = W_2 W_1$\n"
        f"$n=9$,  pair acc = {s3['mlp']['pair_acc']:.0%},  AUC = {s3['mlp']['auc']:.2f}",
        fontsize=10.5, pad=8,
    )
    axes[0].set_xlabel(r'$W_2$ idx $j$', fontsize=9)
    axes[0].set_ylabel(r'$W_1$ idx $i$', fontsize=9)
    plt.colorbar(im, ax=axes[0], shrink=0.78, pad=0.04, label=r'$d(i,j)$')

    # Panel 2: ViT-B MLP
    M = vit_out['mlp']['_M']
    im = axes[1].imshow(M, cmap='magma', aspect='equal', vmin=0)
    axes[1].set_title(
        f"ViT-B-16 MLP\n"
        f"Transformer MLP: $M = W_2 W_1$\n"
        f"$n=12$, pair acc = {vit_out['mlp']['pair_acc']:.0%}, AUC = {vit_out['mlp']['auc']:.2f}",
        fontsize=10.5, pad=8,
    )
    axes[1].set_xlabel(r'$W_2$ idx $j$', fontsize=9)
    plt.colorbar(im, ax=axes[1], shrink=0.78, pad=0.04, label=r'$d(i,j)$')

    # Panel 3: ViT-B Attention V<->O
    M = vit_out['vo']['_M']
    im = axes[2].imshow(M, cmap='magma', aspect='equal', vmin=0)
    axes[2].set_title(
        f"ViT-B-16 Attention V$\\leftrightarrow$O\n"
        f"$M = W_O W_V$\n"
        f"$n=12$, pair acc = {vit_out['vo']['pair_acc']:.0%}, AUC = {vit_out['vo']['auc']:.2f}",
        fontsize=10.5, pad=8,
    )
    axes[2].set_xlabel(r'$W_O$ idx $j$', fontsize=9)
    plt.colorbar(im, ax=axes[2], shrink=0.78, pad=0.04, label=r'$d(i,j)$')

    # Panel 4: ViT-B Attention Q<->K
    M = vit_out['qk']['_M']
    im = axes[3].imshow(M, cmap='magma', aspect='equal', vmin=0)
    axes[3].set_title(
        f"ViT-B-16 Attention Q$\\leftrightarrow$K\n"
        f"$M = W_K^T W_Q$\n"
        f"$n=12$, pair acc = {vit_out['qk']['pair_acc']:.0%}, AUC = {vit_out['qk']['auc']:.2f}",
        fontsize=10.5, pad=8,
    )
    axes[3].set_xlabel(r'$W_K$ idx $j$', fontsize=9)
    plt.colorbar(im, ax=axes[3], shrink=0.78, pad=0.04, label=r'$d(i,j)$')

    for ax in axes:
        ax.tick_params(labelsize=8)

    fig.suptitle('Modern vision architectures: ConvNeXt-T and ViT-B-16',
                 fontsize=12.5, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/fig_modern_vision_pairing.png', dpi=160,
                bbox_inches='tight')
    plt.savefig('figures/fig_modern_vision_pairing.pdf',
                bbox_inches='tight')
    # also sync to paper/figures
    import shutil
    shutil.copy('figures/fig_modern_vision_pairing.png',
                'paper/figures/fig_modern_vision_pairing.png')
    shutil.copy('figures/fig_modern_vision_pairing.pdf',
                'paper/figures/fig_modern_vision_pairing.pdf')
    print("Saved figures/fig_modern_vision_pairing.{png,pdf} (also into paper/figures/)")


# ----------------------------------------------------------------- main
if __name__ == '__main__':
    import torchvision.models as M

    # 1. ConvNeXt-T pretrained
    cnx = run_convnext('convnext_tiny', M.convnext_tiny, 'IMAGENET1K_V1',
                       label='ConvNeXt-T (pretrained ImageNet)')

    # 2. ViT-B-16 pretrained
    vit = run_vit('vit_b_16', M.vit_b_16, 'IMAGENET1K_V1',
                  label='ViT-B-16 (pretrained ImageNet)')

    # 3. Random baselines
    cnx_random = random_baseline_convnext(M.convnext_tiny)
    vit_random = random_baseline_vit(M.vit_b_16)

    # Save (drop _M which is numpy array not JSON-serializable)
    def strip_M(d):
        out = {}
        for k, v in d.items():
            if k.startswith('_'):
                continue
            if isinstance(v, dict):
                out[k] = strip_M(v)
            else:
                out[k] = v
        return out

    output = {
        'convnext_tiny':         strip_M(cnx),
        'convnext_tiny_random':  cnx_random,
        'vit_b_16':              strip_M(vit),
        'vit_b_16_random':       vit_random,
    }
    with open('results/modern_vision_pairing.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved results/modern_vision_pairing.json")

    make_figure(cnx, vit)

    print('\n' + '='*78)
    print('SUMMARY: Modern vision architectures')
    print('='*78)

    s3 = cnx['stages']['stage3']
    cn_acc = s3['mlp']['pair_acc']
    cn_n   = s3['mlp']['n']
    cn_rand = cnx_random['mean_acc']
    print(f"\nConvNeXt-T  stage3 ({cn_n} CNBlocks, MLP $W_2 W_1$):")
    print(f"  trained:    pair_acc={cn_acc:.0%}  (chance {1/cn_n:.0%})")
    print(f"  random:     {cn_rand:.0%}+-{cnx_random['std_acc']:.0%}")
    print(f"  AUC:        trained={s3['mlp']['auc']:.3f}  random={cnx_random['mean_auc']:.3f}")

    print(f"\nViT-B-16    encoder ({vit['n_layers']} blocks, d={vit['d_model']}):")
    for k, label in [('mlp', 'MLP'), ('vo', 'Attn V<->O'), ('qk', 'Attn Q<->K')]:
        t = vit[k]; rb = vit_random[k]
        print(f"  {label:14s}: trained={t['pair_acc']:.0%}  random={rb['mean_acc']:.0%}+-{rb['std_acc']:.0%}  "
              f"AUC={t['auc']:.3f} vs {rb['mean_auc']:.3f}")
