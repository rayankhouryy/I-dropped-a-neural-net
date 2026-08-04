"""
Test diagonal-dominance pairing on SOTA ResNets with BatchNorm.

Issue #5: The "elephant in the room." Every published ResNet uses
BatchNorm; the paper's current evidence is on residual MLPs without
normalization. We need to know whether the diagonal-dominance signal
survives a real Conv + BN residual block.

Pipeline for each Bottleneck (ResNet-50/101/152):

    y = x + BN_3(W_3 * BN_2(W_2 * ReLU(BN_1(W_1 * x))))

At inference, BN with running statistics is just a per-channel affine
transform   BN(z) = gamma * (z - mu) / sqrt(var + eps) + beta. We fold
the gamma/sqrt(var+eps) scale factor into the adjacent conv's OUTPUT
channels, then sum the conv kernels over the 3x3 (or 1x1) spatial
extent to get channel-mixing matrices.

For a Bottleneck block:
    W_in_eff   = (gamma_1 / sqrt(var_1 + eps)) * W_1     (mid, planes, 1, 1)
    W_out_eff  = (gamma_3 / sqrt(var_3 + eps)) * W_3     (planes, mid, 1, 1)

After summing over kernel positions both become channel matrices of
shape (mid, planes) and (planes, mid) respectively. Their product
W_out @ W_in is (planes, planes) -- the analog of our ResNet
W_out @ W_in in the paper.

We pair within each stage (stages have different channel counts, so
cross-stage pairing is not well-defined). Hungarian on the
diagonal-dominance matrix gives a pair_acc per stage.

For BasicBlock (ResNet-18/34), the structure is two convs (no
bottleneck) so we just use W_1 / W_2 directly.

Outputs:
    figures/fig_torchvision_resnet_pairing.{png,pdf}
    results/torchvision_resnet_pairing.json
"""

import json, gc
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

torch.set_grad_enabled(False)


# ---------------------------------------------------------------- BN folding
def fold_bn_scale(conv_weight: torch.Tensor, bn: torch.nn.BatchNorm2d) -> torch.Tensor:
    """Return conv weight with the BN scaling factor folded into output channels.

    BN(z)_c = gamma_c * (z_c - mu_c) / sqrt(var_c + eps) + beta_c.
    For our diagonal-dominance computation we only care about the linear
    multiplicative factor: gamma / sqrt(var + eps) on each output channel.
    The (additive) mean/bias terms wash out of the d(i,j) computation
    because |tr(W_out W_in)| and ||W_out W_in||_F are both linear in W's.
    """
    eps = bn.eps
    scale = bn.weight / torch.sqrt(bn.running_var + eps)  # (C_out,)
    # broadcast scale over (out, in, kH, kW)
    return conv_weight * scale.view(-1, 1, 1, 1)


def conv_channel_matrix(W: torch.Tensor) -> np.ndarray:
    """Collapse a conv kernel (out, in, kH, kW) to a channel-mixing matrix
    (out, in) by summing over kernel positions.

    For 1x1 convs this is a no-op. For 3x3 convs it integrates the kernel
    along the spatial axis -- the natural "DC-channel" projection.
    """
    return W.sum(dim=(2, 3)).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- extraction
def extract_stage_weights(stage, block_type: str):
    """Return W_in_list, W_out_list as numpy channel-mixing matrices.

    Skips the FIRST block of each stage (the "downsample" block) because
    its conv1 input dimension differs from the rest of the stage --
    blocks within a stage need consistent shapes for the d(i,j) matrix
    to be well-defined.
    """
    W_ins, W_outs = [], []
    skipped = 0

    for k, block in enumerate(stage):
        # Skip blocks that change channel dim (always the first block of
        # a stage; identified by having a non-None downsample shortcut).
        if getattr(block, 'downsample', None) is not None:
            skipped += 1
            continue

        if block_type == 'bottleneck':
            # Bottleneck: conv1 (1x1, reduce) -> BN1 -> ReLU -> conv2 (3x3) -> BN2 -> ReLU -> conv3 (1x1, expand) -> BN3
            w1 = fold_bn_scale(block.conv1.weight, block.bn1)   # (mid, planes, 1, 1)
            w3 = fold_bn_scale(block.conv3.weight, block.bn3)   # (planes, mid, 1, 1)
            W_in  = conv_channel_matrix(w1)    # (mid, planes)
            W_out = conv_channel_matrix(w3)    # (planes, mid)
        elif block_type == 'basicblock':
            # BasicBlock: conv1 (3x3) -> BN1 -> ReLU -> conv2 (3x3) -> BN2
            w1 = fold_bn_scale(block.conv1.weight, block.bn1)   # (planes, planes, 3, 3)
            w2 = fold_bn_scale(block.conv2.weight, block.bn2)   # (planes, planes, 3, 3)
            W_in  = conv_channel_matrix(w1)    # (planes, planes)
            W_out = conv_channel_matrix(w2)    # (planes, planes)
        else:
            raise ValueError(block_type)
        W_ins.append(W_in)
        W_outs.append(W_out)

    # Sanity: every kept block must have the same W_in / W_out shape.
    if W_ins:
        shapes_in  = {w.shape for w in W_ins}
        shapes_out = {w.shape for w in W_outs}
        assert len(shapes_in)  == 1, f"inconsistent W_in shapes: {shapes_in}"
        assert len(shapes_out) == 1, f"inconsistent W_out shapes: {shapes_out}"

    return W_ins, W_outs, skipped


# ---------------------------------------------------------------- pairing math
def diag_dominance_matrix(W_ins, W_outs):
    n = len(W_ins)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prod = W_outs[j] @ W_ins[i]
            tr = abs(np.trace(prod))
            fr = np.linalg.norm(prod, 'fro') + 1e-12
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
        'pair_acc':       pair_acc,
        'pair_sep':       pair_sep,
        'mean_correct':   float(diag.mean()),
        'mean_incorrect': float(off.mean()),
        'correct_pairs':  int(pair_acc * n),
        'total_pairs':    n,
        'assignment':     col.tolist(),
    }


def trace_signs(W_ins, W_outs):
    traces = []
    for i in range(len(W_ins)):
        prod = W_outs[i] @ W_ins[i]
        traces.append(float(np.trace(prod)))
    traces = np.array(traces)
    return {
        'traces': traces.tolist(),
        'mean_trace': float(traces.mean()),
        'frac_negative': float((traces < 0).mean()),
    }


# ---------------------------------------------------------------- runner
def run_model(name: str, model, block_type: str):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    out = {'model': name, 'block_type': block_type, 'stages': []}

    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        stage = getattr(model, stage_name)
        W_ins, W_outs, skipped = extract_stage_weights(stage, block_type)
        n_blocks = len(W_ins)
        if n_blocks < 3:
            print(f"  {stage_name:7s}: only {n_blocks} interior blocks (skipped {skipped}); too few")
            continue
        d_planes = W_ins[0].shape[1]  # input channels  (= residual stream width)
        d_mid    = W_ins[0].shape[0]  # bottleneck channels
        M = diag_dominance_matrix(W_ins, W_outs)
        res = evaluate(M)
        tr  = trace_signs(W_ins, W_outs)
        chance = 1.0 / n_blocks

        # Frobenius baseline for comparison
        M_frob = np.zeros_like(M)
        for i in range(n_blocks):
            for j in range(n_blocks):
                M_frob[i, j] = np.linalg.norm(W_outs[j] @ W_ins[i], 'fro')
        _, col_f = linear_sum_assignment(M_frob)  # minimize
        frob_acc = float((col_f == np.arange(n_blocks)).mean())

        entry = {
            'stage': stage_name,
            'n_blocks': n_blocks,
            'planes': d_planes,
            'mid': d_mid,
            'skipped_downsample_blocks': skipped,
            'chance': chance,
            'diag_dominance': res,
            'frobenius_pair_acc': frob_acc,
            'trace_analysis': tr,
        }
        out['stages'].append(entry)
        out[f'_{stage_name}_M'] = M  # not JSON-serializable; for figure only

        print(f"  {stage_name:7s}: n={n_blocks:2d} (skip {skipped}), planes={d_planes:4d}, mid={d_mid:4d}  "
              f"|  diag_dom={res['correct_pairs']}/{n_blocks} "
              f"({res['pair_acc']:.0%}, chance {chance:.1%})  "
              f"sep={res['pair_sep']:+.3f}  "
              f"frob={frob_acc:.0%}  "
              f"neg_traces={tr['frac_negative']:.0%}")

    return out


def run_random_init(builder, name='resnet50'):
    print(f"\n{'='*70}\nRANDOM INIT BASELINE: {name}\n{'='*70}")
    model = builder(weights=None)
    model.eval()
    # use the largest stage (most interior blocks) for the strongest test
    block_type = 'bottleneck'
    best = None
    for stage_name in ['layer3', 'layer2', 'layer4', 'layer1']:
        stage = getattr(model, stage_name)
        W_ins, W_outs, skipped = extract_stage_weights(stage, block_type)
        if len(W_ins) >= 3:
            M = diag_dominance_matrix(W_ins, W_outs)
            res = evaluate(M)
            tr = trace_signs(W_ins, W_outs)
            n = len(W_ins)
            print(f"  {stage_name} (n={n}): pair_acc={res['correct_pairs']}/{n} "
                  f"({res['pair_acc']:.1%}, chance {1/n:.1%}), sep={res['pair_sep']:+.3f}, "
                  f"neg_traces={tr['frac_negative']:.0%}")
            if best is None:
                best = (stage_name, n, res, tr)
    del model; gc.collect()
    if best is None:
        return None
    stage_name, n, res, tr = best
    return {
        'model': f'{name}-random-init',
        'stage': stage_name,
        'pair_acc': res['pair_acc'],
        'pair_sep': res['pair_sep'],
        'frac_negative': tr['frac_negative'],
        'mean_correct':   res['mean_correct'],
        'mean_incorrect': res['mean_incorrect'],
        'n_blocks': n,
    }


# ---------------------------------------------------------------- figure
def make_figure(all_results):
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 4, figsize=(16, 3.6 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, r in enumerate(all_results):
        for sidx, stage_entry in enumerate(r['stages'][:4]):
            stage = stage_entry['stage']
            n = stage_entry['n_blocks']
            ax = axes[idx, sidx]
            M = r.get(f'_{stage}_M')
            if M is None:
                ax.axis('off')
                continue
            im = ax.imshow(M, cmap='magma', aspect='equal')
            dd = stage_entry['diag_dominance']
            ax.set_title(
                f'{r["model"]} · {stage}\n'
                f'n={n}, planes={stage_entry["planes"]}, mid={stage_entry["mid"]}\n'
                f'pair_acc={dd["pair_acc"]:.0%} (chance {stage_entry["chance"]:.0%}), '
                f'sep={dd["pair_sep"]:+.2f}',
                fontsize=8,
            )
            ax.set_xlabel(r'$W_\mathrm{out}$ idx $j$', fontsize=8)
            ax.set_ylabel(r'$W_\mathrm{in}$ idx $i$', fontsize=8)
            ax.tick_params(labelsize=6)
            plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    plt.tight_layout()
    fig.savefig('figures/fig_torchvision_resnet_pairing.png',
                dpi=140, bbox_inches='tight')
    fig.savefig('figures/fig_torchvision_resnet_pairing.pdf',
                bbox_inches='tight')
    print("\nSaved figures/fig_torchvision_resnet_pairing.{png,pdf}")


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    import torchvision.models as M

    # (name, builder, weights_enum, block_type)
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

        # incremental save (drop matrices for JSON)
        serializable = []
        for x in all_results:
            xx = {k: v for k, v in x.items() if not k.startswith('_')}
            serializable.append(xx)
        with open('results/torchvision_resnet_pairing.json', 'w') as f:
            json.dump({'pretrained': serializable}, f, indent=2)

        del model; gc.collect()

    # Random init baseline on resnet50 layer3
    random_baseline = run_random_init(M.resnet50, 'resnet50')

    # Final save
    serializable = []
    for x in all_results:
        xx = {k: v for k, v in x.items() if not k.startswith('_')}
        serializable.append(xx)
    with open('results/torchvision_resnet_pairing.json', 'w') as f:
        json.dump({
            'pretrained': serializable,
            'random_init_baseline': random_baseline,
        }, f, indent=2)
    print("\nSaved results/torchvision_resnet_pairing.json")

    make_figure(all_results)

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for r in all_results:
        print(f"\n{r['model']} ({r['block_type']}):")
        for s in r['stages']:
            dd = s['diag_dominance']
            print(f"  {s['stage']:7s}: {dd['correct_pairs']:2d}/{s['n_blocks']:2d} "
                  f"({dd['pair_acc']:.0%}, chance {s['chance']:.0%})  "
                  f"sep={dd['pair_sep']:+.3f}  "
                  f"neg_tr={s['trace_analysis']['frac_negative']:.0%}  "
                  f"frob={s['frobenius_pair_acc']:.0%}")
    rb = random_baseline
    print(f"\nresnet50 (random init), layer3 (n={rb['n_blocks']}): "
          f"pair_acc={rb['pair_acc']:.1%} (chance {1/rb['n_blocks']:.1%}), "
          f"sep={rb['pair_sep']:+.3f}, neg_tr={rb['frac_negative']:.0%}")
