"""
Random-init control for the ResNet-101 factor ablation.

The critique was sharp and correct: the trained-only factor ablation
showed self-similarity scores (W_i W_i^T) at 100% pair_acc, but that
might be trivially true at random initialization (any matrix is
unique to itself). This script tests that.

For each seed, score the same factor subsets on a RANDOMLY initialized
ResNet-101 layer3:

  cross-factor tests (the meaningful ones):
    W_3 W_1            (endpoint, no W_2)
    W_3 W_2 W_1        (full bottleneck triple)

  self-similarity controls (should trivially work at random init):
    W_1 (W_1)^T
    W_2 (W_2)^T
    W_3 (W_3)^T
    (W_2 W_1) (W_2 W_1)^T
    (W_3 W_2) (W_3 W_2)^T

If self-similarity controls hit ~100% at random init while cross-factor
tests stay at chance, that proves the self-similarity rows are identity
controls rather than training-induced fingerprints.

Outputs:
  results/deepdive_resnet_factors_random.json
"""
import json, gc, importlib.util, os
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

Path("results").mkdir(exist_ok=True)
torch.set_grad_enabled(False)

# load helpers
spec = importlib.util.spec_from_file_location(
    'rea', os.path.join('experiments', 'resnet_extraction_ablation.py'))
_rea = importlib.util.module_from_spec(spec); spec.loader.exec_module(_rea)
fold_bn_scale = _rea.fold_bn_scale
conv_to_matrix_channel_sum = _rea.conv_to_matrix_channel_sum

N_SEEDS = 5


def self_ratio(P):
    return abs(np.trace(P)) / (np.linalg.norm(P, 'fro') + 1e-12)


def diag_dominance_score(blocks):
    """Return a dict of subset_name -> n x n score matrix for a list of
    blocks where each block is a (W_1, W_2, W_3) tuple."""
    n = len(blocks)
    W_1s = [b[0] for b in blocks]
    W_2s = [b[1] for b in blocks]
    W_3s = [b[2] for b in blocks]

    out = {}

    # Cross-factor pairing
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = (W_3s[j] @ W_2s[j]) @ W_1s[i]
            M[i, j] = self_ratio(P)
    out['W_3 W_2 W_1 (full triple)'] = M

    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = W_3s[j] @ W_1s[i]
            M[i, j] = self_ratio(P)
    out['W_3 W_1 (endpoint)'] = M

    # Self-similarity controls
    for name, mats in [
        ('W_1 self-sim', W_1s),
        ('W_2 self-sim', W_2s),
        ('W_3 self-sim', W_3s),
    ]:
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P = mats[j] @ mats[i].T
                M[i, j] = self_ratio(P)
        out[name] = M

    # Pair-composition self-similarity (not cross-block)
    for name, mats in [
        ('W_2 W_1 self-sim', [W_2s[i] @ W_1s[i] for i in range(n)]),
        ('W_3 W_2 self-sim', [W_3s[i] @ W_2s[i] for i in range(n)]),
    ]:
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P = mats[j] @ mats[i].T
                M[i, j] = self_ratio(P)
        out[name] = M

    return out


def evaluate(M):
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M); off = M[~np.eye(n, dtype=bool)]
    pos = diag[:, None]; neg = off[None, :]
    auc = float(((pos > neg).sum() + 0.5*(pos == neg).sum())
                 / (pos.size * neg.size))
    return {'pair_acc': pair_acc, 'auc': auc,
            'mean_correct': float(diag.mean()),
            'mean_incorrect': float(off.mean())}


if __name__ == '__main__':
    import torchvision.models as M

    print(f"Random-init ResNet-101 layer3 factor ablation ({N_SEEDS} seeds)")
    print("=" * 78)

    all_seeds = []
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        model = M.resnet101(weights=None)
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
        scores = diag_dominance_score(blocks)
        seed_res = {}
        for name, score_M in scores.items():
            seed_res[name] = evaluate(score_M)
        all_seeds.append(seed_res)
        # print
        print(f"\n  seed {s} (n={n}):")
        for name, r in seed_res.items():
            print(f"    {name:<32s}  acc={r['pair_acc']:>4.0%}  AUC={r['auc']:.3f}")

    # aggregate
    print()
    print("=" * 78)
    print("AGGREGATE (mean +/- std over seeds)")
    print("=" * 78)
    names = list(all_seeds[0].keys())
    agg = {}
    for name in names:
        accs = np.array([s[name]['pair_acc']     for s in all_seeds])
        aucs = np.array([s[name]['auc']          for s in all_seeds])
        mcs  = np.array([s[name]['mean_correct'] for s in all_seeds])
        mis  = np.array([s[name]['mean_incorrect'] for s in all_seeds])
        agg[name] = {
            'mean_pair_acc': float(accs.mean()), 'std_pair_acc': float(accs.std(ddof=1)),
            'mean_auc':      float(aucs.mean()), 'std_auc':      float(aucs.std(ddof=1)),
            'mean_correct':  float(mcs.mean()),
            'mean_incorrect': float(mis.mean()),
        }
        flag = ''
        if 'self-sim' in name and agg[name]['mean_pair_acc'] > 0.5:
            flag = '   <-- self-sim works EVEN at random init (identity control)'
        elif 'self-sim' not in name and agg[name]['mean_pair_acc'] > 0.5:
            flag = '   <-- cross-factor signal present (would be surprising)'
        print(f"  {name:<32s}  acc={agg[name]['mean_pair_acc']:.0%}+-{agg[name]['std_pair_acc']:.0%}  "
              f"AUC={agg[name]['mean_auc']:.3f}+-{agg[name]['std_auc']:.3f}{flag}")

    json.dump({
        'n_seeds': N_SEEDS,
        'n_blocks': 22,
        'chance': 1.0/22,
        'aggregate': agg,
        'per_seed': all_seeds,
    }, open('results/deepdive_resnet_factors_random.json', 'w'), indent=2)
    print("\nSaved results/deepdive_resnet_factors_random.json")
