"""
ConvNeXt random baseline refinement: 20 seeds + margin diagnostics.

The initial ConvNeXt experiment used only 3 random seeds and reported AUC
~0.62 on random init (above 0.5). To check whether ConvNeXt has a mild
structural bias at initialization (vs. the LayerNorm-based init genuinely
inducing some weak diagonal-dominance structure) we re-run the random
baseline with N_SEEDS=20 and add margin diagnostics: pair separation,
Hungarian assignment gap, per-row top-2 gap.

We also compute the same diagnostics on the trained model for direct
comparison.

Outputs:
  results/convnext_random_extended.json
"""
import json, gc
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

Path("results").mkdir(exist_ok=True)
torch.set_grad_enabled(False)

N_SEEDS = 20


def convnext_extract_stage(stage):
    W1s, W2s = [], []
    for blk in stage:
        w1 = blk.block[3].weight.detach().float().cpu().numpy()   # (4d, d)
        w2 = blk.block[5].weight.detach().float().cpu().numpy()   # (d, 4d)
        W1s.append(w1)
        W2s.append(w2)
    return W1s, W2s


def diag_dominance_matrix(A_list, B_list):
    n = len(A_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = B_list[j] @ A_list[i]
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def diagnostics(M):
    """Compute pair_acc, AUC, pair_sep, Hungarian assignment gap (between
    optimal and 2nd-best assignment cost), and per-row top-2 gap.
    """
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())

    diag = np.diag(M)
    off  = M[~np.eye(n, dtype=bool)]
    # AUC
    pos = diag[:, None]
    neg = off[None, :]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    auc = float(wins / (pos.size * neg.size))

    # Pair separation: min over rows of (diag - max off-diag)
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())

    # Per-row top-2 gap on the SCORE row (diag value minus the highest
    # off-diagonal in that row). Median over rows.
    row_top2_gaps = diag - off_max_per_row
    median_top2_gap = float(np.median(row_top2_gaps))
    mean_top2_gap   = float(np.mean(row_top2_gaps))

    # Hungarian assignment cost gap: cost of optimal vs cost of 2nd-best.
    # 2nd-best heuristic: best assignment that disagrees on at least one row.
    best_cost = -M[np.arange(n), col].sum()
    # Disable best move per row and re-solve: not exact but a reasonable proxy.
    second_best_cost = best_cost
    for k in range(n):
        Mk = M.copy()
        Mk[k, col[k]] = -1e9  # forbid the best assignment for row k
        _, ck = linear_sum_assignment(-Mk)
        cost = -Mk[np.arange(n), ck].sum()
        if cost > second_best_cost:
            second_best_cost = cost
    assignment_gap = best_cost - second_best_cost  # negative => positive gap

    return {
        'n': n,
        'pair_acc':         pair_acc,
        'auc':              auc,
        'pair_sep':         pair_sep,
        'mean_correct':     float(diag.mean()),
        'mean_incorrect':   float(off.mean()),
        'median_top2_gap':  median_top2_gap,
        'mean_top2_gap':    mean_top2_gap,
        'assignment_gap':  -float(assignment_gap),  # report as positive
    }


if __name__ == '__main__':
    import torchvision.models as M

    # ---- trained baseline (diagnostics for the pretrained model) ----
    print("=" * 70)
    print("Trained ConvNeXt-T diagnostics")
    print("=" * 70)
    trained = M.convnext_tiny(weights='IMAGENET1K_V1')
    trained.eval()
    W1s, W2s = convnext_extract_stage(trained.features[5])  # 9-block stage
    M_trained = diag_dominance_matrix(W1s, W2s)
    trained_diag = diagnostics(M_trained)
    print(f"  stage3 (n=9):")
    for k, v in trained_diag.items():
        if k == 'n': continue
        print(f"    {k:18s}: {v:.4f}")
    del trained; gc.collect()

    # ---- random-init baseline with N_SEEDS ----
    print()
    print("=" * 70)
    print(f"Random-init ConvNeXt-T baseline ({N_SEEDS} seeds)")
    print("=" * 70)
    per_seed = []
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        rnd = M.convnext_tiny(weights=None)
        rnd.eval()
        W1s, W2s = convnext_extract_stage(rnd.features[5])
        M_r = diag_dominance_matrix(W1s, W2s)
        d = diagnostics(M_r)
        per_seed.append(d)
        print(f"  seed {s:2d}: pair_acc={d['pair_acc']:.0%}  "
              f"sep={d['pair_sep']:+.3f}  "
              f"AUC={d['auc']:.3f}  "
              f"mean_top2_gap={d['mean_top2_gap']:+.3f}")
        del rnd; gc.collect()

    # ---- aggregate ----
    fields = ['pair_acc', 'auc', 'pair_sep', 'mean_correct', 'mean_incorrect',
              'median_top2_gap', 'mean_top2_gap', 'assignment_gap']
    agg = {}
    for f in fields:
        vals = np.array([d[f] for d in per_seed])
        agg[f] = {
            'mean': float(vals.mean()),
            'std':  float(vals.std(ddof=1)),
            'min':  float(vals.min()),
            'max':  float(vals.max()),
        }

    print()
    print("=" * 70)
    print(f"AGGREGATE OVER {N_SEEDS} SEEDS (random init):")
    print("=" * 70)
    for f in fields:
        a = agg[f]
        print(f"  {f:18s}: {a['mean']:+.4f} +/- {a['std']:.4f}   "
              f"(min={a['min']:+.4f}, max={a['max']:+.4f})")

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    chance = 1.0 / per_seed[0]['n']
    n = per_seed[0]['n']
    print(f"  ConvNeXt-T stage3, n={n}, chance = {chance:.1%}")
    print()
    print(f"  pair_acc:   trained = {trained_diag['pair_acc']:.0%}  "
          f"random = {agg['pair_acc']['mean']:.1%} +/- {agg['pair_acc']['std']:.1%}")
    print(f"  AUC:        trained = {trained_diag['auc']:.3f}  "
          f"random = {agg['auc']['mean']:.3f} +/- {agg['auc']['std']:.3f}")
    print(f"  pair_sep:   trained = {trained_diag['pair_sep']:+.3f}  "
          f"random = {agg['pair_sep']['mean']:+.3f} +/- {agg['pair_sep']['std']:.3f}")
    print(f"  mean_top2:  trained = {trained_diag['mean_top2_gap']:+.3f}  "
          f"random = {agg['mean_top2_gap']['mean']:+.3f} +/- {agg['mean_top2_gap']['std']:.3f}")

    with open('results/convnext_random_extended.json', 'w') as f:
        json.dump({
            'n_blocks': n,
            'chance': chance,
            'n_seeds': N_SEEDS,
            'trained_diagnostics': trained_diag,
            'random_aggregate':    agg,
            'random_per_seed':     per_seed,
        }, f, indent=2)
    print()
    print("Saved results/convnext_random_extended.json")
