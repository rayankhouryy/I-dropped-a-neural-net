"""
Random-init triple baseline for Task 1.

Tests whether the architecture-aware triple product M = W_3 W_2 W_1
identifies blocks ONLY when the network is trained, or whether it
identifies blocks even at random init (which would suggest the rescue
seen on pretrained ResNets is a structural / shape artifact rather
than a training-induced fingerprint).

We run randomly initialized resnet50 / resnet101 / resnet152 at the
same stages where the trained models showed 100% pair_acc under the
triple score, with N_SEEDS independent initializations per model.

Both channel_sum and center_tap projections are tested.

Outputs:
  results/resnet_triple_random_baseline.json
"""

import json, gc
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

Path("results").mkdir(exist_ok=True)
torch.set_grad_enabled(False)

# Import extraction helpers from the ablation script
import sys
sys.path.insert(0, 'experiments')
from resnet_extraction_ablation import (
    extract_bottleneck_block_matrices,
    score_pair_matrix,
    score_triple_endpoint,
    hungarian_accuracy,
    auc_correct_vs_incorrect,
)

N_SEEDS = 3

# (model_builder, name, stages_to_test [(name, expected_chance, n_blocks)])
SPECS = [
    ('resnet50',  'layer3', 5),
    ('resnet101', 'layer3', 22),
    ('resnet152', 'layer2', 7),
    ('resnet152', 'layer3', 35),
]


def run_one(builder, model_name, stage_name, seed, mode):
    torch.manual_seed(seed)
    model = builder(weights=None)
    model.eval()
    stage = getattr(model, stage_name)
    blocks = []
    skipped = 0
    for block in stage:
        if getattr(block, 'downsample', None) is not None:
            skipped += 1
            continue
        tup = extract_bottleneck_block_matrices(block, mode=mode, fold_bn=True)
        blocks.append(tup)

    W_1s = [b[0] for b in blocks]
    W_2s = [b[1] for b in blocks]
    W_3s = [b[2] for b in blocks]

    M_triple    = score_triple_endpoint(W_1s, W_2s, W_3s)
    M_endpoints = score_pair_matrix(W_1s, W_3s)

    acc_t, sep_t = hungarian_accuracy(M_triple)
    acc_e, sep_e = hungarian_accuracy(M_endpoints)
    auc_t = auc_correct_vs_incorrect(M_triple)
    auc_e = auc_correct_vs_incorrect(M_endpoints)

    n = len(W_1s)
    diag_t = np.diag(M_triple)
    off_t  = M_triple[~np.eye(n, dtype=bool)]

    del model; gc.collect()
    return {
        'seed': seed,
        'n_blocks': n,
        'chance': 1.0 / n,
        'triple': {
            'pair_acc': acc_t,
            'pair_sep': sep_t,
            'auc': auc_t,
            'mean_correct': float(diag_t.mean()),
            'mean_incorrect': float(off_t.mean()),
            'acc_over_chance': acc_t * n,
        },
        'endpoints': {
            'pair_acc': acc_e,
            'pair_sep': sep_e,
            'auc': auc_e,
            'acc_over_chance': acc_e * n,
        },
    }


if __name__ == '__main__':
    import torchvision.models as M
    builders = {
        'resnet50':  M.resnet50,
        'resnet101': M.resnet101,
        'resnet152': M.resnet152,
    }

    all_results = []
    for model_name, stage_name, expected_n in SPECS:
        for mode in ['channel_sum', 'center_tap']:
            seeds_data = []
            for seed in range(N_SEEDS):
                print(f"\n[{model_name}/{stage_name}/mode={mode}/seed={seed}]")
                r = run_one(builders[model_name], model_name, stage_name, seed, mode)
                seeds_data.append(r)
                t = r['triple']; e = r['endpoints']
                print(f"  triple   : acc={t['pair_acc']:.0%} ({t['acc_over_chance']:.1f}x chance)  "
                      f"sep={t['pair_sep']:+.3f}  AUC={t['auc']:.3f}")
                print(f"  endpoints: acc={e['pair_acc']:.0%} ({e['acc_over_chance']:.1f}x chance)  "
                      f"sep={e['pair_sep']:+.3f}  AUC={e['auc']:.3f}")

            # aggregate over seeds
            tr_accs   = [d['triple']['pair_acc']        for d in seeds_data]
            tr_aucs   = [d['triple']['auc']             for d in seeds_data]
            tr_seps   = [d['triple']['pair_sep']        for d in seeds_data]
            ep_accs   = [d['endpoints']['pair_acc']     for d in seeds_data]
            ep_aucs   = [d['endpoints']['auc']          for d in seeds_data]
            agg = {
                'model':  model_name,
                'stage':  stage_name,
                'mode':   mode,
                'n_blocks': seeds_data[0]['n_blocks'],
                'chance': seeds_data[0]['chance'],
                'n_seeds': N_SEEDS,
                'triple_mean_acc':   float(np.mean(tr_accs)),
                'triple_std_acc':    float(np.std(tr_accs)),
                'triple_mean_auc':   float(np.mean(tr_aucs)),
                'triple_mean_sep':   float(np.mean(tr_seps)),
                'endpoints_mean_acc': float(np.mean(ep_accs)),
                'endpoints_mean_auc': float(np.mean(ep_aucs)),
                'per_seed': seeds_data,
            }
            all_results.append(agg)

            print(f"  [AVG over {N_SEEDS} seeds] triple_acc={agg['triple_mean_acc']:.0%} "
                  f"+/-{agg['triple_std_acc']:.0%}  "
                  f"AUC={agg['triple_mean_auc']:.3f}  sep={agg['triple_mean_sep']:+.3f}")

            # incremental save
            with open('results/resnet_triple_random_baseline.json', 'w') as f:
                json.dump(all_results, f, indent=2)

    print('\n' + '='*78)
    print('SUMMARY: random-init triple baseline')
    print('='*78)
    for r in all_results:
        chance = r['chance']
        delta = r['triple_mean_acc'] - chance
        flag = ' <-- ABOVE chance' if delta > 0.05 else ''
        print(f"  {r['model']}/{r['stage']}/{r['mode']:11s}  "
              f"n={r['n_blocks']:2d}  chance={chance:.1%}  "
              f"triple acc={r['triple_mean_acc']:.0%}+/-{r['triple_std_acc']:.0%}  "
              f"AUC={r['triple_mean_auc']:.3f}{flag}")
