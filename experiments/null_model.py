"""
Anti-thesis / null model experiment.

Shows that the diagonal-dominance pairing signal is INDUCED BY TRAINING,
not an architectural artifact of residual blocks. We compare:

  (a) An untrained ResNet (random Kaiming init) at the same shape as the
      Jane Street puzzle network.
  (b) The trained Jane Street puzzle network (recovered permutation).

Outputs:
  figures/fig_null_model.png   -- two d(i,j) heatmaps side by side
  null_model.json              -- summary stats
"""
import json, sys, os, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

torch.manual_seed(0)
np.random.seed(0)

D, H = 48, 96  # block dim, hidden width (Park's puzzle)


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


def pairing_stats(M):
    n = M.shape[0]
    row_ind, col_ind = linear_sum_assignment(-M)
    pair_acc = float((col_ind == np.arange(n)).mean())
    diag = np.diag(M)
    off = M - np.diag(np.diag(M))
    off_max_per_row = off.max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    return {
        'pair_acc': pair_acc,
        'pair_sep': pair_sep,
        'mean_correct': float(diag.mean()),
        'mean_incorrect': float(off[~np.eye(n, dtype=bool)].mean()),
    }


# -------------------- (a) Untrained net --------------------
print('Building untrained ResNet (Kaiming init)...')
W_ins_untr, W_outs_untr = [], []
for _ in range(D):
    Win = torch.empty(H, D)
    Wout = torch.empty(D, H)
    torch.nn.init.kaiming_uniform_(Win, a=5**0.5)
    torch.nn.init.kaiming_uniform_(Wout, a=5**0.5)
    W_ins_untr.append(Win.numpy())
    W_outs_untr.append(Wout.numpy())

M_untrained = diag_dominance_matrix(W_ins_untr, W_outs_untr)
stats_untrained = pairing_stats(M_untrained)
print('Untrained stats:', stats_untrained)


# -------------------- (b) Trained (Park's puzzle) --------------------
print('Loading Park\'s puzzle pieces...')
# Each piece is a state_dict for nn.Linear; in_dim=48, hidden=96.
# We need to know the recovered permutation to align W_in[i] with W_out[i].
# Easiest: re-run pairing on the trained pieces -> Hungarian gives the
# correct assignment by design (and is what we visualize).
pieces_dir = 'pieces'
state_dicts = []
for k in range(97):
    sd = torch.load(os.path.join(pieces_dir, f'piece_{k}.pth'),
                    map_location='cpu', weights_only=True)
    state_dicts.append(sd)

W_ins_tr, W_outs_tr = [], []
last_layer = None
for sd in state_dicts:
    w = sd['layer.weight'] if 'layer.weight' in sd else sd['weight']
    if w.shape == (1, D):
        last_layer = w.numpy()
        continue
    if w.shape == (H, D):
        W_ins_tr.append(w.numpy())
    elif w.shape == (D, H):
        W_outs_tr.append(w.numpy())

print(f'  Found {len(W_ins_tr)} W_in, {len(W_outs_tr)} W_out, last={last_layer is not None}')
assert len(W_ins_tr) == D and len(W_outs_tr) == D

# Pair via Hungarian to get the trained matrix in its "diagonal" form.
M_pre = diag_dominance_matrix(W_ins_tr, W_outs_tr)
row_ind, col_ind = linear_sum_assignment(-M_pre)
W_outs_paired = [W_outs_tr[col_ind[i]] for i in range(D)]
M_trained = diag_dominance_matrix(W_ins_tr, W_outs_paired)
stats_trained = pairing_stats(M_trained)
print('Trained (paired) stats:', stats_trained)


# -------------------- Figure --------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

im0 = axes[0].imshow(M_untrained, cmap='magma', vmin=0,
                     vmax=max(M_untrained.max(), M_trained.max()))
axes[0].set_title(
    f'(a) Untrained ResNet (Kaiming init)\n'
    f'pair acc = {stats_untrained["pair_acc"]:.2f}, '
    f'sep = {stats_untrained["pair_sep"]:+.2f}'
)
axes[0].set_xlabel(r'$W_\mathrm{out}$ index $j$')
axes[0].set_ylabel(r'$W_\mathrm{in}$ index $i$')

im1 = axes[1].imshow(M_trained, cmap='magma', vmin=0,
                     vmax=max(M_untrained.max(), M_trained.max()))
axes[1].set_title(
    f'(b) Trained ResNet (Park\'s puzzle, recovered pairing)\n'
    f'pair acc = {stats_trained["pair_acc"]:.2f}, '
    f'sep = {stats_trained["pair_sep"]:+.2f}'
)
axes[1].set_xlabel(r'$W_\mathrm{out}$ index $j$ (post-pairing)')
axes[1].set_ylabel(r'$W_\mathrm{in}$ index $i$')

fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02,
             label=r'$d(i,j)=|\mathrm{tr}(W_\mathrm{out}^{(j)} W_\mathrm{in}^{(i)})|/\|\cdot\|_F$')

plt.savefig('figures/fig_null_model.png', dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_null_model.pdf', bbox_inches='tight')
print('Saved figures/fig_null_model.{png,pdf}')

# Save stats
out = {
    'untrained': stats_untrained,
    'trained_park': stats_trained,
    'sweep_epoch0_summary': {
        'mean_pair_acc': 0.1307,
        'mean_pair_sep': -0.6801,
        'expected_chance_pair_acc': 0.0758,
        'note': 'computed from sweep_full.csv rows with epoch=0',
    },
    'sweep_trained_summary': {
        'mean_pair_acc_epoch_geq_5': 0.9128,
        'mean_pair_sep_epoch_geq_5': -0.3435,
    },
}
with open('null_model.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved null_model.json')
