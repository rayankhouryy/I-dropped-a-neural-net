"""
Deep-dive null model experiment.

GitHub issue: "Show that pairing fails on a random (untrained) network.
This gives you a clean before/after: at init it's random, after 5
epochs it's perfect."

We train a ResNet and snapshot the diagonal-dominance matrix d(i,j)
at five training stages: epochs 0 (untrained), 1, 5, 25, and final.
At each stage we record:

  - pair accuracy (Hungarian on -d)
  - pair separation
  - distributions of correct-pair vs incorrect-pair scores
  - mean diagonal entry vs mean off-diagonal entry
  - the d(i,j) heatmap itself

Output figure tells the whole story in one shot:
  Row 1: five d(i,j) heatmaps along the training trajectory.
  Row 2 (left): pair_acc, pair_sep, mean(diag), mean(off-diag) vs epoch.
  Row 2 (right): histograms of correct/incorrect scores at e=0 vs e=final.

Files written:
  figures/fig_null_deepdive.{png,pdf}
  null_deepdive.json
"""
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment

from pipeline import ResNet, make_data, DEVICE

torch.manual_seed(0)
np.random.seed(0)

# --------------------- config ---------------------
DEPTH     = 24
HIDDEN    = 64
IN_DIM    = 24
EPOCHS    = 300
BATCH     = 256
LR        = 3e-3
CKPTS     = [0, 1, 5, 25, EPOCHS]


# --------------------- pairing math ---------------------
def block_pair_matrix(model):
    """Return the depth x depth diagonal-dominance score matrix d(i,j)
    where row i indexes W_in^{(i)} and column j indexes W_out^{(j)}."""
    Wins  = [b.inp.weight.detach().cpu().numpy() for b in model.blocks]
    Wouts = [b.out.weight.detach().cpu().numpy() for b in model.blocks]
    n = len(Wins)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prod = Wouts[j] @ Wins[i]
            tr = abs(np.trace(prod))
            fr = np.linalg.norm(prod, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def stats(M):
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
        'std_correct':    float(diag.std()),
        'std_incorrect':  float(off.std()),
        'min_correct':    float(diag.min()),
        'max_incorrect':  float(off.max()),
    }, diag.copy(), off.copy()


# --------------------- train + snapshot ---------------------
X, y = make_data(IN_DIM, n=8000, seed=0)
ntr = 6000
Xt, yt = X[:ntr], y[:ntr]

model = ResNet(IN_DIM, HIDDEN, DEPTH).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

snapshots = {}  # epoch -> dict
train_loss_hist = {}

def snapshot(epoch):
    model.eval()
    M = block_pair_matrix(model)
    st, diag, off = stats(M)
    snapshots[epoch] = {'M': M, 'stats': st, 'diag': diag, 'off': off}
    with torch.no_grad():
        pred = model(Xt[:1000].to(DEVICE)).cpu().numpy().ravel()
        tloss = float(((pred - yt[:1000].numpy()) ** 2).mean())
    train_loss_hist[epoch] = tloss
    print(f'  ep={epoch:4d}  train_loss={tloss:.4f}  '
          f'pair_acc={st["pair_acc"]:.3f}  '
          f'sep={st["pair_sep"]:+.3f}  '
          f'mean_corr={st["mean_correct"]:.3f}  '
          f'mean_inc={st["mean_incorrect"]:.3f}')

# Snapshot at the requested checkpoints
t0 = time.time()
snapshot(0)
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(ntr)
    for i in range(0, ntr, BATCH):
        ix = perm[i:i+BATCH]
        xb = Xt[ix].to(DEVICE)
        yb = yt[ix].to(DEVICE).reshape(-1, 1)
        opt.zero_grad()
        F.mse_loss(model(xb), yb).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if epoch in CKPTS:
        snapshot(epoch)
print(f'Training done in {time.time()-t0:.1f}s')


# =====================================================================
# Four independent figures (each with its own axes / scaling).
#   (a) heatmap strip of d(i,j) across the trajectory
#   (b) identifiability trajectory (pair_acc, sep, mean d's)
#   (c) score distribution: untrained vs trained (overlay histogram)
#   (d) per-row margin distribution by epoch
# =====================================================================
vmax = max(s['M'].max() for s in snapshots.values())
eps_sorted = sorted(snapshots.keys())
acc   = [snapshots[e]['stats']['pair_acc']      for e in eps_sorted]
sep   = [snapshots[e]['stats']['pair_sep']      for e in eps_sorted]
mcorr = [snapshots[e]['stats']['mean_correct']  for e in eps_sorted]
minc  = [snapshots[e]['stats']['mean_incorrect'] for e in eps_sorted]

# ============================================================
# (a) Heatmap strip of d(i,j) along the training trajectory
# ============================================================
fig_a, axes_a = plt.subplots(1, 5, figsize=(15, 3.4))
for k, (ax, ep) in enumerate(zip(axes_a, CKPTS)):
    s = snapshots[ep]
    im = ax.imshow(s['M'], cmap='magma', vmin=0, vmax=vmax)
    title_tag = 'epoch 0 (untrained)' if ep == 0 else f'epoch {ep}'
    ax.set_title(
        f'{title_tag}\n'
        f'pair acc = {s["stats"]["pair_acc"]:.2f}, '
        f'sep = {s["stats"]["pair_sep"]:+.2f}',
        fontsize=10,
    )
    ticks = list(range(0, DEPTH, 4))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    if k == 0:
        ax.set_ylabel(r'$W_\mathrm{in}$ idx $i$', fontsize=9)
    ax.set_xlabel(r'$W_\mathrm{out}$ idx $j$', fontsize=9)

fig_a.subplots_adjust(right=0.92, wspace=0.25)
cbar_ax = fig_a.add_axes([0.93, 0.18, 0.011, 0.66])
cb = fig_a.colorbar(im, cax=cbar_ax)
cb.set_label(r'$d(i,j)$', fontsize=9)
fig_a.suptitle(
    '(a) Diagonal-dominance matrix $d(i,j)$ along the training trajectory',
    fontsize=11, y=1.02,
)
fig_a.savefig('figures/fig_null_a_heatmaps.png', dpi=160, bbox_inches='tight')
fig_a.savefig('figures/fig_null_a_heatmaps.pdf',           bbox_inches='tight')

# ============================================================
# (b) Identifiability trajectory
# ============================================================
fig_b, ax_traj = plt.subplots(figsize=(7.5, 5.0))

EP0_POS = 0.5
x_eps = [EP0_POS if e == 0 else e for e in eps_sorted]

# Left axis: pair_acc + pair_sep (in [-1, 1.15])
l1, = ax_traj.plot(x_eps, acc, 'o-', color='C0', lw=2.0, ms=8,
                   label='pair accuracy')
l2, = ax_traj.plot(x_eps, sep, 's-', color='C3', lw=1.8, ms=7,
                   label='pair separation')
ax_traj.axhline(0,   color='k',  lw=0.6, alpha=0.4)
ax_traj.axhline(1.0, color='C0', lw=0.7, ls='--', alpha=0.45)

chance = 1.0 / DEPTH
ax_traj.axhline(chance, color='C0', lw=1.2, ls=':', alpha=0.85)
ax_traj.text(
    50, chance + 0.07,
    rf'$1/D = {chance:.3f}$ (chance)',
    fontsize=9, color='C0', ha='left',
    bbox=dict(boxstyle='round,pad=0.18', fc='white',
              ec='none', alpha=0.9),
)

sep_peak_epoch = eps_sorted[int(np.argmax(sep))]
if sep_peak_epoch not in (0, EPOCHS):
    ax_traj.axvline(sep_peak_epoch, color='C3', lw=0.9, ls=':', alpha=0.7)
    ax_traj.text(
        sep_peak_epoch * 1.08, 0.85,
        f'sep peak\n(ep {sep_peak_epoch})',
        fontsize=8.5, color='C3',
        bbox=dict(boxstyle='round,pad=0.18', fc='white',
                  ec='none', alpha=0.9),
    )

ax_traj.set_xscale('log')
ax_traj.set_ylim(-1.0, 1.18)
ax_traj.set_ylabel('pair accuracy   /   pair separation', fontsize=10)
ax_traj.set_xticks(x_eps)
ax_traj.set_xticklabels([str(e) for e in eps_sorted])
ax_traj.set_xlabel('training epoch (log scale)', fontsize=10)
ax_traj.grid(True, alpha=0.3)
ax_traj.set_title(
    '(b) Identifiability trajectory: random at init, perfect by epoch 5',
    fontsize=11,
)

# Right axis: mean d(i,i), mean d(i,j)
ax_traj2 = ax_traj.twinx()
l3, = ax_traj2.plot(x_eps, mcorr, 'd-', color='C2', lw=1.7, ms=7,
                    alpha=0.9, label=r'mean $d(i,i)$ (correct)')
l4, = ax_traj2.plot(x_eps, minc,  'v-', color='gray', lw=1.5, ms=7,
                    alpha=0.9, label=r'mean $d(i,j)$, $i\neq j$')
ax_traj2.set_ylabel(r'$d$ score', fontsize=10)
ax_traj2.set_ylim(0, max(mcorr) * 1.15)

ax_traj.legend(
    handles=[l1, l2, l3, l4],
    loc='upper center', bbox_to_anchor=(0.5, -0.16),
    ncol=4, fontsize=9, framealpha=0.92,
    handlelength=1.8, columnspacing=1.5, borderaxespad=0.2,
)

fig_b.savefig('figures/fig_null_b_trajectory.png', dpi=160, bbox_inches='tight')
fig_b.savefig('figures/fig_null_b_trajectory.pdf',           bbox_inches='tight')

# ============================================================
# (c) Score distribution: untrained vs trained (overlay)
# ============================================================
fig_c, ax_hist = plt.subplots(figsize=(7.5, 5.0))
bins = np.linspace(0, vmax, 40)
s0, sf = snapshots[0], snapshots[EPOCHS]

ax_hist.hist(s0['off'],  bins=bins, color='gray', alpha=0.35, density=True,
             label='incorrect (off-diag), epoch 0')
ax_hist.hist(s0['diag'], bins=bins, color='C3', alpha=0.50, density=True,
             label='correct (diag), epoch 0')
ax_hist.hist(sf['off'],  bins=bins, color='black', alpha=0.95, density=True,
             histtype='step', lw=2.2,
             label=f'incorrect (off-diag), epoch {EPOCHS}')
ax_hist.hist(sf['diag'], bins=bins, color='C3', alpha=0.95, density=True,
             histtype='step', lw=2.2,
             label=f'correct (diag), epoch {EPOCHS}')
ax_hist.set_xlabel(r'$d$ score', fontsize=10)
ax_hist.set_ylabel('density', fontsize=10)
ax_hist.set_title(
    '(c) Score distribution: untrained vs trained (overlay)',
    fontsize=11,
)
ax_hist.legend(fontsize=9, loc='upper right', framealpha=0.92)
ax_hist.grid(True, alpha=0.3)
fig_c.savefig('figures/fig_null_c_histograms.png', dpi=160, bbox_inches='tight')
fig_c.savefig('figures/fig_null_c_histograms.pdf',           bbox_inches='tight')

# ============================================================
# (d) Per-row margin distribution by epoch
# ============================================================
fig_d, ax_marg = plt.subplots(figsize=(7.5, 5.0))
margins_per_ep = []
for e in eps_sorted:
    M = snapshots[e]['M']
    diag = np.diag(M)
    off = M - np.diag(diag)
    off_max = off.max(axis=1)
    margins_per_ep.append(diag - off_max)

positions = list(range(len(eps_sorted)))
bp = ax_marg.boxplot(
    margins_per_ep, positions=positions, widths=0.55,
    patch_artist=True, showmeans=True,
    meanprops={'marker': 'D', 'markerfacecolor': 'white',
               'markeredgecolor': 'black', 'markersize': 6},
    medianprops={'color': 'black', 'lw': 1.4},
)
box_colors = ['lightgray'] + ['#a3d3a3'] * (len(eps_sorted) - 1)
for patch, c in zip(bp['boxes'], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)

rng_loc = np.random.default_rng(0)
for k, m in enumerate(margins_per_ep):
    jit = rng_loc.normal(0, 0.06, size=len(m))
    ax_marg.scatter(np.full_like(m, k) + jit, m,
                    s=14, color='C0', alpha=0.55, edgecolors='none')

ax_marg.axhline(0, color='k', lw=0.8, alpha=0.6)
ax_marg.set_xticks(positions)
ax_marg.set_xticklabels([str(e) for e in eps_sorted])
ax_marg.set_xlabel('epoch', fontsize=10)
ax_marg.set_ylabel(r'per-row margin: $d(i,i) - \max_{j\neq i} d(i,j)$',
                   fontsize=10)
ax_marg.set_title(
    '(d) Per-row Hungarian margin distribution by epoch',
    fontsize=11,
)
ax_marg.grid(True, alpha=0.3, axis='y')
fig_d.savefig('figures/fig_null_d_margins.png', dpi=160, bbox_inches='tight')
fig_d.savefig('figures/fig_null_d_margins.pdf',           bbox_inches='tight')

print('Saved figures/fig_null_{a,b,c,d}_*.{png,pdf}')


# --------------------- dump JSON ---------------------
out = {
    'config': {
        'depth': DEPTH, 'hidden': HIDDEN, 'in_dim': IN_DIM,
        'epochs': EPOCHS, 'batch': BATCH, 'lr': LR,
    },
    'chance_pair_acc': 1.0 / DEPTH,
    'trajectory': {
        str(e): {
            **snapshots[e]['stats'],
            'train_loss': train_loss_hist[e],
        } for e in eps_sorted
    },
}
with open('null_deepdive.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved null_deepdive.json')
