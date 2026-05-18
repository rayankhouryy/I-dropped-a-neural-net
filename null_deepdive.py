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


# --------------------- figure: 2-row deep-dive ---------------------
fig = plt.figure(figsize=(15, 7.5))
gs = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 0.85],
              hspace=0.55, wspace=0.35)

vmax = max(s['M'].max() for s in snapshots.values())

# Row 1: five d(i,j) heatmaps
heat_axes = []
for k, ep in enumerate(CKPTS):
    ax = fig.add_subplot(gs[0, k])
    s = snapshots[ep]
    im = ax.imshow(s['M'], cmap='magma', vmin=0, vmax=vmax)
    title_tag = 'untrained' if ep == 0 else f'epoch {ep}'
    ax.set_title(
        f'{title_tag}\n'
        f'pair acc = {s["stats"]["pair_acc"]:.2f}, '
        f'sep = {s["stats"]["pair_sep"]:+.2f}',
        fontsize=10,
    )
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.set_ylabel(r'$W_\mathrm{in}$ idx $i$', fontsize=9)
    heat_axes.append(ax)

# Shared colorbar for row 1
cbar_ax = fig.add_axes([0.92, 0.55, 0.012, 0.33])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label(r'$d(i,j)$', fontsize=9)

# Row 2 (left two cols): trajectory plot
ax_traj = fig.add_subplot(gs[1, 0:2])
eps_sorted = sorted(snapshots.keys())
acc   = [snapshots[e]['stats']['pair_acc'] for e in eps_sorted]
sep   = [snapshots[e]['stats']['pair_sep'] for e in eps_sorted]
mcorr = [snapshots[e]['stats']['mean_correct'] for e in eps_sorted]
minc  = [snapshots[e]['stats']['mean_incorrect'] for e in eps_sorted]

x_eps = [max(0.5, e) for e in eps_sorted]  # log-scale safe
ax_traj.plot(x_eps, acc, 'o-', color='C0', label='pair accuracy')
ax_traj.plot(x_eps, sep, 's-', color='C3', label='pair separation')
ax_traj.plot(x_eps, mcorr, 'd-', color='C2', label=r'mean $d(i,i)$')
ax_traj.plot(x_eps, minc,  'v-', color='gray', label=r'mean $d(i,j)$, $i\neq j$')
ax_traj.axhline(0, color='k', lw=0.6, alpha=0.5)
ax_traj.axhline(1.0, color='C0', lw=0.6, ls='--', alpha=0.5)
chance = 1.0 / DEPTH
ax_traj.axhline(chance, color='C0', lw=0.6, ls=':', alpha=0.6)
ax_traj.text(0.55, chance + 0.02, r'$1/D$ (chance)', fontsize=8, color='C0')
ax_traj.set_xscale('log')
ax_traj.set_xlabel('training epoch (log scale, $0\\to0.5$)')
ax_traj.set_ylabel('value')
ax_traj.set_title('(b) Identifiability trajectory', fontsize=10)
ax_traj.legend(loc='center right', fontsize=8, framealpha=0.92)
ax_traj.grid(True, alpha=0.3)

# Row 2 (right three cols): correct vs incorrect score distributions at e=0 and e=final
ax_h0 = fig.add_subplot(gs[1, 2])
ax_hf = fig.add_subplot(gs[1, 3])
ax_summary = fig.add_subplot(gs[1, 4])

def hist_panel(ax, ep, title):
    s = snapshots[ep]
    bins = np.linspace(0, vmax, 35)
    ax.hist(s['off'],  bins=bins, alpha=0.55, color='gray',
            label='incorrect (off-diag)', density=True)
    ax.hist(s['diag'], bins=bins, alpha=0.85, color='C3',
            label='correct (diag)', density=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$d$ score')
    ax.set_ylabel('density')
    ax.legend(fontsize=7, loc='upper right')

hist_panel(ax_h0, 0,    '(c) Score distribution @ untrained')
hist_panel(ax_hf, EPOCHS, f'(d) Score distribution @ epoch {EPOCHS}')

# Summary bar chart: KS-style separation per checkpoint
ax_summary.set_title('(e) Diag/off-diag gap', fontsize=10)
gap = np.array(mcorr) - np.array(minc)
ax_summary.bar(range(len(eps_sorted)), gap,
               color=['gray' if e == 0 else 'C2' for e in eps_sorted])
ax_summary.set_xticks(range(len(eps_sorted)))
ax_summary.set_xticklabels([str(e) for e in eps_sorted])
ax_summary.set_xlabel('epoch')
ax_summary.set_ylabel(r'mean $d(i,i)$ - mean $d(i,j)$')
ax_summary.axhline(0, color='k', lw=0.6, alpha=0.5)

fig.suptitle(
    'Null model deep dive: the diagonal-dominance signal is created '
    'by training, not by residual architecture',
    fontsize=12, y=1.005,
)

plt.savefig('figures/fig_null_deepdive.png', dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_null_deepdive.pdf', bbox_inches='tight')
print('Saved figures/fig_null_deepdive.{png,pdf}')


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
