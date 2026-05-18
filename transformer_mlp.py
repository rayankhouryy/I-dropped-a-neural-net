"""
Transformer MLP pairing experiment.

Tests whether the diagonal-dominance pairing signal extends from
residual MLPs to the MLP sublayers of a trained transformer.

For each TransformerEncoderLayer the residual MLP branch is
   x -> x + W_2 * sigma(W_1 * LayerNorm(x) + b_1) + b_2,
so the analogous pair product is W_2 @ W_1 (shape (d_model, d_model)).
We score
   d(i,j) = |tr(W_2^{(j)} @ W_1^{(i)})| / ||W_2^{(j)} @ W_1^{(i)}||_F
and run Hungarian assignment, just as in the ResNet case.

Outputs:
  figures/fig_transformer_mlp.png   -- side-by-side d(i,j) heatmaps
  transformer_mlp.json              -- summary stats
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

torch.manual_seed(0)
np.random.seed(0)

# ----- hyperparameters (small, fast) -----
D_MODEL = 64
NHEAD = 4
DIM_FF = 128
NUM_LAYERS = 16     # gives a 16x16 pairing matrix
SEQ_LEN = 12
N_EPOCHS = 600
BATCH_SIZE = 128

# ----- model: tiny encoder + linear head -----
encoder_layer = nn.TransformerEncoderLayer(
    d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
    dropout=0.0, batch_first=True, activation='relu',
    norm_first=True,   # pre-norm (modern convention)
)
model = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
head = nn.Linear(D_MODEL, 1)


def extract_mlp_weights(transformer):
    """Return (W1_list, W2_list) where each entry is a numpy array."""
    W1s, W2s = [], []
    for layer in transformer.layers:
        W1s.append(layer.linear1.weight.detach().cpu().numpy().copy())
        W2s.append(layer.linear2.weight.detach().cpu().numpy().copy())
    return W1s, W2s


# Snapshot the untrained model
W1_untr, W2_untr = extract_mlp_weights(model)


# ----- synthetic regression task -----
# Pick a fixed target function: tanh(Ax) read out via the last position.
A_tgt = torch.randn(D_MODEL, D_MODEL) / D_MODEL**0.5

def gen_batch(n):
    x = torch.randn(n, SEQ_LEN, D_MODEL)
    # target: mean over positions of a 1-d nonlinear readout
    z = torch.tanh(x @ A_tgt)
    y = z.mean(dim=1).sum(dim=-1, keepdim=True) * 0.1
    return x, y


# ----- train -----
opt = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()),
    lr=3e-4,
)

t0 = time.time()
loss_hist = []
for ep in range(N_EPOCHS):
    x, y = gen_batch(BATCH_SIZE)
    out = model(x)
    pred = head(out.mean(dim=1))
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(head.parameters()), 1.0
    )
    opt.step()
    loss_hist.append(loss.item())
    if ep % 50 == 0 or ep == N_EPOCHS - 1:
        print(f'  epoch {ep:4d}  loss = {loss.item():.4f}')
print(f'Training done in {time.time()-t0:.1f}s. Final loss = {loss_hist[-1]:.4f}')

# Snapshot the trained model
W1_tr, W2_tr = extract_mlp_weights(model)


# ----- pairing scores -----
def diag_dominance_matrix(W1s, W2s):
    n = len(W1s)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prod = W2s[j] @ W1s[i]      # (d_model, d_model)
            tr = abs(np.trace(prod))
            fr = np.linalg.norm(prod, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def pairing_stats(M):
    n = M.shape[0]
    _, col_ind = linear_sum_assignment(-M)
    pair_acc = float((col_ind == np.arange(n)).mean())
    diag = np.diag(M)
    off = M - np.diag(np.diag(M))
    off_max_per_row = off.max(axis=1)
    sep = float((diag - off_max_per_row).min())
    return {
        'pair_acc': pair_acc,
        'pair_sep': sep,
        'mean_correct': float(diag.mean()),
        'mean_incorrect': float(off[~np.eye(n, dtype=bool)].mean()),
        # signed trace stats: did dynamic isometry push tr(M_ii) negative?
        'frac_negative_diag_trace_correct':
            float(np.mean([np.trace(W2s_j @ W1s_i) < 0
                          for W1s_i, W2s_j in zip(_W1s_ref, _W2s_ref)]))
            if False else None,   # placeholder; computed below
    }


def trace_signs(W1s, W2s):
    return np.array([np.trace(W2s[i] @ W1s[i]) for i in range(len(W1s))])


M_untr = diag_dominance_matrix(W1_untr, W2_untr)
M_tr = diag_dominance_matrix(W1_tr, W2_tr)

stats_untr = pairing_stats(M_untr)
stats_tr = pairing_stats(M_tr)

# Add the dynamic-isometry diagnostic: tr(W_2 W_1) for correct pairs.
tr_untr = trace_signs(W1_untr, W2_untr)
tr_tr = trace_signs(W1_tr, W2_tr)
stats_untr['mean_trace_correct'] = float(tr_untr.mean())
stats_untr['frac_negative_trace_correct'] = float((tr_untr < 0).mean())
stats_tr['mean_trace_correct'] = float(tr_tr.mean())
stats_tr['frac_negative_trace_correct'] = float((tr_tr < 0).mean())

print()
print('UNTRAINED transformer MLPs:')
for k, v in stats_untr.items():
    print(f'  {k}: {v}')
print('TRAINED transformer MLPs:')
for k, v in stats_tr.items():
    print(f'  {k}: {v}')


# ----- figure -----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
vmax = max(M_untr.max(), M_tr.max())

im0 = axes[0].imshow(M_untr, cmap='magma', vmin=0, vmax=vmax)
axes[0].set_title(
    f'(a) Untrained transformer MLPs\n'
    f'pair acc = {stats_untr["pair_acc"]:.2f}, '
    f'sep = {stats_untr["pair_sep"]:+.2f}'
)
axes[0].set_xlabel(r'$W_2$ index $j$')
axes[0].set_ylabel(r'$W_1$ index $i$')

im1 = axes[1].imshow(M_tr, cmap='magma', vmin=0, vmax=vmax)
axes[1].set_title(
    f'(b) Trained transformer MLPs\n'
    f'pair acc = {stats_tr["pair_acc"]:.2f}, '
    f'sep = {stats_tr["pair_sep"]:+.2f}'
)
axes[1].set_xlabel(r'$W_2$ index $j$')
axes[1].set_ylabel(r'$W_1$ index $i$')

fig.colorbar(
    im1, ax=axes, fraction=0.025, pad=0.02,
    label=r'$d(i,j) = |\mathrm{tr}(W_2^{(j)} W_1^{(i)})| / \|W_2^{(j)} W_1^{(i)}\|_F$',
)

plt.savefig('figures/fig_transformer_mlp.png', dpi=160, bbox_inches='tight')
plt.savefig('figures/fig_transformer_mlp.pdf', bbox_inches='tight')
print('Saved figures/fig_transformer_mlp.{png,pdf}')


with open('transformer_mlp.json', 'w') as f:
    json.dump({
        'config': {
            'd_model': D_MODEL, 'nhead': NHEAD, 'dim_ff': DIM_FF,
            'num_layers': NUM_LAYERS, 'seq_len': SEQ_LEN,
            'n_epochs': N_EPOCHS, 'batch': BATCH_SIZE,
            'norm_first': True, 'activation': 'relu',
        },
        'final_train_loss': loss_hist[-1],
        'untrained': stats_untr,
        'trained': stats_tr,
    }, f, indent=2)
print('Saved transformer_mlp.json')
