"""
RQ1 P0: Jacobian Orthogonality Verification (MLP)

Tests whether diagonal dominance (high s(i,i)) corresponds to near-orthogonal
Jacobians (low δ_J), as dynamical isometry theory predicts.

For each block:
  M = W_out @ W_in
  J = I + M
  δ_J = ||J^T J - I||_F / √d  (orthogonality deviation)
  s = |tr(M)| / ||M||_F       (diagonal dominance)

Measures at:
  - Initialization (7 schemes)
  - After training (epochs 5, 300)

Outputs:
  results/rq1_jacobian_mlp.json
  figures/fig_rq1_jacobian_mlp.png
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# --------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))


class ResNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(in_dim, hidden_dim) for _ in range(depth)]
        )
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.last(x)


# --------------------------------------------------------------------- inits
def apply_init(model: nn.Module, scheme: str, seed: int):
    """Reinitialize every nn.Linear in model according to scheme."""
    g = torch.Generator().manual_seed(seed)

    def _init_linear(lin: nn.Linear):
        W = lin.weight
        with torch.no_grad():
            if scheme == "orthogonal":
                tmp = torch.empty_like(W)
                nn.init.orthogonal_(tmp, gain=1.0, generator=g)
                W.copy_(tmp)
            elif scheme == "kaiming_normal":
                fan_in = W.shape[1]
                std = math.sqrt(2.0 / fan_in)
                W.copy_(torch.randn(W.shape, generator=g) * std)
            elif scheme == "kaiming_uniform":
                fan_in = W.shape[1]
                bound = math.sqrt(6.0 / fan_in)
                W.copy_(torch.empty(W.shape).uniform_(-bound, bound, generator=g))
            elif scheme == "xavier_normal":
                fan_in, fan_out = W.shape[1], W.shape[0]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                W.copy_(torch.randn(W.shape, generator=g) * std)
            elif scheme == "xavier_uniform":
                fan_in, fan_out = W.shape[1], W.shape[0]
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                W.copy_(torch.empty(W.shape).uniform_(-bound, bound, generator=g))
            elif scheme == "uniform":
                W.copy_(torch.empty(W.shape).uniform_(-0.1, 0.1, generator=g))
            elif scheme == "gaussian_002":
                W.copy_(torch.randn(W.shape, generator=g) * 0.02)
            else:
                raise ValueError(f"unknown init scheme: {scheme}")
            lin.bias.zero_()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            _init_linear(m)


# --------------------------------------------------------------------- data
def synthetic_target(X, in_dim, key):
    g = torch.Generator().manual_seed(key)
    A = torch.randn(in_dim, 8, generator=g) * 0.5
    B = torch.randn(8, generator=g)
    bias = torch.randn(1, generator=g)
    h = torch.tanh(X @ A)
    return h @ B + bias


def make_data(in_dim, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    y = synthetic_target(X, in_dim, key=seed + 1234)
    return X, y


# --------------------------------------------------------------------- metrics
def jacobian_orthogonality_deviation(M: np.ndarray) -> float:
    """Compute δ_J = ||J^T J - I||_F / √d where J = I + M."""
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    JTJ = J.T @ J
    deviation = np.linalg.norm(JTJ - I, 'fro') / np.sqrt(d)
    return float(deviation)


def diagonal_dominance(M: np.ndarray, eps: float = 1e-12) -> float:
    """Compute s = |tr(M)| / ||M||_F."""
    tr = np.trace(M)
    frob = np.linalg.norm(M, 'fro') + eps
    return float(abs(tr) / frob)


def extract_block_metrics(model: nn.Module):
    """Extract (s, δ_J, trace) for each block."""
    results = []
    d = model.blocks[0].inp.weight.shape[1]  # in_dim

    for i, blk in enumerate(model.blocks):
        W_in = blk.inp.weight.detach().cpu().numpy().astype(np.float64)
        W_out = blk.out.weight.detach().cpu().numpy().astype(np.float64)
        M = W_out @ W_in

        tr = float(np.trace(M))
        results.append({
            'block': i,
            'trace': tr,
            'trace_negative': tr < 0,
            'diag_dominance_s': diagonal_dominance(M),
            'jacobian_deviation': jacobian_orthogonality_deviation(M),
        })

    return results, d


# --------------------------------------------------------------------- train
def train_model(model, X, y, epochs, lr=1e-3, batch=256, grad_clip=1.0):
    """Train and return checkpoints at specified epochs."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]
    n_train = X_train.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            yb_pred = model(X_train[idx]).squeeze(-1)
            loss = loss_fn(yb_pred, y_train[idx])
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

    model.eval()
    with torch.no_grad():
        eval_loss = float(loss_fn(model(X_eval).squeeze(-1), y_eval).item())
    return eval_loss


def run_init_experiment(scheme: str, seed: int, depth: int, in_dim: int,
                        hidden: int, device: torch.device):
    """Measure metrics at initialization (epoch 0) for a given scheme."""
    model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth).to(device)
    apply_init(model, scheme, seed=seed)
    results, d = extract_block_metrics(model)
    return results


def run_training_experiment(seed: int, depth: int, in_dim: int, hidden: int,
                            checkpoint_epochs: list, device: torch.device):
    """Train model and measure metrics at checkpoint epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth).to(device)
    apply_init(model, "kaiming_normal", seed=seed * 1000 + 1)

    X, y = make_data(in_dim=in_dim, n=4000, seed=seed)
    X, y = X.to(device), y.to(device)

    checkpoints = {}

    # Epoch 0 (before training)
    if 0 in checkpoint_epochs:
        results, d = extract_block_metrics(model)
        checkpoints[0] = {'blocks': results, 'eval_loss': None}

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]
    n_train = X_train.shape[0]
    batch = 256

    max_ep = max(checkpoint_epochs)
    for ep in range(1, max_ep + 1):
        model.train()
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            yb_pred = model(X_train[idx]).squeeze(-1)
            loss = loss_fn(yb_pred, y_train[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep in checkpoint_epochs:
            model.eval()
            with torch.no_grad():
                eval_loss = float(loss_fn(model(X_eval).squeeze(-1), y_eval).item())
            results, d = extract_block_metrics(model)
            checkpoints[ep] = {'blocks': results, 'eval_loss': eval_loss}

    return checkpoints


def aggregate_results(block_results: list):
    """Aggregate per-block results."""
    return {
        'mean_s': float(np.mean([r['diag_dominance_s'] for r in block_results])),
        'mean_delta_J': float(np.mean([r['jacobian_deviation'] for r in block_results])),
        'frac_neg_trace': float(np.mean([r['trace_negative'] for r in block_results])),
        'mean_trace': float(np.mean([r['trace'] for r in block_results])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--in-dim", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/rq1_jacobian_mlp.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    print("=" * 60)
    print("RQ1: Jacobian Orthogonality Verification (MLP)")
    print("=" * 60)
    print(f"Architecture: depth={args.depth}, in_dim={args.in_dim}, hidden={args.hidden}")

    init_schemes = [
        "orthogonal", "kaiming_normal", "kaiming_uniform",
        "xavier_normal", "xavier_uniform", "uniform", "gaussian_002"
    ]
    checkpoint_epochs = [0, 5, 300]

    all_results = {
        'config': {
            'depth': args.depth,
            'hidden': args.hidden,
            'in_dim': args.in_dim,
            'seeds': args.seeds,
            'checkpoint_epochs': checkpoint_epochs,
        },
        'init_schemes': {},
        'training': {},
    }

    # Collect all (s, δ_J) pairs for scatter plot
    scatter_data = []

    # 1. Initialization experiments
    print("\n[Phase 1] Initialization experiments (epoch 0)")
    for scheme in init_schemes:
        print(f"\n  {scheme}:", end=" ", flush=True)
        all_blocks = []
        for seed in args.seeds:
            results = run_init_experiment(
                scheme, seed, args.depth, args.in_dim, args.hidden, device)
            all_blocks.extend(results)
            for r in results:
                scatter_data.append({
                    's': r['diag_dominance_s'],
                    'delta_J': r['jacobian_deviation'],
                    'condition': f'{scheme} (init)',
                    'epoch': 0,
                })

        agg = aggregate_results(all_blocks)
        all_results['init_schemes'][scheme] = {
            'aggregate': agg,
            'n_blocks': len(all_blocks),
        }
        print(f"s={agg['mean_s']:.4f}, δ_J={agg['mean_delta_J']:.4f}, neg_tr={agg['frac_neg_trace']:.1%}")

    # 2. Training experiments
    print("\n[Phase 2] Training experiments")
    for seed in args.seeds:
        print(f"\n  Seed {seed}:", flush=True)
        t0 = time.time()
        checkpoints = run_training_experiment(
            seed, args.depth, args.in_dim, args.hidden, checkpoint_epochs, device
        )
        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.1f}s")

        all_results['training'][f'seed_{seed}'] = {}
        for ep, data in checkpoints.items():
            agg = aggregate_results(data['blocks'])
            all_results['training'][f'seed_{seed}'][f'epoch_{ep}'] = {
                'aggregate': agg,
                'eval_loss': data['eval_loss'],
            }
            print(f"    Epoch {ep:3d}: s={agg['mean_s']:.4f}, δ_J={agg['mean_delta_J']:.4f}, "
                  f"neg_tr={agg['frac_neg_trace']:.1%}", end="")
            if data['eval_loss'] is not None:
                print(f", loss={data['eval_loss']:.4f}")
            else:
                print()

            for r in data['blocks']:
                scatter_data.append({
                    's': r['diag_dominance_s'],
                    'delta_J': r['jacobian_deviation'],
                    'condition': f'trained (seed {seed})',
                    'epoch': ep,
                })

    # Compute correlation
    all_s = [d['s'] for d in scatter_data]
    all_delta = [d['delta_J'] for d in scatter_data]
    corr, pval = pearsonr(all_s, all_delta)
    print(f"\n[Correlation] Pearson r(s, δ_J) = {corr:.4f} (p = {pval:.2e})")
    all_results['correlation'] = {'pearson_r': float(corr), 'p_value': float(pval)}

    # Save JSON
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {args.out}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot of s vs δ_J
    ax = axes[0]
    epoch_colors = {0: 'lightgray', 5: 'orange', 300: 'green'}
    for d in scatter_data:
        color = epoch_colors.get(d['epoch'], 'blue')
        ax.scatter(d['s'], d['delta_J'], c=color, alpha=0.5, s=20)

    # Add regression line
    z = np.polyfit(all_s, all_delta, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_s), max(all_s), 100)
    ax.plot(x_line, p(x_line), 'r--', label=f'r = {corr:.3f}')

    ax.set_xlabel(r'Diagonal dominance $s = |tr(M)| / \|M\|_F$')
    ax.set_ylabel(r'Jacobian deviation $\delta_J = \|J^T J - I\|_F / \sqrt{d}$')
    ax.set_title(f'Jacobian Orthogonality vs Diagonal Dominance\n(all blocks, all conditions)')
    ax.legend(loc='upper right')

    # Add legend for epochs
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='Epoch 0 (init)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Epoch 5'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Epoch 300'),
    ]
    ax.legend(handles=legend_elements + [Line2D([0], [0], linestyle='--', color='r', label=f'r = {corr:.3f}')],
              loc='upper right')

    # Right: Bar chart of mean δ_J by condition
    ax = axes[1]
    conditions = list(all_results['init_schemes'].keys()) + ['Trained ep5', 'Trained ep300']
    mean_deltas = [all_results['init_schemes'][s]['aggregate']['mean_delta_J'] for s in init_schemes]

    # Average across seeds for trained
    ep5_deltas = [all_results['training'][f'seed_{s}']['epoch_5']['aggregate']['mean_delta_J']
                  for s in args.seeds]
    ep300_deltas = [all_results['training'][f'seed_{s}']['epoch_300']['aggregate']['mean_delta_J']
                    for s in args.seeds]
    mean_deltas.append(np.mean(ep5_deltas))
    mean_deltas.append(np.mean(ep300_deltas))

    colors = ['lightgray'] * len(init_schemes) + ['orange', 'green']
    bars = ax.bar(range(len(conditions)), mean_deltas, color=colors)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel(r'Mean $\delta_J$')
    ax.set_title('Jacobian Deviation by Condition')

    plt.tight_layout()
    plt.savefig('figures/fig_rq1_jacobian_mlp.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_rq1_jacobian_mlp.pdf', bbox_inches='tight')
    print("Saved figures/fig_rq1_jacobian_mlp.{png,pdf}")


if __name__ == "__main__":
    main()
