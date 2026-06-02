"""
RQ1 P0: Random Labels Experiment

Tests whether the fingerprint requires learning a meaningful task, or if
gradient flow through residual connections alone suffices.

Trains 48-block residual MLP on CIFAR-10 under two conditions:
  - Normal labels: Standard classification
  - Shuffled labels: Random permutation of training labels

If diagonal dominance emerges equally in both conditions, gradient flow
(not task-relevant learning) causes the fingerprint.

Outputs:
  results/rq1_random_labels.json
  figures/fig_rq1_random_labels.png
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# --------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))


class ResNetMLP(nn.Module):
    """Residual MLP for image classification."""

    def __init__(self, input_dim, block_dim, hidden_dim, depth, num_classes):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, block_dim)
        self.blocks = nn.ModuleList(
            [Block(block_dim, hidden_dim) for _ in range(depth)]
        )
        self.head = nn.Linear(block_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.proj_in(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


# --------------------------------------------------------------------- data
def get_cifar_loaders(batch_size: int = 256, shuffle_labels: bool = False,
                      shuffle_seed: int = 42):
    """Load CIFAR-10 with optional label shuffling."""
    tx_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tx_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    root = "data"
    Path(root).mkdir(exist_ok=True)

    train_ds = datasets.CIFAR10(root, train=True, download=True, transform=tx_train)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=tx_eval)

    if shuffle_labels:
        rng = np.random.RandomState(shuffle_seed)
        shuffled = rng.permutation(len(train_ds.targets))
        original_targets = np.array(train_ds.targets)
        train_ds.targets = list(original_targets[shuffled])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader


# --------------------------------------------------------------------- metrics
def diagonal_dominance(M: np.ndarray, eps: float = 1e-12) -> float:
    """Compute s = |tr(M)| / ||M||_F."""
    tr = np.trace(M)
    frob = np.linalg.norm(M, 'fro') + eps
    return float(abs(tr) / frob)


def extract_metrics(model: nn.Module, init_weights: dict = None):
    """Extract diagonal dominance metrics for all blocks.

    Args:
        model: The ResNetMLP model
        init_weights: Optional dict mapping block index to (W_in_init, W_out_init)
                      for computing weight change magnitude
    """
    results = []
    for i, blk in enumerate(model.blocks):
        W_in = blk.inp.weight.detach().cpu().numpy().astype(np.float64)
        W_out = blk.out.weight.detach().cpu().numpy().astype(np.float64)
        M = W_out @ W_in

        tr = float(np.trace(M))
        block_result = {
            'block': i,
            'trace': tr,
            'trace_negative': tr < 0,
            'diag_dominance_s': diagonal_dominance(M),
        }

        # Weight magnitude tracking for P0.5 control
        if init_weights is not None and i in init_weights:
            W_in_init, W_out_init = init_weights[i]
            delta_W_in = np.linalg.norm(W_in - W_in_init, 'fro')
            delta_W_out = np.linalg.norm(W_out - W_out_init, 'fro')
            block_result['delta_W_in_fro'] = float(delta_W_in)
            block_result['delta_W_out_fro'] = float(delta_W_out)
            block_result['total_weight_delta'] = float(delta_W_in + delta_W_out)

        results.append(block_result)

    return results


def aggregate_metrics(block_results: list):
    """Aggregate per-block results."""
    agg = {
        'mean_s': float(np.mean([r['diag_dominance_s'] for r in block_results])),
        'frac_neg_trace': float(np.mean([r['trace_negative'] for r in block_results])),
        'mean_trace': float(np.mean([r['trace'] for r in block_results])),
    }

    # Weight magnitude aggregates (P0.5 control)
    if 'total_weight_delta' in block_results[0]:
        agg['mean_delta_W'] = float(np.mean([r['total_weight_delta'] for r in block_results]))
        agg['sum_delta_W'] = float(np.sum([r['total_weight_delta'] for r in block_results]))

    return agg


# --------------------------------------------------------------------- train
def train_epoch(model, loader, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_samples = 0
    criterion = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)

    return total_loss / n_samples


def evaluate(model, loader, device):
    """Evaluate accuracy on loader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def run_experiment(seed: int, shuffle_labels: bool, depth: int, block_dim: int,
                   hidden_dim: int, epochs: int, checkpoint_epochs: list,
                   device: str):
    """Run a single training experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    condition = "shuffled" if shuffle_labels else "normal"
    print(f"  [{condition}, seed={seed}] Starting...", flush=True)

    # Data
    train_loader, test_loader = get_cifar_loaders(
        batch_size=256,
        shuffle_labels=shuffle_labels,
        shuffle_seed=seed + 9999  # Different seed for label shuffle
    )

    # Model
    input_dim = 3 * 32 * 32  # CIFAR-10 flattened
    model = ResNetMLP(
        input_dim=input_dim,
        block_dim=block_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        num_classes=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Store initial weights for P0.5 weight magnitude control
    init_weights = {}
    for i, blk in enumerate(model.blocks):
        W_in_init = blk.inp.weight.detach().cpu().numpy().astype(np.float64).copy()
        W_out_init = blk.out.weight.detach().cpu().numpy().astype(np.float64).copy()
        init_weights[i] = (W_in_init, W_out_init)

    # Training with checkpoints
    checkpoints = {}

    # Epoch 0 (before training) - no weight delta at init
    if 0 in checkpoint_epochs:
        metrics = extract_metrics(model)  # No init_weights comparison at epoch 0
        test_acc = evaluate(model, test_loader, device)
        checkpoints[0] = {
            'blocks': metrics,
            'test_acc': test_acc,
            'train_loss': None,
            'aggregate': aggregate_metrics(metrics),
        }
        print(f"    Epoch 0: s={checkpoints[0]['aggregate']['mean_s']:.4f}, "
              f"acc={test_acc:.1%}")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        if ep in checkpoint_epochs:
            metrics = extract_metrics(model, init_weights)  # Pass init weights for delta
            test_acc = evaluate(model, test_loader, device)
            agg = aggregate_metrics(metrics)
            checkpoints[ep] = {
                'blocks': metrics,
                'test_acc': test_acc,
                'train_loss': train_loss,
                'aggregate': agg,
            }
            elapsed = time.time() - t0
            delta_str = f", ΔW={agg.get('mean_delta_W', 0):.2f}" if 'mean_delta_W' in agg else ""
            print(f"    Epoch {ep}: s={agg['mean_s']:.4f}, "
                  f"neg_tr={agg['frac_neg_trace']:.1%}, "
                  f"acc={test_acc:.1%}, loss={train_loss:.4f}{delta_str} ({elapsed:.0f}s)")

    return checkpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--block-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/rq1_random_labels.json")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    checkpoint_epochs = [0, 5, 10, 50, 100]
    checkpoint_epochs = [ep for ep in checkpoint_epochs if ep <= args.epochs]

    print("=" * 60)
    print("RQ1: Random Labels Experiment")
    print("=" * 60)
    print(f"Architecture: depth={args.depth}, block_dim={args.block_dim}, hidden={args.hidden_dim}")
    print(f"Training: {args.epochs} epochs, checkpoints at {checkpoint_epochs}")
    print(f"Device: {args.device}")

    all_results = {
        'config': {
            'depth': args.depth,
            'block_dim': args.block_dim,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'seeds': args.seeds,
            'checkpoint_epochs': checkpoint_epochs,
        },
        'normal': {},
        'shuffled': {},
    }

    # Run experiments
    for shuffle_labels in [False, True]:
        condition = "shuffled" if shuffle_labels else "normal"
        print(f"\n[{condition.upper()} LABELS]")

        for seed in args.seeds:
            checkpoints = run_experiment(
                seed=seed,
                shuffle_labels=shuffle_labels,
                depth=args.depth,
                block_dim=args.block_dim,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                checkpoint_epochs=checkpoint_epochs,
                device=args.device,
            )
            all_results[condition][f'seed_{seed}'] = {
                ep: {
                    'aggregate': data['aggregate'],
                    'test_acc': data['test_acc'],
                    'train_loss': data['train_loss'],
                }
                for ep, data in checkpoints.items()
            }

    # Compute trajectory averages
    def avg_metric(condition_data, metric_path):
        """Average a metric across seeds for each epoch."""
        trajectory = {}
        for ep in checkpoint_epochs:
            values = []
            for s in args.seeds:
                key = f'seed_{s}'
                if key in condition_data and ep in condition_data[key]:
                    val = condition_data[key][ep]['aggregate'][metric_path]
                    values.append(val)
            trajectory[ep] = float(np.mean(values)) if values else None
        return trajectory

    normal_s = avg_metric(all_results['normal'], 'mean_s')
    shuffled_s = avg_metric(all_results['shuffled'], 'mean_s')

    # Weight delta trajectories (P0.5 control)
    def avg_metric_optional(condition_data, metric_path):
        """Average a metric across seeds, returning None if metric missing."""
        trajectory = {}
        for ep in checkpoint_epochs:
            if ep == 0:  # No delta at epoch 0
                trajectory[ep] = 0.0
                continue
            values = []
            for s in args.seeds:
                key = f'seed_{s}'
                if key in condition_data and ep in condition_data[key]:
                    agg = condition_data[key][ep]['aggregate']
                    if metric_path in agg:
                        values.append(agg[metric_path])
            trajectory[ep] = float(np.mean(values)) if values else None
        return trajectory

    normal_delta = avg_metric_optional(all_results['normal'], 'mean_delta_W')
    shuffled_delta = avg_metric_optional(all_results['shuffled'], 'mean_delta_W')

    all_results['trajectory_avg'] = {
        'normal_mean_s': normal_s,
        'shuffled_mean_s': shuffled_s,
        'normal_mean_delta_W': normal_delta,
        'shuffled_mean_delta_W': shuffled_delta,
    }

    # P0.5: Compute correlation between s and ||ΔW|| across all blocks/epochs
    s_values = []
    delta_values = []
    for condition in ['normal', 'shuffled']:
        for seed in args.seeds:
            key = f'seed_{seed}'
            for ep in checkpoint_epochs:
                if ep == 0:
                    continue
                if key in all_results[condition] and ep in all_results[condition][key]:
                    agg = all_results[condition][key][ep]['aggregate']
                    if 'mean_delta_W' in agg:
                        s_values.append(agg['mean_s'])
                        delta_values.append(agg['mean_delta_W'])

    if len(s_values) > 2:
        corr = float(np.corrcoef(s_values, delta_values)[0, 1])
        all_results['weight_magnitude_control'] = {
            'correlation_s_deltaW': corr,
            'n_samples': len(s_values),
            'interpretation': (
                'positive: fingerprint may correlate with gradient flow' if corr > 0.5
                else 'negative: fingerprint inversely related to weight change' if corr < -0.5
                else 'weak: fingerprint largely independent of weight magnitude'
            )
        }
        print(f"\nP0.5 Weight Magnitude Control: r(s, ||ΔW||) = {corr:.3f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nFinal results (epoch {}):".format(checkpoint_epochs[-1]))
    print(f"{'Condition':<12} {'Seed':<6} {'Test Acc':<10} {'Mean s':<10} {'Neg Trace':<10}")
    print("-" * 50)

    for condition in ['normal', 'shuffled']:
        for seed in args.seeds:
            final = all_results[condition][f'seed_{seed}'][checkpoint_epochs[-1]]
            print(f"{condition:<12} {seed:<6} {final['test_acc']:.1%}      "
                  f"{final['aggregate']['mean_s']:.4f}     "
                  f"{final['aggregate']['frac_neg_trace']:.1%}")

    print("\nTrajectory (mean across seeds):")
    print(f"{'Epoch':<8} {'Normal s':<12} {'Shuffled s':<12}")
    print("-" * 35)
    for ep in checkpoint_epochs:
        ns = normal_s.get(ep)
        ss = shuffled_s.get(ep)
        ns_str = f"{ns:.4f}" if ns is not None else "N/A"
        ss_str = f"{ss:.4f}" if ss is not None else "N/A"
        print(f"{ep:<8} {ns_str:<12} {ss_str:<12}")

    # Save JSON
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {args.out}")

    # Create figure with 3 panels (added P0.5 weight magnitude control)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: s(i,i) trajectory
    ax = axes[0]
    epochs = checkpoint_epochs
    normal_means = [normal_s[ep] for ep in epochs]
    shuffled_means = [shuffled_s[ep] for ep in epochs]

    ax.plot(epochs, normal_means, 'o-', color='steelblue', label='Normal labels', linewidth=2, markersize=8)
    ax.plot(epochs, shuffled_means, 's--', color='coral', label='Shuffled labels', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Mean diagonal dominance $s(i,i)$')
    ax.set_title('Fingerprint Emergence: Normal vs Shuffled Labels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Test accuracy trajectory
    ax = axes[1]
    normal_acc = [all_results['normal'][f'seed_0'][ep]['test_acc'] for ep in epochs]
    shuffled_acc = [all_results['shuffled'][f'seed_0'][ep]['test_acc'] for ep in epochs]

    ax.plot(epochs, normal_acc, 'o-', color='steelblue', label='Normal labels', linewidth=2, markersize=8)
    ax.plot(epochs, shuffled_acc, 's--', color='coral', label='Shuffled labels', linewidth=2, markersize=8)
    ax.axhline(0.1, color='gray', linestyle=':', label='Chance (10%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Generalization: Normal vs Shuffled Labels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 3: P0.5 Weight Magnitude Control - s vs ||ΔW||
    ax = axes[2]
    if normal_delta and shuffled_delta:
        # Skip epoch 0 (no delta)
        epochs_nonzero = [ep for ep in epochs if ep > 0]
        normal_d = [normal_delta.get(ep, 0) for ep in epochs_nonzero]
        shuffled_d = [shuffled_delta.get(ep, 0) for ep in epochs_nonzero]
        normal_s_nonzero = [normal_s[ep] for ep in epochs_nonzero]
        shuffled_s_nonzero = [shuffled_s[ep] for ep in epochs_nonzero]

        ax.scatter(normal_d, normal_s_nonzero, c='steelblue', s=100, marker='o', label='Normal', edgecolors='black')
        ax.scatter(shuffled_d, shuffled_s_nonzero, c='coral', s=100, marker='s', label='Shuffled', edgecolors='black')

        # Annotate epochs
        for i, ep in enumerate(epochs_nonzero):
            ax.annotate(f'e{ep}', (normal_d[i], normal_s_nonzero[i]), textcoords='offset points',
                        xytext=(5, 5), fontsize=8, color='steelblue')
            ax.annotate(f'e{ep}', (shuffled_d[i], shuffled_s_nonzero[i]), textcoords='offset points',
                        xytext=(5, 5), fontsize=8, color='coral')

        ax.set_xlabel(r'Mean weight change $\|\Delta W\|_F$')
        ax.set_ylabel(r'Mean diagonal dominance $s(i,i)$')

        # Add correlation if computed
        if 'weight_magnitude_control' in all_results:
            corr = all_results['weight_magnitude_control']['correlation_s_deltaW']
            ax.set_title(f'P0.5 Control: s vs Weight Magnitude\n(r = {corr:.3f})')
        else:
            ax.set_title('P0.5 Control: s vs Weight Magnitude')

        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Weight magnitude\ndata not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('P0.5 Control: s vs Weight Magnitude')

    plt.tight_layout()
    plt.savefig('figures/fig_rq1_random_labels.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_rq1_random_labels.pdf', bbox_inches='tight')
    print("Saved figures/fig_rq1_random_labels.{png,pdf}")


if __name__ == "__main__":
    main()
