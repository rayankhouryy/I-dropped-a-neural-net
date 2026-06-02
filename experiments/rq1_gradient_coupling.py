"""
RQ1 Section 9: Gradient Coupling Hypothesis Experiments

Tests whether gradient coupling between W_in and W_out creates the diagonal
dominance fingerprint.

Part A: Measures gradient correlation during normal training
Part B: Ablation with gradient shuffling across blocks
Part C: Synthetic gradient injection to create fingerprint without backprop

Outputs:
  results/rq1_gradient_coupling.json
  figures/fig_rq1_gradient_coupling.png
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
def get_cifar_loaders(batch_size: int = 256):
    """Load CIFAR-10."""
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


def extract_weight_metrics(model: nn.Module):
    """Extract diagonal dominance metrics from weight products M = W_out @ W_in."""
    results = []
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
        })

    return results


def extract_gradient_metrics(model: nn.Module):
    """Extract diagonal dominance metrics from gradient products G = grad_W_out @ grad_W_in.

    Must be called after loss.backward() and before optimizer.step().

    Weight shapes in nn.Linear:
        inp.weight: (hidden_dim, in_dim) - maps in_dim -> hidden_dim
        out.weight: (in_dim, hidden_dim) - maps hidden_dim -> in_dim

    So grad_W_out @ grad_W_in gives shape (in_dim, in_dim), analogous to M = W_out @ W_in.
    """
    results = []
    for i, blk in enumerate(model.blocks):
        if blk.inp.weight.grad is None or blk.out.weight.grad is None:
            continue

        grad_W_in = blk.inp.weight.grad.detach().cpu().numpy().astype(np.float64)
        grad_W_out = blk.out.weight.grad.detach().cpu().numpy().astype(np.float64)

        # G = grad_W_out @ grad_W_in has shape (in_dim, in_dim)
        # grad_W_out: (in_dim, hidden_dim), grad_W_in: (hidden_dim, in_dim)
        G = grad_W_out @ grad_W_in

        tr = float(np.trace(G))
        frob = float(np.linalg.norm(G, 'fro'))
        results.append({
            'block': i,
            'g_trace': tr,
            'g_frob': frob,
            'g_diag': diagonal_dominance(G),
            'g_trace_negative': tr < 0,
        })

    return results


def aggregate_weight_metrics(block_results: list):
    """Aggregate per-block weight metrics."""
    return {
        'mean_s': float(np.mean([r['diag_dominance_s'] for r in block_results])),
        'frac_neg_trace': float(np.mean([r['trace_negative'] for r in block_results])),
        'mean_trace': float(np.mean([r['trace'] for r in block_results])),
    }


def aggregate_gradient_metrics(block_results: list):
    """Aggregate per-block gradient metrics."""
    if not block_results:
        return {'mean_g_diag': 0, 'frac_g_neg_trace': 0}
    return {
        'mean_g_diag': float(np.mean([r['g_diag'] for r in block_results])),
        'frac_g_neg_trace': float(np.mean([r['g_trace_negative'] for r in block_results])),
    }


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


# --------------------------------------------------------------------- Part A
def run_part_a(seed: int, depth: int, block_dim: int, hidden_dim: int,
               epochs: int, sample_every_n_steps: int, device: str):
    """
    Part A: Measure gradient correlation during normal training.

    Tracks gradient diagonal dominance (g_diag) alongside weight diagonal dominance (s).
    """
    print(f"\n[Part A] Gradient correlation during training (seed={seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_cifar_loaders(batch_size=256)

    input_dim = 3 * 32 * 32
    model = ResNetMLP(
        input_dim=input_dim,
        block_dim=block_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        num_classes=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    gradient_samples = []
    weight_checkpoints = []
    global_step = 0

    checkpoint_epochs = [0, 5, 10, 25, 50, epochs]

    # Epoch 0 checkpoint
    w_metrics = extract_weight_metrics(model)
    test_acc = evaluate(model, test_loader, device)
    weight_checkpoints.append({
        'epoch': 0,
        'aggregate': aggregate_weight_metrics(w_metrics),
        'test_acc': test_acc,
    })
    print(f"  Epoch 0: s={weight_checkpoints[-1]['aggregate']['mean_s']:.4f}, acc={test_acc:.1%}")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_samples = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            # Sample gradient metrics before optimizer step
            if global_step % sample_every_n_steps == 0:
                g_metrics = extract_gradient_metrics(model)
                w_metrics = extract_weight_metrics(model)
                gradient_samples.append({
                    'epoch': ep,
                    'step': global_step,
                    'gradient': aggregate_gradient_metrics(g_metrics),
                    'weight': aggregate_weight_metrics(w_metrics),
                })

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            global_step += 1

        # Epoch checkpoint
        if ep in checkpoint_epochs:
            w_metrics = extract_weight_metrics(model)
            test_acc = evaluate(model, test_loader, device)
            agg = aggregate_weight_metrics(w_metrics)
            weight_checkpoints.append({
                'epoch': ep,
                'aggregate': agg,
                'test_acc': test_acc,
                'train_loss': epoch_loss / n_samples,
            })
            elapsed = time.time() - t0
            print(f"  Epoch {ep}: s={agg['mean_s']:.4f}, neg_tr={agg['frac_neg_trace']:.1%}, "
                  f"acc={test_acc:.1%} ({elapsed:.0f}s)")

    # Compute correlation between g_diag and s across all samples
    if gradient_samples:
        g_diags = [s['gradient']['mean_g_diag'] for s in gradient_samples]
        s_vals = [s['weight']['mean_s'] for s in gradient_samples]
        correlation = float(np.corrcoef(g_diags, s_vals)[0, 1])
    else:
        correlation = 0.0

    return {
        'gradient_samples': gradient_samples,
        'weight_checkpoints': weight_checkpoints,
        'correlation_g_s': correlation,
    }


# --------------------------------------------------------------------- Part B
def run_part_b(seed: int, depth: int, block_dim: int, hidden_dim: int,
               epochs: int, shuffle_gradients: bool, device: str):
    """
    Part B: Gradient shuffling ablation.

    If shuffle_gradients=True, permute grad_W_out across blocks before optimizer step.
    """
    condition = "shuffled" if shuffle_gradients else "control"
    print(f"\n[Part B] Gradient shuffle ablation (seed={seed}, {condition})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_cifar_loaders(batch_size=256)

    input_dim = 3 * 32 * 32
    model = ResNetMLP(
        input_dim=input_dim,
        block_dim=block_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        num_classes=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    checkpoints = []
    checkpoint_epochs = [0, 5, 10, 25, 50, epochs]

    # Epoch 0
    w_metrics = extract_weight_metrics(model)
    test_acc = evaluate(model, test_loader, device)
    checkpoints.append({
        'epoch': 0,
        'aggregate': aggregate_weight_metrics(w_metrics),
        'test_acc': test_acc,
    })
    print(f"  Epoch 0: s={checkpoints[-1]['aggregate']['mean_s']:.4f}")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_samples = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            # Shuffle gradients across blocks
            if shuffle_gradients:
                grad_outs = [blk.out.weight.grad.detach().clone()
                             for blk in model.blocks]
                perm = torch.randperm(len(grad_outs))
                with torch.no_grad():
                    for i, blk in enumerate(model.blocks):
                        blk.out.weight.grad.copy_(grad_outs[perm[i]])

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)
            n_samples += X.size(0)

        if ep in checkpoint_epochs:
            w_metrics = extract_weight_metrics(model)
            test_acc = evaluate(model, test_loader, device)
            agg = aggregate_weight_metrics(w_metrics)
            checkpoints.append({
                'epoch': ep,
                'aggregate': agg,
                'test_acc': test_acc,
                'train_loss': epoch_loss / n_samples,
            })
            elapsed = time.time() - t0
            print(f"  Epoch {ep}: s={agg['mean_s']:.4f}, neg_tr={agg['frac_neg_trace']:.1%}, "
                  f"acc={test_acc:.1%} ({elapsed:.0f}s)")

    return {
        'condition': condition,
        'checkpoints': checkpoints,
    }


# --------------------------------------------------------------------- Part C
def run_part_c(seed: int, depth: int, block_dim: int, hidden_dim: int,
               n_steps: int, epsilon: float, device: str):
    """
    Part C: Synthetic weight updates to create diagonal structure.

    Directly perturb weights such that M = W_out @ W_in develops diagonal
    structure with negative trace, simulating what gradient coupling would do.

    For each step:
        M_new = M + delta_M, where delta_M ≈ -epsilon * I + noise
        We achieve this by: W_out_new = W_out + delta_out, W_in unchanged
        where delta_out is chosen so W_out_new @ W_in has more diagonal structure.
    """
    print(f"\n[Part C] Synthetic diagonal injection (seed={seed}, epsilon={epsilon})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = 3 * 32 * 32
    model = ResNetMLP(
        input_dim=input_dim,
        block_dim=block_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        num_classes=10
    ).to(device)

    checkpoints = []
    checkpoint_steps = [0, 100, 500, 1000, 2000, n_steps]

    # Step 0
    w_metrics = extract_weight_metrics(model)
    checkpoints.append({
        'step': 0,
        'aggregate': aggregate_weight_metrics(w_metrics),
    })
    print(f"  Step 0: s={checkpoints[-1]['aggregate']['mean_s']:.4f}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # For each block, add a small diagonal contribution to M = W_out @ W_in
        # We want: M_new = M + delta_M, where delta_M = -epsilon * I + noise
        # Since M = W_out @ W_in, we can add delta_out to W_out:
        #   M_new = (W_out + delta_out) @ W_in = M + delta_out @ W_in
        # So we need: delta_out @ W_in = -epsilon * I + noise
        #   delta_out = (-epsilon * I + noise) @ pinv(W_in)
        with torch.no_grad():
            for blk in model.blocks:
                W_in = blk.inp.weight  # (hidden_dim, in_dim)
                in_dim = W_in.shape[1]

                # Target change to M: diagonal with negative trace plus noise
                noise = torch.randn(in_dim, in_dim, device=device) * 0.001
                delta_M = -epsilon * torch.eye(in_dim, device=device) + noise

                # Compute delta_out such that delta_out @ W_in = delta_M
                # delta_out = delta_M @ pinv(W_in)
                delta_out = delta_M @ torch.linalg.pinv(W_in, rcond=1e-4)

                # Scale down to avoid exploding weights
                delta_out = delta_out * 0.01

                # Update W_out
                blk.out.weight.add_(delta_out)

        if step in checkpoint_steps:
            w_metrics = extract_weight_metrics(model)
            agg = aggregate_weight_metrics(w_metrics)
            checkpoints.append({
                'step': step,
                'aggregate': agg,
            })
            elapsed = time.time() - t0
            print(f"  Step {step}: s={agg['mean_s']:.4f}, "
                  f"neg_tr={agg['frac_neg_trace']:.1%} ({elapsed:.0f}s)")

    return {
        'epsilon': epsilon,
        'checkpoints': checkpoints,
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--block-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50, help="Epochs for Part A and B")
    ap.add_argument("--n-steps", type=int, default=3000, help="Steps for Part C")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample-every", type=int, default=50, help="Sample gradients every N steps")
    ap.add_argument("--out", default="results/rq1_gradient_coupling.json")
    ap.add_argument("--part", choices=["a", "b", "c", "all"], default="all",
                    help="Run specific part or all")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    print("=" * 60)
    print("RQ1 Section 9: Gradient Coupling Hypothesis")
    print("=" * 60)
    print(f"Config: depth={args.depth}, block_dim={args.block_dim}, "
          f"hidden_dim={args.hidden_dim}")
    print(f"Device: {args.device}")

    results = {
        'config': {
            'depth': args.depth,
            'block_dim': args.block_dim,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'n_steps': args.n_steps,
            'seeds': args.seeds,
        },
    }

    # Part A: Gradient correlation during training
    if args.part in ["a", "all"]:
        print("\n" + "=" * 60)
        print("PART A: Gradient Correlation During Training")
        print("=" * 60)
        results['part_a'] = {}
        for seed in args.seeds:
            results['part_a'][f'seed_{seed}'] = run_part_a(
                seed=seed,
                depth=args.depth,
                block_dim=args.block_dim,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                sample_every_n_steps=args.sample_every,
                device=args.device
            )

        # Aggregate correlation across seeds
        correlations = [results['part_a'][f'seed_{s}']['correlation_g_s']
                        for s in args.seeds]
        results['part_a']['aggregate'] = {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
        }
        print(f"\n[Part A Summary] Mean correlation(g_diag, s) = "
              f"{results['part_a']['aggregate']['mean_correlation']:.3f}")

    # Part B: Gradient shuffling ablation
    if args.part in ["b", "all"]:
        print("\n" + "=" * 60)
        print("PART B: Gradient Shuffling Ablation")
        print("=" * 60)
        results['part_b'] = {'control': {}, 'shuffled': {}}

        for seed in args.seeds:
            # Control (normal training)
            results['part_b']['control'][f'seed_{seed}'] = run_part_b(
                seed=seed,
                depth=args.depth,
                block_dim=args.block_dim,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                shuffle_gradients=False,
                device=args.device
            )

            # Shuffled gradients
            results['part_b']['shuffled'][f'seed_{seed}'] = run_part_b(
                seed=seed,
                depth=args.depth,
                block_dim=args.block_dim,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                shuffle_gradients=True,
                device=args.device
            )

        # Aggregate final metrics
        def get_final(condition):
            finals = []
            for seed in args.seeds:
                chkpts = results['part_b'][condition][f'seed_{seed}']['checkpoints']
                finals.append(chkpts[-1]['aggregate'])
            return {
                'mean_s': float(np.mean([f['mean_s'] for f in finals])),
                'mean_neg_trace': float(np.mean([f['frac_neg_trace'] for f in finals])),
            }

        results['part_b']['aggregate'] = {
            'control': get_final('control'),
            'shuffled': get_final('shuffled'),
        }
        print(f"\n[Part B Summary]")
        print(f"  Control:  s={results['part_b']['aggregate']['control']['mean_s']:.3f}, "
              f"neg_tr={results['part_b']['aggregate']['control']['mean_neg_trace']:.1%}")
        print(f"  Shuffled: s={results['part_b']['aggregate']['shuffled']['mean_s']:.3f}, "
              f"neg_tr={results['part_b']['aggregate']['shuffled']['mean_neg_trace']:.1%}")

    # Part C: Synthetic gradient injection
    if args.part in ["c", "all"]:
        print("\n" + "=" * 60)
        print("PART C: Synthetic Gradient Injection")
        print("=" * 60)
        results['part_c'] = {}

        epsilons = [0.0, 0.01, 0.1]  # 0.0 is random control
        for eps in epsilons:
            results['part_c'][f'eps_{eps}'] = {}
            for seed in args.seeds:
                results['part_c'][f'eps_{eps}'][f'seed_{seed}'] = run_part_c(
                    seed=seed,
                    depth=args.depth,
                    block_dim=args.block_dim,
                    hidden_dim=args.hidden_dim,
                    n_steps=args.n_steps,
                    epsilon=eps,
                    device=args.device
                )

        # Aggregate final metrics per epsilon
        results['part_c']['aggregate'] = {}
        for eps in epsilons:
            finals = []
            for seed in args.seeds:
                chkpts = results['part_c'][f'eps_{eps}'][f'seed_{seed}']['checkpoints']
                finals.append(chkpts[-1]['aggregate'])
            results['part_c']['aggregate'][f'eps_{eps}'] = {
                'mean_s': float(np.mean([f['mean_s'] for f in finals])),
                'mean_neg_trace': float(np.mean([f['frac_neg_trace'] for f in finals])),
            }

        print(f"\n[Part C Summary]")
        for eps in epsilons:
            agg = results['part_c']['aggregate'][f'eps_{eps}']
            print(f"  eps={eps}: s={agg['mean_s']:.3f}, neg_tr={agg['mean_neg_trace']:.1%}")

    # Save results
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")

    # Generate figure
    if args.part == "all":
        generate_figure(results, args.seeds)


def generate_figure(results, seeds):
    """Generate 4-panel figure summarizing all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Part A - g_diag trajectory
    ax = axes[0, 0]
    for seed in seeds:
        samples = results['part_a'][f'seed_{seed}']['gradient_samples']
        if samples:
            steps = [s['step'] for s in samples]
            g_diags = [s['gradient']['mean_g_diag'] for s in samples]
            ax.plot(steps, g_diags, alpha=0.7, label=f'seed {seed}')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Diagonal Dominance (g_diag)')
    ax.set_title('Part A: Gradient Structure During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Part A - correlation scatter
    ax = axes[0, 1]
    all_g = []
    all_s = []
    for seed in seeds:
        samples = results['part_a'][f'seed_{seed}']['gradient_samples']
        for s in samples:
            all_g.append(s['gradient']['mean_g_diag'])
            all_s.append(s['weight']['mean_s'])
    ax.scatter(all_g, all_s, alpha=0.3, s=10)
    corr = results['part_a']['aggregate']['mean_correlation']
    ax.set_xlabel('Gradient Diagonal Dominance (g_diag)')
    ax.set_ylabel('Weight Diagonal Dominance (s)')
    ax.set_title(f'Part A: Gradient-Weight Correlation (r={corr:.3f})')
    ax.grid(True, alpha=0.3)

    # Panel 3: Part B - shuffle comparison
    ax = axes[1, 0]
    conditions = ['control', 'shuffled']
    x = np.arange(2)
    width = 0.35

    s_vals = [results['part_b']['aggregate'][c]['mean_s'] for c in conditions]
    neg_tr = [results['part_b']['aggregate'][c]['mean_neg_trace'] for c in conditions]

    bars1 = ax.bar(x - width/2, s_vals, width, label='Mean s', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, neg_tr, width, label='Frac Neg Trace', color='coral')

    ax.set_ylabel('Diagonal Dominance (s)', color='steelblue')
    ax2.set_ylabel('Fraction Negative Trace', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(['Control', 'Shuffled'])
    ax.set_title('Part B: Gradient Shuffling Ablation')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Panel 4: Part C - synthetic gradient trajectory
    ax = axes[1, 1]
    epsilons = [0.0, 0.01, 0.1]
    colors = ['gray', 'orange', 'green']

    for eps, color in zip(epsilons, colors):
        # Average across seeds
        all_steps = None
        all_s_vals = []
        for seed in seeds:
            chkpts = results['part_c'][f'eps_{eps}'][f'seed_{seed}']['checkpoints']
            steps = [c['step'] for c in chkpts]
            s_vals = [c['aggregate']['mean_s'] for c in chkpts]
            if all_steps is None:
                all_steps = steps
            all_s_vals.append(s_vals)

        mean_s = np.mean(all_s_vals, axis=0)
        ax.plot(all_steps, mean_s, 'o-', color=color, label=f'eps={eps}')

    ax.set_xlabel('Step')
    ax.set_ylabel('Diagonal Dominance (s)')
    ax.set_title('Part C: Synthetic Gradient Injection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_rq1_gradient_coupling.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_rq1_gradient_coupling.pdf', bbox_inches='tight')
    print("Saved figures/fig_rq1_gradient_coupling.{png,pdf}")


if __name__ == '__main__':
    main()
