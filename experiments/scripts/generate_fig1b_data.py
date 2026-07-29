"""Generate real score matrix data for Figure 1b.

Trains ResNet and PlainNet using the exact config from nonresidual_baseline.py
that achieved 100% vs 3% accuracy. Saves score matrices at key epochs.

Outputs: results/fig1b_score_matrices.json
"""
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.optimize import linear_sum_assignment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class Block(nn.Module):
    """Residual block: output = x + f(x)"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))


class PlainBlock(nn.Module):
    """Non-residual block: output = f(x), NO skip connection"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.out(F.relu(self.inp(x)))


class ResNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList([Block(in_dim, hidden_dim) for _ in range(depth)])
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.last(x)


class PlainNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList([PlainBlock(in_dim, hidden_dim) for _ in range(depth)])
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.last(x)


def apply_init(model, scheme, seed):
    """Reinitialize all nn.Linear layers according to scheme."""
    g = torch.Generator().manual_seed(seed)

    def _init_linear(lin):
        W = lin.weight
        with torch.no_grad():
            if scheme == "kaiming_normal":
                fan_in = W.shape[1]
                std = math.sqrt(2.0 / fan_in)
                W.copy_(torch.randn(W.shape, generator=g) * std)
            elif scheme == "xavier_normal":
                fan_in, fan_out = W.shape[1], W.shape[0]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                W.copy_(torch.randn(W.shape, generator=g) * std)
            lin.bias.zero_()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            _init_linear(m)


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


def compute_score_matrix(model):
    """Compute cross-block score matrix s(i,j)."""
    W_in_list = [b.inp.weight.detach().cpu().numpy().astype(np.float64) for b in model.blocks]
    W_out_list = [b.out.weight.detach().cpu().numpy().astype(np.float64) for b in model.blocks]

    n = len(W_in_list)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            P = W_out_list[j] @ W_in_list[i]
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, "fro") + 1e-12
            M[i, j] = tr / fr
    return M


def hungarian_accuracy(S):
    """Compute pair accuracy using Hungarian matching."""
    n = S.shape[0]
    _, col = linear_sum_assignment(-S)
    return float((col == np.arange(n)).mean())


def train_with_checkpoints(arch, seed, depth, in_dim, hidden, epochs, lr, checkpoint_epochs, batch=256, grad_clip=1.0):
    """Train a network and save score matrices at checkpoint epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if arch == "resnet":
        model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
        apply_init(model, "kaiming_normal", seed=seed * 1000 + 1)
    else:
        model = PlainNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
        apply_init(model, "xavier_normal", seed=seed * 1000 + 1)

    model = model.to(DEVICE)

    X, y = make_data(in_dim=in_dim, n=4000, seed=seed)
    X_train, y_train = X[1000:], y[1000:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    checkpoints = {}

    for ep in range(epochs + 1):
        # Save checkpoint before training this epoch
        if ep in checkpoint_epochs:
            S = compute_score_matrix(model)
            acc = hungarian_accuracy(S)
            checkpoints[f"epoch_{ep}"] = {
                "score_matrix": S.tolist(),
                "pair_accuracy": float(acc),
                "mean_diagonal": float(np.mean(np.diag(S))),
                "mean_off_diagonal": float(np.mean(S[~np.eye(depth, dtype=bool)]))
            }
            print(f"  {arch} Epoch {ep}: pair_acc={acc:.2%}, mean_diag={np.mean(np.diag(S)):.3f}, mean_offdiag={np.mean(S[~np.eye(depth, dtype=bool)]):.3f}")

        if ep == epochs:
            break

        # Train one epoch
        model.train()
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            yb_pred = model(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

    return checkpoints


def main():
    # Config from nonresidual_baseline.json that achieved 100% vs 3%
    depth = 24
    hidden = 64
    in_dim = 24
    epochs = 200
    seed = 0
    resnet_lr = 0.001
    plainnet_lr = 0.0001

    # Checkpoint at key epochs showing progression
    checkpoint_epochs = [0, 25, 75, 150, 200]

    results = {
        "config": {
            "depth": depth,
            "hidden": hidden,
            "in_dim": in_dim,
            "epochs": epochs,
            "resnet_lr": resnet_lr,
            "plainnet_lr": plainnet_lr,
            "checkpoint_epochs": checkpoint_epochs
        },
        "resnet": {},
        "plainnet": {}
    }

    print("Training ResNet...")
    results["resnet"] = train_with_checkpoints(
        "resnet", seed, depth, in_dim, hidden, epochs, resnet_lr, checkpoint_epochs
    )

    print("\nTraining PlainNet...")
    results["plainnet"] = train_with_checkpoints(
        "plainnet", seed, depth, in_dim, hidden, epochs, plainnet_lr, checkpoint_epochs
    )

    # Save results
    out_path = Path(__file__).parent.parent / "results" / "fig1b_score_matrices.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
