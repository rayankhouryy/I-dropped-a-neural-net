"""Issue #17: Validate diagonal-dominance fingerprint is absent in non-residual networks.

Tests whether the diagonal-dominance pairing signal requires residual connections.
We train both ResNet and PlainNet (identical architecture but without skip connections)
and compare their pairing metrics.

Expected Results:
  - ResNet (trained): pair_acc ~ 100%, strong diagonal in heatmap
  - PlainNet (trained): pair_acc ~ 1/depth (chance), no diagonal structure

This confirms the fingerprint is residual-connection-specific, not just a training artifact.

Outputs:
  results/nonresidual_baseline.json
  figures/fig_nonresidual_baseline.{pdf,png}
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------- models
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


# -------------------------------------------------------------------- init
def apply_init(model: nn.Module, scheme: str, seed: int):
    """Reinitialize all nn.Linear layers according to scheme."""
    g = torch.Generator().manual_seed(seed)

    def _init_linear(lin: nn.Linear):
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
            else:
                raise ValueError(f"unknown init scheme: {scheme}")
            lin.bias.zero_()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            _init_linear(m)


# -------------------------------------------------------------------- data
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


# -------------------------------------------------------------------- scoring
def diag_dominance_matrix(W_in_list, W_out_list):
    """Compute the d(i,j) diagonal-dominance score matrix."""
    n = len(W_in_list)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Ai = W_in_list[i].astype(np.float64)
        for j in range(n):
            Bj = W_out_list[j].astype(np.float64)
            P = Bj @ Ai
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, "fro") + 1e-12
            M[i, j] = tr / fr
    return M


def evaluate_pairing(M):
    """Compute pairing metrics from diagonal-dominance matrix."""
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    pos, neg = diag[:, None], off[None, :]
    auc = float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))
    return {
        "n": n,
        "pair_acc": pair_acc,
        "pair_sep": pair_sep,
        "auc": auc,
        "mean_correct": float(diag.mean()),
        "mean_incorrect": float(off.mean()),
    }


def trace_signs(W_in_list, W_out_list):
    """Compute fraction of correctly paired products with negative trace."""
    traces = []
    for Wi, Wo in zip(W_in_list, W_out_list):
        P = Wo.astype(np.float64) @ Wi.astype(np.float64)
        traces.append(float(np.trace(P)))
    traces = np.asarray(traces)
    return {
        "frac_negative": float((traces < 0).mean()),
        "mean_trace": float(traces.mean()),
    }


# -------------------------------------------------------------------- train
def train_and_score(arch, seed, depth, in_dim, hidden, epochs, lr, grad_clip=1.0, batch=256):
    """Train a network and return pairing metrics + the d(i,j) matrix."""
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
    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    init_loss = None
    final_loss = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        n_seen = 0
        for s in range(0, n_train, batch):
            idx = perm[s : s + batch]
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            yb_pred = model(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item() * idx.numel()
            n_seen += idx.numel()
        ep_loss /= n_seen
        if ep == 0:
            init_loss = ep_loss
        final_loss = ep_loss

    model.eval()
    with torch.no_grad():
        eval_loss = float(loss_fn(model(X_eval.to(DEVICE)).squeeze(-1), y_eval.to(DEVICE)).item())

    W_in_list = [b.inp.weight.detach().cpu().numpy() for b in model.blocks]
    W_out_list = [b.out.weight.detach().cpu().numpy() for b in model.blocks]

    M = diag_dominance_matrix(W_in_list, W_out_list)
    metrics = evaluate_pairing(M)
    metrics["trace"] = trace_signs(W_in_list, W_out_list)
    metrics["init_loss"] = init_loss
    metrics["final_loss"] = final_loss
    metrics["eval_loss"] = eval_loss

    return metrics, M


# -------------------------------------------------------------------- visualization
def make_figure(M_resnet, M_plainnet, stats_resnet, stats_plainnet, out_path):
    """Generate side-by-side heatmap figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    vmax = max(M_resnet.max(), M_plainnet.max())

    axes[0].imshow(M_resnet, cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title(
        f"(a) ResNet (trained)\n"
        f"pair acc = {stats_resnet['pair_acc']:.2f}, "
        f"sep = {stats_resnet['pair_sep']:+.2f}"
    )
    axes[0].set_xlabel(r"$W_\mathrm{out}$ index $j$")
    axes[0].set_ylabel(r"$W_\mathrm{in}$ index $i$")

    im1 = axes[1].imshow(M_plainnet, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(
        f"(b) PlainNet (trained, no skip)\n"
        f"pair acc = {stats_plainnet['pair_acc']:.2f}, "
        f"sep = {stats_plainnet['pair_sep']:+.2f}"
    )
    axes[1].set_xlabel(r"$W_\mathrm{out}$ index $j$")
    axes[1].set_ylabel(r"$W_\mathrm{in}$ index $i$")

    fig.colorbar(
        im1,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label=r"$d(i,j)=|\mathrm{tr}(W_\mathrm{out}^{(j)} W_\mathrm{in}^{(i)})|/\|\cdot\|_F$",
    )

    plt.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    plt.savefig(str(out_path) + ".png", dpi=160, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--in-dim", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--resnet-lr", type=float, default=1e-3)
    ap.add_argument("--plainnet-lr", type=float, default=1e-4)
    ap.add_argument("--out", default="results/nonresidual_baseline.json")
    ap.add_argument("--fig", default="../figures/fig_nonresidual_baseline")
    args = ap.parse_args()

    all_results = {
        "config": {
            "depth": args.depth,
            "hidden": args.hidden,
            "in_dim": args.in_dim,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "resnet_lr": args.resnet_lr,
            "plainnet_lr": args.plainnet_lr,
        },
        "resnet": {"per_run": [], "summary": {}},
        "plainnet": {"per_run": [], "summary": {}},
    }

    M_matrices = {"resnet": [], "plainnet": []}

    t0 = time.time()
    for arch in ["resnet", "plainnet"]:
        lr = args.resnet_lr if arch == "resnet" else args.plainnet_lr
        print(f"\n=== {arch.upper()} (lr={lr}) ===", flush=True)

        for seed in args.seeds:
            tic = time.time()
            metrics, M = train_and_score(
                arch,
                seed,
                depth=args.depth,
                in_dim=args.in_dim,
                hidden=args.hidden,
                epochs=args.epochs,
                lr=lr,
            )
            took = time.time() - tic
            all_results[arch]["per_run"].append({"seed": seed, **metrics})
            M_matrices[arch].append(M)
            print(
                f"  seed {seed}: pair_acc={metrics['pair_acc']:.0%}  "
                f"AUC={metrics['auc']:.3f}  sep={metrics['pair_sep']:+.3f}  "
                f"neg_tr={metrics['trace']['frac_negative']:.0%}  "
                f"eval_loss={metrics['eval_loss']:.4f}  ({took:.1f}s)",
                flush=True,
            )

        per_seed = all_results[arch]["per_run"]
        accs = [r["pair_acc"] for r in per_seed]
        aucs = [r["auc"] for r in per_seed]
        seps = [r["pair_sep"] for r in per_seed]
        ntr = [r["trace"]["frac_negative"] for r in per_seed]
        mc = [r["mean_correct"] for r in per_seed]
        mi = [r["mean_incorrect"] for r in per_seed]
        evlo = [r["eval_loss"] for r in per_seed]

        all_results[arch]["summary"] = {
            "mean_pair_acc": float(np.mean(accs)),
            "std_pair_acc": float(np.std(accs)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_pair_sep": float(np.mean(seps)),
            "std_pair_sep": float(np.std(seps)),
            "mean_frac_negative": float(np.mean(ntr)),
            "mean_correct": float(np.mean(mc)),
            "mean_incorrect": float(np.mean(mi)),
            "mean_eval_loss": float(np.mean(evlo)),
        }
        s = all_results[arch]["summary"]
        print(
            f"  AGG: pair_acc={s['mean_pair_acc']:.0%}+/-{s['std_pair_acc']:.0%}  "
            f"AUC={s['mean_auc']:.3f}+/-{s['std_auc']:.3f}  "
            f"sep={s['mean_pair_sep']:+.3f}+/-{s['std_pair_sep']:.3f}  "
            f"neg_tr={s['mean_frac_negative']:.0%}",
            flush=True,
        )

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    res_acc = all_results["resnet"]["summary"]["mean_pair_acc"]
    plain_acc = all_results["plainnet"]["summary"]["mean_pair_acc"]
    chance = 1.0 / args.depth
    all_results["hypothesis_confirmed"] = res_acc > 0.9 and plain_acc < 0.15
    all_results["conclusion"] = (
        f"ResNet achieves {res_acc:.0%} pair accuracy while PlainNet achieves "
        f"{plain_acc:.0%} (chance={chance:.1%}). "
        f"{'Confirmed' if all_results['hypothesis_confirmed'] else 'Inconclusive'}: "
        f"diagonal-dominance fingerprint requires residual connections."
    )

    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)
    print(f"Conclusion: {all_results['conclusion']}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)

    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    M_resnet = M_matrices["resnet"][0]
    M_plainnet = M_matrices["plainnet"][0]
    stats_resnet = all_results["resnet"]["per_run"][0]
    stats_plainnet = all_results["plainnet"]["per_run"][0]
    make_figure(M_resnet, M_plainnet, stats_resnet, stats_plainnet, fig_path)
    print(f"Wrote {fig_path}.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    main()
