"""Case Study 1: Training Quality Assurance

Tests whether the diagonal dominance fingerprint can serve as an early warning
system for training pathologies. Trains models under various pathological
conditions and tracks DD emergence over epochs.

Pathologies tested:
  1. healthy_baseline - Standard training (control)
  2. lr_too_low       - LR=1e-6 (1000x smaller)
  3. lr_too_high      - LR=1e-2 (10x larger)
  4. no_skip          - PlainNet without skip connections
  5. high_weight_decay - weight_decay=1.0
  6. small_init       - Gaussian(0, 0.02^2) initialization

Outputs:
  case_studies/case_study_1/training_qa_results.csv
  case_studies/case_study_1/training_qa_summary.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
            elif scheme == "gaussian_002":
                W.copy_(torch.randn(W.shape, generator=g) * 0.02)
            elif scheme == "orthogonal":
                tmp = torch.empty_like(W)
                nn.init.orthogonal_(tmp, gain=1.0)
                W.copy_(tmp)
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
    off_max_per_row = (M - np.diag(diag) * np.eye(n)).max(axis=1)
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


def compute_dd_metrics(model):
    """Compute all DD metrics for a model."""
    W_in_list = [b.inp.weight.detach().cpu().numpy() for b in model.blocks]
    W_out_list = [b.out.weight.detach().cpu().numpy() for b in model.blocks]

    M = diag_dominance_matrix(W_in_list, W_out_list)
    metrics = evaluate_pairing(M)
    trace_info = trace_signs(W_in_list, W_out_list)

    metrics["pct_negative_trace"] = trace_info["frac_negative"]
    metrics["mean_trace"] = trace_info["mean_trace"]
    metrics["dd_matrix"] = M

    return metrics


# -------------------------------------------------------------------- configs
PATHOLOGY_CONFIGS = {
    "healthy_baseline": {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "init_scheme": "kaiming_normal",
        "use_skip": True,
        "description": "Standard training (control)",
    },
    "lr_too_low": {
        "lr": 1e-6,
        "weight_decay": 0.0,
        "init_scheme": "kaiming_normal",
        "use_skip": True,
        "description": "LR=1e-6 (1000x smaller)",
    },
    "lr_too_high": {
        "lr": 1e-2,
        "weight_decay": 0.0,
        "init_scheme": "kaiming_normal",
        "use_skip": True,
        "description": "LR=1e-2 (10x larger)",
    },
    "no_skip": {
        "lr": 1e-4,  # PlainNet needs smaller LR
        "weight_decay": 0.0,
        "init_scheme": "xavier_normal",
        "use_skip": False,
        "description": "PlainNet without skip connections",
    },
    "high_weight_decay": {
        "lr": 1e-3,
        "weight_decay": 1.0,
        "init_scheme": "kaiming_normal",
        "use_skip": True,
        "description": "weight_decay=1.0",
    },
    "small_init": {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "init_scheme": "gaussian_002",
        "use_skip": True,
        "description": "Gaussian(0, 0.02^2) initialization",
    },
}

# Dense early checkpoints for early warning detection
CHECKPOINTS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]


# -------------------------------------------------------------------- train
def train_with_pathology(pathology_name, seed, depth=24, hidden=64, in_dim=24,
                         epochs=200, batch=256, grad_clip=1.0, verbose=True):
    """Train a model with a specific pathology and collect metrics at checkpoints."""
    config = PATHOLOGY_CONFIGS[pathology_name]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    if config["use_skip"]:
        model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
    else:
        model = PlainNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)

    apply_init(model, config["init_scheme"], seed=seed * 1000 + 1)
    model = model.to(DEVICE)

    # Data
    X, y = make_data(in_dim=in_dim, n=4000, seed=seed)
    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    rows = []
    checkpoints_set = set(cp for cp in CHECKPOINTS if cp <= epochs)

    for epoch in range(epochs + 1):
        # Checkpoint evaluation
        if epoch in checkpoints_set:
            model.eval()
            with torch.no_grad():
                eval_pred = model(X_eval.to(DEVICE)).squeeze(-1)
                eval_loss = float(loss_fn(eval_pred, y_eval.to(DEVICE)).item())
                train_pred = model(X_train.to(DEVICE)).squeeze(-1)
                train_loss = float(loss_fn(train_pred, y_train.to(DEVICE)).item())

            dd_metrics = compute_dd_metrics(model)

            row = {
                "pathology": pathology_name,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "pair_acc": dd_metrics["pair_acc"],
                "pair_sep": dd_metrics["pair_sep"],
                "mean_dd_score": dd_metrics["mean_correct"],
                "mean_incorrect": dd_metrics["mean_incorrect"],
                "pct_negative_trace": dd_metrics["pct_negative_trace"],
                "auc": dd_metrics["auc"],
            }
            rows.append(row)

            if verbose:
                print(f"  [{pathology_name} s{seed}] epoch={epoch:4d}  "
                      f"loss={train_loss:.4f}  pair_acc={dd_metrics['pair_acc']:.2f}  "
                      f"sep={dd_metrics['pair_sep']:+.3f}  neg_tr={dd_metrics['pct_negative_trace']:.0%}")

        if epoch == epochs:
            break

        # Training step
        model.train()
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            yb_pred = model(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)

            # Check for NaN/Inf (pathology detection)
            if not torch.isfinite(loss):
                if verbose:
                    print(f"  [{pathology_name} s{seed}] DIVERGED at epoch {epoch}")
                # Fill remaining checkpoints with NaN
                for cp in checkpoints_set:
                    if cp > epoch:
                        rows.append({
                            "pathology": pathology_name,
                            "seed": seed,
                            "epoch": cp,
                            "train_loss": float('nan'),
                            "eval_loss": float('nan'),
                            "pair_acc": float('nan'),
                            "pair_sep": float('nan'),
                            "mean_dd_score": float('nan'),
                            "mean_incorrect": float('nan'),
                            "pct_negative_trace": float('nan'),
                            "auc": float('nan'),
                        })
                return rows

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

    return rows


def run_all_pathologies(seeds, output_dir, verbose=True):
    """Run all pathology experiments and save results."""
    all_rows = []
    t0 = time.time()

    for pathology_name in PATHOLOGY_CONFIGS:
        config = PATHOLOGY_CONFIGS[pathology_name]
        print(f"\n=== {pathology_name}: {config['description']} ===", flush=True)

        for seed in seeds:
            tic = time.time()
            rows = train_with_pathology(pathology_name, seed, verbose=verbose)
            all_rows.extend(rows)
            took = time.time() - tic

            # Summary for this seed
            final = [r for r in rows if r["epoch"] == 200]
            if final and not np.isnan(final[0]["pair_acc"]):
                f = final[0]
                print(f"  seed {seed} FINAL: pair_acc={f['pair_acc']:.0%}  "
                      f"eval_loss={f['eval_loss']:.4f}  ({took:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "training_qa_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(all_rows)} rows to {csv_path}")

    # Compute summary statistics
    summary = compute_summary(df)
    summary["elapsed_seconds"] = elapsed

    json_path = output_dir / "training_qa_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

    return df, summary


def compute_summary(df):
    """Compute summary statistics from results DataFrame."""
    summary = {
        "config": {
            "depth": 24,
            "hidden": 64,
            "in_dim": 24,
            "epochs": 200,
            "checkpoints": CHECKPOINTS,
        },
        "pathologies": {},
    }

    for pathology in df["pathology"].unique():
        pdf = df[df["pathology"] == pathology]

        # Final epoch stats
        final = pdf[pdf["epoch"] == 200]

        # Emergence epoch (first epoch with pair_acc > 0.9)
        emergence_epochs = []
        for seed in pdf["seed"].unique():
            sdf = pdf[pdf["seed"] == seed]
            emerged = sdf[sdf["pair_acc"] > 0.9]
            if len(emerged) > 0:
                emergence_epochs.append(int(emerged["epoch"].min()))
            else:
                emergence_epochs.append(None)

        # Early warning metrics (epoch 5 and 10)
        ep5 = pdf[pdf["epoch"] == 5]
        ep10 = pdf[pdf["epoch"] == 10]

        summary["pathologies"][pathology] = {
            "description": PATHOLOGY_CONFIGS[pathology]["description"],
            "final": {
                "mean_pair_acc": float(final["pair_acc"].mean()),
                "std_pair_acc": float(final["pair_acc"].std()),
                "mean_eval_loss": float(final["eval_loss"].mean()),
                "std_eval_loss": float(final["eval_loss"].std()),
                "mean_pct_negative_trace": float(final["pct_negative_trace"].mean()),
            },
            "emergence_epochs": emergence_epochs,
            "mean_emergence_epoch": float(np.nanmean([e for e in emergence_epochs if e is not None])) if any(e is not None for e in emergence_epochs) else None,
            "early_warning": {
                "epoch_5_pair_acc": float(ep5["pair_acc"].mean()) if len(ep5) > 0 else None,
                "epoch_10_pair_acc": float(ep10["pair_acc"].mean()) if len(ep10) > 0 else None,
            },
        }

    return summary


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default="../case_studies/case_study_1")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    output_dir = Path(__file__).parent / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print(f"Running Training QA Case Study")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")

    df, summary = run_all_pathologies(args.seeds, output_dir, verbose=not args.quiet)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Pathology':<20} {'Final Acc':>10} {'Emerge@':>10} {'Ep5 Acc':>10} {'Ep10 Acc':>10}")
    print("-"*70)
    for pathology, stats in summary["pathologies"].items():
        final_acc = f"{stats['final']['mean_pair_acc']:.0%}" if not np.isnan(stats['final']['mean_pair_acc']) else "NaN"
        emerge = f"{stats['mean_emergence_epoch']:.0f}" if stats['mean_emergence_epoch'] is not None else "Never"
        ep5 = f"{stats['early_warning']['epoch_5_pair_acc']:.0%}" if stats['early_warning']['epoch_5_pair_acc'] is not None else "N/A"
        ep10 = f"{stats['early_warning']['epoch_10_pair_acc']:.0%}" if stats['early_warning']['epoch_10_pair_acc'] is not None else "N/A"
        print(f"{pathology:<20} {final_acc:>10} {emerge:>10} {ep5:>10} {ep10:>10}")


if __name__ == "__main__":
    main()
