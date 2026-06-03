"""Issue #23: Test fingerprint behavior under non-isometric initialization.

Trains depth-24 residual MLPs from four different initialization schemes
and measures whether the diagonal-dominance pairing fingerprint still
emerges. Tests whether dynamical isometry is a *requirement* of the
initial state or an *attractor* enforced by gradient descent regardless
of starting point.

Inits compared:
  1. orthogonal      - exact dynamical isometry at init (theoretical sweet spot)
  2. kaiming_normal  - He init, common default for ReLU networks
  3. xavier_normal   - Glorot init, common default for tanh/sigmoid
  4. gaussian_002    - plain Gaussian N(0, 0.02^2), small-scale and explicitly
                       non-isometric (used by GPT-2 / BERT / LLaMA)

For each (init, seed) we train to convergence on the synthetic regression
target from pipeline.py, then score the diagonal-dominance Hungarian
pairing on the residual blocks. We report pair_acc, AUC, pair_sep,
mean correct/incorrect score, and the fraction of correctly paired
products with negative trace.

Output: results/init_scheme_ablation.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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
    """Reinitialize every nn.Linear in `model` according to `scheme`.

    The PyTorch default (used by all prior pipeline.py experiments) is
    kaiming-uniform; we explicitly cover the four common schemes here.
    Biases are zeroed in every scheme.
    """
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
            elif scheme == "xavier_normal":
                fan_in, fan_out = W.shape[1], W.shape[0]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                W.copy_(torch.randn(W.shape, generator=g) * std)
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


# --------------------------------------------------------------------- scoring
def diag_dominance_matrix(W_in_list, W_out_list):
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
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    pos, neg = diag[:, None], off[None, :]
    auc = float(((pos > neg).sum() + 0.5 * (pos == neg).sum())
                / (pos.size * neg.size))
    return {
        "n":              n,
        "pair_acc":       pair_acc,
        "pair_sep":       pair_sep,
        "auc":            auc,
        "mean_correct":   float(diag.mean()),
        "mean_incorrect": float(off.mean()),
    }


def trace_signs(W_in_list, W_out_list):
    traces = []
    for Wi, Wo in zip(W_in_list, W_out_list):
        P = Wo.astype(np.float64) @ Wi.astype(np.float64)
        traces.append(float(np.trace(P)))
    traces = np.asarray(traces)
    return {
        "frac_negative": float((traces < 0).mean()),
        "mean_trace":    float(traces.mean()),
    }


# --------------------------------------------------------------------- train
def train_and_score(scheme, seed, depth=24, in_dim=24, hidden=64,
                    epochs=200, lr=1e-3, batch=256, grad_clip=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
    apply_init(model, scheme, seed=seed * 1000 + 1)

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
            idx = perm[s:s + batch]
            yb_pred = model(X_train[idx]).squeeze(-1)
            loss = loss_fn(yb_pred, y_train[idx])
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

    # eval loss
    model.eval()
    with torch.no_grad():
        eval_loss = float(loss_fn(model(X_eval).squeeze(-1), y_eval).item())

    # extract weights for scoring
    W_in_list  = [b.inp.weight.detach().cpu().numpy() for b in model.blocks]
    W_out_list = [b.out.weight.detach().cpu().numpy() for b in model.blocks]

    M = diag_dominance_matrix(W_in_list, W_out_list)
    metrics = evaluate_pairing(M)
    metrics["trace"] = trace_signs(W_in_list, W_out_list)
    metrics["init_loss"]  = init_loss
    metrics["final_loss"] = final_loss
    metrics["eval_loss"]  = eval_loss
    return metrics


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--in-dim", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out",
                    default="results/init_scheme_ablation.json")
    args = ap.parse_args()

    schemes = ["orthogonal", "kaiming_normal", "xavier_normal", "gaussian_002"]

    all_results = {
        "config": {
            "depth": args.depth, "hidden": args.hidden, "in_dim": args.in_dim,
            "epochs": args.epochs, "seeds": args.seeds,
        },
        "per_run": [],
        "summary": {},
    }

    t0 = time.time()
    for scheme in schemes:
        per_seed = []
        print(f"\n=== {scheme} ===", flush=True)
        for seed in args.seeds:
            tic = time.time()
            m = train_and_score(scheme, seed,
                                depth=args.depth, hidden=args.hidden,
                                in_dim=args.in_dim, epochs=args.epochs)
            took = time.time() - tic
            per_seed.append({"seed": seed, **m})
            print(f"  seed {seed}: pair_acc={m['pair_acc']:.0%}  "
                  f"AUC={m['auc']:.3f}  sep={m['pair_sep']:+.3f}  "
                  f"neg_tr={m['trace']['frac_negative']:.0%}  "
                  f"eval_loss={m['eval_loss']:.4f}  ({took:.1f}s)",
                  flush=True)
            all_results["per_run"].append({"scheme": scheme, **per_seed[-1]})

        # aggregate
        accs = [r["pair_acc"]       for r in per_seed]
        aucs = [r["auc"]            for r in per_seed]
        seps = [r["pair_sep"]       for r in per_seed]
        ntr  = [r["trace"]["frac_negative"] for r in per_seed]
        mc   = [r["mean_correct"]   for r in per_seed]
        mi   = [r["mean_incorrect"] for r in per_seed]
        evlo = [r["eval_loss"]      for r in per_seed]
        all_results["summary"][scheme] = {
            "mean_pair_acc":       float(np.mean(accs)),
            "std_pair_acc":        float(np.std(accs)),
            "mean_auc":            float(np.mean(aucs)),
            "std_auc":             float(np.std(aucs)),
            "mean_pair_sep":       float(np.mean(seps)),
            "std_pair_sep":        float(np.std(seps)),
            "mean_frac_negative":  float(np.mean(ntr)),
            "mean_correct":        float(np.mean(mc)),
            "mean_incorrect":      float(np.mean(mi)),
            "mean_eval_loss":      float(np.mean(evlo)),
        }
        s = all_results["summary"][scheme]
        print(f"  AGG: pair_acc={s['mean_pair_acc']:.0%}±{s['std_pair_acc']:.0%}  "
              f"AUC={s['mean_auc']:.3f}±{s['std_auc']:.3f}  "
              f"sep={s['mean_pair_sep']:+.3f}±{s['std_pair_sep']:.3f}  "
              f"neg_tr={s['mean_frac_negative']:.0%}",
              flush=True)

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed
    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
