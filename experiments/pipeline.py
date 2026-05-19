"""Layer-identifiability experiment infrastructure.

Trains small ResNets on a synthetic regression task, saves checkpoints during
training, and at each checkpoint measures four identifiability metrics:

  1) Pairing accuracy via diagonal-dominance Hungarian (against ground truth)
  2) Diagonal-dominance separation: min correct-pair score - max incorrect-pair score
  3) ||W_out||_F vs true-depth Spearman rho
  4) End-to-end reassembly success: does the hill-climb reach MSE=0?

Outputs a CSV row per (config, seed, checkpoint).

Designed to run on CPU for small configs; scale up on GPU via env vars.
"""
import argparse, json, math, os, time, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- model
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
        self.blocks = nn.ModuleList([Block(in_dim, hidden_dim) for _ in range(depth)])
        self.last = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.last(x)

# ---------------------------------------------------------------- data
def synthetic_target(X, in_dim, key):
    """Smooth random target function. Reproducible from `key`."""
    g = torch.Generator(device="cpu").manual_seed(key)
    A = torch.randn(in_dim, 8, generator=g) * 0.5
    B = torch.randn(8,        generator=g)
    bias = torch.randn(1,     generator=g)
    h = torch.tanh(X @ A)
    y = h @ B + bias
    return y

def make_data(in_dim, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    y = synthetic_target(X, in_dim, key=seed + 1234)
    return X, y

# ---------------------------------------------------------------- metrics
def reassembly_metrics(model, X_eval, y_eval, n_eval=600):
    """All metrics that depend on (a) trained weights, (b) eval data.

    Returns dict of:
      pair_acc           — Hungarian on diag-dominance correctly recovers truth?
      pair_sep           — min(correct-pair d) - max(incorrect-pair d)
      wout_norm_rho      — Spearman of ||W_out||_F with true depth
      wall_C             — slope MSE(k)/k for k=2..8 (Pairing Wall)
      reassembly_success — does ||W_out||_F seed + bubble-sort hit MSE<1e-6?
      reassembly_mse     — final MSE after reassembly
      train_loss         — current training loss on eval set
    """
    model.eval()
    depth = len(model.blocks)
    in_dim = model.blocks[0].inp.in_features
    out_dim = model.last.out_features
    # ground truth: block k has W_in = blocks[k].inp, W_out = blocks[k].out
    true_W_in  = [b.inp.weight.detach().to(torch.float64).cpu() for b in model.blocks]
    true_b_in  = [b.inp.bias  .detach().to(torch.float64).cpu() for b in model.blocks]
    true_W_out = [b.out.weight.detach().to(torch.float64).cpu() for b in model.blocks]
    true_b_out = [b.out.bias  .detach().to(torch.float64).cpu() for b in model.blocks]
    WL = model.last.weight.detach().to(torch.float64).cpu()
    bL = model.last.bias  .detach().to(torch.float64).cpu()

    X = X_eval[:n_eval].to(torch.float64).cpu()
    y = y_eval[:n_eval].to(torch.float64).cpu().reshape(-1, out_dim)

    # ---- 1) diagonal-dominance matrix (square depth x depth)
    cost = np.zeros((depth, depth))
    score = np.zeros((depth, depth))
    for i in range(depth):
        for j in range(depth):
            M = (true_W_out[j] @ true_W_in[i]).numpy()
            s = abs(float(np.trace(M))) / (float(np.linalg.norm(M, 'fro')) + 1e-12)
            score[i, j] = s
            cost[i, j] = -s
    r, c = linear_sum_assignment(cost)
    # Correct pairing under ground truth identity i->i; Hungarian assigns
    # inp_index i to out_index c[i]. Accurate if c == identity.
    pair_acc = float(np.mean(c == np.arange(depth)))
    correct_scores = np.array([score[i, i] for i in range(depth)])
    incorrect_scores = score[~np.eye(depth, dtype=bool)]
    pair_sep = float(correct_scores.min() - incorrect_scores.max())

    # ---- 3) ||W_out||_F vs depth correlation
    wout_F = np.array([float(W.norm()) for W in true_W_out])
    rho, _ = spearmanr(wout_F, np.arange(depth))
    wout_norm_rho = float(rho)

    # forward function with arbitrary in/out sequence
    def forward_seq(in_seq, out_seq):
        h = X
        for k in range(depth):
            W1, b1 = true_W_in[in_seq[k]], true_b_in[in_seq[k]]
            W2, b2 = true_W_out[out_seq[k]], true_b_out[out_seq[k]]
            z = F.relu(h @ W1.T + b1)
            h = h + z @ W2.T + b2
        return h @ WL.T + bL

    def mse_of(in_seq, out_seq):
        return ((forward_seq(in_seq, out_seq) - y) ** 2).mean().item()

    # ---- training loss (on eval slice)
    with torch.no_grad():
        pred = model(X.float())
        train_loss = ((pred - y.float())**2).mean().item()

    # ---- 4) Pairing Wall: with TRUE order, swap k out-partners and measure MSE
    base = mse_of(list(range(depth)), list(range(depth)))
    rng = random.Random(0)
    wall_ks = [2, 4, 8] if depth >= 16 else [2, 4]
    wall_mses = []
    for k in wall_ks:
        ms = []
        for trial in range(8):
            positions = rng.sample(range(depth), k)
            outs = list(positions); rng.shuffle(outs)
            tries = 0
            while any(outs[i] == positions[i] for i in range(k)) and tries < 50:
                rng.shuffle(outs); tries += 1
            in_seq, out_seq = list(range(depth)), list(range(depth))
            for p, o in zip(positions, outs):
                out_seq[p] = o
            ms.append(mse_of(in_seq, out_seq) - base)
        wall_mses.append((k, np.mean(ms)))
    # linear fit slope
    if len(wall_mses) >= 2:
        ks = np.array([w[0] for w in wall_mses])
        ms = np.array([max(w[1], 0.0) for w in wall_mses])
        wall_C = float(np.polyfit(ks, ms, 1)[0])
    else:
        wall_C = float('nan')

    # ---- 5) End-to-end reassembly: use Park's pipeline on these weights
    # pair by diagonal dominance (Hungarian), seed by ||W_out||_F, bubble-sort hill-climb.
    pred_pairing = [(i, int(c[i])) for i in range(depth)]
    seed_order = list(np.argsort(wout_F))
    in_seq  = [pred_pairing[k][0] for k in seed_order]
    out_seq = [pred_pairing[k][1] for k in seed_order]
    cur = mse_of(in_seq, out_seq)
    # bubble-sort hill-climb
    for _ in range(40):
        swaps = 0
        for i in range(depth - 1):
            in_seq[i], in_seq[i+1] = in_seq[i+1], in_seq[i]
            out_seq[i], out_seq[i+1] = out_seq[i+1], out_seq[i]
            m = mse_of(in_seq, out_seq)
            if m < cur - 1e-12:
                cur = m; swaps += 1
            else:
                in_seq[i], in_seq[i+1] = in_seq[i+1], in_seq[i]
                out_seq[i], out_seq[i+1] = out_seq[i+1], out_seq[i]
        if swaps == 0: break
    reassembly_mse = float(cur)
    reassembly_success = bool(cur < 1e-6)

    return {
        "pair_acc": pair_acc,
        "pair_sep": pair_sep,
        "wout_norm_rho": wout_norm_rho,
        "wall_C": wall_C,
        "reassembly_mse": reassembly_mse,
        "reassembly_success": reassembly_success,
        "train_loss": train_loss,
    }

# ---------------------------------------------------------------- train
def train_one(config, seed, checkpoints, X_train, y_train, X_eval, y_eval, log=print):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    in_dim, hidden, depth = config["in_dim"], config["hidden"], config["depth"]
    model = ResNet(in_dim, hidden, depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    epochs = config.get("epochs", 100)
    batch = config.get("batch", 256)
    cp_epochs = sorted(set(int(e) for e in checkpoints if 0 <= e <= epochs))
    rows = []
    for epoch in range(epochs + 1):
        if epoch in cp_epochs:
            metrics = reassembly_metrics(model, X_eval, y_eval)
            row = dict(config); row.update({"seed": seed, "epoch": epoch}); row.update(metrics)
            rows.append(row)
            log(f"  [{config['name']} s{seed}] epoch={epoch:4d}  loss={metrics['train_loss']:.4f}  "
                f"pair_acc={metrics['pair_acc']:.2f}  pair_sep={metrics['pair_sep']:+.3f}  "
                f"wrho={metrics['wout_norm_rho']:+.3f}  reassembly_mse={metrics['reassembly_mse']:.2e}")
        if epoch == epochs: break
        # one training epoch
        model.train()
        perm = torch.randperm(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch):
            ix = perm[i:i+batch]
            xb = X_train[ix].to(DEVICE); yb = y_train[ix].to(DEVICE).reshape(-1, 1)
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return rows

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="identifiability_results.csv")
    ap.add_argument("--small", action="store_true", help="run small PoC sweep")
    ap.add_argument("--full", action="store_true", help="run full sweep (need GPU)")
    args = ap.parse_args()

    if args.small:
        configs = [
            {"name": "d8w32",  "depth":  8, "hidden": 32, "in_dim": 16, "epochs": 120, "lr": 1e-3, "batch": 256},
            {"name": "d16w32", "depth": 16, "hidden": 32, "in_dim": 16, "epochs": 150, "lr": 1e-3, "batch": 256},
            {"name": "d16w64", "depth": 16, "hidden": 64, "in_dim": 24, "epochs": 150, "lr": 1e-3, "batch": 256},
        ]
        seeds = [0, 1]
        checkpoint_pattern = lambda E: sorted(set([0, 1, 2, 5, 10, 20, 40, 70, E//2, E]))
    else:
        # full sweep template; user runs on GPU
        configs = []
        for depth in [8, 16, 32, 48]:
            for hidden in [32, 64, 96]:
                for in_dim in [16, 32, 48]:
                    configs.append({
                        "name": f"d{depth}w{hidden}i{in_dim}",
                        "depth": depth, "hidden": hidden, "in_dim": in_dim,
                        "epochs": 200, "lr": 1e-3, "batch": 256,
                    })
        seeds = [0, 1, 2]
        checkpoint_pattern = lambda E: sorted(set(
            [0, 1, 2, 5, 10, 20, 40, 70, 100, 140, E]
        ))

    rows = []
    t0 = time.time()
    for cfg in configs:
        # Build dataset for this in_dim
        X, y = make_data(cfg["in_dim"], n=4000, seed=0)
        ntr = 3000
        X_train, y_train = X[:ntr], y[:ntr]
        X_eval,  y_eval  = X[ntr:], y[ntr:]
        for seed in seeds:
            print(f"\n=== config {cfg['name']} seed={seed} ===")
            cps = checkpoint_pattern(cfg["epochs"])
            rows.extend(train_one(cfg, seed, cps, X_train, y_train, X_eval, y_eval))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(rows)} rows to {args.out}.  Total time {time.time()-t0:.1f}s")
    print(df.groupby(["name", "seed"]).agg({
        "train_loss": "min", "pair_acc": "max",
        "pair_sep": "max", "wout_norm_rho": "max",
        "reassembly_success": "max", "reassembly_mse": "min",
    }))

if __name__ == "__main__":
    main()
