"""Comprehensive sweep over (depth, hidden) x seeds.

Runs in background, saves incrementally so we can monitor.
"""
import time, random, math, os, sys
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from pipeline import ResNet, make_data, DEVICE

sys.stdout.reconfigure(line_buffering=True)  # so log writes are visible

OUT_CSV = "sweep_full.csv"

def per_block_norms(model):
    return np.array([float(b.out.weight.norm().item()) for b in model.blocks])

def cosine_lr(opt, step, total, base, lo=1e-5):
    cos = 0.5 * (1 + np.cos(np.pi * step / total))
    for g in opt.param_groups: g["lr"] = lo + (base - lo) * cos

def metrics(model, X, y, do_sa=True):
    model.eval()
    depth = len(model.blocks)
    W_in  = [b.inp.weight.detach().to(torch.float64).cpu() for b in model.blocks]
    b_in  = [b.inp.bias  .detach().to(torch.float64).cpu() for b in model.blocks]
    W_out = [b.out.weight.detach().to(torch.float64).cpu() for b in model.blocks]
    b_out = [b.out.bias  .detach().to(torch.float64).cpu() for b in model.blocks]
    WL = model.last.weight.detach().to(torch.float64).cpu()
    bL = model.last.bias  .detach().to(torch.float64).cpu()
    X64 = X.to(torch.float64).cpu()[:1000]
    y64 = y.to(torch.float64).cpu()[:1000].reshape(-1, 1)

    cost = np.zeros((depth, depth)); score = np.zeros((depth, depth))
    for i in range(depth):
        for j in range(depth):
            M = (W_out[j] @ W_in[i]).numpy()
            s = abs(float(np.trace(M))) / (float(np.linalg.norm(M, 'fro')) + 1e-12)
            score[i,j] = s; cost[i,j] = -s
    r, c = linear_sum_assignment(cost)
    pair_acc = float(np.mean(c == np.arange(depth)))
    pair_sep = float(np.array([score[i,i] for i in range(depth)]).min()
                     - score[~np.eye(depth,dtype=bool)].max())
    wout_F = np.array([float(W.norm()) for W in W_out])
    rho, _ = spearmanr(wout_F, np.arange(depth))

    def fseq(in_seq, out_seq):
        h = X64
        for k in range(depth):
            z = F.relu(h @ W_in[in_seq[k]].T + b_in[in_seq[k]])
            h = h + z @ W_out[out_seq[k]].T + b_out[out_seq[k]]
        return h @ WL.T + bL
    def mse_of(s_in, s_out): return ((fseq(s_in, s_out) - y64)**2).mean().item()

    pred_pairs = [(i, int(c[i])) for i in range(depth)]
    asc  = list(np.argsort(wout_F))
    desc = list(np.argsort(-wout_F))

    def hc(order):
        order = list(order)
        cur = mse_of([pred_pairs[k][0] for k in order], [pred_pairs[k][1] for k in order])
        for _ in range(40):
            sw = 0
            for i in range(depth-1):
                order[i], order[i+1] = order[i+1], order[i]
                m = mse_of([pred_pairs[k][0] for k in order], [pred_pairs[k][1] for k in order])
                if m < cur - 1e-12: cur = m; sw += 1
                else: order[i], order[i+1] = order[i+1], order[i]
            if sw == 0: break
        return cur
    def sa(order, iters=3000):
        rng = random.Random(0)
        order = list(order)
        cur = mse_of([pred_pairs[k][0] for k in order], [pred_pairs[k][1] for k in order])
        best = cur
        T = 0.5; alpha = (1e-4/0.5)**(1/iters)
        for _ in range(iters):
            i, j = rng.sample(range(depth), 2)
            order[i], order[j] = order[j], order[i]
            new = mse_of([pred_pairs[k][0] for k in order], [pred_pairs[k][1] for k in order])
            if new < cur or rng.random() < math.exp((cur - new)/max(T, 1e-12)):
                cur = new
            else:
                order[i], order[j] = order[j], order[i]
            best = min(best, cur); T *= alpha
        return best

    out = {"pair_acc": pair_acc, "pair_sep": pair_sep, "wout_rho": float(rho),
           "seed_mse_asc":  mse_of([pred_pairs[k][0] for k in asc],  [pred_pairs[k][1] for k in asc]),
           "seed_mse_desc": mse_of([pred_pairs[k][0] for k in desc], [pred_pairs[k][1] for k in desc]),
           "hc_asc":  hc(asc), "hc_desc": hc(desc),
           "wout_norms": wout_F.tolist()}
    if do_sa:
        out["sa_asc"] = sa(asc); out["sa_desc"] = sa(desc)
    with torch.no_grad():
        out["train_loss"] = float(F.mse_loss(model(X.float()[:1000]),
                                             y.float()[:1000].reshape(-1, 1)).item())
    return out

CONFIGS = [
    {"name": "d8_h32",   "depth":  8, "hidden": 32, "in_dim": 16, "epochs": 200, "lr": 3e-3, "batch": 256},
    {"name": "d16_h32",  "depth": 16, "hidden": 32, "in_dim": 16, "epochs": 300, "lr": 3e-3, "batch": 256},
    {"name": "d16_h64",  "depth": 16, "hidden": 64, "in_dim": 24, "epochs": 400, "lr": 3e-3, "batch": 256},
    {"name": "d24_h64",  "depth": 24, "hidden": 64, "in_dim": 24, "epochs": 500, "lr": 3e-3, "batch": 256},
    {"name": "d32_h64",  "depth": 32, "hidden": 64, "in_dim": 32, "epochs": 500, "lr": 3e-3, "batch": 256},
]
SEEDS = [0, 1, 2]

def run_one(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    X, y = make_data(cfg["in_dim"], n=8000, seed=0)
    ntr = 6000
    Xt, yt = X[:ntr], y[:ntr]
    Xe, ye = X[ntr:], y[ntr:]
    model = ResNet(cfg["in_dim"], cfg["hidden"], cfg["depth"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    total_steps = cfg["epochs"] * (ntr // cfg["batch"])
    step = 0
    E = cfg["epochs"]
    checkpoints = sorted(set([0, 1, 2, 5, 10, 25, 50, 100, 200, E]))
    if E >= 400: checkpoints += [400]
    if E >= 500: checkpoints += [500]
    rows = []
    for epoch in range(E + 1):
        if epoch in checkpoints:
            m = metrics(model, Xe, ye)
            row = dict(cfg); row.update({"seed": seed, "epoch": epoch}); row.update(m)
            rows.append(row)
            print(f"[{cfg['name']} s{seed} ep{epoch:4d}] loss={m['train_loss']:.4f} "
                  f"pair_acc={m['pair_acc']:.2f} sep={m['pair_sep']:+.3f} "
                  f"rho={m['wout_rho']:+.3f} sa_asc={m['sa_asc']:.4f} sa_desc={m['sa_desc']:.4f}",
                  flush=True)
        if epoch == E: break
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, cfg["batch"]):
            ix = perm[i:i+cfg["batch"]]
            xb, yb = Xt[ix].to(DEVICE), yt[ix].to(DEVICE).reshape(-1, 1)
            cosine_lr(opt, step, total_steps, cfg["lr"])
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    return rows

all_rows = []
if os.path.exists(OUT_CSV):
    print(f"resuming from {OUT_CSV}", flush=True)
    all_rows = pd.read_csv(OUT_CSV).to_dict("records")
done = {(r["name"], r["seed"], r["epoch"]) for r in all_rows}

t0 = time.time()
for cfg in CONFIGS:
    for seed in SEEDS:
        # quick check: skip if all checkpoints already present
        E = cfg["epochs"]
        cps = sorted(set([0, 1, 2, 5, 10, 25, 50, 100, 200, E] + ([400] if E>=400 else []) + ([500] if E>=500 else [])))
        if all((cfg["name"], seed, c) in done for c in cps):
            print(f"skip {cfg['name']} s{seed}", flush=True); continue
        print(f"\n=== {cfg['name']} seed={seed} ===", flush=True)
        rows = run_one(cfg, seed)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        print(f"saved {len(all_rows)} rows, elapsed {time.time()-t0:.0f}s", flush=True)

print(f"\nDONE in {time.time()-t0:.0f}s.", flush=True)
