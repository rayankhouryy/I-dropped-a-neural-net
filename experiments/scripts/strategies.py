"""Quick follow-up: at peak identifiability, does sort direction matter?

Also test: simulated annealing as a stronger ordering optimizer when bubble-sort
fails. This will tell us whether the ordering bottleneck is the SEED (sign of
||W_out||_F) or the SEARCH ALGORITHM (bubble-sort vs SA).
"""
import time, random, math
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from pipeline import ResNet, make_data, DEVICE
torch.set_grad_enabled(False)

def train_to_epoch(config, seed, max_epoch):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = make_data(config["in_dim"], n=8000, seed=0)
    ntr = 6000
    Xt, yt = X[:ntr], y[:ntr]
    Xe, ye = X[ntr:], y[ntr:]
    model = ResNet(config["in_dim"], config["hidden"], config["depth"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total_steps = max_epoch * (ntr // config["batch"])
    step = 0
    torch.set_grad_enabled(True)
    for epoch in range(max_epoch):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, config["batch"]):
            ix = perm[i:i+config["batch"]]
            xb, yb = Xt[ix].to(DEVICE), yt[ix].to(DEVICE).reshape(-1, 1)
            cos = 0.5 * (1 + np.cos(np.pi * step / total_steps))
            for g in opt.param_groups: g["lr"] = 1e-5 + (config["lr"] - 1e-5) * cos
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    torch.set_grad_enabled(False)
    return model, Xe, ye

def metrics_with_strategies(model, X_eval, y_eval, n_eval=1000):
    depth = len(model.blocks)
    W_in  = [b.inp.weight.detach().to(torch.float64).cpu() for b in model.blocks]
    b_in  = [b.inp.bias  .detach().to(torch.float64).cpu() for b in model.blocks]
    W_out = [b.out.weight.detach().to(torch.float64).cpu() for b in model.blocks]
    b_out = [b.out.bias  .detach().to(torch.float64).cpu() for b in model.blocks]
    WL = model.last.weight.detach().to(torch.float64).cpu()
    bL = model.last.bias  .detach().to(torch.float64).cpu()
    X = X_eval[:n_eval].to(torch.float64).cpu()
    y = y_eval[:n_eval].to(torch.float64).cpu().reshape(-1, 1)

    # Pairing via diag-dominance Hungarian
    cost = np.zeros((depth, depth))
    for i in range(depth):
        for j in range(depth):
            M = (W_out[j] @ W_in[i]).numpy()
            s = abs(float(np.trace(M))) / (float(np.linalg.norm(M, 'fro')) + 1e-12)
            cost[i, j] = -s
    r, c = linear_sum_assignment(cost)
    pred_pairing = [(i, int(c[i])) for i in range(depth)]
    pair_acc = float(np.mean(c == np.arange(depth)))

    wout_F = np.array([float(W.norm()) for W in W_out])

    def forward_seq(in_seq, out_seq):
        h = X
        for k in range(depth):
            W1, B1 = W_in[in_seq[k]], b_in[in_seq[k]]
            W2, B2 = W_out[out_seq[k]], b_out[out_seq[k]]
            z = F.relu(h @ W1.T + B1)
            h = h + z @ W2.T + B2
        return h @ WL.T + bL
    def mse_of(in_seq, out_seq):
        return ((forward_seq(in_seq, out_seq) - y) ** 2).mean().item()

    def hill_climb(order, pairs, max_rounds=40):
        cur = mse_of([pairs[k][0] for k in order], [pairs[k][1] for k in order])
        order = list(order)
        for _ in range(max_rounds):
            swaps = 0
            for i in range(depth - 1):
                order[i], order[i+1] = order[i+1], order[i]
                m = mse_of([pairs[k][0] for k in order], [pairs[k][1] for k in order])
                if m < cur - 1e-12:
                    cur = m; swaps += 1
                else:
                    order[i], order[i+1] = order[i+1], order[i]
            if swaps == 0: break
        return order, cur

    def simulated_annealing(order, pairs, iters=4000, T0=0.5, Tf=1e-4, seed=0):
        rng = random.Random(seed)
        order = list(order)
        cur = mse_of([pairs[k][0] for k in order], [pairs[k][1] for k in order])
        best = cur; best_order = list(order)
        alpha = (Tf/T0) ** (1.0/iters); T = T0
        for it in range(iters):
            i, j = rng.sample(range(depth), 2)
            order[i], order[j] = order[j], order[i]
            new = mse_of([pairs[k][0] for k in order], [pairs[k][1] for k in order])
            if new < cur or rng.random() < math.exp((cur - new)/max(T, 1e-12)):
                cur = new
            else:
                order[i], order[j] = order[j], order[i]
            if cur < best:
                best = cur; best_order = list(order)
            T *= alpha
        return best_order, best

    seed_asc  = list(np.argsort(wout_F))
    seed_desc = list(np.argsort(-wout_F))

    out = {"pair_acc": pair_acc}
    # initial-MSE for each seed direction
    out["mse_asc_seed"]  = mse_of([pred_pairing[k][0] for k in seed_asc],
                                  [pred_pairing[k][1] for k in seed_asc])
    out["mse_desc_seed"] = mse_of([pred_pairing[k][0] for k in seed_desc],
                                  [pred_pairing[k][1] for k in seed_desc])
    # bubble from each direction
    _, asc_hc = hill_climb(seed_asc, pred_pairing)
    _, desc_hc = hill_climb(seed_desc, pred_pairing)
    out["mse_asc_hc"]  = asc_hc
    out["mse_desc_hc"] = desc_hc
    # SA from each direction
    _, asc_sa = simulated_annealing(seed_asc, pred_pairing, iters=4000)
    _, desc_sa = simulated_annealing(seed_desc, pred_pairing, iters=4000)
    out["mse_asc_sa"]  = asc_sa
    out["mse_desc_sa"] = desc_sa
    # train loss
    with torch.no_grad():
        out["train_loss"] = float(F.mse_loss(model(X.float()), y.float()).item())
    return out

config = {"name": "d24w64i24", "depth": 24, "hidden": 64, "in_dim": 24,
          "epochs": 800, "lr": 3e-3, "batch": 256}

rows = []
for seed in [0, 1]:
    for max_ep in [25, 100, 400, 800]:
        print(f"--- seed={seed}, train to epoch {max_ep} ---")
        model, Xe, ye = train_to_epoch(config, seed, max_ep)
        m = metrics_with_strategies(model, Xe, ye)
        m["seed"] = seed; m["epoch"] = max_ep
        rows.append(m)
        print(f"  loss={m['train_loss']:.4f} pair_acc={m['pair_acc']:.2f}")
        print(f"  ASC : seed={m['mse_asc_seed']:.4f}  HC={m['mse_asc_hc']:.4f}  SA={m['mse_asc_sa']:.4f}")
        print(f"  DESC: seed={m['mse_desc_seed']:.4f} HC={m['mse_desc_hc']:.4f} SA={m['mse_desc_sa']:.4f}")

df = pd.DataFrame(rows)
df.to_csv("strategies.csv", index=False)
print("\nSummary:")
print(df[["seed","epoch","train_loss","pair_acc","mse_asc_hc","mse_desc_hc","mse_asc_sa","mse_desc_sa"]].to_string(index=False))
