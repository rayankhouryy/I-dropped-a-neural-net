"""Focused convergence experiment.

Train deeper + longer networks and watch the identifiability metrics. Goal:
verify there's a regime where reassembly succeeds, and characterize the
training-loss threshold at which the signal emerges.
"""
import time
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from pipeline import ResNet, make_data, reassembly_metrics, DEVICE

def cosine_lr(opt, step, total, base_lr, min_lr=1e-5):
    cos = 0.5 * (1 + np.cos(np.pi * step / total))
    lr = min_lr + (base_lr - min_lr) * cos
    for g in opt.param_groups:
        g["lr"] = lr

def run_one(config, seed, ckpts):
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim, hidden, depth, epochs = (config["in_dim"], config["hidden"],
                                     config["depth"], config["epochs"])
    X, y = make_data(in_dim, n=8000, seed=0)
    ntr = 6000
    Xt, yt = X[:ntr], y[:ntr]
    Xe, ye = X[ntr:], y[ntr:]

    model = ResNet(in_dim, hidden, depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    total_steps = epochs * (ntr // config["batch"])
    step = 0; rows = []; t0 = time.time()
    for epoch in range(epochs + 1):
        if epoch in ckpts:
            m = reassembly_metrics(model, Xe, ye, n_eval=1000)
            row = dict(config); row.update({"seed": seed, "epoch": epoch}); row.update(m)
            rows.append(row)
            print(f"  ep={epoch:4d} loss={m['train_loss']:.4f} pair_acc={m['pair_acc']:.2f} "
                  f"sep={m['pair_sep']:+.3f} wrho={m['wout_norm_rho']:+.3f} "
                  f"reasm_mse={m['reassembly_mse']:.2e} t={time.time()-t0:.0f}s")
        if epoch == epochs: break
        model.train()
        perm = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], config["batch"]):
            ix = perm[i:i+config["batch"]]
            xb, yb = Xt[ix].to(DEVICE), yt[ix].to(DEVICE).reshape(-1, 1)
            cosine_lr(opt, step, total_steps, config["lr"])
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    return rows

# A deeper, fuller-trained config close-ish to Park's setup
config = {"name": "d24w64i24", "depth": 24, "hidden": 64, "in_dim": 24,
          "epochs": 800, "lr": 3e-3, "batch": 256}

ckpts = sorted(set([0, 1, 2, 5, 10, 25, 50, 100, 200, 400, 600, 800]))
all_rows = []
for seed in [0, 1]:
    print(f"\n--- d24w64 seed={seed} ---")
    all_rows.extend(run_one(config, seed, ckpts))

df = pd.DataFrame(all_rows)
df.to_csv("convergence_d24w64.csv", index=False)
print(f"\nSaved {len(df)} rows to convergence_d24w64.csv")
print(df.groupby("seed").agg(min_loss=("train_loss","min"),
                             max_pair_acc=("pair_acc","max"),
                             max_sep=("pair_sep","max"),
                             max_wrho=("wout_norm_rho","max"),
                             reassembly_success_any=("reassembly_success","max")))
