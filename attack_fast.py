"""Fast variant of attack.py — reduced SA iters and incremental save."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8', errors='replace')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, random, math, os
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

torch.set_grad_enabled(False)
DEVICE = "cpu"
torch.set_num_threads(2)  # leave headroom for sweep

# ---- Load Park's puzzle network ---------------------------------------
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"pieces/piece_{i}.pth", map_location="cpu", weights_only=True)
last_id = next(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([1, 48]))
SOL = [43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,
       33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,
       38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,
       90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85]
true_in  = [SOL[2*k]   for k in range(48)]
true_out = [SOL[2*k+1] for k in range(48)]
D = 48

class Block(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden); self.out = nn.Linear(hidden, in_dim)
    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))

class ResNet(nn.Module):
    def __init__(self, blocks, last):
        super().__init__()
        self.blocks = nn.ModuleList(blocks); self.last = last
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.last(x)

def build_park():
    blocks = []
    for k in range(D):
        b = Block(48, 96)
        b.inp.weight.data = pieces[true_in[k]]['weight'].clone()
        b.inp.bias  .data = pieces[true_in[k]]['bias' ].clone()
        b.out.weight.data = pieces[true_out[k]]['weight'].clone()
        b.out.bias  .data = pieces[true_out[k]]['bias' ].clone()
        blocks.append(b)
    last = nn.Linear(48, 1)
    last.weight.data = pieces[last_id]['weight'].clone()
    last.bias  .data = pieces[last_id]['bias'].clone()
    return ResNet(blocks, last)

df = pd.read_csv("historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
y_pred = torch.tensor(df["pred"].values, dtype=torch.float32).reshape(-1, 1)
y_true = torch.tensor(df["true"].values, dtype=torch.float32).reshape(-1, 1)

N_EVAL = 500
SA_ITERS = 500

def measure(model):
    model.eval()
    Win  = [b.inp.weight.detach().to(torch.float64) for b in model.blocks]
    bIn  = [b.inp.bias  .detach().to(torch.float64) for b in model.blocks]
    Wout = [b.out.weight.detach().to(torch.float64) for b in model.blocks]
    bOut = [b.out.bias  .detach().to(torch.float64) for b in model.blocks]
    WL = model.last.weight.detach().to(torch.float64)
    bL = model.last.bias  .detach().to(torch.float64)
    Xe = X[:N_EVAL].to(torch.float64); ye = y_pred[:N_EVAL].to(torch.float64)

    score = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            M = (Wout[j] @ Win[i]).numpy()
            s = abs(float(M.trace())) / (float(np.linalg.norm(M, 'fro')) + 1e-12)
            score[i,j] = s
    _, c = linear_sum_assignment(-score)
    pair_acc = float(np.mean(c == np.arange(D)))
    pair_sep = float(np.array([score[i,i] for i in range(D)]).min()
                     - score[~np.eye(D, dtype=bool)].max())
    wout_F = np.array([float(W.norm()) for W in Wout])
    rho, _ = spearmanr(wout_F, np.arange(D))

    def fseq(in_seq, out_seq):
        h = Xe
        for k in range(D):
            z = F.relu(h @ Win[in_seq[k]].T + bIn[in_seq[k]])
            h = h + z @ Wout[out_seq[k]].T + bOut[out_seq[k]]
        return h @ WL.T + bL
    def mse_of(in_seq, out_seq):
        return ((fseq(in_seq, out_seq) - ye)**2).mean().item()

    pred_pairs = [(i, int(c[i])) for i in range(D)]
    asc  = list(np.argsort(wout_F))
    desc = list(np.argsort(-wout_F))

    def sa(order, iters=SA_ITERS):
        rng = random.Random(0)
        order = list(order)
        cur = mse_of([pred_pairs[k][0] for k in order], [pred_pairs[k][1] for k in order])
        best = cur; T = 0.5; alpha = (1e-4/0.5)**(1.0/iters)
        for _ in range(iters):
            i, j = rng.sample(range(D), 2)
            order[i], order[j] = order[j], order[i]
            new = mse_of([pred_pairs[k][0] for k in order], [pred_pairs[k][1] for k in order])
            if new < cur or rng.random() < math.exp((cur-new)/max(T, 1e-12)):
                cur = new
            else:
                order[i], order[j] = order[j], order[i]
            best = min(best, cur); T *= alpha
        return best

    orig_mse = mse_of(list(range(D)), list(range(D)))
    sa_asc  = sa(asc)
    sa_desc = sa(desc)
    return dict(pair_acc=pair_acc, pair_sep=pair_sep, wout_rho=float(rho),
                orig_mse=orig_mse, sa_asc=sa_asc, sa_desc=sa_desc)

def fine_tune(model, epochs, lr, target=y_pred, batch=256):
    torch.set_grad_enabled(True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            ix = perm[i:i+batch]
            opt.zero_grad()
            loss = F.mse_loss(model(X[ix]), target[ix])
            loss.backward(); opt.step()
    torch.set_grad_enabled(False)

CSV = "attack_results.csv"
rows = []

def emit(r):
    rows.append(r)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"{r['attack']:>10} ep={r['epochs']:>3} lr={r['lr']:.0e}  "
          f"pair_acc={r['pair_acc']:.2f} sep={r['pair_sep']:+.3f} rho={r['wout_rho']:+.3f} "
          f"orig_mse={r['orig_mse']:.2e} sa_asc={r['sa_asc']:.2e} sa_desc={r['sa_desc']:.2e}")

# Sanity check
m0 = measure(build_park())
print(f"Baseline: pair_acc={m0['pair_acc']}, orig_mse={m0['orig_mse']:.2e}")

print("\n=== ft_pred ===")
for ep, lr in [(0, 0.0), (1, 1e-5), (1, 1e-4), (1, 1e-3),
               (5, 1e-4), (5, 1e-3), (20, 1e-4), (20, 1e-3), (50, 1e-3)]:
    model = build_park()
    if ep > 0: fine_tune(model, ep, lr, target=y_pred)
    m = measure(model); m["epochs"] = ep; m["lr"] = lr; m["attack"] = "ft_pred"
    emit(m)

print("\n=== ft_true ===")
for ep, lr in [(1, 1e-4), (1, 1e-3), (5, 1e-3), (20, 1e-3), (50, 1e-3)]:
    model = build_park()
    fine_tune(model, ep, lr, target=y_true)
    m = measure(model); m["epochs"] = ep; m["lr"] = lr; m["attack"] = "ft_true"
    emit(m)

print("\n=== noise ===")
for sigma in [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    model = build_park()
    if sigma > 0:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * sigma * p.std().item())
    m = measure(model); m["epochs"] = 0; m["lr"] = sigma; m["attack"] = "noise"
    emit(m)

print(f"\nDone. {len(rows)} rows in {CSV}")
