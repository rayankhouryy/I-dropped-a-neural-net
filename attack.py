"""Fine-tuning attack experiment.

Adversary scenario: someone steals Park's puzzle ResNet and fine-tunes it for
N epochs to evade identification. Can we still:
  (i)  Pair the blocks (diag-dominance Hungarian)?
  (ii) Recover the ordering (SA)?
  (iii) Match the original model's outputs (reassembly MSE)?

At what fine-tuning step does identifiability break?
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, time, random, math, json, os
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import sys
sys.stdout.reconfigure(line_buffering=True)

torch.set_grad_enabled(False)
DEVICE = "cpu"

# ---- Load Park's puzzle network ---------------------------------------
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"pieces/piece_{i}.pth", map_location="cpu", weights_only=True)
inp_ids = sorted(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([96, 48]))
out_ids = sorted(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([48, 96]))
last_id = next(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([1, 48]))
SOL = [43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,
       33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,
       38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,
       90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85]
true_in  = [SOL[2*k]   for k in range(48)]
true_out = [SOL[2*k+1] for k in range(48)]
D = 48

# ---- Build a torch model from the puzzle pieces -----------------------
class Block(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, in_dim)
    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))

class ResNet(nn.Module):
    def __init__(self, blocks, last):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.last = last
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

# ---- Park puzzle data ------------------------------------------------
df = pd.read_csv("historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
y_pred = torch.tensor(df["pred"].values, dtype=torch.float32).reshape(-1, 1)
y_true = torch.tensor(df["true"].values, dtype=torch.float32).reshape(-1, 1)

# verify the network is correct
model = build_park()
with torch.no_grad():
    out = model(X)
    base_mse = F.mse_loss(out, y_pred).item()
print(f"Built Park model. MSE vs pred = {base_mse:.2e} (should be ~0)")

# ---- Identifiability metrics -----------------------------------------
def measure(model, X_eval=X[:1000], y_target=y_pred[:1000]):
    """Compute identifiability metrics for a model. We *re-pair* and
    *re-order* it from scratch (forgetting depth/pairing) and ask if we
    can recover the original."""
    model.eval()
    Win  = [b.inp.weight.detach().to(torch.float64) for b in model.blocks]
    bIn  = [b.inp.bias  .detach().to(torch.float64) for b in model.blocks]
    Wout = [b.out.weight.detach().to(torch.float64) for b in model.blocks]
    bOut = [b.out.bias  .detach().to(torch.float64) for b in model.blocks]
    WL = model.last.weight.detach().to(torch.float64)
    bL = model.last.bias  .detach().to(torch.float64)
    Xe = X_eval.to(torch.float64)
    ye = y_target.to(torch.float64)

    cost = np.zeros((D, D)); score = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            M = (Wout[j] @ Win[i]).numpy()
            s = abs(float(M.trace())) / (float(np.linalg.norm(M, 'fro')) + 1e-12)
            score[i,j] = s; cost[i,j] = -s
    r, c = linear_sum_assignment(cost)
    # Ground-truth here is identity (we built the model with block k = (true_in[k], true_out[k]))
    pair_acc = float(np.mean(c == np.arange(D)))
    pair_sep = float(np.array([score[i,i] for i in range(D)]).min() - score[~np.eye(D, dtype=bool)].max())
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

    # Reassemble from scratch with Hungarian pairing and SA ordering (both directions)
    pred_pairs = [(i, int(c[i])) for i in range(D)]
    asc  = list(np.argsort(wout_F))
    desc = list(np.argsort(-wout_F))

    def sa(order, iters=2500):
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

    # current reassembled MSE using identity (no shuffle):  this is "how close is the model to the ORIGINAL model's outputs?"
    orig_mse = mse_of(list(range(D)), list(range(D)))
    sa_asc  = sa(asc)
    sa_desc = sa(desc)

    return {
        "pair_acc": pair_acc, "pair_sep": pair_sep, "wout_rho": float(rho),
        "orig_mse": orig_mse, "sa_asc": sa_asc, "sa_desc": sa_desc,
    }

# ---- Fine-tune the network ------------------------------------------
def fine_tune(model, epochs, lr, target=y_pred, batch=256):
    torch.set_grad_enabled(True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            ix = perm[i:i+batch]
            xb, yb = X[ix], target[ix]
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    torch.set_grad_enabled(False)

# ---- Scenarios ------------------------------------------------------
print("\n=== Attack: fine-tune Park's net against `pred` (light retraining) ===")
print(f"{'epochs':>7} {'lr':>7} {'pair_acc':>9} {'pair_sep':>9} {'rho':>7} "
      f"{'orig_mse':>10} {'sa_asc':>10} {'sa_desc':>10}")
rows = []
for ep, lr in [(0, 0.0), (1, 1e-5), (1, 1e-4), (1, 1e-3),
               (5, 1e-4), (5, 1e-3), (20, 1e-4), (20, 1e-3),
               (50, 1e-3), (100, 1e-3)]:
    model = build_park()
    if ep > 0: fine_tune(model, ep, lr, target=y_pred)
    m = measure(model)
    m["epochs"] = ep; m["lr"] = lr; m["attack"] = "ft_pred"
    rows.append(m)
    print(f"{ep:>7} {lr:>7.0e} {m['pair_acc']:>9.2f} {m['pair_sep']:>+9.3f} "
          f"{m['wout_rho']:>+7.3f} {m['orig_mse']:>10.2e} {m['sa_asc']:>10.2e} {m['sa_desc']:>10.2e}")

print("\n=== Attack: fine-tune against `true` (noisy real labels - the adversary's data) ===")
print(f"{'epochs':>7} {'lr':>7} {'pair_acc':>9} {'pair_sep':>9} {'rho':>7} "
      f"{'orig_mse':>10} {'sa_asc':>10} {'sa_desc':>10}")
for ep, lr in [(1, 1e-4), (1, 1e-3), (5, 1e-3), (20, 1e-3), (50, 1e-3), (100, 1e-3)]:
    model = build_park()
    fine_tune(model, ep, lr, target=y_true)
    m = measure(model)
    m["epochs"] = ep; m["lr"] = lr; m["attack"] = "ft_true"
    rows.append(m)
    print(f"{ep:>7} {lr:>7.0e} {m['pair_acc']:>9.2f} {m['pair_sep']:>+9.3f} "
          f"{m['wout_rho']:>+7.3f} {m['orig_mse']:>10.2e} {m['sa_asc']:>10.2e} {m['sa_desc']:>10.2e}")

print("\n=== Attack: add Gaussian noise to weights ===")
print(f"{'sigma':>7} {'pair_acc':>9} {'pair_sep':>9} {'rho':>7} "
      f"{'orig_mse':>10} {'sa_asc':>10} {'sa_desc':>10}")
for sigma in [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    model = build_park()
    if sigma > 0:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * sigma * p.std().item())
    m = measure(model)
    m["epochs"] = 0; m["lr"] = sigma; m["attack"] = "noise"
    rows.append(m)
    print(f"{sigma:>7.0e} {m['pair_acc']:>9.2f} {m['pair_sep']:>+9.3f} "
          f"{m['wout_rho']:>+7.3f} {m['orig_mse']:>10.2e} {m['sa_asc']:>10.2e} {m['sa_desc']:>10.2e}")

pd.DataFrame(rows).to_csv("attack_results.csv", index=False)
print(f"\nsaved attack_results.csv with {len(rows)} rows")
