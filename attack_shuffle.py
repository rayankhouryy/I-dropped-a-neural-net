"""Post-shuffle attack: fine-tune + hidden-unit permutation symmetry.

The only function-preserving symmetry of a ReLU residual block is permutation
of hidden units: W_in -> P W_in, b_in -> P b_in, W_out -> W_out P^T.
Under this symmetry W_out W_in is invariant, so diagonal-dominance pairing
should be invariant. This experiment verifies that empirically and combines
the symmetry with fine-tuning to mimic a forensic-evasion attacker.
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8', errors='replace')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, random
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

torch.set_grad_enabled(False)
torch.set_num_threads(2)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

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
D, H = 48, 96

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(D, H); self.out = nn.Linear(H, D)
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
        b = Block()
        b.inp.weight.data = pieces[true_in[k]]['weight'].clone()
        b.inp.bias  .data = pieces[true_in[k]]['bias' ].clone()
        b.out.weight.data = pieces[true_out[k]]['weight'].clone()
        b.out.bias  .data = pieces[true_out[k]]['bias' ].clone()
        blocks.append(b)
    last = nn.Linear(48, 1)
    last.weight.data = pieces[last_id]['weight'].clone()
    last.bias  .data = pieces[last_id]['bias'].clone()
    return ResNet(blocks, last)

def shuffle_hidden_units(model, rng=None):
    """Apply random per-block hidden-unit permutation. Function-preserving."""
    rng = rng or np.random.RandomState(1)
    for b in model.blocks:
        P = rng.permutation(H)
        with torch.no_grad():
            b.inp.weight.data = b.inp.weight.data[P]
            b.inp.bias.data   = b.inp.bias.data[P]
            b.out.weight.data = b.out.weight.data[:, P]
    return model

df = pd.read_csv("historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
y_pred = torch.tensor(df["pred"].values, dtype=torch.float32).reshape(-1, 1)
y_true = torch.tensor(df["true"].values, dtype=torch.float32).reshape(-1, 1)

N_EVAL = 500
def measure(model):
    Win  = [b.inp.weight.detach().to(torch.float64) for b in model.blocks]
    Wout = [b.out.weight.detach().to(torch.float64) for b in model.blocks]
    bIn  = [b.inp.bias  .detach().to(torch.float64) for b in model.blocks]
    bOut = [b.out.bias  .detach().to(torch.float64) for b in model.blocks]
    WL = model.last.weight.detach().to(torch.float64)
    bL = model.last.bias  .detach().to(torch.float64)

    score = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            M = (Wout[j] @ Win[i]).numpy()
            s = abs(float(M.trace())) / (float(np.linalg.norm(M, 'fro')) + 1e-12)
            score[i,j] = s
    _, c = linear_sum_assignment(-score)
    pair_acc_hun = float(np.mean(c == np.arange(D)))
    greedy = np.argmax(score, axis=1)
    pair_acc_greedy = float(np.mean(greedy == np.arange(D)))
    pair_sep = float(np.array([score[i,i] for i in range(D)]).min()
                     - score[~np.eye(D, dtype=bool)].max())

    wout_F = np.array([float(W.norm()) for W in Wout])
    rho, _ = spearmanr(wout_F, np.arange(D))

    Xe = X[:N_EVAL].to(torch.float64); ye = y_pred[:N_EVAL].to(torch.float64)
    h = Xe
    for k in range(D):
        h = h + (F.relu(h @ Win[k].T + bIn[k]) @ Wout[k].T + bOut[k])
    out = h @ WL.T + bL
    mse_native = ((out - ye)**2).mean().item()

    return dict(pair_acc_hun=pair_acc_hun, pair_acc_greedy=pair_acc_greedy,
                pair_sep=pair_sep, wout_rho=float(rho), mse_native=mse_native)

def fine_tune(model, epochs, lr, target, batch=256):
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

rows = []
def run(label, mod_fn):
    m = build_park()
    mod_fn(m)
    r = measure(m); r["scenario"] = label
    rows.append(r)
    print(f"{label:>30}: pair_acc(hun)={r['pair_acc_hun']:.2f} "
          f"pair_acc(greedy)={r['pair_acc_greedy']:.2f} "
          f"sep={r['pair_sep']:+.3f} mse={r['mse_native']:.2e}")

print("=== Post-shuffle attack: hidden-unit permutation symmetry ===")
run("baseline",                 lambda m: None)
run("shuffle_only",             lambda m: shuffle_hidden_units(m, np.random.RandomState(1)))
run("ft_pred_50ep_1e-3",        lambda m: fine_tune(m, 50, 1e-3, y_pred))

m = build_park()
fine_tune(m, 50, 1e-3, y_pred)
shuffle_hidden_units(m, np.random.RandomState(7))
r = measure(m); r["scenario"] = "ft_pred + shuffle"
rows.append(r)
print(f"{'ft_pred + shuffle':>30}: pair_acc(hun)={r['pair_acc_hun']:.2f} "
      f"pair_acc(greedy)={r['pair_acc_greedy']:.2f} "
      f"sep={r['pair_sep']:+.3f} mse={r['mse_native']:.2e}")

m = build_park()
fine_tune(m, 50, 1e-3, y_true)
shuffle_hidden_units(m, np.random.RandomState(11))
r = measure(m); r["scenario"] = "ft_true + shuffle"
rows.append(r)
print(f"{'ft_true + shuffle':>30}: pair_acc(hun)={r['pair_acc_hun']:.2f} "
      f"pair_acc(greedy)={r['pair_acc_greedy']:.2f} "
      f"sep={r['pair_sep']:+.3f} mse={r['mse_native']:.2e}")

m = build_park()
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.01 * p.std().item())
shuffle_hidden_units(m, np.random.RandomState(13))
r = measure(m); r["scenario"] = "noise_1e-2 + shuffle"
rows.append(r)
print(f"{'noise + shuffle':>30}: pair_acc(hun)={r['pair_acc_hun']:.2f} "
      f"pair_acc(greedy)={r['pair_acc_greedy']:.2f} "
      f"sep={r['pair_sep']:+.3f} mse={r['mse_native']:.2e}")

# Multiple shuffle seeds to confirm invariance
print("\n=== Invariance check: 5 random shuffles, no fine-tune ===")
for sd in range(1, 6):
    m = build_park()
    shuffle_hidden_units(m, np.random.RandomState(sd))
    r = measure(m); r["scenario"] = f"shuffle_seed{sd}"
    rows.append(r)
    print(f"shuffle_seed{sd:>2}: pair_acc(hun)={r['pair_acc_hun']:.2f} "
          f"sep={r['pair_sep']:+.4f} mse={r['mse_native']:.2e}")

pd.DataFrame(rows).to_csv("attack_shuffle_results.csv", index=False)
print(f"\nWrote {len(rows)} rows to attack_shuffle_results.csv")
