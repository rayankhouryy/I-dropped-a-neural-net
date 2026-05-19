"""Experiments for companion paper.

Ground-truth pairing/ordering is known from our exact solution (MSE = 0).

E1. Compare four pairing metrics against ground truth:
    a) SV-L1: |svd(W_in) - svd(W_out)|_1
    b) Subspace overlap: 48 - ||col(W_in)^T row(W_out)||_F^2
    c) -||W_out W_in||_F (negative Frobenius of the product)
    d) Diagonal dominance: |tr(W_out W_in)| / ||W_out W_in||_F
    For each: #correct/48, margin, random-order MSE.

E2. The Pairing Wall: take ground-truth solution, randomly swap k inp-out
    pairs, measure MSE(k). Repeat over multiple seeds.

E3. Seed comparison: delta-norm vs ||W_out||_F vs ||W_in||_F vs other proxies.
    Compute Spearman, Kendall tau, max displacement, inversion count vs true
    depth ordering, and seed MSE.

E4. Hill-climb convergence from each seed: rounds, swaps, final MSE.
"""
import torch, json, numpy as np, pandas as pd, time, random
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, kendalltau
torch.set_grad_enabled(False)

# ---- Load --------------------------------------------------------------
weights, biases = {}, {}
for i in range(97):
    sd = torch.load(f"pieces/piece_{i}.pth", map_location="cpu", weights_only=True)
    weights[i] = sd["weight"].to(torch.float64)
    biases[i]  = sd["bias"].to(torch.float64)
inp_ids = sorted(i for i in range(97) if weights[i].shape == (96, 48))
out_ids = sorted(i for i in range(97) if weights[i].shape == (48, 96))
last_id = [i for i in range(97) if weights[i].shape == (1, 48)][0]
WL = weights[last_id]; bL = biases[last_id]
df = pd.read_csv("historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float64)
y = torch.tensor(df["pred"].values, dtype=torch.float64).reshape(-1,1)

# Ground truth from our solved permutation
SOL = [43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,
       33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,
       38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,
       90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85]
assert SOL[-1] == last_id
true_pairs = [(SOL[2*k], SOL[2*k+1]) for k in range(48)]  # list of (inp, out) at depth k
true_inp_at_depth = [p[0] for p in true_pairs]
true_out_at_depth = [p[1] for p in true_pairs]
print(f"Loaded. {len(true_pairs)} true pairs.")

# inp_id -> depth (0..47), out_id -> depth (0..47)
inp_depth = {ip: k for k, ip in enumerate(true_inp_at_depth)}
out_depth = {op: k for k, op in enumerate(true_out_at_depth)}
true_partner_of_inp = {true_inp_at_depth[k]: true_out_at_depth[k] for k in range(48)}

def forward(in_seq, out_seq, X=X):
    h = X
    for ip, op in zip(in_seq, out_seq):
        z = torch.relu(h @ weights[ip].T + biases[ip])
        h = h + z @ weights[op].T + biases[op]
    return h @ WL.T + bL

def mse(in_seq, out_seq):
    return ((forward(in_seq, out_seq) - y)**2).mean().item()

# Sanity check
gt_mse = mse(true_inp_at_depth, true_out_at_depth)
print(f"Ground-truth MSE: {gt_mse:.2e}")

# ====================================================================
# E1. Comparative pairing study
# ====================================================================
print("\n" + "="*60)
print("E1. Comparative pairing metrics")
print("="*60)

# Precompute SVDs and bases
inp_svals, inp_colspace = {}, {}
for a in inp_ids:
    W = weights[a].cpu().numpy()  # (96, 48)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    inp_svals[a] = np.sort(S)[::-1]
    inp_colspace[a] = U   # orthonormal cols spanning col(W), in R^96, 48 cols

out_svals, out_rowspace = {}, {}
for b in out_ids:
    W = weights[b].cpu().numpy()  # (48, 96)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    out_svals[b] = np.sort(S)[::-1]
    out_rowspace[b] = Vt.T   # rows of W are spans of Vt; we want subspace in R^96

# Build four cost matrices (lower = better pair)
n = 48
cost_sv = np.zeros((n, n))
cost_sub = np.zeros((n, n))
cost_prodF = np.zeros((n, n))
cost_diag = np.zeros((n, n))

for i, ip in enumerate(inp_ids):
    Win = weights[ip].cpu().numpy()
    for j, op in enumerate(out_ids):
        Wout = weights[op].cpu().numpy()
        M = Wout @ Win   # (48, 48)
        # SV L1
        cost_sv[i, j] = float(np.abs(inp_svals[ip] - out_svals[op]).sum())
        # Subspace overlap: lower cost = closer subspaces
        Mqq = inp_colspace[ip].T @ out_rowspace[op]   # (48, 48)
        cost_sub[i, j] = 48.0 - float(np.sum(Mqq*Mqq))
        # -||M||_F (lower = bigger product = more interaction)
        cost_prodF[i, j] = -float(np.linalg.norm(M, 'fro'))
        # Diagonal dominance ratio: higher = better, so cost = -ratio
        cost_diag[i, j] = -abs(float(M.trace())) / (float(np.linalg.norm(M, 'fro')) + 1e-12)

def assess_metric(cost, name):
    row, col = linear_sum_assignment(cost)
    # correct pair count: did Hungarian pick (inp_ids[i], out_ids[col[i]]) matching truth?
    n_correct = 0
    pairs = [(inp_ids[i], out_ids[col[i]]) for i in range(n)]
    for ip, op in pairs:
        if true_partner_of_inp[ip] == op:
            n_correct += 1
    # margin: for each row, best vs runner-up
    margins = []
    for i in range(n):
        c_sorted = np.sort(cost[i])
        margins.append(c_sorted[1] - c_sorted[0])
    margin_min = float(np.min(margins))
    margin_mean = float(np.mean(margins))
    # random-order MSE
    rng = random.Random(0)
    msvs = []
    for t in range(8):
        order = list(range(48)); rng.shuffle(order)
        in_seq  = [pairs[k][0] for k in order]
        out_seq = [pairs[k][1] for k in order]
        msvs.append(mse(in_seq, out_seq))
    print(f"  {name:>30s}: correct={n_correct:2d}/48  margin_min={margin_min:.3f}  margin_mean={margin_mean:.3f}  random-MSE mean={np.mean(msvs):.3f}, min={min(msvs):.3f}")
    return {"name": name, "n_correct": n_correct, "margin_min": margin_min,
            "margin_mean": margin_mean, "rand_mse_mean": float(np.mean(msvs)),
            "rand_mse_min": float(min(msvs)), "pairs": pairs}

print("\nMetric             correct  margin_min  margin_mean  rand-order MSE")
print("-"*78)
r_sv     = assess_metric(cost_sv,     "SV-L1 spectrum")
r_sub    = assess_metric(cost_sub,    "Subspace overlap (principal angles)")
r_prodF  = assess_metric(cost_prodF,  "-||Wout Win||_F (Frobenius)")
r_diag   = assess_metric(cost_diag,   "Diagonal dominance |tr|/||F||")

# ====================================================================
# E2. The Pairing Wall
# ====================================================================
print("\n" + "="*60)
print("E2. Pairing Wall: MSE vs k mis-paired blocks")
print("="*60)
print("(starting from exact solution, randomly swap k out-partners among k random pairs)")

rng = random.Random(42)
wall = {}
for k in [0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48]:
    msvs = []
    for trial in range(20 if k > 0 else 1):
        in_seq = list(true_inp_at_depth)
        out_seq = list(true_out_at_depth)
        if k > 0:
            # pick k positions, randomly permute their out-partners (within those k)
            positions = rng.sample(range(48), k)
            outs = [out_seq[p] for p in positions]
            outs_shuffled = list(outs); rng.shuffle(outs_shuffled)
            # ensure non-trivial permutation (no fixed point)
            tries = 0
            while any(outs[i] == outs_shuffled[i] for i in range(k)) and tries < 100:
                rng.shuffle(outs_shuffled); tries += 1
            for p, o in zip(positions, outs_shuffled):
                out_seq[p] = o
        m = mse(in_seq, out_seq)
        msvs.append(m)
    wall[k] = (float(np.mean(msvs)), float(np.std(msvs)))
    print(f"  k={k:2d} mis-pairs: MSE = {np.mean(msvs):.4f} +- {np.std(msvs):.4f}")

# ====================================================================
# E3. Seed quality: depth-correlation of various proxies
# ====================================================================
print("\n" + "="*60)
print("E3. Seed proxy correlation with true depth")
print("="*60)

# Compute proxies indexed by block (using known pairing)
def block_props(ip, op, X_input):
    W1 = weights[ip]; b1 = biases[ip]
    W2 = weights[op]; b2 = biases[op]
    delta = (torch.relu(X_input @ W1.T + b1) @ W2.T + b2)  # (N, 48)
    return {
        "delta_norm": delta.norm(dim=1).mean().item(),
        "Wout_F": W2.norm().item(),
        "Win_F": W1.norm().item(),
        "WoutWin_F": (W2 @ W1).norm().item(),
        "trace_neg": -float((W2 @ W1).trace().item()),
        "delta_norm_atdepth": None,  # filled below
    }

# delta-norm computed at each block's actual depth (i.e., the "real" residual perturbation)
def compute_actual_delta_norms():
    h = X.clone()
    res = {}
    for k in range(48):
        ip = true_inp_at_depth[k]; op = true_out_at_depth[k]
        W1 = weights[ip]; b1 = biases[ip]
        W2 = weights[op]; b2 = biases[op]
        delta = (torch.relu(h @ W1.T + b1) @ W2.T + b2)
        res[k] = delta.norm(dim=1).mean().item()
        h = h + delta
    return res

actual_delta = compute_actual_delta_norms()

props = []
for k in range(48):
    ip = true_inp_at_depth[k]; op = true_out_at_depth[k]
    p = block_props(ip, op, X)
    p["true_depth"] = k
    p["delta_norm_atdepth"] = actual_delta[k]
    props.append(p)

import pandas as pd2
P = pd2.DataFrame(props)
depths = P["true_depth"].values

for col in ["delta_norm", "Wout_F", "Win_F", "WoutWin_F", "trace_neg", "delta_norm_atdepth"]:
    vals = P[col].values
    rho, _ = spearmanr(vals, depths)
    tau, _ = kendalltau(vals, depths)
    # Use the proxy to predict ranking, count inversions vs truth
    order = np.argsort(vals)  # ascending
    # ranks
    # max displacement
    rank_of_depth = np.empty(48, dtype=int)
    for new_pos, k in enumerate(order):
        rank_of_depth[k] = new_pos
    max_disp = int(np.max(np.abs(rank_of_depth - depths)))
    # mean displacement
    mean_disp = float(np.mean(np.abs(rank_of_depth - depths)))
    print(f"  {col:>22s}: rho={rho:+.3f}  tau={tau:+.3f}  max_disp={max_disp:3d}  mean_disp={mean_disp:.2f}")

# ====================================================================
# E4. Hill-climb convergence from each seed
# ====================================================================
print("\n" + "="*60)
print("E4. Hill-climb convergence from each seed (correct pairing)")
print("="*60)

# Build pair list in canonical order
canon_pairs = [(true_inp_at_depth[k], true_out_at_depth[k]) for k in range(48)]

X_sub, y_sub = X[:1000], y[:1000].squeeze()

def mse_sub(in_seq, out_seq):
    h = X_sub
    for ip, op in zip(in_seq, out_seq):
        z = torch.relu(h @ weights[ip].T + biases[ip])
        h = h + z @ weights[op].T + biases[op]
    out = h @ WL.T + bL
    return ((out.squeeze() - y_sub)**2).mean().item()

def hillclimb_from_seed(seed_perm, max_rounds=30):
    """seed_perm is list of 48 block indices into canon_pairs"""
    current = [canon_pairs[k] for k in seed_perm]
    cur_mse = mse_sub([p[0] for p in current], [p[1] for p in current])
    total_swaps = 0; rounds = 0
    history = [(0, 0, cur_mse)]
    for r in range(1, max_rounds + 1):
        swaps = 0
        for i in range(47):
            trial = list(current); trial[i], trial[i+1] = trial[i+1], trial[i]
            m = mse_sub([p[0] for p in trial], [p[1] for p in trial])
            if m < cur_mse - 1e-12:
                current = trial; cur_mse = m; swaps += 1
        for i in range(46, -1, -1):
            trial = list(current); trial[i], trial[i+1] = trial[i+1], trial[i]
            m = mse_sub([p[0] for p in trial], [p[1] for p in trial])
            if m < cur_mse - 1e-12:
                current = trial; cur_mse = m; swaps += 1
        total_swaps += swaps; rounds += 1
        history.append((r, swaps, cur_mse))
        if swaps == 0: break
    return rounds, total_swaps, cur_mse, history

# Build seeds by each proxy
seeds = {}
for col in ["delta_norm", "Wout_F", "Win_F", "WoutWin_F", "trace_neg", "delta_norm_atdepth"]:
    vals = P[col].values
    order_by_proxy = list(np.argsort(vals))  # depth indices in proxy-order
    seeds[col] = order_by_proxy

print(f"{'Seed':>22s}  {'Seed MSE':>10s}  {'Rounds':>7s}  {'Swaps':>6s}  {'Final MSE':>12s}")
print("-"*78)
hc_results = {}
for name, seed in seeds.items():
    seed_in_seq  = [canon_pairs[k][0] for k in seed]
    seed_out_seq = [canon_pairs[k][1] for k in seed]
    seed_mse = mse_sub(seed_in_seq, seed_out_seq)
    rounds, swaps, final, hist = hillclimb_from_seed(seed, max_rounds=30)
    hc_results[name] = {"seed_mse": seed_mse, "rounds": rounds, "swaps": swaps,
                       "final": final, "history": hist}
    print(f"{name:>22s}  {seed_mse:10.4f}  {rounds:7d}  {swaps:6d}  {final:12.2e}")

# Save all results
out = {
    "ground_truth_mse": gt_mse,
    "E1_pairing_metrics": [
        {k: v for k, v in r.items() if k != "pairs"} for r in [r_sv, r_sub, r_prodF, r_diag]
    ],
    "E2_pairing_wall": {str(k): list(v) for k, v in wall.items()},
    "E3_seed_correlation": [
        {"proxy": col,
         "spearman": float(spearmanr(P[col].values, depths)[0]),
         "kendall": float(kendalltau(P[col].values, depths)[0]),
         "max_disp": int(np.max(np.abs(np.argsort(np.argsort(P[col].values)) - depths))),
         "mean_disp": float(np.mean(np.abs(np.argsort(np.argsort(P[col].values)) - depths)))}
        for col in ["delta_norm", "Wout_F", "Win_F", "WoutWin_F", "trace_neg", "delta_norm_atdepth"]
    ],
    "E4_hillclimb": hc_results,
}
json.dump(out, open("paper_experiments.json", "w"), indent=2, default=str)
print("\nsaved paper_experiments.json")
