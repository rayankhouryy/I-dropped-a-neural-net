"""Clean solver using the diagonal-dominance pairing trick + ||W_out||_F seed + hill-climb."""
import torch, torch.nn as nn, pandas as pd, time, os
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

PIECES_DIR = "pieces"
DATA_CSV = "historical_data.csv"

# Load pieces
pieces = {}
for i in range(97):
    pieces[i] = torch.load(os.path.join(PIECES_DIR, f"piece_{i}.pth"),
                            map_location="cpu", weights_only=True)
inp_pieces = sorted(i for i in pieces if pieces[i]['weight'].shape == torch.Size([96, 48]))
out_pieces = sorted(i for i in pieces if pieces[i]['weight'].shape == torch.Size([48, 96]))
last_piece = next(i for i in pieces if pieces[i]['weight'].shape == torch.Size([1, 48]))
print(f"inp={len(inp_pieces)}, out={len(out_pieces)}, last={last_piece}")

# Load data
df = pd.read_csv(DATA_CSV)
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
y_pred = torch.tensor(df["pred"].values, dtype=torch.float32)
print(f"X={tuple(X.shape)}, y={tuple(y_pred.shape)}")

# Step 1: diagonal-dominance pairing
print("\nStep 1: pairing via diagonal dominance ratio")
n = 48
ratio = torch.zeros(n, n)
for i, ip in enumerate(inp_pieces):
    W_in = pieces[ip]['weight']  # (96,48)
    for j, op in enumerate(out_pieces):
        W_out = pieces[op]['weight']  # (48,96)
        M = W_out @ W_in            # (48,48)
        ratio[i, j] = abs(M.trace().item()) / (M.norm().item() + 1e-10)

cost = (ratio.max() + 1.0 - ratio).numpy()
row, col = linear_sum_assignment(cost)
paired = [(inp_pieces[i], out_pieces[col[i]]) for i in range(n)]
matched_ratios = [ratio[i, col[i]].item() for i in range(n)]
print(f"  matched ratios: min={min(matched_ratios):.3f}, max={max(matched_ratios):.3f}, mean={sum(matched_ratios)/n:.3f}")

# Step 2: seed order by ||W_out||_F
print("\nStep 2: seed order by ||W_out||_F")
order_keys = [(pieces[op]['weight'].norm().item(), k) for k, (ip, op) in enumerate(paired)]
order_keys.sort()
ordered = [paired[k] for _, k in order_keys]

# eval helper
def eval_mse(block_list, X, y):
    h = X
    for ip, op in block_list:
        z = torch.relu(h @ pieces[ip]['weight'].T + pieces[ip]['bias'])
        h = h + z @ pieces[op]['weight'].T + pieces[op]['bias']
    out = h @ pieces[last_piece]['weight'].T + pieces[last_piece]['bias']
    return ((out.squeeze() - y) ** 2).mean().item()

X_sub, y_sub = X[:1000], y_pred[:1000]
print(f"  seed MSE (N=1000): {eval_mse(ordered, X_sub, y_sub):.6f}")

# Step 3: bubble-sort hill-climb
print("\nStep 3: hill-climb")
cur_mse = eval_mse(ordered, X_sub, y_sub)
for r in range(1, 30):
    swaps = 0
    for i in range(len(ordered) - 1):
        trial = list(ordered); trial[i], trial[i+1] = trial[i+1], trial[i]
        m = eval_mse(trial, X_sub, y_sub)
        if m < cur_mse - 1e-10:
            ordered = trial; cur_mse = m; swaps += 1
    for i in range(len(ordered) - 2, -1, -1):
        trial = list(ordered); trial[i], trial[i+1] = trial[i+1], trial[i]
        m = eval_mse(trial, X_sub, y_sub)
        if m < cur_mse - 1e-10:
            ordered = trial; cur_mse = m; swaps += 1
    print(f"  round {r}: {swaps} swaps, MSE = {cur_mse:.10f}")
    if swaps == 0: break

# Gap swaps
for gap in range(2, 6):
    for i in range(len(ordered) - gap):
        trial = list(ordered); trial[i], trial[i+gap] = trial[i+gap], trial[i]
        m = eval_mse(trial, X_sub, y_sub)
        if m < cur_mse - 1e-10:
            ordered = trial; cur_mse = m

final_mse_full = eval_mse(ordered, X, y_pred)
print(f"\nFINAL full MSE: {final_mse_full:.12f}")

# Emit permutation
perm = []
for ip, op in ordered:
    perm += [ip, op]
perm.append(last_piece)
assert len(perm) == 97 and len(set(perm)) == 97
print("\nPERMUTATION:")
print(",".join(map(str, perm)))
