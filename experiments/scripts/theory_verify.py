"""Verify theoretical claims with quick numerical checks."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch, numpy as np, pandas as pd
torch.set_grad_enabled(False)

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
true_in = [SOL[2*k] for k in range(48)]
true_out = [SOL[2*k+1] for k in range(48)]
D, h = 48, 96

# ---- Validate Margin Theorem (sanity) ----
print("="*60)
print("Margin Theorem verification")
print("="*60)
print("Theorem: For correctly paired (i,i),")
print("  d(i,i) = |tr(M)| / ||M||_F = sqrt(d) / sqrt(1 + ||E||_F^2/(eps^2 d))")
print("where M = -eps*I + E, tr(E)=0.")
print(f"Upper bound: sqrt(d) = sqrt({D}) = {np.sqrt(D):.4f}\n")

# Empirical: compute for each correct pair
correct_ratios = []
correct_eps    = []
correct_Enorms = []
for k in range(D):
    Win = pieces[true_in[k]]['weight'].to(torch.float64)
    Wout = pieces[true_out[k]]['weight'].to(torch.float64)
    M = (Wout @ Win).numpy()  # (48, 48)
    tr_M = M.trace()
    F_M = np.linalg.norm(M, 'fro')
    d = abs(tr_M) / F_M
    eps = abs(tr_M) / D
    E = M + eps * np.eye(D)
    F_E = np.linalg.norm(E, 'fro')
    correct_ratios.append(d)
    correct_eps.append(eps)
    correct_Enorms.append(F_E)

correct_ratios = np.array(correct_ratios)
correct_eps = np.array(correct_eps)
correct_Enorms = np.array(correct_Enorms)
print(f"Correct pairs: d in [{correct_ratios.min():.3f}, {correct_ratios.max():.3f}], mean {correct_ratios.mean():.3f}")
print(f"  eps in [{correct_eps.min():.3f}, {correct_eps.max():.3f}]")
print(f"  ||E||_F in [{correct_Enorms.min():.3f}, {correct_Enorms.max():.3f}]")
print(f"  ||E||_F / (eps*sqrt(d)) in [{(correct_Enorms / (correct_eps*np.sqrt(D))).min():.3f}, "
      f"{(correct_Enorms / (correct_eps*np.sqrt(D))).max():.3f}]")
# Predicted ratio: sqrt(d) / sqrt(1 + (||E||_F / (eps*sqrt(d)))^2)
predicted = np.sqrt(D) / np.sqrt(1 + (correct_Enorms / (correct_eps * np.sqrt(D)))**2)
err = abs(predicted - correct_ratios).max()
print(f"  Predicted vs empirical: max abs error = {err:.6f}  (should be ~0)")

# ---- Incorrect pair ratios ----
incorrect_ratios = []
incorrect_F = []
for i in range(D):
    for j in range(D):
        if i == j: continue
        Win = pieces[true_in[i]]['weight'].to(torch.float64)
        Wout = pieces[true_out[j]]['weight'].to(torch.float64)
        M = (Wout @ Win).numpy()
        F_M = np.linalg.norm(M, 'fro')
        incorrect_ratios.append(abs(M.trace()) / F_M)
        incorrect_F.append(F_M)
incorrect_ratios = np.array(incorrect_ratios)
incorrect_F = np.array(incorrect_F)
print(f"\nIncorrect pairs (n={len(incorrect_ratios)}): d in [{incorrect_ratios.min():.3f}, {incorrect_ratios.max():.3f}]")
print(f"  expected ~1/sqrt(d) = {1/np.sqrt(D):.4f}")
print(f"  actual mean = {incorrect_ratios.mean():.4f}, median = {np.median(incorrect_ratios):.4f}")

# Frobenius product margin (the other "metric" that succeeds)
correct_F = []
for k in range(D):
    Win = pieces[true_in[k]]['weight'].to(torch.float64)
    Wout = pieces[true_out[k]]['weight'].to(torch.float64)
    M = (Wout @ Win).numpy()
    correct_F.append(np.linalg.norm(M, 'fro'))
correct_F = np.array(correct_F)
print(f"\n||M||_F for correct pairs: [{correct_F.min():.3f}, {correct_F.max():.3f}]")
print(f"||M||_F for incorrect pairs: [{incorrect_F.min():.3f}, {incorrect_F.max():.3f}]")
print(f"\nMargin comparison (relative):")
print(f"  Diagonal dominance: (min correct - max incorrect) / max incorrect = "
      f"{(correct_ratios.min() - incorrect_ratios.max()) / incorrect_ratios.max():.3f}")
print(f"  Frobenius:          (min correct - max incorrect) / max incorrect = "
      f"{(correct_F.min() - incorrect_F.max()) / incorrect_F.max():.3f}")

# ---- Pairing Wall: first-order theoretical prediction ----
print("\n" + "="*60)
print("Pairing Wall slope: theoretical prediction")
print("="*60)
print("Claim: for k mispairs, in the small-k regime,")
print("  E[MSE(k)] ≈ k * E[||W_L · delta_block||^2 / N]")
print("where delta_block is the change in residual contribution from one mispair.")

# Build full forward
import torch.nn.functional as F
df = pd.read_csv("historical_data.csv")
X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float64)
y = torch.tensor(df["pred"].values, dtype=torch.float64).reshape(-1, 1)

WL = pieces[last_id]['weight'].to(torch.float64)
bL = pieces[last_id]['bias'].to(torch.float64)

def forward_with_block_outputs(in_seq, out_seq):
    h = X
    h_list = [h.clone()]
    for k in range(D):
        Win = pieces[in_seq[k]]['weight'].to(torch.float64)
        bin_ = pieces[in_seq[k]]['bias'].to(torch.float64)
        Wout = pieces[out_seq[k]]['weight'].to(torch.float64)
        bout = pieces[out_seq[k]]['bias'].to(torch.float64)
        z = torch.relu(h @ Win.T + bin_)
        delta = z @ Wout.T + bout
        h = h + delta
        h_list.append(h.clone())
    out = h @ WL.T + bL
    return out, h_list

# Truth: get residual stream
out, h_list = forward_with_block_outputs(true_in, true_out)
base_mse = ((out - y)**2).mean().item()
print(f"Base MSE = {base_mse:.2e}")

# At each depth k, what is the typical ||W_L · delta_block||^2 / N effect of replacing
# block k with a random other block?
import random
rng = random.Random(0)
delta_costs = []
for trial in range(60):
    k = rng.randint(0, D-1)
    j = rng.randint(0, D-1)
    while j == k: j = rng.randint(0, D-1)
    # Swap out-partner at position k with that of position j
    in_seq, out_seq = list(true_in), list(true_out)
    out_seq[k], out_seq[j] = out_seq[j], out_seq[k]  # mispair both positions k and j
    # That's 2 mispairs, so divide by 2
    out_new, _ = forward_with_block_outputs(in_seq, out_seq)
    new_mse = ((out_new - y)**2).mean().item()
    delta_costs.append((new_mse - base_mse) / 2)

print(f"Avg per-mispair MSE cost (60 trials): {np.mean(delta_costs):.5f}")
print(f"Std: {np.std(delta_costs):.5f}")
print(f"Predicted slope C ≈ {np.mean(delta_costs):.5f}")
print(f"(Compare to empirical fit from paper: C ≈ 0.0035-0.004)")
