"""Generate all paper figures.

Reads from:
  - paper_experiments.json    (Park puzzle: pairing wall, metrics, seed corr)
  - convergence_d24w64.csv    (training curves, single config)
  - strategies.csv            (bubble vs SA comparison)
  - sweep_full.csv            (multi-config sweep, if available)
  - pieces/ + historical_data.csv (Park puzzle raw data, for matrix viz)
"""
import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Park puzzle data
torch.set_grad_enabled(False)
pieces = {}
for i in range(97):
    sd = torch.load(f"pieces/piece_{i}.pth", map_location="cpu", weights_only=True)
    pieces[i] = sd
inp_ids = sorted(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([96, 48]))
out_ids = sorted(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([48, 96]))
last_id = next(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([1, 48]))

SOL = [43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,
       33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,
       38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,
       90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85]
true_in  = [SOL[2*k]   for k in range(48)]
true_out = [SOL[2*k+1] for k in range(48)]

# Compute diagonal-dominance matrix for the puzzle, in TRUE order
score_park = np.zeros((48, 48))
for i_dep, ip in enumerate(inp_ids):
    W_in = pieces[ip]['weight'].to(torch.float64)
    for j_dep, op in enumerate(out_ids):
        W_out = pieces[op]['weight'].to(torch.float64)
        M = (W_out @ W_in).numpy()
        score_park[i_dep, j_dep] = abs(M.trace()) / (np.linalg.norm(M, 'fro') + 1e-12)

# Re-index so rows/cols are in TRUE depth order
def reorder_to_truth(score_matrix, true_in_ids, true_out_ids, inp_ids, out_ids):
    inp_pos = {ip: k for k, ip in enumerate(inp_ids)}
    out_pos = {op: k for k, op in enumerate(out_ids)}
    rows = [inp_pos[ip] for ip in true_in_ids]
    cols = [out_pos[op] for op in true_out_ids]
    return score_matrix[np.ix_(rows, cols)]
score_park_ordered = reorder_to_truth(score_park, true_in, true_out, inp_ids, out_ids)

# ---------------------------------------------------------------- FIG 1: pairing matrix
fig, ax = plt.subplots(figsize=(6.5, 5.2))
im = ax.imshow(score_park_ordered, cmap="viridis", aspect="auto")
ax.set_xlabel("$W_{\\mathrm{out}}$ index (true depth order)")
ax.set_ylabel("$W_{\\mathrm{in}}$ index (true depth order)")
ax.set_title("Park's puzzle: diagonal-dominance ratio $d(i,j)$\n(diagonal = correctly paired; bright band visible)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$d(i,j) = |\\mathrm{tr}(W_{\\mathrm{out}} W_{\\mathrm{in}})| / \\|W_{\\mathrm{out}} W_{\\mathrm{in}}\\|_F$")
plt.savefig(f"{OUT}/fig_pairing_matrix_park.pdf"); plt.savefig(f"{OUT}/fig_pairing_matrix_park.png"); plt.close()

# ---------------------------------------------------------------- FIG 2: pairing wall
peJ = json.load(open("paper_experiments.json"))
wall = peJ["E2_pairing_wall"]
ks = sorted(int(k) for k in wall)
mses = [wall[str(k)][0] for k in ks]
stds = [wall[str(k)][1] for k in ks]
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar(ks[:9], mses[:9], yerr=stds[:9], marker="o", capsize=3, linewidth=1.5, color="#c0392b")
# Linear fit on small-k regime
small = [k for k in ks if 0 < k <= 8]
small_mses = [wall[str(k)][0] for k in small]
slope, intercept = np.polyfit(small, small_mses, 1)
xs = np.linspace(0, 10, 50)
ax.plot(xs, slope*xs + intercept, "--", color="#2c3e50", alpha=0.7,
        label=f"Linear fit (k≤8): MSE ≈ {slope:.4f}·k")
ax.set_xlabel("Number of mis-paired blocks $k$")
ax.set_ylabel("MSE on Jane Street puzzle")
ax.set_title("The Pairing Wall: MSE vs.\\ $k$ mis-paired blocks\n(ordering held at ground truth)")
ax.set_xlim(-1, 35); ax.set_ylim(-0.02, 0.85)
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(f"{OUT}/fig_pairing_wall.pdf"); plt.savefig(f"{OUT}/fig_pairing_wall.png"); plt.close()

# ---------------------------------------------------------------- FIG 3: pairing metric comparison
metrics_data = peJ["E1_pairing_metrics"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
names_short = ["SV-L1\nspectrum", "Subspace\noverlap", "$-\\|M\\|_F$", "Diagonal\ndominance"]
colors = ["#7f8c8d", "#e67e22", "#3498db", "#27ae60"]
ax = axes[0]
n_correct = [m["n_correct"] for m in metrics_data]
bars = ax.bar(names_short, n_correct, color=colors)
ax.axhline(48, ls="--", color="black", alpha=0.4, label="ground truth (48)")
ax.set_ylabel("Pairs correctly recovered (/48)")
ax.set_title("Pairing accuracy by metric")
ax.set_ylim(0, 52)
for b, v in zip(bars, n_correct):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, str(v), ha="center", fontsize=10, fontweight="bold")
ax.legend()
ax = axes[1]
margins = [m["margin_min"] for m in metrics_data]
bars = ax.bar(names_short, margins, color=colors)
ax.set_ylabel("Min margin (best vs.\\ runner-up)")
ax.set_title("Robustness: min margin per metric (log scale)")
ax.set_yscale("symlog", linthresh=0.01)
for b, v in zip(bars, margins):
    ax.text(b.get_x() + b.get_width()/2, v*1.15 if v > 0 else 0.02,
            f"{v:.3f}", ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.savefig(f"{OUT}/fig_metric_comparison.pdf"); plt.savefig(f"{OUT}/fig_metric_comparison.png"); plt.close()

# ---------------------------------------------------------------- FIG 4: Wout norm vs depth (Park)
WL = pieces[last_id]['weight'].to(torch.float64)
park_wout = np.array([pieces[op]['weight'].to(torch.float64).norm().item() for op in true_out])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
ax.scatter(np.arange(48), park_wout, s=22, color="#c0392b", alpha=0.85)
rho_park, _ = spearmanr(park_wout, np.arange(48))
z = np.polyfit(np.arange(48), park_wout, 1)
ax.plot(np.arange(48), z[0]*np.arange(48) + z[1], "--", color="black", alpha=0.5,
        label=f"linear fit\nSpearman $\\rho$ = {rho_park:+.3f}")
ax.set_xlabel("True depth $k$"); ax.set_ylabel("$\\|W_{\\mathrm{out}}^{(k)}\\|_F$")
ax.set_title("Park's puzzle network (depth 48): positive $\\rho$")
ax.legend(); ax.grid(alpha=0.3)

# Our networks (will be filled after sweep), but use d24w64 csv for now
try:
    sw = pd.read_csv("sweep_full.csv") if os.path.exists("sweep_full.csv") else None
except Exception:
    sw = None
if sw is not None and "wout_norms" in sw.columns:
    # take last-epoch row per config/seed
    final = sw.sort_values("epoch").groupby(["name", "seed"]).tail(1)
    # pick d24_h64 seed 0 if present, else fall back to focused_run data
    candidate = final[final["name"].str.contains("d24")].head(1) if "name" in final else None
else:
    candidate = None

# Fallback: take d24w64 from focused script's CSV
ax = axes[1]
df_focused = pd.read_csv("convergence_d24w64.csv")
# We saved Wout norms? not in this CSV. So we have to compute from a quick train.
# Instead show wout_norm_rho trajectory across training:
for seed in df_focused["seed"].unique():
    sub = df_focused[df_focused["seed"] == seed].sort_values("epoch")
    ax.plot(sub["epoch"], sub["wout_norm_rho"], marker="o", linewidth=1.5,
            label=f"seed {seed}")
ax.axhline(rho_park, color="#c0392b", linestyle="--", alpha=0.7,
           label=f"Park's puzzle ($\\rho={rho_park:+.3f}$)")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax.set_xscale("log"); ax.set_xlim(0.7, 1000)
ax.set_xlabel("Training epoch"); ax.set_ylabel("Spearman $\\rho(\\|W_{\\mathrm{out}}\\|_F,\\, k)$")
ax.set_title("Our networks (d24, h64): $\\rho$ is NEGATIVE")
ax.legend(); ax.grid(alpha=0.3)
plt.savefig(f"{OUT}/fig_rho_sign_flip.pdf"); plt.savefig(f"{OUT}/fig_rho_sign_flip.png"); plt.close()

# ---------------------------------------------------------------- FIG 5: pair separation & pair acc vs training
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
for seed in df_focused["seed"].unique():
    sub = df_focused[df_focused["seed"] == seed].sort_values("epoch")
    ax.plot(sub["epoch"], sub["pair_acc"], marker="o", linewidth=1.5,
            label=f"seed {seed}")
ax.set_xscale("log"); ax.set_xlim(0.7, 1000); ax.set_ylim(-0.05, 1.05)
ax.axhline(1.0, color="green", ls="--", alpha=0.4)
ax.set_xlabel("Training epoch"); ax.set_ylabel("Pairing accuracy (Hungarian on $-d$)")
ax.set_title("Pair accuracy hits 1.0 by epoch 5 and stays")
ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
for seed in df_focused["seed"].unique():
    sub = df_focused[df_focused["seed"] == seed].sort_values("epoch")
    ax.plot(sub["epoch"], sub["pair_sep"], marker="o", linewidth=1.5,
            label=f"seed {seed}")
ax.axhline(0.0, color="black", linewidth=0.5)
ax.set_xscale("log"); ax.set_xlim(0.7, 1000)
ax.set_xlabel("Training epoch")
ax.set_ylabel("Pair separation $\\min_i d(i,i) - \\max_{i\\neq j} d(i,j)$")
ax.set_title("Pair separation is non-monotonic\n(peaks mid-training, degrades with overtraining)")
ax.legend(); ax.grid(alpha=0.3)
plt.savefig(f"{OUT}/fig_pair_acc_sep.pdf"); plt.savefig(f"{OUT}/fig_pair_acc_sep.png"); plt.close()

# ---------------------------------------------------------------- FIG 6: bubble vs SA convergence
ST = pd.read_csv("strategies.csv")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
W = 0.18
xs = np.arange(len(ST))
ax.bar(xs - 1.5*W, ST["mse_asc_hc"],  W, label="Asc seed + bubble-sort", color="#9b59b6", alpha=0.85)
ax.bar(xs - 0.5*W, ST["mse_desc_hc"], W, label="Desc seed + bubble-sort", color="#8e44ad", alpha=0.85)
ax.bar(xs + 0.5*W, ST["mse_asc_sa"],  W, label="Asc seed + SA",           color="#27ae60", alpha=0.85)
ax.bar(xs + 1.5*W, ST["mse_desc_sa"], W, label="Desc seed + SA",          color="#16a085", alpha=0.85)
ax.plot(xs, ST["train_loss"], "k--", marker="x", linewidth=1.3,
        label="training loss floor", zorder=10)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([f"s{int(r['seed'])}\nep{int(r['epoch'])}" for _, r in ST.iterrows()])
ax.set_ylabel("Final MSE after ordering search (log)")
ax.set_title("Bubble-sort never reaches the training-loss floor; SA reliably does")
ax.legend(ncol=2, fontsize=8.5, loc="upper left")
ax.grid(axis="y", which="both", alpha=0.25)
plt.savefig(f"{OUT}/fig_bubble_vs_sa.pdf"); plt.savefig(f"{OUT}/fig_bubble_vs_sa.png"); plt.close()

# ---------------------------------------------------------------- FIG 7: Park's per-block Wout norms vs depth
fig, ax = plt.subplots(figsize=(7.5, 3.7))
ax.plot(np.arange(48), park_wout, marker="o", color="#c0392b", linewidth=1.5)
ax.set_xlabel("True depth $k$"); ax.set_ylabel("$\\|W_{\\mathrm{out}}^{(k)}\\|_F$")
ax.set_title("Per-block $\\|W_{\\mathrm{out}}\\|_F$ in Park's puzzle network\n"
             f"(ascending sort reflects true order; Spearman $\\rho={rho_park:+.3f}$)")
ax.grid(alpha=0.3)
plt.savefig(f"{OUT}/fig_park_wout_per_block.pdf"); plt.savefig(f"{OUT}/fig_park_wout_per_block.png"); plt.close()

# ---------------------------------------------------------------- FIG 8: pairing wall on log-log
fig, ax = plt.subplots(figsize=(6.5, 4.0))
small_k = [k for k in ks if k >= 2 and k <= 16]
small_m = [wall[str(k)][0] for k in small_k]
ax.loglog(small_k, small_m, "o-", color="#c0392b", linewidth=1.5, label="empirical")
# Fit slope on log-log
log_slope, log_intercept = np.polyfit(np.log(small_k), np.log(small_m), 1)
xs = np.linspace(2, 16, 50)
ax.loglog(xs, np.exp(log_intercept) * xs**log_slope, "--", color="black", alpha=0.6,
          label=f"power-law fit: MSE ~ $k^{{{log_slope:.2f}}}$")
ax.set_xlabel("Number of mis-paired blocks $k$ (log)")
ax.set_ylabel("MSE (log)")
ax.set_title("Pairing Wall on log-log: super-linear growth")
ax.legend(); ax.grid(which="both", alpha=0.3)
plt.savefig(f"{OUT}/fig_pairing_wall_loglog.pdf"); plt.savefig(f"{OUT}/fig_pairing_wall_loglog.png"); plt.close()

# ---------------------------------------------------------------- FIG 9: seed-proxy rank correlation
e3 = peJ["E3_seed_correlation"]
fig, ax = plt.subplots(figsize=(7.5, 4.0))
proxies = [r["proxy"] for r in e3]
rhos = [r["spearman"] for r in e3]
disps = [r["max_disp"] for r in e3]
labels_pretty = {"delta_norm": "$\\delta$-norm (input)",
                 "Wout_F": "$\\|W_{\\mathrm{out}}\\|_F$",
                 "Win_F": "$\\|W_{\\mathrm{in}}\\|_F$",
                 "WoutWin_F": "$\\|W_{\\mathrm{out}} W_{\\mathrm{in}}\\|_F$",
                 "trace_neg": "$-\\mathrm{tr}(W_{\\mathrm{out}} W_{\\mathrm{in}})$",
                 "delta_norm_atdepth": "$\\delta$-norm (at depth)"}
labels = [labels_pretty.get(p, p) for p in proxies]
order_idx = np.argsort(rhos)[::-1]
ax.barh(np.array(labels)[order_idx], np.array(rhos)[order_idx],
        color=["#27ae60" if r > 0.9 else "#e67e22" if r > 0.5 else "#c0392b" for r in np.array(rhos)[order_idx]])
ax.set_xlabel("Spearman $\\rho$ with true depth")
ax.set_title("Seed-proxy rank correlation in Park's puzzle network\n($\\|W_{\\mathrm{out}}\\|_F$ wins; $\\delta$-norm second)")
ax.axvline(0.95, ls="--", color="green", alpha=0.5, label="$\\rho=0.95$")
ax.axvline(0.9,  ls="--", color="orange", alpha=0.5, label="$\\rho=0.90$")
for i, (rho, disp) in enumerate(zip(np.array(rhos)[order_idx], np.array(disps)[order_idx])):
    ax.text(rho + 0.01, i, f"  max disp={disp}", va="center", fontsize=8.5)
ax.legend()
plt.savefig(f"{OUT}/fig_seed_rho.pdf"); plt.savefig(f"{OUT}/fig_seed_rho.png"); plt.close()

print("Done. Wrote figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
