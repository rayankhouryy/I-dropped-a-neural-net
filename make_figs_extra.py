"""Generate new figures: margin theorem prediction, pairing wall theory overlay, attack robustness."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch, numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)

OUT = "figures"
import os; os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ Park network
pieces = {}
for i in range(97):
    pieces[i] = torch.load(f"pieces/piece_{i}.pth", map_location="cpu", weights_only=True)
last_id = next(i for i in range(97) if pieces[i]['weight'].shape == torch.Size([1, 48]))
SOL = [43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,
       33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,
       38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,
       90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85]
true_in  = [SOL[2*k] for k in range(48)]
true_out = [SOL[2*k+1] for k in range(48)]
D = 48

# ============================================================ FIG: Margin theorem
print("Building margin theorem figure...")
emp, pred, eps_arr, E_arr = [], [], [], []
for k in range(D):
    Win = pieces[true_in[k]]['weight'].to(torch.float64)
    Wout = pieces[true_out[k]]['weight'].to(torch.float64)
    M = (Wout @ Win).numpy()
    tr_M = M.trace(); F_M = np.linalg.norm(M, 'fro')
    emp.append(abs(tr_M)/F_M)
    eps = abs(tr_M)/D
    E = M + eps*np.eye(D)
    F_E = np.linalg.norm(E, 'fro')
    pred.append(np.sqrt(D)/np.sqrt(1 + (F_E/(eps*np.sqrt(D)))**2))
    eps_arr.append(eps); E_arr.append(F_E)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.scatter(emp, pred, s=80, alpha=0.8, edgecolors='k')
xy = np.linspace(min(emp+pred)*0.95, max(emp+pred)*1.05, 100)
ax.plot(xy, xy, 'r--', label='Predicted = Empirical')
ax.axhline(np.sqrt(D), ls=':', c='grey', label=fr'$\sqrt{{d}}$ = {np.sqrt(D):.2f} (upper bound)')
ax.set_xlabel(r'Empirical $d(i,i) = |\mathrm{tr}(M)|/\|M\|_F$')
ax.set_ylabel(r'Theorem: $\sqrt{d}/\sqrt{1+\|E\|_F^2/(\varepsilon^2 d)}$')
ax.set_title(f'Margin formula matches empirics\n(max error: {max(abs(np.array(emp)-np.array(pred))):.2e})')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)

ax = axes[1]
ax.bar(range(D), sorted(emp), color='steelblue', label='correct pairs')
ax.axhline(1/np.sqrt(D), color='red', ls='--', label=fr'$1/\sqrt{{d}}\approx{1/np.sqrt(D):.2f}$ (incorrect baseline)')
ax.axhline(np.sqrt(D), color='grey', ls=':', label=fr'$\sqrt{{d}}\approx{np.sqrt(D):.2f}$ (theoretical max)')
ax.set_xlabel('Block index (sorted by $d(i,i)$)')
ax.set_ylabel(r'$d(i,i)$')
ax.set_title(r'Margins for correctly paired blocks vs.\ theoretical bounds')
ax.legend(loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_margin_theorem.png", dpi=150)
plt.savefig(f"{OUT}/fig_margin_theorem.pdf"); plt.close()
print(f"  saved fig_margin_theorem (max abs err {max(abs(np.array(emp)-np.array(pred))):.2e})")

# ============================================================ FIG: Pairing wall with theory line
print("Building pairing wall + theory figure...")
try:
    peJ = json.load(open("paper_experiments.json"))
    wall = peJ["E2_pairing_wall"]
    ks = sorted([int(k) for k in wall.keys() if int(k) > 0])
    mses_mean = [wall[str(k)][0] for k in ks]
    mses_std = [wall[str(k)][1] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    C_theory = 0.00301
    for ax, (xlim, title) in zip(axes, [(8, 'Linear regime ($k\\leq 8$)'), (32, 'Full range ($k\\leq 32$)')]):
        ks_sub = [k for k in ks if k <= xlim]
        mses_mean_sub = [m for k, m in zip(ks, mses_mean) if k <= xlim]
        mses_std_sub = [s for k, s in zip(ks, mses_std) if k <= xlim]
        ax.errorbar(ks_sub, mses_mean_sub, yerr=mses_std_sub, fmt='o-', label='Empirical (20 trials)', capsize=3, color='C0')
        ax.plot(ks_sub, [C_theory*k for k in ks_sub], 'r--', label=fr'First-order theory: $C k$, $C\approx{C_theory:.4f}$', lw=2)
        ax.set_xlabel('Number of mispaired blocks $k$')
        ax.set_ylabel('Mean MSE')
        ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_pairing_wall_theory.png", dpi=150)
    plt.savefig(f"{OUT}/fig_pairing_wall_theory.pdf"); plt.close()
    print("  saved fig_pairing_wall_theory")
except Exception as e:
    print(f"  pairing_wall_theory skipped: {e}")

# ============================================================ FIG: attack robustness
print("Building attack robustness figure...")
if os.path.exists("attack_results.csv"):
    df = pd.read_csv("attack_results.csv")
    df['effort'] = df['epochs'].astype(float) * df['lr'].astype(float).abs()
    # Attack-strength axis: total parameter movement proxy
    # We'll use orig_mse as a unified "attack effect" metric on x-axis,
    # so all 3 attacks fit on same plot.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    attacks = ['ft_pred', 'ft_true', 'noise']
    colors = {'ft_pred': 'C0', 'ft_true': 'C1', 'noise': 'C2'}
    markers = {'ft_pred': 'o', 'ft_true': 's', 'noise': '^'}
    labels = {'ft_pred': 'Fine-tune vs. own predictions',
              'ft_true': 'Fine-tune vs. held-out labels',
              'noise': 'Gaussian weight noise'}

    metrics = [('pair_acc', 'Pair accuracy', (0, 1.05), False),
               ('pair_sep', 'Pair separation', None, False),
               ('wout_rho', r'$\rho(\|W_{out}\|_F, k)$', None, False),
               ('orig_mse', 'Output deviation MSE', None, True),
               ('sa_asc', 'SA-asc reassembly MSE', None, True)]

    for k, (metric, title, ylim, logy) in enumerate(metrics):
        ax = axes.flat[k]
        for a in attacks:
            sub = df[df['attack']==a].copy()
            if len(sub)==0: continue
            sub = sub.sort_values('orig_mse')
            x = sub['orig_mse'].clip(lower=1e-15)
            y = sub[metric]
            ax.plot(x, y, marker=markers[a], color=colors[a], label=labels[a], lw=1.5, ms=7, alpha=0.85)
        ax.set_xscale('log')
        if logy: ax.set_yscale('log')
        if ylim: ax.set_ylim(*ylim)
        ax.set_xlabel('Output deviation from original (MSE)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3, which='both')
        if k == 0: ax.legend(loc='lower left', fontsize=8)
    # last subplot: best-of-two-direction reassembly
    ax = axes.flat[5]
    for a in attacks:
        sub = df[df['attack']==a].copy()
        if len(sub)==0: continue
        sub = sub.sort_values('orig_mse')
        x = sub['orig_mse'].clip(lower=1e-15)
        y = sub[['sa_asc','sa_desc']].min(axis=1)
        ax.plot(x, y, marker=markers[a], color=colors[a], label=labels[a], lw=1.5, ms=7, alpha=0.85)
    ax.plot(x, x, 'k:', alpha=0.5, label='y = x (reassembly = output deviation)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Output deviation from original (MSE)')
    ax.set_ylabel('Best reassembly MSE')
    ax.set_title('Reassembly tracks attack-induced drift')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_attack_robustness.png", dpi=150)
    plt.savefig(f"{OUT}/fig_attack_robustness.pdf"); plt.close()
    print(f"  saved fig_attack_robustness ({len(df)} rows)")
else:
    print("  attack_results.csv not yet present")

# ============================================================ FIG: sweep summary
print("Building sweep summary figure...")
if os.path.exists("sweep_full.csv"):
    df = pd.read_csv("sweep_full.csv")
    configs = df['name'].unique()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = plt.cm.viridis(np.linspace(0, 0.85, len(configs)))
    for ax, metric, title, ylim in zip(
        axes,
        ['pair_acc', 'pair_sep', 'wout_rho'],
        ['Pair accuracy', 'Pair separation', r'$\rho(\|W_{out}\|_F, k)$'],
        [(0, 1.05), None, (-1.05, 1.05)]):
        for c, col in zip(configs, palette):
            sub = df[df['name']==c]
            for s in sub['seed'].unique():
                ss = sub[sub['seed']==s].sort_values('epoch')
                # epoch 0 is random init; start at epoch 1 for log axis
                xs = ss['epoch'].clip(lower=0.5)
                ax.plot(xs, ss[metric], color=col, alpha=0.6, lw=1.3, marker='.')
            ax.plot([], [], color=col, label=c)
        ax.set_xscale('log')
        ax.set_xlabel('Training epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, which='both')
        if metric=='pair_acc':
            ax.axhline(1.0, color='red', ls=':', alpha=0.5)
        if metric=='wout_rho':
            ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.legend(loc='best', fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_sweep_summary.png", dpi=150)
    plt.savefig(f"{OUT}/fig_sweep_summary.pdf"); plt.close()
    print(f"  saved fig_sweep_summary ({len(df)} rows, {len(configs)} configs)")

print("\nDone.")
