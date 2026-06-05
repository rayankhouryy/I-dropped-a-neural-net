"""Plot AUROC comparison across lineage verification baselines (Issue #44)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "lineage_baselines_mlp.json"
OUT = ROOT / "paper" / "figures" / "fig_lineage_baselines_auroc.png"

DISPLAY = {
    "diagonal_dominance": "Diagonal Dominance (ours)",
    "aligned_frobenius": "Aligned Frobenius",
    "singular_value_dist": "Singular Value Dist.",
    "weight_cosine": "Weight Cosine",
    "cka": "CKA",
    "svcca": "SVCCA",
    "ipguard_regr": "IPGuard (regr.)",
}
DATA_FREE = {"diagonal_dominance", "aligned_frobenius", "singular_value_dist", "weight_cosine"}

with open(RES) as f:
    data = json.load(f)

methods = list(DISPLAY.keys())
labels = [DISPLAY[m] for m in methods]
aurocs = [data["auroc"][m] for m in methods]
colors = ["#1f77b4" if m == "diagonal_dominance" else
          ("#2ca02c" if m in DATA_FREE else "#ff7f0e") for m in methods]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1.4]})

# Panel A: AUROC bars
y = np.arange(len(methods))
ax1.barh(y, aurocs, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_yticks(y)
ax1.set_yticklabels(labels)
ax1.invert_yaxis()
ax1.set_xlabel("AUROC")
ax1.set_xlim(0.4, 1.05)
ax1.axvline(1.0, ls=":", color="gray", lw=0.8)
ax1.set_title(f"(A) AUROC on MLP benchmark (n={data['n_pairs']} pairs)")
for i, v in enumerate(aurocs):
    ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

# Panel B: per-kind means (descendant vs non-descendant separation)
DESC_KINDS = ["fine_tune", "fine_tune_new_target", "noise", "prune", "quantize"]
NONDESC_KINDS = ["distilled", "diff_seed_same_task"]
x = np.arange(len(methods))
desc_means = []
nondesc_means = []
for m in methods:
    pk = data["per_kind"][m]
    d = np.mean([pk[k]["mean"] for k in DESC_KINDS if k in pk])
    nd = np.mean([pk[k]["mean"] for k in NONDESC_KINDS if k in pk])
    desc_means.append(d)
    nondesc_means.append(nd)

# Normalize per method to [0,1] for cross-method visual comparison of separation
desc_n, nondesc_n = [], []
for d, nd in zip(desc_means, nondesc_means):
    lo, hi = min(d, nd), max(d, nd)
    rng = hi - lo if hi != lo else 1.0
    desc_n.append((d - lo) / rng)
    nondesc_n.append((nd - lo) / rng)

w = 0.35
ax2.bar(x - w/2, desc_n, w, label="Descendants (mean)", color="#1f77b4", edgecolor="black", linewidth=0.5)
ax2.bar(x + w/2, nondesc_n, w, label="Non-descendants (mean)", color="#d62728", edgecolor="black", linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("Normalized score (per method)")
ax2.set_title("(B) Descendant vs Non-descendant Separation (normalized)")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
ax2.set_ylim(-0.05, 1.15)

fig.suptitle(
    f"Lineage Verification Baselines — MLP Benchmark "
    f"(depth={data['config']['depth']}, d={data['config']['hidden']}, n={data['n_pairs']})",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"Wrote {OUT}")
