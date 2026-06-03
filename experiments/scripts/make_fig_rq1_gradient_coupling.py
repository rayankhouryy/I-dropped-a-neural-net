"""Build the RQ1 Section 9 gradient-coupling figure (3 panels).

Reads experiments/results/rq1_gradient_coupling_{a,b,c}.json and produces
fig_rq1_gradient_coupling.{pdf,png}.

Panels:
  A. Part A — gradient diagonality (g_diag) vs weight diagonality (s) over
     training. Shows that gradients stay flat (~0.15) while weights climb to
     ~4.0; r = -0.12. Counters the "diagonal gradients -> diagonal weights"
     story.
  B. Part B — control vs shuffled-gradient trajectory across epochs {0,5,10,25,50}.
     Shows shuffling cuts the fingerprint by 68% (3.90 -> 1.24) while training
     still proceeds.
  C. Part C — synthetic diagonal-injection trajectory for eps in {0, 0.01, 0.1}.
     Shows that adding diagonal structure directly drives s from 0.08 to 7.57
     with no backprop.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("experiments/results")
OUT_PNG = Path("experiments/figures/fig_rq1_gradient_coupling.png")
OUT_PDF = Path("experiments/figures/fig_rq1_gradient_coupling.pdf")
PAPER_PDF = Path("paper/figures/fig_rq1_gradient_coupling.pdf")


def load_part_a():
    d = json.load(open(RESULTS / "rq1_gradient_coupling_a.json"))
    samples_per_seed = []
    for s in (0, 1, 2):
        samples = d["part_a"][f"seed_{s}"]["gradient_samples"]
        rows = []
        for r in samples:
            ep = r["epoch"]
            step = r.get("step", 0)
            g_diag = r["gradient"]["mean_g_diag"]
            w_s = r["weight"]["mean_s"]
            rows.append((ep, step, g_diag, w_s))
        samples_per_seed.append(rows)
    agg = d["part_a"]["aggregate"]
    return samples_per_seed, agg


def load_part_b():
    d = json.load(open(RESULTS / "rq1_gradient_coupling_b.json"))
    out = {}
    for cond in ("control", "shuffled"):
        per_seed = []
        for s in (0, 1, 2):
            cps = d["part_b"][cond][f"seed_{s}"]["checkpoints"]
            per_seed.append([(c["epoch"], c["aggregate"]["mean_s"]) for c in cps])
        out[cond] = per_seed
    return out, d["part_b"]["aggregate"]


def load_part_c():
    d = json.load(open(RESULTS / "rq1_gradient_coupling_c.json"))
    out = {}
    for eps_key in ("eps_0.0", "eps_0.01", "eps_0.1"):
        per_seed = []
        for s in (0, 1, 2):
            cps = d["part_c"][eps_key][f"seed_{s}"]["checkpoints"]
            per_seed.append([(c.get("step", 0), c["aggregate"]["mean_s"]) for c in cps])
        out[eps_key] = per_seed
    return out, d["part_c"]["aggregate"]


def aggregate(per_seed, key_x=0, key_y=1):
    """Return (xs, mean_ys, std_ys) collapsing across seeds at matching x values."""
    by_x = {}
    for seed in per_seed:
        for row in seed:
            x, y = row[key_x], row[key_y]
            by_x.setdefault(x, []).append(y)
    xs = sorted(by_x.keys())
    means = np.array([np.mean(by_x[x]) for x in xs])
    stds = np.array([np.std(by_x[x]) for x in xs])
    return np.array(xs), means, stds


def panel_a(ax, samples_per_seed, agg):
    # Use seed 0 trajectory for line, but show pooled scatter for context.
    # Aggregate g_diag and weight_s by epoch.
    rows_by_epoch_g = {}
    rows_by_epoch_s = {}
    for seed in samples_per_seed:
        for ep, step, gd, ws in seed:
            rows_by_epoch_g.setdefault(ep, []).append(gd)
            rows_by_epoch_s.setdefault(ep, []).append(ws)
    eps = sorted(rows_by_epoch_g.keys())
    g_mean = np.array([np.mean(rows_by_epoch_g[e]) for e in eps])
    g_std = np.array([np.std(rows_by_epoch_g[e]) for e in eps])
    s_mean = np.array([np.mean(rows_by_epoch_s[e]) for e in eps])
    s_std = np.array([np.std(rows_by_epoch_s[e]) for e in eps])

    ax.plot(eps, g_mean, marker="o", color="#1f77b4",
            label=r"gradient $g_{\mathrm{diag}}$")
    ax.fill_between(eps, g_mean - g_std, g_mean + g_std, alpha=0.2, color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(eps, s_mean, marker="s", color="#d62728",
             label=r"weight $s = |\mathrm{tr}(M)|/\|M\|_F$")
    ax2.fill_between(eps, s_mean - s_std, s_mean + s_std, alpha=0.2, color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$g_{\mathrm{diag}}$", color="#1f77b4")
    ax2.set_ylabel(r"$s$", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("A. Gradients stay flat while weights climb\n"
                 r"Pearson $r(g_{\mathrm{diag}}, s) = "
                 f"{agg['mean_correlation']:.2f} \pm {agg['std_correlation']:.2f}$",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8, framealpha=0.92, ncol=2)


def panel_b(ax, part_b, agg):
    for cond, color, marker in [("control", "#2ca02c", "o"),
                                 ("shuffled", "#ff7f0e", "s")]:
        xs, ys, stds = aggregate(part_b[cond])
        ax.errorbar(xs, ys, yerr=stds, marker=marker, color=color,
                    label=f"{cond} (final $s$={agg[cond]['mean_s']:.2f})",
                    capsize=3, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Mean diagonal dominance $s$")
    drop = 1 - agg["shuffled"]["mean_s"] / agg["control"]["mean_s"]
    ax.set_title(f"B. Shuffling $\\nabla W_{{out}}$ cuts fingerprint by {drop*100:.0f}%\n"
                 "Independent updates break the coupling",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def panel_c(ax, part_c, agg):
    colors = {"eps_0.0": "#7f7f7f", "eps_0.01": "#9467bd", "eps_0.1": "#e377c2"}
    markers = {"eps_0.0": "x", "eps_0.01": "o", "eps_0.1": "^"}
    labels = {
        "eps_0.0":  r"$\varepsilon = 0$ (control, noise only)",
        "eps_0.01": r"$\varepsilon = 0.01$",
        "eps_0.1":  r"$\varepsilon = 0.1$",
    }
    for eps_key, per_seed in part_c.items():
        xs, ys, stds = aggregate(per_seed)
        ax.errorbar(xs, ys, yerr=stds, marker=markers[eps_key],
                    color=colors[eps_key],
                    label=labels[eps_key] + f"  (final $s$={agg[eps_key]['mean_s']:.2f})",
                    capsize=3, linewidth=1.5)
    ax.set_xlabel("Synthetic injection step")
    ax.set_ylabel(r"Mean diagonal dominance $s$")
    ax.set_title("C. Diagonal injection (no backprop) builds the fingerprint\n"
                 "Diagonal structure is sufficient",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.3)


def main():
    samples_per_seed, agg_a = load_part_a()
    part_b, agg_b = load_part_b()
    part_c, agg_c = load_part_c()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panel_a(axes[0], samples_per_seed, agg_a)
    panel_b(axes[1], part_b, agg_b)
    panel_c(axes[2], part_c, agg_c)
    fig.suptitle(
        r"Gradient coupling mechanism (RQ1 §9): "
        r"correlated $\nabla W_{in},\nabla W_{out}$ updates "
        r"build the diagonal fingerprint",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    PAPER_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(PAPER_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {PAPER_PDF}")


if __name__ == "__main__":
    main()
