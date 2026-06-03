"""Build the RQ1 optimizer ablation figure.

Reads results/rq1_optimizer_ablation_cpu.json and produces a 2-panel figure:
  A. Pair accuracy over training, one curve per optimizer (mean over seeds,
     shaded ±std).
  B. Mean diagonal-dominance s over training, same layout.

Highlights the qualitative gap: adaptive optimizers (Adam/AdamW/RMSprop)
form the fingerprint within 1 epoch; vanilla SGD / SGD+momentum at the same
LR fail to develop it within the 30-epoch budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RES = Path("results/rq1_optimizer_ablation_cpu.json")
OUT_PNG = Path("experiments/figures/fig_rq1_optimizer_ablation.png")
OUT_PDF = Path("experiments/figures/fig_rq1_optimizer_ablation.pdf")
PAPER_PDF = Path("paper/figures/fig_rq1_optimizer_ablation.pdf")

LABELS = {
    "sgd": "SGD",
    "sgd_momentum": "SGD+momentum",
    "adam": "Adam",
    "adamw": "AdamW",
    "rmsprop": "RMSprop",
}
COLORS = {
    "sgd": "#7f7f7f",
    "sgd_momentum": "#1f77b4",
    "adam": "#d62728",
    "adamw": "#ff7f0e",
    "rmsprop": "#9467bd",
}


def aggregate(runs, key):
    """Return (epochs, mean, std) across seeds for given history key."""
    epochs = [h["epoch"] for h in runs[0]["history"]]
    arr = np.array([[h[key] for h in r["history"]] for r in runs])
    return np.array(epochs), arr.mean(axis=0), arr.std(axis=0)


def main():
    d = json.load(open(RES))
    runs = d["runs"]
    by_opt: dict = {}
    for r in runs:
        by_opt.setdefault(r["optimizer"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: pair accuracy
    ax = axes[0]
    for name in ("sgd", "sgd_momentum", "adam", "adamw", "rmsprop"):
        if name not in by_opt:
            continue
        x, m, s = aggregate(by_opt[name], "pair_acc")
        ax.plot(x, m, marker="o", markersize=3.5, color=COLORS[name],
                label=LABELS[name], linewidth=1.5)
        ax.fill_between(x, m - s, m + s, alpha=0.18, color=COLORS[name])
    ax.axhline(1.0 / d["config"]["n_blocks"], color="k", linestyle=":",
               alpha=0.5, label=f"chance (1/{d['config']['n_blocks']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pair accuracy")
    ax.set_title("A. Fingerprint emerges only under adaptive optimizers",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.10)
    handles_a, labels_a = ax.get_legend_handles_labels()

    # Panel B: mean diagonal-dominance s
    ax = axes[1]
    for name in ("sgd", "sgd_momentum", "adam", "adamw", "rmsprop"):
        if name not in by_opt:
            continue
        x, m, s = aggregate(by_opt[name], "mean_diag_s")
        ax.plot(x, m, marker="o", markersize=3.5, color=COLORS[name],
                label=LABELS[name], linewidth=1.5)
        ax.fill_between(x, m - s, m + s, alpha=0.18, color=COLORS[name])
    ax.axhline(1.0 / np.sqrt(d["config"]["d"]), color="k", linestyle=":",
               alpha=0.5, label=r"$1/\sqrt{d}$ baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Mean diagonal dominance $s$")
    ax.set_title("B. Diagonal structure climbs only with Adam/AdamW/RMSprop",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    # Append only the sqrt(d) baseline (the 5 optimizer entries are shared with panel A)
    handles_b, labels_b = ax.get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(handles_b, labels_b) if l not in labels_a]
    all_handles = handles_a + [h for h, _ in extra]
    all_labels = labels_a + [l for _, l in extra]

    cfg = d["config"]
    fig.suptitle(
        f"Optimizer ablation (RQ1 §10, Issue #43): {cfg['n_blocks']}-block MLP, "
        f"d={cfg['d']}, lr={cfg['lr']}, CIFAR-10 subset (n={cfg['train_n']}), "
        f"3 seeds",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.legend(all_handles, all_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               fontsize=9, framealpha=0.95, ncol=len(all_labels))
    fig.subplots_adjust(bottom=0.18)

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
