"""Issue #43 item 5: deep dive on AUROC=1.000 for lineage detection.

Loads lineage_phase2_resnet18_cifar.json, breaks down by attack, computes
bootstrap CI on AUROC and Clopper-Pearson CIs on per-class TPR/TNR,
and reports the actual separation margin.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = 0.0
    for p in pos:
        wins += (p > neg).sum() + 0.5 * (p == neg).sum()
    return wins / (len(pos) * len(neg))


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    root = Path(__file__).resolve().parents[2]
    src = root / "results" / "lineage_phase2_resnet18_cifar.json"
    d = json.load(open(src))
    records = d["records"]
    summary = d["summary"]

    desc, indep = [], []
    by_attack = {}
    for r in records:
        kind = r.get("kind", "unknown")
        label = r.get("label", "")
        s = r.get("score")
        if s is None:
            continue
        is_desc = label == "descendant"
        (desc if is_desc else indep).append(s)
        by_attack.setdefault(kind, []).append((s, is_desc))

    print("=" * 72)
    print("AUROC=1.000 deep dive (lineage_phase2_resnet18_cifar)")
    print("=" * 72)
    print()
    print("Record schema: {}".format(list(records[0].keys())))
    print()
    print("Per-kind breakdown:")
    for a in sorted(by_attack):
        entries = by_attack[a]
        vals = [s for s, _ in entries]
        is_d = entries[0][1]
        cls = "descendant" if is_d else "non-desc"
        print(
            "  {:25s} [{}] n={:3d}  mean={:.4f}  min={:.4f}  max={:.4f}".format(
                a, cls, len(vals), float(np.mean(vals)), float(np.min(vals)),
                float(np.max(vals))
            )
        )
    print()
    print("Totals: descendants={}, non-descendants={}".format(len(desc), len(indep)))
    print()

    if not desc or not indep:
        print("Cannot compute AUROC without both classes.")
        return

    emp = auroc(desc, indep)
    print("Empirical AUROC:  {:.6f}".format(emp))
    print("Summary AUROC:    {}".format(summary.get("auroc")))
    print("Summary AUPRC:    {}".format(summary.get("auprc")))

    np.random.seed(0)
    B = 5000
    boot = np.array(
        [
            auroc(
                np.random.choice(desc, size=len(desc), replace=True),
                np.random.choice(indep, size=len(indep), replace=True),
            )
            for _ in range(B)
        ]
    )
    print(
        "Bootstrap AUROC 95% CI (B={}): [{:.4f}, {:.4f}]".format(
            B, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        )
    )
    print("Bootstrap median: {:.4f}".format(float(np.median(boot))))
    print(
        "Bootstrap fraction <1.000: {:.1%}".format(float(np.mean(boot < 1.0)))
    )

    tau = summary.get("config", {}).get("tau_s")
    print()
    print("Threshold tau_s from summary: {}".format(tau))
    print()
    print("Per-kind TPR/TNR with Clopper-Pearson 95% CI:")
    for a in sorted(by_attack):
        entries = by_attack[a]
        is_d = entries[0][1]
        vals = [s for s, _ in entries]
        if is_d:
            k = sum(s > tau for s in vals) if tau is not None else 0
            metric = "TPR"
        else:
            k = sum(s <= tau for s in vals) if tau is not None else 0
            metric = "TNR"
        lo, hi = clopper_pearson(k, len(vals))
        print(
            "  {:25s} {}: {}/{} = {:6.1%}  95% CI: [{:.3f}, {:.3f}]".format(
                a, metric, k, len(vals), k / len(vals), lo, hi
            )
        )

    print()
    print("SEPARATION DIAGNOSTIC")
    print(
        "  Min descendant score: {:.4f} | Max non-desc score: {:.4f} | gap: {:.4f}".format(
            float(np.min(desc)), float(np.max(indep)),
            float(np.min(desc) - np.max(indep)),
        )
    )
    if np.min(desc) > np.max(indep):
        print(
            "  -> Classes are PERFECTLY SEPARABLE; AUROC=1.000 is genuine "
            "(no overlap)."
        )
        print(
            "     Reviewer skepticism is about generalization, not about the "
            "current sample: with descendants n={} and non-desc n={}, the "
            "bootstrap CI quantifies sampling variability.".format(
                len(desc), len(indep)
            )
        )
    else:
        print("  -> Classes overlap; AUROC < 1.0 should be possible.")


if __name__ == "__main__":
    main()
