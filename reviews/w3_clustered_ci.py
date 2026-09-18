#!/usr/bin/env python3
"""
Root-clustered (dyadic) confidence intervals for W3 (reviewer u7g4).

The reviewer's point: the 45-159 "pairs" reuse only 2-3 roots, so a pair-level
i.i.d. bootstrap (resampling checkpoint pairs) understates uncertainty. The
correct interval resamples ROOTS with replacement and rebuilds the pair set from
the resampled roots (dyadic / cluster bootstrap). This script reports the
root-clustered CI alongside the pair-level i.i.d. CI so the width gap is explicit.

Benchmarks:
  gpt2_8root / gpt2_19root : reads a GPT-2 benchmark scores_by_pair.csv, uses the
      `centered_with_gating` column (Eq. 6). Positives = descendants, negatives =
      independent + distilled. CI is on AUROC.
  pythia : reads results/pythia_seed_suite.json (null-only). No positives, so
      AUROC is undefined; CI is on the max and mean independent-pair score.

Usage:
  python w3_clustered_ci.py --benchmark gpt2_8root
  python w3_clustered_ci.py --benchmark pythia
  python w3_clustered_ci.py --benchmark gpt2_19root --csv <path/to/scores_by_pair.csv>
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRIPTS = REPO / "experiments" / "scripts"

DEFAULT_CSV = {
    "gpt2_8root": SCRIPTS / "results/lineage_benchmark_gpt2_paper_v2/laundering_NONE_v2/"
                  "lineage_benchmark_gpt2_paper_v2/laundering_NONE_v2/scores_by_pair.csv",
    # run_benchmark.py --phase evaluate writes benchmark_results.json (pairs w/ "lineage");
    # the older laundering harness wrote scores_by_pair.csv. Loader below handles both.
    "gpt2_19root": SCRIPTS / "results/lineage_benchmark_gpt2_19root/benchmark_results.json",
}
PYTHIA_JSON = SCRIPTS / "results/pythia_seed_suite.json"


def auroc(pos, neg):
    """AUROC via Mann-Whitney U (0.5 credit for ties)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += (p > neg).sum() + 0.5 * (p == neg).sum()
    return wins / (len(pos) * len(neg))


def ci(vals, lo=2.5, hi=97.5):
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


# ---------------------------------------------------------------------------
# GPT-2: AUROC, pair-level i.i.d. CI, root-clustered (dyadic) CI
# ---------------------------------------------------------------------------
def root_idx(name: str) -> int:
    m = re.match(r"root_?(\d+)", name)
    return int(m.group(1))


def load_gpt2(path: Path, score_col: str):
    """Load pairs from either a scores_by_pair.csv (laundering harness, uses
    `score_col`) or a benchmark_results.json (run_benchmark --phase evaluate,
    uses the gated `lineage` score). Both encode independent pairs as
    reference=root_i, suspect=root_j."""
    rows = []
    if str(path).endswith(".json"):
        data = json.loads(Path(path).read_text())
        for p in data["pairs"]:
            ref = root_idx(p["reference"])
            sus = root_idx(p["suspect"]) if p["attack_type"] == "independent" else ref
            rows.append({
                "ref": ref, "sus": sus,
                "pos": p["label"] == "descendant",
                "attack": p["attack_type"],
                "score": float(p["lineage"]),
            })
    else:
        with open(path) as f:
            for r in csv.DictReader(f):
                ref = root_idx(r["ref_id"])
                sus = root_idx(r["sus_id"]) if r["attack_type"] == "independent" else ref
                rows.append({
                    "ref": ref, "sus": sus,
                    "pos": r["label"] == "descendant",
                    "attack": r["attack_type"],
                    "score": float(r[score_col]),
                })
    return rows


def gpt2(csv_path: Path, score_col: str, n_boot: int, seed: int):
    rows = load_gpt2(csv_path, score_col)
    roots = sorted({r["ref"] for r in rows} | {r["sus"] for r in rows})
    pos = [r["score"] for r in rows if r["pos"]]
    neg = [r["score"] for r in rows if not r["pos"]]
    distilled = [r["score"] for r in rows if r["attack"] == "distilled_student"]

    point = auroc(pos, neg)
    gap_z = (np.mean(pos) - np.mean(distilled)) / (np.std(distilled, ddof=0) + 1e-12) if distilled else float("nan")

    rng = np.random.RandomState(seed)
    # pair-level i.i.d. bootstrap (resample checkpoint pairs)
    pair_boot = []
    p, n = np.asarray(pos), np.asarray(neg)
    for _ in range(n_boot):
        pair_boot.append(auroc(rng.choice(p, len(p), replace=True),
                               rng.choice(n, len(n), replace=True)))

    # root-clustered (dyadic) bootstrap: resample roots, rebuild pairs by multiplicity
    pos_by_root = {t: [r["score"] for r in rows if r["pos"] and r["ref"] == t] for t in roots}
    dist_by_root = {t: [r["score"] for r in rows if r["attack"] == "distilled_student" and r["ref"] == t] for t in roots}
    indep = {}  # (i,j) i<j -> score
    for r in rows:
        if r["attack"] == "independent":
            i, j = sorted((r["ref"], r["sus"]))
            indep[(i, j)] = r["score"]

    clus_boot = []
    R = len(roots)
    for _ in range(n_boot):
        draw = rng.choice(roots, size=R, replace=True)
        m = {t: int((draw == t).sum()) for t in roots}
        bpos, bneg = [], []
        for t in roots:
            if m[t]:
                bpos += pos_by_root[t] * m[t]        # descendants of t
                bneg += dist_by_root[t] * m[t]       # distilled negs of t
        for (i, j), s in indep.items():              # independent pair weight m_i * m_j
            w = m[i] * m[j]
            if w:
                bneg += [s] * w
        clus_boot.append(auroc(bpos, bneg))

    return {
        "benchmark": csv_path.parent.name,
        "n_roots": len(roots),
        "n_pos": len(pos), "n_neg": len(neg),
        "auroc": point,
        "max_negative": float(np.max(neg)),
        "min_positive": float(np.min(pos)),
        "gap_z_distilled": float(gap_z),
        "ci_pair_level": ci(pair_boot),
        "ci_root_clustered": ci(clus_boot),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Pythia: null-only -> CI on max / mean independent-pair score
# ---------------------------------------------------------------------------
FULLY_INDEPENDENT = ["pythia-160m"] + [f"pythia-160m-seed{i}" for i in range(1, 10)]


def pythia(n_boot: int, seed: int, subset: str = "independent"):
    data = json.loads(PYTHIA_JSON.read_text())
    all_roots = [r["tag"] for r in data["roots"]]
    if subset == "independent":
        roots = [t for t in FULLY_INDEPENDENT if t in all_roots]
    else:
        roots = all_roots
    rootset = set(roots)
    pair = {}
    for pr in data["pairs"]:
        if pr["a"] not in rootset or pr["b"] not in rootset:
            continue
        i, j = sorted((pr["a"], pr["b"]))
        pair[(i, j)] = pr["L"]
    scores = np.array(list(pair.values()))
    point_max, point_mean = float(scores.max()), float(scores.mean())

    rng = np.random.RandomState(seed)
    # pair-level i.i.d. bootstrap (resample pair scores)
    pl_max, pl_mean = [], []
    for _ in range(n_boot):
        s = rng.choice(scores, len(scores), replace=True)
        pl_max.append(s.max()); pl_mean.append(s.mean())

    # root-clustered (dyadic) bootstrap: resample roots, rebuild pairs by multiplicity
    R = len(roots)
    rc_max, rc_mean = [], []
    for _ in range(n_boot):
        draw = rng.choice(roots, size=R, replace=True)
        m = {t: int((draw == t).sum()) for t in roots}
        vals = []
        for (i, j), s in pair.items():
            w = m[i] * m[j]
            if w:
                vals += [s] * w
        if vals:
            vals = np.asarray(vals)
            rc_max.append(vals.max()); rc_mean.append(vals.mean())
    return {
        "benchmark": f"pythia-160m seed suite (null-only, subset={subset})",
        "roots_used": roots,
        "n_roots": len(roots),
        "n_pairs": len(pair),
        "auroc": "N/A (no descendants / positives)",
        "max_null": point_max,
        "mean_null": point_mean,
        "conformal_p_floor": 1.0 / (len(roots) + 1),
        "max_null_ci_pair_level": ci(pl_max),
        "max_null_ci_root_clustered": ci(rc_max),
        "mean_null_ci_pair_level": ci(pl_mean),
        "mean_null_ci_root_clustered": ci(rc_mean),
        "n_boot": n_boot,
    }


def fmt_ci(t):
    return f"[{t[0]:.4f}, {t[1]:.4f}]  (width {t[1]-t[0]:.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["gpt2_8root", "gpt2_19root", "pythia"])
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--score-col", default="centered_with_gating")
    ap.add_argument("--subset", choices=["independent", "all"], default="independent",
                    help="pythia only: 'independent' = main+seed1-9; 'all' = 22 (incl. partial-independence variants)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.benchmark == "pythia":
        res = pythia(args.n_boot, args.seed, args.subset)
    else:
        csv_path = args.csv or DEFAULT_CSV[args.benchmark]
        if not Path(csv_path).exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        res = gpt2(Path(csv_path), args.score_col, args.n_boot, args.seed)

    suffix = f"_{args.subset}" if args.benchmark == "pythia" else ""
    out = HERE / f"w3_ci_{args.benchmark}{suffix}.json"
    out.write_text(json.dumps(res, indent=2, default=str))

    print("=" * 68)
    print(f"Root-clustered CI  |  benchmark = {args.benchmark}")
    print("=" * 68)
    for k, v in res.items():
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, float) for x in v):
            print(f"  {k:28s}: {fmt_ci(v)}")
        else:
            print(f"  {k:28s}: {v}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
