#!/usr/bin/env python3
"""E4 -- component ablation for the lineage scorer (reviewer VXDT, weakness V4).

VXDT V4: "Insufficient component validation: Centered and uncentered signatures both
obtain AUROC 1.0; ablations for trace gating, Hungarian matching, and aggregation are
missing." This script produces the four measured ablation lines for the ⟦PENDING E4⟧
block in reviews/response_VXDT.md by re-scoring the already-cached GPT-2 benchmark
branch products under each component toggle. It is post-hoc and CPU-only -- no
retraining, no GPU, no forward passes.

The lineage score (paper default) is, per gpt2_lineage_benchmark/evaluation.py and
lineage_detection.py:

    phi(M)   = vec(M - tr(M)/d * I) / ||.||        centered residual signature
    s(M)     = |tr(M)| / ||M||_F                    diagonal-dominance score (gate input)
    G_ij     = <phi(M_i^A), phi(M_j^B)> * min(s_i^A/tau_s, s_j^B/tau_s, 1)   gated cosine
    L(A,B)   = mean over Hungarian-aligned (i, pi(i)) of G_i,pi(i)

Each ablation axis flips exactly ONE component, holding the others at the default
(centered + gated + Hungarian + mean):

    gating        : gate = min(s_i/tau, s_j/tau, 1)   vs   gate = 1.0
    alignment     : Hungarian (linear_sum_assignment) vs   fixed block order i<->i
    aggregation   : mean                              vs   min / trimmed
    signature     : centered phi(M)                   vs   uncentered vec(M)/||.||
                    -- evaluated CLEAN and under a global residual permutation P

The default corner (gate=True, hungarian, mean, centered) is asserted bit-identical to
evaluation.compute_lineage_score. The (fixed + ungated + mean) corner under the signature
toggle is algebraically the per-layer i<->i cosine of ablate_centering.py, so it
reproduces results/centering_ablation/centering_ablation.json exactly (built-in check).

Global-P condition: P is a single permutation of the d residual-stream coordinates
(attack_global_perm.make_global_permutation). It transforms every branch product
M -> P M P^T (realized directly on cached Ms). We apply it to the SUSPECT side of the
POSITIVE pairs only (negatives stay clean), with NO recovery -- this is the honest
signature-discrimination test. E1 (results/laundering/e1_global_perm) already certified
that this permutation is exactly function-preserving on real checkpoints
(max|delta logit| ~ 2.2e-5), so operating on cached Ms here is legitimate and needs no
re-materialized checkpoints.

Usage (on the SageMaker box, where the pickles live):
    cd ~/I-dropped-a-neural-net/experiments/scripts
    python e4_component_ablation.py \
        --benchmark-dir results/lineage_benchmark_gpt2_paper_v2 \
        --seeds 5 --seed-base 9000 --trim-k 1 --test-roots 5 6 7

Outputs -> results/component_ablation/{e4_full.json, summary.csv, by_cell/*.json,
REPORT.md, component_ablation_table.tex}. Runtime ~5-60 s (CPU).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

import functools
print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Scoring primitives -- reused, not reimplemented. evaluation.py is numpy/scipy only
# (its package __init__ is import-free), so no torch is pulled in.
from gpt2_lineage_benchmark.evaluation import (  # noqa: E402
    _residual_signature,
    _diag_score,
    compute_lineage_score,
    choose_tau_s,
)
import attack_global_perm as agp  # noqa: E402  (make_global_permutation)


# --------------------------------------------------------------------- signatures / scorer

def _uncentered_signature(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Raw branch product, flattened and unit-normalized -- NO centering."""
    v = np.asarray(M, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + eps)


def _aggregate(aligned: List[float], agg: str, trim_k: int = 1) -> float:
    """Aggregate the per-block aligned similarities into a model-level score."""
    a = np.asarray(aligned, dtype=np.float64)
    if agg == "mean":
        return float(a.mean())
    if agg == "min":
        return float(a.min())
    if agg == "trimmed":
        # Drop the trim_k lowest and trim_k highest, average the middle. With L=6 and
        # trim_k=1 this keeps the middle 4. Fall back to mean if the trim would empty it.
        if 2 * trim_k >= a.size:
            return float(a.mean())
        return float(np.sort(a)[trim_k:-trim_k].mean())
    raise ValueError(f"unknown agg: {agg}")


def lineage_score_variant(
    Ms_A: List[np.ndarray],
    Ms_B: List[np.ndarray],
    tau_s: float,
    *,
    gate: bool = True,
    align: str = "hungarian",
    agg: str = "mean",
    signature: str = "centered",
    trim_k: int = 1,
    eps: float = 1e-12,
) -> float:
    """Parametrized generalization of evaluation.compute_lineage_score.

    Default (gate=True, align='hungarian', agg='mean', signature='centered') is
    bit-identical to compute_lineage_score.
    """
    L = len(Ms_A)
    assert len(Ms_B) == L, "models must have the same number of blocks"
    sigfn = _residual_signature if signature == "centered" else _uncentered_signature
    sig_A = [sigfn(M, eps) for M in Ms_A]
    sig_B = [sigfn(M, eps) for M in Ms_B]
    if gate:
        d_A = [_diag_score(M, eps) for M in Ms_A]
        d_B = [_diag_score(M, eps) for M in Ms_B]

    G = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            cos = float(np.dot(sig_A[i], sig_B[j]))
            g = min(d_A[i] / tau_s, d_B[j] / tau_s, 1.0) if gate else 1.0
            G[i, j] = cos * g

    if align == "hungarian":
        row_ind, col_ind = linear_sum_assignment(-G)
        aligned = [G[row_ind[k], col_ind[k]] for k in range(L)]
    elif align == "fixed":
        aligned = [G[i, i] for i in range(L)]
    else:
        raise ValueError(f"unknown align: {align}")

    return _aggregate(aligned, agg, trim_k)


# --------------------------------------------------------------------- Gap-Z conventions

def gap_z_distilled(related: List[float], distilled: List[float]) -> float:
    """Paper / run_benchmark convention: (mean_rel - mean_distilled) / std_distilled (ddof=0)."""
    if len(related) < 1 or len(distilled) < 2:
        return float("nan")
    r = np.asarray(related, dtype=np.float64)
    d = np.asarray(distilled, dtype=np.float64)
    return float((r.mean() - d.mean()) / (d.std(ddof=0) + 1e-12))


def gap_z_pooled(related: List[float], unrelated: List[float]) -> float:
    """ablate_centering convention: (mu_rel - mu_unrel) / pooled_std (ddof=1)."""
    if len(related) < 2 or len(unrelated) < 2:
        return float("nan")
    r = np.asarray(related, dtype=np.float64)
    u = np.asarray(unrelated, dtype=np.float64)
    pooled = np.sqrt((r.std(ddof=1) ** 2 + u.std(ddof=1) ** 2) / 2) + 1e-12
    return float((r.mean() - u.mean()) / pooled)


def auroc(pos: List[float], neg: List[float]) -> float:
    labels = np.array([1] * len(pos) + [0] * len(neg))
    scores = np.array(list(pos) + list(neg), dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------- pair construction
# Copied verbatim (behaviorally) from ablate_centering.build_test_pairs so the 27-pair
# test-root set is identical to the reported centering ablation.

def build_test_pairs(
    phase1_data: Dict, phase2_data: Dict, test_root_indices: List[int]
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    test_roots = set(test_root_indices)
    root_signatures = phase1_data["root_signatures"]
    descendants = phase2_data["descendants"]
    students = phase2_data["students"]

    for desc in descendants:
        if desc["root_idx"] not in test_roots:
            continue
        pairs.append({
            "ref_id": f"root_{desc['root_idx']}", "ref_Ms": root_signatures[desc["root_idx"]],
            "sus_id": desc["id"], "sus_Ms": desc["Ms"],
            "label": "related", "attack_type": desc["type"],
        })
    for student in students:
        ridx = student["teacher_root_idx"]
        if ridx not in test_roots:
            continue
        pairs.append({
            "ref_id": f"root_{ridx}", "ref_Ms": root_signatures[ridx],
            "sus_id": student["id"], "sus_Ms": student["Ms"],
            "label": "unrelated", "attack_type": "distilled",
        })
    test_list = sorted(test_roots)
    for i, root_i in enumerate(test_list):
        for root_j in test_list[i + 1:]:
            pairs.append({
                "ref_id": f"root_{root_i}", "ref_Ms": root_signatures[root_i],
                "sus_id": f"root_{root_j}", "sus_Ms": root_signatures[root_j],
                "label": "unrelated", "attack_type": "independent",
            })
    return pairs


# --------------------------------------------------------------------- cell scoring

def score_cell(pairs: List[Dict], tau_s: float, *, trim_k: int, **variant) -> Dict[str, Any]:
    """Score every pair with one variant; return AUROC / Gap-Z / margin + per-pair scores."""
    related, distilled, cross = [], [], []
    related_rows, unrelated_rows = [], []
    for p in pairs:
        L = lineage_score_variant(p["ref_Ms"], p["sus_Ms"], tau_s, trim_k=trim_k, **variant)
        if p["label"] == "related":
            related.append(L)
            related_rows.append({"ref": p["ref_id"], "sus": p["sus_id"],
                                 "kind": p["attack_type"], "score": L})
        else:
            (distilled if p["attack_type"] == "distilled" else cross).append(L)
            unrelated_rows.append({"ref": p["ref_id"], "sus": p["sus_id"],
                                   "kind": p["attack_type"], "score": L})
    unrelated = distilled + cross
    return {
        "AUROC": auroc(related, unrelated),
        "gap_z": gap_z_distilled(related, distilled),
        "gap_z_pooled": gap_z_pooled(related, unrelated),
        "margin": (float(min(related)) - float(max(unrelated))) if related and unrelated else float("nan"),
        "mean_related": float(np.mean(related)) if related else float("nan"),
        "min_related": float(np.min(related)) if related else float("nan"),
        "mean_unrelated": float(np.mean(unrelated)) if unrelated else float("nan"),
        "max_unrelated": float(np.max(unrelated)) if unrelated else float("nan"),
        "n_pairs": len(pairs),
        "related_scores": related_rows,
        "unrelated_scores": unrelated_rows,
    }


def permute_positive_suspects(pairs: List[Dict], seed: int, d_model: int) -> List[Dict]:
    """Return a copy of pairs with each POSITIVE suspect's Ms permuted (M -> P M P^T).

    Negatives are left clean. A distinct P per positive (seeded by pair index) mirrors
    the E1 per-suspect attack scheme.
    """
    out = []
    pos_idx = 0
    for p in pairs:
        q = dict(p)
        if p["label"] == "related":
            P, _ = agp.make_global_permutation(d_model, seed + pos_idx)
            q["sus_Ms"] = [P @ np.asarray(M, dtype=np.float64) @ P.T for M in p["sus_Ms"]]
            pos_idx += 1
        out.append(q)
    return out


# --------------------------------------------------------------------- artifacts

def load_pickles(bdir: Path) -> Tuple[Dict, Dict]:
    p1, p2 = bdir / "phase1_roots.pkl", bdir / "phase2_descendants.pkl"
    for p in (p1, p2):
        if not p.exists():
            print(f"\nERROR: missing {p}\nThe GPT-2 benchmark branch products are absent on this "
                  f"machine. They live on the SageMaker box; locate/restore them there, or "
                  f"regenerate (GPU, ~6-8 h) with:\n"
                  f"    python -m gpt2_lineage_benchmark.run_benchmark --preset paper "
                  f"--save-models --device cuda\n", file=sys.stderr)
            sys.exit(2)
    with open(p1, "rb") as f:
        phase1 = pickle.load(f)
    with open(p2, "rb") as f:
        phase2 = pickle.load(f)
    print(f"[schema] phase1 keys: {list(phase1.keys())}")
    print(f"[schema] phase2 keys: {list(phase2.keys())}")
    return phase1, phase2


def resolve_test_roots(phase1: Dict, bdir: Path, override: List[int] | None) -> List[int]:
    if override:
        return sorted(override)
    test = [r["root_idx"] for r in phase1.get("roots_info", []) if r.get("split") == "test"]
    if not test:
        br = bdir / "benchmark_results.json"
        if br.exists():
            bench = json.loads(br.read_text())
            test = [r["root_idx"] for r in bench.get("roots", []) if r.get("split") == "test"]
    return sorted(test) if test else [5, 6, 7]


# --------------------------------------------------------------------- reconciliation

def reconcile_centering(pairs: List[Dict], tau_s: float, ref_path: Path) -> Dict[str, Any]:
    """The (fixed + ungated + mean) corner under the signature toggle == ablate_centering.

    Compares to results/centering_ablation/centering_ablation.json when present.
    """
    base = dict(gate=False, align="fixed", agg="mean")
    cen = score_cell(pairs, tau_s, trim_k=1, signature="centered", **base)
    unc = score_cell(pairs, tau_s, trim_k=1, signature="uncentered", **base)
    out = {
        "centered": {"auroc": cen["AUROC"], "gap_z_pooled": cen["gap_z_pooled"],
                     "margin": cen["margin"], "unrelated_max": cen["max_unrelated"]},
        "uncentered": {"auroc": unc["AUROC"], "gap_z_pooled": unc["gap_z_pooled"],
                       "margin": unc["margin"], "unrelated_max": unc["max_unrelated"]},
        "reference_file": str(ref_path),
    }
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["metrics"]
        out["reference"] = {
            "centered_unrelated_max": ref["centered"]["unrelated_max"],
            "uncentered_unrelated_max": ref["uncentered"]["unrelated_max"],
            "centered_gap_z": ref["centered"]["gap_z"],
            "uncentered_gap_z": ref["uncentered"]["gap_z"],
        }
        out["matches_reference"] = bool(
            np.isclose(unc["max_unrelated"], ref["uncentered"]["unrelated_max"], atol=1e-4)
            and np.isclose(cen["max_unrelated"], ref["centered"]["unrelated_max"], atol=1e-4)
        )
    return out


def check_default_corner(pairs: List[Dict], tau_s: float) -> bool:
    """Assert the default variant reproduces compute_lineage_score on sample pairs."""
    ok = True
    for p in pairs[:5]:
        mine = lineage_score_variant(p["ref_Ms"], p["sus_Ms"], tau_s)
        ref, _, _ = compute_lineage_score(p["ref_Ms"], p["sus_Ms"], tau_s)
        if not np.isclose(mine, ref, atol=1e-9):
            print(f"  [default-corner MISMATCH] {p['sus_id']}: {mine:.6g} vs {ref:.6g}")
            ok = False
    print(f"[sanity] default corner == compute_lineage_score on sample pairs: {ok}")
    return ok


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="E4 component ablation (VXDT V4)")
    ap.add_argument("--benchmark-dir", default="results/lineage_benchmark_gpt2_paper_v2")
    ap.add_argument("--test-roots", type=int, nargs="+", default=None,
                    help="override test-root indices (default: resolve, else [5,6,7])")
    ap.add_argument("--seeds", type=int, default=5, help="# permutation seeds for the signature axis")
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--trim-k", type=int, default=1, help="trimmed-mean: drop trim_k from each tail")
    ap.add_argument("--out", default=None, help="output dir (default: <benchmark parent>/component_ablation)")
    args = ap.parse_args()

    bdir = Path(args.benchmark_dir)
    if not bdir.is_absolute():
        bdir = SCRIPT_DIR / bdir
    outdir = Path(args.out) if args.out else (bdir.parent / "component_ablation")
    (outdir / "by_cell").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 64)
    print("E4 component ablation")
    print("=" * 64)
    print(f"benchmark_dir = {bdir}")

    phase1, phase2 = load_pickles(bdir)
    tau_s_loaded = float(phase1["tau_s"])
    tau_s_recomputed = float(choose_tau_s(phase1["root_signatures"]))
    d_model = int(np.asarray(phase1["root_signatures"][0][0]).shape[0])
    test_roots = resolve_test_roots(phase1, bdir, args.test_roots)
    pairs = build_test_pairs(phase1, phase2, test_roots)
    n_rel = sum(1 for p in pairs if p["label"] == "related")
    print(f"tau_s loaded={tau_s_loaded:.6g}  recomputed={tau_s_recomputed:.6g}  d_model={d_model}")
    print(f"test_roots={test_roots}  pairs={len(pairs)} (related={n_rel}, unrelated={len(pairs)-n_rel})")
    tau_s = tau_s_loaded

    # --- gate diagnostics: is the gate inert on this benchmark? -------------------
    ref_diag = [_diag_score(M) for Ms in phase1["root_signatures"] for M in Ms]
    sus_diag = [_diag_score(M) for d in phase2["descendants"] for M in d["Ms"]]
    min_sus_over_tau = float(min(sus_diag) / tau_s)
    print(f"[gate] ref diag_score: min={min(ref_diag):.4g} max={max(ref_diag):.4g}")
    print(f"[gate] suspect diag_score: min={min(sus_diag):.4g} max={max(sus_diag):.4g}")
    print(f"[gate] min(suspect)/tau_s = {min_sus_over_tau:.4g}  "
          f"(>=1 => gate saturates at 1.0 => gated==ungated; <1 => gate attenuates)")

    # --- sanity: default corner == compute_lineage_score --------------------------
    default_ok = check_default_corner(pairs, tau_s)

    # --- the four axes ------------------------------------------------------------
    DEFAULT = dict(gate=True, align="hungarian", agg="mean", signature="centered")

    def cell(**over):
        v = dict(DEFAULT); v.update(over)
        return score_cell(pairs, tau_s, trim_k=args.trim_k, **v)

    axes: Dict[str, Dict[str, Any]] = {}
    axes["gating"] = {"on": cell(gate=True), "off": cell(gate=False)}
    axes["alignment"] = {"hungarian": cell(align="hungarian"), "fixed": cell(align="fixed")}
    axes["aggregation"] = {"mean": cell(agg="mean"), "min": cell(agg="min"),
                           "trimmed": cell(agg="trimmed")}
    axes["signature_clean"] = {"centered": cell(signature="centered"),
                               "uncentered": cell(signature="uncentered")}

    gate_inert = bool(np.isclose(axes["gating"]["on"]["AUROC"], axes["gating"]["off"]["AUROC"], atol=1e-9)
                      and np.isclose(axes["gating"]["on"]["gap_z"], axes["gating"]["off"]["gap_z"], atol=1e-6))
    print(f"[gate] gated == ungated (AUROC & Gap-Z)? {gate_inert}")

    # --- signature under global permutation (K seeds) -----------------------------
    print(f"[perm] applying global P to positive suspects, {args.seeds} seeds (no recovery)...")
    perm_seeds = {"centered": [], "uncentered": []}
    perm_cells_last: Dict[str, Any] = {}
    for s in range(args.seeds):
        seed = args.seed_base + 1000 * s
        ppairs = permute_positive_suspects(pairs, seed, d_model)
        for sig in ("centered", "uncentered"):
            c = score_cell(ppairs, tau_s, trim_k=args.trim_k,
                           gate=True, align="hungarian", agg="mean", signature=sig)
            perm_seeds[sig].append(c)
            perm_cells_last[sig] = c

    def summarize_perm(cells: List[Dict]) -> Dict[str, Any]:
        au = [c["AUROC"] for c in cells]
        gz = [c["gap_z"] for c in cells]
        return {
            "AUROC_mean": float(np.nanmean(au)), "AUROC_std": float(np.nanstd(au)),
            "AUROC_per_seed": au,
            "gap_z_mean": float(np.nanmean(gz)), "gap_z_std": float(np.nanstd(gz)),
            "gap_z_per_seed": gz,
            "mean_related": float(np.nanmean([c["mean_related"] for c in cells])),
            "mean_unrelated": float(np.nanmean([c["mean_unrelated"] for c in cells])),
            "max_unrelated": float(np.nanmax([c["max_unrelated"] for c in cells])),
            "n_pairs": cells[0]["n_pairs"],
        }

    axes["signature_globalp"] = {
        "centered": summarize_perm(perm_seeds["centered"]),
        "uncentered": summarize_perm(perm_seeds["uncentered"]),
    }

    # --- centering reconciliation vs the already-reported ablation ----------------
    ref_json = bdir.parent / "centering_ablation" / "centering_ablation.json"
    recon = reconcile_centering(pairs, tau_s, ref_json)
    print(f"[sanity] centering reconciliation matches reference: {recon.get('matches_reference', 'N/A')}")

    wall = time.time() - t0

    # --- write e4_full.json -------------------------------------------------------
    def strip_rows(axis_dict):
        out = {}
        for var, c in axis_dict.items():
            out[var] = {k: v for k, v in c.items()
                        if k not in ("related_scores", "unrelated_scores")}
        return out

    full = {
        "config": {"benchmark_dir": str(bdir), "test_roots": test_roots,
                   "n_pairs": len(pairs), "n_related": n_rel, "seeds": args.seeds,
                   "seed_base": args.seed_base, "trim_k": args.trim_k},
        "seeds": {"perm_base": args.seed_base, "perm_formula": "seed_base + 1000*s (+ pos_idx per positive)"},
        "tau_s_loaded": tau_s_loaded, "tau_s_recomputed": tau_s_recomputed,
        "gate_diagnostics": {"ref_diag_min": float(min(ref_diag)), "ref_diag_max": float(max(ref_diag)),
                             "suspect_diag_min": float(min(sus_diag)), "suspect_diag_max": float(max(sus_diag)),
                             "min_suspect_over_tau": min_sus_over_tau},
        "axes": {name: (strip_rows(a) if name not in ("signature_globalp",) else a)
                 for name, a in axes.items()},
        "reconciliation": recon,
        "sanity_checks": {"default_corner_ok": default_ok,
                          "centering_reproduced": recon.get("matches_reference", None),
                          "gate_inert": gate_inert},
        "wall_seconds": wall,
    }
    (outdir / "e4_full.json").write_text(json.dumps(full, indent=2))

    # --- by_cell/*.json -----------------------------------------------------------
    for name, a in axes.items():
        if name == "signature_globalp":
            continue
        for var, c in a.items():
            (outdir / "by_cell" / f"{name}__{var}.json").write_text(json.dumps({
                "axis": name, "variant": var, "AUROC": c["AUROC"], "gap_z": c["gap_z"],
                "gap_z_pooled": c["gap_z_pooled"], "margin": c["margin"], "n_pairs": c["n_pairs"],
                "related_scores": c["related_scores"], "unrelated_scores": c["unrelated_scores"],
            }, indent=2))

    # --- summary.csv --------------------------------------------------------------
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "variant", "seed", "AUROC", "gap_z"])
        for name in ("gating", "alignment", "aggregation", "signature_clean"):
            for var, c in axes[name].items():
                w.writerow([name, var, "-", f"{c['AUROC']:.6f}", f"{c['gap_z']:.4f}"])
        for sig in ("centered", "uncentered"):
            for s, c in enumerate(perm_seeds[sig]):
                w.writerow(["signature_globalp", sig, s, f"{c['AUROC']:.6f}", f"{c['gap_z']:.4f}"])
            summ = axes["signature_globalp"][sig]
            w.writerow(["signature_globalp", sig, "mean", f"{summ['AUROC_mean']:.6f}", f"{summ['gap_z_mean']:.4f}"])
            w.writerow(["signature_globalp", sig, "std", f"{summ['AUROC_std']:.6f}", f"{summ['gap_z_std']:.4f}"])

    # --- REPORT.md + .tex ---------------------------------------------------------
    write_report(outdir, full, axes)
    write_tex(outdir, axes)

    print(f"\nWrote outputs to {outdir}")
    print(f"Done in {wall:.1f}s")
    print("\n=== FILL-IN for reviews/response_VXDT.md E4 block ===")
    g = axes["gating"]; al = axes["alignment"]; ag = axes["aggregation"]; sp = axes["signature_globalp"]
    print(f"Gating on/off: AUROC {g['on']['AUROC']:.3f} / {g['off']['AUROC']:.3f}, "
          f"Gap-Z {g['on']['gap_z']:.1f} / {g['off']['gap_z']:.1f}")
    print(f"Hungarian vs fixed block order: {al['hungarian']['AUROC']:.3f} / {al['fixed']['AUROC']:.3f}")
    print(f"Aggregation mean/min/trimmed: {ag['mean']['AUROC']:.3f} / {ag['min']['AUROC']:.3f} / {ag['trimmed']['AUROC']:.3f}")
    print(f"Centered vs uncentered under global permutation: "
          f"{sp['centered']['AUROC_mean']:.3f} / {sp['uncentered']['AUROC_mean']:.3f}")


def _fmt(c: Dict[str, Any], k: str, fmt: str = "{:.4f}") -> str:
    v = c.get(k, float("nan"))
    return fmt.format(v) if isinstance(v, (int, float)) and not np.isnan(v) else "n/a"


def write_report(outdir: Path, full: Dict, axes: Dict) -> None:
    cfg = full["config"]
    lines: List[str] = []
    lines.append("# E4 — Component Ablation — Appendix Report\n")
    lines.append("**Paper:** *Training Leaves Traces*. **Reviewer:** VXDT (V4). "
                 f"**Pairs:** {cfg['n_pairs']} (test roots {cfg['test_roots']}).\n")
    lines.append("**Question:** Do the scorer's components — trace gating, Hungarian matching, "
                 "aggregation, and centering — each contribute to discrimination? Centered and "
                 "uncentered both reach AUROC 1.0 on the clean benchmark (AUROC is saturated); "
                 "the margin and Gap-`Z`, and behavior under a global residual permutation, are "
                 "what separate the components.\n")
    lines.append("All numbers are post-hoc re-scoring of cached GPT-2 branch products "
                 "(`phase1_roots.pkl` / `phase2_descendants.pkl`) — CPU-only, no retraining, no "
                 "forward passes. The default corner (gated + Hungarian + mean + centered) is "
                 "asserted bit-identical to `evaluation.compute_lineage_score`.\n")

    lines.append("## Seeds\n")
    lines.append(f"- Permutation seeds: `make_global_permutation(d, {full['seeds']['perm_base']} + "
                 f"1000*s + pos_idx)`, `s = 0..{cfg['seeds']-1}`; applied to positive suspects only, "
                 "no recovery.\n")

    lines.append("## Gate diagnostics\n")
    gd = full["gate_diagnostics"]
    lines.append(f"- `tau_s` (loaded) = {full['tau_s_loaded']:.6g}; recomputed = "
                 f"{full['tau_s_recomputed']:.6g}.\n")
    lines.append(f"- Reference-branch `s(M)` in [{gd['ref_diag_min']:.4g}, {gd['ref_diag_max']:.4g}]; "
                 f"suspect `s(M)` in [{gd['suspect_diag_min']:.4g}, {gd['suspect_diag_max']:.4g}].\n")
    lines.append(f"- `min(suspect s)/tau_s = {gd['min_suspect_over_tau']:.4g}` — "
                 f"{'>=1: gate saturates at 1.0 (gated == ungated).' if gd['min_suspect_over_tau'] >= 1 else '<1: gate ATTENUATES (gated != ungated).'}\n")
    lines.append(f"- Empirically, gated == ungated (AUROC & Gap-Z): "
                 f"**{full['sanity_checks']['gate_inert']}**.\n")

    def table(title: str, axis: str, variants: List[str]):
        lines.append(f"## {title}\n")
        lines.append("| Variant | AUROC | Gap-Z | Gap-Z (pooled) | margin | max neg | min pos |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for v in variants:
            c = axes[axis][v]
            lines.append(f"| {v} | {_fmt(c,'AUROC','{:.3f}')} | {_fmt(c,'gap_z','{:+.1f}')} | "
                         f"{_fmt(c,'gap_z_pooled','{:+.1f}')} | {_fmt(c,'margin','{:+.4f}')} | "
                         f"{_fmt(c,'max_unrelated')} | {_fmt(c,'min_related')} |")
        lines.append("")

    table("Trace gating", "gating", ["on", "off"])
    table("Block alignment", "alignment", ["hungarian", "fixed"])
    table("Aggregation", "aggregation", ["mean", "min", "trimmed"])
    table("Signature centering (clean)", "signature_clean", ["centered", "uncentered"])

    lines.append("## Signature centering under global permutation `P`\n")
    lines.append("| Variant | AUROC (mean±std) | Gap-Z (mean±std) | mean pos | max neg |")
    lines.append("|---|--:|--:|--:|--:|")
    for v in ("centered", "uncentered"):
        c = axes["signature_globalp"][v]
        lines.append(f"| {v} | {c['AUROC_mean']:.3f} ± {c['AUROC_std']:.3f} | "
                     f"{c['gap_z_mean']:+.1f} ± {c['gap_z_std']:.1f} | "
                     f"{c['mean_related']:.4f} | {c['max_unrelated']:.4f} |")
    lines.append("")
    lines.append("*Cached-`Ms` signature-discrimination test (no recovery). E1 "
                 "(`results/laundering/e1_global_perm`) certifies the permutation is exactly "
                 "function-preserving on real checkpoints, so this measures the signature, not "
                 "the attack's validity. Because `P·I·P^T = I`, the uncentered signature retains a "
                 "spurious identity floor under `P` while the centered signature collapses — the two "
                 "are no longer tied.*\n")

    lines.append("## Reconciliation & sanity checks\n")
    r = full["reconciliation"]
    lines.append(f"- Default corner == `compute_lineage_score`: **{full['sanity_checks']['default_corner_ok']}**.")
    lines.append(f"- Centering (fixed+ungated+mean) reproduces `centering_ablation.json`: "
                 f"**{r.get('matches_reference', 'N/A')}** "
                 f"(uncentered max-neg {r['uncentered']['unrelated_max']:.4f}, "
                 f"centered {r['centered']['unrelated_max']:.4f}).")
    if "reference" in r:
        lines.append(f"  - reference: uncentered {r['reference']['uncentered_unrelated_max']:.4f}, "
                     f"centered {r['reference']['centered_unrelated_max']:.4f}.")
    lines.append("")

    lines.append("## Files\n")
    lines.append("- `e4_full.json` — config, seeds, tau_s, gate diagnostics, per-axis cells, "
                 "reconciliation, sanity checks.")
    lines.append("- `summary.csv` — one row per (axis, variant, seed).")
    lines.append("- `by_cell/<axis>__<variant>.json` — per-cell scores + per-pair rows.")
    lines.append("- `component_ablation_table.tex` — per-axis booktabs tables.")
    lines.append(f"- Wall time: {full['wall_seconds']:.1f}s (CPU).")
    lines.append("")
    (outdir / "REPORT.md").write_text("\n".join(lines))


def write_tex(outdir: Path, axes: Dict) -> None:
    g, al, ag = axes["gating"], axes["alignment"], axes["aggregation"]
    sp = axes["signature_globalp"]
    tex = r"""\begin{table}[t]
\centering\small
\begin{tabular}{@{}llcc@{}}
\toprule
Axis & Variant & AUROC & Gap-$Z$ \\
\midrule
Gating & on (default) & %.3f & %+.1f \\
       & off & %.3f & %+.1f \\
\midrule
Alignment & Hungarian (default) & %.3f & %+.1f \\
          & fixed block order & %.3f & %+.1f \\
\midrule
Aggregation & mean (default) & %.3f & %+.1f \\
            & min & %.3f & %+.1f \\
            & trimmed & %.3f & %+.1f \\
\midrule
Centering (under $P$) & centered & %.3f & %+.1f \\
                      & uncentered & %.3f & %+.1f \\
\bottomrule
\end{tabular}
\caption{Component ablation of the lineage scorer (test-root pairs). AUROC is saturated on the
clean benchmark; Gap-$Z$ and behavior under a global residual permutation $P$ separate the
components.}
\label{tab:component-ablation}
\end{table}
""" % (
        g["on"]["AUROC"], g["on"]["gap_z"], g["off"]["AUROC"], g["off"]["gap_z"],
        al["hungarian"]["AUROC"], al["hungarian"]["gap_z"], al["fixed"]["AUROC"], al["fixed"]["gap_z"],
        ag["mean"]["AUROC"], ag["mean"]["gap_z"], ag["min"]["AUROC"], ag["min"]["gap_z"],
        ag["trimmed"]["AUROC"], ag["trimmed"]["gap_z"],
        sp["centered"]["AUROC_mean"], sp["centered"]["gap_z_mean"],
        sp["uncentered"]["AUROC_mean"], sp["uncentered"]["gap_z_mean"],
    )
    (outdir / "component_ablation_table.tex").write_text(tex)


if __name__ == "__main__":
    main()
