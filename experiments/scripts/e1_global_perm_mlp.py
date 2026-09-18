"""E1 (MLP half): global residual-stream permutation attack + recovery defense.

Rebuilds the exact 52-pair Table-6 MLP bank once (deterministic, seed 0), then for
each of --seeds permutation seeds evaluates three conditions on all five methods:

    Clean              -- unlaundered (reproduction; seed-independent)
    Global-P           -- one global residual permutation P per descendant
                          (positives only; negatives stay clean)
    Global-P+recovery  -- recover P via row/col-norm descriptors + Hungarian,
                          undo it, then score (applied to ALL pairs; verifier is
                          blind to which suspect is laundered)

Methods (apples-to-apples with paper Table 5 `tab:laundering`, columns
Ours | Re-Basin | Al.Frob | SVD | W.Cos):
    ours                    = lineage_detection.lineage_score            (on M)
    rebasin_scale           = laundering_baselines_raw.rebasin_scale_frobenius (raw)
    raw_aligned_frobenius   = laundering_baselines_raw.raw_aligned_frobenius   (raw)
    raw_singular_value_dist = laundering_baselines_raw.raw_singular_value_dist (raw)
    raw_weight_cosine       = laundering_baselines_raw.raw_weight_cosine       (raw)

The global P transforms every branch product M_l -> P M_l P^T (unlike the
intra-block hidden-unit P already in the paper, which leaves M invariant). Each
laundered descendant passes the hard function-preservation gate (perm-aware,
< 1e-4). CPU, deterministic, ~a few minutes. Writes results/laundering/e1_global_perm/mlp/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import attack_global_perm as agp  # noqa: E402
import lineage_detection as ldet  # noqa: E402
import laundering_baselines_raw as lraw  # noqa: E402
import laundering_ops as lops  # noqa: E402
from lineage_phase1_mlp import branch_products  # noqa: E402
from laundering_benchmark_mlp import build_bank  # noqa: E402

TAU_S = 0.5
CONDITIONS = ["clean", "global_p", "global_p_recovery"]
METHODS = [
    "ours",
    "rebasin_scale",
    "raw_aligned_frobenius",
    "raw_singular_value_dist",
    "raw_weight_cosine",
]


# ---------------------------------------------------------------- scorers

def _score(method: str, ref, sus) -> float:
    """ref/sus are {'Ms', 'raw'} numpy bundles."""
    if method == "ours":
        return ldet.lineage_score(ref["Ms"], sus["Ms"], TAU_S)[0]
    if method == "rebasin_scale":
        return lraw.rebasin_scale_frobenius(ref["raw"], sus["raw"])
    if method == "raw_aligned_frobenius":
        return lraw.raw_aligned_frobenius(ref["raw"], sus["raw"])
    if method == "raw_singular_value_dist":
        return lraw.raw_singular_value_dist(ref["raw"], sus["raw"])
    if method == "raw_weight_cosine":
        return lraw.raw_weight_cosine(ref["raw"], sus["raw"])
    raise ValueError(method)


def _score_all(ref, sus) -> dict:
    return {m: _score(m, ref, sus) for m in METHODS}


def _mk_pack(Ms, raw) -> dict:
    return {"Ms": Ms, "raw": raw}


def _recover_and_align(ref_pack, sus_pack) -> dict:
    """Blind recovery: recover P from branch products, undo it on M + raw."""
    col_ind = agp.recover_permutation(ref_pack["Ms"], sus_pack["Ms"])
    Ms_al = agp.align_Ms(sus_pack["Ms"], col_ind)
    raw_al = agp.permute_residual_raw_mlp(sus_pack["raw"], col_ind)
    return _mk_pack(Ms_al, raw_al)


# ---------------------------------------------------------------- aggregate

def _aggregate(related, unrelated) -> dict:
    labels = np.array([1] * len(related) + [0] * len(unrelated))
    out = {}
    for m in METHODS:
        rel = np.array([r[m] for r in related], dtype=float)
        unr = np.array([u[m] for u in unrelated], dtype=float)
        scores = np.concatenate([rel, unr])
        try:
            au = float(roc_auc_score(labels, scores))
        except Exception:
            au = float("nan")
        out[m] = {
            "AUROC": au,
            "mean_related": float(rel.mean()), "min_related": float(rel.min()),
            "mean_unrelated": float(unr.mean()), "max_unrelated": float(unr.max()),
            "n_pairs": int(len(scores)),
        }
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--n-refs", type=int, default=2)
    ap.add_argument("--n-per-descendant-type", type=int, default=3)
    ap.add_argument("--n-same-arch-diff-seed", type=int, default=8)
    ap.add_argument("--n-distilled", type=int, default=3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--in-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=30)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/laundering/e1_global_perm/mlp"))
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.out)
    (outdir / "by_cell").mkdir(parents=True, exist_ok=True)

    # ---- build the 52-pair bank once (deterministic seed 0) ----
    bank_args = SimpleNamespace(
        n_refs=args.n_refs, n_per_descendant_type=args.n_per_descendant_type,
        n_same_arch_diff_seed=args.n_same_arch_diff_seed, n_distilled=args.n_distilled,
        depth=args.depth, hidden=args.hidden, in_dim=args.in_dim,
        epochs=args.epochs, ft_epochs=args.ft_epochs,
    )
    print(f"Building 52-pair MLP bank (seed 0)...", flush=True)
    refs, descendants, unrelated = build_bank(bank_args)
    print(f"Bank: {len(descendants)} related, {len(unrelated)} unrelated "
          f"({time.time()-t0:.1f}s)", flush=True)

    probes = lops.make_probes(args.in_dim, n=lops.N_PROBES, seed=12345)

    # Reference packs (Ms + raw) per ref.
    ref_packs = {ri: _mk_pack(refs[ri]["bundle"]["Ms"], refs[ri]["bundle"]["raw"])
                 for ri in refs}
    # Clean descendant / unrelated packs (seed-independent).
    clean_desc = [{"ref": d["ref"], "kind": d["kind"],
                   "pack": _mk_pack(branch_products(d["model"]),
                                    lops.raw_weights(d["model"]))}
                  for d in descendants]
    clean_unrel = [{"ref": u["ref"], "kind": u["kind"],
                    "pack": _mk_pack(u["bundle"]["Ms"], u["bundle"]["raw"])}
                   for u in unrelated]

    # ---- Clean condition (seed-independent; computed once) ----
    clean_related = [_score_all(ref_packs[d["ref"]], d["pack"]) for d in clean_desc]
    clean_unrelated = [_score_all(ref_packs[u["ref"]], u["pack"]) for u in clean_unrel]
    clean_agg = _aggregate(clean_related, clean_unrelated)

    # ---- per-seed Global-P and Global-P+recovery ----
    per_seed = {c: {m: [] for m in METHODS} for c in CONDITIONS}
    for m in METHODS:  # clean identical across seeds
        per_seed["clean"][m] = [clean_agg[m]["AUROC"]] * args.seeds

    gate_devs = []
    recovery_exact_fracs = []       # per (seed, descendant)
    n_desc_exact = []               # per seed: count with fraction == 1.0
    # keep the last-seed cells for by_cell serialization (representative)
    cells = {c: {} for c in CONDITIONS}
    cells["clean"] = {"related": clean_related, "unrelated": clean_unrelated}

    for s in range(args.seeds):
        gp_related, gpr_related = [], []
        exact_this_seed = []
        for i, d in enumerate(clean_desc):
            model = descendants[i]["model"]
            P, perm = agp.make_global_permutation(args.in_dim,
                                                  args.seed_base + 1000 * s + i)
            pmodel = agp.global_residual_permutation_mlp(model, P)
            dev = agp.function_deviation_global_perm_mlp(pmodel, model, P, probes)
            gate_devs.append(dev)
            if not (dev < lops.GATE_THRESHOLD):
                raise AssertionError(
                    f"GATE FAIL seed={s} desc#{i} kind={d['kind']} dev={dev:.3e}")
            att = _mk_pack(branch_products(pmodel), lops.raw_weights(pmodel))
            ref = ref_packs[d["ref"]]
            gp_related.append(_score_all(ref, att))
            # recovery
            col_ind = agp.recover_permutation(ref["Ms"], att["Ms"])
            frac = agp.exact_recovery_fraction(col_ind, perm)
            exact_this_seed.append(frac)
            recovery_exact_fracs.append(frac)
            rec = _mk_pack(agp.align_Ms(att["Ms"], col_ind),
                           agp.permute_residual_raw_mlp(att["raw"], col_ind))
            gpr_related.append(_score_all(ref, rec))
        # negatives: Global-P leaves them clean; recovery runs blindly on them.
        gp_unrelated = clean_unrelated
        gpr_unrelated = [_score_all(ref_packs[u["ref"]],
                                    _recover_and_align(ref_packs[u["ref"]], u["pack"]))
                         for u in clean_unrel]

        gp_agg = _aggregate(gp_related, gp_unrelated)
        gpr_agg = _aggregate(gpr_related, gpr_unrelated)
        for m in METHODS:
            per_seed["global_p"][m].append(gp_agg[m]["AUROC"])
            per_seed["global_p_recovery"][m].append(gpr_agg[m]["AUROC"])
        n_desc_exact.append(int(sum(1 for f in exact_this_seed if f == 1.0)))
        if s == args.seeds - 1:  # representative cells
            cells["global_p"] = {"related": gp_related, "unrelated": gp_unrelated}
            cells["global_p_recovery"] = {"related": gpr_related, "unrelated": gpr_unrelated}
        print(f"[seed {s}] gate_max={max(gate_devs):.2e}  "
              f"ours GP={gp_agg['ours']['AUROC']:.3f} "
              f"GP+rec={gpr_agg['ours']['AUROC']:.3f}  "
              f"exact_desc={n_desc_exact[-1]}/{len(clean_desc)}", flush=True)

    # ---- summarize mean/std over seeds ----
    def mean_std(vals):
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    summary = {c: {m: dict(zip(("mean", "std"), mean_std(per_seed[c][m])))
                   for m in METHODS} for c in CONDITIONS}

    # ---- write by_cell (representative: clean + last-seed GP/recovery) ----
    kinds_desc = [d["kind"] for d in clean_desc]
    refs_desc = [d["ref"] for d in clean_desc]
    kinds_unrel = [u["kind"] for u in clean_unrel]
    refs_unrel = [u["ref"] for u in clean_unrel]
    for c in CONDITIONS:
        rel = cells[c]["related"]
        unr = cells[c]["unrelated"]
        agg = _aggregate(rel, unr)
        for m in METHODS:
            cell = {
                "variant": c, "method": m, **agg[m],
                "AUROC_per_seed": per_seed[c][m],
                "AUROC_mean": summary[c][m]["mean"], "AUROC_std": summary[c][m]["std"],
                "related_scores": [{"ref": refs_desc[k], "kind": kinds_desc[k],
                                    "score": rel[k][m]} for k in range(len(rel))],
                "unrelated_scores": [{"ref": refs_unrel[k], "kind": kinds_unrel[k],
                                      "score": unr[k][m]} for k in range(len(unr))],
            }
            (outdir / "by_cell" / f"{c}__{m}.json").write_text(json.dumps(cell, indent=2))

    # ---- summary.csv (seed column + mean row) ----
    with (outdir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "seed", "AUROC"])
        for m in METHODS:
            for c in CONDITIONS:
                for s in range(args.seeds):
                    w.writerow([m, c, s, f"{per_seed[c][m][s]:.6f}"])
                w.writerow([m, c, "mean", f"{summary[c][m]['mean']:.6f}"])
                w.writerow([m, c, "std", f"{summary[c][m]['std']:.6f}"])

    # ---- full JSON ----
    full = {
        "config": vars(args),
        "benchmark": "MLP 52-pair (depth=16, hidden=48, in_dim=16)",
        "seeds": {"bank": 0, "perm_base": args.seed_base,
                  "perm_formula": "seed_base + 1000*s + desc_idx", "probes": 12345},
        "conditions": CONDITIONS, "methods": METHODS, "tau_s": TAU_S,
        "gate_threshold": lops.GATE_THRESHOLD,
        "gate_max_deviation": float(max(gate_devs)),
        "gate_mean_deviation": float(np.mean(gate_devs)),
        "auroc_per_seed": per_seed,
        "auroc_mean_std": summary,
        "recovery": {
            "exact_coord_fraction_mean": float(np.mean(recovery_exact_fracs)),
            "exact_coord_fraction_min": float(np.min(recovery_exact_fracs)),
            "n_descendants": len(clean_desc),
            "n_desc_exactly_recovered_per_seed": n_desc_exact,
        },
        "wall_seconds": time.time() - t0,
    }
    (outdir / "e1_full.json").write_text(json.dumps(full, indent=2))

    # ---- console table ----
    print(f"\nWrote {outdir}/ ({time.time()-t0:.1f}s)\n")
    print("AUROC (mean over seeds; rows=methods, cols=conditions):")
    print("  " + f"{'method':26s}" + "".join(f"{c:>20s}" for c in CONDITIONS))
    for m in METHODS:
        row = f"  {m:26s}"
        for c in CONDITIONS:
            row += f"{summary[c][m]['mean']:12.3f}±{summary[c][m]['std']:.3f}   "
        print(row)
    print(f"\nGate max deviation: {max(gate_devs):.3e} "
          f"({'PASS' if max(gate_devs) < lops.GATE_THRESHOLD else 'FAIL'})")
    print(f"Recovery: exact-coord fraction mean={np.mean(recovery_exact_fracs):.4f}, "
          f"descendants exactly recovered per seed={n_desc_exact}/{len(clean_desc)}")


if __name__ == "__main__":
    main()
