"""Harder lineage benchmark: layer-grafts and linear merges.

Reviewer concern: on the easy 52-pair benchmark, weight cosine ties with
diagonal dominance at AUROC=1.000. The proposed harder regime is partial
provenance: suspects that combine reference blocks with an independent
model's blocks (layer-graft) or blend reference and independent weights
(linear merge). In both regimes a globally-flat cosine should blur, while
block-localized methods (diagonal dominance, aligned Frobenius) should
track the partial-overlap fraction.

Outputs:
    results/lineage_harder_bench.json   (per-pair scores + AUROC + Spearman)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import lineage_detection as ldet  # noqa: E402
import lineage_baselines as lbase  # noqa: E402
from lineage_phase1_mlp import (  # noqa: E402
    ResNet,
    branch_products,
    fresh_model,
    make_data,
    train_model,
)
from lineage_benchmark_mlp import collect_activations, predictions, model_pack, score_pair  # noqa: E402


METHODS = [
    "diagonal_dominance",
    "aligned_frobenius",
    "singular_value_dist",
    "weight_cosine",
    "cka",
    "svcca",
    "ipguard_regr",
]


def layer_graft(ref: ResNet, donor: ResNet, k_from_ref: int, seed: int = 0) -> ResNet:
    """Build a suspect by replacing (L - k_from_ref) blocks of donor with ref's blocks.

    Block indices to copy from ref are chosen deterministically by seed.
    The suspect is a fresh deep-copy so neither ref nor donor are mutated.
    """
    rng = np.random.RandomState(seed)
    L = len(ref.blocks)
    idx_from_ref = sorted(rng.choice(L, size=k_from_ref, replace=False).tolist())
    suspect = copy.deepcopy(donor)
    for i in idx_from_ref:
        # full-block parameter copy (in proj, activation has no params, out proj)
        suspect.blocks[i].load_state_dict(copy.deepcopy(ref.blocks[i].state_dict()))
    return suspect, idx_from_ref


def linear_merge(ref: ResNet, donor: ResNet, w: float) -> ResNet:
    """Suspect parameters = w * ref + (1 - w) * donor, element-wise."""
    suspect = copy.deepcopy(ref)
    sd_ref = ref.state_dict()
    sd_donor = donor.state_dict()
    sd_out = {}
    for key in sd_ref:
        sd_out[key] = w * sd_ref[key].float() + (1.0 - w) * sd_donor[key].float()
    suspect.load_state_dict(sd_out)
    return suspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, default=2)
    ap.add_argument("--n-donors-per-ref", type=int, default=2)
    ap.add_argument("--graft-ks", type=int, nargs="+", default=[0, 4, 8, 12, 16])
    ap.add_argument("--merge-ws", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--in-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/lineage_harder_bench.json")
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    t0 = time.time()
    pairs = []

    for ref_idx in range(args.n_refs):
        target_key = 42 + ref_idx
        X, y = make_data(args.in_dim, n=2000, seed=ref_idx, target_key=target_key)
        Xv, yv = make_data(args.in_dim, n=400, seed=100 + ref_idx, target_key=target_key)

        ref_model = fresh_model(args.depth, args.hidden, args.in_dim, seed=ref_idx)
        train_model(ref_model, X, y, epochs=args.epochs)
        ref = model_pack(ref_model, Xv)
        print(f"[ref {ref_idx}] trained")

        for donor_idx in range(args.n_donors_per_ref):
            donor_seed = 5000 + ref_idx * 100 + donor_idx
            donor_model = fresh_model(args.depth, args.hidden, args.in_dim, seed=donor_seed)
            # Train donor on same task so the merge / graft is non-trivial
            train_model(donor_model, X, y, epochs=args.epochs)
            print(f"  [donor {donor_idx}] trained (seed={donor_seed})")

            # ---- layer grafts: K of L blocks from ref, rest from donor ----
            for K in args.graft_ks:
                suspect_model, idx_from_ref = layer_graft(ref_model, donor_model, K, seed=donor_seed + K)
                sus = model_pack(suspect_model, Xv)
                scores = score_pair(ref, sus, tau_s=0.5)
                pairs.append({
                    "ref": ref_idx, "donor": donor_idx, "regime": "graft",
                    "param": K, "fraction": K / args.depth,
                    "label": 1 if K >= args.depth // 2 else 0,
                    "blocks_from_ref": idx_from_ref,
                    "scores": scores,
                })

            # ---- linear merges: suspect = w * ref + (1 - w) * donor ----
            for w in args.merge_ws:
                suspect_model = linear_merge(ref_model, donor_model, w)
                sus = model_pack(suspect_model, Xv)
                scores = score_pair(ref, sus, tau_s=0.5)
                pairs.append({
                    "ref": ref_idx, "donor": donor_idx, "regime": "merge",
                    "param": w, "fraction": float(w),
                    "label": 1 if w >= 0.5 else 0,
                    "scores": scores,
                })

        print(f"[ref {ref_idx}] done; running total {len(pairs)} pairs")

    # ----- summarise -----
    auroc_table = {"graft": {}, "merge": {}, "all": {}}
    spearman_table = {"graft": {}, "merge": {}}

    for regime in ("graft", "merge"):
        sub = [p for p in pairs if p["regime"] == regime]
        labels = np.array([p["label"] for p in sub])
        fractions = np.array([p["fraction"] for p in sub])
        for m in METHODS:
            s = np.array([p["scores"][m] for p in sub])
            try:
                au = float(roc_auc_score(labels, s))
            except Exception:
                au = float("nan")
            auroc_table[regime][m] = au
            rho, _ = spearmanr(s, fractions)
            spearman_table[regime][m] = float(rho)

    labels_all = np.array([p["label"] for p in pairs])
    for m in METHODS:
        s = np.array([p["scores"][m] for p in pairs])
        try:
            au = float(roc_auc_score(labels_all, s))
        except Exception:
            au = float("nan")
        auroc_table["all"][m] = au

    out = {
        "config": vars(args),
        "n_pairs": len(pairs),
        "auroc": auroc_table,
        "spearman_with_fraction": spearman_table,
        "pairs": pairs,
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out} ({len(pairs)} pairs, {time.time() - t0:.1f}s)")
    print("\nAUROC by regime:")
    for regime, sub in auroc_table.items():
        print(f"  [{regime}]")
        for m, v in sub.items():
            print(f"    {m:24s} {v:.4f}")
    print("\nSpearman vs ref-fraction:")
    for regime, sub in spearman_table.items():
        print(f"  [{regime}]")
        for m, v in sub.items():
            print(f"    {m:24s} {v:+.3f}")


if __name__ == "__main__":
    main()
