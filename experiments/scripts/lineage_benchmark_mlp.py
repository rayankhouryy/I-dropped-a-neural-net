"""Run all lineage baselines on the synthetic MLP benchmark (issue #44).

Trains a small bank of reference MLPs plus a battery of descendants and
non-descendants, then scores each (ref, suspect) pair under:

    - diagonal_dominance (ours)
    - aligned_frobenius
    - singular_value_distance
    - weight_cosine
    - linear_cka (per-layer activations, Hungarian-aligned)
    - svcca (per-layer activations, Hungarian-aligned)
    - ipguard_regr (output-rank agreement -- regression analogue)

Outputs:
    results/lineage_baselines_mlp.json   (per-pair scores + AUROC table)

Designed to run on CPU in under 10 minutes with default settings; reuses
the train_model / make_data / descendant_* helpers from lineage_phase1_mlp
so the protocol is byte-for-byte identical to our validated harness.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import lineage_detection as ldet  # noqa: E402
import lineage_baselines as lbase  # noqa: E402
from lineage_phase1_mlp import (  # noqa: E402
    ResNet,
    branch_products,
    descendant_fine_tune,
    descendant_finetune_new_target,
    descendant_noise,
    descendant_prune,
    descendant_quantize,
    eval_loss,
    fresh_model,
    make_data,
    nondesc_distilled_student,
    train_model,
)


# --------------------------------------------------------------- activation hooks

def collect_activations(model: ResNet, X: torch.Tensor) -> list[np.ndarray]:
    """Return per-block input activations (length L).

    We hook the *input* to each Block; that is the residual stream snapshot
    the block sees, which is what CKA/SVCCA care about. Sample size is
    capped at 1024 to keep CKA's O(n^2) Gram matrix cheap.
    """
    n = min(X.shape[0], 1024)
    Xs = X[:n]
    acts: list[np.ndarray] = []
    hooks = []
    for blk in model.blocks:
        def hook(_m, inp, _out, sink=acts):
            sink.append(inp[0].detach().to(torch.float32).cpu().numpy())
        hooks.append(blk.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        _ = model(Xs)
    for h in hooks:
        h.remove()
    return acts


def predictions(model: ResNet, X: torch.Tensor) -> np.ndarray:
    n = min(X.shape[0], 2048)
    model.eval()
    with torch.no_grad():
        y = model(X[:n]).detach().to(torch.float32).cpu().numpy()
    return y


# ----------------------------------------------------------------- model packs

def model_pack(model: ResNet, X: torch.Tensor) -> dict:
    return {
        "model": model,
        "Ms": branch_products(model),
        "acts": collect_activations(model, X),
        "preds": predictions(model, X),
    }


# ------------------------------------------------------------------------ main

METHODS = [
    "diagonal_dominance",
    "aligned_frobenius",
    "singular_value_dist",
    "weight_cosine",
    "cka",
    "svcca",
    "ipguard_regr",
]


def score_pair(ref: dict, sus: dict, tau_s: float) -> dict:
    out = {}
    out["diagonal_dominance"], _, _ = ldet.lineage_score(ref["Ms"], sus["Ms"], tau_s)
    out["aligned_frobenius"] = lbase.aligned_frobenius(ref["Ms"], sus["Ms"])
    out["singular_value_dist"] = lbase.singular_value_distance(ref["Ms"], sus["Ms"])
    out["weight_cosine"] = lbase.weight_cosine(ref["Ms"], sus["Ms"])
    out["cka"] = lbase.cka_lineage_score(ref["acts"], sus["acts"])
    out["svcca"] = lbase.svcca_lineage_score(ref["acts"], sus["acts"])
    out["ipguard_regr"] = lbase.ipguard_match_rate(ref["preds"], sus["preds"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, default=2)
    ap.add_argument("--n-per-descendant-type", type=int, default=3)
    ap.add_argument("--n-same-arch-diff-seed", type=int, default=8)
    ap.add_argument("--n-distilled", type=int, default=3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--in-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=30)
    ap.add_argument("--out", default="results/lineage_baselines_mlp.json")
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    t0 = time.time()
    pairs = []
    descendant_kinds = ["fine_tune", "fine_tune_new_target", "noise", "prune", "quantize"]
    nondesc_kinds = ["diff_seed_same_task", "distilled"]

    for ref_idx in range(args.n_refs):
        target_key = 42 + ref_idx
        X, y = make_data(args.in_dim, n=2000, seed=ref_idx, target_key=target_key)
        Xv, yv = make_data(args.in_dim, n=400, seed=100 + ref_idx, target_key=target_key)

        ref_model = fresh_model(args.depth, args.hidden, args.in_dim, seed=ref_idx)
        train_model(ref_model, X, y, epochs=args.epochs)
        ref = model_pack(ref_model, Xv)
        ref["eval_loss"] = eval_loss(ref_model, Xv, yv)
        print(f"[ref {ref_idx}] trained, eval_loss={ref['eval_loss']:.4f}")

        # ----- descendants -----
        for k in range(args.n_per_descendant_type):
            kid = descendant_fine_tune(ref_model, X, y, epochs=args.ft_epochs)
            pairs.append({"ref": ref_idx, "kind": "fine_tune", "label": 1,
                          "scores": score_pair(ref, model_pack(kid, Xv), tau_s=0.5)})
            X2, y2 = make_data(args.in_dim, n=2000, seed=200 + ref_idx * 10 + k,
                               target_key=target_key + 100 + k)
            kid = descendant_finetune_new_target(ref_model, X2, y2, epochs=args.ft_epochs)
            pairs.append({"ref": ref_idx, "kind": "fine_tune_new_target", "label": 1,
                          "scores": score_pair(ref, model_pack(kid, Xv), tau_s=0.5)})
            kid = descendant_noise(ref_model, sigma_rel=0.02, seed=ref_idx * 100 + k)
            pairs.append({"ref": ref_idx, "kind": "noise", "label": 1,
                          "scores": score_pair(ref, model_pack(kid, Xv), tau_s=0.5)})
            kid = descendant_prune(ref_model, sparsity=0.3, seed=ref_idx * 100 + k)
            pairs.append({"ref": ref_idx, "kind": "prune", "label": 1,
                          "scores": score_pair(ref, model_pack(kid, Xv), tau_s=0.5)})
            kid = descendant_quantize(ref_model, levels=256)
            pairs.append({"ref": ref_idx, "kind": "quantize", "label": 1,
                          "scores": score_pair(ref, model_pack(kid, Xv), tau_s=0.5)})
        print(f"[ref {ref_idx}] descendants done")

        # ----- non-descendants: same arch, different seed, same task -----
        for k in range(args.n_same_arch_diff_seed):
            other = fresh_model(args.depth, args.hidden, args.in_dim,
                                seed=1000 + ref_idx * 100 + k)
            train_model(other, X, y, epochs=args.epochs)
            pairs.append({"ref": ref_idx, "kind": "diff_seed_same_task", "label": 0,
                          "scores": score_pair(ref, model_pack(other, Xv), tau_s=0.5)})

        # ----- non-descendants: distilled student -----
        for k in range(args.n_distilled):
            stu = nondesc_distilled_student(ref_model, X, epochs=args.epochs // 2,
                                            seed=9000 + ref_idx * 100 + k,
                                            depth=args.depth, hidden=args.hidden,
                                            in_dim=args.in_dim)
            pairs.append({"ref": ref_idx, "kind": "distilled", "label": 0,
                          "scores": score_pair(ref, model_pack(stu, Xv), tau_s=0.5)})
        print(f"[ref {ref_idx}] non-descendants done; running total {len(pairs)} pairs")

    # ---------- AUROC table ----------
    labels = np.array([p["label"] for p in pairs])
    auroc_table = {}
    perkind_table = {}
    for m in METHODS:
        scores = np.array([p["scores"][m] for p in pairs])
        # Some baselines return distances; AUROC handles direction via sign.
        try:
            au = float(roc_auc_score(labels, scores))
        except Exception:
            au = float("nan")
        auroc_table[m] = au
        perkind = {}
        for kind in descendant_kinds + nondesc_kinds:
            mask = np.array([p["kind"] == kind for p in pairs])
            if mask.sum() == 0:
                continue
            perkind[kind] = {
                "n": int(mask.sum()),
                "mean": float(scores[mask].mean()),
                "std": float(scores[mask].std()),
            }
        perkind_table[m] = perkind

    out = {
        "config": vars(args),
        "n_pairs": len(pairs),
        "auroc": auroc_table,
        "per_kind": perkind_table,
        "pairs": pairs,
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out} ({len(pairs)} pairs, {time.time() - t0:.1f}s)")
    print("\nAUROC by method:")
    for m, v in auroc_table.items():
        print(f"  {m:24s} {v:.4f}")


if __name__ == "__main__":
    main()
