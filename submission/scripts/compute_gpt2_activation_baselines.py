#!/usr/bin/env python3
"""Compute CKA, SVCCA, and IPGuard baselines on GPT-2 benchmark.

Requires model checkpoints (not just branch products).

Usage (on GPU cluster):
    cd experiments/scripts
    python compute_gpt2_activation_baselines.py \
        --results-dir results/lineage_benchmark_gpt2_paper_v2

Output:
    results/lineage_benchmark_gpt2_paper_v2/activation_baselines_auroc.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from gpt2_lineage_benchmark.model import create_model
from gpt2_lineage_benchmark.config import BenchmarkConfig
from gpt2_lineage_benchmark.data import get_tokenizer, create_dataloaders


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between activation matrices."""
    X = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    Y = Y.astype(np.float64) - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    nA = np.linalg.norm(X.T @ X, ord='fro')
    nB = np.linalg.norm(Y.T @ Y, ord='fro')
    return float(cross / (nA * nB + 1e-12))


def svcca(X: np.ndarray, Y: np.ndarray, variance_threshold: float = 0.99) -> float:
    """SVCCA mean correlation."""
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    Yc = Y.astype(np.float64) - Y.mean(axis=0, keepdims=True)
    Ux, sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Yc, full_matrices=False)

    def keep(s):
        c = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-12)
        k = int(np.searchsorted(c, variance_threshold) + 1)
        return max(1, min(k, len(s)))

    kx, ky = keep(sx), keep(sy)
    Ux, Uy = Ux[:, :kx], Uy[:, :ky]
    Qx, _ = np.linalg.qr(Ux)
    Qy, _ = np.linalg.qr(Uy)
    _, corrs, _ = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    return float(np.mean(corrs))


def matched_activation_score(
    actsA: List[np.ndarray], actsB: List[np.ndarray], metric
) -> float:
    """Apply activation metric with Hungarian layer alignment."""
    L = len(actsA)
    S = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            S[i, j] = metric(actsA[i], actsB[j])
    row, col = linear_sum_assignment(-S)
    return float(S[row, col].mean())


def ipguard_match_rate(predsA: np.ndarray, predsB: np.ndarray) -> float:
    """IPGuard-style agreement rate (adapted for language modeling).

    For LM outputs, we use top-k token agreement instead of class agreement.
    """
    # predsA, predsB are logits of shape (n_samples, vocab_size)
    topk = 5
    topA = np.argsort(predsA, axis=-1)[:, -topk:]
    topB = np.argsort(predsB, axis=-1)[:, -topk:]

    # Compute overlap in top-k predictions
    agreements = []
    for a, b in zip(topA, topB):
        overlap = len(set(a) & set(b)) / topk
        agreements.append(overlap)
    return float(np.mean(agreements))


def extract_activations(
    model: torch.nn.Module, dataloader, device: str, max_batches: int = 10
) -> List[np.ndarray]:
    """Extract per-block activations."""
    model = model.to(device).eval()
    activations = defaultdict(list)
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Mean over sequence dimension
            activations[idx].append(output.detach().mean(dim=1).cpu().numpy())
        return hook

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    result = []
    for i in sorted(activations.keys()):
        result.append(np.concatenate(activations[i], axis=0))
    return result


def extract_predictions(
    model: torch.nn.Module, dataloader, device: str, max_batches: int = 10
) -> np.ndarray:
    """Extract model predictions (logits) for IPGuard."""
    model = model.to(device).eval()
    all_logits = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # Take logits at last position
            logits = outputs.logits[:, -1, :].cpu().numpy()
            all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def compute_gap_z(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Compute Gap-Z: (mean_pos - mean_neg) / pooled_std."""
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    mean_pos, mean_neg = pos.mean(), neg.mean()
    std_pos, std_neg = pos.std(), neg.std()
    n_pos, n_neg = len(pos), len(neg)
    pooled_var = ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2)
    pooled_std = np.sqrt(pooled_var / (n_pos + n_neg - 2))
    return float((mean_pos - mean_neg) / (pooled_std + 1e-12))


def load_model(ckpt_path: Path, config) -> torch.nn.Module:
    """Load model from checkpoint."""
    model = create_model(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Compute CKA/SVCCA/IPGuard for GPT-2 baselines"
    )
    parser.add_argument(
        "--results-dir",
        default="results/lineage_benchmark_gpt2_paper_v2",
        help="Path to benchmark results directory"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=15,
        help="Max batches for activation extraction"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / args.results_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Results dir: {results_dir}")

    config = BenchmarkConfig.from_preset("paper")

    # Load data
    print("\nLoading data...")
    tokenizer = get_tokenizer()
    loaders = create_dataloaders(
        dataset_name="tinystories",
        tokenizer=tokenizer,
        max_train_samples=1000,
        max_val_samples=500,
        batch_size=8,
    )
    dataloader = loaders["val"]

    # Find checkpoints
    checkpoint_dir = results_dir / "checkpoints"
    models_dir = results_dir / "models"

    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint dir not found: {checkpoint_dir}")
        return
    if not models_dir.exists():
        print(f"ERROR: Models dir not found: {models_dir}")
        return

    # Load root models (test split: 5, 6, 7)
    test_roots = [5, 6, 7]
    print(f"\nLoading root models for test split: {test_roots}")

    roots = {}
    for idx in test_roots:
        ckpt_path = checkpoint_dir / f"root_{idx}" / "epoch_3.pt"
        if not ckpt_path.exists():
            print(f"  Root {idx}: NOT FOUND at {ckpt_path}")
            continue
        print(f"  Loading root_{idx}...", end=" ", flush=True)
        model = load_model(ckpt_path, config)
        acts = extract_activations(model, dataloader, device, args.max_batches)
        preds = extract_predictions(model, dataloader, device, args.max_batches)
        roots[idx] = {"model": model, "acts": acts, "preds": preds}
        print("done")

    # Load descendant and student models
    print("\nLoading descendant/student models...")
    descendants = {}
    students = {}

    for pt_file in sorted(models_dir.glob("*.pt")):
        name = pt_file.stem
        # Parse: root5_cont_pt_same_0 -> root_idx=5, type=cont_pt_same
        parts = name.split("_")
        root_idx = int(parts[0].replace("root", ""))

        if root_idx not in test_roots:
            continue

        model_type = "_".join(parts[1:-1])  # e.g., cont_pt_same, distilled, lora

        print(f"  Loading {name}...", end=" ", flush=True)
        model = load_model(pt_file, config)
        acts = extract_activations(model, dataloader, device, args.max_batches)
        preds = extract_predictions(model, dataloader, device, args.max_batches)

        data = {
            "acts": acts,
            "preds": preds,
            "root_idx": root_idx,
            "type": model_type,
        }

        if "distilled" in model_type:
            students[name] = data
        else:
            descendants[name] = data
        print("done")

    print(f"\nLoaded: {len(roots)} roots, {len(descendants)} descendants, "
          f"{len(students)} students")

    # Also need independent pairs (cross-root)
    # Load remaining roots for independent comparisons
    print("\nLoading additional roots for independent pairs...")
    all_roots = {}
    for idx in range(8):
        if idx in roots:
            all_roots[idx] = roots[idx]
            continue
        ckpt_path = checkpoint_dir / f"root_{idx}" / "epoch_3.pt"
        if not ckpt_path.exists():
            continue
        print(f"  Loading root_{idx}...", end=" ", flush=True)
        model = load_model(ckpt_path, config)
        acts = extract_activations(model, dataloader, device, args.max_batches)
        preds = extract_predictions(model, dataloader, device, args.max_batches)
        all_roots[idx] = {"acts": acts, "preds": preds}
        print("done")

    # Compute baselines
    print("\n" + "=" * 70)
    print("Computing activation-based baseline scores...")
    print("=" * 70)

    methods = {
        "CKA": lambda a, b: matched_activation_score(a["acts"], b["acts"], linear_cka),
        "SVCCA": lambda a, b: matched_activation_score(a["acts"], b["acts"], svcca),
        "IPGuard": lambda a, b: ipguard_match_rate(a["preds"], b["preds"]),
    }

    all_scores = {m: {"positive": [], "negative": []} for m in methods}

    # Descendant pairs (positive)
    for desc_name, desc_data in descendants.items():
        root_idx = desc_data["root_idx"]
        if root_idx not in roots:
            continue

        root_data = roots[root_idx]
        print(f"  {desc_name} (descendant)...", end=" ", flush=True)

        for method_name, scorer in methods.items():
            score = scorer(root_data, desc_data)
            all_scores[method_name]["positive"].append(score)
        print("done")

    # Student pairs (negative - distilled)
    for student_name, student_data in students.items():
        root_idx = student_data["root_idx"]
        if root_idx not in roots:
            continue

        root_data = roots[root_idx]
        print(f"  {student_name} (distilled)...", end=" ", flush=True)

        for method_name, scorer in methods.items():
            score = scorer(root_data, student_data)
            all_scores[method_name]["negative"].append(score)
        print("done")

    # Independent pairs (negative - cross-root)
    for i in test_roots:
        for j in range(8):
            if i == j or i not in all_roots or j not in all_roots:
                continue

            print(f"  root_{i} vs root_{j} (independent)...", end=" ", flush=True)

            for method_name, scorer in methods.items():
                score = scorer(all_roots[i], all_roots[j])
                all_scores[method_name]["negative"].append(score)
            print("done")

    # Compute AUROC and Gap-Z
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = {}
    for method_name in methods:
        pos = all_scores[method_name]["positive"]
        neg = all_scores[method_name]["negative"]

        if not pos or not neg:
            print(f"  {method_name}: insufficient data")
            continue

        y_true = [1] * len(pos) + [0] * len(neg)
        y_score = pos + neg

        auroc = roc_auc_score(y_true, y_score)
        gap_z = compute_gap_z(pos, neg)

        results[method_name] = {
            "auroc": auroc,
            "gap_z": gap_z,
            "n_positive": len(pos),
            "n_negative": len(neg),
            "pos_mean": float(np.mean(pos)),
            "pos_std": float(np.std(pos)),
            "neg_mean": float(np.mean(neg)),
            "neg_std": float(np.std(neg)),
        }

        print(f"\n{method_name}:")
        print(f"  AUROC = {auroc:.3f}")
        print(f"  Gap-Z = {gap_z:+.1f}")
        print(f"  Positive: n={len(pos)}, mean={np.mean(pos):.4f}")
        print(f"  Negative: n={len(neg)}, mean={np.mean(neg):.4f}")

    print("\n" + "=" * 70)
    print("TABLE FOR PAPER")
    print("=" * 70)
    print(f"{'Method':<20} {'AUROC':>8} {'Gap-Z':>10}")
    print("-" * 40)
    for method_name, r in results.items():
        auroc_str = f"{r['auroc']:.3f}"
        gap_z_str = f"{r['gap_z']:+.1f}"
        print(f"{method_name:<20} {auroc_str:>8} {gap_z_str:>10}")

    # Save results
    output_path = results_dir / "activation_baselines_auroc.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
