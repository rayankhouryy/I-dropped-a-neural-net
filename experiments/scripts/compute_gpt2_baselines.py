#!/usr/bin/env python3
"""Compute baseline methods on GPT-2-Small-Lite benchmark.

Baselines:
1. Weight cosine similarity
2. Aligned Frobenius distance
3. Singular value distance
4. CKA (requires forward passes)

Run on cluster with GPU for CKA computation.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from gpt2_lineage_benchmark.model import create_model
from gpt2_lineage_benchmark.config import BenchmarkConfig
from gpt2_lineage_benchmark.data import get_tokenizer, create_dataloaders


def flatten_weights(model: torch.nn.Module) -> torch.Tensor:
    """Flatten all model weights into a single vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])


def weight_cosine(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    """Compute cosine similarity between flattened weights."""
    w_a = flatten_weights(model_a)
    w_b = flatten_weights(model_b)
    return F.cosine_similarity(w_a.unsqueeze(0), w_b.unsqueeze(0)).item()


def weight_frobenius(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    """Compute normalized Frobenius distance between weights."""
    w_a = flatten_weights(model_a)
    w_b = flatten_weights(model_b)
    dist = torch.norm(w_a - w_b)
    norm = (torch.norm(w_a) + torch.norm(w_b)) / 2
    return (dist / norm).item()


def singular_value_distance(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    """Compute singular value distance between weight matrices.

    For each weight matrix, compare sorted singular values.
    """
    total_dist = 0.0
    total_norm = 0.0

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    for name in params_a:
        if params_a[name].dim() < 2:
            continue

        w_a = params_a[name].detach()
        w_b = params_b[name].detach()

        # Reshape to 2D if needed
        if w_a.dim() > 2:
            w_a = w_a.view(w_a.size(0), -1)
            w_b = w_b.view(w_b.size(0), -1)

        try:
            sv_a = torch.linalg.svdvals(w_a)
            sv_b = torch.linalg.svdvals(w_b)

            # Pad to same length if needed
            max_len = max(len(sv_a), len(sv_b))
            if len(sv_a) < max_len:
                sv_a = F.pad(sv_a, (0, max_len - len(sv_a)))
            if len(sv_b) < max_len:
                sv_b = F.pad(sv_b, (0, max_len - len(sv_b)))

            dist = torch.norm(sv_a - sv_b)
            norm = (torch.norm(sv_a) + torch.norm(sv_b)) / 2

            total_dist += dist.item()
            total_norm += norm.item()
        except:
            continue

    return total_dist / total_norm if total_norm > 0 else 0.0


def extract_activations(model: torch.nn.Module, dataloader, device: str,
                        max_batches: int = 10) -> Dict[str, torch.Tensor]:
    """Extract activations from each layer for CKA computation."""
    model = model.to(device).eval()

    activations = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Take mean over sequence dimension
            activations[name].append(output.detach().mean(dim=1).cpu())
        return hook

    # Register hooks on transformer blocks
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(make_hook(f"layer_{i}")))

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids, attention_mask=attention_mask)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate activations
    result = {}
    for name, acts in activations.items():
        result[name] = torch.cat(acts, dim=0)

    model.cpu()
    return result


def centering_matrix(n: int) -> torch.Tensor:
    """Create centering matrix H = I - 1/n * 11^T."""
    return torch.eye(n) - torch.ones(n, n) / n


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two activation matrices.

    X, Y: (n_samples, n_features)
    """
    n = X.size(0)

    # Center the matrices
    H = centering_matrix(n)

    # Compute kernel matrices
    K = X @ X.T
    L = Y @ Y.T

    # Center the kernel matrices
    K_c = H @ K @ H
    L_c = H @ L @ H

    # Compute HSIC
    hsic_kl = torch.trace(K_c @ L_c)
    hsic_kk = torch.trace(K_c @ K_c)
    hsic_ll = torch.trace(L_c @ L_c)

    # CKA
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return cka.item()


def compute_cka(model_a: torch.nn.Module, model_b: torch.nn.Module,
                dataloader, device: str, max_batches: int = 10) -> float:
    """Compute mean CKA across layers between two models."""
    acts_a = extract_activations(model_a, dataloader, device, max_batches)
    acts_b = extract_activations(model_b, dataloader, device, max_batches)

    cka_scores = []
    for layer in acts_a:
        if layer in acts_b:
            cka = linear_cka(acts_a[layer], acts_b[layer])
            cka_scores.append(cka)

    return np.mean(cka_scores) if cka_scores else 0.0


def load_model_from_checkpoint(ckpt_path: Path, config) -> torch.nn.Module:
    """Load model from checkpoint file."""
    model = create_model(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute baseline methods on GPT-2 benchmark")
    parser.add_argument("--results-dir", default="experiments/scripts/results/lineage_benchmark_gpt2_paper",
                        help="Path to benchmark results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    checkpoint_dir = results_dir / "checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Results dir: {results_dir}")

    # Load config
    config = BenchmarkConfig.from_preset("paper")

    # Load benchmark results for pair information
    with open(results_dir / "benchmark_results.json") as f:
        benchmark = json.load(f)

    # Create dataloader for CKA
    print("\nLoading data for CKA...")
    tokenizer = get_tokenizer()
    loaders = create_dataloaders(
        dataset_name="tinystories",
        tokenizer=tokenizer,
        max_train_samples=1000,
        max_val_samples=500,
        batch_size=8,
    )
    val_loader = loaders["val"]

    # Load all root models from checkpoints
    print("\nLoading root models...")
    roots = {}
    for i in range(8):
        root_dir = checkpoint_dir / f"root_{i}"
        ckpt_path = None
        for epoch in [3, 2, 1, 0]:
            candidate = root_dir / f"epoch_{epoch}.pt"
            if candidate.exists():
                ckpt_path = candidate
                break

        if ckpt_path:
            roots[i] = load_model_from_checkpoint(ckpt_path, config)
            print(f"  Root {i}: loaded from {ckpt_path.name}")
        else:
            print(f"  Root {i}: not found")

    # Load descendants and students from saved models directory
    print("\nLoading descendants and students...")
    import pickle

    descendants = {}
    students = {}

    models_dir = results_dir / "models"

    # Get descendant/student info from benchmark results
    with open(results_dir / "benchmark_results.json") as f:
        benchmark = json.load(f)

    if models_dir.exists():
        # Load from saved model files
        for desc_info in benchmark["descendants"]:
            desc_id = desc_info["id"]
            model_path = models_dir / f"{desc_id}.pt"
            if model_path.exists():
                model = create_model(config.model)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                descendants[desc_id] = {
                    "model": model,
                    "root_idx": desc_info["root_idx"],
                    "type": desc_info["type"],
                }
                print(f"  Descendant {desc_id}: loaded")

        for student_info in benchmark["distilled_students"]:
            student_id = student_info["id"]
            model_path = models_dir / f"{student_id}.pt"
            if model_path.exists():
                model = create_model(config.model)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                students[student_id] = {
                    "model": model,
                    "root_idx": student_info["root_idx"],
                }
                print(f"  Student {student_id}: loaded")
    else:
        print(f"  Models directory not found at {models_dir}")
        print("  Run Phase 2 with --save-models flag to save model weights")

    print(f"\nLoaded: {len(roots)} roots, {len(descendants)} descendants, {len(students)} students")

    if len(roots) < 2:
        print("Not enough models loaded. Check checkpoint paths.")
        return

    # Compute baselines for test-split pairs
    print("\n" + "="*60)
    print("Computing baselines on test-split pairs...")
    print("="*60)

    test_roots = [5, 6, 7]  # Test split roots

    results_by_type = defaultdict(lambda: defaultdict(list))

    # Descendant pairs (root vs its descendants)
    print("\n--- Descendant pairs ---")
    for desc_id, desc_data in descendants.items():
        root_idx = desc_data["root_idx"]
        if root_idx not in test_roots:
            continue
        if root_idx not in roots:
            continue

        root_model = roots[root_idx]
        desc_model = desc_data["model"]
        desc_type = desc_data["type"]

        print(f"  {desc_id}...", end=" ", flush=True)

        wcos = weight_cosine(root_model, desc_model)
        wfrob = weight_frobenius(root_model, desc_model)
        svd = singular_value_distance(root_model, desc_model)

        results_by_type[desc_type]["weight_cosine"].append(wcos)
        results_by_type[desc_type]["frobenius"].append(wfrob)
        results_by_type[desc_type]["svd_dist"].append(svd)
        results_by_type[desc_type]["label"].append("descendant")

        print(f"wcos={wcos:.4f}, frob={wfrob:.4f}, svd={svd:.4f}")

    # Distilled student pairs
    print("\n--- Distilled student pairs ---")
    for student_id, student_data in students.items():
        root_idx = student_data["root_idx"]
        if root_idx not in test_roots:
            continue
        if root_idx not in roots:
            continue

        root_model = roots[root_idx]
        student_model = student_data["model"]

        print(f"  {student_id}...", end=" ", flush=True)

        wcos = weight_cosine(root_model, student_model)
        wfrob = weight_frobenius(root_model, student_model)
        svd = singular_value_distance(root_model, student_model)

        results_by_type["distilled"]["weight_cosine"].append(wcos)
        results_by_type["distilled"]["frobenius"].append(wfrob)
        results_by_type["distilled"]["svd_dist"].append(svd)
        results_by_type["distilled"]["label"].append("non_descendant")

        print(f"wcos={wcos:.4f}, frob={wfrob:.4f}, svd={svd:.4f}")

    # Independent pairs (cross-root)
    print("\n--- Independent pairs ---")
    for i in test_roots:
        for j in range(8):
            if i == j:
                continue
            if i not in roots or j not in roots:
                continue

            print(f"  root_{i} vs root_{j}...", end=" ", flush=True)

            wcos = weight_cosine(roots[i], roots[j])
            wfrob = weight_frobenius(roots[i], roots[j])
            svd = singular_value_distance(roots[i], roots[j])

            results_by_type["independent"]["weight_cosine"].append(wcos)
            results_by_type["independent"]["frobenius"].append(wfrob)
            results_by_type["independent"]["svd_dist"].append(svd)
            results_by_type["independent"]["label"].append("non_descendant")

            print(f"wcos={wcos:.4f}, frob={wfrob:.4f}, svd={svd:.4f}")

    # CKA computation (slower, optional)
    compute_cka_flag = device == "cuda"  # Only compute CKA on GPU
    if compute_cka_flag:
        print("\n--- Computing CKA (GPU) ---")

        # Sample a few pairs for CKA
        for desc_id, desc_data in list(descendants.items())[:6]:
            root_idx = desc_data["root_idx"]
            if root_idx not in test_roots or root_idx not in roots:
                continue

            print(f"  CKA {desc_id}...", end=" ", flush=True)
            cka = compute_cka(roots[root_idx], desc_data["model"], val_loader, device, max_batches=5)
            results_by_type[desc_data["type"]]["cka"].append(cka)
            print(f"{cka:.4f}")

        for student_id, student_data in list(students.items())[:3]:
            root_idx = student_data["root_idx"]
            if root_idx not in test_roots or root_idx not in roots:
                continue

            print(f"  CKA {student_id}...", end=" ", flush=True)
            cka = compute_cka(roots[root_idx], student_data["model"], val_loader, device, max_batches=5)
            results_by_type["distilled"]["cka"].append(cka)
            print(f"{cka:.4f}")

        # A few independent pairs
        for i, j in [(5, 0), (6, 1), (7, 2)]:
            if i in roots and j in roots:
                print(f"  CKA root_{i} vs root_{j}...", end=" ", flush=True)
                cka = compute_cka(roots[i], roots[j], val_loader, device, max_batches=5)
                results_by_type["independent"]["cka"].append(cka)
                print(f"{cka:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary = {}
    for attack_type, metrics in results_by_type.items():
        summary[attack_type] = {}
        for metric_name, values in metrics.items():
            if metric_name == "label":
                continue
            if values:
                summary[attack_type][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
        print(f"\n{attack_type}:")
        for metric_name, stats in summary[attack_type].items():
            print(f"  {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")

    # Save results
    output_path = results_dir / "baseline_methods.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print comparison table
    print("\n" + "="*60)
    print("BASELINE COMPARISON TABLE (for paper)")
    print("="*60)
    print(f"{'Type':<25} {'Weight Cos':>12} {'Frob Dist':>12} {'SVD Dist':>12}")
    print("-" * 65)

    descendant_types = ["quantized", "lora_merge", "continued_pretraining", "pruned"]
    for dtype in descendant_types:
        if dtype in summary:
            s = summary[dtype]
            wcos = s.get("weight_cosine", {}).get("mean", 0)
            frob = s.get("frobenius", {}).get("mean", 0)
            svd = s.get("svd_dist", {}).get("mean", 0)
            print(f"{dtype:<25} {wcos:>12.4f} {frob:>12.4f} {svd:>12.4f}")

    print("-" * 65)
    for dtype in ["distilled", "independent"]:
        if dtype in summary:
            s = summary[dtype]
            wcos = s.get("weight_cosine", {}).get("mean", 0)
            frob = s.get("frobenius", {}).get("mean", 0)
            svd = s.get("svd_dist", {}).get("mean", 0)
            print(f"{dtype:<25} {wcos:>12.4f} {frob:>12.4f} {svd:>12.4f}")


if __name__ == "__main__":
    main()
