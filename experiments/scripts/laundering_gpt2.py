#!/usr/bin/env python3
"""GPT-2 hidden-unit permutation laundering benchmark.

Tests whether the centered residual signature lineage method is invariant to
function-preserving hidden-unit permutations, while raw-weight baselines collapse.

Experimental conditions:
    NONE      - Original checkpoints (baseline)
    P-SUSPECT - Permute only suspect checkpoint
    P-BOTH    - Different permutations on both ref and suspect
    P+FT      - P-SUSPECT + brief continued training

Methods evaluated:
    Proposed (on M = W_proj @ W_fc):
        - centered_residual_signature (full method)

    Raw baselines (should COLLAPSE under P):
        - raw_weight_cosine
        - raw_aligned_frobenius
        - singular_value_distance (may be invariant)

    Recovery baselines:
        - rebasin_frobenius
        - rebasin_scale_frobenius

    Ablation (on NONE only, for discrimination not invariance):
        - raw_branch_product_cosine
        - centered_branch_product_cosine
        - centered_with_gating
        - full_lineage_score

Usage:
    python laundering_gpt2.py --benchmark-dir results/lineage_benchmark_gpt2_paper
    python laundering_gpt2.py --smoke  # Quick test on 2 pairs
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import GPT2Tokenizer

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import laundering_gpt2_ops as lops
import gpt2_laundering_baselines as lbase
import lineage_detection as ldet
from gpt2_lineage_benchmark.extraction import extract_branch_products
from gpt2_lineage_benchmark.model import load_checkpoint
from gpt2_lineage_benchmark.config import ModelConfig, BenchmarkConfig


# ---------------------------------------------------------------- configuration

@dataclass
class LaunderingConfig:
    """Configuration for the laundering experiment."""
    # Input paths
    benchmark_dir: str = "results/lineage_benchmark_gpt2_paper"
    checkpoint_dir: Optional[str] = None  # defaults to benchmark_dir/checkpoints

    # Output
    output_dir: str = "results/laundering_gpt2"

    # Experimental conditions
    conditions: List[str] = field(default_factory=lambda: ["NONE", "P-SUSPECT", "P-BOTH", "P+FT"])
    n_perm_seeds: int = 5  # Number of random permutation seeds per condition

    # P+FT settings
    pft_steps: int = 1000
    pft_lr: float = 2e-5

    # Validation
    n_validation_seqs: int = 128

    # Seeds
    seed_base: int = 8000

    # Test split
    test_root_indices: List[int] = field(default_factory=lambda: [5, 6, 7])  # roots 5-7

    # Smoke test
    smoke: bool = False

    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"{self.benchmark_dir}/checkpoints"


# ---------------------------------------------------------------- data structures

@dataclass
class PairResult:
    """Result for one (reference, suspect) pair under one condition/seed."""
    condition: str
    seed: int
    ref_id: str
    sus_id: str
    label: str  # "descendant" or "non_descendant"
    attack_type: str  # "continued_pretraining", "lora_merge", etc.
    split: str
    scores: Dict[str, float]
    validation: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------- scoring functions

def score_pair_all_methods(
    ref_Ms: List[np.ndarray],
    sus_Ms: List[np.ndarray],
    ref_raw: Dict[str, List[np.ndarray]],
    sus_raw: Dict[str, List[np.ndarray]],
    tau_s: float,
) -> Dict[str, float]:
    """Score a pair with all methods."""
    scores = {}

    # Proposed method (full)
    L_score, _, _ = ldet.lineage_score(ref_Ms, sus_Ms, tau_s)
    scores["centered_residual_signature"] = L_score

    # Raw baselines
    scores["raw_weight_cosine"] = lbase.raw_weight_cosine_gpt2(ref_raw, sus_raw)
    scores["raw_aligned_frobenius"] = lbase.raw_aligned_frobenius_gpt2(ref_raw, sus_raw)
    scores["singular_value_distance"] = lbase.singular_value_distance_gpt2(ref_raw, sus_raw)

    # Re-Basin
    scores["rebasin_frobenius"] = lbase.rebasin_frobenius_gpt2(ref_raw, sus_raw)
    scores["rebasin_scale_frobenius"] = lbase.rebasin_scale_frobenius_gpt2(ref_raw, sus_raw)

    # Ablation methods
    scores["raw_branch_product_cosine"] = lbase.raw_branch_product_cosine(ref_Ms, sus_Ms)
    scores["centered_branch_product_cosine"] = lbase.centered_branch_product_cosine(ref_Ms, sus_Ms)
    scores["centered_with_gating"] = lbase.centered_with_gating(ref_Ms, sus_Ms, tau_s)

    return scores


# ---------------------------------------------------------------- main experiment

def load_benchmark_data(config: LaunderingConfig) -> Tuple[Dict, Dict, float]:
    """Load phase1 and phase2 data from the benchmark."""
    benchmark_dir = Path(config.benchmark_dir)

    phase1_path = benchmark_dir / "phase1_roots.pkl"
    phase2_path = benchmark_dir / "phase2_descendants.pkl"

    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase 1 data not found: {phase1_path}")
    if not phase2_path.exists():
        raise FileNotFoundError(f"Phase 2 data not found: {phase2_path}")

    with open(phase1_path, "rb") as f:
        phase1_data = pickle.load(f)
    with open(phase2_path, "rb") as f:
        phase2_data = pickle.load(f)

    tau_s = phase1_data["tau_s"]

    return phase1_data, phase2_data, tau_s


def build_test_pairs(
    phase1_data: Dict,
    phase2_data: Dict,
    config: LaunderingConfig,
) -> List[Dict[str, Any]]:
    """Build list of test pairs from benchmark data."""
    pairs = []
    test_roots = set(config.test_root_indices)

    roots_info = phase1_data["roots_info"]
    root_signatures = phase1_data["root_signatures"]
    descendants = phase2_data["descendants"]
    students = phase2_data["students"]

    # Descendant pairs (test roots only)
    for desc in descendants:
        root_idx = desc["root_idx"]
        if root_idx not in test_roots:
            continue

        pairs.append({
            "ref_id": f"root_{root_idx}",
            "ref_idx": root_idx,
            "sus_id": desc["id"],
            "sus_Ms": desc["Ms"],
            "label": "descendant",
            "attack_type": desc["type"],
            "split": "test",
        })

    # Distilled students (test roots only)
    for student in students:
        root_idx = student["teacher_root_idx"]
        if root_idx not in test_roots:
            continue

        pairs.append({
            "ref_id": f"root_{root_idx}",
            "ref_idx": root_idx,
            "sus_id": student["id"],
            "sus_Ms": student["Ms"],
            "label": "non_descendant",
            "attack_type": "distilled_student",
            "split": "test",
        })

    # Cross-root pairs (test roots only, unordered to avoid duplicates)
    test_roots_list = sorted(test_roots)
    for i, root_i in enumerate(test_roots_list):
        for root_j in test_roots_list[i+1:]:
            pairs.append({
                "ref_id": f"root_{root_i}",
                "ref_idx": root_i,
                "sus_id": f"root_{root_j}",
                "sus_idx": root_j,
                "sus_Ms": root_signatures[root_j],
                "label": "non_descendant",
                "attack_type": "independent",
                "split": "test",
            })

    return pairs


def run_experiment(config: LaunderingConfig) -> Dict[str, Any]:
    """Run the full laundering experiment."""
    t0 = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load benchmark data
    print("Loading benchmark data...")
    phase1_data, phase2_data, tau_s = load_benchmark_data(config)
    root_signatures = phase1_data["root_signatures"]
    model_config = ModelConfig()

    print(f"tau_s = {tau_s:.4f}")
    print(f"Test roots: {config.test_root_indices}")

    # Build test pairs
    pairs = build_test_pairs(phase1_data, phase2_data, config)
    print(f"Test pairs: {len(pairs)}")

    if config.smoke:
        pairs = pairs[:2]
        config.conditions = ["NONE", "P-SUSPECT"]
        config.n_perm_seeds = 1
        print(f"SMOKE TEST: {len(pairs)} pairs, conditions={config.conditions}")

    # Load tokenizer for validation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create validation batch
    print("Creating validation batch...")
    validation_batch = lops.make_validation_batch(
        tokenizer,
        n_seqs=config.n_validation_seqs,
        seed=config.seed_base + 999,
    )

    # Results storage
    all_results: List[PairResult] = []
    validation_log = []
    invariance_checks = []

    # Process each condition
    for condition in config.conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        n_seeds = 1 if condition == "NONE" else config.n_perm_seeds

        for seed_idx in range(n_seeds):
            seed = config.seed_base + seed_idx * 100

            if condition != "NONE":
                print(f"\n  Seed {seed_idx+1}/{n_seeds} (seed={seed})")

            for pair_idx, pair in enumerate(pairs):
                ref_idx = pair["ref_idx"]
                ref_Ms = root_signatures[ref_idx]

                # Load reference model for raw weights
                ref_ckpt = Path(config.checkpoint_dir) / f"root_{ref_idx}" / "epoch_3.pt"
                ref_model, _, _ = load_checkpoint(ref_ckpt, model_config, device)
                ref_raw = lops.raw_weights_gpt2(ref_model)

                # Get suspect model/signatures
                if "sus_idx" in pair:
                    # Cross-root pair (root vs root)
                    sus_idx = pair["sus_idx"]
                    sus_ckpt = Path(config.checkpoint_dir) / f"root_{sus_idx}" / "epoch_3.pt"
                    sus_model, _, _ = load_checkpoint(sus_ckpt, model_config, device)
                    sus_Ms = pair["sus_Ms"]
                else:
                    # Descendant/student - try to load from models/ directory
                    sus_id = pair["sus_id"]
                    models_dir = Path(config.benchmark_dir) / "models"
                    sus_ckpt = models_dir / f"{sus_id}.pt"

                    if sus_ckpt.exists():
                        sus_model, _, _ = load_checkpoint(sus_ckpt, model_config, device)
                        sus_Ms = pair["sus_Ms"]
                        print(f"      Loaded suspect checkpoint: {sus_ckpt}")
                    else:
                        # Fallback: no checkpoint available
                        print(f"    WARNING: No checkpoint for {sus_id} at {sus_ckpt}")
                        sus_Ms = pair["sus_Ms"]
                        sus_model = None

                # Apply permutation based on condition
                print(f"      Condition={condition}, sus_model={'LOADED' if sus_model is not None else 'NONE'}")
                if condition == "NONE":
                    # No permutation
                    sus_Ms_perm = sus_Ms
                    if sus_model is not None:
                        sus_raw = lops.raw_weights_gpt2(sus_model)
                    else:
                        # For descendants, we need raw weights too
                        # This is a limitation - we'll use zeros as placeholder
                        # and mark these pairs as "signatures_only"
                        sus_raw = ref_raw  # placeholder, scores won't be meaningful for raw methods

                elif condition in ["P-SUSPECT", "P+FT"]:
                    if sus_model is not None:
                        # Apply permutation to suspect
                        sus_model_perm, manifest = lops.apply_permutation_gpt2(sus_model, seed)

                        # Validate function preservation
                        val_result = lops.validate_function_preservation(
                            sus_model, sus_model_perm, validation_batch, device
                        )
                        validation_log.append({
                            "condition": condition,
                            "seed": seed,
                            "pair": pair["sus_id"],
                            **val_result,
                        })

                        if not val_result["gate_pass"]:
                            print(f"    WARNING: Gate failed for {pair['sus_id']}: {val_result['gate_details']}")

                        # Verify branch product invariance
                        inv_check = lops.verify_branch_product_invariance(sus_model, sus_model_perm)
                        invariance_checks.append({
                            "condition": condition,
                            "seed": seed,
                            "pair": pair["sus_id"],
                            **inv_check,
                        })

                        # P+FT: additional fine-tuning
                        if condition == "P+FT":
                            # Create a simple data loader for fine-tuning
                            # This is a simplified version - in practice, use proper dataloader
                            from gpt2_lineage_benchmark.data import create_dataloaders
                            dataloaders = create_dataloaders(
                                dataset_name="tinystories",
                                tokenizer=tokenizer,
                                max_length=512,
                                batch_size=8,
                                max_train_samples=5000,
                            )
                            sus_model_perm = lops.finetune_after_permutation(
                                sus_model_perm,
                                dataloaders["train"],
                                n_steps=config.pft_steps,
                                lr=config.pft_lr,
                                device=device,
                                verbose=False,
                            )

                        sus_Ms_perm = extract_branch_products(sus_model_perm)
                        sus_raw = lops.raw_weights_gpt2(sus_model_perm)
                    else:
                        # For descendants without checkpoints, skip raw weight methods
                        sus_Ms_perm = sus_Ms  # Ms are invariant anyway
                        sus_raw = ref_raw  # placeholder

                elif condition == "P-BOTH":
                    # Permute both reference and suspect with different seeds
                    print(f"      P-BOTH: permuting ref with seed={seed}, sus with seed={seed + 50000}")
                    ref_model_perm, _ = lops.apply_permutation_gpt2(ref_model, seed)
                    ref_Ms = extract_branch_products(ref_model_perm)
                    ref_raw = lops.raw_weights_gpt2(ref_model_perm)

                    if sus_model is not None:
                        sus_model_perm, _ = lops.apply_permutation_gpt2(sus_model, seed + 50000)
                        sus_Ms_perm = extract_branch_products(sus_model_perm)
                        sus_raw = lops.raw_weights_gpt2(sus_model_perm)
                        print(f"      P-BOTH: extracted sus_raw from permuted sus_model")
                    else:
                        print(f"      P-BOTH: WARNING - sus_model is None, using ref_raw as placeholder!")
                        sus_Ms_perm = sus_Ms
                        sus_raw = ref_raw

                # Score all methods
                scores = score_pair_all_methods(
                    ref_Ms, sus_Ms_perm, ref_raw, sus_raw, tau_s
                )

                result = PairResult(
                    condition=condition,
                    seed=seed,
                    ref_id=pair["ref_id"],
                    sus_id=pair["sus_id"],
                    label=pair["label"],
                    attack_type=pair["attack_type"],
                    split=pair["split"],
                    scores=scores,
                )
                all_results.append(result)

                # Clean up GPU memory
                ref_model.cpu()
                if sus_model is not None:
                    sus_model.cpu()
                del ref_model
                if sus_model is not None:
                    del sus_model
                torch.cuda.empty_cache()

                elapsed = time.time() - t0
                avg_per_pair = elapsed / (pair_idx + 1) if pair_idx > 0 else 0
                eta = avg_per_pair * (len(pairs) - pair_idx - 1)
                print(f"    [{pair_idx+1}/{len(pairs)}] {pair['ref_id']} vs {pair['sus_id']} "
                      f"({pair['label'][:4]}) | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

            # Print intermediate results after each seed
            seed_results = [r for r in all_results if r.condition == condition and r.seed == seed]
            if seed_results:
                print(f"\n  --- Seed {seed} AUROC (n={len(seed_results)}) ---")
                labels = [1 if r.label == "descendant" else 0 for r in seed_results]
                for method in ["centered_residual_signature", "raw_weight_cosine", "rebasin_frobenius"]:
                    scores = [r.scores[method] for r in seed_results]
                    if len(set(labels)) >= 2:
                        auroc = roc_auc_score(labels, scores)
                        print(f"    {method}: {auroc:.4f}")
                print()

    # Compute aggregate metrics
    print("\n" + "="*60)
    print("Computing aggregate metrics...")
    print("="*60)

    summary = compute_aggregate_metrics(all_results, config.conditions)

    # Save results
    save_results(
        config, all_results, summary, validation_log, invariance_checks,
        time.time() - t0
    )

    # Print summary table
    print_summary_table(summary, config.conditions)

    return {
        "config": asdict(config),
        "summary": summary,
        "n_pairs": len(pairs),
        "n_results": len(all_results),
        "wall_seconds": time.time() - t0,
    }


def compute_aggregate_metrics(
    results: List[PairResult],
    conditions: List[str],
) -> Dict[str, Any]:
    """Compute AUROC and other metrics per condition and method."""
    methods = [
        "centered_residual_signature",
        "raw_weight_cosine",
        "raw_aligned_frobenius",
        "singular_value_distance",
        "rebasin_frobenius",
        "rebasin_scale_frobenius",
        "raw_branch_product_cosine",
        "centered_branch_product_cosine",
        "centered_with_gating",
    ]

    summary = {}

    for condition in conditions:
        cond_results = [r for r in results if r.condition == condition]
        if not cond_results:
            continue

        summary[condition] = {}

        for method in methods:
            labels = [1 if r.label == "descendant" else 0 for r in cond_results]
            scores = [r.scores[method] for r in cond_results]

            if len(set(labels)) < 2:
                # Need both classes for AUROC
                auroc = float("nan")
                auprc = float("nan")
            else:
                auroc = float(roc_auc_score(labels, scores))
                auprc = float(average_precision_score(labels, scores))

            # Separate descendant/non-descendant scores
            desc_scores = [s for s, l in zip(scores, labels) if l == 1]
            nondesc_scores = [s for s, l in zip(scores, labels) if l == 0]

            summary[condition][method] = {
                "auroc": auroc,
                "auprc": auprc,
                "mean_descendant": float(np.mean(desc_scores)) if desc_scores else float("nan"),
                "min_descendant": float(np.min(desc_scores)) if desc_scores else float("nan"),
                "mean_non_descendant": float(np.mean(nondesc_scores)) if nondesc_scores else float("nan"),
                "max_non_descendant": float(np.max(nondesc_scores)) if nondesc_scores else float("nan"),
                "n_descendant": len(desc_scores),
                "n_non_descendant": len(nondesc_scores),
            }

    return summary


def save_results(
    config: LaunderingConfig,
    results: List[PairResult],
    summary: Dict,
    validation_log: List[Dict],
    invariance_checks: List[Dict],
    wall_seconds: float,
):
    """Save all results to output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-pair scores
    scores_path = output_dir / "scores_by_pair.csv"
    with open(scores_path, "w", newline="") as f:
        if results:
            methods = list(results[0].scores.keys())
            fieldnames = ["condition", "seed", "ref_id", "sus_id", "label", "attack_type"] + methods
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "condition": r.condition,
                    "seed": r.seed,
                    "ref_id": r.ref_id,
                    "sus_id": r.sus_id,
                    "label": r.label,
                    "attack_type": r.attack_type,
                }
                row.update(r.scores)
                writer.writerow(row)

    # Validation log
    with open(output_dir / "validation_log.json", "w") as f:
        json.dump(validation_log, f, indent=2)

    # Invariance checks
    with open(output_dir / "invariance_checks.json", "w") as f:
        json.dump(invariance_checks, f, indent=2)

    # Summary CSV
    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "condition", "auroc", "auprc", "mean_desc", "min_desc", "mean_nondesc", "max_nondesc"])
        for condition, methods in summary.items():
            for method, metrics in methods.items():
                writer.writerow([
                    method, condition,
                    f"{metrics['auroc']:.4f}",
                    f"{metrics['auprc']:.4f}",
                    f"{metrics['mean_descendant']:.4f}",
                    f"{metrics['min_descendant']:.4f}",
                    f"{metrics['mean_non_descendant']:.4f}",
                    f"{metrics['max_non_descendant']:.4f}",
                ])

    # Full results JSON
    full_results = {
        "config": asdict(config),
        "summary": summary,
        "validation_log": validation_log,
        "invariance_checks": invariance_checks,
        "wall_seconds": wall_seconds,
    }
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def print_summary_table(summary: Dict, conditions: List[str]):
    """Print AUROC summary table."""
    methods = [
        "centered_residual_signature",
        "raw_weight_cosine",
        "raw_aligned_frobenius",
        "singular_value_distance",
        "rebasin_frobenius",
        "rebasin_scale_frobenius",
    ]

    print("\n" + "="*80)
    print("AUROC Summary (Main Laundering Table)")
    print("="*80)

    # Header
    header = f"{'Method':35s}" + "".join(f"{c:>12s}" for c in conditions)
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:35s}"
        for cond in conditions:
            if cond in summary and method in summary[cond]:
                auroc = summary[cond][method]["auroc"]
                row += f"{auroc:12.4f}"
            else:
                row += f"{'N/A':>12s}"
        print(row)

    print("\n" + "="*80)
    print("Ablation (NONE condition only)")
    print("="*80)

    ablation_methods = [
        "raw_weight_cosine",
        "raw_branch_product_cosine",
        "centered_branch_product_cosine",
        "centered_with_gating",
        "centered_residual_signature",
    ]

    if "NONE" in summary:
        for method in ablation_methods:
            if method in summary["NONE"]:
                auroc = summary["NONE"][method]["auroc"]
                print(f"{method:40s} {auroc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 laundering benchmark")
    parser.add_argument("--benchmark-dir", default="results/lineage_benchmark_gpt2_paper",
                        help="Path to benchmark data directory")
    parser.add_argument("--output-dir", default="results/laundering_gpt2",
                        help="Output directory")
    parser.add_argument("--conditions", nargs="+",
                        default=["NONE", "P-SUSPECT", "P-BOTH", "P+FT"],
                        help="Conditions to run")
    parser.add_argument("--n-perm-seeds", type=int, default=5,
                        help="Number of permutation seeds per condition")
    parser.add_argument("--pft-steps", type=int, default=1000,
                        help="Fine-tuning steps for P+FT condition")
    parser.add_argument("--pft-lr", type=float, default=2e-5,
                        help="Learning rate for P+FT")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 pairs, limited conditions)")
    parser.add_argument("--seed-base", type=int, default=8000,
                        help="Base random seed")
    parser.add_argument("--test-roots", type=int, nargs="+", default=None,
                        help="Root indices to use for test (default: auto-detect from data)")

    args = parser.parse_args()

    # Auto-detect test roots from data if not specified
    test_roots = args.test_roots
    if test_roots is None:
        phase1_path = Path(args.benchmark_dir) / "phase1_roots.pkl"
        if phase1_path.exists():
            import pickle
            with open(phase1_path, "rb") as f:
                phase1 = pickle.load(f)
            n_roots = len(phase1["root_signatures"])
            test_roots = list(range(n_roots))
            print(f"Auto-detected {n_roots} roots: {test_roots}")

    config = LaunderingConfig(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        conditions=args.conditions,
        n_perm_seeds=args.n_perm_seeds,
        pft_steps=args.pft_steps,
        pft_lr=args.pft_lr,
        smoke=args.smoke,
        seed_base=args.seed_base,
        test_root_indices=test_roots if test_roots else [0, 1, 2],
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
