"""Main orchestration script for the GPT-2 lineage benchmark."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np

from .config import BenchmarkConfig, ModelConfig, TrainingConfig
from .data import create_dataloaders, create_domain_shift_loader, get_tokenizer
from .model import create_model, load_checkpoint, get_model_info
from .train import train_root
from .descendants import generate_all_descendants
from .distillation import generate_distilled_students
from .extraction import extract_branch_products
from .evaluation import (
    compute_lineage_score,
    choose_tau_s,
    evaluate_lineage_benchmark,
    compute_gap_z,
)


def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
    device: str = "cuda",
    skip_training: bool = False,
    verbose: bool = True,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Run the full GPT-2 lineage benchmark.

    Args:
        config: Benchmark configuration
        device: GPU device (cuda, mps, or cpu)
        skip_training: Load roots from checkpoints if available
        verbose: Print progress
        num_workers: Number of parallel workers for CPU-bound tasks
    """
    if config is None:
        config = BenchmarkConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = {
        "config": asdict(config),
        "roots": [],
        "descendants": [],
        "distilled_students": [],
        "pairs": [],
        "metrics": {},
    }

    # Get tokenizer and dataloaders
    if verbose:
        print("Loading data...")
    tokenizer = get_tokenizer()
    dataloaders = create_dataloaders(
        dataset_name=config.dataset,
        tokenizer=tokenizer,
        max_length=config.model.max_seq_len,
        batch_size=config.training.batch_size,
        max_train_samples=config.max_train_samples,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    domain_shift_loader = create_domain_shift_loader(
        tokenizer=tokenizer,
        max_length=config.model.max_seq_len,
        batch_size=config.training.batch_size,
    )

    if verbose:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Model config: {get_model_info(create_model(config.model, seed=0, device='cpu'))}")

    # Phase 1: Train root models
    roots = []
    root_Ms = []

    for root_idx in range(config.n_roots):
        split = config.get_split(root_idx)
        root_ckpt = checkpoint_dir / f"root_{root_idx}" / f"epoch_{config.training.epochs}.pt"

        if skip_training and root_ckpt.exists():
            if verbose:
                print(f"\n[root-{root_idx}] ({split}) Loading from checkpoint...")
            model, epoch, metrics = load_checkpoint(root_ckpt, config.model, device)
            root_info = {
                "root_idx": root_idx,
                "split": split,
                "final_val_ppl": metrics.get("val_ppl", 0.0) if metrics else 0.0,
                "loaded_from_checkpoint": True,
            }
        else:
            if verbose:
                print(f"\n[root-{root_idx}] ({split}) Training from scratch...")
            result = train_root(
                root_idx=root_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=config.model,
                train_config=config.training,
                checkpoint_dir=checkpoint_dir,
                device=device,
                verbose=verbose,
            )
            model = result["model"]
            root_info = {
                "root_idx": root_idx,
                "split": split,
                "final_val_ppl": result["final_val_ppl"],
                "training_time_seconds": result["training_time_seconds"],
                "seed": result["seed"],
            }

        # Extract branch products
        Ms = extract_branch_products(model)
        root_Ms.append(Ms)
        root_info["mean_diag_score"] = float(np.mean([
            abs(np.trace(M)) / np.linalg.norm(M, 'fro') for M in Ms
        ]))

        roots.append({"info": root_info, "model": model, "Ms": Ms})
        results["roots"].append(root_info)

        if verbose:
            print(f"  Mean diag score: {root_info['mean_diag_score']:.4f}")

    # Compute tau_s from all roots
    tau_s = choose_tau_s(root_Ms)
    results["tau_s"] = tau_s
    if verbose:
        print(f"\ntau_s = {tau_s:.4f}")

    # Phase 2: Generate descendants
    all_descendants = []
    for root_idx, root in enumerate(roots):
        if verbose:
            print(f"\n[root-{root_idx}] Generating descendants...")

        descendants = generate_all_descendants(
            parent=root["model"],
            root_idx=root_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            domain_shift_loader=domain_shift_loader,
            config=config.descendant,
            device=device,
            verbose=verbose,
            parallel_cpu=True,
        )

        # Extract branch products in parallel (CPU-bound)
        def extract_and_annotate(desc):
            desc["Ms"] = extract_branch_products(desc["model"])
            desc["mean_diag_score"] = float(np.mean([
                abs(np.trace(M)) / np.linalg.norm(M, 'fro') for M in desc["Ms"]
            ]))
            return desc

        if verbose:
            print(f"  Extracting signatures for {len(descendants)} descendants...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            descendants = list(executor.map(extract_and_annotate, descendants))

        for desc in descendants:
            # Remove model to save memory, keep Ms for scoring
            desc_info = {k: v for k, v in desc.items() if k != "model"}
            desc_info.pop("Ms", None)
            results["descendants"].append(desc_info)
            del desc["model"]  # Free GPU/CPU memory
            all_descendants.append(desc)

        # Clear parent model from GPU
        root["model"].cpu()
        torch.cuda.empty_cache()

    # Phase 3: Generate distilled students
    all_students = []
    for root_idx, root in enumerate(roots):
        if verbose:
            print(f"\n[root-{root_idx}] Generating distilled students...")

        root["model"].to(device)
        students = generate_distilled_students(
            teacher=root["model"],
            root_idx=root_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.distillation,
            model_config=config.model,
            device=device,
            verbose=verbose,
        )

        # Extract branch products in parallel
        def extract_student(student):
            student["Ms"] = extract_branch_products(student["model"])
            student["mean_diag_score"] = float(np.mean([
                abs(np.trace(M)) / np.linalg.norm(M, 'fro') for M in student["Ms"]
            ]))
            return student

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            students = list(executor.map(extract_student, students))

        for student in students:
            student_info = {
                k: v for k, v in student.items() if k not in ["model", "Ms"]
            }
            results["distilled_students"].append(student_info)
            del student["model"]
            all_students.append(student)

        root["model"].cpu()
        torch.cuda.empty_cache()

    # Phase 4: Score all pairs (parallelized)
    if verbose:
        print("\n" + "="*60)
        print("Scoring all pairs (parallel)...")

    # Build list of scoring tasks
    scoring_tasks = []
    for root_idx, root in enumerate(roots):
        ref_Ms = root["Ms"]
        split = root["info"]["split"]

        # Score descendants of this root
        for desc in all_descendants:
            if desc["root_idx"] == root_idx:
                scoring_tasks.append({
                    "ref_Ms": ref_Ms,
                    "sus_Ms": desc["Ms"],
                    "reference": f"root_{root_idx}",
                    "suspect": desc["id"],
                    "label": "descendant",
                    "attack_type": desc["type"],
                    "split": split,
                })

        # Score distilled students
        for student in all_students:
            if student["teacher_root_idx"] == root_idx:
                scoring_tasks.append({
                    "ref_Ms": ref_Ms,
                    "sus_Ms": student["Ms"],
                    "reference": f"root_{root_idx}",
                    "suspect": student["id"],
                    "label": "non_descendant",
                    "attack_type": "distilled_student",
                    "split": split,
                    "quality_metrics": student.get("quality_metrics", {}),
                })

        # Score other roots as non-descendants
        for other_idx, other_root in enumerate(roots):
            if other_idx != root_idx:
                scoring_tasks.append({
                    "ref_Ms": ref_Ms,
                    "sus_Ms": other_root["Ms"],
                    "reference": f"root_{root_idx}",
                    "suspect": f"root_{other_idx}",
                    "label": "non_descendant",
                    "attack_type": "independent",
                    "split": split,
                })

    # Score in parallel using threads (CPU-bound numpy ops)
    def score_task(task: Dict) -> Dict:
        L_score, _, _ = compute_lineage_score(
            task["ref_Ms"], task["sus_Ms"], tau_s
        )
        result = {k: v for k, v in task.items() if k not in ["ref_Ms", "sus_Ms"]}
        result["lineage"] = L_score
        return result

    pairs = []
    n_score_workers = min(num_workers * 2, len(scoring_tasks))
    with ThreadPoolExecutor(max_workers=n_score_workers) as executor:
        pairs = list(executor.map(score_task, scoring_tasks))

    if verbose:
        print(f"  Scored {len(pairs)} pairs using {n_score_workers} workers")

    results["pairs"] = pairs

    # Phase 5: Evaluate by split
    if verbose:
        print("\nEvaluating metrics...")

    for split_name in ["calibration", "development", "test"]:
        split_pairs = [p for p in pairs if p["split"] == split_name]
        if split_pairs:
            metrics = evaluate_lineage_benchmark(split_pairs)
            results["metrics"][split_name] = metrics

            if verbose:
                print(f"\n[{split_name.upper()}] AUROC={metrics['auroc']:.4f} "
                      f"({metrics['auroc_ci']['ci_lower']:.4f}, {metrics['auroc_ci']['ci_upper']:.4f})")
                print(f"  TPR@1%FPR={metrics['tpr_at_1pct_fpr']:.2%}, "
                      f"TPR@10%FPR={metrics['tpr_at_10pct_fpr']:.2%}")
                print(f"  Pos mean={metrics['pos_mean']:.4f}, Neg mean={metrics['neg_mean']:.4f}")

    # Overall metrics
    all_metrics = evaluate_lineage_benchmark(pairs)
    results["metrics"]["overall"] = all_metrics

    # Gap-Z for distillation
    related_scores = [p["lineage"] for p in pairs if p["label"] == "descendant"]
    distilled_scores = [p["lineage"] for p in pairs if p["attack_type"] == "distilled_student"]
    if related_scores and distilled_scores:
        gap_z = compute_gap_z(related_scores, distilled_scores)
        results["metrics"]["gap_z_distilled"] = gap_z
        if verbose:
            print(f"\nGap-Z (related vs distilled): {gap_z:.1f}")

    # Timing
    results["total_seconds"] = time.time() - t0
    if verbose:
        print(f"\nTotal time: {results['total_seconds']/3600:.1f} hours")

    # Save results
    output_path = output_dir / "benchmark_results.json"

    # Remove numpy arrays for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)

    if verbose:
        print(f"\nSaved results to {output_path}")
        print("\n" + "="*60)
        print("HEADLINE RESULTS")
        print("="*60)
        print(f"Overall AUROC: {all_metrics['auroc']:.4f}")
        print(f"Test AUROC: {results['metrics'].get('test', {}).get('auroc', 'N/A')}")
        if gap_z:
            print(f"Gap-Z (distillation): {gap_z:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run GPT-2 lineage benchmark")
    parser.add_argument("--n-calibration-roots", type=int, default=3)
    parser.add_argument("--n-development-roots", type=int, default=3)
    parser.add_argument("--n-test-roots", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="results/lineage_benchmark_gpt2")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load roots from checkpoints if available")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples (for debugging)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers for CPU tasks")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    config = BenchmarkConfig(
        n_calibration_roots=args.n_calibration_roots,
        n_development_roots=args.n_development_roots,
        n_test_roots=args.n_test_roots,
        output_dir=args.output_dir,
        checkpoint_dir=f"{args.output_dir}/checkpoints",
        max_train_samples=args.max_train_samples,
    )
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size

    run_benchmark(
        config=config,
        device=args.device,
        skip_training=args.skip_training,
        verbose=not args.quiet,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
