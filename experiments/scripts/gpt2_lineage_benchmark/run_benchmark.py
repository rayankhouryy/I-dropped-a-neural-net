"""Main orchestration script for the GPT-2 lineage benchmark."""
import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

from .config import BenchmarkConfig
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


def run_phase_roots(
    config: BenchmarkConfig,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase 1: Train root models (~1 hour for paper preset)."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if verbose:
        print("=" * 60)
        print("PHASE 1: Training root models")
        print("=" * 60)
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

    if verbose:
        info = get_model_info(create_model(config.model, seed=0, device='cpu'))
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Model: {info['total_params_millions']:.1f}M params")

    roots_info = []
    root_signatures = []

    for root_idx in range(config.n_roots):
        split = config.get_split(root_idx)
        ckpt_path = checkpoint_dir / f"root_{root_idx}"
        final_ckpt = ckpt_path / f"epoch_{config.training.epochs}.pt"

        if final_ckpt.exists():
            if verbose:
                print(f"\n[root-{root_idx}] ({split}) Already trained, loading...")
            model, _, metrics = load_checkpoint(final_ckpt, config.model, device)
            root_info = {
                "root_idx": root_idx,
                "split": split,
                "final_val_ppl": metrics.get("val_ppl", 0.0) if metrics else 0.0,
            }
        else:
            if verbose:
                print(f"\n[root-{root_idx}] ({split}) Training...")
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
            }

        # Extract and save signatures
        Ms = extract_branch_products(model)
        root_info["mean_diag_score"] = float(np.mean([
            abs(np.trace(M)) / np.linalg.norm(M, 'fro') for M in Ms
        ]))
        roots_info.append(root_info)
        root_signatures.append(Ms)

        if verbose:
            print(f"  Mean diag score: {root_info['mean_diag_score']:.4f}")

        # Free GPU memory
        model.cpu()
        del model
        torch.cuda.empty_cache()

    # Save phase 1 results
    tau_s = choose_tau_s(root_signatures)
    phase1_data = {
        "roots_info": roots_info,
        "root_signatures": root_signatures,
        "tau_s": tau_s,
        "config": asdict(config),
    }
    phase1_path = output_dir / "phase1_roots.pkl"
    with open(phase1_path, "wb") as f:
        pickle.dump(phase1_data, f)

    if verbose:
        print(f"\ntau_s = {tau_s:.4f}")
        print(f"Phase 1 completed in {(time.time()-t0)/60:.1f} min")
        print(f"Saved to {phase1_path}")

    return phase1_data


def run_phase_descendants(
    config: BenchmarkConfig,
    device: str = "cuda",
    verbose: bool = True,
    num_workers: int = 4,
    fast_descendants: bool = False,
    save_models: bool = False,
) -> Dict[str, Any]:
    """Phase 2: Generate descendants + distilled students.

    Args:
        fast_descendants: Use 5K samples for descendants (not distillation)
        save_models: Save full model state_dicts for baseline comparisons
    """
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)

    # Load phase 1 data
    phase1_path = output_dir / "phase1_roots.pkl"
    if not phase1_path.exists():
        raise RuntimeError(f"Phase 1 not complete. Run --phase roots first.")

    with open(phase1_path, "rb") as f:
        phase1_data = pickle.load(f)

    roots_info = phase1_data["roots_info"]
    root_signatures = phase1_data["root_signatures"]

    t0 = time.time()

    if verbose:
        print("=" * 60)
        print("PHASE 2: Generating descendants and distilled students")
        print("=" * 60)
        if fast_descendants:
            print("FAST MODE: Using 5K samples for descendants")
        print("Loading data...")

    tokenizer = get_tokenizer()

    # Full data for distillation (need good imitation)
    full_dataloaders = create_dataloaders(
        dataset_name=config.dataset,
        tokenizer=tokenizer,
        max_length=config.model.max_seq_len,
        batch_size=config.training.batch_size,
        max_train_samples=config.max_train_samples,
    )

    # Smaller data for descendants if fast mode
    if fast_descendants:
        desc_dataloaders = create_dataloaders(
            dataset_name=config.dataset,
            tokenizer=tokenizer,
            max_length=config.model.max_seq_len,
            batch_size=config.training.batch_size,
            max_train_samples=5000,
        )
        train_loader = desc_dataloaders["train"]
        if verbose:
            print(f"Descendant batches: {len(train_loader)}")
    else:
        train_loader = full_dataloaders["train"]

    val_loader = full_dataloaders["val"]

    domain_shift_loader = None
    if config.descendant.cont_pt_shift_epochs:
        domain_shift_loader = create_domain_shift_loader(
            tokenizer=tokenizer,
            max_length=config.model.max_seq_len,
            batch_size=config.training.batch_size,
        )

    all_descendants = []
    all_students = []

    for root_idx, root_info in enumerate(roots_info):
        # Load root model
        ckpt = checkpoint_dir / f"root_{root_idx}" / f"epoch_{config.training.epochs}.pt"
        model, _, _ = load_checkpoint(ckpt, config.model, device)

        if verbose:
            print(f"\n[root-{root_idx}] Generating descendants...")

        # Generate descendants
        descendants = generate_all_descendants(
            parent=model,
            root_idx=root_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            domain_shift_loader=domain_shift_loader,
            config=config.descendant,
            device=device,
            verbose=verbose,
            parallel_cpu=True,
        )

        # Extract signatures and optionally save models
        models_dir = Path(config.output_dir) / "models"
        if save_models:
            models_dir.mkdir(parents=True, exist_ok=True)

        def extract_desc(desc):
            desc["Ms"] = extract_branch_products(desc["model"])
            desc["mean_diag_score"] = float(np.mean([
                abs(np.trace(M)) / np.linalg.norm(M, 'fro') for M in desc["Ms"]
            ]))
            if save_models:
                # Save model state dict
                model_path = models_dir / f"{desc['id']}.pt"
                torch.save(desc["model"].state_dict(), model_path)
            del desc["model"]
            return desc

        if verbose:
            print(f"  Extracting {len(descendants)} descendant signatures...")
            if save_models:
                print(f"  Saving models to {models_dir}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            descendants = list(executor.map(extract_desc, descendants))
        all_descendants.extend(descendants)

        # Generate distilled students
        if verbose:
            print(f"  Generating distilled students...")

        # Use full data for distillation to ensure good imitation
        students = generate_distilled_students(
            teacher=model,
            root_idx=root_idx,
            train_loader=full_dataloaders["train"],
            val_loader=val_loader,
            config=config.distillation,
            model_config=config.model,
            device=device,
            verbose=verbose,
        )

        def extract_student(s):
            s["Ms"] = extract_branch_products(s["model"])
            s["mean_diag_score"] = float(np.mean([
                abs(np.trace(M)) / np.linalg.norm(M, 'fro') for M in s["Ms"]
            ]))
            if save_models:
                model_path = models_dir / f"{s['id']}.pt"
                torch.save(s["model"].state_dict(), model_path)
            del s["model"]
            return s

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            students = list(executor.map(extract_student, students))
        all_students.extend(students)

        model.cpu()
        del model
        torch.cuda.empty_cache()

    # Save phase 2 results
    phase2_data = {
        "descendants": all_descendants,
        "students": all_students,
    }
    phase2_path = output_dir / "phase2_descendants.pkl"
    with open(phase2_path, "wb") as f:
        pickle.dump(phase2_data, f)

    if verbose:
        print(f"\nPhase 2 completed in {(time.time()-t0)/60:.1f} min")
        print(f"  {len(all_descendants)} descendants, {len(all_students)} students")
        print(f"Saved to {phase2_path}")

    return phase2_data


def run_phase_evaluate(
    config: BenchmarkConfig,
    verbose: bool = True,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Phase 3: Score all pairs and compute metrics (~15 min)."""
    output_dir = Path(config.output_dir)

    # Load phase 1 and 2 data
    phase1_path = output_dir / "phase1_roots.pkl"
    phase2_path = output_dir / "phase2_descendants.pkl"

    if not phase1_path.exists():
        raise RuntimeError("Phase 1 not complete. Run --phase roots first.")
    if not phase2_path.exists():
        raise RuntimeError("Phase 2 not complete. Run --phase descendants first.")

    with open(phase1_path, "rb") as f:
        phase1_data = pickle.load(f)
    with open(phase2_path, "rb") as f:
        phase2_data = pickle.load(f)

    roots_info = phase1_data["roots_info"]
    root_signatures = phase1_data["root_signatures"]
    tau_s = phase1_data["tau_s"]
    all_descendants = phase2_data["descendants"]
    all_students = phase2_data["students"]

    t0 = time.time()

    if verbose:
        print("=" * 60)
        print("PHASE 3: Scoring and evaluation")
        print("=" * 60)
        print(f"tau_s = {tau_s:.4f}")

    # Build scoring tasks
    scoring_tasks = []
    for root_idx, root_info in enumerate(roots_info):
        ref_Ms = root_signatures[root_idx]
        split = root_info["split"]

        # Descendants
        for desc in all_descendants:
            if desc["root_idx"] == root_idx:
                scoring_tasks.append({
                    "ref_Ms": ref_Ms, "sus_Ms": desc["Ms"],
                    "reference": f"root_{root_idx}", "suspect": desc["id"],
                    "label": "descendant", "attack_type": desc["type"],
                    "split": split,
                })

        # Distilled students
        for student in all_students:
            if student["teacher_root_idx"] == root_idx:
                scoring_tasks.append({
                    "ref_Ms": ref_Ms, "sus_Ms": student["Ms"],
                    "reference": f"root_{root_idx}", "suspect": student["id"],
                    "label": "non_descendant", "attack_type": "distilled_student",
                    "split": split,
                    "quality_metrics": student.get("quality_metrics", {}),
                })

        # Cross-root pairs
        for other_idx in range(len(roots_info)):
            if other_idx != root_idx:
                scoring_tasks.append({
                    "ref_Ms": ref_Ms, "sus_Ms": root_signatures[other_idx],
                    "reference": f"root_{root_idx}", "suspect": f"root_{other_idx}",
                    "label": "non_descendant", "attack_type": "independent",
                    "split": split,
                })

    if verbose:
        print(f"Scoring {len(scoring_tasks)} pairs...")

    def score_task(task):
        L_score, _, _ = compute_lineage_score(task["ref_Ms"], task["sus_Ms"], tau_s)
        result = {k: v for k, v in task.items() if k not in ["ref_Ms", "sus_Ms"]}
        result["lineage"] = L_score
        return result

    with ThreadPoolExecutor(max_workers=num_workers * 2) as executor:
        pairs = list(executor.map(score_task, scoring_tasks))

    # Compute metrics
    results = {
        "config": phase1_data["config"],
        "tau_s": tau_s,
        "roots": roots_info,
        "descendants": [{k: v for k, v in d.items() if k != "Ms"}
                        for d in all_descendants],
        "distilled_students": [{k: v for k, v in s.items() if k != "Ms"}
                               for s in all_students],
        "pairs": pairs,
        "metrics": {},
    }

    for split_name in ["calibration", "development", "test"]:
        split_pairs = [p for p in pairs if p["split"] == split_name]
        if split_pairs:
            metrics = evaluate_lineage_benchmark(split_pairs)
            results["metrics"][split_name] = metrics
            if verbose:
                ci = metrics['auroc_ci']
                print(f"\n[{split_name.upper()}] AUROC={metrics['auroc']:.4f} "
                      f"({ci['ci_lower']:.4f}, {ci['ci_upper']:.4f})")

    all_metrics = evaluate_lineage_benchmark(pairs)
    results["metrics"]["overall"] = all_metrics

    # Gap-Z
    related = [p["lineage"] for p in pairs if p["label"] == "descendant"]
    distilled = [p["lineage"] for p in pairs if p["attack_type"] == "distilled_student"]
    gap_z = compute_gap_z(related, distilled) if related and distilled else None
    if gap_z:
        results["metrics"]["gap_z_distilled"] = gap_z

    results["total_seconds"] = time.time() - t0

    # Save final results
    def clean_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(v) for v in obj]
        return obj

    output_path = output_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(clean_json(results), f, indent=2)

    if verbose:
        print(f"\nPhase 3 completed in {(time.time()-t0)/60:.1f} min")
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Overall AUROC: {all_metrics['auroc']:.4f}")
        test_auroc = results['metrics'].get('test', {}).get('auroc', 'N/A')
        print(f"Test AUROC: {test_auroc}")
        if gap_z:
            print(f"Gap-Z (distillation): {gap_z:.1f}")
        print(f"\nSaved to {output_path}")

    return results


def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
    device: str = "cuda",
    verbose: bool = True,
    num_workers: int = 4,
    phase: Optional[str] = None,
    fast_descendants: bool = False,
    save_models: bool = False,
) -> Dict[str, Any]:
    """Run the GPT-2 lineage benchmark (all phases or specific phase)."""
    if config is None:
        config = BenchmarkConfig()

    if phase == "roots":
        return run_phase_roots(config, device, verbose)
    elif phase == "descendants":
        return run_phase_descendants(
            config, device, verbose, num_workers, fast_descendants, save_models
        )
    elif phase == "evaluate":
        return run_phase_evaluate(config, verbose, num_workers)
    else:
        # Run all phases
        run_phase_roots(config, device, verbose)
        run_phase_descendants(
            config, device, verbose, num_workers, fast_descendants, save_models
        )
        return run_phase_evaluate(config, verbose, num_workers)


def main():
    parser = argparse.ArgumentParser(description="Run GPT-2 lineage benchmark")
    parser.add_argument("--preset", choices=["smoke", "paper", "full"],
                        help="Preset config (smoke=2min, paper=3-4hr, full=15-20hr)")
    parser.add_argument("--phase", choices=["roots", "descendants", "evaluate"],
                        help="Run specific phase only")
    parser.add_argument("--n-calibration-roots", type=int, default=None)
    parser.add_argument("--n-development-roots", type=int, default=None)
    parser.add_argument("--n-test-roots", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fast-descendants", action="store_true",
                        help="Use 5K samples for descendants (not distillation)")
    parser.add_argument("--save-models", action="store_true",
                        help="Save descendant/student model weights for baseline comparisons")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.preset:
        config = BenchmarkConfig.from_preset(args.preset)
        default_output = f"results/lineage_benchmark_gpt2_{args.preset}"
    else:
        config = BenchmarkConfig()
        default_output = "results/lineage_benchmark_gpt2"

    if args.n_calibration_roots is not None:
        config.n_calibration_roots = args.n_calibration_roots
    if args.n_development_roots is not None:
        config.n_development_roots = args.n_development_roots
    if args.n_test_roots is not None:
        config.n_test_roots = args.n_test_roots
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples

    config.output_dir = args.output_dir or default_output
    config.checkpoint_dir = f"{config.output_dir}/checkpoints"

    run_benchmark(
        config=config,
        device=args.device,
        verbose=not args.quiet,
        num_workers=args.num_workers,
        phase=args.phase,
        fast_descendants=args.fast_descendants,
        save_models=args.save_models,
    )


if __name__ == "__main__":
    main()
