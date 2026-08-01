#!/usr/bin/env python3
"""Compute top-1 agreement between independent roots for distillation control."""

import json
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from gpt2_lineage_benchmark.data import get_tokenizer, create_dataloaders
from gpt2_lineage_benchmark.model import create_model
from gpt2_lineage_benchmark.config import BenchmarkConfig

def compute_agreement(model_a, model_b, loader, device, max_batches=50):
    """Compute top-1 token agreement between two models."""
    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()

    total_agree = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_a = model_a(input_ids, attention_mask=attention_mask).logits
            logits_b = model_b(input_ids, attention_mask=attention_mask).logits

            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)

            # Only count non-padded positions
            mask = attention_mask.bool()
            agree = ((pred_a == pred_b) & mask).sum().item()
            total = mask.sum().item()

            total_agree += agree
            total_tokens += total

    model_a.cpu()
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return total_agree / total_tokens if total_tokens > 0 else 0


def compute_kl_divergence(model_a, model_b, loader, device, max_batches=50):
    """Compute KL divergence between two models."""
    import torch.nn.functional as F

    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()

    total_kl = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_a = model_a(input_ids, attention_mask=attention_mask).logits
            logits_b = model_b(input_ids, attention_mask=attention_mask).logits

            # KL(B || A) - how different is B from A
            log_probs_a = F.log_softmax(logits_a, dim=-1)
            probs_b = F.softmax(logits_b, dim=-1)

            kl = F.kl_div(log_probs_a, probs_b, reduction='none').sum(dim=-1)

            mask = attention_mask.bool()
            total_kl += (kl * mask).sum().item()
            total_tokens += mask.sum().item()

    model_a.cpu()
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return total_kl / total_tokens if total_tokens > 0 else 0


def main():
    results_dir = Path("experiments/scripts/results/lineage_benchmark_gpt2_paper")
    checkpoint_dir = results_dir / "checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load config
    config = BenchmarkConfig.from_preset("paper")

    # Create tokenizer and val loader
    tokenizer = get_tokenizer()
    loaders = create_dataloaders(
        dataset_name="tinystories",
        tokenizer=tokenizer,
        max_train_samples=5000,
        max_val_samples=2000,
        batch_size=8,
    )
    val_loader = loaders["val"]

    # Load root models
    print("\nLoading root models...")
    roots = []
    for i in range(8):
        root_dir = checkpoint_dir / f"root_{i}"
        # Find the latest epoch checkpoint
        ckpt_path = None
        for epoch in [3, 2, 1, 0]:  # Try latest first
            candidate = root_dir / f"epoch_{epoch}.pt"
            if candidate.exists():
                ckpt_path = candidate
                break

        if ckpt_path is None:
            print(f"  Root {i}: no checkpoint found in {root_dir}")
            continue

        model = create_model(config.model)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # Handle both formats: full checkpoint dict or raw state_dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        roots.append(model)
        print(f"  Root {i}: loaded from {ckpt_path.name}")

    if len(roots) < 2:
        print("Not enough roots found. Trying to estimate from JSON results...")
        # Fallback: use the lineage scores as a proxy
        with open(results_dir / "benchmark_results.json") as f:
            results = json.load(f)

        # Independent pairs already have lineage scores near 0
        # We can report that as a baseline
        indep_pairs = [p for p in results["pairs"] if p["attack_type"] == "independent"]
        print(f"\nIndependent root pairs: {len(indep_pairs)}")
        print(f"Lineage scores: mean={sum(p['lineage'] for p in indep_pairs)/len(indep_pairs):.4f}")
        return

    # Compute pairwise agreement between independent roots
    print("\n=== INDEPENDENT ROOT AGREEMENT ===")
    agreements = []
    kl_divs = []

    for i in range(len(roots)):
        for j in range(i+1, len(roots)):
            agree = compute_agreement(roots[i], roots[j], val_loader, device, max_batches=30)
            kl = compute_kl_divergence(roots[i], roots[j], val_loader, device, max_batches=30)
            agreements.append(agree)
            kl_divs.append(kl)
            print(f"  Root {i} vs Root {j}: agreement={agree*100:.1f}%, KL={kl:.3f}")

    print(f"\nIndependent roots:")
    print(f"  Agreement: mean={sum(agreements)/len(agreements)*100:.1f}%, range={min(agreements)*100:.1f}%-{max(agreements)*100:.1f}%")
    print(f"  KL div: mean={sum(kl_divs)/len(kl_divs):.3f}, range={min(kl_divs):.3f}-{max(kl_divs):.3f}")

    # Compare with distilled student results
    with open(results_dir / "benchmark_results.json") as f:
        results = json.load(f)

    distilled = results["distilled_students"]
    dist_agreements = [d["quality_metrics"]["top1_agreement"] for d in distilled]
    dist_kls = [d["quality_metrics"]["kl_divergence"] for d in distilled]

    print(f"\nDistilled students (from JSON):")
    print(f"  Agreement: mean={sum(dist_agreements)/len(dist_agreements)*100:.1f}%, range={min(dist_agreements)*100:.1f}%-{max(dist_agreements)*100:.1f}%")
    print(f"  KL div: mean={sum(dist_kls)/len(dist_kls):.3f}, range={min(dist_kls):.3f}-{max(dist_kls):.3f}")

    # Save results
    control_results = {
        "independent": {
            "n_pairs": len(agreements),
            "agreement_mean": sum(agreements)/len(agreements),
            "agreement_min": min(agreements),
            "agreement_max": max(agreements),
            "kl_mean": sum(kl_divs)/len(kl_divs),
            "kl_min": min(kl_divs),
            "kl_max": max(kl_divs),
        },
        "distilled": {
            "n": len(dist_agreements),
            "agreement_mean": sum(dist_agreements)/len(dist_agreements),
            "agreement_min": min(dist_agreements),
            "agreement_max": max(dist_agreements),
            "kl_mean": sum(dist_kls)/len(dist_kls),
            "kl_min": min(dist_kls),
            "kl_max": max(dist_kls),
        }
    }

    output_path = results_dir / "distillation_control.json"
    with open(output_path, "w") as f:
        json.dump(control_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
