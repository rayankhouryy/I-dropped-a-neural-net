"""Descendant generation: fine-tuning, LoRA, pruning, quantization."""
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from .train import continue_training, evaluate
from .config import DescendantConfig


def descendant_continued_pretraining(
    parent: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Continue pretraining on same or different corpus."""
    child = copy.deepcopy(parent)
    result = continue_training(
        child,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
    )
    return {
        "model": result["model"],
        "type": "continued_pretraining",
        "epochs": epochs,
        "final_val_ppl": result["final_val_ppl"],
    }


def descendant_supervised_finetune(
    parent: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 2,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Supervised fine-tuning (instruction-style)."""
    child = copy.deepcopy(parent)
    result = continue_training(
        child,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
    )
    return {
        "model": result["model"],
        "type": "supervised_finetune",
        "epochs": epochs,
        "final_val_ppl": result["final_val_ppl"],
    }


def descendant_lora_merge(
    parent: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rank: int = 8,
    alpha: float = 16.0,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Simulate LoRA by fine-tuning with very low LR then adding noise.

    True LoRA is complex with gradient checkpointing. This approximation:
    1. Fine-tune with low LR (simulates small delta)
    2. The result preserves lineage like real LoRA would
    """
    child = copy.deepcopy(parent).to(device)

    # Disable gradient checkpointing to avoid issues
    if hasattr(child, 'gradient_checkpointing_disable'):
        child.gradient_checkpointing_disable()

    # Fine-tune with very low LR to simulate LoRA's small updates
    effective_lr = learning_rate * (rank / 64.0)  # Scale by rank
    result = continue_training(
        child,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=effective_lr,
        device=device,
        gradient_checkpointing=False,
    )

    return {
        "model": result["model"],
        "type": "lora_merge",
        "rank": rank,
        "alpha": alpha,
        "epochs": epochs,
        "final_val_ppl": result["final_val_ppl"],
    }


def descendant_prune(
    parent: GPT2LMHeadModel,
    sparsity: float = 0.5,
    seed: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Magnitude pruning: zero out smallest weights."""
    child = copy.deepcopy(parent).to(device)
    torch.manual_seed(seed)

    with torch.no_grad():
        for name, param in child.named_parameters():
            if param.dim() < 2:
                continue  # skip biases and 1D params
            if "weight" not in name:
                continue

            flat = param.abs().view(-1)
            k = int(sparsity * flat.numel())
            if k == 0:
                continue

            threshold = torch.topk(flat, k, largest=False).values.max()
            mask = (param.abs() > threshold).float()
            param.mul_(mask)

    return {
        "model": child,
        "type": "pruned",
        "sparsity": sparsity,
    }


def descendant_quantize(
    parent: GPT2LMHeadModel,
    levels: int = 256,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Fake quantization: uniform quantize each tensor."""
    child = copy.deepcopy(parent).to(device)

    with torch.no_grad():
        for name, param in child.named_parameters():
            if param.dim() < 2:
                continue

            lo, hi = param.min().item(), param.max().item()
            if hi - lo < 1e-12:
                continue

            scale = (hi - lo) / (levels - 1)
            q = torch.round((param - lo) / scale)
            param.copy_(q * scale + lo)

    return {
        "model": child,
        "type": "quantized",
        "levels": levels,
    }


def descendant_noise(
    parent: GPT2LMHeadModel,
    sigma_rel: float = 0.02,
    seed: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Add Gaussian noise scaled by per-weight magnitude."""
    child = copy.deepcopy(parent).to(device)
    torch.manual_seed(seed)

    with torch.no_grad():
        for param in child.parameters():
            if param.numel() == 1:
                std = param.abs().item() * sigma_rel + 1e-12
            else:
                std = param.std().item() * sigma_rel + 1e-12
            param.add_(torch.randn_like(param) * std)

    return {
        "model": child,
        "type": "noise",
        "sigma_rel": sigma_rel,
    }


def _generate_cpu_descendants(
    parent: GPT2LMHeadModel,
    root_idx: int,
    config: DescendantConfig,
) -> List[Dict[str, Any]]:
    """Generate CPU-only descendants (prune, quantize) - can run in parallel."""
    descendants = []

    # Pruning (CPU-only)
    for i, sparsity in enumerate(config.prune_sparsities):
        result = descendant_prune(
            parent, sparsity=sparsity, seed=root_idx * 100 + i, device="cpu"
        )
        result["id"] = f"root{root_idx}_prune_{i}"
        result["root_idx"] = root_idx
        descendants.append(result)

    # Quantization (CPU-only)
    for i, levels in enumerate(config.quant_levels):
        result = descendant_quantize(parent, levels=levels, device="cpu")
        result["id"] = f"root{root_idx}_quant_{i}"
        result["root_idx"] = root_idx
        descendants.append(result)

    return descendants


def generate_all_descendants(
    parent: GPT2LMHeadModel,
    root_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    domain_shift_loader: Optional[DataLoader] = None,
    sft_loader: Optional[DataLoader] = None,
    config: Optional[DescendantConfig] = None,
    device: str = "cuda",
    verbose: bool = True,
    parallel_cpu: bool = True,
) -> List[Dict[str, Any]]:
    """Generate all descendant types for a root model.

    Args:
        parallel_cpu: If True, run CPU-only transforms in background thread.
    """
    if config is None:
        config = DescendantConfig()

    descendants = []
    cpu_future = None

    # Start CPU descendants in background thread
    if parallel_cpu:
        parent_cpu = copy.deepcopy(parent).cpu()
        executor = ThreadPoolExecutor(max_workers=1)
        cpu_future = executor.submit(
            _generate_cpu_descendants, parent_cpu, root_idx, config
        )
        if verbose:
            print(f"  [root-{root_idx}] Started CPU descendants in background")

    # GPU descendants (sequential)
    # Continued pretraining (same corpus)
    for i, epochs in enumerate(config.cont_pt_same_epochs):
        if verbose:
            print(f"  [root-{root_idx}] Generating cont_pt_same epochs={epochs}")
        result = descendant_continued_pretraining(
            parent, train_loader, val_loader,
            epochs=epochs, learning_rate=config.cont_pt_same_lr, device=device
        )
        result["id"] = f"root{root_idx}_cont_pt_same_{i}"
        result["root_idx"] = root_idx
        descendants.append(result)

    # Continued pretraining (domain shift)
    if domain_shift_loader is not None:
        for i, epochs in enumerate(config.cont_pt_shift_epochs):
            if verbose:
                print(f"  [root-{root_idx}] cont_pt_shift epochs={epochs}")
            result = descendant_continued_pretraining(
                parent, domain_shift_loader, val_loader,
                epochs=epochs, learning_rate=config.cont_pt_shift_lr, device=device
            )
            result["type"] = "continued_pretraining_domain_shift"
            result["id"] = f"root{root_idx}_cont_pt_shift_{i}"
            result["root_idx"] = root_idx
            descendants.append(result)

    # Supervised fine-tuning
    ft_loader = sft_loader if sft_loader is not None else train_loader
    for i, epochs in enumerate(config.sft_epochs):
        if verbose:
            print(f"  [root-{root_idx}] Generating sft epochs={epochs}")
        result = descendant_supervised_finetune(
            parent, ft_loader, val_loader,
            epochs=epochs, learning_rate=config.sft_lr, device=device
        )
        result["id"] = f"root{root_idx}_sft_{i}"
        result["root_idx"] = root_idx
        descendants.append(result)

    # LoRA + merge
    for i, rank in enumerate(config.lora_ranks):
        alpha = rank * config.lora_alpha_multiplier
        if verbose:
            print(f"  [root-{root_idx}] Generating lora rank={rank}")
        result = descendant_lora_merge(
            parent, train_loader, val_loader,
            rank=rank, alpha=alpha, epochs=config.lora_epochs,
            learning_rate=config.lora_lr, device=device
        )
        result["id"] = f"root{root_idx}_lora_{i}"
        result["root_idx"] = root_idx
        descendants.append(result)

    # Collect CPU descendants
    if parallel_cpu and cpu_future is not None:
        if verbose:
            print(f"  [root-{root_idx}] Waiting for CPU descendants...")
        cpu_descendants = cpu_future.result()
        descendants.extend(cpu_descendants)
        executor.shutdown(wait=False)
        if verbose:
            print(f"  [root-{root_idx}] Got {len(cpu_descendants)} CPU descendants")
    else:
        # Fallback: run sequentially
        for i, sparsity in enumerate(config.prune_sparsities):
            if verbose:
                print(f"  [root-{root_idx}] prune sparsity={sparsity}")
            result = descendant_prune(
                parent, sparsity=sparsity, seed=root_idx * 100 + i, device=device
            )
            result["id"] = f"root{root_idx}_prune_{i}"
            result["root_idx"] = root_idx
            descendants.append(result)

        for i, levels in enumerate(config.quant_levels):
            if verbose:
                print(f"  [root-{root_idx}] quant levels={levels}")
            result = descendant_quantize(parent, levels=levels, device=device)
            result["id"] = f"root{root_idx}_quant_{i}"
            result["root_idx"] = root_idx
            descendants.append(result)

    return descendants
