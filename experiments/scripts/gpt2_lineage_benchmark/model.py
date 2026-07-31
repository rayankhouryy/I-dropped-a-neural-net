"""GPT-2-Small-Lite model creation and utilities."""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Optional
from pathlib import Path

from .config import ModelConfig


def create_model(
    config: Optional[ModelConfig] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> GPT2LMHeadModel:
    """Create a fresh GPT-2-Small-Lite model."""
    if config is None:
        config = ModelConfig()

    if seed is not None:
        torch.manual_seed(seed)

    hf_config = config.to_hf_config()
    model = GPT2LMHeadModel(hf_config)

    if seed is not None:
        _reinit_weights(model, seed)

    return model.to(device)


def _reinit_weights(model: nn.Module, seed: int):
    """Reinitialize weights with controlled randomness."""
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def save_checkpoint(
    model: GPT2LMHeadModel,
    path: Path,
    epoch: int,
    optimizer_state: Optional[dict] = None,
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
):
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    config: Optional[ModelConfig] = None,
    device: str = "cuda",
    load_optimizer: bool = False,
) -> tuple:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = create_model(config, seed=None, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result = (model, checkpoint.get("epoch"), checkpoint.get("metrics"))

    if load_optimizer and "optimizer_state_dict" in checkpoint:
        result = (*result, checkpoint["optimizer_state_dict"])

    return result


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enable_gradient_checkpointing(model: GPT2LMHeadModel):
    """Enable gradient checkpointing for memory efficiency."""
    model.gradient_checkpointing_enable()


def get_model_info(model: GPT2LMHeadModel) -> dict:
    """Get model architecture info."""
    config = model.config
    return {
        "num_layers": config.n_layer,
        "d_model": config.n_embd,
        "d_ff": config.n_inner,
        "num_heads": config.n_head,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.n_positions,
        "total_params": count_parameters(model),
        "total_params_millions": count_parameters(model) / 1e6,
    }
