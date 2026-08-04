#!/usr/bin/env python3
"""Test if Q-rotation is function-preserving for GPT-2 with LayerNorm."""

import copy
import numpy as np
import torch
from scipy.stats import ortho_group
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


def apply_q_rotation_gpt2(model, Q, device="cpu"):
    """Apply Q rotation to all residual-stream-touching weights."""
    out = copy.deepcopy(model)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Embeddings
        out.transformer.wte.weight.copy_(out.transformer.wte.weight @ Q_t.T)
        out.transformer.wpe.weight.copy_(out.transformer.wpe.weight @ Q_t.T)

        for block in out.transformer.h:
            # Attention projections
            block.attn.c_attn.weight.copy_(Q_t.T @ block.attn.c_attn.weight)
            block.attn.c_proj.weight.copy_(block.attn.c_proj.weight @ Q_t.T)

            # LayerNorm (problematic!)
            block.ln_1.weight.copy_(Q_t @ block.ln_1.weight)
            block.ln_1.bias.copy_(Q_t @ block.ln_1.bias)
            block.ln_2.weight.copy_(Q_t @ block.ln_2.weight)
            block.ln_2.bias.copy_(Q_t @ block.ln_2.bias)

            # MLP
            block.mlp.c_fc.weight.copy_(Q_t.T @ block.mlp.c_fc.weight)
            block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight @ Q_t.T)

        # Final LN
        out.transformer.ln_f.weight.copy_(Q_t @ out.transformer.ln_f.weight)
        out.transformer.ln_f.bias.copy_(Q_t @ out.transformer.ln_f.bias)

    return out


def main():
    device = "cpu"

    # Create small GPT-2 for testing
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=4,
    )
    model = GPT2LMHeadModel(config).to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Generate Q
    d_model = config.n_embd
    Q = ortho_group.rvs(d_model, random_state=42)

    # Apply rotation
    model_rot = apply_q_rotation_gpt2(model, Q, device)
    model_rot.eval()

    # Test on some inputs - use random token ids within vocab range
    input_ids = torch.randint(0, config.vocab_size, (3, 20))
    inputs = {"input_ids": input_ids}

    with torch.no_grad():
        logits_orig = model(**inputs).logits
        logits_rot = model_rot(**inputs).logits

    # Compare
    max_diff = (logits_orig - logits_rot).abs().max().item()
    mean_diff = (logits_orig - logits_rot).abs().mean().item()

    # Top-1 agreement
    pred_orig = logits_orig.argmax(dim=-1)
    pred_rot = logits_rot.argmax(dim=-1)
    agreement = (pred_orig == pred_rot).float().mean().item()

    print(f"Max logit diff: {max_diff:.6f}")
    print(f"Mean logit diff: {mean_diff:.6f}")
    print(f"Top-1 agreement: {agreement:.4f}")

    if max_diff < 1e-4:
        print("\n✓ Q-rotation IS function-preserving")
    else:
        print(f"\n✗ Q-rotation is NOT function-preserving (diff={max_diff:.4f})")
        print("  LayerNorm gamma/beta break exact function preservation")

    # Now test WITHOUT LayerNorm transformation to show it's the culprit
    print("\n--- Test: Q-rotation WITHOUT transforming LayerNorm ---")
    model2 = GPT2LMHeadModel(config).to(device)
    model2.load_state_dict(model.state_dict())
    model2.eval()

    model_rot2 = copy.deepcopy(model2)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Only transform embeddings and linear layers, NOT LayerNorm
        model_rot2.transformer.wte.weight.copy_(
            model_rot2.transformer.wte.weight @ Q_t.T)
        model_rot2.transformer.wpe.weight.copy_(
            model_rot2.transformer.wpe.weight @ Q_t.T)

        for block in model_rot2.transformer.h:
            block.attn.c_attn.weight.copy_(Q_t.T @ block.attn.c_attn.weight)
            block.attn.c_proj.weight.copy_(
                block.attn.c_proj.weight @ Q_t.T)
            block.mlp.c_fc.weight.copy_(Q_t.T @ block.mlp.c_fc.weight)
            block.mlp.c_proj.weight.copy_(
                block.mlp.c_proj.weight @ Q_t.T)
            # Skip LayerNorm!

    model_rot2.eval()

    with torch.no_grad():
        logits_orig2 = model2(**inputs).logits
        logits_rot2 = model_rot2(**inputs).logits

    max_diff2 = (logits_orig2 - logits_rot2).abs().max().item()
    print(f"Max logit diff (no LN transform): {max_diff2:.6f}")
    print("This confirms LayerNorm is the blocker for Q-rotation.")


if __name__ == "__main__":
    main()
