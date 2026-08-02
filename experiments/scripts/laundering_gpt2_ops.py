"""GPT-2 checkpoint laundering operators (permutation only).

GELU is NOT positively homogeneous, so rescaling operators from the MLP
benchmark cannot be used without approximation. This module focuses on
EXACT function-preserving permutation only.

GPT-2 Conv1D weight conventions:
    c_fc.weight:   (d_model, d_ff) - input projection
    c_fc.bias:     (d_ff,) - per hidden unit
    c_proj.weight: (d_ff, d_model) - output projection
    c_proj.bias:   (d_model,) - on residual stream, UNTOUCHED

Permutation operator for hidden dimension d_ff:
    c_fc.weight[:, p]     - permute columns (hidden outputs)
    c_fc.bias[p]          - permute bias
    c_proj.weight[p, :]   - permute rows (hidden inputs)
    c_proj.bias           - UNCHANGED (residual stream dimension)

Branch product invariance:
    M = W_proj.T @ W_fc.T (in standard math convention)
    After permutation: M' = W_proj.T @ P^T @ P @ W_fc.T = W_proj.T @ W_fc.T = M
    The branch product is EXACTLY invariant to hidden-unit permutation.
"""
from __future__ import annotations

import copy
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Gate thresholds for function preservation validation
GATE_THRESHOLD_LOGIT = 1e-4  # max |logit_laundered - logit_original| in fp32
GATE_THRESHOLD_PPL = 1e-5    # relative perplexity change
N_VALIDATION_SEQS = 128
MAX_VALIDATION_LEN = 256


def _rng(*ints) -> np.random.Generator:
    """Deterministic Generator seeded by a structured integer sequence."""
    return np.random.default_rng([int(x) for x in ints])


# --------------------------------------------------------------------- operators

def apply_permutation_gpt2_block(block, perm: torch.LongTensor) -> None:
    """Apply hidden-unit permutation to one GPT-2 MLP block IN PLACE.

    Args:
        block: A transformer block (model.transformer.h[i])
        perm: Permutation index tensor of shape (d_ff,)

    Conv1D stores: c_fc.weight (d_model, d_ff), c_proj.weight (d_ff, d_model)
    Permutation p acts on d_ff dimension:
        c_fc.weight[:, p]     - permute columns (hidden outputs)
        c_fc.bias[p]          - permute bias
        c_proj.weight[p, :]   - permute rows (hidden inputs)
        c_proj.bias           - UNCHANGED (residual stream dimension)
    """
    with torch.no_grad():
        # c_fc: d_model -> d_ff (stored as d_model, d_ff)
        block.mlp.c_fc.weight.copy_(block.mlp.c_fc.weight[:, perm])
        block.mlp.c_fc.bias.copy_(block.mlp.c_fc.bias[perm])

        # c_proj: d_ff -> d_model (stored as d_ff, d_model)
        block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight[perm, :])
        # c_proj.bias is on residual stream dimension - DO NOT TOUCH


def apply_permutation_gpt2(
    model: GPT2LMHeadModel,
    seed: int,
    block_indices: Optional[List[int]] = None,
) -> Tuple[GPT2LMHeadModel, Dict[str, Any]]:
    """Permute hidden units of GPT-2 MLP blocks (function-preserving).

    Args:
        model: Original GPT-2 model
        seed: Random seed for permutation generation
        block_indices: Which blocks to permute (default: all)

    Returns:
        (permuted_model, manifest) where manifest contains the permutations used
    """
    out = copy.deepcopy(model)
    n_blocks = len(out.transformer.h)

    if block_indices is None:
        block_indices = list(range(n_blocks))

    manifest = {
        "seed": seed,
        "block_indices": block_indices,
        "permutations": {},
    }

    for bi in block_indices:
        block = out.transformer.h[bi]
        d_ff = block.mlp.c_fc.weight.shape[1]

        # Generate permutation for this block
        perm_np = _rng(seed, bi).permutation(d_ff)
        perm = torch.as_tensor(perm_np, dtype=torch.long, device=block.mlp.c_fc.weight.device)

        apply_permutation_gpt2_block(block, perm)
        manifest["permutations"][bi] = perm_np.tolist()

    return out, manifest


# --------------------------------------------------- function-preservation gate

def make_validation_batch(
    tokenizer: GPT2Tokenizer,
    n_seqs: int = N_VALIDATION_SEQS,
    max_len: int = MAX_VALIDATION_LEN,
    seed: int = 54321,
) -> Dict[str, torch.Tensor]:
    """Create held-out validation sequences for function preservation testing.

    Uses a fixed set of prompts to ensure reproducibility.
    """
    # Fixed prompts for validation (deterministic)
    base_prompts = [
        "Once upon a time, there was a little",
        "The cat sat on the mat and",
        "In a galaxy far far away",
        "The quick brown fox jumps over",
        "To be or not to be, that is",
        "It was the best of times, it was",
        "All happy families are alike",
        "Call me Ishmael. Some years ago",
        "The sun rose over the mountains",
        "She opened the door and saw",
        "He walked into the room and",
        "The rain fell softly on the",
        "In the beginning, there was",
        "Once there lived a young girl",
        "The forest was dark and quiet",
        "A long time ago in a kingdom",
    ]

    # Expand prompts with seeded variations
    rng = np.random.default_rng(seed)
    prompts = []
    for i in range(n_seqs):
        base = base_prompts[i % len(base_prompts)]
        # Add some variation
        suffix_words = ["the", "a", "an", "one", "some", "this", "that", "my", "her", "his"]
        suffix = " " + suffix_words[rng.integers(len(suffix_words))]
        prompts.append(base + suffix * (i // len(base_prompts)))

    # Tokenize
    encodings = tokenizer(
        prompts[:n_seqs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }


def compute_logits(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute logits in eval mode with no gradient."""
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    return outputs.logits


def compute_perplexity(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute perplexity over the validation batch."""
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    return float(torch.exp(outputs.loss).item())


def validate_function_preservation(
    original: GPT2LMHeadModel,
    laundered: GPT2LMHeadModel,
    validation_batch: Dict[str, torch.Tensor],
    device: str = "cuda",
) -> Dict[str, Any]:
    """Comprehensive function-preservation validation.

    Args:
        original: Original model before permutation
        laundered: Model after permutation
        validation_batch: Dict with input_ids and attention_mask
        device: Device for computation

    Returns:
        {
            "max_logit_diff": float,
            "mean_logit_diff": float,
            "max_relative_logit_diff": float,
            "ppl_original": float,
            "ppl_laundered": float,
            "relative_ppl_change": float,
            "top1_agreement": float,  # should be 1.0
            "gate_pass": bool,
            "gate_details": str,
        }
    """
    input_ids = validation_batch["input_ids"].to(device)
    attention_mask = validation_batch["attention_mask"].to(device)

    original = original.to(device).float()  # fp32 for validation
    laundered = laundered.to(device).float()

    # Compute logits
    logits_orig = compute_logits(original, input_ids, attention_mask).float()
    logits_laun = compute_logits(laundered, input_ids, attention_mask).float()

    # Logit differences
    diff = (logits_orig - logits_laun).abs()
    max_logit_diff = float(diff.max().item())
    mean_logit_diff = float(diff.mean().item())

    # Relative logit difference (avoid div by zero)
    rel_diff = diff / (logits_orig.abs() + 1e-8)
    max_relative_logit_diff = float(rel_diff.max().item())

    # Perplexity
    ppl_orig = compute_perplexity(original, input_ids, attention_mask)
    ppl_laun = compute_perplexity(laundered, input_ids, attention_mask)
    relative_ppl_change = abs(ppl_laun - ppl_orig) / (ppl_orig + 1e-8)

    # Top-1 agreement
    pred_orig = logits_orig.argmax(dim=-1)
    pred_laun = logits_laun.argmax(dim=-1)
    # Only compare non-padded positions
    mask = attention_mask.bool()
    n_positions = mask.sum().item()
    n_agree = ((pred_orig == pred_laun) & mask).sum().item()
    top1_agreement = n_agree / n_positions if n_positions > 0 else 1.0

    # Gate check
    gate_pass = True
    gate_details = []

    if max_logit_diff >= GATE_THRESHOLD_LOGIT:
        gate_pass = False
        gate_details.append(f"max_logit_diff={max_logit_diff:.2e} >= {GATE_THRESHOLD_LOGIT}")

    if relative_ppl_change >= GATE_THRESHOLD_PPL:
        gate_pass = False
        gate_details.append(f"relative_ppl_change={relative_ppl_change:.2e} >= {GATE_THRESHOLD_PPL}")

    if top1_agreement < 1.0:
        gate_pass = False
        gate_details.append(f"top1_agreement={top1_agreement:.6f} < 1.0")

    return {
        "max_logit_diff": max_logit_diff,
        "mean_logit_diff": mean_logit_diff,
        "max_relative_logit_diff": max_relative_logit_diff,
        "ppl_original": ppl_orig,
        "ppl_laundered": ppl_laun,
        "relative_ppl_change": relative_ppl_change,
        "top1_agreement": top1_agreement,
        "gate_pass": gate_pass,
        "gate_details": "; ".join(gate_details) if gate_details else "PASS",
    }


# ------------------------------------------------- branch product verification

def verify_branch_product_invariance(
    original: GPT2LMHeadModel,
    laundered: GPT2LMHeadModel,
    eps: float = 1e-10,
) -> Dict[str, Any]:
    """Verify that branch products M = W_proj.T @ W_fc.T are unchanged.

    Returns per-block cosine similarities (should all be 1.0).
    """
    from gpt2_lineage_benchmark.extraction import extract_branch_products

    Ms_orig = extract_branch_products(original)
    Ms_laun = extract_branch_products(laundered)

    cosines = []
    max_abs_diff = []

    for M_o, M_l in zip(Ms_orig, Ms_laun):
        # Flatten and compute cosine
        v_o = M_o.flatten()
        v_l = M_l.flatten()
        cos = float(np.dot(v_o, v_l) / (np.linalg.norm(v_o) * np.linalg.norm(v_l) + eps))
        cosines.append(cos)

        # Max absolute difference
        max_abs_diff.append(float(np.abs(M_o - M_l).max()))

    return {
        "per_block_cosine": cosines,
        "min_cosine": min(cosines),
        "max_cosine": max(cosines),
        "per_block_max_diff": max_abs_diff,
        "overall_max_diff": max(max_abs_diff),
        "invariant": all(c > 1.0 - eps for c in cosines),
    }


# ------------------------------------------------- raw-weight extraction

def raw_weights_gpt2(model: GPT2LMHeadModel) -> Dict[str, List[np.ndarray]]:
    """Extract raw weights per block for Re-Basin baselines.

    Returns:
        {
            "Wins": list of (d_ff, d_model+1) with bias folded as trailing column
            "Wouts": list of (d_model, d_ff)
        }

    The bias is folded into Wins so that permutation of hidden units acts on
    whole rows consistently.
    """
    Wins, Wouts = [], []

    for block in model.transformer.h:
        # c_fc: (d_model, d_ff) in Conv1D storage
        # Standard form: W_fc is (d_ff, d_model), so transpose
        W_fc = block.mlp.c_fc.weight.detach().cpu().numpy().astype(np.float64).T  # (d_ff, d_model)
        b_fc = block.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float64)      # (d_ff,)

        # Fold bias as trailing column: (d_ff, d_model+1)
        W_fc_aug = np.concatenate([W_fc, b_fc[:, None]], axis=1)
        Wins.append(W_fc_aug)

        # c_proj: (d_ff, d_model) in Conv1D storage
        # Standard form: W_proj is (d_model, d_ff), so transpose
        W_proj = block.mlp.c_proj.weight.detach().cpu().numpy().astype(np.float64).T  # (d_model, d_ff)
        Wouts.append(W_proj)

    return {"Wins": Wins, "Wouts": Wouts}


# ------------------------------------------------- P+FT fine-tuning

def finetune_after_permutation(
    model: GPT2LMHeadModel,
    train_loader,
    n_steps: int = 1000,
    lr: float = 2e-5,
    device: str = "cuda",
    verbose: bool = False,
) -> GPT2LMHeadModel:
    """Brief fine-tuning after permutation to test robustness.

    This tests whether the method remains stable after the algebraic
    invariance is perturbed by additional training.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    losses = []

    while step < n_steps:
        for batch in train_loader:
            if step >= n_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if verbose and step % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"  [P+FT] step {step}/{n_steps}, loss={avg_loss:.4f}")

    model.eval()
    return model
