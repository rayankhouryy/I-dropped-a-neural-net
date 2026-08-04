#!/usr/bin/env python3
"""Q-rotation laundering attack on centered residual signatures.

Implements the orthogonal rotation attack on the residual stream. For a residual
model with branch product M = W_out @ W_in per block, an orthogonal rotation Q
applied to the residual stream transforms:

    W_in  <- W_in @ Q^T     (input from rotated residual)
    W_out <- Q @ W_out      (output back to rotated residual)

Therefore M <- Q @ M @ Q^T (similarity transform).

This PRESERVES trace(M) but CHANGES the centered residual signature
R = M - (tr(M)/d) I, since Q @ R @ Q^T != R in general.

LayerNorm constraint:
    LayerNorm(x) centers and normalizes each position vector. For Q-rotation
    to commute with LayerNorm, Q must satisfy:
        LN(Qx) = Q LN(x)  for all x

    This holds iff Q stabilizes the all-ones vector: Q @ 1 = 1.
    Equivalently, Q must be in the stabilizer subgroup Stab_O(d)(1).

    Construction: Take any (d-1)x(d-1) orthogonal matrix U, embed it as the
    block Q = [[1, 0], [0, U]] in the basis where the first vector is 1/sqrt(d).
    More efficiently, generate Householder reflections in the 1-orthogonal subspace.

Usage:
    python laundering_q_rotation.py --benchmark-dir results/lineage_benchmark_gpt2_paper
    python laundering_q_rotation.py --smoke  # Quick test
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from transformers import GPT2Tokenizer

import functools
print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import laundering_gpt2_ops as lops
import gpt2_laundering_baselines as lbase
import lineage_detection as ldet
from gpt2_lineage_benchmark.extraction import extract_branch_products
from gpt2_lineage_benchmark.model import load_checkpoint
from gpt2_lineage_benchmark.config import ModelConfig


# ---------------------------------------------------------------- Q-rotation math

def generate_orthogonal_stabilizing_ones(d: int, seed: int) -> np.ndarray:
    """Generate random orthogonal Q in O(d) that stabilizes the all-ones vector.

    The all-ones vector 1 = [1,1,...,1]^T must satisfy Q @ 1 = 1.

    Construction:
    1. Normalize 1 to get e_1 = 1/sqrt(d)
    2. Generate a random (d-1)x(d-1) orthogonal matrix U
    3. Embed U in the orthogonal complement of e_1
    4. Q = I restricted to span{e_1} + U restricted to e_1^perp

    Efficiently: use Householder to map e_1 to first coordinate, apply block
    diag(1, U), then map back.
    """
    rng = np.random.default_rng(seed)

    # The all-ones vector, normalized
    ones = np.ones(d) / np.sqrt(d)

    # Householder reflector H such that H @ ones = e_1 (first standard basis vector)
    # H = I - 2 v v^T / (v^T v) where v = ones - e_1
    e1 = np.zeros(d)
    e1[0] = 1.0
    v = ones - e1
    v_norm_sq = v @ v
    if v_norm_sq < 1e-12:
        # ones is already e_1 (only happens if d=1)
        H = np.eye(d)
    else:
        H = np.eye(d) - 2 * np.outer(v, v) / v_norm_sq

    # Generate random (d-1)x(d-1) orthogonal matrix
    if d > 1:
        U = ortho_group.rvs(d - 1, random_state=rng)
    else:
        U = np.array([[1.0]])

    # Block matrix: diag(1, U) in standard basis
    block = np.eye(d)
    block[1:, 1:] = U

    # Q = H^T @ block @ H  (since H is symmetric, H^T = H)
    Q = H @ block @ H

    # Verify: Q @ ones should equal ones
    check = Q @ (np.ones(d) / np.sqrt(d))
    assert np.allclose(check, ones, atol=1e-10), f"Q does not stabilize 1: error = {np.linalg.norm(check - ones)}"

    # Verify orthogonality
    assert np.allclose(Q @ Q.T, np.eye(d), atol=1e-10), "Q is not orthogonal"

    return Q


def verify_q_stabilizes_ones(Q: np.ndarray, atol: float = 1e-10) -> bool:
    """Verify that Q @ 1 = 1 (unnormalized)."""
    d = Q.shape[0]
    ones = np.ones(d)
    return np.allclose(Q @ ones, ones, atol=atol)


# ---------------------------------------------------------------- applying Q-rotation

def apply_q_rotation_gpt2_block_mlp(block, Q: torch.Tensor, Q_T: torch.Tensor) -> None:
    """Apply Q-rotation to MLP within a transformer block IN PLACE.

    The MLP computes: h + W_proj @ sigma(W_fc @ LN(h) + b_fc) + b_proj

    If h -> Q @ h, then for the residual to remain consistent:
        - LN(Q @ h) = Q @ LN(h)  [requires Q stabilizes 1, checked separately]
        - W_fc @ LN(Q @ h) = W_fc @ Q @ LN(h), so W_fc <- W_fc @ Q^T to absorb Q
        - b_fc is per hidden-unit, unchanged
        - sigma(.) unchanged (element-wise)
        - W_proj @ sigma(.) contributes to rotated residual, so W_proj <- Q @ W_proj
        - b_proj is on residual stream, so b_proj <- Q @ b_proj

    Conv1D storage: c_fc.weight is (d_model, d_ff), c_proj.weight is (d_ff, d_model)
    Standard math: W_fc is (d_ff, d_model), W_proj is (d_model, d_ff)
    """
    with torch.no_grad():
        # c_fc: stored as (d_model, d_ff)
        # Standard W_fc (d_ff, d_model) = c_fc.weight.T
        # New W_fc = W_fc @ Q^T = c_fc.weight.T @ Q^T
        # New c_fc.weight = (W_fc @ Q^T).T = Q @ c_fc.weight
        block.mlp.c_fc.weight.copy_(Q @ block.mlp.c_fc.weight)
        # c_fc.bias is per hidden-unit (d_ff,), unchanged

        # c_proj: stored as (d_ff, d_model)
        # Standard W_proj (d_model, d_ff) = c_proj.weight.T
        # New W_proj = Q @ W_proj = Q @ c_proj.weight.T
        # New c_proj.weight = (Q @ c_proj.weight.T).T = c_proj.weight @ Q^T
        block.mlp.c_proj.weight.copy_(block.mlp.c_proj.weight @ Q_T)
        # c_proj.bias is on residual stream (d_model,), needs rotation
        block.mlp.c_proj.bias.copy_(Q @ block.mlp.c_proj.bias)


def apply_q_rotation_gpt2_block_attn(block, Q: torch.Tensor, Q_T: torch.Tensor) -> None:
    """Apply Q-rotation to attention within a transformer block IN PLACE.

    Attention: h + W_o @ (softmax(QK^T/sqrt(d)) @ V)

    The c_attn projects: [Q, K, V] = W_qkv @ LN(h)

    If h -> Q @ h:
        - LN(Q @ h) = Q @ LN(h) [requires Q stabilizes 1]
        - W_qkv @ LN(Q @ h) = W_qkv @ Q @ LN(h), so W_qkv <- W_qkv @ Q^T
        - Q/K/V projections rotate, but Q^T K attention pattern unchanged
          (because Q_attn @ K^T = (W_q Q x)(W_k Q x)^T = W_q Q x x^T Q^T W_k^T,
           and Q/K weights both get the same Q rotation, so it factors out)
        - W_o @ V contributes to rotated residual, so W_o <- Q @ W_o
        - c_proj.bias on residual stream, so bias <- Q @ bias

    Conv1D c_attn.weight: (d_model, 3*d_model) for combined Q,K,V
    Conv1D c_proj.weight: (d_model, d_model) for output projection
    """
    with torch.no_grad():
        # c_attn: stored as (d_model, 3*d_model)
        # New c_attn.weight = Q @ c_attn.weight
        block.attn.c_attn.weight.copy_(Q @ block.attn.c_attn.weight)
        # c_attn.bias: (3*d_model,) per attention-unit, unchanged by residual rotation
        # Wait - this needs more care. The bias is added after projection,
        # b_q, b_k, b_v are per-head. Since Q/K/V all rotate together,
        # the biases don't need rotation (they're in the Q/K/V space, not residual)

        # c_proj: stored as (d_model, d_model)
        # Standard W_o (d_model, d_model) = c_proj.weight.T
        # New W_o = Q @ W_o = Q @ c_proj.weight.T
        # New c_proj.weight = (Q @ c_proj.weight.T).T = c_proj.weight @ Q^T
        block.attn.c_proj.weight.copy_(block.attn.c_proj.weight @ Q_T)
        # c_proj.bias on residual stream
        block.attn.c_proj.bias.copy_(Q @ block.attn.c_proj.bias)


def apply_q_rotation_gpt2_embeddings(model, Q: torch.Tensor) -> None:
    """Apply Q-rotation to embedding and LM head.

    - wte (word embeddings): output is added to residual stream, so rotate
    - wpe (position embeddings): output is added to residual stream, so rotate
    - lm_head: maps from residual stream to vocab logits
        logits = lm_head @ h, if h -> Q @ h, then need lm_head <- lm_head @ Q^T

    GPT-2 ties lm_head to wte, so we handle both by rotating wte only;
    the tied lm_head will automatically use the rotated embeddings.
    """
    with torch.no_grad():
        # wte.weight: (vocab_size, d_model)
        # Each row is an embedding vector in the residual stream
        # If residual -> Q @ residual, embeddings should be Q @ emb
        # wte.weight[v, :] <- Q @ wte.weight[v, :].T, i.e., wte.weight <- wte.weight @ Q^T
        model.transformer.wte.weight.copy_(model.transformer.wte.weight @ Q.T)

        # wpe.weight: (max_seq_len, d_model)
        model.transformer.wpe.weight.copy_(model.transformer.wpe.weight @ Q.T)

        # lm_head is tied to wte, so it's automatically updated


def apply_q_rotation_gpt2_layernorms(model, Q: torch.Tensor) -> None:
    """Apply Q-rotation to LayerNorm parameters.

    LayerNorm: gamma * (x - mean) / std + beta

    If x -> Q @ x, and Q stabilizes 1 (so mean is preserved):
        LN(Q @ x) = gamma * (Q @ x - mean(Q @ x)) / std(Q @ x) + beta
                 = gamma * Q @ (x - mean) / std + beta

    This equals Q @ LN(x) only if:
        gamma * Q @ (x - mean) / std + beta = Q @ (gamma * (x - mean) / std + beta)

    Expanding: the gamma terms match if gamma is applied after Q, but LayerNorm
    applies gamma element-wise before the rotation would happen.

    Actually, for Q @ LN(x) = LN(Q @ x), we need gamma and beta to transform:
        gamma_new = Q^T @ diag(gamma) @ Q applied element-wise... this gets messy.

    The clean solution: LayerNorm with Q-rotation is only exactly preserved if
    gamma = 1, beta = 0 (identity affine). In practice, trained models have
    non-trivial gamma/beta.

    For GPT-2 with trained LayerNorm, we need to transform:
        - ln_f (final layernorm before lm_head)
        - ln_1, ln_2 in each block (attention and MLP prenorms)

    The transformation for element-wise affine: if output y = gamma * norm(x) + beta,
    and we want Q @ y, then Q @ (gamma * norm(x) + beta).

    Since norm(x) is scalar-normalized (zero mean, unit var), and Q preserves norm,
    norm(Q @ x) = Q @ norm(x) when Q stabilizes 1.

    So Q @ y = Q @ gamma * Q @ norm(x) + Q @ beta
            = (Q @ gamma) * norm(Q @ x) + Q @ beta  [since Q @ norm(x) = norm(Q @ x)]

    Wait, that's not right either. gamma is applied element-wise: y_i = gamma_i * norm(x)_i + beta_i

    Let me reconsider. LayerNorm computes:
        y = gamma * (x - mu) / sigma + beta
    where mu = mean(x), sigma = std(x), and * is element-wise.

    For Q @ y = LN'(Q @ x), we need LN' such that:
        Q @ y = Q @ [gamma * (x - mu) / sigma + beta]

    Note that mu(Q @ x) = sum(Q @ x) / d. If Q @ 1 = 1, then 1^T @ Q = 1^T, so
    1^T @ Q @ x = 1^T @ x = d * mu(x). Hence mu is invariant.

    Similarly, var(Q @ x) = ||Q @ x - mu||^2 / d = ||Q @ (x - mu 1)||^2 / d = ||x - mu 1||^2 / d = var(x)
    since Q is orthogonal. So sigma is also invariant.

    Thus: Q @ y = Q @ [gamma * (x - mu) / sigma + beta]
                = gamma * Q @ [(x - mu) / sigma] + Q @ beta  [if gamma constant]

    But gamma is a vector applied element-wise! So Q @ [gamma * z] != gamma * (Q @ z) in general.

    The solution: transform gamma and beta. Define:
        gamma_new such that gamma_new * (Q @ z) = Q @ (gamma * z)

    For element-wise mult: (gamma_new)_i * (Q @ z)_i = sum_j Q_ij gamma_j z_j
    This should hold for all z, so we need (gamma_new)_i * Q_ij = Q_ij * gamma_j for all j.
    This is only possible if gamma is constant (or Q is diagonal).

    Conclusion: Exact function preservation under Q-rotation requires LayerNorm
    with constant gamma and zero beta (or the identity affine). For trained GPT-2
    models with non-trivial gamma/beta, Q-rotation is NOT exactly function-preserving.

    HOWEVER, for the attack to be meaningful, we can:
    1. Accept small approximation error (LayerNorm affine is typically close to identity)
    2. Focus on RMSNorm models where this is exact
    3. Transform gamma/beta as best-effort approximation

    For this implementation, we'll transform gamma and beta as:
        gamma_new = Q @ gamma  (rotate the scale vector)
        beta_new = Q @ beta    (rotate the shift vector)

    This is the "as if gamma/beta were on the residual stream" interpretation,
    which is approximately correct when gamma is close to 1 and beta close to 0.
    """
    with torch.no_grad():
        # Final layer norm before LM head
        model.transformer.ln_f.weight.copy_(Q @ model.transformer.ln_f.weight)
        model.transformer.ln_f.bias.copy_(Q @ model.transformer.ln_f.bias)

        # Per-block layer norms
        for block in model.transformer.h:
            # ln_1: before attention
            block.ln_1.weight.copy_(Q @ block.ln_1.weight)
            block.ln_1.bias.copy_(Q @ block.ln_1.bias)
            # ln_2: before MLP
            block.ln_2.weight.copy_(Q @ block.ln_2.weight)
            block.ln_2.bias.copy_(Q @ block.ln_2.bias)


def apply_q_rotation_gpt2(
    model,
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Apply Q-rotation to entire GPT-2 model (approximately function-preserving).

    The same Q is used across all blocks (since they share the residual stream).

    Returns (rotated_model, manifest).
    """
    out = copy.deepcopy(model)
    d_model = out.config.n_embd

    # Generate Q that stabilizes the all-ones vector
    Q_np = generate_orthogonal_stabilizing_ones(d_model, seed)
    Q = torch.from_numpy(Q_np).to(out.transformer.wte.weight.dtype).to(out.transformer.wte.weight.device)
    Q_T = Q.T

    manifest = {
        "seed": seed,
        "d_model": d_model,
        "Q_stabilizes_ones": bool(verify_q_stabilizes_ones(Q_np)),
        "Q_is_orthogonal": bool(np.allclose(Q_np @ Q_np.T, np.eye(d_model), atol=1e-10)),
    }

    # Apply to embeddings and LM head
    apply_q_rotation_gpt2_embeddings(out, Q)

    # Apply to each transformer block
    for block in out.transformer.h:
        apply_q_rotation_gpt2_block_attn(block, Q, Q_T)
        apply_q_rotation_gpt2_block_mlp(block, Q, Q_T)

    # Apply to layer norms
    apply_q_rotation_gpt2_layernorms(out, Q)

    return out, manifest


# ---------------------------------------------------------------- verification

def compute_branch_product_under_q_rotation(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute how branch product transforms under Q-rotation.

    M' = Q @ M @ Q^T (similarity transform).
    """
    return Q @ M @ Q.T


def verify_trace_preservation(M_orig: np.ndarray, M_rot: np.ndarray, atol: float = 1e-10) -> bool:
    """Verify that trace is preserved under similarity transform."""
    return np.abs(np.trace(M_orig) - np.trace(M_rot)) < atol


def verify_centered_signature_change(M_orig: np.ndarray, M_rot: np.ndarray, atol: float = 1e-6) -> Dict[str, float]:
    """Compute how much the centered residual signature changes.

    The centered signature phi(M) = vec(M - tr(M)/d I) / ||...||

    Under Q-rotation: M' = Q M Q^T, so tr(M') = tr(M) (preserved).
    R' = M' - tr(M')/d I = Q M Q^T - tr(M)/d I
       = Q (M - tr(M)/d I) Q^T + tr(M)/d (Q I Q^T - I)
       = Q R Q^T  (since Q I Q^T = I for orthogonal Q)

    So R' = Q R Q^T, which means phi(M') != phi(M) in general.
    The cosine between phi(M) and phi(M') measures the signature change.
    """
    d = M_orig.shape[0]

    # Centered residual signatures
    alpha_orig = np.trace(M_orig) / d
    alpha_rot = np.trace(M_rot) / d

    R_orig = M_orig - alpha_orig * np.eye(d)
    R_rot = M_rot - alpha_rot * np.eye(d)

    phi_orig = R_orig.flatten()
    phi_orig = phi_orig / (np.linalg.norm(phi_orig) + 1e-12)

    phi_rot = R_rot.flatten()
    phi_rot = phi_rot / (np.linalg.norm(phi_rot) + 1e-12)

    cosine = float(phi_orig @ phi_rot)

    return {
        "cosine_phi": cosine,
        "trace_orig": float(np.trace(M_orig)),
        "trace_rot": float(np.trace(M_rot)),
        "trace_preserved": abs(np.trace(M_orig) - np.trace(M_rot)) < atol,
    }


# ---------------------------------------------------------------- experiment runner

@dataclass
class QRotationConfig:
    """Configuration for Q-rotation experiment."""
    benchmark_dir: str = "results/lineage_benchmark_gpt2_paper"
    checkpoint_dir: Optional[str] = None
    output_dir: str = "results/laundering_q_rotation"

    n_seeds: int = 5
    seed_base: int = 9000

    test_root_indices: List[int] = field(default_factory=lambda: [5, 6, 7])
    n_validation_seqs: int = 128

    smoke: bool = False


def run_q_rotation_experiment(config: QRotationConfig) -> Dict[str, Any]:
    """Run the Q-rotation laundering experiment."""
    import pickle

    t0 = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if config.checkpoint_dir is None:
        # Check if benchmark_dir already contains checkpoints directly
        if (Path(config.benchmark_dir) / "root_0").exists():
            config.checkpoint_dir = config.benchmark_dir
        else:
            config.checkpoint_dir = f"{config.benchmark_dir}/checkpoints"

    # Load benchmark data
    print("Loading benchmark data...")
    benchmark_dir = Path(config.benchmark_dir)

    with open(benchmark_dir / "phase1_roots.pkl", "rb") as f:
        phase1_data = pickle.load(f)
    with open(benchmark_dir / "phase2_descendants.pkl", "rb") as f:
        phase2_data = pickle.load(f)

    tau_s = phase1_data["tau_s"]
    root_signatures = phase1_data["root_signatures"]
    model_config = ModelConfig()

    print(f"tau_s = {tau_s:.4f}")

    # Load tokenizer for validation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    validation_batch = lops.make_validation_batch(
        tokenizer, n_seqs=config.n_validation_seqs, seed=config.seed_base + 999
    )

    # Build test pairs (same as laundering_gpt2.py)
    test_roots = set(config.test_root_indices)
    descendants = phase2_data["descendants"]
    students = phase2_data["students"]

    pairs = []

    # Descendant pairs
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
        })

    # Distilled students
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
        })

    # Cross-root pairs
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
            })

    print(f"Test pairs: {len(pairs)}")

    if config.smoke:
        pairs = pairs[:3]
        config.n_seeds = 2
        print(f"SMOKE TEST: {len(pairs)} pairs, {config.n_seeds} seeds")

    # Run experiment
    all_results = []
    validation_logs = []
    signature_analysis = []

    for seed_idx in range(config.n_seeds):
        seed = config.seed_base + seed_idx * 100
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{config.n_seeds} (seed={seed})")
        print(f"{'='*60}")

        for pair_idx, pair in enumerate(pairs):
            ref_idx = pair["ref_idx"]
            ref_Ms_orig = root_signatures[ref_idx]

            # Load reference model
            ref_ckpt = Path(config.checkpoint_dir) / f"root_{ref_idx}" / "epoch_3.pt"
            ref_model, _, _ = load_checkpoint(ref_ckpt, model_config, device)

            # Apply Q-rotation to reference
            ref_model_rot, manifest = apply_q_rotation_gpt2(ref_model, seed)

            # Validate function preservation
            val_result = lops.validate_function_preservation(
                ref_model, ref_model_rot, validation_batch, device
            )
            validation_logs.append({
                "seed": seed,
                "model": pair["ref_id"],
                **val_result,
            })

            # Extract rotated branch products
            ref_Ms_rot = extract_branch_products(ref_model_rot)

            # Analyze signature change per block
            for bi, (M_orig, M_rot) in enumerate(zip(ref_Ms_orig, ref_Ms_rot)):
                analysis = verify_centered_signature_change(M_orig, M_rot)
                analysis["seed"] = seed
                analysis["model"] = pair["ref_id"]
                analysis["block"] = bi
                signature_analysis.append(analysis)

            # Get suspect signatures
            if "sus_idx" in pair:
                sus_Ms = pair["sus_Ms"]  # Original signatures
            else:
                sus_Ms = pair["sus_Ms"]

            # Compute lineage scores
            # Case 1: Original reference vs original suspect (baseline)
            L_orig, _, _ = ldet.lineage_score(ref_Ms_orig, sus_Ms, tau_s)

            # Case 2: Q-rotated reference vs original suspect (attack scenario)
            # This simulates: attacker has rotated their model (originally from ref),
            # defender compares against original ref
            L_rot_vs_orig, _, _ = ldet.lineage_score(ref_Ms_rot, sus_Ms, tau_s)

            # Case 3: Q-rotated reference vs Q-rotated suspect
            # (if suspect is also rotated with same Q - shouldn't happen in practice)
            # Skip this - not realistic attack scenario

            result = {
                "seed": seed,
                "ref_id": pair["ref_id"],
                "sus_id": pair["sus_id"],
                "label": pair["label"],
                "attack_type": pair["attack_type"],
                "L_original": L_orig,
                "L_after_Q_rotation": L_rot_vs_orig,
                "validation": val_result,
            }
            all_results.append(result)

            # Clean up
            ref_model.cpu()
            ref_model_rot.cpu()
            del ref_model, ref_model_rot
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            print(f"  [{pair_idx+1}/{len(pairs)}] {pair['ref_id']} vs {pair['sus_id']} "
                  f"({pair['label'][:4]}) | L={L_orig:.4f} -> L_rot={L_rot_vs_orig:.4f}")

    # Compute aggregate metrics
    print("\n" + "="*60)
    print("Computing aggregate metrics...")
    print("="*60)

    # AUROC for original vs after Q-rotation
    labels = [1 if r["label"] == "descendant" else 0 for r in all_results]
    scores_orig = [r["L_original"] for r in all_results]
    scores_rot = [r["L_after_Q_rotation"] for r in all_results]

    auroc_orig = roc_auc_score(labels, scores_orig) if len(set(labels)) >= 2 else float("nan")
    auroc_rot = roc_auc_score(labels, scores_rot) if len(set(labels)) >= 2 else float("nan")

    # Gap-Z analysis (how much the gap shrinks)
    desc_scores_orig = [s for s, l in zip(scores_orig, labels) if l == 1]
    nondesc_scores_orig = [s for s, l in zip(scores_orig, labels) if l == 0]
    desc_scores_rot = [s for s, l in zip(scores_rot, labels) if l == 1]
    nondesc_scores_rot = [s for s, l in zip(scores_rot, labels) if l == 0]

    gap_orig = np.mean(desc_scores_orig) - np.mean(nondesc_scores_orig)
    gap_rot = np.mean(desc_scores_rot) - np.mean(nondesc_scores_rot)

    # Signature cosine analysis
    mean_cosine = np.mean([a["cosine_phi"] for a in signature_analysis])
    min_cosine = np.min([a["cosine_phi"] for a in signature_analysis])
    max_cosine = np.max([a["cosine_phi"] for a in signature_analysis])

    summary = {
        "n_pairs": len(pairs),
        "n_seeds": config.n_seeds,
        "n_results": len(all_results),

        "auroc_original": auroc_orig,
        "auroc_after_Q_rotation": auroc_rot,

        "gap_original": gap_orig,
        "gap_after_Q_rotation": gap_rot,

        "mean_desc_score_orig": np.mean(desc_scores_orig),
        "mean_desc_score_rot": np.mean(desc_scores_rot),
        "mean_nondesc_score_orig": np.mean(nondesc_scores_orig),
        "mean_nondesc_score_rot": np.mean(nondesc_scores_rot),

        "signature_cosine_mean": mean_cosine,
        "signature_cosine_min": min_cosine,
        "signature_cosine_max": max_cosine,

        "validation_pass_rate": np.mean([v["gate_pass"] for v in validation_logs]),
        "mean_max_logit_diff": np.mean([v["max_logit_diff"] for v in validation_logs]),

        "wall_seconds": time.time() - t0,
    }

    # Print summary
    print("\n" + "="*60)
    print("Q-ROTATION ATTACK RESULTS")
    print("="*60)
    print(f"AUROC (original):         {auroc_orig:.4f}")
    print(f"AUROC (after Q-rotation): {auroc_rot:.4f}")
    print(f"Gap (original):           {gap_orig:.4f}")
    print(f"Gap (after Q-rotation):   {gap_rot:.4f}")
    print(f"Signature cosine (mean):  {mean_cosine:.4f}")
    print(f"Signature cosine (min):   {min_cosine:.4f}")
    print(f"Validation pass rate:     {summary['validation_pass_rate']:.2%}")
    print(f"Mean max logit diff:      {summary['mean_max_logit_diff']:.2e}")

    # Save results
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "all_results.json", "w") as f:
        # Convert validation dicts for JSON
        results_for_json = []
        for r in all_results:
            r_copy = dict(r)
            r_copy["validation"] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                                    for k, v in r["validation"].items()}
            results_for_json.append(r_copy)
        json.dump(results_for_json, f, indent=2)

    with open(output_dir / "signature_analysis.json", "w") as f:
        # Convert numpy bools to Python bools for JSON serialization
        sig_for_json = [{k: bool(v) if isinstance(v, (np.bool_, bool)) and not isinstance(v, int)
                         else float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in a.items()} for a in signature_analysis]
        json.dump(sig_for_json, f, indent=2)

    with open(output_dir / "validation_logs.json", "w") as f:
        val_logs_for_json = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in log.items()} for log in validation_logs]
        json.dump(val_logs_for_json, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Q-rotation laundering attack")
    parser.add_argument("--benchmark-dir", default="results/lineage_benchmark_gpt2_paper",
                        help="Path to benchmark data directory")
    parser.add_argument("--output-dir", default="results/laundering_q_rotation",
                        help="Output directory")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random Q-rotation seeds")
    parser.add_argument("--seed-base", type=int, default=9000,
                        help="Base random seed")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test")
    parser.add_argument("--test-roots", type=int, nargs="+", default=None,
                        help="Root indices for test (default: auto-detect)")

    args = parser.parse_args()

    # Auto-detect test roots
    test_roots = args.test_roots
    if test_roots is None:
        import pickle
        phase1_path = Path(args.benchmark_dir) / "phase1_roots.pkl"
        if phase1_path.exists():
            with open(phase1_path, "rb") as f:
                phase1 = pickle.load(f)
            n_roots = len(phase1["root_signatures"])
            test_roots = list(range(n_roots))
            print(f"Auto-detected {n_roots} roots: {test_roots}")

    config = QRotationConfig(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        n_seeds=args.n_seeds,
        seed_base=args.seed_base,
        smoke=args.smoke,
        test_root_indices=test_roots if test_roots else [0, 1, 2],
    )

    run_q_rotation_experiment(config)


if __name__ == "__main__":
    main()
