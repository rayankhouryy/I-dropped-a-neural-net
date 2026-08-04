#!/usr/bin/env python3
"""Test Q-rotation attack on centered residual signatures.

The reviewer points out that orthogonal rotation Q of the residual stream:
    W_in <- W_in @ Q^T
    W_out <- Q @ W_out

Transforms the branch product as:
    M = W_out @ W_in -> Q @ M @ Q^T

This preserves trace (tr(QMQ^T) = tr(M)) but changes the centered signature:
    <vec(Q R_A Q^T), vec(R_B)> != <vec(R_A), vec(R_B)>

For RMSNorm models, this is exactly function-preserving.
For LayerNorm, Q must fix the all-ones vector (Q @ 1 = 1).

This script tests whether this attack breaks our method.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score

import functools
print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gpt2_lineage_benchmark.extraction import extract_branch_products
from gpt2_lineage_benchmark.model import load_checkpoint
from gpt2_lineage_benchmark.config import ModelConfig
import lineage_detection as ldet
import laundering_gpt2_ops as lops


def generate_orthogonal_matrix(d: int, seed: int) -> np.ndarray:
    """Generate a random orthogonal matrix of size d x d."""
    rng = np.random.default_rng(seed)
    return ortho_group.rvs(d, random_state=rng)


def generate_layernorm_compatible_Q(d: int, seed: int) -> np.ndarray:
    """Generate orthogonal Q that fixes the all-ones vector (for LayerNorm).

    LayerNorm subtracts the mean, so we need Q @ 1 = 1.
    We generate Q in the (d-1)-dimensional subspace orthogonal to 1,
    then extend it to include 1 as an eigenvector.
    """
    rng = np.random.default_rng(seed)

    # 1/sqrt(d) * ones is a unit vector
    ones_normalized = np.ones(d) / np.sqrt(d)

    # Generate random orthogonal matrix in d-1 dimensions
    Q_sub = ortho_group.rvs(d - 1, random_state=rng)

    # Build basis for subspace orthogonal to ones
    # Use Householder to get orthonormal basis
    basis = np.eye(d)
    basis[:, 0] = ones_normalized
    Q_full, _ = np.linalg.qr(basis)

    # Q_full[:, 0] is parallel to ones
    # Q_full[:, 1:] spans the orthogonal complement

    # Build final Q: fix ones direction, rotate in orthogonal complement
    Q = np.zeros((d, d))
    Q[:, 0] = Q_full[:, 0]  # Keep ones direction
    Q[:, 1:] = Q_full[:, 1:] @ Q_sub  # Rotate in complement

    # Make it a proper rotation (Q @ Q^T = I)
    Q = Q_full @ np.block([
        [np.array([[1]])],
        [np.zeros((d-1, 1))]
    ]).T @ np.eye(d)

    # Actually simpler: just use Q_full with Q_sub embedded
    Q = Q_full.copy()
    Q[:, 1:] = Q_full[:, 1:] @ Q_sub

    return Q


def apply_q_rotation_gpt2(
    model,
    Q: np.ndarray,
    device: str = "cuda",
) -> Tuple[Any, Dict[str, Any]]:
    """Apply residual-stream rotation Q to all layers.

    For GPT-2, the residual stream dimension is d_model.
    We need to transform:
    - Embeddings: wte.weight, wpe.weight -> @ Q^T
    - LayerNorm (handled by mean-subtraction, but scale/bias need care)
    - Attention: Q/K/V projections, output projection
    - MLP: c_fc (input), c_proj (output)
    - Final LN and lm_head

    For MLP blocks specifically:
        x_in -> residual stream
        c_fc: x_in @ W_fc (Conv1D: d_model -> d_ff)
        c_proj: h @ W_proj (Conv1D: d_ff -> d_model)

    If we rotate residual stream by Q:
        x_in' = Q @ x_in  (or x_in @ Q^T in row-vector convention)

    For the MLP to produce the same rotated output:
        c_fc.weight <- Q @ c_fc.weight  (if Q acts on input)
        c_proj.weight <- c_proj.weight @ Q^T  (if Q acts on output)

    But GPT-2 uses row-vector convention and Conv1D, so:
        c_fc.weight: (d_model, d_ff) - input is left-multiplied
        c_proj.weight: (d_ff, d_model) - output is right-multiplied

    Rotation of residual stream x -> x @ Q^T means:
        c_fc needs: (x @ Q^T) @ W_fc = x @ (Q^T @ W_fc), so W_fc <- Q^T @ W_fc
        But W_fc stored as (d_model, d_ff), so W_fc <- Q.T @ W_fc

        c_proj needs: h @ W_proj -> (h @ W_proj) @ Q^T, so W_proj <- W_proj @ Q.T
        But W_proj stored as (d_ff, d_model), so W_proj <- W_proj @ Q.T
    """
    out = copy.deepcopy(model)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Token embeddings: output is x = wte[token_id], need x @ Q^T
        # wte.weight: (vocab, d_model), so wte <- wte @ Q^T
        out.transformer.wte.weight.copy_(out.transformer.wte.weight @ Q_t.T)

        # Position embeddings: same
        out.transformer.wpe.weight.copy_(out.transformer.wpe.weight @ Q_t.T)

        # Each transformer block
        for block in out.transformer.h:
            # Attention c_attn: (d_model, 3*d_model) projects to Q,K,V
            # Input is x @ c_attn.weight, need (x @ Q^T) @ c_attn = x @ (Q^T @ c_attn)
            # So c_attn.weight <- Q.T @ c_attn.weight... but that's wrong dimension
            # c_attn.weight is (d_model, 3*d_model)
            # Q.T is (d_model, d_model)
            # Q.T @ c_attn.weight doesn't work

            # Actually in Conv1D, x @ W means W is (in, out)
            # So for input rotation x -> x @ Q^T, we need:
            # (x @ Q^T) @ W = x @ (Q^T @ W)... no, matrix mult is associative
            # x @ Q^T @ W = x @ (Q^T @ W), so W <- Q^T @ W? No...
            #
            # Let's be careful. x is (batch, seq, d_model). W is (d_model, out).
            # x @ W gives (batch, seq, out).
            # If x' = x @ Q^T, then x' @ W = x @ Q^T @ W.
            # We want this to equal (x @ W) @ Some_transform for attention...
            #
            # Actually for function preservation, we need the OUTPUT of attention
            # (after c_proj) to be rotated the same way.

            # Let me think more carefully about the full block.
            # Skip connection: y = x + Attn(LN(x))
            # If we rotate: y' = x' + Attn'(LN(x'))
            # We need y' = y @ Q^T, i.e., y @ Q^T = x @ Q^T + Attn'(LN(x @ Q^T))
            # So Attn'(LN(x @ Q^T)) = Attn(LN(x)) @ Q^T

            # For LayerNorm: LN(x @ Q^T) = LN(x) @ Q^T if Q fixes ones
            # (because LN subtracts mean and divides by std, both scalar ops)
            # Actually no - LN operates per-position on the d_model dimension.
            # LN(x)[i] = (x[i] - mean(x[i])) / std(x[i]) * gamma + beta
            # If x' = x @ Q^T, then LN(x') uses mean and std of rotated vector.
            # mean(x @ Q^T) = sum(x @ Q^T) / d = x @ Q^T @ 1 / d
            # If Q @ 1 = 1, then Q^T @ 1 = 1, so mean(x @ Q^T) = mean(x).
            # Similarly for std. So LN(x @ Q^T) = LN(x) @ Q^T. ✓

            # For attention: simplified, out = softmax(QK^T/sqrt(d)) @ V @ W_proj
            # The QKV come from c_attn. Let z = LN(x). z' = z @ Q^T.
            # c_attn: z @ W_attn = [z@W_q, z@W_k, z@W_v]
            # With rotation: z' @ W_attn = z @ Q^T @ W_attn
            # We want this to give rotated Q,K,V... but attention is invariant to rotation
            # of Q,K,V if we rotate all of them the same way.
            #
            # Actually the attention output goes through c_proj back to residual stream.
            # Let's just transform c_attn input and c_proj output.

            # c_attn: (d_model, 3*d_model). Input needs Q^T.
            # (x @ Q^T) @ W = x @ (Q^T @ W)... but dims don't match.
            # Q^T is (d, d), W is (d, 3d).
            # Oh wait, (d,d) @ (d, 3d) = (d, 3d). ✓
            # So c_attn.weight <- Q.T @ c_attn.weight
            W_attn = block.attn.c_attn.weight  # (d_model, 3*d_model)
            block.attn.c_attn.weight.copy_(Q_t.T @ W_attn)

            # c_proj (attention output): (d_model, d_model) in Conv1D
            # Internal attention output @ c_proj.weight, then add to residual
            # We need output @ Q^T, so c_proj.weight <- c_proj.weight @ Q^T
            W_proj_attn = block.attn.c_proj.weight  # (d_model, d_model)
            block.attn.c_proj.weight.copy_(W_proj_attn @ Q_t.T)

            # LayerNorm 1 (before attention): gamma, beta are (d_model,)
            # LN_out = (x - mean) / std * gamma + beta
            # If input is rotated, and Q fixes 1, then mean/std unchanged.
            # But gamma and beta are element-wise, so they need rotation too.
            # gamma' @ (rotated_normalized) = gamma @ normalized @ Q^T
            # Hmm, this is tricky. Let's think...
            # normalized = (x - mean) / std, shape (batch, seq, d)
            # output = normalized * gamma + beta
            # If normalized' = normalized @ Q^T, we want output' = output @ Q^T
            # output' = normalized' * gamma + beta = (normalized @ Q^T) * gamma + beta
            # We want this = (normalized * gamma + beta) @ Q^T
            #              = normalized @ diag(gamma) @ Q^T + beta @ Q^T
            # For element-wise: (normalized @ Q^T) * gamma = normalized @ Q^T @ diag(gamma)
            # We need Q^T @ diag(gamma) = diag(gamma') @ Q^T for some gamma'
            # That means Q^T must commute with diag(gamma), which is only true if Q is diagonal.
            #
            # So LayerNorm gamma/beta break the exact function preservation!
            # Unless gamma = constant (all same value) and beta = constant.
            #
            # For standard LN with learned gamma/beta, Q-rotation is NOT function-preserving.
            #
            # However, for RMSNorm (no mean subtraction, gamma only, no beta):
            # RMSNorm(x) = x / rms(x) * gamma
            # If we use gamma = 1 (or constant), then it's just normalization.
            # GPT-2 uses LayerNorm with learned gamma and beta.

            # Let's check what happens if we transform gamma and beta too.
            # Actually, we can absorb the rotation into gamma/beta:
            # Want: LN'(x @ Q^T) @ Q = LN(x)? No, that's backwards.
            #
            # I think the clean approach is:
            # - Transform gamma: gamma' = Q @ gamma (as a vector)
            # - Transform beta: beta' = Q @ beta
            # Then LN'(x') where x' = x @ Q^T:
            # normalized' = (x' - mean(x')) / std(x')
            # If Q fixes 1: normalized' = normalized @ Q^T
            # output' = normalized' * gamma' + beta'
            #         = (normalized @ Q^T) * (Q @ gamma) + Q @ beta
            # For this to equal output @ Q^T = (normalized * gamma + beta) @ Q^T:
            # (normalized @ Q^T) * (Q @ gamma) should equal (normalized * gamma) @ Q^T
            #
            # Let n = normalized (d-dim vector per position).
            # (n @ Q^T) * (Q @ gamma) = element-wise product
            # (n * gamma) @ Q^T = n @ diag(gamma) @ Q^T
            # These aren't equal in general.

            # OK so the key insight is: Q-rotation is NOT exactly function-preserving
            # for LayerNorm architectures like GPT-2 unless gamma/beta are constant.

            # Let's proceed anyway to see how much the function changes.
            # We'll transform the weights and measure the function change.

            # LayerNorm 1 gamma/beta
            block.ln_1.weight.copy_(Q_t @ block.ln_1.weight)
            block.ln_1.bias.copy_(Q_t @ block.ln_1.bias)

            # LayerNorm 2 gamma/beta
            block.ln_2.weight.copy_(Q_t @ block.ln_2.weight)
            block.ln_2.bias.copy_(Q_t @ block.ln_2.bias)

            # MLP c_fc: (d_model, d_ff)
            # Input is LN(x), which after our transforms is rotated.
            # (z @ Q^T) @ W_fc = z @ (Q^T @ W_fc)
            # Hmm but W_fc is (d_model, d_ff), Q^T is (d_model, d_model)
            # Q^T @ W_fc doesn't work (dim mismatch)
            # Oh wait, Q.T @ W_fc: (d,d) @ (d, d_ff) = (d, d_ff). ✓
            W_fc = block.mlp.c_fc.weight  # (d_model, d_ff)
            block.mlp.c_fc.weight.copy_(Q_t.T @ W_fc)

            # MLP c_proj: (d_ff, d_model)
            # Output goes to residual stream, need @ Q^T
            W_proj = block.mlp.c_proj.weight  # (d_ff, d_model)
            block.mlp.c_proj.weight.copy_(W_proj @ Q_t.T)

        # Final LayerNorm
        out.transformer.ln_f.weight.copy_(Q_t @ out.transformer.ln_f.weight)
        out.transformer.ln_f.bias.copy_(Q_t @ out.transformer.ln_f.bias)

        # LM head: (d_model, vocab) - but usually tied to embeddings
        # If tied, already handled by wte transform.
        # If not tied, need to transform.
        if not out.config.tie_word_embeddings:
            out.lm_head.weight.copy_(out.lm_head.weight @ Q_t.T)

    manifest = {
        "Q_shape": Q.shape,
        "Q_det": float(np.linalg.det(Q)),
        "Q_orthogonality_error": float(np.linalg.norm(Q @ Q.T - np.eye(Q.shape[0]))),
    }

    return out, manifest


def check_branch_product_transform(
    model_orig,
    model_rotated,
    Q: np.ndarray,
) -> Dict[str, Any]:
    """Verify that M -> Q @ M @ Q^T under rotation."""
    Ms_orig = extract_branch_products(model_orig)
    Ms_rot = extract_branch_products(model_rotated)

    results = []
    for i, (M_o, M_r) in enumerate(zip(Ms_orig, Ms_rot)):
        # Expected: M_r = Q @ M_o @ Q^T
        M_expected = Q @ M_o @ Q.T

        diff = np.abs(M_r - M_expected).max()
        trace_orig = np.trace(M_o)
        trace_rot = np.trace(M_r)
        trace_diff = abs(trace_orig - trace_rot)

        results.append({
            "block": i,
            "max_diff_from_expected": float(diff),
            "trace_orig": float(trace_orig),
            "trace_rot": float(trace_rot),
            "trace_diff": float(trace_diff),
        })

    return {
        "per_block": results,
        "max_diff": max(r["max_diff_from_expected"] for r in results),
        "max_trace_diff": max(r["trace_diff"] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description="Q-rotation attack test")
    parser.add_argument("--benchmark-dir", default="results/lineage_benchmark_gpt2_paper",
                        help="Path to benchmark data")
    parser.add_argument("--output", default="results/attack_q_rotation.json",
                        help="Output file")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of Q seeds to test")
    parser.add_argument("--seed-base", type=int, default=9000, help="Base seed")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load benchmark data
    print("Loading benchmark data...")
    benchmark_dir = Path(args.benchmark_dir)

    with open(benchmark_dir / "phase1_roots.pkl", "rb") as f:
        phase1 = pickle.load(f)
    with open(benchmark_dir / "phase2_descendants.pkl", "rb") as f:
        phase2 = pickle.load(f)

    tau_s = phase1["tau_s"]
    root_signatures = phase1["root_signatures"]
    model_config = ModelConfig()

    # Get d_model from first root
    d_model = root_signatures[0][0].shape[0]
    print(f"d_model = {d_model}")

    # Build test pairs (simplified - just use roots 5,6,7)
    test_roots = [5, 6, 7]
    pairs = []

    # Add descendant pairs
    for desc in phase2["descendants"]:
        if desc["root_idx"] in test_roots:
            pairs.append({
                "ref_idx": desc["root_idx"],
                "sus_id": desc["id"],
                "sus_Ms": desc["Ms"],
                "label": "descendant",
            })

    # Add distilled students
    for student in phase2["students"]:
        if student["teacher_root_idx"] in test_roots:
            pairs.append({
                "ref_idx": student["teacher_root_idx"],
                "sus_id": student["id"],
                "sus_Ms": student["Ms"],
                "label": "non_descendant",
            })

    # Add cross-root pairs
    for i, ri in enumerate(test_roots):
        for rj in test_roots[i+1:]:
            pairs.append({
                "ref_idx": ri,
                "sus_idx": rj,
                "sus_Ms": root_signatures[rj],
                "label": "non_descendant",
            })

    print(f"Test pairs: {len(pairs)}")

    # Test Q-rotation
    results = []

    for seed_idx in range(args.n_seeds):
        seed = args.seed_base + seed_idx * 100
        print(f"\n{'='*60}")
        print(f"Q-rotation seed {seed}")
        print(f"{'='*60}")

        # Generate Q
        Q = generate_orthogonal_matrix(d_model, seed)
        print(f"Q orthogonality error: {np.linalg.norm(Q @ Q.T - np.eye(d_model)):.2e}")

        for pair_idx, pair in enumerate(pairs):
            ref_idx = pair["ref_idx"]
            ref_Ms = root_signatures[ref_idx]
            sus_Ms = pair["sus_Ms"]

            # Apply Q rotation to branch products directly
            # M -> Q @ M @ Q^T
            ref_Ms_rot = [Q @ M @ Q.T for M in ref_Ms]
            # Leave suspect unchanged (or rotate with different Q)

            # Compute lineage scores
            score_orig, _, _ = ldet.lineage_score(ref_Ms, sus_Ms, tau_s)
            score_rot, _, _ = ldet.lineage_score(ref_Ms_rot, sus_Ms, tau_s)

            results.append({
                "seed": seed,
                "ref_idx": ref_idx,
                "sus_id": pair.get("sus_id", f"root_{pair.get('sus_idx')}"),
                "label": pair["label"],
                "score_orig": float(score_orig),
                "score_rot": float(score_rot),
                "score_change": float(score_rot - score_orig),
            })

            print(f"  [{pair_idx+1}/{len(pairs)}] {pair['label'][:4]}: "
                  f"orig={score_orig:.4f} rot={score_rot:.4f} "
                  f"delta={score_rot - score_orig:+.4f}")

    # Compute AUROC
    print("\n" + "="*60)
    print("AUROC Summary")
    print("="*60)

    labels = [1 if r["label"] == "descendant" else 0 for r in results]
    scores_orig = [r["score_orig"] for r in results]
    scores_rot = [r["score_rot"] for r in results]

    auroc_orig = roc_auc_score(labels, scores_orig)
    auroc_rot = roc_auc_score(labels, scores_rot)

    print(f"AUROC (original):  {auroc_orig:.4f}")
    print(f"AUROC (Q-rotated): {auroc_rot:.4f}")

    # Gap analysis
    desc_orig = [r["score_orig"] for r in results if r["label"] == "descendant"]
    desc_rot = [r["score_rot"] for r in results if r["label"] == "descendant"]
    nondesc_orig = [r["score_orig"] for r in results if r["label"] == "non_descendant"]
    nondesc_rot = [r["score_rot"] for r in results if r["label"] == "non_descendant"]

    print(f"\nDescendant scores:")
    print(f"  Original:  mean={np.mean(desc_orig):.4f} min={np.min(desc_orig):.4f}")
    print(f"  Q-rotated: mean={np.mean(desc_rot):.4f} min={np.min(desc_rot):.4f}")

    print(f"\nNon-descendant scores:")
    print(f"  Original:  mean={np.mean(nondesc_orig):.4f} max={np.max(nondesc_orig):.4f}")
    print(f"  Q-rotated: mean={np.mean(nondesc_rot):.4f} max={np.max(nondesc_rot):.4f}")

    # Save results
    output = {
        "auroc_original": auroc_orig,
        "auroc_rotated": auroc_rot,
        "desc_mean_orig": float(np.mean(desc_orig)),
        "desc_mean_rot": float(np.mean(desc_rot)),
        "nondesc_mean_orig": float(np.mean(nondesc_orig)),
        "nondesc_mean_rot": float(np.mean(nondesc_rot)),
        "per_pair_results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
