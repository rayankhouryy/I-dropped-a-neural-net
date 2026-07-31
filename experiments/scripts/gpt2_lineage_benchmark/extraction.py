"""Branch product extraction for GPT-2-style transformers."""
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import GPT2LMHeadModel


def extract_branch_products(model: GPT2LMHeadModel) -> List[np.ndarray]:
    """Extract MLP branch products M = W_proj @ W_fc for each transformer block.

    GPT-2 Conv1D layers store weights as (in_features, out_features), so we
    need to transpose to get standard (out_features, in_features) format.

    For each block:
        W_fc: (d_model, d_ff) -> input projection
        W_proj: (d_ff, d_model) -> output projection
        M = W_proj @ W_fc: (d_model, d_model)
    """
    Ms = []
    for block in model.transformer.h:
        # GPT-2 uses Conv1D which stores as (in, out)
        # c_fc: d_model -> d_ff
        # c_proj: d_ff -> d_model
        W_fc = block.mlp.c_fc.weight.detach().cpu().numpy().astype(np.float64)
        W_proj = block.mlp.c_proj.weight.detach().cpu().numpy().astype(np.float64)

        # Conv1D stores as (in, out), so W_fc is (d_model, d_ff)
        # and W_proj is (d_ff, d_model)
        # M = W_proj @ W_fc would be (d_ff, d_ff) which is wrong
        # We want M = W_proj.T @ W_fc.T = (d_model, d_ff) @ (d_ff, d_model) = (d_model, d_model)
        # Actually: W_proj.T is (d_model, d_ff), W_fc.T is (d_ff, d_model)
        # M = W_proj.T @ W_fc.T = (d_model, d_ff) @ (d_ff, d_model) - wrong dimensions

        # Let's think again: for residual x + W_proj(relu(W_fc(x)))
        # W_fc maps d_model -> d_ff, so it's (d_ff, d_model) in standard form
        # W_proj maps d_ff -> d_model, so it's (d_model, d_ff) in standard form
        # Conv1D stores as (in, out), so:
        #   c_fc.weight is (d_model, d_ff) - need transpose to get (d_ff, d_model)
        #   c_proj.weight is (d_ff, d_model) - need transpose to get (d_model, d_ff)
        # M = W_proj @ W_fc = (d_model, d_ff) @ (d_ff, d_model) = (d_model, d_model)

        W_fc_std = W_fc.T  # (d_ff, d_model)
        W_proj_std = W_proj.T  # (d_model, d_ff)
        M = W_proj_std @ W_fc_std  # (d_model, d_model)

        Ms.append(M)

    return Ms


def extract_attention_products(model: GPT2LMHeadModel) -> Dict[str, List[np.ndarray]]:
    """Extract attention projection products (optional, for extended analysis).

    Returns:
        Dict with 'qk' and 'vo' products for each layer.
    """
    qk_products = []
    vo_products = []

    for block in model.transformer.h:
        attn = block.attn

        # GPT-2 uses combined c_attn for Q, K, V
        # c_attn.weight is (d_model, 3 * d_model)
        c_attn_weight = attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)
        d_model = c_attn_weight.shape[0]

        # Split into Q, K, V projections
        W_q = c_attn_weight[:, :d_model].T  # (d_model, d_model)
        W_k = c_attn_weight[:, d_model:2*d_model].T  # (d_model, d_model)
        W_v = c_attn_weight[:, 2*d_model:].T  # (d_model, d_model)

        # Output projection
        W_o = attn.c_proj.weight.detach().cpu().numpy().astype(np.float64).T  # (d_model, d_model)

        # Q/K product (bilinear form)
        M_qk = W_q @ W_k.T  # (d_model, d_model)
        qk_products.append(M_qk)

        # V/O product
        M_vo = W_o @ W_v  # (d_model, d_model)
        vo_products.append(M_vo)

    return {"qk": qk_products, "vo": vo_products}


def compute_diagonal_score(M: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized trace concentration s(M) = |tr(M)| / ||M||_F."""
    trace = np.trace(M)
    frob = np.linalg.norm(M, 'fro')
    return abs(trace) / (frob + eps)


def compute_residual_signature(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute centered residual signature phi(M).

    R = M - (tr(M)/d) * I
    phi = vec(R) / ||vec(R)||
    """
    d = M.shape[0]
    alpha = np.trace(M) / d
    R = M - alpha * np.eye(d)
    R_vec = R.flatten()
    norm = np.linalg.norm(R_vec)
    return R_vec / (norm + eps)


def extract_all_signatures(
    model: GPT2LMHeadModel,
) -> Dict[str, Any]:
    """Extract branch products and compute signatures for a model."""
    Ms = extract_branch_products(model)

    signatures = []
    diag_scores = []

    for M in Ms:
        signatures.append(compute_residual_signature(M))
        diag_scores.append(compute_diagonal_score(M))

    return {
        "branch_products": Ms,
        "signatures": signatures,
        "diag_scores": diag_scores,
        "mean_diag_score": float(np.mean(diag_scores)),
    }


def signatures_to_numpy(signatures: List[np.ndarray]) -> np.ndarray:
    """Stack signatures into a single matrix for efficient comparison."""
    return np.stack(signatures, axis=0)
