"""Raw-weight baselines + Git-Re-Basin baselines for GPT-2 laundering benchmark.

The proposed method operates on the branch product M = W_proj @ W_fc, which is
EXACTLY invariant to hidden-unit permutation. To make the baseline comparison
meaningful we run raw-weight baselines that should COLLAPSE under permutation:

    raw_weight_cosine_gpt2      -- cosine of flattened raw MLP weights
    raw_aligned_frobenius_gpt2  -- Frobenius distance with block alignment
    singular_value_distance_gpt2 -- may be invariant (eigenvalues unchanged)

Track B upgrades the raw baselines with Git-Re-Basin-style unit matching
(Ainsworth et al., 2023, weight-matching variant, data-free):

    rebasin_frobenius_gpt2       -- per-block Hungarian unit alignment, then Frobenius
    rebasin_scale_frobenius_gpt2 -- alignment + per-unit least-squares scale removal
                                    (NOTE: approximate for GELU, exact only for ReLU)

Adapted from laundering_baselines_raw.py for GPT-2 Conv1D conventions.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- raw baselines

def raw_weight_cosine_gpt2(A: Dict[str, List[np.ndarray]], B: Dict[str, List[np.ndarray]]) -> float:
    """Cosine similarity of flattened raw MLP weights.

    Should COLLAPSE under permutation since weights are scrambled.
    """
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        na = np.linalg.norm(a_flat) + 1e-12
        nb = np.linalg.norm(b_flat) + 1e-12
        return float(np.dot(a_flat, b_flat) / (na * nb))

    L = len(A["Wins"])
    cosines_in = [_cosine(A["Wins"][i], B["Wins"][i]) for i in range(L)]
    cosines_out = [_cosine(A["Wouts"][i], B["Wouts"][i]) for i in range(L)]

    # Average over all blocks and both weight matrices
    return float(np.mean(cosines_in + cosines_out))


def raw_aligned_frobenius_gpt2(A: Dict[str, List[np.ndarray]], B: Dict[str, List[np.ndarray]]) -> float:
    """Frobenius distance with block-level Hungarian alignment.

    Returns similarity (negated normalized distance) for consistent AUROC.
    Should COLLAPSE under permutation.
    """
    L = len(A["Wins"])

    # Build cost matrix: Frobenius distance between each block pair
    cost = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            d_in = np.linalg.norm(A["Wins"][i] - B["Wins"][j], ord='fro')
            d_out = np.linalg.norm(A["Wouts"][i] - B["Wouts"][j], ord='fro')
            cost[i, j] = d_in + d_out

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dist = cost[row_ind, col_ind]

    # Normalize by average norm
    norms = []
    for i in range(L):
        norms.append(np.linalg.norm(A["Wins"][i], ord='fro'))
        norms.append(np.linalg.norm(A["Wouts"][i], ord='fro'))
    mean_norm = float(np.mean(norms)) + 1e-12

    # Return as similarity (negated distance)
    return -float(np.mean(matched_dist)) / mean_norm


def singular_value_distance_gpt2(A: Dict[str, List[np.ndarray]], B: Dict[str, List[np.ndarray]]) -> float:
    """Wasserstein-1 distance between sorted singular value spectra.

    NOTE: This may be INVARIANT to permutation since singular values are
    permutation-invariant. This is a legitimate strong baseline.
    """
    L = len(A["Wins"])

    def _svd_dist(a: np.ndarray, b: np.ndarray) -> float:
        sa = np.linalg.svd(a, compute_uv=False)
        sb = np.linalg.svd(b, compute_uv=False)
        # Pad to same length
        max_len = max(len(sa), len(sb))
        sa_pad = np.concatenate([np.sort(sa)[::-1], np.zeros(max_len - len(sa))])
        sb_pad = np.concatenate([np.sort(sb)[::-1], np.zeros(max_len - len(sb))])
        return float(np.abs(sa_pad - sb_pad).sum())

    dists = []
    for i in range(L):
        dists.append(_svd_dist(A["Wins"][i], B["Wins"][i]))
        dists.append(_svd_dist(A["Wouts"][i], B["Wouts"][i]))

    # Normalize by average spectral norm
    norms = []
    for i in range(L):
        norms.append(np.linalg.svd(A["Wins"][i], compute_uv=False)[0])
        norms.append(np.linalg.svd(A["Wouts"][i], compute_uv=False)[0])
    mean_norm = float(np.mean(norms)) + 1e-12

    # Return as similarity (negated distance)
    return -float(np.mean(dists)) / mean_norm


# --------------------------------------------------------------- Git-Re-Basin

def _match_units_gpt2(
    Win_A: np.ndarray,
    Wout_A: np.ndarray,
    Win_B: np.ndarray,
    Wout_B: np.ndarray,
) -> np.ndarray:
    """Weight-matching unit assignment (Ainsworth et al. 2023, data-free).

    Maximize sum of per-unit weight inner products across BOTH incident weight
    groups: C[i,j] = <Win_A[i], Win_B[j]> + <Wout_A[:,i], Wout_B[:,j]>.

    Args:
        Win_A, Win_B: (d_ff, d_model+1) with bias folded
        Wout_A, Wout_B: (d_model, d_ff)

    Returns:
        col_ind: reference unit i is matched to suspect unit col_ind[i]
    """
    # Win: rows are hidden units
    # Wout: columns are hidden units
    C = Win_A @ Win_B.T + Wout_A.T @ Wout_B  # (d_ff, d_ff)
    row_ind, col_ind = linear_sum_assignment(-C)  # maximize similarity
    return col_ind


def _rebasin_block_gpt2(
    Win_A: np.ndarray,
    Wout_A: np.ndarray,
    Win_B: np.ndarray,
    Wout_B: np.ndarray,
    scale_aware: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align suspect block B's units to reference A, optionally removing scale.

    Returns (Win_B_aligned, Wout_B_aligned).

    For GELU, scale removal is APPROXIMATE (GELU is not positively homogeneous).
    For ReLU, it would be exact.
    """
    col = _match_units_gpt2(Win_A, Wout_A, Win_B, Wout_B)

    # Permute hidden units
    Win_Bp = Win_B[col, :].copy()    # permute rows (units) of W_in
    Wout_Bp = Wout_B[:, col].copy()  # permute cols (units) of W_out

    if scale_aware:
        # Per unit i, LS scale d_i minimizing ||Win_A[i] - d_i * Win_Bp[i]||:
        #   d_i = <Win_Bp[i], Win_A[i]> / ||Win_Bp[i]||^2
        # Apply: Win_Bp[i] *= d_i, Wout_Bp[:,i] /= d_i
        # NOTE: This is approximate for GELU since GELU(d*x) != d*GELU(x)
        num = np.sum(Win_Bp * Win_A, axis=1)
        den = np.sum(Win_Bp * Win_Bp, axis=1) + 1e-12
        d = num / den
        d = np.where(np.abs(d) < 1e-8, 1.0, d)  # guard degenerate units
        Win_Bp = Win_Bp * d[:, None]
        Wout_Bp = Wout_Bp / d[None, :]

    return Win_Bp, Wout_Bp


def rebasin_frobenius_gpt2(A: Dict[str, List[np.ndarray]], B: Dict[str, List[np.ndarray]]) -> float:
    """Block-index-aligned Frobenius similarity after per-block unit matching.

    Should RECOVER lineage detection after permutation by solving the alignment.
    """
    L = len(A["Wins"])
    dists, norms = [], []

    for i in range(L):
        Win_A, Wout_A = A["Wins"][i], A["Wouts"][i]
        Win_B, Wout_B = B["Wins"][i], B["Wouts"][i]

        Win_Bp, Wout_Bp = _rebasin_block_gpt2(Win_A, Wout_A, Win_B, Wout_B, scale_aware=False)

        d_in = np.linalg.norm(Win_A - Win_Bp, ord='fro')
        d_out = np.linalg.norm(Wout_A - Wout_Bp, ord='fro')
        dists.append(0.5 * (d_in + d_out))
        norms.append(0.5 * (np.linalg.norm(Win_A, ord='fro') + np.linalg.norm(Wout_A, ord='fro')))

    mean_dist = float(np.mean(dists))
    mean_norm = float(np.mean(norms)) + 1e-12

    return -mean_dist / mean_norm


def rebasin_scale_frobenius_gpt2(A: Dict[str, List[np.ndarray]], B: Dict[str, List[np.ndarray]]) -> float:
    """Re-Basin with per-unit scale removal.

    NOTE: Scale removal is APPROXIMATE for GELU (exact only for ReLU).
    """
    L = len(A["Wins"])
    dists, norms = [], []

    for i in range(L):
        Win_A, Wout_A = A["Wins"][i], A["Wouts"][i]
        Win_B, Wout_B = B["Wins"][i], B["Wouts"][i]

        Win_Bp, Wout_Bp = _rebasin_block_gpt2(Win_A, Wout_A, Win_B, Wout_B, scale_aware=True)

        d_in = np.linalg.norm(Win_A - Win_Bp, ord='fro')
        d_out = np.linalg.norm(Wout_A - Wout_Bp, ord='fro')
        dists.append(0.5 * (d_in + d_out))
        norms.append(0.5 * (np.linalg.norm(Win_A, ord='fro') + np.linalg.norm(Wout_A, ord='fro')))

    mean_dist = float(np.mean(dists))
    mean_norm = float(np.mean(norms)) + 1e-12

    return -mean_dist / mean_norm


# ---------------------------------------------------------------- ablation methods

def raw_branch_product_cosine(Ms_A: List[np.ndarray], Ms_B: List[np.ndarray]) -> float:
    """Cosine of flattened branch products WITHOUT centering.

    This is trivially P-invariant since M = W_proj @ W_fc is invariant.
    Used for ablation to show what centering adds to discrimination.
    """
    L = min(len(Ms_A), len(Ms_B))
    cosines = []

    for i in range(L):
        a = Ms_A[i].flatten()
        b = Ms_B[i].flatten()
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        cosines.append(float(np.dot(a, b) / (na * nb)))

    return float(np.mean(cosines))


def centered_branch_product_cosine(Ms_A: List[np.ndarray], Ms_B: List[np.ndarray]) -> float:
    """Centered residual signature cosine WITHOUT trace gating.

    Subtracts tr(M)/d * I before computing cosine.
    """
    L = min(len(Ms_A), len(Ms_B))
    cosines = []

    for i in range(L):
        M_a = Ms_A[i].astype(np.float64)
        M_b = Ms_B[i].astype(np.float64)

        d = M_a.shape[0]
        alpha_a = np.trace(M_a) / d
        alpha_b = np.trace(M_b) / d

        R_a = (M_a - alpha_a * np.eye(d)).flatten()
        R_b = (M_b - alpha_b * np.eye(d)).flatten()

        na = np.linalg.norm(R_a) + 1e-12
        nb = np.linalg.norm(R_b) + 1e-12
        cosines.append(float(np.dot(R_a, R_b) / (na * nb)))

    return float(np.mean(cosines))


def centered_with_gating(
    Ms_A: List[np.ndarray],
    Ms_B: List[np.ndarray],
    tau_s: float,
) -> float:
    """Centered residual signature with trace gating but NO Hungarian alignment.

    Uses identity block matching (block i in A matches block i in B).
    """
    L = min(len(Ms_A), len(Ms_B))
    gated_scores = []

    for i in range(L):
        M_a = Ms_A[i].astype(np.float64)
        M_b = Ms_B[i].astype(np.float64)

        d = M_a.shape[0]
        alpha_a = np.trace(M_a) / d
        alpha_b = np.trace(M_b) / d

        R_a = (M_a - alpha_a * np.eye(d)).flatten()
        R_b = (M_b - alpha_b * np.eye(d)).flatten()

        na = np.linalg.norm(R_a) + 1e-12
        nb = np.linalg.norm(R_b) + 1e-12
        cos = float(np.dot(R_a, R_b) / (na * nb))

        # Diagonal dominance scores
        s_a = abs(np.trace(M_a)) / (np.linalg.norm(M_a, 'fro') + 1e-12)
        s_b = abs(np.trace(M_b)) / (np.linalg.norm(M_b, 'fro') + 1e-12)

        # Gate
        gate = min(s_a / (tau_s + 1e-12), s_b / (tau_s + 1e-12), 1.0)
        gated_scores.append(cos * gate)

    return float(np.mean(gated_scores))


# ---------------------------------------------------------------- Registry

RAW_METHODS = {
    "raw_weight_cosine": raw_weight_cosine_gpt2,
    "raw_aligned_frobenius": raw_aligned_frobenius_gpt2,
    "singular_value_distance": singular_value_distance_gpt2,
}

REBASIN_METHODS = {
    "rebasin_frobenius": rebasin_frobenius_gpt2,
    "rebasin_scale_frobenius": rebasin_scale_frobenius_gpt2,
}

# Combined registry for all methods that take raw weights
RAW_SCORERS = {**RAW_METHODS, **REBASIN_METHODS}

# Ablation methods (take branch products Ms, not raw weights)
ABLATION_METHODS = {
    "raw_branch_product_cosine": raw_branch_product_cosine,
    "centered_branch_product_cosine": centered_branch_product_cosine,
    # centered_with_gating needs tau_s, handled separately
}
