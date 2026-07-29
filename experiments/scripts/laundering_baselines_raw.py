"""Raw-weight baselines + Git-Re-Basin baselines for the laundering benchmark.

The existing weight-space baselines (lineage_baselines.py) all consume the
branch product M = W_out @ W_in, which is EXACTLY invariant to hidden-unit
permutation/rescaling -- so on M they cannot see laundering at all. To make the
baseline comparison meaningful we run them on RAW per-block weights instead:

    raw_f(A, B) = 0.5 * ( f(A.Wins, B.Wins) + f(A.Wouts, B.Wouts) )

where f is one of aligned_frobenius / singular_value_distance / weight_cosine
(reused verbatim from lineage_baselines). Wins/Wouts come from
laundering_ops.raw_weights (b_in folded into Wins).

Track B upgrades the weakest raw baseline (block-level aligned Frobenius) with
Git-Re-Basin-style unit matching (Ainsworth et al., 2023, weight-matching
variant, data-free):

    rebasin_frobenius        -- per-block Hungarian unit alignment, then Frobenius
    rebasin_scale_frobenius  -- alignment + per-unit least-squares scale removal

If the scale-aware version recovers D, that is a real finding: the paper's
framing shifts to "baseline requires solving a per-block alignment+scale
optimization; ours needs none," stated with the measured numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

import lineage_baselines as lbase

RAW_METHODS = ["raw_aligned_frobenius", "raw_singular_value_dist", "raw_weight_cosine"]
REBASIN_METHODS = ["rebasin_frobenius", "rebasin_scale_frobenius"]


def _both(fn, A: dict, B: dict) -> float:
    """Average an existing list-of-matrices baseline over Win and Wout streams."""
    return 0.5 * (fn(A["Wins"], B["Wins"]) + fn(A["Wouts"], B["Wouts"]))


def raw_aligned_frobenius(A: dict, B: dict) -> float:
    return _both(lbase.aligned_frobenius, A, B)


def raw_singular_value_dist(A: dict, B: dict) -> float:
    return _both(lbase.singular_value_distance, A, B)


def raw_weight_cosine(A: dict, B: dict) -> float:
    return _both(lbase.weight_cosine, A, B)


# --------------------------------------------------------------- Git-Re-Basin

def _match_units(Win_A, Wout_A, Win_B, Wout_B):
    """Weight-matching unit assignment (Ainsworth et al. 2023, data-free).

    Maximize sum of per-unit weight inner products across BOTH incident weight
    groups:  C[i,j] = <Win_A[i], Win_B[j]> + <Wout_A[:,i], Wout_B[:,j]>.
    Returns col_ind: reference unit i is matched to suspect unit col_ind[i].
    """
    C = Win_A @ Win_B.T + Wout_A.T @ Wout_B          # (h, h)
    row, col = linear_sum_assignment(-C)             # maximize similarity
    return col


def _rebasin_block(Win_A, Wout_A, Win_B, Wout_B, scale_aware: bool):
    """Align suspect block B's units to reference A, optionally removing scale.

    Returns (Win_B_aligned, Wout_B_aligned). Inverts a permutation for free and,
    when scale_aware, a positive per-unit diagonal via least squares.
    """
    col = _match_units(Win_A, Wout_A, Win_B, Wout_B)
    Win_Bp = Win_B[col, :].copy()      # permute rows (units) of W_in
    Wout_Bp = Wout_B[:, col].copy()    # permute cols (units) of W_out

    if scale_aware:
        # Per unit i, LS scale d_i minimizing ||Win_A[i] - d_i * Win_Bp[i]||:
        #   d_i = <Win_Bp[i], Win_A[i]> / ||Win_Bp[i]||^2
        # Undo it: Win_Bp[i] *= d_i, Wout_Bp[:,i] /= d_i (keeps M unchanged).
        num = np.sum(Win_Bp * Win_A, axis=1)
        den = np.sum(Win_Bp * Win_Bp, axis=1) + 1e-12
        d = num / den
        d = np.where(np.abs(d) < 1e-8, 1.0, d)   # guard degenerate units
        Win_Bp = Win_Bp * d[:, None]
        Wout_Bp = Wout_Bp / d[None, :]
    return Win_Bp, Wout_Bp


def _rebasin_score(A: dict, B: dict, scale_aware: bool) -> float:
    """Block-index-aligned Frobenius similarity after per-block unit matching.

    Returns a similarity (negated, scale-normalized mean Frobenius distance),
    matching lineage_baselines.aligned_frobenius's convention/normalization.
    """
    L = len(A["Wins"])
    dists, norms = [], []
    for i in range(L):
        Win_A, Wout_A = A["Wins"][i], A["Wouts"][i]
        Win_B, Wout_B = B["Wins"][i], B["Wouts"][i]
        Win_Bp, Wout_Bp = _rebasin_block(Win_A, Wout_A, Win_B, Wout_B, scale_aware)
        d_in = np.linalg.norm(Win_A - Win_Bp, ord="fro")
        d_out = np.linalg.norm(Wout_A - Wout_Bp, ord="fro")
        dists.append(0.5 * (d_in + d_out))
        norms.append(0.5 * (np.linalg.norm(Win_A, ord="fro")
                            + np.linalg.norm(Wout_A, ord="fro")))
    mean_dist = float(np.mean(dists))
    mean_norm = float(np.mean(norms)) + 1e-12
    return -mean_dist / mean_norm


def rebasin_frobenius(A: dict, B: dict) -> float:
    return _rebasin_score(A, B, scale_aware=False)


def rebasin_scale_frobenius(A: dict, B: dict) -> float:
    return _rebasin_score(A, B, scale_aware=True)


# Registry: name -> fn(A_raw, B_raw) taking dicts from laundering_ops.raw_weights.
RAW_SCORERS = {
    "raw_aligned_frobenius": raw_aligned_frobenius,
    "raw_singular_value_dist": raw_singular_value_dist,
    "raw_weight_cosine": raw_weight_cosine,
    "rebasin_frobenius": rebasin_frobenius,
    "rebasin_scale_frobenius": rebasin_scale_frobenius,
}
