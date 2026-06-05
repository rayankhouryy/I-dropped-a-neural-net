"""Lineage verification baselines (issue #44).

All implementations are pure NumPy / SciPy except CKA/SVCCA which optionally
take a probe-activation tensor. None of these require a classification head,
so they all work on our existing regression MLP / ResNet benchmarks.

Conventions:
- Higher score = "more likely descendant" (so distances are returned negated).
- Functions that operate on branch products take a list of np.ndarray of
  matching shape per layer; the lists must have the same length L.

Baselines included:
    Weight-space (data-free, comparable to ours):
        aligned_frobenius
        singular_value_distance
        weight_cosine
    Activation-space (require probe data):
        linear_cka
        svcca
    Decision-boundary (require classifier):
        ipguard_match_rate
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------- helpers

def _f64(M: np.ndarray) -> np.ndarray:
    return np.asarray(M, dtype=np.float64)


# --------------------------------------------------------------- weight-space

def aligned_frobenius(Ms_A: Sequence[np.ndarray],
                      Ms_B: Sequence[np.ndarray]) -> float:
    """Frobenius distance with Hungarian layer alignment, returned as a
    similarity (negated mean distance, normalized to be roughly in [-1, 1])."""
    L = len(Ms_A)
    assert len(Ms_B) == L
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            D[i, j] = np.linalg.norm(_f64(Ms_A[i]) - _f64(Ms_B[j]), ord="fro")
    row, col = linear_sum_assignment(D)
    mean_dist = float(D[row, col].mean())
    # normalize by mean ||A||_F to be scale-invariant -> distance ratio
    mean_norm = float(np.mean([np.linalg.norm(_f64(M), ord="fro") for M in Ms_A]))
    return -mean_dist / (mean_norm + 1e-12)


def singular_value_distance(Ms_A: Sequence[np.ndarray],
                            Ms_B: Sequence[np.ndarray]) -> float:
    """Wasserstein-1 between sorted singular-value spectra, averaged over the
    Hungarian-matched layer assignment, returned as similarity (negated)."""
    L = len(Ms_A)
    assert len(Ms_B) == L
    sA = [np.sort(np.linalg.svd(_f64(M), compute_uv=False))[::-1] for M in Ms_A]
    sB = [np.sort(np.linalg.svd(_f64(M), compute_uv=False))[::-1] for M in Ms_B]
    D = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            k = min(len(sA[i]), len(sB[j]))
            D[i, j] = float(np.mean(np.abs(sA[i][:k] - sB[j][:k])))
    row, col = linear_sum_assignment(D)
    mean_dist = float(D[row, col].mean())
    mean_scale = float(np.mean([s[0] for s in sA]) + 1e-12)
    return -mean_dist / mean_scale


def weight_cosine(Ms_A: Sequence[np.ndarray],
                  Ms_B: Sequence[np.ndarray]) -> float:
    """Cosine similarity of flattened branch products with Hungarian
    alignment, returned as mean cosine over matched pairs."""
    L = len(Ms_A)
    assert len(Ms_B) == L
    fA = [_f64(M).ravel() for M in Ms_A]
    fB = [_f64(M).ravel() for M in Ms_B]
    D = np.zeros((L, L))  # negated cosine for argmin alignment
    for i in range(L):
        for j in range(L):
            na = np.linalg.norm(fA[i]) + 1e-12
            nb = np.linalg.norm(fB[j]) + 1e-12
            D[i, j] = -float(fA[i] @ fB[j]) / (na * nb)
    row, col = linear_sum_assignment(D)
    return float(-D[row, col].mean())


# ----------------------------------------------------------- activation-space

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., ICML 2019).

    X: (n_samples, d_x), Y: (n_samples, d_y)
    Returns scalar in [0, 1].
    """
    X = _f64(X) - _f64(X).mean(axis=0, keepdims=True)
    Y = _f64(Y) - _f64(Y).mean(axis=0, keepdims=True)
    # HSIC via the unbiased linear-kernel trick: ||X^T Y||_F^2 / (...)
    # (equivalent to using Gram matrices, but cheaper when n >> d)
    cross = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    nA = np.linalg.norm(X.T @ X, ord="fro")
    nB = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(cross / (nA * nB + 1e-12))


def svcca(X: np.ndarray, Y: np.ndarray, variance_threshold: float = 0.99) -> float:
    """SVCCA mean correlation (Raghu et al., NeurIPS 2017).

    Keep singular vectors of X and Y that explain `variance_threshold`
    fraction of variance, then return mean canonical correlation.
    """
    Xc = _f64(X) - _f64(X).mean(axis=0, keepdims=True)
    Yc = _f64(Y) - _f64(Y).mean(axis=0, keepdims=True)
    Ux, sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Yc, full_matrices=False)
    def keep(s):
        c = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-12)
        k = int(np.searchsorted(c, variance_threshold) + 1)
        return max(1, min(k, len(s)))
    kx, ky = keep(sx), keep(sy)
    Ux, Uy = Ux[:, :kx], Uy[:, :ky]
    Qx, _ = np.linalg.qr(Ux)
    Qy, _ = np.linalg.qr(Uy)
    _, corrs, _ = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    return float(np.mean(corrs))


def matched_activation_score(actsA: Sequence[np.ndarray],
                             actsB: Sequence[np.ndarray],
                             metric=linear_cka) -> float:
    """Apply a per-layer activation metric (CKA / SVCCA) with Hungarian
    layer alignment; return mean matched similarity."""
    L = len(actsA)
    assert len(actsB) == L
    S = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            S[i, j] = metric(actsA[i], actsB[j])
    # Hungarian minimizes cost, so feed negative similarity
    row, col = linear_sum_assignment(-S)
    return float(S[row, col].mean())


def cka_lineage_score(actsA, actsB) -> float:
    return matched_activation_score(actsA, actsB, metric=linear_cka)


def svcca_lineage_score(actsA, actsB) -> float:
    return matched_activation_score(actsA, actsB, metric=svcca)


# --------------------------------------------------------- decision-boundary

def ipguard_match_rate(predsA: np.ndarray, predsB: np.ndarray) -> float:
    """Fraction of inputs on which the two classifiers agree.

    For our regression benchmark we adapt this to: fraction of probe points
    whose ranks under both predictors fall in the same quantile bin
    (8 bins by default). This generalizes IPGuard's classification-agreement
    rate to regression outputs.
    """
    predsA = _f64(predsA).ravel()
    predsB = _f64(predsB).ravel()
    if predsA.dtype.kind in ("i", "u"):
        return float(np.mean(predsA == predsB))
    bins = np.quantile(predsA, np.linspace(0, 1, 9))
    bins[0], bins[-1] = -np.inf, np.inf
    bA = np.digitize(predsA, bins) - 1
    bB = np.digitize(predsB, bins) - 1
    return float(np.mean(bA == bB))


# --------------------------------------------------------------------- main

ALL_METHODS = {
    "aligned_frobenius": ("weight", aligned_frobenius),
    "singular_value_dist": ("weight", singular_value_distance),
    "weight_cosine": ("weight", weight_cosine),
    "cka": ("activation", cka_lineage_score),
    "svcca": ("activation", svcca_lineage_score),
    "ipguard_regr": ("predictions", ipguard_match_rate),
}


if __name__ == "__main__":
    # Smoke test: identical inputs -> max similarity for each method.
    rng = np.random.default_rng(0)
    Ms = [rng.standard_normal((8, 8)) for _ in range(5)]
    acts = [rng.standard_normal((100, 16)) for _ in range(5)]
    preds = rng.standard_normal(100)
    print("aligned_frobenius (self):", aligned_frobenius(Ms, Ms))
    print("singular_value_dist (self):", singular_value_distance(Ms, Ms))
    print("weight_cosine (self):", weight_cosine(Ms, Ms))
    print("cka (self):", cka_lineage_score(acts, acts))
    print("svcca (self):", svcca_lineage_score(acts, acts))
    print("ipguard_regr (self):", ipguard_match_rate(preds, preds))
