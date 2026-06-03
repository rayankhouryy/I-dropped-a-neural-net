"""Core algorithm for model-level lineage detection (Issue #30).

Implements the residual-signature lineage metric proposed by @singh96aman:

  M_l                  branch product
  alpha_l = tr(M_l)/d   best-fit identity scalar
  R_l = M_l - alpha_l I residual signature (the "E" matrix from Prop margin)
  phi(M_l)             = vec(R_l) / ||vec(R_l)||
  s(M_l) = |tr(M_l)|/||M_l||_F   diagonal-dominance score
  C_lm(A,B) = <phi(M_l^A), phi(M_m^B)>
  G_lm(A,B) = C_lm * min(s(M_l^A)/tau_s, s(M_m^B)/tau_s, 1)
  L(A,B)   = mean over Hungarian-aligned (l,pi(l)) of G_l,pi(l)
  Z(A,B)   = (L(A,B) - mu(null)) / sigma(null)

All functions are pure numpy; the lineage score does not depend on any
specific architecture beyond requiring square branch products of equal
size between the reference A and suspect B.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------- core
def best_identity_scalar(M: np.ndarray) -> float:
    """alpha* = argmin_alpha ||M - alpha I||_F  =  tr(M) / d.

    For dynamic-isometry-trained residual branches we expect this to be
    close to -epsilon < 0; the residual signature R = M - alpha* I is the
    branch-specific deviation from the generic isometry component.
    """
    d = M.shape[0]
    return float(np.trace(M)) / d


def residual_signature(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """phi(M) = vec(M - (tr(M)/d) I) / ||...||_2 .

    Returns a unit-norm 1-D float64 vector; the leading scalar identity
    component has been subtracted before flattening.
    """
    M = np.asarray(M, dtype=np.float64)
    d = M.shape[0]
    alpha = np.trace(M) / d
    R = M - alpha * np.eye(d, dtype=np.float64)
    v = R.reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps)


def diag_score(M: np.ndarray, eps: float = 1e-12) -> float:
    """s(M) = |tr(M)| / ||M||_F  (the dimensionless diagonal-dominance ratio)."""
    M = np.asarray(M, dtype=np.float64)
    return float(abs(np.trace(M))) / (float(np.linalg.norm(M, 'fro')) + eps)


def gated_branch_score(M_a: np.ndarray, M_b: np.ndarray, tau_s: float,
                       eps: float = 1e-12) -> float:
    """G(M_a, M_b) = cos(phi_a, phi_b) * min(s_a/tau, s_b/tau, 1).

    The gate downweights branches that are not strongly trained (i.e., not
    sufficiently diagonal-dominant). When both s_a, s_b >= tau_s the gate
    is 1.0 and the score reduces to the cosine of residual signatures.
    """
    phi_a = residual_signature(M_a, eps)
    phi_b = residual_signature(M_b, eps)
    cos = float(phi_a @ phi_b)
    s_a = diag_score(M_a, eps)
    s_b = diag_score(M_b, eps)
    gate = min(s_a / (tau_s + eps), s_b / (tau_s + eps), 1.0)
    return cos * gate


def compute_score_matrix(Ms_A, Ms_B, tau_s, eps: float = 1e-12) -> np.ndarray:
    """G_lm(A, B) for every (reference branch, suspect branch) pair."""
    L_a, L_b = len(Ms_A), len(Ms_B)
    G = np.zeros((L_a, L_b), dtype=np.float64)
    # Precompute signatures and diag scores once.
    phi_A = [residual_signature(M, eps) for M in Ms_A]
    phi_B = [residual_signature(M, eps) for M in Ms_B]
    s_A   = [diag_score(M, eps) for M in Ms_A]
    s_B   = [diag_score(M, eps) for M in Ms_B]
    for i in range(L_a):
        for j in range(L_b):
            cos = float(phi_A[i] @ phi_B[j])
            gate = min(s_A[i] / (tau_s + eps),
                       s_B[j] / (tau_s + eps), 1.0)
            G[i, j] = cos * gate
    return G


def lineage_score(Ms_A, Ms_B, tau_s, eps: float = 1e-12):
    """Hungarian-aligned mean of gated branch scores.

    Returns (lineage_score, permutation, full_score_matrix).
    The permutation is the column index assigned to each row (i.e.
    reference branch i is matched to suspect branch perm[i]).
    """
    G = compute_score_matrix(Ms_A, Ms_B, tau_s, eps)
    # scipy minimizes; negate to maximize.
    row_ind, col_ind = linear_sum_assignment(-G)
    # We want the mean of G[i, perm(i)] for each reference row.
    # When L_a == L_b this is a full matching.
    matched = G[row_ind, col_ind]
    return float(matched.mean()), col_ind, G


# --------------------------------------------------------------------- baselines
def diag_only_score(Ms_A, Ms_B, **kw) -> float:
    """Mean diagonal-dominance score of the SUSPECT (does not use A at all).

    This baseline tests whether B is a trained residual model; it cannot
    in principle distinguish descendant B from an independently-trained
    same-architecture B' since both have similar mean diagonal-dominance.
    """
    return float(np.mean([diag_score(M) for M in Ms_B]))


def raw_cos_score(Ms_A, Ms_B, **kw) -> float:
    """Mean cosine of flattened (raw) branch products, identity alignment."""
    L = min(len(Ms_A), len(Ms_B))
    out = 0.0
    for i in range(L):
        a = np.asarray(Ms_A[i], dtype=np.float64).reshape(-1)
        b = np.asarray(Ms_B[i], dtype=np.float64).reshape(-1)
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        out += float(a @ b) / (na * nb)
    return out / L


def frob_distance(Ms_A, Ms_B, **kw) -> float:
    """Negative mean Frobenius distance over identity-aligned branches.

    Sign chosen so 'higher = more similar' to match the lineage metric.
    """
    L = min(len(Ms_A), len(Ms_B))
    out = 0.0
    for i in range(L):
        out -= float(np.linalg.norm(
            np.asarray(Ms_A[i], dtype=np.float64)
            - np.asarray(Ms_B[i], dtype=np.float64),
            ord='fro'))
    return out / L


# --------------------------------------------------------------------- calibration
def calibrate_z_score(score: float, null_scores) -> float:
    """Z(A,B) = (L(A,B) - mu_N) / sigma_N where N is the null distribution."""
    null_scores = np.asarray(null_scores, dtype=np.float64)
    mu = float(null_scores.mean())
    sigma = float(null_scores.std(ddof=0))
    return (score - mu) / (sigma + 1e-12)


def choose_tau_s(reference_Ms_list) -> float:
    """Pick the gate threshold tau_s.

    Strategy: the gate should equal 1.0 for every branch of every reference
    model. We choose tau_s = minimum s(M) across all branches and all
    references, so reference branches always pass the gate at strength 1.
    """
    all_s = []
    for Ms in reference_Ms_list:
        for M in Ms:
            all_s.append(diag_score(M))
    return float(np.min(all_s))


# --------------------------------------------------------------------- evaluation
def evaluate_lineage(records):
    """Compute AUROC, AUPRC, TPR@1%FPR from a list of records.

    Each record must have keys: 'label' ('descendant' or 'non_descendant'),
    'score' (the lineage score or z-score to threshold).

    Returns a dict with metric values plus the per-threshold ROC curve.
    """
    labels = np.asarray([1 if r['label'] == 'descendant' else 0
                         for r in records], dtype=np.int64)
    scores = np.asarray([r['score'] for r in records], dtype=np.float64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {'error': 'need both classes in records'}

    # Sort scores descending and sweep
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    # AUROC by trapezoidal rule (numpy 1.x had trapz, 2.x renamed to trapezoid)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    auroc = float(trapz(np.concatenate([[0], tpr, [1]]),
                        np.concatenate([[0], fpr, [1]])))

    # AUPRC
    precision = tp / np.maximum(tp + fp, 1)
    auprc = float(trapz(precision, tpr))

    # TPR at 1% FPR (find largest TPR with FPR <= 0.01)
    mask = fpr <= 0.01
    tpr_at_1pct = float(tpr[mask].max()) if mask.any() else 0.0
    mask10 = fpr <= 0.10
    tpr_at_10pct = float(tpr[mask10].max()) if mask10.any() else 0.0

    return {
        'auroc':         auroc,
        'auprc':         auprc,
        'tpr_at_1pct':   tpr_at_1pct,
        'tpr_at_10pct':  tpr_at_10pct,
        'n_positive':    n_pos,
        'n_negative':    n_neg,
        'roc_fpr':       fpr.tolist(),
        'roc_tpr':       tpr.tolist(),
    }
