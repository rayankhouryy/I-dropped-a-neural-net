"""Evaluation: lineage scoring, AUROC, bootstrap CIs."""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_lineage_score(
    Ms_A: List[np.ndarray],
    Ms_B: List[np.ndarray],
    tau_s: float = 0.5,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute lineage score L(A, B) with Hungarian alignment.

    Args:
        Ms_A: Branch products for model A
        Ms_B: Branch products for model B
        tau_s: Gating threshold for diagonal dominance
        eps: Numerical stability

    Returns:
        (lineage_score, alignment, score_matrix)
    """
    L = len(Ms_A)
    assert len(Ms_B) == L, "Models must have same number of layers"

    # Compute signatures and diagonal scores
    sigs_A = [_residual_signature(M, eps) for M in Ms_A]
    sigs_B = [_residual_signature(M, eps) for M in Ms_B]
    scores_A = [_diag_score(M, eps) for M in Ms_A]
    scores_B = [_diag_score(M, eps) for M in Ms_B]

    # Build gated similarity matrix
    G = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            cos_sim = np.dot(sigs_A[i], sigs_B[j])
            gate = min(scores_A[i] / tau_s, scores_B[j] / tau_s, 1.0)
            G[i, j] = cos_sim * gate

    # Hungarian alignment (maximize)
    row_ind, col_ind = linear_sum_assignment(-G)
    alignment = col_ind

    # Lineage score = mean of aligned similarities
    lineage = np.mean([G[i, alignment[i]] for i in range(L)])

    return float(lineage), alignment, G


def _residual_signature(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute centered residual signature."""
    d = M.shape[0]
    alpha = np.trace(M) / d
    R = M - alpha * np.eye(d)
    R_vec = R.flatten()
    norm = np.linalg.norm(R_vec)
    return R_vec / (norm + eps)


def _diag_score(M: np.ndarray, eps: float = 1e-12) -> float:
    """Compute diagonal dominance score."""
    return abs(np.trace(M)) / (np.linalg.norm(M, 'fro') + eps)


def choose_tau_s(reference_Ms_list: List[List[np.ndarray]]) -> float:
    """Choose tau_s as minimum diagonal score across all reference branches."""
    all_scores = []
    for Ms in reference_Ms_list:
        for M in Ms:
            all_scores.append(_diag_score(M))
    return float(np.min(all_scores))


def calibrate_z_score(score: float, null_scores: List[float]) -> float:
    """Calibrate score to z-score against null distribution."""
    null_arr = np.array(null_scores)
    mu = float(null_arr.mean())
    sigma = float(null_arr.std(ddof=0))
    return (score - mu) / (sigma + 1e-12)


def compute_auroc(
    pos_scores: List[float],
    neg_scores: List[float],
) -> float:
    """Compute AUROC from positive and negative scores."""
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)

    # Concatenate with labels
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Compute TPR and FPR
    n_pos = len(pos)
    n_neg = len(neg)

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # AUROC via trapezoidal rule
    auroc = float(np.trapz(np.concatenate([[0], tpr, [1]]),
                           np.concatenate([[0], fpr, [1]])))

    return auroc


def compute_tpr_at_fpr(
    pos_scores: List[float],
    neg_scores: List[float],
    target_fpr: float = 0.01,
) -> float:
    """Compute TPR at a given FPR threshold."""
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)

    # Find threshold for target FPR
    threshold = np.percentile(neg, 100 * (1 - target_fpr))

    # TPR = fraction of positives above threshold
    tpr = float((pos >= threshold).mean())
    return tpr


def bootstrap_auroc_ci(
    pos_scores: List[float],
    neg_scores: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute AUROC with bootstrap confidence interval."""
    rng = np.random.RandomState(seed)

    pos = np.array(pos_scores)
    neg = np.array(neg_scores)

    bootstrap_aurocs = []
    for _ in range(n_bootstrap):
        pos_sample = rng.choice(pos, size=len(pos), replace=True)
        neg_sample = rng.choice(neg, size=len(neg), replace=True)
        auroc = compute_auroc(pos_sample.tolist(), neg_sample.tolist())
        bootstrap_aurocs.append(auroc)

    bootstrap_aurocs = np.array(bootstrap_aurocs)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_aurocs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_aurocs, 100 * (1 - alpha / 2)))

    return {
        "auroc": float(np.mean(bootstrap_aurocs)),
        "auroc_std": float(np.std(bootstrap_aurocs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
    }


def evaluate_lineage_benchmark(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate full benchmark results.

    Args:
        records: List of dicts with 'label' (descendant/non_descendant),
                 'lineage' score, and optionally 'attack_type'.

    Returns:
        Dict with AUROC, TPR@FPR metrics, per-attack breakdown.
    """
    pos_scores = [r["lineage"] for r in records if r["label"] == "descendant"]
    neg_scores = [r["lineage"] for r in records if r["label"] == "non_descendant"]

    # Overall metrics
    auroc = compute_auroc(pos_scores, neg_scores)
    auroc_ci = bootstrap_auroc_ci(pos_scores, neg_scores)
    tpr_1pct = compute_tpr_at_fpr(pos_scores, neg_scores, 0.01)
    tpr_10pct = compute_tpr_at_fpr(pos_scores, neg_scores, 0.10)

    # Per-attack breakdown
    attack_types = set(r.get("attack_type", "unknown") for r in records)
    per_attack = {}
    for attack in attack_types:
        attack_records = [r for r in records if r.get("attack_type") == attack]
        scores = [r["lineage"] for r in attack_records]
        label = attack_records[0]["label"] if attack_records else "unknown"
        per_attack[attack] = {
            "label": label,
            "n": len(scores),
            "mean": float(np.mean(scores)) if scores else 0.0,
            "std": float(np.std(scores)) if scores else 0.0,
            "min": float(np.min(scores)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0,
        }

    return {
        "auroc": auroc,
        "auroc_ci": auroc_ci,
        "tpr_at_1pct_fpr": tpr_1pct,
        "tpr_at_10pct_fpr": tpr_10pct,
        "n_positives": len(pos_scores),
        "n_negatives": len(neg_scores),
        "pos_mean": float(np.mean(pos_scores)),
        "pos_std": float(np.std(pos_scores)),
        "neg_mean": float(np.mean(neg_scores)),
        "neg_std": float(np.std(neg_scores)),
        "per_attack": per_attack,
    }


def compute_gap_z(
    related_scores: List[float],
    distilled_scores: List[float],
) -> float:
    """Compute Gap-Z: standardized margin between related and distilled.

    Gap-Z = (mean_related - mean_distilled) / std_distilled
    """
    related = np.array(related_scores)
    distilled = np.array(distilled_scores)

    mean_related = related.mean()
    mean_distilled = distilled.mean()
    std_distilled = distilled.std(ddof=0)

    return float((mean_related - mean_distilled) / (std_distilled + 1e-12))
