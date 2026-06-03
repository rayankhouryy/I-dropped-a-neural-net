"""Parent Retrieval via Diagonal Profile Fingerprints POC

Tests whether the centered diagonal fingerprint can retrieve the exact parent
checkpoint of a modified model from a gallery of independently trained references.

Research Question: Given R independently trained reference models and a suspect
checkpoint B, can we identify which reference produced B?

Key insight: While diagonal dominance is generic across trained residual models,
the *centered* diagonal profile captures checkpoint-specific structure that
persists through post-training transformations.

Outputs:
  case_studies/model_lineage/parent_retrieval_results.csv
  case_studies/model_lineage/parent_retrieval_summary.json
  case_studies/model_lineage/fig_parent_retrieval_poc.png
"""
import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Section 1: Model Architecture (adapted from compression_audit.py)
# ============================================================================

class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))


class ResNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList([Block(in_dim, hidden_dim) for _ in range(depth)])
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.last(x)


def apply_init(model: nn.Module, scheme: str, seed: int):
    g = torch.Generator().manual_seed(seed)

    def _init_linear(lin: nn.Linear):
        W = lin.weight
        with torch.no_grad():
            if scheme == "kaiming_normal":
                fan_in = W.shape[1]
                std = math.sqrt(2.0 / fan_in)
                W.copy_(torch.randn(W.shape, generator=g) * std)
            elif scheme == "xavier_normal":
                fan_in, fan_out = W.shape[1], W.shape[0]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                W.copy_(torch.randn(W.shape, generator=g) * std)
            else:
                raise ValueError(f"unknown init scheme: {scheme}")
            lin.bias.zero_()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            _init_linear(m)


# ============================================================================
# Section 2: Data Generation
# ============================================================================

def synthetic_target(X, in_dim, key):
    g = torch.Generator().manual_seed(key)
    A = torch.randn(in_dim, 8, generator=g) * 0.5
    B = torch.randn(8, generator=g)
    bias = torch.randn(1, generator=g)
    h = torch.tanh(X @ A)
    return h @ B + bias


def make_data(in_dim, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    y = synthetic_target(X, in_dim, key=seed + 1234)
    return X, y


# ============================================================================
# Section 3: Training
# ============================================================================

def train_model(seed, depth=24, hidden=64, in_dim=24, epochs=200, lr=1e-3,
                batch=256, grad_clip=1.0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
    apply_init(model, "kaiming_normal", seed=seed * 1000 + 1)
    model = model.to(DEVICE)

    X, y = make_data(in_dim=in_dim, n=4000, seed=seed)
    X_train, y_train = X[1000:], y[1000:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_train = X_train.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            yb_pred = model(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        if verbose and (ep + 1) % 50 == 0:
            print(f"    Epoch {ep+1}/{epochs}, loss={loss.item():.4f}")

    return model


# ============================================================================
# Section 4: Centered Diagonal Fingerprint Functions
# ============================================================================

def extract_branch_products(model: nn.Module) -> List[np.ndarray]:
    products = []
    for block in model.blocks:
        W_in = block.inp.weight.detach().cpu().numpy().astype(np.float64)
        W_out = block.out.weight.detach().cpu().numpy().astype(np.float64)
        products.append(W_out @ W_in)
    return products


def centered_diag_fingerprint(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.diag(M)
    d_centered = d - d.mean()
    norm = np.linalg.norm(d_centered)
    return d_centered / (norm + eps)


def diag_dominance_score(M: np.ndarray, eps: float = 1e-12) -> float:
    return abs(float(np.trace(M))) / (float(np.linalg.norm(M, 'fro')) + eps)


def extract_fingerprints(model: nn.Module, eps: float = 1e-12) -> List[np.ndarray]:
    return [centered_diag_fingerprint(M, eps) for M in extract_branch_products(model)]


def extract_dd_scores(model: nn.Module, eps: float = 1e-12) -> List[float]:
    return [diag_dominance_score(M, eps) for M in extract_branch_products(model)]


# ============================================================================
# Section 5: Lineage Scoring Functions
# ============================================================================

def branch_similarity(psi_a: np.ndarray, psi_b: np.ndarray,
                      s_a: float = None, s_b: float = None,
                      tau_s: float = None) -> float:
    C = float(np.dot(psi_a, psi_b))
    if tau_s is not None and s_a is not None and s_b is not None:
        gate = min(s_a / tau_s, s_b / tau_s, 1.0)
        return C * gate
    return C


def compute_similarity_matrix(fingerprints_a: List[np.ndarray],
                              fingerprints_b: List[np.ndarray],
                              scores_a: List[float] = None,
                              scores_b: List[float] = None,
                              tau_s: float = None) -> np.ndarray:
    L = len(fingerprints_a)
    G = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            s_a = scores_a[i] if scores_a else None
            s_b = scores_b[j] if scores_b else None
            G[i, j] = branch_similarity(fingerprints_a[i], fingerprints_b[j],
                                        s_a, s_b, tau_s)
    return G


def lineage_score_diagonal(model_a: nn.Module, model_b: nn.Module,
                           tau_s: float = None, use_hungarian: bool = False) -> float:
    psi_a = extract_fingerprints(model_a)
    psi_b = extract_fingerprints(model_b)

    if tau_s is not None:
        s_a = extract_dd_scores(model_a)
        s_b = extract_dd_scores(model_b)
    else:
        s_a, s_b = None, None

    G = compute_similarity_matrix(psi_a, psi_b, s_a, s_b, tau_s)

    if use_hungarian:
        row_ind, col_ind = linear_sum_assignment(-G)
        return float(G[row_ind, col_ind].mean())
    else:
        return float(np.diag(G).mean())


# ============================================================================
# Section 6: Baseline Methods
# ============================================================================

def frobenius_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    products_a = extract_branch_products(model_a)
    products_b = extract_branch_products(model_b)
    L = len(products_a)
    total = 0.0
    for i in range(L):
        diff = products_a[i] - products_b[i]
        total -= np.linalg.norm(diff, 'fro')
    return total / L


def raw_cosine_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    products_a = extract_branch_products(model_a)
    products_b = extract_branch_products(model_b)
    L = len(products_a)
    total = 0.0
    for i in range(L):
        vec_a = products_a[i].flatten()
        vec_b = products_b[i].flatten()
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 1e-12 and norm_b > 1e-12:
            total += np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return total / L


def diagonal_dominance_only(model_b: nn.Module) -> float:
    scores = extract_dd_scores(model_b)
    return float(np.mean(scores))


def raw_diagonal_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    products_a = extract_branch_products(model_a)
    products_b = extract_branch_products(model_b)
    L = len(products_a)
    total = 0.0
    eps = 1e-12
    for i in range(L):
        d_a = np.diag(products_a[i])
        d_b = np.diag(products_b[i])
        norm_a = np.linalg.norm(d_a)
        norm_b = np.linalg.norm(d_b)
        if norm_a > eps and norm_b > eps:
            total += np.dot(d_a, d_b) / (norm_a * norm_b)
    return total / L


# ============================================================================
# Section 7: Model Transformations
# ============================================================================

def magnitude_prune(model: nn.Module, sparsity: float) -> nn.Module:
    pruned = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in pruned.named_parameters():
            if 'weight' in name:
                flat = param.abs().flatten()
                k = int(sparsity * flat.numel())
                if k > 0:
                    threshold = torch.kthvalue(flat, k).values
                    mask = param.abs() >= threshold
                    param.mul_(mask.float())
    return pruned


def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for param in quantized.parameters():
            if param.numel() > 1:
                max_val = param.abs().max()
                if max_val > 0:
                    if bits == 16:
                        param.copy_(param.half().float())
                    else:
                        levels = 2 ** (bits - 1) - 1
                        scale = max_val / levels
                        quantized_vals = torch.round(param / scale) * scale
                        param.copy_(quantized_vals)
    return quantized


def fine_tune_model(model: nn.Module, epochs=20, lr=1e-4, seed=999) -> nn.Module:
    torch.manual_seed(seed)
    fine_tuned = copy.deepcopy(model).to(DEVICE)
    in_dim = model.blocks[0].inp.in_features
    X, y = make_data(in_dim=in_dim, n=2000, seed=seed + 5000)
    opt = torch.optim.Adam(fine_tuned.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        fine_tuned.train()
        perm = torch.randperm(X.shape[0])
        for s in range(0, X.shape[0], 256):
            idx = perm[s:s + 256]
            xb = X[idx].to(DEVICE)
            yb = y[idx].to(DEVICE)
            yb_pred = fine_tuned(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return fine_tuned


def distill_model(teacher: nn.Module, student_seed: int, epochs=100, lr=1e-3,
                  verbose=False) -> nn.Module:
    torch.manual_seed(student_seed)
    depth = len(teacher.blocks)
    in_dim = teacher.blocks[0].inp.in_features
    hidden = teacher.blocks[0].inp.out_features

    student = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
    apply_init(student, "kaiming_normal", seed=student_seed * 1000 + 1)
    student = student.to(DEVICE)
    teacher = teacher.to(DEVICE)
    teacher.eval()

    X, _ = make_data(in_dim=in_dim, n=4000, seed=student_seed + 9999)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        student.train()
        perm = torch.randperm(X.shape[0])
        for s in range(0, X.shape[0], 256):
            idx = perm[s:s + 256]
            xb = X[idx].to(DEVICE)
            with torch.no_grad():
                teacher_out = teacher(xb)
            student_out = student(xb)
            loss = loss_fn(student_out, teacher_out)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and (ep + 1) % 50 == 0:
            print(f"    Distill epoch {ep+1}/{epochs}, loss={loss.item():.6f}")

    return student


def add_weight_noise(model: nn.Module, sigma: float, seed: int = None) -> nn.Module:
    noisy = copy.deepcopy(model)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        for p in noisy.parameters():
            if p.numel() > 1:
                p.add_(torch.randn_like(p) * sigma * p.std().item())
    return noisy


def apply_lora_and_merge(model: nn.Module, rank: int = 4, alpha: float = 1.0,
                          fine_tune_epochs: int = 10, seed: int = None) -> nn.Module:
    if seed is not None:
        torch.manual_seed(seed)

    merged = copy.deepcopy(model).to(DEVICE)
    in_dim = model.blocks[0].inp.in_features

    lora_params = []
    for block in merged.blocks:
        for lin in [block.inp, block.out]:
            out_f, in_f = lin.weight.shape
            A = torch.randn(rank, in_f, device=DEVICE) * 0.01
            B = torch.zeros(out_f, rank, device=DEVICE)
            A.requires_grad = True
            B.requires_grad = True
            lora_params.append((lin, A, B))

    X, y = make_data(in_dim=in_dim, n=2000, seed=(seed or 0) + 7777)
    opt = torch.optim.Adam([p for _, A, B in lora_params for p in [A, B]], lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(fine_tune_epochs):
        merged.train()
        perm = torch.randperm(X.shape[0])
        for s in range(0, X.shape[0], 256):
            idx = perm[s:s + 256]
            xb = X[idx].to(DEVICE)
            yb = y[idx].to(DEVICE)

            for lin, A, B in lora_params:
                lin.weight.data += (alpha / rank) * (B @ A)

            yb_pred = merged(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)

            for lin, A, B in lora_params:
                lin.weight.data -= (alpha / rank) * (B @ A)

            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        for lin, A, B in lora_params:
            lin.weight.add_((alpha / rank) * (B @ A))

    return merged


# ============================================================================
# Section 8: Evaluation
# ============================================================================

@dataclass
class RetrievalResult:
    suspect_id: str
    suspect_type: str
    true_parent_idx: int
    predicted_parent_idx: int
    all_scores: List[float]
    rank_of_true_parent: int
    is_top1_correct: bool
    reciprocal_rank: float
    margin: float
    method: str


def retrieve_parent(references: List[nn.Module], suspect: nn.Module,
                    scoring_fn: Callable) -> Tuple[int, List[float]]:
    scores = [scoring_fn(ref, suspect) for ref in references]
    return int(np.argmax(scores)), scores


def compute_margin(scores: List[float]) -> float:
    sorted_scores = sorted(scores, reverse=True)
    if len(sorted_scores) >= 2:
        return sorted_scores[0] - sorted_scores[1]
    return 0.0


def compute_rank(scores: List[float], true_idx: int) -> int:
    if true_idx < 0:
        return -1
    true_score = scores[true_idx]
    rank = 1
    for i, s in enumerate(scores):
        if i != true_idx and s > true_score:
            rank += 1
    return rank


def evaluate_retrieval(references: List[nn.Module],
                       suspects: List[Tuple[nn.Module, str, str, int]],
                       scoring_fn: Callable,
                       method_name: str) -> List[RetrievalResult]:
    results = []
    for suspect_model, suspect_id, suspect_type, true_parent_idx in suspects:
        pred_idx, scores = retrieve_parent(references, suspect_model, scoring_fn)

        if true_parent_idx >= 0:
            rank = compute_rank(scores, true_parent_idx)
            is_correct = (pred_idx == true_parent_idx)
            rr = 1.0 / rank
        else:
            rank = -1
            is_correct = False
            rr = 0.0

        margin = compute_margin(scores)

        results.append(RetrievalResult(
            suspect_id=suspect_id,
            suspect_type=suspect_type,
            true_parent_idx=true_parent_idx,
            predicted_parent_idx=pred_idx,
            all_scores=scores,
            rank_of_true_parent=rank,
            is_top1_correct=is_correct,
            reciprocal_rank=rr,
            margin=margin,
            method=method_name
        ))

    return results


def compute_auroc(pos_scores: List[float], neg_scores: List[float]) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    pos = np.array(pos_scores)[:, None]
    neg = np.array(neg_scores)[None, :]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    total = len(pos_scores) * len(neg_scores)
    return float(wins / total)


def aggregate_metrics(results: List[RetrievalResult]) -> Dict:
    descendants = [r for r in results if r.true_parent_idx >= 0]
    non_descendants = [r for r in results if r.true_parent_idx < 0]

    if descendants:
        top1 = np.mean([r.is_top1_correct for r in descendants])
        mrr = np.mean([r.reciprocal_rank for r in descendants])
    else:
        top1, mrr = 0.0, 0.0

    if descendants and non_descendants:
        desc_max_scores = [max(r.all_scores) for r in descendants]
        non_desc_max_scores = [max(r.all_scores) for r in non_descendants]
        auroc = compute_auroc(desc_max_scores, non_desc_max_scores)
    else:
        auroc = 0.5

    return {
        "top1": float(top1),
        "mrr": float(mrr),
        "auroc": float(auroc),
        "n_descendants": len(descendants),
        "n_non_descendants": len(non_descendants),
    }


# ============================================================================
# Section 9: Visualization
# ============================================================================

def create_figure(all_results: Dict[str, List[RetrievalResult]],
                  references: List[nn.Module],
                  output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    diag_results = all_results.get("diag_centered", [])

    # Panel A: Parent-retrieval heatmap
    ax = axes[0]
    descendants = [r for r in diag_results if r.true_parent_idx >= 0]
    if descendants:
        n_suspects = len(descendants)
        n_refs = len(references)
        heatmap = np.zeros((n_suspects, n_refs))
        for i, r in enumerate(descendants):
            heatmap[i, :] = r.all_scores

        im = ax.imshow(heatmap, aspect='auto', cmap='viridis')
        ax.set_xlabel('Reference Index')
        ax.set_ylabel('Suspect Index')
        ax.set_title('Panel A: Lineage Scores\n(descendants only)')
        plt.colorbar(im, ax=ax, label='L_diag(A, B)')

        for i, r in enumerate(descendants):
            if r.true_parent_idx >= 0:
                ax.plot(r.true_parent_idx, i, 'rx', markersize=8, markeredgewidth=2)

    # Panel B: Score distributions
    ax = axes[1]
    true_parent_scores = []
    wrong_parent_scores = []
    independent_scores = []
    distilled_scores = []

    for r in diag_results:
        if r.suspect_type == "distilled":
            distilled_scores.append(max(r.all_scores))
        elif r.suspect_type == "independent":
            independent_scores.append(max(r.all_scores))
        elif r.true_parent_idx >= 0:
            true_parent_scores.append(r.all_scores[r.true_parent_idx])
            for i, s in enumerate(r.all_scores):
                if i != r.true_parent_idx:
                    wrong_parent_scores.append(s)

    bins = np.linspace(-0.5, 1.0, 30)
    if true_parent_scores:
        ax.hist(true_parent_scores, bins=bins, alpha=0.6, label='True parent', color='green')
    if wrong_parent_scores:
        ax.hist(wrong_parent_scores, bins=bins, alpha=0.6, label='Wrong parent', color='orange')
    if independent_scores:
        ax.hist(independent_scores, bins=bins, alpha=0.6, label='Independent', color='red')
    if distilled_scores:
        ax.hist(distilled_scores, bins=bins, alpha=0.6, label='Distilled', color='blue')

    ax.set_xlabel('Lineage Score')
    ax.set_ylabel('Count')
    ax.set_title('Panel B: Score Distributions')
    ax.legend()

    # Panel C: Comparison across methods
    ax = axes[2]
    methods = list(all_results.keys())
    metrics_by_method = {}
    for method in methods:
        results = all_results[method]
        desc = [r for r in results if r.true_parent_idx >= 0]
        if desc:
            metrics_by_method[method] = {
                "top1": np.mean([r.is_top1_correct for r in desc]),
                "mrr": np.mean([r.reciprocal_rank for r in desc]),
            }

    if metrics_by_method:
        x = np.arange(len(metrics_by_method))
        width = 0.35
        method_names = list(metrics_by_method.keys())
        top1_vals = [metrics_by_method[m]["top1"] for m in method_names]
        mrr_vals = [metrics_by_method[m]["mrr"] for m in method_names]

        ax.bar(x - width/2, top1_vals, width, label='Top-1 Acc')
        ax.bar(x + width/2, mrr_vals, width, label='MRR')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Panel C: Method Comparison')
        ax.legend()
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


# ============================================================================
# Section 10: Main Experiment Runner
# ============================================================================

def run_experiment(output_dir: Path, config: Dict, verbose: bool = True):
    print("=" * 70)
    print("Parent Retrieval via Diagonal Profile Fingerprints - POC")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print()

    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_refs = config["n_references"]
    depth = config["depth"]
    hidden = config["hidden"]
    in_dim = config["in_dim"]
    epochs = config["epochs"]

    # Phase 1: Train reference models
    print(f"Phase 1: Training {n_refs} reference models...")
    references = []
    for i in range(n_refs):
        print(f"  Reference {i+1}/{n_refs} (seed={i})...")
        model = train_model(seed=i, depth=depth, hidden=hidden, in_dim=in_dim,
                           epochs=epochs, verbose=False)
        references.append(model)
        if verbose:
            dd_scores = extract_dd_scores(model)
            print(f"    Mean DD score: {np.mean(dd_scores):.3f}")

    # Phase 2: Generate descendants
    print(f"\nPhase 2: Generating descendants...")
    suspects = []

    for ref_idx, ref_model in enumerate(references):
        print(f"  Reference {ref_idx}: generating 5 descendants...")

        # Fine-tuned
        ft = fine_tune_model(ref_model, epochs=20, lr=1e-4, seed=ref_idx + 100)
        suspects.append((ft, f"ref{ref_idx}_ft", "fine_tuned", ref_idx))

        # Quantized
        qt = quantize_model(ref_model, bits=8)
        suspects.append((qt, f"ref{ref_idx}_q8", "quantized", ref_idx))

        # Pruned
        pr = magnitude_prune(ref_model, sparsity=0.3)
        suspects.append((pr, f"ref{ref_idx}_pr30", "pruned", ref_idx))

        # Noisy
        ns = add_weight_noise(ref_model, sigma=0.01, seed=ref_idx + 200)
        suspects.append((ns, f"ref{ref_idx}_noise", "noisy", ref_idx))

        # LoRA merged
        lora = apply_lora_and_merge(ref_model, rank=4, fine_tune_epochs=10, seed=ref_idx + 300)
        suspects.append((lora, f"ref{ref_idx}_lora", "lora_merged", ref_idx))

    # Phase 3: Generate controls
    print(f"\nPhase 3: Generating controls...")

    # Independent models
    n_indep = config["n_independents"]
    print(f"  Training {n_indep} independent models...")
    for i in range(n_indep):
        seed = 1000 + i
        model = train_model(seed=seed, depth=depth, hidden=hidden, in_dim=in_dim,
                           epochs=epochs, verbose=False)
        suspects.append((model, f"indep_{i}", "independent", -1))

    # Random-init models
    n_random = config["n_random_init"]
    print(f"  Creating {n_random} random-init models...")
    for i in range(n_random):
        model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
        apply_init(model, "kaiming_normal", seed=2000 + i)
        suspects.append((model, f"random_{i}", "random_init", -1))

    # Distilled models
    n_distilled = config["n_distilled"]
    print(f"  Training {n_distilled} distilled students...")
    for i in range(n_distilled):
        teacher_idx = i % n_refs
        student = distill_model(references[teacher_idx], student_seed=3000 + i,
                               epochs=100, verbose=False)
        suspects.append((student, f"distill_{i}_from_{teacher_idx}", "distilled", -1))

    # Phase 4: Compute scores for all methods
    print(f"\nPhase 4: Computing scores ({len(suspects)} suspects x {n_refs} refs)...")

    scoring_methods = {
        "diag_centered": lambda a, b: lineage_score_diagonal(a, b, tau_s=config.get("tau_s")),
        "frobenius": frobenius_similarity,
        "raw_cosine": raw_cosine_similarity,
        "raw_diagonal": raw_diagonal_similarity,
    }

    all_results = {}
    for method_name, scoring_fn in scoring_methods.items():
        print(f"  Method: {method_name}...")
        results = evaluate_retrieval(references, suspects, scoring_fn, method_name)
        all_results[method_name] = results

    # Phase 5: Aggregate and report
    print(f"\nPhase 5: Aggregating results...")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "elapsed_seconds": time.time() - t0,
        "n_references": n_refs,
        "n_suspects": len(suspects),
        "results_by_method": {},
    }

    for method_name, results in all_results.items():
        overall = aggregate_metrics(results)

        by_type = {}
        for suspect_type in ["fine_tuned", "quantized", "pruned", "noisy", "lora_merged",
                            "independent", "random_init", "distilled"]:
            type_results = [r for r in results if r.suspect_type == suspect_type]
            if type_results:
                by_type[suspect_type] = aggregate_metrics(type_results)

        summary["results_by_method"][method_name] = {
            "overall": overall,
            "by_suspect_type": by_type,
        }

    # POC success criteria
    diag_metrics = summary["results_by_method"]["diag_centered"]
    poc_targets = {
        "top1_fine_tuned_>90%": diag_metrics["by_suspect_type"].get("fine_tuned", {}).get("top1", 0) > 0.9,
        "mrr_overall_>0.9": diag_metrics["overall"]["mrr"] > 0.9,
        "auroc_>0.95": diag_metrics["overall"]["auroc"] > 0.95,
    }
    summary["poc_targets"] = poc_targets
    summary["poc_success"] = all(poc_targets.values())

    # Save outputs
    json_path = output_dir / "parent_retrieval_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    csv_path = output_dir / "parent_retrieval_results.csv"
    rows = []
    for method_name, results in all_results.items():
        for r in results:
            rows.append({
                "suspect_id": r.suspect_id,
                "suspect_type": r.suspect_type,
                "true_parent_idx": r.true_parent_idx,
                "predicted_parent_idx": r.predicted_parent_idx,
                "rank_of_true": r.rank_of_true_parent,
                "is_correct": r.is_top1_correct,
                "reciprocal_rank": r.reciprocal_rank,
                "margin": r.margin,
                "max_score": max(r.all_scores),
                "true_parent_score": r.all_scores[r.true_parent_idx] if r.true_parent_idx >= 0 else None,
                "method": r.method,
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")

    fig_path = output_dir / "fig_parent_retrieval_poc.png"
    create_figure(all_results, references, fig_path)
    print(f"  Saved: {fig_path}")

    # Print summary
    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'Top-1':>8} {'MRR':>8} {'AUROC':>8}")
    print("-" * 50)
    for method_name in all_results:
        m = summary["results_by_method"][method_name]["overall"]
        print(f"{method_name:<20} {m['top1']:>8.1%} {m['mrr']:>8.3f} {m['auroc']:>8.3f}")
    print("-" * 50)
    print()
    print("By suspect type (diag_centered):")
    for stype, metrics in summary["results_by_method"]["diag_centered"]["by_suspect_type"].items():
        print(f"  {stype:<15} Top-1: {metrics['top1']:.1%}, MRR: {metrics['mrr']:.3f}")
    print()
    print("POC Success Criteria:")
    for target, achieved in poc_targets.items():
        status = "PASS" if achieved else "FAIL"
        print(f"  [{status}] {target}")
    print()
    print(f"Overall POC: {'SUCCESS' if summary['poc_success'] else 'NEEDS WORK'}")
    print(f"Total time: {elapsed:.1f}s")

    return summary, all_results


# ============================================================================
# Section 11: Robustness Sweep
# ============================================================================

def run_robustness_sweep(output_dir: Path, config: Dict, verbose: bool = True):
    """
    Test increasingly aggressive transformations to find where methods fail.

    Sweeps over:
    - Fine-tuning: 10, 50, 100, 200, 500 epochs
    - Pruning: 10%, 30%, 50%, 70%, 90%, 95%
    - Weight noise: 1%, 5%, 10%, 20%, 50%
    - Quantization: 16, 8, 4, 2 bits
    """
    print("=" * 70)
    print("ROBUSTNESS SWEEP - Finding Method Limits")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_refs = config.get("n_references", 5)
    depth = config.get("depth", 16)
    hidden = config.get("hidden", 48)
    in_dim = config.get("in_dim", 20)
    epochs = config.get("epochs", 150)

    # Train reference models
    print(f"Training {n_refs} reference models...")
    references = []
    for i in range(n_refs):
        model = train_model(seed=i, depth=depth, hidden=hidden, in_dim=in_dim,
                           epochs=epochs, verbose=False)
        references.append(model)
        if verbose:
            print(f"  Reference {i+1}/{n_refs} done")

    # Also train some independent models for negative controls
    print(f"Training {n_refs} independent models (negative controls)...")
    independents = []
    for i in range(n_refs):
        model = train_model(seed=100 + i, depth=depth, hidden=hidden, in_dim=in_dim,
                           epochs=epochs, verbose=False)
        independents.append(model)

    # Define scoring methods
    scoring_methods = {
        "diag_centered": lambda a, b: lineage_score_diagonal(a, b, tau_s=0.3),
        "frobenius": frobenius_similarity,
        "raw_cosine": raw_cosine_similarity,
        "raw_diagonal": raw_diagonal_similarity,
    }

    # Define transformation sweeps - include extreme values to find failure points
    sweeps = {
        "fine_tune_epochs": [10, 50, 100, 200, 500, 1000],
        "prune_sparsity": [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995],
        "noise_sigma": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
        "quantize_bits": [16, 8, 4, 2, 1],
    }

    all_sweep_results = []

    # Fine-tuning sweep
    print("\n--- Fine-tuning sweep ---")
    for ft_epochs in sweeps["fine_tune_epochs"]:
        print(f"  Fine-tune epochs={ft_epochs}...")
        suspects = []
        for ref_idx, ref_model in enumerate(references):
            modified = fine_tune_model(ref_model, epochs=ft_epochs, lr=1e-4, seed=ref_idx + 1000)
            suspects.append((modified, f"ft_{ft_epochs}_ref{ref_idx}", "fine_tuned", ref_idx))

        # Add independents as negatives
        for i, indep in enumerate(independents):
            suspects.append((indep, f"indep_{i}", "independent", -1))

        for method_name, scoring_fn in scoring_methods.items():
            results = evaluate_retrieval(references, suspects, scoring_fn, method_name)
            metrics = aggregate_metrics(results)

            # Also compute mean score for true parents
            desc_results = [r for r in results if r.true_parent_idx >= 0]
            mean_true_score = np.mean([r.all_scores[r.true_parent_idx] for r in desc_results])

            all_sweep_results.append({
                "transform": "fine_tune",
                "param": ft_epochs,
                "method": method_name,
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "auroc": metrics["auroc"],
                "mean_true_score": mean_true_score,
            })

    # Pruning sweep
    print("\n--- Pruning sweep ---")
    for sparsity in sweeps["prune_sparsity"]:
        print(f"  Prune sparsity={sparsity:.0%}...")
        suspects = []
        for ref_idx, ref_model in enumerate(references):
            modified = magnitude_prune(ref_model, sparsity=sparsity)
            suspects.append((modified, f"prune_{sparsity}_ref{ref_idx}", "pruned", ref_idx))

        for i, indep in enumerate(independents):
            suspects.append((indep, f"indep_{i}", "independent", -1))

        for method_name, scoring_fn in scoring_methods.items():
            results = evaluate_retrieval(references, suspects, scoring_fn, method_name)
            metrics = aggregate_metrics(results)
            desc_results = [r for r in results if r.true_parent_idx >= 0]
            mean_true_score = np.mean([r.all_scores[r.true_parent_idx] for r in desc_results])

            all_sweep_results.append({
                "transform": "prune",
                "param": sparsity,
                "method": method_name,
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "auroc": metrics["auroc"],
                "mean_true_score": mean_true_score,
            })

    # Noise sweep
    print("\n--- Weight noise sweep ---")
    for sigma in sweeps["noise_sigma"]:
        print(f"  Noise sigma={sigma:.0%}...")
        suspects = []
        for ref_idx, ref_model in enumerate(references):
            modified = add_weight_noise(ref_model, sigma=sigma, seed=ref_idx + 2000)
            suspects.append((modified, f"noise_{sigma}_ref{ref_idx}", "noisy", ref_idx))

        for i, indep in enumerate(independents):
            suspects.append((indep, f"indep_{i}", "independent", -1))

        for method_name, scoring_fn in scoring_methods.items():
            results = evaluate_retrieval(references, suspects, scoring_fn, method_name)
            metrics = aggregate_metrics(results)
            desc_results = [r for r in results if r.true_parent_idx >= 0]
            mean_true_score = np.mean([r.all_scores[r.true_parent_idx] for r in desc_results])

            all_sweep_results.append({
                "transform": "noise",
                "param": sigma,
                "method": method_name,
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "auroc": metrics["auroc"],
                "mean_true_score": mean_true_score,
            })

    # Quantization sweep
    print("\n--- Quantization sweep ---")
    for bits in sweeps["quantize_bits"]:
        print(f"  Quantize bits={bits}...")
        suspects = []
        for ref_idx, ref_model in enumerate(references):
            modified = quantize_model(ref_model, bits=bits)
            suspects.append((modified, f"quant_{bits}_ref{ref_idx}", "quantized", ref_idx))

        for i, indep in enumerate(independents):
            suspects.append((indep, f"indep_{i}", "independent", -1))

        for method_name, scoring_fn in scoring_methods.items():
            results = evaluate_retrieval(references, suspects, scoring_fn, method_name)
            metrics = aggregate_metrics(results)
            desc_results = [r for r in results if r.true_parent_idx >= 0]
            mean_true_score = np.mean([r.all_scores[r.true_parent_idx] for r in desc_results])

            all_sweep_results.append({
                "transform": "quantize",
                "param": bits,
                "method": method_name,
                "top1": metrics["top1"],
                "mrr": metrics["mrr"],
                "auroc": metrics["auroc"],
                "mean_true_score": mean_true_score,
            })

    # Save results
    csv_path = output_dir / "robustness_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(all_sweep_results)
    print(f"\nSaved: {csv_path}")

    # Create robustness figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    transforms = ["fine_tune", "prune", "noise", "quantize"]
    titles = ["Fine-tuning (epochs)", "Pruning (sparsity)", "Weight Noise (σ)", "Quantization (bits)"]
    xlabels = ["Epochs", "Sparsity", "Noise σ", "Bits"]

    for idx, (transform, title, xlabel) in enumerate(zip(transforms, titles, xlabels)):
        ax = axes[idx // 2, idx % 2]

        for method in scoring_methods.keys():
            data = [r for r in all_sweep_results if r["transform"] == transform and r["method"] == method]
            if data:
                params = [r["param"] for r in data]
                top1s = [r["top1"] for r in data]
                ax.plot(params, top1s, 'o-', label=method, linewidth=2, markersize=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 Retrieval Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1/n_refs, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.grid(True, alpha=0.3)

        if transform == "quantize":
            ax.invert_xaxis()

    plt.tight_layout()
    fig_path = output_dir / "fig_robustness_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Print summary table
    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("ROBUSTNESS SWEEP SUMMARY")
    print("=" * 70)
    print()

    for transform in transforms:
        print(f"\n{transform.upper()}:")
        print(f"{'Param':<10} {'diag_centered':>15} {'frobenius':>15} {'raw_cosine':>15} {'raw_diagonal':>15}")
        print("-" * 75)

        params = sorted(set(r["param"] for r in all_sweep_results if r["transform"] == transform))
        for param in params:
            row = f"{param:<10}"
            for method in ["diag_centered", "frobenius", "raw_cosine", "raw_diagonal"]:
                data = [r for r in all_sweep_results
                       if r["transform"] == transform and r["param"] == param and r["method"] == method]
                if data:
                    row += f" {data[0]['top1']:>14.1%}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    print()
    print(f"Total time: {elapsed:.1f}s")

    return all_sweep_results


# ============================================================================
# Section 12: Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parent Retrieval POC")
    parser.add_argument("--out-dir", default="case_studies/model_lineage",
                        help="Output directory")
    parser.add_argument("--n-refs", type=int, default=10, help="Number of reference models")
    parser.add_argument("--n-indep", type=int, default=20, help="Number of independent controls")
    parser.add_argument("--n-random", type=int, default=10, help="Number of random-init controls")
    parser.add_argument("--n-distilled", type=int, default=5, help="Number of distilled controls")
    parser.add_argument("--depth", type=int, default=24, help="Model depth")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--in-dim", type=int, default=24, help="Input dimension")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--tau-s", type=float, default=0.3, help="DD gating threshold")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced scale")
    parser.add_argument("--sweep", action="store_true",
                        help="Run robustness sweep instead of standard experiment")
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.out_dir

    if args.sweep:
        if args.quick:
            config = {
                "n_references": 3,
                "depth": 8,
                "hidden": 32,
                "in_dim": 16,
                "epochs": 50,
            }
        else:
            config = {
                "n_references": args.n_refs,
                "depth": args.depth,
                "hidden": args.hidden,
                "in_dim": args.in_dim,
                "epochs": args.epochs,
            }
        run_robustness_sweep(output_dir, config)
    else:
        if args.quick:
            config = {
                "n_references": 3,
                "n_independents": 3,
                "n_random_init": 2,
                "n_distilled": 2,
                "depth": 8,
                "hidden": 32,
                "in_dim": 16,
                "epochs": 50,
                "tau_s": 0.3,
            }
        else:
            config = {
                "n_references": args.n_refs,
                "n_independents": args.n_indep,
                "n_random_init": args.n_random,
                "n_distilled": args.n_distilled,
                "depth": args.depth,
                "hidden": args.hidden,
                "in_dim": args.in_dim,
                "epochs": args.epochs,
                "tau_s": args.tau_s,
            }
        run_experiment(output_dir, config)


if __name__ == "__main__":
    main()
