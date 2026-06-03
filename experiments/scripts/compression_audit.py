"""Case Study 3: Model Compression Auditing

Verify whether compressed models (pruned/quantized/distilled) derive from an
original using the diagonal dominance fingerprint.

Key insight: The E matrix (M + εI where M = W_out @ W_in) captures training-specific
structure. We test how this survives different compression methods:
  - Pruning: Should preserve E (removes weights, but structure remains)
  - Quantization: Should preserve E (precision reduction, structure intact)
  - Distillation: Should NOT preserve E (fresh training, new E matrices)

Audit Protocol:
  1. Train original model and extract E matrices
  2. Apply compression method
  3. Extract E matrices from compressed model
  4. Compute E correlation to determine derivation

Outputs:
  case_studies/case_study_3/compression_audit_results.csv
  case_studies/case_study_3/compression_audit_summary.json
  case_studies/case_study_3/figures/
"""
import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------- models
class Block(nn.Module):
    """Residual block: output = x + f(x)"""
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


# -------------------------------------------------------------------- init
def apply_init(model: nn.Module, scheme: str, seed: int):
    """Reinitialize all nn.Linear layers according to scheme."""
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


# -------------------------------------------------------------------- data
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


# -------------------------------------------------------------------- training
def train_model(seed, depth=24, hidden=64, in_dim=24, epochs=200, lr=1e-3,
                batch=256, grad_clip=1.0, verbose=True):
    """Train a ResNet and return the model."""
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
            print(f"  Epoch {ep+1}/{epochs}, loss={loss.item():.4f}")

    return model


def distill_model(teacher, student_seed, epochs=100, lr=1e-3, verbose=True):
    """Train a student model to mimic teacher outputs (knowledge distillation)."""
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
            print(f"  Distill epoch {ep+1}/{epochs}, loss={loss.item():.6f}")

    return student


# -------------------------------------------------------------------- compression methods

def magnitude_prune(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Apply magnitude pruning to all weights.
    Sets smallest `sparsity` fraction of weights to zero.
    """
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


def structured_prune_channels(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Apply structured pruning by zeroing out channels.
    For each block, prune the smallest-norm channels.
    """
    pruned = copy.deepcopy(model)

    with torch.no_grad():
        for block in pruned.blocks:
            W_in = block.inp.weight  # [hidden, in_dim]
            W_out = block.out.weight  # [in_dim, hidden]

            channel_norms = W_in.norm(dim=1)
            k = int(sparsity * channel_norms.numel())
            if k > 0:
                threshold = torch.kthvalue(channel_norms, k).values
                mask = channel_norms >= threshold
                W_in.mul_(mask.unsqueeze(1).float())
                W_out.mul_(mask.unsqueeze(0).float())

    return pruned


def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    """
    Simulate quantization by rounding weights to `bits` precision.
    Uses symmetric quantization around zero.
    """
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
    """Fine-tune an existing model on slightly different data."""
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


# -------------------------------------------------------------------- E matrix extraction

def extract_E_matrix(W_in: np.ndarray, W_out: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Extract the E matrix from W_out @ W_in.

    M = W_out @ W_in = -ε*I + E
    where ε = |tr(M)| / d and tr(E) ≈ 0

    Returns: (epsilon, E, dd_score)
    """
    M = W_out.astype(np.float64) @ W_in.astype(np.float64)
    d = M.shape[0]
    trace = np.trace(M)
    epsilon = abs(trace) / d

    if trace < 0:
        E = M + epsilon * np.eye(d)
    else:
        E = M - epsilon * np.eye(d)

    frob_norm = np.linalg.norm(M, 'fro') + 1e-12
    dd_score = abs(trace) / frob_norm

    return epsilon, E, dd_score


def extract_all_E_matrices(model: nn.Module) -> List[np.ndarray]:
    """Extract E matrices for all blocks."""
    E_list = []
    for block in model.blocks:
        W_in = block.inp.weight.detach().cpu().numpy()
        W_out = block.out.weight.detach().cpu().numpy()
        _, E, _ = extract_E_matrix(W_in, W_out)
        E_list.append(E)
    return E_list


def compute_dd_scores(model: nn.Module) -> List[float]:
    """Compute DD scores for all blocks."""
    scores = []
    for block in model.blocks:
        W_in = block.inp.weight.detach().cpu().numpy()
        W_out = block.out.weight.detach().cpu().numpy()
        _, _, dd_score = extract_E_matrix(W_in, W_out)
        scores.append(dd_score)
    return scores


def compute_trace_signs(model: nn.Module) -> List[bool]:
    """Check if trace is negative for all blocks."""
    signs = []
    for block in model.blocks:
        W_in = block.inp.weight.detach().cpu().numpy()
        W_out = block.out.weight.detach().cpu().numpy()
        M = W_out.astype(np.float64) @ W_in.astype(np.float64)
        signs.append(np.trace(M) < 0)
    return signs


# -------------------------------------------------------------------- audit protocol

@dataclass
class AuditReport:
    """Result of compression audit."""
    compression_method: str
    compression_params: Dict
    verdict: str  # "DERIVED", "LIKELY_DERIVED", "INCONCLUSIVE", "NOT_DERIVED"
    confidence: float
    mean_E_correlation: float
    per_block_correlations: List[float]
    mean_dd_score_original: float
    mean_dd_score_compressed: float
    pct_negative_trace_original: float
    pct_negative_trace_compressed: float
    notes: str = ""


def compute_E_correlation(E_orig: np.ndarray, E_comp: np.ndarray) -> float:
    """Compute correlation between two E matrices."""
    flat_orig = E_orig.flatten()
    flat_comp = E_comp.flatten()

    if np.std(flat_orig) < 1e-12 or np.std(flat_comp) < 1e-12:
        return 0.0

    corr, _ = pearsonr(flat_orig, flat_comp)
    return float(corr) if not np.isnan(corr) else 0.0


def audit_compressed_model(original: nn.Module,
                           compressed: nn.Module,
                           compression_method: str,
                           compression_params: Dict) -> AuditReport:
    """
    Audit whether a compressed model derives from the original.

    Decision thresholds:
      > 0.90: DEFINITELY DERIVED
      0.70 - 0.90: LIKELY DERIVED
      0.30 - 0.70: INCONCLUSIVE
      < 0.30: NOT DERIVED
    """
    E_orig_list = extract_all_E_matrices(original)
    E_comp_list = extract_all_E_matrices(compressed)

    correlations = []
    for E_orig, E_comp in zip(E_orig_list, E_comp_list):
        corr = compute_E_correlation(E_orig, E_comp)
        correlations.append(corr)

    mean_corr = np.mean(correlations)

    if mean_corr > 0.90:
        verdict = "DERIVED"
        confidence = mean_corr
    elif mean_corr > 0.70:
        verdict = "LIKELY_DERIVED"
        confidence = mean_corr
    elif mean_corr > 0.30:
        verdict = "INCONCLUSIVE"
        confidence = 0.5
    else:
        verdict = "NOT_DERIVED"
        confidence = 1 - mean_corr

    dd_orig = compute_dd_scores(original)
    dd_comp = compute_dd_scores(compressed)

    trace_orig = compute_trace_signs(original)
    trace_comp = compute_trace_signs(compressed)

    return AuditReport(
        compression_method=compression_method,
        compression_params=compression_params,
        verdict=verdict,
        confidence=confidence,
        mean_E_correlation=mean_corr,
        per_block_correlations=correlations,
        mean_dd_score_original=np.mean(dd_orig),
        mean_dd_score_compressed=np.mean(dd_comp),
        pct_negative_trace_original=np.mean(trace_orig),
        pct_negative_trace_compressed=np.mean(trace_comp),
    )


# -------------------------------------------------------------------- experiments

def run_pruning_experiments(model: nn.Module, verbose=True) -> List[AuditReport]:
    """Test various pruning levels."""
    reports = []
    sparsities = [0.30, 0.50, 0.70, 0.90]

    for sparsity in sparsities:
        if verbose:
            print(f"  Magnitude pruning: {int(sparsity*100)}% sparsity...")

        pruned = magnitude_prune(model, sparsity)
        report = audit_compressed_model(
            model, pruned,
            compression_method="magnitude_pruning",
            compression_params={"sparsity": sparsity}
        )
        reports.append(report)

        if verbose:
            print(f"    E correlation: {report.mean_E_correlation:.3f} -> {report.verdict}")

    for sparsity in [0.30, 0.50]:
        if verbose:
            print(f"  Structured pruning: {int(sparsity*100)}% channels...")

        pruned = structured_prune_channels(model, sparsity)
        report = audit_compressed_model(
            model, pruned,
            compression_method="structured_pruning",
            compression_params={"sparsity": sparsity}
        )
        reports.append(report)

        if verbose:
            print(f"    E correlation: {report.mean_E_correlation:.3f} -> {report.verdict}")

    return reports


def run_quantization_experiments(model: nn.Module, verbose=True) -> List[AuditReport]:
    """Test various quantization levels."""
    reports = []
    bit_widths = [16, 8, 4]

    for bits in bit_widths:
        if verbose:
            print(f"  INT{bits} quantization...")

        quantized = quantize_model(model, bits)
        report = audit_compressed_model(
            model, quantized,
            compression_method="quantization",
            compression_params={"bits": bits}
        )
        reports.append(report)

        if verbose:
            print(f"    E correlation: {report.mean_E_correlation:.3f} -> {report.verdict}")

    return reports


def run_distillation_experiment(teacher: nn.Module, verbose=True) -> AuditReport:
    """Test distillation (should NOT preserve fingerprint)."""
    if verbose:
        print("  Knowledge distillation...")

    student = distill_model(teacher, student_seed=777, epochs=100, verbose=verbose)

    report = audit_compressed_model(
        teacher, student,
        compression_method="distillation",
        compression_params={"epochs": 100, "student_seed": 777}
    )

    if verbose:
        print(f"    E correlation: {report.mean_E_correlation:.3f} -> {report.verdict}")

    return report


def run_fine_tuning_experiment(model: nn.Module, verbose=True) -> List[AuditReport]:
    """Test fine-tuning with different epoch counts."""
    reports = []
    epoch_counts = [5, 20, 50]

    for epochs in epoch_counts:
        if verbose:
            print(f"  Fine-tuning: {epochs} epochs...")

        fine_tuned = fine_tune_model(model, epochs=epochs, lr=1e-4, seed=999)
        report = audit_compressed_model(
            model, fine_tuned,
            compression_method="fine_tuning",
            compression_params={"epochs": epochs, "lr": 1e-4}
        )
        reports.append(report)

        if verbose:
            print(f"    E correlation: {report.mean_E_correlation:.3f} -> {report.verdict}")

    return reports


def run_independent_model_experiment(original: nn.Module, verbose=True) -> AuditReport:
    """Test independent model (different seed, same architecture)."""
    if verbose:
        print("  Independent model (different training run)...")

    independent = train_model(seed=999, epochs=200, verbose=False)

    report = audit_compressed_model(
        original, independent,
        compression_method="independent_training",
        compression_params={"seed": 999}
    )

    if verbose:
        print(f"    E correlation: {report.mean_E_correlation:.3f} -> {report.verdict}")

    return report


# -------------------------------------------------------------------- main experiment runner

def run_all_experiments(output_dir: Path, seed: int = 0, verbose: bool = True):
    """Run all compression audit experiments."""
    print("=" * 60)
    print("Case Study 3: Model Compression Auditing")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")
    print()

    t0 = time.time()
    all_reports = []

    print("Training original model...")
    original = train_model(seed=seed, epochs=200, verbose=verbose)

    dd_orig = compute_dd_scores(original)
    trace_orig = compute_trace_signs(original)
    print(f"  Original model DD score: {np.mean(dd_orig):.3f}")
    print(f"  Original model negative trace: {np.mean(trace_orig)*100:.1f}%")
    print()

    print("Running pruning experiments...")
    all_reports.extend(run_pruning_experiments(original, verbose))
    print()

    print("Running quantization experiments...")
    all_reports.extend(run_quantization_experiments(original, verbose))
    print()

    print("Running fine-tuning experiments...")
    all_reports.extend(run_fine_tuning_experiment(original, verbose))
    print()

    print("Running distillation experiment...")
    all_reports.append(run_distillation_experiment(original, verbose))
    print()

    print("Running independent model experiment...")
    all_reports.append(run_independent_model_experiment(original, verbose))
    print()

    elapsed = time.time() - t0

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_reports:
        row = {
            "method": r.compression_method,
            "params": json.dumps(r.compression_params),
            "verdict": r.verdict,
            "confidence": r.confidence,
            "mean_E_corr": r.mean_E_correlation,
            "dd_score_orig": r.mean_dd_score_original,
            "dd_score_comp": r.mean_dd_score_compressed,
            "neg_trace_orig": r.pct_negative_trace_original,
            "neg_trace_comp": r.pct_negative_trace_compressed,
        }
        rows.append(row)

    import csv
    csv_path = output_dir / "compression_audit_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "seed": seed,
        "n_experiments": len(all_reports),
        "original_model": {
            "mean_dd_score": float(np.mean(dd_orig)),
            "pct_negative_trace": float(np.mean(trace_orig)),
        },
        "results_by_method": {},
        "decision_thresholds": {
            "DERIVED": "> 0.90",
            "LIKELY_DERIVED": "0.70 - 0.90",
            "INCONCLUSIVE": "0.30 - 0.70",
            "NOT_DERIVED": "< 0.30",
        },
    }

    for r in all_reports:
        key = f"{r.compression_method}_{json.dumps(r.compression_params)}"
        summary["results_by_method"][key] = {
            "verdict": r.verdict,
            "mean_E_correlation": r.mean_E_correlation,
            "per_block_correlations": r.per_block_correlations,
            "confidence": r.confidence,
        }

    verdict_counts = {}
    for r in all_reports:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1
    summary["verdict_distribution"] = verdict_counts

    pruning_derived = all(
        r.verdict in ["DERIVED", "LIKELY_DERIVED"]
        for r in all_reports
        if "pruning" in r.compression_method and r.compression_params.get("sparsity", 0) <= 0.7
    )
    quantization_derived = all(
        r.verdict in ["DERIVED", "LIKELY_DERIVED"]
        for r in all_reports
        if r.compression_method == "quantization"
    )
    distillation_not_derived = all(
        r.verdict == "NOT_DERIVED"
        for r in all_reports
        if r.compression_method == "distillation"
    )
    independent_not_derived = all(
        r.verdict == "NOT_DERIVED"
        for r in all_reports
        if r.compression_method == "independent_training"
    )

    summary["hypothesis_validation"] = {
        "pruning_preserves_fingerprint": pruning_derived,
        "quantization_preserves_fingerprint": quantization_derived,
        "distillation_erases_fingerprint": distillation_not_derived,
        "independent_has_different_fingerprint": independent_not_derived,
        "all_hypotheses_confirmed": pruning_derived and quantization_derived and distillation_not_derived and independent_not_derived,
    }

    json_path = output_dir / "compression_audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON: {json_path}")

    print()
    print("=" * 60)
    print("COMPRESSION AUDIT - SUMMARY")
    print("=" * 60)
    print()
    print("Method                          | E Corr | Verdict")
    print("-" * 60)
    for r in all_reports:
        params_str = json.dumps(r.compression_params)
        method_str = f"{r.compression_method} {params_str}"[:30].ljust(30)
        print(f"{method_str} | {r.mean_E_correlation:6.3f} | {r.verdict}")
    print("-" * 60)
    print()
    print("Hypothesis Validation:")
    for key, val in summary["hypothesis_validation"].items():
        status = "✓" if val else "✗"
        print(f"  {status} {key}: {val}")
    print()
    print(f"Total time: {elapsed:.1f}s")

    return all_reports, summary


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../case_studies/case_study_3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    output_dir = Path(__file__).parent / args.out_dir
    run_all_experiments(output_dir, seed=args.seed, verbose=not args.quiet)


if __name__ == "__main__":
    main()
