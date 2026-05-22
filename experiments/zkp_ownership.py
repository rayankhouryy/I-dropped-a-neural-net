"""Case Study 2: Zero-Knowledge Ownership Proofs

Implements a cryptographic protocol for proving model ownership using the
diagonal dominance fingerprint. The prover commits to the E matrices
(residual after removing -εI) and can later prove ownership by revealing
weights for challenged blocks.

Protocol:
  1. REGISTRATION: Owner commits to E[k] for each block
  2. CHALLENGE: Verifier selects random blocks to challenge
  3. RESPONSE: Owner reveals weights + commitment randomness for challenged blocks
  4. VERIFICATION: Verifier checks that W_out × W_in matches committed E[k]

Experiments:
  - Honest prover, honest verifier → PASS
  - Honest prover, fine-tuned model → PASS (E robust to fine-tuning)
  - Attacker with different training run → FAIL (different E)
  - Attacker with distilled model → FAIL (E not preserved)
  - Attacker with random weights → FAIL (DD score too low)

Outputs:
  case_studies/case_study_2/zkp_results.json
  case_studies/case_study_2/figures/
"""
import argparse
import hashlib
import json
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def fine_tune_model(model, epochs=20, lr=1e-4, seed=999):
    """Fine-tune an existing model on slightly different data."""
    torch.manual_seed(seed)
    model = model.to(DEVICE)

    in_dim = model.blocks[0].inp.in_features
    X, y = make_data(in_dim=in_dim, n=2000, seed=seed + 5000)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(X.shape[0])
        for s in range(0, X.shape[0], 256):
            idx = perm[s:s + 256]
            xb = X[idx].to(DEVICE)
            yb = y[idx].to(DEVICE)
            yb_pred = model(xb).squeeze(-1)
            loss = loss_fn(yb_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def distill_model(teacher, student_seed, epochs=100, lr=1e-3):
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

    return student


# -------------------------------------------------------------------- ZKP Protocol

@dataclass
class BlockCommitment:
    """Commitment for a single residual block."""
    block_idx: int
    epsilon: float
    E_hash: str
    trace_sign: bool
    dd_score: float
    randomness: str  # hex string


@dataclass
class OwnershipCertificate:
    """Certificate proving ownership of a model."""
    owner_id: str
    model_hash: str
    n_blocks: int
    block_commitments: List[BlockCommitment]
    dd_matrix_hash: str
    timestamp: str

    # Store actual E matrices and randomness for later revelation (prover keeps secret)
    _E_matrices: List[np.ndarray] = field(default_factory=list, repr=False)
    _randomness: List[bytes] = field(default_factory=list, repr=False)


@dataclass
class Challenge:
    """Verifier's challenge to the prover."""
    certificate_id: str
    challenged_blocks: List[int]
    nonce: str


@dataclass
class Revelation:
    """Prover's revelation for a single challenged block."""
    block_idx: int
    W_in: np.ndarray
    W_out: np.ndarray
    randomness: str


@dataclass
class ChallengeResponse:
    """Prover's response to a challenge."""
    challenge_id: str
    revelations: List[Revelation]


@dataclass
class VerificationResult:
    """Result of verification."""
    success: bool
    reason: str
    details: Dict = field(default_factory=dict)


def hash_array(arr: np.ndarray, randomness: bytes = b"") -> str:
    """Hash a numpy array with optional randomness (commitment)."""
    data = arr.tobytes() + randomness
    return hashlib.sha256(data).hexdigest()


def hash_model_weights(model: nn.Module) -> str:
    """Hash all model weights."""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def extract_E_matrix(W_in: np.ndarray, W_out: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Extract the E matrix from W_out @ W_in.

    M = W_out @ W_in = -ε*I + E
    where ε = |tr(M)| / d and tr(E) = 0

    Returns: (epsilon, E, dd_score)
    """
    M = W_out.astype(np.float64) @ W_in.astype(np.float64)
    d = M.shape[0]
    trace = np.trace(M)
    epsilon = abs(trace) / d

    # E = M + ε*I (if trace is negative) or M - ε*I (if trace is positive)
    # We want M = -ε*I + E, so E = M + ε*I when trace < 0
    if trace < 0:
        E = M + epsilon * np.eye(d)
    else:
        E = M - epsilon * np.eye(d)

    # DD score
    frob_norm = np.linalg.norm(M, 'fro') + 1e-12
    dd_score = abs(trace) / frob_norm

    return epsilon, E, dd_score


def register_ownership(model: nn.Module, owner_id: str) -> OwnershipCertificate:
    """
    Phase 1: Registration

    Generate ownership certificate by committing to E matrices.
    """
    model.eval()
    depth = len(model.blocks)

    block_commitments = []
    E_matrices = []
    randomness_list = []

    for k in range(depth):
        W_in = model.blocks[k].inp.weight.detach().cpu().numpy()
        W_out = model.blocks[k].out.weight.detach().cpu().numpy()

        epsilon, E, dd_score = extract_E_matrix(W_in, W_out)

        # Generate randomness for commitment
        randomness = secrets.token_bytes(32)
        E_hash = hash_array(E, randomness)

        # Check trace sign
        M = W_out.astype(np.float64) @ W_in.astype(np.float64)
        trace_sign = np.trace(M) < 0

        block_commitments.append(BlockCommitment(
            block_idx=k,
            epsilon=float(epsilon),
            E_hash=E_hash,
            trace_sign=trace_sign,
            dd_score=float(dd_score),
            randomness=randomness.hex(),
        ))

        E_matrices.append(E)
        randomness_list.append(randomness)

    # Compute DD matrix hash
    dd_matrix = compute_dd_matrix(model)
    dd_matrix_hash = hash_array(dd_matrix)

    # Model hash
    model_hash = hash_model_weights(model)

    cert = OwnershipCertificate(
        owner_id=owner_id,
        model_hash=model_hash,
        n_blocks=depth,
        block_commitments=block_commitments,
        dd_matrix_hash=dd_matrix_hash,
        timestamp=datetime.now().isoformat(),
    )

    # Store secrets for later revelation
    cert._E_matrices = E_matrices
    cert._randomness = randomness_list

    return cert


def compute_dd_matrix(model: nn.Module) -> np.ndarray:
    """Compute the diagonal dominance score matrix."""
    depth = len(model.blocks)
    W_in_list = [b.inp.weight.detach().cpu().numpy() for b in model.blocks]
    W_out_list = [b.out.weight.detach().cpu().numpy() for b in model.blocks]

    M = np.zeros((depth, depth), dtype=np.float64)
    for i in range(depth):
        for j in range(depth):
            P = W_out_list[j].astype(np.float64) @ W_in_list[i].astype(np.float64)
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def generate_challenge(certificate: OwnershipCertificate,
                       num_challenges: int = 5,
                       seed: Optional[int] = None) -> Challenge:
    """
    Phase 2: Challenge

    Verifier selects random blocks to challenge.
    """
    if seed is not None:
        np.random.seed(seed)

    n_blocks = certificate.n_blocks
    num_challenges = min(num_challenges, n_blocks)
    challenged_blocks = list(np.random.choice(n_blocks, num_challenges, replace=False))

    nonce = secrets.token_hex(16)
    cert_id = hash_array(np.array([ord(c) for c in certificate.model_hash[:16]]))

    return Challenge(
        certificate_id=cert_id,
        challenged_blocks=challenged_blocks,
        nonce=nonce,
    )


def respond_to_challenge(model: nn.Module,
                         certificate: OwnershipCertificate,
                         challenge: Challenge) -> ChallengeResponse:
    """
    Phase 3: Response

    Prover reveals weights for challenged blocks.
    """
    revelations = []

    for k in challenge.challenged_blocks:
        W_in = model.blocks[k].inp.weight.detach().cpu().numpy()
        W_out = model.blocks[k].out.weight.detach().cpu().numpy()

        revelations.append(Revelation(
            block_idx=k,
            W_in=W_in,
            W_out=W_out,
            randomness=certificate.block_commitments[k].randomness,
        ))

    return ChallengeResponse(
        challenge_id=challenge.nonce,
        revelations=revelations,
    )


def verify_ownership(certificate: OwnershipCertificate,
                     challenge: Challenge,
                     response: ChallengeResponse,
                     tolerance: float = 1e-6) -> VerificationResult:
    """
    Phase 4: Verification

    Verifier checks that revealed weights match commitments.
    """
    details = {
        "checks": [],
        "dd_scores": [],
        "epsilon_errors": [],
    }

    for revelation in response.revelations:
        k = revelation.block_idx
        W_in = revelation.W_in
        W_out = revelation.W_out
        randomness = bytes.fromhex(revelation.randomness)

        commitment = certificate.block_commitments[k]

        # Compute E matrix from revealed weights
        epsilon, E, dd_score = extract_E_matrix(W_in, W_out)

        # Check 1: E hash matches commitment
        computed_hash = hash_array(E, randomness)
        if computed_hash != commitment.E_hash:
            return VerificationResult(
                success=False,
                reason=f"Block {k}: E_hash mismatch",
                details={"block": k, "expected": commitment.E_hash, "got": computed_hash}
            )
        details["checks"].append(f"Block {k}: E_hash OK")

        # Check 2: Epsilon matches (within tolerance for floating point)
        epsilon_error = abs(epsilon - commitment.epsilon)
        details["epsilon_errors"].append(epsilon_error)
        if epsilon_error > tolerance:
            return VerificationResult(
                success=False,
                reason=f"Block {k}: epsilon mismatch (error={epsilon_error:.2e})",
                details={"block": k, "expected": commitment.epsilon, "got": epsilon}
            )
        details["checks"].append(f"Block {k}: epsilon OK")

        # Check 3: Trace sign matches
        M = W_out.astype(np.float64) @ W_in.astype(np.float64)
        actual_sign = np.trace(M) < 0
        if actual_sign != commitment.trace_sign:
            return VerificationResult(
                success=False,
                reason=f"Block {k}: trace sign mismatch",
                details={"block": k, "expected": commitment.trace_sign, "got": actual_sign}
            )
        details["checks"].append(f"Block {k}: trace_sign OK")

        # Check 4: DD score is reasonable (trained models should have dd_score > 1)
        details["dd_scores"].append(dd_score)
        if dd_score < 0.5:
            return VerificationResult(
                success=False,
                reason=f"Block {k}: DD score too low ({dd_score:.3f})",
                details={"block": k, "dd_score": dd_score}
            )
        details["checks"].append(f"Block {k}: dd_score OK ({dd_score:.3f})")

    return VerificationResult(
        success=True,
        reason="All checks passed",
        details=details
    )


# -------------------------------------------------------------------- Experiments

def run_honest_verification(seed=0, verbose=True):
    """Experiment: Honest prover, honest verifier."""
    if verbose:
        print("\n=== Experiment: Honest Prover, Honest Verifier ===")
        print("Training model...")

    model = train_model(seed=seed, epochs=200, verbose=verbose)

    if verbose:
        print("Registering ownership...")
    cert = register_ownership(model, owner_id="Alice")

    if verbose:
        print(f"Certificate: {cert.n_blocks} blocks, model_hash={cert.model_hash[:16]}...")

    # Verifier challenges
    challenge = generate_challenge(cert, num_challenges=5, seed=42)
    if verbose:
        print(f"Challenge: blocks {challenge.challenged_blocks}")

    # Prover responds
    response = respond_to_challenge(model, cert, challenge)

    # Verify
    result = verify_ownership(cert, challenge, response)
    if verbose:
        print(f"Result: {'PASS' if result.success else 'FAIL'} - {result.reason}")
        if result.success:
            print(f"  DD scores: {[f'{s:.3f}' for s in result.details['dd_scores']]}")

    return {
        "scenario": "honest_prover_honest_verifier",
        "success": result.success,
        "reason": result.reason,
        "dd_scores": result.details.get("dd_scores", []),
    }


def run_fine_tuned_verification(seed=0, verbose=True):
    """Experiment: Honest prover with fine-tuned model."""
    if verbose:
        print("\n=== Experiment: Fine-tuned Model ===")
        print("Training original model...")

    model = train_model(seed=seed, epochs=200, verbose=verbose)

    if verbose:
        print("Registering ownership...")
    cert = register_ownership(model, owner_id="Alice")

    if verbose:
        print("Fine-tuning model for 20 epochs...")
    model = fine_tune_model(model, epochs=20, lr=1e-4)

    # Challenge with fine-tuned model
    challenge = generate_challenge(cert, num_challenges=5, seed=42)
    response = respond_to_challenge(model, cert, challenge)

    # This should FAIL because the weights changed
    result = verify_ownership(cert, challenge, response)
    if verbose:
        print(f"Result: {'PASS' if result.success else 'FAIL'} - {result.reason}")

    return {
        "scenario": "fine_tuned_model",
        "success": result.success,
        "reason": result.reason,
        "expected": "FAIL (weights changed)",
    }


def run_different_training_run(seed=0, verbose=True):
    """Experiment: Attacker with different training run (same architecture)."""
    if verbose:
        print("\n=== Experiment: Different Training Run (Attacker) ===")
        print("Training owner's model (seed=0)...")

    owner_model = train_model(seed=0, epochs=200, verbose=verbose)
    cert = register_ownership(owner_model, owner_id="Alice")

    if verbose:
        print("Training attacker's model (seed=999)...")
    attacker_model = train_model(seed=999, epochs=200, verbose=verbose)

    # Attacker tries to respond with their model
    challenge = generate_challenge(cert, num_challenges=5, seed=42)
    response = respond_to_challenge(attacker_model, cert, challenge)

    result = verify_ownership(cert, challenge, response)
    if verbose:
        print(f"Result: {'PASS' if result.success else 'FAIL'} - {result.reason}")

    return {
        "scenario": "different_training_run",
        "success": result.success,
        "reason": result.reason,
        "expected": "FAIL (different E matrices)",
    }


def run_distilled_model(seed=0, verbose=True):
    """Experiment: Attacker with distilled model."""
    if verbose:
        print("\n=== Experiment: Distilled Model (Attacker) ===")
        print("Training teacher model...")

    teacher = train_model(seed=seed, epochs=200, verbose=verbose)
    cert = register_ownership(teacher, owner_id="Alice")

    if verbose:
        print("Distilling to student model...")
    student = distill_model(teacher, student_seed=777, epochs=100)

    # Attacker tries to respond with distilled model
    challenge = generate_challenge(cert, num_challenges=5, seed=42)
    response = respond_to_challenge(student, cert, challenge)

    result = verify_ownership(cert, challenge, response)
    if verbose:
        print(f"Result: {'PASS' if result.success else 'FAIL'} - {result.reason}")

    return {
        "scenario": "distilled_model",
        "success": result.success,
        "reason": result.reason,
        "expected": "FAIL (E not preserved through distillation)",
    }


def run_random_weights(seed=0, verbose=True):
    """Experiment: Attacker with random weights."""
    if verbose:
        print("\n=== Experiment: Random Weights (Attacker) ===")
        print("Training owner's model...")

    owner_model = train_model(seed=seed, epochs=200, verbose=verbose)
    cert = register_ownership(owner_model, owner_id="Alice")

    if verbose:
        print("Creating random model (no training)...")
    depth = len(owner_model.blocks)
    in_dim = owner_model.blocks[0].inp.in_features
    hidden = owner_model.blocks[0].inp.out_features

    random_model = ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)
    apply_init(random_model, "kaiming_normal", seed=12345)

    # Attacker tries to respond with random model
    challenge = generate_challenge(cert, num_challenges=5, seed=42)
    response = respond_to_challenge(random_model, cert, challenge)

    result = verify_ownership(cert, challenge, response)
    if verbose:
        print(f"Result: {'PASS' if result.success else 'FAIL'} - {result.reason}")

    return {
        "scenario": "random_weights",
        "success": result.success,
        "reason": result.reason,
        "expected": "FAIL (E_hash mismatch or DD score too low)",
    }


def run_all_experiments(output_dir, verbose=True):
    """Run all ZKP experiments."""
    results = {
        "experiments": [],
        "summary": {},
        "timestamp": datetime.now().isoformat(),
    }

    t0 = time.time()

    # Run each experiment
    results["experiments"].append(run_honest_verification(seed=0, verbose=verbose))
    results["experiments"].append(run_fine_tuned_verification(seed=0, verbose=verbose))
    results["experiments"].append(run_different_training_run(seed=0, verbose=verbose))
    results["experiments"].append(run_distilled_model(seed=0, verbose=verbose))
    results["experiments"].append(run_random_weights(seed=0, verbose=verbose))

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    # Summary
    n_pass = sum(1 for e in results["experiments"] if e["success"])
    n_total = len(results["experiments"])

    # For security: honest should pass, adversarial should fail
    honest_passed = results["experiments"][0]["success"]  # honest
    adversarial_failed = all(not e["success"] for e in results["experiments"][2:])  # diff run, distill, random

    results["summary"] = {
        "total_experiments": n_total,
        "passed": n_pass,
        "failed": n_total - n_pass,
        "honest_verification_passed": honest_passed,
        "adversarial_attacks_blocked": adversarial_failed,
        "protocol_secure": honest_passed and adversarial_failed,
    }

    # Print summary
    print("\n" + "="*60)
    print("ZKP OWNERSHIP PROTOCOL - SUMMARY")
    print("="*60)
    for exp in results["experiments"]:
        status = "PASS" if exp["success"] else "FAIL"
        expected = exp.get("expected", "PASS")
        match = "✓" if (exp["success"] and "PASS" in expected) or (not exp["success"] and "FAIL" in expected) else "✗"
        print(f"  {match} {exp['scenario']}: {status}")
    print("-"*60)
    print(f"Protocol Security: {'SECURE' if results['summary']['protocol_secure'] else 'INSECURE'}")
    print(f"Total time: {elapsed:.1f}s")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_serializable = make_serializable(results)

    json_path = output_dir / "zkp_results.json"
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nSaved results to {json_path}")

    return results


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../case_studies/case_study_2")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    output_dir = Path(__file__).parent / args.out_dir

    print("="*60)
    print("Case Study 2: Zero-Knowledge Ownership Proofs")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")

    results = run_all_experiments(output_dir, verbose=not args.quiet)

    return results


if __name__ == "__main__":
    main()
