"""Detect whether a model has been subjected to lineage-suppression attack.

Hypothesis: An attacked model was originally trained (has diagonal-dominant
branch products) but then perturbed to break lineage. A genuinely unrelated
model was trained from scratch — its structure is coherent.

We compare:
1. Reference model A (trained)
2. Attacked model A+Δ at various λ
3. Genuinely independent model I (trained from scratch)

Detection signals to test:
- Diagonal dominance score s(M): attacked models should STILL have high s(M)
  because the attack minimizes lineage cosine, not diagonal dominance
- Weight norm ratios: attack might create unusual scale patterns
- Singular value entropy: attack perturbations might be low-rank
- Residual centering consistency: attacked model's E_ℓ should look like
  noise added to A's E_ℓ, not like a fresh E_ℓ from training
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lineage_phase1_mlp import (Block, ResNet, synthetic_target, make_data,
                                 train_model, fresh_model, eval_loss,
                                 branch_products)
import lineage_detection as ldet


def diagonal_dominance_scores(model):
    """Return list of s(M_ℓ) = |tr(M)|/||M||_F for each block."""
    scores = []
    for blk in model.blocks:
        W_in = blk.inp.weight.detach()
        W_out = blk.out.weight.detach()
        M = W_out @ W_in
        tr = torch.trace(M).abs()
        frob = torch.norm(M, 'fro')
        scores.append(float(tr / frob))
    return scores


def weight_norms(model):
    """Return dict of weight statistics per block."""
    stats = []
    for i, blk in enumerate(model.blocks):
        W_in = blk.inp.weight.detach()
        W_out = blk.out.weight.detach()
        stats.append({
            'block': i,
            'W_in_frob': float(torch.norm(W_in, 'fro')),
            'W_out_frob': float(torch.norm(W_out, 'fro')),
            'W_in_spectral': float(torch.linalg.norm(W_in, ord=2)),
            'W_out_spectral': float(torch.linalg.norm(W_out, ord=2)),
        })
    return stats


def singular_value_entropy(W):
    """Entropy of normalized singular values (higher = more spread out)."""
    s = torch.linalg.svdvals(W)
    s = s / s.sum()
    s = s[s > 1e-10]  # avoid log(0)
    return float(-(s * torch.log(s)).sum())


def sv_entropy_per_block(model):
    """Return SV entropy for each block's W_in and W_out."""
    stats = []
    for i, blk in enumerate(model.blocks):
        W_in = blk.inp.weight.detach()
        W_out = blk.out.weight.detach()
        stats.append({
            'block': i,
            'W_in_sv_entropy': singular_value_entropy(W_in),
            'W_out_sv_entropy': singular_value_entropy(W_out),
        })
    return stats


def centered_signature_norms(model):
    """Return ||E_ℓ||_F for each block (the traceless remainder)."""
    norms = []
    for blk in model.blocks:
        W_in = blk.inp.weight.detach()
        W_out = blk.out.weight.detach()
        M = W_out @ W_in
        d = M.shape[0]
        alpha = torch.trace(M) / d
        E = M - alpha * torch.eye(d)
        norms.append(float(torch.norm(E, 'fro')))
    return norms


def gradient_attack(reference, X_eval, y_eval, tau_s, ref_Ms,
                    lambda_utility=1.0, n_steps=200, lr=1e-3):
    """Reproduce the attack and return the attacked model."""
    suspect = copy.deepcopy(reference)
    for p in suspect.parameters():
        p.requires_grad_(True)

    ref_outputs = reference(X_eval).detach().squeeze(-1)
    ref_blocks = [
        (blk.inp.weight.detach().clone(), blk.out.weight.detach().clone())
        for blk in reference.blocks
    ]

    opt = torch.optim.Adam(suspect.parameters(), lr=lr)

    for step in range(n_steps):
        cos_sum = 0.0
        L_blocks = len(suspect.blocks)
        for i, blk in enumerate(suspect.blocks):
            W_in_ref, W_out_ref = ref_blocks[i]
            W_in = blk.inp.weight
            W_out = blk.out.weight
            M_ref = W_out_ref @ W_in_ref
            M_sus = W_out @ W_in
            d = M_ref.shape[0]
            alpha_ref = torch.trace(M_ref) / d
            alpha_sus = torch.trace(M_sus) / d
            R_ref = (M_ref - alpha_ref * torch.eye(d)).reshape(-1)
            R_sus = (M_sus - alpha_sus * torch.eye(d)).reshape(-1)
            phi_ref = R_ref / (R_ref.norm() + 1e-12)
            phi_sus = R_sus / (R_sus.norm() + 1e-12)
            cos_sum = cos_sum + (phi_ref * phi_sus).sum()
        cos_mean = cos_sum / L_blocks

        utility_loss = F.mse_loss(suspect(X_eval).squeeze(-1), ref_outputs)
        loss = cos_mean + lambda_utility * utility_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    for p in suspect.parameters():
        p.requires_grad_(False)

    return suspect


def analyze_model(model, name):
    """Compute all detection metrics for a model."""
    dd_scores = diagonal_dominance_scores(model)
    w_norms = weight_norms(model)
    sv_ent = sv_entropy_per_block(model)
    e_norms = centered_signature_norms(model)

    return {
        'name': name,
        'dd_scores': dd_scores,
        'dd_mean': float(np.mean(dd_scores)),
        'dd_std': float(np.std(dd_scores)),
        'weight_norms': w_norms,
        'sv_entropy': sv_ent,
        'sv_entropy_W_in_mean': float(np.mean([s['W_in_sv_entropy'] for s in sv_ent])),
        'sv_entropy_W_out_mean': float(np.mean([s['W_out_sv_entropy'] for s in sv_ent])),
        'E_norms': e_norms,
        'E_norm_mean': float(np.mean(e_norms)),
        'E_norm_std': float(np.std(e_norms)),
    }


def main():
    Path('results').mkdir(parents=True, exist_ok=True)
    out_path = Path('results/attack_detection.json')

    depth, hidden, in_dim = 24, 64, 24
    epochs_root = 200

    # Create data
    X, y = make_data(in_dim=in_dim, n=4000, seed=0, target_key=42)
    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]

    print("Training reference model A...")
    A = fresh_model(depth, hidden, in_dim, seed=100)
    train_model(A, X_train, y_train, epochs=epochs_root)
    ref_loss = eval_loss(A, X_eval, y_eval)
    print(f"  Reference eval_loss: {ref_loss:.4f}")

    ref_Ms = branch_products(A)
    tau_s = ldet.choose_tau_s([ref_Ms])

    # Train independent models
    print("\nTraining 5 independent models...")
    independents = []
    for seed in [500, 501, 502, 503, 504]:
        model = fresh_model(depth, hidden, in_dim, seed=seed)
        train_model(model, X_train, y_train, epochs=epochs_root)
        independents.append(model)
        L, _, _ = ldet.lineage_score(ref_Ms, branch_products(model), tau_s)
        print(f"  Seed {seed}: L={L:.4f}, eval_loss={eval_loss(model, X_eval, y_eval):.4f}")

    # Run attacks at different lambda
    print("\nRunning attacks...")
    attacked_models = {}
    for lam in [0.01, 0.1, 1.0]:
        print(f"  λ={lam}...")
        attacked = gradient_attack(A, X_eval, y_eval, tau_s, ref_Ms,
                                   lambda_utility=lam, n_steps=200, lr=1e-3)
        attacked_models[lam] = attacked
        Ms_att = branch_products(attacked)
        L, _, _ = ldet.lineage_score(ref_Ms, Ms_att, tau_s)
        att_loss = eval_loss(attacked, X_eval, y_eval)
        print(f"    L={L:.4f}, eval_loss={att_loss:.4f}")

    # Analyze all models
    print("\nAnalyzing models...")
    results = {
        'reference': analyze_model(A, 'reference'),
        'independents': [analyze_model(m, f'independent_{i}') for i, m in enumerate(independents)],
        'attacked': {str(lam): analyze_model(m, f'attacked_λ={lam}')
                     for lam, m in attacked_models.items()},
    }

    # Compute lineage scores for all
    results['lineage_scores'] = {
        'reference_vs_reference': 1.0,
        'independents': [float(ldet.lineage_score(ref_Ms, branch_products(m), tau_s)[0])
                        for m in independents],
        'attacked': {str(lam): float(ldet.lineage_score(ref_Ms, branch_products(m), tau_s)[0])
                    for lam, m in attacked_models.items()},
    }

    # Print summary
    print("\n" + "="*70)
    print("DETECTION ANALYSIS")
    print("="*70)

    print("\nDiagonal Dominance (s(M)) - should be HIGH for trained models:")
    print(f"  Reference:     mean={results['reference']['dd_mean']:.3f} ± {results['reference']['dd_std']:.3f}")
    for i, ind in enumerate(results['independents']):
        print(f"  Independent_{i}: mean={ind['dd_mean']:.3f} ± {ind['dd_std']:.3f}")
    for lam, att in results['attacked'].items():
        print(f"  Attacked λ={lam}: mean={att['dd_mean']:.3f} ± {att['dd_std']:.3f}")

    print("\nLineage Score L (vs reference):")
    print(f"  Reference:     L=1.000")
    for i, L in enumerate(results['lineage_scores']['independents']):
        print(f"  Independent_{i}: L={L:.4f}")
    for lam, L in results['lineage_scores']['attacked'].items():
        print(f"  Attacked λ={lam}: L={L:.4f}")

    print("\nCentered Signature Norm ||E|| (traceless remainder):")
    print(f"  Reference:     mean={results['reference']['E_norm_mean']:.3f} ± {results['reference']['E_norm_std']:.3f}")
    for i, ind in enumerate(results['independents']):
        print(f"  Independent_{i}: mean={ind['E_norm_mean']:.3f} ± {ind['E_norm_std']:.3f}")
    for lam, att in results['attacked'].items():
        print(f"  Attacked λ={lam}: mean={att['E_norm_mean']:.3f} ± {att['E_norm_std']:.3f}")

    print("\nSV Entropy (W_in) - higher = more spread singular values:")
    print(f"  Reference:     {results['reference']['sv_entropy_W_in_mean']:.3f}")
    for i, ind in enumerate(results['independents']):
        print(f"  Independent_{i}: {ind['sv_entropy_W_in_mean']:.3f}")
    for lam, att in results['attacked'].items():
        print(f"  Attacked λ={lam}: {att['sv_entropy_W_in_mean']:.3f}")

    # Key insight check
    print("\n" + "="*70)
    print("KEY INSIGHT: Can we detect attack via diagonal dominance?")
    print("="*70)
    print("\nIf attacked models retain HIGH diagonal dominance but LOW lineage,")
    print("that's suspicious - genuine unrelated models should have BOTH from scratch.\n")

    att_01 = results['attacked']['0.1']
    ind_mean_dd = np.mean([ind['dd_mean'] for ind in results['independents']])
    ind_mean_L = np.mean(results['lineage_scores']['independents'])

    print(f"Attacked λ=0.1:  dd={att_01['dd_mean']:.3f}, L={results['lineage_scores']['attacked']['0.1']:.4f}")
    print(f"Independents:    dd={ind_mean_dd:.3f}, L={ind_mean_L:.4f}")
    print(f"Reference:       dd={results['reference']['dd_mean']:.3f}, L=1.000")

    if att_01['dd_mean'] > ind_mean_dd * 0.9:
        print("\n→ Attacked model has SIMILAR diagonal dominance to independents")
        print("  Detection via dd alone: UNLIKELY")
    else:
        print("\n→ Attacked model has DIFFERENT diagonal dominance")
        print("  Detection via dd: POSSIBLE")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
