"""Issue #36 / AAAI Section 6: Adaptive evasion and boundary cases.

Three experiments that stress-test the residual-signature lineage score
under deliberate attacks. The first two are theorem-level invariance
checks; the third is a true adaptive attack.

Experiment 1: Orthogonal-rotation invariance.
============================================
For each block l, sample a random R_l in O(h) and apply
  W_in,l  -> R_l @ W_in,l
  W_out,l -> W_out,l @ R_l^T
The block's function is preserved EXACTLY (the rotation cancels at the
hidden state), and the branch product W_out @ W_in is also exactly
preserved because (W_out R^T)(R W_in) = W_out (R^T R) W_in = W_out W_in.

Prediction: L(A, A_rotated) == 1.000 to numerical precision.

Experiment 2: Hidden-unit permutation invariance.
=================================================
Same as 1 but with R_l = permutation matrix P_l. Same identity-cancel
argument applies. Permutations are easier to apply in practice (they
just shuffle indices) and are a known function-preserving symmetry of
MLP-style blocks.

Prediction: L(A, A_permuted) == 1.000 to numerical precision.

Experiment 3: Gradient-based fingerprint suppression.
=====================================================
Adversary searches for a perturbation Delta that *minimizes* the
lineage score L(A, A + Delta) subject to a utility constraint
||model_output(A+Delta) - model_output(A)|| <= eps.

We solve this as an unconstrained optimization with a tradeoff lambda:
  loss(Delta) = L(A, A + Delta) + lambda * utility_drift(A, A + Delta)
and sweep lambda to trace the Pareto frontier of
(lineage suppression, utility loss).

Prediction: at any utility-preserving Delta the lineage score remains
high; only when utility collapses does lineage drop.

Output: results/lineage_section6_attacks.json
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lineage_detection as ldet
from lineage_phase1_mlp import (Block, ResNet, synthetic_target, make_data,
                                 train_model, fresh_model, eval_loss,
                                 branch_products)


# --------------------------------------------------------------------- attacks
def orthogonal_rotate_blocks(model, seed=0):
    """Apply a per-block random orthogonal rotation R_l in O(h).

    For each block l with hidden dim h:
      W_in  : (h, d_in)  ->  R_l @ W_in
      W_out : (d_in, h)  ->  W_out @ R_l^T
      b_in  : (h,)       ->  R_l @ b_in
    Function preserved exactly: the rotation cancels at the ReLU input.
    (Technically only an exact symmetry for the LINEAR pre-activation;
    ReLU breaks general rotations unless they are signed permutations.
    We test BOTH cases below: rotation = function-preserving for the
    *linear* part, permutation = function-preserving for the FULL block
    including ReLU.)
    """
    g = torch.Generator().manual_seed(seed)
    rotated = copy.deepcopy(model)
    for blk in rotated.blocks:
        h = blk.inp.out_features
        # Random orthogonal matrix via QR decomposition.
        A = torch.randn(h, h, generator=g)
        Q, _ = torch.linalg.qr(A)
        with torch.no_grad():
            blk.inp.weight.copy_(Q @ blk.inp.weight)
            blk.inp.bias.copy_(Q @ blk.inp.bias)
            blk.out.weight.copy_(blk.out.weight @ Q.T)
    return rotated


def permute_hidden_units(model, seed=0):
    """Apply a random permutation of hidden units in each block.

    This IS a function-preserving symmetry of the full ReLU block,
    because P @ ReLU(x) = ReLU(P @ x) when P is a permutation.
    """
    g = torch.Generator().manual_seed(seed)
    permuted = copy.deepcopy(model)
    for blk in permuted.blocks:
        h = blk.inp.out_features
        perm = torch.randperm(h, generator=g)
        with torch.no_grad():
            blk.inp.weight.copy_(blk.inp.weight[perm, :])
            blk.inp.bias.copy_(blk.inp.bias[perm])
            blk.out.weight.copy_(blk.out.weight[:, perm])
    return permuted


def gradient_attack(reference, X_eval, y_eval, tau_s, ref_Ms,
                    lambda_utility=1.0, n_steps=200, lr=1e-3,
                    seed=0, verbose=False):
    """Train Delta to minimize lineage(reference, ref + Delta).

    The objective is approximate-differentiable: we approximate the
    lineage score by mean cosine of identity-aligned residual signatures
    (skipping Hungarian, which is non-differentiable). The signatures
    themselves are differentiable through PyTorch.

      L_attack(Delta) = mean_l cos(phi_l(M_l^ref), phi_l(M_l^(ref+Delta)))
                         + lambda_utility * ||f(ref+Delta) - f(ref)||^2

    minimization drives lineage cosine DOWN while keeping outputs close
    to the reference.
    """
    suspect = copy.deepcopy(reference)
    for p in suspect.parameters():
        p.requires_grad_(True)
    # We accumulate Delta implicitly: optimize the model parameters
    # directly with reference frozen as targets.
    ref_outputs = reference(X_eval).detach().squeeze(-1)
    ref_blocks = [
        (blk.inp.weight.detach().clone(), blk.out.weight.detach().clone())
        for blk in reference.blocks
    ]

    opt = torch.optim.Adam(suspect.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for step in range(n_steps):
        # Compute branch products with gradients
        cos_sum = 0.0
        L_blocks = len(suspect.blocks)
        for i, blk in enumerate(suspect.blocks):
            W_in_ref, W_out_ref = ref_blocks[i]
            W_in = blk.inp.weight
            W_out = blk.out.weight
            M_ref = W_out_ref @ W_in_ref         # (d, d), no grad
            M_sus = W_out @ W_in                 # (d, d), with grad
            # Residual signatures (differentiable)
            d = M_ref.shape[0]
            alpha_ref = torch.trace(M_ref) / d
            alpha_sus = torch.trace(M_sus) / d
            R_ref = (M_ref - alpha_ref * torch.eye(d)).reshape(-1)
            R_sus = (M_sus - alpha_sus * torch.eye(d)).reshape(-1)
            phi_ref = R_ref / (R_ref.norm() + 1e-12)
            phi_sus = R_sus / (R_sus.norm() + 1e-12)
            cos_sum = cos_sum + (phi_ref * phi_sus).sum()
        cos_mean = cos_sum / L_blocks
        # Utility drift
        utility_loss = loss_fn(suspect(X_eval).squeeze(-1), ref_outputs)
        # Adversary minimizes cosine (negate) but penalizes utility drift
        loss = cos_mean + lambda_utility * utility_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == n_steps - 1:
            with torch.no_grad():
                # measure full lineage (Hungarian) and utility
                Ms_sus = branch_products(suspect)
                L_score, _, _ = ldet.lineage_score(ref_Ms, Ms_sus, tau_s)
                u_loss = float(utility_loss.item())
                history.append({
                    'step': step,
                    'cos_objective': float(cos_mean.item()),
                    'lineage_full': L_score,
                    'utility_drift': u_loss,
                })
                if verbose:
                    print(f"  step {step:>3d}: cos={cos_mean.item():+.4f}  "
                          f"L={L_score:+.4f}  util_drift={u_loss:.4f}",
                          flush=True)
    # Final eval and freeze
    for p in suspect.parameters():
        p.requires_grad_(False)
    Ms_final = branch_products(suspect)
    L_final, _, _ = ldet.lineage_score(ref_Ms, Ms_final, tau_s)
    with torch.no_grad():
        u_final = float(F.mse_loss(suspect(X_eval).squeeze(-1),
                                    ref_outputs).item())
        eval_y_loss = float(F.mse_loss(suspect(X_eval).squeeze(-1),
                                        y_eval).item())
    return {
        'lambda_utility':  lambda_utility,
        'final_lineage':   L_final,
        'utility_drift':   u_final,
        'eval_y_loss':     eval_y_loss,
        'history':         history,
    }


# --------------------------------------------------------------------- main
def main():
    Path('results').mkdir(parents=True, exist_ok=True)
    out_path = Path('results/lineage_section6_attacks.json')

    depth, hidden, in_dim = 24, 64, 24
    epochs_root = 200

    # Build a reference checkpoint and 5 independent same-arch controls
    t0 = time.time()
    X, y = make_data(in_dim=in_dim, n=4000, seed=0, target_key=42)
    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]

    print("Training reference model A...", flush=True)
    A = fresh_model(depth, hidden, in_dim, seed=100)
    train_model(A, X_train, y_train, epochs=epochs_root)
    print(f"  done ({time.time()-t0:.1f}s)  eval_loss={eval_loss(A, X_eval, y_eval):.4f}",
          flush=True)

    ref_Ms = branch_products(A)
    tau_s = ldet.choose_tau_s([ref_Ms])
    print(f"  tau_s = {tau_s:.4f}", flush=True)

    # --- baseline non-descendant for context (1 independent model)
    print("Training 1 independent baseline for context...", flush=True)
    indep = fresh_model(depth, hidden, in_dim, seed=500)
    train_model(indep, X_train, y_train, epochs=epochs_root)
    L_indep, _, _ = ldet.lineage_score(ref_Ms, branch_products(indep), tau_s)
    print(f"  L(A, independent) = {L_indep:+.4f}", flush=True)

    results = {
        'config': {'depth': depth, 'hidden': hidden, 'in_dim': in_dim,
                    'epochs_root': epochs_root, 'tau_s': tau_s},
        'reference_eval_loss': eval_loss(A, X_eval, y_eval),
        'baseline_independent_lineage': L_indep,
    }

    # ============================================================
    # Experiment 1: orthogonal rotation invariance
    # ============================================================
    print("\n=== Experiment 1: orthogonal rotation invariance ===", flush=True)
    exp1 = []
    for seed in range(5):
        A_rot = orthogonal_rotate_blocks(A, seed=seed)
        Ms_rot = branch_products(A_rot)
        L_score, perm, _ = ldet.lineage_score(ref_Ms, Ms_rot, tau_s)
        # Utility check: rotation breaks ReLU symmetry, so outputs may shift
        out_drift = float(F.mse_loss(A_rot(X_eval).squeeze(-1),
                                      A(X_eval).squeeze(-1)).item())
        # Sanity: identity permutation expected
        id_perm = int((perm == np.arange(len(perm))).sum())
        exp1.append({
            'seed': seed,
            'lineage_L':    L_score,
            'output_drift': out_drift,
            'id_perm_count': id_perm,
            'n_layers': len(perm),
        })
        print(f"  seed {seed}: L(A, R(A))={L_score:+.6f}  "
              f"out_drift={out_drift:.4e}  id_perm={id_perm}/{len(perm)}",
              flush=True)
    results['exp1_orthogonal_rotation'] = {
        'description': 'Per-block random orthogonal rotation of hidden units.'
                       ' Function-preserving for linear part; ReLU breaks'
                       ' general rotations so outputs may drift, but the'
                       ' weight product W_out @ W_in is exactly invariant.',
        'runs': exp1,
        'mean_lineage':  float(np.mean([r['lineage_L'] for r in exp1])),
        'min_lineage':   float(np.min([r['lineage_L'] for r in exp1])),
    }

    # ============================================================
    # Experiment 2: hidden-unit permutation invariance
    # ============================================================
    print("\n=== Experiment 2: hidden-unit permutation invariance ===",
          flush=True)
    exp2 = []
    for seed in range(5):
        A_perm = permute_hidden_units(A, seed=seed)
        Ms_perm = branch_products(A_perm)
        L_score, perm, _ = ldet.lineage_score(ref_Ms, Ms_perm, tau_s)
        out_drift = float(F.mse_loss(A_perm(X_eval).squeeze(-1),
                                       A(X_eval).squeeze(-1)).item())
        id_perm = int((perm == np.arange(len(perm))).sum())
        exp2.append({
            'seed': seed,
            'lineage_L':    L_score,
            'output_drift': out_drift,
            'id_perm_count': id_perm,
            'n_layers': len(perm),
        })
        print(f"  seed {seed}: L(A, P(A))={L_score:+.6f}  "
              f"out_drift={out_drift:.4e}  id_perm={id_perm}/{len(perm)}",
              flush=True)
    results['exp2_hidden_permutation'] = {
        'description': 'Per-block random hidden-unit permutation. Function-'
                       'preserving for the full ReLU block (P commutes with'
                       ' ReLU). Branch product exactly invariant. Both the'
                       ' fingerprint score and the model output should be'
                       ' identical to the reference.',
        'runs': exp2,
        'mean_lineage':  float(np.mean([r['lineage_L'] for r in exp2])),
        'min_lineage':   float(np.min([r['lineage_L'] for r in exp2])),
    }

    # ============================================================
    # Experiment 3: gradient-based suppression attack
    # ============================================================
    print("\n=== Experiment 3: gradient-based suppression attack ===",
          flush=True)
    print(f"  baseline L(A, A) = 1.000, L(A, indep) = {L_indep:+.4f}",
          flush=True)
    exp3 = []
    # Sweep lambda_utility from very low (adversary unconstrained -> may
    # destroy utility) to high (adversary heavily constrained -> can't
    # move).
    lambda_grid = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    for lam in lambda_grid:
        print(f"\n  lambda_utility = {lam}", flush=True)
        result = gradient_attack(A, X_eval, y_eval, tau_s, ref_Ms,
                                  lambda_utility=lam, n_steps=200, lr=1e-3,
                                  verbose=True)
        exp3.append(result)
    results['exp3_gradient_attack'] = {
        'description': 'Adversary trains a perturbation Delta to minimize'
                       ' the lineage cosine objective while a utility-drift'
                       ' penalty discourages function change. Sweeping the'
                       ' tradeoff parameter lambda traces the Pareto frontier'
                       ' between utility-preservation and lineage-suppression.',
        'lambda_grid':  lambda_grid,
        'runs':         exp3,
    }

    results['total_seconds'] = time.time() - t0
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    # === Headline ==========================================================
    print("\n=== HEADLINE ===")
    e1 = results['exp1_orthogonal_rotation']
    e2 = results['exp2_hidden_permutation']
    print(f"  Experiment 1 (orthogonal rotation, 5 seeds):")
    print(f"     mean L = {e1['mean_lineage']:+.6f}   min L = {e1['min_lineage']:+.6f}")
    print(f"     output drifts (ReLU breaks general rotation):")
    for r in e1['runs']:
        print(f"        seed {r['seed']}: out_drift = {r['output_drift']:.4e}")
    print(f"  Experiment 2 (hidden-unit permutation, 5 seeds):")
    print(f"     mean L = {e2['mean_lineage']:+.6f}   min L = {e2['min_lineage']:+.6f}")
    print(f"     output drifts (P commutes with ReLU, should be ~0):")
    for r in e2['runs']:
        print(f"        seed {r['seed']}: out_drift = {r['output_drift']:.4e}")
    print(f"  Experiment 3 (gradient suppression, lambda sweep):")
    print(f"   lambda  final_L  utility_drift  eval_y_loss")
    for r in exp3:
        print(f"   {r['lambda_utility']:>7.3f}  {r['final_lineage']:+.4f}  "
              f"{r['utility_drift']:.4e}    {r['eval_y_loss']:.4f}")


if __name__ == '__main__':
    main()
