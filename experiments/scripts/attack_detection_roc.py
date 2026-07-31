#!/usr/bin/env python3
"""Attack Detection ROC Analysis - MLP Benchmark.

Formalizes the attack detection test: can we distinguish adversarially-attacked
models from genuinely unrelated models based on forensic traces?

Key insight: The gradient attack that erases lineage (minimizes L) leaves traces:
- Attacked models have LOW diagonal dominance (dd ≈ 0.15 vs ≈ 0.28)
- Attacked models have HIGH ||E|| norm (≈ 3.4 vs ≈ 1.8)

This script:
1. Generates a population of related, unrelated, and attacked models
2. Computes detection metrics (dd, ||E||, ratio)
3. Trains classifiers (ratio threshold, logistic regression)
4. Reports ROC analysis and visualizations

Output: results/attack_detection_roc.json, figures/attack_detection_roc.pdf
"""

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lineage_phase1_mlp import (Block, ResNet, make_data, train_model,
                                 fresh_model, eval_loss, branch_products)
import lineage_detection as ldet


DEPTH, HIDDEN, IN_DIM = 24, 64, 24
EPOCHS_ROOT = 200
ATTACK_STEPS = 200
ATTACK_LR = 1e-3


def diagonal_dominance_scores(model):
    """Return list of s(M_l) = |tr(M)|/||M||_F for each block."""
    scores = []
    for blk in model.blocks:
        W_in = blk.inp.weight.detach()
        W_out = blk.out.weight.detach()
        M = W_out @ W_in
        tr = torch.trace(M).abs()
        frob = torch.norm(M, 'fro')
        scores.append(float(tr / frob))
    return scores


def centered_signature_norms(model):
    """Return ||E_l||_F for each block (the traceless remainder)."""
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


def compute_metrics(model, name):
    """Compute all detection metrics for a model."""
    dd_scores = diagonal_dominance_scores(model)
    e_norms = centered_signature_norms(model)
    dd_mean = float(np.mean(dd_scores))
    e_mean = float(np.mean(e_norms))
    ratio = e_mean / (dd_mean + 1e-8)
    return {
        'name': name,
        'dd_mean': dd_mean,
        'dd_std': float(np.std(dd_scores)),
        'E_norm_mean': e_mean,
        'E_norm_std': float(np.std(e_norms)),
        'ratio': ratio,
    }


def gradient_attack(reference, X_eval, lambda_utility, n_steps=ATTACK_STEPS, lr=ATTACK_LR):
    """Run gradient attack to minimize lineage score."""
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


def process_reference(args):
    """Process one reference model - generate related, unrelated, attacked."""
    ref_seed, X_train, y_train, X_eval, y_eval, lambda_values = args

    results = {'ref_seed': ref_seed, 'related': [], 'unrelated': [], 'attacked': []}

    # Train reference
    ref = fresh_model(DEPTH, HIDDEN, IN_DIM, seed=ref_seed)
    train_model(ref, X_train, y_train, epochs=EPOCHS_ROOT)
    ref_Ms = branch_products(ref)
    tau_s = ldet.choose_tau_s([ref_Ms])

    ref_metrics = compute_metrics(ref, f'ref_{ref_seed}')
    L_ref, _, _ = ldet.lineage_score(ref_Ms, ref_Ms, tau_s)
    ref_metrics['lineage'] = L_ref
    results['reference'] = ref_metrics

    # Related: fine-tune variants
    for ft_seed in range(5):
        ft_model = copy.deepcopy(ref)
        train_model(ft_model, X_train, y_train, epochs=20, lr=1e-4)
        metrics = compute_metrics(ft_model, f'finetune_{ft_seed}')
        L, _, _ = ldet.lineage_score(ref_Ms, branch_products(ft_model), tau_s)
        metrics['lineage'] = L
        metrics['class'] = 'related'
        results['related'].append(metrics)

    # Related: noise variants
    for noise_seed in range(5):
        noise_model = copy.deepcopy(ref)
        torch.manual_seed(noise_seed + 1000)
        with torch.no_grad():
            for p in noise_model.parameters():
                p.add_(torch.randn_like(p) * 0.01 * p.abs().mean())
        metrics = compute_metrics(noise_model, f'noise_{noise_seed}')
        L, _, _ = ldet.lineage_score(ref_Ms, branch_products(noise_model), tau_s)
        metrics['lineage'] = L
        metrics['class'] = 'related'
        results['related'].append(metrics)

    # Unrelated: independent training
    for ind_seed in range(10):
        ind_model = fresh_model(DEPTH, HIDDEN, IN_DIM, seed=500 + ref_seed * 100 + ind_seed)
        train_model(ind_model, X_train, y_train, epochs=EPOCHS_ROOT)
        metrics = compute_metrics(ind_model, f'independent_{ind_seed}')
        L, _, _ = ldet.lineage_score(ref_Ms, branch_products(ind_model), tau_s)
        metrics['lineage'] = L
        metrics['class'] = 'unrelated'
        results['unrelated'].append(metrics)

    # Attacked: gradient attack at various lambda
    for lam in lambda_values:
        for att_seed in range(2):
            torch.manual_seed(att_seed + 2000)
            attacked = gradient_attack(ref, X_eval, lambda_utility=lam)
            metrics = compute_metrics(attacked, f'attacked_lam{lam}_s{att_seed}')
            L, _, _ = ldet.lineage_score(ref_Ms, branch_products(attacked), tau_s)
            metrics['lineage'] = L
            metrics['lambda'] = lam
            metrics['class'] = 'attacked'
            results['attacked'].append(metrics)

    return results


def compute_roc_metrics(y_true, y_scores):
    """Compute ROC metrics including TPR@1%FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)

    # TPR at 1% FPR
    idx = np.searchsorted(fpr, 0.01)
    tpr_at_1pct_fpr = tpr[min(idx, len(tpr)-1)]

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_threshold = thresholds[opt_idx]

    return {
        'auroc': auroc,
        'tpr_at_1pct_fpr': tpr_at_1pct_fpr,
        'optimal_threshold': float(opt_threshold),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }


def main():
    t0 = time.time()
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('figures').mkdir(parents=True, exist_ok=True)

    # Generate data once
    X, y = make_data(in_dim=IN_DIM, n=4000, seed=0, target_key=42)
    X_eval, y_eval = X[:1000], y[:1000]
    X_train, y_train = X[1000:], y[1000:]

    # Lambda values for attack
    lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5]

    # Reference seeds
    ref_seeds = [100, 101, 102, 103, 104]

    print(f"Running attack detection ROC analysis...")
    print(f"  {len(ref_seeds)} reference models")
    print(f"  {len(lambda_values)} attack lambda values")
    print(f"  Parallelizing with ProcessPoolExecutor\n")

    # Prepare args for parallel execution
    args_list = [
        (seed, X_train, y_train, X_eval, y_eval, lambda_values)
        for seed in ref_seeds
    ]

    all_results = []
    with ProcessPoolExecutor(max_workers=min(len(ref_seeds), 4)) as executor:
        futures = {executor.submit(process_reference, args): args[0] for args in args_list}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"  Completed ref_seed={seed}")
            except Exception as e:
                print(f"  Failed ref_seed={seed}: {e}")

    # Aggregate all samples
    related_samples = []
    unrelated_samples = []
    attacked_samples = []

    for res in all_results:
        related_samples.extend(res['related'])
        unrelated_samples.extend(res['unrelated'])
        attacked_samples.extend(res['attacked'])

    print(f"\nTotal samples:")
    print(f"  Related: {len(related_samples)}")
    print(f"  Unrelated: {len(unrelated_samples)}")
    print(f"  Attacked: {len(attacked_samples)}")

    # Build feature matrix for classification
    all_samples = related_samples + unrelated_samples + attacked_samples
    X_features = np.array([[s['dd_mean'], s['E_norm_mean']] for s in all_samples])
    y_labels = np.array([0] * len(related_samples) + [0] * len(unrelated_samples) +
                        [1] * len(attacked_samples))  # 1 = attacked

    # Method 1: Ratio-based detection
    ratios = np.array([s['ratio'] for s in all_samples])
    ratio_roc = compute_roc_metrics(y_labels, ratios)

    # Method 2: Logistic regression on (dd, E_norm)
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_features, y_labels)
    logreg_probs = logreg.predict_proba(X_features)[:, 1]
    logreg_roc = compute_roc_metrics(y_labels, logreg_probs)

    print(f"\n{'='*60}")
    print("ATTACK DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"\nMethod 1: Ratio (||E|| / dd)")
    print(f"  AUROC: {ratio_roc['auroc']:.4f}")
    print(f"  TPR@1%FPR: {ratio_roc['tpr_at_1pct_fpr']:.4f}")
    print(f"  Optimal threshold: {ratio_roc['optimal_threshold']:.2f}")

    print(f"\nMethod 2: Logistic Regression on (dd, ||E||)")
    print(f"  AUROC: {logreg_roc['auroc']:.4f}")
    print(f"  TPR@1%FPR: {logreg_roc['tpr_at_1pct_fpr']:.4f}")
    print(f"  Coefficients: dd={logreg.coef_[0][0]:.3f}, E={logreg.coef_[0][1]:.3f}")

    # Summary stats by class
    print(f"\n{'='*60}")
    print("CLASS STATISTICS")
    print(f"{'='*60}")
    for name, samples in [('Related', related_samples), ('Unrelated', unrelated_samples),
                          ('Attacked', attacked_samples)]:
        dd_vals = [s['dd_mean'] for s in samples]
        e_vals = [s['E_norm_mean'] for s in samples]
        ratio_vals = [s['ratio'] for s in samples]
        print(f"\n{name}:")
        print(f"  dd_mean:  {np.mean(dd_vals):.3f} ± {np.std(dd_vals):.3f}")
        print(f"  ||E||:    {np.mean(e_vals):.3f} ± {np.std(e_vals):.3f}")
        print(f"  ratio:    {np.mean(ratio_vals):.2f} ± {np.std(ratio_vals):.2f}")

    # Per-lambda breakdown for attacked
    print(f"\n{'='*60}")
    print("ATTACKED BY LAMBDA")
    print(f"{'='*60}")
    for lam in lambda_values:
        lam_samples = [s for s in attacked_samples if s.get('lambda') == lam]
        if lam_samples:
            dd_vals = [s['dd_mean'] for s in lam_samples]
            e_vals = [s['E_norm_mean'] for s in lam_samples]
            L_vals = [s['lineage'] for s in lam_samples]
            print(f"\nλ={lam}:")
            print(f"  lineage L: {np.mean(L_vals):.4f} ± {np.std(L_vals):.4f}")
            print(f"  dd_mean:   {np.mean(dd_vals):.3f} ± {np.std(dd_vals):.3f}")
            print(f"  ||E||:     {np.mean(e_vals):.3f} ± {np.std(e_vals):.3f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: ROC curves
    ax = axes[0]
    ax.plot(ratio_roc['fpr'], ratio_roc['tpr'], 'b-', linewidth=2,
            label=f"Ratio (AUROC={ratio_roc['auroc']:.3f})")
    ax.plot(logreg_roc['fpr'], logreg_roc['tpr'], 'r--', linewidth=2,
            label=f"LogReg (AUROC={logreg_roc['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1)
    ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, label='1% FPR')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Attack Detection ROC')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Panel 2: Scatter plot (dd vs E_norm)
    ax = axes[1]
    colors = {'related': 'green', 'unrelated': 'blue', 'attacked': 'red'}
    for name, samples in [('related', related_samples), ('unrelated', unrelated_samples),
                          ('attacked', attacked_samples)]:
        dd_vals = [s['dd_mean'] for s in samples]
        e_vals = [s['E_norm_mean'] for s in samples]
        ax.scatter(dd_vals, e_vals, c=colors[name], alpha=0.6, label=name.capitalize(), s=50)
    ax.set_xlabel('Diagonal Dominance (dd)')
    ax.set_ylabel('Signature Norm ||E||')
    ax.set_title('Feature Space')
    ax.legend()

    # Panel 3: Per-lambda attacked scatter
    ax = axes[2]
    cmap = plt.cm.viridis
    for i, lam in enumerate(lambda_values):
        lam_samples = [s for s in attacked_samples if s.get('lambda') == lam]
        if lam_samples:
            dd_vals = [s['dd_mean'] for s in lam_samples]
            e_vals = [s['E_norm_mean'] for s in lam_samples]
            color = cmap(i / len(lambda_values))
            ax.scatter(dd_vals, e_vals, c=[color], alpha=0.7,
                      label=f'λ={lam}', s=80, marker='x')
    # Add unrelated for reference
    dd_vals = [s['dd_mean'] for s in unrelated_samples]
    e_vals = [s['E_norm_mean'] for s in unrelated_samples]
    ax.scatter(dd_vals, e_vals, c='blue', alpha=0.3, label='Unrelated', s=30)
    ax.set_xlabel('Diagonal Dominance (dd)')
    ax.set_ylabel('Signature Norm ||E||')
    ax.set_title('Attacked by λ vs Unrelated')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = Path('figures/attack_detection_roc.pdf')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {fig_path}")

    # Save results
    results = {
        'config': {
            'depth': DEPTH, 'hidden': HIDDEN, 'in_dim': IN_DIM,
            'epochs_root': EPOCHS_ROOT, 'attack_steps': ATTACK_STEPS,
            'lambda_values': lambda_values, 'ref_seeds': ref_seeds,
        },
        'counts': {
            'related': len(related_samples),
            'unrelated': len(unrelated_samples),
            'attacked': len(attacked_samples),
        },
        'ratio_roc': {
            'auroc': ratio_roc['auroc'],
            'tpr_at_1pct_fpr': ratio_roc['tpr_at_1pct_fpr'],
            'optimal_threshold': ratio_roc['optimal_threshold'],
        },
        'logreg_roc': {
            'auroc': logreg_roc['auroc'],
            'tpr_at_1pct_fpr': logreg_roc['tpr_at_1pct_fpr'],
            'coefficients': {'dd': float(logreg.coef_[0][0]),
                           'E_norm': float(logreg.coef_[0][1])},
        },
        'class_stats': {
            'related': {
                'dd_mean': float(np.mean([s['dd_mean'] for s in related_samples])),
                'dd_std': float(np.std([s['dd_mean'] for s in related_samples])),
                'E_norm_mean': float(np.mean([s['E_norm_mean'] for s in related_samples])),
                'E_norm_std': float(np.std([s['E_norm_mean'] for s in related_samples])),
            },
            'unrelated': {
                'dd_mean': float(np.mean([s['dd_mean'] for s in unrelated_samples])),
                'dd_std': float(np.std([s['dd_mean'] for s in unrelated_samples])),
                'E_norm_mean': float(np.mean([s['E_norm_mean'] for s in unrelated_samples])),
                'E_norm_std': float(np.std([s['E_norm_mean'] for s in unrelated_samples])),
            },
            'attacked': {
                'dd_mean': float(np.mean([s['dd_mean'] for s in attacked_samples])),
                'dd_std': float(np.std([s['dd_mean'] for s in attacked_samples])),
                'E_norm_mean': float(np.mean([s['E_norm_mean'] for s in attacked_samples])),
                'E_norm_std': float(np.std([s['E_norm_mean'] for s in attacked_samples])),
            },
        },
        'per_lambda': {},
        'all_samples': all_samples,
        'elapsed_seconds': time.time() - t0,
    }

    for lam in lambda_values:
        lam_samples = [s for s in attacked_samples if s.get('lambda') == lam]
        if lam_samples:
            results['per_lambda'][str(lam)] = {
                'lineage_mean': float(np.mean([s['lineage'] for s in lam_samples])),
                'dd_mean': float(np.mean([s['dd_mean'] for s in lam_samples])),
                'E_norm_mean': float(np.mean([s['E_norm_mean'] for s in lam_samples])),
            }

    out_path = Path('results/attack_detection_roc.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {out_path}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
