"""
GPT-2 MLP Layer Pairing Experiment.

Tests whether the diagonal-dominance pairing signal from trained ResNets
extends to the MLP sublayers of pretrained GPT-2 transformers.

Hypothesis: GPT-2 MLP blocks have the same structure as ResNet blocks:
  ResNet:  x + W_out · ReLU(W_in · x)
  GPT-2:   x + W_2 · GELU(W_1 · x)

If dynamic isometry holds, W_2 @ W_1 ≈ -εI + E and diagonal-dominance
pairing should recover correct layer assignments.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# Ensure output directories exist
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


def extract_mlp_weights(model_name):
    """Extract MLP W_1 (c_fc) and W_2 (c_proj) from GPT-2."""
    from transformers import GPT2LMHeadModel

    print(f"  Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    W1s, W2s = [], []
    for block in model.transformer.h:
        # HuggingFace Conv1D stores weights as (in_features, out_features)
        # c_fc: input d_model -> output 4*d_model, stored as (d_model, 4*d_model)
        # c_proj: input 4*d_model -> output d_model, stored as (4*d_model, d_model)
        # For matrix multiply we need:
        #   W_1: (4*d_model, d_model) to expand
        #   W_2: (d_model, 4*d_model) to contract
        W1 = block.mlp.c_fc.weight.detach().cpu().numpy().T    # (4*d, d)
        W2 = block.mlp.c_proj.weight.detach().cpu().numpy().T  # (d, 4*d)
        W1s.append(W1)
        W2s.append(W2)

    return W1s, W2s, model.config


def diag_dominance_matrix(W1s, W2s):
    """Compute d(i,j) = |tr(W2[j] @ W1[i])| / ||W2[j] @ W1[i]||_F"""
    n = len(W1s)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prod = W2s[j] @ W1s[i]  # (d_model, d_model)
            tr = abs(np.trace(prod))
            fr = np.linalg.norm(prod, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def frobenius_cost_matrix(W1s, W2s):
    """Baseline: pair by minimizing ||W2[j] @ W1[i]||_F"""
    n = len(W1s)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.linalg.norm(W2s[j] @ W1s[i], 'fro')
    return M


def sv_distance_matrix(W1s, W2s):
    """Baseline: match by L1 distance on sorted singular values."""
    def sv_signature(W):
        return np.sort(np.linalg.svd(W, compute_uv=False))[::-1]

    sig1 = [sv_signature(W) for W in W1s]
    sig2 = [sv_signature(W) for W in W2s]

    n = len(W1s)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            min_len = min(len(sig1[i]), len(sig2[j]))
            M[i, j] = np.abs(sig1[i][:min_len] - sig2[j][:min_len]).sum()
    return M


def evaluate_pairing(M, minimize=False):
    """
    Run Hungarian matching and compute accuracy (ground truth = identity).
    """
    n = M.shape[0]
    if minimize:
        _, col_ind = linear_sum_assignment(M)
    else:
        _, col_ind = linear_sum_assignment(-M)

    pair_acc = float((col_ind == np.arange(n)).mean())

    diag = np.diag(M)
    off_diag = M[~np.eye(n, dtype=bool)]

    if minimize:
        sep = float(off_diag.min() - diag.max())
    else:
        sep = float(diag.min() - off_diag.max())

    return {
        'pair_acc': pair_acc,
        'pair_sep': sep,
        'mean_correct': float(diag.mean()),
        'mean_incorrect': float(off_diag.mean()),
        'correct_pairs': int(pair_acc * n),
        'total_pairs': n,
        'assignment': col_ind.tolist()
    }


def analyze_trace_signs(W1s, W2s):
    """
    Check if tr(W2[i] @ W1[i]) < 0 for correct pairs.
    Dynamic isometry predicts negative traces.
    """
    traces = []
    frob_norms = []
    for i in range(len(W1s)):
        prod = W2s[i] @ W1s[i]
        traces.append(np.trace(prod))
        frob_norms.append(np.linalg.norm(prod, 'fro'))
    traces = np.array(traces)
    frob_norms = np.array(frob_norms)

    return {
        'traces': traces.tolist(),
        'mean_trace': float(traces.mean()),
        'std_trace': float(traces.std()),
        'frac_negative': float((traces < 0).mean()),
        'min_trace': float(traces.min()),
        'max_trace': float(traces.max()),
        'mean_frob': float(frob_norms.mean()),
        'epsilon': float(np.abs(traces).mean() / (W1s[0].shape[1])),  # |tr|/d_model
    }


def run_experiment(model_name):
    """Run the full pairing experiment on a GPT-2 model."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print('='*60)

    W1s, W2s, config = extract_mlp_weights(model_name)
    n_layers = len(W1s)
    d_model = config.n_embd

    print(f"  Layers: {n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  W1 shape: {W1s[0].shape} (4*d_model, d_model)")
    print(f"  W2 shape: {W2s[0].shape} (d_model, 4*d_model)")
    print(f"  Product W2@W1 shape: ({d_model}, {d_model})")

    # Compute metrics
    print("\n  Computing diagonal-dominance matrix...")
    M_diag = diag_dominance_matrix(W1s, W2s)

    print("  Computing Frobenius cost matrix...")
    M_frob = frobenius_cost_matrix(W1s, W2s)

    print("  Computing SV distance matrix...")
    M_sv = sv_distance_matrix(W1s, W2s)

    # Evaluate pairing methods
    results = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'diag_dominance': evaluate_pairing(M_diag, minimize=False),
        'frobenius': evaluate_pairing(M_frob, minimize=True),
        'sv_distance': evaluate_pairing(M_sv, minimize=True),
        'trace_analysis': analyze_trace_signs(W1s, W2s),
        'random_baseline': {'expected_acc': 1.0 / n_layers}
    }

    # Print summary
    print(f"\n  === RESULTS ===")
    print(f"  Diagonal Dominance: {results['diag_dominance']['correct_pairs']}/{n_layers} correct ({results['diag_dominance']['pair_acc']:.1%})")
    print(f"  Frobenius:          {results['frobenius']['correct_pairs']}/{n_layers} correct ({results['frobenius']['pair_acc']:.1%})")
    print(f"  SV Distance:        {results['sv_distance']['correct_pairs']}/{n_layers} correct ({results['sv_distance']['pair_acc']:.1%})")
    print(f"  Random baseline:    {1.0/n_layers:.1%} expected accuracy")

    print(f"\n  === TRACE ANALYSIS (Dynamic Isometry Check) ===")
    ta = results['trace_analysis']
    print(f"  Mean trace:       {ta['mean_trace']:.2f}")
    print(f"  Std trace:        {ta['std_trace']:.2f}")
    print(f"  Frac negative:    {ta['frac_negative']:.1%}")
    print(f"  ε = |tr|/d:       {ta['epsilon']:.4f}")

    print(f"\n  === SEPARATION ===")
    print(f"  Diag dom separation:  {results['diag_dominance']['pair_sep']:.4f}")
    print(f"  Mean correct:         {results['diag_dominance']['mean_correct']:.4f}")
    print(f"  Mean incorrect:       {results['diag_dominance']['mean_incorrect']:.4f}")

    return results, M_diag, M_frob, M_sv


def plot_results(all_results, all_matrices):
    """Create comparison figure for all models."""
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 4.5*n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (results, (M_diag, M_frob, M_sv)) in enumerate(zip(all_results, all_matrices)):
        model_name = results['model']
        n = results['n_layers']

        # Left: Diagonal dominance matrix
        ax = axes[idx, 0]
        im = ax.imshow(M_diag, cmap='magma', aspect='equal')
        ax.set_title(f'{model_name} - Diagonal Dominance Matrix\n'
                    f'Pair Accuracy: {results["diag_dominance"]["pair_acc"]:.1%}')
        ax.set_xlabel('W₂ index (output projection)')
        ax.set_ylabel('W₁ index (input projection)')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Middle: Trace values per layer
        ax = axes[idx, 1]
        traces = results['trace_analysis']['traces']
        colors = ['green' if t < 0 else 'red' for t in traces]
        ax.bar(range(n), traces, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Layer index')
        ax.set_ylabel('tr(W₂ @ W₁)')
        ax.set_title(f'{model_name} - Trace Values\n'
                    f'{results["trace_analysis"]["frac_negative"]:.0%} negative (green)')

        # Right: Method comparison
        ax = axes[idx, 2]
        methods = ['Diag\nDom', 'Frob', 'SV\nDist', 'Random']
        accs = [
            results['diag_dominance']['pair_acc'],
            results['frobenius']['pair_acc'],
            results['sv_distance']['pair_acc'],
            results['random_baseline']['expected_acc']
        ]
        bar_colors = ['green' if a > 0.9 else 'orange' if a > 0.5 else 'red' for a in accs[:3]] + ['gray']
        bars = ax.bar(methods, accs, color=bar_colors, alpha=0.8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Pair Accuracy')
        ax.set_title(f'{model_name} - Method Comparison')
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)

        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                   f'{v:.0%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig_gpt2_mlp_pairing.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_gpt2_mlp_pairing.pdf', bbox_inches='tight')
    print('\nSaved figures/fig_gpt2_mlp_pairing.{png,pdf}')


def run_random_init_baseline(model_name='gpt2'):
    """Compare against randomly initialized GPT-2 (should fail)."""
    from transformers import GPT2LMHeadModel, GPT2Config

    print(f"\n{'='*60}")
    print(f"BASELINE: Randomly initialized {model_name}")
    print('='*60)

    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel(config)  # Random init, no pretrained weights

    W1s, W2s = [], []
    for block in model.transformer.h:
        W1 = block.mlp.c_fc.weight.detach().cpu().numpy().T
        W2 = block.mlp.c_proj.weight.detach().cpu().numpy().T
        W1s.append(W1)
        W2s.append(W2)

    n = len(W1s)
    M_diag = diag_dominance_matrix(W1s, W2s)
    result = evaluate_pairing(M_diag, minimize=False)
    trace_info = analyze_trace_signs(W1s, W2s)

    print(f"  Pair accuracy: {result['correct_pairs']}/{n} ({result['pair_acc']:.1%})")
    print(f"  Expected random: {1/n:.1%}")
    print(f"  Frac negative traces: {trace_info['frac_negative']:.1%}")
    print(f"  Mean trace: {trace_info['mean_trace']:.2f}")

    return {
        'model': f'{model_name}-random-init',
        'pair_acc': result['pair_acc'],
        'frac_negative': trace_info['frac_negative'],
        'mean_trace': trace_info['mean_trace']
    }


if __name__ == '__main__':
    MODELS = ['gpt2', 'gpt2-medium']

    all_results = []
    all_matrices = []

    for model_name in MODELS:
        results, M_diag, M_frob, M_sv = run_experiment(model_name)
        all_results.append(results)
        all_matrices.append((M_diag, M_frob, M_sv))

    # Run random init baseline
    random_baseline = run_random_init_baseline('gpt2')

    # Save results
    output = {
        'pretrained': all_results,
        'random_init_baseline': random_baseline
    }
    with open('results/gpt2_mlp_pairing.json', 'w') as f:
        json.dump(output, f, indent=2)
    print('\nSaved results/gpt2_mlp_pairing.json')

    # Create visualization
    plot_results(all_results, all_matrices)

    # Print final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"\n{r['model']}:")
        print(f"  Diagonal-dominance pairing: {r['diag_dominance']['correct_pairs']}/{r['n_layers']} ({r['diag_dominance']['pair_acc']:.1%})")
        print(f"  Dynamic isometry signal: {r['trace_analysis']['frac_negative']:.0%} negative traces")
