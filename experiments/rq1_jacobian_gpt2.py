"""
RQ1 P0: Jacobian Orthogonality Verification (GPT-2)

Validates that the diagonal dominance mechanism (M ≈ -εI → J ≈ orthogonal)
operates in real pretrained transformers, not just controlled toy experiments.

For each MLP block in GPT-2-small:
  M = W_proj @ W_fc  (768×768)
  J = I + M
  δ_J = ||J^T J - I||_F / √768  (orthogonality deviation)
  s = |tr(M)| / ||M||_F         (diagonal dominance)

Compares pretrained (should have low δ_J, high s) vs random-init (control).

Outputs:
  results/rq1_jacobian_gpt2.json
  figures/fig_rq1_jacobian_gpt2.png
"""

import json
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def extract_mlp_products(model_name: str, pretrained: bool = True):
    """Extract M = W_proj @ W_fc for each MLP block."""
    from transformers import GPT2LMHeadModel, GPT2Config

    if pretrained:
        print(f"  Loading pretrained {model_name}...")
        model = GPT2LMHeadModel.from_pretrained(model_name, low_cpu_mem_usage=True)
    else:
        print(f"  Loading random-init {model_name}...")
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModel(config)

    model.eval()
    d_model = model.config.n_embd

    Ms = []
    for block in model.transformer.h:
        W1 = block.mlp.c_fc.weight.detach().float().cpu().numpy().T    # (4d, d)
        W2 = block.mlp.c_proj.weight.detach().float().cpu().numpy().T  # (d, 4d)
        M = W2 @ W1  # (d, d)
        Ms.append(M)

    del model
    gc.collect()
    return Ms, d_model


def jacobian_orthogonality_deviation(M: np.ndarray) -> float:
    """Compute δ_J = ||J^T J - I||_F / √d where J = I + M."""
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    JTJ = J.T @ J
    deviation = np.linalg.norm(JTJ - I, 'fro') / np.sqrt(d)
    return float(deviation)


def diagonal_dominance(M: np.ndarray, eps: float = 1e-12) -> float:
    """Compute s = |tr(M)| / ||M||_F."""
    tr = np.trace(M)
    frob = np.linalg.norm(M, 'fro') + eps
    return float(abs(tr) / frob)


def analyze_model(model_name: str, pretrained: bool = True):
    """Analyze all MLP blocks in a model."""
    Ms, d_model = extract_mlp_products(model_name, pretrained=pretrained)

    results = []
    for i, M in enumerate(Ms):
        tr = float(np.trace(M))
        results.append({
            'layer': i,
            'trace': tr,
            'trace_negative': tr < 0,
            'diag_dominance_s': diagonal_dominance(M),
            'jacobian_deviation': jacobian_orthogonality_deviation(M),
            'd_model': d_model,
        })

    return results


def main():
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    model_name = 'gpt2'  # GPT-2-small (124M, 12 layers, d=768)

    print("=" * 60)
    print("RQ1: Jacobian Orthogonality Verification (GPT-2)")
    print("=" * 60)

    # Analyze pretrained
    print("\n[1/2] Analyzing pretrained GPT-2...")
    pretrained_results = analyze_model(model_name, pretrained=True)

    # Analyze random-init
    print("\n[2/2] Analyzing random-init GPT-2...")
    random_results = analyze_model(model_name, pretrained=False)

    # Aggregate stats
    def aggregate(results):
        return {
            'mean_s': float(np.mean([r['diag_dominance_s'] for r in results])),
            'mean_delta_J': float(np.mean([r['jacobian_deviation'] for r in results])),
            'frac_neg_trace': float(np.mean([r['trace_negative'] for r in results])),
            'mean_trace': float(np.mean([r['trace'] for r in results])),
        }

    pretrained_agg = aggregate(pretrained_results)
    random_agg = aggregate(random_results)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nPretrained GPT-2:")
    print(f"  Mean s(i,i):     {pretrained_agg['mean_s']:.4f}")
    print(f"  Mean δ_J:        {pretrained_agg['mean_delta_J']:.4f}")
    print(f"  Frac neg trace:  {pretrained_agg['frac_neg_trace']:.1%}")
    print(f"  Mean trace:      {pretrained_agg['mean_trace']:.2f}")

    print("\nRandom-init GPT-2:")
    print(f"  Mean s(i,i):     {random_agg['mean_s']:.4f}")
    print(f"  Mean δ_J:        {random_agg['mean_delta_J']:.4f}")
    print(f"  Frac neg trace:  {random_agg['frac_neg_trace']:.1%}")
    print(f"  Mean trace:      {random_agg['mean_trace']:.2f}")

    print("\nPer-layer details:")
    print(f"{'Layer':>5} | {'Pre s':>8} | {'Pre δ_J':>8} | {'Pre tr':>10} | {'Rand s':>8} | {'Rand δ_J':>8}")
    print("-" * 65)
    for p, r in zip(pretrained_results, random_results):
        print(f"{p['layer']:>5} | {p['diag_dominance_s']:>8.4f} | {p['jacobian_deviation']:>8.4f} | "
              f"{p['trace']:>10.2f} | {r['diag_dominance_s']:>8.4f} | {r['jacobian_deviation']:>8.4f}")

    # Save JSON
    output = {
        'model': model_name,
        'd_model': pretrained_results[0]['d_model'],
        'n_layers': len(pretrained_results),
        'pretrained': {
            'per_layer': pretrained_results,
            'aggregate': pretrained_agg,
        },
        'random_init': {
            'per_layer': random_results,
            'aggregate': random_agg,
        },
    }

    with open('results/rq1_jacobian_gpt2.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved results/rq1_jacobian_gpt2.json")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    layers = [r['layer'] for r in pretrained_results]
    pre_delta = [r['jacobian_deviation'] for r in pretrained_results]
    rand_delta = [r['jacobian_deviation'] for r in random_results]
    pre_s = [r['diag_dominance_s'] for r in pretrained_results]
    rand_s = [r['diag_dominance_s'] for r in random_results]

    # Left: δ_J comparison
    x = np.arange(len(layers))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, pre_delta, width, label='Pretrained', color='steelblue')
    ax.bar(x + width/2, rand_delta, width, label='Random-init', color='salmon')
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Jacobian deviation $\delta_J = \|J^T J - I\|_F / \sqrt{d}$')
    ax.set_title('Jacobian Orthogonality: Pretrained vs Random-init')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.axhline(0, color='k', linewidth=0.5)

    # Right: s comparison
    ax = axes[1]
    ax.bar(x - width/2, pre_s, width, label='Pretrained', color='steelblue')
    ax.bar(x + width/2, rand_s, width, label='Random-init', color='salmon')
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Diagonal dominance $s = |tr(M)| / \|M\|_F$')
    ax.set_title('Diagonal Dominance: Pretrained vs Random-init')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/fig_rq1_jacobian_gpt2.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/fig_rq1_jacobian_gpt2.pdf', bbox_inches='tight')
    print("Saved figures/fig_rq1_jacobian_gpt2.{png,pdf}")


if __name__ == '__main__':
    main()
