"""RQ2 Pending Experiment: LLaMA-2 Scaling Analysis (7B → 13B).

Extends the GPT-2 d^0.87 scaling fit to larger models. Tests whether the
fingerprint scaling exponent holds at d = 5120 (LLaMA-2-13B).

From RQ.md:
  - GPT-2 scaling: mean s ∝ d^0.87 (faster than theoretical √d)
  - LLaMA-2-13B has d = 5120, which would extend the fit beyond d = 1600 (GPT-2-XL)

Metrics computed:
  - s = |tr(M)| / ||M||_F  (diagonal dominance)
  - δ_J^norm = ||J^T J / ||J||_F^2 - I/d||_F  (normalized Jacobian deviation)
  - Pair accuracy via Hungarian matching

Resource estimate: ~30 GB GPU memory for LLaMA-2-13B in fp16.

Outputs:
  results/rq2_llama_scaling.json
  figures/fig_rq2_llama_scaling.png
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np


def diag_dom(M):
    return float(abs(np.trace(M)) / (np.linalg.norm(M, 'fro') + 1e-12))


def delta_J_abs(M):
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    return float(np.linalg.norm(J.T @ J - I, 'fro') / np.sqrt(d))


def delta_J_norm(M):
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    JTJ = J.T @ J
    fro2 = float(np.linalg.norm(J, 'fro') ** 2)
    return float(np.linalg.norm(JTJ / fro2 - I / d, 'fro'))


def extract_llama_mlp(model_name, use_safetensors=False):
    """Extract M = W_down @ W_up for each MLP block in LLaMA-like model."""
    import torch

    if use_safetensors:
        return extract_llama_safetensors(model_name)

    from transformers import AutoModelForCausalLM

    print(f"  Loading {model_name} (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    model.eval()
    cfg = model.config

    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    print(f"  Config: {n_layers} layers, d={d_model}")

    base = model.model if hasattr(model, 'model') else model

    Ms = []
    for i, layer in enumerate(base.layers):
        W_up = layer.mlp.up_proj.weight.detach().float().cpu().numpy()
        W_down = layer.mlp.down_proj.weight.detach().float().cpu().numpy()
        M = W_down @ W_up
        Ms.append(M)

        base.layers[i] = None
        if (i + 1) % 8 == 0:
            gc.collect()
            print(f"    layer {i+1}/{n_layers}", flush=True)

    del model
    gc.collect()

    return Ms, d_model, n_layers


def extract_llama_safetensors(model_name, hf_cache_root=None):
    """Direct safetensors extraction for LLaMA models."""
    import os
    from pathlib import Path
    from safetensors import safe_open
    import torch

    if hf_cache_root is None:
        hf_cache_root = Path(os.environ.get('HF_HOME',
                                            Path.home() / '.cache' / 'huggingface')) / 'hub'
    else:
        hf_cache_root = Path(hf_cache_root)

    repo_dir = hf_cache_root / f"models--{model_name.replace('/', '--')}"
    snap_root = repo_dir / 'snapshots'
    if not snap_root.exists():
        raise FileNotFoundError(
            f"No HF cache at {snap_root}; pre-download with: "
            f"huggingface-cli download {model_name}")
    snap = next(snap_root.iterdir())

    cfg = json.loads((snap / 'config.json').read_text())
    n_layers = cfg['num_hidden_layers']
    d_model = cfg['hidden_size']

    print(f"  (safetensors-direct; snap={snap.name[:8]}...)")
    print(f"  Config: {n_layers} layers, d={d_model}")

    idx_file = snap / 'model.safetensors.index.json'
    if idx_file.exists():
        idx = json.loads(idx_file.read_text())['weight_map']
    else:
        single = snap / 'model.safetensors'
        with safe_open(str(single), framework='pt') as f:
            idx = {k: 'model.safetensors' for k in f.keys()}

    shard_to_keys = {}
    for k, shard in idx.items():
        shard_to_keys.setdefault(shard, []).append(k)

    W_up = [None] * n_layers
    W_down = [None] * n_layers

    def _to_fp32_np(t):
        return t.to(torch.float32).numpy()

    for shard, keys in shard_to_keys.items():
        wanted = [k for k in keys if 'mlp' in k and ('up_proj' in k or 'down_proj' in k)]
        if not wanted:
            continue

        with safe_open(str(snap / shard), framework='pt') as f:
            for k in wanted:
                parts = k.split('.')
                layer_idx = int(parts[2])
                if 'up_proj' in k:
                    W_up[layer_idx] = _to_fp32_np(f.get_tensor(k))
                elif 'down_proj' in k:
                    W_down[layer_idx] = _to_fp32_np(f.get_tensor(k))

        gc.collect()
        print(f"    processed {shard}", flush=True)

    Ms = []
    for i in range(n_layers):
        M = W_down[i] @ W_up[i]
        Ms.append(M)

    return Ms, d_model, n_layers


def analyze_model(model_name, use_safetensors=False):
    """Compute all metrics for a single model."""
    Ms, d_model, n_layers = extract_llama_mlp(model_name, use_safetensors)

    per_layer = []
    for i, M in enumerate(Ms):
        per_layer.append({
            'layer': i,
            's': diag_dom(M),
            'delta_J': delta_J_abs(M),
            'delta_J_norm': delta_J_norm(M),
            'trace': float(np.trace(M)),
        })

    return {
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'per_layer': per_layer,
        'mean_s': float(np.mean([r['s'] for r in per_layer])),
        'median_s': float(np.median([r['s'] for r in per_layer])),
        'mean_delta_J': float(np.mean([r['delta_J'] for r in per_layer])),
        'mean_delta_J_norm': float(np.mean([r['delta_J_norm'] for r in per_layer])),
        'median_delta_J_norm': float(np.median([r['delta_J_norm'] for r in per_layer])),
        'frac_neg_trace': float(np.mean([r['trace'] < 0 for r in per_layer])),
    }


def fit_scaling(results_list):
    """Fit power-law scaling: metric = a * d^b"""
    ds = np.array([r['d_model'] for r in results_list])

    fits = {}
    for key, label in [('mean_s', 's'), ('mean_delta_J_norm', 'delta_J_norm')]:
        ys = np.array([r[key] for r in results_list])
        b, log_a = np.polyfit(np.log(ds), np.log(ys), 1)
        fits[label] = {'exponent': float(b), 'coefficient': float(np.exp(log_a))}

    return fits


def create_figure(results, gpt2_results, out_path):
    """Create scaling comparison figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Combine GPT-2 and LLaMA results
    all_results = gpt2_results + results
    ds = [r['d_model'] for r in all_results]
    mean_s = [r['mean_s'] for r in all_results]
    mean_delta = [r['mean_delta_J_norm'] for r in all_results]

    # Color by family
    colors = ['steelblue'] * len(gpt2_results) + ['darkgreen'] * len(results)
    labels = [r['model'].split('/')[-1] for r in all_results]

    # 1. Mean s vs d (log-log)
    ax = axes[0]
    for i, (d, s, c, l) in enumerate(zip(ds, mean_s, colors, labels)):
        ax.scatter(d, s, c=c, s=100, zorder=3)
        ax.annotate(l, (d, s), textcoords='offset points', xytext=(5, 5), fontsize=8)

    # Fit line
    log_d = np.log(ds)
    log_s = np.log(mean_s)
    b, a = np.polyfit(log_d, log_s, 1)
    d_fit = np.linspace(min(ds) * 0.9, max(ds) * 1.1, 100)
    s_fit = np.exp(a) * d_fit ** b
    ax.plot(d_fit, s_fit, 'k--', alpha=0.5, label=f's ∝ d^{b:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model dimension (d)')
    ax.set_ylabel('Mean diagonal dominance (s)')
    ax.set_title('Fingerprint Scaling: s vs d')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean delta_J_norm vs d
    ax = axes[1]
    for i, (d, delta, c, l) in enumerate(zip(ds, mean_delta, colors, labels)):
        ax.scatter(d, delta, c=c, s=100, zorder=3)
        ax.annotate(l, (d, delta), textcoords='offset points', xytext=(5, 5), fontsize=8)

    log_delta = np.log(mean_delta)
    b_d, a_d = np.polyfit(log_d, log_delta, 1)
    delta_fit = np.exp(a_d) * d_fit ** b_d
    ax.plot(d_fit, delta_fit, 'k--', alpha=0.5, label=f'δ_J^norm ∝ d^{b_d:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model dimension (d)')
    ax.set_ylabel('Mean δ_J^norm')
    ax.set_title('Jacobian Deviation Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Per-layer s for each model
    ax = axes[2]
    for r in all_results:
        layers = [p['layer'] for p in r['per_layer']]
        s_vals = [p['s'] for p in r['per_layer']]
        label = r['model'].split('/')[-1]
        ax.plot(layers, s_vals, 'o-', markersize=3, label=label, alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Diagonal dominance (s)')
    ax.set_title('Per-Layer Fingerprint')
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved {out_path}")


def main():
    ap = argparse.ArgumentParser(description='LLaMA-2 Scaling Analysis (RQ2)')
    ap.add_argument('--models', nargs='+',
                    default=['NousResearch/Llama-2-7b-hf', 'NousResearch/Llama-2-13b-hf'],
                    help='LLaMA models to analyze')
    ap.add_argument('--use-safetensors', action='store_true',
                    help='Use direct safetensors extraction')
    ap.add_argument('--include-gpt2', action='store_true',
                    help='Include GPT-2 results for combined scaling fit')
    args = ap.parse_args()

    Path('results').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("RQ2: LLaMA-2 Scaling Analysis")
    print("=" * 70)

    # Analyze LLaMA models
    results = []
    for model_name in args.models:
        print(f"\n[Analyzing {model_name}]")
        try:
            r = analyze_model(model_name, args.use_safetensors)
            results.append(r)

            # Save incrementally
            with open('results/rq2_llama_scaling.json', 'w') as f:
                json.dump({'llama_models': results}, f, indent=2)
            print(f"  Saved partial results")

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

    # Load GPT-2 results if available
    gpt2_results = []
    gpt2_path = Path('results/gpt2_scaling_normalized.json')
    if args.include_gpt2 and gpt2_path.exists():
        print("\n[Loading GPT-2 results for combined fit]")
        gpt2_data = json.load(open(gpt2_path))
        gpt2_results = gpt2_data.get('per_model', [])

    # Combined scaling fit
    all_results = gpt2_results + results
    if len(all_results) >= 2:
        fits = fit_scaling(all_results)
        print("\n" + "=" * 70)
        print("SCALING FIT (combined GPT-2 + LLaMA)")
        print("=" * 70)
        print(f"  s ∝ d^{fits['s']['exponent']:.3f}  (√d would be 0.5)")
        print(f"  δ_J^norm ∝ d^{fits['delta_J_norm']['exponent']:.3f}")
    else:
        fits = None

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'d':>6} {'L':>4} {'mean_s':>8} {'med_s':>8} {'δ_J^norm':>10} {'neg_tr':>8}")
    print("-" * 85)
    for r in all_results:
        name = r['model'].split('/')[-1][:30]
        print(f"{name:<35} {r['d_model']:>6} {r['n_layers']:>4} "
              f"{r['mean_s']:>8.3f} {r['median_s']:>8.3f} "
              f"{r['mean_delta_J_norm']:>10.4f} {r['frac_neg_trace']:>7.0%}")

    # Save final results
    output = {
        'llama_models': results,
        'gpt2_models': gpt2_results,
        'combined_scaling_fit': fits,
    }
    with open('results/rq2_llama_scaling.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results/rq2_llama_scaling.json")

    # Create figure
    if results and gpt2_results:
        create_figure(results, gpt2_results, 'figures/fig_rq2_llama_scaling.png')


if __name__ == '__main__':
    main()
