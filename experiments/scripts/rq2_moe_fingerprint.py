"""RQ2 Pending Experiment: MoE / Mixtral-8x7B Fingerprint Analysis.

Tests whether the diagonal-dominance fingerprint extends to Mixture-of-Experts
architectures. Key questions from RQ.md:
  - Do experts inherit the fingerprint independently?
  - Do experts share fingerprint structure via the gate?
  - Can we pair experts across layers?

Mixtral-8x7B architecture:
  - 32 layers, each with 8 experts
  - Each expert: SwiGLU MLP with W_gate (w1), W_up (w3), W_down (w2)
  - Router selects top-2 experts per token
  - d_model = 4096, per-expert d_ff = 14336
  - GQA: 32 attention heads, 8 KV heads

Analysis paths:
  1. Per-expert MLP: M = W_down @ W_up for each expert (8 per layer, 256 total)
  2. Per-expert gate: M = W_down @ W_gate
  3. Aggregate per-layer: average fingerprint across experts in each layer
  4. Cross-expert similarity: do experts in the same layer have similar fingerprints?
  5. Standard attention paths: W_O @ W_V, W_Q @ W_K^T (same as dense models)

Resource estimate: ~80 GB GPU memory for fp16 inference.

Outputs:
  results/rq2_moe_mixtral.json
  figures/fig_rq2_moe_mixtral.png
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# Detect GPU availability once at module load
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def diag_dominance(M):
    """s = |tr(M)| / ||M||_F"""
    return float(abs(np.trace(M)) / (np.linalg.norm(M, 'fro') + 1e-12))


def delta_J_norm(M):
    """Scale-invariant Jacobian orthogonality: ||J^T J / ||J||_F^2 - I/d||_F"""
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    JTJ = J.T @ J
    fro2 = float(np.linalg.norm(J, 'fro') ** 2)
    return float(np.linalg.norm(JTJ / fro2 - I / d, 'fro'))


def diag_dominance_matrix(A_list, B_list):
    """Compute pairwise diagonal dominance: d(i,j) = |tr(B[j] @ A[i])| / ||B[j] @ A[i]||_F

    Uses GPU (CUDA) if available for fast matrix operations.
    """
    n = len(A_list)
    M = np.zeros((n, n), dtype=np.float64)

    if DEVICE == 'cuda':
        print(f"    (using GPU for scoring)", flush=True)
        A_tensors = [torch.from_numpy(a.astype(np.float32)).to(DEVICE) for a in A_list]
        B_tensors = [torch.from_numpy(b.astype(np.float32)).to(DEVICE) for b in B_list]

        for i in range(n):
            for j in range(n):
                P = B_tensors[j] @ A_tensors[i]
                tr = abs(float(torch.trace(P)))
                fr = float(torch.linalg.norm(P, 'fro')) + 1e-12
                M[i, j] = tr / fr

        del A_tensors, B_tensors
        torch.cuda.empty_cache()
    else:
        for i in range(n):
            for j in range(n):
                P = B_list[j].astype(np.float32) @ A_list[i].astype(np.float32)
                tr = abs(np.trace(P))
                fr = np.linalg.norm(P, 'fro') + 1e-12
                M[i, j] = tr / fr
    return M


def evaluate(M):
    """Evaluate pairing accuracy via Hungarian matching."""
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    pos = diag[:, None]
    neg = off[None, :]
    auc = float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))
    return {
        'n': n,
        'chance': 1.0 / n,
        'pair_acc': pair_acc,
        'pair_sep': pair_sep,
        'auc': auc,
        'mean_correct': float(diag.mean()),
        'mean_incorrect': float(off.mean()),
    }


def trace_signs(A_list, B_list):
    """Compute trace statistics for correct pairs. Uses GPU if available."""
    if DEVICE == 'cuda':
        A_tensors = [torch.from_numpy(a.astype(np.float32)).to(DEVICE) for a in A_list]
        B_tensors = [torch.from_numpy(b.astype(np.float32)).to(DEVICE) for b in B_list]
        tr = np.array([
            float(torch.trace(B_tensors[i] @ A_tensors[i]))
            for i in range(len(A_list))
        ])
        del A_tensors, B_tensors
        torch.cuda.empty_cache()
    else:
        tr = np.array([
            float(np.trace(B_list[i].astype(np.float32) @ A_list[i].astype(np.float32)))
            for i in range(len(A_list))
        ])
    return {
        'mean_trace': float(tr.mean()),
        'frac_negative': float((tr < 0).mean()),
        'traces': tr.tolist(),
    }


def expand_gqa(W_kv, n_heads, n_kv_heads, head_dim):
    """Expand GQA KV weights to match Q heads."""
    if n_heads == n_kv_heads:
        return W_kv
    repeat = n_heads // n_kv_heads
    d = W_kv.shape[1]
    return np.repeat(W_kv.reshape(n_kv_heads, head_dim, d), repeat, axis=0).reshape(n_heads * head_dim, d)


def extract_mixtral_streaming(model_name='mistralai/Mixtral-8x7B-v0.1'):
    """Extract weights from Mixtral MoE model with streaming to manage memory.

    Returns dict with:
      - experts[layer][expert] = {'W_gate', 'W_up', 'W_down'}
      - attention weights: W_Q, W_K, W_V, W_O lists
      - config metadata
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"  Loading {model_name} (fp16, low_cpu_mem_usage)...")
    print("  This requires ~80GB GPU memory or will use CPU offloading...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    model.eval()
    cfg = model.config

    n_layers = cfg.num_hidden_layers
    n_experts = cfg.num_local_experts
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    head_dim = cfg.hidden_size // n_heads

    print(f"  Config: {n_layers} layers, {n_experts} experts/layer, d={cfg.hidden_size}, d_ff={cfg.intermediate_size}")
    print(f"  Attention: {n_heads} heads, {n_kv_heads} KV heads, head_dim={head_dim}")

    W = {
        'experts': [[None for _ in range(n_experts)] for _ in range(n_layers)],
        'W_Q': [], 'W_K': [], 'W_V': [], 'W_O': [],
    }

    base = model.model if hasattr(model, 'model') else model

    print(f"  Extracting {n_layers} layers...")
    for i in range(n_layers):
        layer = base.layers[i]

        # Extract attention weights
        sa = layer.self_attn
        W['W_Q'].append(sa.q_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_K'].append(sa.k_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_V'].append(sa.v_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_O'].append(sa.o_proj.weight.detach().to(torch.float16).cpu().numpy())

        # Extract MoE expert weights
        moe = layer.block_sparse_moe
        for e in range(n_experts):
            expert = moe.experts[e]
            W['experts'][i][e] = {
                'W_gate': expert.w1.weight.detach().to(torch.float16).cpu().numpy(),  # gate_proj
                'W_up': expert.w3.weight.detach().to(torch.float16).cpu().numpy(),    # up_proj
                'W_down': expert.w2.weight.detach().to(torch.float16).cpu().numpy(),  # down_proj
            }

        # Free layer memory
        base.layers[i] = None
        if (i + 1) % 4 == 0:
            gc.collect()
            print(f"    layer {i+1}/{n_layers} extracted", flush=True)

    del model
    gc.collect()

    W['n_layers'] = n_layers
    W['n_experts'] = n_experts
    W['d_model'] = cfg.hidden_size
    W['d_ff'] = cfg.intermediate_size
    W['n_heads'] = n_heads
    W['n_kv_heads'] = n_kv_heads
    W['head_dim'] = head_dim

    return W


def extract_mixtral_safetensors(model_name='mistralai/Mixtral-8x7B-v0.1', hf_cache_root=None):
    """Direct safetensors extraction - bypasses full model loading.

    More memory-efficient: reads weights directly from disk without
    instantiating the full model.
    """
    import json as json_lib
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

    cfg = json_lib.loads((snap / 'config.json').read_text())
    n_layers = cfg['num_hidden_layers']
    n_experts = cfg['num_local_experts']

    print(f"  (safetensors-direct path; snap={snap.name[:8]}...)")
    print(f"  Extracting {n_layers} layers x {n_experts} experts from safetensors...")

    idx_file = snap / 'model.safetensors.index.json'
    if idx_file.exists():
        idx = json_lib.loads(idx_file.read_text())['weight_map']
    else:
        single = snap / 'model.safetensors'
        if not single.exists():
            raise FileNotFoundError(f"No safetensors at {snap}")
        with safe_open(str(single), framework='pt') as f:
            idx = {k: 'model.safetensors' for k in f.keys()}

    shard_to_keys = {}
    for k, shard in idx.items():
        shard_to_keys.setdefault(shard, []).append(k)

    def _to_fp16_np(t):
        return t.to(torch.float32).to(torch.float16).numpy()

    W = {
        'experts': [[{} for _ in range(n_experts)] for _ in range(n_layers)],
        'W_Q': [None] * n_layers,
        'W_K': [None] * n_layers,
        'W_V': [None] * n_layers,
        'W_O': [None] * n_layers,
    }

    # MoE weight patterns
    expert_patterns = {
        'block_sparse_moe.experts.{e}.w1.weight': 'W_gate',
        'block_sparse_moe.experts.{e}.w2.weight': 'W_down',
        'block_sparse_moe.experts.{e}.w3.weight': 'W_up',
    }
    attn_patterns = {
        'self_attn.q_proj.weight': 'W_Q',
        'self_attn.k_proj.weight': 'W_K',
        'self_attn.v_proj.weight': 'W_V',
        'self_attn.o_proj.weight': 'W_O',
    }

    extracted = 0
    total_weights = n_layers * (4 + n_experts * 3)

    for shard, keys in shard_to_keys.items():
        wanted = [k for k in keys if k.startswith('model.layers.')]
        if not wanted:
            continue

        with safe_open(str(snap / shard), framework='pt') as f:
            for k in wanted:
                parts = k.split('.')
                layer_idx = int(parts[2])

                # Check attention weights
                for suffix, pool in attn_patterns.items():
                    if k.endswith(suffix):
                        W[pool][layer_idx] = _to_fp16_np(f.get_tensor(k))
                        extracted += 1
                        break

                # Check expert weights
                if 'block_sparse_moe.experts' in k:
                    expert_idx = int(parts[5])
                    weight_type = parts[6]
                    if weight_type == 'w1':
                        W['experts'][layer_idx][expert_idx]['W_gate'] = _to_fp16_np(f.get_tensor(k))
                    elif weight_type == 'w2':
                        W['experts'][layer_idx][expert_idx]['W_down'] = _to_fp16_np(f.get_tensor(k))
                    elif weight_type == 'w3':
                        W['experts'][layer_idx][expert_idx]['W_up'] = _to_fp16_np(f.get_tensor(k))
                    extracted += 1

        gc.collect()
        print(f"    {shard}: {extracted}/{total_weights} weights", flush=True)

    W['n_layers'] = n_layers
    W['n_experts'] = n_experts
    W['d_model'] = cfg['hidden_size']
    W['d_ff'] = cfg['intermediate_size']
    W['n_heads'] = cfg['num_attention_heads']
    W['n_kv_heads'] = cfg.get('num_key_value_heads', cfg['num_attention_heads'])
    W['head_dim'] = cfg['hidden_size'] // cfg['num_attention_heads']

    return W


def random_init_mixtral(cfg_dict, seed):
    """Generate random baseline with Mixtral shapes."""
    rng = np.random.default_rng(seed)

    n_layers = cfg_dict['n_layers']
    n_experts = cfg_dict['n_experts']
    d = cfg_dict['d_model']
    d_ff = cfg_dict['d_ff']
    n_heads = cfg_dict['n_heads']
    n_kv = cfg_dict['n_kv_heads']
    head_dim = cfg_dict['head_dim']
    init_std = 0.02

    W = {
        'experts': [[{} for _ in range(n_experts)] for _ in range(n_layers)],
        'W_Q': [], 'W_K': [], 'W_V': [], 'W_O': [],
    }

    kv_d = n_kv * head_dim

    for _ in range(n_layers):
        W['W_Q'].append((rng.standard_normal((d, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_K'].append((rng.standard_normal((kv_d, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_V'].append((rng.standard_normal((kv_d, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_O'].append((rng.standard_normal((d, d), dtype=np.float32) * init_std).astype(np.float16))

    for i in range(n_layers):
        for e in range(n_experts):
            W['experts'][i][e] = {
                'W_gate': (rng.standard_normal((d_ff, d), dtype=np.float32) * init_std).astype(np.float16),
                'W_up': (rng.standard_normal((d_ff, d), dtype=np.float32) * init_std).astype(np.float16),
                'W_down': (rng.standard_normal((d, d_ff), dtype=np.float32) * init_std).astype(np.float16),
            }

    W.update(cfg_dict)
    return W


def diag_dominance_gpu(M_tensor):
    """GPU version: s = |tr(M)| / ||M||_F"""
    tr = abs(float(torch.trace(M_tensor)))
    fr = float(torch.linalg.norm(M_tensor, 'fro')) + 1e-12
    return tr / fr


def delta_J_norm_gpu(M_tensor):
    """GPU version: scale-invariant Jacobian orthogonality."""
    d = M_tensor.shape[0]
    I = torch.eye(d, device=M_tensor.device, dtype=M_tensor.dtype)
    J = I + M_tensor
    JTJ = J.T @ J
    fro2 = float(torch.linalg.norm(J, 'fro') ** 2)
    return float(torch.linalg.norm(JTJ / fro2 - I / d, 'fro'))


def score_moe(W):
    """Compute all fingerprint metrics for MoE model."""
    n_layers = W['n_layers']
    n_experts = W['n_experts']
    n_heads = W['n_heads']
    n_kv = W['n_kv_heads']
    head_dim = W['head_dim']

    results = {}

    # 1. Per-expert fingerprint (s, delta_J_norm for each expert)
    print("  Computing per-expert fingerprints...", flush=True)
    if DEVICE == 'cuda':
        print(f"    (using GPU for {n_layers * n_experts} expert blocks)", flush=True)

    expert_metrics = []
    for i in range(n_layers):
        layer_experts = []
        for e in range(n_experts):
            exp = W['experts'][i][e]

            if DEVICE == 'cuda':
                W_down = torch.from_numpy(exp['W_down'].astype(np.float32)).to(DEVICE)
                W_up = torch.from_numpy(exp['W_up'].astype(np.float32)).to(DEVICE)
                W_gate = torch.from_numpy(exp['W_gate'].astype(np.float32)).to(DEVICE)

                M_up = W_down @ W_up
                M_gate = W_down @ W_gate

                layer_experts.append({
                    'layer': i,
                    'expert': e,
                    's_down_up': diag_dominance_gpu(M_up),
                    's_down_gate': diag_dominance_gpu(M_gate),
                    'delta_J_norm_up': delta_J_norm_gpu(M_up),
                    'delta_J_norm_gate': delta_J_norm_gpu(M_gate),
                    'trace_up': float(torch.trace(M_up)),
                    'trace_gate': float(torch.trace(M_gate)),
                })

                del W_down, W_up, W_gate, M_up, M_gate
            else:
                M_up = exp['W_down'].astype(np.float32) @ exp['W_up'].astype(np.float32)
                M_gate = exp['W_down'].astype(np.float32) @ exp['W_gate'].astype(np.float32)

                layer_experts.append({
                    'layer': i,
                    'expert': e,
                    's_down_up': diag_dominance(M_up),
                    's_down_gate': diag_dominance(M_gate),
                    'delta_J_norm_up': delta_J_norm(M_up),
                    'delta_J_norm_gate': delta_J_norm(M_gate),
                    'trace_up': float(np.trace(M_up)),
                    'trace_gate': float(np.trace(M_gate)),
                })
        expert_metrics.append(layer_experts)
        if DEVICE == 'cuda' and (i + 1) % 8 == 0:
            torch.cuda.empty_cache()
            print(f"    layer {i+1}/{n_layers} done", flush=True)

    results['per_expert'] = expert_metrics

    # 2. Aggregate per-layer statistics
    print("  Computing per-layer aggregates...")
    layer_agg = []
    for i in range(n_layers):
        s_vals = [expert_metrics[i][e]['s_down_up'] for e in range(n_experts)]
        delta_vals = [expert_metrics[i][e]['delta_J_norm_up'] for e in range(n_experts)]
        trace_vals = [expert_metrics[i][e]['trace_up'] for e in range(n_experts)]

        layer_agg.append({
            'layer': i,
            'mean_s': float(np.mean(s_vals)),
            'std_s': float(np.std(s_vals)),
            'mean_delta_J_norm': float(np.mean(delta_vals)),
            'frac_neg_trace': float(np.mean([t < 0 for t in trace_vals])),
        })
    results['layer_aggregate'] = layer_agg

    # 3. Cross-expert similarity within each layer
    print("  Computing cross-expert similarity...", flush=True)
    cross_expert = []
    for i in range(n_layers):
        # Compute s for all expert pairs within this layer
        s_matrix = np.zeros((n_experts, n_experts))

        if DEVICE == 'cuda':
            # Load all experts for this layer to GPU
            W_downs = [torch.from_numpy(W['experts'][i][e]['W_down'].astype(np.float32)).to(DEVICE)
                       for e in range(n_experts)]
            W_ups = [torch.from_numpy(W['experts'][i][e]['W_up'].astype(np.float32)).to(DEVICE)
                     for e in range(n_experts)]

            for e1 in range(n_experts):
                for e2 in range(n_experts):
                    M = W_downs[e2] @ W_ups[e1]
                    s_matrix[e1, e2] = diag_dominance_gpu(M)

            del W_downs, W_ups
            torch.cuda.empty_cache()
        else:
            for e1 in range(n_experts):
                for e2 in range(n_experts):
                    M = W['experts'][i][e2]['W_down'].astype(np.float32) @ \
                        W['experts'][i][e1]['W_up'].astype(np.float32)
                    s_matrix[e1, e2] = diag_dominance(M)

        diag_mean = float(np.diag(s_matrix).mean())
        off_diag = s_matrix[~np.eye(n_experts, dtype=bool)]
        off_mean = float(off_diag.mean())

        cross_expert.append({
            'layer': i,
            'diag_mean': diag_mean,
            'off_diag_mean': off_mean,
            'separation': diag_mean - off_mean,
        })
    results['cross_expert_similarity'] = cross_expert

    # 4. Layer-level pairing using aggregated expert fingerprints
    print("  Computing layer-level pairing...")
    # Aggregate W_up and W_down across experts for each layer (simple average)
    W_up_agg = []
    W_down_agg = []
    for i in range(n_layers):
        up_stack = np.stack([W['experts'][i][e]['W_up'].astype(np.float32)
                            for e in range(n_experts)], axis=0)
        down_stack = np.stack([W['experts'][i][e]['W_down'].astype(np.float32)
                              for e in range(n_experts)], axis=0)
        W_up_agg.append(up_stack.mean(axis=0))
        W_down_agg.append(down_stack.mean(axis=0))

    M_layer = diag_dominance_matrix(W_up_agg, W_down_agg)
    results['layer_pairing'] = {
        **evaluate(M_layer),
        'trace': trace_signs(W_up_agg, W_down_agg),
    }

    # 5. Attention paths (same as dense models)
    print("  Computing attention path fingerprints...")
    W_V_exp = [expand_gqa(v, n_heads, n_kv, head_dim) for v in W['W_V']]
    M_vo = diag_dominance_matrix(W_V_exp, W['W_O'])
    results['attn_WO_WV'] = {
        **evaluate(M_vo),
        'trace': trace_signs(W_V_exp, W['W_O']),
    }

    W_K_exp = [expand_gqa(k, n_heads, n_kv, head_dim) for k in W['W_K']]
    W_K_T = [k.T for k in W_K_exp]
    M_qk = diag_dominance_matrix(W['W_Q'], W_K_T)
    results['attn_WQ_WK'] = {
        **evaluate(M_qk),
        'trace': trace_signs(W['W_Q'], W_K_T),
    }

    # 6. Global summary
    all_s = [expert_metrics[i][e]['s_down_up']
             for i in range(n_layers) for e in range(n_experts)]
    all_delta = [expert_metrics[i][e]['delta_J_norm_up']
                 for i in range(n_layers) for e in range(n_experts)]
    all_traces = [expert_metrics[i][e]['trace_up']
                  for i in range(n_layers) for e in range(n_experts)]

    results['global_summary'] = {
        'n_layers': n_layers,
        'n_experts': n_experts,
        'total_expert_blocks': n_layers * n_experts,
        'mean_s': float(np.mean(all_s)),
        'std_s': float(np.std(all_s)),
        'mean_delta_J_norm': float(np.mean(all_delta)),
        'frac_neg_trace': float(np.mean([t < 0 for t in all_traces])),
    }

    return results


def print_results(scores, label='TRAINED'):
    """Pretty-print scoring results."""
    print(f"\n  === {label} ===")

    gs = scores['global_summary']
    print(f"\n  Global ({gs['total_expert_blocks']} expert blocks):")
    print(f"    Mean s:            {gs['mean_s']:.4f} ± {gs['std_s']:.4f}")
    print(f"    Mean δ_J^norm:     {gs['mean_delta_J_norm']:.4f}")
    print(f"    Frac neg trace:    {gs['frac_neg_trace']:.1%}")

    lp = scores['layer_pairing']
    print(f"\n  Layer-level pairing (aggregate across experts):")
    print(f"    Pair accuracy:     {lp['pair_acc']:.0%}")
    print(f"    AUC:               {lp['auc']:.3f}")
    print(f"    Separation:        {lp['pair_sep']:+.4f}")

    print(f"\n  Attention paths:")
    for path in ['attn_WO_WV', 'attn_WQ_WK']:
        r = scores[path]
        nt = r['trace']['frac_negative']
        print(f"    {path:<14s}  acc={r['pair_acc']:.0%}  AUC={r['auc']:.3f}  neg_tr={nt:.0%}")

    print(f"\n  Cross-expert similarity (first 4 layers):")
    for ce in scores['cross_expert_similarity'][:4]:
        print(f"    Layer {ce['layer']:2d}: diag={ce['diag_mean']:.3f}, "
              f"off={ce['off_diag_mean']:.3f}, sep={ce['separation']:+.3f}")


def create_figure(scores, random_scores, out_path):
    """Create visualization of MoE fingerprint results."""
    import matplotlib.pyplot as plt

    n_layers = scores['global_summary']['n_layers']
    n_experts = scores['global_summary']['n_experts']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Per-expert s values heatmap
    ax = axes[0, 0]
    s_matrix = np.array([[scores['per_expert'][i][e]['s_down_up']
                          for e in range(n_experts)] for i in range(n_layers)])
    im = ax.imshow(s_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('Expert')
    ax.set_ylabel('Layer')
    ax.set_title('Diagonal Dominance (s) per Expert')
    plt.colorbar(im, ax=ax)

    # 2. Layer aggregate s: trained vs random
    ax = axes[0, 1]
    layers = range(n_layers)
    trained_s = [scores['layer_aggregate'][i]['mean_s'] for i in layers]
    random_s = [random_scores['layer_aggregate'][i]['mean_s'] for i in layers]
    ax.plot(layers, trained_s, 'b-o', label='Trained', markersize=4)
    ax.plot(layers, random_s, 'r--x', label='Random-init', markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean s (across experts)')
    ax.set_title('Layer-wise Fingerprint Strength')
    ax.legend()

    # 3. Cross-expert separation
    ax = axes[0, 2]
    sep = [scores['cross_expert_similarity'][i]['separation'] for i in layers]
    ax.bar(layers, sep, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Separation (diag - off-diag)')
    ax.set_title('Cross-Expert Fingerprint Separation')

    # 4. Per-expert delta_J_norm heatmap
    ax = axes[1, 0]
    delta_matrix = np.array([[scores['per_expert'][i][e]['delta_J_norm_up']
                              for e in range(n_experts)] for i in range(n_layers)])
    im = ax.imshow(delta_matrix, aspect='auto', cmap='magma')
    ax.set_xlabel('Expert')
    ax.set_ylabel('Layer')
    ax.set_title('δ_J^norm per Expert')
    plt.colorbar(im, ax=ax)

    # 5. Attention path comparison
    ax = axes[1, 1]
    paths = ['layer_pairing', 'attn_WO_WV', 'attn_WQ_WK']
    labels = ['MLP\n(aggregate)', 'Attn\nV/O', 'Attn\nQ/K']
    trained_acc = [scores[p]['pair_acc'] for p in paths]
    random_acc = [random_scores[p]['pair_acc'] for p in paths]
    x = np.arange(len(paths))
    width = 0.35
    ax.bar(x - width/2, trained_acc, width, label='Trained', color='steelblue')
    ax.bar(x + width/2, random_acc, width, label='Random-init', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Pair Accuracy')
    ax.set_title('Pairing Accuracy by Path')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # 6. Global summary comparison
    ax = axes[1, 2]
    metrics = ['mean_s', 'mean_delta_J_norm', 'frac_neg_trace']
    labels = ['Mean s', 'Mean δ_J^norm', 'Frac neg tr']
    trained_vals = [scores['global_summary'][m] for m in metrics]
    random_vals = [random_scores['global_summary'][m] for m in metrics]
    x = np.arange(len(metrics))
    ax.bar(x - width/2, trained_vals, width, label='Trained', color='steelblue')
    ax.bar(x + width/2, random_vals, width, label='Random-init', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Global Fingerprint Metrics')
    ax.legend()

    plt.suptitle(f'MoE Fingerprint Analysis: {n_layers} layers × {n_experts} experts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved {out_path}")


def main():
    ap = argparse.ArgumentParser(description='MoE Fingerprint Analysis (RQ2)')
    ap.add_argument('--model', default='mixtral-8x7b',
                    choices=['mixtral-8x7b'],
                    help='MoE model to evaluate')
    ap.add_argument('--use-safetensors', action='store_true',
                    help='Use direct safetensors extraction (more memory efficient)')
    ap.add_argument('--seeds-random', type=int, default=1,
                    help='Number of random init seeds for baseline')
    ap.add_argument('--skip-trained', action='store_true',
                    help='Skip trained model (for testing with random only)')
    args = ap.parse_args()

    Path('results').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)

    model_name = 'mistralai/Mixtral-8x7B-v0.1'

    print("=" * 70)
    print("RQ2: MoE / Mixtral-8x7B Fingerprint Analysis")
    print("=" * 70)

    if not args.skip_trained:
        # Extract trained model
        print(f"\n[1] Extracting trained {model_name}...")
        if args.use_safetensors:
            W = extract_mixtral_safetensors(model_name)
        else:
            W = extract_mixtral_streaming(model_name)

        # Score trained model
        print("\n[2] Scoring trained model...")
        scores = score_moe(W)
        print_results(scores, 'TRAINED')

        # Save config for random baseline
        cfg_dict = {k: W[k] for k in ['n_layers', 'n_experts', 'd_model', 'd_ff',
                                       'n_heads', 'n_kv_heads', 'head_dim']}
        del W
        gc.collect()
    else:
        # Use Mixtral config for random baseline
        cfg_dict = {
            'n_layers': 32, 'n_experts': 8, 'd_model': 4096, 'd_ff': 14336,
            'n_heads': 32, 'n_kv_heads': 8, 'head_dim': 128,
        }
        scores = None

    # Random baseline
    print(f"\n[3] Computing random baseline ({args.seeds_random} seeds)...")
    random_all = []
    for seed in range(args.seeds_random):
        print(f"  Seed {seed}...")
        W_r = random_init_mixtral(cfg_dict, seed=seed)
        scores_r = score_moe(W_r)
        random_all.append(scores_r)
        del W_r
        gc.collect()

    # Aggregate random results (use first seed as representative)
    random_scores = random_all[0]
    if args.seeds_random > 1:
        # Average global summary across seeds
        for key in random_scores['global_summary']:
            if isinstance(random_scores['global_summary'][key], float):
                vals = [r['global_summary'][key] for r in random_all]
                random_scores['global_summary'][key] = float(np.mean(vals))

    print_results(random_scores, 'RANDOM-INIT')

    # Save results
    output = {
        'model': model_name,
        'config': cfg_dict,
        'trained': scores,
        'random': random_scores,
        'random_all_seeds': random_all if args.seeds_random > 1 else None,
    }

    out_path = 'results/rq2_moe_mixtral.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    # Create figure
    if scores is not None:
        create_figure(scores, random_scores, 'figures/fig_rq2_moe_mixtral.png')

    # Print headline
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    if scores:
        gs = scores['global_summary']
        gs_r = random_scores['global_summary']
        print(f"\nMixtral-8x7B ({gs['n_layers']} layers × {gs['n_experts']} experts = {gs['total_expert_blocks']} MLP blocks)")
        print(f"  Mean s:        {gs['mean_s']:.3f} (trained) vs {gs_r['mean_s']:.3f} (random) — {gs['mean_s']/gs_r['mean_s']:.1f}× ratio")
        print(f"  δ_J^norm:      {gs['mean_delta_J_norm']:.4f} (trained) vs {gs_r['mean_delta_J_norm']:.4f} (random)")
        print(f"  Neg trace:     {gs['frac_neg_trace']:.0%} (trained) vs {gs_r['frac_neg_trace']:.0%} (random)")
        print(f"\n  Layer pairing: {scores['layer_pairing']['pair_acc']:.0%} accuracy, AUC={scores['layer_pairing']['auc']:.3f}")
        print(f"  Attention V/O: {scores['attn_WO_WV']['pair_acc']:.0%} accuracy, AUC={scores['attn_WO_WV']['auc']:.3f}")


if __name__ == '__main__':
    main()
