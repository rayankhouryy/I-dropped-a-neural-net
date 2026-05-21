"""
Transformer-family pairing experiment (issue #10) - BERT + Mistral.

Tests diagonal-dominance pairing across diverse transformer architectures:

  BERT-base-uncased  : encoder-only, MLM, bidirectional attention,
                       GELU MLP                                        ~440 MB fp32
  Mistral-7B-v0.1    : decoder-only, causal LM, SwiGLU MLP, GQA,
                       RMSNorm                                         ~14 GB fp16

Paths scored:

  BERT:
    mlp_W2W1      : W_2 @ W_1
    attn_WO_WV    : W_O @ W_V
    attn_WQ_WK    : W_Q @ W_K^T

  Mistral (SwiGLU is a gated 3-matrix MLP):
    mlp_down_up   : W_down @ W_up
    mlp_down_gate : W_down @ W_gate
    mlp_joint     : per-cell average of the two SwiGLU scores
    attn_WO_WV    : W_O @ W_V    (with GQA expansion: 8 KV heads
                                  replicated 4x to match 32 Q heads)
    attn_WQ_WK    : W_Q @ W_K^T  (W_K also GQA-expanded)

Random-init baseline (3 seeds) for both. For Mistral, the random
baseline is generated directly from N(0, initializer_range) with the
correct shapes -- avoids loading the full HF model and saves ~14 GB
per seed.

Memory strategy for Mistral:
  - low_cpu_mem_usage=True + torch_dtype=float16
  - streaming layer-by-layer extraction: copy each layer's weights into
    fp16 numpy arrays, then SET THE LAYER TO None in the HF model and
    gc.collect() periodically. Peak resident memory stays near 14 GB
    instead of doubling to ~28 GB during the naive all-at-once extract.
  - score in fp16 numpy (matmul auto-upcasts to fp32 internally)

Outputs:
  results/transformer_family_pairing_<model>.json
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- math
def diag_dominance_matrix(A_list, B_list):
    """d(i, j) = |tr(B[j] @ A[i])| / ||B[j] @ A[i]||_F"""
    n = len(A_list)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            # auto-upcast: fp16 * fp16 in numpy uses fp32 internally
            P = B_list[j].astype(np.float32) @ A_list[i].astype(np.float32)
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def evaluate(M):
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off  = M[~np.eye(n, dtype=bool)]
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    pair_sep = float((diag - off_max_per_row).min())
    pos = diag[:, None]; neg = off[None, :]
    auc = float(((pos > neg).sum() + 0.5 * (pos == neg).sum())
                / (pos.size * neg.size))
    return {
        'n':              n,
        'chance':         1.0 / n,
        'pair_acc':       pair_acc,
        'acc_over_chance': pair_acc * n,
        'pair_sep':       pair_sep,
        'auc':            auc,
        'mean_correct':   float(diag.mean()),
        'mean_incorrect': float(off.mean()),
    }


def trace_signs(A_list, B_list):
    tr = np.array([
        float(np.trace(B_list[i].astype(np.float32) @ A_list[i].astype(np.float32)))
        for i in range(len(A_list))
    ])
    return {
        'mean_trace':    float(tr.mean()),
        'frac_negative': float((tr < 0).mean()),
        'traces':        tr.tolist(),
    }


# --------------------------------------------------------------------------- BERT
def extract_bert(model):
    if hasattr(model, 'bert'):
        layers = list(model.bert.encoder.layer)
    else:
        layers = list(model.encoder.layer)
    out = {'W_Q': [], 'W_K': [], 'W_V': [], 'W_O': [], 'W_mlp1': [], 'W_mlp2': []}
    for blk in layers:
        sa = blk.attention.self
        ao = blk.attention.output.dense
        mi = blk.intermediate.dense
        mo = blk.output.dense
        out['W_Q'].append(sa.query.weight.detach().float().cpu().numpy())
        out['W_K'].append(sa.key.weight.detach().float().cpu().numpy())
        out['W_V'].append(sa.value.weight.detach().float().cpu().numpy())
        out['W_O'].append(ao.weight.detach().float().cpu().numpy())
        out['W_mlp1'].append(mi.weight.detach().float().cpu().numpy())
        out['W_mlp2'].append(mo.weight.detach().float().cpu().numpy())
    out['n_layers'] = len(layers)
    out['d_model']  = out['W_Q'][0].shape[1]
    return out


def score_bert(W):
    out = {}
    M = diag_dominance_matrix(W['W_mlp1'], W['W_mlp2'])
    out['mlp_W2W1']   = {**evaluate(M), 'trace': trace_signs(W['W_mlp1'], W['W_mlp2'])}
    M = diag_dominance_matrix(W['W_V'], W['W_O'])
    out['attn_WO_WV'] = {**evaluate(M), 'trace': trace_signs(W['W_V'], W['W_O'])}
    W_K_T = [w.T for w in W['W_K']]
    M = diag_dominance_matrix(W['W_Q'], W_K_T)
    out['attn_WQ_WK'] = {**evaluate(M), 'trace': trace_signs(W['W_Q'], W_K_T)}
    return out


# --------------------------------------------------------------------------- Mistral / LLaMA
def extract_llama_like_streaming(model):
    """Stream layer-by-layer extraction, dropping each layer from the
    HF model after extraction. Peak memory stays near the HF model size
    rather than doubling.

    Weights kept as fp16 numpy to halve resident footprint. Matmuls
    automatically upcast to fp32 during scoring.
    """
    base = model.model if hasattr(model, 'model') else model
    cfg = model.config
    n_layers = len(base.layers)

    W = {key: [] for key in ['W_Q', 'W_K', 'W_V', 'W_O',
                              'W_gate', 'W_up', 'W_down']}

    print(f"  Extracting {n_layers} layers...", flush=True)
    for i in range(n_layers):
        blk = base.layers[i]
        sa  = blk.self_attn
        mlp = blk.mlp
        W['W_Q'].append(   sa.q_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_K'].append(   sa.k_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_V'].append(   sa.v_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_O'].append(   sa.o_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_gate'].append(mlp.gate_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_up'].append(  mlp.up_proj.weight.detach().to(torch.float16).cpu().numpy())
        W['W_down'].append(mlp.down_proj.weight.detach().to(torch.float16).cpu().numpy())
        # Free this layer's HF storage.
        base.layers[i] = None
        if (i + 1) % 4 == 0:
            gc.collect()
            print(f"    layer {i+1}/{n_layers} extracted", flush=True)

    gc.collect()
    W['n_layers']   = n_layers
    W['d_model']    = cfg.hidden_size
    W['n_heads']    = cfg.num_attention_heads
    W['n_kv_heads'] = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    W['head_dim']   = cfg.hidden_size // cfg.num_attention_heads
    W['d_ff']       = cfg.intermediate_size
    return W


def random_init_llama_like(cfg, seed):
    """Cheap random baseline: directly generate fp16 N(0, init_range)
    arrays matching Mistral's _init_weights. Avoids loading the full
    14 GB HF model per seed.
    """
    rng = np.random.default_rng(seed)
    d        = cfg.hidden_size
    d_ff     = cfg.intermediate_size
    n_heads  = cfg.num_attention_heads
    n_kv     = getattr(cfg, 'num_key_value_heads', n_heads)
    head_dim = d // n_heads
    n_layers = cfg.num_hidden_layers
    init_std = getattr(cfg, 'initializer_range', 0.02)

    W = {key: [] for key in ['W_Q', 'W_K', 'W_V', 'W_O',
                              'W_gate', 'W_up', 'W_down']}
    kv_d = n_kv * head_dim
    for _ in range(n_layers):
        W['W_Q'].append(   (rng.standard_normal((d,    d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_K'].append(   (rng.standard_normal((kv_d, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_V'].append(   (rng.standard_normal((kv_d, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_O'].append(   (rng.standard_normal((d,    d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_gate'].append((rng.standard_normal((d_ff, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_up'].append(  (rng.standard_normal((d_ff, d), dtype=np.float32) * init_std).astype(np.float16))
        W['W_down'].append((rng.standard_normal((d, d_ff), dtype=np.float32) * init_std).astype(np.float16))
    W.update({'n_layers': n_layers, 'd_model': d, 'n_heads': n_heads,
              'n_kv_heads': n_kv, 'head_dim': head_dim, 'd_ff': d_ff})
    return W


def expand_gqa(W_kv, n_heads, n_kv_heads, head_dim):
    """(n_kv_heads*head_dim, d) -> (n_heads*head_dim, d) by replicating
    each KV head's rows for the n_heads/n_kv_heads query heads in its
    group."""
    if n_heads == n_kv_heads:
        return W_kv
    repeat = n_heads // n_kv_heads
    d = W_kv.shape[1]
    return np.repeat(W_kv.reshape(n_kv_heads, head_dim, d), repeat, axis=0) \
            .reshape(n_heads * head_dim, d)


def score_llama_like(W):
    out = {}
    n_heads, n_kv, head_dim = W['n_heads'], W['n_kv_heads'], W['head_dim']

    # SwiGLU MLP: down @ up, down @ gate, joint
    M_du = diag_dominance_matrix(W['W_up'],   W['W_down'])
    out['mlp_down_up']   = {**evaluate(M_du),
                             'trace': trace_signs(W['W_up'],   W['W_down'])}

    M_dg = diag_dominance_matrix(W['W_gate'], W['W_down'])
    out['mlp_down_gate'] = {**evaluate(M_dg),
                             'trace': trace_signs(W['W_gate'], W['W_down'])}

    M_joint = 0.5 * (M_du + M_dg)
    out['mlp_joint']     = evaluate(M_joint)

    # Attention V/O with GQA expansion
    W_V_exp = [expand_gqa(v, n_heads, n_kv, head_dim) for v in W['W_V']]
    M_vo = diag_dominance_matrix(W_V_exp, W['W_O'])
    out['attn_WO_WV'] = {**evaluate(M_vo),
                         'trace': trace_signs(W_V_exp, W['W_O'])}

    # Attention Q/K with GQA expansion
    W_K_exp = [expand_gqa(k, n_heads, n_kv, head_dim) for k in W['W_K']]
    W_K_T   = [k.T for k in W_K_exp]
    M_qk = diag_dominance_matrix(W['W_Q'], W_K_T)
    out['attn_WQ_WK'] = {**evaluate(M_qk),
                         'trace': trace_signs(W['W_Q'], W_K_T)}

    out['_meta'] = {'n_heads': n_heads, 'n_kv_heads': n_kv,
                    'gqa_expanded': n_heads != n_kv}
    return out


# --------------------------------------------------------------------------- runners
def run_bert(seeds_random=3):
    from transformers import BertForMaskedLM, BertConfig
    print("\n=== BERT-base-uncased ===")
    model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                            low_cpu_mem_usage=True)
    model.eval()
    W = extract_bert(model)
    print(f"  layers: {W['n_layers']}, d_model: {W['d_model']}")
    del model; gc.collect()

    scores = score_bert(W)
    print("\n  TRAINED:")
    for p in ['mlp_W2W1', 'attn_WO_WV', 'attn_WQ_WK']:
        r = scores[p]; nt = r['trace']['frac_negative']
        print(f"    {p:<14s}  acc={r['pair_acc']:.0%}  AUC={r['auc']:.3f}  "
              f"sep={r['pair_sep']:+.3f}  neg_tr={nt:.0%}")

    print(f"\n  RANDOM INIT ({seeds_random} seeds):")
    cfg = BertConfig.from_pretrained('bert-base-uncased')
    random_per_seed = []
    for s in range(seeds_random):
        torch.manual_seed(s)
        rnd = BertForMaskedLM(cfg); rnd.eval()
        W_r = extract_bert(rnd)
        del rnd; gc.collect()
        scores_r = score_bert(W_r)
        random_per_seed.append(scores_r)
        print(f"    seed {s}: " + "  ".join(
            f"{p}={scores_r[p]['pair_acc']:.0%}/AUC={scores_r[p]['auc']:.2f}"
            for p in ['mlp_W2W1', 'attn_WO_WV', 'attn_WQ_WK']), flush=True)

    return {
        'model':           'bert-base-uncased',
        'family':          'BERT',
        'arch':            'encoder-only, MLM, bidirectional attention',
        'n_layers':        W['n_layers'],
        'd_model':         W['d_model'],
        'paths_tested':    ['mlp_W2W1', 'attn_WO_WV', 'attn_WQ_WK'],
        'trained':         scores,
        'random_per_seed': random_per_seed,
        'random_aggregate': aggregate_random(random_per_seed,
                                              ['mlp_W2W1', 'attn_WO_WV', 'attn_WQ_WK']),
    }


def run_mistral(seeds_random=3, model_name='mistralai/Mistral-7B-v0.1',
                family='Mistral'):
    from transformers import AutoModelForCausalLM, AutoConfig
    print(f"\n=== {family} ({model_name}) ===")
    print("  Loading model (fp16, low_cpu_mem_usage)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    )
    model.eval()
    W = extract_llama_like_streaming(model)
    del model; gc.collect()
    print(f"  layers: {W['n_layers']}, d_model: {W['d_model']}, "
          f"n_heads: {W['n_heads']}, n_kv_heads: {W['n_kv_heads']}, "
          f"head_dim: {W['head_dim']}, d_ff: {W['d_ff']}")

    paths = ['mlp_down_up', 'mlp_down_gate', 'mlp_joint',
             'attn_WO_WV', 'attn_WQ_WK']
    print("  Scoring trained paths...", flush=True)
    scores = score_llama_like(W)
    print("\n  TRAINED:")
    for p in paths:
        r = scores[p]
        line = f"    {p:<16s}  acc={r['pair_acc']:.0%}  AUC={r['auc']:.3f}  sep={r['pair_sep']:+.3f}"
        if 'trace' in r:
            line += f"  neg_tr={r['trace']['frac_negative']:.0%}"
        print(line)

    # Free trained weights before random baseline
    del W; gc.collect()

    print(f"\n  RANDOM INIT ({seeds_random} seeds):")
    cfg = AutoConfig.from_pretrained(model_name)
    random_per_seed = []
    for s in range(seeds_random):
        print(f"    seed {s}: generating + scoring...", flush=True)
        W_r = random_init_llama_like(cfg, seed=s)
        scores_r = score_llama_like(W_r)
        del W_r; gc.collect()
        random_per_seed.append(scores_r)
        print(f"      " + "  ".join(
            f"{p}={scores_r[p]['pair_acc']:.0%}/AUC={scores_r[p]['auc']:.2f}"
            for p in paths), flush=True)

    return {
        'model':           model_name,
        'family':          family,
        'arch':            'decoder-only, causal LM, SwiGLU MLP, GQA, RMSNorm',
        'n_layers':        cfg.num_hidden_layers,
        'd_model':         cfg.hidden_size,
        'n_heads':         cfg.num_attention_heads,
        'n_kv_heads':      getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        'd_ff':            cfg.intermediate_size,
        'paths_tested':    paths,
        'trained':         scores,
        'random_per_seed': random_per_seed,
        'random_aggregate': aggregate_random(random_per_seed, paths),
    }


def aggregate_random(per_seed, paths):
    agg = {}
    for p in paths:
        accs = np.array([s[p]['pair_acc'] for s in per_seed])
        aucs = np.array([s[p]['auc']      for s in per_seed])
        agg[p] = {
            'mean_pair_acc': float(accs.mean()),
            'std_pair_acc':  float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            'mean_auc':      float(aucs.mean()),
            'std_auc':       float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='bert-base',
                    choices=['bert-base', 'mistral-7b', 'llama2-7b', 'llama2-7b-chat', 'tinyllama'],
                    help='which model to evaluate')
    ap.add_argument('--seeds-random', type=int, default=3)
    args = ap.parse_args()

    Path('results').mkdir(parents=True, exist_ok=True)

    if args.model == 'bert-base':
        out = run_bert(args.seeds_random)
        out_key = 'bert'
    elif args.model == 'mistral-7b':
        out = run_mistral(args.seeds_random,
                          model_name='mistralai/Mistral-7B-v0.1',
                          family='Mistral')
        out_key = 'mistral_7b'
    elif args.model == 'llama2-7b':
        out = run_mistral(args.seeds_random,
                          model_name='NousResearch/Llama-2-7b-hf',
                          family='LLaMA-2')
        out_key = 'llama2_7b'
    elif args.model == 'llama2-7b-chat':
        out = run_mistral(args.seeds_random,
                          model_name='NousResearch/Llama-2-7b-chat-hf',
                          family='LLaMA-2-chat')
        out_key = 'llama2_7b_chat'
    elif args.model == 'tinyllama':
        out = run_mistral(args.seeds_random,
                          model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                          family='TinyLlama')
        out_key = 'tinyllama'

    out_path = f'results/transformer_family_pairing_{out_key}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')

    print("\n=== HEADLINE ===")
    print(f"  Model: {out['family']} ({out['model']}), n_layers={out['n_layers']}")
    print(f"  {'path':<16s}  {'trained acc':>11s}  {'random acc':>14s}  "
          f"{'trained AUC':>11s}  {'random AUC':>14s}")
    for p in out['paths_tested']:
        t = out['trained'][p]
        r = out['random_aggregate'][p]
        print(f"  {p:<16s}  {t['pair_acc']:>10.0%}   "
              f"{r['mean_pair_acc']:.0%}+-{r['std_pair_acc']:.0%}".rjust(14)
              + f"  {t['auc']:>10.3f}  "
              + f"{r['mean_auc']:.3f}+-{r['std_auc']:.3f}".rjust(14))


if __name__ == '__main__':
    main()
