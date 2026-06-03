"""
GPT-2 Attention Block Pairing Experiment (Issue #9).

Tests whether the diagonal-dominance pairing signal extends from MLP
sublayers (Section 6, 100% pair accuracy) to ATTENTION sublayers.

A GPT-2 attention block has four matrices:
    Q = W_Q x,  K = W_K x,  V = W_V x
    output = softmax(QK^T / sqrt(d_head)) V
    final  = W_O output                       -- writes back to residual stream

The most "residual-like" sub-path is V -> O:
    attn_residual_branch(x) = W_O @ softmax(...) @ W_V @ x
so the analog of W_out @ W_in for ResNets is the product

    M_VO(i,j) := W_O^{(j)} @ W_V^{(i)}    in R^{d_model x d_model}

If dynamic isometry constrains the attention residual flow the same way
it constrains MLP residual flow, M_VO should exhibit a diagonal-dominance
signature for correctly paired layers.

We also try W_Q <-> W_K (more speculative -- there's no clean residual
argument because QK only enters as a softmax-weighted attention matrix,
not as a linear residual contribution). Included as a control.

In GPT-2, attention is implemented via:
    block.attn.c_attn.weight   -- (d_model, 3*d_model)
                                  packs Q, K, V concatenated on output axis
    block.attn.c_proj.weight   -- (d_model, d_model)   == W_O (stored transposed)

After transposing to standard (rows = output, cols = input) form:
    W_Q, W_K, W_V each have shape (d_model, d_model)
    W_O           has shape         (d_model, d_model)

Outputs:
  figures/fig_gpt2_attention_pairing.{png,pdf}
  results/gpt2_attention_pairing.json
"""

import json
import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# weight extraction
# ----------------------------------------------------------------------
def extract_attention_weights(model_name, dtype=None):
    """Pull W_Q, W_K, W_V, W_O from every transformer block of a GPT-2 model.

    Returns:
        W_Qs, W_Ks, W_Vs, W_Os: lists of (d_model, d_model) float32 numpy arrays
        config: hf model config
    """
    from transformers import GPT2LMHeadModel

    kwargs = {'low_cpu_mem_usage': True}
    if dtype is not None:
        kwargs['torch_dtype'] = dtype
    print(f"  Loading {model_name}"
          f"{' (fp16)' if dtype == torch.float16 else ''}...")
    model = GPT2LMHeadModel.from_pretrained(model_name, **kwargs)
    model.eval()
    cfg = model.config
    d = cfg.n_embd

    W_Qs, W_Ks, W_Vs, W_Os = [], [], [], []
    for block in model.transformer.h:
        # c_attn stores Q, K, V concatenated on the OUTPUT axis.
        # HuggingFace Conv1D stores weights as (in_features, out_features):
        # c_attn.weight has shape (d_model, 3*d_model).
        # After .T it's (3*d_model, d_model); split into three (d_model, d_model).
        W_attn = block.attn.c_attn.weight.detach().float().cpu().numpy().T  # (3d, d)
        W_q, W_k, W_v = np.split(W_attn, 3, axis=0)                         # each (d, d)
        W_o = block.attn.c_proj.weight.detach().float().cpu().numpy().T     # (d, d)
        W_Qs.append(W_q)
        W_Ks.append(W_k)
        W_Vs.append(W_v)
        W_Os.append(W_o)

    del model
    gc.collect()
    return W_Qs, W_Ks, W_Vs, W_Os, cfg


# ----------------------------------------------------------------------
# pairing scores
# ----------------------------------------------------------------------
def diag_dominance_matrix(A_list, B_list):
    """d(i,j) = |tr(B[j] @ A[i])| / ||B[j] @ A[i]||_F"""
    n = len(A_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            prod = B_list[j] @ A_list[i]
            tr = abs(np.trace(prod))
            fr = np.linalg.norm(prod, 'fro') + 1e-12
            M[i, j] = tr / fr
    return M


def per_head_diag_dominance_matrix(A_list, B_list, n_head):
    """Aggregated per-head margin: for each (i, j) average the
    diagonal-dominance ratio across the n_head head blocks.

    Inside an attention layer the heads operate on disjoint slices of
    the value space, so the natural object is the n_head head-products
    rather than the single (d_model, d_model) tensor product.
    """
    n = len(A_list)
    d = A_list[0].shape[0]
    d_head = d // n_head
    M = np.zeros((n, n))
    for i in range(n):
        # split W_V^{(i)} columns/rows into n_head slabs; layout depends on B vs A
        A_heads = [A_list[i][h*d_head:(h+1)*d_head, :] for h in range(n_head)]
        for j in range(n):
            B_heads = [B_list[j][:, h*d_head:(h+1)*d_head] for h in range(n_head)]
            scores = []
            for h in range(n_head):
                prod = B_heads[h] @ A_heads[h]  # (d, d) -- still d_model x d_model
                tr = abs(np.trace(prod))
                fr = np.linalg.norm(prod, 'fro') + 1e-12
                scores.append(tr / fr)
            M[i, j] = float(np.mean(scores))
    return M


def evaluate(M, minimize=False):
    n = M.shape[0]
    _, col = linear_sum_assignment(M if minimize else -M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    off_max_per_row = (M - np.diag(diag)).max(axis=1)
    if minimize:
        sep = float(off.min() - diag.max())
    else:
        sep = float((diag - off_max_per_row).min())
    return {
        'pair_acc': pair_acc,
        'pair_sep': sep,
        'mean_correct':   float(diag.mean()),
        'mean_incorrect': float(off.mean()),
        'correct_pairs':  int(pair_acc * n),
        'total_pairs':    n,
        'assignment':     col.tolist(),
    }


def trace_signs(A_list, B_list):
    """Compute tr(B[i] @ A[i]) for every i. Dynamic isometry predicts negative."""
    traces = []
    fr_norms = []
    for i in range(len(A_list)):
        prod = B_list[i] @ A_list[i]
        traces.append(float(np.trace(prod)))
        fr_norms.append(float(np.linalg.norm(prod, 'fro')))
    traces = np.array(traces)
    return {
        'traces': traces.tolist(),
        'mean_trace': float(traces.mean()),
        'std_trace':  float(traces.std()),
        'frac_negative': float((traces < 0).mean()),
        'mean_frob': float(np.mean(fr_norms)),
    }


# ----------------------------------------------------------------------
# main per-model experiment
# ----------------------------------------------------------------------
def run_model(model_name, dtype=None):
    print(f"\n{'='*60}\nAnalyzing {model_name}\n{'='*60}")
    Wq, Wk, Wv, Wo, cfg = extract_attention_weights(model_name, dtype=dtype)
    n = len(Wv)
    d = cfg.n_embd
    n_head = cfg.n_head

    print(f"  Layers:   {n}")
    print(f"  d_model:  {d}")
    print(f"  n_head:   {n_head}  (d_head = {d // n_head})")

    print("\n  [VO] Computing W_O @ W_V diagonal-dominance...")
    M_vo = diag_dominance_matrix(Wv, Wo)
    res_vo = evaluate(M_vo, minimize=False)
    trace_vo = trace_signs(Wv, Wo)

    print("\n  [QK] Computing W_K^T @ W_Q diagonal-dominance...")
    # Q acts on x to produce queries; K acts on x to produce keys.
    # Their attention interaction is via Q K^T, so the natural product
    # at the input side is W_K^T @ W_Q in R^{d_model x d_model}.
    Wk_T = [w.T for w in Wk]
    M_qk = diag_dominance_matrix(Wq, Wk_T)
    res_qk = evaluate(M_qk, minimize=False)
    trace_qk = trace_signs(Wq, Wk_T)

    print("\n  [VO per-head] aggregated diagonal-dominance over heads...")
    M_vo_ph = per_head_diag_dominance_matrix(Wv, Wo, n_head)
    res_vo_ph = evaluate(M_vo_ph, minimize=False)

    print(f"\n  === RESULTS (W_V <-> W_O) ===")
    print(f"  diag-dom pair_acc:  {res_vo['correct_pairs']}/{n} ({res_vo['pair_acc']:.1%})")
    print(f"  pair_sep:           {res_vo['pair_sep']:+.4f}")
    print(f"  mean correct:       {res_vo['mean_correct']:.4f}")
    print(f"  mean incorrect:     {res_vo['mean_incorrect']:.4f}")
    print(f"  ratio correct/incorrect: {res_vo['mean_correct'] / max(res_vo['mean_incorrect'], 1e-9):.2f}x")
    print(f"  frac negative traces:    {trace_vo['frac_negative']:.1%}")
    print(f"  mean trace:              {trace_vo['mean_trace']:.2f}")

    print(f"\n  === RESULTS (W_Q <-> W_K, control) ===")
    print(f"  diag-dom pair_acc:  {res_qk['correct_pairs']}/{n} ({res_qk['pair_acc']:.1%})")
    print(f"  pair_sep:           {res_qk['pair_sep']:+.4f}")
    print(f"  frac negative traces:    {trace_qk['frac_negative']:.1%}")

    print(f"\n  === RESULTS (W_V <-> W_O per-head) ===")
    print(f"  diag-dom pair_acc:  {res_vo_ph['correct_pairs']}/{n} ({res_vo_ph['pair_acc']:.1%})")
    print(f"  pair_sep:           {res_vo_ph['pair_sep']:+.4f}")

    out = {
        'model': model_name,
        'n_layers': n,
        'd_model': d,
        'n_head': n_head,
        'load_dtype': str(dtype) if dtype is not None else 'float32',
        'random_baseline': 1.0 / n,
        'VO_full':     res_vo,
        'VO_trace':    trace_vo,
        'VO_per_head': res_vo_ph,
        'QK':          res_qk,
        'QK_trace':    trace_qk,
    }
    del Wq, Wk, Wv, Wo, Wk_T
    gc.collect()
    return out, M_vo, M_qk, M_vo_ph


def run_random_baseline(model_name='gpt2'):
    """Sanity check: attention pairing on a randomly initialized GPT-2."""
    from transformers import GPT2LMHeadModel, GPT2Config

    print(f"\n{'='*60}\nRANDOM BASELINE: untrained {model_name}\n{'='*60}")
    cfg = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel(cfg)
    Wq, Wk, Wv, Wo = [], [], [], []
    for block in model.transformer.h:
        W_attn = block.attn.c_attn.weight.detach().float().cpu().numpy().T
        q, k, v = np.split(W_attn, 3, axis=0)
        Wq.append(q); Wk.append(k); Wv.append(v)
        Wo.append(block.attn.c_proj.weight.detach().float().cpu().numpy().T)
    del model; gc.collect()

    M_vo = diag_dominance_matrix(Wv, Wo)
    res = evaluate(M_vo, minimize=False)
    tr = trace_signs(Wv, Wo)
    print(f"  VO pair_acc: {res['correct_pairs']}/{cfg.n_layer} ({res['pair_acc']:.1%})  (chance = {1/cfg.n_layer:.1%})")
    print(f"  frac negative traces: {tr['frac_negative']:.1%}")
    return {
        'model': f'{model_name}-random-init',
        'VO_pair_acc': res['pair_acc'],
        'VO_pair_sep': res['pair_sep'],
        'frac_negative': tr['frac_negative'],
    }


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
def make_figure(all_results, all_matrices):
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 3, figsize=(15, 4.6 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (r, (M_vo, M_qk, M_vo_ph)) in enumerate(zip(all_results, all_matrices)):
        # column 0: VO matrix
        ax = axes[idx, 0]
        im = ax.imshow(M_vo, cmap='magma', aspect='equal')
        ax.set_title(f'{r["model"]} -- $W_O \\circ W_V$ diag-dominance\n'
                     f'pair acc = {r["VO_full"]["pair_acc"]:.0%}, '
                     f'sep = {r["VO_full"]["pair_sep"]:+.2f}')
        ax.set_xlabel(r'$W_O$ index $j$')
        ax.set_ylabel(r'$W_V$ index $i$')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # column 1: trace bar chart
        ax = axes[idx, 1]
        traces = r['VO_trace']['traces']
        colors = ['#2ca02c' if t < 0 else '#d62728' for t in traces]
        ax.bar(range(len(traces)), traces, color=colors, alpha=0.85)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('layer index')
        ax.set_ylabel(r'$\mathrm{tr}(W_O W_V)$')
        ax.set_title(f'{r["model"]} -- VO trace per layer\n'
                     f'{r["VO_trace"]["frac_negative"]:.0%} negative')

        # column 2: method comparison
        ax = axes[idx, 2]
        labels = ['VO\nfull', 'VO\nper-head', 'QK\n(control)', 'chance']
        vals = [
            r['VO_full']['pair_acc'],
            r['VO_per_head']['pair_acc'],
            r['QK']['pair_acc'],
            r['random_baseline'],
        ]
        bar_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', 'gray']
        bars = ax.bar(labels, vals, color=bar_colors, alpha=0.85)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('pair accuracy')
        ax.set_title(f'{r["model"]} -- method comparison')
        ax.axhline(1.0, color='k', lw=0.5, ls='--', alpha=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                    f'{v:.0%}', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig('figures/fig_gpt2_attention_pairing.png',
                dpi=160, bbox_inches='tight')
    fig.savefig('figures/fig_gpt2_attention_pairing.pdf',
                bbox_inches='tight')
    print('\nSaved figures/fig_gpt2_attention_pairing.{png,pdf}')


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Same memory strategy as the MLP script
    MODEL_SPECS = [
        ('gpt2',         None),
        ('gpt2-medium',  None),
        ('gpt2-large',   torch.float16),
        ('gpt2-xl',      torch.float16),
    ]

    all_results = []
    all_matrices = []

    for model_name, dtype in MODEL_SPECS:
        r, M_vo, M_qk, M_vo_ph = run_model(model_name, dtype=dtype)
        all_results.append(r)
        all_matrices.append((M_vo, M_qk, M_vo_ph))

        # incremental save
        with open('results/gpt2_attention_pairing.json', 'w') as f:
            json.dump({
                'pretrained': all_results,
                'random_init_baseline': None,
            }, f, indent=2)

    random_baseline = run_random_baseline('gpt2')

    with open('results/gpt2_attention_pairing.json', 'w') as f:
        json.dump({
            'pretrained': all_results,
            'random_init_baseline': random_baseline,
        }, f, indent=2)
    print('Saved results/gpt2_attention_pairing.json')

    make_figure(all_results, all_matrices)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for r in all_results:
        print(f'\n{r["model"]}:')
        print(f'  VO full     : {r["VO_full"]["correct_pairs"]}/{r["n_layers"]} ({r["VO_full"]["pair_acc"]:.0%})  sep={r["VO_full"]["pair_sep"]:+.3f}  neg_traces={r["VO_trace"]["frac_negative"]:.0%}')
        print(f'  VO per-head : {r["VO_per_head"]["correct_pairs"]}/{r["n_layers"]} ({r["VO_per_head"]["pair_acc"]:.0%})  sep={r["VO_per_head"]["pair_sep"]:+.3f}')
        print(f'  QK control  : {r["QK"]["correct_pairs"]}/{r["n_layers"]} ({r["QK"]["pair_acc"]:.0%})  neg_traces={r["QK_trace"]["frac_negative"]:.0%}')
