"""Phase 0 of Issue #30: zero-compute lineage detection on cached transformer
weights.

We have safetensors-extractable weights for four production transformers
in the local HuggingFace cache: LLaMA-2-7B, LLaMA-2-7B-chat, Mistral-7B,
and DeepSeek-R1-Distill-Llama-8B. The first two form a verified
descendant pair (LLaMA-2 base -> RLHF chat). The last two share an
identical residual shape (32 layers, d=4096, d_ff=14336, GQA n_kv=8)
but were trained from different ancestors (Mistral pretraining vs
LLaMA-3.1 then R1-distilled), giving us a clean same-shape independent
non-descendant pair.

We compute the SwiGLU MLP branch product M_l = W_down @ W_up per layer,
then apply the residual-signature lineage metric from
:mod:`lineage_detection`. We report:

  - self-similarity sanity checks (L(A, A) = 1)
  - descendant   L(LLaMA-2-base, LLaMA-2-chat)
  - independent  L(Mistral, DeepSeek) [non-descendant control]
  - baselines:   diag-only, raw cosine, Frobenius

Output: results/lineage_phase0_transformers.json
"""
import gc
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment  # noqa: F401 - imported transitively

import lineage_detection as ldet
from transformer_family_pairing import extract_from_safetensors  # noqa: E402


def compute_branch_signatures(W_dict, path='mlp_down_up'):
    """For each layer, compute M_l = W_down @ W_up and reduce to
    (phi, diag_score). Returns three lists: branch products (fp16 to save
    memory), residual signatures (fp32), and diag-dominance scores (fp64).

    'path' selects which branch product to use:
      mlp_down_up   - SwiGLU up-projection path (strongest in our sweep)
      attn_WO_WV    - attention V/O path (requires GQA expansion when n_h>n_kv)
    """
    Ms, phis, ss = [], [], []
    n = W_dict['n_layers']
    n_h = W_dict['n_heads']
    n_kv = W_dict['n_kv_heads']
    head_dim = W_dict['head_dim']

    for i in range(n):
        if path == 'mlp_down_up':
            # M_l = W_down @ W_up , shape (d, d)
            W_up   = W_dict['W_up'][i].astype(np.float32)
            W_down = W_dict['W_down'][i].astype(np.float32)
            M = W_down @ W_up
        elif path == 'attn_WO_WV':
            W_V = W_dict['W_V'][i].astype(np.float32)
            W_O = W_dict['W_O'][i].astype(np.float32)
            if n_h != n_kv:
                repeat = n_h // n_kv
                d = W_V.shape[1]
                W_V = (np.repeat(W_V.reshape(n_kv, head_dim, d), repeat, axis=0)
                       .reshape(n_h * head_dim, d))
            M = W_O @ W_V
        else:
            raise ValueError(f"unknown path: {path}")

        Ms.append(M.astype(np.float16))   # store as fp16 to halve memory
        phis.append(ldet.residual_signature(M).astype(np.float16))
        ss.append(ldet.diag_score(M))

    return Ms, phis, ss


def precomputed_lineage(phis_A, phis_B, ss_A, ss_B, tau_s, eps=1e-12):
    """Lineage score using precomputed signatures + diag scores."""
    L_a, L_b = len(phis_A), len(phis_B)
    G = np.zeros((L_a, L_b), dtype=np.float64)
    # Promote signatures to fp32 for the dot product.
    for i in range(L_a):
        pa = phis_A[i].astype(np.float32)
        for j in range(L_b):
            pb = phis_B[j].astype(np.float32)
            cos = float(pa @ pb)
            gate = min(ss_A[i] / (tau_s + eps),
                       ss_B[j] / (tau_s + eps), 1.0)
            G[i, j] = cos * gate

    row_ind, col_ind = linear_sum_assignment(-G)
    matched = G[row_ind, col_ind]
    return float(matched.mean()), col_ind, G


def precomputed_diag_only(_phis_A, _phis_B, _ss_A, ss_B, **kw) -> float:
    return float(np.mean(ss_B))


def precomputed_raw_cos(Ms_A, Ms_B) -> float:
    """Raw branch-product cosine, identity-aligned."""
    L = min(len(Ms_A), len(Ms_B))
    out = 0.0
    for i in range(L):
        a = Ms_A[i].astype(np.float32).reshape(-1)
        b = Ms_B[i].astype(np.float32).reshape(-1)
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        out += float(a @ b) / (na * nb)
    return out / L


def precomputed_frob_dist(Ms_A, Ms_B) -> float:
    """Negative mean Frobenius distance, identity-aligned (sign flipped)."""
    L = min(len(Ms_A), len(Ms_B))
    out = 0.0
    for i in range(L):
        out -= float(np.linalg.norm(
            Ms_A[i].astype(np.float32) - Ms_B[i].astype(np.float32),
            ord='fro'))
    return out / L


# --------------------------------------------------------------------- main
def main():
    Path('results').mkdir(parents=True, exist_ok=True)
    out_path = Path('results/lineage_phase0_transformers.json')

    # Production checkpoints available locally
    checkpoints = {
        'llama2-7b':       'NousResearch/Llama-2-7b-hf',
        'llama2-7b-chat':  'NousResearch/Llama-2-7b-chat-hf',
        'mistral-7b':      'mistralai/Mistral-7B-v0.1',
        'deepseek-distill': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    }

    # The pairs we care about
    pairs = [
        # label, ref, suspect, hypothesis
        ('llama2-base self',           'llama2-7b',        'llama2-7b',        'self'),
        ('mistral self',               'mistral-7b',       'mistral-7b',       'self'),
        ('llama2 base->chat (RLHF)',   'llama2-7b',        'llama2-7b-chat',   'descendant'),
        ('llama2 chat->base reverse',  'llama2-7b-chat',   'llama2-7b',        'descendant'),
        ('mistral vs deepseek',        'mistral-7b',       'deepseek-distill', 'non_descendant'),
        ('deepseek vs mistral',        'deepseek-distill', 'mistral-7b',       'non_descendant'),
    ]

    # Extract weights for each model + compute signatures.
    # We do this lazily: extract one model, compute its signatures + diag
    # scores, then drop the raw weights before extracting the next.
    cache = {}
    needed_models = set()
    for _, ref, sus, _ in pairs:
        needed_models.add(ref)
        needed_models.add(sus)

    paths = ['mlp_down_up']   # primary path; attn_WO_WV is optional

    extracted = {}  # key -> {'Ms': list, 'phis': list, 'ss': list, 'meta': dict}
    for key in sorted(needed_models):
        repo = checkpoints[key]
        print(f"\n[extract] {key}  ({repo})", flush=True)
        W = extract_from_safetensors(repo)
        meta = {k: W[k] for k in ['n_layers', 'd_model', 'n_heads',
                                   'n_kv_heads', 'head_dim', 'd_ff']}
        per_path = {}
        for p in paths:
            print(f"  [signature] {p}", flush=True)
            Ms, phis, ss = compute_branch_signatures(W, path=p)
            per_path[p] = {'Ms': Ms, 'phis': phis, 'ss': ss}
        extracted[key] = {'paths': per_path, 'meta': meta}
        # Drop raw weights ASAP
        del W
        gc.collect()
        print(f"  done. {key} n_layers={meta['n_layers']}, "
              f"d={meta['d_model']}, d_ff={meta['d_ff']}, "
              f"n_kv={meta['n_kv_heads']}", flush=True)

    # Compute tau_s globally from the strongest 'reference' model branches.
    # We use LLaMA-2 base as the canonical reference.
    tau_s_by_path = {}
    for p in paths:
        all_s = []
        for key in ['llama2-7b', 'mistral-7b', 'deepseek-distill']:
            all_s.extend(extracted[key]['paths'][p]['ss'])
        tau_s_by_path[p] = float(min(all_s))
        print(f"\ntau_s ({p}) = {tau_s_by_path[p]:.4f}  "
              f"(min across {len(all_s)} reference branches)")

    # Score every pair
    results = []
    for label, ref_key, sus_key, hyp in pairs:
        print(f"\n[pair] {label}  ({ref_key} vs {sus_key}, hyp={hyp})")
        ref_meta = extracted[ref_key]['meta']
        sus_meta = extracted[sus_key]['meta']
        for p in paths:
            tau = tau_s_by_path[p]
            phis_A = extracted[ref_key]['paths'][p]['phis']
            phis_B = extracted[sus_key]['paths'][p]['phis']
            ss_A   = extracted[ref_key]['paths'][p]['ss']
            ss_B   = extracted[sus_key]['paths'][p]['ss']
            Ms_A   = extracted[ref_key]['paths'][p]['Ms']
            Ms_B   = extracted[sus_key]['paths'][p]['Ms']

            # Skip pairs with mismatched dimensions
            if (ref_meta['d_model'] != sus_meta['d_model']
                    or Ms_A[0].shape != Ms_B[0].shape):
                print(f"  [{p}] SKIP - shape mismatch "
                      f"{Ms_A[0].shape} vs {Ms_B[0].shape}")
                continue

            score_L, perm, G = precomputed_lineage(
                phis_A, phis_B, ss_A, ss_B, tau)
            score_diag = precomputed_diag_only(phis_A, phis_B, ss_A, ss_B)
            score_rawcos = precomputed_raw_cos(Ms_A, Ms_B)
            score_frob = precomputed_frob_dist(Ms_A, Ms_B)

            n_identity = int((perm == np.arange(len(perm))).sum())
            entry = {
                'label':         label,
                'reference':     ref_key,
                'suspect':       sus_key,
                'hypothesis':    hyp,
                'path':          p,
                'tau_s':         tau,
                'lineage':       score_L,
                'diag_only':     score_diag,
                'raw_cos':       score_rawcos,
                'frob_dist':     score_frob,
                'n_identity_perm':   n_identity,
                'n_layers':      len(perm),
                'matched_score_per_layer': G[np.arange(len(perm)), perm].tolist(),
            }
            results.append(entry)
            print(f"  [{p}] lineage={score_L:+.4f}  diag_only={score_diag:.3f}  "
                  f"raw_cos={score_rawcos:+.4f}  frob_d={score_frob:+.3e}  "
                  f"id_perm={n_identity}/{len(perm)}")

    # Calibration: z-score the descendant against the non-descendant
    # baseline (we have one non-descendant pair to use as null with n=1
    # so this is a sanity check only).
    z_by_path = {}
    for p in paths:
        desc = [r['lineage'] for r in results
                if r['hypothesis'] == 'descendant' and r['path'] == p]
        nondesc = [r['lineage'] for r in results
                   if r['hypothesis'] == 'non_descendant' and r['path'] == p]
        if desc and nondesc:
            mu = float(np.mean(nondesc))
            sd = float(np.std(nondesc, ddof=0))
            z = [(d - mu) / (sd + 1e-12) for d in desc]
            z_by_path[p] = {
                'descendant_mean_L': float(np.mean(desc)),
                'nondesc_mean_L':    mu,
                'nondesc_std_L':     sd,
                'descendant_z_scores': z,
                'effective_separation':
                    float(np.mean(desc) - np.mean(nondesc)),
            }

    out = {
        'description': 'Phase 0 lineage detection on cached production transformers',
        'paths_tested': paths,
        'tau_s_by_path': tau_s_by_path,
        'pairs': results,
        'calibration_by_path': z_by_path,
        'checkpoints': checkpoints,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Headline
    print("\n=== HEADLINE ===")
    for p in paths:
        print(f"\n[{p}]")
        for r in results:
            if r['path'] == p:
                tag = ('SELF      ' if r['hypothesis'] == 'self'
                       else 'DESCENDANT' if r['hypothesis'] == 'descendant'
                       else 'INDEPENDENT')
                print(f"  {tag} {r['label']:<32s}  "
                      f"lineage={r['lineage']:+.4f}  "
                      f"diag={r['diag_only']:.3f}  "
                      f"raw_cos={r['raw_cos']:+.4f}")
        if p in z_by_path:
            c = z_by_path[p]
            print(f"  >>> descendant z over independent null: "
                  f"mean L_desc={c['descendant_mean_L']:.4f}, "
                  f"L_indep={c['nondesc_mean_L']:.4f}, "
                  f"sep={c['effective_separation']:+.4f}")


if __name__ == '__main__':
    main()
