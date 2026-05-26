"""
Issue #10: Whisper encoder/decoder pairing test.

Adds the speech-recognition modality to the cross-architecture fingerprint
sweep (transformer-family table). Whisper is a speech-to-text model whose
encoder and decoder are standard pre-LN transformer blocks with GELU
activation, full bidirectional attention in the encoder, and causal
attention in the decoder.

Each Whisper block has:
  fc1:        (d_ff, d)        -- up-projection
  fc2:        (d,    d_ff)     -- down-projection
  q_proj,k_proj,v_proj: (d,d)  -- attention input
  out_proj:    (d, d)          -- attention output

We score:
  MLP path : M = fc2 . fc1           expected positive trace
  V<->O    : M = out_proj . v_proj   expected positive trace
  Q<->K    : M = q_proj . k_proj^T   architecturally unpaired control

Three model sizes:
  openai/whisper-tiny   (39M params, encoder 4 layers, decoder 4 layers, d=384)
  openai/whisper-base   (74M params, 6/6, d=512)
  openai/whisper-small  (244M params, 12/12, d=768)

Each model gets a random-init baseline (3 seeds) to confirm the structure
emerges from training rather than initialization.

Outputs:
  results/whisper_pairing.json
"""

import json, gc, os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

Path("results").mkdir(exist_ok=True)
torch.set_grad_enabled(False)

N_SEEDS = 3
DEVICE  = torch.device("cpu")


# --------------------------------------------------------------- scoring
def diag_dominance_matrix(A_list, B_list):
    n = len(A_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = B_list[j] @ A_list[i]
            tr = abs(np.trace(P))
            fr = np.linalg.norm(P, "fro") + 1e-12
            M[i, j] = tr / fr
    return M


def evaluate(M):
    n = M.shape[0]
    _, col = linear_sum_assignment(-M)
    pair_acc = float((col == np.arange(n)).mean())
    diag = np.diag(M)
    off  = M[~np.eye(n, dtype=bool)]
    off_max = (M - np.diag(diag)).max(axis=1)
    return {
        "n": n,
        "chance": 1.0 / n,
        "pair_acc":        pair_acc,
        "acc_over_chance": pair_acc * n,
        "pair_sep":        float((diag - off_max).min()),
        "mean_correct":    float(diag.mean()),
        "mean_incorrect":  float(off.mean()),
    }


def auc_correct_vs_incorrect(M):
    diag = np.diag(M)
    off  = M[~np.eye(M.shape[0], dtype=bool)]
    pos = diag[:, None]; neg = off[None, :]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    return float(wins / (pos.size * neg.size))


def trace_signs(A_list, B_list):
    traces = np.array([float(np.trace(B_list[i] @ A_list[i])) for i in range(len(A_list))])
    return {
        "mean_trace": float(traces.mean()),
        "frac_negative": float((traces < 0).mean()),
        "traces": traces.tolist(),
    }


# --------------------------------------------------------------- whisper extract
def extract_block(layer):
    """Extract (Q, K, V, O, fc1, fc2) numpy weights from a single Whisper
    encoder/decoder layer (HF transformers naming)."""
    q = layer.self_attn.q_proj.weight.detach().float().cpu().numpy()
    k = layer.self_attn.k_proj.weight.detach().float().cpu().numpy()
    v = layer.self_attn.v_proj.weight.detach().float().cpu().numpy()
    o = layer.self_attn.out_proj.weight.detach().float().cpu().numpy()
    w1 = layer.fc1.weight.detach().float().cpu().numpy()  # (d_ff, d)
    w2 = layer.fc2.weight.detach().float().cpu().numpy()  # (d,    d_ff)
    return q, k, v, o, w1, w2


def score_layers(layers, label):
    Qs, Ks, Vs, Os, W1s, W2s = [], [], [], [], [], []
    for layer in layers:
        q, k, v, o, w1, w2 = extract_block(layer)
        Qs.append(q); Ks.append(k); Vs.append(v); Os.append(o)
        W1s.append(w1); W2s.append(w2)

    # MLP: M = W_2 . W_1 (paired)
    M_mlp = diag_dominance_matrix(W1s, W2s)
    # V <-> O: M = O . V (paired)
    M_vo  = diag_dominance_matrix(Vs, Os)
    # Q <-> K: M = Q . K^T (control, architecturally unpaired)
    Kt = [k.T for k in Ks]
    M_qk  = diag_dominance_matrix(Kt, Qs)

    out = {
        "label":    label,
        "n_layers": len(layers),
        "d_model":  W1s[0].shape[1],
        "d_ff":     W1s[0].shape[0],
        "mlp":   {**evaluate(M_mlp), "auc": auc_correct_vs_incorrect(M_mlp),
                  "trace": trace_signs(W1s, W2s)},
        "vo":    {**evaluate(M_vo),  "auc": auc_correct_vs_incorrect(M_vo),
                  "trace": trace_signs(Vs, Os)},
        "qk":    {**evaluate(M_qk),  "auc": auc_correct_vs_incorrect(M_qk),
                  "trace": trace_signs(Kt, Qs)},
    }
    return out


def randomize_weights(model, seed):
    """Re-initialize every Linear weight in the model in-place with the
    same seed, matching shapes -- used for random-init baseline."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            d_out, d_in = m.weight.shape
            std = (2.0 / d_in) ** 0.5   # Kaiming-uniform-ish
            m.weight.data = torch.randn(d_out, d_in, generator=g) * std
            if m.bias is not None:
                m.bias.data.zero_()
    return model


# --------------------------------------------------------------- runner
def run_model(hf_name, short_name):
    print(f"\n{'='*78}\n{short_name} ({hf_name})\n{'='*78}")
    from transformers import WhisperModel
    model = WhisperModel.from_pretrained(hf_name, torch_dtype=torch.float32)
    model.eval()

    enc_layers = list(model.encoder.layers)
    dec_layers = list(model.decoder.layers)

    result = {
        "model":   short_name,
        "hf_name": hf_name,
        "trained": {
            "encoder": score_layers(enc_layers, f"{short_name}-encoder-trained"),
            "decoder": score_layers(dec_layers, f"{short_name}-decoder-trained"),
        },
        "random_baseline": {"encoder": [], "decoder": []},
    }

    for path, layers, key in [
        ("encoder", enc_layers, "encoder"),
        ("decoder", dec_layers, "decoder"),
    ]:
        r = result["trained"][key]
        print(f"  {path} (n={r['n_layers']}, d={r['d_model']}, d_ff={r['d_ff']}):")
        for tag in ["mlp", "vo", "qk"]:
            s = r[tag]
            print(f"    {tag:3s}  pair_acc={s['pair_acc']:.0%} ({s['acc_over_chance']:.1f}x)  "
                  f"sep={s['pair_sep']:+.3f}  AUC={s['auc']:.3f}  "
                  f"neg_tr={s['trace']['frac_negative']:.0%}")

    # Random-init baselines (re-init in place; cheaper than re-instantiating)
    print(f"  random baselines (n={N_SEEDS} seeds)")
    for seed in range(N_SEEDS):
        randomize_weights(model, seed=seed)
        result["random_baseline"]["encoder"].append(
            score_layers(list(model.encoder.layers), f"{short_name}-encoder-rand{seed}"))
        result["random_baseline"]["decoder"].append(
            score_layers(list(model.decoder.layers), f"{short_name}-decoder-rand{seed}"))

    for path in ["encoder", "decoder"]:
        for tag in ["mlp", "vo", "qk"]:
            accs = [r[tag]["pair_acc"] for r in result["random_baseline"][path]]
            seps = [r[tag]["pair_sep"] for r in result["random_baseline"][path]]
            print(f"    rand-{path:7s} {tag:3s}  mean pair_acc={np.mean(accs):.0%}  "
                  f"mean sep={np.mean(seps):+.3f}")

    del model; gc.collect()
    return result


def strip_matrices(d):
    """Recursively drop any np.ndarray to keep JSON small."""
    if isinstance(d, dict):
        return {k: strip_matrices(v) for k, v in d.items() if not isinstance(v, np.ndarray)}
    if isinstance(d, list):
        return [strip_matrices(x) for x in d]
    return d


def main():
    targets = [
        ("openai/whisper-tiny",  "Whisper-tiny"),
        ("openai/whisper-base",  "Whisper-base"),
        ("openai/whisper-small", "Whisper-small"),
    ]
    results = {}
    for hf_name, short in targets:
        try:
            results[short] = run_model(hf_name, short)
        except Exception as e:
            print(f"  !! failed on {short}: {e!r}")
            results[short] = {"error": repr(e), "hf_name": hf_name}

    out_path = Path("results/whisper_pairing.json")
    out_path.write_text(json.dumps(strip_matrices(results), indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
