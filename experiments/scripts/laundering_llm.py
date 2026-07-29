#!/usr/bin/env python3
"""Track C -- Real-LLM checkpoint laundering (Llama-2 SwiGLU). GPU/cluster only.

Applies function-preserving laundering to the SwiGLU intermediate dimension of
a suspect model (Llama-2-7b-chat) in every layer, verifies function preservation
with a probe forward pass, then scores ours (L) and weight cosine on
(base vs laundered-chat). A laundered unrelated control (base vs laundered
OpenLLaMA-7B) confirms the null does not move.

SwiGLU MLP:  y = W_down @ ( silu(W_gate @ x) (.) (W_up @ x) ),  intermediate dim I.
    W_gate, W_up : (I, d)      W_down : (d, I)

Function-preserving operators on the I hidden units (verified algebra):
    P (permutation pi):  rows of W_gate AND W_up by pi; cols of W_down by pi.
        silu and (.) are elementwise so they commute with pi; W_down pi^T pi = I.
    D (positive scale d): rows of W_up by d; cols of W_down by 1/d; GATE UNTOUCHED.
        gate feeds silu (nonlinear) so it must not be scaled; the up-branch scale
        d cancels against the down-branch 1/d. D-mild = LogUniform[0.5, 2].

M-based signature M_l = W_down @ W_up is exactly invariant to both (as in the MLP
track), so ours is laundering-invariant; raw weight cosine collapses under P.

This machine has no Llama weights and no CUDA -- run on the cluster where the
Llama-2 / OpenLLaMA weights are downloaded. The forward passes use transformers
so attention/RoPE/RMSNorm are exact and only MLP weights are perturbed.

Usage (cluster):
    python laundering_llm.py \
        --ref-base  NousResearch/Llama-2-7b-hf \
        --suspect   NousResearch/Llama-2-7b-chat-hf \
        --unrelated openlm-research/open_llama_7b \
        --variants P,D-mild --probe-blocks 4 --out results/laundering/
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

GATE_THRESHOLD = 1e-4
N_PROBE_SEQ = 32
PROBE_SEQ_LEN = 16
LOGUNIFORM = {"D-mild": (0.5, 2.0), "D-strong": (0.1, 10.0)}


def _rng(*ints):
    return np.random.default_rng([int(x) for x in ints])


# ------------------------------------------------------------------ laundering

@torch.no_grad()
def launder_layer_mlp(mlp, variant: str, seed: int, layer: int):
    """Launder one decoder layer's SwiGLU MLP in place (function-preserving).

    Works on gate_proj/up_proj/down_proj .weight (transformers Llama/OpenLLaMA).
    Computes in fp32 then casts back to the parameters' dtype.
    """
    Wg = mlp.gate_proj.weight
    Wu = mlp.up_proj.weight
    Wd = mlp.down_proj.weight
    I = Wu.shape[0]
    dev, dt = Wu.device, Wu.dtype

    if variant == "P":
        perm = torch.as_tensor(_rng(seed, 0, layer).permutation(I),
                               dtype=torch.long, device=dev)
        Wg.copy_(Wg[perm, :])
        Wu.copy_(Wu[perm, :])
        Wd.copy_(Wd[:, perm])
    elif variant in ("D-mild", "D-strong"):
        lo, hi = LOGUNIFORM[variant]
        d_np = np.exp(_rng(seed, 1, layer).uniform(np.log(lo), np.log(hi), I))
        d = torch.as_tensor(d_np, device=dev, dtype=torch.float32)
        Wu.copy_((Wu.to(torch.float32) * d[:, None]).to(dt))          # up rows * d
        Wd.copy_((Wd.to(torch.float32) * (1.0 / d)[None, :]).to(dt))  # down cols / d
        # gate untouched (feeds nonlinear silu)
    elif variant == "PD":
        launder_layer_mlp(mlp, "P", seed, layer)
        launder_layer_mlp(mlp, "D-strong", seed, layer)
    else:
        raise ValueError(f"unknown variant {variant}")


@torch.no_grad()
def launder_model(model, variant: str, seed: int):
    """Launder every decoder layer's MLP in place."""
    for li, layer in enumerate(model.model.layers):
        launder_layer_mlp(layer.mlp, variant, seed, li)


# ------------------------------------------------------------ probe / gate

def make_probe_ids(vocab_size: int, n: int = N_PROBE_SEQ,
                   seqlen: int = PROBE_SEQ_LEN, seed: int = 12345):
    g = _rng(seed)
    ids = g.integers(1, vocab_size, size=(n, seqlen))  # avoid 0 = pad/bos edge
    return torch.as_tensor(ids, dtype=torch.long)


@torch.no_grad()
def hidden_after_blocks(model, input_ids, n_blocks: int) -> torch.Tensor:
    """Residual-stream hidden state after `n_blocks` decoder layers, fp32.

    Uses output_hidden_states so attention/RoPE/RMSNorm are exact; hidden_states
    is a tuple (embed, after_layer_0, after_layer_1, ...), so index n_blocks.
    """
    out = model(input_ids.to(model.device), output_hidden_states=True,
                use_cache=False)
    hs = out.hidden_states[n_blocks]
    return hs.detach().to(torch.float32).cpu()


# ------------------------------------------------------- signatures / scoring

@torch.no_grad()
def _layer_signatures(layer):
    """(phi, raw) as fp32 1-D CPU tensors for one decoder layer's SwiGLU MLP.

    phi = vec(M - (tr/d) I) / ||.||   with M = W_down @ W_up  (ours; M-based)
    raw = [W_gate; W_up; W_down] flattened                    (raw weight cosine)

    The fp32 upcast + matmul happen on-device; only the two 1-D results move to
    CPU. Shared by the reference extractor and the streaming suspect scorer so
    both compute byte-identical signatures.
    """
    Wg = layer.mlp.gate_proj.weight.detach().to(torch.float32)
    Wu = layer.mlp.up_proj.weight.detach().to(torch.float32)
    Wd = layer.mlp.down_proj.weight.detach().to(torch.float32)
    M = Wd @ Wu                          # (d, d)
    d = M.shape[0]
    tr = torch.trace(M)
    R = M - (tr / d) * torch.eye(d, dtype=M.dtype, device=M.device)
    phi = (R / torch.linalg.matrix_norm(R)).flatten().cpu()
    raw = torch.cat([Wg.flatten(), Wu.flatten(), Wd.flatten()]).cpu()
    del Wg, Wu, Wd, M, R
    return phi, raw


@torch.no_grad()
def ref_signatures(model):
    """Per-layer (phi, raw) for the REFERENCE, stored as fp16 numpy to save host RAM.

    For a 7B model this is ~1 GB (phis) + ~8.7 GB (raw) in fp16, versus ~19 GB in
    fp32 -- the difference between fitting and OOMing a 16 GB host (e.g. an
    ml.g6.xlarge, which pairs a 24 GB L4 GPU with only 16 GB system RAM). The
    suspect is never stored in full; see score_against_ref.
    """
    phis, raws = [], []
    for layer in model.model.layers:
        phi, raw = _layer_signatures(layer)
        phis.append(phi.to(torch.float16).numpy())
        raws.append(raw.to(torch.float16).numpy())
        del phi, raw
        gc.collect()
    return phis, raws


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


@torch.no_grad()
def score_against_ref(model, ref_phis, ref_raws):
    """Stream the suspect layer-by-layer -> (ours_L, weight_cosine_raw).

    Each layer is reduced to two scalar cosines immediately and discarded, so the
    host-RAM peak is the (fp16) reference plus a single fp32 layer (~0.6 GB), not
    a second full 7B signature set. Identity-aligned mean over layers (P/D leave
    M invariant, so ours needs no unit alignment; the raw cosine does not either,
    which is exactly why it collapses under P).
    """
    n_ref, n_sus = len(ref_phis), len(model.model.layers)
    if n_ref != n_sus:
        print(f"[warn] layer-count mismatch ref={n_ref} suspect={n_sus}; "
              f"scoring the first {min(n_ref, n_sus)} layers", flush=True)
    L_terms, wcos_terms = [], []
    for li, layer in enumerate(model.model.layers):
        if li >= n_ref:
            break
        phi, raw = _layer_signatures(layer)
        L_terms.append(_cosine(ref_phis[li].astype(np.float32), phi.numpy()))
        wcos_terms.append(_cosine(ref_raws[li].astype(np.float32), raw.numpy()))
        del phi, raw
        gc.collect()
    return float(np.mean(L_terms)), float(np.mean(wcos_terms))


# ------------------------------------------------------------------------ main

def load_model(repo: str, device: str, dtype):
    """Load a causal LM, minimizing the host-RAM peak.

    On CUDA we first try to stream shards straight onto the GPU via device_map
    (needs `accelerate`), which keeps CPU RAM low -- important on boxes with
    little system memory (e.g. g6.xlarge has only 16 GB RAM vs a 13.5 GB fp16
    7B checkpoint). If accelerate is unavailable we fall back to a plain load
    followed by .to(device).
    """
    from transformers import AutoModelForCausalLM
    print(f"[load] {repo}", flush=True)
    if device.startswith("cuda"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=dtype, low_cpu_mem_usage=True,
                device_map={"": device})
            m.eval()
            return m
        except (ImportError, ValueError) as e:
            print(f"[load] device_map unavailable ({e.__class__.__name__}); "
                  f"falling back to plain load + .to({device})", flush=True)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype,
                                             low_cpu_mem_usage=True)
    m.to(device)
    m.eval()
    return m


def run_probe_gate(repo, variant, seed, device, dtype, n_blocks, vocab_fallback):
    """Load ONE copy, probe it, launder IN PLACE, probe again.

    Returns (max_deviation, laundered_model). Laundering is in-place and exactly
    function-preserving, so a single resident model suffices: we record the
    block-`n_blocks` hidden states before laundering, launder in place, and
    record them again after. Peak GPU memory is therefore ONE model (~13.5 GB
    for a 7B in fp16 -- fits a 24 GB L4), not two copies as before; this also
    halves the number of 7B loads per pair.
    """
    model = load_model(repo, device, dtype)
    vocab = getattr(model.config, "vocab_size", vocab_fallback)
    probe = make_probe_ids(vocab, seed=12345)
    h_ref = hidden_after_blocks(model, probe, n_blocks)   # before laundering

    launder_model(model, variant, seed)                   # function-preserving, in place
    h_l = hidden_after_blocks(model, probe, n_blocks)      # after laundering

    dev = float((h_ref - h_l).abs().max().item())
    del h_ref, h_l
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return dev, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-base", required=True, help="base reference repo/path")
    ap.add_argument("--suspect", required=True, help="descendant to launder (chat)")
    ap.add_argument("--unrelated", default=None, help="unrelated control to launder")
    ap.add_argument("--variants", default="P,D-mild")
    ap.add_argument("--probe-blocks", type=int, default=4,
                    help="compare hidden state after this many decoder layers "
                         "(use a large value / >= n_layers for full forward)")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default="results/laundering/")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reference signatures once (unlaundered base); model freed before any suspect loads.
    ref = load_model(args.ref_base, args.device, dtype)
    n_layers = len(ref.model.layers)
    probe_blocks = min(args.probe_blocks, n_layers)
    ref_phis, ref_raw = ref_signatures(ref)
    del ref
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    results = {"config": vars(args), "n_layers": n_layers,
               "gate_threshold": GATE_THRESHOLD, "probe_blocks": probe_blocks,
               "seeds": {"laundering_base": args.seed_base, "probe": 12345},
               "pairs": []}

    def process(tag, repo, expected, variant, seed):
        dev, model = run_probe_gate(repo, variant, seed, args.device, dtype,
                                    probe_blocks, vocab_fallback=32000)
        gate_pass = dev < GATE_THRESHOLD
        L, wcos = score_against_ref(model, ref_phis, ref_raw)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        rec = {"tag": tag, "repo": repo, "expected": expected, "variant": variant,
               "gate_max_deviation": dev, "gate_pass": bool(gate_pass),
               "ours_L": L, "weight_cosine_raw": wcos}
        results["pairs"].append(rec)
        print(f"[{tag:22s} {variant:7s}] gate_dev={dev:.3e} "
              f"{'PASS' if gate_pass else 'FAIL(reported)'}  "
              f"ours_L={L:+.4f}  weight_cosine_raw={wcos:+.4f}", flush=True)

    for i, v in enumerate(variants):
        process("chat(laundered)", args.suspect, "DESCENDANT", v,
                args.seed_base + i)
        if args.unrelated:
            process("unrelated(laundered)", args.unrelated, "NON-DESCENDANT", v,
                    args.seed_base + 100 + i)

    out_path = outdir / "laundering_llm.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print("\nNOTE: weights are fp16; the gate upcasts laundered matmuls to fp32 but "
          "the forward pass runs at the model dtype. If gate_dev exceeds 1e-4 due to "
          "fp16 rounding, the MEASURED deviation is reported (not forced) -- rerun "
          "with --dtype float32 for a stricter check if memory allows.")


if __name__ == "__main__":
    main()
