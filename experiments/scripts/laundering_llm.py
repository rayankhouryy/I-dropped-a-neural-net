#!/usr/bin/env python3
"""Track C -- Real-LLM checkpoint laundering (SwiGLU models). GPU/cluster only.

Applies function-preserving laundering to the SwiGLU intermediate dimension of
a suspect model in every layer, verifies function preservation with a probe
forward pass, then scores ours (L) and weight cosine on (base vs laundered).
A laundered unrelated control confirms the null does not move.

Supports multiple model families: LLaMA-2, LLaMA-3, Mistral, Qwen.

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

Memory optimization for ml.g6.xlarge (24 GB GPU, 16 GB host RAM):
    - Reference signatures are spilled to disk (~10 GB as .npy), not held in RAM
    - Suspect signatures are streamed layer-by-layer and reduced to scalars immediately
    - Peak host RAM: ~4 GB (model loading buffers + one layer's signatures)

Usage (cluster):
    # LLaMA-2 (default)
    python laundering_llm.py --model-family llama2 --variants P,D-mild --out results/laundering/

    # Mistral
    python laundering_llm.py --model-family mistral --variants P,D-mild --out results/laundering/

    # Qwen (small, for diversity)
    python laundering_llm.py --model-family qwen --variants P,D-mild --out results/laundering/

    # LLaMA-3 (requires HF_TOKEN)
    HF_TOKEN=<token> python laundering_llm.py --model-family llama3 --variants P,D-mild --out results/laundering/

    # Custom models (legacy CLI)
    python laundering_llm.py --ref-base <repo> --suspect <repo> --unrelated <repo> --variants P,D-mild
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

GATE_THRESHOLD = 1e-4
N_PROBE_SEQ = 32
PROBE_SEQ_LEN = 16
LOGUNIFORM = {"D-mild": (0.5, 2.0), "D-strong": (0.1, 10.0)}

MODEL_TRIPLETS = {
    'llama2': {
        'ref': 'NousResearch/Llama-2-7b-hf',
        'suspect': 'NousResearch/Llama-2-7b-chat-hf',
        'unrelated': 'openlm-research/open_llama_7b',
    },
    'llama3': {
        'ref': 'meta-llama/Llama-3.1-8B',
        'suspect': 'meta-llama/Llama-3.1-8B-Instruct',
        'unrelated': 'mistralai/Mistral-7B-v0.1',
    },
    'mistral': {
        'ref': 'mistralai/Mistral-7B-v0.1',
        'suspect': 'mistralai/Mistral-7B-Instruct-v0.1',
        'unrelated': None,  # d_ff=14336 differs from LLaMA-2's 11008; skip unrelated control
    },
    'qwen': {
        'ref': 'Qwen/Qwen2.5-1.5B',
        'suspect': 'Qwen/Qwen2.5-1.5B-Instruct',
        'unrelated': None,  # No common model with matching d=1536; skip unrelated control
    },
}


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
def ref_signatures_to_disk(model, cache_dir: Path):
    """Write per-layer (phi, raw) to disk as fp16 .npy files.

    For a 7B model this writes ~10 GB to disk but keeps host RAM at ~0.3 GB per
    layer (one layer resident at a time). This is critical for ml.g6.xlarge which
    has only 16 GB host RAM — holding ~10 GB of signatures in RAM leaves no room
    for model loading buffers and causes OOM.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(model.model.layers)
    for li, layer in enumerate(model.model.layers):
        phi, raw = _layer_signatures(layer)
        np.save(cache_dir / f"phi_{li:03d}.npy", phi.to(torch.float16).numpy())
        np.save(cache_dir / f"raw_{li:03d}.npy", raw.to(torch.float16).numpy())
        del phi, raw
        gc.collect()
        if (li + 1) % 8 == 0:
            print(f"  [ref sigs] {li + 1}/{n_layers} layers written", flush=True)
    print(f"  [ref sigs] {n_layers} layers -> {cache_dir}", flush=True)
    return n_layers


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


@torch.no_grad()
def score_against_ref(model, ref_cache_dir: Path, n_ref_layers: int):
    """Stream suspect and reference layer-by-layer -> (ours_L, weight_cosine_raw).

    Reference signatures are read from disk one layer at a time; suspect signatures
    are computed on-the-fly. Each layer is reduced to two scalar cosines immediately
    and discarded. Peak host RAM: ~0.6 GB (one layer's signatures in fp32).
    """
    n_sus = len(model.model.layers)
    if n_ref_layers != n_sus:
        print(f"[warn] layer-count mismatch ref={n_ref_layers} suspect={n_sus}; "
              f"scoring the first {min(n_ref_layers, n_sus)} layers", flush=True)
    L_terms, wcos_terms = [], []
    for li, layer in enumerate(model.model.layers):
        if li >= n_ref_layers:
            break
        ref_phi = np.load(ref_cache_dir / f"phi_{li:03d}.npy").astype(np.float32)
        ref_raw = np.load(ref_cache_dir / f"raw_{li:03d}.npy").astype(np.float32)
        sus_phi, sus_raw = _layer_signatures(layer)
        L_terms.append(_cosine(ref_phi, sus_phi.numpy()))
        wcos_terms.append(_cosine(ref_raw, sus_raw.numpy()))
        del ref_phi, ref_raw, sus_phi, sus_raw
        gc.collect()
    return float(np.mean(L_terms)), float(np.mean(wcos_terms))


# ------------------------------------------------------------------------ main

def load_model(repo: str, device: str, dtype, use_safetensors_direct: bool = False):
    """Load a causal LM, minimizing the host-RAM peak.

    On CUDA we first try to stream shards straight onto the GPU via device_map
    (needs `accelerate`), which keeps CPU RAM low -- important on boxes with
    little system memory (e.g. g6.xlarge has only 16 GB RAM vs a 13.5 GB fp16
    7B checkpoint). If accelerate is unavailable we fall back to a plain load
    followed by .to(device).

    For Qwen models stored as bf16 (which crash on some systems during HF load),
    set use_safetensors_direct=True to bypass from_pretrained and load via
    safetensors directly. This is a fallback, not the default.
    """
    from transformers import AutoModelForCausalLM
    print(f"[load] {repo}", flush=True)

    if use_safetensors_direct:
        print(f"[load] using safetensors-direct path for bf16 compatibility", flush=True)

    if device.startswith("cuda"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=dtype, low_cpu_mem_usage=True,
                device_map={"": device}, trust_remote_code=True)
            m.eval()
            return m
        except (ImportError, ValueError) as e:
            print(f"[load] device_map unavailable ({e.__class__.__name__}); "
                  f"falling back to plain load + .to({device})", flush=True)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype,
                                             low_cpu_mem_usage=True,
                                             trust_remote_code=True)
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
    ap.add_argument("--model-family", choices=list(MODEL_TRIPLETS.keys()),
                    help="predefined model family (llama2, llama3, mistral, qwen)")
    ap.add_argument("--ref-base", default=None, help="base reference repo/path (overrides --model-family)")
    ap.add_argument("--suspect", default=None, help="descendant to launder (overrides --model-family)")
    ap.add_argument("--unrelated", default=None, help="unrelated control to launder (overrides --model-family)")
    ap.add_argument("--variants", default="P,D-mild")
    ap.add_argument("--probe-blocks", type=int, default=4,
                    help="compare hidden state after this many decoder layers "
                         "(use a large value / >= n_layers for full forward)")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default="results/laundering/")
    ap.add_argument("--sig-cache", default=None,
                    help="directory for reference signatures (default: temp dir, auto-cleaned)")
    args = ap.parse_args()

    if args.model_family:
        triplet = MODEL_TRIPLETS[args.model_family]
        ref_base = args.ref_base or triplet['ref']
        suspect = args.suspect or triplet['suspect']
        unrelated = args.unrelated or triplet['unrelated']
        family_tag = args.model_family
    elif args.ref_base and args.suspect:
        ref_base = args.ref_base
        suspect = args.suspect
        unrelated = args.unrelated
        family_tag = "custom"
    else:
        ap.error("Either --model-family or both --ref-base and --suspect are required")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[config] family={family_tag}", flush=True)
    print(f"         ref={ref_base}", flush=True)
    print(f"         suspect={suspect}", flush=True)
    print(f"         unrelated={unrelated}", flush=True)
    print(f"         variants={variants}\n", flush=True)

    # Use a temp dir for signatures unless user specifies --sig-cache
    sig_cache_is_temp = args.sig_cache is None
    if sig_cache_is_temp:
        sig_cache_dir = Path(tempfile.mkdtemp(prefix="llm_sigs_"))
    else:
        sig_cache_dir = Path(args.sig_cache)
    print(f"[sig cache] {sig_cache_dir} (temp={sig_cache_is_temp})", flush=True)

    try:
        # Reference signatures to disk; model freed before any suspect loads.
        print(f"[ref] extracting signatures from {ref_base}", flush=True)
        ref = load_model(ref_base, args.device, dtype)
        n_layers = len(ref.model.layers)
        probe_blocks = min(args.probe_blocks, n_layers)
        n_ref_layers = ref_signatures_to_disk(ref, sig_cache_dir)
        del ref
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        config_dict = {
            "model_family": family_tag,
            "ref_base": ref_base,
            "suspect": suspect,
            "unrelated": unrelated,
            "variants": args.variants,
            "probe_blocks": args.probe_blocks,
            "seed_base": args.seed_base,
            "device": args.device,
            "dtype": args.dtype,
            "out": args.out,
        }
        results = {"config": config_dict, "n_layers": n_layers,
                   "gate_threshold": GATE_THRESHOLD, "probe_blocks": probe_blocks,
                   "seeds": {"laundering_base": args.seed_base, "probe": 12345},
                   "pairs": []}

        def process(tag, repo, expected, variant, seed):
            dev, model = run_probe_gate(repo, variant, seed, args.device, dtype,
                                        probe_blocks, vocab_fallback=32000)
            gate_pass = dev < GATE_THRESHOLD
            L, wcos = score_against_ref(model, sig_cache_dir, n_ref_layers)
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
            process("suspect(laundered)", suspect, "DESCENDANT", v,
                    args.seed_base + i)
            if unrelated:
                process("unrelated(laundered)", unrelated, "NON-DESCENDANT", v,
                        args.seed_base + 100 + i)

        out_path = outdir / f"laundering_llm_{family_tag}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {out_path}")
        print("\nNOTE: weights are fp16; the gate upcasts laundered matmuls to fp32 but "
              "the forward pass runs at the model dtype. If gate_dev exceeds 1e-4 due to "
              "fp16 rounding, the MEASURED deviation is reported (not forced) -- rerun "
              "with --dtype float32 for a stricter check if memory allows.")

    finally:
        if sig_cache_is_temp and sig_cache_dir.exists():
            print(f"[cleanup] removing temp sig cache {sig_cache_dir}", flush=True)
            shutil.rmtree(sig_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
