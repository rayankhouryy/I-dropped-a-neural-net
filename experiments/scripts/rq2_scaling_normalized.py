"""Issue #43 item 2: GPT-2 sqrt(d) scaling redo with normalized delta_J.

Extends rq1_jacobian_gpt2.py to all four GPT-2 sizes (small, medium, large, xl).
For each model:
  - extract M_l = W_proj @ W_fc for each MLP block
  - compute s_l = |tr(M_l)| / ||M_l||_F
  - compute delta_J_l        = ||J_l^T J_l - I||_F / sqrt(d)
  - compute delta_J_norm_l   = ||J_l^T J_l / ||J_l||_F^2 - I/d||_F

Saves per-layer JSON and prints scaling fit for s, delta_J, delta_J_norm.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np


def diag_dom(M):
    return float(abs(np.trace(M)) / (np.linalg.norm(M, "fro") + 1e-12))


def delta_J_abs(M):
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    return float(np.linalg.norm(J.T @ J - I, "fro") / np.sqrt(d))


def delta_J_norm(M):
    d = M.shape[0]
    I = np.eye(d, dtype=np.float64)
    J = I + M
    JTJ = J.T @ J
    fro2 = float(np.linalg.norm(J, "fro") ** 2)
    return float(np.linalg.norm(JTJ / fro2 - I / d, "fro"))


def extract_M_list(model_name):
    """Return list of M = W_proj @ W_fc (one per block) and d_model."""
    from transformers import GPT2LMHeadModel

    print(f"  [{model_name}] loading...", flush=True)
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained(model_name, low_cpu_mem_usage=True)
    model.eval()
    d_model = model.config.n_embd
    print(f"  [{model_name}] loaded in {time.time()-t0:.1f}s, d={d_model}, "
          f"n_layers={len(model.transformer.h)}", flush=True)

    Ms = []
    for i, block in enumerate(model.transformer.h):
        # GPT-2 Conv1D stores weight as (in, out); transpose to get (out, in)
        W1 = block.mlp.c_fc.weight.detach().float().cpu().numpy().T   # (4d, d)
        W2 = block.mlp.c_proj.weight.detach().float().cpu().numpy().T # (d, 4d)
        M = W2 @ W1
        Ms.append(M)
    del model
    gc.collect()
    return Ms, d_model


def analyze(model_name):
    Ms, d = extract_M_list(model_name)
    per_layer = []
    t0 = time.time()
    for i, M in enumerate(Ms):
        per_layer.append({
            "layer": i,
            "s": diag_dom(M),
            "delta_J": delta_J_abs(M),
            "delta_J_norm": delta_J_norm(M),
            "trace": float(np.trace(M)),
        })
    print(f"  [{model_name}] metrics done in {time.time()-t0:.1f}s",
          flush=True)
    return {
        "model": model_name,
        "d_model": d,
        "n_layers": len(Ms),
        "per_layer": per_layer,
        "mean_s": float(np.mean([r["s"] for r in per_layer])),
        "median_s": float(np.median([r["s"] for r in per_layer])),
        "mean_delta_J": float(np.mean([r["delta_J"] for r in per_layer])),
        "median_delta_J": float(np.median([r["delta_J"] for r in per_layer])),
        "mean_delta_J_norm": float(np.mean([r["delta_J_norm"] for r in per_layer])),
        "median_delta_J_norm": float(np.median([r["delta_J_norm"] for r in per_layer])),
        "q1_delta_J_norm": float(np.percentile([r["delta_J_norm"] for r in per_layer], 25)),
        "q3_delta_J_norm": float(np.percentile([r["delta_J_norm"] for r in per_layer], 75)),
        "frac_neg_trace": float(np.mean([r["trace"] < 0 for r in per_layer])),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    out_path = root / "results" / "gpt2_scaling_normalized.json"
    out_path.parent.mkdir(exist_ok=True)

    models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    if len(sys.argv) > 1:
        models = sys.argv[1:]

    results = []
    # Incremental save so we don't lose progress if xl OOMs.
    for m in models:
        try:
            r = analyze(m)
            results.append(r)
            json.dump({"per_model": results}, open(out_path, "w"), indent=2)
            print(f"  [{m}] saved partial -> {out_path.name}", flush=True)
        except Exception as e:
            print(f"  [{m}] FAILED: {type(e).__name__}: {e}", flush=True)

    # Report scaling
    print()
    print("=" * 78)
    print("GPT-2 scaling: s, delta_J, delta_J_norm vs d")
    print("=" * 78)
    print("{:<14} {:>5} {:>4} {:>8} {:>8} {:>9} {:>9} {:>10} {:>10}".format(
        "model", "d", "L", "mean_s", "med_s",
        "mean_d_J", "med_d_J", "mean_d_Jn", "med_d_Jn"))
    for r in results:
        print("{:<14} {:>5} {:>4} {:>8.3f} {:>8.3f} {:>9.2f} {:>9.2f} "
              "{:>10.4f} {:>10.4f}".format(
            r["model"], r["d_model"], r["n_layers"],
            r["mean_s"], r["median_s"],
            r["mean_delta_J"], r["median_delta_J"],
            r["mean_delta_J_norm"], r["median_delta_J_norm"]))

    # Fit log-log for each metric
    if len(results) >= 2:
        ds = np.array([r["d_model"] for r in results])
        print()
        print("Power-law fit  metric = a * d^b")
        for key, label in [("mean_s", "s"),
                            ("mean_delta_J", "delta_J"),
                            ("mean_delta_J_norm", "delta_J_norm")]:
            ys = np.array([r[key] for r in results])
            b, a = np.polyfit(np.log(ds), np.log(ys), 1)
            print(f"  {label:14s}  exponent b = {b:+.3f}   (sqrt(d) would be +0.5)")


if __name__ == "__main__":
    main()
