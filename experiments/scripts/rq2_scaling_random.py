"""Random-init baseline for GPT-2 scaling redo (Issue #43 item 2)."""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rq2_scaling_normalized import delta_J_abs, delta_J_norm, diag_dom  # noqa


def main():
    from transformers import GPT2Config, GPT2LMHeadModel

    out = []
    for name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        t0 = time.time()
        print(f"[{name}] random-init...", flush=True)
        cfg = GPT2Config.from_pretrained(name)
        m = GPT2LMHeadModel(cfg)
        m.eval()
        d = m.config.n_embd
        per = []
        for block in m.transformer.h:
            W1 = block.mlp.c_fc.weight.detach().float().cpu().numpy().T
            W2 = block.mlp.c_proj.weight.detach().float().cpu().numpy().T
            M = W2 @ W1
            per.append({"s": diag_dom(M), "d_J": delta_J_abs(M),
                        "d_Jn": delta_J_norm(M)})
        del m
        gc.collect()
        r = {
            "model": name, "d": d, "n": len(per),
            "mean_s": float(np.mean([p["s"] for p in per])),
            "mean_d_J": float(np.mean([p["d_J"] for p in per])),
            "mean_d_Jn": float(np.mean([p["d_Jn"] for p in per])),
            "median_d_Jn": float(np.median([p["d_Jn"] for p in per])),
        }
        out.append(r)
        print("  done in {:.1f}s: s={:.4f} d_J={:.3f} d_Jn={:.4f}".format(
            time.time() - t0, r["mean_s"], r["mean_d_J"], r["mean_d_Jn"]
        ), flush=True)

    root = Path(__file__).resolve().parents[2]
    outp = root / "results" / "gpt2_scaling_normalized_random.json"
    json.dump(out, open(outp, "w"), indent=2)
    print()
    print("Random-init scaling:")
    for r in out:
        print("  {:<14} d={:>5}  s={:.4f}  d_J={:.4f}  d_Jn={:.4f}".format(
            r["model"], r["d"], r["mean_s"], r["mean_d_J"], r["mean_d_Jn"]
        ))


if __name__ == "__main__":
    main()
