#!/usr/bin/env python3
"""Real-LLM lineage test: base, chat, real independent, plus local transforms.

Pairs evaluated:
    DESCENDANT pairs:
      - (Llama-2-7b-base, Llama-2-7b-chat)            <- the canonical RLHF pair
      - (Llama-2-7b-base, base+quant-int8)
      - (Llama-2-7b-base, base+prune-30)
      - (Llama-2-7b-base, base+noise-1pct)
    NON-DESCENDANT pairs:
      - (Llama-2-7b-base, DeepSeek-R1-Distill-Llama-8B)   real independent
    NULL: within-model block-shuffle of the base reference (n=20).

All five real-LLM models share the 32-layer, d=4096, SwiGLU-MLP shape so
signatures are directly comparable. Llama-3-8B (DeepSeek base) uses the
same MLP shape as Llama-2-7B; this is the strongest readily-cached real
independent for a same-shape rejection test.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_lineage as core  # noqa: E402

REF_TAG = "llama2-7b-base"
REF_REPO = "NousResearch/Llama-2-7b-hf"
CHAT_TAG = "llama2-7b-chat"
CHAT_REPO = "NousResearch/Llama-2-7b-chat-hf"
INDEP_TAG = "deepseek-r1-distill-llama-8b"
INDEP_REPO = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LOCAL = {
    "base+quant-int8": "quant",
    "base+prune-30":   "prune",
    "base+noise-1pct": "noise",
}

core.SIG_DIR = Path("sigs_real_llm")


def main():
    print("== Extracting signatures ==\n")
    core.extract(REF_TAG, REF_REPO, device="cpu", cleanup=False)
    core.extract(CHAT_TAG, CHAT_REPO, device="cpu", cleanup=False)
    core.extract(INDEP_TAG, INDEP_REPO, device="cpu", cleanup=False)
    for tag, kind in LOCAL.items():
        core.extract(tag, REF_REPO, device="cpu", transform=kind, cleanup=False)

    print("\n== Lineage scores ==\n")
    rows = []

    # within-model block-shuffle null on the reference (synthetic but
    # real-scale: 32 blocks at d=4096)
    phi_ref, s_ref = core.load_sig(REF_TAG)
    rng = np.random.default_rng(0)
    shuffles = []
    for _ in range(20):
        perm = rng.permutation(len(phi_ref))
        while np.array_equal(perm, np.arange(len(perm))):
            perm = rng.permutation(len(phi_ref))
        cos = phi_ref @ phi_ref[perm].T
        shuffles.append(float(np.diag(cos).mean()))
    mu_null = float(np.mean(shuffles))
    sd_null = float(np.std(shuffles, ddof=1))
    print(f"[null]  within-model block-shuffle on base (n=20)  "
          f"mu={mu_null:+.4f}  sigma={sd_null:.4f}\n")

    def add(kind, b_tag, expected):
        L, pb = core.lineage(REF_TAG, b_tag)
        z = (L - mu_null) / max(sd_null, 1e-6)
        verdict = ("DESCENDANT" if z > 3.0 else
                   "NON-DESCENDANT" if z < 1.645 else "INCONCLUSIVE")
        print(f"[{kind:>16s}]  base vs {b_tag:30s}  "
              f"L={L:+.4f}  z={z:+7.1f}  {verdict:<14s} "
              f"blocks=[{pb.min():+.3f}, {pb.max():+.3f}]  "
              f"(expected {expected})")
        rows.append((kind, REF_TAG, b_tag, expected, L, z, verdict,
                     float(pb.min()), float(pb.max())))

    add("descendant", CHAT_TAG, "DESCENDANT")
    for tag in LOCAL:
        add("desc-local", tag, "DESCENDANT")
    add("independent", INDEP_TAG, "NON-DESCENDANT")

    # diagonal-dominance s scores per block, mean for context
    print(f"\n[s(M)]  mean diagonal-dominance score on base: "
          f"{s_ref.mean():.4f}  (range [{s_ref.min():.3f}, {s_ref.max():.3f}])")

    import csv
    out = Path("results") / "lineage_real_llm.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_a", "model_b", "expected",
                    "L", "z", "verdict", "min_block", "max_block"])
        w.writerows(rows)

    summary = Path("results") / "lineage_real_llm_summary.txt"
    summary.write_text(
        f"Null mu={mu_null:.4f} sigma={sd_null:.4f}\n"
        + "\n".join(f"{r[0]:>16s}  base vs {r[2]:30s}  L={r[4]:+.4f}  "
                    f"z={r[5]:+7.1f}  {r[6]}  (expected {r[3]})" for r in rows)
    )
    print(f"\n-> {out}")
    print(f"-> {summary}")


if __name__ == "__main__":
    main()
