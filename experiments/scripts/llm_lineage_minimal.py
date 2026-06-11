#!/usr/bin/env python3
"""Minimal real-LLM lineage test using only models already cached locally.

Reference: NousResearch/Llama-2-7b-chat-hf (32 layers, d=4096, SwiGLU MLP).
Descendants: 3 local transforms (int8 quant, 30% prune, 1% noise).
Negative: deepseek-ai/DeepSeek-R1-Distill-Llama-8B (Llama-3 8B arch:
  32 layers, d=4096, SwiGLU MLP -- same shape so signatures are directly
  comparable, but a completely different pre-training family / lineage).

This gives 4 real-LLM lineage data points with no additional downloads
beyond what is already in the HuggingFace cache. The point is to show
that the centered residual signature works on a real 7B foundation
model -- both surviving the paper's transformation grid at scale, and
cleanly rejecting a real same-shape independent.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_lineage as core  # noqa: E402

# ---- overrides ---------------------------------------------------------
core.REFERENCE = ("llama2-7b-chat", "NousResearch/Llama-2-7b-chat-hf")
core.DESCENDANTS = []  # no remote descendants in this minimal test
core.NEGATIVES = [
    ("deepseek-r1-distill-llama-8b",
     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
]
core.LOCAL_TRANSFORMS = {
    "chat+quant-int8": "quant",
    "chat+prune-30":   "prune",
    "chat+noise-1pct": "noise",
}
core.SIG_DIR = Path("sigs_minimal")


def main():
    print("== Extracting signatures (using cached HF blobs) ==\n")
    core.extract(*core.REFERENCE, device="cpu", cleanup=False)
    for tag, repo in core.NEGATIVES:
        core.extract(tag, repo, device="cpu", cleanup=False)
    for tag, kind in core.LOCAL_TRANSFORMS.items():
        core.extract(tag, core.REFERENCE[1], device="cpu",
                     transform=kind, cleanup=False)

    print("\n== Pairwise lineage scores ==\n")
    ref = core.REFERENCE[0]
    rows = []

    # null: reference vs one real independent (single sample, so we'll
    # also build a within-checkpoint block-shuffle null for context).
    L_indep, pb_indep = core.lineage(ref, core.NEGATIVES[0][0])
    print(f"[indep]    {ref} vs {core.NEGATIVES[0][0]:32s}  L={L_indep:+.4f}")
    rows.append(("independent", ref, core.NEGATIVES[0][0], "NON-DESCENDANT",
                 L_indep, float(pb_indep.min()), float(pb_indep.max())))

    # Within-model block-shuffle null: shuffle the reference's own block
    # signatures and score; gives a sense of chance-level alignment at
    # real-LLM scale (32 blocks, d=4096).
    phi_ref, _ = core.load_sig(ref)
    rng = np.random.default_rng(0)
    shuffles = []
    for _ in range(20):
        perm = rng.permutation(len(phi_ref))
        cos = phi_ref @ phi_ref[perm].T
        shuffles.append(float(np.diag(cos).mean()))
    mu_null = float(np.mean(shuffles))
    sd_null = float(np.std(shuffles, ddof=1))
    print(f"[null]     within-model block-shuffle (n=20)  "
          f"mu={mu_null:+.4f}  sigma={sd_null:.4f}")

    for tag in core.LOCAL_TRANSFORMS:
        L, pb = core.lineage(ref, tag)
        z = (L - mu_null) / max(sd_null, 1e-6)
        print(f"[desc]     {ref} vs {tag:32s}  "
              f"L={L:+.4f}  z={z:+7.1f}  blocks=[{pb.min():+.3f}, {pb.max():+.3f}]")
        rows.append(("descendant-local", ref, tag, "DESCENDANT",
                     L, float(pb.min()), float(pb.max())))

    z_indep = (L_indep - mu_null) / max(sd_null, 1e-6)
    print(f"\n[indep z]  {ref} vs {core.NEGATIVES[0][0]:32s}  z={z_indep:+.2f}")

    import csv
    with open("lineage_scores_minimal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_a", "model_b", "expected",
                    "L", "min_block", "max_block"])
        w.writerows(rows)
    print(f"\n-> lineage_scores_minimal.csv  (null mu={mu_null:.4f}, sigma={sd_null:.4f})")


if __name__ == "__main__":
    main()
