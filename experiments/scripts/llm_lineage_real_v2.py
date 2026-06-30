#!/usr/bin/env python3
"""Real-LLM lineage test, v2: real cross-model null from multiple independents.

Upgrades llm_lineage_real.py by replacing the within-model block-shuffle
proxy null with a genuine cross-model null estimated from four real,
independently trained 7B-class checkpoints (none descended from LLaMA-2):

    Negatives (reference vs each, all branch-product-compatible d=4096, L=32):
      - DeepSeek-R1-Distill-Llama-8B   (Llama-3.1-8B weights)
      - Mistral-7B-v0.1                (Mistral, from scratch)
      - Qwen1.5-7B                     (Qwen, from scratch)
      - Yi-6B                          (Yi, from scratch)

    mu_null, sigma_null = mean, std of those four L(reference, independent)
    scores. Descendant z-scores are computed against this real null.

    Descendants (reference vs each):
      - Llama-2-7B-chat                (official RLHF descendant)
      - base + int8 quant / 30% prune / 1% noise   (local transforms)

The within-model block-shuffle null is still printed as a supplementary
chance-level reference, but the verdicts now use the cross-model null.

All signatures are reused from sigs_real_llm/ when present; new negatives
are extracted on first run. Signatures are fp32-renormalized on load
(llm_lineage.load_sig) so L stays in [-1, 1].
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_lineage as core  # noqa: E402

core.SIG_DIR = Path("sigs_real_llm")

REF_TAG = "llama2-7b-base"
REF_REPO = "NousResearch/Llama-2-7b-hf"

DESCENDANTS = [
    ("llama2-7b-chat", "NousResearch/Llama-2-7b-chat-hf", None, "RLHF (official)"),
    ("base+quant-int8", REF_REPO, "quant", "int8 quant"),
    ("base+prune-30", REF_REPO, "prune", "30% prune"),
    ("base+noise-1pct", REF_REPO, "noise", "1% noise"),
]

# Real independents (cross-model null). None descend from LLaMA-2.
NEGATIVES = [
    ("deepseek-r1-distill-llama-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B"),
    ("mistral-7b-v0.1", "mistralai/Mistral-7B-v0.1", "Mistral"),
    ("qwen1.5-7b", "Qwen/Qwen1.5-7B", "Qwen"),
    ("yi-6b", "01-ai/Yi-6B", "Yi"),
]


def ensure_signatures():
    print("== Ensuring signatures (reuse cached, extract missing) ==\n")
    core.extract(REF_TAG, REF_REPO, device="cpu", cleanup=False)
    for tag, repo, kind, _ in DESCENDANTS:
        core.extract(tag, repo, device="cpu", transform=kind, cleanup=False)
    for tag, repo, _ in NEGATIVES:
        core.extract(tag, repo, device="cpu", cleanup=False)


def main():
    ensure_signatures()

    print("\n== Cross-model null (reference vs real independents) ==\n")
    null_scores = []
    for tag, _, family in NEGATIVES:
        L, pb = core.lineage(REF_TAG, tag)
        null_scores.append(L)
        print(f"  {REF_TAG} vs {tag:30s} ({family:12s})  L={L:+.6f}  "
              f"blocks=[{pb.min():+.4f}, {pb.max():+.4f}]")
    null_scores = np.array(null_scores)
    mu = float(null_scores.mean())
    sd = float(max(null_scores.std(ddof=1), 1e-6))
    print(f"\n  cross-model null (n={len(null_scores)}): "
          f"mu={mu:+.6f}  sigma={sd:.6f}  max={null_scores.max():+.6f}")

    # supplementary within-model shuffle null (chance-level reference)
    phi_ref, _ = core.load_sig(REF_TAG)
    rng = np.random.default_rng(0)
    sh = []
    for _ in range(20):
        perm = rng.permutation(len(phi_ref))
        while np.array_equal(perm, np.arange(len(perm))):
            perm = rng.permutation(len(phi_ref))
        sh.append(float(np.diag(phi_ref @ phi_ref[perm].T).mean()))
    print(f"  [suppl.] within-model block-shuffle null (n=20): "
          f"mu={np.mean(sh):+.4f}  sigma={np.std(sh, ddof=1):.4f}")

    print("\n== Verdicts (z against cross-model null) ==\n")
    rows = []

    def verdict(z):
        return ("DESCENDANT" if z > 3.0 else
                "NON-DESCENDANT" if z < 1.645 else "INCONCLUSIVE")

    print("-- descendants --")
    for tag, _, _, rel in DESCENDANTS:
        L, pb = core.lineage(REF_TAG, tag)
        z = (L - mu) / sd
        v = verdict(z)
        print(f"  base vs {tag:30s}  L={L:+.6f}  z={z:+8.1f}  {v:<15s} ({rel})")
        rows.append(("descendant", REF_TAG, tag, "DESCENDANT", L, z, v,
                     float(pb.min()), float(pb.max())))

    print("\n-- independents (held-in null members; z by construction near 0) --")
    for tag, _, family in NEGATIVES:
        L, pb = core.lineage(REF_TAG, tag)
        z = (L - mu) / sd
        v = verdict(z)
        print(f"  base vs {tag:30s}  L={L:+.6f}  z={z:+8.1f}  {v:<15s} ({family})")
        rows.append(("independent", REF_TAG, tag, "NON-DESCENDANT", L, z, v,
                     float(pb.min()), float(pb.max())))

    import csv
    out = Path("results") / "lineage_real_llm_v2.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_a", "model_b", "expected",
                    "L", "z", "verdict", "min_block", "max_block"])
        w.writerows(rows)

    summary = Path("results") / "lineage_real_llm_v2_summary.txt"
    summary.write_text(
        f"Cross-model null (n={len(null_scores)}): mu={mu:.6f} sigma={sd:.6f} "
        f"max={null_scores.max():.6f}\n"
        + "\n".join(f"{r[0]:>11s}  base vs {r[2]:30s}  L={r[4]:+.6f}  "
                    f"z={r[5]:+8.1f}  {r[6]}  (expected {r[3]})" for r in rows)
    )
    print(f"\n-> {out}\n-> {summary}")


if __name__ == "__main__":
    main()
