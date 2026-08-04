#!/usr/bin/env python3
"""
Real-LLM lineage validation for the residual-signature provenance method.

Computes, for each model, the per-layer SwiGLU branch product
    M_l = W_down^(l) @ W_up^(l)   in R^{d x d}
then the centered residual signature
    R_l = M_l - (tr(M_l)/d) I,   phi_l = vec(R_l)/||vec(R_l)||
and the diagonal-dominance score s_l = |tr(M_l)| / ||M_l||_F.

Lineage score L(A,B) = mean_l <phi_l^A, phi_l^{pi(l)}^B>, with alignment pi
either identity (default, blocks are in order in real checkpoints) or
Hungarian on the 32x32 signature-cosine matrix (--hungarian, cheap since
signatures are precomputed).

Memory: never loads a full model. Streams individual tensors out of
safetensors shards via safe_open; peak RAM ~1.5 GB per layer in fp32.

Disk: each 7B model is ~13 GB to download. With --cleanup the shard cache
is deleted after signature extraction, so peak disk = one model + ~1 GB
of signatures per model already processed.

Usage:
    pip install torch safetensors huggingface_hub scipy numpy
    export HF_TOKEN=...            # only needed for gated meta-llama repos
    python llm_lineage.py extract --all [--cleanup] [--device cuda]
    python llm_lineage.py score   [--hungarian]

Outputs:
    sigs/<model_tag>.npz          per-model signatures (fp16) + s values
    lineage_scores.csv            all pairwise L, z-scores, verdicts
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

# --------------------------------------------------------------------------
# Pair list.
# All models: 32 transformer layers, hidden d = 4096, SwiGLU MLP with
# tensors model.layers.{i}.mlp.{up_proj,down_proj}.weight.
# Intermediate sizes differ across families (11008 LLaMA-2 / 14336 Mistral
# / 11008 Yi / 11008 Qwen1.5) -- irrelevant, the product is always d x d.
# Attention paths are deliberately excluded: Mistral/Yi use GQA, so
# W_Q W_K^T is not square there; the MLP path is the uniform comparator.
# --------------------------------------------------------------------------
REFERENCE = ("llama2-7b-base", "NousResearch/Llama-2-7b-hf")
# NousResearch mirror is ungated; swap to meta-llama/Llama-2-7b-hf if you
# have access approved (identical weights).

DESCENDANTS = [
    # tag, repo, expected relation to reference
    ("llama2-7b-chat",   "NousResearch/Llama-2-7b-chat-hf", "RLHF (official)"),
    ("vicuna-7b-v1.5",   "lmsys/vicuna-7b-v1.5",            "full FT (ShareGPT)"),
    ("hermes-llama2-7b", "NousResearch/Nous-Hermes-llama-2-7b", "full FT"),
    ("codellama-7b",     "codellama/CodeLlama-7b-hf",       "continued pretrain, ~500B tokens"),
]

NEGATIVES = [
    # independent, same shape family (32 layers, d=4096), trained from scratch
    ("mistral-7b-v0.1",  "mistralai/Mistral-7B-v0.1"),
    ("openllama-7b",     "openlm-research/open_llama_7b"),
    ("yi-6b",            "01-ai/Yi-6B"),
    ("qwen1.5-7b",       "Qwen/Qwen1.5-7B"),
    ("amber-7b",         "LLM360/Amber"),
]

N_LAYERS = 32
D_MODEL = 4096
SIG_DIR = Path("sigs")

# Synthetic descendants applied locally to the reference weights, mirroring
# the paper's transformation grid on real 7B weights at zero training cost.
LOCAL_TRANSFORMS = {
    "ref+quant-int8": "quant",   # per-channel symmetric int8 round-trip
    "ref+prune-30":   "prune",   # 30% global magnitude pruning (per matrix)
    "ref+noise-1pct": "noise",   # Gaussian noise, sigma = 1% of per-matrix std
}


# --------------------------------------------------------------------------
# Streaming tensor access
# --------------------------------------------------------------------------
def _shard_map(repo: str) -> tuple[Path, dict]:
    """Return (local_dir, tensor_name -> shard_filename)."""
    try:
        idx_path = hf_hub_download(repo, "model.safetensors.index.json")
        with open(idx_path) as f:
            index = json.load(f)["weight_map"]
        local_dir = Path(idx_path).parent
        return local_dir, index
    except Exception:
        # single-file checkpoint
        st_path = hf_hub_download(repo, "model.safetensors")
        local_dir = Path(st_path).parent
        with safe_open(st_path, framework="pt") as f:
            names = list(f.keys())
        return local_dir, {n: "model.safetensors" for n in names}


def get_tensor(local_dir: Path, shard_of: dict, name: str) -> torch.Tensor:
    shard = local_dir / shard_of[name]
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(name)


def predownload(repo: str):
    """Pull all safetensors shards up front (resumable)."""
    snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])


# --------------------------------------------------------------------------
# Transforms for local synthetic descendants
# --------------------------------------------------------------------------
def apply_transform(w: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    if kind == "quant":  # per-output-channel symmetric int8
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        return (w / scale).round().clamp(-127, 127) * scale
    if kind == "prune":  # 30% magnitude pruning per matrix
        k = int(0.30 * w.numel())
        thresh = w.abs().flatten().kthvalue(k).values
        return torch.where(w.abs() <= thresh, torch.zeros_like(w), w)
    if kind == "noise":
        sigma = 0.01 * w.std()
        return w + sigma * torch.randn(w.shape, generator=gen, dtype=w.dtype)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# Signature extraction
# --------------------------------------------------------------------------
def extract(tag: str, repo: str, device: str = "cpu",
            transform: str | None = None, cleanup: bool = False):
    out = SIG_DIR / f"{tag}.npz"
    if out.exists():
        print(f"[skip] {tag} (exists)")
        return
    print(f"[extract] {tag} <- {repo}" + (f" (+{transform})" if transform else ""))
    predownload(repo)
    local_dir, shard_of = _shard_map(repo)
    gen = torch.Generator().manual_seed(0)

    phis, scores = [], []
    eye = torch.eye(D_MODEL, device=device)
    for i in range(N_LAYERS):
        wu = get_tensor(local_dir, shard_of, f"model.layers.{i}.mlp.up_proj.weight").float()
        wd = get_tensor(local_dir, shard_of, f"model.layers.{i}.mlp.down_proj.weight").float()
        if transform:
            wu = apply_transform(wu, transform, gen)
            wd = apply_transform(wd, transform, gen)
        wu, wd = wu.to(device), wd.to(device)
        m = wd @ wu                          # (d, I) @ (I, d) -> (d, d)
        tr = torch.trace(m)
        fro = torch.linalg.matrix_norm(m)
        scores.append((tr.abs() / fro).item())
        r = m - (tr / D_MODEL) * eye
        phi = (r / torch.linalg.matrix_norm(r)).flatten()
        phis.append(phi.to(torch.float16).cpu().numpy())
        del wu, wd, m, r, phi
        print(f"  layer {i:02d}  s={scores[-1]:.4f}  tr_sign={'-' if tr < 0 else '+'}")

    SIG_DIR.mkdir(exist_ok=True)
    np.savez_compressed(out, phi=np.stack(phis), s=np.array(scores))
    print(f"  -> {out}  (mean s = {np.mean(scores):.4f})")

    if cleanup:
        # nuke this repo's blobs from the HF cache to bound disk usage
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for r in info.repos:
            if r.repo_id == repo:
                shutil.rmtree(r.repo_path, ignore_errors=True)
                print(f"  [cleanup] removed cache for {repo}")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def load_sig(tag: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(SIG_DIR / f"{tag}.npz")
    phi = z["phi"].astype(np.float32)
    # Signatures are stored in fp16; rounding perturbs unit norm by ~1e-3,
    # which can push the dot of two such vectors above 1 and violate the
    # method's bounded-in-[-1,1] claim. Renormalize on load (fp32).
    phi /= np.linalg.norm(phi, axis=1, keepdims=True).clip(min=1e-12)
    return phi, z["s"]


def lineage(tag_a: str, tag_b: str, hungarian: bool = False) -> tuple[float, np.ndarray]:
    pa, _ = load_sig(tag_a)
    pb, _ = load_sig(tag_b)
    cos = pa @ pb.T                          # (L, L) signature-cosine matrix
    if hungarian:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-cos)
        per_block = cos[rows, cols]
        perm_ok = np.array_equal(cols, np.arange(len(cols)))
        if not perm_ok:
            print(f"  [note] Hungarian alignment for ({tag_a},{tag_b}) "
                  f"is non-identity: {cols.tolist()}")
    else:
        per_block = np.diag(cos)
    return float(per_block.mean()), per_block


def score_all(hungarian: bool = False):
    ref = REFERENCE[0]
    desc_tags = [t for t, _, _ in DESCENDANTS] + list(LOCAL_TRANSFORMS)
    neg_tags = [t for t, _ in NEGATIVES]

    # null distribution: reference vs each independent model
    null = np.array([lineage(ref, t, hungarian)[0] for t in neg_tags])
    mu, sd = null.mean(), max(null.std(ddof=1), 1e-6)
    print(f"\nNull (n={len(null)}): mu={mu:.4f}  sigma={sd:.4f}  "
          f"max={null.max():.4f}")

    rows = []
    def add(kind, a, b, expected):
        L, pb = lineage(a, b, hungarian)
        z = (L - mu) / sd
        verdict = ("DESCENDANT" if z > 3.0 else
                   "NON-DESCENDANT" if z < 1.645 else "INCONCLUSIVE")
        rows.append((kind, a, b, expected, L, z, verdict,
                     float(pb.min()), float(pb.max())))
        print(f"  {a:>18s} vs {b:<18s}  L={L:+.4f}  z={z:+7.1f}  "
              f"{verdict:<15s} (expected {expected})")

    print("\n-- reference vs descendants --")
    for t, _, rel in DESCENDANTS:
        add("descendant", ref, t, "DESCENDANT")
    for t in LOCAL_TRANSFORMS:
        add("descendant-local", ref, t, "DESCENDANT")

    print("\n-- reference vs independents --")
    for t in neg_tags:
        add("independent", ref, t, "NON-DESCENDANT")

    print("\n-- sibling pairs (shared available ancestor) --")
    sib = [t for t, _, _ in DESCENDANTS]
    for i in range(len(sib)):
        for j in range(i + 1, len(sib)):
            add("sibling", sib[i], sib[j], "high-but-below-parent")

    import csv
    with open("lineage_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_a", "model_b", "expected",
                    "L", "z", "verdict", "min_block", "max_block"])
        w.writerows(rows)
    print("\n-> lineage_scores.csv")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--all", action="store_true")
    pe.add_argument("--model", help="extract a single tag")
    pe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pe.add_argument("--cleanup", action="store_true",
                    help="delete HF shard cache after each model")
    ps = sub.add_parser("score")
    ps.add_argument("--hungarian", action="store_true")
    args = p.parse_args()

    if args.cmd == "extract":
        todo = [REFERENCE] + [(t, r) for t, r, _ in DESCENDANTS] + NEGATIVES
        if args.model:
            todo = [(t, r) for (t, r) in todo if t == args.model]
            if not todo:
                sys.exit(f"unknown tag {args.model}")
        for tag, repo in todo:
            extract(tag, repo, args.device, cleanup=args.cleanup)
        # local synthetic descendants are derived from the reference repo
        if args.all or (args.model in LOCAL_TRANSFORMS if args.model else False):
            for tag, kind in LOCAL_TRANSFORMS.items():
                extract(tag, REFERENCE[1], args.device,
                        transform=kind, cleanup=False)
    else:
        score_all(hungarian=args.hungarian)


if __name__ == "__main__":
    main()
