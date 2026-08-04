#!/usr/bin/env python3
"""
Expanded null models for Table 4: Public checkpoint case study (LLaMA-2 family).

Adds 7 exact architectural clones (32 layers, d=4096, intermediate=11008, SwiGLU)
that are independently trained from scratch (no LLaMA-2 ancestry):

    1. openlm-research/open_llama_7b       - OpenLLaMA v1
    2. openlm-research/open_llama_7b_v2    - OpenLLaMA v2 (separate training run)
    3. LLM360/Amber                        - Fully documented from-scratch run
    4. baichuan-inc/Baichuan-7B            - Baichuan v1
    5. baichuan-inc/Baichuan2-7B-Base      - Baichuan v2
    6. internlm/internlm-7b                - InternLM
    7. 01-ai/Yi-6B                         - Yi (exact dims despite "6B" name)

Usage (run one model at a time to manage disk/memory):
    cd experiments/scripts
    export HF_TOKEN=...

    # Extract each model (downloads ~13GB, extracts sig, deletes weights)
    python llm_lineage_table4_nulls.py extract --model openllama-7b
    python llm_lineage_table4_nulls.py extract --model openllama-7b-v2
    python llm_lineage_table4_nulls.py extract --model amber-7b
    python llm_lineage_table4_nulls.py extract --model baichuan-7b
    python llm_lineage_table4_nulls.py extract --model baichuan2-7b
    python llm_lineage_table4_nulls.py extract --model internlm-7b
    python llm_lineage_table4_nulls.py extract --model yi-6b

    # After all extracted, compute scores
    python llm_lineage_table4_nulls.py score

Outputs:
    sigs_table4/<tag>.npz                   per-model signatures
    results/table4_null_expansion.json      scores and verdicts
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download, scan_cache_dir
from safetensors import safe_open

# Reference: LLaMA-2-7B
REF_TAG = "llama2-7b-base"
REF_REPO = "NousResearch/Llama-2-7b-hf"

# Known descendants (for context, already in paper)
DESCENDANTS = [
    ("llama2-7b-chat", "NousResearch/Llama-2-7b-chat-hf", "RLHF (official)"),
    ("vicuna-7b", "lmsys/vicuna-7b-v1.5", "full FT"),
    ("codellama-7b", "codellama/CodeLlama-7b-hf", "code PT"),
]

# New null models: exact architectural clones, trained from scratch
NULL_MODELS = [
    ("openllama-7b", "openlm-research/open_llama_7b", "OpenLLaMA v1"),
    ("openllama-7b-v2", "openlm-research/open_llama_7b_v2", "OpenLLaMA v2"),
    ("amber-7b", "LLM360/Amber", "LLM360 Amber"),
    ("baichuan-7b", "baichuan-inc/Baichuan-7B", "Baichuan v1"),
    ("baichuan2-7b", "baichuan-inc/Baichuan2-7B-Base", "Baichuan v2"),
    ("internlm-7b", "internlm/internlm-7b", "InternLM"),
    ("yi-6b", "01-ai/Yi-6B", "Yi"),
]

N_LAYERS = 32
D_MODEL = 4096
SIG_DIR = Path("sigs_table4")

# Weight name patterns vary by model family
WEIGHT_PATTERNS = {
    "default": ("model.layers.{i}.mlp.up_proj.weight", "model.layers.{i}.mlp.down_proj.weight"),
    "baichuan": ("model.layers.{i}.mlp.up_proj.weight", "model.layers.{i}.mlp.down_proj.weight"),
    "internlm": ("model.layers.{i}.mlp.up_proj.weight", "model.layers.{i}.mlp.down_proj.weight"),
}


def get_weight_names(tag: str, layer: int) -> tuple[str, str]:
    return (f"model.layers.{layer}.mlp.up_proj.weight",
            f"model.layers.{layer}.mlp.down_proj.weight")


def _shard_map(repo: str) -> tuple[Path, dict, str]:
    """Return (local_dir, tensor_name -> shard_filename, format).

    Supports both safetensors and pytorch .bin formats.
    """
    # Try safetensors first
    try:
        idx_path = hf_hub_download(repo, "model.safetensors.index.json")
        with open(idx_path) as f:
            index = json.load(f)["weight_map"]
        return Path(idx_path).parent, index, "safetensors"
    except Exception:
        pass

    # Try single safetensors file
    try:
        st_path = hf_hub_download(repo, "model.safetensors")
        local_dir = Path(st_path).parent
        with safe_open(st_path, framework="pt") as f:
            names = list(f.keys())
        return local_dir, {n: "model.safetensors" for n in names}, "safetensors"
    except Exception:
        pass

    # Try pytorch .bin sharded
    try:
        idx_path = hf_hub_download(repo, "pytorch_model.bin.index.json")
        with open(idx_path) as f:
            index = json.load(f)["weight_map"]
        return Path(idx_path).parent, index, "pytorch"
    except Exception:
        pass

    # Try single pytorch file
    pt_path = hf_hub_download(repo, "pytorch_model.bin")
    local_dir = Path(pt_path).parent
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    return local_dir, {n: "pytorch_model.bin" for n in state.keys()}, "pytorch"


def get_tensor(local_dir: Path, shard_of: dict, name: str,
               fmt: str = "safetensors") -> torch.Tensor:
    shard = local_dir / shard_of[name]
    if fmt == "safetensors":
        with safe_open(str(shard), framework="pt") as f:
            return f.get_tensor(name)
    else:
        state = torch.load(str(shard), map_location="cpu", weights_only=True)
        return state[name]


def predownload(repo: str):
    """Pull all model shards (resumable). Supports safetensors and pytorch."""
    snapshot_download(repo, allow_patterns=[
        "*.safetensors", "*.bin", "*.json"
    ])


def cleanup_cache(repo: str):
    """Delete this repo's blobs from HF cache."""
    try:
        info = scan_cache_dir()
        for r in info.repos:
            if r.repo_id == repo:
                shutil.rmtree(r.repo_path, ignore_errors=True)
                print(f"  [cleanup] removed cache for {repo}")
                return
    except Exception as e:
        print(f"  [cleanup] warning: {e}")


def extract(tag: str, repo: str, device: str = "cpu", cleanup: bool = True):
    """Extract signatures for a single model."""
    out = SIG_DIR / f"{tag}.npz"
    if out.exists():
        print(f"[skip] {tag} (already extracted)")
        return

    print(f"\n{'='*60}")
    print(f"[extract] {tag}")
    print(f"  repo: {repo}")
    print(f"  device: {device}")
    print(f"{'='*60}\n")

    print("Downloading weights...")
    predownload(repo)
    local_dir, shard_of, fmt = _shard_map(repo)
    print(f"  format: {fmt}")

    phis, scores = [], []
    eye = torch.eye(D_MODEL, device=device)

    print("\nExtracting signatures...")
    for i in range(N_LAYERS):
        up_name, down_name = get_weight_names(tag, i)
        wu = get_tensor(local_dir, shard_of, up_name, fmt).float()
        wd = get_tensor(local_dir, shard_of, down_name, fmt).float()
        wu, wd = wu.to(device), wd.to(device)

        m = wd @ wu  # (d, intermediate) @ (intermediate, d) -> (d, d)
        tr = torch.trace(m)
        fro = torch.linalg.matrix_norm(m)
        scores.append((tr.abs() / fro).item())
        r = m - (tr / D_MODEL) * eye
        phi = (r / torch.linalg.matrix_norm(r)).flatten()
        phis.append(phi.to(torch.float16).cpu().numpy())

        del wu, wd, m, r, phi
        print(f"  layer {i:02d}  s={scores[-1]:.4f}")

    SIG_DIR.mkdir(exist_ok=True)
    np.savez_compressed(out, phi=np.stack(phis), s=np.array(scores))
    print(f"\n  -> saved {out}  (mean s = {np.mean(scores):.4f})")

    if cleanup:
        print("\nCleaning up downloaded weights...")
        cleanup_cache(repo)

    print(f"\n[done] {tag}\n")


def load_sig(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and renormalize signatures."""
    z = np.load(SIG_DIR / f"{tag}.npz")
    phi = z["phi"].astype(np.float32)
    phi /= np.linalg.norm(phi, axis=1, keepdims=True).clip(min=1e-12)
    return phi, z["s"]


def lineage(tag_a: str, tag_b: str) -> tuple[float, np.ndarray]:
    """Compute lineage score between two models."""
    pa, _ = load_sig(tag_a)
    pb, _ = load_sig(tag_b)
    cos = pa @ pb.T
    per_block = np.diag(cos)
    return float(per_block.mean()), per_block


def score_all():
    """Compute all scores and generate results."""
    print("\n" + "="*60)
    print("Computing lineage scores for Table 4 expansion")
    print("="*60 + "\n")

    # Check which signatures exist
    available = []
    missing = []

    if not (SIG_DIR / f"{REF_TAG}.npz").exists():
        missing.append((REF_TAG, REF_REPO))

    for tag, repo, _ in NULL_MODELS:
        if (SIG_DIR / f"{tag}.npz").exists():
            available.append(tag)
        else:
            missing.append((tag, repo))

    if missing:
        print("Missing signatures (run extract first):")
        for tag, repo in missing:
            print(f"  - {tag}: python llm_lineage_table4_nulls.py extract --model {tag}")
        print()

    if not available:
        print("No null model signatures found. Extract at least one first.")
        return

    print(f"Available null models: {len(available)}/{len(NULL_MODELS)}")

    # Compute null distribution from available models
    print("\n-- Null distribution (LLaMA-2 vs independent models) --\n")
    null_scores = []
    null_details = []

    for tag, repo, family in NULL_MODELS:
        if tag not in available:
            continue
        L, pb = lineage(REF_TAG, tag)
        null_scores.append(L)
        null_details.append({
            "tag": tag,
            "family": family,
            "L": L,
            "min_block": float(pb.min()),
            "max_block": float(pb.max()),
        })
        print(f"  {REF_TAG} vs {tag:<18s} ({family:<15s})  L={L:+.6f}  "
              f"blocks=[{pb.min():+.4f}, {pb.max():+.4f}]")

    null_arr = np.array(null_scores)
    mu = float(null_arr.mean())
    sd = float(max(null_arr.std(ddof=1), 1e-6))

    print(f"\n  Null (n={len(null_arr)}): mu={mu:+.6f}  sigma={sd:.6f}  "
          f"range=[{null_arr.min():+.6f}, {null_arr.max():+.6f}]")

    # Compute descendant scores if available
    print("\n-- Descendants (for comparison) --\n")
    desc_results = []

    for tag, repo, rel in DESCENDANTS:
        if not (SIG_DIR / f"{tag}.npz").exists():
            print(f"  [skip] {tag} (not extracted)")
            continue
        L, pb = lineage(REF_TAG, tag)
        z = (L - mu) / sd
        verdict = "DESCENDANT" if z > 3.0 else ("NON-DESCENDANT" if z < 1.645 else "INCONCLUSIVE")
        desc_results.append({
            "tag": tag,
            "relation": rel,
            "L": L,
            "z": z,
            "verdict": verdict,
        })
        print(f"  {REF_TAG} vs {tag:<18s}  L={L:+.6f}  z={z:+8.1f}  {verdict}")

    # Save results
    results = {
        "reference": REF_TAG,
        "null_distribution": {
            "n": len(null_arr),
            "mu": mu,
            "sigma": sd,
            "min": float(null_arr.min()),
            "max": float(null_arr.max()),
        },
        "null_models": null_details,
        "descendants": desc_results,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "table4_null_expansion.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n-> {out_path}")

    # Print LaTeX snippet for paper
    print("\n" + "="*60)
    print("LaTeX table rows (copy to paper):")
    print("="*60 + "\n")

    for d in null_details:
        tag_fmt = d["tag"].replace("-", " ").replace("7b", "7B").replace("6b", "6B")
        print(f"{tag_fmt:<20s} & scratch & ${d['L']:.0e}$ & -- & UNR. \\\\")


def main():
    p = argparse.ArgumentParser(description="Table 4 null model expansion")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract signatures for a model")
    pe.add_argument("--model",
                    choices=[REF_TAG] + [t for t, _, _ in NULL_MODELS] + [t for t, _, _ in DESCENDANTS],
                    help="Model tag to extract (or use --all)")
    pe.add_argument("--all", action="store_true", help="Extract reference + all null models sequentially")
    pe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pe.add_argument("--no-cleanup", action="store_true", help="Keep downloaded weights")

    sub.add_parser("score", help="Compute scores from extracted signatures")
    sub.add_parser("list", help="List all models and their status")

    args = p.parse_args()

    if args.cmd == "extract":
        if args.all:
            # Run reference + all null models sequentially
            todo = [(REF_TAG, REF_REPO)] + [(t, r) for t, r, _ in NULL_MODELS]
            for tag, repo in todo:
                extract(tag, repo, args.device, cleanup=not args.no_cleanup)
            print("\n[all done] Run 'python llm_lineage_table4_nulls.py score'")
        elif args.model:
            tag = args.model
            repo = None
            if tag == REF_TAG:
                repo = REF_REPO
            else:
                for t, r, _ in NULL_MODELS + DESCENDANTS:
                    if t == tag:
                        repo = r
                        break
            if not repo:
                sys.exit(f"Unknown model tag: {tag}")
            extract(tag, repo, args.device, cleanup=not args.no_cleanup)
        else:
            sys.exit("Specify --model <tag> or --all")

    elif args.cmd == "score":
        score_all()

    elif args.cmd == "list":
        print("\nReference:")
        status = "✓" if (SIG_DIR / f"{REF_TAG}.npz").exists() else "✗"
        print(f"  [{status}] {REF_TAG:<20s}  {REF_REPO}")

        print("\nNull models (independent, exact architecture):")
        for tag, repo, family in NULL_MODELS:
            status = "✓" if (SIG_DIR / f"{tag}.npz").exists() else "✗"
            print(f"  [{status}] {tag:<20s}  {repo:<45s}  ({family})")

        print("\nDescendants:")
        for tag, repo, rel in DESCENDANTS:
            status = "✓" if (SIG_DIR / f"{tag}.npz").exists() else "✗"
            print(f"  [{status}] {tag:<20s}  {repo:<45s}  ({rel})")
        print()


if __name__ == "__main__":
    main()
