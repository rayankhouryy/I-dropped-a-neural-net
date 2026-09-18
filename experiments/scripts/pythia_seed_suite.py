#!/usr/bin/env python3
"""
Pythia-160M independent-root null benchmark (W3 response, reviewer u7g4).

Real-LM analogue of the LLaMA-2 Table-4 null study, at 160M scale, using the
EleutherAI Pythia seed suite as a large set of INDEPENDENTLY TRAINED roots.
The scoring math is byte-identical to `llm_lineage_table4_nulls.py` (the paper's
Table-4 null path): per-block MLP branch product M = W_down @ W_up, centered
residual signature phi = (M - tr(M)/d * I)/||.||, per-block cosine with DIAGONAL
alignment (block i vs block i), model-level score = mean over blocks. UNGATED.
Only the architecture plumbing differs (GPTNeoX 2-matrix GELU MLP, L=12, d=768),
so no method component, scoring function, or gate is changed.

The seed suite has NO descendants, so this is a NULL-ONLY benchmark: every model
pair is a pair of independent roots. We report N, the max independent-pair score,
and the conformal p-value floor 1/(N+1), plus the full N x N score matrix.

Usage:
    python pythia_seed_suite.py enumerate                 # discover repos, dims, commit SHAs
    python pythia_seed_suite.py extract --all --workers 4 # download (safetensors-only) + sigs
    python pythia_seed_suite.py extract --tag pythia-160m-seed1
    python pythia_seed_suite.py score                     # full matrix, max null, 1/(N+1)

Outputs:
    sigs_pythia/manifest.json               repo -> {sha, L, d, ff, fmt}
    sigs_pythia/<tag>.npz                    per-model signatures (phi, s)
    results/pythia_seed_suite.json           N, max_null, conformal_floor, pair list
    ../../reviews/w3_pythia_matrix.csv        full N x N lineage-score matrix
"""
import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, scan_cache_dir
from safetensors import safe_open

EXPECT_L = 12
EXPECT_D = 768
EXPECT_FF = 3072

SIG_DIR = Path("sigs_pythia")
RESULTS = Path("results/pythia_seed_suite.json")
MATRIX_CSV = Path("../../reviews/w3_pythia_matrix.csv")

# Keep only HF-format GPTNeoX *language models* named pythia-160m*. Exclude the
# sparse-autoencoder repos (SAE/SST/ST/sae-) and the raw GPT-NeoX-library
# checkpoints (neox-ckpt-*), which are not HF GPTNeoXForCausalLM checkpoints.
KEEP = re.compile(r"^EleutherAI/pythia-160m(-[a-z0-9\-]+)?$")
DROP = re.compile(r"(sae|sst|-st-|neox-ckpt)", re.IGNORECASE)


def candidate_repos():
    api = HfApi()
    ids = sorted({m.id for m in api.list_models(author="EleutherAI", search="pythia-160m")})
    return [r for r in ids if KEEP.match(r) and not DROP.search(r)]


def tag_of(repo: str) -> str:
    return repo.split("/")[-1]


# ---------------------------------------------------------------------------
# enumerate
# ---------------------------------------------------------------------------
def cmd_enumerate(args):
    api = HfApi()
    SIG_DIR.mkdir(exist_ok=True)
    manifest = {}
    repos = candidate_repos()
    print(f"Checking {len(repos)} candidate pythia-160m* repos...\n")
    kept, skipped = [], []
    for repo in repos:
        try:
            info = api.model_info(repo)
            sha = info.sha
            files = {s.rfilename for s in info.siblings}
            cfg_path = hf_hub_download(repo, "config.json", revision=sha)
            cfg = json.loads(Path(cfg_path).read_text())
            L = cfg.get("num_hidden_layers")
            d = cfg.get("hidden_size")
            ff = cfg.get("intermediate_size")
            mt = cfg.get("model_type")
            archs = cfg.get("architectures", [])
            ok = (mt == "gpt_neox" and L == EXPECT_L and d == EXPECT_D and ff == EXPECT_FF)
            if "model.safetensors" in files:
                fmt = "st"
            elif "pytorch_model.bin" in files:
                fmt = "bin"
            else:
                fmt = None
            if ok and fmt:
                manifest[tag_of(repo)] = {
                    "repo": repo, "sha": sha, "L": L, "d": d, "ff": ff,
                    "model_type": mt, "architectures": archs, "fmt": fmt,
                }
                kept.append(tag_of(repo))
                print(f"  [keep] {tag_of(repo):<28s} L={L} d={d} ff={ff} fmt={fmt} sha={sha[:12]}")
            else:
                skipped.append((tag_of(repo), f"mt={mt} L={L} d={d} ff={ff} fmt={fmt}"))
                print(f"  [skip] {tag_of(repo):<28s} ({mt} L={L} d={d} ff={ff} fmt={fmt})")
        except Exception as e:
            skipped.append((tag_of(repo), f"error: {e}"))
            print(f"  [skip] {tag_of(repo):<28s} (error: {e})")

    (SIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nKept {len(kept)} independent roots -> {SIG_DIR/'manifest.json'}")
    print(f"Skipped {len(skipped)}.")
    print("\nNOTE (flag, not fix): weight-seedN share data order and data-seedN share init "
          "with the base recipe, so they are only PARTIALLY independent; dropout/v0 variants "
          "are separate training regimes. All are scored as roots per the task; caveat reported.")


def load_manifest() -> dict:
    p = SIG_DIR / "manifest.json"
    if not p.exists():
        sys.exit("No manifest. Run: python pythia_seed_suite.py enumerate")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def _tensor(local_dir: Path, fmt: str, name: str):
    if fmt == "st":
        with safe_open(str(local_dir / "model.safetensors"), framework="pt") as f:
            return f.get_tensor(name)
    sd = torch.load(str(local_dir / "pytorch_model.bin"), map_location="cpu",
                    weights_only=True, mmap=True)
    return sd[name]


def _cleanup(repo: str):
    try:
        for r in scan_cache_dir().repos:
            if r.repo_id == repo:
                shutil.rmtree(r.repo_path, ignore_errors=True)
                return
    except Exception as e:
        print(f"    [cleanup warn] {e}")


def extract_one(tag: str, meta: dict, cleanup: bool = True) -> str:
    out = SIG_DIR / f"{tag}.npz"
    if out.exists():
        return f"[skip] {tag} (exists)"
    repo, sha, fmt = meta["repo"], meta["sha"], meta["fmt"]
    patterns = ["*.safetensors", "config.json"] if fmt == "st" else ["pytorch_model.bin", "config.json"]
    snapshot_download(repo, revision=sha, allow_patterns=patterns)
    # resolve the snapshot dir for this revision
    cfg_path = hf_hub_download(repo, "config.json", revision=sha)
    local_dir = Path(cfg_path).parent

    eye = np.eye(EXPECT_D, dtype=np.float64)
    phis, scores = [], []
    for i in range(EXPECT_L):
        wu = _tensor(local_dir, fmt, f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight").to(torch.float32).numpy().astype(np.float64)
        wd = _tensor(local_dir, fmt, f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight").to(torch.float32).numpy().astype(np.float64)
        # wu: (4h, h) = (3072,768); wd: (h, 4h) = (768,3072); M = wd @ wu = (h,h)
        M = wd @ wu
        tr = np.trace(M)
        fro = np.linalg.norm(M, "fro")
        scores.append(float(abs(tr) / (fro + 1e-12)))
        R = M - (tr / EXPECT_D) * eye
        phi = (R / (np.linalg.norm(R) + 1e-12)).flatten().astype(np.float16)
        phis.append(phi)
    SIG_DIR.mkdir(exist_ok=True)
    np.savez_compressed(out, phi=np.stack(phis), s=np.array(scores))
    if cleanup:
        _cleanup(repo)
    return f"[done] {tag}  mean_s={np.mean(scores):.4f}"


def cmd_extract(args):
    manifest = load_manifest()
    if args.tag:
        todo = {args.tag: manifest[args.tag]}
    elif args.all:
        todo = manifest
    else:
        sys.exit("Specify --all or --tag <tag>")
    print(f"Extracting {len(todo)} models (fmt-only download, stream+delete)...")
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(extract_one, t, m, not args.no_cleanup): t for t, m in todo.items()}
            for f in as_completed(futs):
                print("  " + f.result())
    else:
        for t, m in todo.items():
            print("  " + extract_one(t, m, not args.no_cleanup))
    print("Done. Next: python pythia_seed_suite.py score")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def load_sig(tag: str) -> np.ndarray:
    z = np.load(SIG_DIR / f"{tag}.npz")
    phi = z["phi"].astype(np.float32)
    phi /= np.linalg.norm(phi, axis=1, keepdims=True).clip(min=1e-12)
    return phi


def lineage(pa: np.ndarray, pb: np.ndarray) -> float:
    """Ungated, diagonal-aligned, mean-over-blocks cosine (Table-4 convention)."""
    return float(np.diag(pa @ pb.T).mean())


def cmd_score(args):
    manifest = load_manifest()
    tags = [t for t in manifest if (SIG_DIR / f"{t}.npz").exists()]
    tags.sort()
    missing = [t for t in manifest if t not in tags]
    if missing:
        print(f"WARNING: {len(missing)} models not yet extracted: {missing}")
    n = len(tags)
    if n < 2:
        sys.exit("Need >=2 extracted models to score pairs.")
    sigs = {t: load_sig(t) for t in tags}

    mat = np.full((n, n), np.nan)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            L = lineage(sigs[tags[i]], sigs[tags[j]])
            mat[i, j] = L
            if i < j:
                pairs.append({"a": tags[i], "b": tags[j], "L": L})

    scores = np.array([p["L"] for p in pairs])
    max_null = float(scores.max())
    argmax = pairs[int(scores.argmax())]
    conformal_floor = 1.0 / (n + 1)

    result = {
        "benchmark": "pythia-160m seed suite",
        "convention": "ungated, diagonal alignment, mean-over-blocks (Table-4 null path)",
        "n_roots": n,
        "n_independent_pairs": len(pairs),
        "max_independent_pair_score": max_null,
        "argmax_pair": argmax,
        "mean_independent_pair_score": float(scores.mean()),
        "sd_independent_pair_score": float(scores.std(ddof=1)),
        "conformal_p_floor_1_over_Nplus1": conformal_floor,
        "roots": [{"tag": t, "repo": manifest[t]["repo"], "sha": manifest[t]["sha"]} for t in tags],
        "pairs": pairs,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2))

    MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(MATRIX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + tags)
        for i, t in enumerate(tags):
            w.writerow([t] + [f"{mat[i, j]:.6f}" for j in range(n)])

    print("=" * 64)
    print(f"Pythia-160M independent-root null benchmark")
    print("=" * 64)
    print(f"  N independent roots            : {n}")
    print(f"  independent pairs              : {len(pairs)}")
    print(f"  max independent-pair score     : {max_null:.6f}  ({argmax['a']} vs {argmax['b']})")
    print(f"  mean +/- sd                    : {scores.mean():.6f} +/- {scores.std(ddof=1):.6f}")
    print(f"  conformal p-floor 1/(N+1)      : {conformal_floor:.4f}")
    print(f"\n  -> {RESULTS}\n  -> {MATRIX_CSV}")


def main():
    p = argparse.ArgumentParser(description="Pythia-160M independent-root null benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enumerate")
    pe = sub.add_parser("extract")
    pe.add_argument("--all", action="store_true")
    pe.add_argument("--tag")
    pe.add_argument("--workers", type=int, default=1)
    pe.add_argument("--no-cleanup", action="store_true")
    sub.add_parser("score")
    args = p.parse_args()
    {"enumerate": cmd_enumerate, "extract": cmd_extract, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
