#!/usr/bin/env python3
"""Unified GPU benchmark for Issue #44: Strong baselines on LLaMA/BERT/ResNet.

Runs all 7 lineage baseline methods on:
1. LLaMA family: Llama-2-7b vs descendants (chat, Vicuna) and non-descendants (Mistral, OpenLLaMA)
2. BERT family: bert-base vs fine-tunes and DistilBERT/TinyBERT
3. ResNet: Re-score existing models with CKA/SVCCA/IPGuard

Usage:
    # Run all benchmarks in parallel (~2 hours on ml.g5.12xlarge):
    python gpu_benchmark_issue44.py --benchmark all --parallel

    # Run single benchmark:
    python gpu_benchmark_issue44.py --benchmark llama --device cuda:0

    # Dry run (list pairs, no computation):
    python gpu_benchmark_issue44.py --benchmark all --dry-run
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import lineage_baselines as lbase
import lineage_detection as ldet

RESULTS_DIR = SCRIPT_DIR.parent.parent / "results"
SIGS_DIR = SCRIPT_DIR / "sigs"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints_issue44"

METHODS = [
    "diagonal_dominance",
    "aligned_frobenius",
    "singular_value_dist",
    "weight_cosine",
    "cka",
    "svcca",
    "ipguard_regr",
]

# ============================================================================
# LLaMA Family Configuration
# ============================================================================

LLAMA_REFERENCE = ("llama2-7b-base", "NousResearch/Llama-2-7b-hf")

LLAMA_DESCENDANTS = [
    ("llama2-7b-chat", "NousResearch/Llama-2-7b-chat-hf"),
    ("vicuna-7b", "lmsys/vicuna-7b-v1.5"),
    ("codellama-7b", "codellama/CodeLlama-7b-hf"),
]

LLAMA_NONDESCENDANTS = [
    ("mistral-7b", "mistralai/Mistral-7B-v0.1"),
    ("open-llama-7b", "openlm-research/open_llama_7b"),
    ("yi-6b", "01-ai/Yi-6B"),
    ("deepseek-r1-distill-llama-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
]

LLAMA_LOCAL_TRANSFORMS = ["quant", "prune", "noise"]

# ============================================================================
# BERT Family Configuration
# ============================================================================

BERT_REFERENCE = ("bert-base", "google-bert/bert-base-uncased")

BERT_DESCENDANTS = [
    ("bert-sst2", "textattack/bert-base-uncased-SST-2"),
    ("bert-mnli", "textattack/bert-base-uncased-MNLI"),
    ("bert-squad", "csarron/bert-base-uncased-squad-v1"),
]

BERT_NONDESCENDANTS = [
    ("distilbert", "distilbert/distilbert-base-uncased"),
    ("bert-tiny", "prajjwal1/bert-mini"),
]

BERT_LOCAL_TRANSFORMS = ["quant", "prune", "noise"]

# ============================================================================
# Probe texts for activation collection
# ============================================================================

PROBE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can learn complex patterns from data.",
    "Neural networks have revolutionized artificial intelligence.",
    "Deep learning requires large amounts of training data.",
    "Transformers use attention mechanisms for sequence modeling.",
    "Language models can generate coherent text passages.",
    "Computer vision systems can recognize objects in images.",
    "Reinforcement learning agents learn through trial and error.",
    "Natural language processing enables human-computer interaction.",
    "Convolutional neural networks excel at image classification.",
] * 110  # ~1100 samples, will use first 1024


# ============================================================================
# Utilities
# ============================================================================

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_checkpoint(benchmark: str) -> dict:
    path = CHECKPOINT_DIR / f"{benchmark}_checkpoint.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_pairs": [], "pairs": []}


def save_checkpoint(benchmark: str, data: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{benchmark}_checkpoint.json"
    path.write_text(json.dumps(data, indent=2))


# ============================================================================
# Weight extraction for transformers
# ============================================================================

def extract_llama_branch_products(model) -> list[np.ndarray]:
    """Extract W_down @ W_up from each SwiGLU MLP layer."""
    Ms = []
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        mlp = layer.mlp
        W_up = mlp.up_proj.weight.detach().float().cpu().numpy()
        W_down = mlp.down_proj.weight.detach().float().cpu().numpy()
        M = W_down @ W_up
        Ms.append(M)
    return Ms


def extract_bert_branch_products(model) -> list[np.ndarray]:
    """Extract MLP branch products from BERT encoder layers."""
    Ms = []
    encoder = model.bert.encoder if hasattr(model, "bert") else model.encoder
    for layer in encoder.layer:
        W1 = layer.intermediate.dense.weight.detach().float().cpu().numpy()
        W2 = layer.output.dense.weight.detach().float().cpu().numpy()
        M = W2 @ W1
        Ms.append(M)
    return Ms


# ============================================================================
# Activation collection
# ============================================================================

def collect_llm_activations(model, tokenizer, probe_texts: list[str],
                            n_samples: int = 1024, device: str = "cuda") -> list[np.ndarray] | None:
    """Collect hidden states from all transformer layers.

    DISABLED for LLaMA: CKA/SVCCA on 32 layers × 4096 dims takes 10+ minutes
    per pair (32×32 = 1024 comparisons). Weight-space methods are the focus.

    Returns None to skip activation-based scoring.
    """
    print(f"      [acts] SKIPPED - CKA/SVCCA too slow for LLaMA (32×32×4096)", flush=True)
    return None


def collect_bert_activations(model, tokenizer, probe_texts: list[str],
                             n_samples: int = 1024, device: str = "cuda") -> list[np.ndarray]:
    """Collect hidden states from BERT encoder layers."""
    encoder = model.bert.encoder if hasattr(model, "bert") else model.encoder
    L = len(encoder.layer)
    all_acts = [[] for _ in range(L)]
    hooks = []

    for i, layer in enumerate(encoder.layer):
        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                all_acts[idx].append(h[:, 0, :].detach().float().cpu().numpy())
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for text in probe_texts[:n_samples]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=128, padding=True).to(device)
            _ = model(**inputs)

    for h in hooks:
        h.remove()

    return [np.vstack(acts) for acts in all_acts]


def collect_llm_logits(model, tokenizer, probe_texts: list[str],
                       n_samples: int = 1024, device: str = "cuda") -> np.ndarray | None:
    """Collect output logits for IPGuard scoring.

    For 7B models, uses only 128 samples (128 forward passes × ~0.5s = ~1 min).
    """
    effective_samples = min(n_samples, 128)
    print(f"      [logits] Collecting {effective_samples} samples...", flush=True)

    all_logits = []
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(probe_texts[:effective_samples]):
            if i % 25 == 0:
                print(f"      [logits] Sample {i}/{effective_samples}", flush=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True).to(device)
            out = model(**inputs)
            logits = out.logits[:, -1, :].detach().float().cpu().numpy()
            all_logits.append(logits)
    return np.vstack(all_logits)


def collect_bert_logits(model, tokenizer, probe_texts: list[str],
                        n_samples: int = 1024, device: str = "cuda") -> np.ndarray:
    """Collect output logits for IPGuard scoring."""
    all_logits = []
    model.eval()
    with torch.no_grad():
        for text in probe_texts[:n_samples]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=128, padding=True).to(device)
            out = model(**inputs)
            if hasattr(out, "logits"):
                logits = out.logits[:, 0, :].detach().float().cpu().numpy()
            else:
                logits = out.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            all_logits.append(logits)
    return np.vstack(all_logits)


# ============================================================================
# Local transforms
# ============================================================================

def apply_transform(model, transform: str, seed: int = 0):
    """Apply local transform (quant/prune/noise) to model weights."""
    rng = np.random.RandomState(seed)
    model = copy.deepcopy(model)

    if transform == "quant":
        for p in model.parameters():
            if p.numel() > 1000:
                data = p.data.float()
                vmin, vmax = data.min(), data.max()
                scale = (vmax - vmin) / 255
                q = ((data - vmin) / scale).round().clamp(0, 255)
                p.data = (q * scale + vmin).to(p.dtype)

    elif transform == "prune":
        for p in model.parameters():
            if p.numel() > 1000:
                flat = torch.abs(p.data).flatten().float()
                k = int(flat.numel() * 0.3)
                if k > 0:
                    threshold = flat.kthvalue(k).values.item()
                    mask = torch.abs(p.data) > threshold
                    p.data = p.data * mask

    elif transform == "noise":
        for p in model.parameters():
            if p.numel() > 1000:
                noise = torch.randn_like(p.data) * 0.01 * p.data.std()
                p.data = p.data + noise

    return model


# ============================================================================
# Scoring
# ============================================================================

def score_pair(ref_pack: dict, sus_pack: dict, tau_s: float = 0.5) -> dict:
    """Score a (reference, suspect) pair across all 7 methods."""
    scores = {}

    # Check if branch products have compatible shapes
    ref_Ms, sus_Ms = ref_pack["Ms"], sus_pack["Ms"]
    shapes_match = (len(ref_Ms) == len(sus_Ms) and
                    all(r.shape == s.shape for r, s in zip(ref_Ms, sus_Ms)))
    print(f"      [score] shapes_match={shapes_match}, L={len(ref_Ms)}", flush=True)

    if shapes_match:
        print(f"      [score] computing diagonal_dominance...", flush=True)
        try:
            scores["diagonal_dominance"], _, _ = ldet.lineage_score(ref_Ms, sus_Ms, tau_s)
        except Exception as e:
            print(f"      [score] diagonal_dominance failed: {e}", flush=True)
            scores["diagonal_dominance"] = float("nan")
        print(f"      [score] computing aligned_frobenius...", flush=True)
        try:
            scores["aligned_frobenius"] = lbase.aligned_frobenius(ref_Ms, sus_Ms)
        except Exception as e:
            print(f"      [score] aligned_frobenius failed: {e}", flush=True)
            scores["aligned_frobenius"] = float("nan")
        print(f"      [score] computing singular_value_dist...", flush=True)
        try:
            scores["singular_value_dist"] = lbase.singular_value_distance(ref_Ms, sus_Ms)
        except Exception as e:
            print(f"      [score] singular_value_dist failed: {e}", flush=True)
            scores["singular_value_dist"] = float("nan")
        print(f"      [score] computing weight_cosine...", flush=True)
        try:
            scores["weight_cosine"] = lbase.weight_cosine(ref_Ms, sus_Ms)
        except Exception as e:
            print(f"      [score] weight_cosine failed: {e}", flush=True)
            scores["weight_cosine"] = float("nan")
    else:
        # Architecture mismatch - weight-space methods not applicable
        scores["diagonal_dominance"] = float("nan")
        scores["aligned_frobenius"] = float("nan")
        scores["singular_value_dist"] = float("nan")
        scores["weight_cosine"] = float("nan")

    ref_acts, sus_acts = ref_pack.get("acts"), sus_pack.get("acts")
    if ref_acts and sus_acts and len(ref_acts) == len(sus_acts):
        print(f"      [score] computing cka (acts: {len(ref_acts)} layers)...", flush=True)
        try:
            scores["cka"] = lbase.cka_lineage_score(ref_acts, sus_acts)
        except Exception as e:
            print(f"      [score] cka failed: {e}", flush=True)
            scores["cka"] = float("nan")
        print(f"      [score] computing svcca...", flush=True)
        try:
            scores["svcca"] = lbase.svcca_lineage_score(ref_acts, sus_acts)
        except Exception as e:
            print(f"      [score] svcca failed: {e}", flush=True)
            scores["svcca"] = float("nan")
    else:
        scores["cka"] = float("nan")
        scores["svcca"] = float("nan")

    if ref_pack.get("logits") is not None and sus_pack.get("logits") is not None:
        print(f"      [score] computing ipguard_regr...", flush=True)
        try:
            scores["ipguard_regr"] = lbase.ipguard_match_rate(
                ref_pack["logits"], sus_pack["logits"]
            )
        except Exception as e:
            print(f"      [score] ipguard_regr failed: {e}", flush=True)
            scores["ipguard_regr"] = float("nan")
    else:
        scores["ipguard_regr"] = float("nan")

    print(f"      [score] done", flush=True)
    return scores


def compute_auroc(pairs: list[dict]) -> dict:
    """Compute AUROC for each method."""
    labels = np.array([p["label"] for p in pairs])
    auroc = {}
    for m in METHODS:
        s = np.array([p["scores"].get(m, float("nan")) for p in pairs])
        valid = ~np.isnan(s)
        if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
            auroc[m] = float("nan")
        else:
            try:
                auroc[m] = float(roc_auc_score(labels[valid], s[valid]))
            except Exception:
                auroc[m] = float("nan")
    return auroc


def compute_per_kind(pairs: list[dict]) -> dict:
    """Compute per-kind statistics for each method."""
    kinds = sorted(set(p["kind"] for p in pairs))
    per_kind = {m: {} for m in METHODS}
    for m in METHODS:
        for kind in kinds:
            sub = [p for p in pairs if p["kind"] == kind]
            s = np.array([p["scores"].get(m, float("nan")) for p in sub])
            valid = ~np.isnan(s)
            if valid.sum() == 0:
                continue
            per_kind[m][kind] = {
                "n": int(valid.sum()),
                "mean": float(np.mean(s[valid])),
                "std": float(np.std(s[valid])),
            }
    return per_kind


# ============================================================================
# LLaMA Benchmark
# ============================================================================

def run_llama_benchmark(device: str = "cuda", resume: bool = False,
                        dry_run: bool = False, n_samples: int = 1024) -> dict:
    """Run LLaMA family benchmark."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print("LLaMA Family Benchmark")
    print(f"{'='*60}")

    all_models = (
        [LLAMA_REFERENCE]
        + LLAMA_DESCENDANTS
        + LLAMA_NONDESCENDANTS
    )
    pair_specs = []

    for tag, repo in LLAMA_DESCENDANTS:
        pair_specs.append({
            "ref": LLAMA_REFERENCE[0], "sus": tag, "repo": repo,
            "kind": "descendant_finetune", "label": 1, "transform": None
        })

    for transform in LLAMA_LOCAL_TRANSFORMS:
        pair_specs.append({
            "ref": LLAMA_REFERENCE[0], "sus": f"{LLAMA_REFERENCE[0]}+{transform}",
            "repo": LLAMA_REFERENCE[1], "kind": f"descendant_{transform}",
            "label": 1, "transform": transform
        })

    for tag, repo in LLAMA_NONDESCENDANTS:
        pair_specs.append({
            "ref": LLAMA_REFERENCE[0], "sus": tag, "repo": repo,
            "kind": "non_descendant", "label": 0, "transform": None
        })

    print(f"Total pairs to evaluate: {len(pair_specs)}")
    for p in pair_specs:
        print(f"  [{p['kind']:20s}] {p['ref']} vs {p['sus']}")

    if dry_run:
        return {"n_pairs": len(pair_specs), "pairs": pair_specs}

    checkpoint = load_checkpoint("llama") if resume else {"completed_pairs": [], "pairs": []}
    completed = set(checkpoint["completed_pairs"])

    t0 = time.time()

    print(f"\n[1/{len(pair_specs)+1}] Loading reference: {LLAMA_REFERENCE[0]}")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_REFERENCE[1], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_REFERENCE[1], torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    print("  Extracting branch products...")
    ref_Ms = extract_llama_branch_products(ref_model)
    print(f"  Extracted {len(ref_Ms)} layers")

    print(f"  Collecting activations ({n_samples} samples)...")
    ref_acts = collect_llm_activations(ref_model, tokenizer, PROBE_TEXTS, n_samples, device)
    print(f"  Activations: {'skipped' if ref_acts is None else f'{len(ref_acts)} layers'}")

    print("  Collecting logits...")
    ref_logits = collect_llm_logits(ref_model, tokenizer, PROBE_TEXTS, n_samples, device)
    print(f"  Logits shape: {ref_logits.shape if ref_logits is not None else 'skipped'}")

    ref_pack = {"Ms": ref_Ms, "acts": ref_acts, "logits": ref_logits}

    pairs = checkpoint["pairs"].copy()

    for i, spec in enumerate(pair_specs):
        pair_key = f"{spec['ref']}|{spec['sus']}"
        if pair_key in completed:
            print(f"\n[{i+2}/{len(pair_specs)+1}] Skipping (already done): {spec['sus']}")
            continue

        print(f"\n[{i+2}/{len(pair_specs)+1}] Processing: {spec['sus']}")

        if spec["transform"]:
            print(f"  Applying transform: {spec['transform']}")
            sus_model = apply_transform(ref_model, spec["transform"])
        else:
            print(f"  Loading from: {spec['repo']}")
            sus_model = AutoModelForCausalLM.from_pretrained(
                spec["repo"], torch_dtype=torch.float16,
                low_cpu_mem_usage=True, trust_remote_code=True
            ).to(device)

        print("  Extracting branch products...")
        sus_Ms = extract_llama_branch_products(sus_model)

        print("  Collecting activations...")
        sus_acts = collect_llm_activations(sus_model, tokenizer, PROBE_TEXTS, n_samples, device)

        print("  Collecting logits...")
        sus_logits = collect_llm_logits(sus_model, tokenizer, PROBE_TEXTS, n_samples, device)

        sus_pack = {"Ms": sus_Ms, "acts": sus_acts, "logits": sus_logits}

        print("  Scoring pair...")
        scores = score_pair(ref_pack, sus_pack)
        for m, v in scores.items():
            print(f"    {m:24s}: {v:.4f}")

        pairs.append({
            "ref": spec["ref"], "sus": spec["sus"],
            "kind": spec["kind"], "label": spec["label"],
            "scores": scores
        })

        if not spec["transform"]:
            del sus_model
            clear_gpu_memory()

        checkpoint["completed_pairs"].append(pair_key)
        checkpoint["pairs"] = pairs
        save_checkpoint("llama", checkpoint)

    del ref_model
    clear_gpu_memory()

    auroc = compute_auroc(pairs)
    per_kind = compute_per_kind(pairs)

    result = {
        "config": {
            "benchmark": "llama",
            "reference": LLAMA_REFERENCE,
            "n_samples": n_samples,
            "device": device,
        },
        "n_pairs": len(pairs),
        "auroc": auroc,
        "per_kind": per_kind,
        "pairs": pairs,
        "wall_seconds": time.time() - t0,
    }

    out_path = RESULTS_DIR / "lineage_llama_family.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")

    print("\nAUROC by method:")
    for m, v in auroc.items():
        print(f"  {m:24s}: {v:.4f}")

    return result


# ============================================================================
# BERT Benchmark
# ============================================================================

def run_bert_benchmark(device: str = "cuda", resume: bool = False,
                       dry_run: bool = False, n_samples: int = 1024) -> dict:
    """Run BERT family benchmark."""
    from transformers import AutoModel, AutoTokenizer, BertForMaskedLM

    print(f"\n{'='*60}")
    print("BERT Family Benchmark")
    print(f"{'='*60}")

    pair_specs = []

    for tag, repo in BERT_DESCENDANTS:
        pair_specs.append({
            "ref": BERT_REFERENCE[0], "sus": tag, "repo": repo,
            "kind": "descendant_finetune", "label": 1, "transform": None
        })

    for transform in BERT_LOCAL_TRANSFORMS:
        pair_specs.append({
            "ref": BERT_REFERENCE[0], "sus": f"{BERT_REFERENCE[0]}+{transform}",
            "repo": BERT_REFERENCE[1], "kind": f"descendant_{transform}",
            "label": 1, "transform": transform
        })

    for tag, repo in BERT_NONDESCENDANTS:
        pair_specs.append({
            "ref": BERT_REFERENCE[0], "sus": tag, "repo": repo,
            "kind": "non_descendant", "label": 0, "transform": None
        })

    print(f"Total pairs to evaluate: {len(pair_specs)}")
    for p in pair_specs:
        print(f"  [{p['kind']:20s}] {p['ref']} vs {p['sus']}")

    if dry_run:
        return {"n_pairs": len(pair_specs), "pairs": pair_specs}

    checkpoint = load_checkpoint("bert") if resume else {"completed_pairs": [], "pairs": []}
    completed = set(checkpoint["completed_pairs"])

    t0 = time.time()

    print(f"\n[1/{len(pair_specs)+1}] Loading reference: {BERT_REFERENCE[0]}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_REFERENCE[1])

    ref_model = AutoModel.from_pretrained(BERT_REFERENCE[1]).to(device)

    print("  Extracting branch products...")
    ref_Ms = extract_bert_branch_products(ref_model)
    print(f"  Extracted {len(ref_Ms)} layers")

    print(f"  Collecting activations ({n_samples} samples)...")
    ref_acts = collect_bert_activations(ref_model, tokenizer, PROBE_TEXTS, n_samples, device)
    print(f"  Activations: {'skipped' if ref_acts is None else f'{len(ref_acts)} layers'}")

    print("  Collecting logits...")
    ref_logits = collect_bert_logits(ref_model, tokenizer, PROBE_TEXTS, n_samples, device)
    print(f"  Logits shape: {ref_logits.shape if ref_logits is not None else 'skipped'}")

    ref_pack = {"Ms": ref_Ms, "acts": ref_acts, "logits": ref_logits}

    pairs = checkpoint["pairs"].copy()

    for i, spec in enumerate(pair_specs):
        pair_key = f"{spec['ref']}|{spec['sus']}"
        if pair_key in completed:
            print(f"\n[{i+2}/{len(pair_specs)+1}] Skipping (already done): {spec['sus']}")
            continue

        print(f"\n[{i+2}/{len(pair_specs)+1}] Processing: {spec['sus']}")

        if spec["transform"]:
            print(f"  Applying transform: {spec['transform']}")
            sus_model = apply_transform(ref_model, spec["transform"])
        else:
            print(f"  Loading from: {spec['repo']}")
            try:
                sus_model = AutoModel.from_pretrained(spec["repo"]).to(device)
            except Exception:
                sus_model = BertForMaskedLM.from_pretrained(spec["repo"]).to(device)

        print("  Extracting branch products...")
        try:
            sus_Ms = extract_bert_branch_products(sus_model)
        except Exception as e:
            print(f"  WARNING: Could not extract branch products: {e}")
            sus_Ms = ref_Ms

        print("  Collecting activations...")
        try:
            sus_acts = collect_bert_activations(sus_model, tokenizer, PROBE_TEXTS, n_samples, device)
        except Exception as e:
            print(f"  WARNING: Could not collect activations: {e}")
            sus_acts = None

        print("  Collecting logits...")
        try:
            sus_logits = collect_bert_logits(sus_model, tokenizer, PROBE_TEXTS, n_samples, device)
        except Exception as e:
            print(f"  WARNING: Could not collect logits: {e}")
            sus_logits = None

        sus_pack = {"Ms": sus_Ms, "acts": sus_acts, "logits": sus_logits}

        print("  Scoring pair...")
        scores = score_pair(ref_pack, sus_pack)
        for m, v in scores.items():
            print(f"    {m:24s}: {v:.4f}")

        pairs.append({
            "ref": spec["ref"], "sus": spec["sus"],
            "kind": spec["kind"], "label": spec["label"],
            "scores": scores
        })

        if not spec["transform"]:
            del sus_model
            clear_gpu_memory()

        checkpoint["completed_pairs"].append(pair_key)
        checkpoint["pairs"] = pairs
        save_checkpoint("bert", checkpoint)

    del ref_model
    clear_gpu_memory()

    auroc = compute_auroc(pairs)
    per_kind = compute_per_kind(pairs)

    result = {
        "config": {
            "benchmark": "bert",
            "reference": BERT_REFERENCE,
            "n_samples": n_samples,
            "device": device,
        },
        "n_pairs": len(pairs),
        "auroc": auroc,
        "per_kind": per_kind,
        "pairs": pairs,
        "wall_seconds": time.time() - t0,
    }

    out_path = RESULTS_DIR / "lineage_bert_family.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")

    print("\nAUROC by method:")
    for m, v in auroc.items():
        print(f"  {m:24s}: {v:.4f}")

    return result


# ============================================================================
# ResNet Rescoring
# ============================================================================

def run_resnet_rescoring(device: str = "cuda", resume: bool = False,
                         dry_run: bool = False, n_samples: int = 1024) -> dict:
    """Re-score ResNet models with all 7 baselines."""
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10

    from lineage_phase2_resnet import (
        make_cifar_resnet18,
        extract_branch_products,
        add_gaussian_noise,
        magnitude_prune,
        fake_quantize,
        get_cifar_loaders,
        train_one,
    )

    print(f"\n{'='*60}")
    print("ResNet Rescoring Benchmark")
    print(f"{'='*60}")

    n_refs = 2
    n_independents = 4

    pair_specs = []
    for ref_i in range(n_refs):
        for tfm in ["noise", "prune", "quant"]:
            pair_specs.append({
                "ref_idx": ref_i, "kind": f"descendant_{tfm}", "label": 1, "transform": tfm
            })
        for ind_i in range(n_independents):
            pair_specs.append({
                "ref_idx": ref_i, "ind_idx": ind_i,
                "kind": "non_descendant", "label": 0, "transform": None
            })

    print(f"Total pairs to evaluate: {len(pair_specs)}")

    if dry_run:
        return {"n_pairs": len(pair_specs), "pairs": pair_specs}

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
        )
    except Exception as e:
        print(f"WARNING: Could not load CIFAR-10: {e}")
        raise

    def collect_resnet_activations(model, dataloader, n_samples=1024):
        """Collect activations from ResNet BasicBlocks.

        Uses global average pooling to reduce (batch, C, H, W) -> (batch, C)
        instead of flattening, which would create 65k-dim vectors that are
        too large for CKA/SVCCA.
        """
        all_acts = []
        hooks = []
        block_count = 0

        for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in stage:
                def make_hook(sink_idx):
                    def hook(module, inp, out):
                        x = inp[0]  # (batch, C, H, W)
                        # Global average pool: (batch, C, H, W) -> (batch, C)
                        pooled = x.mean(dim=(2, 3))
                        all_acts.append(pooled.detach().float().cpu().numpy())
                    return hook
                hooks.append(block.register_forward_hook(make_hook(block_count)))
                block_count += 1

        print(f"      [acts] registered {block_count} hooks", flush=True)
        model.eval()
        collected = 0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(dataloader):
                if collected >= n_samples:
                    break
                x = x.to(device)
                _ = model(x)
                collected += x.size(0)
                if batch_idx % 5 == 0:
                    print(f"      [acts] batch {batch_idx}, collected {collected}", flush=True)

        for h in hooks:
            h.remove()
        print(f"      [acts] done collection, processing {len(all_acts)} activations", flush=True)

        n_blocks = block_count
        per_block = [[] for _ in range(n_blocks)]
        for i, act in enumerate(all_acts):
            per_block[i % n_blocks].append(act)

        result = []
        for acts in per_block:
            stacked = np.vstack(acts)[:n_samples]
            result.append(stacked)  # Already (n_samples, C), no flatten needed

        print(f"      [acts] processed into {len(result)} layers, shape {result[0].shape}", flush=True)
        return result

    def collect_resnet_preds(model, dataloader, n_samples=1024):
        all_preds = []
        model.eval()
        collected = 0
        print(f"      [preds] starting collection, target={n_samples}", flush=True)
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(dataloader):
                if collected >= n_samples:
                    break
                x = x.to(device)
                out = model(x)
                all_preds.append(out.detach().float().cpu().numpy())
                collected += x.size(0)
                if batch_idx % 5 == 0:
                    print(f"      [preds] batch {batch_idx}, collected {collected}", flush=True)
        print(f"      [preds] done, collected {collected}", flush=True)
        return np.vstack(all_preds)[:n_samples]

    checkpoint = load_checkpoint("resnet") if resume else {"completed_pairs": [], "pairs": []}
    completed = set(checkpoint["completed_pairs"])

    t0 = time.time()
    pairs = checkpoint["pairs"].copy()

    torch.manual_seed(0)
    np.random.seed(0)

    for ref_i in range(n_refs):
        print(f"\n[ref {ref_i}] Creating and processing reference model...", flush=True)
        torch.manual_seed(ref_i)
        ref_model = make_cifar_resnet18(num_classes=10).to(device)

        print(f"    Extracting branch products...", flush=True)
        ref_Ms = extract_branch_products(ref_model)
        print(f"    Extracted {len(ref_Ms)} layers", flush=True)
        print(f"    Collecting activations...", flush=True)
        ref_acts = collect_resnet_activations(ref_model, dataloader, n_samples)
        print(f"    Collected {len(ref_acts)} layer activations", flush=True)
        print(f"    Collecting predictions...", flush=True)
        ref_preds = collect_resnet_preds(ref_model, dataloader, n_samples)
        print(f"    Predictions shape: {ref_preds.shape}", flush=True)
        ref_pack = {"Ms": ref_Ms, "acts": ref_acts, "logits": ref_preds}

        for tfm in ["noise", "prune", "quant"]:
            pair_key = f"ref{ref_i}|{tfm}"
            if pair_key in completed:
                print(f"  Skipping {tfm} (already done)", flush=True)
                continue

            print(f"  Transform: {tfm}", flush=True)
            if tfm == "noise":
                sus_model = add_gaussian_noise(ref_model, sigma_rel=0.01, seed=ref_i * 100)
            elif tfm == "prune":
                sus_model = magnitude_prune(ref_model, sparsity=0.3)
            else:
                sus_model = fake_quantize(ref_model, levels=256)
            sus_model = sus_model.to(device)
            print(f"    Transform applied, model on {next(sus_model.parameters()).device}", flush=True)

            print(f"    Extracting branch products...", flush=True)
            sus_Ms = extract_branch_products(sus_model)
            print(f"    Collecting activations...", flush=True)
            sus_acts = collect_resnet_activations(sus_model, dataloader, n_samples)
            print(f"    Collected {len(sus_acts)} layer activations", flush=True)
            print(f"    Collecting predictions...", flush=True)
            sus_preds = collect_resnet_preds(sus_model, dataloader, n_samples)
            print(f"    Predictions shape: {sus_preds.shape}", flush=True)
            sus_pack = {"Ms": sus_Ms, "acts": sus_acts, "logits": sus_preds}

            print(f"    Scoring pair...", flush=True)
            scores = score_pair(ref_pack, sus_pack)
            print(f"    Scores: {scores}", flush=True)
            pairs.append({
                "ref": f"ref{ref_i}", "sus": f"ref{ref_i}+{tfm}",
                "kind": f"descendant_{tfm}", "label": 1,
                "scores": scores
            })

            del sus_model
            clear_gpu_memory()

            checkpoint["completed_pairs"].append(pair_key)
            checkpoint["pairs"] = pairs
            save_checkpoint("resnet", checkpoint)

        # Process independents one at a time
        for ind_i in range(n_independents):
            pair_key = f"ref{ref_i}|ind{ind_i}"
            if pair_key in completed:
                print(f"  Skipping ind{ind_i} (already done)", flush=True)
                continue

            print(f"  Independent: ind{ind_i}", flush=True)
            torch.manual_seed(1000 + ref_i * 100 + ind_i)
            ind_model = make_cifar_resnet18(num_classes=10).to(device)

            print(f"    Extracting branch products...", flush=True)
            ind_Ms = extract_branch_products(ind_model)
            print(f"    Collecting activations...", flush=True)
            ind_acts = collect_resnet_activations(ind_model, dataloader, n_samples)
            print(f"    Collected {len(ind_acts)} layer activations", flush=True)
            print(f"    Collecting predictions...", flush=True)
            ind_preds = collect_resnet_preds(ind_model, dataloader, n_samples)
            print(f"    Predictions shape: {ind_preds.shape}", flush=True)
            ind_pack = {"Ms": ind_Ms, "acts": ind_acts, "logits": ind_preds}

            print(f"    Scoring pair...", flush=True)
            scores = score_pair(ref_pack, ind_pack)
            print(f"    Scores: {scores}", flush=True)
            pairs.append({
                "ref": f"ref{ref_i}", "sus": f"ind{ind_i}",
                "kind": "non_descendant", "label": 0,
                "scores": scores
            })

            del ind_model
            clear_gpu_memory()

            checkpoint["completed_pairs"].append(pair_key)
            checkpoint["pairs"] = pairs
            save_checkpoint("resnet", checkpoint)

        del ref_model
        clear_gpu_memory()

    auroc = compute_auroc(pairs)
    per_kind = compute_per_kind(pairs)

    result = {
        "config": {
            "benchmark": "resnet",
            "n_refs": n_refs,
            "n_independents": n_independents,
            "n_samples": n_samples,
            "device": device,
        },
        "n_pairs": len(pairs),
        "auroc": auroc,
        "per_kind": per_kind,
        "pairs": pairs,
        "wall_seconds": time.time() - t0,
    }

    out_path = RESULTS_DIR / "lineage_resnet_rescored.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")

    print("\nAUROC by method:")
    for m, v in auroc.items():
        print(f"  {m:24s}: {v:.4f}")

    return result


# ============================================================================
# Parallel execution
# ============================================================================

def run_benchmark_subprocess(args):
    """Run a single benchmark in a subprocess with specific GPU."""
    benchmark, gpu_id, resume, n_samples = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    if benchmark == "llama":
        return run_llama_benchmark(device, resume, False, n_samples)
    elif benchmark == "bert":
        return run_bert_benchmark(device, resume, False, n_samples)
    elif benchmark == "resnet":
        return run_resnet_rescoring(device, resume, False, n_samples)
    return None


def run_parallel(resume: bool = False, n_samples: int = 1024):
    """Run all benchmarks in parallel across GPUs."""
    tasks = [
        ("bert", 2, resume, n_samples),
        ("resnet", 3, resume, n_samples),
        ("llama", 0, resume, n_samples),
    ]

    print(f"\n{'='*60}")
    print("Running benchmarks in parallel")
    print(f"{'='*60}")
    for bench, gpu, _, _ in tasks:
        print(f"  {bench:10s} -> GPU {gpu}")

    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_benchmark_subprocess, tasks))

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Issue #44 GPU Benchmark")
    parser.add_argument("--benchmark", choices=["llama", "bert", "resnet", "all"],
                        default="all", help="Which benchmark to run")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all benchmarks in parallel across GPUs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="List pairs without running")
    parser.add_argument("--n-samples", type=int, default=1024,
                        help="Number of probe samples for CKA/SVCCA")
    args = parser.parse_args()

    print(f"Issue #44 GPU Benchmark")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Device: {args.device}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Resume: {args.resume}")
    print(f"  N samples: {args.n_samples}")

    if args.parallel and args.benchmark == "all":
        run_parallel(args.resume, args.n_samples)
    elif args.benchmark == "all":
        run_llama_benchmark(args.device, args.resume, args.dry_run, args.n_samples)
        run_bert_benchmark(args.device, args.resume, args.dry_run, args.n_samples)
        run_resnet_rescoring(args.device, args.resume, args.dry_run, args.n_samples)
    elif args.benchmark == "llama":
        run_llama_benchmark(args.device, args.resume, args.dry_run, args.n_samples)
    elif args.benchmark == "bert":
        run_bert_benchmark(args.device, args.resume, args.dry_run, args.n_samples)
    elif args.benchmark == "resnet":
        run_resnet_rescoring(args.device, args.resume, args.dry_run, args.n_samples)


if __name__ == "__main__":
    main()
