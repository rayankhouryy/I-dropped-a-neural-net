#!/usr/bin/env python3
"""Measure per-method latency on GPT-2 benchmark (single seed, quick run).

Usage:
    python measure_gpt2_latency.py --benchmark-dir results/lineage_benchmark_gpt2_paper_v2

Outputs latency_stats.json with mean/std/min/max per method.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import gpt2_laundering_baselines as lbase
import lineage_detection as ldet
import laundering_gpt2_ops as lops
from gpt2_lineage_benchmark.extraction import extract_branch_products
from gpt2_lineage_benchmark.model import load_checkpoint
from gpt2_lineage_benchmark.config import ModelConfig


def score_with_latency(
    ref_Ms: List[np.ndarray],
    sus_Ms: List[np.ndarray],
    ref_raw: Dict[str, List[np.ndarray]],
    sus_raw: Dict[str, List[np.ndarray]],
    tau_s: float,
) -> Dict[str, float]:
    """Score a pair and measure latency for each method."""
    latencies = {}

    # Ours (centered residual signature)
    t0 = time.perf_counter()
    ldet.lineage_score(ref_Ms, sus_Ms, tau_s)
    latencies["centered_residual_signature"] = (time.perf_counter() - t0) * 1000

    # Raw weight cosine
    t0 = time.perf_counter()
    lbase.raw_weight_cosine_gpt2(ref_raw, sus_raw)
    latencies["raw_weight_cosine"] = (time.perf_counter() - t0) * 1000

    # Raw aligned Frobenius
    t0 = time.perf_counter()
    lbase.raw_aligned_frobenius_gpt2(ref_raw, sus_raw)
    latencies["raw_aligned_frobenius"] = (time.perf_counter() - t0) * 1000

    # Singular value distance
    t0 = time.perf_counter()
    lbase.singular_value_distance_gpt2(ref_raw, sus_raw)
    latencies["singular_value_distance"] = (time.perf_counter() - t0) * 1000

    # Re-Basin + scale
    t0 = time.perf_counter()
    lbase.rebasin_scale_frobenius_gpt2(ref_raw, sus_raw)
    latencies["rebasin_scale_frobenius"] = (time.perf_counter() - t0) * 1000

    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir",
                        default="results/lineage_benchmark_gpt2_paper_v2",
                        help="Path to benchmark data")
    parser.add_argument("--n-pairs", type=int, default=20,
                        help="Number of pairs to measure (default: 20)")
    parser.add_argument("--output", default="results/gpt2_latency_stats.json")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load benchmark data
    phase1_path = benchmark_dir / "phase1_roots.pkl"
    phase2_path = benchmark_dir / "phase2_descendants.pkl"

    with open(phase1_path, "rb") as f:
        phase1_data = pickle.load(f)
    with open(phase2_path, "rb") as f:
        phase2_data = pickle.load(f)

    tau_s = phase1_data["tau_s"]
    root_signatures = phase1_data["root_signatures"]
    descendants = phase2_data["descendants"]
    model_config = ModelConfig()

    print(f"Loaded {len(root_signatures)} roots, {len(descendants)} descendants")
    print(f"tau_s = {tau_s:.4f}")

    # Collect latencies
    all_latencies = defaultdict(list)
    checkpoint_dir = benchmark_dir / "checkpoints"

    n_pairs = min(args.n_pairs, len(descendants))
    print(f"Measuring latency on {n_pairs} pairs...")

    for i, desc in enumerate(descendants[:n_pairs]):
        root_idx = desc["root_idx"]

        # Load reference
        ref_ckpt = checkpoint_dir / f"root_{root_idx}" / "epoch_3.pt"
        ref_model, _, _ = load_checkpoint(ref_ckpt, model_config, device)
        ref_Ms = root_signatures[root_idx]
        ref_raw = lops.raw_weights_gpt2(ref_model)

        # Get suspect Ms and raw weights
        sus_Ms = desc["Ms"]
        # For raw weights, we need to load the model if available
        models_dir = benchmark_dir / "models"
        sus_ckpt = models_dir / f"{desc['id']}.pt"
        if sus_ckpt.exists():
            sus_model, _, _ = load_checkpoint(sus_ckpt, model_config, device)
            sus_raw = lops.raw_weights_gpt2(sus_model)
            sus_model.cpu()
            del sus_model
        else:
            # Use ref as placeholder (scores won't be meaningful but latency is)
            sus_raw = ref_raw

        # Measure latency
        latencies = score_with_latency(ref_Ms, sus_Ms, ref_raw, sus_raw, tau_s)
        for method, lat in latencies.items():
            all_latencies[method].append(lat)

        # Cleanup
        ref_model.cpu()
        del ref_model
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n_pairs}] done")

    # Compute statistics
    stats = {}
    print("\n" + "=" * 50)
    print("Latency Statistics (ms)")
    print("=" * 50)
    print(f"{'Method':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 62)

    for method in ["centered_residual_signature", "rebasin_scale_frobenius",
                   "raw_aligned_frobenius", "singular_value_distance", "raw_weight_cosine"]:
        lats = all_latencies[method]
        stats[method] = {
            "mean_ms": float(np.mean(lats)),
            "std_ms": float(np.std(lats)),
            "min_ms": float(np.min(lats)),
            "max_ms": float(np.max(lats)),
            "n_calls": len(lats),
        }
        print(f"{method:<30} {stats[method]['mean_ms']:8.2f} "
              f"{stats[method]['std_ms']:8.2f} {stats[method]['min_ms']:8.2f} "
              f"{stats[method]['max_ms']:8.2f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
