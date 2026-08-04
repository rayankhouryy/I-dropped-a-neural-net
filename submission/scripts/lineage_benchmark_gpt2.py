#!/usr/bin/env python3
"""Controlled GPT-2 Lineage Benchmark for ACL Paper.

This script trains GPT-2-style language models from scratch with known ancestry
and evaluates the lineage verification method on language models.

Key improvements over the MLP benchmark:
- NLP-relevant: actual language models on TinyStories
- Controlled ancestry: known ground truth for all descendants
- Root-disjoint splits: calibration/development/test
- Distillation quality metrics: validates that rejected students actually imitate well

Architecture: GPT-2-Small-Lite (~35M params)
- 6 layers, d_model=384, d_ff=1536, 6 heads
- Trained on TinyStories

Descendants per root (15 total):
- Continued pretraining (same corpus): 3
- Continued pretraining (domain shift): 2
- Supervised fine-tuning: 2
- LoRA + merge: 3
- Pruning: 3
- Quantization: 2

Plus 2 distilled students per root with quality metrics.

Outputs:
- results/lineage_benchmark_gpt2/benchmark_results.json
- results/lineage_benchmark_gpt2/checkpoints/ (all model checkpoints)

Usage:
    # Full benchmark (10 roots, ~24-36 hours)
    python lineage_benchmark_gpt2.py

    # Quick test (2 roots, ~2-4 hours)
    python lineage_benchmark_gpt2.py --n-calibration-roots 1 --n-development-roots 1 --n-test-roots 0 --epochs 2

    # Resume from checkpoints
    python lineage_benchmark_gpt2.py --skip-training

    # Debug mode (limited samples)
    python lineage_benchmark_gpt2.py --max-train-samples 1000 --epochs 1
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gpt2_lineage_benchmark.run_benchmark import main

if __name__ == "__main__":
    main()
