# Supplementary Code: Centered Residual Signatures for Language Model Lineage Verification

This repository contains the code and results to reproduce the experiments in the paper.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- Hugging Face Transformers (for LLM experiments)

Install dependencies:
```bash
pip install torch numpy scipy transformers datasets
```

## Directory Structure

```
submission/
├── scripts/           # Experiment scripts
│   ├── lineage_benchmark_mlp.py      # Table 3 (MLP benchmark)
│   ├── lineage_benchmark_gpt2.py     # Table 3 (GPT-2 benchmark)
│   ├── laundering_benchmark_mlp.py   # Table 5 (laundering, MLP)
│   ├── gpt2_laundering_baselines.py  # Table 5 (laundering, GPT-2)
│   ├── llm_lineage_real_v2.py        # Table 4 (LLaMA-2 case study)
│   ├── transformer_family_pairing.py # Table 2 (transformer families)
│   ├── init_scheme_ablation.py       # Table 12 (initialization ablation)
│   ├── ablate_centering.py           # Table 11 (centering ablation)
│   ├── make_fig_hero_real.py         # Figure 1
│   ├── fig2_redesign.py              # Figure 2
│   └── ...
├── results/           # Pre-computed results (JSON format)
└── README.md
```

## Reproducing Main Results

### Table 2: Transformer Family Pairing
```bash
python scripts/transformer_family_pairing.py
```

### Table 3: Baseline Comparison (MLP + GPT-2)
```bash
python scripts/lineage_benchmark_mlp.py
python scripts/lineage_benchmark_gpt2.py
python scripts/lineage_baselines.py
```

### Table 4: LLaMA-2 Case Study
```bash
python scripts/llm_lineage_real_v2.py
python scripts/llm_lineage_table4_nulls.py
```
Note: Requires ~50GB disk space for downloading LLaMA-2 and related checkpoints.

### Table 5: Laundering Robustness
```bash
python scripts/laundering_benchmark_mlp.py
python scripts/gpt2_laundering_baselines.py
```

### Appendix Experiments
```bash
python scripts/init_scheme_ablation.py      # Table 12
python scripts/ablate_centering.py          # Table 11
python scripts/lineage_harder_bench.py      # Table 8
python scripts/torchvision_resnet_pairing.py # ResNet results
python scripts/whisper_pairing.py           # Table 14
```

## Pre-computed Results

The `results/` directory contains JSON files with pre-computed experiment outputs:
- `lineage_benchmark_gpt2_paper/` - GPT-2 benchmark results
- `centering_ablation/` - Centering ablation data
- `laundering_latency/` - Latency measurements
- `table4_null_expansion.json` - Expanded null model results

## Hardware

Experiments were run on a single NVIDIA L4 GPU (24GB VRAM). The MLP benchmarks can run on CPU. LLM experiments require GPU with sufficient VRAM for model loading.

## Reproducibility

All scripts use fixed random seeds for reproducibility:
- `torch.manual_seed(0)`
- `np.random.seed(0)`
