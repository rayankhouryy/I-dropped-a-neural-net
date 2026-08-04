#!/usr/bin/env python3
"""Generate figure for GPT-2-Small-Lite lineage benchmark results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent / "results/lineage_benchmark_gpt2_paper/benchmark_results.json"
with open(results_path) as f:
    results = json.load(f)

# Extract scores by type
pairs = results["pairs"]
scores_by_type = {}
for pair in pairs:
    attack_type = pair["attack_type"]
    if attack_type not in scores_by_type:
        scores_by_type[attack_type] = []
    scores_by_type[attack_type].append(pair["lineage"])

# Define order and colors
descendant_types = ["quantized", "lora_merge", "continued_pretraining", "pruned"]
non_descendant_types = ["distilled_student", "independent"]

descendant_labels = ["Quantized\n(INT8/6)", "LoRA\n(rank-8)", "Fine-tuned\n(1 epoch)", "Pruned\n(30-70%)"]
non_descendant_labels = ["Distilled\nstudent", "Independent"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]})

# Panel A: Descendants vs Non-descendants violin/box plot
ax1 = axes[0]

# Prepare data
all_types = descendant_types + non_descendant_types
all_labels = descendant_labels + non_descendant_labels
all_scores = [scores_by_type[t] for t in all_types]

positions = np.arange(len(all_types))
colors = ['#2ecc71'] * len(descendant_types) + ['#e74c3c'] * len(non_descendant_types)

bp = ax1.boxplot(all_scores, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_xticks(positions)
ax1.set_xticklabels(all_labels, fontsize=9)
ax1.set_ylabel("Lineage Score $\mathcal{L}$", fontsize=11)
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(y=0.004, color='gray', linestyle='--', linewidth=1, label='Max null (0.004)')
ax1.axvline(x=3.5, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.7, label='Descendants (related)'),
    Patch(facecolor='#e74c3c', alpha=0.7, label='Non-descendants (unrelated)'),
]
ax1.legend(handles=legend_elements, loc='center right', fontsize=9)

ax1.set_title("(a) Lineage Scores by Transformation Type", fontsize=11)

# Panel B: Distillation quality vs lineage score
ax2 = axes[1]

# Extract distillation quality metrics
distilled = results["distilled_students"]
agreements = [d["quality_metrics"]["top1_agreement"] * 100 for d in distilled]
lineage_scores = [
    pair["lineage"] for pair in pairs
    if pair["attack_type"] == "distilled_student"
]

ax2.scatter(agreements, lineage_scores, s=80, c='#e74c3c', alpha=0.8, edgecolors='black', linewidth=0.5)
ax2.axhline(y=0.855, color='#2ecc71', linestyle='--', linewidth=1.5, label='Min descendant (0.855)')
ax2.axhline(y=0.004, color='gray', linestyle='--', linewidth=1, label='Max null (0.004)')

ax2.set_xlabel("Top-1 Agreement with Teacher (%)", fontsize=10)
ax2.set_ylabel("Lineage Score $\mathcal{L}$", fontsize=10)
ax2.set_xlim(78, 80)
ax2.set_ylim(-0.01, 0.01)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_title("(b) Distillation: Modest Behavioral\nGain, Low Weight Lineage", fontsize=11)

plt.tight_layout()

# Save
output_dir = Path(__file__).parent.parent.parent / "figures"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "fig_gpt2_lineage_benchmark.pdf"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved to {output_path}")

# Also save PNG for preview
plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=150)
print(f"Saved PNG preview")

plt.close()
