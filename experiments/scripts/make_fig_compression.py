"""Generate figures for Case Study 3: Model Compression Auditing.

Creates:
  fig_compression_overview.pdf - Bar chart of E correlation by method
  fig_pruning_robustness.pdf - E correlation vs sparsity level
  fig_quantization_robustness.pdf - E correlation vs bit width
  fig_derivation_detection.pdf - Comparison: compression vs non-derived
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

METHOD_COLORS = {
    "magnitude_pruning": "#3498db",
    "structured_pruning": "#2980b9",
    "quantization": "#2ecc71",
    "fine_tuning": "#f39c12",
    "distillation": "#e74c3c",
    "independent_training": "#9b59b6",
}

VERDICT_COLORS = {
    "DERIVED": "#2ecc71",
    "LIKELY_DERIVED": "#27ae60",
    "INCONCLUSIVE": "#f39c12",
    "NOT_DERIVED": "#e74c3c",
}


def load_data(case_study_dir):
    """Load experiment results."""
    csv_path = case_study_dir / "compression_audit_results.csv"
    json_path = case_study_dir / "compression_audit_summary.json"

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        summary = json.load(f)

    return df, summary


def fig_compression_overview(df, summary, output_path):
    """Bar chart showing E correlation for all methods."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = []
    correlations = []
    colors = []
    verdicts = []

    for _, row in df.iterrows():
        params = json.loads(row['params'])
        if row['method'] == 'magnitude_pruning':
            label = f"Mag Prune {int(params['sparsity']*100)}%"
        elif row['method'] == 'structured_pruning':
            label = f"Struct Prune {int(params['sparsity']*100)}%"
        elif row['method'] == 'quantization':
            label = f"INT{params['bits']}"
        elif row['method'] == 'fine_tuning':
            label = f"Fine-tune {params['epochs']}ep"
        elif row['method'] == 'distillation':
            label = "Distillation"
        elif row['method'] == 'independent_training':
            label = "Independent\nTraining"
        else:
            label = row['method']

        methods.append(label)
        correlations.append(row['mean_E_corr'])
        colors.append(VERDICT_COLORS[row['verdict']])
        verdicts.append(row['verdict'])

    x = np.arange(len(methods))
    bars = ax.bar(x, correlations, color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0.90, color='#2ecc71', linestyle='--', linewidth=1.5, label='DERIVED threshold')
    ax.axhline(y=0.70, color='#27ae60', linestyle=':', linewidth=1.5, label='LIKELY threshold')
    ax.axhline(y=0.30, color='#e74c3c', linestyle='--', linewidth=1.5, label='NOT_DERIVED threshold')

    ax.set_ylabel("E Matrix Correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Compression Audit: E Matrix Correlation by Method", fontsize=12, fontweight='bold')

    for i, (bar, corr, verdict) in enumerate(zip(bars, correlations, verdicts)):
        y_pos = max(corr + 0.03, 0.05)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{corr:.2f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    derived_patch = mpatches.Patch(color='#2ecc71', label='DERIVED (>0.90)')
    likely_patch = mpatches.Patch(color='#27ae60', label='LIKELY (0.70-0.90)')
    inconc_patch = mpatches.Patch(color='#f39c12', label='INCONCLUSIVE')
    not_patch = mpatches.Patch(color='#e74c3c', label='NOT_DERIVED (<0.30)')
    ax.legend(handles=[derived_patch, likely_patch, inconc_patch, not_patch],
              loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_pruning_robustness(df, output_path):
    """Line plot: E correlation vs sparsity for pruning methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mag_df = df[df['method'] == 'magnitude_pruning'].copy()
    mag_df['sparsity'] = mag_df['params'].apply(lambda x: json.loads(x)['sparsity'])
    mag_df = mag_df.sort_values('sparsity')

    struct_df = df[df['method'] == 'structured_pruning'].copy()
    struct_df['sparsity'] = struct_df['params'].apply(lambda x: json.loads(x)['sparsity'])
    struct_df = struct_df.sort_values('sparsity')

    ax.plot(mag_df['sparsity'] * 100, mag_df['mean_E_corr'], 'o-',
            color=METHOD_COLORS['magnitude_pruning'], linewidth=2, markersize=10,
            label='Magnitude Pruning')

    ax.plot(struct_df['sparsity'] * 100, struct_df['mean_E_corr'], 's-',
            color=METHOD_COLORS['structured_pruning'], linewidth=2, markersize=10,
            label='Structured Pruning')

    ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.70, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=0.30, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.text(92, 0.92, 'DERIVED', fontsize=8, color='gray')
    ax.text(92, 0.72, 'LIKELY', fontsize=8, color='gray')
    ax.text(92, 0.32, 'NOT_DERIVED', fontsize=8, color='gray')

    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("E Matrix Correlation")
    ax.set_xlim(25, 95)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Pruning Robustness: E Correlation vs Sparsity", fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')

    ax.fill_between([25, 95], 0.90, 1.1, alpha=0.1, color='#2ecc71')
    ax.fill_between([25, 95], 0.70, 0.90, alpha=0.1, color='#27ae60')
    ax.fill_between([25, 95], -0.1, 0.30, alpha=0.1, color='#e74c3c')

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_quantization_robustness(df, output_path):
    """Bar chart: E correlation for different bit widths."""
    fig, ax = plt.subplots(figsize=(7, 5))

    quant_df = df[df['method'] == 'quantization'].copy()
    quant_df['bits'] = quant_df['params'].apply(lambda x: json.loads(x)['bits'])
    quant_df = quant_df.sort_values('bits', ascending=False)

    x = np.arange(len(quant_df))
    colors = [VERDICT_COLORS[v] for v in quant_df['verdict']]

    bars = ax.bar(x, quant_df['mean_E_corr'], color=colors, edgecolor='black', linewidth=1)

    ax.axhline(y=0.90, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0.70, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.7)

    bit_labels = [f"INT{b}" if b < 16 else "FP16" for b in quant_df['bits']]
    ax.set_xticks(x)
    ax.set_xticklabels(bit_labels, fontsize=11)
    ax.set_ylabel("E Matrix Correlation")
    ax.set_ylim(0, 1.1)
    ax.set_title("Quantization Robustness: E Correlation vs Bit Width", fontsize=12, fontweight='bold')

    for i, (bar, corr) in enumerate(zip(bars, quant_df['mean_E_corr'])):
        ax.text(bar.get_x() + bar.get_width()/2, corr + 0.02, f"{corr:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.text(2.5, 0.92, 'DERIVED', fontsize=9, color='#2ecc71', fontweight='bold')
    ax.text(2.5, 0.72, 'LIKELY', fontsize=9, color='#27ae60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_derivation_detection(df, summary, output_path):
    """Comparison: compression methods vs non-derived models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    derived_methods = df[df['method'].isin(['magnitude_pruning', 'structured_pruning', 'quantization', 'fine_tuning'])]
    not_derived_methods = df[df['method'].isin(['distillation', 'independent_training'])]

    derived_corrs = derived_methods['mean_E_corr'].values
    not_derived_corrs = not_derived_methods['mean_E_corr'].values

    bp1 = ax1.boxplot([derived_corrs, not_derived_corrs], positions=[1, 2],
                       patch_artist=True, widths=0.6)
    bp1['boxes'][0].set_facecolor('#2ecc71')
    bp1['boxes'][1].set_facecolor('#e74c3c')

    ax1.scatter(np.ones(len(derived_corrs)) + np.random.randn(len(derived_corrs))*0.05,
                derived_corrs, color='#27ae60', alpha=0.7, s=50)
    ax1.scatter(np.ones(len(not_derived_corrs))*2 + np.random.randn(len(not_derived_corrs))*0.05,
                not_derived_corrs, color='#c0392b', alpha=0.7, s=50)

    ax1.axhline(y=0.30, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.text(2.5, 0.32, 'Threshold', fontsize=8, color='gray')

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Compressed\n(Prune/Quant/FT)', 'Non-Derived\n(Distill/Indep)'])
    ax1.set_ylabel("E Matrix Correlation")
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_title("Derivation Detection: Clear Separation", fontsize=11, fontweight='bold')

    ax2 = axes[1]

    prune_corrs = df[df['method'].isin(['magnitude_pruning', 'structured_pruning'])]['mean_E_corr']
    quant_corrs = df[df['method'] == 'quantization']['mean_E_corr']
    ft_corrs = df[df['method'] == 'fine_tuning']['mean_E_corr']
    distill_corrs = df[df['method'] == 'distillation']['mean_E_corr']
    indep_corrs = df[df['method'] == 'independent_training']['mean_E_corr']

    categories = ['Pruning', 'Quantization', 'Fine-tune', 'Distillation', 'Independent']
    means = [prune_corrs.mean(), quant_corrs.mean(), ft_corrs.mean(),
             distill_corrs.mean(), indep_corrs.mean()]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

    bars = ax2.bar(categories, means, color=colors, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.30, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + 0.03, f"{mean:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel("Mean E Correlation")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Mean E Correlation by Method Category", fontsize=11, fontweight='bold')
    ax2.set_xticklabels(categories, rotation=20, ha='right')

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_hypothesis_validation(summary, output_path):
    """Visual summary of hypothesis validation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    hypotheses = [
        ("Pruning preserves fingerprint", summary['hypothesis_validation']['pruning_preserves_fingerprint'],
         "E correlation stays high even at 70% sparsity"),
        ("Quantization preserves fingerprint", summary['hypothesis_validation']['quantization_preserves_fingerprint'],
         "E correlation stays >0.95 down to INT4"),
        ("Distillation erases fingerprint", summary['hypothesis_validation']['distillation_erases_fingerprint'],
         "Distilled model has near-zero E correlation"),
        ("Independent training has different E", summary['hypothesis_validation']['independent_has_different_fingerprint'],
         "Same architecture, different training = different E"),
    ]

    ax.text(0.5, 0.95, "Hypothesis Validation Summary", ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)

    for i, (hyp, result, note) in enumerate(hypotheses):
        y = 0.80 - i * 0.18

        status = "✓ CONFIRMED" if result else "✗ REJECTED"
        color = '#2ecc71' if result else '#e74c3c'

        ax.add_patch(mpatches.FancyBboxPatch((0.05, y-0.06), 0.9, 0.14,
                                              boxstyle="round,pad=0.02",
                                              facecolor='#ecf0f1', edgecolor=color,
                                              linewidth=2, transform=ax.transAxes))

        ax.text(0.08, y + 0.02, hyp, ha='left', va='center',
                fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.08, y - 0.03, note, ha='left', va='center',
                fontsize=9, color='gray', transform=ax.transAxes)
        ax.text(0.92, y, status, ha='right', va='center',
                fontsize=11, fontweight='bold', color=color, transform=ax.transAxes)

    all_confirmed = summary['hypothesis_validation']['all_hypotheses_confirmed']
    final_status = "ALL HYPOTHESES CONFIRMED" if all_confirmed else "SOME HYPOTHESES REJECTED"
    final_color = '#2ecc71' if all_confirmed else '#e74c3c'

    ax.text(0.5, 0.08, final_status, ha='center', va='center',
            fontsize=14, fontweight='bold', color=final_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=final_color, linewidth=3),
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def main():
    case_study_dir = Path(__file__).parent / "../case_studies/case_study_3"
    figures_dir = case_study_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, summary = load_data(case_study_dir)

    print("\nGenerating figures...")
    fig_compression_overview(df, summary, figures_dir / "fig_compression_overview")
    fig_pruning_robustness(df, figures_dir / "fig_pruning_robustness")
    fig_quantization_robustness(df, figures_dir / "fig_quantization_robustness")
    fig_derivation_detection(df, summary, figures_dir / "fig_derivation_detection")
    fig_hypothesis_validation(summary, figures_dir / "fig_hypothesis_validation")

    print("\nDone!")


if __name__ == "__main__":
    main()
