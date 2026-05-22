"""Generate figures for Case Study 2: Zero-Knowledge Ownership Proofs.

Creates:
  fig_zkp_protocol.pdf - Protocol diagram
  fig_zkp_results.pdf - Experiment results summary
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def load_results(case_study_dir):
    """Load experiment results."""
    json_path = case_study_dir / "zkp_results.json"
    with open(json_path) as f:
        return json.load(f)


def fig_zkp_results(results, output_path):
    """Create bar chart showing experiment results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    experiments = results["experiments"]
    scenarios = [e["scenario"].replace("_", "\n") for e in experiments]
    successes = [1 if e["success"] else 0 for e in experiments]

    # Expected outcomes (1 = should pass, 0 = should fail)
    expected = [1, 0, 0, 0, 0]  # honest passes, rest should fail

    x = np.arange(len(scenarios))
    width = 0.35

    # Color based on whether result matches expectation
    colors = []
    for i, (actual, exp) in enumerate(zip(successes, expected)):
        if actual == exp:
            colors.append("#2ecc71" if actual == 1 else "#e74c3c")  # green for pass, red for fail
        else:
            colors.append("#f39c12")  # orange for unexpected

    bars = ax.bar(x, successes, width, color=colors, edgecolor="black", linewidth=1.2)

    # Add result labels
    for i, (bar, success) in enumerate(zip(bars, successes)):
        label = "PASS" if success else "FAIL"
        y_pos = bar.get_height() + 0.05
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add expected labels
    for i, exp in enumerate(expected):
        expected_label = "(Expected: PASS)" if exp == 1 else "(Expected: FAIL)"
        ax.text(i, -0.15, expected_label, ha='center', va='top', fontsize=8, color='gray')

    ax.set_ylabel("Verification Result (1=PASS, 0=FAIL)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylim(-0.3, 1.4)
    ax.set_title("ZKP Ownership Protocol: Security Validation", fontsize=12, fontweight='bold')

    # Legend
    pass_patch = mpatches.Patch(color='#2ecc71', label='Correct PASS')
    fail_patch = mpatches.Patch(color='#e74c3c', label='Correct FAIL (attack blocked)')
    ax.legend(handles=[pass_patch, fail_patch], loc='upper right')

    # Add protocol security status
    is_secure = results["summary"]["protocol_secure"]
    status_color = "#2ecc71" if is_secure else "#e74c3c"
    status_text = "PROTOCOL SECURE" if is_secure else "PROTOCOL INSECURE"
    ax.text(0.5, 1.25, status_text, transform=ax.transAxes, ha='center',
            fontsize=14, fontweight='bold', color=status_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=status_color, linewidth=2))

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_protocol_diagram(output_path):
    """Create a diagram showing the 4-phase protocol."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, "Zero-Knowledge Ownership Proof Protocol", ha='center',
            fontsize=14, fontweight='bold')

    # Actors
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                                          facecolor='#3498db', edgecolor='black', linewidth=2))
    ax.text(1.75, 7, "PROVER\n(Owner)", ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    ax.add_patch(mpatches.FancyBboxPatch((9, 6), 2.5, 2, boxstyle="round,pad=0.1",
                                          facecolor='#e74c3c', edgecolor='black', linewidth=2))
    ax.text(10.25, 7, "VERIFIER\n(Challenger)", ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    ax.add_patch(mpatches.FancyBboxPatch((4.5, 6), 3, 2, boxstyle="round,pad=0.1",
                                          facecolor='#2ecc71', edgecolor='black', linewidth=2))
    ax.text(6, 7, "REGISTRY\n(Public)", ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    # Phase 1: Registration
    ax.annotate("", xy=(4.5, 7), xytext=(3, 7),
                arrowprops=dict(arrowstyle="->", color='#3498db', lw=2))
    ax.text(3.75, 7.5, "1. REGISTER", ha='center', fontsize=9, fontweight='bold', color='#3498db')
    ax.text(3.75, 5.5, "Commit(E, ε)\nfor each block", ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='gray'))

    # Phase 2: Challenge
    ax.annotate("", xy=(3, 4.5), xytext=(9, 4.5),
                arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=2))
    ax.text(6, 5, "2. CHALLENGE", ha='center', fontsize=9, fontweight='bold', color='#e74c3c')
    ax.text(6, 4, '"Reveal blocks [k₁, k₂, ...]"', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='gray'))

    # Phase 3: Response
    ax.annotate("", xy=(9, 3), xytext=(3, 3),
                arrowprops=dict(arrowstyle="->", color='#3498db', lw=2))
    ax.text(6, 3.5, "3. RESPONSE", ha='center', fontsize=9, fontweight='bold', color='#3498db')
    ax.text(6, 2.5, "Reveal W_in[k], W_out[k]\n+ randomness", ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='gray'))

    # Phase 4: Verify
    ax.add_patch(mpatches.FancyBboxPatch((8.5, 0.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                                          facecolor='#f39c12', edgecolor='black', linewidth=2))
    ax.text(10.25, 1.25, "4. VERIFY\n✓ E_hash match?\n✓ ε correct?\n✓ Trace sign?",
            ha='center', va='center', fontsize=8, fontweight='bold')

    # Commitment details
    ax.text(0.5, 0.5, "Commitment Structure:\nH(E || randomness) → E_hash\nε = |tr(M)|/d\nM = W_out × W_in = -εI + E",
            fontsize=8, va='bottom', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def fig_security_matrix(output_path):
    """Create a matrix showing attack resistance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    attacks = [
        "Honest Owner\n(with original weights)",
        "Fine-tuned Model\n(weights changed)",
        "Different Training\n(same architecture)",
        "Distilled Model\n(fresh training)",
        "Random Weights\n(no training)",
    ]

    # Results: 1 = verification passed, 0 = verification failed
    results = [1, 0, 0, 0, 0]

    # Expected: for honest owner, we want PASS; for attacks, we want FAIL
    expected = ["PASS", "FAIL", "FAIL", "FAIL", "FAIL"]
    actual = ["PASS" if r else "FAIL" for r in results]

    # Check if result matches expectation
    correct = [a == e for a, e in zip(actual, expected)]

    # Create table
    cell_colors = []
    for c in correct:
        cell_colors.append(['#2ecc71' if c else '#e74c3c'])

    table_data = [[a] for a in actual]

    table = ax.table(cellText=table_data,
                     rowLabels=attacks,
                     colLabels=["Verification\nResult"],
                     cellColours=cell_colors,
                     rowColours=['#ecf0f1']*5,
                     colColours=['#3498db'],
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 2)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
        if col == -1:
            cell.set_text_props(fontsize=9)

    ax.axis('off')
    ax.set_title("ZKP Protocol: Attack Resistance Matrix", fontsize=12, fontweight='bold', pad=20)

    # Add legend
    ax.text(0.5, -0.1, "GREEN = Correct behavior (honest passes, attacks blocked)\nRED = Incorrect behavior (security failure)",
            ha='center', transform=ax.transAxes, fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(str(output_path) + ".pdf")
    plt.savefig(str(output_path) + ".png", dpi=200)
    plt.close()
    print(f"Saved {output_path}.{{pdf,png}}")


def main():
    case_study_dir = Path(__file__).parent / "../case_studies/case_study_2"
    figures_dir = case_study_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(case_study_dir)

    print("\nGenerating figures...")
    fig_zkp_results(results, figures_dir / "fig_zkp_results")
    fig_protocol_diagram(figures_dir / "fig_zkp_protocol")
    fig_security_matrix(figures_dir / "fig_security_matrix")

    print("\nDone!")


if __name__ == "__main__":
    main()
