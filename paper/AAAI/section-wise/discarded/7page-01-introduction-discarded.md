# Introduction — Discarded Content

Content removed from original ~999 words to achieve ~550 word target.

---

## Extended Dynamical Isometry Preview (moved to Section 3)

Original text:
> In two-factor residual MLPs ($x + W_{\mathrm{out}} \phi(W_{\mathrm{in}} x)$), this constraint pulls the branch product toward a negative-identity-like component, $W_{\mathrm{out}} W_{\mathrm{in}} \approx -\varepsilon I$~\citep{he2016deep}, and admits a closed-form margin analysis. In multi-factor and attention-style branches (e.g., $W_3 W_2 W_1$, $W_O W_V$, $W_Q W_K^{\!\top}$, SwiGLU joint products), the same training pressure produces trace concentration in the architecture-appropriate product without necessarily taking the strict negative-identity form.

---

## Detailed Attack Configuration Numbers

Original text:
> The fingerprint maintains 100\% block-pair accuracy across 21 attack configurations (fine-tuning, quantization, pruning, weight noise up to ${\sim}20\%$)

Simplified to: "surviving fine-tuning, quantization, and pruning while degrading only when perturbations destroy model utility"

---

## Extended Prior Fingerprinting Discussion

Original text:
> Prior intrinsic fingerprinting methods have asked whether weight statistics---means, variances, or histogram features---can identify models~\citep{zheng2022fingerprinting,zhao2020shaping}, but these aggregate properties lack verification precision and are easily disrupted. Training dynamics research has established that gradient descent enforces near-orthogonal Jacobians in residual networks~\citep{pennington2017resurrecting,saxe2014exact}, yet this work focused on trainability, not on what structural traces the process leaves behind.

Condensed: removed fingerprinting prior work detail (citations kept); dynamical isometry context retained but shortened.

---

## Verbose Contributions List

Original contributions were ~150 words with full section references. Condensed to ~80 words in inline format.

Original:
> \item \textbf{We formalize the diagonal-dominance signal and derive a separation margin that grows with width.} Building on \citet{park2026}'s observation that training induces negative diagonal structure in residual-branch products, we define the score $s(i,j) = |\mathrm{tr}(M)|/\|M\|_F$ and prove that, in the residual-MLP regime captured by Proposition~\ref{prop:margin}, it achieves $\Theta(\sqrt{d})$ for correct pairs versus $\Theta(1/\sqrt{d})$ for mismatched pairs; GPT-2 MLPs follow this $\sqrt{d}$ envelope empirically (Section~\ref{sec:diagonal-dominance}).

---

## Extended Two-Tasks Explanation

Original text:
> Both tasks share the same underlying mechanism, but they have different inputs, outputs, and failure modes, and they should not be conflated.

Kept core distinction, removed redundant elaboration.

---

## Specific LLaMA Timeline

Original: "March 2023"
Removed: specific date (not essential)

---

## Cost Detail

Original: "each costing under \$300 to create"
Removed: cost figure (interesting but not essential for space)
