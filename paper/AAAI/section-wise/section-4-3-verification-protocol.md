# Section 4.3: Verification Protocol

## Overview

This section presents an end-to-end protocol for passive weight-level provenance verification. The protocol enables a practitioner---model hub auditor, intellectual property investigator, or regulatory body---to determine whether a suspect checkpoint descends from a reference model using only the model weights, without requiring watermarks, metadata, or cooperation from the original trainer.

---

## 4.3.1 Protocol Definition

We formalize the verification procedure as Algorithm~1. The protocol takes as input a reference checkpoint $A$, a suspect checkpoint $B$, and a pre-calibrated null distribution $\mathcal{N}$, and outputs one of three verdicts: **DESCENDANT**, **NON-DESCENDANT**, or **INCONCLUSIVE**.

```latex
\begin{algorithm}[t]
\caption{Diagonal-Dominance Provenance Verification}
\label{alg:verification}
\begin{algorithmic}[1]
\REQUIRE Reference checkpoint $A$, suspect checkpoint $B$
\REQUIRE Null distribution $\mathcal{N}$ (scores from unrelated model pairs)
\REQUIRE Architecture extractor $\mathcal{E}$, thresholds $\tau_{\text{upper}}, \tau_{\text{lower}}$
\ENSURE Verdict $\in \{\textsc{Descendant}, \textsc{Non-Descendant}, \textsc{Inconclusive}\}$

\STATE \textbf{Step 1: Architecture Compatibility Check}
\IF{$\text{arch}(A) \neq \text{arch}(B)$}
    \RETURN \textsc{Incompatible}
\ENDIF

\STATE \textbf{Step 2: Extract Branch Products}
\STATE $\{M_\ell^A\}_{\ell=1}^{L} \gets \mathcal{E}(A)$ \COMMENT{Architecture-aware extraction}
\STATE $\{M_m^B\}_{m=1}^{L} \gets \mathcal{E}(B)$

\STATE \textbf{Step 3: Compute Residual Signatures}
\FOR{$\ell = 1, \ldots, L$}
    \STATE $\alpha_\ell^A \gets \mathrm{tr}(M_\ell^A)/d$
    \STATE $R_\ell^A \gets M_\ell^A - \alpha_\ell^A I$
    \STATE $\phi_\ell^A \gets \mathrm{vec}(R_\ell^A) / (\|\mathrm{vec}(R_\ell^A)\|_2 + \delta)$
\ENDFOR
\STATE Compute $\{\phi_m^B\}_{m=1}^{L}$ analogously

\STATE \textbf{Step 4: Compute Gated Similarity Matrix}
\FOR{$\ell, m = 1, \ldots, L$}
    \STATE $C_{\ell m} \gets \langle \phi_\ell^A, \phi_m^B \rangle$
    \STATE $g_{\ell m} \gets \min(s(M_\ell^A)/\tau_s, s(M_m^B)/\tau_s, 1)$
    \STATE $G_{\ell m} \gets C_{\ell m} \cdot g_{\ell m}$
\ENDFOR

\STATE \textbf{Step 5: Align Branches via Hungarian Matching}
\STATE $\hat{\pi} \gets \arg\max_{\pi \in S_L} \sum_{\ell} G_{\ell, \pi(\ell)}$

\STATE \textbf{Step 6: Aggregate to Model-Level Score}
\STATE $\mathcal{L}(A,B) \gets \frac{1}{L} \sum_{\ell=1}^{L} G_{\ell, \hat{\pi}(\ell)}$

\STATE \textbf{Step 7: Calibrate Against Null Distribution}
\STATE $Z(A,B) \gets \frac{\mathcal{L}(A,B) - \mu(\mathcal{N})}{\sigma(\mathcal{N}) + \delta}$

\STATE \textbf{Step 8: Decision}
\IF{$Z(A,B) > \tau_{\text{upper}}$}
    \RETURN \textsc{Descendant}
\ELSIF{$Z(A,B) < \tau_{\text{lower}}$}
    \RETURN \textsc{Non-Descendant}
\ELSE
    \RETURN \textsc{Inconclusive}
\ENDIF
\end{algorithmic}
\end{algorithm}
```

---

## 4.3.2 Architecture Compatibility Check

The verification protocol requires that the reference and suspect checkpoints share the same architecture family. Specifically:

1. **Identical layer count**: $L_A = L_B$
2. **Matching hidden dimensions**: $d_A = d_B$
3. **Compatible residual structure**: Both models use the same residual block type (BasicBlock, Bottleneck, Transformer MLP, etc.)

When architectures differ, the protocol returns **INCOMPATIBLE** rather than a provenance verdict. Cross-architecture lineage (e.g., from a large teacher to a smaller distilled student) erases the weight-level fingerprint by design---distillation transfers function, not weight structure---and such cases should be addressed by behavioral verification methods.

**Rationale**: The diagonal-dominance fingerprint captures relational structure between paired weight matrices within a residual branch. When architectures differ, there is no principled correspondence between branches, and alignment becomes undefined.

---

## 4.3.3 Architecture-Aware Extraction

The extraction procedure $\mathcal{E}$ computes branch products according to the residual block structure. Table~\ref{tab:extraction} specifies the extraction formula for each architecture family.

```latex
\begin{table}[t]
\centering
\caption{Architecture-aware branch product extraction. Each formula computes the composite transformation $M \in \mathbb{R}^{d \times d}$ for one residual branch.}
\label{tab:extraction}
\begin{tabular}{lll}
\toprule
\textbf{Architecture} & \textbf{Component} & \textbf{Branch Product Formula} \\
\midrule
\multicolumn{3}{l}{\textit{Convolutional Residual Networks}} \\
\quad BasicBlock & Conv path & $M = W_{\text{conv2}} W_{\text{conv1}}$ \\
\quad Bottleneck & Conv path & $M = W_3 W_2 W_1$ \\
\midrule
\multicolumn{3}{l}{\textit{Transformer MLP Blocks}} \\
\quad Standard MLP & FFN & $M = W_{\text{down}} W_{\text{up}}$ \\
\quad SwiGLU MLP & FFN & $M = W_{\text{down}} \odot W_{\text{gate}} \cdot W_{\text{up}}$\textsuperscript{*} \\
\midrule
\multicolumn{3}{l}{\textit{Transformer Attention}} \\
\quad Value-Output & V/O path & $M = W_O W_V$ \\
\quad Query-Key & Q/K path & $M = W_Q W_K^\top$ \\
\bottomrule
\end{tabular}
\begin{flushleft}
\textsuperscript{*}For SwiGLU, the effective branch product uses the gated combination; see Appendix for derivation.
\end{flushleft}
\end{table}
```

**Critical finding**: Using the wrong extraction formula yields chance-level detection accuracy. In our experiments on GPT-2 (12 layers, $d=768$), applying the correct MLP formula ($M = W_{\text{down}} W_{\text{up}}$) achieves 100\% branch detection, while applying an incorrect formula (e.g., treating attention weights as MLP) reduces accuracy to the $1/L \approx 8\%$ random baseline.

---

## 4.3.4 Null Distribution Construction

The null distribution $\mathcal{N}$ provides the statistical reference against which suspect scores are compared. Construction proceeds as follows:

**Definition**: Let $\mathcal{U} = \{U_1, \ldots, U_n\}$ be a set of $n$ independently trained models sharing the same architecture as the reference $A$, where no $U_j$ descends from $A$. The null distribution is:
$$
\mathcal{N}_A = \left\{ \mathcal{L}(A, U_j) : U_j \in \mathcal{U} \right\}_{j=1}^{n}
$$

**Construction requirements**:
1. **Independence**: Each $U_j$ must be trained from a different random initialization
2. **Task diversity**: Include models trained on different datasets or objectives
3. **Seed diversity**: Multiple random seeds per configuration
4. **Sufficient sample size**: We recommend $n \geq 20$ for stable variance estimates

**Sources for null models**:
- Different random seeds on the same task
- Same architecture trained on different datasets
- Publicly available checkpoints from independent training runs
- Fresh random initializations (provide a lower bound on unrelated scores)

**Empirical distribution characteristics**: In our experiments with GPT-2 family models, the null distribution exhibits:
- Mean: $\mu(\mathcal{N}) \approx 0.02$ (near zero cosine similarity)
- Standard deviation: $\sigma(\mathcal{N}) \approx 0.03$
- Shape: Approximately Gaussian with light tails

---

## 4.3.5 Threshold Selection and Decision Criteria

Given a calibrated null distribution $\mathcal{N}$ with mean $\mu$ and standard deviation $\sigma$, we convert the raw lineage score to a z-score:
$$
Z(A,B) = \frac{\mathcal{L}(A,B) - \mu(\mathcal{N})}{\sigma(\mathcal{N}) + \delta}
$$
where $\delta = 10^{-8}$ prevents division by zero.

**Threshold selection**: We define two thresholds corresponding to confidence levels:

| Threshold | Value | Interpretation |
|-----------|-------|----------------|
| $\tau_{\text{upper}}$ | $Z_{0.95} \approx 1.645$ | 95\% confidence for positive verdict |
| $\tau_{\text{lower}}$ | $Z_{0.05} \approx -1.645$ | 95\% confidence for negative verdict |

Under the assumption that the null distribution is approximately Gaussian:
- $Z > 1.645$ implies $p < 0.05$ under $H_0$ (unrelated models)
- $Z > 2.576$ implies $p < 0.005$ (99\% confidence)
- $Z > 3.291$ implies $p < 0.0005$ (99.95\% confidence)

**Conservative recommendation**: For high-stakes applications (legal disputes, regulatory audits), we recommend $\tau_{\text{upper}} = 3.0$ to minimize false positives.

---

## 4.3.6 Output Categories

The protocol produces one of four outputs:

```latex
\begin{description}
\item[\textsc{Descendant}] ($Z > \tau_{\text{upper}}$): The suspect checkpoint preserves branch-specific weight structure from the reference with high confidence. The z-score provides a quantitative measure of evidence strength.

\item[\textsc{Non-Descendant}] ($Z < \tau_{\text{lower}}$): The suspect checkpoint shows no more similarity to the reference than expected between independently trained models. This rules out direct descent through fine-tuning, quantization, or pruning.

\item[\textsc{Inconclusive}] ($\tau_{\text{lower}} \leq Z \leq \tau_{\text{upper}}$): The evidence is ambiguous. This may occur when:
\begin{itemize}
    \item The suspect has undergone extensive modification
    \item The reference model was itself derived from a third model
    \item Noise or quantization has partially degraded the fingerprint
\end{itemize}

\item[\textsc{Incompatible}] (architecture mismatch): The models cannot be compared because they have different architectures. No provenance verdict is possible.
\end{description}
```

---

## 4.3.7 Robustness Guarantees

The protocol maintains reliability under common model transformations:

```latex
\begin{table}[t]
\centering
\caption{Verification robustness under model transformations. Detection accuracy is the fraction of true descendants correctly identified at $\tau_{\text{upper}} = 1.645$.}
\label{tab:robustness}
\begin{tabular}{lcc}
\toprule
\textbf{Transformation} & \textbf{Parameters} & \textbf{Detection} \\
\midrule
Fine-tuning & LR $10^{-5}$--$10^{-3}$, 50 epochs & 100\% \\
Gaussian noise & $\sigma \leq 0.3$ & 100\% \\
Gaussian noise & $\sigma = 0.5$ & 96\% \\
Gaussian noise & $\sigma = 1.0$ & 85\% \\
Quantization & 16-bit, 8-bit & 100\% \\
Quantization & 6-bit & 94\% \\
Quantization & 4-bit & 45\% \\
Pruning & Up to 30\% sparsity & 100\% \\
LoRA merge & Rank 8--64 & 100\% \\
\bottomrule
\end{tabular}
\end{table}
```

**Degradation boundary**: The fingerprint degrades gracefully with perturbation intensity, collapsing only when perturbations also substantially degrade model utility. At $\sigma = 1.0$ noise (where model perplexity increases by $>50\%$), diagonal similarity remains at 85\% versus 53\% for full-matrix methods.

---

## 4.3.8 Failure Modes and Limitations

The protocol cannot provide a verdict in the following cases:

**1. Architecture Mismatch**
Cross-architecture lineage (e.g., knowledge distillation from a larger teacher to a smaller student) erases the weight-level fingerprint. The protocol correctly abstains rather than producing a false negative.

**2. Extreme Perturbation**
When weight perturbation is severe enough to destroy model utility (e.g., 2-bit quantization, $\sigma > 1.0$ noise), the fingerprint may be unrecoverable. However, such perturbations also render the model functionally unusable.

**3. Shared Ancestry**
If both the reference and suspect descend from a common ancestor (e.g., both fine-tuned from a public base model), the protocol may report similarity arising from the shared grandparent rather than direct descent. Practitioners should compare against the earliest available checkpoint in the lineage.

**4. Insufficient Null Distribution**
With fewer than 20 null samples, variance estimates may be unreliable, leading to miscalibrated z-scores. We recommend constructing null distributions specific to each architecture family.

**5. Adversarial Evasion**
A motivated adversary with white-box access to both checkpoints could potentially craft perturbations that remove the fingerprint while preserving functionality. We analyze such adaptive attacks in Section~6.

---

## 4.3.9 Practical Implementation

**Computational cost**: For a model with $L$ layers and hidden dimension $d$:
- Branch product computation: $O(L \cdot d^2 \cdot k)$ where $k$ is branch depth
- Similarity matrix: $O(L^2 \cdot d^2)$
- Hungarian matching: $O(L^3)$
- Total: Dominated by similarity matrix, but with $L \leq 100$ and precomputed fingerprints, verification completes in under 1 second on commodity hardware.

**Memory**: Each branch product $M_\ell \in \mathbb{R}^{d \times d}$ requires $d^2 \cdot 4$ bytes (float32). For GPT-2 ($d=768$, $L=12$), this is approximately 28 MB per model.

**Reference implementation**: We provide a Python implementation using PyTorch and SciPy that verifies a (reference, suspect) pair in:
- 0.3 seconds for GPT-2 (124M parameters)
- 1.2 seconds for GPT-2-XL (1.5B parameters)

---

## 4.3.10 Protocol Summary

The diagonal-dominance verification protocol provides:

1. **Passive verification**: No watermarks or metadata required; works retroactively on any residual model
2. **Calibrated decisions**: Z-scores provide quantitative evidence strength with interpretable confidence levels
3. **Three-way output**: Explicit handling of inconclusive cases prevents overconfident verdicts
4. **Robustness**: Survives fine-tuning, quantization, pruning, and noise at levels that preserve model utility
5. **Efficiency**: Sub-second verification on commodity hardware

The protocol fills a critical gap in model provenance: it enables auditing of checkpoints that were deployed without foresight, where external evidence is unavailable or untrustworthy.

---

## LaTeX Formulas for Direct Inclusion

### Core Definitions

```latex
% Branch product extraction
M_\ell = W_{\mathrm{out},\ell} W_{\mathrm{in},\ell} \in \mathbb{R}^{d \times d}

% Diagonal dominance score
s(M) = \frac{|\mathrm{tr}(M)|}{\|M\|_F}

% Residual signature
\alpha_\ell^\star = \frac{\mathrm{tr}(M_\ell)}{d}, \qquad R_\ell = M_\ell - \alpha_\ell^\star I

% Normalized signature
\phi(M_\ell) = \frac{\mathrm{vec}(R_\ell)}{\|\mathrm{vec}(R_\ell)\|_2 + \delta}

% Gated branch similarity
G_{\ell m}(A,B) = \langle \phi(M_\ell^A), \phi(M_m^B) \rangle \cdot \min\left(\frac{s(M_\ell^A)}{\tau_s}, \frac{s(M_m^B)}{\tau_s}, 1\right)

% Hungarian alignment
\hat{\pi} = \arg\max_{\pi \in S_L} \sum_{\ell=1}^{L} G_{\ell,\pi(\ell)}(A,B)

% Model-level lineage score
\mathcal{L}(A,B) = \frac{1}{L} \sum_{\ell=1}^{L} G_{\ell,\hat{\pi}(\ell)}(A,B)

% Calibrated z-score
Z(A,B) = \frac{\mathcal{L}(A,B) - \mu(\mathcal{N}_A)}{\sigma(\mathcal{N}_A) + \delta}

% Decision rule
\text{Verdict} = \begin{cases}
\textsc{Descendant} & \text{if } Z(A,B) > \tau_{\text{upper}} \\
\textsc{Non-Descendant} & \text{if } Z(A,B) < \tau_{\text{lower}} \\
\textsc{Inconclusive} & \text{otherwise}
\end{cases}
```

### Null Distribution Definition

```latex
\mathcal{N}_A = \left\{ \mathcal{L}(A, U_j) : U_j \notin \mathcal{D}(A) \right\}_{j=1}^{n}
```

### Decision Thresholds (95% Confidence)

```latex
\tau_{\text{upper}} = 1.645 \quad \text{(descendant if } p < 0.05 \text{)}
\tau_{\text{lower}} = -1.645 \quad \text{(non-descendant if } p < 0.05 \text{)}
```

---

## Comparison with Watermarking Verification Protocols

Unlike watermarking approaches (Uchida et al. 2017, Adi et al. 2018, Rouhani et al. 2019), the diagonal-dominance protocol:

| Property | Watermarking | Diagonal Dominance |
|----------|-------------|-------------------|
| Insertion required | Yes, during training | No, passive |
| Retroactive verification | No | Yes |
| Survives fine-tuning | Often not (Lukas et al. 2022) | Yes |
| False positive control | Explicit key space | Statistical calibration |
| Adversarial removal | Black-box attacks exist | Requires utility destruction |

The key distinction is that watermarks embed *artificial* signals that can be removed without affecting model function, while diagonal dominance arises from the optimization process itself and cannot be removed without degrading the trained structure that enables model performance.
