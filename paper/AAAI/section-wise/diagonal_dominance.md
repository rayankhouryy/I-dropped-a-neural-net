# Section 3: Training-Induced Diagonal Dominance - Research Document

This document consolidates research findings for Section 3 of the AAAI 2026 paper.

---

## Section 3.1: Empirical Observation

### Key Findings from ACML Paper

**Core Phenomenon (Figure 1, lines 91-99):**
1. At initialization (epoch 0), no diagonal structure is visible in the pairing matrix
2. By epoch 5, the diagonal is fully separated and pair accuracy reaches 100%
3. The signal emerges from gradient descent, not architectural constraints

**Training Dynamics (Section RQ5, lines 460-461):**
- Mean correct-pair score rises monotonically from 0.19 at epoch 0 to 2.07 at epoch 300
- Mean incorrect-pair score remains flat at approximately 0.16 throughout training
- The transformation occurs within the first few epochs of training

**Null Model Results (lines 219-234):**
- Untrained network (depth 48, hidden dim 96, Kaiming init): pair accuracy 6.25% (chance is 2.08%)
- Pair separation at init: -0.42 (negative = no separation)
- Trained network: pair accuracy 100%, pair separation +1.18
- Ratio between correct-pair and incorrect-pair means: 21x after training

**Initialization Scheme Ablation (Table 5, lines 440-458):**
All seven initialization schemes show chance-level accuracy at initialization:
- Kaiming uniform: 2.1% untrained -> 98.6% trained
- Kaiming normal: 2.1% -> 97.2%
- Xavier uniform: 2.1% -> 100%
- Xavier normal: 2.1% -> 98.6%
- Gaussian: 2.1% -> 75.0%
- Uniform: 2.1% -> 93.8%
- Orthogonal: 0.0% -> 100% (critical: even with dynamical isometry at init, no pairing signal)

**Non-Residual Control (lines 471-486):**
- ResNet (with skip connections): 100% pair accuracy, AUC 0.98, 100% negative traces
- PlainNet (no skip connections): 3% pair accuracy (chance ~4.2%), AUC 0.51, no diagonal structure
- Both networks achieve comparable eval loss (1.35 vs 1.56), confirming PlainNet learns the task

### Figures to Include

**Primary Figure - Training Dynamics Heatmaps (Figure 1 in ACML):**
```latex
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../figures/fig_null_a_heatmaps.pdf}\\[0.5em]
{\small (a) Training dynamics: 24-block ResNet}\\[1em]
\includegraphics[width=\textwidth]{../figures/fig_modern_vision_pairing.pdf}\\[0.5em]
{\small (b) Modern vision architectures: ViT-B/16 and ConvNeXt-T}
\caption{Diagonal-dominance matrices $s(i,j)$ reveal training-induced structure. \textbf{(a)} At initialization (epoch 0), no diagonal structure is visible. By epoch 5, the diagonal is fully separated and pair accuracy reaches 100\%. \textbf{(b)} ViT-B/16 (MLP, V/O attention, Q/K attention paths) and ConvNeXt-T all achieve 100\% pairing accuracy with clear diagonal structure.}
\label{fig:heatmaps}
\end{figure>
```

**Figure File Paths:**
- `/Users/amanzing/i-dropped-a-neural-network/I-dropped-a-neural-net/paper/figures/fig_null_a_heatmaps.pdf`
- `/Users/amanzing/i-dropped-a-neural-network/I-dropped-a-neural-net/paper/figures/fig_modern_vision_pairing.pdf`
- `/Users/amanzing/i-dropped-a-neural-network/I-dropped-a-neural-net/paper/figures/fig_nonresidual_baseline.pdf` (for non-residual control)
- `/Users/amanzing/i-dropped-a-neural-network/I-dropped-a-neural-net/paper/figures/fig_init_ablation.pdf` (for initialization ablation)

**Non-Residual Baseline Figure (Figure 11 in ACML):**
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{../figures/fig_nonresidual_baseline.pdf}
\caption{Non-residual control experiment. \textbf{(a)} Trained ResNet: clear diagonal structure, 100\% pair accuracy. \textbf{(b)} Trained PlainNet (no skip connections): uniform noise, 0\% pair accuracy (3\% averaged over seeds). Both networks achieve comparable eval loss, confirming PlainNet learns the task but lacks the structural fingerprint.}
\label{fig:nonresidual}
\end{figure>
```

### Draft Sentences with Citations

**Opening - phenomenon introduction:**
> Training a deep residual network imprints a characteristic structure on its weights. We first demonstrate this phenomenon empirically before formalizing it.

**Initialization observation:**
> At initialization, no diagonal structure appears in the pairing matrix $s(i,j)$. Across seven initialization schemes---including Kaiming, Xavier, Gaussian, and orthogonal---pair accuracy remains at chance level ($\leq 2.1\%$; Table~\ref{tab:init_ablation}). Critically, orthogonal initialization provides dynamical isometry at initialization by construction \citep{saxe2014exact}, yet shows \emph{zero} correct pairs before training, ruling out the hypothesis that near-orthogonal Jacobians alone create the fingerprint.

**Training emergence:**
> During training, diagonal structure emerges rapidly. By epoch 5, the diagonal is fully separated and Hungarian matching recovers 100\% of correct pairs (Figure~\ref{fig:heatmaps}a). The mean correct-pair score rises monotonically from 0.19 at epoch 0 to 2.07 at epoch 300, while the mean incorrect-pair score remains flat at $\approx 0.16$---a 21$\times$ ratio at convergence.

**Not explained by architecture alone:**
> The effect is not explained by matrix shape or residual architecture alone. A null model with randomly initialized weights matching our 48-block network achieves only 6.25\% pair accuracy (chance is 2.08\%) with negative pair separation (-0.42). Three alternative explanations are ruled out: (a) diagonal dominance is not an artifact of compatible matrix shapes; (b) the residual connection itself does not create the signal; (c) any nontrivial loss level does not suffice.

**Non-residual control:**
> To confirm the fingerprint requires residual training specifically, we train a PlainNet---architecturally identical except that skip connections are removed. While both networks converge to similar eval loss (1.35 vs 1.56), ResNet achieves 100\% pair accuracy with clear diagonal structure, while PlainNet achieves only 3\%---at chance baseline---with no discernible pattern (Figure~\ref{fig:nonresidual}). The skip connection creates a functional constraint that imprints the characteristic trace structure.

### Metrics/Numbers

| Metric | Value | Context |
|--------|-------|---------|
| Emergence epoch | 5 | When 100% pair accuracy first achieved |
| Mean $s(i,i)$ at epoch 0 | 0.19 | Untrained correct-pair score |
| Mean $s(i,i)$ at epoch 300 | 2.07 | Trained correct-pair score |
| Mean $s(i,j)$ incorrect | ~0.16 | Remains flat during training |
| Correct/incorrect ratio | 21x | At convergence |
| Null model pair accuracy | 6.25% | Untrained (chance = 2.08%) |
| Null model pair separation | -0.42 | Negative = no separation |
| Trained pair separation | +1.18 | Positive = clear separation |
| Orthogonal init untrained | 0.0% | Even with dynamical isometry |
| PlainNet pair accuracy | 3% | No skip connections (chance ~4.2%) |
| ResNet pair accuracy | 100% | With skip connections |
| PlainNet AUC | 0.51 | Random classifier level |
| ResNet AUC | 0.98 | Near-perfect separation |

### Established vs Novel Claims

| Claim | Status | Citation/Evidence |
|-------|--------|-------------------|
| Dynamical isometry enables deep network training | Established | \citet{pennington2017isometry}, \citet{saxe2014exact} |
| Residual networks achieve dynamical isometry | Established | \citet{tarnowski2019dynamical} |
| Skip connections preserve gradient norms | Established | \citet{zaeemzadeh2020norm} |
| BatchNorm biases residual blocks toward identity | Established | \citet{de2020batch} |
| $W_{out}W_{in} \approx -\varepsilon I$ emerges from training | Novel | ACML paper empirical results |
| Diagonal structure absent at initialization | Novel | Null model experiments |
| Signal emerges by epoch 5 | Novel | Training dynamics experiments |
| Orthogonal init shows 0% pairing before training | Novel | Initialization ablation (Table 5) |
| Non-residual networks show no fingerprint | Novel | PlainNet control experiment |
| 21x correct/incorrect ratio after training | Novel | ACML paper empirical results |

### Key Citations for Section 3.1

1. **Dynamical isometry theory:**
   - `\citet{pennington2017isometry}` - Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry
   - `\citet{saxe2014exact}` - Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Networks

2. **ResNets and gradient flow:**
   - `\citet{he2016deep}` - Deep Residual Learning for Image Recognition
   - `\citet{tarnowski2019dynamical}` - Dynamical Isometry is Achieved in Residual Networks
   - `\citet{zaeemzadeh2020norm}` - Norm-Preservation: Why Residual Networks Can Become Extremely Deep

3. **BatchNorm and identity bias:**
   - `\citet{de2020batch}` - Batch Normalization Biases Residual Blocks Towards the Identity Function

4. **Original puzzle network:**
   - `\citet{park2026}` - I Dropped a Neural Net (arXiv:2602.19845)

---

## Section 3.2: Diagonal-Dominance Fingerprint

### Key Definition from ACML Paper

**LaTeX equation (Equation 3 in ACML paper, labeled eq:dd-score):**
```latex
s(i,j) \;=\; \frac{|\mathrm{tr}(M)|}{\|M\|_F}
```

**Context from ACML Section 3.1 (lines 155-168):**

For a residual branch of depth K with weight matrices W_1, ..., W_K, compute the full branch product:
```latex
M = W_K \cdots W_1 \in \mathbb{R}^{d \times d}
```

The score definition explanation from ACML:
- **Numerator** `|tr(M)|` = "diagonal mass" - captures how much of the matrix energy concentrates on the diagonal
- **Denominator** `||M||_F` = "total matrix energy" - normalizes by scale (Frobenius norm)

**Key insight stated in ACML (line 165-167):**
> "This ratio isolates structural fingerprint from scale: a matrix with energy uniformly spread has s(i,j) = Theta(1/sqrt(d)), while a matrix proportional to the identity achieves s(i,j) = sqrt(d)."

### Geometric Intuition

**Why the formula works (from ACML Section 3.1, lines 166-168):**

1. **Correct pairs** (all K matrices from same residual branch): Score at Theta(sqrt(d))
2. **Incorrect pairings**: Remain at Theta(1/sqrt(d)) random baseline
3. **Signal-to-noise gap**: Grows linearly with hidden dimension d

**The trained decomposition (from ACML Section 3.2, lines 172-177):**

Dynamic isometry predicts that a well-trained residual block has Jacobian J = I + W_out W_in close to orthogonal. For the block to act as a near-identity perturbation of the residual stream, the product must approximate:
```latex
M := W_{\mathrm{out}}^{(i)} W_{\mathrm{in}}^{(i)} = -\varepsilon I + E
```
where:
- epsilon := |tr(M)|/d > 0 captures diagonal strength
- E is the residual with tr(E) = 0
- E contains all off-diagonal structure plus the zero-trace portion of the diagonal
- Dynamic isometry implies ||E||_F << epsilon*sqrt(d) for well-trained blocks

**Why dividing by Frobenius norm matters (from ACML RQ3, lines 416-417):**
> "Dividing by ||M||_F normalizes away scale variation and isolates the trace contribution that only correctly paired products possess."

### Draft Sentences with Citations

**Opening definition (can use directly):**
"For a candidate residual branch product M_{ij}, we define the diagonal-dominance score:
s(i,j) = |tr(M_{ij})| / ||M_{ij}||_F.
The numerator captures diagonal mass while the denominator normalizes by total matrix energy, isolating structural fingerprint from scale."

**Geometric interpretation sentence:**
"A matrix with energy uniformly spread achieves s(i,j) = Theta(1/sqrt(d)), while a matrix proportional to the identity achieves s(i,j) = sqrt(d). The key insight is that correctly paired blocks---where all K matrices come from the same residual branch---score at Theta(sqrt(d)), while incorrect pairings remain at the random baseline Theta(1/sqrt(d))."

**Relational structure sentence (from ACML Section 2.3, line 151):**
"Crucially, diagonal dominance captures *relational* structure between paired matrices---how they jointly encode the near-identity mapping---rather than aggregate properties of individual matrices."

**Citations needed:**
- \citet{pennington2017isometry} - for dynamical isometry prediction that J = I + W_out W_in close to orthogonal
- Background on trace-based metrics: No direct citation found in ACML paper; this appears to be a novel formulation

### Key Numbers

**Score values from ACML experiments:**

| Model | Hidden dim d | Mean s(i,i) | sqrt(d) theoretical max |
|-------|-------------|-------------|------------------------|
| GPT-2 | 768 | 4.18 | 27.7 |
| GPT-2-medium | - | 5.46 | - |
| GPT-2-large | - | 6.98 | - |
| GPT-2-xl | 1600 | 7.77 | 40.0 |
| Park puzzle network (48 blocks) | 48 | 1.76-3.23 | 6.93 |

**Incorrect pair baseline:**
- E[s(i,j)] approximately 1/sqrt(d) for i != j
- GPT-2 random init: mean s(i,i) = 0.12

**Fitted values from Park puzzle network (line 200):**
- ||E||_F / (epsilon*sqrt(d)) ranges from 1.90 to 3.80 across blocks
- This yields s(i,i) in [1.76, 3.23]

**Signal-to-noise ratio:**
- Trained puzzle network: 21x ratio between correct-pair and incorrect-pair means (line 231)

### Established vs Novel Claims

| Claim | Status | Citation/Evidence |
|-------|--------|-------------------|
| Residual blocks enforce J close to orthogonal | ESTABLISHED | \citet{pennington2017isometry} dynamical isometry |
| W_out W_in approximates -epsilon I for trained blocks | ESTABLISHED | Consequence of dynamical isometry theory |
| The specific score formula s = |tr(M)| / ||M||_F | NOVEL | No prior citation found; appears original to this work |
| Score separates correct vs incorrect at Theta(sqrt(d)) vs Theta(1/sqrt(d)) | NOVEL | Derived in Proposition 3.1 (margin formula) |
| Signal-to-noise gap grows linearly with d | NOVEL | Corollary 3.2 (signal-to-baseline ratio) |
| Diagonal dominance captures relational structure, not aggregate statistics | NOVEL | Key differentiator from prior fingerprinting work |
| Division by Frobenius norm isolates trace contribution | NOVEL | Methodological insight |

### Additional Context for Section 3.2

**What Section 3.2 should establish (based on paper structure):**

Section 3.2 defines the statistic. The formal margin analysis (Proposition) belongs in Section 3.3. Section 3.2 should:

1. Present the formula with clear notation
2. Explain numerator = diagonal mass
3. Explain denominator = scale normalization
4. State (without proving) that correct products preserve diagonal concentration
5. State (without proving) that incorrect products behave like random products
6. Emphasize that the statistic captures relational structure between matrices

**Connection to why this works (bridge to Section 3.3):**

The ACML paper's decomposition M = -epsilon I + E explains why the score works:
- For trained blocks: epsilon d = |tr(M)|, and ||M||_F^2 = epsilon^2 d + ||E||_F^2
- When ||E||_F is small relative to epsilon*sqrt(d), the score approaches sqrt(d)
- For random/incorrect pairs: no such structure exists, score stays at 1/sqrt(d)

---

## Section 3.3: Margin Analysis and Null Model

### Theorems/Proofs from ACML Paper

#### Proposition (Margin Formula) - ACML Section 3.2, Proposition 3.1

```latex
\begin{proposition}[Margin Formula]\label{prop:margin}
Let $M = -\varepsilon I + E$ with $\varepsilon = |\mathrm{tr}(M)|/d$ and $\mathrm{tr}(E) = 0$. Then the diagonal-dominance score satisfies
\begin{equation}\label{eq:margin-formula}
  s(i,i) \;=\; \frac{\sqrt{d}}{\sqrt{1 + \|E\|_F^2 / (\varepsilon^2 d)}} \;\leq\; \sqrt{d},
\end{equation}
with equality if and only if $E = 0$, i.e., $M = -\varepsilon I$.
\end{proposition}
```

#### Proof of Proposition

```latex
\begin{proof}
By construction, $|\mathrm{tr}(M)| = \varepsilon d$. For the Frobenius norm, expand
\begin{align*}
  \|M\|_F^2 &= \|-\varepsilon I + E\|_F^2 = \varepsilon^2 \|I\|_F^2 - 2\varepsilon \langle I, E \rangle + \|E\|_F^2 \\
            &= \varepsilon^2 d - 2\varepsilon \,\mathrm{tr}(E) + \|E\|_F^2 = \varepsilon^2 d + \|E\|_F^2,
\end{align*}
where the last step uses $\mathrm{tr}(E) = 0$. Substituting into \eqref{eq:dd-score}:
\[
  s(i,i) = \frac{\varepsilon d}{\sqrt{\varepsilon^2 d + \|E\|_F^2}} = \frac{\sqrt{d}}{\sqrt{1 + \|E\|_F^2/(\varepsilon^2 d)}}.
\]
The bound $s(i,i) \leq \sqrt{d}$ follows immediately, with equality iff $\|E\|_F = 0$.
\end{proof}
```

#### Corollary (Signal-to-Baseline Ratio) - ACML Corollary 3.2

```latex
\begin{corollary}[Signal-to-Baseline Ratio]\label{cor:baseline}
Suppose $(W_{\mathrm{in}}^{(i)}, W_{\mathrm{out}}^{(j)})$ with $i \neq j$ are independent random matrices with i.i.d.\ zero-mean entries. Then
\begin{equation}\label{eq:baseline}
  \mathbb{E}[s(i,j)^2] = \frac{1}{d}.
\end{equation}
Consequently, the expected score for incorrect pairs is $\mathbb{E}[s(i,j)] \approx 1/\sqrt{d}$, while correct pairs score $s(i,i) = O(\sqrt{d})$ by Proposition~\ref{prop:margin}. The signal-to-baseline ratio scales as $d$.
\end{corollary}
```

#### Corollary (Null Model) - ACML Corollary 3.3

```latex
\begin{corollary}[Null Model]\label{cor:null-model}
For randomly initialized weights with no training, all pairs---correct and incorrect---satisfy
\begin{equation}\label{eq:null}
  \mathbb{E}[s(i,j)] = \frac{1}{\sqrt{d}} + o(1)
\end{equation}
uniformly in $(i,j)$. Hungarian matching on $-d$ recovers correct pairs at the chance rate $1/D$.
\end{corollary}
```

### Figures

#### Margin Theorem Verification Figure (fig_margin_theorem.pdf)

```latex
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../figures/fig_margin_theorem.pdf}
\caption{Proposition~\ref{prop:margin} verification. \textbf{Left:} Predicted $s(i,i)$ from the margin formula vs.\ empirical values for all 48 blocks---points lie on the identity line. \textbf{Right:} Sorted correct-pair margins (blue) vs.\ incorrect-pair baseline (red dashed) and theoretical upper bound $\sqrt{d}$ (grey dotted).}
\label{fig:margin}
\end{figure}
```

### Draft Sentences with Citations

**Opening claim (key theoretical statement):**
> "If training induces $M \approx -\varepsilon I + E$, then diagonal dominance separates correct products from random incorrect products."

**Margin formula explanation:**
> "The margin formula reveals the geometry of diagonal dominance. The numerator $\sqrt{d}$ is the score achievable by a perfect negative-scalar identity; the denominator penalizes deviation from this ideal via the dimensionless ratio $\|E\|_F^2/(\varepsilon^2 d)$."

**Signal-to-noise scaling:**
> "Correct pairs achieve scores of order $\sqrt{d}$ (diminished from the maximum only by off-diagonal energy $\|E\|_F$), while incorrect pairs score at the random baseline $1/\sqrt{d}$. The gap grows with dimension, providing increasingly robust separation for deeper or wider networks."

**Null model motivation:**
> "A skeptical reader might wonder whether diagonal dominance is an artifact of matrix shapes or the residual architecture itself, rather than a learned property. We rule this out with a null model."

**Null model result:**
> "Without training, there is no mechanism coupling $W_{\mathrm{in}}^{(i)}$ with $W_{\mathrm{out}}^{(i)}$ more than with any other $W_{\mathrm{out}}^{(j)}$. The decomposition $M = -\varepsilon I + E$ does not hold because $\varepsilon$ is not systematically positive for $i = j$."

**What the null model rules out:**
> "The null model eliminates three alternative explanations: (a) diagonal dominance is not an artifact of compatible matrix shapes; (b) the residual connection itself does not create the signal; (c) any nontrivial loss level does not suffice. The diagonal-dominance fingerprint is a \emph{learned} property induced by gradient descent, exactly as dynamical isometry theory predicts."

**Honest theoretical caveat:**
> "The theory explains why the observed structure is identifiable once induced; controlled experiments show that training induces it."

**Dynamical isometry citation:**
> "Dynamic isometry \citep{pennington2017isometry} predicts that a well-trained residual block has Jacobian $J = I + W_{\mathrm{out}} W_{\mathrm{in}}$ close to orthogonal."

### Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Correct pair score | $s(i,i) = \Theta(\sqrt{d})$ | Proposition 3.1 |
| Incorrect pair score | $\mathbb{E}[s(i,j)] = 1/\sqrt{d}$ | Corollary 3.2 |
| Signal-to-baseline ratio | Scales as $d$ | Corollary 3.2 |
| $\|E\|_F/(\varepsilon\sqrt{d})$ range | $[1.9, 3.8]$ (Park's puzzle network) | ACML empirical |
| $s(i,i)$ range | $[1.76, 3.23]$ | ACML empirical |
| Theoretical maximum | $\sqrt{48} \approx 6.93$ (for $d=48$) | ACML empirical |
| Null model pair accuracy | $6.25\%$ (chance $\approx 2.08\%$) | ACML empirical |
| Null model pair separation | $-0.42$ | ACML empirical |
| Trained model pair separation | $+1.18$ | ACML empirical |
| Correct/incorrect mean ratio | $21\times$ | ACML empirical |

### Established vs Novel Claims

| Claim | Status | Citation/Evidence |
|-------|--------|-------------------|
| Dynamical isometry in ResNets | **Established** | \citet{pennington2017isometry}, \citet{tarnowski2019dynamical} |
| $J = I + W_{\mathrm{out}} W_{\mathrm{in}}$ near orthogonal for trained blocks | **Established** | \citet{pennington2017isometry} |
| BatchNorm biases blocks toward identity | **Established** | \citet{de2020batch} |
| Skip connections preserve gradient norms | **Established** | \citet{zaeemzadeh2020norm} |
| $M = -\varepsilon I + E$ decomposition for trained residual blocks | **Novel formalization** | This work |
| Margin formula: $s(i,i) = \sqrt{d}/\sqrt{1 + \|E\|_F^2/(\varepsilon^2 d)}$ | **Novel** | Proposition 3.1 |
| Correct pairs at $\Theta(\sqrt{d})$, incorrect at $\Theta(1/\sqrt{d})$ | **Novel** | Corollary 3.2 |
| Signal-to-noise gap scales linearly with $d$ | **Novel** | Corollary 3.2 |
| Null model: untrained networks show no signal | **Novel empirical** | Corollary 3.3, experiments |
| Training induces the structure (not architecture alone) | **Novel empirical** | Null model + initialization ablation |
| Gradient descent always induces $M \approx -\varepsilon I$ | **NOT claimed** | We claim training induces it; we do not prove GD necessarily converges to this |

### Key Theoretical Contributions

1. **Margin Formula** - Exact closed-form expression for diagonal-dominance score in terms of the decomposition $M = -\varepsilon I + E$. The formula reproduces empirical scores to floating-point precision.

2. **Signal-to-Baseline Ratio** - Random matrices have expected score $1/\sqrt{d}$; trained correct pairs achieve $\sqrt{d}$. The gap is linear in dimension, explaining why wider models have more robust fingerprints.

3. **Null Model** - Establishes that the fingerprint is training-induced, not architectural. Three alternative explanations are ruled out: (a) shape compatibility, (b) residual architecture, (c) any loss level.

### Important Caveats to Include

- The theory explains **why the structure is identifiable once induced**, not that gradient descent always induces it
- Empirical verification (null model, training dynamics, initialization ablation) shows training induces the structure
- The decomposition $M = -\varepsilon I + E$ with $\text{tr}(E) = 0$ is motivated by dynamical isometry but the connection is not rigorously proven

---
