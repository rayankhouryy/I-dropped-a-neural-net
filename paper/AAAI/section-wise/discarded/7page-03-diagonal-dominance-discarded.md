# Diagonal Dominance — Discarded Content

Content removed from original ~1,335 words to achieve ~700 word target.

---

## Extended Empirical Observation (initialization ablation detail)

Original text:
> Across seven initialization schemes---Kaiming uniform/normal, Xavier uniform/normal, Gaussian, uniform, and orthogonal---pair accuracy remains at chance level ($\leq 2.1\%$).

Condensed to: single sentence about orthogonal init showing zero pairs.

---

## Detailed Training Dynamics

Original text:
> The mean correct-pair score rises monotonically from 0.19 at epoch 0 to 2.07 at epoch 300, while the mean incorrect-pair score remains flat at ${\approx}0.16$---a $21\times$ ratio at convergence. The transformation occurs within the first few epochs, well before training loss plateaus.

Condensed to: "signal emerges rapidly, well before loss plateaus"

---

## Extended PlainNet Discussion

Original text:
> To confirm the fingerprint requires residual training specifically, we train a PlainNet---architecturally identical except that skip connections are removed. While both networks converge to similar evaluation loss (1.35 vs 1.56), ResNet achieves 100\% pair accuracy (AUC 0.98) with 100\% of correctly paired products exhibiting negative trace, while PlainNet achieves only 3\%---at the chance baseline of $1/24 \approx 4.2\%$---with AUC 0.51 and no discernible diagonal structure. The skip connection creates a functional constraint---the block must act as a near-identity perturbation to preserve gradient flow---that imprints the characteristic trace structure.

Condensed to: single sentence with key numbers (3% vs 100%).

---

## Verbose Geometric Interpretation

Original text:
> This ratio has a natural geometric interpretation. A matrix with energy uniformly spread across entries achieves $s = \Theta(1/\sqrt{d})$, while a matrix proportional to the identity achieves $s = \sqrt{d}$. The key insight is that correctly paired blocks---where all $K$ matrices come from the same residual branch---score at $\Theta(\sqrt{d})$, while incorrect pairings remain at the random baseline $\Theta(1/\sqrt{d})$. This yields a signal-to-noise gap that grows linearly with hidden dimension $d$, making correct pairs increasingly separable in wider networks.

Condensed to: two sentences stating the key insight.

---

## Extended Decomposition Discussion

Original text:
> The matrix $E$ contains all off-diagonal structure plus the zero-trace portion of the diagonal. Dynamical isometry implies $\|E\|_F \ll \varepsilon\sqrt{d}$ for well-trained blocks; the following proposition quantifies the margin exactly. The decomposition is exact for any matrix, but its predictive power depends on the regime: in two-factor residual MLPs the negative-identity component dominates, while in multi-factor and attention-style products (Section~\ref{sec:experiments}) the trained product retains the same trace-concentration signal even when $E$ is too large for the negative-identity geometry to hold cleanly.

Removed entirely (regime discussion deferred to experiments).

---

## Verbose Margin Formula Discussion

Original text:
> The margin formula reveals the geometry of diagonal dominance. The numerator $\sqrt{d}$ is the score achievable by a perfect negative-scalar identity; the denominator penalizes deviation from this ideal via the dimensionless ratio $\|E\|_F^2/(\varepsilon^2 d)$. For typical trained blocks, $\|E\|_F/(\varepsilon\sqrt{d}) \in [1.9, 3.8]$, yielding $s(i,i) \in [1.76, 3.23]$---well above the baseline but below the theoretical maximum of $\sqrt{d}$.

Removed: typical ranges detail.

---

## Corollary 2 (Null Model) - Full Treatment

Original text:
> \begin{corollary}[Null Model]\label{cor:null-model}
> For randomly initialized weights with no training, all pairs satisfy $\mathbb{E}[s(i,j)] = 1/\sqrt{d} + o(1)$ uniformly in $(i,j)$. Hungarian matching recovers correct pairs at the chance rate $1/L$.
> \end{corollary}
>
> The null model rules out three alternative explanations: (a) diagonal dominance is not an artifact of compatible matrix shapes; (b) the residual connection itself does not create the signal; (c) any nontrivial loss level does not suffice. The diagonal-dominance fingerprint is a \emph{learned} property induced by gradient descent. The theory explains why the observed structure is identifiable once induced; controlled experiments---including the initialization ablation showing orthogonal init at 0\% pairing and the PlainNet control at 3\%---confirm that training induces it

Removed: Corollary 2 and extended discussion (key insight retained in main text).

---

## Relational Structure Paragraph

Original text:
> Crucially, diagonal dominance captures \emph{relational} structure between paired matrices---how they jointly encode the near-identity mapping---rather than aggregate properties of individual matrices. This relational fingerprint is preserved under fine-tuning and noise because it reflects the functional constraint that the residual block must approximate.

Removed: implicit in the mechanism description.
