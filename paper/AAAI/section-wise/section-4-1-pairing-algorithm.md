# Section 4.1: Pairing Algorithm

## Problem Setup

Given a reference model $R$ with $L$ residual blocks and a suspect model $S$ with $L$ residual blocks, we seek to determine which block in $S$ corresponds to which block in $R$. Let $\{(W_{\text{in}}^{(i)}, W_{\text{out}}^{(i)})\}_{i=1}^L$ denote the input and output projections of $R$'s residual branches, and $\{(\tilde{W}_{\text{in}}^{(j)}, \tilde{W}_{\text{out}}^{(j)})\}_{j=1}^L$ denote those of $S$. If $S$ descends from $R$---through fine-tuning, quantization, or other post-training modifications---then a permutation $\pi: [L] \to [L]$ should exist such that block $j$ in $S$ corresponds to block $\pi(j)$ in $R$.

For models sharing identical architectures (the common case when $S$ is derived from $R$), we expect $\pi$ to be the identity. However, block reordering can occur through model surgery, layer-wise fine-tuning, or deliberate obfuscation. The pairing algorithm must recover the correct correspondence from weights alone, without metadata or architectural assumptions.

## Score Matrix Construction

We construct a score matrix $\mathbf{S} \in \mathbb{R}^{L \times L}$ where each entry $s(i,j)$ quantifies the diagonal-dominance evidence that block $i$ in the reference corresponds to block $j$ in the suspect. Following the fingerprint definition from Section 3, we compute:
$$
s(i,j) = \frac{|\mathrm{tr}(M_{ij})|}{\|M_{ij}\|_F}, \quad \text{where } M_{ij} = \tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}.
$$

The cross-model product $M_{ij}$ tests whether the output projection from suspect block $j$ and the input projection from reference block $i$ exhibit the trained coupling characteristic of a true residual branch. If $S$ descends from $R$ and block $j$ corresponds to block $i$ (after any modifications), then $M_{ij}$ inherits the diagonal-dominant structure of the original branch product, yielding $s(i,j) = \Theta(\sqrt{d})$. For non-corresponding pairs, the matrices lack this coupling, and $s(i,j)$ remains at the random baseline $\Theta(1/\sqrt{d})$.

**Remark.** We use the cross-model product $\tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}$ rather than comparing branch products directly (i.e., $\tilde{W}_{\text{out}}^{(j)} \tilde{W}_{\text{in}}^{(j)}$ vs. $W_{\text{out}}^{(i)} W_{\text{in}}^{(i)}$) because the diagonal-dominance property is a *joint* property of the paired matrices. The cross-product explicitly tests whether suspect block $j$'s output projection "completes" reference block $i$'s input projection in the same way it did in the original trained model.

## Assignment Algorithm

Given the score matrix $\mathbf{S}$, we seek a one-to-one assignment $\pi: [L] \to [L]$ that maximizes total evidence for correct correspondences. This is the classical optimal assignment problem, solvable via the Hungarian algorithm \citep{kuhn1955hungarian}:
$$
\pi^* = \arg\max_{\pi \in \mathcal{P}_L} \sum_{i=1}^{L} s(i, \pi(i)),
$$
where $\mathcal{P}_L$ denotes the set of all permutations on $[L]$.

For derived models with preserved block ordering, the optimal assignment recovers the identity permutation: $\pi^*(i) = i$ for all $i \in [L]$. Deviations from identity indicate either block reordering in the suspect model or, if many blocks fail to match, evidence against the lineage hypothesis.

**Alternative: Greedy Matching.** When $L$ is large or real-time verification is required, a greedy approximation offers a practical alternative. Starting from an empty assignment, iteratively select the pair $(i,j)$ with maximum score among unassigned indices:
1. Initialize $\mathcal{U}_R = \mathcal{U}_S = [L]$
2. While $\mathcal{U}_R \neq \emptyset$:
   - $(i^*, j^*) = \arg\max_{i \in \mathcal{U}_R, j \in \mathcal{U}_S} s(i,j)$
   - Set $\pi(i^*) = j^*$
   - Remove $i^*$ from $\mathcal{U}_R$ and $j^*$ from $\mathcal{U}_S$

Greedy matching achieves $O(L^2)$ comparisons but may produce suboptimal assignments when score differences are small. In our experiments, both methods achieve identical pair accuracy because the $\Theta(\sqrt{d})$ vs. $\Theta(1/\sqrt{d})$ separation creates unambiguous diagonal dominance in the score matrix.

## Complete Algorithm

Algorithm 1 presents the complete pairing procedure.

---

**Algorithm 1: Residual Block Pairing via Diagonal Dominance**

---

**Input:** Reference model $R$ with blocks $\{(W_{\text{in}}^{(i)}, W_{\text{out}}^{(i)})\}_{i=1}^L$, suspect model $S$ with blocks $\{(\tilde{W}_{\text{in}}^{(j)}, \tilde{W}_{\text{out}}^{(j)})\}_{j=1}^L$

**Output:** Assignment $\pi: [L] \to [L]$, pair accuracy $\alpha$, aggregate lineage score $\mathcal{L}$

---

1. **Construct score matrix:**
   - **for** $i = 1$ to $L$ **do**
     - **for** $j = 1$ to $L$ **do**
       - $M_{ij} \leftarrow \tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}$ $\quad$ // Cross-model branch product
       - $s(i,j) \leftarrow |\mathrm{tr}(M_{ij})| / \|M_{ij}\|_F$

2. **Solve optimal assignment:**
   - $\pi^* \leftarrow \text{Hungarian}(-\mathbf{S})$ $\quad$ // Maximize via min-cost assignment on negated matrix

3. **Compute pair accuracy:**
   - $\alpha \leftarrow \frac{1}{L} \sum_{i=1}^{L} \mathbf{1}[\pi^*(i) = i]$

4. **Compute aggregate lineage score:**
   - $\mathcal{L} \leftarrow \frac{1}{L} \sum_{i=1}^{L} s(i, \pi^*(i))$

5. **return** $\pi^*, \alpha, \mathcal{L}$

---

## Complexity Analysis

**Score matrix computation.** For each of the $L^2$ entries, we compute the matrix product $M_{ij} = \tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}$, its trace, and its Frobenius norm. If $W_{\text{in}}^{(i)} \in \mathbb{R}^{h \times d}$ and $W_{\text{out}}^{(j)} \in \mathbb{R}^{d \times h}$, the product $M_{ij} \in \mathbb{R}^{d \times d}$ requires $O(d^2 h)$ operations. The trace and Frobenius norm each require $O(d^2)$ operations. Total cost: $O(L^2 d^2 h)$.

For standard transformer MLP blocks with expansion ratio 4 (i.e., $h = 4d$), this simplifies to $O(L^2 d^3)$. For GPT-2-xl with $L = 48$ and $d = 1600$, this amounts to approximately $10^{13}$ floating-point operations---substantial but tractable on modern hardware (roughly 10 seconds on a single GPU).

**Optimal assignment.** The Hungarian algorithm runs in $O(L^3)$ time \citep{kuhn1955hungarian}. For typical model depths ($L \leq 100$), this is negligible compared to score matrix computation.

**Total complexity:** $O(L^2 d^2 h + L^3) = O(L^2 d^2 h)$ for $d \gg L$.

**Memory.** The algorithm requires storing the $L \times L$ score matrix ($O(L^2)$ floats) and, temporarily, one $d \times d$ product matrix at a time ($O(d^2)$ floats). Weight matrices need not be held simultaneously; they can be loaded block-by-block from disk. Peak memory: $O(L^2 + d^2 + dh)$ floats.

## Interpretation of Outputs

The algorithm produces three outputs with distinct interpretive roles:

1. **Assignment $\pi^*$**: The recovered block correspondence. For derived models, $\pi^* = \text{id}$ confirms preserved architecture. Non-identity assignments indicate block reordering or partial derivation.

2. **Pair accuracy $\alpha$**: The fraction of blocks correctly matched to their expected positions. For models with identical architecture and the identity ground-truth permutation:
   - $\alpha = 1.0$: Perfect block correspondence, strong lineage evidence
   - $\alpha \approx 1/L$: Chance-level matching, no lineage signal (consistent with independent training or random initialization)

3. **Aggregate lineage score $\mathcal{L}$**: The mean diagonal-dominance score along the optimal assignment. This continuous measure quantifies fingerprint strength:
   - $\mathcal{L} = \Theta(\sqrt{d})$: Strong trained coupling, consistent with shared lineage
   - $\mathcal{L} = \Theta(1/\sqrt{d})$: Baseline noise, no evidence of derivation

The combination of $\alpha$ and $\mathcal{L}$ enables nuanced verification decisions. High $\alpha$ with low $\mathcal{L}$ (theoretically possible if scores barely exceed baseline) would warrant caution, while high $\mathcal{L}$ confirms strong fingerprint evidence.

## Connection to Related Work

The pairing algorithm draws on optimal assignment methods that have appeared in neural network analysis, though for different purposes. \citet{ainsworth2023git} use permutation-based alignment to identify weight-space symmetries for model merging, finding permutations that minimize interpolation loss barriers. \citet{singh2020model} apply optimal transport for layer-wise neuron alignment in federated model fusion. Our approach differs in objective: rather than aligning neurons within a layer to enable averaging, we match entire residual blocks across models to establish lineage evidence. The diagonal-dominance score provides a principled similarity metric grounded in the trained structure of residual branches, whereas prior alignment methods optimize for functional equivalence or interpolation smoothness.

## Empirical Validation

On GPT-2 ($L = 12$, $d = 768$), the pairing algorithm achieves:
- **Trained model:** Pair accuracy $\alpha = 100\%$, mean diagonal score $\bar{s}(i,i) = 4.18$, mean off-diagonal score $\bar{s}(i,j) = 0.12$ for $i \neq j$
- **Random initialization:** Pair accuracy $\alpha = 6.25\%$ (chance = $1/12 \approx 8.3\%$), scores uniformly at baseline

The score matrix exhibits clear diagonal structure after training (Figure~\ref{fig:heatmaps}), with correct pairs achieving scores 35$\times$ higher than incorrect pairs. Hungarian matching on this matrix is unambiguous: the diagonal entries dominate their respective rows and columns, leaving no room for assignment errors.

Across the full GPT-2 scaling series (124M to 1.5B parameters), pair accuracy remains at 100\% while the mean correct-pair score increases from 4.18 to 7.77, tracking the theoretical $\sqrt{d}$ dependence (Table~\ref{tab:gpt2_scaling}).

---

## References for Section 4.1

- \citet{kuhn1955hungarian}: Kuhn, H.W. "The Hungarian Method for the Assignment Problem." *Naval Research Logistics Quarterly*, 1955.
- \citet{ainsworth2023git}: Ainsworth, S.K., Hayase, J., and Srinivasa, S. "Git Re-Basin: Merging Models modulo Permutation Symmetries." *ICLR*, 2023.
- \citet{singh2020model}: Singh, S.P. and Jaggi, M. "Model Fusion via Optimal Transport." *NeurIPS*, 2020.
