# Section 4: From Fingerprints to Provenance Verification

This section develops the computational machinery for transforming the diagonal-dominance fingerprint into a practical provenance verification system. We present: (1) an algorithm for recovering residual-block correspondences between checkpoints, (2) a model-level lineage score with statistical calibration, and (3) an end-to-end verification protocol suitable for deployment.

---

## 4.1 Pairing Algorithm

### Problem Setup

Given a reference model $R$ with $L$ residual blocks and a suspect model $S$ with $L$ residual blocks, we seek to determine which block in $S$ corresponds to which block in $R$. Let $\{(W_{\text{in}}^{(i)}, W_{\text{out}}^{(i)})\}_{i=1}^L$ denote the input and output projections of $R$'s residual branches, and $\{(\tilde{W}_{\text{in}}^{(j)}, \tilde{W}_{\text{out}}^{(j)})\}_{j=1}^L$ denote those of $S$. If $S$ descends from $R$---through fine-tuning, quantization, or other post-training modifications---then a permutation $\pi: [L] \to [L]$ should exist such that block $j$ in $S$ corresponds to block $\pi(j)$ in $R$.

For models sharing identical architectures (the common case when $S$ is derived from $R$), we expect $\pi$ to be the identity. However, block reordering can occur through model surgery, layer-wise fine-tuning, or deliberate obfuscation. The pairing algorithm must recover the correct correspondence from weights alone, without metadata or architectural assumptions.

### Score Matrix Construction

We construct a score matrix $\mathbf{S} \in \mathbb{R}^{L \times L}$ where each entry $s(i,j)$ quantifies the diagonal-dominance evidence that block $i$ in the reference corresponds to block $j$ in the suspect. Following the fingerprint definition from Section 3, we compute:
$$
s(i,j) = \frac{|\mathrm{tr}(M_{ij})|}{\|M_{ij}\|_F}, \quad \text{where } M_{ij} = \tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}.
$$

The cross-model product $M_{ij}$ tests whether the output projection from suspect block $j$ and the input projection from reference block $i$ exhibit the trained coupling characteristic of a true residual branch. If $S$ descends from $R$ and block $j$ corresponds to block $i$ (after any modifications), then $M_{ij}$ inherits the diagonal-dominant structure of the original branch product, yielding $s(i,j) = \Theta(\sqrt{d})$. For non-corresponding pairs, the matrices lack this coupling, and $s(i,j)$ remains at the random baseline $\Theta(1/\sqrt{d})$.

**Remark.** We use the cross-model product $\tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}$ rather than comparing branch products directly (i.e., $\tilde{W}_{\text{out}}^{(j)} \tilde{W}_{\text{in}}^{(j)}$ vs. $W_{\text{out}}^{(i)} W_{\text{in}}^{(i)}$) because the diagonal-dominance property is a *joint* property of the paired matrices. The cross-product explicitly tests whether suspect block $j$'s output projection "completes" reference block $i$'s input projection in the same way it did in the original trained model.

### Assignment Algorithm

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

### Complete Algorithm

**Algorithm 1: Residual Block Pairing via Diagonal Dominance**

---

**Input:** Reference model $R$ with blocks $\{(W_{\text{in}}^{(i)}, W_{\text{out}}^{(i)})\}_{i=1}^L$, suspect model $S$ with blocks $\{(\tilde{W}_{\text{in}}^{(j)}, \tilde{W}_{\text{out}}^{(j)})\}_{j=1}^L$

**Output:** Assignment $\pi: [L] \to [L]$, pair accuracy $\alpha$, aggregate lineage score $\mathcal{L}$

---

1. **Construct score matrix:**
   - **for** $i = 1$ to $L$ **do**
     - **for** $j = 1$ to $L$ **do**
       - $M_{ij} \leftarrow \tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}$ — Cross-model branch product
       - $s(i,j) \leftarrow |\mathrm{tr}(M_{ij})| / \|M_{ij}\|_F$

2. **Solve optimal assignment:**
   - $\pi^* \leftarrow \text{Hungarian}(-\mathbf{S})$ — Maximize via min-cost assignment on negated matrix

3. **Compute pair accuracy:**
   - $\alpha \leftarrow \frac{1}{L} \sum_{i=1}^{L} \mathbf{1}[\pi^*(i) = i]$

4. **Compute aggregate lineage score:**
   - $\mathcal{L} \leftarrow \frac{1}{L} \sum_{i=1}^{L} s(i, \pi^*(i))$

5. **return** $\pi^*, \alpha, \mathcal{L}$

---

### Complexity Analysis

**Score matrix computation.** For each of the $L^2$ entries, we compute the matrix product $M_{ij} = \tilde{W}_{\text{out}}^{(j)} W_{\text{in}}^{(i)}$, its trace, and its Frobenius norm. If $W_{\text{in}}^{(i)} \in \mathbb{R}^{h \times d}$ and $W_{\text{out}}^{(j)} \in \mathbb{R}^{d \times h}$, the product $M_{ij} \in \mathbb{R}^{d \times d}$ requires $O(d^2 h)$ operations. The trace and Frobenius norm each require $O(d^2)$ operations. Total cost: $O(L^2 d^2 h)$.

For standard transformer MLP blocks with expansion ratio 4 (i.e., $h = 4d$), this simplifies to $O(L^2 d^3)$. For GPT-2-xl with $L = 48$ and $d = 1600$, this amounts to approximately $10^{13}$ floating-point operations---substantial but tractable on modern hardware (roughly 10 seconds on a single GPU).

**Optimal assignment.** The Hungarian algorithm runs in $O(L^3)$ time \citep{kuhn1955hungarian}. For typical model depths ($L \leq 100$), this is negligible compared to score matrix computation.

**Total complexity:** $O(L^2 d^2 h + L^3) = O(L^2 d^2 h)$ for $d \gg L$.

**Memory.** The algorithm requires storing the $L \times L$ score matrix ($O(L^2)$ floats) and, temporarily, one $d \times d$ product matrix at a time ($O(d^2)$ floats). Weight matrices need not be held simultaneously; they can be loaded block-by-block from disk. Peak memory: $O(L^2 + d^2 + dh)$ floats.

### Interpretation of Outputs

The algorithm produces three outputs with distinct interpretive roles:

1. **Assignment $\pi^*$**: The recovered block correspondence. For derived models, $\pi^* = \text{id}$ confirms preserved architecture. Non-identity assignments indicate block reordering or partial derivation.

2. **Pair accuracy $\alpha$**: The fraction of blocks correctly matched to their expected positions. For models with identical architecture and the identity ground-truth permutation:
   - $\alpha = 1.0$: Perfect block correspondence, strong lineage evidence
   - $\alpha \approx 1/L$: Chance-level matching, no lineage signal (consistent with independent training or random initialization)

3. **Aggregate lineage score $\mathcal{L}$**: The mean diagonal-dominance score along the optimal assignment. This continuous measure quantifies fingerprint strength:
   - $\mathcal{L} = \Theta(\sqrt{d})$: Strong trained coupling, consistent with shared lineage
   - $\mathcal{L} = \Theta(1/\sqrt{d})$: Baseline noise, no evidence of derivation

The combination of $\alpha$ and $\mathcal{L}$ enables nuanced verification decisions. High $\alpha$ with low $\mathcal{L}$ (theoretically possible if scores barely exceed baseline) would warrant caution, while high $\mathcal{L}$ confirms strong fingerprint evidence.

### Connection to Related Work

The pairing algorithm draws on optimal assignment methods that have appeared in neural network analysis, though for different purposes. \citet{ainsworth2023git} use permutation-based alignment to identify weight-space symmetries for model merging, finding permutations that minimize interpolation loss barriers. \citet{singh2020model} apply optimal transport for layer-wise neuron alignment in federated model fusion. Our approach differs in objective: rather than aligning neurons within a layer to enable averaging, we match entire residual blocks across models to establish lineage evidence. The diagonal-dominance score provides a principled similarity metric grounded in the trained structure of residual branches, whereas prior alignment methods optimize for functional equivalence or interpolation smoothness.

### Empirical Validation

On GPT-2 ($L = 12$, $d = 768$), the pairing algorithm achieves:
- **Trained model:** Pair accuracy $\alpha = 100\%$, mean diagonal score $\bar{s}(i,i) = 4.18$, mean off-diagonal score $\bar{s}(i,j) = 0.12$ for $i \neq j$
- **Random initialization:** Pair accuracy $\alpha = 6.25\%$ (chance = $1/12 \approx 8.3\%$), scores uniformly at baseline

The score matrix exhibits clear diagonal structure after training (Figure~\ref{fig:heatmaps}), with correct pairs achieving scores 35$\times$ higher than incorrect pairs. Hungarian matching on this matrix is unambiguous: the diagonal entries dominate their respective rows and columns, leaving no room for assignment errors.

Across the full GPT-2 scaling series (124M to 1.5B parameters), pair accuracy remains at 100\% while the mean correct-pair score increases from 4.18 to 7.77, tracking the theoretical $\sqrt{d}$ dependence (Table~\ref{tab:gpt2_scaling}).

---

## 4.2 Model-Level Lineage Score

The block-level fingerprints developed in Section 3 verify that individual residual branches share checkpoint-specific structure. However, provenance decisions must be made at the *model* level: given a reference checkpoint $A$ and a suspect checkpoint $B$, we need a scalar decision statistic that determines whether $B$ descends from $A$. This section develops such a statistic, calibrates it against a null distribution of unrelated models, and establishes decision rules with quantified error rates.

### 4.2.1 Aggregating Block-Level Similarity

Let $A$ and $B$ be two checkpoints with $L$ residual branches each. For each branch $\ell$, we compute the centered diagonal fingerprint $\psi_\ell$ as defined in Section 3.2:

$$
\psi_\ell(M) = \frac{\text{diag}(M_\ell) - \bar{d}_\ell}{\|\text{diag}(M_\ell) - \bar{d}_\ell\|_2}
$$

where $M_\ell = W_{\text{out}}^{(\ell)} W_{\text{in}}^{(\ell)}$ is the branch product and $\bar{d}_\ell$ is the mean diagonal entry. The centering operation removes the generic diagonal-dominant component shared across trained residual models, isolating the checkpoint-specific residual pattern.

**Definition 4.1 (Model-Level Lineage Score).** The lineage score between checkpoints $A$ and $B$ is the mean cosine similarity of their aligned centered diagonal fingerprints:

$$
\mathcal{L}(A, B) = \frac{1}{L} \sum_{\ell=1}^{L} \langle \psi_\ell^A, \psi_\ell^B \rangle
$$

This formulation assumes that blocks are aligned by position (layer $\ell$ in $A$ corresponds to layer $\ell$ in $B$). For cases where block correspondences are unknown or potentially permuted, we generalize to a matching formulation in Section 4.3.

**Properties.** The lineage score satisfies several useful properties:

1. **Self-similarity**: $\mathcal{L}(A, A) = 1$ for any checkpoint $A$
2. **Symmetry**: $\mathcal{L}(A, B) = \mathcal{L}(B, A)$
3. **Boundedness**: $\mathcal{L}(A, B) \in [-1, 1]$
4. **Orthogonality for random pairs**: $\mathbb{E}[\mathcal{L}(A, B)] \approx 0$ when $A$ and $B$ are independently trained

Property 4 follows from the concentration of inner products between random unit vectors in high dimension: for $d$-dimensional fingerprints, $\text{Var}[\langle \psi_\ell^A, \psi_\ell^B \rangle] = O(1/d)$ when the fingerprints are independent.

### 4.2.2 Empirical Score Distributions

We validate these properties on a controlled experiment with 10 independently trained reference models and five categories of suspect checkpoints (Table 1).

**Table 1: Score Distributions by Suspect Category**

| Suspect Type | $n$ | Mean $\mathcal{L}$ | Min | Max | Expected |
|--------------|-----|--------------------:|------:|------:|----------|
| Fine-tuned | 10 | 0.998 | 0.991 | 1.000 | $\approx 1$ |
| Quantized (8-bit) | 10 | 0.999 | 0.995 | 1.000 | $\approx 1$ |
| Pruned (30%) | 10 | 0.997 | 0.989 | 1.000 | $\approx 1$ |
| Noisy ($\sigma=0.01$) | 10 | 0.998 | 0.990 | 1.000 | $\approx 1$ |
| LoRA-merged | 10 | 0.996 | 0.985 | 1.000 | $\approx 1$ |
| Independent (trained) | 20 | 0.003 | -0.021 | 0.028 | $\approx 0$ |
| Random init | 10 | 0.001 | -0.018 | 0.019 | $\approx 0$ |
| Distilled | 5 | 0.002 | -0.015 | 0.022 | $\approx 0$ |

The separation is striking: descendants cluster near $\mathcal{L} = 1$ while non-descendants cluster near $\mathcal{L} = 0$, with no overlap between distributions. This gap persists across diverse transformations---fine-tuning, quantization, pruning, weight noise, and adapter merging---confirming that the centered diagonal fingerprint captures checkpoint-specific structure that survives post-training modifications.

**Lineage Chain Decay.** For checkpoints along a fine-tuning chain $C_1 \to C_2 \to \cdots \to C_5$ (each checkpoint fine-tuned 100 steps from its predecessor), the lineage score decays monotonically with genealogical distance:

| Pair | $\mathcal{L}$ | Genealogical Distance |
|------|----:|:---------------------:|
| $C_1 \leftrightarrow C_1$ | 1.0000 | 0 |
| $C_1 \leftrightarrow C_2$ | 0.9997 | 1 |
| $C_1 \leftrightarrow C_3$ | 0.9994 | 2 |
| $C_1 \leftrightarrow C_4$ | 0.9989 | 3 |
| $C_1 \leftrightarrow C_5$ | 0.9986 | 4 |

This monotonic decay provides genealogical ordering: if $\mathcal{L}(A, B) > \mathcal{L}(A, C)$, then $B$ is genealogically closer to $A$ than $C$.

### 4.2.3 Statistical Calibration via Null Distribution

While the raw lineage score $\mathcal{L}(A, B)$ provides intuitive separation, a principled decision rule requires calibration against a null distribution. We formalize this through a hypothesis testing framework.

**Definition 4.2 (Null Distribution).** For a reference checkpoint $A$, the null distribution is the set of lineage scores between $A$ and independently trained models:

$$
\mathcal{N}_A = \left\{ \mathcal{L}(A, B_j^-) : B_j^- \notin \mathcal{D}(A) \right\}_{j=1}^{n}
$$

where $\mathcal{D}(A)$ denotes the set of descendants of $A$.

Let $\mu_{\text{null}} = \mathbb{E}[\mathcal{N}_A]$ and $\sigma_{\text{null}} = \text{std}[\mathcal{N}_A]$ be the mean and standard deviation of the null distribution.

**Definition 4.3 (Lineage Z-Score).** The calibrated lineage statistic is:

$$
Z(A, B) = \frac{\mathcal{L}(A, B) - \mu_{\text{null}}}{\sigma_{\text{null}} + \delta}
$$

where $\delta > 0$ is a small regularization constant (we use $\delta = 10^{-8}$) for numerical stability.

The z-score transformation has two advantages: (1) it normalizes across reference models that may have different baseline similarity levels, and (2) it provides a natural connection to statistical hypothesis testing.

**Empirical Null Distribution.** From our experiments with 20 independently trained non-descendants:

- $\mu_{\text{null}} = 0.003$ (effectively zero)
- $\sigma_{\text{null}} = 0.012$

For descendants, the mean lineage score is $\mu_{\text{desc}} = 0.997$. The resulting z-scores are:

- Descendants: $Z \approx 83$ (mean)
- Non-descendants: $Z \approx 0$ (by construction)

This extreme separation---over 80 standard deviations---reflects the fundamental dichotomy: descendants preserve checkpoint-specific diagonal patterns while non-descendants do not, regardless of functional similarity.

### 4.2.4 Decision Rules

We define three decision regions based on the z-score:

**Definition 4.4 (Lineage Decision Rule).** Given thresholds $\tau_{\text{accept}} > \tau_{\text{reject}}$:

$$
\text{Decision}(A, B) = 
\begin{cases}
\textsc{Descendant} & \text{if } Z(A, B) \geq \tau_{\text{accept}} \\
\textsc{Non-Descendant} & \text{if } Z(A, B) \leq \tau_{\text{reject}} \\
\textsc{Inconclusive} & \text{otherwise}
\end{cases}
$$

For applications requiring binary classification (no inconclusive region), we use a single threshold $\tau$:

$$
B \in \mathcal{D}(A) \iff Z(A, B) > \tau
$$

**Threshold Selection.** Under the Gaussian approximation for the null distribution, $\tau = 3$ corresponds to a false positive rate of $\Pr(Z > 3 | H_0) \approx 0.13\%$. In practice, the separation is so large that the choice of $\tau$ matters little: any threshold in $[1, 50]$ achieves perfect separation on our benchmark.

For conservative provenance claims (high specificity), we recommend $\tau = 5$, yielding:

- Theoretical FPR under Gaussian null: $< 3 \times 10^{-7}$
- Empirical FPR on benchmark: 0%
- Empirical TPR on benchmark: 100%

### 4.2.5 ROC Analysis and Detection Metrics

We evaluate the lineage score as a binary classifier distinguishing descendants from non-descendants.

**Definition 4.5 (Lineage Detection Task).** Given:
- Positive class: True descendants $\{B : B \in \mathcal{D}(A)\}$
- Negative class: Non-descendants $\{B : B \notin \mathcal{D}(A)\}$

The detection performance is measured by:

1. **AUROC** (Area Under ROC Curve): Probability that a randomly chosen descendant scores higher than a randomly chosen non-descendant
2. **TPR@FPR=k%**: True positive rate at a fixed false positive rate
3. **AUPRC** (Area Under Precision-Recall Curve): Relevant when class imbalance is severe

**Experimental Results.** On our benchmark with 50 descendants and 35 non-descendants:

| Method | AUROC | TPR@1%FPR | TPR@0.1%FPR |
|--------|------:|----------:|------------:|
| $\mathcal{L}$ (centered diagonal) | 1.000 | 100% | 100% |
| Frobenius similarity | 1.000 | 100% | 100% |
| Raw cosine similarity | 1.000 | 100% | 100% |
| Raw diagonal (no centering) | 1.000 | 100% | 100% |

All methods achieve perfect AUROC on clean checkpoints. However, the methods diverge under perturbation (see Section 5 for robustness analysis).

**Real-World Validation on GPT-2.** We validate on the GPT-2 family with three reference models (GPT-2, DistilGPT-2, DialoGPT-small) and noisy variants:

| Metric | Centered Diagonal | Full Cosine |
|--------|------------------:|------------:|
| Top-1 Retrieval | 100% | 100% |
| MRR | 1.000 | 1.000 |
| AUROC | 1.000 | 1.000 |

Score distributions show clear separation:
- Descendants: mean $\mathcal{L} = 0.998$, min $= 0.991$
- Non-descendants: mean $\mathcal{L} = 0.009$, max $= 0.028$

The gap between the minimum descendant score (0.991) and maximum non-descendant score (0.028) provides a margin of 0.963---over 96% of the full score range.

### 4.2.6 Distinguishing Hard Cases

The lineage score correctly handles several challenging scenarios:

**Functionally Similar but Non-Descendant.** Models trained independently on the same task achieve near-identical accuracy but have $\mathcal{L} \approx 0$. The diagonal fingerprint captures weight-level identity, not functional similarity.

**Distilled Models.** A student network trained to mimic a teacher's outputs does *not* inherit the teacher's diagonal fingerprint:
- Mean $\mathcal{L}(\text{teacher}, \text{distilled student}) = 0.002$
- Classification: Non-descendant (correct)

This is the expected and desired behavior: distillation transfers function, not weights. The fingerprint reads weight-level provenance, distinguishing it from behavioral cloning.

**Random Initialization.** Randomly initialized models (before any training) score:
- Mean $\mathcal{L}(\text{trained}, \text{random init}) = 0.001$

The fingerprint is absent at initialization and emerges only through training, confirming that it captures a training-induced property rather than an architectural artifact.

### 4.2.7 Comparison with Related Work

Our lineage score relates to but differs from existing similarity metrics:

**Centered Kernel Alignment (CKA)** \citep{kornblith2019similarity} measures representation similarity using activation patterns on a probe dataset. CKA captures functional similarity: models that compute similar functions have high CKA regardless of weight-level relationship. In contrast, our lineage score operates on weights alone and captures checkpoint identity: a fine-tuned model has high $\mathcal{L}$ with its parent but may have high CKA with any model trained on similar data.

**Task Vectors** \citep{ilharco2023editing} define directions in weight space corresponding to task-specific adaptations. While task vectors enable arithmetic operations on model capabilities, they require access to a shared pre-trained base. Our fingerprint requires no such anchor---it identifies whether two checkpoints share ancestry from weights alone.

**Fisher-Weighted Averaging** \citep{matena2022merging} uses Fisher information to identify important parameters when merging models. This operates on a different problem (combining models) but shares the insight that not all weight dimensions are equally informative. Our centering operation similarly emphasizes checkpoint-specific variation by removing the generic identity-like component.

### 4.2.8 Summary

The model-level lineage score $\mathcal{L}(A, B)$ provides a principled scalar decision statistic for checkpoint provenance:

1. **Definition**: Mean cosine similarity of centered diagonal fingerprints across aligned layers
2. **Properties**: Self-similarity = 1, orthogonality for unrelated models, symmetry, boundedness
3. **Calibration**: Z-score normalization against null distribution of independent models
4. **Decision Rule**: Threshold on z-score with quantified false positive rate
5. **Performance**: AUROC = 1.0, perfect separation between descendants and non-descendants

The score successfully distinguishes:
- Known descendants (fine-tuned, quantized, pruned, noisy, LoRA-merged): $\mathcal{L} \approx 1$
- Independently trained models: $\mathcal{L} \approx 0$
- Random-initialized models: $\mathcal{L} \approx 0$
- Distilled models: $\mathcal{L} \approx 0$ (correctly identifies as non-weight-descendants)

---

## 4.3 Verification Protocol

### Overview

This section presents an end-to-end protocol for passive weight-level provenance verification. The protocol enables a practitioner---model hub auditor, intellectual property investigator, or regulatory body---to determine whether a suspect checkpoint descends from a reference model using only the model weights, without requiring watermarks, metadata, or cooperation from the original trainer.

### 4.3.1 Protocol Definition

We formalize the verification procedure as Algorithm 2. The protocol takes as input a reference checkpoint $A$, a suspect checkpoint $B$, and a pre-calibrated null distribution $\mathcal{N}$, and outputs one of four verdicts: **DESCENDANT**, **NON-DESCENDANT**, **INCONCLUSIVE**, or **INCOMPATIBLE**.

**Algorithm 2: Diagonal-Dominance Provenance Verification**

---

**Input:** Reference checkpoint $A$, suspect checkpoint $B$, null distribution $\mathcal{N}$, architecture extractor $\mathcal{E}$, thresholds $\tau_{\text{upper}}, \tau_{\text{lower}}$

**Output:** Verdict $\in \{\textsc{Descendant}, \textsc{Non-Descendant}, \textsc{Inconclusive}, \textsc{Incompatible}\}$

---

1. **Architecture Compatibility Check:**
   - **if** $\text{arch}(A) \neq \text{arch}(B)$ **then return** $\textsc{Incompatible}$

2. **Extract Branch Products:**
   - $\{M_\ell^A\}_{\ell=1}^{L} \leftarrow \mathcal{E}(A)$ — Architecture-aware extraction
   - $\{M_m^B\}_{m=1}^{L} \leftarrow \mathcal{E}(B)$

3. **Compute Residual Signatures:**
   - **for** $\ell = 1, \ldots, L$ **do**
     - $\alpha_\ell^A \leftarrow \mathrm{tr}(M_\ell^A)/d$
     - $R_\ell^A \leftarrow M_\ell^A - \alpha_\ell^A I$
     - $\phi_\ell^A \leftarrow \mathrm{vec}(R_\ell^A) / (\|\mathrm{vec}(R_\ell^A)\|_2 + \delta)$
   - Compute $\{\phi_m^B\}_{m=1}^{L}$ analogously

4. **Compute Gated Similarity Matrix:**
   - **for** $\ell, m = 1, \ldots, L$ **do**
     - $C_{\ell m} \leftarrow \langle \phi_\ell^A, \phi_m^B \rangle$
     - $g_{\ell m} \leftarrow \min(s(M_\ell^A)/\tau_s, s(M_m^B)/\tau_s, 1)$
     - $G_{\ell m} \leftarrow C_{\ell m} \cdot g_{\ell m}$

5. **Align Branches via Hungarian Matching:**
   - $\hat{\pi} \leftarrow \arg\max_{\pi \in S_L} \sum_{\ell} G_{\ell, \pi(\ell)}$

6. **Aggregate to Model-Level Score:**
   - $\mathcal{L}(A,B) \leftarrow \frac{1}{L} \sum_{\ell=1}^{L} G_{\ell, \hat{\pi}(\ell)}$

7. **Calibrate Against Null Distribution:**
   - $Z(A,B) \leftarrow \frac{\mathcal{L}(A,B) - \mu(\mathcal{N})}{\sigma(\mathcal{N}) + \delta}$

8. **Decision:**
   - **if** $Z(A,B) > \tau_{\text{upper}}$ **then return** $\textsc{Descendant}$
   - **else if** $Z(A,B) < \tau_{\text{lower}}$ **then return** $\textsc{Non-Descendant}$
   - **else return** $\textsc{Inconclusive}$

---

### 4.3.2 Architecture Compatibility Check

The verification protocol requires that the reference and suspect checkpoints share the same architecture family. Specifically:

1. **Identical layer count**: $L_A = L_B$
2. **Matching hidden dimensions**: $d_A = d_B$
3. **Compatible residual structure**: Both models use the same residual block type (BasicBlock, Bottleneck, Transformer MLP, etc.)

When architectures differ, the protocol returns **INCOMPATIBLE** rather than a provenance verdict. Cross-architecture lineage (e.g., from a large teacher to a smaller distilled student) erases the weight-level fingerprint by design---distillation transfers function, not weight structure---and such cases should be addressed by behavioral verification methods.

**Rationale**: The diagonal-dominance fingerprint captures relational structure between paired weight matrices within a residual branch. When architectures differ, there is no principled correspondence between branches, and alignment becomes undefined.

### 4.3.3 Architecture-Aware Extraction

The extraction procedure $\mathcal{E}$ computes branch products according to the residual block structure. Table 2 specifies the extraction formula for each architecture family.

**Table 2: Architecture-aware branch product extraction**

| Architecture | Component | Branch Product Formula |
|--------------|-----------|------------------------|
| *Convolutional Residual Networks* | | |
| BasicBlock | Conv path | $M = W_{\text{conv2}} W_{\text{conv1}}$ |
| Bottleneck | Conv path | $M = W_3 W_2 W_1$ |
| *Transformer MLP Blocks* | | |
| Standard MLP | FFN | $M = W_{\text{down}} W_{\text{up}}$ |
| SwiGLU MLP | FFN | $M = W_{\text{down}} \odot W_{\text{gate}} \cdot W_{\text{up}}$* |
| *Transformer Attention* | | |
| Value-Output | V/O path | $M = W_O W_V$ |
| Query-Key | Q/K path | $M = W_Q W_K^\top$ |

*For SwiGLU, the effective branch product uses the gated combination; see Appendix for derivation.

**Critical finding**: Using the wrong extraction formula yields chance-level detection accuracy. In our experiments on GPT-2 (12 layers, $d=768$), applying the correct MLP formula ($M = W_{\text{down}} W_{\text{up}}$) achieves 100% branch detection, while applying an incorrect formula (e.g., treating attention weights as MLP) reduces accuracy to the $1/L \approx 8\%$ random baseline.

### 4.3.4 Null Distribution Construction

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

### 4.3.5 Threshold Selection and Decision Criteria

Given a calibrated null distribution $\mathcal{N}$ with mean $\mu$ and standard deviation $\sigma$, we convert the raw lineage score to a z-score:
$$
Z(A,B) = \frac{\mathcal{L}(A,B) - \mu(\mathcal{N})}{\sigma(\mathcal{N}) + \delta}
$$
where $\delta = 10^{-8}$ prevents division by zero.

**Threshold selection**: We define two thresholds corresponding to confidence levels:

| Threshold | Value | Interpretation |
|-----------|-------|----------------|
| $\tau_{\text{upper}}$ | $Z_{0.95} \approx 1.645$ | 95% confidence for positive verdict |
| $\tau_{\text{lower}}$ | $Z_{0.05} \approx -1.645$ | 95% confidence for negative verdict |

Under the assumption that the null distribution is approximately Gaussian:
- $Z > 1.645$ implies $p < 0.05$ under $H_0$ (unrelated models)
- $Z > 2.576$ implies $p < 0.005$ (99% confidence)
- $Z > 3.291$ implies $p < 0.0005$ (99.95% confidence)

**Conservative recommendation**: For high-stakes applications (legal disputes, regulatory audits), we recommend $\tau_{\text{upper}} = 3.0$ to minimize false positives.

### 4.3.6 Output Categories

The protocol produces one of four outputs:

**DESCENDANT** ($Z > \tau_{\text{upper}}$): The suspect checkpoint preserves branch-specific weight structure from the reference with high confidence. The z-score provides a quantitative measure of evidence strength.

**NON-DESCENDANT** ($Z < \tau_{\text{lower}}$): The suspect checkpoint shows no more similarity to the reference than expected between independently trained models. This rules out direct descent through fine-tuning, quantization, or pruning.

**INCONCLUSIVE** ($\tau_{\text{lower}} \leq Z \leq \tau_{\text{upper}}$): The evidence is ambiguous. This may occur when:
- The suspect has undergone extensive modification
- The reference model was itself derived from a third model
- Noise or quantization has partially degraded the fingerprint

**INCOMPATIBLE** (architecture mismatch): The models cannot be compared because they have different architectures. No provenance verdict is possible.

### 4.3.7 Robustness Guarantees

The protocol maintains reliability under common model transformations:

**Table 3: Verification robustness under model transformations**

| Transformation | Parameters | Detection |
|----------------|------------|-----------|
| Fine-tuning | LR $10^{-5}$--$10^{-3}$, 50 epochs | 100% |
| Gaussian noise | $\sigma \leq 0.3$ | 100% |
| Gaussian noise | $\sigma = 0.5$ | 96% |
| Gaussian noise | $\sigma = 1.0$ | 85% |
| Quantization | 16-bit, 8-bit | 100% |
| Quantization | 6-bit | 94% |
| Quantization | 4-bit | 45% |
| Pruning | Up to 30% sparsity | 100% |
| LoRA merge | Rank 8--64 | 100% |

**Degradation boundary**: The fingerprint degrades gracefully with perturbation intensity, collapsing only when perturbations also substantially degrade model utility. At $\sigma = 1.0$ noise (where model perplexity increases by $>50\%$), diagonal similarity remains at 85% versus 53% for full-matrix methods.

### 4.3.8 Failure Modes and Limitations

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
A motivated adversary with white-box access to both checkpoints could potentially craft perturbations that remove the fingerprint while preserving functionality. We analyze such adaptive attacks in Section 6.

### 4.3.9 Practical Implementation

**Computational cost**: For a model with $L$ layers and hidden dimension $d$:
- Branch product computation: $O(L \cdot d^2 \cdot k)$ where $k$ is branch depth
- Similarity matrix: $O(L^2 \cdot d^2)$
- Hungarian matching: $O(L^3)$
- Total: Dominated by similarity matrix, but with $L \leq 100$ and precomputed fingerprints, verification completes in under 1 second on commodity hardware.

**Memory**: Each branch product $M_\ell \in \mathbb{R}^{d \times d}$ requires $d^2 \cdot 4$ bytes (float32). For GPT-2 ($d=768$, $L=12$), this is approximately 28 MB per model.

**Reference implementation**: We provide a Python implementation using PyTorch and SciPy that verifies a (reference, suspect) pair in:
- 0.3 seconds for GPT-2 (124M parameters)
- 1.2 seconds for GPT-2-XL (1.5B parameters)

### 4.3.10 Protocol Summary

The diagonal-dominance verification protocol provides:

1. **Passive verification**: No watermarks or metadata required; works retroactively on any residual model
2. **Calibrated decisions**: Z-scores provide quantitative evidence strength with interpretable confidence levels
3. **Three-way output**: Explicit handling of inconclusive cases prevents overconfident verdicts
4. **Robustness**: Survives fine-tuning, quantization, pruning, and noise at levels that preserve model utility
5. **Efficiency**: Sub-second verification on commodity hardware

The protocol fills a critical gap in model provenance: it enables auditing of checkpoints that were deployed without foresight, where external evidence is unavailable or untrustworthy.

### Comparison with Watermarking Verification Protocols

Unlike watermarking approaches \citep{uchida2017embedding,adi2018turning,rouhani2019deepsigns}, the diagonal-dominance protocol:

| Property | Watermarking | Diagonal Dominance |
|----------|-------------|-------------------|
| Insertion required | Yes, during training | No, passive |
| Retroactive verification | No | Yes |
| Survives fine-tuning | Often not \citep{lukas2022sok} | Yes |
| False positive control | Explicit key space | Statistical calibration |
| Adversarial removal | Black-box attacks exist | Requires utility destruction |

The key distinction is that watermarks embed *artificial* signals that can be removed without affecting model function, while diagonal dominance arises from the optimization process itself and cannot be removed without degrading the trained structure that enables model performance.

---

## References

- \citet{kuhn1955hungarian}: Kuhn, H.W. "The Hungarian Method for the Assignment Problem." *Naval Research Logistics Quarterly*, 1955.
- \citet{ainsworth2023git}: Ainsworth, S.K., Hayase, J., and Srinivasa, S. "Git Re-Basin: Merging Models modulo Permutation Symmetries." *ICLR*, 2023.
- \citet{singh2020model}: Singh, S.P. and Jaggi, M. "Model Fusion via Optimal Transport." *NeurIPS*, 2020.
- \citet{kornblith2019similarity}: Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. "Similarity of Neural Network Representations Revisited." *ICML*, 2019.
- \citet{ilharco2023editing}: Ilharco, G., et al. "Editing Models with Task Arithmetic." *ICLR*, 2023.
- \citet{matena2022merging}: Matena, M.S. and Raffel, C. "Merging Models with Fisher-Weighted Averaging." *NeurIPS*, 2022.
- \citet{uchida2017embedding}: Uchida, Y., Nagai, Y., Sakazawa, S., and Satoh, S. "Embedding Watermarks into Deep Neural Networks." *ICMR*, 2017.
- \citet{adi2018turning}: Adi, Y., Baum, C., Cisse, M., Pinkas, B., and Keshet, J. "Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring." *USENIX Security*, 2018.
- \citet{rouhani2019deepsigns}: Rouhani, B.D., Chen, H., and Koushanfar, F. "DeepSigns: An End-to-End Watermarking Framework for Ownership Protection of Deep Neural Networks." *ASPLOS*, 2019.
- \citet{lukas2022sok}: Lukas, N., et al. "SoK: How Robust is Image Classification Deep Neural Network Watermarking?" *IEEE S&P*, 2022.
