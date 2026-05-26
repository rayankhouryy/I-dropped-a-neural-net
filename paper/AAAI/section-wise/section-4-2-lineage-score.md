# Section 4.2: Model-Level Lineage Score

The block-level fingerprints developed in Section 3 verify that individual residual branches share checkpoint-specific structure. However, provenance decisions must be made at the *model* level: given a reference checkpoint $A$ and a suspect checkpoint $B$, we need a scalar decision statistic that determines whether $B$ descends from $A$. This section develops such a statistic, calibrates it against a null distribution of unrelated models, and establishes decision rules with quantified error rates.

## 4.2.1 Aggregating Block-Level Similarity

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

## 4.2.2 Empirical Score Distributions

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

## 4.2.3 Statistical Calibration via Null Distribution

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

## 4.2.4 Decision Rules

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

## 4.2.5 ROC Analysis and Detection Metrics

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

## 4.2.6 Distinguishing Hard Cases

The lineage score correctly handles several challenging scenarios:

**Functionally Similar but Non-Descendant.** Models trained independently on the same task achieve near-identical accuracy but have $\mathcal{L} \approx 0$. The diagonal fingerprint captures weight-level identity, not functional similarity.

**Distilled Models.** A student network trained to mimic a teacher's outputs does *not* inherit the teacher's diagonal fingerprint:
- Mean $\mathcal{L}(\text{teacher}, \text{distilled student}) = 0.002$
- Classification: Non-descendant (correct)

This is the expected and desired behavior: distillation transfers function, not weights. The fingerprint reads weight-level provenance, distinguishing it from behavioral cloning.

**Random Initialization.** Randomly initialized models (before any training) score:
- Mean $\mathcal{L}(\text{trained}, \text{random init}) = 0.001$

The fingerprint is absent at initialization and emerges only through training, confirming that it captures a training-induced property rather than an architectural artifact.

## 4.2.7 Comparison with Related Work

Our lineage score relates to but differs from existing similarity metrics:

**Centered Kernel Alignment (CKA)** [Kornblith et al. 2019] measures representation similarity using activation patterns on a probe dataset. CKA captures functional similarity: models that compute similar functions have high CKA regardless of weight-level relationship. In contrast, our lineage score operates on weights alone and captures checkpoint identity: a fine-tuned model has high $\mathcal{L}$ with its parent but may have high CKA with any model trained on similar data.

**Task Vectors** [Ilharco et al. 2023] define directions in weight space corresponding to task-specific adaptations. While task vectors enable arithmetic operations on model capabilities, they require access to a shared pre-trained base. Our fingerprint requires no such anchor---it identifies whether two checkpoints share ancestry from weights alone.

**Fisher-Weighted Averaging** [Matena and Raffel 2022] uses Fisher information to identify important parameters when merging models. This operates on a different problem (combining models) but shares the insight that not all weight dimensions are equally informative. Our centering operation similarly emphasizes checkpoint-specific variation by removing the generic identity-like component.

## 4.2.8 Summary

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

This completes the bridge from block-level fingerprints to model-level provenance decisions. Section 4.3 extends the framework to handle unknown block correspondences via optimal transport, and Section 5 evaluates robustness under adversarial and realistic perturbations.

---

## Key Equations Reference

**Lineage Score:**
$$\mathcal{L}(A, B) = \frac{1}{L} \sum_{\ell=1}^{L} \langle \psi_\ell^A, \psi_\ell^B \rangle$$

**Null Distribution:**
$$\mathcal{N}_A = \left\{ \mathcal{L}(A, B_j^-) : B_j^- \notin \mathcal{D}(A) \right\}_{j=1}^{n}$$

**Z-Score:**
$$Z(A, B) = \frac{\mathcal{L}(A, B) - \mu_{\text{null}}}{\sigma_{\text{null}} + \delta}$$

**Decision Rule:**
$$B \in \mathcal{D}(A) \iff Z(A, B) > \tau$$

---

## Connections to Prior Sections

- **Section 3.2**: Defines the centered diagonal fingerprint $\psi_\ell$
- **Section 3.3**: Establishes margin analysis showing correct pairs score at $\Theta(\sqrt{d})$
- **Section 4.1**: Introduces block-level pairing and Hungarian matching
- **Section 4.3** (next): Extends to unknown block correspondences
- **Section 5**: Robustness experiments validating the score under perturbation
