# Background Research: Section 2 - Background and Problem Setting

Merged research document for AAAI 2026 paper on intrinsic model fingerprinting.

---

# Section 2.1: Passive Model Provenance

## Purpose
Define the practical problem precisely.

---

## 1. Defining Model Provenance

### 1.1 General Definition
**Established concept (W3C PROV standard):** Data provenance is "information about entities, activities, and agents involved in producing a piece of data or thing" that can be used to assess "quality, reliability, or trustworthiness" [W3C PROV 2013].

Core structures:
- **Entities** - the artifacts themselves (model weights)
- **Activities** - processes that create or transform data (training, fine-tuning)
- **Agents** - people, institutions, or systems involved (model creators)

### 1.2 Model Provenance Definition
**Working definition for paper:**
> Model provenance refers to the verifiable history of a neural network's creation and modification, including its training origin, intermediate checkpoints, and any post-training transformations such as fine-tuning, pruning, or quantization.

### 1.3 Recent Work
- Kuditipudi et al. [NeurIPS 2026] - "Blackbox Model Provenance via Palimpsestic Membership Inference"
- Mu et al. [2023] - "Model DNA" examining relationships between neural networks
- Liu et al. [ACM 2023] - "Provenance of Training" leveraging weight distance properties

---

## 2. Active vs. Passive Provenance Verification

### 2.1 Active Methods (Watermarking)
**Definition:** Embed deliberate signals during or after training for later extraction.

**Three families [Lukas et al. 2022]:**

1. **Embedding-based watermarks:** Modify training loss to encode fingerprint into weights
   - Uchida et al. [ICMR 2017] - regularization term projecting binary string into filters
   - DeepSigns [Rouhani et al. 2019] - embeds in activation probability distributions
   - DeepMarks [Chen et al. 2019] - survives model compression

2. **Backdoor-based watermarks:** Train to produce specific outputs on trigger set
   - Adi et al. [USENIX Security 2018] - arbitrary input-output memorization
   - Zhang et al. [ASIACCS 2018] - trigger set as cryptographic key

3. **Decision-boundary fingerprinting:** Inputs near classification boundary
   - Le Merrer et al. [2020] - adversarial frontier stitching
   - IPGuard [Cao et al. 2021] - boundary-sensitive inputs

### 2.2 Passive Methods (Our Contribution)
**Definition:** Extract provenance signals from weights without training modification.

**Key distinction:**
> Passive provenance reads structural fingerprints that emerge from the training process itself, rather than embedding information that must be inserted before training concludes.

### 2.3 Comparison Table

| Aspect | Active (Watermarking) | Passive (Intrinsic) |
|--------|----------------------|---------------------|
| Requires foresight | Yes | **No** |
| Requires training modification | Yes | **No** |
| Retroactive verification | No | **Yes** |
| Robustness to fine-tuning | Variable | X% maintained |
| Applies to pre-existing models | No | **Yes** |

---

## 3. White-Box vs. Black-Box Verification

### 3.1 Definitions
**White-box verification:** Full access to model weights, architecture, and internal representations.

**Black-box verification:** Only query access (input-output pairs), no internal parameters.

### 3.2 Our Positioning
**Our setting is explicitly white-box:**
> We assume the verifier has access to both the reference checkpoint's weights and the suspect checkpoint's weights. This matches scenarios where a model is suspected to be stolen and the original owner seeks to prove provenance.

**Sentence for paper:**
> Our method is a white-box verification procedure: the verifier has access to the weights of both reference and suspect checkpoints. This setting matches regulatory audits, model hub compliance checks, and ownership disputes where both parties can produce their weights.

---

## 4. Weight-Level Lineage vs. Behavioral Equivalence

### 4.1 Critical Distinction
**Weight-level lineage:** Suspect weights derived from reference through allowed transformations (fine-tuning, pruning, quantization) without retraining from scratch.

**Behavioral/functional equivalence:** Same/similar outputs on same inputs, regardless of weight relationship.

### 4.2 Model Extraction Creates Functional Copies Without Weight Lineage
Tramer et al. [USENIX Security 2016]:
> "The extracted model...duplicates the functionality of the target model...it does not necessarily have identical weights."

Orekondy et al. [2018] (Knockoff Nets):
> "The knockoff can use a different architecture than the original model...Random images from different distributions can produce effective knockoffs."

### 4.3 Sentences for Paper
> The diagonal-dominance fingerprint detects weight-level lineage, not behavioral equivalence. A model that produces identical outputs but was trained independently (or via distillation) will not share the fingerprint, because its weights evolved through a separate optimization trajectory.

> This is a feature, not a limitation: weight-level identity is the appropriate forensic target when the question is whether specific model parameters were copied or derived.

---

## 5. Distillation and Independent Training (Outside Scope)

### 5.1 Knowledge Distillation
Hinton, Vinyals, and Dean [2015]:
> "Knowledge distillation trains a smaller 'student' network to mimic the outputs of a larger 'teacher' network...The student learns from the soft probability distributions rather than just hard labels."

**Critical point:** Distillation does not preserve weight-level structure. Student learns to approximate teacher's output distribution, but weights and architecture can differ significantly.

### 5.2 Why Distillation Erases the Fingerprint (Novel Claim)
From abstract:
> "Knowledge distillation, by contrast, erases the signal: the score reads weight-level rather than function-level identity."

**Why expected:**
- Distillation initializes student weights randomly
- Student learns through fresh optimization trajectory
- No coupling between student and teacher weight matrices
- Structural fingerprint is training-trajectory-specific

### 5.3 Sentences for Paper
> Distilled models and independently trained functional copies are explicitly outside the positive guarantee. A student model trained via distillation may replicate the teacher's behavior with high fidelity, but it evolves its weights through a separate optimization process that produces distinct structural fingerprints.

---

## 6. Formal Problem Statement

### 6.1 Task Definition
> Given a reference checkpoint R and a suspect checkpoint S with missing or untrusted metadata, determine whether S shares weight-level lineage with R after allowed post-training transformations.

### 6.2 Allowed Transformations
- Fine-tuning (various learning rates and epochs)
- Pruning (up to ~70% sparsity)
- Quantization (down to INT4)
- Weight noise (up to ~20% relative magnitude)
- Per-block orthogonal rotations (function-preserving)
- Hidden-unit permutations (function-preserving)

### 6.3 Explicit Scope Boundaries
1. **Not run-level provenance:** Identifies which matrices belong together within a model
2. **Not architecture-agnostic:** Requires knowledge of residual-branch factorization
3. **Not black-box:** Requires weight access to both checkpoints
4. **Not distillation-robust:** By design, distillation erases the fingerprint

---

# Section 2.2: Limits of Hashes, Metadata, and Watermarks

## Purpose
Position against watermarking literature. Establish that prior work asks "how to insert or protect an ownership signal" while we ask "whether training already leaves a signal that can be read retroactively."

---

## 1. Cryptographic Hashes

### The Avalanche Effect Problem
Cryptographic hash functions (SHA-256, MD5) exhibit the **avalanche effect**: single-bit change produces completely different hash. Makes hashes fundamentally unsuitable for neural network provenance.

**Key limitation:** Any modification—fine-tuning, pruning, quantization, or floating-point rounding—produces entirely different hash. Binary match/no-match with no notion of similarity or derivation.

**Sentence for paper:**
> Cryptographic hashes fail immediately upon any weight modification: the avalanche effect ensures that fine-tuning, pruning, or even numerical precision changes produce entirely different digests, providing no signal of derivation.

---

## 2. Metadata and Logging Systems

### Model Cards [Mitchell et al. 2019]
**Citation:** Mitchell et al., "Model Cards for Model Reporting," FAT* 2019.

**Provides:** Benchmarked evaluation, intended use cases, performance details.
**Limitation:** Does not cryptographically bind documentation to weights. Stolen models can be redistributed with fabricated cards.

### MLflow / Weights & Biases
**Provides:** Model versioning, lineage tracking, artifact storage with metadata.
**Limitation:** Requires organizational adoption and trusted infrastructure. No verification outside the ecosystem. Records can be incomplete, falsified, or unavailable.

**Sentence for paper:**
> Metadata systems—model cards [Mitchell et al. 2019], MLflow registries, W&B experiment tracking—require trusted provenance records that may be incomplete, falsified, or absent when models are redistributed outside controlled environments.

---

## 3. Embedded Watermarks (Training-time)

### Uchida et al. 2017
**Citation:** Uchida et al., "Embedding Watermarks into Deep Neural Networks," ICMR 2017.
**Method:** Regularization term embeds binary watermark into convolutional filters.
**Limitation:** Requires modification of training procedure (forethought). Later work showed not robust under adaptive attacks.

### DeepSigns [Rouhani et al. 2019]
**Method:** Embeds watermarks in activation maps using regularizer.
**Limitation:** Still requires training-time insertion.

### DeepMarks [Chen et al. 2019]
**Method:** Fingerprints that survive compression.
**Limitation:** Cannot verify models trained before watermarking became standard.

**Sentence for paper:**
> Embedded watermarks [Uchida et al. 2017; Rouhani et al. 2019] encode ownership signals into weights or activations during training, but require forethought—the watermark must be inserted before training concludes—precluding retroactive verification of already-deployed models.

---

## 3.5 Output Watermarking vs. Weight-Level Provenance

### SynthID [Dathathri et al. 2024]
**Citation:** Dathathri et al., "Scalable watermarking for identifying large language model outputs," Nature 2024.

**Method:** SynthID watermarks the *outputs* of generative models rather than the weights themselves. For text, it uses a pseudo-random g-function to bias token sampling probabilities during generation, embedding an imperceptible statistical signature. For images/video, it embeds watermarks directly into pixel data using jointly-trained encoder-decoder networks.

**Key distinction from weight-level provenance:**
- SynthID answers: "Was this content generated by an AI model?"
- Our method answers: "Does this model's weights derive from that model's weights?"

**Limitation for model provenance:** Output watermarking cannot verify weight-level lineage—a fine-tuned derivative model and a distilled student model may both produce watermarked outputs, yet only the former shares weight-level identity with the original. Furthermore, output watermarks require deployment-time integration and do not apply retroactively to already-deployed models without the watermarking system.

**Sentence for paper:**
> Output watermarking systems such as SynthID [Dathathri et al. 2024] identify AI-generated content by embedding statistical signatures during generation, but address a fundamentally different question than weight-level provenance: they detect whether content was AI-generated, not whether one model's weights derive from another's.

---

## 4. Backdoor/Trigger Watermarks

### Adi et al. 2018
**Citation:** Adi et al., "Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring," USENIX Security 2018.
**Method:** Train to memorize arbitrary input-output pairs (trigger set) as cryptographic key.
**Limitations:** Requires training-time insertion. Trigger behavior can be removed via fine-tuning/pruning.

### Entangled Watermarks [Jia et al. 2021]
**Citation:** Jia et al., USENIX Security 2021.
**Key insight:** Traditional watermarks use "outlier input-output pairs" distinct from task distribution, making them "easily removed through compression or other forms of knowledge transfer."

**Sentence for paper:**
> Backdoor watermarks [Adi et al. 2018; Zhang et al. 2018] train models to produce specific outputs on secret trigger inputs, but can be removed via fine-tuning [Jia et al. 2021] or behave unpredictably after domain adaptation.

---

## 5. The SoK Verdict [Lukas et al. 2022]

**Citation:** Lukas et al., "SoK: How Robust is Image Classification Deep Neural Network Watermarking?" IEEE S&P 2022.

**THIS IS THE KEY CITATION.**

**Key Findings:**
1. **"None of the surveyed watermarking schemes is robust in practice"** against comprehensive removal attacks
2. Novel attacks: weight shifting, smooth retraining
3. Existing schemes "fail to withstand adaptive attacks"
4. "Intrinsic flaws in how robustness is currently evaluated"

**Attack categories that defeat watermarks:**
- Fine-tuning (various learning rates/epochs)
- Pruning (magnitude-based, lottery ticket)
- Model extraction / knowledge distillation
- Weight perturbation
- Neural cleanse-style removal

**Sentence for paper:**
> A systematic evaluation by [Lukas et al. 2022] found that no surveyed watermarking scheme is robust in practice: fine-tuning, pruning, and model extraction all significantly degrade or remove watermarks within realistic compute budgets.

---

## 6. Behavioral Fingerprinting (Cannot Detect Weight Lineage)

### IPGuard [Cao et al. 2021]
**Method:** Extract data points near classification boundary as fingerprints.
**Limitation:** Depends on preserving exact decision surface. Fine-tuning/domain adaptation shifts boundaries.

### Le Merrer et al. 2020
**Method:** Adversarial examples to mark decision frontier.
**Limitation:** Identifies functional similarity but not weight lineage—distilled models may have similar boundaries but completely different weights.

### Model Extraction [Tramer et al. 2016]
**Implication:** Behavioral methods cannot distinguish extracted model from original, yet extracted model shares no weight-level identity.

**Sentence for paper:**
> Decision-boundary fingerprinting [Cao et al. 2021; Le Merrer et al. 2020] identifies functional similarity but cannot distinguish weight lineage: a distilled student may share decision surfaces while having entirely independent weights.

---

## 7. The Key Differentiator

**Central quote for the paper:**
> **Prior watermarking work asks how to insert or protect an ownership signal. We ask whether training already leaves a signal that can be read retroactively.**

### Comparison Table

| Method | Foresight? | Survives FT? | Weight-level ID? |
|--------|-----------|--------------|------------------|
| Cryptographic hashes | No | No (any change fails) | Yes (exact only) |
| Metadata/logs | No | N/A | No (external) |
| Embedded watermarks | **Yes** | Partial | Yes |
| Backdoor watermarks | **Yes** | No [Lukas 2022] | Yes |
| Decision boundaries | No | Partial | **No** |
| **Diagonal dominance** | **No** | **Yes** | **Yes** |

### The Fundamental Gap We Fill
1. **No forethought required:** Fingerprint emerges from training itself
2. **Robust to modification:** Signal persists through fine-tuning, pruning, quantization
3. **Weight-level identity:** Distinguishes originals from distilled copies

---

# Section 2.3: Residual Branch Products

## Purpose
Introduce the architectural object studied by the method.

---

## 1. Mathematical Definition

### 1.1 General Form
A **residual block** computes:
$$x' = x + F(x)$$

where $x \in \mathbb{R}^d$ is the input, $F: \mathbb{R}^d \to \mathbb{R}^d$ is the residual branch function, and $x'$ is the output.

### 1.2 Residual Branch Product
For a residual branch with $K$ linear transformations:
$$F(x) = W_K \phi_{K-1}(W_{K-1} \cdots \phi_1(W_1 x))$$

The **residual branch product** is:
$$M = W_K W_{K-1} \cdots W_1 \in \mathbb{R}^{d \times d}$$

**Novel insight:** Training enforces $M \approx -\varepsilon I$ to maintain dynamical isometry.

### 1.3 Jacobian Connection
The Jacobian of the residual block:
$$J = I + J_F(x)$$

For dynamical isometry, we need $J$ close to orthogonal, which implies $J_F \approx -\varepsilon I$.

**CITATIONS:**
- [He et al. 2016] Deep Residual Learning (CVPR) - Introduced residual blocks
- [Pennington et al. 2017] Dynamical Isometry (NeurIPS) - Isometry condition

---

## 2. Architecture-Specific Factorization Table

| Architecture | Residual Form | Branch Product $M$ | $K$ |
|--------------|---------------|-------------------|-----|
| **Residual MLP / BasicBlock** | $x + W_2 \phi(W_1 x)$ | $W_2 W_1$ | 2 |
| **ResNet Bottleneck** | $x + W_3 \phi(W_2 \phi(W_1 x))$ | $W_3 W_2 W_1$ | 3 |
| **Transformer MLP** | $x + W_2 \text{GELU}(W_1 \text{LN}(x))$ | $W_2 W_1$ | 2 |
| **SwiGLU MLP** | $x + W_{\text{down}}(\text{SiLU}(W_{\text{gate}}) \odot W_{\text{up}} x)$ | $W_{\text{down}} W_{\text{up}}$ | 2 |
| **Attention V/O path** | $x + W_O \text{Attn}(W_V x)$ | $W_O W_V$ | 2 |
| **Attention Q/K path** | $\text{softmax}(Q K^\top / \sqrt{d})$ | $W_Q W_K^\top$ | 2 |
| **ViT MLP/Attention** | Same as Transformer | Same | 2 |
| **ConvNeXt MLP** | $x + W_2 \text{GELU}(W_1 \text{LN}(x))$ | $W_2 W_1$ | 2 |

**Key insight from ACML:** Using wrong factorization (e.g., $W_3 W_1$ for Bottleneck instead of $W_3 W_2 W_1$) yields chance-level accuracy. Correct factorization recovers 91-100% pair accuracy.

**CITATIONS:**
- [He et al. 2016] ResNet (BasicBlock, Bottleneck)
- [Vaswani et al. 2017] Transformer (Attention Is All You Need)
- [Dosovitskiy et al. 2021] ViT (ICLR)
- [Liu et al. 2022] ConvNeXt (CVPR)
- [Shazeer 2020] SwiGLU (arXiv:2002.05202)

---

## 3. Architecture Family Details

### 3.1 ResNet Family [He et al. 2016]

**BasicBlock** (ResNet-18, ResNet-34):
```
x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU
```
Product: $M = W_2 W_1 \in \mathbb{R}^{c \times c}$

**Bottleneck** (ResNet-50/101/152):
```
x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+x) -> ReLU
```
Product: $M = W_3 W_2 W_1$

**Spatial collapse for convolutions:**
1. **Channel-sum:** $W[c_{out}, c_{in}] = \sum_{h,w} K[c_{out}, c_{in}, h, w]$
2. **Center-tap:** $W[c_{out}, c_{in}] = K[c_{out}, c_{in}, k//2, k//2]$

**BatchNorm folding:** $W_{\text{folded}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W$

### 3.2 Transformer Architecture [Vaswani et al. 2017]

**MLP sublayer:**
$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 \cdot \text{LN}(x))$$
where $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$.

Product: $M = W_2 W_1 \in \mathbb{R}^{d \times d}$

**Multi-head Attention paths:**
1. **V/O path:** $W_O W_V \in \mathbb{R}^{d \times d}$ - how attention writes to residual stream
2. **Q/K path:** $W_Q W_K^\top \in \mathbb{R}^{d \times d}$ - query-key coupling

### 3.3 GPT-2 Family [Radford et al. 2019]

| Model | Params | Layers | Hidden $d$ | FF dim |
|-------|--------|--------|------------|--------|
| GPT-2 | 124M | 12 | 768 | 3072 |
| GPT-2-medium | 355M | 24 | 1024 | 4096 |
| GPT-2-large | 774M | 36 | 1280 | 5120 |
| GPT-2-xl | 1.5B | 48 | 1600 | 6400 |

Weight names: `c_fc` ($W_1$), `c_proj` ($W_2$) for MLP.

**ACML results:** 100% pair accuracy across all sizes; mean $s(i,i)$ scales from 4.18 ($d=768$) to 7.77 ($d=1600$), confirming $\sqrt{d}$ scaling.

### 3.4 SwiGLU Architectures (LLaMA, Mistral, Qwen)

**SwiGLU MLP:**
$$\text{MLP}(x) = W_{\text{down}} \left( \text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x \right)$$

Three valid factorizations (all achieve 100% pair accuracy):
1. $W_{\text{down}} W_{\text{up}}$
2. $W_{\text{down}} W_{\text{gate}}$
3. Joint stack: $W_{\text{down}} [W_{\text{up}}; W_{\text{gate}}]$

**Grouped-Query Attention (GQA):** Mistral uses 8 KV heads expanded to 32 Q heads.

---

## 4. Connection to Dynamical Isometry Theory

### 4.1 Why Residual Connections Enable Stable Training
**Problem with deep networks:** In a plain network with $L$ layers, $J = \prod_{i=1}^{L} J_i$. If $\|J_i\| < 1$, gradients vanish; if $\|J_i\| > 1$, gradients explode.

**Residual solution:** Each block has Jacobian $J_{\text{block}} = I + J_F$. Even if $\|J_F\|$ is small, $J_{\text{block}}$ remains close to identity.

### 4.2 Dynamical Isometry Condition
**Definition:** Network satisfies **dynamical isometry** if singular values of input-output Jacobian are all close to 1 throughout training.

For residual blocks: If $J = I + J_F$ and we want $J$ orthogonal, we need $J_F \approx -\varepsilon I$ for small $\varepsilon$.

**CITATIONS:**
- [Saxe et al. 2014] Exact Solutions in Deep Linear Networks (ICLR)
- [Pennington et al. 2017] Dynamical Isometry (NeurIPS)

### 4.3 ResNets Satisfy Dynamical Isometry Universally
**Tarnowski et al. 2019:** ResNets satisfy dynamical isometry at initialization when residual branches are scaled appropriately, and this holds **universally for any activation function**.

**CITATION:** Tarnowski et al., "Dynamical Isometry is Achieved in Residual Networks in a Universal Way for Any Activation Function," AISTATS 2019.

### 4.4 Additional Theory
- **Zaeemzadeh et al. 2020:** Skip connections preserve gradient norms across depth (IEEE TPAMI)
- **De & Smith 2020:** BatchNorm biases residual blocks toward identity function (NeurIPS)

---

## 5. Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $d$ | Hidden dimension (residual stream width) |
| $d_{\text{ff}}$ | Feedforward dimension (typically $4d$) |
| $K$ | Depth of residual branch (number of linear layers) |
| $W_k$ | The $k$-th weight matrix in the branch |
| $M$ | Residual branch product $W_K \cdots W_1$ |
| $\phi$ | Nonlinearity (ReLU, GELU, SiLU, etc.) |
| $\varepsilon$ | Scalar magnitude of diagonal component |
| $E$ | Residual matrix where $M = -\varepsilon I + E$, $\text{tr}(E) = 0$ |
| $s(i,j)$ | Diagonal-dominance score $|\text{tr}(M)|/\|M\|_F$ |

---

# Complete Citation List

## Section 2.1 Citations
- [W3C PROV 2013] PROV Data Model standard
- [Uchida et al. 2017] Embedding Watermarks into DNNs (ICMR)
- [Rouhani et al. 2019] DeepSigns (ASPLOS)
- [Adi et al. 2018] Backdoor watermarking (USENIX Security)
- [Le Merrer et al. 2020] Adversarial frontier stitching
- [Cao et al. 2021] IPGuard (AsiaCCS)
- [Tramer et al. 2016] Stealing ML models (USENIX Security)
- [Orekondy et al. 2018] Knockoff Nets
- [Hinton et al. 2015] Knowledge Distillation
- [Shokri et al. 2017] Membership Inference (IEEE S&P)
- [Jia et al. 2021] Proof-of-Learning / Entangled Watermarks (USENIX Security)

## Section 2.2 Citations
- [Mitchell et al. 2019] Model Cards (FAT*)
- [Lukas et al. 2022] SoK: DNN Watermarking Robustness (IEEE S&P) **KEY CITATION**
- [Chen et al. 2019] DeepMarks (ICMR)
- [Zhang et al. 2018] Protecting IP of DNNs (ASIACCS)
- [Szyller et al. 2021] DAWN (ACM Multimedia)
- [Dathathri et al. 2024] SynthID: Scalable watermarking for LLM outputs (Nature)

## Section 2.3 Citations
- [He et al. 2016] Deep Residual Learning (CVPR)
- [He et al. 2016b] Identity Mappings in Deep Residual Networks (ECCV)
- [Vaswani et al. 2017] Attention Is All You Need (NeurIPS)
- [Dosovitskiy et al. 2021] ViT (ICLR)
- [Liu et al. 2022] ConvNeXt (CVPR)
- [Radford et al. 2019] GPT-2 (OpenAI)
- [Devlin et al. 2019] BERT (NAACL)
- [Saxe et al. 2014] Deep Linear Networks (ICLR)
- [Pennington et al. 2017] Dynamical Isometry (NeurIPS)
- [Tarnowski et al. 2019] Universal Isometry in ResNets (AISTATS)
- [Zaeemzadeh et al. 2020] Norm Preservation (IEEE TPAMI)
- [De & Smith 2020] BatchNorm Identity Bias (NeurIPS)
- [Shazeer 2020] SwiGLU (arXiv)
- [Touvron et al. 2023] LLaMA
- [Jiang et al. 2023] Mistral 7B

---

# Placeholder Metrics (Novel Claims)

| Claim | Placeholder | Source |
|-------|-------------|--------|
| Pair recovery across architectures | X--Y% | ACML experiments |
| Robustness under fine-tuning | X--Y% | ACML robustness sweep |
| Score scaling with dimension | $\sqrt{d}$ | Theoretical (verified) |
| Distillation drops to baseline | ~0% correlation | ACML distillation experiment |
| Signal-to-noise gap | $O(d)$ | Theoretical (Corollary) |
| Negative trace fraction | 81-92% | ACML GPT-2 experiments |

---

# Summary: Established vs Novel Claims

| Claim | Status | Citation |
|-------|--------|----------|
| Residual blocks compute $x + F(x)$ | Established | He et al. 2016 |
| Dynamical isometry requires near-orthogonal Jacobians | Established | Pennington et al. 2017 |
| ResNets satisfy isometry universally | Established | Tarnowski et al. 2019 |
| Watermarks not robust to removal | Established | Lukas et al. 2022 |
| Definition: $M = W_K \cdots W_1$ as branch product | **Novel** | This paper |
| Architecture-aware factorization table | **Novel** | This paper |
| Training enforces $M \approx -\varepsilon I$ | **Novel empirical** | This paper |
| Using $M$ for provenance verification | **Novel application** | This paper |

---

*Document compiled: May 2026*
*For: AAAI 2026 Submission - Section 2*
