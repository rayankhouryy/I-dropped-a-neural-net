# Model Lineage Tracing via Diagonal Dominance Fingerprints

## 1. Background: Diagonal Dominance in Trained Networks

### 1.1 Residual Block Structure

Modern neural networks use residual blocks of the form:

$$x_{\ell+1} = x_\ell + \sigma(W_\ell^{\text{in}} x_\ell) W_\ell^{\text{out}}$$

where $W^{\text{in}} \in \mathbb{R}^{d \times 4d}$ expands the representation and $W^{\text{out}} \in \mathbb{R}^{4d \times d}$ contracts it back.

### 1.2 Branch Product

The **branch product** is the composition of these weight matrices:

$$M_\ell = W_\ell^{\text{out}} W_\ell^{\text{in}} \in \mathbb{R}^{d \times d}$$

This matrix characterizes how the residual branch transforms the hidden state.

### 1.3 Diagonal Dominance Score

We define the **diagonal dominance (DD) score** as:

$$s(M) = \frac{|\text{tr}(M)|}{\|M\|_F}$$

where $\text{tr}(M) = \sum_i M_{ii}$ is the trace and $\|M\|_F = \sqrt{\sum_{ij} M_{ij}^2}$ is the Frobenius norm.

**Interpretation:**
- For a random matrix: $s(M) \approx 1/\sqrt{d}$ (near zero for large $d$)
- For a scaled identity $M = cI$: $s(M) = \sqrt{d}$ (maximum possible)

### 1.4 Empirical Finding: Training Induces Diagonal Dominance

| Model | Before Training | After Training | Amplification |
|-------|-----------------|----------------|---------------|
| GPT-2 (d=768) | 0.03 | 4.18 | **135x** |
| Random baseline | 0.036 | -- | -- |
| Theoretical max | -- | 27.7 | -- |

Training transforms the branch product from a random matrix toward a scaled identity structure:

$$M_\ell \approx -\varepsilon_\ell I + E_\ell$$

where $\varepsilon_\ell > 0$ and $E_\ell$ is a small residual matrix.

---

## 2. Diagonal Fingerprints for Lineage Tracing

### 2.1 Motivation

If training induces checkpoint-specific diagonal structure, the diagonal entries of $M_\ell$ should serve as a **fingerprint** that:
1. Is unique to each training trajectory
2. Persists through fine-tuning
3. Can identify parent-child relationships between checkpoints

### 2.2 Centered Diagonal Fingerprint

For a model with $L$ layers, we extract the **centered diagonal fingerprint**:

$$\psi_\ell(M) = \frac{\text{diag}(M_\ell) - \bar{d}_\ell}{\|\text{diag}(M_\ell) - \bar{d}_\ell\|_2}$$

where:
- $\text{diag}(M_\ell) \in \mathbb{R}^d$ extracts the diagonal entries
- $\bar{d}_\ell = \frac{1}{d}\sum_i M_{\ell,ii}$ is the mean diagonal value
- The result is a unit vector in $\mathbb{R}^d$

**Centering rationale:** Removes the shared $-\varepsilon I$ component, isolating the checkpoint-specific residual.

### 2.3 Lineage Score

Given two models $A$ and $B$ with fingerprints $\{\psi_\ell^A\}$ and $\{\psi_\ell^B\}$, the **lineage score** is:

$$\mathcal{L}(A, B) = \frac{1}{L} \sum_{\ell=1}^{L} \langle \psi_\ell^A, \psi_\ell^B \rangle$$

This is the mean cosine similarity across aligned layers.

**Properties:**
- $\mathcal{L}(A, A) = 1$ (self-similarity)
- $\mathcal{L}(A, B) \approx 0$ for unrelated models
- $\mathcal{L}(A, B) \approx 1$ for fine-tuned descendants

### 2.4 Baseline: Full Branch Product Fingerprint

For comparison, we define the **full fingerprint**:

$$\phi_\ell(M) = \frac{\text{vec}(M_\ell)}{\|\text{vec}(M_\ell)\|_2} \in \mathbb{R}^{d^2}$$

This uses all $d^2$ entries of the branch product, not just the $d$ diagonal entries.

---

## 3. Experiments

### 3.1 Experimental Setup

**Base model:** GPT-2 (124M parameters, 12 layers, $d=768$)

**Lineage chain creation:**
1. Start with pretrained GPT-2 as $C_1$
2. Fine-tune for $N$ steps to produce $C_2$
3. Fine-tune $C_2$ for $N$ steps to produce $C_3$
4. Continue to produce $C_4$, $C_5$, etc.

**Perturbations tested:**
- Gaussian noise: $W \leftarrow W + \sigma \cdot \text{std}(W) \cdot \mathcal{N}(0, I)$
- Quantization: Round weights to $k$-bit precision

### 3.2 Experiment 1: Lineage Chain Tracing

**Setup:** Chain of 5 checkpoints, each fine-tuned 100 steps from the previous.

**Result: Similarity Decay with Lineage Distance**

| Pair | Diagonal Similarity | Full Similarity |
|------|---------------------|-----------------|
| $C_1 \leftrightarrow C_1$ | 1.0000 | 1.0000 |
| $C_1 \leftrightarrow C_2$ | 0.9997 | 0.9997 |
| $C_1 \leftrightarrow C_3$ | 0.9994 | 0.9995 |
| $C_1 \leftrightarrow C_4$ | 0.9989 | 0.9989 |
| $C_1 \leftrightarrow C_5$ | 0.9986 | 0.9987 |

**Finding:** Both methods show monotonic decay with lineage distance. Similarity decreases as checkpoints diverge further from the ancestor.

### 3.3 Experiment 2: Branching Lineage Tree

**Setup:** Two branches fine-tuned on different domains from the same base.

```
           C1 (GPT-2 base)
          /              \
       C2a                C2b
   (healthcare)        (finance)
       |                  |
      C3a                C3b
```

**Task:** Given $C_{3a}$, identify its parent as $C_{2a}$ (not $C_{2b}$).

**Results:**

| Test | Diagonal | Full |
|------|----------|------|
| $C_{3a}$ parent detection | Correct | Correct |
| $C_{3b}$ parent detection | Correct | Correct |
| Branch detection accuracy | 100% | 100% |

**Discrimination Margin** (higher = better separation):

| Child | Diagonal Margin | Full Margin |
|-------|-----------------|-------------|
| $C_{3a}$ | 0.000071 | 0.000504 |
| $C_{3b}$ | 0.000051 | 0.000378 |
| **Mean** | **0.000061** | **0.000441** |

**Finding:** Full fingerprints provide 7x larger discrimination margins for clean checkpoints.

### 3.4 Experiment 3: Robustness Under Perturbation

**Setup:** Compare fingerprint preservation when original model is perturbed.

**Noise Robustness** (similarity between original and noisy model):

| Noise Level ($\sigma$) | Diagonal | Full | Winner |
|------------------------|----------|------|--------|
| 0.01 | 0.9999 | 0.9999 | Tie |
| 0.10 | 0.9986 | 0.9908 | **Diagonal** |
| 0.30 | 0.9867 | 0.9237 | **Diagonal** |
| 0.50 | 0.9622 | 0.8153 | **Diagonal** |
| 1.00 | 0.8530 | 0.5298 | **Diagonal** |

**Quantization Robustness** (similarity between original and quantized model):

| Bits | Diagonal | Full | Winner |
|------|----------|------|--------|
| 16 | 1.0000 | 1.0000 | Tie |
| 8 | 0.9988 | 0.9897 | **Diagonal** |
| 6 | 0.9433 | 0.8140 | **Diagonal** |
| 4 | 0.4505 | 0.2322 | **Diagonal** |
| 2 | 0.0874 | 0.0055 | **Diagonal** |

**Finding:** Diagonal fingerprints are significantly more robust to perturbation. At $\sigma=1.0$ noise, diagonal preserves 85% similarity vs. full's 53%. At 4-bit quantization, diagonal preserves 45% vs. full's 23%.

---

## 4. Summary of Findings

### 4.1 Key Results

| Property | Diagonal Fingerprint | Full Fingerprint |
|----------|---------------------|------------------|
| Dimensions | $d$ (768) | $d^2$ (589,824) |
| Lineage chain ordering | Correct | Correct |
| Branch detection | 100% | 100% |
| Discrimination margin | Lower | **Higher (7x)** |
| Noise robustness ($\sigma=0.5$) | **0.962** | 0.815 |
| Quantization robustness (4-bit) | **0.450** | 0.232 |
| Compute cost | $O(Ld)$ | $O(Ld^2)$ |

### 4.2 Interpretation

1. **Diagonal dominance is real:** Training amplifies the DD score by 135x in GPT-2, confirming the theoretical prediction that branch products converge toward $-\varepsilon I$.

2. **Both methods detect lineage:** When checkpoints are unperturbed, both diagonal and full fingerprints successfully trace lineage chains and identify branching relationships.

3. **Diagonal wins on robustness:** The diagonal captures the stable structural signal induced by training. Off-diagonal entries, while informative, are more susceptible to noise and quantization.

4. **Full wins on discrimination:** For distinguishing between similar clean checkpoints, full fingerprints provide larger margins because they use more information.

### 4.3 Practical Recommendations

| Use Case | Recommended Method |
|----------|-------------------|
| Lineage detection with quantized models | **Diagonal** |
| Lineage detection with noisy/attacked weights | **Diagonal** |
| Fine-grained checkpoint discrimination | **Full** |
| Resource-constrained settings | **Diagonal** (768x fewer dimensions) |

---

## 5. Formulas Reference

### Definitions

| Symbol | Definition |
|--------|------------|
| $M_\ell = W_\ell^{\text{out}} W_\ell^{\text{in}}$ | Branch product at layer $\ell$ |
| $s(M) = \frac{|\text{tr}(M)|}{\|M\|_F}$ | Diagonal dominance score |
| $\psi_\ell = \frac{\text{diag}(M_\ell) - \bar{d}_\ell}{\|\text{diag}(M_\ell) - \bar{d}_\ell\|}$ | Centered diagonal fingerprint |
| $\phi_\ell = \frac{\text{vec}(M_\ell)}{\|\text{vec}(M_\ell)\|}$ | Full branch product fingerprint |
| $\mathcal{L}(A,B) = \frac{1}{L}\sum_\ell \langle \psi_\ell^A, \psi_\ell^B \rangle$ | Lineage score |

### Key Equations

**Diagonal dominance emergence:**
$$M_\ell \approx -\varepsilon_\ell I + E_\ell \quad \text{where } \|E_\ell\|_F \ll |\varepsilon_\ell| \sqrt{d}$$

**Expected DD score:**
- Random matrix: $\mathbb{E}[s(M)] \approx 1/\sqrt{d}$
- Trained model: $s(M) \approx O(\sqrt{d})$

**Lineage similarity decay:**
$$\mathcal{L}(C_1, C_k) < \mathcal{L}(C_1, C_j) \quad \text{for } k > j$$

---

## 6. Experimental Code

The experiments are implemented in:
- `experiments/gpt2_parent_retrieval.py` -- Parent retrieval on GPT-2 family
- `experiments/parent_retrieval_poc.py` -- Synthetic model experiments
- `case_studies/model_lineage/` -- Results and figures

Key functions:
```python
def centered_diag_fingerprint(M):
    d = np.diag(M)
    d_centered = d - d.mean()
    return d_centered / np.linalg.norm(d_centered)

def lineage_score(model_a, model_b):
    fps_a = [centered_diag_fingerprint(M) for M in extract_branch_products(model_a)]
    fps_b = [centered_diag_fingerprint(M) for M in extract_branch_products(model_b)]
    return np.mean([np.dot(a, b) for a, b in zip(fps_a, fps_b)])
```
