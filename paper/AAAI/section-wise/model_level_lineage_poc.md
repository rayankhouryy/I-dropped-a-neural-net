# Proof-of-Concept Experiment: Model-Level Lineage from Diagonal-Dominance Fingerprints

## Core correction

The scalar diagonal-dominance score alone is not sufficient for model-level lineage.

A high diagonal-dominance score can show that a model is a trained residual model, but not necessarily that it descends from a particular reference checkpoint. An independently trained ResNet may also have high diagonal dominance.

The stronger lineage claim is:

> Descendant checkpoints preserve not only the existence of diagonal dominance, but also the branch-specific residual pattern around the diagonal-dominant component.

This proof-of-concept experiment tests that stronger claim.

---

# 1. Goal

Given a reference checkpoint \(A\) and a suspect checkpoint \(B\), decide whether \(B\) descends from \(A\) using only model weights.

The experiment extends the within-model block-matching result to a between-model lineage-detection task.

---

# 2. Experimental Setup

## 2.1 Reference model

Start with a small, fast setting.

Recommended first option:

\[
A = \text{ResNet-18 trained on CIFAR-10}.
\]

Alternative faster option:

\[
A = \text{depth-24 residual MLP trained on synthetic data or CIFAR features}.
\]

## 2.2 Descendant suspects

Create positive examples from \(A\):

\[
B^+ \in \mathcal{D}(A).
\]

Examples:

- fine-tuned \(A\);
- quantized \(A\);
- pruned \(A\);
- noisy \(A\);
- LoRA/adapters merged into \(A\);
- partially fine-tuned \(A\).

## 2.3 Non-descendant suspects

Create negative controls:

\[
B^- \notin \mathcal{D}(A).
\]

Examples:

- same architecture, same dataset, different seed;
- same architecture, different dataset;
- random initialization;
- distilled student from \(A\);
- independently trained model matched in accuracy.

This is critical. Without strong non-descendant controls, the result only shows trained-model structure, not lineage.

---

# 3. Step 1: Compute Residual Branch Products

For each residual branch \(\ell \in \{1,\dots,L\}\), compute the full branch product:

\[
M_\ell^{A} = W_{\ell,K}^{A} \cdots W_{\ell,1}^{A},
\qquad
M_m^{B} = W_{m,K}^{B} \cdots W_{m,1}^{B}.
\]

For a two-layer branch:

\[
M_\ell = W_{\mathrm{out},\ell} W_{\mathrm{in},\ell}.
\]

For bottleneck branches:

\[
M_\ell = W_{3,\ell} W_{2,\ell} W_{1,\ell}.
\]

The key principle is architecture-aware factorization: compute the fingerprint on the complete residual branch product, not merely on endpoint matrices.

---

# 4. Step 2: Remove the Generic Identity-Like Component

The diagonal-dominant component is partly generic across trained residual models. The useful lineage signal should come from the branch-specific residual pattern after removing this generic identity-like component.

For each branch product \(M_\ell\), define:

\[
\alpha_\ell^\star
=
\arg\min_{\alpha \in \mathbb{R}}
\left\|M_\ell - \alpha I\right\|_F
=
\frac{\operatorname{tr}(M_\ell)}{d}.
\]

Then define the residual signature:

\[
R_\ell = M_\ell - \alpha_\ell^\star I.
\]

Interpretation:

- \(\alpha_\ell^\star I\) is the generic diagonal-dominant component;
- \(R_\ell\) is the branch-specific residual signature;
- \(R_\ell\) is what should be preserved by descendants and differ for independently trained models.

---

# 5. Step 3: Define Branch-Level Lineage Similarity

Normalize the residual signature:

\[
\phi(M_\ell)
=
\frac{\operatorname{vec}(R_\ell)}
{\left\|\operatorname{vec}(R_\ell)\right\|_2 + \delta}.
\]

Then define the branch similarity between reference branch \(\ell\) and suspect branch \(m\):

\[
C_{\ell m}(A,B)
=
\left\langle
\phi(M_\ell^A),
\phi(M_m^B)
\right\rangle.
\]

This is cosine similarity of branch-specific residual signatures after removing the generic identity-like component.

---

# 6. Step 4: Gate Similarity by Diagonal Dominance

The branch similarity should be trusted only when both branch products exhibit a valid diagonal-dominance fingerprint.

The diagonal-dominance score is:

\[
s(M) = \frac{|\operatorname{tr}(M)|}{\|M\|_F}.
\]

Define a gated branch score:

\[
G_{\ell m}(A,B)
=
C_{\ell m}(A,B)
\cdot
\min
\left(
\frac{s(M_\ell^A)}{\tau_s},
\frac{s(M_m^B)}{\tau_s},
1
\right).
\]

Here, \(\tau_s\) is a validation threshold chosen from trained reference models or held-out descendants.

Interpretation:

- \(C_{\ell m}(A,B)\) asks whether the branches share the same residual signature;
- the gate asks whether both products are valid diagonal-dominance fingerprints.

---

# 7. Step 5: Align Branches Across Checkpoints

Compute the cross-checkpoint score matrix:

\[
G(A,B) \in \mathbb{R}^{L \times L}.
\]

Then solve the assignment problem:

\[
\hat{\pi}
=
\arg\max_{\pi \in S_L}
\sum_{\ell=1}^{L}
G_{\ell,\pi(\ell)}(A,B).
\]

This can be implemented using Hungarian matching on \(-G\).

---

# 8. Step 6: Define Model-Level Lineage Score

Define the checkpoint-level ancestry score:

\[
\mathcal{L}(A,B)
=
\frac{1}{L}
\sum_{\ell=1}^{L}
G_{\ell,\hat{\pi}(\ell)}(A,B).
\]

This is the main lineage score.

Interpretation:

- high \(\mathcal{L}(A,B)\): suspect checkpoint preserves branch-specific structure from reference checkpoint;
- low \(\mathcal{L}(A,B)\): suspect checkpoint does not preserve reference-specific weight lineage.

---

# 9. Main Formula Block for the Paper

```latex
\[
\alpha_\ell^\star
=
\arg\min_{\alpha \in \mathbb{R}}
\left\|M_\ell - \alpha I\right\|_F
=
\frac{\operatorname{tr}(M_\ell)}{d},
\qquad
R_\ell = M_\ell - \alpha_\ell^\star I.
\]

\[
\phi(M_\ell)
=
\frac{\operatorname{vec}(R_\ell)}
{\left\|\operatorname{vec}(R_\ell)\right\|_2 + \delta}.
\]

\[
C_{\ell m}(A,B)
=
\left\langle
\phi(M_\ell^A),
\phi(M_m^B)
\right\rangle.
\]

\[
s(M) = \frac{|\operatorname{tr}(M)|}{\|M\|_F}.
\]

\[
G_{\ell m}(A,B)
=
C_{\ell m}(A,B)
\cdot
\min
\left(
\frac{s(M_\ell^A)}{\tau_s},
\frac{s(M_m^B)}{\tau_s},
1
\right).
\]

\[
\hat{\pi}
=
\arg\max_{\pi \in S_L}
\sum_{\ell=1}^{L}
G_{\ell,\pi(\ell)}(A,B).
\]

\[
\mathcal{L}(A,B)
=
\frac{1}{L}
\sum_{\ell=1}^{L}
G_{\ell,\hat{\pi}(\ell)}(A,B).
\]
```

---

# 10. Step 7: Calibrate Against a Null Distribution

For each reference checkpoint \(A\), build a null distribution using independently trained non-descendants:

\[
\mathcal{N}_A
=
\left\{
\mathcal{L}(A,B_j^-)
:
B_j^- \notin \mathcal{D}(A)
\right\}_{j=1}^{n}.
\]

Convert the raw lineage score into a normalized lineage z-score:

\[
Z(A,B)
=
\frac{
\mathcal{L}(A,B) - \mu(\mathcal{N}_A)
}{
\sigma(\mathcal{N}_A) + \delta
}.
\]

Decision rule:

\[
B \text{ descends from } A
\quad \Longleftrightarrow \quad
Z(A,B) > \tau_Z.
\]

LaTeX block:

```latex
\[
\mathcal{N}_A
=
\left\{
\mathcal{L}(A,B_j^-)
:
B_j^- \notin \mathcal{D}(A)
\right\}_{j=1}^{n}.
\]

\[
Z(A,B)
=
\frac{
\mathcal{L}(A,B)-\mu(\mathcal{N}_A)
}{
\sigma(\mathcal{N}_A)+\delta
}.
\]

\[
B \in \mathcal{D}(A)
\iff
Z(A,B) > \tau_Z.
\]
```

---

# 11. Minimal POC Matrix

Start with a small but convincing experiment.

| Category | Models |
|---|---|
| Reference | 5 independently trained ResNet-18 checkpoints |
| Descendants | 5 fine-tuned, 5 quantized, 5 pruned, 5 noisy, 5 LoRA/adapted |
| Non-descendants | 20 same-architecture different-seed, 10 random-init, 10 distilled, 10 different-dataset |

For every pair \((A,B)\), compute:

\[
\mathcal{L}(A,B), \qquad Z(A,B).
\]

Define labels:

\[
y(A,B)=
\begin{cases}
1, & B \in \mathcal{D}(A),\\
0, & B \notin \mathcal{D}(A).
\end{cases}
\]

Compute:

- AUROC;
- AUPRC;
- TPR at 1% FPR;
- TPR at 0.1% FPR if enough negatives;
- false positives among same-architecture independently trained models.

---

# 12. Main Expected Figure

Create one figure with three panels.

## Panel A: Score Distributions

Plot distributions of \(Z(A,B)\) or \(\mathcal{L}(A,B)\):

- descendants;
- same-architecture different-seed models;
- random-initialized models;
- distilled models.

Expected result:

- descendants high;
- non-descendants low;
- distilled models near non-descendants.

## Panel B: ROC Curve

Plot descendant-versus-non-descendant detection using:

- proposed lineage score;
- Frobenius baseline;
- cosine baseline;
- diagonal-dominance-only ablation;
- activation or behavioral baseline if available.

## Panel C: Utility-Fingerprint Tradeoff

Plot:

- x-axis: model utility degradation;
- y-axis: lineage score or z-score.

Expected result:

- mild transformations preserve the fingerprint;
- fingerprint collapse occurs only when task utility also degrades substantially.

---

# 13. Baselines

## 13.1 Frobenius Similarity

\[
\mathrm{FrobSim}(A,B)
=
-\frac{1}{L}
\sum_{\ell=1}^L
\left\|
M_\ell^A - M_\ell^B
\right\|_F.
\]

## 13.2 Cosine Similarity

\[
\mathrm{CosSim}(A,B)
=
\frac{1}{L}
\sum_{\ell=1}^L
\frac{
\left\langle \operatorname{vec}(M_\ell^A), \operatorname{vec}(M_\ell^B)\right\rangle
}{
\|M_\ell^A\|_F \|M_\ell^B\|_F
}.
\]

## 13.3 Diagonal-Dominance-Only Ablation

\[
\mathrm{DiagOnly}(B)
=
\frac{1}{L}
\sum_{\ell=1}^L
s(M_\ell^B).
\]

This baseline should usually fail for lineage because it measures whether \(B\) is a trained residual model, not whether it descends from \(A\).

## 13.4 Optional Baselines

Add if feasible:

- weight cosine over raw layer weights;
- CKA/SVCCA activation similarity;
- behavioral agreement on held-out data;
- spectral-statistic fingerprint;
- direct layer-wise Frobenius matching.

---

# 14. What Would Count as a Convincing POC?

| Metric | Strong POC Target |
|---|---:|
| Descendant vs non-descendant AUROC | \(>0.95\) |
| TPR at 1% FPR | \(>90\%\) |
| Same-architecture/different-seed false positives | Near 0 |
| Distilled models | Near non-descendant distribution |
| Quantized/pruned/fine-tuned descendants | High lineage score |
| Utility tradeoff | Fingerprint collapse only after major accuracy/perplexity degradation |

---

# 15. Why This Experiment Matters

Pair accuracy answers:

> Does this model contain diagonal-dominant residual structure?

Model-level lineage answers:

> Does this suspect checkpoint preserve branch-specific structure from this reference checkpoint?

That distinction is crucial.

A trained but unrelated residual network may have diagonal dominance. A descendant should preserve reference-specific residual signatures. This POC tests the latter and turns the diagonal-dominance phenomenon into a real model-provenance result.

---

# 16. Minimal Implementation Pseudocode

```python
def branch_product(block):
    # Architecture-specific.
    # Two-layer branch:
    # return W_out @ W_in
    # Bottleneck branch:
    # return W3 @ W2 @ W1
    raise NotImplementedError


def residual_signature(M, eps=1e-12):
    d = M.shape[0]
    alpha = torch.trace(M) / d
    R = M - alpha * torch.eye(d, device=M.device, dtype=M.dtype)
    phi = R.flatten() / (torch.norm(R.flatten()) + eps)
    return phi


def diag_score(M, eps=1e-12):
    return torch.abs(torch.trace(M)) / (torch.norm(M, p="fro") + eps)


def gated_branch_score(M_a, M_b, tau_s, eps=1e-12):
    phi_a = residual_signature(M_a, eps)
    phi_b = residual_signature(M_b, eps)
    C = torch.dot(phi_a, phi_b)

    gate = min(
        float(diag_score(M_a, eps) / tau_s),
        float(diag_score(M_b, eps) / tau_s),
        1.0,
    )
    return C * gate


def lineage_score(model_a, model_b, tau_s):
    M_a = [branch_product(block) for block in model_a.residual_blocks]
    M_b = [branch_product(block) for block in model_b.residual_blocks]

    L = len(M_a)
    G = torch.zeros((L, L))

    for i in range(L):
        for j in range(L):
            G[i, j] = gated_branch_score(M_a[i], M_b[j], tau_s)

    # Hungarian solves min-cost, so use -G.
    row_ind, col_ind = scipy.optimize.linear_sum_assignment((-G).cpu().numpy())

    return G[row_ind, col_ind].mean().item()
```

---

# 17. Suggested Paper Wording

Use this phrasing in the methodology section:

> Diagonal dominance alone indicates that a residual branch has acquired a trained identity-like structure, but this is not sufficient for checkpoint-level lineage: independently trained residual networks may exhibit the same generic property. We therefore remove the best-fitting identity component from each branch product and compare the residual signatures that remain. These signatures encode branch-specific deviations from the generic diagonal-dominant form. We align residual branches across checkpoints with Hungarian matching and aggregate the matched similarities into a checkpoint-level lineage score calibrated against independently trained non-descendants.

This wording makes the method scientifically honest and much harder to attack.
