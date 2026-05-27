# POC 2: Parent Retrieval via Diagonal Profile Fingerprints

## Core idea

This proof-of-concept tests **model-level lineage** through a parent-retrieval task.

The previous lineage POC compared the **off-identity residual signature**

\[
R_\ell = M_\ell - \alpha I.
\]

This POC is different: it uses the **centered diagonal profile** of each trained residual branch product and asks whether a modified suspect checkpoint can be matched back to its exact source checkpoint among many independently trained candidates.

The task is no longer:

> Does this model exhibit diagonal dominance?

The task is:

> Given many possible reference checkpoints, can we retrieve the exact ancestor of a modified suspect checkpoint?

That is a direct model-level provenance experiment.

---

# 1. Research Question

Given a gallery of independently trained reference models

\[
\mathcal{A}=\{A_1,\dots,A_R\},
\]

and a suspect checkpoint \(B\), can we identify which reference model \(A_r\) produced \(B\)?

This is stronger than binary descendant detection because the negatives are same-architecture, same-dataset, independently trained models.

---

# 2. Key Hypothesis

Diagonal dominance itself may be generic across trained residual models. However, the **layerwise diagonal profile** of each residual branch product may contain checkpoint-specific structure.

For a residual branch product

\[
M_\ell = W_{\ell,K}\cdots W_{\ell,1},
\]

extract the diagonal vector:

\[
d_\ell = \operatorname{diag}(M_\ell).
\]

Remove the generic mean identity component:

\[
\bar d_\ell = \frac{1}{d}\mathbf{1}^{\top}d_\ell,
\qquad
\tilde d_\ell = d_\ell - \bar d_\ell \mathbf{1}.
\]

Normalize:

\[
\psi(M_\ell)
=
\frac{\tilde d_\ell}
{\|\tilde d_\ell\|_2+\delta}.
\]

Here, \(\psi(M_\ell)\) is the **centered diagonal fingerprint** of branch \(\ell\).

---

# 3. Branch-Level Similarity

For reference model \(A\) and suspect model \(B\), compare branch \(\ell\) in \(A\) to branch \(m\) in \(B\):

\[
C_{\ell m}^{\mathrm{diag}}(A,B)
=
\left\langle
\psi(M_\ell^A),
\psi(M_m^B)
\right\rangle.
\]

This is cosine similarity between centered diagonal fingerprints.

Optionally, gate the similarity by diagonal-dominance validity:

\[
s(M)=\frac{|\operatorname{tr}(M)|}{\|M\|_F}.
\]

\[
G_{\ell m}^{\mathrm{diag}}(A,B)
=
C_{\ell m}^{\mathrm{diag}}(A,B)
\cdot
\min\left(
\frac{s(M_\ell^A)}{\tau_s},
\frac{s(M_m^B)}{\tau_s},
1
\right).
\]

Interpretation:

- \(C_{\ell m}^{\mathrm{diag}}\) asks whether the branches have similar centered diagonal fingerprints.
- The gate asks whether both branches actually exhibit a valid diagonal-dominance signal.

---

# 4. Model-Level Lineage Score

## 4.1 If layer order is preserved

Use direct alignment:

\[
\mathcal{L}_{\mathrm{diag}}(A,B)
=
\frac{1}{L}
\sum_{\ell=1}^{L}
G_{\ell\ell}^{\mathrm{diag}}(A,B).
\]

## 4.2 If layer order may be permuted

Use Hungarian matching:

\[
\hat{\pi}
=
\arg\max_{\pi\in S_L}
\sum_{\ell=1}^{L}
G_{\ell,\pi(\ell)}^{\mathrm{diag}}(A,B),
\]

\[
\mathcal{L}_{\mathrm{diag}}(A,B)
=
\frac{1}{L}
\sum_{\ell=1}^{L}
G_{\ell,\hat{\pi}(\ell)}^{\mathrm{diag}}(A,B).
\]

This gives a checkpoint-level similarity score between reference \(A\) and suspect \(B\).

---

# 5. Parent Retrieval Task

Train \(R\) independent reference models:

\[
\mathcal{A}=\{A_1,\dots,A_R\}.
\]

For each reference \(A_r\), generate descendants:

\[
B_{r,t}^{+}=T_t(A_r),
\]

where \(T_t\) is a post-training transformation such as:

- fine-tuning;
- quantization;
- pruning;
- weight noise;
- LoRA/adapters merged into the base model;
- partial layer fine-tuning.

Given a suspect \(B\), retrieve its parent by:

\[
\hat r(B)
=
\arg\max_{r\in\{1,\dots,R\}}
\mathcal{L}_{\mathrm{diag}}(A_r,B).
\]

Success criterion:

\[
\hat r(B)=r^\star,
\]

where \(r^\star\) is the true parent.

---

# 6. Metrics

## 6.1 Top-1 parent retrieval

\[
\mathrm{Top\text{-}1}
=
\frac{1}{|\mathcal{B}|}
\sum_{B\in\mathcal{B}}
\mathbf{1}[\hat r(B)=r^\star].
\]

## 6.2 Mean reciprocal rank

\[
\mathrm{MRR}
=
\frac{1}{|\mathcal{B}|}
\sum_{B\in\mathcal{B}}
\frac{1}{\operatorname{rank}(r^\star)}.
\]

## 6.3 Retrieval margin

For open-world detection, define:

\[
\mathrm{margin}(B)
=
\max_r \mathcal{L}_{\mathrm{diag}}(A_r,B)
-
\max_{r\neq \hat r}
\mathcal{L}_{\mathrm{diag}}(A_r,B).
\]

A large positive margin means the suspect strongly matches one reference checkpoint over all alternatives.

## 6.4 Open-world descendant detection

Use unknown non-descendants to test rejection.

Report:

- AUROC;
- AUPRC;
- TPR at 1% FPR;
- TPR at 0.1% FPR if enough negatives are available;
- false positive rate among same-architecture independently trained models.

---

# 7. Minimal Experiment

Use **ResNet-18 on CIFAR-10** first.

| Component | Count |
|---|---:|
| Independent reference models | 10--20 |
| Descendants per reference | 5--10 |
| Same-architecture non-descendants | 20 |
| Random-init controls | 10 |
| Distilled students | 5--10 |

For each suspect checkpoint, compute \(\mathcal{L}_{\mathrm{diag}}(A_r,B)\) against every reference \(A_r\).

Expected output table:

| Suspect type | Top-1 parent retrieval | MRR | AUROC descendant detection |
|---|---:|---:|---:|
| Fine-tuned descendant | X% | X | X |
| Quantized descendant | X% | X | X |
| Pruned descendant | X% | X | X |
| Noisy descendant | X% | X | X |
| LoRA-merged descendant | X% | X | X |
| Independent same-architecture / different-seed | near chance | low | low |
| Distilled student | near non-descendant | low | low |

---

# 8. Main Figure

Create one figure with three panels.

## Panel A: Parent-retrieval heatmap

Rows: suspect checkpoints.  
Columns: reference checkpoints.  
Cell value:

\[
\mathcal{L}_{\mathrm{diag}}(A_r,B).
\]

Expected result:

> Descendants light up only under their true parent.

## Panel B: Score distributions

Plot distributions for:

- true parent scores;
- wrong parent scores;
- independent model scores;
- distilled student scores.

Expected result:

> True-parent scores separate from wrong-parent and non-descendant scores.

## Panel C: Robustness curve

For each transformation strength, plot:

- x-axis: modification severity;
- y-axis: Top-1 parent retrieval or lineage score;
- overlay model accuracy/perplexity.

Expected result:

> Parent retrieval remains strong under mild-to-moderate post-training transformations and degrades only when model utility meaningfully degrades.

---

# 9. Baselines

Compare against simple and strong alternatives.

## 9.1 Frobenius similarity

\[
\mathrm{FrobSim}(A,B)
=
-
\frac{1}{L}
\sum_{\ell=1}^{L}
\left\|
M_\ell^A-M_\ell^B
\right\|_F.
\]

## 9.2 Raw branch-product cosine similarity

\[
\mathrm{CosSim}(A,B)
=
\frac{1}{L}
\sum_{\ell=1}^{L}
\frac{
\left\langle \operatorname{vec}(M_\ell^A),\operatorname{vec}(M_\ell^B)\right\rangle
}{
\|M_\ell^A\|_F\|M_\ell^B\|_F
}.
\]

## 9.3 Diagonal-dominance-only ablation

\[
\mathrm{DiagOnly}(B)
=
\frac{1}{L}
\sum_{\ell=1}^{L}
s(M_\ell^B).
\]

This baseline should fail for parent retrieval because it measures whether \(B\) is a trained residual model, not whether it descends from \(A\).

## 9.4 Raw diagonal profile without centering

\[
\psi_{\mathrm{raw}}(M_\ell)
=
\frac{\operatorname{diag}(M_\ell)}
{\|\operatorname{diag}(M_\ell)\|_2+\delta}.
\]

This ablation tests whether centering out the identity-like component matters.

## 9.5 Optional baselines

Add if feasible:

- raw weight cosine similarity;
- CKA/SVCCA activation similarity;
- behavioral agreement on held-out data;
- spectral-statistic fingerprint;
- direct layer-wise Frobenius matching.

---

# 10. What Would Count as a Convincing POC?

| Metric | Strong POC Target |
|---|---:|
| Top-1 parent retrieval for fine-tuned descendants | \(>90\%\) |
| MRR | \(>0.9\) |
| AUROC descendant detection | \(>0.95\) |
| Same-architecture/different-seed false positives | Near chance / near 0 |
| Distilled students | Near non-descendant distribution |
| Quantized/pruned/noisy descendants | High parent retrieval |
| Robustness curve | Retrieval degrades only with meaningful utility loss |

---

# 11. Why This POC Is Strong

This experiment directly supports model-level provenance.

It asks:

> Can a suspect checkpoint be linked back to its exact source checkpoint among many independently trained same-architecture models?

That is much stronger than showing that a trained model has diagonal dominance.

It also avoids relying only on the generic fact that trained residual models have diagonal dominance. The centered diagonal profile tests whether the model preserves **checkpoint-specific structure**, not merely trained-model structure.

---

# 12. Minimal Implementation Pseudocode

```python
import torch
import scipy.optimize


def branch_product(block):
    """Architecture-specific branch product.

    Two-layer residual branch:
        return W_out @ W_in

    Bottleneck residual branch:
        return W3 @ W2 @ W1
    """
    raise NotImplementedError


def diag_score(M, eps=1e-12):
    return torch.abs(torch.trace(M)) / (torch.norm(M, p="fro") + eps)


def centered_diag_fingerprint(M, eps=1e-12):
    d = torch.diag(M)
    d_centered = d - d.mean()
    return d_centered / (torch.norm(d_centered) + eps)


def branch_diag_similarity(M_a, M_b, tau_s=None, eps=1e-12):
    psi_a = centered_diag_fingerprint(M_a, eps)
    psi_b = centered_diag_fingerprint(M_b, eps)
    C = torch.dot(psi_a, psi_b)

    if tau_s is None:
        return C

    gate = min(
        float(diag_score(M_a, eps) / tau_s),
        float(diag_score(M_b, eps) / tau_s),
        1.0,
    )
    return C * gate


def lineage_score_diag(model_a, model_b, tau_s=None, use_hungarian=True):
    M_a = [branch_product(block) for block in model_a.residual_blocks]
    M_b = [branch_product(block) for block in model_b.residual_blocks]

    L = len(M_a)
    G = torch.zeros((L, L), device=M_a[0].device)

    for i in range(L):
        for j in range(L):
            G[i, j] = branch_diag_similarity(M_a[i], M_b[j], tau_s=tau_s)

    if use_hungarian:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment((-G).detach().cpu().numpy())
        return G[row_ind, col_ind].mean().item()

    return torch.diag(G).mean().item()


def retrieve_parent(references, suspect, tau_s=None):
    scores = [lineage_score_diag(ref, suspect, tau_s=tau_s) for ref in references]
    parent_idx = int(torch.tensor(scores).argmax().item())
    return parent_idx, scores
```

---

# 13. Suggested Paper Wording

Use this in Section 4 or Section 5:

> Diagonal dominance alone is a generic signature of trained residual branches and is therefore insufficient to identify a specific ancestor checkpoint. To test model-level lineage, we extract a branch-specific diagonal profile from each residual-branch product after subtracting its mean identity component. We then compare centered diagonal fingerprints across reference and suspect checkpoints, align branches using Hungarian matching when layer order is unknown, and aggregate the matched similarities into a checkpoint-level parent-retrieval score. This evaluates whether a modified suspect checkpoint can be linked to its exact source among independently trained same-architecture references.

---

# 14. How This Differs from the Previous Lineage POC

| Aspect | Previous POC | This POC |
|---|---|---|
| Signature | Off-identity residual matrix \(R_\ell=M_\ell-\alpha I\) | Centered diagonal profile \(\psi(M_\ell)\) |
| Signal type | Full residual-pattern similarity | Diagonal-profile similarity |
| Task | Binary descendant detection / lineage scoring | Parent retrieval among many references |
| Strength | Uses full matrix information | Simpler, more interpretable, strong retrieval framing |
| Main risk | May be too close to raw weight similarity | May discard useful off-diagonal lineage signal |

Both POCs are useful. If both work, the paper becomes stronger because model-level lineage is supported by two related but distinct fingerprint constructions.
