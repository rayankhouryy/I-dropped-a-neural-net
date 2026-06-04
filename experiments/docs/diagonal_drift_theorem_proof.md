# Theorem Proof: Coupled Gradient Updates Induce Diagonal Drift

This section gives a narrow theoretical result explaining why same-block coupled updates to the input and output projections of a residual branch can induce diagonal dominance in the branch product.

The result should not be framed as a universal proof for arbitrary deep networks. It is a sufficient-condition argument: under a linearized residual block and isotropy assumptions, coupled gradient descent induces a coherent negative-identity drift in the product

\[
M = W_{\text{out}} W_{\text{in}}.
\]

---

## Theoretical Statement

```latex
\subsection{Why Coupled Gradient Updates Induce Diagonal Drift}

We provide a simple mechanism showing why coupled updates to the input and
output projections of a residual branch can induce diagonal dominance in the
branch product. The result is not intended as a universal theorem for arbitrary
deep networks; rather, it identifies a sufficient condition under which the
observed fingerprint emerges.

Consider a linearized residual block
\[
    y = x + BAx,
\]
where \(x \in \mathbb{R}^d\), \(A \in \mathbb{R}^{m \times d}\),
\(B \in \mathbb{R}^{d \times m}\), and
\[
    M = BA \in \mathbb{R}^{d \times d}
\]
is the branch product whose diagonal-dominance score is
\[
    s(M) = \frac{|\operatorname{tr}(M)|}{\|M\|_F}.
\]
Let
\[
    g = \nabla_y \ell(y)
\]
denote the upstream gradient for one example, and define the population
cross-covariance
\[
    C = \mathbb{E}[g x^\top].
\]

For the population loss, the gradients of the two branch projections are
\[
    \nabla_B \mathcal{L}
    =
    \mathbb{E}[g(Ax)^\top]
    =
    \mathbb{E}[g x^\top] A^\top
    =
    CA^\top,
\]
and
\[
    \nabla_A \mathcal{L}
    =
    \mathbb{E}[B^\top g x^\top]
    =
    B^\top C.
\]
A gradient descent step with learning rate \(\eta\) gives
\[
    B^+ = B - \eta CA^\top,
    \qquad
    A^+ = A - \eta B^\top C.
\]
The updated product is
\[
    M^+ = B^+A^+.
\]
Expanding,
\[
\begin{aligned}
    M^+ - M
    &=
    (B - \eta CA^\top)(A - \eta B^\top C) - BA \\
    &=
    -\eta CA^\top A
    -\eta BB^\top C
    + \eta^2 CA^\top B^\top C.
\end{aligned}
\]
Thus, to first order in the learning rate,
\[
    \Delta M
    =
    M^+ - M
    \approx
    -\eta\left(CA^\top A + BB^\top C\right).
\]

We now state the isotropic-drift condition. Suppose that, at a given stage of
training,
\[
    C = \kappa I + E_C,
\]
\[
    A^\top A = \alpha I + E_A,
\]
and
\[
    BB^\top = \beta I + E_B,
\]
where \(\kappa > 0\), \(\alpha,\beta > 0\), and
\(E_C,E_A,E_B\) are residual anisotropy terms. Substituting into the first-order
update gives
\[
\begin{aligned}
    \Delta M
    &\approx
    -\eta
    \left[
        (\kappa I + E_C)(\alpha I + E_A)
        +
        (\beta I + E_B)(\kappa I + E_C)
    \right] \\
    &=
    -\eta\kappa(\alpha+\beta)I
    -
    \eta R,
\end{aligned}
\]
where
\[
    R
    =
    \kappa E_A
    + \alpha E_C
    + E_CE_A
    + \kappa E_B
    + \beta E_C
    + E_BE_C.
\]
Therefore, when the anisotropy residual \(R\) is small relative to the scalar
term, the expected product update contains a coherent negative-identity drift:
\[
    \Delta M
    \approx
    -\eta\kappa(\alpha+\beta)I.
\]

Taking traces,
\[
    \operatorname{tr}(\Delta M)
    \approx
    -\eta\kappa(\alpha+\beta)d,
\]
so the trace accumulates coherently across all \(d\) coordinates. After \(T\)
steps, ignoring higher-order terms,
\[
    M_T
    \approx
    M_0
    -
    \left(
        \sum_{t=0}^{T-1}
        \eta_t\kappa_t(\alpha_t+\beta_t)
    \right)I
    +
    N_T,
\]
where \(N_T\) collects anisotropic and stochastic residual terms. Let
\[
    \Lambda_T
    =
    \sum_{t=0}^{T-1}
    \eta_t\kappa_t(\alpha_t+\beta_t).
\]
Then
\[
    M_T \approx M_0 - \Lambda_T I + N_T.
\]
If the scalar component dominates the residual term, the diagonal-dominance
score satisfies
\[
    s(M_T)
    =
    \frac{|\operatorname{tr}(M_T)|}{\|M_T\|_F}
    \approx
    \frac{\Lambda_T d}{\Lambda_T \sqrt{d}}
    =
    \sqrt{d}.
\]
By contrast, for an untrained random product with no coherent scalar-identity
component, \(|\operatorname{tr}(M)|\) scales like the standard deviation of a
sum of independent diagonal terms, while \(\|M\|_F\) scales with all \(d^2\)
entries, yielding the random baseline
\[
    s(M) = O(d^{-1/2}).
\]
Thus, coupled gradient descent can move the product from the random baseline
toward a diagonal-dominant scalar-identity structure.

\paragraph{Effect of gradient shuffling.}
This derivation also predicts why shuffling one side of the update weakens, but
does not necessarily eliminate, the fingerprint. Suppose the \(B\)-gradient for
block \(i\) is replaced by the \(B\)-gradient from an independent block \(j\).
Then the \(B\)-side contribution becomes
\[
    -\eta C_j A_j^\top A_i.
\]
If different blocks are approximately independent and isotropic, then
\[
    \mathbb{E}[A_j^\top A_i] \approx 0
    \qquad
    \text{for } i \neq j,
\]
so this contribution loses its coherent identity drift. However, the \(A\)-side
term
\[
    -\eta B_iB_i^\top C_i
\]
remains coupled within block \(i\). Consequently, shuffling one side should
reduce the fingerprint substantially but not necessarily return it to the
random-initialization baseline. This matches the empirical observation that
gradient shuffling weakens the diagonal-dominance score while leaving a residual
signal.
```

---

## Reviewer-Safe Claim

Use this in the paper instead of claiming that gradient coupling universally proves diagonal dominance.

```latex
These calculations show that same-block gradient coupling is sufficient to
produce a coherent negative-identity drift in the residual-branch product under
standard isotropy assumptions. This does not prove that all diagonal dominance
in trained networks arises from this mechanism, but it gives a testable
prediction: the observed product update \(\Delta M\) should align with
\(-\eta(CA^\top A + BB^\top C)\), and its trace should be negative whenever
\(\operatorname{tr}(C)/d > 0\). We verify this prediction empirically by
measuring the cosine similarity and trace agreement between the predicted and
actual product updates during training.
```

---

## Plain-English Interpretation

The theorem says that if three things are approximately true during training:

1. The residual-stream gradient/input covariance is approximately scalar-identity:

\[
C = \mathbb{E}[gx^\top] \approx \kappa I
\]

2. The input projection is approximately isotropic:

\[
A^\top A \approx \alpha I
\]

3. The output projection is approximately isotropic:

\[
BB^\top \approx \beta I
\]

then coupled gradient descent adds the following drift to the product:

\[
\Delta M \approx -\eta\kappa(\alpha + \beta)I.
\]

That is exactly the diagonal-dominant, negative-trace structure observed empirically.

The theory also explains why shuffling one side of the update weakens but does not necessarily eliminate the signal: it destroys one source of coherent drift while leaving the other source partially intact.
