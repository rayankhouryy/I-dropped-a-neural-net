# Background — Discarded Content

Content removed from original ~793 words to achieve ~300 word target.

---

## Full "Passive Model Provenance" Subsection

Original text (removed, condensed to 2 sentences):
> We define \emph{passive model provenance} as follows: given a reference checkpoint $R$ and a suspect checkpoint $S$ with missing or untrusted metadata, determine whether $S$ shares weight-level lineage with $R$ after allowed post-training transformations. This is a \emph{white-box} verification task---the verifier has access to both sets of weights. The setting matches regulatory audits, model hub compliance checks, and ownership disputes where both parties can produce their checkpoints.
>
> The method detects \emph{weight-level lineage}, not behavioral equivalence. Two models share weight-level lineage if the suspect's parameters can be derived from the reference through transformations such as fine-tuning, pruning, or quantization---without retraining from scratch. This is distinct from functional equivalence, where models produce similar outputs regardless of whether their weights share any relationship. Model extraction attacks create functional copies with entirely fresh weights; such models are behaviorally similar but share no weight-level identity.
>
> Distilled models and independently trained functional copies are explicitly outside the positive guarantee. A student trained via knowledge distillation may replicate the teacher's behavior with high fidelity, but it evolves its weights through a separate optimization process that produces distinct structural fingerprints. This boundary is intentional: proving that weights were copied or derived requires distinguishing ``same functionality'' from ``same parameters (up to transformation).''

---

## Full "Limits of Existing Methods" Subsection

Original text (replaced by comparison table):
> Model provenance verification has relied on three families of approaches, each with fundamental limitations.
>
> \textbf{Cryptographic hashes} fail immediately upon any weight modification. The avalanche effect ensures that fine-tuning, pruning, or even numerical precision changes produce entirely different digests, providing no signal of derivation.
>
> \textbf{Metadata and logging systems}---model cards, experiment trackers like MLflow and Weights \& Biases---require trusted provenance records that may be incomplete, falsified, or absent when models are redistributed outside controlled environments.
>
> \textbf{Watermarking methods} embed ownership signals into weights or train models to produce specific outputs on trigger inputs. Both require forethought: the watermark must be inserted before training concludes, precluding retroactive verification of already-deployed models. A systematic evaluation found that no surveyed watermarking scheme is robust in practice against fine-tuning, pruning, and model extraction attacks.
>
> \textbf{Output watermarking} systems such as SynthID identify AI-generated content by embedding statistical signatures during generation, but address a fundamentally different question: they detect whether content was AI-generated, not whether one model's weights derive from another's.
>
> \textbf{Decision-boundary fingerprinting} identifies functional similarity via adversarial examples near classification boundaries, but cannot distinguish weight lineage: a distilled student may share decision surfaces while having entirely independent weights.
>
> The key differentiator of our approach: prior watermarking work asks how to \emph{insert or protect} an ownership signal. We ask whether training already \emph{leaves} a signal that can be read retroactively.

---

## Extended Residual Branch Products Discussion

Original text (condensed):
> Modern deep networks use \emph{residual connections}---shortcuts that allow information to bypass layers---to enable stable training of very deep architectures. A residual block computes $x' = x + F(x)$, where $x$ is the input and $F$ is the residual branch function. For these blocks to maintain stable gradient flow, the Jacobian $J = I + J_F$ must remain close to orthogonal throughout training---a condition formalized as \emph{dynamical isometry}.
>
> This product captures how the branch transforms the input space. In the two-factor residual-MLP regime, the dynamical isometry constraint admits a clean characterization: $M$ approximates a negative scaled identity, $M \approx -\varepsilon I$ for small $\varepsilon > 0$. We use this as the working model for our margin analysis; the broader empirical observation that trained branch products exhibit trace/Frobenius concentration extends to multi-factor and attention-style products even when the strict negative-identity geometry does not hold.

---

## Verbose Table (Residual Form column removed)

Original table had 3 columns including "Residual Form" with equations like `$x + W_2 \phi(W_1 x)$`. Simplified to 2-column format showing only Architecture and Product.
