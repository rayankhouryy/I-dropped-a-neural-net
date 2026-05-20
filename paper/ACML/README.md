# Training Leaves Traces: Diagonal Dominance as a Neural Network Fingerprint

**ACML 2026 Submission**

## Abstract

Verifying the provenance of neural network weights is difficult: existing watermarking schemes must be embedded during training, and can be removed by fine-tuning. We show that training itself leaves an intrinsic fingerprint requiring no such foresight. Residual networks initialized for dynamical isometry develop a distinctive structure: after training, each block's weight product settles near negative identity, consistent with the network maintaining stable gradient flow by keeping layer Jacobians close to orthogonal. This leaves a detectable trace: the diagonal-dominance score of correctly paired input and output weights is high, while incorrect pairings score near zero. We show this signal reliably identifies which weights belong to the same block in residual networks and transformers, even when layers are shuffled or the model is subsequently fine-tuned. Because the fingerprint emerges from training rather than being deliberately embedded, it applies retroactively to any already-deployed model. This enables post-hoc verification—reconstructing models from corrupted or unlabeled weights, confirming a deployed model matches a registered checkpoint, or detecting unauthorized derivatives.

We formalize the diagonal-dominance fingerprint and stress-test its limits. A closed-form decomposition shows that the gap between correct and incorrect pairs scales as O(√d) with matrix dimension d. Benchmarking against alternatives, diagonal-dominance is the only metric achieving perfect layer recovery; Frobenius matching plateaus at 47–67% on transformers, while singular-value distance performs at chance. The signal transfers across architectures: 100% accuracy on GPT-2 models from 124M to 1.5B parameters across MLP and attention paths, with signal strength increasing at larger model sizes. Vision Transformers (ViT-B/16) achieve 100% accuracy on all paths, with attention showing the strongest separation. ImageNet ResNets with BatchNorm achieve 91–100% accuracy using architecture-aware factorization. Modern ConvNets (ConvNeXt-T) also achieve 100% accuracy. Robustness holds under adversarial conditions: pairing accuracy stays at 100% across 21 attack configurations including fine-tuning up to 50 epochs and weight noise, degrading only when perturbations exceed ~20% relative weight magnitude—at which point classification accuracy drops below 50%.

---

## Key Results

### Training Induces Diagonal Dominance

![Training Dynamics](../../figures/fig_null_a_heatmaps.pdf)

*At initialization (epoch 0), no diagonal structure is visible. By epoch 5, the diagonal is fully separated and pair accuracy reaches 100%.*

### Works Across Modern Architectures

![Modern Vision Architectures](../../figures/fig_modern_vision_pairing.pdf)

*ViT-B/16 (MLP, V/O attention, Q/K attention paths) and ConvNeXt-T all achieve 100% pairing accuracy.*

### Robust to Adversarial Attacks

![Attack Robustness](../../figures/fig_attack_robustness.pdf)

*Pair accuracy remains 100% across 21 attack configurations until noise exceeds ~20% relative magnitude.*

---

## Key Findings

| Finding | Result |
|---------|--------|
| **GPT-2 (124M–1.5B)** | 100% pairing accuracy on MLP and attention paths |
| **ViT-B/16** | 100% accuracy; V/O path has strongest signal (+4.84 separation) |
| **ConvNeXt-T** | 100% accuracy |
| **ImageNet ResNets** | 91–100% with architecture-aware factorization |
| **Robustness** | 100% across 21 attack configs until ~20% perturbation |
| **Alternatives** | Frobenius: 47–67%, SV-distance: chance level |

---

## Method

The diagonal-dominance score is computed as:

```
d(i,j) = |tr(W_out × W_in)| / ||W_out × W_in||_F
```

For correctly paired layers, this score is high (O(√d)); for incorrect pairings, it's near zero (O(1/√d)).

**Why it works:** Training enforces dynamical isometry—adjacent layers learn to partially cancel each other out, so their product concentrates near -εI. This is a *learned* property, not an architectural artifact.

---

## Citation

```bibtex
@inproceedings{anonymous2026training,
  title={Training Leaves Traces: Diagonal Dominance as a Neural Network Fingerprint},
  author={Anonymous},
  booktitle={ACML},
  year={2026}
}
```
