# Abstract — Discarded Content

Content removed from the original 409-word abstract to achieve 248-word target.

---

## Detailed Architecture Enumeration (moved to Master Results Table)

Original text:
> Empirically, across $12$ model architectures (GPT-2 124M--1.5B, BERT-base, Mistral-7B, LLaMA-2-7B base and RLHF chat, Qwen2.5-7B, DeepSeek-R1-Distill-Llama-8B, Whisper tiny/base/small, ResNet-18/50/101/152, ViT-B/16, and ConvNeXt-T)

Replaced with: "12 architectures spanning 5 families"

---

## Specific Epoch/Noise Thresholds

Original text:
> The fingerprint is absent at initialization, appears within the first five epochs of training, follows the residual-MLP margin formula's $\sqrt{d}$ envelope on GPT-2 MLPs

Removed: epoch-specific emergence timing

---

## Detailed Score Distribution Numbers

Original text:
> while knowledge-distilled students remain near the null distribution ($\mathcal{L} \approx 0.086$) despite functional imitation

Simplified to: "correctly classify as non-descendants"

---

## Extended Architecture-Aware Product Description

Original text:
> for deeper or non-MLP branches we measure the trace on the architecture-appropriate product (e.g.\ $W_3W_2W_1$ in Bottleneck ResNets, $W_OW_V$ and $W_QW_K^{\!\top}$ in attention, joint SwiGLU products)

Moved to: Background section (Table 1: Architecture-aware factorization)

---

## Post-Training Modification List

Original text:
> $5$ post-training weight modifications

Now implicit in: "persists under fine-tuning, quantization, pruning, RLHF, and reasoning distillation"

---

## Boundary Case Detail

Original text:
> Boundary cases clarify the signal: independently trained same-architecture models do not falsely match ($\mathcal{L} \le 0.20$)

Removed: specific threshold numbers; retained core message about distillation boundary

---

## Reasoning for Cuts

1. **Architecture list** — Creates visual clutter; readers get the scope from "12 architectures, 5 families" and can see details in experiments
2. **Epoch thresholds** — Too granular for abstract; belongs in method/experiments
3. **Score numbers** — AUROC=1.000 is the headline; internal score values are implementation detail
4. **Architecture-aware products** — Technical detail that belongs in Background
5. **Boundary thresholds** — The distillation distinction is the key insight; specific numbers belong in experiments
