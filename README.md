# I Dropped a Neural Net

A solver for [Jane Street's January 2026 puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet):
given 97 shuffled linear layers from a 48-block ResNet and 10,000 data points,
recover the exact original ordering.

The solver reconstructs the network in under 30 seconds, with final
`MSE = 0.000000000000` (exact reconstruction).

## Paper

This repo contains our AAAI 2026 submission:

> **Training Leaves Traces: Diagonal Dominance as a Neural Network Fingerprint.**
> AAAI 2026.
> [`paper/AAAI/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.pdf`](paper/AAAI/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.pdf)

**Abstract:** Verifying the provenance of neural network weights is difficult: existing watermarking schemes must be embedded during training, and can be removed by fine-tuning. We show that training itself leaves an intrinsic fingerprint requiring no such foresight. Residual networks initialized for dynamical isometry develop a distinctive structure: after training, each block's weight product settles near negative identity. This leaves a detectable trace: the diagonal-dominance score of correctly paired weights is high, while incorrect pairings score near zero.

**Key Results:**
- **100% pair accuracy** on every architecture-aware path of GPT-2 (124M–1.5B), ViT-B/16, ConvNeXt-T, BERT-base, Mistral-7B, LLaMA-2-7B (base + RLHF chat), DeepSeek-R1-Distill-Llama-8B, and Whisper (tiny/base/small, encoder + decoder); Qwen2.5-7B and DeepSeek hit 4/5 paths with the joint SwiGLU score rescuing the gate-only path
- **91–100% accuracy** on ImageNet ResNets (ResNet-50/101/152) with architecture-aware factorization
- **Robust** across 21 attack configurations (fine-tuning up to 50 epochs, weight noise to 20%); RLHF and R1 reasoning distillation both preserve the fingerprint
- **Model-level lineage detection: AUROC = 1.000** on both synthetic depth-24 MLPs (252 reference–suspect pairs across 5 attack types) *and* real CNNs on natural images (ResNet-18 / CIFAR-10, 22 descendants vs 6 same-arch independents). Worst-case descendant ($80\%$ pruning) still beats best independent by ~170× separation
- **Initialization-agnostic:** the fingerprint develops from orthogonal, Kaiming, and Xavier inits (100% negative trace across all three); requires only that residual blocks be non-degenerately used during training
- **Three application case studies:** training-quality early warning, zero-knowledge ownership proofs, model-compression auditing
- Signal scales as **O(√d)** with hidden dimension

Earlier versions are archived at [`paper/deprecated/`](paper/deprecated/).

## Generalization beyond the puzzle

The Jane Street puzzle solved itself the moment we noticed the diagonal-
dominance pattern. The interesting question is whether the pattern is
specific to that one network — and it is not. We measured it on every
GPT-2 size that HuggingFace ships, on both the MLP sublayer and the
attention sublayer:

| Model | Layers | MLP (W₂·W₁) | Attention V↔O | Attention Q↔K | Random init |
|---|---|---|---|---|---|
| gpt2 (124M) | 12 | **12/12** | **12/12** | **12/12** | 1–2/12 (chance) |
| gpt2-medium (355M) | 24 | **24/24** | **24/24** | **24/24** | — |
| gpt2-large (774M) | 36 | **36/36** | **36/36** | **36/36** | — |
| gpt2-xl (1.5B) | 48 | **48/48** | **48/48** | **48/48** | — |

All recovered by the same one-line Hungarian-on-`|tr|/‖·‖_F` pipeline that
solves the original puzzle. The signal-to-noise ratio (mean correct /
mean incorrect score) grows monotonically with model size: 68× → 122×
for MLP; 144× → 255× for attention. Reproducible via:

```bash
python experiments/gpt2_mlp_pairing.py        # full family, ~30 min on CPU
python experiments/gpt2_attention_pairing.py  # full family, ~10 min on CPU
```

## Cross-family transformer sweep

To rule out the possibility that the fingerprint is specific to GPT-style
causal decoders with GELU MLPs, we extended the sweep to six additional
transformer families spanning encoder-only / decoder-only paradigms,
GELU / SwiGLU activations, full / grouped-query / sliding-window
attention, and three post-training regimes (pretraining, RLHF chat,
R1 reasoning distillation):

| Family | n_layers / d_model | Distinctive features | All paths 100%? |
|---|---|---|---|
| BERT-base | 12 / 768 | encoder-only, bidirectional, MLM-pretrained, GELU | ✅ 3/3 |
| Mistral-7B | 32 / 4096 | SwiGLU, GQA 4:1, sliding-window attention (2023) | ✅ 5/5 |
| LLaMA-2-7B | 32 / 4096 | SwiGLU, full MHA, RMSNorm (2023 pretraining) | ✅ 5/5 |
| LLaMA-2-7B-chat | 32 / 4096 | + RLHF + instruction tuning | ✅ 5/5 |
| Qwen2.5-7B | 28 / 3584 | SwiGLU, GQA 7:1, RoPE, Q/K/V biases (2024) | 4/5 + joint rescues |
| DeepSeek-R1-Distill-Llama-8B | 32 / 4096 | + R1 reasoning distillation (2024) | 4/5 + joint rescues |
| Whisper-tiny/base/small | 4–12 / 384–768 | encoder + decoder, cross-attention, mel-spectrogram input (speech) | ✅ 3/3 in **both** encoder and decoder, all sizes |

**Four structural confounds ruled out at once:**
- *Not specific to GELU activations* — SwiGLU works across four SwiGLU families
- *Not specific to causal masking* — bidirectional (BERT) and sliding-window (Mistral) both work
- *Not specific to full attention* — 4:1, 7:1, and 8:1 GQA all give AUC 1.000 on $W_O W_V$
- *Not specific to text or self-attention* — Whisper's speech encoder + cross-attention decoder both fingerprint at 100% on MLP, V/O, and Q/K paths across tiny/base/small

**Three post-training regimes preserve the fingerprint on LLaMA architectures:**
pretraining (LLaMA-2-7B), RLHF chat-tuning (LLaMA-2-7B-chat), and
reasoning-trace distillation (DeepSeek-R1-Distill-Llama-8B) all hit 100%
pair accuracy on every attention path and on the joint SwiGLU stack.

**Graceful degradation through factorization redundancy:** the only sub-100%
data points in the sweep are Qwen2.5-7B and DeepSeek-R1-Distill-Llama-8B
on the gate-only path $W_{\text{down}}W_{\text{gate}}$ (68% and 84% pair
accuracy respectively). Both are 2024-era models trained with more
aggressive schedules. On both, the up-only path $W_{\text{down}}W_{\text{up}}$
becomes unusually strong (separation +2.869 and +1.469 vs +0.5 for 2023
baselines), and the joint SwiGLU factorization
$W_{\text{down}}[W_{\text{up}};W_{\text{gate}}]$ recovers all layers
exactly. The fingerprint degrades *gracefully*: when one architecture-aware
sub-path weakens, the joint factorization captures the union.

Reproducible (model weights pre-downloaded via `hf download`):

```bash
python experiments/transformer_family_pairing.py --model bert-base
python experiments/transformer_family_pairing.py --model mistral-7b
python experiments/transformer_family_pairing.py --model llama2-7b
python experiments/transformer_family_pairing.py --model llama2-7b-chat
python experiments/transformer_family_pairing.py --model qwen2.5-7b              # safetensors-direct
python experiments/transformer_family_pairing.py --model deepseek-r1-distill-llama-8b
python experiments/whisper_pairing.py                                            # tiny + base + small
```

The unified runner uses streaming layer-by-layer weight extraction (peak
RAM ≈ model size, not 2×) and a safetensors-direct loader for models
with bfloat16-stored weights that crash `AutoModelForCausalLM.from_pretrained`
on Windows. Random-init scoring is computed directly in numpy without
loading the full HF model. Full results in
[`results/transformer_family_pairing_*.json`](results/).

A few non-trivial side findings from the cross-family sweep (full
discussion in the paper):

- **Negative trace is sufficient but not necessary** for identifiability.
  BERT MLP has only 8% negative traces, and several attention paths
  across families have <10%, yet still recover 100% via Hungarian
  assignment. The broader fingerprint is the global concentration of
  diagonal mass in the correct branch product, which the global
  assignment optimum exploits even when row-wise dominance is absent.
- **Hungarian rescue regime** — paths with slightly negative pair
  separation (e.g. Mistral $W_{\text{down}}W_{\text{gate}}$ sep −0.025,
  LLaMA-2 same path sep −0.025, DeepSeek attn $W_QW_K^\top$ sep +0.045
  with 6% negative trace) still recover 100% via global assignment.
- **RLHF vs reasoning distillation** — both preserve the fingerprint
  on the LLaMA-architecture base, but they perturb different paths.
  RLHF chat-tuning shrinks attn $W_OW_V$ separation from +0.141
  (base) to +0.036; R1 reasoning distillation actually grows it to
  +0.280.

## Initialization-agnostic emergence

A skeptical reading of the dynamical-isometry theory is that the
fingerprint requires the network to begin training in a near-isometric
state — e.g. from explicit orthogonal initialization. We tested four
common schemes on a 24-block residual MLP (3 seeds each, 200 epochs):

| Init scheme | Pair acc | AUC | Frac neg trace | Eval loss |
|---|---|---|---|---|
| Orthogonal | 82% ± 20% | 0.947 | **100%** | 0.87 |
| Kaiming-normal | 97% ± 4% | 0.981 | **100%** | 1.35 |
| Xavier-normal | **100% ± 0%** | 0.990 | **100%** | 0.96 |
| Gaussian σ=0.02 | 13% ± 10% | 0.671 | 32% | 0.12 |

Three of four schemes converge to a clean fingerprint with **100% negative
trace** — training enforces the dynamical-isometry condition from any
non-trivial starting point. The σ=0.02 small-init case is a clean
*negative control*: with such small initial weights the residual blocks
collapse to near-zero contribution and the network shortcuts the task
through the skip path and the readout layer, so no Jacobian condition
develops. Real LLMs with σ=0.02 init (GPT-2, BERT, Mistral, LLaMA-2)
still recover 100% because the language-modeling objective is hard
enough that residual blocks must contribute non-trivially. The honest
claim is therefore: **the fingerprint emerges whenever training drives
residual blocks into non-degenerate use, regardless of init scheme.**

```bash
python experiments/init_scheme_ablation.py    # 4 schemes × 3 seeds, ~30 min
```

## Application case studies

Three end-to-end applications of the fingerprint (paper §6, code under
`case_studies/` and `experiments/`):

1. **Training Quality Assurance** — pair-accuracy at epoch 10 is an early-
   warning indicator: `pair_acc < 50%` flags 4 of 5 pathological training
   conditions (LR too high/low, no skip connections, high weight decay,
   small init) before they show in the loss curve.

2. **Zero-Knowledge Ownership Proofs** — a 4-phase Register / Challenge /
   Response / Verify protocol that commits to a hash of the per-block
   error matrix $E$ and only reveals challenged blocks. Passes all 5
   security scenarios (honest passes, attacks blocked).

3. **Model Compression Auditing** — compression operations (FP16/INT8/INT4
   quantization, 30–90% pruning, fine-tuning) preserve the $E$ correlation
   to the original model at >0.75, while knowledge distillation erases
   the correlation to ≈0. Clear separation enables derivation detection
   on stolen / distilled checkpoints.

```bash
python experiments/training_qa_case_study.py
python experiments/zkp_ownership.py
python experiments/compression_audit.py
```

## Model-level lineage detection

The pair-accuracy fingerprint identifies *which weights pair with which inside
a single model*. The natural follow-up: can the same residual-signature
machinery decide *whether one model descends from another*? We answer this on
two benchmarks.

**Benchmark 1 — Synthetic depth-24 residual MLPs.** Each reference has 75
descendants spanning 5 attack types (same-target fine-tune, target-shift
fine-tune, weight noise 1–15%, magnitude pruning 10–85%, fake-int8
quantization at 16–256 levels) and is paired against 67 non-descendants
(45 same-task independents, 5 different-task models, 5 random inits, 3
distilled students per reference). Across 252 reference–suspect pairs:

| Metric | Value |
| --- | --- |
| **AUROC** | **1.000** |
| **AUPRC** | 0.987 |
| **TPR @ 1% FPR** | **100%** |
| **TPR @ 10% FPR** | 100% |
| Descendant minimum $\mathcal{L}$ | 0.81 |
| Non-descendant maximum $\mathcal{L}$ | 0.20 |
| Distilled student false positives | 1/9 at 1% FPR (function-similar but weight-fresh) |

The score also recovers multi-generation ancestry on branching trees:
parent (0.95) > grandparent (0.91) > sibling (0.90) > uncle (0.87) >
cousin (0.85) ≫ independent (0.08). Cousins that diverged at a common
ancestor through entirely different fine-tune branches still inherit a
detectable shared residual signature.

**Benchmark 2 — ResNet-18 / CIFAR-10 (real CNN, natural images).** Two
ResNet-18 references and three same-architecture / different-seed
independents trained from scratch on CIFAR-10. Branch products extracted
per BasicBlock after BatchNorm folding, with channel-sum projection of the
3×3 kernels (block dims are heterogeneous across stages: [64, 64, 128, 256,
512], so Hungarian alignment is replaced by canonical block-index matching).
Each reference yields 11 cheap descendants (no extra training): Gaussian
weight noise at $\sigma_{\mathrm{rel}} \in \{1, 2, 5, 10\}\%$, magnitude
pruning at $\{20, 40, 60, 80\}\%$ sparsity, fake-int8 quantization at
$\{16, 64, 256\}$ levels.

| Metric | Value |
| --- | --- |
| **AUROC** | **1.000** |
| **TPR @ 1% FPR** | **100%** |
| Descendant $\mathcal{L}$ — noise (mean / min) | 0.983 / 0.916 |
| Descendant $\mathcal{L}$ — quantization (mean / min) | 0.954 / 0.854 |
| Descendant $\mathcal{L}$ — pruning (mean / min) | 0.864 / 0.695 |
| Independent baseline (mean / max) | **0.0014 / 0.004** |
| Worst-descendant / best-independent gap | ~170× |

The same residual-signature mechanism works on the convolutional branch
product, on natural images, with heterogeneous stage dimensions —
confirming that lineage discrimination is not a synthetic-MLP artifact.

**Adaptive evasion is hard.** Function-preserving per-block hidden-unit
permutations and per-block orthogonal rotations are mathematically exact
invariants of $W_{\mathrm{out}} W_{\mathrm{in}}$, so both give
$\mathcal{L}(A, T(A)) = 1.0$ to numerical precision (verified empirically).
Direct gradient-based suppression of $\mathcal{L}$ traces a Pareto frontier
between utility-preservation and fingerprint suppression; the score can
only be driven below threshold by destroying model utility.

```bash
python experiments/lineage_detection.py             # synthetic-MLP benchmark
python experiments/lineage_phase2_resnet.py         # ResNet-18 / CIFAR-10, ~1 hour on CPU
python experiments/lineage_phase2_resnet.py --reuse_ckpts   # cheap re-score from saved ckpts
```

## The Problem

A 48-block Residual Network was trained on a financial dataset. Each block
consists of two linear layers (input projection $48 \to 96$ and output
projection $96 \to 48$) connected by a ReLU, with a residual connection.
There is also a final linear layer ($48 \to 1$) producing the prediction.

All 97 layers were extracted and shuffled. The task: put them back in order.

The submission is verified by SHA-256 hash — there is exactly one correct
permutation, no MSE threshold to "settle" for. The search space is

$$
48! \times 48! \;\approx\; 10^{121}.
$$

You must solve two coupled problems simultaneously:

1. **Pairing** — for each block, which $W_\text{in}$ goes with which $W_\text{out}$?
2. **Ordering** — in what sequence do the 48 paired blocks execute?

## Architecture

```
Input (48) → [Block_1 → Block_2 → … → Block_48] → LastLayer → Output (1)

Block_k:  x → x + W_out · ReLU(W_in · x + b_in) + b_out
          W_in:  (96, 48)
          W_out: (48, 96)
```

## The Solution

### Step 1 — Pairing via Dynamic Isometry (< 1s)

Well-trained ResNets exhibit *dynamic isometry*: each block's Jacobian
$I + W_\text{out} W_\text{in}$ stays close to the identity so gradients
neither vanish nor explode. Equivalently,

$$
W_\text{out}^{(\star)} \, W_\text{in}^{(\star)} \;\approx\; -I + \varepsilon
$$

for the *correctly paired* indices. For wrong pairs, the product is essentially
unstructured noise. We score every candidate pair $(i,j)$ by the **diagonal
dominance ratio**:

$$
d(i, j) \;=\; \frac{\bigl|\mathrm{tr}\!\left(W_\text{out}^{(j)} W_\text{in}^{(i)}\right)\bigr|}
                   {\bigl\| W_\text{out}^{(j)} W_\text{in}^{(i)} \bigr\|_F}.
$$

This dimensionless ratio is maximised when the product is diagonal-dominant.
A maximum-weight bipartite matching (Hungarian algorithm) over the $48 \times 48$
score matrix recovers all 48 correct pairs in one shot. On this puzzle the
matched ratios are tightly clustered in $[1.76, 3.23]$, while the runner-up for
each row is below $1$.

### Step 2 — Seed Ordering by $\|W_\text{out}\|_F$ (instant)

In ResNets, earlier blocks tend to make smaller perturbations to the residual
stream than later ones. Sorting the 48 paired blocks ascending by
$\|W_\text{out}\|_F$ produces a seed permutation whose MSE on $N=1000$ samples
is already around $0.076$.

### Step 3 — Bubble-Sort Hill-Climb (~10s)

From the seed, sweep adjacent positions and swap any pair whose swap strictly
decreases MSE on a 1,000-sample subset. Then a pass of wider-gap swaps
(gaps $2$–$5$) cleans up stragglers. Converges to exact MSE $= 0$ in 3 rounds.

## Deepdive: What Didn't Work (and Why)

The successful approach was the *fifth* pairing strategy I tried. Most failed
approaches got stuck at MSE $\approx 0.03$–$0.5$ — the diagnostic signature of
"good enough ordering, but a few pairings are still wrong."

### Attempt 1 — SVD Singular-Value Fingerprinting (failed)

> Compute the sorted singular values of each $W_\text{in}$ and each
> $W_\text{out}$. Match by $\ell_1$ distance on the spectra.

Intuition: paired layers were trained together so they should share "effective
rank" and thus singular value spectra. **Wrong.** Random orderings under this
pairing gave MSE $86$–$556$. Singular values are well-controlled in magnitude
but say nothing about *which directions* in the 96-dim hidden space are coupled
between the two halves of a block.

With this pairing, simulated annealing on ordering plateaued at MSE
$\approx 0.49$ — no order can recover from broken pairings.

### Attempt 2 — Joint SA over (Pairing, Ordering) (failed)

Adding two move types to SA:

- swap $W_\text{in}$ pieces between two positions (re-pairs);
- swap $W_\text{out}$ pieces between two positions (re-pairs).

This explores the full $48!^2$ space. It improved to about $0.45$ but couldn't
break through — the joint landscape is too rugged and SA mixes too slowly
across two interleaved permutations.

### Attempt 3 — First-Order Algebraic Decomposition (failed)

If block contributions are small, then to first order

$$
\mathrm{pred} \;\approx\; \mathrm{LastLayer}\Bigl( X + \sum_k \mathrm{contrib}_k(X) \Bigr),
$$

which is order-invariant. Pairing becomes a clean Hungarian assignment on the
scalar features

$$
f_{ab}(x) \;=\; W_L \cdot \Bigl( W_\text{out}^{(b)} \, \mathrm{ReLU}\bigl(W_\text{in}^{(a)} x + b_\text{in}^{(a)}\bigr) + b_\text{out}^{(b)} \Bigr).
$$

Elegant in theory, useless in practice: residual contributions in this network
are *not* small (total cumulative change is roughly $6\times$ the input norm),
so first-order error dominates. Even with greedy peeling and 2-opt refinement,
the $\ell_2$ residual stayed above $\|\text{target}\|^2$. MSE after this
pairing: about $4.9$.

### Attempt 4 — Subspace Pairing (close, but stuck)

> Paired $(W_\text{in}, W_\text{out})$ share the hidden-dim subspace they
> communicate through. The column space of $W_\text{in}$ (its 48-dim range in
> $\mathbb{R}^{96}$) should match the row space of $W_\text{out}$ (what it
> reads from). Score by principal angles between subspaces.

Close. Random orderings under this pairing gave MSE $0.66$–$0.92$, a $50\times$
improvement over SV-matching. SA + 2-opt drove it to MSE $0.035$, then stalled.
Repair sweeps (inp-swap, out-swap, slice-reverse on full data) brought it to
$0.017$ but couldn't reach zero — the subspace metric agreed with the eventual
correct pairing on only about $38/48$ pairs.

### Attempt 5 — Diagonal Dominance Pairing (exact)

The right answer comes from the *structure of training*, not from generic
linear algebra. Dynamic isometry pins down both the magnitude *and* the
diagonal structure of $W_\text{out} W_\text{in}$. The ratio
$|\mathrm{tr}(M)|/\|M\|_F$ is dimensionless, perfectly diagonal-sensitive, and
unambiguous. Hungarian on this metric nails all 48 pairs on the first try.

## Lessons

- **Theory beats heuristics.** Singular-value matching, subspace overlap, and
  first-order Taylor expansions are all generic. They each found about half the
  right pairs and got stuck. The metric derived from *why these layers were
  trained that way* (dynamic isometry / near-identity Jacobian) nailed all 48
  pairs instantly.
- **Pairing is the bottleneck, not ordering.** Once pairing is exact, ordering
  falls to plain bubble-sort. Until pairing is exact, no amount of SA /
  Gumbel-Sinkhorn / 3-opt can reach zero.
- **MSE plateau $\approx 0.03$–$0.5$ is diagnostic.** It almost always means
  "a handful of pairings are wrong." Don't keep grinding the ordering — fix the
  pairing.
- **Target `pred`, not `true`.** The model deterministically produces `pred`;
  the `true` column has irreducible noise
  ($\mathrm{MSE}(\mathrm{pred}, \mathrm{true}) \approx 0.107$). Targeting
  `true` caps your loss at $0.107$ and hides whether you've actually
  reconstructed the model.

## Usage

```bash
pip install -r requirements.txt
python solutions/solve_dynamic_isometry.py
```

### Expected output

```
Step 1: pairing via diagonal dominance ratio
  matched ratios: min=1.764, max=3.232, mean=2.785

Step 2: seed order by ||W_out||_F
  seed MSE (N=1000): 0.075716

Step 3: hill-climb
  round 1: 52 swaps, MSE = 0.0021263796
  round 2: 15 swaps, MSE = 0.0003789640
  round 3:  5 swaps, MSE = 0.0000000000

FINAL full MSE: 0.000000000000
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── paper/
│   ├── AAAI/                     # AAAI 2026 submission
│   ├── deprecated/               # Archived versions (ACML, original note)
│   └── figures/                  # Figures used in the paper
│
├── puzzle_artifacts/
│   ├── pieces/                   # piece_0.pth … piece_96.pth (puzzle input)
│   └── historical_data.csv       # 10,000 × (48 measurements + pred + true)
│
├── solutions/
│   ├── solve_dynamic_isometry.py # Park's diagonal-dominance pipeline
│   └── solve_annealing.py        # Alternative solver (stub, see issue #2)
│
├── utils/
│   ├── __init__.py
│   ├── data.py                   # load_pieces, load_data, get_piece_groups
│   └── eval.py                   # eval_mse, build_model_from_blocks
│
├── results/                      # Cross-architecture pairing results (JSON)
│   ├── gpt2_mlp_pairing.json     # MLP pairing across 4 GPT-2 sizes
│   ├── gpt2_attention_pairing.json
│   ├── transformer_family_pairing_bert.json
│   ├── transformer_family_pairing_mistral_7b.json
│   ├── transformer_family_pairing_llama2_7b.json       # base (slim — random budget-limited)
│   ├── transformer_family_pairing_llama2_7b_chat.json  # RLHF chat
│   ├── transformer_family_pairing_qwen2_5_7b.json
│   ├── transformer_family_pairing_deepseek_r1_distill_llama_8b.json  # slim
│   ├── whisper_pairing.json      # Whisper tiny/base/small (encoder + decoder)
│   ├── lineage_phase2_resnet18_cifar.json  # ResNet-18/CIFAR-10 lineage POC
│   └── init_scheme_ablation.json
│
├── case_studies/                 # Application case studies (paper §6)
│   ├── case_study_1/             # Training Quality Assurance
│   ├── case_study_2/             # Zero-Knowledge Ownership Proofs
│   └── case_study_3/             # Model Compression Auditing
│
├── figures/                      # Working copies of generated figures
│
└── experiments/
    ├── data/                     # Pre-computed configs (JSON)
    ├── results/                  # Output CSVs from sweeps and attacks
    │
    ├── paper_experiments.py      # Puzzle-side experiments (pairing matrix,
    │                             #   pairing wall, seed proxies)
    ├── theory_verify.py          # Numerical verification of Prop 1 + Prop 2
    ├── strategies.py             # Four-strategy reassembly comparison
    ├── pipeline.py               # Generalized pipeline implementation
    │
    ├── sweep_full.py             # 4 (depth,width) × 3 seeds × ~10 ckpts sweep
    ├── focused_run.py            # Focused training run (24-block ResNet)
    │
    ├── null_model.py             # Untrained-network baseline (issue #1)
    ├── null_deepdive.py          # Training trajectory of the pairing signal
    ├── init_scheme_ablation.py   # 4 init schemes (orthog/Kaiming/Xavier/
    │                             #   Gaussian) × 3 seeds (issue #23)
    ├── nonresidual_baseline.py   # Non-residual control (issue #17)
    │
    ├── gpt2_mlp_pairing.py       # GPT-2 MLP layer pairing
    │                             #   (gpt2, medium, large, xl; issue #8)
    ├── gpt2_attention_pairing.py # GPT-2 attention block pairing
    │                             #   (V↔O, Q↔K, per-head; issue #9)
    ├── transformer_family_pairing.py  # Unified BERT/Mistral/LLaMA-2/
    │                             #   Qwen2.5/DeepSeek runner with
    │                             #   streaming extraction + safetensors-
    │                             #   direct path for bf16 models (#10)
    ├── whisper_pairing.py        # Whisper encoder + decoder pairing
    │                             #   (tiny/base/small, MLP/V-O/Q-K; #10)
    ├── lineage_detection.py      # Core residual-signature lineage metric
    │                             #   (residual_signature, lineage_score,
    │                             #    evaluate_lineage; pure numpy; #30)
    ├── lineage_phase2_resnet.py  # Phase 2: ResNet-18/CIFAR-10 lineage POC
    │                             #   (BN-folded BasicBlock products,
    │                             #    --reuse_ckpts for cheap re-scoring; #30)
    ├── transformer_mlp.py        # Earlier from-scratch transformer
    │                             #   (superseded by gpt2_*_pairing.py)
    │
    ├── attack.py                 # Fine-tuning + noise attacks
    ├── attack_fast.py            # Fast attack variant
    ├── attack_shuffle.py         # Post-fine-tune shuffle robustness
    │
    ├── training_qa_case_study.py # Application 1: training-quality early warning
    ├── zkp_ownership.py          # Application 2: zero-knowledge ownership proofs
    ├── compression_audit.py      # Application 3: model-compression auditing
    │
    ├── make_figs.py              # Generates the main-text figures
    └── make_figs_extra.py        # Theory + sweep + attack figures
```

## License

MIT
