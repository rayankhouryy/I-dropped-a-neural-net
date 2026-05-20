# I Dropped a Neural Net

A solver for [Jane Street's January 2026 puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet):
given 97 shuffled linear layers from a 48-block ResNet and 10,000 data points,
recover the exact original ordering.

The solver reconstructs the network in under 30 seconds, with final
`MSE = 0.000000000000` (exact reconstruction).

## Paper

This repo contains our ACML 2026 submission:

> **Training Leaves Traces: Diagonal Dominance as a Neural Network Fingerprint.**
> ACML 2026.
> [`paper/ACML_camera_ready/ACML_camera_ready/acml26_submission_template.pdf`](paper/ACML_camera_ready/ACML_camera_ready/acml26_submission_template.pdf) · source: [`paper/ACML_camera_ready/ACML_camera_ready/acml26_submission_template.tex`](paper/ACML_camera_ready/ACML_camera_ready/acml26_submission_template.tex)

**Abstract:** Verifying the provenance of neural network weights is difficult: existing watermarking schemes must be embedded during training, and can be removed by fine-tuning. We show that training itself leaves an intrinsic fingerprint requiring no such foresight. Residual networks initialized for dynamical isometry develop a distinctive structure: after training, each block's weight product settles near negative identity. This leaves a detectable trace: the diagonal-dominance score of correctly paired weights is high, while incorrect pairings score near zero.

**Key Results:**
- **100% accuracy** on GPT-2 (124M–1.5B), ViT-B/16, and ConvNeXt-T
- **91–100% accuracy** on ImageNet ResNets with architecture-aware factorization
- **Robust** across 21 attack configurations (fine-tuning, weight noise)
- Signal scales as **O(√d)** with hidden dimension

We also include the original research note exploring the phenomenon:

> **Layer Identifiability in Trained Neural Networks: From ResNets to Transformers.**
> [`paper/paper.pdf`](paper/paper.pdf) · source: [`paper/paper.tex`](paper/paper.tex)

Key contributions of the research note:

1. **Theory** — a closed-form margin formula for the diagonal-dominance ratio
   and a first-order derivation of the *Pairing Wall* slope; both numerically
   tight on Park's puzzle network. A null-model corollary shows the signal
   collapses to chance on randomly initialized networks.
2. **ResNet empirics** — a sweep over depths, widths, and seeds. Pairing
   transfers; ordering proxies do not. Identifiability is non-monotonic
   in training time.
3. **Transformers** — the diagonal-dominance signal also identifies layers
   in the **full GPT-2 family** (124M → 1.5B parameters): 100% pair accuracy
   on MLP sublayers, attention sublayers (both V↔O and Q↔K paths), and the
   per-head decomposition. See "Generalization beyond the puzzle" below.
4. **Forensic application** — pair recovery is robust to fine-tuning and
   noise attacks, making the signal a candidate for model fingerprinting
   and provenance.

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

A few non-trivial side findings (full discussion in the paper):

- **Attention pairing is ~2× sharper than MLP pairing**, consistently across
  the entire family. Likely because W_V and W_O are tied through the head
  structure and therefore more tightly co-trained.
- **Attention V→O traces sign-flip with depth** — early layers have negative
  traces (dynamic isometry holds) and late layers have positive traces.
  MLP traces grow monotonically *more* negative with depth. The two
  sublayer types specialise in opposite directions during training.
- **The pairing signal does not require dynamic isometry.** Attention has
  ~50% negative traces (chance) but 100% pair accuracy — so the diagonal-
  dominance ratio captures a more universal property than the negative-
  trace theorem that originally motivated it.

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
│   ├── paper.tex                 # Research note source
│   ├── paper.pdf                 # Compiled PDF (22 pages)
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
├── results/                      # GPT-2 family results (JSON)
│   ├── gpt2_mlp_pairing.json     # MLP pairing across 4 GPT-2 sizes
│   └── gpt2_attention_pairing.json
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
    │
    ├── gpt2_mlp_pairing.py       # GPT-2 MLP layer pairing
    │                             #   (gpt2, medium, large, xl; issue #8)
    ├── gpt2_attention_pairing.py # GPT-2 attention block pairing
    │                             #   (V↔O, Q↔K, per-head; issue #9)
    ├── transformer_mlp.py        # Earlier from-scratch transformer
    │                             #   (superseded by gpt2_*_pairing.py)
    │
    ├── attack.py                 # Fine-tuning + noise attacks
    ├── attack_fast.py            # Fast attack variant
    ├── attack_shuffle.py         # Post-fine-tune shuffle robustness
    │
    ├── make_figs.py              # Generates the main-text figures
    └── make_figs_extra.py        # Theory + sweep + attack figures
```

## License

MIT
