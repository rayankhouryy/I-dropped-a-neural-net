# I Dropped a Neural Net

Solution to Jane Street's January 2026 puzzle
[**"I Dropped a Neural Net"**](https://huggingface.co/spaces/jane-street/droppedaneuralnet):
a 48-block residual network was "dropped" and its 97 linear layers shuffled.
The goal is to find the unique permutation of `[0..96]` that reassembles the model.

**Final MSE on the dataset: `0.000000000000` (exact reconstruction).**
Total runtime: about 10 seconds on a laptop CPU.

The submitted permutation:

```
43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,
33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,
38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,
90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85
```

## Usage

```bash
pip install -r requirements.txt
python solve.py
```

## The Problem

You're given 97 `piece_*.pth` files (each a single `nn.Linear`) and a CSV with
10,000 rows of `48 measurements + pred + true`. The original architecture is:

```python
class Block(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=96):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)        # (96, 48)
        self.out = nn.Linear(hidden_dim, in_dim)        # (48, 96)
    def forward(self, x):
        return x + self.out(torch.relu(self.inp(x)))    # residual

class LastLayer(nn.Module):                             # (1, 48)
    def __init__(self, in_dim=48, out_dim=1):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
```

So the 97 pieces decompose into **48 `inp` layers + 48 `out` layers + 1 `last`
layer**. You need to solve two coupled problems simultaneously:

1. **Pairing** — which `inp` belongs with which `out` to form each block? (48! options)
2. **Ordering** — in what sequence do the 48 blocks run? (another 48!)

Total search space is roughly 48!² ≈ 10¹²¹. Brute force is hopeless. Submissions
are verified by SHA-256 hash, so we need exact MSE = 0, not "small enough."

## The Solution (3 steps, ~10 seconds)

### Step 1 — Pairing via Dynamic Isometry (< 1s)

The killer insight. Trained ResNet blocks satisfy *dynamic isometry*: each block's
Jacobian `I + W_out · W_in` stays close to the identity so gradients don't blow
up. That forces the product

```
M = W_out · W_in   ≈   -I + (small)
```

So **for the true pair `(inp_i, out_j)`, the matrix `M` has a strongly dominant
negative diagonal**. For a wrong pair, the off-diagonal is just noise. Score
every candidate by the diagonal dominance ratio:

```
d(i, j) = |tr(W_out_j · W_in_i)| / ||W_out_j · W_in_i||_F
```

and solve a max-weight bipartite matching with the Hungarian algorithm. On this
puzzle the matched ratios cluster around **2.8** for correct pairs versus much
less than 1 for random pairs, giving a unique error-free assignment.

### Step 2 — Seed Ordering by ‖W_out‖_F (instant)

In residual networks, earlier blocks tend to make smaller perturbations to the
signal than later ones. Sorting the 48 paired blocks by `||W_out||_F` ascending
gives a seed permutation whose MSE on 1,000 samples is already about **0.076** —
close enough that local search can finish the job.

### Step 3 — Bubble-Sort Hill-Climb (~10s)

Repeatedly sweep adjacent positions, swapping any pair whose swap strictly
lowers MSE. Then a sweep of wider-gap swaps (gaps 2–5) for stragglers. Converges
to exact 0 in 3 rounds:

```
round 1: 52 swaps, MSE = 0.0021263796
round 2: 15 swaps, MSE = 0.0003789640
round 3:  5 swaps, MSE = 0.0000000000
```

That's it. The whole solver is around 120 lines.

## Deepdive: What Didn't Work (and Why)

The successful approach was the *fifth* pairing strategy I tried. The journey is
instructive — most failed approaches got stuck at MSE ≈ 0.03–0.5, which is the
signature of "good enough ordering, but a few pairings are still wrong."

### Attempt 1 — SVD Singular-Value Fingerprinting (failed)

> Compute the sorted singular values of each `inp` and each `out`. Match by L1
> distance on the spectra.

Heuristic intuition: paired layers were trained together so they should share
"effective rank" and thus singular value spectra. **Wrong.** Random orderings
under this pairing gave MSE 86–556. The singular values are individually
well-controlled but say nothing about *which directions* in the 96-dim hidden
space are coupled between the two halves of a block.

With SVD pairing, simulated annealing on ordering plateaued at MSE ≈ **0.49** —
far from zero, because no order can recover from broken pairings.

### Attempt 2 — Joint SA over (Pairing, Ordering) (failed)

Recognising the pairing was bad, I added two move types to SA:

- swap `inp` pieces between two positions (re-pairs)
- swap `out` pieces between two positions (re-pairs)

This explores the full 48!² space. It improved to about 0.45 but couldn't break
through — the joint landscape is too rugged and SA mixes too slowly across two
interleaved permutations.

### Attempt 3 — First-Order Algebraic Decomposition (failed)

> If block contributions are small, then to first order
> `pred ≈ LastLayer(X + Σₖ contribₖ(X))`, which is *order-invariant* in the
> pairing. So pairing becomes a clean Hungarian assignment on scalar features
> `f_ab(x) = W_L · (W2_b · ReLU(W1_a x + b1_a) + b2_b)`.

Elegant in theory, useless in practice: residual contributions in this network
are *not* small — total cumulative change is roughly 6× the input norm — so
first-order error dominates. Even with greedy peeling and 2-opt refinement, the
L2 residual stayed above `||target||²`. MSE after this pairing: about 4.9.

### Attempt 4 — Subspace Pairing (close, but stuck)

> Paired `(W_in, W_out)` share the hidden-dim subspace they communicate
> through. The column space of `W_in` (its 48-dim range in R⁹⁶) should match
> the row space of `W_out` (what it reads from). Score by principal angles
> between subspaces.

This was *close* — random orderings under this pairing gave MSE 0.66–0.92, a 50×
improvement over SVD-value matching. SA + 2-opt drove this to MSE 0.035, then
stalled. Repair sweeps (inp-swap, out-swap, slice-reverse on full data) brought
it to 0.017 but couldn't reach zero. The subspace metric agreed with the
eventual correct pairing on only about 38 / 48 pairs — the remaining ones get
stuck in basins that local search can't escape.

### Attempt 5 — Diagonal Dominance Pairing (exact)

The right answer comes from the *structure of training*, not from generic linear
algebra. Dynamic isometry pins down both the magnitude **and** the diagonal
structure of `W_out · W_in`. The ratio `|tr(M)| / ||M||_F` is dimensionless,
perfectly diagonal-sensitive, and unambiguous — matched ratios are 1.7–3.2
versus near-zero for wrong pairs. Hungarian on this metric nails all 48 pairs
on the first try.

## Lessons

- **Theory beats heuristics.** Singular-value matching, subspace overlap, and
  first-order Taylor expansions are all generic. They each found about half the
  right pairs and got stuck. The metric derived from *why these layers were
  trained that way* (dynamic isometry / near-identity Jacobian) nailed all 48
  pairs instantly.
- **Pairing is the bottleneck, not ordering.** Once pairing is exact, ordering
  falls to plain bubble-sort. Until pairing is exact, no amount of SA /
  Gumbel-Sinkhorn / 3-opt can reach zero.
- **MSE plateau ≈ 0.03–0.5 is diagnostic.** It almost always means "a handful
  of pairings are wrong." Don't keep grinding the ordering — fix the pairing.
- **Target `pred`, not `true`.** The model deterministically produces `pred`;
  the `true` column has irreducible noise (MSE(pred, true) ≈ 0.107). Targeting
  `true` caps your loss at 0.107 and hides whether you've actually reconstructed
  the model.

## Files

```
.
├── README.md               # this file
├── requirements.txt        # torch, numpy, pandas, scipy
├── solve.py                # the ~120-line solver
├── historical_data.csv     # 10,000 x (48 measurements + pred + true)
└── pieces/
    ├── piece_0.pth
    ├── ...
    └── piece_96.pth
```