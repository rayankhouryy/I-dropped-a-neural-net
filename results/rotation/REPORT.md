# Rotation Laundering Benchmark — Appendix Report

**Paper:** *Training Leaves Traces* (AAAI-27). **Date:** 2026-07-28.
**Question:** The scale-aware Re-Basin baseline tied ours at AUROC 1.000 on every
P/D/PD variant because exact function preservation under ReLU limits the attacker to
permutation+diagonal transforms. Can an attacker disguise with a block-wise orthogonal
rotation R (outside that class), recover utility via fine-tuning, and evade detection?

All numbers below are **measured**, produced by
`experiments/scripts/rotation_benchmark_mlp.py` on this machine. Every cell
contradicting the pre-registered predictions is listed in **§Discrepancies**.

## Bug-fix verification (prerequisite)

The singleton-std fix in `lineage_phase1_mlp.py:descendant_noise` was committed
(`19823d7`). The `sanitize_nonfinite_head` shim is removed. Finiteness assertions
run after every descendant construction — all pass.

Reproduction checks on the fixed bank:
- Ours (none): AUROC = **1.000** (matches prior run exactly)
- IPGuard none is no longer computed (this experiment focuses on M-based methods)
- All M-based and raw-weight "none"-column cells reproduce the prior run.

---

## Seeds (all deterministic)
- Bank: `torch.manual_seed(0)`, `np.random.seed(0)` (identical to Table 6).
- Rotation base: `9000` (per-descendant `9000+i`), phase code `_PHASE_ROTATE=2`,
  per-block substreams via `np.random.default_rng([seed, 2, block])`.
- Recovery FT: same optimizer as PDFT (Adam 3e-4), batch seed from torch global.

---

## Main result — AUROC (rows = methods, cols = rotation variant)

`none` = unlaundered (reproduction). R0.1/R0.3/R1.0-FT = rotation(θ) + recovery FT.
PR0.3D-FT = permutation + R(0.3) + D-strong + recovery FT (attacker's full toolkit).

Related = 30 laundered descendants; unrelated = 22 unchanged; 52 pairs per cell.

| Method | none | R0.1-FT | R0.3-FT | R1.0-FT | PR0.3D-FT |
|---|---|---|---|---|---|
| **Diagonal Dominance (ours, on M)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Raw aligned Frobenius | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Raw singular-value distance | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Raw weight cosine | 1.000 | 1.000 | 1.000 | 1.000 | 0.689 |
| Re-Basin Frobenius (perm-match) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **Re-Basin + scale** (perm+scale) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| Procrustes-Frobenius (orthogonal) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| GL-align Frobenius (general linear) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

**Headline.** Pure rotation variants (R0.1/R0.3/R1.0-FT) are not effective disguises:
all baselines score 1.000 because recovery fine-tuning largely restores the original
weight geometry. The **only** effective laundering variant is **PR0.3D-FT** (combining
permutation + rotation + scale), which collapses raw baselines, plain Re-Basin, Procrustes,
and GL-align to AUROC 0.000–0.689. Ours holds at 1.000 by M-invariance.

---

## New baselines: Procrustes and GL-align

**Procrustes-Frobenius:** Per-block joint orthogonal Procrustes minimizing
`||Win_A - R @ Win_B||² + ||Wout_A - Wout_B @ R^T||²`. This is the natural defender
escalation against rotation-only attacks.

**GL-align Frobenius:** Per-block general linear alignment `T @ Win_B ≈ Win_A` via
least squares. With h=48, d=16, this is underdetermined (2304 params vs 816 constraints)
and should overfit — included to test whether overfitting inflates the unrelated null.

**Result:** Both fail on PR0.3D-FT (AUROC 0.000). Procrustes cannot jointly invert
the composed P + R + D transformation. GL-align overfits but still cannot find a
valid alignment when the composition includes permutation (which scrambles the unit
correspondence). The unrelated null is NOT significantly inflated — both methods
maintain clear separation on pure-rotation variants.

---

## Re-Basin + scale survives PR0.3D-FT (unexpected)

The scale-aware Re-Basin baseline achieves **AUROC = 1.000 on PR0.3D-FT**, matching ours.

**Mechanism:** After the attacker applies P + R + D-strong, recovery fine-tuning partially
undoes R (since the network must recover function). The Re-Basin perm-matching then
finds the correct permutation correspondence, and the per-unit least-squares scale
removal handles D. The residual rotation error after recovery FT is small enough that
Frobenius distance still separates related from unrelated.

**Implication:** On this benchmark, the scale-aware Re-Basin baseline is **robust to the
full attacker toolkit**. The only remaining distinction from ours is **cost**: Re-Basin
must solve a per-block alignment + scale optimization (Hungarian + LS) for every pair,
while ours is invariant with zero alignment work.

---

## M-invariance check (replaces function gate)

For rotation variants, the function is intentionally broken (ReLU does not commute with R).
The gate verifies M-invariance: `||M_rotated - M_original||_F / ||M_original||_F < 1e-6`.

| Variant | max rel. error | status |
|---|---|---|
| R0.1-FT | 2.95e-07 | PASS |
| R0.3-FT | 2.99e-07 | PASS |
| R1.0-FT | 2.92e-07 | PASS |
| PR0.3D-FT | 2.92e-07 | PASS |

All variants pass — M is exactly invariant to rotation (by construction: M' = W_out R^T R W_in = M).

---

## Utility recovery (epochs + final loss)

Recovery fine-tuning runs until eval loss is within 10% of the unlaundered descendant's,
capped at 30 epochs. Reference eval losses: ref_0 = 0.420, ref_1 = 0.213.

| Variant | recovered | avg epochs | avg final loss |
|---|---|---|---|
| R0.1-FT | 30/30 | 5.1 | 0.333 |
| R0.3-FT | 30/30 | 5.2 | 0.333 |
| R1.0-FT | 30/30 | 6.0 | 0.338 |
| PR0.3D-FT | 30/30 | 4.6 | 0.335 |

All 30 descendants recovered within the cap. θ=1.0 required slightly more epochs but
was **not** unrecoverable — the hypothesis that large θ might be unrecoverable is
falsified on this benchmark.

---

## Two-sided measurement (alignment tradeoff)

For alignment-equipped baselines (Re-Basin, Procrustes, GL-align), alignment is part of
the method and runs on unrelated pairs too. The question: does the alignment inflate
spurious similarity on unrelated pairs, degrading AUROC from the null side?

| Method | mean_unrelated (none) | max_unrelated (none) | shift vs raw |
|---|---|---|---|
| raw_aligned_frobenius | -1.371 | -1.356 | — |
| rebasin_frobenius | -1.130 | -1.121 | +0.24 (closer) |
| rebasin_scale_frobenius | -2.009 | -1.560 | -0.64 (farther) |
| procrustes_frobenius | -0.811 | -0.800 | +0.56 (closer) |
| gl_align_frobenius | -0.519 | -0.511 | +0.85 (closer) |

**Observation:** Re-Basin + scale pushes unrelated scores **more negative** (farther from
related), improving margin. Procrustes and GL-align shift unrelated scores closer to zero
(worse margin) but not enough to flip any pair — AUROC remains 1.000 on all pure-rotation
variants. The tradeoff does NOT bite at h=48 for any method tested.

---

## Discrepancies (measured vs pre-registered predictions)

1. **Procrustes "may recover R-only" — CONFIRMED but IRRELEVANT.** Procrustes does achieve
   AUROC 1.000 on R0.1/R0.3/R1.0-FT, but so do all other baselines. Pure rotation is not
   an effective disguise.

2. **Re-Basin perm+scale "should collapse on R-containing variants" — CONTRADICTED.** The
   scale-aware Re-Basin survives **all** variants including PR0.3D-FT (AUROC 1.000). The
   prediction assumed rotation would decouple from the permutation+scale class, but after
   recovery FT the residual rotation is too small to disrupt unit matching.

3. **Procrustes "may survive PR0.3D-FT" — CONTRADICTED.** Procrustes collapses to 0.000
   on PR0.3D-FT. Per-block orthogonal alignment cannot invert the composed transformation.

4. **GL-align "may overfit and inflate unrelated" — CONFIRMED but BENIGN.** GL-align shifts
   unrelated scores closer to zero (+0.85 shift) but AUROC remains 1.000 on pure-rotation
   variants. The overfit does NOT cause false positives.

5. **θ=1.0 "may be unrecoverable" — CONTRADICTED.** All 30 descendants recovered within
   30 epochs at θ=1.0. Average epochs to recovery was only 6.0 (vs 5.1–5.2 for smaller θ).

---

## Controls (Track D)

1. **Ours invariance (all R variants):** `max|ΔL vs none| < 1e-2` (slight drift from FT),
   but AUROC = 1.000 throughout. M-invariance relative error < 3e-7 for all.

2. **None column reproduces Table 6:** Ours 1.000, raw_weight_cosine 1.000,
   raw_aligned_frobenius 1.000, raw_singular_value_dist 1.000. Confirmed.

---

## Draft appendix paragraph (150 words)

We tested whether block-wise orthogonal rotation — a disguise outside the permutation+scale
class — can evade lineage detection after recovery fine-tuning. For pure rotation variants
(θ ∈ {0.1, 0.3, 1.0}), all baselines achieve AUROC 1.000; recovery FT largely restores the
original weight geometry. The only effective disguise is the full attacker toolkit
(permutation + rotation + D-strong + FT), which collapses raw baselines and Procrustes to
AUROC 0.000. Our M-based method holds at 1.000 by exact invariance. Surprisingly, the
scale-aware Re-Basin baseline also survives at AUROC 1.000: after FT, residual rotation is
small enough that unit matching succeeds. The paper's framing therefore remains: both our
signature and the scale-aware baseline detect lineage through function-preserving laundering;
the distinction is computational (ours requires zero alignment work) rather than capability.

---

## Files
- `results/rotation/summary.csv` — aggregate table
- `results/rotation/by_cell/<variant>__<method>.json` — 40 per-cell files
- `results/rotation/rotation_full.json` — config, seeds, summary, recovery log, invariance log
