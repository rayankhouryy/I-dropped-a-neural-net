# Checkpoint Laundering Benchmark — Report

**Paper:** *Training Leaves Traces* (AAAI-27). **Date:** 2026-07-28.
**Question:** the 52-pair MLP benchmark (Table 6) is too easy — four data-free
weight-space methods all reach AUROC = 1.000. Do they survive *function-preserving
laundering* that decouples raw weight proximity from lineage?

All numbers below are **measured**, produced by
`experiments/scripts/laundering_benchmark_mlp.py` (Tracks A/B/D, CPU) on this
machine. Track C (real Llama-2) is cluster-only and was **not run here** (no Llama
weights, no CUDA on this host) — its code is complete and validated for import;
commands are at the end. Every cell contradicting the pre-registered predictions is
listed in **§Discrepancies**, including numbers that favor a baseline.

## Seeds (all deterministic)
- Bank construction: `torch.manual_seed(0)`, `np.random.seed(0)` (identical to Table 6).
- Laundering randomness base: `7000` (per-descendant `7000+i`, per-block substreams
  via `np.random.default_rng([seed, phase, block])`).
- Probe inputs: `12345` (512 × N(0,1), dim 16). PDFT fine-tuning: minibatch seed `seed+777`.
- Track D1 (launder unrelated): base `7000+5000+rank`.

---

## Main result — AUROC (rows = methods, cols = laundering variant)

`none` = unlaundered descendants (reproduction column). Related pairs = 30 laundered
descendants; unrelated = 22 reused unchanged; 52 pairs per cell.

| Method | none | P | D-mild | D-strong | PD | PDFT |
|---|---|---|---|---|---|---|
| **Diagonal Dominance (ours, on M)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Raw aligned Frobenius | 1.000 | 0.495 | 1.000 | 0.000 | 0.000 | 0.000 |
| Raw singular-value distance | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Raw weight cosine | 1.000 | 0.856 | 1.000 | 1.000 | 0.805 | 0.805 |
| CKA (activations) | 0.847 | 0.847 | 0.847 | 0.847 | 0.847 | 0.994 |
| SVCCA (activations) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| IPGuard (regression) | 0.700 | 0.700 | 0.700 | 0.700 | 0.700 | 0.827 |
| **Track B: Re-Basin Frobenius** (perm-match) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| **Track B: Re-Basin + scale** (perm+scale) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| *aligned Frobenius on M (completeness)* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| *singular-value dist on M (completeness)* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.986 |
| *weight cosine on M (completeness)* | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Design note (critical).** In the shipped harness, *all* weight-space baselines
consume the **branch product** `M = W_out·W_in`, which is **exactly invariant** to
hidden-unit permutation (PᵀP=I) and rescaling (D⁻¹D=I). Run that way they trivially
score 1.000 on P/D/PD (bottom three *italic* rows, shown for completeness). To make
the baseline comparison meaningful we run the three baselines on **raw per-block
weights** (`raw_*` rows) — what an attacker-facing method actually has, absent M.
The invariance of M is *our contribution*; the `*_M` rows show baselines only survive
by borrowing it.

**Headline.** Ours is invariant to P/D/PD **by construction** — measured
`max|ΔL| ≈ 4.6e-8` and residual-signature cosine `= 1.0000000000` (all blocks, all
variants; §Invariance). It also holds at 1.000 through **PDFT** (5 epochs of fine-tuning
after P+D-strong), the only variant whose outcome was not algebraically determined.

---

## Track B — Re-Basin baselines (measured, not predicted)

**Did the scale-aware Re-Basin baseline recover D? YES — fully.**

- **Plain Re-Basin (weight-matching permutation only):** recovers **P** (AUROC 1.000,
  vs raw Frobenius 0.495) but **collapses under any D** (D-strong/PD/PDFT = 0.000).
  Permutation matching cannot undo per-unit rescaling.
- **Scale-aware Re-Basin (permutation + per-unit least-squares scale):** **AUROC =
  1.000 on P, D-mild, D-strong, PD, *and* PDFT.** It matches ours on every variant.

**Honest reframing (as instructed).** On this benchmark the scale-aware baseline is not
less *capable* than ours — it ties at 1.000. The distinction is **cost and guarantee**:
ours is invariant with *zero* alignment work (`ΔL ≈ 1e-8`, provably, signature cosine
exactly 1), whereas the Re-Basin baseline must **solve a per-block alignment + per-unit
scale optimization** for every pair (L Hungarian assignments + L least-squares fits).
The paper's framing should therefore be, verbatim to the measured result:
*"a raw-weight baseline can also survive laundering, but only by explicitly solving the
per-block permutation+scale problem that our signature is invariant to for free."*

---

## Function-preservation gate (HARD gate — max |Δ| over 512 probes, float32)

Threshold `< 1e-4`. Gate is on the **block-stack output** `g(x)` (residual stream after
all blocks, before the final head) — see §Pre-existing-bug for why. **All variants PASS.**

| Variant | none | P | D-mild | D-strong | PD | PDFT (pre-FT stage) |
|---|---|---|---|---|---|---|
| max deviation | 0.0 | 4.77e-6 | 5.72e-6 | 4.77e-6 | 4.29e-6 | 4.29e-6 |
| gate | PASS | PASS | PASS | PASS | PASS | PASS |

Per-checkpoint deviations for all 150 laundered related + 4 laundered unrelated are in
`laundering_full.json → gate_deviations`.

## PDFT utility (eval-loss on the reference validation set, per kind)

PDFT = P + D-strong then 5 epochs Adam @ lr 3e-4 (same optimizer/flat schedule as the
descendants' fine-tuning). Utility is **comparable or better** than the unlaundered
descendant — laundering + light FT does not degrade the model:

| kind | unlaundered | PDFT-laundered |
|---|---|---|
| fine_tune | 0.322 | 0.322 |
| fine_tune_new_target | 13.836 | 0.659 |
| noise | 0.348 | 0.329 |
| prune | 0.635 | 0.323 |
| quantize | 0.318 | 0.328 |

(`fine_tune_new_target` was trained on a *different* target so it scores poorly on the
reference's val set; 5 epochs on the reference data pull it back — expected, not a defect.)

---

## Track D — controls

1. **Launder unrelated (4 diff-seed checkpoints, PD):** all 7 methods still score them
   **unrelated**; the null does not move. Example (ours / raw-cosine): PD-laundered
   scores equal the unlaundered scores to 3 dp (`0.030→0.030`, `0.053→0.053`), and
   Re-Basin-scale drives them *more* negative (`−1.56 → −40.8`), i.e. further from
   descendant. No false positives manufactured. Gate max dev `3.8e-6`. (Full data:
   `laundering_full.json → track_d1_unrelated_pd`.)
2. **Ours invariance (P/D/PD):** `max|ΔL vs none|` = 3.6e-8 / 2.3e-8 / 1.7e-8 / 4.6e-8;
   `min signature cosine = 1.0000000000` in every case. *(Note: L(ref,kid)=0.371, not
   1.0 — the `τ_s=0.5` dominance gate caps weakly-trained branches; the correct invariance
   statement is "laundering does not change the score," ΔL≈1e-8, and signature cosine=1,
   both confirmed. The plan's shorthand "assert L=1.0" was imprecise and is corrected here.)*
3. **Harness-unchanged (`none` column):** reproduces Table 6 for the M-based/function
   methods — ours 1.000, SVCCA 1.000, aligned-Frobenius-M 1.000, SVD-M 1.000,
   weight-cosine-M 1.000. (CKA differs by cross-version drift; IPGuard now 0.700 after
   the root-cause fix below — see §Discrepancies.)

---

## Discrepancies (measured vs pre-registered predictions)

Predictions were structural only. Contradictions are findings.

1. **SVD distance, "D-mild may be survivable" — CONTRADICTED.** Raw singular-value
   distance survives P (1.000, as predicted) but **collapses at D-mild too** (AUROC
   0.000, not just D-strong). LogUniform[0.5,2] row-scaling already shifts the raw
   singular spectrum enough to destroy separation. Both D strengths kill it.
2. **Weight cosine "degrades on D-strong" — PARTIALLY CONTRADICTED.** Raw weight cosine
   holds **AUROC = 1.000 on both D-mild and D-strong** (mean related score does drop,
   0.996→0.929→0.655, but unrelated is ~0.06, so ranking/AUROC is preserved). It collapses
   only on P-containing variants (P 0.856, PD/PDFT 0.805) — and even there degrades rather
   than fully collapsing to 0.5.
3. **IPGuard `none` = 0.700, not the Table-6 0.500 — and the 0.500 was itself a bug.**
   The committed 0.500 came from a NaN: `descendant_noise` scaled noise by `p.std()`,
   and torch's Bessel-corrected `std()` on the 1-element `last.bias` returns NaN, so the
   6 noise descendants emitted NaN predictions → IPGuard 0.000 on every noise pair →
   AUROC dragged to chance. We fixed the **root cause** (singleton guard in
   `descendant_noise`, §Root-cause-fix), so noise descendants now emit finite predictions
   that agree with the parent: IPGuard noise per-kind mean **0.000 → 0.941**, AUROC
   **0.500 → 0.700**, and the Gap-Z the paper reports flips **−4.9 → +3.1** (i.e. IPGuard
   no longer spuriously *inverts* on this easy benchmark). A same-torch pre/post diff
   confirms **every other method is bit-identical** (all 6 non-IPGuard rows, AUROC and
   every per-kind mean) and only IPGuard's `noise` cell moved — the fix is surgical.
   IPGuard remains identical across none/P/D/PD (0.700), as expected for a function-space
   method under function-preserving laundering.
4. **CKA `none` = 0.847 vs committed Table-6 0.829 (Δ0.018).** Cross-version numerical
   drift in 120-epoch Adam training (committed run predates torch 2.12 / numpy 2.3). CKA
   is the only non-saturated Table-6 cell and the only one sensitive to this. Internal
   consistency is unaffected: CKA is identical across none/P/D-*/PD (0.847) by
   function-preservation, and rises to 0.994 under PDFT.
5. **Re-Basin scale-aware ties ours at 1.000 everywhere (incl. PDFT).** Predicted
   "genuinely unknown — measure." Measured: it fully recovers D and P (and PDFT). See
   Track B for the honest reframing — the contribution is zero-cost invariance vs a
   solved optimization, not a detection gap on this benchmark.
6. **PDFT for ours = 1.000 (predicted "expected high, measure it").** Confirmed high;
   invariance survives 5 epochs of fine-tuning. `singular_value_dist_M` dips to 0.986
   under PDFT (FT perturbs M slightly) — the one M-based cell that moves.

Predictions **confirmed:** ours 1.000 on P/D/PD by invariance; raw weight cosine
collapses on P-containing variants; raw block-level Frobenius fails P (0.495); raw SVD
survives P; CKA/SVCCA/IPGuard unchanged by function-preserving laundering; plain Re-Basin
fixes P but not D.

---

## Track C — Real-LLM (Llama-2), cluster-only — NOT RUN HERE

Code complete and import-validated: `experiments/scripts/laundering_llm.py`. Launders
chat's SwiGLU per layer (P: permute rows of W_gate & W_up + cols of W_down; D-mild: scale
W_up rows / W_down cols, **gate untouched** since SiLU is nonlinear), gates on the
hidden state after `--probe-blocks` layers (32 probe sequences), then scores ours (L) and
raw weight cosine on (base vs laundered-chat) plus an unrelated control (base vs
laundered **OpenLLaMA-7B**). fp16 caveat is handled honestly: the gate upcasts laundered
matmuls to fp32 and **reports the measured deviation** rather than forcing the 1e-4
threshold; rerun `--dtype float32` for a stricter check if memory allows.

**Expectation to verify on the cluster (not asserted):** ours(L) ≈ unchanged base-vs-chat
value; raw weight cosine collapses under P; unrelated control stays ≈ 0.

---

## Files
- `results/laundering/summary.csv` — aggregate table (method, variant, AUROC,
  mean/min_related, mean/max_unrelated, n_pairs).
- `results/laundering/by_cell/<variant>__<method>.json` — 72 per-cell files with every
  per-pair score.
- `results/laundering/laundering_full.json` — config, seeds, full summary, per-checkpoint
  gate deviations, ours-invariance, PDFT utility, Track D1, head-sanitization count.
- `results/laundering/laundering_llm.json` — **written by Track C on the cluster.**
