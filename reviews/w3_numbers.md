# W3 — numbers produced (add-roots only)

Scoring unchanged. Pythia uses the paper's real-LM Table-4 convention (ungated, diagonal
block-i↔block-i cosine of the centered residual signature `M = W_down·W_up`, mean over blocks).
GPT-2 uses the paper's `centered_with_gating` (Eq. 6) column. Bootstraps: 10,000 resamples, seed 42.
`NOT RUN` = not yet produced (never estimated).

## Task 1 — Pythia-160M real-LM independent-root null benchmark

Source: `experiments/scripts/pythia_seed_suite.py` (repos + commit SHAs in `sigs_pythia/manifest.json`;
full 22×22 matrix in `reviews/w3_pythia_matrix.csv`; subsets in `results/pythia_seed_suite_subsets.json`).

| Root set | N | indep. pairs | max indep-pair score | mean | conformal floor 1/(N+1) |
|---|--:|--:|--:|--:|--:|
| **Fully independent (pythia-160m + seed1–9)** | **10** | 45 | **0.000896** | 7.3×10⁻⁵ | **0.0909** |
| seed1–9 only | 9 | 36 | 0.000797 | 1.6×10⁻⁵ | 0.1000 |
| + deduped | 11 | 55 | 0.1887 | 3.6×10⁻³ | 0.0833 |
| ALL 22 (incl. partial-independence variants) | 22 | 231 | 0.3613 | 0.0547 | 0.0435 |

Reference scale: paper's descendants ≈ 0.34–1.0; paper's prior GPT-2 max null ≈ 0.004 (d=384).
The fully-independent Pythia null (d=768, L=12) max = **8.96×10⁻⁴**.

## Task 3 — confidence intervals (pair-level i.i.d. vs root-clustered/dyadic)

| Benchmark | statistic | point | pair-level i.i.d. 95% CI (width) | root-clustered 95% CI (width) |
|---|---|--:|---|---|
| Pythia indep (N=10) | max null | 8.96×10⁻⁴ | [5×10⁻⁴, 9×10⁻⁴] (3×10⁻⁴) | [4×10⁻⁴, 9×10⁻⁴] (5×10⁻⁴) |
| Pythia indep (N=10) | mean null | 7.3×10⁻⁵ | [−0.0×10⁻⁴, 2×10⁻⁴] (2×10⁻⁴) | [−1×10⁻⁴, 2×10⁻⁴] (3×10⁻⁴) |
| Pythia ALL (N=22) | mean null | 0.0547 | [0.0439, 0.0663] (0.0224) | [0.0208, 0.1013] (**0.0805**) |
| Pythia ALL (N=22) | max null | 0.3613 | [0.2605, 0.3613] (0.1008) | [0.2390, 0.3613] (0.1223) |
| GPT-2 8-root (clean) | AUROC | 1.000 | [1.000, 1.000] (0.000) | [1.000, 1.000] (0.000) |

Files: `reviews/w3_ci_pythia_independent.json`, `reviews/w3_ci_pythia_all.json`, `reviews/w3_ci_gpt2_8root.json`.

## Task 2 — GPT-2 roots, 8 → 19 (side by side)

| Setting | N roots | AUROC | Gap-Z (distilled) | max negative | min positive | pair-level CI | root-clustered CI |
|---|--:|--:|--:|--:|--:|---|---|
| Original 8-root (cached, clean) | 8 | 1.000 | 793.6 | 0.002575 | 0.85530 | [1.000,1.000] | [1.000,1.000] |
| Expanded 19-root | 19 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Expanded run is gated on the SageMaker GPU pass (reuses the 8 existing roots, trains 11 new). Its
matrix will be `reviews/w3_gpt2_19root_matrix.csv`.

## Flags (found; not fixed, per task)

1. **Pythia has no descendants** → null-only benchmark. No AUROC / min-descendant / TPR-FPR is
   computable for Pythia; the "Pythia AUROC" placeholder in `response_u7g4.md` cannot be filled from
   the seed suite. Reported statistics are max/mean of the independent-pair null + conformal floor.
2. **Only 10 of the 22 Pythia-160M repos are FULLY independent** (`pythia-160m` + `seed1–9`, which
   vary both init and data order). The elevated scores (0.15–0.36) come exclusively from Pythia's
   *seed-controlled* families: `weight-seedN` (vary init only, share data order), `data-seedN`
   (vary data order only, share init), the dropout regimes, and the `v0` re-releases — all of which
   share half their training randomness, so they are **not** valid independent-null members. Their
   scores reflect shared-randomness convergence, not descent. Cross-scores of `seedN` against every
   other group are ≈0 (≤1.8×10⁻³). The defensible real-LM null is therefore the N=10 set
   (max 8.96×10⁻⁴, floor 0.091); the N=22 numbers are reported only for completeness with this caveat.
3. **Scoring-convention split (pre-existing in the paper):** Pythia/Table-4 nulls are ungated+diagonal
   while the GPT-2 benchmark and the LLaMA-2 descendant scores are gated+Hungarian. Do NOT directly
   compare the Pythia 0.36 (ungated) against the CodeLlama descendant 0.336 (gated) — different
   estimators. (This is the "three conventions" item already in the evidence ledger.)
4. **CI width difference is only visible under score overlap.** GPT-2 clean has perfect separation
   (AUROC 1.0), so both bootstraps give [1.0,1.0] and the pair-vs-cluster width gap is 0. The gap is
   visible on the Pythia-ALL mean null: root-clustered CI (width 0.081) is **3.6× wider** than
   pair-level (0.022) — the reviewer's exact point that reusing roots understates uncertainty.
