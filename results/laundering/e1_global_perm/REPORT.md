# E1 — Global Residual-Stream Permutation Benchmark — Appendix Report

**Paper:** *Training Leaves Traces*. **Reviewer:** u7g4 (W2). **Date:** 2026-09-18.
**Question:** A global permutation `P` of the `d` residual-stream coordinates — applied
consistently to embeddings/input, output head, every residual-facing projection, and the
LayerNorm gain+bias — preserves the model function exactly but sends every branch product
`M_l -> P M_l P^T`, which scrambles the centered signature. Does it defeat the method, and
does the row/col-norm descriptor recovery restore detection?

This is **not** the intra-block hidden-unit permutation already in the paper (that leaves
`M = W_out W_in` invariant). The global `P` does not.

MLP numbers below are **measured** on this machine (CPU) by
`experiments/scripts/e1_global_perm_mlp.py`. GPT-2 numbers are produced by
`experiments/scripts/e1_global_perm_gpt2.py` on the SageMaker box (GPU + trained
artifacts); see §GPT-2. Cells contradicting the rebuttal's (a)/(b) claims are in §Discrepancies.

## Seeds (all deterministic)
- MLP bank: `torch.manual_seed(0)`, `np.random.seed(0)` (identical to Table 6 / the laundering bench).
- Attack permutations: `make_global_permutation(d, seed_base + 1000*s + i)`, `seed_base = 9000`,
  `s` = seed index (0..4), `i` = descendant/suspect index. Structured seeding via
  `np.random.default_rng`.
- Probes for the perm-aware gate: seed `12345`, `N_PROBES = 512`.

## Attack + defense implementation (new, `experiments/scripts/attack_global_perm.py`)
- `global_residual_permutation_mlp` / `apply_global_permutation_gpt2` realize `M_l -> P M_l P^T`.
  For GPT-2 a permutation is fed through the existing `apply_q_rotation_gpt2_*` tensor ops with
  `Q = P`; the LayerNorm transform `γ->Pγ, β->Pβ` is EXACT for a permutation.
- `recover_permutation`: `d x 2L` descriptor = per-coordinate row+col norms of centered `R_l`
  stacked over blocks, L2 row-normalized; Hungarian match; `align_Ms` / `permute_residual_raw`
  undo `P` before scoring (all pairs). Self-tested: exact recovery + exact `M`/raw restoration.

---

## MLP result — AUROC (52 pairs, mean ± std over 5 seeds)

Methods match paper Table 5 (`tab:laundering`): Ours | Re-Basin+scale | Al.Frob | SVD | W.Cos
(the last three are the raw-weight baselines). Positives = 30 laundered descendants; negatives =
22 independently-trained / distilled (kept clean; recovery still runs on them).

| Method | Clean | Global-`P` (no defense) | Global-`P` + recovery |
|---|---|---|---|
| **Ours (centered residual sig.)** | **1.000 ± 0.000** | **0.420 ± 0.029** | **1.000 ± 0.000** |
| Re-Basin+scale | 1.000 ± 0.000 | 0.593 ± 0.025 | 1.000 ± 0.000 |
| Aligned Frobenius (raw) | 1.000 ± 0.000 | 0.855 ± 0.020 | 1.000 ± 0.000 |
| SVD distance (raw) | 1.000 ± 0.000 | **1.000 ± 0.000** | 1.000 ± 0.000 |
| Weight cosine (raw) | 1.000 ± 0.000 | 0.980 ± 0.017 | 1.000 ± 0.000 |

**Function-preservation gate (perm-aware):** max block-stack deviation over all laundered
descendants × seeds = **4.77e-06** (mean 2.88e-06), threshold `1e-4` → **PASS**. The permutation
is exactly function-preserving up to input relabeling; the residual is pure fp32 summation-order
roundoff.

**Recovery:** exact-coordinate fraction = **1.0000**; **30 / 30** descendants exactly recovered
in every seed (all `d = 16` coordinates). Negatives after blind recovery stay unrelated
(`max_unrelated` unchanged; no false positives). Wall time 235 s (bank build 223 s + 5-seed
sweep 12 s), CPU.

**Headline (MLP).** The global `P` collapses ours from 1.000 to **0.420** (≈chance) and drops
Re-Basin+scale to 0.593 and aligned-Frobenius to 0.855 — confirming the attack defeats the
published method and the alignment baselines. The row/col-norm recovery restores **every**
method to 1.000. SVD distance is invariant (see §Discrepancies #2).

---

## GPT-2 result — AUROC (45 pairs) — RUN ON SAGEMAKER

No GPU on the dev-dsk; the trained paper benchmark (pickles + `checkpoints/` + `models/`, 17 GB)
lives on the SageMaker box. No retraining needed — E1 is post-hoc. Run:

```bash
cd ~/I-dropped-a-neural-net/experiments/scripts
python e1_global_perm_gpt2.py \
    --benchmark-dir results/lineage_benchmark_gpt2_paper_v2 \
    --epochs 3 --seeds 5
```

Expected runtime: **a few minutes** (numpy scoring on cached `Ms` + ~15 gate forward passes on
GPU). Output: `results/laundering/e1_global_perm/gpt2/{summary.csv, by_cell/, e1_full.json}`.
Fill the table below from `gpt2/summary.csv` (rows `variant=…, seed=mean/std`). The pure-numpy
attack/recover/score path is unit-tested; only the gate needs the GPU.

| Method | Clean | Global-`P` (no defense) | Global-`P` + recovery |
|---|---|---|---|
| **Ours** | _pending_ | _pending_ | _pending_ |
| Re-Basin+scale | _pending_ | _pending_ | _pending_ |
| Aligned Frobenius (raw) | _pending_ | _pending_ | _pending_ |
| SVD distance (raw) | _pending_ | _pending_ | _pending_ |
| Weight cosine (raw) | _pending_ | _pending_ | _pending_ |

Gate max |Δlogit|: _pending_ (expect ~1e-6–1e-5, PASS). Recovery: exact `n`/21 descendants: _pending_.

---

## Discrepancies (measured vs the rebuttal's claimed (a)/(b) values)

1. **Attack collapse — CONFIRMED qualitatively; exact value is benchmark-specific.** The rebuttal
   (a) reports GPT-2 ours AUROC 1.000 → **0.631**. On the MLP, ours collapses to **0.420 ± 0.029**
   (≈chance). The *collapse to near-chance is robust*; the exact figure differs by benchmark and is
   not pinned. Confirm the GPT-2 0.631 from the SageMaker run.

2. **SVD distance does NOT collapse under Global-`P` — reported honestly.** Singular values are
   invariant under the orthogonal similarity `P M P^T` (and under permuting rows/cols of the raw
   weights), so raw SVD distance holds at **1.000** across all three conditions. This is a genuine
   property, not a bug: it echoes rebuttal (d) — spectral/invariant statistics survive the
   permutation but are otherwise weak discriminators. One baseline column legitimately does not move.

3. **Recovery restores AUROC to 1.000 — CONFIRMED on MLP (stronger than the GPT-2 claim).** Rebuttal
   (b) reports **18 / 21** exact recovery on GPT-2 (`d = 384`). On the MLP (`d = 16`) recovery is
   **30 / 30** exact every seed and AUROC returns to 1.000 for all methods. Recovery is expected to
   be *harder* at `d = 384` (only `2L = 12` descriptor features per coordinate → some ties), so the
   GPT-2 "18/21" and possibly sub-1.000 restoration for aggressively-pruned pairs is plausible;
   confirm from the SageMaker run.

4. **Descendant score `~0.002` (rebuttal a) is GPT-2-specific.** Not directly measurable on the MLP
   (different `d`, different score scale). The MLP shows the analogous collapse via the AUROC drop.

5. **No false positives from blind recovery — CONFIRMED (MLP).** Recovery is applied to all pairs;
   negatives' `max_unrelated` is unchanged and no negative crosses a positive → AUROC 1.000 after
   recovery. The norm-based (not signature-value-based) descriptor avoids null inflation as intended.

---

## Files
- `results/laundering/e1_global_perm/mlp/summary.csv` — per (method, condition, seed) + mean/std
- `results/laundering/e1_global_perm/mlp/by_cell/<condition>__<method>.json` — 15 per-cell files
- `results/laundering/e1_global_perm/mlp/e1_full.json` — config, seeds, gate, per-seed AUROC, recovery
- `results/laundering/e1_global_perm/gpt2/…` — produced by the SageMaker run (same schema)
- Code: `experiments/scripts/{attack_global_perm.py, e1_global_perm_mlp.py, e1_global_perm_gpt2.py}`
