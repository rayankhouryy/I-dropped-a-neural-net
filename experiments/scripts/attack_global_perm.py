"""Global residual-stream coordinate-permutation attack + recovery defense (E1).

Reviewer u7g4 (W2) flagged a function-preserving laundering transform the paper
never tests: a SINGLE permutation ``P`` of the ``d`` residual-stream coordinates,
applied consistently to embeddings/input, the output head, every residual-facing
projection, and the LayerNorm gain+bias. It preserves the model function exactly
but transforms every branch product ``M_l -> P M_l P^T``, which scrambles the
centered signature ``R_l = M_l - (tr(M_l)/d) I`` and destroys cosine similarity.

This is NOT the intra-block hidden-unit permutation in laundering_ops.apply_permutation
/ laundering_gpt2_ops.apply_permutation_gpt2 (those leave M = W_out W_in invariant).
The global P does NOT leave M invariant.

Permutation convention (used throughout):
    make_global_permutation(d, seed) -> (P, perm) with P = eye(d)[perm], i.e.
    (P v)_i = v_{perm[i]}. For the row-vector nn.Linear convention a residual
    x_row maps to (P x)_row = x_row @ P.T.

Attack:
    global_residual_permutation_mlp(model, P)  -- ResNet (no embedding/LayerNorm)
    apply_global_permutation_gpt2(model, seed)  -- HF GPT2LMHeadModel (reuses the
        laundering_q_rotation block funcs with Q = P; for a permutation the
        LayerNorm transform gamma->P gamma, beta->P beta is EXACT).

Defense (recover P, undo before scoring):
    coordinate_descriptor(Ms) -> (d, 2L) row/col-norms of centered R_l, row-normed
    recover_permutation(Ms_ref, Ms_sus) -> col_ind aligning suspect coords to ref
    align_Ms(Ms, col_ind) / permute_residual_raw_{mlp,gpt2}(raw, col_ind) -- undo P
    exact_recovery_fraction(col_ind, perm) -- fraction of coords exactly recovered

GPT-2 imports (torch/transformers/laundering_q_rotation) are lazy so the MLP path
stays light. Run ``python attack_global_perm.py`` for the numeric self-tests.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------- permutation matrix

def make_global_permutation(d: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P, perm): P = eye(d)[perm] (float64), so (P v)_i = v_{perm[i]}.

    Deterministic, structured seeding matching laundering_ops._rng conventions.
    """
    perm = np.random.default_rng([int(seed)]).permutation(d)
    P = np.eye(d, dtype=np.float64)[perm]
    return P, perm


# ---------------------------------------------------------------- MLP attack

def global_residual_permutation_mlp(model, P: np.ndarray):
    """Apply one global residual permutation P to a ResNet (in a fresh copy).

    Per block:  inp.weight <- inp.weight @ P^T   (read the permuted residual Px)
                out.weight <- P @ out.weight     (write into the permuted basis)
                out.bias   <- P @ out.bias        (on the residual stream)
                inp.bias   unchanged              (per-hidden-unit)
    Head:       last.weight <- last.weight @ P^T
                last.bias    unchanged

    Yields M_l -> P M_l P^T for every block. The MLP has no embedding, so the
    function is preserved UP TO input relabeling: f'(Px) = f(x).
    """
    import torch  # local: keep the numpy-only path import-light

    out = copy.deepcopy(model)
    d = P.shape[0]
    dtype = out.blocks[0].inp.weight.dtype
    Pt_t = torch.as_tensor(P.T, dtype=dtype)
    P_t = torch.as_tensor(P, dtype=dtype)
    with torch.no_grad():
        for blk in out.blocks:
            assert blk.inp.weight.shape[1] == d, "P dim must equal residual in_dim"
            blk.inp.weight.copy_(blk.inp.weight @ Pt_t)   # (h,d) @ (d,d)
            blk.out.weight.copy_(P_t @ blk.out.weight)    # (d,d) @ (d,h)
            blk.out.bias.copy_(P_t @ blk.out.bias)        # (d,d) @ (d,)
        out.last.weight.copy_(out.last.weight @ Pt_t)     # (1,d) @ (d,d)
    return out


def function_deviation_global_perm_mlp(perm_model, orig_model, P: np.ndarray,
                                       probes) -> float:
    """Perm-aware gate: max_x || g'(Px) - P g(x) ||_inf over probes, in fp32.

    g = block-stack output (residual before the head). Because the global P
    permutes the residual basis, feeding Px to the laundered model yields
    g'(Px) = P g(x); we compare against P g(x). ~fp32 roundoff (<< 1e-4).
    """
    import torch
    import laundering_ops as lops

    dtype = probes.dtype
    Pt_t = torch.as_tensor(P.T, dtype=dtype)   # (Px)_row = x_row @ P.T
    P_t = torch.as_tensor(P, dtype=dtype)
    g_perm = lops.block_stack_output(perm_model, probes @ Pt_t).to(torch.float32)
    g_ref = lops.block_stack_output(orig_model, probes).to(torch.float32)
    g_ref_perm = g_ref @ P_t.to(torch.float32).T   # (P g)_row = g_row @ P.T
    return float((g_perm - g_ref_perm).abs().max().item())


# ---------------------------------------------------------------- GPT-2 attack

def apply_global_permutation_gpt2(model, seed: int, device: str = "cpu"):
    """Apply one global residual permutation P to an HF GPT2LMHeadModel (copy).

    Reuses laundering_q_rotation's per-tensor application with Q = P. A permutation
    trivially fixes the all-ones vector, and gamma->P gamma / beta->P beta is EXACT
    for a permutation (unlike a dense rotation), so the transform is exactly
    function-preserving on identical token inputs. Returns (perm_model, manifest).
    """
    import torch
    import laundering_q_rotation as lqr

    out = copy.deepcopy(model)
    d = out.config.n_embd
    P_np, perm = make_global_permutation(d, seed)
    wdtype = out.transformer.wte.weight.dtype
    P = torch.from_numpy(P_np).to(wdtype).to(device)
    Pt = P.T
    lqr.apply_q_rotation_gpt2_embeddings(out, P)
    for blk in out.transformer.h:
        lqr.apply_q_rotation_gpt2_block_attn(blk, P, Pt)
        lqr.apply_q_rotation_gpt2_block_mlp(blk, P, Pt)
    lqr.apply_q_rotation_gpt2_layernorms(out, P)
    manifest = {"seed": int(seed), "d_model": int(d), "perm": perm.tolist()}
    return out, manifest


# ---------------------------------------------------------------- defense

def _centered(M: np.ndarray) -> np.ndarray:
    """R = M - (tr(M)/d) I, raw scale (NOT unit-normalized)."""
    M = np.asarray(M, dtype=np.float64)
    d = M.shape[0]
    return M - (np.trace(M) / d) * np.eye(d, dtype=np.float64)


def coordinate_descriptor(Ms: Sequence[np.ndarray]) -> np.ndarray:
    """Per-coordinate row/col norms of centered R_l stacked over blocks.

    Returns a (d, 2L) matrix, L2 row-normalized. Permutation-EQUIVARIANT: if
    M_l -> P M_l P^T for all l then descriptor row i moves to perm-inverse of i
    (a pure row permutation), so Hungarian matching on it recovers P.
    """
    d = np.asarray(Ms[0]).shape[0]
    L = len(Ms)
    F = np.empty((d, 2 * L), dtype=np.float64)
    for l, M in enumerate(Ms):
        R = _centered(M)
        F[:, l] = np.linalg.norm(R, axis=1)       # row norm of coord i
        F[:, L + l] = np.linalg.norm(R, axis=0)   # col norm of coord i
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return F


def recover_permutation(Ms_ref: Sequence[np.ndarray],
                        Ms_sus: Sequence[np.ndarray]) -> np.ndarray:
    """Recover the coordinate permutation aligning the suspect to the reference.

    Returns col_ind (length d): reference coordinate i is matched to suspect
    coordinate col_ind[i]. If Ms_sus == [P M P^T for M in Ms_ref] exactly, then
    col_ind == perm^{-1} == argsort(perm), and align_Ms(Ms_sus, col_ind) == Ms_ref.
    Cost is cosine (rows are already unit-norm, so C = F_A @ F_B^T).
    """
    F_A = coordinate_descriptor(Ms_ref)
    F_B = coordinate_descriptor(Ms_sus)
    C = F_A @ F_B.T
    row_ind, col_ind = linear_sum_assignment(-C)   # maximize; row_ind == arange(d)
    return col_ind


def align_Ms(Ms: Sequence[np.ndarray], col_ind: np.ndarray) -> List[np.ndarray]:
    """Undo the residual permutation on branch products: P M P^T -> M."""
    idx = np.ix_(col_ind, col_ind)
    return [np.asarray(M, dtype=np.float64)[idx] for M in Ms]


def permute_residual_raw(raw: Dict, index: np.ndarray) -> Dict:
    """Reindex the residual dimension of raw weights by ``index``.

    Works for BOTH the MLP bank (Wins (h, d+1), Wouts (d, h)) and GPT-2
    (Wins (d_ff, d_model+1), Wouts (d_model, d_ff)); the residual dim d is read
    from Wouts.shape[0] and the trailing bias column of Wins is left untouched.

    A global residual permutation P permutes the d residual columns of W_in and
    the d rows of W_out. Pass ``index = perm`` to APPLY the attack and
    ``index = col_ind`` (= perm^{-1}) to UNDO it:
        Wins[:, :d] <- Wins[:, :d][:, index];  Wouts <- Wouts[index, :].
    """
    Wins, Wouts = [], []
    for Win, Wout in zip(raw["Wins"], raw["Wouts"]):
        Win = np.asarray(Win, dtype=np.float64).copy()
        d = np.asarray(Wout).shape[0]
        Win[:, :d] = Win[:, :d][:, index]
        Wins.append(Win)
        Wouts.append(np.asarray(Wout, dtype=np.float64)[index, :].copy())
    return {"Wins": Wins, "Wouts": Wouts}


# Backwards-compatible alias (the MLP driver imports this name).
permute_residual_raw_mlp = permute_residual_raw


def exact_recovery_fraction(col_ind: np.ndarray, perm: np.ndarray) -> float:
    """Fraction of coordinates correctly recovered (col_ind == perm^{-1})."""
    inv = np.argsort(perm)
    return float(np.mean(np.asarray(col_ind) == inv))


# ------------------------------------------------------------------- self-test

def _selftest() -> None:
    rng = np.random.default_rng(0)
    d, L = 16, 8
    Ms = [rng.standard_normal((d, d)) for _ in range(L)]

    P, perm = make_global_permutation(d, seed=123)
    Ms_att = [P @ M @ P.T for M in Ms]

    # 1) descriptor is a pure row permutation under the attack.
    F = coordinate_descriptor(Ms)
    F_att = coordinate_descriptor(Ms_att)
    assert np.allclose(F_att, F[perm], atol=1e-9), "descriptor not equivariant"

    # 2) recovery is exact and undoes the attack on branch products.
    col_ind = recover_permutation(Ms, Ms_att)
    assert exact_recovery_fraction(col_ind, perm) == 1.0, "perm not fully recovered"
    Ms_rec = align_Ms(Ms_att, col_ind)
    err = max(float(np.abs(a - b).max()) for a, b in zip(Ms_rec, Ms))
    assert err < 1e-9, f"align_Ms did not restore M (err={err:.2e})"

    # 3) MLP attack realizes M -> P M P^T and is function-preserving up to input P.
    import torch
    import lineage_phase1_mlp as p1
    import laundering_ops as lops
    torch.manual_seed(0)
    model = p1.ResNet(in_dim=d, hidden_dim=24, depth=L)
    Ms_clean = p1.branch_products(model)
    pmodel = global_residual_permutation_mlp(model, P)
    Ms_perm = p1.branch_products(pmodel)
    perr = max(float(np.abs(mp - (P @ mc @ P.T)).max())
               for mp, mc in zip(Ms_perm, Ms_clean))
    assert perr < 1e-4, f"MLP attack != P M P^T (err={perr:.2e})"
    probes = lops.make_probes(d, n=64, seed=999)
    gate = function_deviation_global_perm_mlp(pmodel, model, P, probes)
    assert gate < 1e-4, f"MLP function-preservation gate failed (dev={gate:.2e})"

    # 4) raw-weight de-permutation restores clean raw weights.
    raw = lops.raw_weights(model)
    raw_att = lops.raw_weights(pmodel)
    raw_rec = permute_residual_raw_mlp(raw_att, col_ind)
    rerr = max(float(np.abs(a - b).max())
               for a, b in zip(raw_rec["Wins"], raw["Wins"]))
    rerr = max(rerr, max(float(np.abs(a - b).max())
                         for a, b in zip(raw_rec["Wouts"], raw["Wouts"])))
    assert rerr < 1e-4, f"raw de-permutation did not restore weights (err={rerr:.2e})"

    print(f"[selftest] OK  d={d} L={L}  recover=1.000  "
          f"align_err={err:.1e}  MLP_M_err={perr:.1e}  gate={gate:.1e}  raw_err={rerr:.1e}")


if __name__ == "__main__":
    _selftest()
