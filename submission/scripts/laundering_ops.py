"""Checkpoint-laundering operators + function-preservation gate (AAAI-27).

All operators act ONLY on the intermediate (hidden) dimension of each residual
branch, never on the residual-stream dimension. Because ReLU is positively
homogeneous and permutations/positive-diagonal rescalings of the hidden units
cancel between W_in and W_out, every operator is EXACTLY function-preserving.

A direct corollary is that the branch product M = W_out @ W_in is left exactly
invariant:

    P:  M' = W_out P^T P W_in = W_out W_in = M     (P^T P = I)
    D:  M' = W_out D^-1 D W_in = W_out W_in = M     (D^-1 D = I)

so any M-based method (ours, and the CKA/SVCCA/IPGuard function-space methods)
is laundering-invariant for free, while a raw-weight baseline must first solve a
per-block unit alignment (+ scale) problem to recover it (see Track B).

Model (from lineage_phase1_mlp.py):
    Block:  x + W_out @ relu(W_in @ x + b_in) + b_out
    W_in  = block.inp.weight   shape (hidden, in_dim)   -- "h x d"
    W_out = block.out.weight   shape (in_dim, hidden)    -- "d x h"
    b_in  = block.inp.bias     shape (hidden,)           -- per-unit, permuted/scaled
    b_out = block.out.bias     shape (in_dim,)           -- on residual dim, untouched
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from lineage_phase1_mlp import eval_loss, train_model

# LogUniform ranges for the rescaling operator D.
LOGUNIFORM_RANGES = {
    "D-mild": (0.5, 2.0),
    "D-strong": (0.1, 10.0),
}

VARIANTS = ["P", "D-mild", "D-strong", "PD", "PDFT"]

# Sub-stream indices so P and D draw independent randomness from one base seed.
_PHASE_PERM = 0
_PHASE_RESCALE = 1

GATE_THRESHOLD = 1e-4  # float32 max |f_laundered - f_original|
N_PROBES = 512


def _rng(*ints) -> np.random.Generator:
    """Deterministic Generator seeded by a structured integer sequence."""
    return np.random.default_rng([int(x) for x in ints])


def _loguniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    """Strictly-positive LogUniform samples in [lo, hi]."""
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


# --------------------------------------------------------------------- operators

def apply_permutation(model, seed: int):
    """P: permute hidden units of every block independently (function-preserving).

    W_in <- P W_in (row permute), b_in <- P b_in, W_out <- W_out P^T (col permute).
    """
    out = copy.deepcopy(model)
    for bi, blk in enumerate(out.blocks):
        h = blk.inp.weight.shape[0]
        perm = torch.as_tensor(_rng(seed, _PHASE_PERM, bi).permutation(h),
                               dtype=torch.long)
        with torch.no_grad():
            blk.inp.weight.copy_(blk.inp.weight[perm, :])
            blk.inp.bias.copy_(blk.inp.bias[perm])
            blk.out.weight.copy_(blk.out.weight[:, perm])
    return out


def apply_rescale(model, seed: int, strength: str):
    """D: rescale hidden units by strictly-positive LogUniform factors.

    W_in <- diag(d) W_in (row scale), b_in <- d * b_in,
    W_out <- W_out diag(1/d) (col scale). ReLU-compatible because d_i > 0.
    """
    lo, hi = LOGUNIFORM_RANGES[strength]
    out = copy.deepcopy(model)
    for bi, blk in enumerate(out.blocks):
        h = blk.inp.weight.shape[0]
        d_np = _loguniform(_rng(seed, _PHASE_RESCALE, bi), lo, hi, h)
        d = torch.as_tensor(d_np, dtype=blk.inp.weight.dtype)
        with torch.no_grad():
            blk.inp.weight.mul_(d[:, None])          # scale rows of W_in
            blk.inp.bias.mul_(d)                     # scale b_in
            blk.out.weight.mul_((1.0 / d)[None, :])  # scale cols of W_out
    return out


def launder(model, variant: str, seed: int, ft_data=None,
            ft_epochs: int = 5, ft_lr: float = 3e-4):
    """Apply a laundering variant. Returns (laundered_model, pre_ft_model).

    pre_ft_model is the function-preserving stage that the correctness gate is
    run against (identical to laundered_model for P/D/PD; the pre-fine-tuning
    P+D-strong stage for PDFT). ft_data = (X, y) is required for PDFT.
    """
    if variant == "P":
        m = apply_permutation(model, seed)
        return m, m
    if variant == "D-mild":
        m = apply_rescale(model, seed, "D-mild")
        return m, m
    if variant == "D-strong":
        m = apply_rescale(model, seed, "D-strong")
        return m, m
    if variant == "PD":
        m = apply_permutation(model, seed)
        m = apply_rescale(m, seed, "D-strong")
        return m, m
    if variant == "PDFT":
        pre = apply_permutation(model, seed)
        pre = apply_rescale(pre, seed, "D-strong")
        if ft_data is None:
            raise ValueError("PDFT requires ft_data=(X, y)")
        X, y = ft_data
        ft = copy.deepcopy(pre)
        torch.manual_seed(seed + 777)  # deterministic FT minibatch order
        train_model(ft, X, y, epochs=ft_epochs, lr=ft_lr)
        return ft, pre
    raise ValueError(f"unknown variant: {variant}")


# --------------------------------------------------- function-preservation gate

def make_probes(in_dim: int, n: int = N_PROBES, seed: int = 12345) -> torch.Tensor:
    """Seeded N(0,1) probe inputs, matching make_data's input distribution."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, in_dim, generator=g)


def block_stack_output(model, X: torch.Tensor) -> torch.Tensor:
    """g(x): the residual stream after all blocks, BEFORE the `last` head.

    This is exactly the sub-function the laundering operators act on. Since no
    operator touches `model.last`, preserving g is equivalent to preserving the
    full model function f = last(g(.)); comparing g (rather than f) keeps the gate
    focused on precisely what the operators can change.
    """
    model.eval()
    with torch.no_grad():
        x = X
        for b in model.blocks:
            x = b(x)
    return x


def function_deviation(model_a, model_b, probes: torch.Tensor) -> float:
    """max_x |g_a(x) - g_b(x)| over probe inputs, computed in float32.

    Compares the block-stack output (see block_stack_output) rather than the
    final head output, so the gate measures precisely what the operators can
    affect and is not confounded by the pre-existing singleton-bias NaN.
    """
    ga = block_stack_output(model_a, probes).to(torch.float32)
    gb = block_stack_output(model_b, probes).to(torch.float32)
    return float((ga - gb).abs().max().item())


def gate_ok(deviation: float) -> bool:
    return deviation < GATE_THRESHOLD


# ------------------------------------------------- raw-weight extraction (baselines)

def raw_weights(model) -> dict:
    """Per-block RAW weights for the raw-weight baselines (Track A/B).

    Wins:  list of (h, d+1) -- W_in with b_in folded as a trailing column, so
           permutation/scaling of hidden units acts on whole rows.
    Wouts: list of (d, h)   -- W_out (columns are hidden units).

    b_out lives on the untouched residual dimension and is excluded. Ours still
    consumes the branch product M via lineage_phase1_mlp.branch_products.
    """
    Wins, Wouts = [], []
    for blk in model.blocks:
        W_in = blk.inp.weight.detach().to(torch.float32).cpu().numpy()   # (h, d)
        b_in = blk.inp.bias.detach().to(torch.float32).cpu().numpy()     # (h,)
        Wins.append(np.concatenate([W_in, b_in[:, None]], axis=1))       # (h, d+1)
        Wouts.append(blk.out.weight.detach().to(torch.float32).cpu().numpy())  # (d, h)
    return {"Wins": Wins, "Wouts": Wouts}


def utility(model, X, y) -> float:
    """Task MSE on (X, y); thin wrapper around eval_loss for the PDFT check."""
    return eval_loss(model, X, y)
