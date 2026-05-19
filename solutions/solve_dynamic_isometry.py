"""Solver using diagonal-dominance pairing + ||W_out||_F seed + bubble-sort hill-climb.

Based on Park (2026) dynamic isometry approach.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy.optimize import linear_sum_assignment

from utils import load_pieces, load_data, get_piece_groups, eval_mse

torch.set_grad_enabled(False)


def solve():
    # Load pieces and data
    pieces = load_pieces()
    inp_pieces, out_pieces, last_piece = get_piece_groups(pieces)
    X, y_pred = load_data()
    print(f"inp={len(inp_pieces)}, out={len(out_pieces)}, last={last_piece}")
    print(f"X={tuple(X.shape)}, y={tuple(y_pred.shape)}")

    # Step 1: diagonal-dominance pairing
    print("\nStep 1: pairing via diagonal dominance ratio")
    n = 48
    ratio = torch.zeros(n, n)
    for i, ip in enumerate(inp_pieces):
        W_in = pieces[ip]["weight"]  # (96,48)
        for j, op in enumerate(out_pieces):
            W_out = pieces[op]["weight"]  # (48,96)
            M = W_out @ W_in  # (48,48)
            ratio[i, j] = abs(M.trace().item()) / (M.norm().item() + 1e-10)

    cost = (ratio.max() + 1.0 - ratio).numpy()
    row, col = linear_sum_assignment(cost)
    paired = [(inp_pieces[i], out_pieces[col[i]]) for i in range(n)]
    matched_ratios = [ratio[i, col[i]].item() for i in range(n)]
    print(f"  matched ratios: min={min(matched_ratios):.3f}, max={max(matched_ratios):.3f}, mean={sum(matched_ratios)/n:.3f}")

    # Step 2: seed order by ||W_out||_F
    print("\nStep 2: seed order by ||W_out||_F")
    order_keys = [(pieces[op]["weight"].norm().item(), k) for k, (ip, op) in enumerate(paired)]
    order_keys.sort()
    ordered = [paired[k] for _, k in order_keys]

    X_sub, y_sub = X[:1000], y_pred[:1000]
    print(f"  seed MSE (N=1000): {eval_mse(ordered, pieces, last_piece, X_sub, y_sub):.6f}")

    # Step 3: bubble-sort hill-climb
    print("\nStep 3: hill-climb")
    cur_mse = eval_mse(ordered, pieces, last_piece, X_sub, y_sub)
    for r in range(1, 30):
        swaps = 0
        for i in range(len(ordered) - 1):
            trial = list(ordered)
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            m = eval_mse(trial, pieces, last_piece, X_sub, y_sub)
            if m < cur_mse - 1e-10:
                ordered = trial
                cur_mse = m
                swaps += 1
        for i in range(len(ordered) - 2, -1, -1):
            trial = list(ordered)
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            m = eval_mse(trial, pieces, last_piece, X_sub, y_sub)
            if m < cur_mse - 1e-10:
                ordered = trial
                cur_mse = m
                swaps += 1
        print(f"  round {r}: {swaps} swaps, MSE = {cur_mse:.10f}")
        if swaps == 0:
            break

    # Gap swaps
    for gap in range(2, 6):
        for i in range(len(ordered) - gap):
            trial = list(ordered)
            trial[i], trial[i + gap] = trial[i + gap], trial[i]
            m = eval_mse(trial, pieces, last_piece, X_sub, y_sub)
            if m < cur_mse - 1e-10:
                ordered = trial
                cur_mse = m

    final_mse_full = eval_mse(ordered, pieces, last_piece, X, y_pred)
    print(f"\nFINAL full MSE: {final_mse_full:.12f}")

    # Emit permutation
    perm = []
    for ip, op in ordered:
        perm += [ip, op]
    perm.append(last_piece)
    assert len(perm) == 97 and len(set(perm)) == 97
    print("\nPERMUTATION:")
    print(",".join(map(str, perm)))

    return ordered, final_mse_full


if __name__ == "__main__":
    solve()
