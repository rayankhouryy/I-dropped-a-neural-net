"""Evaluation utilities for puzzle solutions."""
import torch


def eval_mse(
    block_list: list[tuple[int, int]],
    pieces: dict,
    last_piece: int,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Evaluate MSE for a given block ordering.

    Args:
        block_list: List of (input_piece_idx, output_piece_idx) tuples defining block order
        pieces: Dictionary mapping piece indices to weight/bias dicts
        last_piece: Index of the final (head) piece
        X: Input tensor of shape (N, 48)
        y: Target tensor of shape (N,)

    Returns:
        Mean squared error as float
    """
    h = X
    for ip, op in block_list:
        z = torch.relu(h @ pieces[ip]["weight"].T + pieces[ip]["bias"])
        h = h + z @ pieces[op]["weight"].T + pieces[op]["bias"]
    out = h @ pieces[last_piece]["weight"].T + pieces[last_piece]["bias"]
    return ((out.squeeze() - y) ** 2).mean().item()


def build_model_from_blocks(
    block_list: list[tuple[int, int]],
    pieces: dict,
    last_piece: int,
) -> callable:
    """Build a forward function from block ordering.

    Returns a callable that takes X and returns predictions.
    """
    def forward(X: torch.Tensor) -> torch.Tensor:
        h = X
        for ip, op in block_list:
            z = torch.relu(h @ pieces[ip]["weight"].T + pieces[ip]["bias"])
            h = h + z @ pieces[op]["weight"].T + pieces[op]["bias"]
        return h @ pieces[last_piece]["weight"].T + pieces[last_piece]["bias"]
    return forward
