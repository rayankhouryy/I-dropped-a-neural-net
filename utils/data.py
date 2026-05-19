"""Data loading utilities for the puzzle."""
import os
import torch
import pandas as pd

PUZZLE_DIR = os.path.join(os.path.dirname(__file__), "..", "puzzle_artifacts")
PIECES_DIR = os.path.join(PUZZLE_DIR, "pieces")
DATA_CSV = os.path.join(PUZZLE_DIR, "historical_data.csv")


def load_pieces(pieces_dir: str = PIECES_DIR) -> dict[int, dict]:
    """Load all 97 puzzle pieces from disk."""
    pieces = {}
    for i in range(97):
        pieces[i] = torch.load(
            os.path.join(pieces_dir, f"piece_{i}.pth"),
            map_location="cpu",
            weights_only=True,
        )
    return pieces


def get_piece_groups(pieces: dict) -> tuple[list[int], list[int], int]:
    """Categorize pieces by shape into input, output, and last (head) pieces."""
    inp_pieces = sorted(i for i in pieces if pieces[i]["weight"].shape == torch.Size([96, 48]))
    out_pieces = sorted(i for i in pieces if pieces[i]["weight"].shape == torch.Size([48, 96]))
    last_piece = next(i for i in pieces if pieces[i]["weight"].shape == torch.Size([1, 48]))
    return inp_pieces, out_pieces, last_piece


def load_data(data_csv: str = DATA_CSV) -> tuple[torch.Tensor, torch.Tensor]:
    """Load historical data CSV and return X, y tensors."""
    df = pd.read_csv(data_csv)
    X = torch.tensor(df[[f"measurement_{i}" for i in range(48)]].values, dtype=torch.float32)
    y_pred = torch.tensor(df["pred"].values, dtype=torch.float32)
    return X, y_pred
