"""Optimizer ablation for RQ1 (Issue #43, item 4).

Train a 48-block residual MLP on CIFAR-10 with each of:
  SGD, SGD+momentum, Adam, AdamW, RMSprop

For each optimizer record:
  - final pair accuracy (Hungarian matching on s = |tr(M)|/||M||_F)
  - emergence epoch (first epoch where pair accuracy >= 0.9)
  - eval loss

Tests whether the gradient-descent coupling mechanism that builds the
diagonal-dominance fingerprint is Adam-specific or general to any
first-order optimizer.

DEFERRED: requires GPU. Output schema documented so any GPU run drops
straight into `results/rq1_optimizer_ablation.json` without code changes.

Run:
  python experiments/scripts/rq1_optimizer_ablation.py --epochs 100 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# torchvision is the only "extra" dep here. requirements.txt must pin it.
from torchvision import datasets, transforms


N_BLOCKS = 48
D = 128
HIDDEN = 256
BATCH = 128


class ResBlock(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, h, bias=False)
        self.fc2 = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(x)))


class ResMLP(nn.Module):
    def __init__(self, n_blocks: int = N_BLOCKS, d: int = D, h: int = HIDDEN,
                 n_in: int = 32 * 32 * 3, n_out: int = 10) -> None:
        super().__init__()
        self.embed = nn.Linear(n_in, d)
        self.blocks = nn.ModuleList([ResBlock(d, h) for _ in range(n_blocks)])
        self.head = nn.Linear(d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x.flatten(1))
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def diag_dominance(M: np.ndarray) -> float:
    fro = np.linalg.norm(M)
    if fro < 1e-12:
        return 0.0
    return float(abs(np.trace(M)) / fro)


def pair_accuracy(model: ResMLP) -> float:
    blocks = list(model.blocks)
    L = len(blocks)
    score = np.zeros((L, L))
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            W_in = bi.fc1.weight.detach().cpu().numpy()
            W_out = bj.fc2.weight.detach().cpu().numpy()
            score[i, j] = diag_dominance(W_out @ W_in)
    row_ind, col_ind = linear_sum_assignment(-score)
    return float(np.mean(row_ind == col_ind))


def build_optimizer(name: str, params, lr: float = 1e-3) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def train_one(name: str, epochs: int, seed: int, data_root: str,
              device: str) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    model = ResMLP().to(device)
    opt = build_optimizer(name, model.parameters())
    history: List[Dict[str, Any]] = []
    emergence_epoch = None
    t0 = time.time()

    for ep in range(epochs + 1):
        if ep > 0:
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            losses = [F.cross_entropy(model(x.to(device)), y.to(device)).item()
                      for x, y in test_dl]
        eval_loss = float(np.mean(losses))
        pa = pair_accuracy(model)
        history.append({"epoch": ep, "eval_loss": eval_loss, "pair_acc": pa})
        if emergence_epoch is None and pa >= 0.9:
            emergence_epoch = ep
        print(f"[{name}] ep={ep:3d}  loss={eval_loss:.3f}  pair_acc={pa:.3f}",
              flush=True)

    return {
        "optimizer": name, "seed": seed, "epochs": epochs,
        "final_pair_acc": history[-1]["pair_acc"],
        "final_eval_loss": history[-1]["eval_loss"],
        "emergence_epoch": emergence_epoch,
        "history": history,
        "wall_seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="./data/cifar10")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--optimizers", nargs="+",
                    default=["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"])
    ap.add_argument("--out", default="results/rq1_optimizer_ablation.json")
    args = ap.parse_args()

    if args.device == "cpu":
        print("WARNING: running on CPU; 48-block MLP * 5 optimizers will be "
              "slow. Recommended to run on GPU.")

    results = []
    for name in args.optimizers:
        results.append(train_one(name, args.epochs, args.seed,
                                 args.data_root, args.device))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    print()
    print("Summary:")
    for r in results:
        print(f"  {r['optimizer']:<14}  final_pair_acc={r['final_pair_acc']:.3f}"
              f"  emergence_epoch={r['emergence_epoch']}"
              f"  eval_loss={r['final_eval_loss']:.3f}")


if __name__ == "__main__":
    main()
