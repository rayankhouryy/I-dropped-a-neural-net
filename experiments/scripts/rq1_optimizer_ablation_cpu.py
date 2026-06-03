"""CPU-feasible variant of rq1_optimizer_ablation.py (Issue #43, item 4).

Reduced configuration to make the sweep tractable on CPU:
  - depth 24 instead of 48 (still matches paper's depth-24 examples)
  - 5000-sample CIFAR-10 train subset (vs 50000)
  - 1000-sample test subset
  - 1 seed
  - 30 epochs default

Still tests all 5 optimizers: SGD, SGD+momentum, Adam, AdamW, RMSprop.
Tracks final pair accuracy, emergence epoch (pair_acc >= 0.9), final eval loss,
final mean diagonal-dominance s, and mean fraction negative trace.
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


N_BLOCKS = 24
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


def block_metrics(model: ResMLP) -> Dict[str, float]:
    blocks = list(model.blocks)
    L = len(blocks)
    score = np.zeros((L, L))
    diag_s = np.zeros(L)
    neg_trace = 0
    for i, bi in enumerate(blocks):
        Wi = bi.fc1.weight.detach().cpu().numpy()
        for j, bj in enumerate(blocks):
            Wo = bj.fc2.weight.detach().cpu().numpy()
            M = Wo @ Wi
            score[i, j] = diag_dominance(M)
            if i == j:
                diag_s[i] = score[i, j]
                if np.trace(M) < 0:
                    neg_trace += 1
    row_ind, col_ind = linear_sum_assignment(-score)
    return {
        "pair_acc": float(np.mean(row_ind == col_ind)),
        "mean_diag_s": float(np.mean(diag_s)),
        "frac_neg_trace": float(neg_trace / L),
    }


def build_optimizer(name: str, params, lr: float = 1e-3) -> torch.optim.Optimizer:
    n = name.lower()
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if n == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if n == "adam":
        return torch.optim.Adam(params, lr=lr)
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def train_one(name: str, epochs: int, seed: int, train_dl, test_dl,
              device: str) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        m = block_metrics(model)
        row = {"epoch": ep, "eval_loss": eval_loss, **m}
        history.append(row)
        if emergence_epoch is None and m["pair_acc"] >= 0.9:
            emergence_epoch = ep
        print(f"[{name}] ep={ep:3d}  loss={eval_loss:.3f}  "
              f"pair_acc={m['pair_acc']:.3f}  s={m['mean_diag_s']:.3f}  "
              f"neg={m['frac_neg_trace']:.2f}", flush=True)

    return {
        "optimizer": name, "seed": seed, "epochs": epochs,
        "final_pair_acc": history[-1]["pair_acc"],
        "final_eval_loss": history[-1]["eval_loss"],
        "final_mean_diag_s": history[-1]["mean_diag_s"],
        "final_frac_neg_trace": history[-1]["frac_neg_trace"],
        "emergence_epoch": emergence_epoch,
        "history": history,
        "wall_seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="If set, runs each optimizer with each seed (multi-seed sweep).")
    ap.add_argument("--train-n", type=int, default=5000)
    ap.add_argument("--test-n", type=int, default=1000)
    ap.add_argument("--data-root", default="./data/cifar10")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--optimizers", nargs="+",
                    default=["sgd", "sgd_momentum", "adam", "adamw", "rmsprop"])
    ap.add_argument("--out", default="results/rq1_optimizer_ablation_cpu.json")
    args = ap.parse_args()

    tfm = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
    full_test = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(full_train), size=args.train_n, replace=False).tolist()
    test_idx = rng.choice(len(full_test), size=args.test_n, replace=False).tolist()
    train_ds = Subset(full_train, train_idx)
    test_ds = Subset(full_test, test_idx)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    seeds = args.seeds if args.seeds else [args.seed]
    results = []
    for name in args.optimizers:
        for seed in seeds:
            results.append(train_one(name, args.epochs, seed,
                                     train_dl, test_dl, args.device))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump({
                "config": {
                    "n_blocks": N_BLOCKS, "d": D, "hidden": HIDDEN,
                    "epochs": args.epochs, "seeds": seeds,
                    "train_n": args.train_n, "test_n": args.test_n,
                    "device": args.device, "batch": BATCH, "lr": 1e-3,
                    "note": "CPU-feasible reduced-scale optimizer ablation"
                },
                "runs": results
            }, open(args.out, "w"), indent=2)

    # Aggregate
    by_opt: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_opt.setdefault(r["optimizer"], []).append(r)
    print()
    print("Summary (mean +/- std across seeds):")
    for name, runs in by_opt.items():
        pa = np.array([r["final_pair_acc"] for r in runs])
        s = np.array([r["final_mean_diag_s"] for r in runs])
        nt = np.array([r["final_frac_neg_trace"] for r in runs])
        ee = [r["emergence_epoch"] for r in runs]
        ll = np.array([r["final_eval_loss"] for r in runs])
        print(f"  {name:<14}  pair_acc={pa.mean():.3f}+/-{pa.std():.3f}  "
              f"s={s.mean():.3f}+/-{s.std():.3f}  "
              f"neg={nt.mean():.2f}  "
              f"emerge_ep={ee}  "
              f"loss={ll.mean():.3f}")


if __name__ == "__main__":
    main()
