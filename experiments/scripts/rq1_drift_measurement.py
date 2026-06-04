#!/usr/bin/env python3
"""
RQ1 Evidence 13: Drift Measurement Validation

Validates the theoretical diagonal drift theorem by measuring whether the
predicted first-order update matches the actual product update during training.

Theory: For linear residual block y = x + BAx where M = BA:
  - Predicted update: ΔM_pred = -η(CA^TA + BB^TC) where C = E[gx^T]
  - Under isotropy: ΔM ≈ -ηκ(α+β)I

Key metrics:
  - cos(ΔM_actual, ΔM_pred): should be ≈ 1
  - tr(ΔM_actual), tr(ΔM_pred): both should be negative
  - s(M): should increase over training

Outputs:
  results/rq1_drift_measurement.json
  figures/fig_rq1_drift_measurement.png
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --------------------------------------------------------------------- utilities

def fro_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x)


def mat_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return torch.dot(a_flat, b_flat) / (torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat) + eps)


def diag_score(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.abs(torch.trace(m)) / (fro_norm(m) + eps)


def isotropy_error(c: torch.Tensor, eps: float = 1e-12):
    """
    Measures how close C is to a scalar multiple of identity.
    Returns (iso_err, kappa) where kappa = tr(C)/d.
    """
    d = c.shape[0]
    eye = torch.eye(d, device=c.device, dtype=c.dtype)
    kappa = torch.trace(c) / d
    residual = c - kappa * eye
    return fro_norm(residual) / (fro_norm(c) + eps), kappa


def relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return fro_norm(a - b) / (fro_norm(b) + eps)


# --------------------------------------------------------------------- model

class ResidualLinearBlock(nn.Module):
    """
    Linear residual block: y = x + B(A(x))

    No activation function - matches the theory derivation exactly.
    Stores last_x and last_y for gradient access after backward().
    """

    def __init__(self, d: int, h: int, init_std: float):
        super().__init__()
        self.A = nn.Linear(d, h, bias=False)
        self.B = nn.Linear(h, d, bias=False)

        nn.init.normal_(self.A.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.B.weight, mean=0.0, std=init_std)

        self.last_x = None
        self.last_y = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_x = x
        y = x + self.B(self.A(x))
        self.last_y = y
        self.last_y.retain_grad()
        return y

    @torch.no_grad()
    def product(self) -> torch.Tensor:
        return self.B.weight @ self.A.weight


class ResidualLinearNet(nn.Module):
    def __init__(self, d: int, h: int, layers: int, init_std: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualLinearBlock(d=d, h=h, init_std=init_std)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# --------------------------------------------------------------------- metrics

@dataclass
class BlockDriftStats:
    step: int
    block: int
    loss: float

    s_before: float
    s_after: float

    trace_M_before: float
    trace_M_after: float

    trace_actual_delta: float
    trace_pred_delta: float

    scalar_drift_actual: float
    scalar_drift_pred: float

    cos_actual_pred: float

    c_kappa: float
    c_isotropy_error: float

    grad_A_relative_error: float
    grad_B_relative_error: float

    second_order_fraction: float


def measure_block_drift(
    *,
    step: int,
    block_idx: int,
    block: ResidualLinearBlock,
    A0: torch.Tensor,
    B0: torch.Tensor,
    A1: torch.Tensor,
    B1: torch.Tensor,
    lr: float,
    loss_value: float,
) -> BlockDriftStats:
    """
    Measures whether the first-order theoretical drift predicts the actual update.

    Theory for linear residual block y = x + BAx:
        C = g^T x
        grad_B = C A^T
        grad_A = B^T C
        Delta M_pred = -lr * (C A^T A + B B^T C)
    """
    x = block.last_x.detach()
    g = block.last_y.grad.detach()

    d = x.shape[1]

    C = g.T @ x

    M0 = B0 @ A0
    M1 = B1 @ A1
    actual_delta_M = M1 - M0

    pred_delta_M = -lr * (C @ (A0.T @ A0) + (B0 @ B0.T) @ C)

    dA = A1 - A0
    dB = B1 - B0
    second_order = dB @ dA

    grad_B_pred = C @ A0.T
    grad_A_pred = B0.T @ C

    grad_B_actual = block.B.weight.grad.detach()
    grad_A_actual = block.A.weight.grad.detach()

    grad_B_err = relative_error(grad_B_actual, grad_B_pred)
    grad_A_err = relative_error(grad_A_actual, grad_A_pred)

    c_iso, c_kappa = isotropy_error(C)

    scalar_actual = torch.trace(actual_delta_M) / d
    scalar_pred = torch.trace(pred_delta_M) / d

    second_order_frac = fro_norm(second_order) / (fro_norm(actual_delta_M) + 1e-12)

    return BlockDriftStats(
        step=step,
        block=block_idx,
        loss=float(loss_value),

        s_before=float(diag_score(M0)),
        s_after=float(diag_score(M1)),

        trace_M_before=float(torch.trace(M0)),
        trace_M_after=float(torch.trace(M1)),

        trace_actual_delta=float(torch.trace(actual_delta_M)),
        trace_pred_delta=float(torch.trace(pred_delta_M)),

        scalar_drift_actual=float(scalar_actual),
        scalar_drift_pred=float(scalar_pred),

        cos_actual_pred=float(mat_cos(actual_delta_M, pred_delta_M)),

        c_kappa=float(c_kappa),
        c_isotropy_error=float(c_iso),

        grad_A_relative_error=float(grad_A_err),
        grad_B_relative_error=float(grad_B_err),

        second_order_fraction=float(second_order_frac),
    )


# --------------------------------------------------------------------- experiment

def run_experiment(
    *,
    d: int = 64,
    h: int = 128,
    layers: int = 4,
    batch_size: int = 512,
    steps: int = 200,
    lr: float = 1e-2,
    init_std: float = 0.02,
    log_every: int = 10,
    seed: int = 0,
    device: str = "cuda",
    shuffle_b_gradients: bool = False,
):
    """
    Main experiment loop.

    Args:
        shuffle_b_gradients: If True, shuffle B.weight.grad across blocks before
                            optimizer step. This breaks the coupling prediction.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = ResidualLinearNet(
        d=d,
        h=h,
        layers=layers,
        init_std=init_std,
    ).to(dev)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.0,
        weight_decay=0.0,
    )

    all_stats = []

    for step in range(steps):
        x = torch.randn(batch_size, d, device=dev)

        optimizer.zero_grad(set_to_none=True)

        y = model(x)

        loss = 0.5 * (y ** 2).sum(dim=1).mean()

        A0 = [block.A.weight.detach().clone() for block in model.blocks]
        B0 = [block.B.weight.detach().clone() for block in model.blocks]

        loss.backward()

        if shuffle_b_gradients:
            b_grads = [block.B.weight.grad.detach().clone() for block in model.blocks]
            perm = torch.randperm(layers).tolist()
            for i, block in enumerate(model.blocks):
                block.B.weight.grad = b_grads[perm[i]]

        optimizer.step()

        A1 = [block.A.weight.detach().clone() for block in model.blocks]
        B1 = [block.B.weight.detach().clone() for block in model.blocks]

        if step % log_every == 0 or step == steps - 1:
            step_stats = []
            for i, block in enumerate(model.blocks):
                stats = measure_block_drift(
                    step=step,
                    block_idx=i,
                    block=block,
                    A0=A0[i],
                    B0=B0[i],
                    A1=A1[i],
                    B1=B1[i],
                    lr=lr,
                    loss_value=float(loss.item()),
                )
                all_stats.append(stats)
                step_stats.append(stats)

            mean_cos = sum(s.cos_actual_pred for s in step_stats) / len(step_stats)
            mean_trace_actual = sum(s.trace_actual_delta for s in step_stats) / len(step_stats)
            mean_trace_pred = sum(s.trace_pred_delta for s in step_stats) / len(step_stats)
            mean_s_before = sum(s.s_before for s in step_stats) / len(step_stats)
            mean_s_after = sum(s.s_after for s in step_stats) / len(step_stats)
            mean_iso = sum(s.c_isotropy_error for s in step_stats) / len(step_stats)

            mode = "shuffled" if shuffle_b_gradients else "control"
            print(
                f"[{mode}] step={step:04d} "
                f"loss={loss.item():.4f} "
                f"cos(Δ,Δ̂)={mean_cos:+.4f} "
                f"tr_ΔM={mean_trace_actual:+.2e} "
                f"tr_Δ̂M={mean_trace_pred:+.2e} "
                f"s={mean_s_before:.3f}→{mean_s_after:.3f} "
                f"C_iso={mean_iso:.3f}"
            )

    return all_stats


def aggregate_final_stats(all_stats, steps, log_every, layers):
    """Aggregate stats from the final logged step."""
    final_step = ((steps - 1) // log_every) * log_every
    if (steps - 1) % log_every != 0:
        final_step = steps - 1

    final_stats = [s for s in all_stats if s.step == final_step]

    if not final_stats:
        final_stats = all_stats[-layers:]

    return {
        "cos_actual_pred": np.mean([s.cos_actual_pred for s in final_stats]),
        "trace_actual_delta": np.mean([s.trace_actual_delta for s in final_stats]),
        "trace_pred_delta": np.mean([s.trace_pred_delta for s in final_stats]),
        "final_s": np.mean([s.s_after for s in final_stats]),
        "c_isotropy_error": np.mean([s.c_isotropy_error for s in final_stats]),
        "grad_A_rel_error": np.mean([s.grad_A_relative_error for s in final_stats]),
        "grad_B_rel_error": np.mean([s.grad_B_relative_error for s in final_stats]),
        "second_order_fraction": np.mean([s.second_order_fraction for s in final_stats]),
    }


def extract_trajectory(all_stats, layers):
    """Extract trajectory of key metrics across steps."""
    steps_seen = sorted(set(s.step for s in all_stats))
    trajectory = []
    for step in steps_seen:
        step_stats = [s for s in all_stats if s.step == step]
        trajectory.append({
            "step": step,
            "mean_cos": np.mean([s.cos_actual_pred for s in step_stats]),
            "mean_trace_actual": np.mean([s.trace_actual_delta for s in step_stats]),
            "mean_trace_pred": np.mean([s.trace_pred_delta for s in step_stats]),
            "mean_s": np.mean([s.s_after for s in step_stats]),
            "mean_c_iso": np.mean([s.c_isotropy_error for s in step_stats]),
            "loss": step_stats[0].loss,
        })
    return trajectory


# --------------------------------------------------------------------- plotting

def generate_figures(results: dict, output_path: Path):
    """Generate validation figures."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for condition in ["control", "shuffled"]:
        if condition not in results:
            continue
        traj = results[condition]["trajectory_avg"]
        steps = [t["step"] for t in traj]

        label = condition.capitalize()
        ls = "-" if condition == "control" else "--"

        ax = axes[0, 0]
        ax.plot(steps, [t["mean_trace_actual"] for t in traj], ls, label=f"{label} actual", alpha=0.8)
        ax.plot(steps, [t["mean_trace_pred"] for t in traj], ls, label=f"{label} pred", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("tr(ΔM)")
        ax.set_title("Trace of Product Update")
        ax.legend()
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

        ax = axes[0, 1]
        ax.plot(steps, [t["mean_cos"] for t in traj], ls, label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("cos(ΔM_actual, ΔM_pred)")
        ax.set_title("Cosine Similarity: Actual vs Predicted")
        ax.legend()
        ax.set_ylim([-0.1, 1.1])
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

        ax = axes[1, 0]
        ax.plot(steps, [t["mean_s"] for t in traj], ls, label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("s(M) = |tr(M)| / ||M||_F")
        ax.set_title("Diagonal Dominance Score")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(steps, [t["mean_c_iso"] for t in traj], ls, label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("C isotropy error")
        ax.set_title("Covariance Isotropy Error")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Drift Measurement Validation Experiment")
    parser.add_argument("--d", type=int, default=64, help="Residual dimension")
    parser.add_argument("--h", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir.parent

    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    config = {
        "d": args.d,
        "h": args.h,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "init_std": args.init_std,
        "log_every": args.log_every,
        "seeds": args.seeds,
        "device": args.device,
    }

    results = {"config": config}

    for condition, shuffle in [("control", False), ("shuffled", True)]:
        print(f"\n{'='*60}")
        print(f"Running {condition} condition (shuffle_b_gradients={shuffle})")
        print(f"{'='*60}\n")

        seed_results = []
        all_trajectories = []

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            stats = run_experiment(
                d=args.d,
                h=args.h,
                layers=args.layers,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                init_std=args.init_std,
                log_every=args.log_every,
                seed=seed,
                device=args.device,
                shuffle_b_gradients=shuffle,
            )

            final_agg = aggregate_final_stats(stats, args.steps, args.log_every, args.layers)
            seed_results.append(final_agg)

            traj = extract_trajectory(stats, args.layers)
            all_trajectories.append(traj)

            results[f"{condition}_seed_{seed}"] = {
                "final": final_agg,
                "trajectory": traj,
                "all_stats": [asdict(s) for s in stats],
            }

        avg_final = {
            k: {
                "mean": float(np.mean([sr[k] for sr in seed_results])),
                "std": float(np.std([sr[k] for sr in seed_results])),
            }
            for k in seed_results[0].keys()
        }
        results[f"{condition}_final_avg"] = avg_final

        n_steps = len(all_trajectories[0])
        avg_traj = []
        for i in range(n_steps):
            avg_traj.append({
                "step": all_trajectories[0][i]["step"],
                "mean_cos": float(np.mean([t[i]["mean_cos"] for t in all_trajectories])),
                "mean_trace_actual": float(np.mean([t[i]["mean_trace_actual"] for t in all_trajectories])),
                "mean_trace_pred": float(np.mean([t[i]["mean_trace_pred"] for t in all_trajectories])),
                "mean_s": float(np.mean([t[i]["mean_s"] for t in all_trajectories])),
                "mean_c_iso": float(np.mean([t[i]["mean_c_iso"] for t in all_trajectories])),
                "loss": float(np.mean([t[i]["loss"] for t in all_trajectories])),
            })
        results[condition] = {"trajectory_avg": avg_traj}

        print(f"\n{condition.upper()} final (mean ± std across {len(args.seeds)} seeds):")
        for k, v in avg_final.items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    json_path = results_dir / "rq1_drift_measurement.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    fig_path = figures_dir / "fig_rq1_drift_measurement.png"
    generate_figures(results, fig_path)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if "control_final_avg" in results and "shuffled_final_avg" in results:
        ctrl = results["control_final_avg"]
        shuf = results["shuffled_final_avg"]

        print(f"\nControl condition:")
        print(f"  cos(ΔM_actual, ΔM_pred) = {ctrl['cos_actual_pred']['mean']:.4f} ± {ctrl['cos_actual_pred']['std']:.4f}")
        print(f"  Final s(M) = {ctrl['final_s']['mean']:.4f} ± {ctrl['final_s']['std']:.4f}")
        print(f"  C isotropy error = {ctrl['c_isotropy_error']['mean']:.4f} ± {ctrl['c_isotropy_error']['std']:.4f}")
        print(f"  grad_A relative error = {ctrl['grad_A_rel_error']['mean']:.2e}")
        print(f"  grad_B relative error = {ctrl['grad_B_rel_error']['mean']:.2e}")

        print(f"\nShuffled condition:")
        print(f"  cos(ΔM_actual, ΔM_pred) = {shuf['cos_actual_pred']['mean']:.4f} ± {shuf['cos_actual_pred']['std']:.4f}")
        print(f"  Final s(M) = {shuf['final_s']['mean']:.4f} ± {shuf['final_s']['std']:.4f}")

        s_reduction = (ctrl['final_s']['mean'] - shuf['final_s']['mean']) / ctrl['final_s']['mean'] * 100
        print(f"\nFingerprint reduction from shuffling: {s_reduction:.1f}%")


if __name__ == "__main__":
    main()
