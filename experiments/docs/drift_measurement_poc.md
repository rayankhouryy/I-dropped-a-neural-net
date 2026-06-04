# POC: Drift Measurement Experiment

This proof-of-concept measures whether the first-order theoretical drift predicted by coupled gradient descent matches the actual update to the residual-branch product.

For a linear residual block,

\[
y = x + BAx,
\]

where

\[
A = W_{\text{in}}, \qquad B = W_{\text{out}}, \qquad M = BA,
\]

the predicted first-order SGD update is

\[
\widehat{\Delta M}
=
-\eta
\left[
C A^\top A + BB^\top C
\right],
\]

where

\[
C = \mathbb{E}[gx^\top], \qquad g = \frac{\partial \mathcal{L}}{\partial y}.
\]

The experiment compares this predicted update against the actual update

\[
\Delta M = M_{t+1} - M_t.
\]

## Metrics Measured

The POC reports:

1. Actual update:

\[
\Delta M = M_{t+1} - M_t
\]

2. Predicted first-order update:

\[
\widehat{\Delta M}
=
-\eta
\left[
C A^\top A + BB^\top C
\right]
\]

3. Cosine similarity:

\[
\cos(\Delta M, \widehat{\Delta M})
\]

4. Trace drift:

\[
\operatorname{tr}(\Delta M), \qquad \operatorname{tr}(\widehat{\Delta M})
\]

5. Isotropy error of \(C\):

\[
\frac{\left\|C - \frac{\operatorname{tr}(C)}{d}I\right\|_F}{\|C\|_F}
\]

6. Fingerprint score:

\[
s(M) = \frac{|\operatorname{tr}(M)|}{\|M\|_F}
\]

## Expected Healthy Pattern

A successful run should show:

- \(\cos(\Delta M, \widehat{\Delta M})\) close to \(+1\)
- \(\operatorname{tr}(\Delta M) < 0\)
- \(\operatorname{tr}(\widehat{\Delta M}) < 0\)
- \(s(M)\) increasing over training
- low-to-moderate isotropy error for \(C\)
- gradient formula sanity checks near numerical precision

The strongest evidence is not merely that \(s(M)\) increases. The key validation is:

\[
\cos(\Delta M, \widehat{\Delta M}) \approx 1
\]

and

\[
\operatorname{tr}(\Delta M) \approx \operatorname{tr}(\widehat{\Delta M}) < 0.
\]

That directly validates the drift equation.

---

## Script: `drift_measurement_poc.py`

```python
#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn


# -----------------------------
# Utilities
# -----------------------------

def fro_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x)


def mat_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + eps)


def diag_score(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.abs(torch.trace(m)) / (fro_norm(m) + eps)


def isotropy_error(c: torch.Tensor, eps: float = 1e-12):
    """
    Measures how close C is to a scalar multiple of identity.

    iso_err = ||C - kI||_F / ||C||_F
    k       = tr(C) / d
    """
    d = c.shape[0]
    eye = torch.eye(d, device=c.device, dtype=c.dtype)
    kappa = torch.trace(c) / d
    residual = c - kappa * eye
    return fro_norm(residual) / (fro_norm(c) + eps), kappa


def relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return fro_norm(a - b) / (fro_norm(b) + eps)


# -----------------------------
# Model
# -----------------------------

class ResidualLinearBlock(nn.Module):
    """
    Residual block:

        y = x + B A x

    A: d -> h
    B: h -> d

    We store the block input x and block output y so that after backward()
    we can access g = dL/dy via y.grad.
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


# -----------------------------
# Drift measurement
# -----------------------------

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

    where g = dL/dy and M = BA.
    """

    x = block.last_x.detach()
    g = block.last_y.grad.detach()

    d = x.shape[1]

    # Important: do NOT divide by batch size.
    # PyTorch's g already contains the reduction scaling from the loss.
    C = g.T @ x

    M0 = B0 @ A0
    M1 = B1 @ A1
    actual_delta_M = M1 - M0

    # First-order theory prediction under vanilla SGD.
    pred_delta_M = -lr * (C @ (A0.T @ A0) + (B0 @ B0.T) @ C)

    # Exact first-order decomposition from actual parameter deltas.
    dA = A1 - A0
    dB = B1 - B0
    second_order = dB @ dA

    # Gradient formula sanity check:
    # For y = x + BAx:
    #   grad_B = C A^T
    #   grad_A = B^T C
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
    device: str = "cpu",
):
    torch.manual_seed(seed)

    model = ResidualLinearNet(
        d=d,
        h=h,
        layers=layers,
        init_std=init_std,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.0,
        weight_decay=0.0,
    )

    all_stats = []

    for step in range(steps):
        # Whitened synthetic inputs.
        # Target is zero, so early in training g roughly aligns with x,
        # giving C = E[gx^T] with positive identity component.
        x = torch.randn(batch_size, d, device=device)

        optimizer.zero_grad(set_to_none=True)

        y = model(x)

        # 1/2 ||y||^2 objective.
        # This makes residual branches learn to cancel the identity map.
        loss = 0.5 * (y ** 2).sum(dim=1).mean()

        # Snapshot parameters before update.
        A0 = [block.A.weight.detach().clone() for block in model.blocks]
        B0 = [block.B.weight.detach().clone() for block in model.blocks]

        loss.backward()

        optimizer.step()

        # Snapshot parameters after update.
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

            print(
                f"step={step:04d} "
                f"loss={loss.item():.4f} "
                f"cos(actual,pred)={mean_cos:+.4f} "
                f"tr_dM_actual={mean_trace_actual:+.4e} "
                f"tr_dM_pred={mean_trace_pred:+.4e} "
                f"s={mean_s_before:.4f}->{mean_s_after:.4f} "
                f"C_iso_err={mean_iso:.4f}"
            )

    return all_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_experiment(
        d=args.d,
        h=args.h,
        layers=args.layers,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        init_std=args.init_std,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

---

## Run Command

```bash
python drift_measurement_poc.py --steps 200 --d 64 --h 128 --layers 4 --lr 1e-2
```

---

## Paper Reporting Template

Use a table like this in the paper:

| Quantity | Control | Shuffled-\(B\)-Gradient |
|---|---:|---:|
| Mean \(\cos(\Delta M, \widehat{\Delta M})\) | high | lower |
| Mean \(\operatorname{tr}(\Delta M)\) | strongly negative | less negative |
| Mean final \(s(M)\) | high | reduced |
| Mean \(C\)-isotropy error | low/moderate | similar |
| Mean \(\kappa = \operatorname{tr}(C)/d\) | positive | positive |
| Pair accuracy | high | reduced |
| Test accuracy | matched or reported |

The strongest plot is:

\[
\operatorname{tr}(\Delta M_t)
\quad \text{vs.} \quad
\operatorname{tr}(\widehat{\Delta M}_t)
\]

over training steps.

A second useful plot is:

\[
s(M_t)
\quad \text{vs.} \quad
\sum_{\tau \le t}
-\operatorname{tr}(\widehat{\Delta M}_\tau).
\]

This tests whether accumulated predicted negative drift explains the fingerprint score.
