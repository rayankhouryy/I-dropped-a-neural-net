"""Phase 2 of Issue #30: ResNet-18 / CIFAR-10 lineage POC.

Confirms the residual-signature lineage metric on a real CNN architecture
with BatchNorm-folded BasicBlocks (W2*W1 branch product), under CPU-only
training constraints (3 epochs, batch 256).

Reference checkpoints:        2 ResNet-18 trained from scratch
Same-arch/diff-seed indep.:   4 ResNet-18 trained from scratch
Descendants per ref:          11 (cheap transforms only -- no extra training):
    - Gaussian noise sigma_rel in {0.01, 0.02, 0.05, 0.10}    (4)
    - Magnitude pruning at sparsity in {0.20, 0.40, 0.60, 0.80} (4)
    - Fake-int8 quantization at levels {16, 64, 256}            (3)

For every pair (A, B):
    1. Extract BN-folded BasicBlock weights per stage.
    2. Build branch products M_l = W2 . W1  using channel-sum projection
       (sum over the 3x3 kernel, then flatten to a (planes, planes) matrix).
    3. Compute lineage score L(A, B) via residual-signature Hungarian
       alignment (lineage_detection.py), aggregated across stages.

Outputs:
    results/lineage_phase2_resnet18_cifar.json
    figures/fig_lineage_phase2_resnet18.{png,pdf}        # via separate fig script
"""

from __future__ import annotations

import argparse, copy, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Make sibling experiments importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_detection as ldet


# ---------------------------------------------------------------- CIFAR resnet
def make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Standard CIFAR-10 ResNet-18: 3x3 stem, no initial maxpool."""
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


# ---------------------------------------------------------------- BN folding
def fold_bn_scale(conv_weight: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    eps = bn.eps
    scale = bn.weight / torch.sqrt(bn.running_var + eps)   # (C_out,)
    return conv_weight * scale.view(-1, 1, 1, 1)


def conv_channel_matrix(W: torch.Tensor) -> np.ndarray:
    """(out, in, kH, kW) -> (out, in) by summing over spatial dims."""
    return W.sum(dim=(2, 3)).detach().cpu().numpy().astype(np.float64)


def extract_branch_products(model: nn.Module) -> list[np.ndarray]:
    """For every non-downsampling BasicBlock, return W2 @ W1 (planes x planes).

    We concatenate branches across all 4 stages so the lineage matcher
    sees one ordered list of branch products per model.
    """
    Ms: list[np.ndarray] = []
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(model, stage_name)
        for block in stage:
            if getattr(block, "downsample", None) is not None:
                continue
            w1 = fold_bn_scale(block.conv1.weight, block.bn1)
            w2 = fold_bn_scale(block.conv2.weight, block.bn2)
            W1 = conv_channel_matrix(w1)        # (planes, planes)
            W2 = conv_channel_matrix(w2)        # (planes, planes)
            Ms.append(W2 @ W1)
    return Ms


# ---------------------------------------------------------------- training
def get_cifar_loaders(batch_size: int = 256, num_workers: int = 0):
    from torchvision import transforms, datasets
    tx_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tx_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = os.environ.get("CIFAR_ROOT", "data")
    Path(root).mkdir(exist_ok=True)
    tr = datasets.CIFAR10(root, train=True,  download=True, transform=tx_train)
    ev = datasets.CIFAR10(root, train=False, download=True, transform=tx_eval)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
            torch.utils.data.DataLoader(ev, batch_size=512,       shuffle=False, num_workers=num_workers))


def train_one(seed: int, epochs: int, label: str,
              loaders=None, lr: float = 0.05, init_state: dict | None = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_cifar_resnet18()
    if init_state is not None:
        model.load_state_dict(init_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if loaders is None:
        loaders = get_cifar_loaders()
    tr, ev = loaders
    t0 = time.time()
    for ep in range(epochs):
        ep_loss, n = 0.0, 0
        for x, y in tr:
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            ep_loss += float(loss) * x.size(0)
            n += x.size(0)
        sched.step()
        acc = eval_accuracy(model, ev)
        print(f"  [{label}] epoch {ep+1}/{epochs}  train_loss={ep_loss/n:.3f}  "
              f"val_acc={acc:.3f}  elapsed={time.time()-t0:.0f}s", flush=True)
    model.eval()
    return model, acc


def eval_accuracy(model: nn.Module, ev_loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in ev_loader:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total   += x.size(0)
    model.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------- transforms
def add_gaussian_noise(model: nn.Module, sigma_rel: float, seed: int) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() >= 2:
                std = p.detach().std().item() * sigma_rel
                noise = torch.randn(p.shape, generator=g).to(p.device) * std
                p.add_(noise)
    return m


def magnitude_prune(model: nn.Module, sparsity: float) -> nn.Module:
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() >= 2:
                k = int(p.numel() * sparsity)
                if k == 0:
                    continue
                flat = p.detach().abs().reshape(-1)
                threshold = flat.kthvalue(k).values
                mask = p.detach().abs() > threshold
                p.mul_(mask.float())
    return m


def fake_quantize(model: nn.Module, levels: int) -> nn.Module:
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.dim() >= 2:
                lo, hi = p.detach().min(), p.detach().max()
                scale = (hi - lo) / max(levels - 1, 1)
                q = torch.round((p.detach() - lo) / (scale + 1e-12))
                p.copy_(q * scale + lo)
    return m


# ---------------------------------------------------------------- layer-aligned scoring
def layer_aligned_lineage_score(Ms_A, Ms_B, tau_s, eps: float = 1e-12):
    """Per-block gated cosine, averaged across blocks.

    ResNet-18 branches have heterogeneous dims per stage ([64,64,128,256,512]),
    so the Hungarian matcher in lineage_detection.lineage_score (which assumes
    uniform dim and full bipartite matching) does not apply. Here the block
    index is itself part of the architecture, so we directly score block i of A
    against block i of B and average.

    Returns (lineage_score, per_block_array).
    """
    assert len(Ms_A) == len(Ms_B), "block count mismatch"
    per_block = np.zeros(len(Ms_A), dtype=np.float64)
    for k, (Ma, Mb) in enumerate(zip(Ms_A, Ms_B)):
        assert Ma.shape == Mb.shape, f"block {k} dim mismatch: {Ma.shape} vs {Mb.shape}"
        per_block[k] = ldet.gated_branch_score(Ma, Mb, tau_s, eps)
    return float(per_block.mean()), per_block


# ---------------------------------------------------------------- pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int, default=3,  help="epochs per model")
    parser.add_argument("--n_refs",    type=int, default=2)
    parser.add_argument("--n_indep",   type=int, default=4)
    parser.add_argument("--out_root",  type=str, default="results")
    parser.add_argument("--ckpt_dir",  type=str, default="checkpoints/phase2")
    parser.add_argument("--reuse_ckpts", action="store_true",
                        help="Skip training; load existing ref_i.pt / indep_j.pt from --ckpt_dir.")
    args = parser.parse_args()

    Path(args.out_root).mkdir(exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"Phase 2 ResNet-18/CIFAR-10  epochs={args.epochs}  "
          f"refs={args.n_refs}  indep={args.n_indep}  reuse={args.reuse_ckpts}",
          flush=True)

    refs, ref_accs = [], []
    indeps, indep_accs = [], []

    if args.reuse_ckpts:
        print(">>> Reusing saved checkpoints (no retraining)", flush=True)
        for i in range(args.n_refs):
            m = make_cifar_resnet18()
            m.load_state_dict(torch.load(f"{args.ckpt_dir}/ref_{i}.pt", weights_only=True))
            m.eval()
            refs.append(m); ref_accs.append(None)
        for j in range(args.n_indep):
            m = make_cifar_resnet18()
            m.load_state_dict(torch.load(f"{args.ckpt_dir}/indep_{j}.pt", weights_only=True))
            m.eval()
            indeps.append(m); indep_accs.append(None)
    else:
        print("Loading CIFAR-10 ...", flush=True)
        loaders = get_cifar_loaders()

        # ------ Train references
        for i in range(args.n_refs):
            seed = 100 + i
            print(f"\n>>> Training REFERENCE A_{i} (seed={seed})", flush=True)
            model, acc = train_one(seed, args.epochs, f"ref_{i}", loaders)
            torch.save(model.state_dict(), f"{args.ckpt_dir}/ref_{i}.pt")
            refs.append(model); ref_accs.append(acc)

        # ------ Train same-arch / different-seed independents
        for j in range(args.n_indep):
            seed = 500 + j
            print(f"\n>>> Training INDEPENDENT I_{j} (seed={seed})", flush=True)
            model, acc = train_one(seed, args.epochs, f"indep_{j}", loaders)
            torch.save(model.state_dict(), f"{args.ckpt_dir}/indep_{j}.pt")
            indeps.append(model); indep_accs.append(acc)

    # ------ Generate cheap descendants per reference
    desc_per_ref: list[list[dict]] = []
    for i, A in enumerate(refs):
        desc = []
        for sigma in [0.01, 0.02, 0.05, 0.10]:
            B = add_gaussian_noise(A, sigma, seed=2000 + i*100 + int(sigma*1000))
            desc.append({"kind": "noise", "sigma_rel": sigma, "model": B})
        for sparsity in [0.20, 0.40, 0.60, 0.80]:
            B = magnitude_prune(A, sparsity)
            desc.append({"kind": "prune", "sparsity": sparsity, "model": B})
        for levels in [16, 64, 256]:
            B = fake_quantize(A, levels)
            desc.append({"kind": "quant", "levels": levels, "model": B})
        desc_per_ref.append(desc)

    # ------ Extract branch products
    print("\n>>> Extracting BN-folded BasicBlock branch products", flush=True)
    Ms_refs   = [extract_branch_products(m) for m in refs]
    Ms_indeps = [extract_branch_products(m) for m in indeps]
    Ms_desc   = [[extract_branch_products(d["model"]) for d in dlist]
                 for dlist in desc_per_ref]
    print(f"  branches per model: {len(Ms_refs[0])}  "
          f"dims: {[M.shape[0] for M in Ms_refs[0]]}", flush=True)

    # ------ Choose tau_s from all reference branch diag-dominance scores
    tau_s = ldet.choose_tau_s(Ms_refs)
    print(f"  tau_s = {tau_s:.4f}", flush=True)

    # ------ Score every (A, B) pair
    records = []
    null_scores_per_ref = [[] for _ in refs]

    for i, Ms_A in enumerate(Ms_refs):
        # null: same-arch / diff-seed independents
        for j, Ms_B in enumerate(Ms_indeps):
            L, per_blk = layer_aligned_lineage_score(Ms_A, Ms_B, tau_s)
            null_scores_per_ref[i].append(L)
            records.append({
                "ref": i, "suspect": f"indep_{j}", "label": "non_descendant",
                "kind": "independent_same_arch",
                "score": L, "per_block": per_blk.tolist(),
            })

        # diff-ref negative: cross-reference comparison (other reference is also "non-descendant")
        for j, Ms_B in enumerate(Ms_refs):
            if i == j: continue
            L, per_blk = layer_aligned_lineage_score(Ms_A, Ms_B, tau_s)
            null_scores_per_ref[i].append(L)
            records.append({
                "ref": i, "suspect": f"ref_{j}", "label": "non_descendant",
                "kind": "other_reference",
                "score": L, "per_block": per_blk.tolist(),
            })

    # Convert null scores to z-scores, then evaluate descendants
    for i, Ms_A in enumerate(Ms_refs):
        for k, (d, Ms_B) in enumerate(zip(desc_per_ref[i], Ms_desc[i])):
            L, per_blk = layer_aligned_lineage_score(Ms_A, Ms_B, tau_s)
            z = ldet.calibrate_z_score(L, null_scores_per_ref[i])
            tag = d["kind"]
            params = {kk: v for kk, v in d.items() if kk not in ("model", "kind")}
            records.append({
                "ref": i, "suspect": f"desc_{i}_{tag}_{k}",
                "label": "descendant",
                "kind": tag,
                "params": params,
                "score": L, "z": z, "per_block": per_blk.tolist(),
            })

    # ------ Self-reference sanity
    for i, Ms_A in enumerate(Ms_refs):
        L, per_blk = layer_aligned_lineage_score(Ms_A, Ms_A, tau_s)
        records.append({"ref": i, "suspect": f"ref_{i}_self",
                        "label": "descendant", "kind": "self",
                        "score": L, "per_block": per_blk.tolist()})

    # ------ Aggregate metrics
    metrics = ldet.evaluate_lineage(records)

    # Per-attack-type detection rates (descendants only)
    by_kind: dict[str, list[float]] = {}
    for r in records:
        if r["label"] == "descendant" and r["kind"] != "self":
            by_kind.setdefault(r["kind"], []).append(r["score"])
    indep_scores = [r["score"] for r in records
                    if r["label"] == "non_descendant" and r["kind"] == "independent_same_arch"]

    summary = {
        "config": {"epochs": args.epochs, "n_refs": args.n_refs,
                    "n_indep": args.n_indep, "tau_s": tau_s},
        "ref_accuracies":   ref_accs,
        "indep_accuracies": indep_accs,
        "by_attack": {
            k: {"n": len(v), "mean": float(np.mean(v)),
                 "min":  float(np.min(v)),  "max":  float(np.max(v))}
            for k, v in by_kind.items()
        },
        "independent_same_arch_baseline": {
            "n": len(indep_scores),
            "mean": float(np.mean(indep_scores)) if indep_scores else None,
            "max":  float(np.max(indep_scores))  if indep_scores else None,
        },
        "auroc":         metrics.get("auroc"),
        "auprc":         metrics.get("auprc"),
        "tpr_at_1pct":   metrics.get("tpr_at_1pct"),
        "tpr_at_10pct":  metrics.get("tpr_at_10pct"),
    }

    out_path = Path(args.out_root) / "lineage_phase2_resnet18_cifar.json"
    out_path.write_text(json.dumps({"records": records, "summary": summary}, indent=2))
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
