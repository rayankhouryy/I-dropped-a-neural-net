"""Phase 1 of Issue #30: synthetic residual-MLP lineage detection POC.

Trains a controlled bestiary of depth-24 residual MLPs and tests whether
the residual-signature lineage metric (lineage_detection.py) separates:

  - DESCENDANTS of a reference checkpoint:
      same-target fine-tune
      different-target fine-tune
      Gaussian weight noise
      magnitude pruning
      fake int8 quantization (round to 256 levels per matrix)

  - NON-DESCENDANTS (independent training, same architecture):
      same-target, different seed
      different-target, different seed
      random initialization (no training)
      distilled student (trained to mimic reference outputs)

Per Aman's spec (#30), the residual-signature score should separate
descendants from all non-descendants; the diagonal-dominance-only
baseline should fail on the same-target/different-seed negatives because
both descendant and non-descendant models exhibit similar diagonal
dominance after training.

Output:
  results/lineage_phase1_mlp.json
  figures/fig_lineage_score_distributions.{png,pdf}
  figures/fig_lineage_roc.{png,pdf}
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lineage_detection as ldet


# --------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return x + self.out(F.relu(self.inp(x)))


class ResNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(in_dim, hidden_dim) for _ in range(depth)])
        self.last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.last(x)


# --------------------------------------------------------------------- data
def synthetic_target(X, in_dim, key):
    g = torch.Generator().manual_seed(key)
    A = torch.randn(in_dim, 8, generator=g) * 0.5
    B = torch.randn(8, generator=g)
    bias = torch.randn(1, generator=g)
    h = torch.tanh(X @ A)
    return h @ B + bias


def make_data(in_dim, n=4000, seed=0, target_key=None):
    if target_key is None:
        target_key = seed + 1234
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    y = synthetic_target(X, in_dim, key=target_key)
    return X, y


# --------------------------------------------------------------------- training
def train_model(model, X, y, epochs=200, lr=1e-3, batch=256, grad_clip=1.0,
                soft_targets=None):
    """If soft_targets is provided, use MSE against the soft targets
    (distillation); otherwise use MSE against y."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_train = X.shape[0]
    final_loss = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        n_seen = 0
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            yp = model(X[idx]).squeeze(-1)
            target = soft_targets[idx] if soft_targets is not None else y[idx]
            loss = loss_fn(yp, target)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item() * idx.numel()
            n_seen += idx.numel()
        final_loss = ep_loss / n_seen
    return final_loss


def fresh_model(depth, hidden, in_dim, seed):
    torch.manual_seed(seed)
    return ResNet(in_dim=in_dim, hidden_dim=hidden, depth=depth)


def eval_loss(model, X, y):
    model.eval()
    with torch.no_grad():
        return float(F.mse_loss(model(X).squeeze(-1), y).item())


# --------------------------------------------------------------------- attacks
def descendant_fine_tune(parent, X, y, epochs=50, lr=3e-4):
    """Continue training from parent on same target."""
    child = copy.deepcopy(parent)
    train_model(child, X, y, epochs=epochs, lr=lr)
    return child


def descendant_finetune_new_target(parent, X, y_new, epochs=50, lr=3e-4):
    """Fine-tune on a different target (more aggressive drift)."""
    child = copy.deepcopy(parent)
    train_model(child, X, y_new, epochs=epochs, lr=lr)
    return child


def descendant_noise(parent, sigma_rel=0.02, seed=0):
    """Add Gaussian noise scaled by per-weight magnitude."""
    child = copy.deepcopy(parent)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in child.parameters():
            # std() Bessel-corrects (÷ n-1), so on a 1-element tensor (the head
            # bias) it returns NaN; fall back to |value| as the noise scale for
            # singletons. Multi-element params keep the exact p.std().item()
            # path, so every non-singleton stays bit-identical to prior banks.
            spread = p.detach().std() if p.numel() > 1 else p.detach().abs()
            std = spread.item() * sigma_rel + 1e-12
            p.add_(torch.randn(p.shape, generator=g) * std)
    return child


def descendant_prune(parent, sparsity=0.5, seed=0):
    """Magnitude pruning: zero out the smallest |w| entries."""
    child = copy.deepcopy(parent)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in child.parameters():
            if p.dim() < 2:
                continue   # skip biases
            flat = p.detach().abs().reshape(-1)
            k = int(sparsity * flat.numel())
            if k == 0:
                continue
            thresh = torch.topk(flat, k, largest=False).values.max()
            mask = (p.detach().abs() > thresh).to(p.dtype)
            p.mul_(mask)
    return child


def descendant_quantize(parent, levels=256):
    """Fake int8 quantization: uniform-quantize each tensor to ``levels``."""
    child = copy.deepcopy(parent)
    with torch.no_grad():
        for p in child.parameters():
            lo, hi = p.min().item(), p.max().item()
            if hi - lo < 1e-12:
                continue
            scale = (hi - lo) / (levels - 1)
            q = torch.round((p - lo) / scale)
            p.copy_(q * scale + lo)
    return child


def nondesc_distilled_student(parent, X, epochs=200, lr=1e-3, seed=999,
                              depth=24, hidden=64, in_dim=24):
    """Train a fresh-init student to match parent outputs on X."""
    parent.eval()
    with torch.no_grad():
        y_soft = parent(X).detach().squeeze(-1)
    student = fresh_model(depth, hidden, in_dim, seed)
    train_model(student, X, y_soft, epochs=epochs, lr=lr,
                soft_targets=y_soft)
    return student


# --------------------------------------------------------------------- branch products
def branch_products(model):
    """M_l = W_out @ W_in (in fp32) for every block in the model."""
    Ms = []
    for blk in model.blocks:
        W_in = blk.inp.weight.detach().to(torch.float32).cpu().numpy()
        W_out = blk.out.weight.detach().to(torch.float32).cpu().numpy()
        Ms.append(W_out @ W_in)
    return Ms


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-refs', type=int, default=3,
                    help='number of independent reference checkpoints')
    ap.add_argument('--n-per-descendant-type', type=int, default=5,
                    help='descendant count per (ref, attack-type)')
    ap.add_argument('--n-same-arch-diff-seed', type=int, default=15)
    ap.add_argument('--n-diff-task', type=int, default=5)
    ap.add_argument('--n-distilled', type=int, default=5)
    ap.add_argument('--depth', type=int, default=24)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--in-dim', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--ft-epochs', type=int, default=50)
    ap.add_argument('--out', default='results/lineage_phase1_mlp.json')
    args = ap.parse_args()

    Path('results').mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    t0 = time.time()
    # Training data is shared by all reference checkpoints (same task,
    # different seeds). Differents-task negatives use a different key.
    X_main, y_main = make_data(in_dim=args.in_dim, n=4000, seed=0,
                                target_key=42)

    # --- Train reference models
    refs = []
    for r in range(args.n_refs):
        m = fresh_model(args.depth, args.hidden, args.in_dim, seed=100 + r)
        loss = train_model(m, X_main, y_main, epochs=args.epochs)
        refs.append({'id': f'ref-{r}', 'model': m,
                     'loss': loss, 'eval_loss': eval_loss(m, X_main, y_main)})
        print(f"[ref-{r}] trained ({time.time()-t0:.1f}s)  loss={loss:.4f}",
              flush=True)

    # --- Generate descendants per reference
    descendants_records = []   # {id, ref_id, attack_type, model, utility}
    attack_specs = [
        ('finetune_same',   {'epochs': args.ft_epochs, 'lr': 3e-4}),
        ('finetune_diff',   {'epochs': args.ft_epochs, 'lr': 3e-4}),
        ('noise',           {'sigma_rel_list': [0.01, 0.02, 0.04, 0.08, 0.15]}),
        ('prune',           {'sparsity_list': [0.10, 0.30, 0.50, 0.70, 0.85]}),
        ('quantize',        {'levels_list':   [256, 128, 64, 32, 16]}),
    ]

    for ref_idx, ref in enumerate(refs):
        seed_offset = 200 + ref_idx * 1000
        for atk_type, spec in attack_specs:
            if atk_type == 'finetune_same':
                for k in range(args.n_per_descendant_type):
                    g = torch.Generator().manual_seed(seed_offset + k)
                    Xk = X_main + torch.randn(X_main.shape, generator=g) * 0.05
                    yk = synthetic_target(Xk, args.in_dim, key=42)
                    child = descendant_fine_tune(ref['model'], Xk, yk,
                                                  epochs=spec['epochs'],
                                                  lr=spec['lr'])
                    descendants_records.append({
                        'id': f'desc-{ref_idx}-{atk_type}-{k}',
                        'ref_id': ref['id'],
                        'attack_type': atk_type,
                        'model': child,
                        'utility': eval_loss(child, X_main, y_main),
                    })
            elif atk_type == 'finetune_diff':
                for k in range(args.n_per_descendant_type):
                    yk = synthetic_target(X_main, args.in_dim,
                                          key=99 + k * 11)
                    child = descendant_finetune_new_target(
                        ref['model'], X_main, yk,
                        epochs=spec['epochs'], lr=spec['lr'])
                    descendants_records.append({
                        'id': f'desc-{ref_idx}-{atk_type}-{k}',
                        'ref_id': ref['id'],
                        'attack_type': atk_type,
                        'model': child,
                        'utility': eval_loss(child, X_main, y_main),
                    })
            elif atk_type == 'noise':
                for k, sigma in enumerate(spec['sigma_rel_list']
                                          [:args.n_per_descendant_type]):
                    child = descendant_noise(ref['model'], sigma_rel=sigma,
                                              seed=seed_offset + k)
                    descendants_records.append({
                        'id': f'desc-{ref_idx}-{atk_type}-{k}',
                        'ref_id': ref['id'],
                        'attack_type': atk_type,
                        'sigma_rel': sigma,
                        'model': child,
                        'utility': eval_loss(child, X_main, y_main),
                    })
            elif atk_type == 'prune':
                for k, sp in enumerate(spec['sparsity_list']
                                       [:args.n_per_descendant_type]):
                    child = descendant_prune(ref['model'], sparsity=sp,
                                              seed=seed_offset + k)
                    descendants_records.append({
                        'id': f'desc-{ref_idx}-{atk_type}-{k}',
                        'ref_id': ref['id'],
                        'attack_type': atk_type,
                        'sparsity': sp,
                        'model': child,
                        'utility': eval_loss(child, X_main, y_main),
                    })
            elif atk_type == 'quantize':
                for k, lv in enumerate(spec['levels_list']
                                       [:args.n_per_descendant_type]):
                    child = descendant_quantize(ref['model'], levels=lv)
                    descendants_records.append({
                        'id': f'desc-{ref_idx}-{atk_type}-{k}',
                        'ref_id': ref['id'],
                        'attack_type': atk_type,
                        'levels': lv,
                        'model': child,
                        'utility': eval_loss(child, X_main, y_main),
                    })
        print(f"[ref-{ref_idx}] descendants done ({time.time()-t0:.1f}s)",
              flush=True)

    # --- Non-descendants (same arch, different seed, same task)
    nondesc_records = []
    for k in range(args.n_same_arch_diff_seed):
        m = fresh_model(args.depth, args.hidden, args.in_dim, seed=500 + k)
        train_model(m, X_main, y_main, epochs=args.epochs)
        nondesc_records.append({
            'id': f'nondesc-sameseed-{k}',
            'attack_type': 'independent_same_task',
            'model': m,
            'utility': eval_loss(m, X_main, y_main),
        })
        print(f"[nondesc same-task {k}] ({time.time()-t0:.1f}s)", flush=True)

    for k in range(args.n_diff_task):
        Xk, yk = make_data(in_dim=args.in_dim, n=4000, seed=700 + k,
                            target_key=900 + k * 13)
        m = fresh_model(args.depth, args.hidden, args.in_dim, seed=700 + k)
        train_model(m, Xk, yk, epochs=args.epochs)
        nondesc_records.append({
            'id': f'nondesc-difftask-{k}',
            'attack_type': 'independent_diff_task',
            'model': m,
            'utility': eval_loss(m, Xk, yk),
        })
        print(f"[nondesc diff-task {k}] ({time.time()-t0:.1f}s)", flush=True)

    # Distilled students of each reference (trained on parent outputs)
    for r_idx, ref in enumerate(refs):
        n_per_ref = max(1, args.n_distilled // args.n_refs)
        for k in range(n_per_ref):
            student = nondesc_distilled_student(
                ref['model'], X_main, epochs=args.epochs, lr=1e-3,
                seed=900 + r_idx * 100 + k,
                depth=args.depth, hidden=args.hidden, in_dim=args.in_dim)
            nondesc_records.append({
                'id': f'nondesc-distill-{r_idx}-{k}',
                'attack_type': 'distilled_student',
                'parent_ref_id': ref['id'],
                'model': student,
                'utility': eval_loss(student, X_main, y_main),
            })
            print(f"[nondesc distill ref-{r_idx} k={k}] ({time.time()-t0:.1f}s)",
                  flush=True)

    # Random-init (untrained) baselines
    for k in range(5):
        m = fresh_model(args.depth, args.hidden, args.in_dim, seed=1000 + k)
        nondesc_records.append({
            'id': f'nondesc-randinit-{k}',
            'attack_type': 'random_init',
            'model': m,
            'utility': eval_loss(m, X_main, y_main),
        })

    # --- Compute branch products for everyone, drop models from RAM
    print(f"\nTotal training time: {time.time()-t0:.1f}s")
    print("Computing branch products...", flush=True)
    for r in refs:
        r['Ms'] = branch_products(r['model'])
        del r['model']
    for d in descendants_records:
        d['Ms'] = branch_products(d['model'])
        del d['model']
    for n in nondesc_records:
        n['Ms'] = branch_products(n['model'])
        del n['model']

    # tau_s from reference branches
    tau_s = ldet.choose_tau_s([r['Ms'] for r in refs])
    print(f"\ntau_s = {tau_s:.4f}")

    # --- Score every (reference, suspect) pair
    metrics_per_pair = []
    for ref_idx, ref in enumerate(refs):
        # Null distribution for this reference: all non-descendants
        # except distilled-of-this-ref (those are non-desc-of-everyone, but
        # using them in their own ref's null would bias).
        per_ref_nulls = []
        # Score the descendants of this reference
        for d in descendants_records:
            if d['ref_id'] != ref['id']:
                continue
            L_score, _, _ = ldet.lineage_score(ref['Ms'], d['Ms'], tau_s)
            metrics_per_pair.append({
                'reference': ref['id'],
                'suspect':   d['id'],
                'label':     'descendant',
                'attack_type': d['attack_type'],
                'utility':   d.get('utility'),
                'lineage':   L_score,
                'diag_only': float(np.mean([ldet.diag_score(M) for M in d['Ms']])),
                'raw_cos':   ldet.raw_cos_score(ref['Ms'], d['Ms']),
                'frob_dist': ldet.frob_distance(ref['Ms'], d['Ms']),
                **{k: d[k] for k in ['sigma_rel', 'sparsity', 'levels']
                   if k in d},
            })

        # Score the non-descendants and build null
        for n in nondesc_records:
            L_score, _, _ = ldet.lineage_score(ref['Ms'], n['Ms'], tau_s)
            per_ref_nulls.append(L_score)
            metrics_per_pair.append({
                'reference': ref['id'],
                'suspect':   n['id'],
                'label':     'non_descendant',
                'attack_type': n['attack_type'],
                'utility':   n.get('utility'),
                'lineage':   L_score,
                'diag_only': float(np.mean([ldet.diag_score(M) for M in n['Ms']])),
                'raw_cos':   ldet.raw_cos_score(ref['Ms'], n['Ms']),
                'frob_dist': ldet.frob_distance(ref['Ms'], n['Ms']),
                **{k: n[k] for k in ['parent_ref_id'] if k in n},
            })

        # Calibrate descendants vs this reference's null
        for entry in metrics_per_pair:
            if entry['reference'] != ref['id']:
                continue
            entry['z_lineage'] = ldet.calibrate_z_score(
                entry['lineage'], per_ref_nulls)
        print(f"[ref-{ref_idx}] scored {len(per_ref_nulls)} nulls + descendants "
              f"({time.time()-t0:.1f}s)", flush=True)

    # --- Aggregate AUROC etc. across all (reference, suspect) pairs
    overall = {}
    for score_key in ['lineage', 'diag_only', 'raw_cos', 'frob_dist',
                      'z_lineage']:
        rs = [{'label': e['label'], 'score': e.get(score_key, 0.0)}
              for e in metrics_per_pair
              if score_key in e]
        if not rs:
            continue
        overall[score_key] = ldet.evaluate_lineage(rs)
        m = overall[score_key]
        print(f"[{score_key:>10s}] AUROC={m['auroc']:.4f}  "
              f"AUPRC={m['auprc']:.4f}  TPR@1%FPR={m['tpr_at_1pct']:.0%}  "
              f"TPR@10%FPR={m['tpr_at_10pct']:.0%}")

    # --- Per-attack-type detection rate at the threshold matching 1% FPR
    # of the GLOBAL non-descendant population
    nondesc_scores = sorted([e['lineage'] for e in metrics_per_pair
                              if e['label'] == 'non_descendant'])
    fpr_1pct_thresh = (nondesc_scores[-max(1, int(0.01 * len(nondesc_scores)))]
                       if nondesc_scores else 1.0)
    fpr_10pct_thresh = (nondesc_scores[-max(1, int(0.10 * len(nondesc_scores)))]
                        if nondesc_scores else 1.0)
    per_attack = {}
    for atk in set(e['attack_type'] for e in metrics_per_pair):
        scores = [e['lineage'] for e in metrics_per_pair
                  if e['attack_type'] == atk]
        label = next(e['label'] for e in metrics_per_pair
                     if e['attack_type'] == atk)
        per_attack[atk] = {
            'label': label,
            'n': len(scores),
            'mean_lineage': float(np.mean(scores)),
            'std_lineage':  float(np.std(scores)),
            'detected_at_1pct_FPR':  int(sum(s >= fpr_1pct_thresh for s in scores)),
            'detected_at_10pct_FPR': int(sum(s >= fpr_10pct_thresh for s in scores)),
        }

    out = {
        'config': vars(args),
        'tau_s':  tau_s,
        'overall_metrics': overall,
        'per_attack_summary': per_attack,
        'thresholds': {
            'global_1pct_FPR':  float(fpr_1pct_thresh),
            'global_10pct_FPR': float(fpr_10pct_thresh),
        },
        'pairs': [{k: v for k, v in e.items() if k != 'Ms'}
                  for e in metrics_per_pair],
        'total_seconds': time.time() - t0,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")

    print("\n=== HEADLINE ===")
    print(f"  lineage AUROC = {overall['lineage']['auroc']:.4f}")
    print(f"  diag_only AUROC (should be lower if metric is real) = "
          f"{overall['diag_only']['auroc']:.4f}")
    print(f"  z_lineage  AUROC = {overall['z_lineage']['auroc']:.4f}")
    print("\n  Per-attack lineage-score detection at global 1%/10% FPR:")
    for atk, s in sorted(per_attack.items(),
                          key=lambda kv: -kv[1]['mean_lineage']):
        print(f"   {atk:<26s}  label={s['label']:<15s}  "
              f"n={s['n']:>3d}  mean_L={s['mean_lineage']:+.3f}  "
              f"det@1%={s['detected_at_1pct_FPR']}/{s['n']}  "
              f"det@10%={s['detected_at_10pct_FPR']}/{s['n']}")


if __name__ == '__main__':
    main()
