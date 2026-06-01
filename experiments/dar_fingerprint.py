"""Issue #40: does the diagonal-dominance fingerprint survive cross-layer routing
(DAR-style softmax-weighted aggregation) instead of standard x + f(x) residual?

DAR (Xu et al., 2026, arXiv:2605.20708) replaces the standard residual update
    h_{l+1} = h_l + f_l(h_l)
with a softmax-weighted aggregation over previous sublayer outputs
    v_i = f_i(h_i),  k_i = RMSNorm(v_i),
    alpha_{i->l} = softmax_i( q_l . k_i / sqrt(d) ),
    h_l = sum_{i < l} alpha_{i->l} * v_i.

Our paper's fingerprint mechanism relies on the per-block identity path
forcing W_out W_in -> -eps I + E through dynamic-isometry pressure. DAR
removes the per-block identity path and replaces it with a learnable router
*between* blocks. Aman's question (#40): does the per-block W_out W_in
fingerprint survive this?

This script trains matched residual-MLP and DAR-MLP populations on the same
synthetic regression task and measures, for each population:
  - mean diagonal-dominance score s(M) = |tr(M)| / ||M||_F on correct pairs
  - mean diagonal score on off-diagonal pairs
  - Hungarian block-pair accuracy (within-model)
  - fraction of correct pairs with negative trace (the dynamic-isometry signature)
  - lineage AUROC (descendants vs same-arch independents) using the
    centered residual-signature score from lineage_detection.py

Output: results/dar_fingerprint.json + a printed summary table.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lineage_detection as ldet


# --------------------------------------------------------------------- models
class StandardBlock(nn.Module):
    """Standard residual MLP block: h_{l+1} = h_l + W_out ReLU(W_in h_l)."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, in_dim)

    def sublayer(self, x):
        return self.out(F.relu(self.inp(x)))

    def forward(self, x):
        return x + self.sublayer(x)


class StandardResNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StandardBlock(in_dim, hidden_dim) for _ in range(depth)])
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class DARNet(nn.Module):
    """DAR-style cross-layer-routed MLP.

    Each block exposes its sublayer (W_out ReLU(W_in)) just like the standard
    block, so branch_products() can still be defined as W_out @ W_in. The
    difference is the inter-block aggregation: the input to sublayer l is a
    softmax-weighted combination of all previous sublayer outputs, not a
    cumulative sum.

    This is the *static* DAR variant from the paper (no timestep injection,
    since we are not training a diffusion model), instantiated for an MLP
    stack rather than a transformer stack. The point is to isolate the
    cross-layer routing change, not to reproduce a DiT.
    """
    def __init__(self, in_dim, hidden_dim, depth, out_dim=1):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.blocks = nn.ModuleList(
            [StandardBlock(in_dim, hidden_dim) for _ in range(depth)])
        # one learnable query vector per block (static DAR; the paper notes
        # "static" matches "dynamic" on LLMs and is the simpler parameterisation)
        self.queries = nn.Parameter(torch.randn(depth, in_dim) * 0.02)
        self.head = nn.Linear(in_dim, out_dim)

    @staticmethod
    def _rmsnorm(v, eps=1e-6):
        # RMSNorm over the per-token feature dim (last axis)
        rms = v.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
        return v / rms

    def forward(self, x):
        # source list: v_0 is the input embedding, v_l for l >= 1 is the
        # output of sublayer l-1. This matches Eq. (5) of the DAR paper.
        v_sources = [x]  # v_0 = h_0
        d = float(self.in_dim)
        for l, blk in enumerate(self.blocks):
            stacked = torch.stack(v_sources, dim=0)        # [S, ..., in_dim]
            keys = self._rmsnorm(stacked)                  # [S, ..., in_dim]
            q = self.queries[l]                             # [in_dim]
            # broadcast: scores = q . k_i / sqrt(d) along the feature axis,
            # producing one scalar per source per sample.
            logits = (keys * q).sum(dim=-1) / (d ** 0.5)    # [S, ...]
            alpha = torch.softmax(logits, dim=0)            # [S, ...]
            # weighted sum over sources -> aggregated h_l
            h_l = (alpha.unsqueeze(-1) * stacked).sum(dim=0)
            v_l = blk.sublayer(h_l)
            v_sources.append(v_l)
        # final hidden state: same softmax-routed aggregation over the full
        # source set, then the linear head.
        stacked = torch.stack(v_sources, dim=0)
        keys = self._rmsnorm(stacked)
        # one final query (we just reuse the last one for the head input)
        q = self.queries[-1]
        logits = (keys * q).sum(dim=-1) / (d ** 0.5)
        alpha = torch.softmax(logits, dim=0)
        h_final = (alpha.unsqueeze(-1) * stacked).sum(dim=0)
        return self.head(h_final)


# --------------------------------------------------------------------- data
def synthetic_target(X, in_dim, key):
    g = torch.Generator().manual_seed(key)
    A = torch.randn(in_dim, 8, generator=g) * 0.5
    B = torch.randn(8, generator=g)
    bias = torch.randn(1, generator=g)
    return torch.tanh(X @ A) @ B + bias


def make_data(in_dim, n=4000, seed=0, target_key=42):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    y = synthetic_target(X, in_dim, key=target_key)
    return X, y


# --------------------------------------------------------------------- train
def train_model(model, X, y, epochs=200, lr=1e-3, batch=256, grad_clip=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    final = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        s_loss, s_n = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            yp = model(X[idx]).squeeze(-1)
            loss = F.mse_loss(yp, y[idx])
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            s_loss += loss.item() * idx.numel()
            s_n += idx.numel()
        final = s_loss / s_n
    return final


def eval_loss(model, X, y):
    model.eval()
    with torch.no_grad():
        return float(F.mse_loss(model(X).squeeze(-1), y).item())


# --------------------------------------------------------------------- attacks
def attack_noise(parent, sigma_rel=0.05, seed=0):
    child = copy.deepcopy(parent)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in child.parameters():
            std = p.detach().std().item() * sigma_rel + 1e-12
            p.add_(torch.randn(p.shape, generator=g) * std)
    return child


def attack_prune(parent, sparsity=0.3, seed=0):
    child = copy.deepcopy(parent)
    with torch.no_grad():
        for p in child.parameters():
            if p.dim() < 2:
                continue
            flat = p.detach().abs().reshape(-1)
            k = int(sparsity * flat.numel())
            if k == 0:
                continue
            thresh = torch.topk(flat, k, largest=False).values.max()
            mask = (p.detach().abs() > thresh).to(p.dtype)
            p.mul_(mask)
    return child


def attack_quant(parent, levels=64):
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


# --------------------------------------------------------------------- fingerprint metrics
def branch_products(model):
    Ms = []
    for blk in model.blocks:
        W_in = blk.inp.weight.detach().to(torch.float32).cpu().numpy()
        W_out = blk.out.weight.detach().to(torch.float32).cpu().numpy()
        Ms.append(W_out @ W_in)
    return Ms


def diag_score(M, eps=1e-12):
    return float(abs(np.trace(M))) / (float(np.linalg.norm(M, 'fro')) + eps)


def cross_score_matrix(Ws_in, Ws_out):
    """s(i, j) = |tr(W_out_j W_in_i)| / ||W_out_j W_in_i||_F."""
    L = len(Ws_in)
    S = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            M = Ws_out[j] @ Ws_in[i]
            S[i, j] = diag_score(M)
    return S


def block_pair_acc(model):
    """Hungarian on the cross-score matrix; fraction of correct (i, i) pairs."""
    from scipy.optimize import linear_sum_assignment
    Ws_in = [b.inp.weight.detach().to(torch.float32).cpu().numpy()
             for b in model.blocks]
    Ws_out = [b.out.weight.detach().to(torch.float32).cpu().numpy()
              for b in model.blocks]
    S = cross_score_matrix(Ws_in, Ws_out)
    row, col = linear_sum_assignment(-S)  # max
    acc = float(np.mean(row == col))
    diag = float(np.mean(np.diag(S)))
    off = float((S.sum() - np.trace(S)) / (S.size - S.shape[0]))
    neg_trace = float(np.mean([
        np.trace(Ws_out[i] @ Ws_in[i]) < 0 for i in range(len(Ws_in))]))
    return {'pair_acc': acc, 'mean_diag_s': diag, 'mean_off_s': off,
            'neg_trace_frac': neg_trace, 'separation': diag - off}


# --------------------------------------------------------------------- per-arch run
def run_arch(arch_name, model_cls, args, X, y, t0):
    print(f"\n========== {arch_name} ==========", flush=True)

    # train refs and indeps
    refs, indeps = [], []
    for r in range(args.n_refs):
        torch.manual_seed(100 + r)
        m = model_cls(in_dim=args.in_dim, hidden_dim=args.hidden,
                       depth=args.depth)
        loss = train_model(m, X, y, epochs=args.epochs)
        ev = eval_loss(m, X, y)
        refs.append({'id': f'{arch_name}-ref-{r}', 'model': m,
                     'loss': loss, 'eval_loss': ev})
        print(f"  [ref-{r}] loss={loss:.4f}  eval={ev:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    for r in range(args.n_indeps):
        torch.manual_seed(500 + r)
        m = model_cls(in_dim=args.in_dim, hidden_dim=args.hidden,
                       depth=args.depth)
        loss = train_model(m, X, y, epochs=args.epochs)
        ev = eval_loss(m, X, y)
        indeps.append({'id': f'{arch_name}-indep-{r}', 'model': m,
                       'loss': loss, 'eval_loss': ev})
        print(f"  [indep-{r}] loss={loss:.4f}  eval={ev:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # fingerprint metrics on each trained ref
    ref_fps = [block_pair_acc(r['model']) for r in refs]
    indep_fps = [block_pair_acc(r['model']) for r in indeps]
    fps = ref_fps + indep_fps
    print(f"  mean diag-s on correct pairs:    "
          f"{np.mean([f['mean_diag_s'] for f in fps]):.4f}", flush=True)
    print(f"  mean diag-s on off-diag pairs:   "
          f"{np.mean([f['mean_off_s'] for f in fps]):.4f}", flush=True)
    print(f"  Hungarian block-pair accuracy:   "
          f"{np.mean([f['pair_acc'] for f in fps]):.4f}", flush=True)
    print(f"  negative-trace fraction:         "
          f"{np.mean([f['neg_trace_frac'] for f in fps]):.4f}", flush=True)

    # descendants: apply 3 attacks at 3 strengths each, per ref
    desc_records = []
    attack_grid = [
        ('noise', [0.02, 0.05, 0.10], 'sigma_rel'),
        ('prune', [0.20, 0.50, 0.70], 'sparsity'),
        ('quant', [128, 64, 32], 'levels'),
    ]
    for ri, ref in enumerate(refs):
        for atk, strengths, key in attack_grid:
            for k, val in enumerate(strengths):
                if atk == 'noise':
                    child = attack_noise(ref['model'], sigma_rel=val,
                                          seed=1000 + ri*100 + k)
                elif atk == 'prune':
                    child = attack_prune(ref['model'], sparsity=val,
                                          seed=1000 + ri*100 + k)
                else:
                    child = attack_quant(ref['model'], levels=val)
                ev = eval_loss(child, X, y)
                desc_records.append({
                    'id': f'{arch_name}-desc-{ri}-{atk}-{k}',
                    'ref_id': ref['id'], 'attack': atk, key: val,
                    'model': child, 'utility': ev,
                })

    # compute residual-signature lineage scores
    ref_Ms = {r['id']: branch_products(r['model']) for r in refs}
    indep_Ms = [(r['id'], branch_products(r['model'])) for r in indeps]
    desc_Ms = [(d['ref_id'], d['id'], d['attack'], branch_products(d['model']),
                d['utility']) for d in desc_records]

    # tau_s = min diag_score across all reference branches
    tau_s = min(ldet.diag_score(M) for Ms in ref_Ms.values() for M in Ms)

    # descendant scores: each descendant against its OWN reference
    desc_scores = []
    for ref_id, did, atk, Ms_B, utility in desc_Ms:
        s, _, _ = ldet.lineage_score(ref_Ms[ref_id], Ms_B, tau_s=tau_s)
        desc_scores.append({'id': did, 'ref_id': ref_id, 'attack': atk,
                            'score': s, 'utility': utility, 'label': 1})

    # non-descendant scores: each ref vs each indep
    indep_scores = []
    for ref_id, ref_M in ref_Ms.items():
        for indep_id, indep_M in indep_Ms:
            s, _, _ = ldet.lineage_score(ref_M, indep_M, tau_s=tau_s)
            indep_scores.append({'id': f'{ref_id}|{indep_id}',
                                 'ref_id': ref_id,
                                 'attack': 'indep',
                                 'score': s, 'utility': None, 'label': 0})

    # AUROC
    all_scores = [r['score'] for r in desc_scores + indep_scores]
    all_labels = [r['label'] for r in desc_scores + indep_scores]
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(all_labels, all_scores))
    except Exception:
        auroc = None

    print(f"  tau_s = {tau_s:.4f}", flush=True)
    print(f"  descendants  (n={len(desc_scores)}): "
          f"mean L={np.mean([r['score'] for r in desc_scores]):.4f}  "
          f"min={np.min([r['score'] for r in desc_scores]):.4f}", flush=True)
    print(f"  independents (n={len(indep_scores)}): "
          f"mean L={np.mean([r['score'] for r in indep_scores]):.4f}  "
          f"max={np.max([r['score'] for r in indep_scores]):.4f}", flush=True)
    print(f"  AUROC = {auroc}", flush=True)

    return {
        'arch': arch_name,
        'tau_s': tau_s,
        'eval_loss_refs': [r['eval_loss'] for r in refs],
        'eval_loss_indeps': [r['eval_loss'] for r in indeps],
        'fingerprint': {
            'mean_diag_s_correct': float(np.mean([f['mean_diag_s'] for f in fps])),
            'mean_diag_s_off':     float(np.mean([f['mean_off_s'] for f in fps])),
            'separation':          float(np.mean([f['separation'] for f in fps])),
            'pair_acc':            float(np.mean([f['pair_acc'] for f in fps])),
            'neg_trace_frac':      float(np.mean([f['neg_trace_frac'] for f in fps])),
            'per_model': fps,
        },
        'lineage': {
            'auroc': auroc,
            'desc_mean': float(np.mean([r['score'] for r in desc_scores])),
            'desc_min':  float(np.min([r['score'] for r in desc_scores])),
            'indep_mean': float(np.mean([r['score'] for r in indep_scores])),
            'indep_max':  float(np.max([r['score'] for r in indep_scores])),
            'desc_records': [{k: v for k, v in r.items()} for r in desc_scores],
            'indep_records': [{k: v for k, v in r.items()} for r in indep_scores],
        },
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=24)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--in-dim', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--n-refs', type=int, default=3)
    ap.add_argument('--n-indeps', type=int, default=3)
    ap.add_argument('--out', default='results/dar_fingerprint.json')
    args = ap.parse_args()

    Path('results').mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, y = make_data(in_dim=args.in_dim, n=4000, seed=0, target_key=42)

    results = {}
    for name, cls in [('standard', StandardResNet), ('dar', DARNet)]:
        results[name] = run_arch(name, cls, args, X, y, t0)

    # summary table
    print("\n=================== SUMMARY ===================")
    print(f"{'arch':<10}  {'pair_acc':>9}  {'diag_corr':>9}  "
          f"{'diag_off':>9}  {'neg_trace':>9}  {'AUROC':>7}")
    for name in ('standard', 'dar'):
        r = results[name]
        fp = r['fingerprint']
        au = r['lineage']['auroc']
        print(f"{name:<10}  {fp['pair_acc']:>9.4f}  {fp['mean_diag_s_correct']:>9.4f}  "
              f"{fp['mean_diag_s_off']:>9.4f}  {fp['neg_trace_frac']:>9.4f}  "
              f"{au:>7.4f}")

    # strip torch models before dumping
    out = {
        'config': vars(args),
        'wall_time_s': time.time() - t0,
        'results': results,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, default=lambda o: str(o))
    print(f"\nWrote {args.out} ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
