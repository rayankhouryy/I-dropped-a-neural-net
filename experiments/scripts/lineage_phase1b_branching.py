"""Phase 1b of Issue #30: branching lineage experiment.

Tests whether the residual-signature lineage score can recover a
multi-generation branching ancestry tree:

    C1  --->  C2  --->  C4
       \\
        \\-->  C3  --->  C5

C1 is the root checkpoint, C2 and C3 are siblings (both fine-tuned from
C1 on different drifted targets), C4 is the child of C2, C5 is the child
of C3. Plus 5 independent same-arch checkpoints trained on the C1 task
from different seeds as a non-descendant control.

RQ1: For each descendant Cx, can the lineage score recover the true
     ancestry chain?
       expected: parent > grandparent > uncle > cousin > independent
       for C5:   C3 > C1 > C2 > C4 > independent
       for C4:   C2 > C1 > C3 > C5 > independent

RQ2: How are C4 and C5 (cousins through different branches) similar?
     They share C1 ancestry but diverged through different fine-tune
     paths.
       expected:  lineage(C4, C5) > lineage(C4, independent)
                  lineage(C4, C5) < min(lineage(C4, C2), lineage(C5, C3))

Output:
  results/lineage_phase1b_branching.json
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lineage_detection as ldet
from lineage_phase1_mlp import (Block, ResNet, synthetic_target, make_data,
                                 train_model, fresh_model, eval_loss,
                                 branch_products)


def main():
    Path('results').mkdir(parents=True, exist_ok=True)
    out_path = Path('results/lineage_phase1b_branching.json')

    depth, hidden, in_dim = 24, 64, 24
    epochs_root = 200       # C1 trains from scratch
    epochs_branch = 80      # C2, C3 fine-tune from C1
    epochs_leaf = 80        # C4, C5 fine-tune from C2/C3
    n_indep = 5

    t0 = time.time()

    # --- Shared data; each "task" is a different synthetic target on
    # the same X. C1's task is target_key=42; C2 drifts to 50, C3 to 70.
    # C4 drifts further to 55, C5 to 75. Independents use 42 (same as C1).
    X, y_C1 = make_data(in_dim=in_dim, n=4000, seed=0, target_key=42)
    y_C2   = synthetic_target(X, in_dim, key=50)
    y_C3   = synthetic_target(X, in_dim, key=70)
    y_C4   = synthetic_target(X, in_dim, key=55)
    y_C5   = synthetic_target(X, in_dim, key=75)

    checkpoints = {}

    # === C1: root, trained from scratch on y_C1 ===========================
    C1 = fresh_model(depth, hidden, in_dim, seed=100)
    train_model(C1, X, y_C1, epochs=epochs_root, lr=1e-3)
    checkpoints['C1'] = {
        'model':  C1,
        'parent': None,
        'task':   'y_C1 (key=42)',
        'eval_C1_loss': eval_loss(C1, X, y_C1),
    }
    print(f"[C1] root trained  ({time.time()-t0:.1f}s)  "
          f"loss(C1 task)={checkpoints['C1']['eval_C1_loss']:.4f}",
          flush=True)

    # === C2: child of C1, fine-tuned on y_C2 (mild drift) =================
    C2 = copy.deepcopy(C1)
    train_model(C2, X, y_C2, epochs=epochs_branch, lr=3e-4)
    checkpoints['C2'] = {
        'model':  C2,
        'parent': 'C1',
        'task':   'y_C2 (key=50)',
        'eval_C1_loss': eval_loss(C2, X, y_C1),
        'eval_self_loss': eval_loss(C2, X, y_C2),
    }
    print(f"[C2] child of C1 trained  ({time.time()-t0:.1f}s)  "
          f"loss(C2 task)={checkpoints['C2']['eval_self_loss']:.4f}",
          flush=True)

    # === C3: sibling of C2, fine-tuned on y_C3 (different drift) ==========
    C3 = copy.deepcopy(C1)
    train_model(C3, X, y_C3, epochs=epochs_branch, lr=3e-4)
    checkpoints['C3'] = {
        'model':  C3,
        'parent': 'C1',
        'task':   'y_C3 (key=70)',
        'eval_C1_loss': eval_loss(C3, X, y_C1),
        'eval_self_loss': eval_loss(C3, X, y_C3),
    }
    print(f"[C3] sibling of C2 trained  ({time.time()-t0:.1f}s)  "
          f"loss(C3 task)={checkpoints['C3']['eval_self_loss']:.4f}",
          flush=True)

    # === C4: child of C2, fine-tuned further on y_C4 ======================
    C4 = copy.deepcopy(C2)
    train_model(C4, X, y_C4, epochs=epochs_leaf, lr=3e-4)
    checkpoints['C4'] = {
        'model':  C4,
        'parent': 'C2',
        'task':   'y_C4 (key=55)',
        'eval_self_loss': eval_loss(C4, X, y_C4),
    }
    print(f"[C4] child of C2 trained  ({time.time()-t0:.1f}s)  "
          f"loss(C4 task)={checkpoints['C4']['eval_self_loss']:.4f}",
          flush=True)

    # === C5: child of C3, fine-tuned further on y_C5 ======================
    C5 = copy.deepcopy(C3)
    train_model(C5, X, y_C5, epochs=epochs_leaf, lr=3e-4)
    checkpoints['C5'] = {
        'model':  C5,
        'parent': 'C3',
        'task':   'y_C5 (key=75)',
        'eval_self_loss': eval_loss(C5, X, y_C5),
    }
    print(f"[C5] child of C3 trained  ({time.time()-t0:.1f}s)  "
          f"loss(C5 task)={checkpoints['C5']['eval_self_loss']:.4f}",
          flush=True)

    # === Independent baselines: same architecture, same task (y_C1),
    # different seeds. These are the strongest non-descendant control.
    for k in range(n_indep):
        m = fresh_model(depth, hidden, in_dim, seed=500 + k)
        train_model(m, X, y_C1, epochs=epochs_root, lr=1e-3)
        checkpoints[f'I{k}'] = {
            'model':  m,
            'parent': None,
            'task':   f'independent (seed {500+k})',
            'eval_C1_loss': eval_loss(m, X, y_C1),
        }
        print(f"[I{k}] independent trained  ({time.time()-t0:.1f}s)  "
              f"loss(C1 task)={checkpoints[f'I{k}']['eval_C1_loss']:.4f}",
              flush=True)

    print(f"\nTotal training time: {time.time()-t0:.1f}s")

    # === Branch products ===================================================
    print("\nComputing branch products...", flush=True)
    Ms_by_id = {}
    for cid, info in checkpoints.items():
        Ms_by_id[cid] = branch_products(info['model'])
        del info['model']    # free RAM

    # tau_s from the canonical reference (C1)
    tau_s = ldet.choose_tau_s([Ms_by_id['C1']])
    print(f"tau_s (from C1) = {tau_s:.4f}")

    # === Full pairwise lineage matrix ======================================
    all_ids = list(checkpoints.keys())
    n = len(all_ids)
    L_mat = np.zeros((n, n), dtype=np.float64)
    print("\nComputing pairwise lineage matrix...", flush=True)
    for i, a in enumerate(all_ids):
        for j, b in enumerate(all_ids):
            if i == j:
                L_mat[i, j] = 1.0  # self
                continue
            L_score, _, _ = ldet.lineage_score(Ms_by_id[a], Ms_by_id[b], tau_s)
            L_mat[i, j] = L_score
        print(f"  row {a}  done", flush=True)

    # === Define the true ancestry sets ======================================
    # All ancestors of each node (including self):
    ancestors = {
        'C1': {'C1'},
        'C2': {'C1', 'C2'},
        'C3': {'C1', 'C3'},
        'C4': {'C1', 'C2', 'C4'},
        'C5': {'C1', 'C3', 'C5'},
    }
    for k in range(n_indep):
        ancestors[f'I{k}'] = {f'I{k}'}

    # Relationship classifier between two nodes
    def relationship(a, b):
        """How is b related to a?"""
        if a == b:
            return 'self'
        a_anc = ancestors.get(a, {a})
        b_anc = ancestors.get(b, {b})
        if b in a_anc:
            depth_a = {'C1': 0, 'C2': 1, 'C3': 1, 'C4': 2, 'C5': 2}.get(a, 0)
            depth_b = {'C1': 0, 'C2': 1, 'C3': 1, 'C4': 2, 'C5': 2}.get(b, 0)
            d = depth_a - depth_b
            return f'ancestor_dist{d}'   # b is d generations above a
        if a in b_anc:
            depth_a = {'C1': 0, 'C2': 1, 'C3': 1, 'C4': 2, 'C5': 2}.get(a, 0)
            depth_b = {'C1': 0, 'C2': 1, 'C3': 1, 'C4': 2, 'C5': 2}.get(b, 0)
            d = depth_b - depth_a
            return f'descendant_dist{d}'
        # common ancestor exists?
        common = a_anc & b_anc
        if common and not (a.startswith('I') or b.startswith('I')):
            # Cousins or uncles
            depth_a = {'C1': 0, 'C2': 1, 'C3': 1, 'C4': 2, 'C5': 2}.get(a, 0)
            depth_b = {'C1': 0, 'C2': 1, 'C3': 1, 'C4': 2, 'C5': 2}.get(b, 0)
            if depth_a == depth_b:
                return 'cousin' if depth_a == 2 else 'sibling'
            else:
                return 'uncle'   # 'uncle' covers grand-uncle etc. too
        return 'independent'

    # === RQ1: For each leaf Cx in {C4, C5}, rank candidates by lineage =====
    rq1 = {}
    for leaf in ['C4', 'C5']:
        i = all_ids.index(leaf)
        row = [(other, L_mat[i, all_ids.index(other)],
                relationship(leaf, other))
               for other in all_ids if other != leaf]
        row.sort(key=lambda t: -t[1])
        rq1[leaf] = row
        true_chain = list(ancestors[leaf] - {leaf})
        true_parent = checkpoints[leaf]['parent']
        true_grandparent = checkpoints.get(true_parent, {}).get('parent') if true_parent else None
        rq1[f'{leaf}_truth'] = {
            'parent':       true_parent,
            'grandparent':  true_grandparent,
            'ranked_by_lineage': [(t[0], float(t[1]), t[2]) for t in row],
            'parent_rank':  next(i for i, t in enumerate(row)
                                 if t[0] == true_parent) + 1,
            'grandparent_rank': (
                next(i for i, t in enumerate(row)
                     if t[0] == true_grandparent) + 1
                if true_grandparent else None),
        }

    # === RQ2: Cousin similarity (C4, C5) ===================================
    rq2 = {
        'L_C4_C5': float(L_mat[all_ids.index('C4'), all_ids.index('C5')]),
        'L_C4_C2_parent':       float(L_mat[all_ids.index('C4'), all_ids.index('C2')]),
        'L_C5_C3_parent':       float(L_mat[all_ids.index('C5'), all_ids.index('C3')]),
        'L_C4_C1_grandparent':  float(L_mat[all_ids.index('C4'), all_ids.index('C1')]),
        'L_C5_C1_grandparent':  float(L_mat[all_ids.index('C5'), all_ids.index('C1')]),
        'L_C4_indep_mean':      float(np.mean(
            [L_mat[all_ids.index('C4'), all_ids.index(f'I{k}')]
             for k in range(n_indep)])),
        'L_C5_indep_mean':      float(np.mean(
            [L_mat[all_ids.index('C5'), all_ids.index(f'I{k}')]
             for k in range(n_indep)])),
    }
    rq2['cousins_above_indep'] = (rq2['L_C4_C5']
                                   > max(rq2['L_C4_indep_mean'],
                                         rq2['L_C5_indep_mean']))
    rq2['cousins_below_parents'] = (rq2['L_C4_C5']
                                     < min(rq2['L_C4_C2_parent'],
                                           rq2['L_C5_C3_parent']))

    # === Aggregate by relationship category ================================
    by_rel = {}
    for i, a in enumerate(all_ids):
        for j, b in enumerate(all_ids):
            if i == j:
                continue
            rel = relationship(a, b)
            by_rel.setdefault(rel, []).append(float(L_mat[i, j]))
    rel_summary = {
        rel: {
            'n':    len(scores),
            'mean': float(np.mean(scores)),
            'std':  float(np.std(scores)),
            'min':  float(np.min(scores)),
            'max':  float(np.max(scores)),
        }
        for rel, scores in by_rel.items()
    }

    out = {
        'description': 'Phase 1b branching lineage tree (#30)',
        'tree': {
            'C1': {'parent': None, 'role': 'root'},
            'C2': {'parent': 'C1', 'role': 'sibling-of-C3'},
            'C3': {'parent': 'C1', 'role': 'sibling-of-C2'},
            'C4': {'parent': 'C2', 'role': 'cousin-of-C5'},
            'C5': {'parent': 'C3', 'role': 'cousin-of-C4'},
        },
        'config': {
            'depth': depth, 'hidden': hidden, 'in_dim': in_dim,
            'epochs_root': epochs_root,
            'epochs_branch': epochs_branch,
            'epochs_leaf': epochs_leaf,
            'n_independent': n_indep,
        },
        'tau_s': tau_s,
        'all_ids': all_ids,
        'lineage_matrix': L_mat.tolist(),
        'checkpoints': {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                        for k, v in checkpoints.items()},
        'rq1_ancestry_tracing': rq1,
        'rq2_cousin_similarity': rq2,
        'by_relationship': rel_summary,
        'total_seconds': time.time() - t0,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # === Headline ==========================================================
    print("\n=== RQ1: Lineage tracing ===")
    for leaf in ['C4', 'C5']:
        print(f"\nFor {leaf} (true parent={checkpoints[leaf]['parent']}, "
              f"true grandparent={checkpoints.get(checkpoints[leaf]['parent'], {}).get('parent')}):")
        ranking = rq1[leaf]
        for rank, (other, score, rel) in enumerate(ranking, 1):
            marker = '<-- PARENT' if rel == 'ancestor_dist1' \
                     else ('<-- GRANDPARENT' if rel == 'ancestor_dist2'
                           else '')
            print(f"  {rank}. {other:>4s}  L={score:+.4f}  ({rel}) {marker}")

    print("\n=== RQ2: Cousin similarity (C4 vs C5) ===")
    print(f"  L(C4, C5)         = {rq2['L_C4_C5']:+.4f}  (cousins)")
    print(f"  L(C4, C2)         = {rq2['L_C4_C2_parent']:+.4f}  (parent)")
    print(f"  L(C5, C3)         = {rq2['L_C5_C3_parent']:+.4f}  (parent)")
    print(f"  L(C4, C1)         = {rq2['L_C4_C1_grandparent']:+.4f}  (grandparent)")
    print(f"  L(C5, C1)         = {rq2['L_C5_C1_grandparent']:+.4f}  (grandparent)")
    print(f"  L(C4, indep mean) = {rq2['L_C4_indep_mean']:+.4f}  (independents)")
    print(f"  L(C5, indep mean) = {rq2['L_C5_indep_mean']:+.4f}  (independents)")
    print(f"  cousins detectably more similar than independents: "
          f"{rq2['cousins_above_indep']}")
    print(f"  cousins less similar than direct parents: "
          f"{rq2['cousins_below_parents']}")

    print("\n=== Lineage by relationship category ===")
    rel_order = ['self', 'descendant_dist1', 'descendant_dist2',
                 'ancestor_dist1', 'ancestor_dist2', 'sibling', 'cousin',
                 'uncle', 'independent']
    for rel in rel_order:
        if rel in rel_summary:
            s = rel_summary[rel]
            print(f"  {rel:<18s}  n={s['n']:>2d}  "
                  f"mean L = {s['mean']:+.4f}  "
                  f"(min={s['min']:+.4f}, max={s['max']:+.4f})")


if __name__ == '__main__':
    main()
