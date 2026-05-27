"""
Parent Retrieval Experiment on Real Pretrained GPT-2 Family Models.

Tests whether centered diagonal fingerprints can identify which pretrained
model a fine-tuned/modified checkpoint descended from.

References: gpt2, distilgpt2, DialoGPT-small (all d_model=768)
Descendants: noise injection, weight perturbation variants
Controls: random init, cross-model comparisons
"""

import json
import copy
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy.optimize import linear_sum_assignment

# Ensure output directory exists
OUTPUT_DIR = Path("case_studies/model_lineage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_mlp_fingerprints(model, max_layers=None):
    """Extract centered diagonal fingerprints from MLP branch products."""
    fingerprints = []
    blocks = list(model.transformer.h)
    if max_layers:
        blocks = blocks[:max_layers]

    for block in blocks:
        W1 = block.mlp.c_fc.weight.detach().float().cpu().numpy().T    # (4d, d)
        W2 = block.mlp.c_proj.weight.detach().float().cpu().numpy().T  # (d, 4d)
        prod = W2 @ W1  # (d, d)

        d = np.diag(prod)
        d_centered = d - d.mean()
        norm = np.linalg.norm(d_centered) + 1e-12
        fingerprints.append(d_centered / norm)

    return fingerprints


def extract_full_fingerprints(model, max_layers=None):
    """Extract full flattened branch product (baseline method)."""
    fingerprints = []
    blocks = list(model.transformer.h)
    if max_layers:
        blocks = blocks[:max_layers]

    for block in blocks:
        W1 = block.mlp.c_fc.weight.detach().float().cpu().numpy().T
        W2 = block.mlp.c_proj.weight.detach().float().cpu().numpy().T
        prod = W2 @ W1
        vec = prod.flatten()
        fingerprints.append(vec / (np.linalg.norm(vec) + 1e-12))

    return fingerprints


def lineage_score(fp_a, fp_b):
    """Mean cosine similarity across aligned layers."""
    n = min(len(fp_a), len(fp_b))
    scores = [np.dot(fp_a[i], fp_b[i]) for i in range(n)]
    return np.mean(scores)


def add_weight_noise(model, sigma, seed=None):
    """Add Gaussian noise proportional to parameter std."""
    noisy = copy.deepcopy(model)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        for p in noisy.parameters():
            if p.numel() > 1:
                p.add_(torch.randn_like(p) * sigma * p.std().item())
    return noisy


def load_model(name):
    """Load a pretrained model."""
    from transformers import GPT2LMHeadModel, AutoModelForCausalLM

    print(f"  Loading {name}...")
    if name in ['gpt2', 'distilgpt2']:
        model = GPT2LMHeadModel.from_pretrained(name)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model


def create_random_init(config_name='gpt2'):
    """Create randomly initialized model with same architecture."""
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config.from_pretrained(config_name)
    return GPT2LMHeadModel(config)


def run_experiment():
    print("=" * 70)
    print("PARENT RETRIEVAL EXPERIMENT - Real GPT-2 Family Models")
    print("=" * 70)

    # Reference models (all have d_model=768)
    reference_names = [
        'gpt2',                      # OpenAI GPT-2 small
        'distilgpt2',                # Distilled GPT-2
        'microsoft/DialoGPT-small',  # Dialog-trained GPT-2
    ]

    # Use minimum layer count for fair comparison
    min_layers = 6  # distilgpt2 has 6 layers

    print(f"\nPhase 1: Loading {len(reference_names)} reference models...")
    references = {}
    ref_fingerprints_diag = {}
    ref_fingerprints_full = {}

    for name in reference_names:
        model = load_model(name)
        references[name] = model
        ref_fingerprints_diag[name] = extract_mlp_fingerprints(model, max_layers=min_layers)
        ref_fingerprints_full[name] = extract_full_fingerprints(model, max_layers=min_layers)

    # Generate descendants
    print(f"\nPhase 2: Generating descendants...")
    suspects = []

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    for ref_name, model in references.items():
        for sigma in noise_levels:
            for seed in [42, 123]:
                noisy = add_weight_noise(model, sigma, seed=seed)
                suspect_name = f"{ref_name.split('/')[-1]}_noise{sigma}_s{seed}"
                fp_diag = extract_mlp_fingerprints(noisy, max_layers=min_layers)
                fp_full = extract_full_fingerprints(noisy, max_layers=min_layers)
                suspects.append({
                    'name': suspect_name,
                    'true_parent': ref_name,
                    'type': 'descendant',
                    'transform': f'noise_{sigma}',
                    'fp_diag': fp_diag,
                    'fp_full': fp_full,
                })
                del noisy
        print(f"    {ref_name}: {len(noise_levels)*2} noisy descendants")

    # Control: random init
    print("\nPhase 3: Generating controls...")
    for seed in range(5):
        torch.manual_seed(seed)
        rand_model = create_random_init('gpt2')
        fp_diag = extract_mlp_fingerprints(rand_model, max_layers=min_layers)
        fp_full = extract_full_fingerprints(rand_model, max_layers=min_layers)
        suspects.append({
            'name': f'random_init_s{seed}',
            'true_parent': None,
            'type': 'random_init',
            'transform': 'none',
            'fp_diag': fp_diag,
            'fp_full': fp_full,
        })
        del rand_model
    print(f"    5 random init controls")

    # Parent retrieval
    print(f"\nPhase 4: Parent retrieval ({len(suspects)} suspects x {len(references)} refs)...")

    results = []
    for suspect in suspects:
        # Compute scores against all references
        scores_diag = {ref: lineage_score(ref_fingerprints_diag[ref], suspect['fp_diag'])
                       for ref in reference_names}
        scores_full = {ref: lineage_score(ref_fingerprints_full[ref], suspect['fp_full'])
                       for ref in reference_names}

        # Predict parent (highest score)
        pred_diag = max(scores_diag, key=scores_diag.get)
        pred_full = max(scores_full, key=scores_full.get)

        # Check correctness
        true_parent = suspect['true_parent']
        correct_diag = (pred_diag == true_parent) if true_parent else False
        correct_full = (pred_full == true_parent) if true_parent else False

        # Compute rank of true parent
        if true_parent:
            sorted_diag = sorted(scores_diag.items(), key=lambda x: -x[1])
            rank_diag = [r[0] for r in sorted_diag].index(true_parent) + 1
            sorted_full = sorted(scores_full.items(), key=lambda x: -x[1])
            rank_full = [r[0] for r in sorted_full].index(true_parent) + 1
        else:
            rank_diag = rank_full = -1

        results.append({
            'suspect': suspect['name'],
            'true_parent': true_parent,
            'type': suspect['type'],
            'transform': suspect['transform'],
            'pred_diag': pred_diag,
            'pred_full': pred_full,
            'correct_diag': correct_diag,
            'correct_full': correct_full,
            'rank_diag': rank_diag,
            'rank_full': rank_full,
            'max_score_diag': max(scores_diag.values()),
            'max_score_full': max(scores_full.values()),
            'scores_diag': scores_diag,
            'scores_full': scores_full,
        })

    # Compute metrics
    print("\nPhase 5: Computing metrics...")

    descendants = [r for r in results if r['type'] == 'descendant']
    controls = [r for r in results if r['type'] != 'descendant']

    top1_diag = np.mean([r['correct_diag'] for r in descendants])
    top1_full = np.mean([r['correct_full'] for r in descendants])

    mrr_diag = np.mean([1.0/r['rank_diag'] for r in descendants])
    mrr_full = np.mean([1.0/r['rank_full'] for r in descendants])

    # AUROC: descendant max-scores vs control max-scores
    desc_scores_diag = [r['max_score_diag'] for r in descendants]
    ctrl_scores_diag = [r['max_score_diag'] for r in controls]
    desc_scores_full = [r['max_score_full'] for r in descendants]
    ctrl_scores_full = [r['max_score_full'] for r in controls]

    def compute_auroc(pos, neg):
        from itertools import product
        n_pos, n_neg = len(pos), len(neg)
        count = sum(1 for p, n in product(pos, neg) if p > n)
        ties = sum(1 for p, n in product(pos, neg) if p == n)
        return (count + 0.5 * ties) / (n_pos * n_neg)

    auroc_diag = compute_auroc(desc_scores_diag, ctrl_scores_diag)
    auroc_full = compute_auroc(desc_scores_full, ctrl_scores_full)

    # Results by transformation
    by_transform = {}
    for transform in set(r['transform'] for r in descendants):
        subset = [r for r in descendants if r['transform'] == transform]
        by_transform[transform] = {
            'n': len(subset),
            'top1_diag': np.mean([r['correct_diag'] for r in subset]),
            'top1_full': np.mean([r['correct_full'] for r in subset]),
        }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nMethod           Top-1      MRR      AUROC")
    print("-" * 50)
    print(f"diag_centered    {top1_diag:6.1%}    {mrr_diag:.3f}    {auroc_diag:.3f}")
    print(f"full_cosine      {top1_full:6.1%}    {mrr_full:.3f}    {auroc_full:.3f}")

    print(f"\nBy noise level (diag_centered):")
    for transform in sorted(by_transform.keys()):
        stats = by_transform[transform]
        print(f"  {transform:15s} Top-1: {stats['top1_diag']:.1%} (n={stats['n']})")

    print(f"\nScore distributions:")
    print(f"  Descendants (diag):  mean={np.mean(desc_scores_diag):.4f}, min={np.min(desc_scores_diag):.4f}")
    print(f"  Controls (diag):     mean={np.mean(ctrl_scores_diag):.4f}, max={np.max(ctrl_scores_diag):.4f}")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'references': reference_names,
            'min_layers': min_layers,
            'noise_levels': noise_levels,
            'n_descendants': len(descendants),
            'n_controls': len(controls),
        },
        'metrics': {
            'diag_centered': {
                'top1': top1_diag,
                'mrr': mrr_diag,
                'auroc': auroc_diag,
            },
            'full_cosine': {
                'top1': top1_full,
                'mrr': mrr_full,
                'auroc': auroc_full,
            },
        },
        'by_transform': by_transform,
        'score_distributions': {
            'desc_diag': {'mean': float(np.mean(desc_scores_diag)), 'min': float(np.min(desc_scores_diag)), 'max': float(np.max(desc_scores_diag))},
            'ctrl_diag': {'mean': float(np.mean(ctrl_scores_diag)), 'min': float(np.min(ctrl_scores_diag)), 'max': float(np.max(ctrl_scores_diag))},
        },
    }

    with open(OUTPUT_DIR / 'gpt2_parent_retrieval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'gpt2_parent_retrieval_summary.json'}")

    # Detailed CSV
    import csv
    with open(OUTPUT_DIR / 'gpt2_parent_retrieval_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'suspect', 'type', 'transform', 'true_parent',
            'pred_diag', 'correct_diag', 'rank_diag', 'max_score_diag',
            'pred_full', 'correct_full', 'rank_full', 'max_score_full',
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in r.items() if k not in ['scores_diag', 'scores_full', 'fp_diag', 'fp_full']})
    print(f"Saved: {OUTPUT_DIR / 'gpt2_parent_retrieval_results.csv'}")

    return summary


if __name__ == '__main__':
    run_experiment()
