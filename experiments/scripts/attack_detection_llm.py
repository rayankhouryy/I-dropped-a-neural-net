#!/usr/bin/env python3
"""Attack Detection LLM Validation - GPU Required.

Validates the attack detection findings from MLP on real LLM checkpoints.
Tests whether we can distinguish adversarially-attacked LLMs from genuinely
unrelated ones using the (dd, ||E||) forensic signature.

Usage (on ml.g6.xlarge):
    # Qwen 1.5B (faster, for validation)
    python attack_detection_llm.py --model-family qwen --attack-lambdas 0.1,1.0

    # LLaMA-2 7B (main experiment)
    python attack_detection_llm.py --model-family llama2 --attack-lambdas 0.1,1.0

Output: results/attack_detection_llm_{family}.json
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_CONFIGS = {
    'qwen': {
        'base': 'Qwen/Qwen2.5-1.5B',
        'related': 'Qwen/Qwen2.5-1.5B-Instruct',
        'unrelated': None,  # No matching architecture
        'dtype': torch.bfloat16,
    },
    'llama2': {
        'base': 'NousResearch/Llama-2-7b-hf',
        'related': 'NousResearch/Llama-2-7b-chat-hf',
        'unrelated': 'openlm-research/open_llama_7b',
        'dtype': torch.float16,
    },
}


def load_model(repo: str, device: str, dtype):
    """Load model with memory optimization."""
    from transformers import AutoModelForCausalLM
    print(f"[load] {repo}", flush=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map={"": device}, trust_remote_code=True)
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=True)
        m.to(device)
    m.eval()
    return m


@torch.no_grad()
def compute_layer_metrics(layer):
    """Compute dd and ||E|| for one decoder layer's SwiGLU MLP.

    Returns: (dd_score, E_norm, lineage_phi)
    """
    Wu = layer.mlp.up_proj.weight.detach().to(torch.float32)
    Wd = layer.mlp.down_proj.weight.detach().to(torch.float32)
    M = Wd @ Wu  # (d, d)
    d = M.shape[0]

    # Diagonal dominance
    tr = torch.trace(M)
    frob = torch.linalg.matrix_norm(M, 'fro')
    dd = float(tr.abs() / frob)

    # Centered signature norm ||E||
    alpha = tr / d
    E = M - alpha * torch.eye(d, dtype=M.dtype, device=M.device)
    E_norm = float(torch.linalg.matrix_norm(E, 'fro'))

    # Lineage signature (unit vector)
    phi = (E / torch.linalg.matrix_norm(E)).flatten().cpu()

    del Wu, Wd, M, E
    return dd, E_norm, phi


@torch.no_grad()
def compute_model_metrics(model):
    """Compute aggregate metrics for entire model."""
    dd_scores = []
    E_norms = []
    phis = []

    for layer in model.model.layers:
        dd, E_norm, phi = compute_layer_metrics(layer)
        dd_scores.append(dd)
        E_norms.append(E_norm)
        phis.append(phi)

    return {
        'dd_mean': float(np.mean(dd_scores)),
        'dd_std': float(np.std(dd_scores)),
        'E_norm_mean': float(np.mean(E_norms)),
        'E_norm_std': float(np.std(E_norms)),
        'dd_scores': dd_scores,
        'E_norms': E_norms,
        'phis': phis,
    }


def lineage_score(phis_a, phis_b):
    """Compute lineage score as mean cosine of signatures."""
    scores = []
    for phi_a, phi_b in zip(phis_a, phis_b):
        cos = float((phi_a * phi_b).sum())
        scores.append(cos)
    return float(np.mean(scores))


def gradient_attack_llm(model, lambda_utility, n_steps=100, lr=1e-4, device='cuda'):
    """Run gradient attack on LLM to minimize lineage score.

    Attacks only the MLP up_proj and down_proj weights (SwiGLU path).
    """
    print(f"  Running gradient attack (λ={lambda_utility}, steps={n_steps})...", flush=True)

    # Store reference signatures
    ref_metrics = compute_model_metrics(model)
    ref_phis = ref_metrics['phis']

    # Enable gradients on MLP weights only
    for layer in model.model.layers:
        layer.mlp.up_proj.weight.requires_grad_(True)
        layer.mlp.down_proj.weight.requires_grad_(True)

    # Collect parameters to optimize
    params = []
    for layer in model.model.layers:
        params.append(layer.mlp.up_proj.weight)
        params.append(layer.mlp.down_proj.weight)

    # Store reference outputs for utility preservation
    vocab_size = model.config.vocab_size
    probe_ids = torch.randint(1, vocab_size, (4, 16), device=device)
    with torch.no_grad():
        ref_logits = model(probe_ids).logits.detach()

    opt = torch.optim.Adam(params, lr=lr)

    for step in range(n_steps):
        # Compute lineage objective (mean cosine of centered signatures)
        cos_sum = 0.0
        n_layers = len(model.model.layers)

        for i, layer in enumerate(model.model.layers):
            Wu = layer.mlp.up_proj.weight
            Wd = layer.mlp.down_proj.weight
            M = Wd @ Wu
            d = M.shape[0]
            alpha = torch.trace(M) / d
            E = M - alpha * torch.eye(d, dtype=M.dtype, device=M.device)
            phi = (E / (torch.linalg.matrix_norm(E) + 1e-12)).flatten()

            ref_phi = ref_phis[i].to(device)
            cos_sum = cos_sum + (phi * ref_phi).sum()

        cos_mean = cos_sum / n_layers

        # Utility objective
        curr_logits = model(probe_ids).logits
        utility_loss = F.mse_loss(curr_logits, ref_logits)

        # Total loss: minimize cos (to break lineage) + preserve utility
        loss = cos_mean + lambda_utility * utility_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == n_steps - 1:
            print(f"    step {step}: cos={float(cos_mean):.4f}, util={float(utility_loss):.4e}",
                  flush=True)

    # Disable gradients
    for layer in model.model.layers:
        layer.mlp.up_proj.weight.requires_grad_(False)
        layer.mlp.down_proj.weight.requires_grad_(False)

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-family", choices=list(MODEL_CONFIGS.keys()), required=True)
    ap.add_argument("--attack-lambdas", default="0.1,1.0",
                    help="Comma-separated lambda values for attack")
    ap.add_argument("--attack-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    Path('results').mkdir(parents=True, exist_ok=True)

    config = MODEL_CONFIGS[args.model_family]
    lambda_values = [float(x) for x in args.attack_lambdas.split(',')]

    print(f"\n{'='*60}")
    print(f"Attack Detection LLM Validation: {args.model_family}")
    print(f"{'='*60}")
    print(f"Base: {config['base']}")
    print(f"Related: {config['related']}")
    print(f"Unrelated: {config['unrelated']}")
    print(f"Attack lambdas: {lambda_values}")
    print(f"Device: {args.device}\n")

    results = {
        'config': {
            'model_family': args.model_family,
            'base': config['base'],
            'related': config['related'],
            'unrelated': config['unrelated'],
            'attack_lambdas': lambda_values,
            'attack_steps': args.attack_steps,
        },
        'metrics': {},
    }

    # Load base model and compute metrics
    print("[1/N] Loading base model...")
    base_model = load_model(config['base'], args.device, config['dtype'])
    base_metrics = compute_model_metrics(base_model)
    base_phis = base_metrics['phis']
    results['metrics']['base'] = {
        'dd_mean': base_metrics['dd_mean'],
        'dd_std': base_metrics['dd_std'],
        'E_norm_mean': base_metrics['E_norm_mean'],
        'E_norm_std': base_metrics['E_norm_std'],
        'lineage': 1.0,
    }
    print(f"  dd_mean={base_metrics['dd_mean']:.4f}, ||E||={base_metrics['E_norm_mean']:.4f}")

    # Related checkpoint
    if config['related']:
        print(f"\n[2/N] Loading related checkpoint: {config['related']}")
        del base_model
        gc.collect()
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None

        related = load_model(config['related'], args.device, config['dtype'])
        related_metrics = compute_model_metrics(related)
        L_related = lineage_score(base_phis, related_metrics['phis'])
        results['metrics']['related'] = {
            'dd_mean': related_metrics['dd_mean'],
            'dd_std': related_metrics['dd_std'],
            'E_norm_mean': related_metrics['E_norm_mean'],
            'E_norm_std': related_metrics['E_norm_std'],
            'lineage': L_related,
        }
        print(f"  dd_mean={related_metrics['dd_mean']:.4f}, ||E||={related_metrics['E_norm_mean']:.4f}, L={L_related:.4f}")
        del related
        gc.collect()
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None

    # Unrelated checkpoint
    if config['unrelated']:
        print(f"\n[3/N] Loading unrelated checkpoint: {config['unrelated']}")
        unrelated = load_model(config['unrelated'], args.device, config['dtype'])
        unrelated_metrics = compute_model_metrics(unrelated)
        L_unrelated = lineage_score(base_phis, unrelated_metrics['phis'])
        results['metrics']['unrelated'] = {
            'dd_mean': unrelated_metrics['dd_mean'],
            'dd_std': unrelated_metrics['dd_std'],
            'E_norm_mean': unrelated_metrics['E_norm_mean'],
            'E_norm_std': unrelated_metrics['E_norm_std'],
            'lineage': L_unrelated,
        }
        print(f"  dd_mean={unrelated_metrics['dd_mean']:.4f}, ||E||={unrelated_metrics['E_norm_mean']:.4f}, L={L_unrelated:.4f}")
        del unrelated
        gc.collect()
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None

    # Attacked variants
    results['metrics']['attacked'] = {}
    for i, lam in enumerate(lambda_values):
        print(f"\n[{4+i}/N] Running attack with λ={lam}...")

        # Reload base for attack
        attack_model = load_model(config['base'], args.device, config['dtype'])

        # Run attack
        attack_model = gradient_attack_llm(
            attack_model, lambda_utility=lam,
            n_steps=args.attack_steps, device=args.device
        )

        # Compute metrics
        attacked_metrics = compute_model_metrics(attack_model)
        L_attacked = lineage_score(base_phis, attacked_metrics['phis'])

        results['metrics']['attacked'][str(lam)] = {
            'dd_mean': attacked_metrics['dd_mean'],
            'dd_std': attacked_metrics['dd_std'],
            'E_norm_mean': attacked_metrics['E_norm_mean'],
            'E_norm_std': attacked_metrics['E_norm_std'],
            'lineage': L_attacked,
        }
        print(f"  dd_mean={attacked_metrics['dd_mean']:.4f}, ||E||={attacked_metrics['E_norm_mean']:.4f}, L={L_attacked:.4f}")

        del attack_model
        gc.collect()
        torch.cuda.empty_cache() if args.device.startswith('cuda') else None

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'dd_mean':>10} {'||E||':>10} {'L':>10}")
    print("-" * 52)
    for name, m in results['metrics'].items():
        if name == 'attacked':
            for lam, am in m.items():
                print(f"{'attacked_λ='+lam:<20} {am['dd_mean']:>10.4f} {am['E_norm_mean']:>10.4f} {am['lineage']:>10.4f}")
        else:
            print(f"{name:<20} {m['dd_mean']:>10.4f} {m['E_norm_mean']:>10.4f} {m['lineage']:>10.4f}")

    # Detection check
    print(f"\n{'='*60}")
    print("ATTACK DETECTION CHECK")
    print(f"{'='*60}")

    # From MLP results: threshold ratio ≈ 8.7
    ratio_threshold = 8.7

    for name, m in results['metrics'].items():
        if name == 'attacked':
            for lam, am in m.items():
                ratio = am['E_norm_mean'] / (am['dd_mean'] + 1e-8)
                detected = ratio > ratio_threshold
                print(f"attacked_λ={lam}: ratio={ratio:.2f}, detected={detected}")
        else:
            ratio = m['E_norm_mean'] / (m['dd_mean'] + 1e-8)
            detected = ratio > ratio_threshold
            print(f"{name}: ratio={ratio:.2f}, detected={detected}")

    results['elapsed_seconds'] = time.time() - t0

    out_path = Path(f'results/attack_detection_llm_{args.model_family}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {results['elapsed_seconds']:.1f}s")


if __name__ == '__main__':
    main()
