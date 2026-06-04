# POC: Strong Baselines for Lineage Verification

**Status:** Draft  
**Author:** Auto-generated from review feedback  
**Date:** 2026-06-03

## Executive Summary

The external review identified **baseline quality** as a key weakness: our current comparisons (Frobenius matching, singular-value distance) are too weak. Reviewers will ask why we didn't compare against representation-similarity methods (CKA, SVCCA) and provenance-specific baselines (IPGuard, UAP fingerprinting). This POC outlines:

1. **Strong baselines** to implement
2. **Public benchmarks** to evaluate on
3. **Implementation plan** with concrete code paths

---

## Part 1: Strong Baselines to Add

### 1.1 CKA (Centered Kernel Alignment)

**What it is:** A representation similarity metric that measures alignment between internal activations of two networks. Unlike our weight-space method, CKA requires forward passes on a shared probe dataset.

**Why it matters:** CKA is the standard academic baseline for "are these two networks similar?" questions. Showing we beat CKA on lineage discrimination would be a strong result.

**Key paper:** Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019, arXiv:1905.00414)

**Implementation:**
```python
# Reference: google-research/google-research/representation_similarity
def linear_CKA(X, Y):
    """CKA with linear kernel (fastest, most common).
    
    X: (n_samples, d_x) activations from network A
    Y: (n_samples, d_y) activations from network B
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T
    
    # HSIC estimator
    hsic = np.sum(K * L)
    norm_x = np.sqrt(np.sum(K * K))
    norm_y = np.sqrt(np.sum(L * L))
    
    return hsic / (norm_x * norm_y + 1e-12)
```

**Lineage protocol:**
1. Forward reference model on probe set → activations per layer
2. Forward suspect model on same probe set → activations per layer  
3. Compute CKA(ref_layer_i, sus_layer_j) for all pairs
4. Hungarian match layers by CKA, report mean matched CKA as lineage score
5. Threshold: descendants should have CKA > τ; independents should have CKA ≈ random baseline

**Expected outcome:** CKA will work for same-architecture descendants but:
- Requires probe data (we're data-free)
- May fail on quantization/pruning (activation patterns change)
- Cannot handle architecture changes

### 1.2 SVCCA (Singular Vector CCA)

**What it is:** SVD the activations first (dimensionality reduction), then CCA to find aligned directions.

**Key paper:** Raghu et al., "SVCCA: Singular Vector Canonical Correlation Analysis" (NeurIPS 2017, arXiv:1706.05806)

**Implementation:**
```python
# Reference: google/svcca (archived)
def svcca_similarity(X, Y, k=None):
    """SVCCA similarity.
    
    X: (n_samples, d_x)
    Y: (n_samples, d_y)
    k: number of singular vectors to keep (default: 99% variance)
    """
    # SVD step
    Ux, sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Y, full_matrices=False)
    
    # Keep top-k by variance
    if k is None:
        k_x = np.searchsorted(np.cumsum(sx**2) / np.sum(sx**2), 0.99) + 1
        k_y = np.searchsorted(np.cumsum(sy**2) / np.sum(sy**2), 0.99) + 1
    else:
        k_x = k_y = k
    
    Ux, Uy = Ux[:, :k_x], Uy[:, :k_y]
    
    # CCA step
    Qx, _ = np.linalg.qr(Ux)
    Qy, _ = np.linalg.qr(Uy)
    
    _, cca_corrs, _ = np.linalg.svd(Qx.T @ Qy)
    return np.mean(cca_corrs)
```

**Lineage protocol:** Same as CKA but use SVCCA similarity.

### 1.3 Weight-Space Frobenius with Hungarian Alignment

**What it is:** Our current Frobenius baseline but with proper layer alignment (not identity).

**Current gap:** Our `frob_distance` uses identity alignment. For fair comparison, we should Hungarian-match layers by Frobenius distance.

```python
def aligned_frobenius_similarity(Ms_A, Ms_B):
    """Frobenius similarity with Hungarian alignment.
    
    Returns negative distance (higher = more similar).
    """
    L = len(Ms_A)
    dist_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            dist_matrix[i, j] = np.linalg.norm(
                Ms_A[i].astype(np.float64) - Ms_B[j].astype(np.float64), 'fro')
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return -dist_matrix[row_ind, col_ind].mean()
```

### 1.4 IPGuard-style Decision Boundary Fingerprinting

**What it is:** Generate adversarial examples on the reference model's decision boundary; check if suspect model classifies them similarly.

**Key paper:** Cao et al., "IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary" (AsiaCCS 2021)

**Why it matters:** This is the main provenance baseline. If we beat IPGuard, we have a strong claim.

**Implementation sketch:**
```python
def ipguard_fingerprint(model, X_seed, n_fingerprints=100, epsilon=0.1):
    """Generate decision-boundary fingerprints.
    
    1. Pick seed samples near class boundaries
    2. Generate adversarial perturbations that flip predictions
    3. The (input, prediction) pairs are the fingerprint
    """
    fingerprints = []
    for x in X_seed[:n_fingerprints]:
        x_adv = pgd_attack(model, x, epsilon=epsilon)
        pred = model(x_adv).argmax()
        fingerprints.append((x_adv, pred))
    return fingerprints

def ipguard_similarity(fingerprints, suspect_model):
    """Fraction of fingerprints with matching predictions."""
    matches = 0
    for x_adv, ref_pred in fingerprints:
        sus_pred = suspect_model(x_adv).argmax()
        if sus_pred == ref_pred:
            matches += 1
    return matches / len(fingerprints)
```

**Limitations:**
- Requires classification task (not regression)
- Requires input data
- Adversarial examples may not transfer to heavily modified models

### 1.5 UAP Fingerprinting

**What it is:** Characterize decision boundary via Universal Adversarial Perturbations. Models with shared lineage should have similar UAP subspaces.

**Key paper:** Peng et al., "Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations" (CVPR 2022, arXiv:2202.08602)

```python
def compute_uap_subspace(model, X, n_uaps=20, epsilon=0.03):
    """Compute UAP subspace for fingerprinting."""
    uaps = []
    for _ in range(n_uaps):
        uap = universal_perturbation(model, X, epsilon=epsilon)
        uaps.append(uap.flatten())
    
    # SVD to get subspace
    U, s, _ = np.linalg.svd(np.stack(uaps), full_matrices=False)
    return U[:, :10]  # Top-10 directions

def uap_subspace_similarity(U_ref, U_sus):
    """Grassmann distance between UAP subspaces."""
    _, s, _ = np.linalg.svd(U_ref.T @ U_sus)
    return np.sum(s)  # Principal angles
```

---

## Part 2: Public Benchmarks

### 2.1 Known Lineage Chains (Hugging Face)

These model families have documented lineage we can verify:

#### LLaMA Family
```
meta-llama/Llama-2-7b-hf (base)
├── meta-llama/Llama-2-7b-chat-hf (RLHF fine-tune) → DESCENDANT
├── NousResearch/Llama-2-7b-hf (community copy) → DESCENDANT  
├── openlm-research/open_llama_7b (independent retrain) → NON-DESCENDANT
└── mistralai/Mistral-7B-v0.1 (different architecture) → NON-DESCENDANT
```

#### BERT Distillation Family
```
google-bert/bert-base-uncased (base)
├── distilbert/distilbert-base-uncased (distilled) → NON-DESCENDANT (no weight inheritance)
├── huawei-noah/TinyBERT_General_4L_312D (distilled) → NON-DESCENDANT
├── prajjwal1/bert-tiny (smaller, retrained) → NON-DESCENDANT
└── bert-base-uncased fine-tunes (task-specific) → DESCENDANT
```

**Implementation:**
```python
MODEL_LINEAGE_GROUND_TRUTH = {
    # (reference, suspect): is_descendant
    ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"): True,
    ("meta-llama/Llama-2-7b-hf", "NousResearch/Llama-2-7b-hf"): True,
    ("meta-llama/Llama-2-7b-hf", "openlm-research/open_llama_7b"): False,
    ("meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"): False,
    ("google-bert/bert-base-uncased", "distilbert/distilbert-base-uncased"): False,
    # ... extend with more pairs
}
```

### 2.2 Model Zoos Dataset

**Paper:** Schürholt et al., "Model Zoos: A Dataset of Diverse Populations of Neural Network Weights" (NeurIPS 2022 Datasets Track, arXiv:2209.14764)

**What it offers:**
- 50,360 unique models across 27 model zoos
- Systematic hyperparameter variations
- Sparsified versions included
- Documented training trajectories

**URL:** modelzoos.cc (data accessible via paper)

**Use case:** Test if our method can distinguish:
- Same architecture, different seeds (hard negative)
- Same architecture, different hyperparameters (hard negative)
- Checkpoints from same training run (should be descendants of each other)

### 2.3 FingerBench

**Paper:** NaturalFinger (arXiv:2305.17868)

**What it offers:**
- 154 models for fingerprinting evaluation
- Standard ARUC metric

**Use case:** Compare our method against NaturalFinger and MetaV baselines.

### 2.4 Cisco Model Provenance Kit

**GitHub:** https://github.com/cisco-ai-defense/model-provenance-kit

**What it offers:**
- Database of ~150 base models across 45+ families
- 8 provenance signals
- CLI for scanning models
- 111-pair calibration benchmark with Cohen's d

**Use case:** Compare our diagonal-dominance method against their 8 signals.

---

## Part 3: Implementation Plan

### Phase 1: Add Baselines to `lineage_detection.py`

```python
# New file: experiments/scripts/lineage_baselines.py

"""Baseline methods for lineage verification comparison."""

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------- Weight-space baselines (data-free, like ours) ----------

def aligned_frobenius(Ms_A, Ms_B):
    """Frobenius distance with Hungarian alignment."""
    ...

def singular_value_distance(Ms_A, Ms_B):
    """Compare singular value spectra of branch products."""
    ...

def weight_cosine_similarity(Ms_A, Ms_B):
    """Cosine similarity of flattened weight matrices."""
    ...

# ---------- Activation-space baselines (require probe data) ----------

def cka_lineage_score(model_A, model_B, probe_data, layer_pairs=None):
    """CKA-based lineage score with layer alignment."""
    ...

def svcca_lineage_score(model_A, model_B, probe_data, layer_pairs=None):
    """SVCCA-based lineage score."""
    ...

# ---------- Decision-boundary baselines (require classification task) ----------

def ipguard_similarity(model_A, model_B, probe_data, n_fingerprints=100):
    """IPGuard decision-boundary fingerprint match rate."""
    ...

def uap_subspace_similarity(model_A, model_B, probe_data, n_uaps=20):
    """UAP subspace alignment score."""
    ...
```

### Phase 2: Evaluation Script

```python
# New file: experiments/scripts/lineage_benchmark.py

"""Benchmark lineage methods on public model pairs."""

METHODS = {
    'diagonal_dominance': our_lineage_score,
    'aligned_frobenius': aligned_frobenius,
    'cka': cka_lineage_score,
    'svcca': svcca_lineage_score,
    'ipguard': ipguard_similarity,
}

BENCHMARKS = {
    'mlp_synthetic': load_mlp_benchmark(),      # Our existing benchmark
    'resnet_cifar': load_resnet_benchmark(),    # Our existing benchmark
    'llama_family': load_hf_lineage_pairs('llama'),
    'bert_family': load_hf_lineage_pairs('bert'),
    'model_zoos': load_model_zoos_subset(),
}

def run_benchmark(method_name, benchmark_name):
    """Run one method on one benchmark, return AUROC + per-kind breakdown."""
    ...
```

### Phase 3: Results Table Format

| Method | MLP (n=159) | ResNet (n=32) | LLaMA (n=40) | BERT (n=40) | Mean |
|--------|-------------|---------------|--------------|-------------|------|
| **Diagonal Dominance (ours)** | 1.000 | 1.000 | ? | ? | ? |
| Aligned Frobenius | ? | ? | ? | ? | ? |
| CKA | ? | ? | ? | ? | ? |
| SVCCA | ? | ? | ? | ? | ? |
| IPGuard | N/A (regression) | ? | ? | ? | ? |

---

## Part 4: Sample Sizes (from Review)

The review recommends:
- **36 descendants + 36 non-descendants per model family** for 95% CI lower bound > 0.90
- Target **40-50 per family** in practice

For the LLaMA family, this means:
- 36+ fine-tunes (same-task, different-task, LoRA, QLoRA)
- 36+ non-descendants (independent retrains, different architectures, distilled students)

---

## Part 5: Dependencies

```
# requirements_baselines.txt
torch>=2.0
transformers>=4.30
scipy>=1.10
numpy>=1.24
scikit-learn>=1.2
```

For CKA/SVCCA: no external deps, pure numpy.
For IPGuard: need adversarial attack library (foolbox or custom PGD).
For UAP: need iterative attack implementation.

---

## Appendix: Key Papers

| Method | Paper | arXiv | Venue |
|--------|-------|-------|-------|
| CKA | Similarity of Neural Network Representations Revisited | 1905.00414 | ICML 2019 |
| SVCCA | SVCCA for Deep Learning Dynamics | 1706.05806 | NeurIPS 2017 |
| IPGuard | Protecting IP via Classification Boundary Fingerprinting | - | AsiaCCS 2021 |
| UAP Fingerprint | Fingerprinting DNNs via UAPs | 2202.08602 | CVPR 2022 |
| Model Zoos | Dataset of Diverse NN Weights | 2209.14764 | NeurIPS 2022 D&B |
| NaturalFinger | FingerBench benchmark | 2305.17868 | - |
| GhostSpec | LLM Spectral Signatures | 2511.06390 | AAAI 2026 |
| TensorGuard | Gradient-based LLM Fingerprinting | 2506.01631 | - |

---

## Next Steps

1. [ ] Implement CKA/SVCCA baselines (2 days)
2. [ ] Implement aligned Frobenius baseline (0.5 day)
3. [ ] Set up HuggingFace model loading for LLaMA/BERT families (1 day)
4. [ ] Run diagonal dominance on LLaMA family, verify it works (1 day)
5. [ ] Run all baselines on all benchmarks (2 days compute)
6. [ ] Write results table and update RQ.md (1 day)

**Total estimate:** 1 week to strong baseline comparison.

---

## Tracking

GitHub Issue: https://github.com/rayankhouryy/I-dropped-a-neural-net/issues/44
