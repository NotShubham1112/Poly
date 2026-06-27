# Poly -- Polymer Property Prediction Optimization Design

**Date**: 2026-06-27
**Current LB Score**: 0.888 (rank 21)
**Target**: 0.994+ (rank 1)
**Strategy**: Feature-Rich Deep Ensemble (Approach 2)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Feature Engineering](#2-feature-engineering)
3. [Experiment Scheduler](#3-experiment-scheduler)
4. [Model Zoo](#4-model-zoo)
5. [Training Pipeline](#5-training-pipeline)
6. [Validation & Leakage Prevention](#6-validation--leakage-prevention)
7. [Ensemble Strategy](#7-ensemble-strategy)
8. [Phase X: Kaggle Hardware Optimization](#8-phase-x-kaggle-hardware-optimization)
9. [Implementation Phases](#9-implementation-phases)
10. [Success Metrics](#10-success-metrics)

---

## 1. Architecture Overview

```
Raw Data (train.csv / test.csv)
         │
         ▼
Feature Engineering (parallel extraction)
         │
         ├── Tabular features ──► features/ (parquet cache, global)
         ├── Graph features   ──► graph_cache/ (separate from tabular)
         └── SMILES embeddings ──► features/ (config-optional, Phase 2)
         │
         ▼
Global Feature Cache
         │
         ▼
Experiment Scheduler
         │
         ├── Allocate GPU/CPU, launch order, resume failed runs
         ├── Per-fold: fit scaler/imputer/PCA on train, transform val
         └── Log metrics to manifest.json
         │
         ▼
Cross-Validation (5-fold scaffold-aware)
         │
         ├── Tabular models (CPU)  ───┐
         ├── GNN models (GPU)     ───┤── Parallel overlap
         └── PolyChain variants (GPU) ┘
         │
         ▼
OOF Predictions (N_train × 15 model types)
         │
         ▼
Flexible Meta-Learner
         ├── Candidates: RidgeCV | XGBoost | CatBoost | ElasticNet
         ├── Per-target selection (tg / egc independent)
         └── Stage-2 only if CV confirms improvement
         │
         ▼
Submission CSV
```

### Design Principles

- **Feature cache avoids recomputation**: global features computed once per SMILES (no label leakage). Graph features in separate storage.
- **GPU always saturated**: schedule CPU-tabular models in parallel with GPU-graph models.
- **Leakage prevention**: global cache is label-free. All fitted transforms (scaler, imputer, PCA) are fold-local.
- **Flexible stacking**: meta-model candidates evaluated by CV; stage-2 only added if empirically beneficial.
- **Resilient to crashes**: manifest.json tracks completed runs; Experiment Scheduler resumes from last checkpoint.
- **Config-gated components**: SMILES embeddings and two-level stacking are optional, toggled via config.

---

## 2. Feature Engineering

### 2A. Existing Features (reused from current pipeline)

| Feature Group | Dim | Description |
|--------------|-----|-------------|
| RDKit Descriptors | 208 | `DescriptorsCalculator` output |
| Morgan Fingerprints | 2048 | Radius 2, bit-vector |
| MACCS Keys | 166 | Structural keys |
| Multi-Scale Graphs | — | Monomer/dimer/trimer/periodic (PolyChain input) |
| PyG Graphs | — | For GCN/GAT/GraphTransformer |

### 2B. Polymer-Specific Descriptors (NEW -- highest feature ROI)

Implemented in `features/polymer_descriptors.py`:

| Descriptor | Method | Rationale |
|-----------|--------|-----------|
| Chain length estimate | Molecular weight / repeat unit MW | Tg scales with chain length |
| Tacticity indicator | SMILES stereochemistry pattern match | Affects Tg (isotactic vs atactic) |
| Ring statistics | Count rings, ring types, aromatic fraction | Rigidity affects both properties |
| Branching metrics | Terminal `*` count, branch points | Crosslinking affects Tg |
| Element composition | Si/F/Cl/O/N content ratios | Egc affected by heteroatoms |
| Rotatable bond fraction | RDKit `NumRotatableBonds / NumBonds` | Chain flexibility |
| TPSA / logP per repeat unit | RDKit | Polarity affects interactions |
| Flexibility index | Rotatable bonds / total bonds | Inverse correlation with Tg |

These are computed per-SMILES (no label), cached globally.

### 2C. Expanded Graph Features (NEW)

- **Atom features extended**: add formal charge, hybridisation, degree, implicit valence, ring membership
- **Bond features extended**: add conjugation flag, ring membership, stereo configuration
- **Tortuosity features**: path-length distribution between `*` attachment points

### 2D. Interaction Features (NEW)

- Pairwise descriptor interactions: top-50 by mutual information with target (fitted on train fold)
- PCA of high-correlation blocks: retain 95% variance (fitted on train fold)

### 2E. Embeddings (Phase 2, config-optional)

- **ChemBERTa-2** (`DeepChem/ChemBERTa-77M-MLM`): extract [CLS] token embedding (768-dim)
- Added as additional features (concatenated with descriptors)
- Config toggle: `features.use_embeddings: true/false`
- Cache computed embeddings to avoid re-running

### 2F. Feature Cache Strategy

- **Global cache** (no label leakage):
  - `features/features_train.parquet` -- SMILES → all descriptors, fingerprints, embeddings
  - `features/features_test.parquet` -- same transform
  - `graph_cache/train/` -- serialized PyG Data objects per SMILES
  - `graph_cache/test/` -- serialized PyG Data objects per SMILES
- **Fold-local artifacts** (fitted on train fold only):
  - Scaler (StandardScaler) -- saved per fold
  - Imputer (SimpleImputer strategy='median') -- saved per fold
  - PCA (if enabled) -- saved per fold
  - Feature selector -- saved per fold

No dataflows from val fold leak into train fold transforms.

---

## 3. Experiment Scheduler

### 3.1 Responsibilities

- Allocate GPU to GNN/PolyChain jobs, CPU to tabular jobs
- Launch jobs in optimal order (CPU-GPU overlap)
- Resume failed runs: check `manifest.json` for completed (model, fold, target) tuples
- Respect Kaggle runtime budget: if <30 min remaining, skip lowest-ROI models
- Log all metrics to `experiments/manifest.json`

### 3.2 Launch Order

```
Batch 1 (parallel): ridge (CPU) + gcn (GPU)
Batch 2 (parallel): xgb (CPU) + gat (GPU)
Batch 3 (parallel): lgb (CPU) + graph_transformer (GPU)
Batch 4 (parallel): catboost (CPU) + PC-base (GPU)  ← longest GPU job
Batch 5 (parallel): rf (CPU) + PC-deep (GPU)
Batch 6 (parallel): mlp (GPU) + PC-wide (GPU)  ← if 2×T4, split across GPUs
Batch 7: PC-light, PC-mt (GPU only)
```

On 2×T4: batches 4-7 can launch two GPU jobs simultaneously with `CUDA_VISIBLE_DEVICES=0/1`.

### 3.3 Manifest Tracking

```json
{
  "experiment": "v2",
  "target": "tg",
  "model_type": "polychain",
  "fold": 0,
  "score": 0.9123,
  "checkpoint": "outputs/checkpoints/v2_tg_polychain_fold0_best.pt",
  "duration_sec": 1862,
  "seed": 42,
  "config_path": "config.yaml",
  "completed": true
}
```

On startup, scheduler reads all completed entries and skips them.

### 3.4 Runtime Budget Enforcement

```
Check remaining time after each batch:
- If >60 min: continue with full schedule
- If 30-60 min: skip PolyChain variants (keep PC-base only)
- If <30 min: skip all GNN/PolyChain, use only existing tabular OOF
```

---

## 4. Model Zoo

### 4.1 Model Inventory (15 model types × 5 folds = 75 runs)

#### Tabular (6 models)

| Model | Config Source | Training Device |
|-------|--------------|-----------------|
| Ridge | `config.yaml` | CPU |
| XGBoost | Optuna-tuned (50 trials) | CPU |
| LightGBM | Optuna-tuned (50 trials) | CPU |
| CatBoost | Optuna-tuned (50 trials) | CPU |
| Random Forest | `config.yaml` | CPU |
| MLP | `config.yaml` | GPU (if T4×2) or CPU |

#### GNN Baselines (3 models)

| Model | Config Source | Training Device |
|-------|--------------|-----------------|
| GCN | `config.yaml` | GPU |
| GAT | `config.yaml` | GPU |
| Graph Transformer | `config.yaml` | GPU |

#### PolyChain Family (4 models)

| Model | Layers (backbone) | HAMF layers | Hidden dim | Params est. |
|-------|-------------------|-------------|------------|-------------|
| PC-base | 4 | 2 | 256 | ~2.5M |
| PC-deep | 6 | 3 | 384 | ~7M |
| PC-wide | 4 | 2 | 512 | ~5M |
| PC-light | 3 | 1 | 128 | ~0.8M |

PolyChain variants share the graph cache -- graphs are built once and reused.

#### PolyChain Multi-Task (2 experimental, time-permitting)

| Model | Architecture |
|-------|-------------|
| PC-mt | Shared backbone, dual output heads (tg + egc) |
| PC-mt-w | Wider shared backbone (512 hidden) |

### 4.2 PolyChain Architecture Details

Each PolyChain variant uses:
1. **GIN-S backbone**: message-passing with sum pooling + virtual node
2. **HAMF**: cross-attention across monomer/dimer/trimer scales
3. **PECGN**: periodic boundary operator for equivariant chain reasoning
4. **CST**: SMILES-derived chain statistics (32-dim, normalized)

Training config stored in `models/polychain/configs/` (base.yaml).

---

## 5. Training Pipeline

### 5.1 Tabular Models

- Input: features from parquet cache + fold-specific scaler/imputer
- Optuna HPO runs first (50 trials, single fold, full data), best params cached to `configs/{model}_tuned.yaml`
- Best params applied to all 5 folds
- Early stopping: patience 20 for tree models
- Checkpoints saved to `outputs/checkpoints/`

### 5.2 GNN Models

- PyG DataLoader with `num_workers=2`, `pin_memory=True`
- AMP (`torch.cuda.amp.autocast`) enabled
- Gradient clipping at 1.0
- Cosine LR scheduler with 5-epoch warmup
- Batch size: binary search (try 32/64/128, pick largest fitting P100)
- Early stopping: patience 30

### 5.3 PolyChain Models

- Multi-scale graph construction (shared cache across variants)
- Same PyG DataLoader setup as GNNs
- AMP + gradient checkpointing (trades compute for memory)
- Gradient accumulation: 2 steps if batch_size limited (target: effective batch 32)
- Batch size: target 16, fallback to 8 if OOM
- CST calibration statistics computed on train fold only

### 5.4 Mixed Precision (AMP)

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    pred = model(batch)
    loss = criterion(pred, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Applied to all GPU-trained models (GNN, PolyChain, MLP).

### 5.5 Multi-GPU Strategy (2×T4)

- Simple `DataParallel` for individual models too small for DDP benefits
- Primary strategy: **independent job parallelism** via `CUDA_VISIBLE_DEVICES`
- E.g., PC-deep on GPU:0, PC-wide on GPU:1 simultaneously

---

## 6. Validation & Leakage Prevention

### 6.1 Scaffold-Aware Splitting

1. Compute Murcko scaffold for each SMILES (RDKit)
2. Cluster by scaffold (identical scaffold → same cluster)
3. Stratified split by target bin (5 bins)
4. Each fold: whole clusters assigned to train or val (no scaffold leakage)

Implementation: `training/splits.py` with pre-computed split files.

### 6.2 Leakage Prevention Rules

| Artifact | Scope | Storage |
|----------|-------|---------|
| Descriptors | Global (no target) | `features/*.parquet` |
| Fingerprints | Global (no target) | `features/*.parquet` |
| Polymer descriptors | Global (no target) | `features/*.parquet` |
| Graph objects | Global (no target) | `graph_cache/*/` |
| Embeddings | Global (no target) | `features/*.parquet` |
| Scaler | Fold-local | `outputs/scalers/` |
| Imputer | Fold-local | `outputs/scalers/` |
| PCA | Fold-local | `outputs/scalers/` |
| Feature selector | Fold-local | `outputs/scalers/` |
| HPO trials | Single fold (not all data) | `experiments/optuna/` |

### 6.3 Metrics Tracked

Per model per fold:
- R², RMSE, MAE, Spearman ρ
- Training time, peak GPU memory
- Checkpoint path

Per ensemble:
- CV R² (mean ± std across 5 folds)
- Per-target CV R² (tg, egc)
- Overall Mean R² = (tg_R² + egc_R²) / 2

---

## 7. Ensemble Strategy

### 7.1 OOF Matrix Construction

- Load all `{exp}_{target}_{model}_fold*.pkl` files
- Per sample: mean prediction across folds per model type
- Result: N_train × M matrix (M = number of completed model types)
- Drop models with incomplete OOF coverage (NaNs → impute with column mean)

### 7.2 Meta-Model Selection

For each target (tg, egc) independently:

```python
candidates = {
    "ridge": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
    "xgb": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1),
    "catboost": CatBoostRegressor(iterations=500, depth=3, verbose=0),
    "elasticnet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1]),
}

best_cv = -inf
best_name = None
for name, meta in candidates.items():
    scores = cross_val_score(meta, oof, y, cv=5, scoring="r2")
    if scores.mean() > best_cv:
        best_cv = scores.mean()
        best_name = name
```

### 7.3 Stage-2 Stacking (Conditional)

If Stage-2 improves CV R² by >0.002 on both targets:
1. Train XGBoost on OOF → predict OOF²
2. Train RidgeCV on OOF² → final blend

Otherwise, use Stage-1 only.

### 7.4 Submission Strategy

Three candidate submissions evaluated by CV:
1. **Weighted blend** (inverse-RMSE weights per model)
2. **Stacking blend** (best meta-model)
3. **Blended stacking** (linear combination of 1 and 2)

Submit the combination with highest CV Mean R².

---

## 8. Phase X: Kaggle Hardware Optimization

### 8.1 Environment Detection

```python
import subprocess
gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader", shell=True).decode()
# "Tesla P100-PCIE-16GB, 6.0" → sm_60
# "Tesla T4, 7.5" → sm_75
```

Based on detected GPU:
- P100 (sm_60): PyTorch 2.5.1+cu121
- T4 (sm_75): PyTorch 2.6.0+cu124
- Install via subprocess `--no-deps --force-reinstall` to bypass Kaggle kernel cache

### 8.2 GPU Memory Budget

| Component | P100 (16 GB) | T4 (16 GB) |
|-----------|-------------|------------|
| PyTorch + CUDA overhead | ~1.5 GB | ~1.5 GB |
| Model params (PC-deep, ~7M) | ~0.5 GB | ~0.5 GB |
| Graph batch (batch_size=16) | ~4 GB | ~4 GB |
| Gradients + optimizer states | ~2 GB | ~2 GB |
| AMP scaler + activations | ~3 GB | ~3 GB |
| **Total estimated** | **~11 GB** | **~11 GB** |
| Headroom | ~5 GB | ~5 GB |

### 8.3 Async DataLoader Configuration

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,           # CPU workers for graph preprocessing
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True,  # Avoid worker recreation per epoch
    prefetch_factor=2,        # Prefetch 2 batches per worker
)
```

### 8.4 Gradient Checkpointing (PolyChain)

```python
from torch.utils.checkpoint import checkpoint

# In backbone forward:
for layer in self.layers:
    h = checkpoint(layer, h, edge_index, edge_attr)
```

Reduces activation memory by ~40% at ~20% compute overhead.

### 8.5 CPU-GPU Overlap Schedule

```
Time ──────────────────────────────────────────────────────►
GPU:  [gcn fold 0] [gcn fold 1] [gcn fold 2] ... [PC-base fold 0] ...
CPU:  [ridge fold 0..4] [xgb fold 0..4] [lgb fold 0..4] ...
      ├────────────────── Overlap ────────────────────────┤
```

Tabular models (CPU) run in parallel with graph models (GPU).
MLP runs on CPU when GPU busy with GNN. On 2×T4, MLP moves to GPU.

### 8.6 Runtime Budget

| Phase | Est. Time (P100) | Est. Time (2×T4) |
|-------|------------------|-------------------|
| Environment Setup | 3 min | 3 min |
| Feature Cache | 5 min | 5 min |
| Optuna HPO (50 trials) | 15 min | 15 min |
| Tabular (6×5 folds, CPU) | 10 min | 10 min |
| GNN (3×5 folds, GPU) | 45 min | 25 min |
| PolyChain (4×5 folds, GPU) | 120 min | 60 min |
| Multi-task (2×5 folds, GPU) | 30 min | 15 min |
| Ensemble + Submission | 2 min | 2 min |
| Diagnostics | 1 min | 1 min |
| **Total** | **~231 min** | **~136 min** |

Kaggle limit: 540 min (9 hours). Well within limits with 2× margin.

### 8.7 GPU Utilization Monitoring

```python
# Captured every 30s during training
import subprocess, threading, time

def log_gpu():
    while training_running:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
            shell=True
        ).decode().strip()
        with open("logs/gpu_util.csv", "a") as f:
            f.write(f"{time.time()},{result}\n")
        time.sleep(30)

threading.Thread(target=log_gpu, daemon=True).start()
```

### 8.8 Resume from Crash

On every training run completion, `manifest.json` is updated. Scheduler startup:
1. Load manifest
2. Check which (model, target, fold) entries have `"completed": true`
3. Skip completed runs
4. Resume incomplete runs from their recovery checkpoint (`*_recovery.pt`)

### 8.9 Memory Profiling

```python
# Before and after each training run
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
```

Recorded in manifest alongside metrics.

---

## 9. Implementation Phases

### Phase 1: Core Pipeline Stabilization (Days 1-2)

| Task | Files | Effort |
|------|-------|--------|
| Implement polymer-specific descriptors | `features/polymer_descriptors.py` | Medium |
| Expand atom/bond features | `features/graphs.py` | Small |
| Implement Experiment Scheduler | `training/scheduler.py` | Medium |
| Enhance stacking ensemble (multi-candidate) | `ensemble/stacking_ensemble.py` | Small |
| Add scaffold-aware splitting | `training/splits.py` | Medium |
| Verify PolyChain trains on Kaggle P100 | Notebook Cell 4-6 | Small |
| Run full feature cache + all tabular models | Notebook | Small |

### Phase 2: Modeling Expansion (Days 3-4)

| Task | Files | Effort |
|------|-------|--------|
| Optuna HPO for all tree models | `training/tune.py` (existing) | Small |
| Train GNN baselines on GPU | Notebook | Medium |
| Train PolyChain family (4 variants) | Notebook | Large (GPU time) |
| Verify ensemble improvements | `ensemble/` | Small |
| Run diagnostics on GPU utilization | Notebook Cell 8 | Small |

### Phase 3: Kaggle Optimization (Days 5-6)

| Task | Files | Effort |
|------|-------|--------|
| AMP integration for all GPU models | `training/train.py` | Small |
| Gradient checkpointing for PolyChain | `models/polychain/backbone.py` | Small |
| Async DataLoader tuning | `training/train.py` | Small |
| Multi-GPU parallelism (2×T4) | `training/scheduler.py` | Medium |
| Runtime budget enforcement | `training/scheduler.py` | Small |
| Resume checkpoint testing | `training/train.py` | Small |
| GPU utilization logging | `training/scheduler.py` | Small |

### Phase 4: Final Push (Days 7-8)

| Task | Files | Effort |
|------|-------|--------|
| Interaction features (mutual info) | `features/interactions.py` | Medium |
| Two-level stacking (conditional) | `ensemble/stacking_ensemble.py` | Small |
| Full pipeline run on Kaggle | Notebook | Large (GPU time) |
| Error analysis on OOF residuals | Notebook / analysis | Medium |
| ChemBERTa embeddings (if time) | `features/embeddings.py` | Large |
| Multi-task PolyChain (if time) | `models/polychain/polychain_model.py` | Large |

---

## 10. Success Metrics

### Leaderboard Targets

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Mean R² (public LB) | 0.888 | 0.994+ | 0.106 |
| Tg R² | — | 0.990+ | — |
| Egc R² | — | 0.990+ | — |

### Internal CV Targets

| Milestone | Expected CV R² | Confidence |
|-----------|----------------|------------|
| Current (6 models, simple blend) | ~0.900 | High |
| + polymer-specific features | ~0.915 | High |
| + HPO | ~0.930 | High |
| + GNN baselines | ~0.940 | Medium |
| + PolyChain (PC-base) | ~0.955 | Medium |
| + PolyChain variants + stacking | ~0.970 | Medium |
| + Kaggle hardware optimizations | ~0.975 | Low (GPU-dependent) |
| + ChemBERTa / multi-task | ~0.985 | Low |
| **Target** | **0.990+** | — |

### Runtime Targets

| Metric | Target |
|--------|--------|
| P100 GPU utilization | >90% during graph training |
| 2×T4 GPU utilization | >80% each (parallel jobs) |
| End-to-end runtime (P100) | <180 min |
| End-to-end runtime (2×T4) | <120 min |
| No OOM failures | 100% success rate |
| Resume after crash | Resume within 2 minutes |
