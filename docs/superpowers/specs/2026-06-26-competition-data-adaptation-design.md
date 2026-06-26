# Competition Data Adaptation — Design Spec

**Date:** 2026-06-26
**Status:** Draft
**Version:** 1.0

## 1. Motivation

The competition host has released the official ANRF AISEHack 2.0 Polymer Property Prediction
dataset. The current pipeline was built on a 903-row placeholder with a single `property` column.
The real dataset has 6171 training rows, two target properties (Tg, Egc), and a `target_type`
column distinguishing them. The entire pipeline must be adapted while maintaining compatibility
with Kaggle notebook-only submission rules.

---

## 2. Data Overview

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 6171 | `smiles, target, target_type` | Tg: 4143, Egc: 2028 |
| `test.csv` | 4115 | `id, smiles, target_type` | Tg: 2763, Egc: 1352 |
| `sample_submission.csv` | 10 | `id, target` | Format spec |

The `target_type` column tells us which property each row measures (tg or egc). For the test
set, it tells us which property to predict for that sample. The submission format is a simple
`id, target` CSV — one prediction per test row.

---

## 3. Architecture

```
Raw Data
   │
   ▼
polymer_competition/data/train.csv
   │
   ▼
Split by target_type
   │
 ┌──┴─────────┐
 │             │
 ▼             ▼
train_tg.csv  train_egc.csv
4143 rows     2028 rows
   │             │
   └──────┬──────┘
          ▼
build_features.py
(one pass on all 6171 SMILES)
          │
          ▼
Feature Cache
(fingerprints + descriptors + custom polymer features)
          │
   ┌──────┴──────┐
   │             │
   ▼             ▼
splits_tg.pkl   splits_egc.pkl
(Scaffold GKF)  (Scaffold GKF)
   │             │
   ▼             ▼
5-fold CV      5-fold CV
11 model types  11 model types
   │             │
   ▼             ▼
Fold preds      Fold preds
(v1_tg_*.pkl)   (v1_egc_*.pkl)
   │             │
   ▼             ▼
Ensemble tg     Ensemble egc
   │             │
   ▼             ▼
tg_preds.csv    egc_preds.csv
   └──────┬──────┘
          ▼
merge_submissions.py
          ▼
submission.csv
(id, target)
```

### 3.1 Key Design Principle

Feature extraction is performed exactly once for every molecule using only its SMILES
representation (no target values are used during feature generation). The cached features
are subsequently split by target type, after which all cross-validation, model training,
ensembling, and inference are conducted independently for each prediction task.

This guarantees no target leakage between properties and avoids redundant computation.

---

## 4. Configuration (config.yaml)

```yaml
data:
  raw_path: "data/train.csv"
  smiles_col: "smiles"
  id_col: "id"

targets:
  tg:
    type: "tg"
    target_col: "target"
  egc:
    type: "egc"
    target_col: "target"

metric: "mean_r2"

experiment:
  version: "v1"
```

`n_samples` is derived automatically from data rather than hard-coded.

---

## 5. Data Preparation

### 5.1 Input Normalization

Column names are normalized (lowercase `smiles` → uppercase `SMILES`) immediately after
loading, keeping the rest of the pipeline backward-compatible. This is a column-name
normalization only — no chemical standardization at this stage.

### 5.2 SMILES Canonicalization

SMILES strings are canonicalized using RDKit before feature caching. This ensures that
equivalent SMILES representations produce identical feature vectors and avoids duplicate
cache entries.

### 5.3 Per-Target Splitting

`data/split_by_target.py` generates target-specific datasets by filtering `target_type`,
preserving original row IDs for later reconstruction of the competition submission.

Outputs per target:
- `data/tg/train.csv` — training samples with target values
- `data/tg/test.csv` — test samples with IDs
- `data/egc/train.csv`
- `data/egc/test.csv`

### 5.4 Cross-Validation Splits

Two independent `splits_tg.pkl` and `splits_egc.pkl` using scaffold-based GroupKFold within
each property's sample set. Split metadata includes:

```python
{
    "folds": [...],
    "n_samples": 4143,
    "cv": "ScaffoldGroupKFold",
    "random_state": 42,
    "created_at": "2026-06-26T12:00:00",
    "scaffold_algorithm": "Murcko"
}
```

Using scaffold-based grouping independently for each property ensures:
- Molecules from one property never influence folds of the other
- Scaffold leakage is prevented within each task
- Evaluation matches the competition objective (averaging two R² scores)

---

## 6. Feature Engineering

### 6.1 Pipeline Order

1. Normalize column names
2. Canonicalize SMILES (RDKit). Invalid SMILES are logged and excluded
   (or handled with a predefined fallback policy). Cache generation reports
   the total number of parsing failures so the pipeline is deterministic.
3. Generate cached features on ALL SMILES — produced separately for train
   (6171 molecules) and test (4115 molecules) to avoid accidental mixing:
   - `data/processed/features_train.parquet` — train cache
   - `data/processed/features_test.parquet` — test cache
4. Split feature cache by target_type

### 6.2 Feature Types

| Feature | Computation | Notes |
|---------|------------|-------|
| Morgan fingerprints | rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect | Radius 2, 2048 bits |
| MACCS keys | rdkit.Chem.MACCSkeys.GenMACCSKeys | 166-bit |
| RDKit FP | rdkit.Chem.RDKFingerprint | |
| RDKit descriptors | rdkit.Chem.Descriptors.CalcMolDescriptors | 200+ descriptors |
| Custom polymer features | features/custom_polymer.py | Repeat length, branching, rings |

### 6.3 Feature Cache

Two separate caches (train / test):

| Cache | Path | Rows |
|-------|------|------|
| Train | `data/processed/features_train.parquet` | 6171 |
| Test | `data/processed/features_test.parquet` | 4115 |

Cache columns include:
- `canonical_smiles`
- All fingerprint bits and descriptor values
- `feature_version` — version string for cache invalidation
- `git_commit` — current commit hash at build time
- `config_hash` — SHA256 of relevant config sections
- `rdkit_version` — RDKit version used for generation

If cache exists and version matches, `build_features.py` skips recomputation.

### 6.4 Graph Model Features

Molecular graphs are constructed on demand from canonical SMILES during training and
cached in memory within each training process. Graph objects are intentionally excluded
from the persistent feature cache due to their large serialized footprint.

### 6.5 Random Seeds

All seeds are defined centrally in `config.yaml` and inherited by every model:

```yaml
seed:
  global: 42
  numpy: 42
  torch: 42
  python: 42
  xgboost: 42
  catboost: 42
  lightgbm: 42
```

Every model instance applies its framework-specific seed at construction time.

---

## 7. Training Orchestration

### 7.1 Script

`generate_all.py` accepts a `--targets tg,egc` argument.

### 7.2 Loop Order

```
for target in tg, egc:
    prepare folds
    for model_type in ridge, xgb, ..., polychain:
        for fold in 0..4:
            train --model_type $m --fold $f --target $t
        validate
    ensemble
```

This naturally groups outputs per model and makes orchestration easier.

### 7.3 Total Model Count

11 model types × 5 folds × 2 targets = 110 training runs.

### 7.4 Checkpointing & Resume

```
if checkpoint exists:
    skip training
else:
    train model
    save checkpoint
```

Checkpoints saved to `outputs/checkpoints/v1_tg_xgb_fold0_best.pt` etc. Every N epochs,
recovery checkpoints (including optimizer + scheduler state) are saved for automatic
resume after disconnection.

### 7.5 Experiment Manifest

Location: `experiments/manifest.json`

```json
{
    "target": "tg",
    "model": "xgb",
    "fold": 3,
    "status": "completed",
    "score": 0.8842,
    "checkpoint": "outputs/checkpoints/v1_tg_xgb_fold3_best.pt",
    "duration": 531,
    "seed": 42
}
```

This enables:
- Identifying failed runs for re-execution
- Cross-experiment comparison
- Reproducible publication

---

## 8. Inference

For each target, a dedicated inference pass generates test-set predictions:

```
for target in tg, egc:
    for model_type in ridge, xgb, ..., polychain:
        for fold in 0..4:
            load checkpoint
            predict on target's test subset
            save fold predictions
    average fold predictions per model_type
    store aggregated predictions for ensemble
```

Model checkpoints are loaded from `outputs/checkpoints/v1_{target}_{model}_fold{fold}_best.pt`.
Fold predictions are averaged to produce one test prediction per model per target. These
aggregated predictions are consumed by the ensemble step.

Tabular models (ridge, xgb, lgb, etc.) apply the same `StandardScaler` and
`feature_cols` serialized at training time.

---

## 9. Ensemble & Submission

### 9.1 Artifact Flow

```
OOF predictions (per model × fold)
        ↓
Weight optimization (OOF matrix only)
        ↓
Fold test predictions (per model × fold)
        ↓
Blended test prediction (weighted average)
```

Ensemble weights are estimated using **only out-of-fold (OOF) predictions** generated
during cross-validation. The learned weights are subsequently applied unchanged to the
corresponding test-set predictions. This guarantees no leakage.

### 9.2 Per-Target Ensembles

Two independent ensemble runs, one per target.

Prediction files:
```
predictions/
  v1_tg_ridge_fold0.pkl
  v1_tg_xgb_fold0.pkl
  ...
  v1_egc_ridge_fold0.pkl
  v1_egc_xgb_fold0.pkl
```

Weight files:
```
ensembles/
  v1_tg_weights.json
  v1_egc_weights.json
```

Example weight file:
```json
{
  "strategy": "Nelder-Mead",
  "weights": {
    "ridge": 0.12,
    "xgb": 0.31,
    "catboost": 0.18,
    "polychain": 0.39
  },
  "cv_score": 0.8871
}
```

### 9.3 Stacking Protocol

1. Generate OOF predictions for every base model
2. Train the meta-model only on the OOF matrix
3. Generate fold-wise test predictions from each base model
4. Average fold predictions per base model
5. Apply the trained meta-model to averaged test predictions

### 9.4 Ensemble Performance Tracking

`ensembles/ensemble_results.csv`:

```
target,strategy,cv_score,public_lb,private_lb,timestamp,experiment
tg,Nelder-Mead,0.8871,,,2026-06-26T12:00:00,v1
egc,Nelder-Mead,0.7643,,,2026-06-26T12:00:00,v1
```

### 9.5 Competition Metric

The evaluation metric is the mean of per-target R² scores:

\[
\text{MeanR}^2 = \frac{R^2_{Tg} + R^2_{Egc}}{2}
\]

Each R² is computed independently per target using `sklearn.metrics.r2_score`.

### 9.6 Submission Validation

Automatic checks run at each pipeline stage:

| Stage | Check | Action |
|-------|-------|--------|
| Data load | No duplicate IDs | Assert |
| Canonicalization | All SMILES parse successfully | Log failures, assert ≥ 99% |
| Feature build | No missing descriptors (after imputation) | Assert |
| Feature build | Feature dimensions match train/test | Assert |
| CV split | All samples assigned to exactly one fold | Assert |
| Training | OOF pred count == val set size | Assert |
| Ensemble | All model types present in OOF matrix | Assert |
| Submission | Prediction count == test rows (4115) | Assert |
| Submission | Columns exactly `id, target` | Assert |

### 9.7 Submission Assembly

`merge_submissions.py`:
1. Validates that all competition samples are covered exactly once
2. Concatenates TG and EGC predictions
3. Sorts by original competition IDs
4. Verifies expected row count (4115)
5. Writes `submission.csv` with only `id, target` columns

---

## 10. Reproducibility & Dependencies

### 10.1 Seeded Randomness

All random seeds are defined in `config.yaml` (see §6.5). Every training run,
split generation, and data shuffle uses these seeds.

### 10.2 Pinned Dependencies

Major packages pinned to specific versions for the Kaggle notebook:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| RDKit | 2024.03+ | Molecular featurization |
| PyTorch | 2.1+ | Neural network models |
| PyTorch Geometric | 2.5+ | Graph neural networks |
| XGBoost | 2.0+ | Tree model |
| CatBoost | 1.2+ | Tree model |
| LightGBM | 4.3+ | Tree model |
| scikit-learn | 1.4+ | Baselines, metrics, CV |

Versions are recorded in `experiments/environment.txt` at each run.

### 10.3 Deterministic Execution

`config.yaml` includes `deterministic: true`, enabling:
- `torch.backends.cudnn.deterministic = True`
- `torch.use_deterministic_algorithms(True)`
- Python `hashseed` fixed via `PYTHONHASHSEED`

---

## 11. Kaggle Notebook Adaptation

### 11.1 Requirements

- All code must run within a single Kaggle notebook execution
- No external data or pretrained weights
- Reproducible with pinned notebook version

### 11.2 Strategy

The full pipeline (feature extraction → training → ensemble → submission) will be
wrapped into a single orchestration notebook. Key adaptations:
- Minimal dependencies, all installable via `!pip` in the notebook
- Feature cache eliminates redundant computation across notebook runs
- Checkpointing enables resume if Kaggle session times out
- All random seeds explicitly set for reproducibility

---

## 12. Directory Layout

```
polymer_competition/
├── config.yaml                  # Global config (dual targets, seeds, paths)
├── data/
│   ├── train.csv                # Original competition data (6171 rows)
│   ├── test.csv                 # Original test data (4115 rows)
│   ├── tg/
│   │   ├── train.csv            # Filtered by target_type=tg
│   │   └── test.csv
│   ├── egc/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   ├── features_train.parquet  # Train feature cache
│   │   └── features_test.parquet   # Test feature cache
│   ├── splits_tg.pkl            # CV splits for Tg
│   └── splits_egc.pkl           # CV splits for Egc
├── features/                    # FP, descriptors, graphs
├── models/                      # All model architectures
├── training/                    # train.py, train_utils.py, configs/
├── ensemble/                    # build_ensemble.py, weight_optimizer.py
├── predictions/                 # OOF + test prediction .pkl files
├── outputs/
│   ├── checkpoints/             # Model checkpoints
│   ├── submissions/             # submission.csv
│   └── logs/                    # Training logs
├── ensembles/                   # Weight files + ensemble_results.csv
├── experiments/                 # manifest.json, environment.txt
├── reports/                     # Plots, SHAP, model summaries
├── notebooks/                   # kaggle_pipeline.ipynb
└── docs/                        # Architecture documentation
```

---

## 13. Files to Create/Modify

### New files
| File | Purpose |
|------|---------|
| `data/split_by_target.py` | Filter data by target_type |
| `data/merge_submissions.py` | Concatenate tg/egc predictions → submission.csv |
| `notebooks/kaggle_pipeline.ipynb` | Single notebook for Kaggle submission |
| `experiments/manifest.json` | Experiment tracking manifest |

### Modified files
| File | Changes |
|------|---------|
| `config.yaml` | Dual target config, experiment version |
| `data/generate_splits.py` | Accept target param, lowercase smiles |
| `features/build_features.py` | Feature cache, canonicalization, per-target split |
| `training/train.py` | Accept `--target` arg, lowercase smiles |
| `generate_all.py` | `--targets` flag, hierarchical loop |
| `ensemble/build_ensemble.py` | Per-target operation |

---

## 14. Open Questions

At the time of writing, no unresolved architectural decisions remain.
Future revisions may address performance optimizations or additional model
families without changing the overall pipeline structure.
