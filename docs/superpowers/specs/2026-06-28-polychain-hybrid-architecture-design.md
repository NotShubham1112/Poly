# Hybrid Architecture for AISEHack 2.0 Polymer Property Prediction

**Date:** 2026-06-28  
**Author:** PolyChain Research Team  
**Status:** Design Specification  

---

## 1. Objective

Maximize leaderboard performance (Mean R²) on the AISEHack 2.0 Polymer Property Prediction competition by combining complementary modeling paradigms: graph neural networks (PolyChain), gradient-boosted tree ensembles, and handcrafted molecular/polymer features. Each component is validated through controlled ablation experiments before inclusion in the final ensemble. All models are trained from scratch on competition data only — no pre-trained weights or external datasets are permitted.

**Metric:** Mean R² = (R²_Tg + R²_Egc) / 2

---

## 2. Competition Constraints

| Constraint | Detail |
|---|---|
| **Data** | Only competition-provided data (train.csv: 6171 samples, test.csv: 4115 samples) |
| **Submission** | Notebook-only on Kaggle — all submissions must be backed by a compliant Kaggle Notebook |
| **Reproducibility** | Notebook must reproduce submitted results end-to-end |
| **External data & artifacts** | **No private artifacts, external datasets, or pre-trained weights allowed.** All models must be trained from scratch on competition data only. RDKit (listed as an approved tool) is acceptable for feature computation. |
| **GPU quota** | 30 hours/week/participant of Kaggle GPU compute. Local development is permitted but final submission must run within this limit. |
| **Targets** | Two properties: Tg (Glass Transition Temperature) in °C and Egc (Chain Band Gap) in eV |

---

## 3. Architecture

### 3.1 Data Flow

```
SMILES
   │
   ├── Fingerprints (Morgan 2048 + MACCS 167)
   ├── RDKit molecular descriptors (~210)
   ├── Polymer-specific descriptors (repeat length, branching, ring stats, end-groups)
   └── Graph construction (monomer/dimer/trimer for PolyChain)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                    FEATURE MATRIX (trees)                     │
│         Fingerprints + RDKit desc + Polymer desc             │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│              TREE ENSEMBLE (XGB, LGB, CatBoost, RF, MLP)     │
│              5-fold CV per model, Optuna hyperparameter tune │
│              Out-of-fold predictions                          │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│                    POLYCHAIN (GNN)                            │
│  Multi-scale graphs + HAMF + PECGN + CST                     │
│  5-fold CV, per-target independent models                    │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│              MULTI-LEVEL STACKING                             │
│  OOF predictions → meta-model (Ridge per target)             │
│  Weight optimization (Nelder-Mead per target)                │
│  Leakage-free: stacking within each fold's held-out set      │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│                    SUBMISSION                                 │
│  submission.csv (id, target)                                  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Feature Tiers

| Tier | Features | Dim | For Trees | For PolyChain |
|---|---|---|---|---|
| T1 | Morgan fingerprints (radius=2, 2048 bits) + MACCS keys (167) | 2215 | ✓ | — |
| T2 | RDKit molecular descriptors (~210) | ~210 | ✓ | — |
| T3 | Polymer-specific: repeat length, branching indicator, ring counts/sizes, end-group counts, aromatic fraction, backbone atoms | ~20-50 | ✓ | ✓ (as CST) |
| T4 | Multi-scale graphs: monomer, dimer, trimer with bond features + asterisk flags | — | — | ✓ |

No pretrained transformer embeddings — competition rules prohibit external pre-trained weights. All features are computed via RDKit (approved tool) or engineered from SMILES directly.

#### Feature Preprocessing

Before feeding features into models, apply the following pipeline (configurable per target):

| Step | Method | Rationale |
|---|---|---|
| Missing values | Zero-fill for RDKit descriptors (invalid SMILES) | Consistent with competition baseline |
| Variance threshold | Remove features with zero variance across training set | No information content |
| Correlation filter | Remove features with pairwise correlation > 0.95 | Reduces multicollinearity for Ridge/MLP |
| Scaling | StandardScaler (for Ridge/MLP only); tree models are scale-invariant | Some models are sensitive to feature scale |
| Inf/nan handling | Replace inf with column median; nan with 0 | Prevents numerical errors in training |

All preprocessing is fit on training data only, then applied to test data (no leakage).

### 3.3 Model Components

#### PolyChain (Graph Neural Network)
- **Backbone**: GIN-S with edge-aware message passing (4 layers, hidden_dim=256)
- **HAMF**: Hierarchy-Aware Multi-Scale Fusion (cross-attention across monomer/dimer/trimer)
- **PECGN**: Periodic Equivariant Chain-Growth Network (learned boundary operator)
- **CST**: Chain Statistics Token (repeat length, branching, rings)
- **Training**: 200 epochs, early stopping patience 30, batch_size 32
- **Per-target**: Independent models for Tg and Egc (different structural patterns)

#### Tree Ensemble (Tabular Models)
- **Primary models**: XGBoost, LightGBM, CatBoost
- **Optional**: Random Forest (weakest performer, only if compute allows)
- **Neural**: MLP (FingerprintMLP)
- **Features**: T1 + T2 + T3 concatenated
- **Tuning**: Optuna Bayesian optimization, number of trials determined by available runtime budget
- **CV**: 5-fold, out-of-fold predictions collected for stacking
- **No pretrained embeddings**: Competition rules prohibit external pre-trained weights; all features must be computed from SMILES using RDKit or engineered from scratch

### 3.4 Ensemble & Stacking

| Level | Input | Method | Output |
|---|---|---|---|
| L1 | OOF predictions per model per fold | Ridge regression (per target) | L1 meta-features |
| L2 | L1 features | Nelder-Mead weight optimization | Final ensemble weights |
| Calibration | Ensemble predictions | Residual correction per target (regression-specific) | Calibrated predictions |

Stacking is leakage-free: for each fold, the meta-model is trained on OOF predictions from the other 4 folds.

---

## 4. Execution Phases

Expected runtime depends on configuration and hardware (local RTX 3050 6GB, Colab T4 16GB, Kaggle T4/P100).

### Phase 1: Feature Engineering (Local CPU)
- Build fingerprint + RDKit descriptor + polymer-specific feature pipeline
- Generate 5-fold CV splits (GroupKFold by SMILES scaffold)
- Cache features to `data/processed/features_*.parquet`
- **Verification**: Confirm 0 NaN, correct dimensions, distribution plots

### Phase 2: Tree Model Training (Local CPU)
- Optuna hyperparameter tuning for each model per target
- 5-fold CV, collect OOF predictions + test predictions
- Models: XGB, LGB, CatBoost, RF, MLP (5 × 2 = 10 training runs)
- **Verification**: CV R² per model per fold, compare to baseline (0.888)

### Phase 3: PolyChain Training (Local GPU, Jupyter)
- 5 folds × 2 targets = 10 training runs
- Sequential execution (auto-resume via checkpoint)
- Per-target independent models
- **Verification**: CV R² per fold, compare to tree models

### Phase 4: Ensemble & Stacking (Local CPU)
- Multi-level stacking with leakage-free OOF meta-features
- Weight optimization per target
- Build final ensemble predictions on test set
- **Verification**: Ensemble CV R² > best individual model CV R²

### Phase 5: Kaggle Submission
- Reproduce final model in Kaggle notebook
- All training must complete within 30 hrs/week Kaggle GPU quota
- Full training notebook — run all phases end-to-end on Kaggle infrastructure
- Do NOT assume inference-only notebooks are allowed unless competition rules explicitly permit uploading trained checkpoints as notebook datasets
- Generate `submission.csv`
- **Verification**: Notebook reproduces submission identically on Kaggle

---

## 5. Ablation Experiments

Every major addition must justify itself through controlled experiments:

| Experiment | Description | Expected Δ (reference) |
|---|---|---|
| Baseline | Morgan + MACCS + XGB (reproduce 0.888) | — |
| + RDKit descriptors | Add RDKit molecular descriptors to features | TBD |
| + Polymer descriptors | Add polymer-specific features | TBD |
| + PolyChain | Add OOF predictions from GNN to ensemble | TBD |
| + Optuna tuning | Replace default params with tuned | TBD |
| + Stacking | Replace simple average with learned weights | TBD |
| + Calibration | Post-hoc per-target residual correction | TBD |

**PolyChain component ablations** (each removes one component from the full PolyChain):

| Ablation | What is removed | Purpose |
|---|---|---|
| Base GIN only | HAMF, PECGN, CST removed | Quantifies backbone contribution |
| + HAMF | Add multi-scale fusion | Quantifies cross-scale attention |
| + PECGN | Add periodic boundary operator | Quantifies periodicity equivariance |
| + CST | Add chain statistics token | Quantifies polymer-specific features |

No pretrained transformer embeddings — competition rules prohibit external pre-trained weights. All features are RDKit-based or engineered from SMILES.

Each row adds to the previous. Stop adding when ΔR² < 0.002.

---

## 6. Production-Ready Visualizations

Generated by `reports/run_all_visuals.py` after training completes:

| Plot | File Prefix | Purpose |
|---|---|---|
| Pred vs Actual (per model per target) | `pred_vs_actual_` | Scatter + diagonal + R² annotation |
| Residuals | `residuals_` | Distribution + Q-Q plot + heteroskedasticity check |
| CV per fold (bar + error) | `cv_per_fold_` | Mean R² ± std across folds per model |
| Model comparison | `model_comparison_` | Side-by-side R² bar chart with error bars |
| Ablation | `ablation_` | Stacked bar showing ΔR² per component |
| Feature importance | `shap_summary_` | SHAP summary plot (top 20 features) |
| Target distribution | `target_dist_` | Histogram of Tg and Egc (train vs test) |
| Calibration | `calibration_` | Observed vs predicted decile plot per target |
| Prediction correlation | `pred_corr_` | Correlation matrix of all model predictions |
| Ensemble diversity | `ensemble_diversity_` | Heatmap of pairwise prediction correlations |
| Ensemble weights | `ensemble_weights_` | Pie/bar chart of optimized blend weights |
| Error analysis | `error_analysis_` | Error vs SMILES length, ring count, etc. |

All plots:
- Publication-quality (matplotlib rcParams: font size, DPI, colorblind-friendly palette)
- Saved as PNG + PDF in `reports/plots/`
- Also displayed inline in Jupyter notebook

---

## 7. Experiment Tracking

Every experiment must produce the following artifacts for reproducibility:

| Artifact | Purpose |
|---|---|
| Configuration snapshot | YAML/JSON of all hyperparameters, features used, model settings |
| Random seed | Explicit seed for each run (fixed per experiment) |
| Git commit hash | Code version at time of run |
| Validation score | Mean R² + per-fold R² for each target |
| Runtime | Wall-clock time and GPU/CPU utilization |
| Model weights | Saved checkpoint (for local development only — not used in submission) |
| Predictions | OOF predictions and test predictions saved to `predictions/` directory |

Use a structured experiment directory:
```
outputs/experiments/
├── {experiment_id}/
│   ├── config.yaml
│   ├── metrics.json
│   ├── runtime.json
│   ├── predictions/
│   └── checkpoints/
```

Experiment IDs are sequential: `exp_001_baseline`, `exp_002_rdkit_desc`, etc.

---

## 8. Rules Compliance Checklist

- [ ] **Only competition-provided data used** — train.csv and test.csv only. No external polymer datasets.
- [ ] **No pre-trained weights** — all models trained from scratch on competition data. RDKit (approved tool) used only for feature computation, not as a pretrained model.
- [ ] **No external artifacts** — no loaded `.pkl`, `.pt`, or other weight files from the internet or local development in the final Kaggle notebook.
- [ ] **Notebook-only submission** — final submission is a Kaggle Notebook, not a CSV upload.
- [ ] **Reproducibility** — notebook must reproduce results end-to-end on Kaggle infrastructure.
- [ ] **30 hrs/week GPU quota** — final Kaggle notebook must complete within this limit. Local development is separate.
- [ ] **Submission format** — `submission.csv` with `id` and `target` columns.
- [ ] **Phase deadlines respected** — Phase 1: Jun 16 – Jul 16, 2026; Phase 2: Jul 20 – Aug 7, 2026.

---

## 9. Success Criteria

| Criterion | Target |
|---|---|
| Cross-validation Mean R² | > 0.900 (exceeds baseline 0.888) |
| Ablation: each component | ΔR² > 0.002 to justify inclusion |
| Leaderboard score | Measurable improvement over baseline; best achievable within competition constraints |
| Reproducibility | Kaggle notebook generates identical submission |
| Visualizations | All 12+ plot types generated without errors |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| PolyChain underperforms trees | Medium | Still valuable for ensemble diversity |
| Kaggle notebook exceeds 30 hr/week quota | Medium | Profile runtime early; reduce Optuna trials or epochs if needed |
| Kaggle notebook RAM/GPU limits | Low | Test on Kaggle early, reduce batch size if needed |
| Reproducibility failure | Low | Document random seeds, env, and data splits precisely |
| Local development can't be replicated on Kaggle | Medium | Use same libraries/versions; test final notebook on Kaggle early |
