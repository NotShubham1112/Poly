# Poly — Project Memory

## Project Overview
Polymer property prediction project for **ANRF AISEHack 2.0** (Kaggle competition). The competition track is **Polymer Property Prediction** contributed by IIT Madras. Two target properties: **Tg** (Glass Transition Temperature) and **Egc** (Chain Band Gap). Evaluation metric: **Mean R²** = (R²_Tg + R²_Egc) / 2.

## Folder Structure
```
Poly/
├── README.md                              # PolyChain architecture research proposal
├── ANRF_AISEHack_2.0_Polymer_Property_Prediction.md  # Competition overview doc
├── AGENTS.md                              # This file - project memory
├── Beginners/                             # 13 beginner guide markdown files
├── Material/                              # 6 research paper PDFs
└── polymer_competition/                   # Full codebase
    ├── config.yaml                        # Global config (5-fold CV, regression task)
    ├── generate_all.py                    # Master pipeline orchestrator
    ├── data/                              # train.csv (903 samples), test.csv (227 samples)
    ├── features/                          # Graphs, fingerprints, RDKit descriptors
    ├── models/                            # PolyChain (HAMF, PECGN, CST) + baselines
    ├── training/                          # train.py, run_all_folds.py, run_ablation.py
    ├── inference/                         # PolymerPredictor + Streamlit chat UI
    ├── ensemble/                          # build_ensemble.py, weight_optimizer.py
    ├── reports/                           # Visualizations (8 plot types)
    ├── notebooks/                         # PolyChain_Colab.ipynb
    ├── tests/                             # Unit tests
    └── docs/                              # Architecture docs
```

## Key Architecture: PolyChain
- **HAMF** (Hierarchy-Aware Multi-Scale Fusion): Cross-attention across monomer/dimer/trimer scales
- **PECGN** (Periodic Equivariant Chain-Growth Network): Learned boundary operator for periodic invariance
- **CST** (Chain Statistics Token): SMILES-derived polymer features (repeat length, branching, rings)

## Model Types Available
`ridge`, `xgb`, `lgb`, `catboost`, `rf`, `mlp`, `gcn`, `gat`, `mpnn`, `graph_transformer`, `polychain`

## Training Status (v29 — merged to main)
| Model | TG 5-fold | EGC 5-fold | Mean R² (TG) | Mean R² (EGC) |
|-------|-----------|------------|---------------|----------------|
| xgb | DONE | DONE | ~0.859 | ~0.906 |
| lgb | DONE | DONE | ~0.860 | ~0.899 |
| catboost | DONE | DONE | ~0.852 | ~0.902 |
| rf | DONE | DONE | ~0.835 | ~0.881 |
| mlp | DONE | DONE | ~0.849 | ~0.881 |
| mlp (multi-seed) | NOT RETRAINED | NOT RETRAINED | — | — |
| gcn | DONE | DONE | ~0.684 | ~0.722 |
| gat | DONE | DONE | ~0.706 | ~0.701 |
| mpnn | DONE | DONE | ~0.668 | ~0.796 |
| polychain | 1/5 (TG only) | 0/5 | ~0.847 (1 fold) | N/A |
| ridge | SKIPPED | SKIPPED | -0.019 (useless) | N/A |
| graph_transformer | NOT TRAINED | NOT TRAINED | N/A | N/A |

Ensemble: 8-model weighted average (xgb, lgb, catboost, rf, mlp, gcn, gat, mpnn)
Submission: `outputs/submissions/submission.csv` (4115 rows)

## v29 Improvements (merged Jun 29)
1. **Target transforms**: Yeo-Johnson + RankGauss in `features/target_transforms.py`, auto-selected via `select_best_transform()`
2. **Topological graph invariants**: 16 RDKit descriptors (BalabanJ, BertzCT, Chi0-4n/v, Kappa1-3, HallKierAlpha) in `features/advanced_descriptors.py`
3. **GNN get_embedding()**: GCN/GAT/DMPNN return pooled graph embeddings via `model.get_embedding(data)` in `models/gnn.py`
4. **Multi-seed MLP ensemble**: `--n_seeds 5` and `--loss huber` args in `training/train.py`, per-seed checkpoints
5. **GNN embeddings as features**: Wired into `features/build_features.py` via `load_gnn_embeddings()` with SMILES-space mapping
6. **Multi-task learning**: `run_multitask.py` with uncertainty-weighted masking, dual-output head
7. **Level-2 stacking**: `ensemble/weight_optimizer.py` with diverse meta-learners (Ridge, Lasso, RF, XGB)
8. **Preprocessing pipeline**: `features/preprocessing.py` with zero-variance removal, NaN/Inf handling, correlation filter, MI-based selection

## Training Commands
- Full pipeline: `cd polymer_competition && python generate_all.py`
- Single model: `python -m training.train --model_type xgb --fold 0`
- Target-specific training: `python -m training.train --model_type mlp --fold 0 --target tg`
- Multi-seed MLP: `python -m training.train --model_type mlp --fold 0 --target tg --n_seeds 5 --loss huber`
- All folds: `python -m training.run_all_folds --models ridge,xgb,gcn,polychain`
- GNN embedding extraction: auto-runs when `--target` is set during GNN training
- Multi-task training: `python run_multitask.py --epochs 80`
- Submission: `python run_submission.py`
- Ablation: `python -m training.run_ablation --fold 0 --epochs 50`
- Ensemble: `python -m ensemble.build_ensemble --config config.yaml`
- Tests: `python -m pytest tests/`
- Tests (all but pre-existing failures): `python -m pytest tests/ --ignore=tests/test_ablate.py --ignore=tests/test_run_submission.py`

## v29 Status: Topological Invariants FAILED — v27 Remains Best

### Critical Session Learnings (Jun 29)
1. **v27 and v28 use IDENTICAL features** — `features_train.parquet` (6394 cols, no GNN embeddings). The ONLY difference is the splits file.
2. **v27 splits ≠ v28 splits** — Only 12-15% val_idx overlap per fold. Cannot naively ensemble OOF predictions across versions.
3. **Topological invariants HURT performance** — Adding 16 RDKit topological descriptors (BalabanJ, BertzCT, Chi0-4n/v, Kappa1-3, HallKierAlpha) via `FORCE_KEEP_COLS` in `preprocessing.py` degraded all models because the descriptors are computed on polymer SMILES with wildcards (`*`), producing unreliable values.
4. **Feature rebuild breaks models** — Rebuilding `features_train.parquet` via `build_features.py` changed periodic feature names and preprocessing, degrading ALL tree models by 0.004–0.030 R².
5. **Target transforms (quantile) auto-applied** — The `training.target_transforms.enabled: true` in config applies quantile transforms automatically. This was also present during v27 training.
6. **GNNs get zero weight** — The ensemble optimizer assigns 0% to GCN/GAT/MPNN because their individual R² (0.70–0.80) is far below trees (0.88–0.91).

### Current Best Ensemble (v27 — baseline 0.896)
| Target | Ensemble R² | Top Model | Top Weight |
|--------|-------------|-----------|------------|
| TG     | 0.8754      | lgb       | 36.7%      |
| EGC    | 0.9121      | xgb       | 40.2%      |
| Mean   | 0.8938      |           |            |

### Files Modified This Session
- `features/preprocessing.py` — Added `FORCE_KEEP_COLS` (topological invariants bypass MI selection). **REVERT THIS if you want v27 behavior.**
- `config.yaml` — Experiment version toggled. Currently set to `v27`.
- `data/splits_egc.pkl` and `data/splits_tg.pkl` — Restored from v27 reconstructed splits. Keys converted to `train`/`val` format.
- `data/splits_egc_v27.pkl` and `data/splits_tg_v27.pkl` — Reconstructed v27 splits (from prediction val_idx). Keys use `train_idx`/`val_idx`.

### v27 vs v29 Model Comparison (EGC)
| Model    | v27 R²  | v29 R² (topo added) | Delta   |
|----------|---------|---------------------|---------|
| xgb      | 0.9067  | 0.9027              | -0.004  |
| lgb      | 0.8986  | 0.8925              | -0.006  |
| catboost | 0.9021  | 0.8860              | -0.016  |
| rf       | 0.8811  | 0.8833              | +0.002  |
| mlp      | 0.8816  | 0.8523              | -0.030  |

### Next Steps to Reach 0.92
- **Multi-seed MLP** (10 seeds): Reduces variance, expected +0.003–0.005
- **Target transforms for TG only**: Yeo-Johnson may help TG (currently 0.875)
- **Retrain MLP with Huber loss**: `--loss huber` for robustness to outliers
- **Consider GNN retraining**: Current GNNs use v27 splits but their R² is too low. A better GNN architecture or training regime could add diversity.
- **Do NOT rebuild features**: The current `features_train.parquet` (6394 cols) is the best feature set. Rebuilding changes feature names and preprocessing, degrading performance.

## Evaluation Metric
**Mean R²** = (R²_Tg + R²_Egc) / 2. Submission format: CSV with `id` and `target` columns.

## Competition Rules
- Notebook-only submissions on Kaggle
- Reproducibility required
- 30 hrs/week GPU compute
- No external data or pre-trained weights
- Must use only competition-provided data

## Cognee Memory
Cognee (v1.2.1) is installed for long-term project memory. Use:
- `cognee.remember("text")` to store context
- `cognee.recall("query")` to retrieve context

## Data Source
Current `polymer_competition/data/train.csv` has 903 samples with a single `property` column (placeholder). Real AISEHack 2.0 competition data requires Kaggle invite (register at https://precog.iiit.ac.in/aisehack).

## Key URLs
- Kaggle: https://www.kaggle.com/competitions/aisehack-2-0
- Registration: https://docs.google.com/forms/d/12EzeczXIiSJUyazcDZtvDaYsM2Q1dHbOsP3kxStrn30/viewform
- Website: https://precog.iiit.ac.in/aisehack
- GitHub: https://github.com/NotShubham1112/Poly
