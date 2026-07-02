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

## Training Status (v27)
| Model | TG 5-fold | EGC 5-fold | Mean R² (TG) | Mean R² (EGC) |
|-------|-----------|------------|---------------|----------------|
| xgb | DONE | DONE | ~0.859 | ~0.906 |
| lgb | DONE | DONE | ~0.860 | ~0.899 |
| catboost | DONE | DONE | ~0.852 | ~0.902 |
| rf | DONE | DONE | ~0.835 | ~0.881 |
| mlp | DONE | DONE | ~0.849 | ~0.881 |
| gcn | DONE | DONE | ~0.684 | ~0.722 |
| gat | DONE | DONE | ~0.706 | ~0.701 |
| mpnn | DONE | DONE | ~0.668 | ~0.796 |
| polychain | 1/5 (TG only) | 0/5 | ~0.847 (1 fold) | N/A |
| ridge | SKIPPED | SKIPPED | -0.019 (useless) | N/A |
| graph_transformer | NOT TRAINED | NOT TRAINED | N/A | N/A |

Ensemble: 8-model weighted average (xgb, lgb, catboost, rf, mlp, gcn, gat, mpnn)
Submission: `outputs/submissions/submission.csv` (4115 rows)

## Training Commands
- Full pipeline: `cd polymer_competition && python generate_all.py`
- Single model: `python -m training.train --model_type xgb --fold 0`
- All folds: `python -m training.run_all_folds --models ridge,xgb,gcn,polychain`
- Ablation: `python -m training.run_ablation --fold 0 --epochs 50`
- Ensemble: `python -m ensemble.build_ensemble --config config.yaml`

## Evaluation Metric
**Mean R²** = (R²_Tg + R²_Egc) / 2. Submission format: CSV with `id` and `target` columns.

## Competition Rules
- Notebook-only submissions on Kaggle
- Reproducibility required
- 30 hrs/week GPU compute
- No external data or pre-trained weights
- Must use only competition-provided data

## Final Results & Key Lessons (Post-Competition)
- **Best competition score**: 0.900 (Ridge meta-stacker: GIN + XGB + Hybrid → 3-model Ridge, LB verified)
- **Previous best**: 0.896 (Arch A = v27 ensemble: xgb+lgb+catboost+rf+mlp)
- **Post-competition OOF**: 0.8972 (Ridge on GIN 24.5 + XGB 40.7 + Hybrid 38.4)
- **TG: 0.8780** (Ridge on GIN 24.5 + XGB 40.7 + Hybrid 38.4)
- **EGC: 0.9163** (Ridge on GIN 0.43 + XGB 0.68 + Hybrid 0.39)
- **Best submission file**: `outputs/final_submission/submission_final.csv` (4115 rows)

### GIN / Hybrid / Meta-Stacker Results
| Model | TG OOF | EGC OOF | Mean |
|-------|--------|---------|------|
| GIN (GINEConv+AttentionalPool) | 0.8473 | 0.8900 | 0.8686 |
| XGB (6394 features) | 0.8647 | 0.9048 | 0.8847 |
| Hybrid (GIN+Tabular+LayerNormFusion) | 0.8608 | 0.8914 | 0.8761 |
| Ridge meta-stacker (GIN+XGB+Hybrid) | **0.8780** | **0.9163** | **0.8972** |

### Key Findings
1. **GIN beats GCN/GAT convincingly**: GIN (0.87 mean) vs GCN (0.68) / GAT (0.71) — isomorphism awareness matters for polymer graphs.
2. **Hybrid requires careful optimization**: Two-stage training (freeze GIN, then fine-tune) + LayerNormProjectedFusion was critical. Naive concat underperformed GIN alone.
3. **GIN embeddings are not tree-friendly**: 128-dim GIN embeddings as XGBoost features gave 0.32 R² (vs 0.86 for tabular alone). Dense neural embeddings don't align with tree split logic.
4. **GIN predictions are complementary**: Despite r(pred)=0.97 between all models, r(error)=0.68-0.82 → models make different errors, enabling ensemble gains.
5. **Residual correlations**: GIN vs Hybrid r(error)=0.68 for EGC (lowest), meaning GIN captures genuinely different signal than tabular+hybrid approaches.
6. **Meta-stacker beats simple blending**: Ridge on OOF predictions gives +0.001-0.004 over optimal 2-way/3-way blends.
7. **Ceiling insight remains**: 0.897 is the practical ceiling for 2D descriptors + monomer graphs. To break 0.92, need genuinely new information (3D conformers, molecular weight, self-supervised pretraining on graph encoder, or external data).

### Model Implementation Details
- **GIN**: 3-layer GINEConv (hidden_dim varies: fold 0=512, folds 1-4=256), embed_dim=128, AttentionalAggregation pooling, output_proj=128. Training: Adam(lr=0.001, wd=5e-4), 100 epochs, early stopping patience=20.
- **Hybrid**: GINEncoder (hidden=512, embed=128) + TabularEncoder (6394→1024→512) + LayerNormProjectedFusion (proj=256, head=512→256→128→1). Two-stage: freeze GIN 30 epochs, then full fine-tune lr/10. AdamW(lr=0.001).
- **Meta-Stacker**: Ridge(alpha=1.0) on OOF predictions from GIN, XGB, Hybrid. Test predictions use Ridge trained on per-fold OOF + scaled test predictions.

### Files Written During Post-Competition
- `models/gnn.py`: GINEncoder, GINRegressor
- `models/tabular.py`: TabularEncoder (6394→1024→512)
- `models/fusion.py`: ConcatFusion, ProjectedConcatFusion, LayerNormProjectedFusion
- `models/hybrid.py`: HybridNet (GIN+Tabular+Fusion)
- `training/train_gin.py`: Standalone GIN training
- `training/train_hybrid.py`: Hybrid training with 2-stage learning
- `training/train_gin_xgb.py`: GIN embeddings → XGBoost (failed: 0.43 R²)
- `training/final_blend.py`: Multi-model OOF blend analysis
- `training/layer2_embeddings.py`: Embedding extraction for Layer 2 models
- `training/meta_stacker.py`: Ridge/XGB meta-stacker on OOF predictions
- `training/generate_final_submission.py`: Final submission generator
- `outputs/hybrid/`: Hybrid checkpoints, OOF, submissions per target
- `outputs/final_submission/submission_final.csv`: Best submission (0.8972 mean R²)

## Cognee Memory
Cognee (v1.2.1) is installed for long-term project memory. Use:
- `cognee.remember("text")` to store context
- `cognee.recall("query")` to retrieve context

## Physics Ensemble Experiments (v4)
### Experiment 1: Full 6394 features + 3 physics features
- **Script**: `run_physics_ensemble.py` (v2)
- **Features**: Precomputed 6394 features (fingerprints, descriptors, interactions, ratios) + MolWt, NumRotBonds, RingCount
- **Models**: Ridge(alpha=1.0) + XGB(300 trees) + MLP(128→64)
- **Split**: Murcko scaffold 5-fold
- **Results**: TG(Ridge=0.25, XGB=0.84, MLP=0.80, Blend=0.85), EGC(Ridge=0.60, XGB=0.82, MLP=0.42, Blend=0.82)
- **Mean OOF**: 0.83, Expected LB: ~0.845

### Experiment 2: RDKit 2D descriptors only (~200 + physics)
- **Script**: `run_physics_ensemble.py` (v3)
- **Features**: 192 RDKit 2D descriptors + NumRotBonds, NumHBD, NumHBA
- **Models**: Same as Exp 1
- **Results**: TG(Ridge=0.45, XGB=0.81, MLP=0.74, Blend=0.81), EGC(Ridge=-7.56, XGB=0.76, MLP=-2.02, Blend=0.76)
- **Mean OOF**: 0.79 → Blueprint FAILS — RDKit descriptors alone are insufficient

### Experiment 3: XGB-Conditioned GIN (FiLM, Concat, Uncertainty)
- **Script**: `training/train_gin_conditioned.py`
- **Variants**: B (concat GIN + XGB pred), C (FiLM modulation), D (+ uncertainty)
- **Results (TG fold 0)**: All ~0.877 — tied. But r(error)=0.999 vs XGB — GIN simply copies XGB prediction, ignores graph structure.
- **Residual training**: GIN predicts y - XGB_pred. R²≈0 — no signal remains beyond XGB.
- **Conclusion**: Monomer graphs contain NO information beyond what the 6394 features capture.

### Experiment 4: Self-Supervised GIN Pretraining
- **Script**: `training/pretrain_gin.py`, `training/train_gin.py` (modified)
- **Task**: Masked atom prediction (15% nodes) on all 10264 unique SMILES
- **Training**: 200 epochs, loss 0.31→0.09
- **Downstream**: Fine-tuned on TG fold 0: pretrained GIN 0.8628 vs scratch GIN 0.8651
- **Frozen encoder**: Only 0.397 R² — pretrained features not directly usable
- **Conclusion**: Masked atom prediction doesn't align with regression task. No improvement over scratch.

### Experiment 5: Multi-Task GIN (shared encoder + TG + EGC heads)
- **Script**: `training/train_multitask_gin.py`
- **Design**: Shared GIN encoder + separate TG/EGC heads, alternating batches
- **Results (fold 0)**: TG R²=0.78, EGC R²=0.71 — WORSE than single-task for both
- **Conclusion**: Shared encoder confuses tasks. Two different physical properties (Tg vs Egc) don't benefit from parameter sharing at this data scale.

### Experiment 6: 3D Conformer Descriptors
- **Script**: `features/descriptors_3d.py` via temp scripts
- **Descriptors**: 12 basic (PMI, Rg, Asphericity) + 114 WHIM + 210 RDF + 273 GETAWAY + 224 MORSE + 80 AUTOCORR3D = 913 total
- **Generation**: 99.3% success, ~9.5 min for 10264 SMILES
- **XGB ablation (TG fold 0)**: base=0.8734, base+3D=0.8672 — 3D descriptors HURT
- **Conclusion**: 3D conformer descriptors are already implicit in the 6394 2D features. The 2D fingerprints, 3D pharmacophore, and shape descriptors in the 6394 set subsume 3D information.

### Key Findings
1. **Precomputed 6394 features outperform 195 RDKit descriptors** by ~0.04 R² — fingerprints, interactions, and ratios add critical signal
2. **Physics features (MolWt, NumRotBonds)** add negligible value on top of 6394 features because they're already represented in the rich feature set
3. **Scaffold split produces highly uneven folds** for EGC (e.g., [226, 257, 184, 243, 1118]) — models on small folds (184 samples) cannot generalize with 195+ features
4. **Ridge and MLP fail on EGC with scaffold split** — Ridge OOF=-7.56, MLP OOF=-2.02 on the 195-feature set
5. **Best combo**: Full 6394 features + XGB only (no Ridge/MLP blending needed)
6. **0.92 ceiling is NOT breakable with any 2D/3D descriptor combination from monomer SMILES** — the 6394 feature set is information-exhaustive for monomers
7. **Six independent approaches failed to improve over 0.900 LB**: (a) XGB-conditioned GIN, (b) residual prediction, (c) self-supervised pretraining, (d) multi-task GIN, (e) multi-scale polymer graphs, (f) 3D conformer descriptors
8. **Practical ceiling confirmed**: ~0.897 OOF / ~0.900 LB is the limit for 2D descriptors + monomer graphs from this data. To break 0.92, need genuinely new information (polymer chain context beyond monomer, 3D conformers at the polymer level, self-supervised pretraining on a vastly larger SMILES corpus, or experimental data augmentation)
9. **Best submission**: `outputs/final_submission/submission_final.csv` (0.900 LB, 0.897 OOF) — Ridge meta-stacker on GIN + XGB + Hybrid

## Key URLs
- Kaggle: https://www.kaggle.com/competitions/aisehack-2-0
- Registration: https://docs.google.com/forms/d/12EzeczXIiSJUyazcDZtvDaYsM2Q1dHbOsP3kxStrn30/viewform
- Website: https://precog.iiit.ac.in/aisehack
- GitHub: https://github.com/NotShubham1112/Poly
