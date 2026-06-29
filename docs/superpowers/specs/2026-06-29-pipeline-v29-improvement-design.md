# Pipeline v29 — Push Mean R² to 0.905+

## Problem
Current Mean R² ~0.894 (TG: 0.876, EGC: 0.912). TG is the bottleneck. Need +0.01–0.02 without external data or rule violations.

## Changes

### 1. New Target Transforms
- Add `yeo_johnson_transform()` and `rank_gauss_transform()` to `features/target_transforms.py`
- Update `select_best_transform()` to include Yeo-Johnson in the candidate pool
- All models benefit, especially MLP on TG's heavy-tailed distribution

### 2. Topological Graph Invariants
- Add to `features/advanced_descriptors.py`: Balaban J, Wiener index, Chi0/Chi1/Chi cluster/path-cluster, Kappa1/Kappa2/Kappa3, Hall-Kier alpha
- All computed via `rdkit.Chem.Descriptors` — zero external data
- Captures chain stiffness / branching topology that current features miss

### 3. GNN Embeddings as MLP Features
- Add `get_embedding(data)` method to GCN, GAT, DMPNN in `models/gnn.py`
- Returns pooled graph embedding `g` before final regression head
- During GNN retraining, extract OOF embeddings per fold → save as extra feature columns
- MLP ensemble trains on fingerprints + descriptors + GNN embeddings

### 4. Multi-Seed MLP Ensemble
- Modify MLP training in `training/train.py` to accept `--n_seeds` parameter
- Train 3–5 seeds with different random seeds and hidden dims
- Average OOF and test predictions across seeds
- Reduces variance on noisy TG

### 5. Full Multi-Task Training
- Run 80-epoch uncertainty-weighted multi-task (already implemented)
- Must run AFTER feature cache rebuild (point 6)

### 6. Execution Order
1. Implement RankGauss/Yeo-Johnson, topological invariants, GNN embedding extraction
2. Rebuild feature cache
3. Retrain all models (tree, GNN, MLP ensemble) 5-fold
4. Run 80-epoch multi-task
5. Stack with per-target meta-learners
6. Submit

## Expected Outcome
- TG OOF: 0.876 → ~0.890 (+0.014)
- EGC OOF: hold at ~0.912
- Mean R²: ~0.894 → ~0.905–0.910
- Public score: 0.90+ (clean, reproducible, rule-compliant)
