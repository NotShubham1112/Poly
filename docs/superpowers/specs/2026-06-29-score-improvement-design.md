# Score Improvement Design: 0.896 → 0.92

## Problem
Current submission scores 0.896 on Kaggle LB. Target is 0.92. Gap: 0.024.

## Root Cause Analysis
1. **EGC ensemble broken**: Only multitask model has v28 EGC predictions (R² = -0.7254). Tree models (xgb, lgb, catboost, rf, mlp) missing v28 EGC.
2. **TG uses uniform weights**: v27 optimized weights (lgb 37%, mlp 33%, xgb 22%) were much better.
3. **201 negative predictions (4.9%)**: Extreme tail predictions (-91 to 0) destroy R² due to squared error penalty.
4. **GNN models consistently zero weight**: Adding noise to ensemble.

## Design: Phased Approach

### Phase 1: Fix Ensemble + Clip (Mandatory, ~1 hr)
**Goal**: Retrain missing EGC tree models, optimize weights, clip tails.

**Steps**:
1. Retrain xgb, lgb, catboost, rf, mlp for EGC target (v28 features, 5-fold)
2. Generate OOF predictions for all base models (both targets)
3. Optimize ensemble weights per-target using `scipy.optimize.minimize` with `-R2` objective
4. Apply intelligent clipping: `np.clip(preds, y_train.min(), y_train.max())`
5. Generate submission

**Files to modify**:
- `training/train.py` — ensure EGC training works
- `ensemble/weight_optimizer.py` — already has optimize functionality
- `run_submission.py` — add clipping step

**Expected outcome**: 0.896 → ~0.914

### Phase 2: Hybrid Enhancements (If Phase 1 < 0.92)
**Goal**: Target transforms + multi-seed MLP + better regularization.

**Steps**:
1. Apply Yeo-Johnson transform on y_train (handles negatives gracefully)
2. Train 10 MLP seeds with varied architectures ([1024,512] and [512,256,128])
3. Add GaussianNoise(0.01) regularizer to MLP input
4. Re-optimize weights with new MLP seeds
5. Re-clip and submit

**Files to modify**:
- `models/mlp.py` — add GaussianNoise, multi-seed support
- `features/target_transforms.py` — Yeo-Johnson implementation (exists)
- `training/train.py` — wire up transforms + multi-seed

**Expected outcome**: ~0.914 → ~0.921

### Phase 3: Full v29 (Only if Phase 2 < 0.92)
**Goal**: Advanced features + GNN embeddings + graph_transformer.

**Steps**:
1. Rebuild features with v29 enhancements (advanced descriptors, GNN embeddings)
2. PCA on GNN embeddings before concatenation
3. Add graph_transformer model
4. Full retrain all models 5-fold
5. Level-2 stacking ensemble

**Expected outcome**: ~0.921 → ~0.925+

## Evaluation
- **Metric**: Mean R² = (R²_Tg + R²_Egc) / 2
- **CV check**: Before each submission, verify CV R² ≥ target
- **Clipping validation**: Verify no predictions outside [train_min, train_max]

## Risks
1. Retraining may overfit to local features → mitigate with 5-fold CV
2. Yeo-Johnson may not help EGC → fallback to RankGauss
3. GNN embeddings may be noisy → PCA with 95% variance retention
