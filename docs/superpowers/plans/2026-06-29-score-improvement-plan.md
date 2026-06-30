# Score Improvement Implementation Plan: 0.896 → 0.92

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Kaggle score from 0.896 to 0.92 by fixing the broken EGC ensemble, optimizing weights, and clipping extreme tail predictions.

**Architecture:** Phased approach — Phase 1 (fix ensemble + clip) is mandatory and must complete before Phase 2 (hybrid enhancements). Phase 3 (full v29) is conditional on Phase 2 results.

**Tech Stack:** Python, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (MLP), scipy.optimize

## Global Constraints
- Experiment version: v28 (existing features)
- CV: 5-fold, scaffold group split
- Targets: tg (glass transition temperature), egc (chain band gap)
- Evaluation metric: Mean R² = (R²_Tg + R²_Egc) / 2
- No external data or pre-trained weights allowed
- Must use only competition-provided data

---

## Phase 1: Fix Ensemble + Clip (Mandatory)

### Task 1: Retrain EGC Tree Models (v28 Features)

**Files:**
- Modify: `polymer_competition/training/train.py`
- Modify: `polymer_competition/config.yaml`
- Test: `polymer_competition/tests/test_train_target.py`

**Interfaces:**
- Consumes: v28 feature cache from `features/` directory
- Produces: OOF + test predictions for xgb, lgb, catboost, rf, mlp on EGC target

- [ ] **Step 1: Verify EGC training works with v28 features**

```bash
cd D:\Parth\Poly\polymer_competition
python -m training.train --model_type xgb --fold 0 --target egc --exp_ver v28
```

Expected: Creates `predictions/v28_egc_xgb_fold0.pkl` and `predictions/v28_egc_xgb_fold0_test.pkl`

- [ ] **Step 2: Train all missing EGC tree models (5-fold)**

```bash
python -m training.run_all_folds --models xgb,lgb,catboost,rf,mlp --target egc --exp_ver v28
```

Expected: 5 new pickle files per model (25 total) in `predictions/` directory

- [ ] **Step 3: Verify EGC predictions exist**

```bash
python -c "from pathlib import Path; p = Path('predictions'); files = list(p.glob('v28_egc_*_fold*.pkl')); print(f'Found {len(files)} EGC prediction files')"
```

Expected: At least 40 files (8 models × 5 folds × 2 [OOF + test])

- [ ] **Step 4: Commit EGC training results**

```bash
git add predictions/v28_egc_*
git commit -m "feat: retrain EGC tree models with v28 features"
```

### Task 2: Optimize Ensemble Weights Per-Target

**Files:**
- Modify: `polymer_competition/ensemble/weight_optimizer.py`
- Modify: `polymer_competition/run_submission.py`
- Test: `polymer_competition/tests/test_weight_optimizer.py`

**Interfaces:**
- Consumes: OOF predictions from `predictions/` directory
- Produces: Optimized weight vectors for TG and EGC

- [ ] **Step 1: Write test for weight optimization**

```python
# tests/test_weight_optimizer.py
def test_optimize_weights_improves_over_uniform():
    import numpy as np
    from ensemble.weight_optimizer import get_weights, r2_score
    
    # Create synthetic OOF data
    np.random.seed(42)
    n_samples = 100
    y = np.random.randn(n_samples) * 50 + 100
    oof = np.column_stack([
        y + np.random.randn(n_samples) * 5,  # Good model
        y + np.random.randn(n_samples) * 10,  # Medium model
        np.random.randn(n_samples) * 50 + 100  # Bad model
    ])
    
    # Uniform weights
    w_uniform = np.ones(3) / 3
    r2_uniform = r2_score(y, oof @ w_uniform)
    
    # Optimized weights
    w_opt = get_weights("optimize", oof, y)
    r2_opt = r2_score(y, oof @ w_opt)
    
    assert r2_opt >= r2_uniform, f"Optimized R² ({r2_opt:.4f}) < uniform R² ({r2_uniform:.4f})"
    assert np.isclose(w_opt.sum(), 1.0), "Weights must sum to 1"
    assert np.all(w_opt >= 0), "Weights must be non-negative"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_weight_optimizer.py::test_optimize_weights_improves_over_uniform -v
```

Expected: PASS (existing implementation already works)

- [ ] **Step 3: Add weight visualization for debugging**

```python
# Add to ensemble/weight_optimizer.py after line 154
def print_weight_breakdown(weights_dict, model_names):
    """Print detailed weight breakdown for debugging."""
    print("\n=== Ensemble Weight Breakdown ===")
    for model, w in sorted(weights_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 50)
        print(f"  {model:20s}: {w:.4f} {bar}")
    print(f"  {'TOTAL':20s}: {sum(weights_dict.values()):.4f}")
```

- [ ] **Step 4: Run weight optimization for both targets**

```bash
python -m ensemble.weight_optimizer --all --exp_ver v28 --strategy optimize
```

Expected: Creates `ensembles/v28_tg_weights.json` and `ensembles/v28_egc_weights.json` with optimized weights

- [ ] **Step 5: Commit weight optimization**

```bash
git add ensemble/weight_optimizer.py ensembles/v28_*
git commit -m "feat: optimize ensemble weights for v28 models"
```

### Task 3: Add Intelligent Clipping to Submission Pipeline

**Files:**
- Modify: `polymer_competition/run_submission.py`
- Test: `polymer_competition/tests/test_run_submission.py`

**Interfaces:**
- Consumes: Blended predictions from ensemble
- Produces: Clipped predictions within [train_min, train_max]

- [ ] **Step 1: Write test for clipping**

```python
# tests/test_run_submission.py
def test_clip_predictions_to_training_range():
    import numpy as np
    from run_submission import clip_predictions
    
    # Synthetic data
    train_min, train_max = 10.0, 300.0
    preds = np.array([-50.0, 5.0, 150.0, 350.0, 400.0])
    
    clipped = clip_predictions(preds, train_min, train_max)
    
    assert clipped[0] == train_min, f"Below-min not clipped: {clipped[0]}"
    assert clipped[1] == 5.0, f"Valid prediction changed: {clipped[1]}"
    assert clipped[2] == 150.0, f"Valid prediction changed: {clipped[2]}"
    assert clipped[3] == train_max, f"Above-max not clipped: {clipped[3]}"
    assert clipped[4] == train_max, f"Above-max not clipped: {clipped[4]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_run_submission.py::test_clip_predictions_to_training_range -v
```

Expected: FAIL with "name 'clip_predictions' is not defined"

- [ ] **Step 3: Implement clipping function**

```python
# Add to run_submission.py after line 30
def clip_predictions(preds: np.ndarray, train_min: float, train_max: float) -> np.ndarray:
    """Clip predictions to observed training range.
    
    This prevents extreme tail predictions from destroying R².
    R² penalizes squared errors, so a single -90 prediction when
    the true value is 100 contributes a residual of 190² = 36,100.
    """
    return np.clip(preds, train_min, train_max)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_run_submission.py::test_clip_predictions_to_training_range -v
```

Expected: PASS

- [ ] **Step 5: Wire clipping into blend_target function**

```python
# Modify run_submission.py blend_target function (around line 160)
# After: blended = matrix @ w_for_test
# Add:
    # Load training targets for clipping bounds
    train_df = pd.read_csv(REPO_ROOT / "data" / "train.csv")
    train_targets = train_df[train_df["target_col"] == target]["property"]
    train_min = float(train_targets.min())
    train_max = float(train_targets.max())
    
    # Apply intelligent clipping
    blended = clip_predictions(blended, train_min, train_max)
    print(f"  Clipped predictions to [{train_min:.2f}, {train_max:.2f}]")
```

- [ ] **Step 6: Commit clipping implementation**

```bash
git add run_submission.py tests/test_run_submission.py
git commit -m "feat: add intelligent clipping to submission pipeline"
```

### Task 4: Generate Improved Submission

**Files:**
- Modify: `polymer_competition/run_submission.py`
- Output: `outputs/submissions/submission_v29_phase1.csv`

**Interfaces:**
- Consumes: Optimized weights + clipped predictions
- Produces: New submission CSV

- [ ] **Step 1: Run full submission pipeline**

```bash
cd D:\Parth\Poly\polymer_competition
python run_submission.py --exp_ver v28 --strategy optimize
```

Expected: Creates `outputs/submissions/submission.csv` with clipped predictions

- [ ] **Step 2: Verify no extreme negatives**

```bash
python -c "import pandas as pd; df = pd.read_csv('outputs/submissions/submission.csv'); neg = (df['target'] < 0).sum(); print(f'Negative predictions: {neg}'); assert neg == 0, f'Expected 0 negatives, got {neg}'"
```

Expected: 0 negative predictions

- [ ] **Step 3: Save as phase1 backup**

```bash
cp outputs/submissions/submission.csv outputs/submissions/submission_v29_phase1.csv
```

- [ ] **Step 4: Commit phase 1 submission**

```bash
git add outputs/submissions/submission_v29_phase1.csv
git commit -m "feat: phase 1 submission - fixed ensemble + clipping"
```

- [ ] **Step 5: Upload to Kaggle and check score**

Upload `outputs/submissions/submission_v29_phase1.csv` to Kaggle.

Expected score: ~0.914 (if successful, proceed to Phase 2)

---

## Phase 2: Hybrid Enhancements (Conditional on Phase 1 Score)

**Proceed only if Phase 1 score is ≥ 0.914 and < 0.92**

### Task 5: Add Yeo-Johnson Target Transform

**Files:**
- Modify: `polymer_competition/features/target_transforms.py`
- Modify: `polymer_competition/training/train.py`
- Test: `polymer_competition/tests/test_target_transforms.py`

**Interfaces:**
- Consumes: Raw y_train values
- Produces: Transformed y_train + inverse_transform function

- [ ] **Step 1: Write test for Yeo-Johnson transform**

```python
# tests/test_target_transforms.py
def test_yeo_johnson_handles_negatives():
    import numpy as np
    from features.target_transforms import YeoJohnsonTransform
    
    # Data with negatives (like EGC target)
    y = np.array([-50.0, -10.0, 0.0, 50.0, 100.0, 200.0])
    
    transform = YeoJohnsonTransform()
    y_transformed = transform.fit_transform(y)
    y_inverse = transform.inverse_transform(y_transformed)
    
    # Should handle negatives gracefully
    assert not np.any(np.isnan(y_transformed)), "Transform produced NaN"
    assert not np.any(np.isinf(y_transformed)), "Transform produced Inf"
    assert np.allclose(y, y_inverse, atol=1e-6), "Inverse transform failed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_target_transforms.py::test_yeo_johnson_handles_negatives -v
```

Expected: FAIL (if YeoJohnsonTransform doesn't exist)

- [ ] **Step 3: Implement Yeo-Johnson transform**

```python
# Add to features/target_transforms.py
class YeoJohnsonTransform:
    """Yeo-Johnson power transform for targets with negative values.
    
    Unlike Box-Cox, Yeo-Johnson handles zero and negative values
    gracefully, making it suitable for EGC targets.
    """
    def __init__(self, lmbda='auto'):
        from sklearn.preprocessing import PowerTransformer
        self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        self.lmbda_ = None
    
    def fit_transform(self, y):
        y = np.asarray(y).reshape(-1, 1)
        result = self.transformer.fit_transform(y)
        self.lmbda_ = self.transformer.lambdas_[0]
        return result.ravel()
    
    def inverse_transform(self, y_transformed):
        y_transformed = np.asarray(y_transformed).reshape(-1, 1)
        return self.transformer.inverse_transform(y_transformed).ravel()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_target_transforms.py::test_yeo_johnson_handles_negatives -v
```

Expected: PASS

- [ ] **Step 5: Commit Yeo-Johnson transform**

```bash
git add features/target_transforms.py tests/test_target_transforms.py
git commit -m "feat: add Yeo-Johnson target transform for negatives"
```

### Task 6: Add GaussianNoise Regularizer to MLP

**Files:**
- Modify: `polymer_competition/models/mlp.py`
- Test: `polymer_competition/tests/test_mlp.py`

**Interfaces:**
- Consumes: Feature matrix X
- Produces: MLP with GaussianNoise(0.01) on input layer

- [ ] **Step 1: Write test for GaussianNoise**

```python
# tests/test_mlp.py
def test_mlp_has_gaussian_noise():
    from models.mlp import build_mlp
    
    input_dim = 100
    model = build_mlp(input_dim, hidden_dims=[64, 32], dropout=0.2, noise_std=0.01)
    
    # Check that first layer is GaussianNoise
    from torch.nn import GaussianNoise
    first_layer = model[0]
    assert isinstance(first_layer, GaussianNoise), f"First layer is {type(first_layer)}, expected GaussianNoise"
    assert first_layer.std == 0.01, f"Noise std is {first_layer.std}, expected 0.01"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_mlp.py::test_mlp_has_gaussian_noise -v
```

Expected: FAIL (if noise parameter doesn't exist)

- [ ] **Step 3: Add noise parameter to MLP**

```python
# Modify models/mlp.py build_mlp function
def build_mlp(input_dim, hidden_dims=[256, 128, 64], dropout=0.3, 
              activation='relu', noise_std=0.0):
    """Build MLP with optional GaussianNoise regularizer.
    
    Args:
        noise_std: Standard deviation of Gaussian noise on input.
                   0.0 disables noise (default).
    """
    from torch.nn import GaussianNoise
    
    layers = []
    if noise_std > 0:
        layers.append(GaussianNoise(noise_std))
    
    # ... rest of existing layer construction
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_mlp.py::test_mlp_has_gaussian_noise -v
```

Expected: PASS

- [ ] **Step 5: Commit GaussianNoise addition**

```bash
git add models/mlp.py tests/test_mlp.py
git commit -m "feat: add GaussianNoise regularizer to MLP"
```

### Task 7: Train 10-Seed MLP Ensemble

**Files:**
- Modify: `polymer_competition/training/train.py`
- Output: 10 new MLP OOF + test prediction files

**Interfaces:**
- Consumes: v28 features + Yeo-Johnson transformed targets
- Produces: 10 sets of OOF predictions per target

- [ ] **Step 1: Train 10 MLP seeds for TG**

```bash
cd D:\Parth\Poly\polymer_competition
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python -m training.train --model_type mlp --fold $((seed % 5)) --target tg --exp_ver v28 --seed $seed --noise_std 0.01
done
```

Expected: 10 new prediction files per fold (50 total for TG)

- [ ] **Step 2: Train 10 MLP seeds for EGC**

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python -m training.train --model_type mlp --fold $((seed % 5)) --target egc --exp_ver v28 --seed $seed --noise_std 0.01
done
```

Expected: 10 new prediction files per fold (50 total for EGC)

- [ ] **Step 3: Verify MLP seed predictions exist**

```bash
python -c "from pathlib import Path; p = Path('predictions'); files = list(p.glob('v28_*_mlp_*_fold*.pkl')); print(f'Found {len(files)} MLP prediction files')"
```

Expected: At least 80 files (10 seeds × 2 targets × 5 folds × 2 [OOF + test])

- [ ] **Step 4: Commit MLP seed training**

```bash
git add predictions/v28_*_mlp_*
git commit -m "feat: train 10-seed MLP ensemble with GaussianNoise"
```

### Task 8: Re-optimize Weights with New MLP Seeds

**Files:**
- Modify: `polymer_competition/ensemble/weight_optimizer.py`
- Output: Updated ensemble weights

**Interfaces:**
- Consumes: OOF predictions from all models including new MLP seeds
- Produces: Optimized weights incorporating MLP diversity

- [ ] **Step 1: Run weight optimization with all models**

```bash
cd D:\Parth\Poly\polymer_competition
python -m ensemble.weight_optimizer --all --exp_ver v28 --strategy optimize
```

Expected: Updated `ensembles/v28_tg_weights.json` and `ensembles/v28_egc_weights.json`

- [ ] **Step 2: Verify MLP seeds have non-zero weight**

```bash
python -c "import json; w = json.load(open('ensembles/v28_tg_weights.json')); mlp_weights = {k:v for k,v in w['weights'].items() if 'mlp' in k}; print(f'MLP weights: {mlp_weights}'); assert sum(mlp_weights.values()) > 0.1, 'MLP should have >10% total weight'"
```

Expected: MLP seeds collectively have >10% weight

- [ ] **Step 3: Commit updated weights**

```bash
git add ensembles/v28_*
git commit -m "feat: re-optimize ensemble weights with 10-seed MLP"
```

### Task 9: Generate Phase 2 Submission

**Files:**
- Modify: `polymer_competition/run_submission.py`
- Output: `outputs/submissions/submission_v29_phase2.csv`

**Interfaces:**
- Consumes: Updated weights + Yeo-Johnson transforms + clipping
- Produces: Phase 2 submission CSV

- [ ] **Step 1: Run submission with all enhancements**

```bash
cd D:\Parth\Poly\polymer_competition
python run_submission.py --exp_ver v28 --strategy optimize
```

Expected: Creates submission with transformed targets + optimized weights + clipping

- [ ] **Step 2: Save as phase2 backup**

```bash
cp outputs/submissions/submission.csv outputs/submissions/submission_v29_phase2.csv
```

- [ ] **Step 3: Verify improvements over phase 1**

```bash
python -c "
import pandas as pd
p1 = pd.read_csv('outputs/submissions/submission_v29_phase1.csv')
p2 = pd.read_csv('outputs/submissions/submission_v29_phase2.csv')
print(f'Phase 1 range: [{p1.target.min():.2f}, {p1.target.max():.2f}]')
print(f'Phase 2 range: [{p2.target.min():.2f}, {p2.target.max():.2f}]')
print(f'Phase 1 negatives: {(p1.target < 0).sum()}')
print(f'Phase 2 negatives: {(p2.target < 0).sum()}')
"
```

Expected: Phase 2 has fewer negatives and tighter range

- [ ] **Step 4: Commit phase 2 submission**

```bash
git add outputs/submissions/submission_v29_phase2.csv
git commit -m "feat: phase 2 submission - hybrid enhancements"
```

- [ ] **Step 5: Upload to Kaggle and check score**

Upload `outputs/submissions/submission_v29_phase2.csv` to Kaggle.

Expected score: ~0.921 (if successful, done!)

---

## Phase 3: Full v29 (Only if Phase 2 Score < 0.92)

**Proceed only if Phase 2 score is < 0.92**

### Task 10: Rebuild Features with v29 Enhancements

**Files:**
- Modify: `polymer_competition/features/build_features.py`
- Modify: `polymer_competition/features/advanced_descriptors.py`
- Output: New feature cache with GNN embeddings

**Interfaces:**
- Consumes: Raw SMILES + molecular graphs
- Produces: Feature matrix with 16 topological invariants + GNN embeddings

- [ ] **Step 1: Run v29 feature build**

```bash
cd D:\Parth\Poly\polymer_competition
python -m features.build_features --exp_ver v29 --include_gnn_embeddings --include_advanced_descriptors
```

Expected: New feature cache with 50+ additional features

- [ ] **Step 2: Verify GNN embeddings exist**

```bash
python -c "from pathlib import Path; p = Path('features/cache'); files = list(p.glob('v29_*_gnn_*.pkl')); print(f'Found {len(files)} GNN embedding files')"
```

Expected: At least 2 files (TG + EGC)

- [ ] **Step 3: Commit v29 features**

```bash
git add features/cache/v29_*
git commit -m "feat: rebuild features with v29 enhancements"
```

### Task 11: PCA on GNN Embeddings

**Files:**
- Modify: `polymer_competition/features/build_features.py`
- Test: `polymer_competition/tests/test_features.py`

**Interfaces:**
- Consumes: Raw GNN embeddings (high-dimensional)
- Produces: PCA-reduced embeddings (95% variance retained)

- [ ] **Step 1: Write test for PCA reduction**

```python
# tests/test_features.py
def test_pca_reduces_gnn_dimensions():
    import numpy as np
    from features.build_features import apply_pca_to_embeddings
    
    # Synthetic GNN embeddings
    n_samples = 100
    n_features = 256
    embeddings = np.random.randn(n_samples, n_features)
    
    reduced = apply_pca_to_embeddings(embeddings, variance_retained=0.95)
    
    assert reduced.shape[0] == n_samples, "Sample count changed"
    assert reduced.shape[1] < n_features, f"Dimensions not reduced: {reduced.shape[1]}"
    assert reduced.shape[1] >= 10, f"Too few dimensions: {reduced.shape[1]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_features.py::test_pca_reduces_gnn_dimensions -v
```

Expected: FAIL (if function doesn't exist)

- [ ] **Step 3: Implement PCA reduction**

```python
# Add to features/build_features.py
def apply_pca_to_embeddings(embeddings, variance_retained=0.95):
    """Apply PCA to GNN embeddings to reduce dimensionality.
    
    Retains specified variance while removing noisy dimensions.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=variance_retained, random_state=42)
    reduced = pca.fit_transform(embeddings)
    
    print(f"  PCA: {embeddings.shape[1]} → {reduced.shape[1]} dimensions "
          f"({variance_retained*100:.0f}% variance retained)")
    
    return reduced
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_features.py::test_pca_reduces_gnn_dimensions -v
```

Expected: PASS

- [ ] **Step 5: Commit PCA implementation**

```bash
git add features/build_features.py tests/test_features.py
git commit -m "feat: add PCA reduction for GNN embeddings"
```

### Task 12: Add Graph Transformer Model

**Files:**
- Modify: `polymer_competition/models/graph_transformer.py`
- Test: `polymer_competition/tests/test_gnn.py`

**Interfaces:**
- Consumes: Molecular graph data
- Produces: Graph Transformer predictions

- [ ] **Step 1: Write test for graph transformer**

```python
# tests/test_gnn.py
def test_graph_transformer_forward_pass():
    import torch
    from models.graph_transformer import GraphTransformer
    
    # Create synthetic graph data
    batch_size = 4
    n_nodes = 10
    n_features = 64
    
    x = torch.randn(batch_size, n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, 20))
    
    model = GraphTransformer(input_dim=n_features, hidden_dim=128, n_heads=4)
    output = model(x, edge_index)
    
    assert output.shape == (batch_size, 1), f"Output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_gnn.py::test_graph_transformer_forward_pass -v
```

Expected: FAIL (if GraphTransformer doesn't exist)

- [ ] **Step 3: Implement Graph Transformer**

```python
# Add to models/graph_transformer.py
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

class GraphTransformer(nn.Module):
    """Graph Transformer for molecular property prediction.
    
    Uses multi-head attention to capture long-range dependencies
    in molecular graphs.
    """
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads)
            for _ in range(n_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)
        
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        if batch is None:
            x = x.mean(dim=0)
        else:
            x = global_mean_pool(x, batch)
        
        return self.classifier(x)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_gnn.py::test_graph_transformer_forward_pass -v
```

Expected: PASS

- [ ] **Step 5: Commit Graph Transformer**

```bash
git add models/graph_transformer.py tests/test_gnn.py
git commit -m "feat: add Graph Transformer model"
```

### Task 13: Full v29 Retrain

**Files:**
- Modify: `polymer_competition/training/train.py`
- Output: All model predictions with v29 features

**Interfaces:**
- Consumes: v29 features (advanced descriptors + GNN embeddings + PCA)
- Produces: Complete OOF + test predictions for all models

- [ ] **Step 1: Retrain all models with v29 features**

```bash
cd D:\Parth\Poly\polymer_competition
python -m training.run_all_folds --models xgb,lgb,catboost,rf,mlp,gcn,gat,mpnn,graph_transformer --exp_ver v29
```

Expected: All model predictions updated with v29 features

- [ ] **Step 2: Verify v29 predictions exist**

```bash
python -c "from pathlib import Path; p = Path('predictions'); files = list(p.glob('v29_*_fold*.pkl')); print(f'Found {len(files)} v29 prediction files')"
```

Expected: At least 180 files (10 models × 2 targets × 5 folds × 2 [OOF + test])

- [ ] **Step 3: Commit v29 training**

```bash
git add predictions/v29_*
git commit -m "feat: full v29 retrain with advanced features"
```

### Task 14: Level-2 Stacking Ensemble

**Files:**
- Modify: `polymer_competition/ensemble/stacking_ensemble.py`
- Test: `polymer_competition/tests/test_stacking.py`

**Interfaces:**
- Consumes: OOF predictions from all v29 models
- Produces: Stacked ensemble predictions

- [ ] **Step 1: Write test for stacking**

```python
# tests/test_stacking.py
def test_stacking_improves_over_single_model():
    import numpy as np
    from ensemble.stacking_ensemble import StackingEnsemble
    
    # Synthetic data
    np.random.seed(42)
    n_samples = 200
    y = np.random.randn(n_samples) * 50 + 100
    
    # 3 base model predictions
    oof_matrix = np.column_stack([
        y + np.random.randn(n_samples) * 5,
        y + np.random.randn(n_samples) * 8,
        y + np.random.randn(n_samples) * 12
    ])
    
    stacker = StackingEnsemble(meta_learner='ridge')
    stacker.fit(oof_matrix, y)
    stacked_pred = stacker.predict(oof_matrix)
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2_single = r2_score(y, oof_matrix[:, 0])
    r2_stacked = r2_score(y, stacked_pred)
    
    assert r2_stacked >= r2_single, f"Stacking R² ({r2_stacked:.4f}) < single model R² ({r2_single:.4f})"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_stacking.py::test_stacking_improves_over_single_model -v
```

Expected: FAIL (if StackingEnsemble doesn't exist)

- [ ] **Step 3: Implement stacking ensemble**

```python
# Add to ensemble/stacking_ensemble.py
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

class StackingEnsemble:
    """Level-2 stacking ensemble with Ridge meta-learner."""
    
    def __init__(self, meta_learner='ridge', cv=5):
        self.meta_learner = Ridge(alpha=1.0) if meta_learner == 'ridge' else meta_learner
        self.cv = cv
        self.is_fitted = False
    
    def fit(self, oof_matrix, y):
        """Fit meta-learner on OOF predictions."""
        # Use cross-val predict to avoid overfitting
        self.meta_oof = cross_val_predict(
            self.meta_learner, oof_matrix, y, cv=self.cv
        )
        # Refit on full data
        self.meta_learner.fit(oof_matrix, y)
        self.is_fitted = True
        return self
    
    def predict(self, oof_matrix):
        """Predict using fitted meta-learner."""
        assert self.is_fitted, "Must call fit() first"
        return self.meta_learner.predict(oof_matrix)
    
    def get_weights(self):
        """Get normalized meta-learner weights."""
        if not self.is_fitted:
            return None
        weights = np.clip(self.meta_learner.coef_, 0, None)
        return weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd D:\Parth\Poly\polymer_competition
python -m pytest tests/test_stacking.py::test_stacking_improves_over_single_model -v
```

Expected: PASS

- [ ] **Step 5: Commit stacking ensemble**

```bash
git add ensemble/stacking_ensemble.py tests/test_stacking.py
git commit -m "feat: add level-2 stacking ensemble"
```

### Task 15: Generate Final v29 Submission

**Files:**
- Modify: `polymer_competition/run_submission.py`
- Output: `outputs/submissions/submission_v29_final.csv`

**Interfaces:**
- Consumes: Stacked ensemble + all enhancements
- Produces: Final submission CSV

- [ ] **Step 1: Run final submission pipeline**

```bash
cd D:\Parth\Poly\polymer_competition
python run_submission.py --exp_ver v29 --strategy stacking
```

Expected: Creates final submission with all v29 enhancements

- [ ] **Step 2: Save as final backup**

```bash
cp outputs/submissions/submission.csv outputs/submissions/submission_v29_final.csv
```

- [ ] **Step 3: Verify final improvements**

```bash
python -c "
import pandas as pd
p1 = pd.read_csv('outputs/submissions/submission_v29_phase1.csv')
pf = pd.read_csv('outputs/submissions/submission_v29_final.csv')
print(f'Phase 1 range: [{p1.target.min():.2f}, {p1.target.max():.2f}]')
print(f'Final range: [{pf.target.min():.2f}, {pf.target.max():.2f}]')
print(f'Phase 1 negatives: {(p1.target < 0).sum()}')
print(f'Final negatives: {(pf.target < 0).sum()}')
"
```

Expected: Final has tightest range and fewest negatives

- [ ] **Step 4: Commit final submission**

```bash
git add outputs/submissions/submission_v29_final.csv
git commit -m "feat: final v29 submission - all enhancements"
```

- [ ] **Step 5: Upload to Kaggle and check final score**

Upload `outputs/submissions/submission_v29_final.csv` to Kaggle.

Expected score: ~0.925 (if successful, mission accomplished!)

---

## Success Criteria

| Phase | Target Score | Status |
|-------|--------------|--------|
| Phase 1 | ≥ 0.914 | ⬜ Pending |
| Phase 2 | ≥ 0.920 | ⬜ Pending |
| Phase 3 | ≥ 0.925 | ⬜ Pending |

## Risk Mitigation

1. **If Phase 1 score < 0.914**: Skip to Phase 3 (full v29)
2. **If Yeo-Johnson hurts EGC**: Fallback to RankGauss
3. **If GNN embeddings are noisy**: Increase PCA variance retained to 0.99
4. **If stacking overfits**: Use uniform weights instead

## Time Estimates

| Phase | Estimated Time | Cumulative |
|-------|----------------|------------|
| Phase 1 | 1 hour | 1 hour |
| Phase 2 | 3 hours | 4 hours |
| Phase 3 | 4 hours | 8 hours |
