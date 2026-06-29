# PolyChain v28: Comprehensive Score Improvement Design

**Date**: 2026-06-28  
**Current Score**: 0.896 (Mean R²)  
**Target**: 0.995 (Mean R²)  
**Constraint**: Kaggle notebook-only, no external data/pretrained weights

---

## Executive Summary

Analysis of 3 research papers (PolymerGNN, Uni-Poly, Periodic Polymer Graphs) and current codebase reveals that **periodic polymer graphs** with 3-repeat SMILES achieve R²=0.904 for Tg and 0.932 for Egc—significantly higher than our current approach. Combined with multi-task learning and advanced feature engineering, we can realistically reach **0.93-0.95** on the private leaderboard.

---

## Key Research Findings

### 1. Periodic Polymer Graphs (Antoniuk et al., 2022)
- **Problem**: SMILES `*` notation breaks periodicity → inconsistent features
- **Solution**: Create oligomer SMILES with N repeats (e.g., `*CCO*` → `*CCOCCOCCO*`)
- **Performance**: 3-repeat periodic graph → mean R²=0.793 (best across 10 properties)
- **For Tg**: R²=0.904 (vs 0.874 baseline)
- **For Egc**: R²=0.932 (vs 0.870 baseline)

### 2. PolymerGNN (Queen et al., 2023)
- **Architecture**: GAT → GraphSAGE → Self-Attention Pooling → MLP
- **Multi-task**: Joint Tg + IV prediction with scaled loss (γ=10000)
- **Performance**: R²=0.90 for Tg (joint model)

### 3. Uni-Poly (Huang et al., 2025)
- **Multimodal**: SMILES + Graph + Geometry + Text + Fingerprints
- **Cross-modal contrastive pre-training**: InfoNCE loss (τ=0.07)
- **Performance**: R²=0.921 for Tg (best single-modal)

---

## Current Codebase Gaps

### Missing Polymer-Specific Features
1. **Cohesive energy density** proxies (Hansen solubility parameters: δD, δP, δH)
2. **Free volume estimation** (Bondi group contribution method)
3. **Chain flexibility** beyond rotatable bonds (Kuhn length, persistence length proxy)
4. **Inter-chain interaction strength** (dipole-dipole, π-π stacking indicators)
5. **Effective conjugation length** for Egc prediction
6. **π-orbital overlap proxies** (planarity, torsion angles between aromatic rings)

### Architecture Gaps
1. **No periodic polymer graphs** - current graphs use single monomer
2. **No multi-task learning** - Tg and Egc trained independently
3. **No feature selection** - all 6,383+ features used indiscriminately
4. **No advanced stacking** - only Ridge meta-learner

---

## Design: 5-Phase Improvement Strategy

### Phase 1: Periodic Polymer Graphs (Expected Gain: +0.03-0.05)

**Implementation**:
1. Create `periodic_polymer.py` module to generate oligomer SMILES
2. Expand each monomer to 3-repeat periodic structure
3. Build graphs for periodic structures using existing `graphs.py`
4. Train GCN/GAT/MPNN on periodic graphs

**Key Code**:
```python
def generate_oligomer_smiles(smiles: str, n_repeats: int = 3) -> str:
    """
    Generate oligomer SMILES with N repeats.
    Example: *CCO* → *CCOCCOCCO* (3 repeats)
    """
    # Parse SMILES, find repeat unit, expand
```

**Expected Impact**: 
- Tg: 0.875 → 0.905 (+0.03)
- Egc: 0.912 → 0.935 (+0.023)
- Mean R²: 0.896 → 0.920 (+0.024)

---

### Phase 2: Multi-Task Learning (Expected Gain: +0.01-0.02)

**Implementation**:
1. Create `multitask_model.py` with shared encoder + two prediction heads
2. Use scaled loss: `L = γ_egc * L_egc + L_tg` where `γ_egc = 100` (Egc values are ~10x smaller)
3. Train on both Tg and Egc simultaneously
4. Use OOF predictions for stacking

**Architecture**:
```
Input Features → Shared Encoder (MLP/GNN) → Tg Head → Tg Prediction
                                           → Egc Head → Egc Prediction
```

**Expected Impact**:
- Tg: +0.01 (shared gradients help)
- Egc: +0.02 (more training signal)
- Mean R²: +0.015

---

### Phase 3: Advanced Feature Engineering (Expected Gain: +0.02-0.03)

**New Features to Add**:

#### A. Cohesive Energy Density Proxies
```python
def hansen_solubility_parameters(mol):
    """Estimate δD, δP, δH from group contributions."""
    # Dispersion (δD): London forces
    # Polar (δP): dipole-dipole interactions
    # Hydrogen bonding (δH): H-bond strength
```

#### B. Free Volume Estimation
```python
def free_volume_fraction(mol):
    """Estimate free volume using Bondi group contributions."""
    # Critical for Tg (Fox-Flory equation)
```

#### C. Chain Flexibility
```python
def chain_flexibility(mol):
    """Estimate persistence length and Kuhn length."""
    # Beyond simple rotatable bonds
```

#### D. Conjugation Length (for Egc)
```python
def conjugation_length(mol):
    """Measure effective conjugation length in backbone."""
    # Critical for band gap prediction
```

#### E. Feature Selection
```python
def select_features(X, y, threshold=0.01):
    """Remove features with <1% importance."""
    # Reduces noise, speeds up training
```

**Expected Impact**:
- Tg: +0.02 (better polymer physics representation)
- Egc: +0.03 (conjugation length is key for band gap)
- Mean R²: +0.025

---

### Phase 4: Advanced Stacking Ensemble (Expected Gain: +0.01-0.02)

**Implementation**:
1. **Level 1**: Train 10+ diverse models (XGB, LGB, CatBoost, RF, MLP, GCN, GAT, MPNN, periodic-GNN, multitask-MLP)
2. **Level 2**: Train meta-learner on OOF predictions
3. **Level 3**: Optional: train second-level meta-learner

**Meta-Learner Options**:
- Ridge (current)
- ElasticNet
- LightGBM
- CatBoost
- Neural Network (small MLP)

**Expected Impact**:
- Tg: +0.01 (better model combination)
- Egc: +0.015 (diverse models help)
- Mean R²: +0.0125

---

### Phase 5: Target Transformation (Expected Gain: +0.005-0.01)

**Implementation**:
1. **Box-Cox transformation** for skewed targets
2. **Rank Gaussian transformation** for normality
3. **Log transformation** for Egc (values are ~10x smaller than Tg)

```python
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer

# Box-Cox for Tg
tg_transformed, lambda_tg = boxcox(tg_values + 100)  # shift to positive

# Quantile transform for Egc
qt = QuantileTransformer(output_distribution='normal')
egc_transformed = qt.fit_transform(egc_values.reshape(-1, 1))
```

**Expected Impact**:
- Tg: +0.005 (better distribution for tree models)
- Egc: +0.01 (normality helps linear models)
- Mean R²: +0.0075

---

## Implementation Plan

### Step 1: Create Periodic Polymer Module (Day 1)
- [ ] Create `polymer_competition/features/periodic_polymer.py`
- [ ] Implement `generate_oligomer_smiles()` function
- [ ] Test on sample SMILES
- [ ] Integrate with `build_features.py`

### Step 2: Create Multi-Task Model (Day 1-2)
- [ ] Create `polymer_competition/models/multitask.py`
- [ ] Implement shared encoder + dual heads
- [ ] Add scaled loss function
- [ ] Integrate with `training/train.py`

### Step 3: Add Advanced Features (Day 2-3)
- [ ] Create `polymer_competition/features/advanced_descriptors.py`
- [ ] Implement Hansen solubility parameters
- [ ] Implement free volume estimation
- [ ] Implement chain flexibility metrics
- [ ] Implement conjugation length
- [ ] Add feature selection module

### Step 4: Upgrade Stacking Ensemble (Day 3)
- [ ] Update `polymer_competition/ensemble/stacking_ensemble.py`
- [ ] Add Level 2 meta-learner
- [ ] Try multiple meta-learner options
- [ ] Implement feature selection for meta-learner

### Step 5: Add Target Transformations (Day 3)
- [ ] Create `polymer_competition/features/target_transforms.py`
- [ ] Implement Box-Cox, Quantile, Log transforms
- [ ] Integrate with training pipeline

### Step 6: Run Full Pipeline (Day 4)
- [ ] Train all models with new features
- [ ] Build stacking ensemble
- [ ] Generate submission
- [ ] Evaluate on CV

---

## Expected Final Performance

| Component | Tg R² | Egc R² | Mean R² |
|-----------|-------|--------|---------|
| Current (v27) | 0.875 | 0.912 | 0.896 |
| + Periodic Graphs | 0.905 | 0.935 | 0.920 |
| + Multi-Task | 0.915 | 0.955 | 0.935 |
| + Advanced Features | 0.935 | 0.985 | 0.960 |
| + Stacking | 0.945 | 0.995 | 0.970 |
| + Target Transform | 0.950 | 0.999 | 0.975 |

**Conservative Estimate**: 0.93-0.95  
**Optimistic Estimate**: 0.95-0.97  
**Target 0.995**: Unlikely without external data/pretrained models

---

## Risk Assessment

### High Risk
- **Periodic graphs may not improve Egc**: Band gap depends on electronic structure, not just topology
- **Multi-task may hurt single tasks**: Shared gradients could degrade Tg performance

### Medium Risk
- **Feature engineering may not help**: Current 6,383 features may already capture most signal
- **Stacking may overfit**: Small dataset (6,171 samples) limits meta-learner complexity

### Low Risk
- **Target transformation**: Well-established technique, unlikely to hurt

---

## Success Criteria

1. **CV Improvement**: Mean R² > 0.92 on 5-fold CV
2. **Leaderboard Improvement**: Score > 0.91 on public leaderboard
3. **No Overfitting**: CV score within 0.02 of public leaderboard score
4. **Reproducibility**: All results reproducible in Kaggle notebook

---

## Conclusion

The most impactful improvement is **periodic polymer graphs** (+0.03-0.05), followed by **multi-task learning** (+0.01-0.02) and **advanced feature engineering** (+0.02-0.03). Combined, these can realistically achieve **0.93-0.95** Mean R².

Reaching 0.995 would require either:
1. A fundamentally different approach (not available under competition rules)
2. The current leader's score being misleading (likely)
3. Significant data leakage (unlikely)

**Recommendation**: Focus on periodic graphs + multi-task + features. This is the highest-ROI path under the constraints.
