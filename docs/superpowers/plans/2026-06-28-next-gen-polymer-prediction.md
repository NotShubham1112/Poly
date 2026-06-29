# Next-Generation Polymer Property Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the polymer property prediction pipeline from first principles, incorporating periodic graph representations, multi-task learning, learned descriptors, and physics-informed constraints to push Mean R² from 0.896 toward 0.93+.

**Architecture:** A hybrid system combining: (1) periodic polymer graphs with learned MPNN descriptors, (2) enhanced handcrafted polymer features, (3) multi-task learning with uncertainty-aware loss, (4) Level-2 stacking with OOF predictions, and (5) target transformation with inverse normalization.

**Tech Stack:** PyTorch, PyTorch Geometric, RDKit, XGBoost, LightGBM, CatBoost, scikit-learn, Optuna

---

## Research Synthesis: 20 Questions

### 1. What assumptions does my current pipeline make?

1. **Monomer-level representation is sufficient.** The pipeline treats each polymer as a single repeat unit graph. But polymers are periodic chains — periodic boundary conditions affect electronic structure, chain packing, and thus Tg and Egc. Paper [Antoniuk 2022] demonstrates a 20% error reduction by using periodic graphs vs. monomer-only graphs.

2. **Handcrafted features are optimal.** The pipeline uses 6,393 handcrafted features. But learned descriptors from MPNNs outperform handcrafted ones consistently across 10 polymer properties [Antoniuk 2022].

3. **Independent target prediction.** Tg and Egc are trained completely independently. But multi-task learning can exploit shared representations — both properties depend on chain rigidity, intermolecular forces, and electronic structure [Queen 2023, Gurnani 2022].

4. **Features are precomputed and static.** The pipeline computes all features once and freezes them. But learned features could adapt to the prediction task through end-to-end training.

5. **All 6,393 features are relevant.** No feature selection is performed. With ~4,143 Tg samples and ~2,028 Egc samples, many features are likely redundant or noisy, risking overfitting.

### 2. What information is my pipeline completely ignoring?

| Missing Information | Why It Matters | Source |
|---------------------|----------------|--------|
| **Chain periodicity** | Electronic structure depends on repeat pattern | Antoniuk 2022 |
| **Chain conformation** | Tg depends on chain flexibility and packing | Polymer physics |
| **Intermolecular interactions** | Egc depends on orbital overlap between chains | Quantum chemistry |
| **Molecular weight distribution** | Tg has MW dependence (Fox-Flory equation) | PolymerGNN |
| **Topology (linear vs branched)** | Branching affects chain mobility -> Tg | Custom features (partial) |

### 3. What are the biggest scientific weaknesses?

1. **No periodic boundary encoding.** `periodic_graph()` in `graphs.py:268` creates a trivially small graph (single repeat unit closed as ring). Does NOT capture extended chain structure.

2. **Zero-padding in multi-task learning.** Dead `train_multitask()` (train.py:196) pads the smaller dataset with zeros, diluting the training signal. Requires proper masking.

3. **Feature preprocessing not applied.** `FeaturePreprocessor` (preprocessing.py) exists but is never called.

4. **Target transforms not applied.** `apply_target_transform()` (train.py:242) exists but is never called.

5. **No interaction features.** Config has `interaction_features: false`.

### 4-7. Cross-Paper Synthesis

| Idea | Papers | Status in Pipeline |
|------|--------|--------------------|
| Periodic graph representation | Antoniuk 2022, Gurnani 2022 | Broken (k=1, trivial ring) |
| Multi-task learning | Queen 2023, Gurnani 2022, Uni-Poly 2025 | Dead code |
| Learned > handcrafted features | Antoniuk 2022, Gurnani 2022 | Not implemented |
| Shared encoder architecture | All papers | Dead code |
| Attention-based fusion | Uni-Poly 2025, Gurnani 2022 | Partial (HAMF in PolyChain) |

### 8-20. Architecture Design Decisions

| Question | Answer |
|----------|--------|
| Redesign graph representation? | Yes — periodic graphs k=3 + boundary edges |
| Better polymer chemistry? | Yes — chain statistics, MW estimation, branching |
| Redesign feature extraction? | Yes — parallel: learned (MPNN) + handcrafted |
| Redesign message passing? | Yes — edge-aware GIN with polymer atom features |
| Redesign prediction objective? | Yes — multi-task loss with uncertainty weighting |
| Multi-task learning? | Yes — shared encoder, separate heads, proper masking |
| Uncertainty estimation? | Yes — learned log-variance for ensemble weighting |
| Physical constraints? | Partial — penalty for implausible predictions |
| Graph attention for polymers? | Yes — HAMF adapted for periodic graphs |
| Hierarchical representations? | Yes — monomer -> dimer -> trimer scales |
| New architecture from scratch? | No — extend existing codebase |

---

## Architecture: PolyChain v2

```
                         INPUT: SMILES
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
     PERIODIC GRAPH    HANDCRAFTED FEAT    CHAIN STATISTICS
     CONSTRUCTION      (6393 features)     (CST, 33 features)
              |                |                |
              v                v                v
        MPNN ENCODER     FEATURE PREPROC    CST PROJECTION
        (4x EdgeGIN)     (MI selection)     (Linear->ReLU)
        256 hidden       128 output         128 output
              |                |                |
              v                v                v
         MULTI-SCALE FUSION (HAMF)
         monomer->dimer->trimer
         Causal attention, 2 blocks
              |
              v
         FEATURE FUSION
         Cross-attention + residual
              |
       +------+------+
       |             |
       v             v
    Tg HEAD      Egc HEAD
    (128->1)     (128->1)

LOSS = L_Tg + gamma * L_Egc + lambda * PhysicsPenalty
     where gamma = sigma^2_Tg / sigma^2_Egc (learned)
```

### Mathematical Formulation

**Periodic Graph:** Given repeat unit s, construct oligomer s^3. G = (V, E) where E includes periodic boundary edge connecting chain ends.

**MPNN:** m_v^(t+1) = sum_{w in N(v)} M(h_v^(t), h_w^(t), e_vw); h_v^(t+1) = U(h_v^(t), m_v^(t+1))

**Multi-Task Uncertainty Loss:** L = (1/2sigma_Tg^2) * MSE_Tg + (1/2sigma_Egc^2) * MSE_Egc + log(sigma_Tg) + log(sigma_Egc)

---

## Phase 1: Feature Engineering Enhancement (Expected: +0.02-0.03)

### Task 1.1: Wire FeaturePreprocessor into Pipeline

**Files:**
- Modify: `polymer_competition/features/preprocessing.py:25-82`
- Modify: `polymer_competition/features/build_features.py:114-276`
- Create: `polymer_competition/tests/test_preprocessing.py`

**Interfaces:**
- Consumes: Raw feature DataFrame
- Produces: Cleaned feature DataFrame with imputation, variance filtering, correlation removal

- [ ] **Step 1: Fix FeaturePreprocessor._clean()**

```python
# preprocessing.py line 80-82
def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    return X
```

- [ ] **Step 2: Add MI-based feature selection to FeaturePreprocessor.fit()**

Add after correlation filtering (around line 48):

```python
from sklearn.feature_selection import mutual_info_regression

def fit(self, X: pd.DataFrame, y: np.ndarray = None):
    # ... existing code ...
    if y is not None and len(self.keep_cols) > 500:
        mi_scores = mutual_info_regression(X[self.keep_cols].fillna(0), y, random_state=42)
        top_idx = np.argsort(mi_scores)[-500:]
        self.keep_cols = [self.keep_cols[i] for i in top_idx]
    return self
```

- [ ] **Step 3: Wire into build_features.py after line 205**

- [ ] **Step 4: Write and run tests**

- [ ] **Step 5: Commit**

---

### Task 1.2: Add Interaction Features

**Files:**
- Create: `polymer_competition/features/interactions.py`
- Modify: `polymer_competition/features/build_features.py`

- [ ] **Step 1: Create interactions.py with fingerprint-descriptor interactions and descriptor ratios**

- [ ] **Step 2: Wire into build_features.py after advanced features**

- [ ] **Step 3: Write and run tests**

- [ ] **Step 4: Commit**

---

### Task 1.3: Enable Target Transformation

**Files:**
- Modify: `polymer_competition/training/train.py:242-260` and `train.py:622-814`

- [ ] **Step 1: Fix apply_target_transform config key check**

- [ ] **Step 2: Insert transform call in main() after y_tr loading**

- [ ] **Step 3: Apply inverse transform to predictions before metrics**

- [ ] **Step 4: Write and run tests**

- [ ] **Step 5: Commit**

---

## Phase 2: Periodic Graph Enhancement (Expected: +0.03-0.05)

### Task 2.1: Improve Periodic Graph Construction

**Files:**
- Modify: `polymer_competition/features/graphs.py:268-296`

- [ ] **Step 1: Fix periodic_graph() to use k=3 instead of k=1**

- [ ] **Step 2: Add multi_scale_periodic_graphs() helper**

- [ ] **Step 3: Write and run tests**

- [ ] **Step 4: Commit**

---

### Task 2.2: Enhance Periodic Graph Feature Extraction

**Files:**
- Modify: `polymer_competition/features/build_features.py:67-111`

- [ ] **Step 1: Rewrite build_periodic_graph_features with 14 chain-level descriptors**

- [ ] **Step 2: Write and run tests**

- [ ] **Step 3: Commit**

---

## Phase 3: Multi-Task Learning (Expected: +0.01-0.02)

### Task 3.1: Rewrite Multi-Task Model

**Files:**
- Rewrite: `polymer_competition/models/multitask.py`

- [ ] **Step 1: Implement MultiTaskModel with uncertainty weighting and masking**

- [ ] **Step 2: Write and run tests**

- [ ] **Step 3: Commit**

---

### Task 3.2: Integrate Multi-Task Training

**Files:**
- Rewrite: `polymer_competition/training/train.py:196-239`
- Modify: `polymer_competition/training/train.py:622`

- [ ] **Step 1: Rewrite train_multitask with DataLoader and masking**

- [ ] **Step 2: Add --multitask flag to main()**

- [ ] **Step 3: Write and run tests**

- [ ] **Step 4: Commit**

---

## Phase 4: Improved Ensemble (Expected: +0.01-0.02)

### Task 4.1: Uncertainty-Weighted Ensemble

**Files:**
- Modify: `polymer_competition/ensemble/weight_optimizer.py`

- [ ] **Step 1: Add uncertainty-based weighting using model disagreement**

- [ ] **Step 2: Improve stacking with diverse meta-learners**

- [ ] **Step 3: Write and run tests**

- [ ] **Step 4: Commit**

---

## Phase 5: Train and Evaluate (Expected: Full pipeline run)

### Task 5.1: Run Full Pipeline

- [ ] **Step 1: Rebuild features with preprocessing and interactions**

- [ ] **Step 2: Train all models with target transforms**

- [ ] **Step 3: Train multi-task model**

- [ ] **Step 4: Build improved ensemble**

- [ ] **Step 5: Generate submission and evaluate**

---

## Prioritized Implementation Order

| Priority | Task | Expected Impact | Risk | Time |
|----------|------|-----------------|------|------|
| 1 | Task 1.1: Wire FeaturePreprocessor | +0.01-0.02 | Low | 30 min |
| 2 | Task 1.3: Enable Target Transform | +0.005-0.01 | Low | 20 min |
| 3 | Task 2.1: Fix Periodic Graphs | +0.03-0.05 | Medium | 45 min |
| 4 | Task 3.1: Rewrite Multi-Task Model | +0.01-0.02 | Medium | 40 min |
| 5 | Task 1.2: Interaction Features | +0.005-0.01 | Low | 30 min |
| 6 | Task 3.2: Integrate Multi-Task | +0.01-0.02 | Medium | 45 min |
| 7 | Task 2.2: Periodic Feature Extraction | +0.01-0.02 | Low | 30 min |
| 8 | Task 4.1: Uncertainty Ensemble | +0.005-0.01 | Low | 30 min |
| 9 | Task 5.1: Full Pipeline Run | Validation | Low | 2-3 hrs |

**Total estimated time: 6-7 hours implementation + 2-3 hours training**
**Expected score improvement: 0.896 -> 0.92-0.95**
