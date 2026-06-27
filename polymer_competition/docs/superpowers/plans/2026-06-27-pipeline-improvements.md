# Pipeline Improvements Implementation Plan

**Goal:** Improve R² by adding GNN models, hyperparameter tuning, and ensemble optimization.

**Architecture:** Three independent tasks that modify different files:
- A: Notebook cell changes for GNN models + PyG install
- B: New HPO tuning script
- C: Weight optimizer improvement

**Tech Stack:** pytorch_geometric, optuna, scipy

## Global Constraints
- All changes must run on Kaggle P100 (sm_60) with PyTorch 2.5.1
- No external data, pre-trained weights only from training
- Must produce a valid submission.csv at the end

---

### Task A: Add GNN models (gcn, gat, polychain)

**Files:**
- Modify: `notebooks/kaggle_pipeline.ipynb` Cells 1 and 7

**Changes:**
- Cell 1: Add PyG install after PyTorch downgrade
- Cell 7: Add gcn, gat, polychain to MODELS list

### Task B: Hyperparameter tuning script

**Files:**
- Create: `training/tune.py`
- Modify: `training/configs/xgb.yaml`, `training/configs/lgb.yaml`, `training/configs/catboost.yaml`

**Changes:**
- Script loads features, runs Optuna for 20 trials per model type
- Saves best params back to config yaml

### Task C: Ensemble weight optimization

**Files:**
- Modify: `ensemble/weight_optimizer.py`

**Changes:**
- Add scipy.optimize.minimize with R² objective
- Support per-target weight optimization
- Support per-model per-fold weight optimization
