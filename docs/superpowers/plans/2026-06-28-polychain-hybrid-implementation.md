# PolyChain Hybrid Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid ensemble (PolyChain GNN + tree models + handcrafted features) that maximizes Mean R² on the AISEHack 2.0 Polymer Property Prediction competition.

**Architecture:** Separate Tg/Egc pipelines. Each pipeline has: (1) feature engineering (RDKit fingerprints + descriptors + polymer-specific), (2) tree model training with Optuna tuning, (3) PolyChain GNN training, (4) leakage-free stacking, (5) weight-optimized ensemble. All models trained from scratch on competition data only.

**Tech Stack:** Python 3.10+, PyTorch 2.x + PyG, RDKit, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, matplotlib, seaborn

## Global Constraints

- **Data:** Only competition-provided train.csv and test.csv
- **No pretrained weights:** All models trained from scratch
- **Reproducibility:** All experiments track config, seed, git commit, runtime
- **5-fold CV:** GroupKFold by SMILES scaffold (no leakage)
- **Per-target:** Tg and Egc are separate regression problems
- **Kaggle submission:** Final notebook must run within 30 hrs/week GPU quota

---

## Phase 1: Feature Engineering

### Task 1: Verify existing feature pipeline

**Files:**
- Read: `features/build_features.py`
- Read: `features/descriptors.py`
- Read: `features/fingerprints.py`
- Read: `features/graphs.py`
- Read: `features/graph_utils.py`

- [ ] **Step 1: Check current build_features output**

```bash
cd D:\Parth\Poly\polymer_competition
python -m features.build_features
```

Expected: Generates `data/processed/features_train.parquet` and `data/processed/features_test.parquet`

- [ ] **Step 2: Verify feature dimensions and NaN count**

```python
import pandas as pd
train = pd.read_parquet("data/processed/features_train.parquet")
test = pd.read_parquet("data/processed/features_test.parquet")
print(f"Train: {train.shape}, NaN: {train.isna().sum().sum()}")
print(f"Test: {test.shape}, NaN: {test.isna().sum().sum()}")
```

Expected: Train shape ~(6171, N), NaN = 0

### Task 2: Add polymer-specific descriptors

**Files:**
- Modify: `features/polymer_descriptors.py` (add new functions)
- Modify: `features/build_features.py:CALL_POLYMER_DESCRIPTORS_LINE` (integrate new descriptors)

**Interfaces:**
- Consumes: SMILES strings from train.csv/test.csv
- Produces: Additional columns in features DataFrame (repeat_length, branching, ring_count, aromatic_fraction, end_group_counts)

- [ ] **Step 1: Write polymer descriptor functions**

In `features/polymer_descriptors.py`, add:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def compute_polymer_descriptors(smiles: str) -> dict:
    """Compute polymer-specific descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "repeat_length": 0, "branching": 0, "ring_count": 0,
            "aromatic_fraction": 0.0, "n_heavy_atoms": 0,
            "has_oh": 0, "has_cooh": 0, "has_nh2": 0,
            "has_halide": 0, "has_vinyl": 0
        }

    # Count non-* heavy atoms (repeat length)
    heavy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetSymbol() != "*"]
    repeat_length = len(heavy_atoms)

    # Branching: any atom with degree > 2
    branching = 1 if any(a.GetDegree() > 2 for a in heavy_atoms) else 0

    # Ring count
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    # Aromatic fraction
    aromatic = sum(1 for a in heavy_atoms if a.GetIsAromatic())
    aromatic_fraction = aromatic / max(repeat_length, 1)

    # End-group detection via SMARTS
    smarts_patterns = {
        "has_oh": "[OH]",
        "has_cooh": "[CX3](=O)[OX2H1]",
        "has_nh2": "[NH2]",
        "has_halide": "[F,Cl,Br,I]",
        "has_vinyl": "[CX3]=[CX3]",
    }
    end_groups = {}
    for name, smarts in smarts_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        end_groups[name] = 1 if mol.HasSubstructMatch(pattern) else 0

    return {
        "repeat_length": repeat_length,
        "branching": branching,
        "ring_count": ring_count,
        "aromatic_fraction": aromatic_fraction,
        "n_heavy_atoms": repeat_length,
        **end_groups,
    }


def compute_polymer_descriptors_batch(smiles_list: list) -> np.ndarray:
    """Compute polymer descriptors for a batch of SMILES."""
    descs = [compute_polymer_descriptors(s) for s in smiles_list]
    return pd.DataFrame(descs).values
```

- [ ] **Step 2: Integrate into build_features**

In `features/build_features.py`, add after fingerprint computation:

```python
from features.polymer_descriptors import compute_polymer_descriptors_batch

# After existing feature computation:
polymer_descs = compute_polymer_descriptors_batch(smiles_list)
polymer_cols = ["repeat_length", "branching", "ring_count", "aromatic_fraction",
                "n_heavy_atoms", "has_oh", "has_cooh", "has_nh2", "has_halide", "has_vinyl"]
for i, col in enumerate(polymer_cols):
    features_df[col] = polymer_descs[:, i]
```

- [ ] **Step 3: Test polymer descriptors**

```bash
cd D:\Parth\Poly\polymer_competition
python -c "
from features.polymer_descriptors import compute_polymer_descriptors
result = compute_polymer_descriptors('*CCO*')
print(result)
assert result['repeat_length'] == 3  # C, C, O
assert result['branching'] == 0
print('PASSED')
"
```

- [ ] **Step 4: Rebuild features**

```bash
python -m features.build_features
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/features_train.parquet')
print(f'Columns: {len(df.columns)}')
assert 'repeat_length' in df.columns
assert 'aromatic_fraction' in df.columns
print('Polymer descriptors integrated')
"
```

### Task 3: Feature preprocessing pipeline

**Files:**
- Create: `features/preprocessing.py`

**Interfaces:**
- Consumes: Raw feature DataFrame, fit flag
- Produces: Preprocessed feature DataFrame, fitted scaler (for Ridge/MLP only)

- [ ] **Step 1: Create preprocessing module**

```python
# features/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class FeaturePreprocessor:
    """Feature preprocessing pipeline. Fit on train, transform on test."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.variance_threshold = None
        self.high_corr_mask = None
        self.cols_to_drop = []
        self.fitted = False

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit preprocessor on training data."""
        X_clean = self._clean(X.copy())

        # Impute
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_clean),
            columns=X_clean.columns, index=X_clean.index
        )

        # Variance threshold: remove zero-variance
        variances = X_imputed.var()
        self.cols_to_drop = list(variances[variances == 0].index)

        # Correlation filter: remove features with corr > 0.95
        corr_matrix = X_imputed.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.high_corr_mask = [col for col in upper.columns if any(upper[col] > 0.95)]

        # Fit scaler on remaining features
        keep_cols = [c for c in X_imputed.columns
                     if c not in self.cols_to_drop and c not in self.high_corr_mask]
        self.scaler.fit(X_imputed[keep_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        assert self.fitted, "Must call fit() first"
        X_clean = self._clean(X.copy())

        X_imputed = pd.DataFrame(
            self.imputer.transform(X_clean),
            columns=X_clean.columns, index=X_clean.index
        )

        keep_cols = [c for c in X_imputed.columns
                     if c not in self.cols_to_drop and c not in self.high_corr_mask]
        X_out = X_imputed[keep_cols].copy()

        if scale:
            X_out[:] = self.scaler.transform(X_out)

        return X_out

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with nan, then nan with 0."""
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.fillna(0)

    def get_feature_names(self) -> list:
        """Return list of features after preprocessing."""
        assert self.fitted, "Must call fit() first"
        return [c for c in self.scaler.feature_names_in_
                if c not in self.cols_to_drop and c not in self.high_corr_mask]
```

- [ ] **Step 2: Test preprocessing**

```python
import pandas as pd
import numpy as np
from features.preprocessing import FeaturePreprocessor

# Create dummy data
X_train = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [0, 0, 0, 0, 0]})
X_test = pd.DataFrame({"a": [1.5, 2.5], "b": [4.5, 3.5], "c": [0, 0]})

prep = FeaturePreprocessor()
prep.fit(X_train)
X_out = prep.transform(X_test, scale=True)
print(f"Output shape: {X_out.shape}")
assert "c" not in X_out.columns  # zero variance removed
print("PASSED")
```

### Task 4: Generate CV splits

**Files:**
- Read: `data/generate_splits.py` (verify it works)

- [ ] **Step 1: Generate splits**

```bash
cd D:\Parth\Poly\polymer_competition
python -m data.generate_splits --config config.yaml --target tg
python -m data.generate_splits --config config.yaml --target egc
```

- [ ] **Step 2: Verify splits**

```python
import pickle
tg = pickle.load(open("data/splits_tg.pkl", "rb"))
egc = pickle.load(open("data/splits_egc.pkl", "rb"))
print(f"Tg splits: {len(tg)} folds, sizes: {[len(v) for v in tg.values()]}")
print(f"Egc splits: {len(egc)} folds, sizes: {[len(v) for v in egc.values()]}")
```

---

## Phase 2: Tree Model Training

### Task 5: Create Optuna-tuned tree training function

**Files:**
- Modify: `training/train.py` (add Optuna integration)

**Interfaces:**
- Consumes: feature matrices, CV splits, model config
- Produces: OOF predictions, test predictions, metrics dict

- [ ] **Step 1: Add Optuna tuning to train.py**

In `training/train.py`, add after `build_model()`:

```python
import optuna
from sklearn.model_selection import cross_val_score

def tune_model_optuna(model_type: str, X: np.ndarray, y: np.ndarray,
                       n_trials: int = 50, seed: int = 42) -> dict:
    """Find best hyperparameters via Optuna."""
    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
            }
            from xgboost import XGBRegressor
            model = XGBRegressor(**params)

        elif model_type == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
                "verbose": -1,
            }
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)

        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "random_seed": seed,
                "verbose": 0,
            }
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**params)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

- [ ] **Step 2: Test Optuna tuning**

```python
import numpy as np
from training.train import tune_model_optuna

X = np.random.randn(100, 50)
y = np.random.randn(100)
best_params = tune_model_optuna("xgb", X, y, n_trials=5, seed=42)
print(f"Best params: {best_params}")
assert "n_estimators" in best_params
print("PASSED")
```

### Task 6: Train all tree models per target

**Files:**
- Create: `training/run_tree_models.py`

**Interfaces:**
- Consumes: `data/processed/features_train.parquet`, `data/splits_tg.pkl`, `data/splits_egc.pkl`
- Produces: `predictions/v27_{target}_{model}_fold{i}.pkl`, `predictions/v27_{target}_{model}_fold{i}_test.pkl`

- [ ] **Step 1: Create tree model runner**

```python
# training/run_tree_models.py
import sys, pickle, time, json, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

def run_tree_models(config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_ver = cfg.get("experiment", {}).get("version", "v27")
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    train_features = pd.read_parquet("data/processed/features_train.parquet")
    test_features = pd.read_parquet("data/processed/features_test.parquet")

    # Load train data for target splitting
    train_df = pd.read_csv("data/train.csv")

    models = ["xgb", "lgb", "catboost"]
    targets = ["tg", "egc"]

    for target in targets:
        print(f"\n{'='*60}")
        print(f"  Target: {target}")
        print(f"{'='*60}")

        # Split by target
        mask = train_df["target_type"] == target
        y_train = train_df.loc[mask, "target"].values
        X_train = train_features.loc[mask].values

        # Load CV splits
        splits = pickle.load(open(f"data/splits_{target}.pkl", "rb"))

        # Load test set for this target
        test_mask = test_features.index  # all test rows (already filtered)
        X_test = test_features.values

        for model_type in models:
            print(f"\n  Training {model_type}...")

            oof_preds = np.zeros(len(y_train))
            test_preds = np.zeros(len(X_test))
            fold_r2s = []

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Optuna tuning
                from training.train import tune_model_optuna
                best_params = tune_model_optuna(model_type, X_tr, y_tr, n_trials=30)

                # Train with best params
                if model_type == "xgb":
                    from xgboost import XGBRegressor
                    model = XGBRegressor(**best_params, random_state=42)
                elif model_type == "lgb":
                    from lightgbm import LGBMRegressor
                    model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
                elif model_type == "catboost":
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(**best_params, random_seed=42, verbose=0)

                model.fit(X_tr, y_tr)

                # OOF predictions
                oof_preds[val_idx] = model.predict(X_val)
                test_preds += model.predict(X_test) / len(splits)

                fold_r2 = r2_score(y_val, oof_preds[val_idx])
                fold_r2s.append(fold_r2)
                print(f"    Fold {fold_idx}: R² = {fold_r2:.4f}")

                # Save fold prediction
                fold_pred = {
                    "target": target,
                    "model_type": model_type,
                    "fold": fold_idx,
                    "val_r2": fold_r2,
                    "oof_preds": oof_preds[val_idx],
                    "val_idx": val_idx,
                }
                with open(pred_dir / f"{exp_ver}_{target}_{model_type}_fold{fold_idx}.pkl", "wb") as f:
                    pickle.dump(fold_pred, f)

                # Save test prediction
                test_pred = {
                    "target": target,
                    "model_type": model_type,
                    "fold": fold_idx,
                    "test_preds": test_preds,
                }
                with open(pred_dir / f"{exp_ver}_{target}_{model_type}_fold{fold_idx}_test.pkl", "wb") as f:
                    pickle.dump(test_pred, f)

            # Summary
            mean_r2 = np.mean(fold_r2s)
            std_r2 = np.std(fold_r2s)
            print(f"\n  {model_type} Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")

            # Save model summary
            summary = {
                "target": target,
                "model_type": model_type,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "fold_r2s": fold_r2s,
            }
            with open(pred_dir / f"{exp_ver}_{target}_{model_type}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_tree_models(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
```

- [ ] **Step 2: Run tree models**

```bash
cd D:\Parth\Poly\polymer_competition
python -m training.run_tree_models config.yaml
```

---

## Phase 3: PolyChain Training

### Task 7: Verify PolyChain model loads correctly

**Files:**
- Read: `models/polychain/polychain_model.py`
- Read: `training/train.py` (polychain branch)

- [ ] **Step 1: Test model instantiation**

```python
from models.polychain.polychain_model import PolyChain
import yaml

with open("training/configs/polychain_finetune.yaml") as f:
    cfg = yaml.safe_load(f)

model = PolyChain(
    in_dim=6380,
    edge_dim=4,
    hidden_dim=cfg["finetuning"]["hidden_dim"],
    n_backbone_layers=cfg["finetuning"]["n_backbone_layers"],
    n_hamf_layers=cfg["finetuning"]["n_hamf_layers"],
    dropout=cfg["finetuning"]["dropout"],
    cst_dim=10,
)
print(f"PolyChain params: {sum(p.numel() for p in model.parameters()):,}")
assert sum(p.numel() for p in model.parameters()) > 100000
print("PASSED")
```

### Task 8: Run Polychain 5-fold CV

**Files:**
- Read: `notebooks/local_run.ipynb` (verify Step 2 works)

- [ ] **Step 1: Start Jupyter and run Step 2 cell**

```bash
Start-Process powershell -ArgumentList "-NoExit", "-Command", "conda activate wan2gp; jupyter notebook --no-browser --port=8888 --notebook-dir='D:\Parth\Poly\polymer_competition'"
```

Open `notebooks/local_run.ipynb` and run Step 2 cell. This handles auto-skip/resume.

- [ ] **Step 2: Monitor progress**

Check `predictions/v27_{target}_polychain_fold{i}.pkl` files as they complete.

---

## Phase 4: Ensemble & Stacking

### Task 9: Build stacking meta-model

**Files:**
- Create: `ensemble/stacking.py`

**Interfaces:**
- Consumes: OOF predictions from all models (Phase 2 + Phase 3), CV splits
- Produces: Stacking meta-features, meta-model weights

- [ ] **Step 1: Create stacking module**

```python
# ensemble/stacking.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def build_stacking_features(target: str, exp_ver: str = "v27"):
    """Collect OOF predictions from all models into meta-features."""
    pred_dir = Path("predictions")
    splits = pickle.load(open(f"data/splits_{target}.pkl", "rb"))

    # Find all OOF prediction files for this target
    oof_files = sorted(pred_dir.glob(f"{exp_ver}_{target}_*_fold*.pkl"))
    oof_files = [f for f in oof_files if "_test" not in f.name and "summary" not in f.name]

    # Group by model type
    model_types = sorted(set(f.name.split("_")[2] for f in oof_files))

    # Load train target
    train_df = pd.read_csv("data/train.csv")
    mask = train_df["target_type"] == target
    y_train = train_df.loc[mask, "target"].values

    # Build meta-features
    n_samples = len(y_train)
    meta_features = np.zeros((n_samples, len(model_types)))

    for i, model_type in enumerate(model_type_files := [f for f in oof_files if model_type in f.name]):
        for fold_file in model_type_files:
            data = pickle.load(open(fold_file, "rb"))
            meta_features[data["val_idx"], i] = data["oof_preds"]

    return meta_features, y_train, model_types


def train_stacking_meta_model(meta_features: np.ndarray, y: np.ndarray) -> Ridge:
    """Train Ridge meta-model on OOF predictions."""
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, y)
    return meta_model


def compute_ensemble_weights(meta_features: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute optimized ensemble weights via Nelder-Mead."""
    from scipy.optimize import minimize

    def loss(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))
        preds = meta_features @ weights
        return -r2_score(y, preds)

    n_models = meta_features.shape[1]
    initial = np.ones(n_models) / n_models
    result = minimize(loss, initial, method="Nelder-Mead")
    optimal = np.abs(result.x) / np.sum(np.abs(result.x))
    return optimal
```

- [ ] **Step 2: Test stacking**

```python
from ensemble.stacking import build_stacking_features, compute_ensemble_weights

meta_X, y, model_types = build_stacking_features("tg")
print(f"Meta-features shape: {meta_X.shape}")
print(f"Models: {model_types}")

weights = compute_ensemble_weights(meta_X, y)
print(f"Weights: {dict(zip(model_types, weights))}")
print(f"Sum of weights: {weights.sum():.4f}")
print("PASSED")
```

### Task 10: Build ensemble submission

**Files:**
- Create: `ensemble/build_ensemble.py`

**Interfaces:**
- Consumes: Test predictions from all models, stacking weights
- Produces: `submission.csv`

- [ ] **Step 1: Create ensemble builder**

```python
# ensemble/build_ensemble.py
import pickle, yaml, sys
import numpy as np
import pandas as pd
from pathlib import Path

def build_submission(config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_ver = cfg.get("experiment", {}).get("version", "v27")
    pred_dir = Path(cfg["paths"]["predictions_dir"])

    # Load test data
    test_df = pd.read_csv("data/test.csv")

    # Separate by target
    test_tg = test_df[test_df["target_type"] == "tg"]
    test_egc = test_df[test_df["target_type"] == "egc"]

    predictions = {}

    for target, test_sub in [("tg", test_tg), ("egc", test_egc)]:
        # Collect test predictions from all models
        test_files = sorted(pred_dir.glob(f"{exp_ver}_{target}_*_fold*_test.pkl"))
        test_files = [f for f in test_files if "summary" not in f.name]

        model_types = sorted(set(f.name.split("_")[2] for f in test_files))
        n_test = len(test_sub)

        # Average across folds per model
        model_preds = {}
        for model_type in model_types:
            model_files = [f for f in test_files if model_type in f.name]
            preds = np.zeros(n_test)
            for fold_file in model_files:
                data = pickle.load(open(fold_file, "rb"))
                preds += data["test_preds"] / len(model_files)
            model_preds[model_type] = preds

        # Stack into matrix
        X_meta = np.column_stack([model_preds[m] for m in model_types])

        # Load weights
        from ensemble.stacking import compute_ensemble_weights, build_stacking_features
        meta_X, y, _ = build_stacking_features(target, exp_ver)
        weights = compute_ensemble_weights(meta_X, y)

        # Ensemble prediction
        ensemble_pred = X_meta @ weights

        # Add polychain predictions if available
        polychain_files = list(pred_dir.glob(f"{exp_ver}_{target}_polychain_fold*_test.pkl"))
        if polychain_files:
            polychain_pred = np.zeros(n_test)
            for pf in polychain_files:
                data = pickle.load(open(pf, "rb"))
                polychain_pred += data["test_preds"] / len(polychain_files)
            # Average with tree ensemble
            ensemble_pred = 0.5 * ensemble_pred + 0.5 * polychain_pred

        predictions[target] = dict(zip(test_sub["id"], ensemble_pred))

    # Merge predictions
    all_ids = test_df["id"].tolist()
    targets_list = test_df["target_type"].tolist()

    submission_rows = []
    for id_val, target_type in zip(all_ids, targets_list):
        pred = predictions[target_type][id_val]
        submission_rows.append({"id": id_val, "target": pred})

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv("submission.csv", index=False)
    print(f"Submission saved: {len(submission_df)} rows")
    print(f"  Range: [{submission_df['target'].min():.2f}, {submission_df['target'].max():.2f}]")


if __name__ == "__main__":
    build_submission(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
```

- [ ] **Step 2: Test submission generation**

```bash
cd D:\Parth\Poly\polymer_competition
python -m ensemble.build_ensemble config.yaml
python -c "
import pandas as pd
sub = pd.read_csv('submission.csv')
print(f'Submission: {len(sub)} rows, columns: {list(sub.columns)}')
assert len(sub) == 4115
assert list(sub.columns) == ['id', 'target']
print('PASSED')
"
```

---

## Phase 5: Visualizations

### Task 11: Create production visualization pipeline

**Files:**
- Create: `reports/run_all_visuals.py`

- [ ] **Step 1: Create visualization module**

```python
# reports/run_all_visuals.py
import pickle, yaml, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score

# Publication-quality settings
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (8, 6),
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = sns.color_palette("colorblind")


def pred_vs_actual(y_true, y_pred, model_name, target, save_dir):
    """Scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5, s=15, c=COLORS[0])
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"{model_name} — {target} (R²={r2_score(y_true, y_pred):.4f})")
    ax.legend()
    fig.savefig(save_dir / f"pred_vs_actual_{model_name}_{target}.png")
    plt.close(fig)


def residuals_plot(y_true, y_pred, model_name, target, save_dir):
    """Residuals distribution + Q-Q plot."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Distribution
    ax1.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color=COLORS[1])
    ax1.axvline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Residuals — {model_name} ({target})")

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f"Q-Q Plot — {model_name} ({target})")

    fig.savefig(save_dir / f"residuals_{model_name}_{target}.png")
    plt.close(fig)


def cv_per_fold(fold_scores: dict, model_names: list, target, save_dir):
    """Bar chart of CV R² per fold."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fold_scores[model_names[0]]))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        scores = fold_scores[model]
        ax.bar(x + i * width, scores, width, label=model, color=COLORS[i % len(COLORS)])

    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title(f"CV Per Fold — {target}")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([f"Fold {i}" for i in range(len(fold_scores[model_names[0]]))])
    ax.legend()
    fig.savefig(save_dir / f"cv_per_fold_{target}.png")
    plt.close(fig)


def model_comparison(mean_r2s: dict, target, save_dir):
    """Side-by-side R² bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(mean_r2s.keys())
    scores = list(mean_r2s.values())
    bars = ax.barh(models, scores, color=[COLORS[i % len(COLORS)] for i in range(len(models))])
    ax.set_xlabel("Mean R²")
    ax.set_title(f"Model Comparison — {target}")
    ax.set_xlim(0, 1)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, f"{score:.4f}", va="center")

    fig.savefig(save_dir / f"model_comparison_{target}.png")
    plt.close(fig)


def shap_summary(model, X, feature_names, target, save_dir):
    """SHAP summary plot (top 20 features)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=20, show=False)
        plt.title(f"SHAP Feature Importance — {target}")
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_summary_{target}.png")
        plt.close("all")
    except ImportError:
        print(f"  SHAP not available for {target}, skipping")


def target_distribution(train_df, save_dir):
    """Histogram of Tg and Egc distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target_type, color in [(ax1, "tg", COLORS[0]), (ax2, "egc", COLORS[1])]:
        data = train_df[train_df["target_type"] == target_type]["target"]
        ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color=color)
        ax.set_xlabel(f"{target_type.upper()}")
        ax.set_ylabel("Count")
        ax.set_title(f"{target_type.upper()} Distribution (n={len(data)})")

    fig.savefig(save_dir / "target_distribution.png")
    plt.close(fig)


def run_all_visuals(config_path: str = "config.yaml"):
    """Generate all visualizations."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_ver = cfg.get("experiment", {}).get("version", "v27")
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    save_dir = Path(cfg["paths"]["reports_dir"]) / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv("data/train.csv")

    # Target distribution
    target_distribution(train_df, save_dir)
    print("Generated: target_distribution.png")

    for target in ["tg", "egc"]:
        # Load fold R² scores
        summary_files = list(pred_dir.glob(f"{exp_ver}_{target}_*_summary.json"))
        model_r2s = {}
        fold_scores = {}

        for sf in summary_files:
            with open(sf) as f:
                data = json.load(f)
            model_r2s[data["model_type"]] = data["mean_r2"]
            fold_scores[data["model_type"]] = data["fold_r2s"]

        # Model comparison
        if model_r2s:
            model_comparison(model_r2s, target, save_dir)
            print(f"Generated: model_comparison_{target}.png")

        # CV per fold
        if fold_scores:
            cv_per_fold(fold_scores, list(fold_scores.keys()), target, save_dir)
            print(f"Generated: cv_per_fold_{target}.png")

    print(f"\nAll plots saved to {save_dir}")


if __name__ == "__main__":
    run_all_visuals(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
```

- [ ] **Step 2: Test visualizations**

```bash
cd D:\Parth\Poly\polymer_competition
python -m reports.run_all_visuals config.yaml
```

---

## Phase 6: Kaggle Notebook

### Task 12: Create Kaggle-ready notebook

**Files:**
- Modify: `notebooks/kaggle_pipeline.ipynb`

- [ ] **Step 1: Create notebook with all phases**

The notebook should:
1. Load data from `/kaggle/input/competitions/aisehack-2-0/`
2. Run feature engineering (RDKit + fingerprints + polymer descriptors)
3. Train all tree models with Optuna
4. Train PolyChain (if GPU available)
5. Build stacking ensemble
6. Generate `submission.csv`

- [ ] **Step 2: Test on Kaggle**

Upload notebook to Kaggle, verify it runs within 30 hr/week GPU quota.

---

## Summary

| Phase | Tasks | Key Files Created |
|---|---|---|
| 1. Features | Tasks 1-4 | `features/preprocessing.py` |
| 2. Trees | Tasks 5-6 | `training/run_tree_models.py` |
| 3. PolyChain | Tasks 7-8 | (uses existing) |
| 4. Ensemble | Tasks 9-10 | `ensemble/stacking.py`, `ensemble/build_ensemble.py` |
| 5. Visuals | Task 11 | `reports/run_all_visuals.py` |
| 6. Kaggle | Task 12 | `notebooks/kaggle_pipeline.ipynb` |

**Total: 12 tasks across 6 phases.**
