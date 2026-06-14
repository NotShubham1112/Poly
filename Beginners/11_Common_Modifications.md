# Chapter 11: Common Modifications

## Introduction

This chapter explains how to make common changes to the PolyChain project. Think of it as a "cookbook" — each recipe tells you exactly what to change and where.

---

## Core Concepts

### Modification Safety

Before making changes:
1. **Back up** your code (git commit)
2. **Understand** what you're changing
3. **Test** your changes incrementally
4. **Document** what you changed

---

## Modification 1: Adding a New Model

### Step 1: Create Model Class

Create `models/my_model.py`:

```python
"""
models/my_model.py
My custom model for polymer property prediction.
"""
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, dropout=0.2):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        x = F.relu(self.encoder(x))
        return self.head(x).squeeze(-1)
```

### Step 2: Add to Model Factory

Edit `training/train.py`:

```python
def build_model(model_type, cfg, in_dim=None, edge_dim=None, n_features=None):
    # ... existing code ...
    if model_type == "my_model":
        from models.my_model import MyModel
        return MyModel(in_dim=n_features, **cfg), True
    # ... existing code ...
```

### Step 3: Add to Model List

Edit `generate_all.py`:

```python
ALL_MODEL_TYPES = [
    "ridge", "xgb", "lgb", "catboost", "rf", "mlp",
    "gcn", "gat", "mpnn", "graph_transformer", "polychain",
    "my_model",  # Add here
]
```

### Step 4: Add Config (Optional)

Create `training/configs/my_model.yaml`:

```yaml
my_model:
  hidden_dim: 256
  dropout: 0.3
  epochs: 100
  lr: 0.001
```

### Step 5: Test

```bash
python -m training.train --model_type my_model --fold 0
```

---

## Modification 2: Adding a New Feature

### Step 1: Add Feature Function

Edit `features/custom_polymer.py`:

```python
def my_custom_feature(smiles: str) -> float:
    """Compute my custom feature."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0.0
    # Your feature computation here
    return float(len(mol.GetAtoms()))
```

### Step 2: Add to Feature Suite

Edit `features/custom_polymer.py`:

```python
def compute_all_custom_features(smiles_list):
    rows = []
    for smi in smiles_list:
        row = {
            # ... existing features ...
            "my_custom_feature": my_custom_feature(smi),  # Add here
        }
        rows.append(row)
    return pd.DataFrame(rows)
```

### Step 3: Test

```python
from features.custom_polymer import my_custom_feature

print(my_custom_feature("*CCO*"))  # Should print a number
```

---

## Modification 3: Changing Hyperparameters

### Option 1: Edit Config File

Edit `config.yaml`:

```yaml
cv:
  n_folds: 3  # Change from 5 to 3

device:
  use_cuda: false  # Change from true to false
```

### Option 2: Command Line Arguments

```bash
# Override number of folds
python generate_all.py --n-folds 3

# Override person name
python generate_all.py --person myname
```

### Option 3: Model-Specific Config

Edit `training/configs/polychain_finetune.yaml`:

```yaml
finetuning:
  epochs: 100  # Change from 200
  batch_size: 64  # Change from 32
  lr: 0.001  # Change from 0.0001
```

---

## Modification 4: Changing Cross-Validation Strategy

### Current Strategy

```yaml
cv:
  split_type: "group"  # GroupKFold by SMILES scaffold
  group_strategy: "scaffold"
```

### Change to Random Split

Edit `config.yaml`:

```yaml
cv:
  split_type: "random"  # Random KFold
```

### Change to Stratified Split

Edit `config.yaml`:

```yaml
cv:
  split_type: "stratified"  # Stratified by target bins
```

---

## Modification 5: Adding a New Ensemble Strategy

### Step 1: Add Weight Function

Edit `ensemble/weight_optimizer.py`:

```python
def my_weight_strategy(oof_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """My custom weight strategy."""
    # Your weight computation here
    n_models = oof_preds.shape[1]
    return np.ones(n_models) / n_models  # Equal weights
```

### Step 2: Add to Strategy Selector

Edit `ensemble/weight_optimizer.py`:

```python
def get_weights(strategy, oof_preds, y_true):
    # ... existing code ...
    if strategy == "my_strategy":
        return my_weight_strategy(oof_preds, y_true)
    # ... existing code ...
```

### Step 3: Use New Strategy

```bash
python -m ensemble.build_ensemble --strategy my_strategy
```

---

## Modification 6: Changing the Target Property

### Current Target

```yaml
target:
  column: "property"
  task: "regression"
```

### Change to Different Property

Edit `config.yaml`:

```yaml
target:
  column: "Tg"  # Or "Tm", "density", etc.
  task: "regression"
```

**Note**: Ensure the column exists in `data/train.csv`

---

## Modification 7: Using a Different Pretrained Model

### Current: ChemBERTa

```python
# models/chemberta.py
model_name = "DeepChem/ChemBERTa-77M-MTR"
```

### Change to Different Model

Edit `models/chemberta.py`:

```python
# Use a different model
model_name = "seyonec/ChemBERTa-zinc-base-v1"
# or
model_name = "DeepChem/ChemBERTa-1M-MTR"
```

---

## Modification 8: Adding Data Augmentation

### Current Augmentation

The project uses SMILES randomization (5× augmentation).

### Add New Augmentation

Edit `training/train.py`:

```python
def augment_smiles(smiles_list, n_augmentations=5):
    """Add custom augmentation."""
    augmented = []
    for smi in smiles_list:
        augmented.append(smi)  # Original
        # Add your augmentation here
        # Example: reverse SMILES
        augmented.append(smi[::-1])
    return augmented
```

---

## Modification 9: Changing Early Stopping

### Current Settings

```yaml
early_stopping:
  enabled: true
  patience: 30
  metric: "val_rmse"
```

### Change Patience

Edit `training/configs/polychain_finetune.yaml`:

```yaml
early_stopping:
  enabled: true
  patience: 50  # Change from 30
  metric: "val_rmse"
```

### Disable Early Stopping

Edit `training/configs/polychain_finetune.yaml`:

```yaml
early_stopping:
  enabled: false
```

---

## Modification 10: Adding a New Report

### Step 1: Create Report Function

Edit `reports/generate_reports.py`:

```python
def generate_my_report(pred_dir: Path, output_dir: Path):
    """Generate my custom report."""
    # Your report generation here
    df = load_all_predictions(pred_dir)
    # ... analysis ...
    plt.savefig(output_dir / "my_report.png")
```

### Step 2: Add to Main Function

Edit `reports/generate_reports.py`:

```python
def main():
    # ... existing code ...
    print("\n[4/4] My Report")
    generate_my_report(pred_dir, output_dir)
```

---

## Modification 11: Changing the Loss Function

### Current Loss

```python
criterion = nn.MSELoss()
```

### Change to MAE Loss

Edit `training/train.py`:

```python
criterion = nn.L1Loss()  # MAE instead of MSE
```

### Change to Huber Loss

Edit `training/train.py`:

```python
criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
```

---

## Modification 12: Adding a New Preprocessing Step

### Step 1: Create Preprocessing Function

Create `data/preprocess.py`:

```python
"""
data/preprocess.py
Custom preprocessing steps.
"""
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply custom preprocessing."""
    # Remove duplicates
    df = df.drop_duplicates(subset=["SMILES"])
    
    # Filter by SMILES length
    df = df[df["SMILES"].str.len() < 100]
    
    return df
```

### Step 2: Integrate into Pipeline

Edit `generate_all.py`:

```python
def step_2_features(config):
    """Build the feature matrix."""
    # Add preprocessing
    from data.preprocess import preprocess_data
    train = pd.read_csv("data/train.csv")
    train = preprocess_data(train)
    train.to_csv("data/train_processed.csv", index=False)
    
    run_cmd(
        [sys.executable, "-m", "features.build_features", "--config", config],
        desc="Step 2: Build Feature Matrix",
    )
```

---

## Examples

### Example: Complete Model Addition

```python
# 1. Create model
# models/my_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MyGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)
        return self.head(g).squeeze(-1)

# 2. Add to factory (training/train.py)
# 3. Add to model list (generate_all.py)
# 4. Test
```

---

## Common Mistakes

1. **Not testing changes**: Always test with a single fold before running full pipeline
2. **Breaking existing functionality**: Check that existing models still work
3. **Forgetting to update configs**: New features often need new config options
4. **Not documenting changes**: Add comments explaining what you changed and why

---

## Summary

- Adding a new model: Create class → add to factory → add to list → test
- Adding a new feature: Add function → add to suite → test
- Changing hyperparameters: Edit config files or use command line
- Most changes are localized to 1-2 files

---

## Key Takeaways

- Most modifications follow a pattern: create → register → test
- Always test changes incrementally before running full pipeline
- Config files control most behavior without code changes
- Document your changes for future reference
- Back up your code before making major changes
