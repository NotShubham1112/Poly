# Chapter 7: APIs and Services

## Introduction

This chapter explains the external interfaces, services, and APIs in the PolyChain project. An API (Application Programming Interface) is like a restaurant menu — it tells you what you can order (call) and what you'll get back (return).

---

## Core Concepts

### What is an API?

An API defines:
- **Inputs**: What you send to the service
- **Outputs**: What you get back
- **Protocol**: How they communicate (HTTP, Python function, CLI)

### Types of APIs in This Project

1. **Python Class APIs**: Classes you import and use in code
2. **CLI APIs**: Commands you run in the terminal
3. **Web APIs**: HTTP endpoints (Streamlit app)

---

## Python Class APIs

### 1. `PolymerPredictor` — Prediction API

**File**: `inference/predictor.py`

**Purpose**: Load a trained model and predict on new SMILES

**Usage**:
```python
from inference.predictor import PolymerPredictor

# Initialize
pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")

# Predict
result = pred.predict(["*CCO*", "*c1ccc(*)cc1*"])
print(result)  # [350.2, 420.5]
```

**Parameters**:
- `checkpoint_path` (str): Path to trained model checkpoint
- `device` (str): `"cpu"` or `"cuda"` (default: `"cpu"`)

**Returns**:
- `np.ndarray`: Predicted property values

**Example**:
```python
# Single prediction
pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")
yhat = pred.predict(["*CCO*"])
print(f"Predicted Tg: {yhat[0]:.1f} K")

# Batch prediction
smiles_list = ["*CCO*", "*c1ccc(*)cc1*", "*CC(c1ccccc1)*"]
predictions = pred.predict(smiles_list)
for smi, pred_val in zip(smiles_list, predictions):
    print(f"{smi}: {pred_val:.1f}")
```

---

### 2. `PolyChain` — Model API

**File**: `models/polychain/polychain_model.py`

**Purpose**: The complete PolyChain model architecture

**Usage**:
```python
from models.polychain import PolyChain
from features.graph_utils import build_multiscale, collate_multiscale
from models.polychain.cst import compute_cst_batch

# Build model
model = PolyChain(
    in_atom_dim=60,
    in_edge_dim=6,
    hidden_dim=256,
    n_backbone_layers=4,
    n_hamf_layers=2,
    dropout=0.2
)

# Prepare input
smiles = ["*CCO*"]
samples = [build_multiscale(s) for s in smiles]
batch = collate_multiscale(samples)
batch["cst"] = torch.tensor(compute_cst_batch(smiles))

# Forward pass
with torch.no_grad():
    prediction = model(batch)
print(prediction)  # tensor([350.2])
```

**Parameters**:
- `in_atom_dim` (int): Input atom feature dimension (default: 60)
- `in_edge_dim` (int): Input edge feature dimension (default: 6)
- `hidden_dim` (int): Hidden layer dimension (default: 256)
- `n_backbone_layers` (int): Number of GIN layers (default: 4)
- `n_hamf_layers` (int): Number of HAMF blocks (default: 2)
- `dropout` (float): Dropout rate (default: 0.2)

**Returns**:
- `torch.Tensor`: Predicted property values, shape `(batch_size,)`

---

### 3. Feature Engineering APIs

#### `compute_all_custom_features()`

**File**: `features/custom_polymer.py`

**Purpose**: Compute all polymer-specific features

**Usage**:
```python
from features.custom_polymer import compute_all_custom_features

smiles_list = ["*CCO*", "*c1ccc(*)cc1*"]
df = compute_all_custom_features(smiles_list)
print(df.columns)  # ['SMILES', 'n_asterisks', 'repeat_length', ...]
```

#### `morgan_fingerprints()`

**File**: `features/fingerprints.py`

**Purpose**: Compute Morgan fingerprints

**Usage**:
```python
from features.fingerprints import morgan_fingerprints

smiles_list = ["*CCO*", "*c1ccc(*)cc1*"]
fps = morgan_fingerprints(smiles_list, radius=2, n_bits=2048)
print(fps.shape)  # (2, 2048)
```

#### `compute_descriptors()`

**File**: `features/descriptors.py`

**Purpose**: Compute RDKit 2D descriptors

**Usage**:
```python
from features.descriptors import compute_descriptors

smiles_list = ["*CCO*", "*c1ccc(*)cc1*"]
descs = compute_descriptors(smiles_list)
print(descs.columns)  # ['SMILES', 'MW', 'LogP', 'TPSA', ...]
```

#### `compute_cst()`

**File**: `models/polychain/cst.py`

**Purpose**: Compute Chain Statistics Token

**Usage**:
```python
from models.polychain.cst import compute_cst, CST_DIM

cst = compute_cst("*CCO*")
print(cst.shape)  # (33,)
print(f"CST has {CST_DIM} features")
```

---

### 4. Graph Construction APIs

#### `build_multiscale()`

**File**: `features/graph_utils.py`

**Purpose**: Build multi-scale graphs for PolyChain

**Usage**:
```python
from features.graph_utils import build_multiscale

sample = build_multiscale("*CCO*", y=350.2)
print(sample.monomer)  # PyG Data object
print(sample.dimer)    # PyG Data object
print(sample.trimer)   # PyG Data object
print(sample.periodic) # PyG Data object
```

#### `collate_multiscale()`

**File**: `features/graph_utils.py`

**Purpose**: Batch multiple multi-scale samples

**Usage**:
```python
from features.graph_utils import build_multiscale, collate_multiscale

samples = [build_multiscale("*CCO*"), build_multiscale("*c1ccc(*)cc1*")]
batch = collate_multiscale(samples)
print(batch.keys())  # dict_keys(['monomer', 'dimer', 'trimer', 'periodic', 'y', 'smiles'])
```

---

## CLI APIs

### `generate_all.py` — Pipeline CLI

**Usage**:
```bash
# Full pipeline
python generate_all.py

# Specific steps
python generate_all.py --steps 1,2,3

# Specific models
python generate_all.py --steps 3 --models xgb,lgb,polychain

# Custom person name
python generate_all.py --person myname

# Custom number of folds
python generate_all.py --n-folds 3
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to global config |
| `--steps` | `1,2,3,4,5` | Steps to run |
| `--models` | All 11 | Models to train |
| `--person` | `team` | Name for prediction files |
| `--n-folds` | `5` | Number of CV folds |

---

### `training/train.py` — Training CLI

**Usage**:
```bash
# Train XGBoost on fold 0
python -m training.train --model_type xgb --fold 0

# Train PolyChain on fold 0
python -m training.train --model_type polychain --fold 0 --person myname

# With custom config
python -m training.train --model_type polychain --fold 0 --model_config training/configs/polychain_finetune.yaml
```

**Arguments**:
| Argument | Required | Description |
|----------|----------|-------------|
| `--model_type` | Yes | Model type (ridge, xgb, lgb, etc.) |
| `--fold` | No | Fold number (default: 0) |
| `--config` | No | Global config path |
| `--model_config` | No | Model-specific config |
| `--person` | No | Name for prediction files |

---

### `data/generate_splits.py` — Split Generation CLI

**Usage**:
```bash
# Generate splits with default config
python -m data.generate_splits

# With custom config
python -m data.generate_splits --config config.yaml

# Override strategy
python -m data.generate_splits --strategy random
```

---

### `ensemble/build_ensemble.py` — Ensemble CLI

**Usage**:
```bash
# Build ensemble
python -m ensemble.build_ensemble

# With custom strategy
python -m ensemble.build_ensemble --strategy nelder_mead
```

---

### `notebooks/eda_report.py` — EDA CLI

**Usage**:
```bash
# Run EDA
python notebooks/eda_report.py

# With custom paths
python notebooks/eda_report.py --train data/train.csv --test data/test.csv
```

---

## Web API (Streamlit App)

### `inference/chat_interface.py` — Web Interface

**Purpose**: Interactive web UI for polymer property prediction

**How to run**:
```bash
streamlit run inference/chat_interface.py
# or
streamlit run demo/app.py
```

**Opens at**: `http://localhost:8501`

**Features**:

#### Tab 1: Predict
- Input: SMILES string
- Output: Molecule visualization, feature summary, property prediction

#### Tab 2: Chat
- Input: Natural language with SMILES
- Output: Feature summary, property prediction

#### Tab 3: Similar (RAG)
- Input: SMILES string
- Output: Similar polymers from training set (placeholder)

#### Tab 4: About
- Architecture documentation
- MCP integration info

**Example interaction**:
```
User: *CCO*
App: [Shows molecule image]
     [Shows features: Connection points: 2, Repeat length: 3, ...]
     [Predicted property: 350.2]
```

---

## Error Handling

### Prediction Errors

```python
from inference.predictor import PolymerPredictor

try:
    pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")
    result = pred.predict(["invalid_smiles"])
except FileNotFoundError:
    print("Checkpoint not found. Train the model first.")
except Exception as e:
    print(f"Prediction failed: {e}")
```

### Training Errors

```bash
# If model_type is invalid
python -m training.train --model_type invalid
# Error: ValueError: Unknown model_type: invalid

# If fold number is invalid
python -m training.train --model_type xgb --fold 10
# Error: IndexError: list index out of range
```

---

## Examples

### Example: Complete Prediction Pipeline

```python
from inference.predictor import PolymerPredictor
from features.custom_polymer import compute_all_custom_features
import pandas as pd

# 1. Load model
pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")

# 2. Prepare data
test_smiles = ["*CCO*", "*c1ccc(*)cc1*", "*CC(c1ccccc1)*"]

# 3. Get predictions
predictions = pred.predict(test_smiles)

# 4. Get features for analysis
features = compute_all_custom_features(test_smiles)

# 5. Combine results
results = pd.DataFrame({
    "SMILES": test_smiles,
    "predicted_property": predictions,
    "repeat_length": features["repeat_length"],
    "rigidity_index": features["rigidity_index"]
})
print(results)
```

### Example: Batch Training

```python
import subprocess
import sys

models = ["ridge", "xgb", "lgb", "catboost", "rf", "mlp", "gcn", "gat", "mpnn", "graph_transformer", "polychain"]

for model in models:
    for fold in range(5):
        cmd = [
            sys.executable, "-m", "training.train",
            "--model_type", model,
            "--fold", str(fold),
            "--person", "batch_run"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed: {model} fold {fold}")
            print(result.stderr)
```

---

## Common Mistakes

1. **Wrong checkpoint path**: Ensure the path to the checkpoint is correct
2. **Missing dependencies**: Some features require RDKit, PyTorch, etc.
3. **Wrong SMILES format**: SMILES must use `*` for connection points
4. **Running from wrong directory**: Always run from `polymer_competition/`

---

## Summary

- **Python Class APIs**: `PolymerPredictor`, `PolyChain`, feature functions
- **CLI APIs**: `generate_all.py`, `training/train.py`, `data/generate_splits.py`
- **Web API**: Streamlit app at `localhost:8501`
- Each API has clear inputs, outputs, and error handling

---

## Key Takeaways

- Use `PolymerPredictor` for making predictions in code
- Use CLI scripts for training and pipeline execution
- Use Streamlit app for interactive exploration
- Always handle errors gracefully
- Checkpoint paths must be correct for predictions to work
