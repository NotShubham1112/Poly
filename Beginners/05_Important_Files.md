# Chapter 5: Important Files

## Introduction

This chapter explains every major file in the project, its purpose, when it runs, who calls it, and what it imports/exports.

---

## Core Concepts

### File Organization

Files are organized by responsibility:
- **Entry points**: Files you run directly (e.g., `generate_all.py`)
- **Modules**: Files imported by other files (e.g., `models/polychain/`)
- **Configs**: YAML files with settings (e.g., `config.yaml`)
- **Data**: CSV and PKL files with data

---

## Entry Point Files

### `generate_all.py` — Master Pipeline Script

**Purpose**: Orchestrates the entire 5-step pipeline

**When it runs**: When you execute `python generate_all.py`

**Who calls it**: The user (you)

**What it imports**:
- `argparse`, `os`, `subprocess`, `sys`, `pathlib`
- Calls other scripts via `subprocess.run()`

**What it exports**: Nothing (runs as a script)

**Key functions**:
- `run_cmd()` — executes a command and streams output
- `step_1_splits()` — generates CV splits
- `step_2_features()` — builds feature matrix
- `step_3_train()` — trains all models across all folds
- `step_4_ensemble()` — builds ensemble
- `step_5_reports()` — generates reports

**Code snippet**:
```python
# generate_all.py:step_3_train()
for model_type in models:
    for fold in range(n_folds):
        cmd = [
            sys.executable, "-m", "training.train",
            "--model_type", model_type,
            "--fold", str(fold),
            "--config", config,
        ]
        run_cmd(cmd, desc=f"Train {model_type} fold {fold}")
```

**What happens if removed**: **Cannot run the pipeline**

---

### `training/train.py` — Main Training Entry Point

**Purpose**: Trains any of the 11 model types on a specific fold

**When it runs**: Called by `generate_all.py` step 3, or directly via `python -m training.train`

**Who calls it**: `generate_all.py`, or the user

**What it imports**:
- `training.train_utils` — metrics, checkpointing
- `models.baselines`, `models.tree_models`, `models.mlp`, `models.gnn`, `models.graph_transformer`, `models.polychain`
- `features.graphs`, `features.graph_utils`, `models.polychain.cst`

**What it exports**: Nothing (runs as a script), but saves `.pkl` files to `predictions/`

**Key functions**:
- `build_model()` — factory that creates any model type
- `train_tabular()` — trains sklearn-style models
- `train_graph()` — trains PyTorch graph models
- `move_to_device()` — moves batch to GPU
- `main()` — entry point

**Code snippet**:
```python
# training/train.py:build_model()
def build_model(model_type, cfg, in_dim, edge_dim, n_features):
    if model_type == "ridge":
        return get_linear_model("ridge"), False
    if model_type == "xgb":
        return get_tree_model("xgb", **cfg), False
    if model_type == "polychain":
        return PolyChain(in_atom_dim=in_dim, ...), True
    raise ValueError(f"Unknown model_type: {model_type}")
```

**What happens if removed**: **Cannot train any model**

---

### `inference/chat_interface.py` — Streamlit Web Interface

**Purpose**: Provides a web UI for predicting polymer properties

**When it runs**: When you execute `streamlit run inference/chat_interface.py`

**Who calls it**: The user (via browser)

**What it imports**:
- `streamlit` — web framework
- `inference.predictor.PolymerPredictor` — prediction engine
- `features.custom_polymer` — feature computation
- `rdkit` — molecule visualization

**What it exports**: Nothing (runs as a script)

**Key features**:
- Tab 1: Predict — molecule visualization + prediction
- Tab 2: Chat — natural language Q&A
- Tab 3: Similar — RAG retrieval (placeholder)
- Tab 4: About — architecture documentation

**What happens if removed**: **No web interface**

---

## Model Files

### `models/polychain/polychain_model.py` — End-to-End PolyChain

**Purpose**: Defines the complete PolyChain architecture

**When it runs**: During training and inference

**Who calls it**: `training/train.py`, `inference/predictor.py`

**What it imports**:
- `models.polychain.backbone.GINBackbone`
- `models.polychain.hamf.HAMF`
- `models.polychain.pecgn.PECGN`
- `models.polychain.cst.CSTNormalizer`

**What it exports**: `PolyChain` class

**Key class**:
```python
class PolyChain(nn.Module):
    def __init__(self, in_atom_dim, in_edge_dim, hidden_dim=256, ...):
        self.backbone = GINBackbone(...)
        self.hamf = HAMF(...)
        self.cst_norm = CSTNormalizer(...)
        self.pecgn = PECGN(...)
        self.head = nn.Sequential(...)
    
    def forward(self, batch_dict):
        h1 = self.encode_scale(batch_dict["monomer"])
        h2 = self.encode_scale(batch_dict["dimer"])
        h3 = self.encode_scale(batch_dict["trimer"])
        fused = self.hamf([h1, h2, h3])
        cst_emb = self.cst_norm(batch_dict["cst"])
        periodic = self.pecgn(fused, cst_emb)
        cat = torch.cat([periodic, cst_emb], dim=-1)
        return self.head(cat).squeeze(-1)
```

**What happens if removed**: **PolyChain cannot run**

---

### `models/polychain/backbone.py` — GIN-S Encoder

**Purpose**: Shared graph encoder for all scales

**When it runs**: During PolyChain forward pass

**Who calls it**: `polychain_model.py`

**What it imports**:
- `torch`, `torch.nn`
- `torch_geometric.nn.GINConv`, `global_add_pool`
- `torch_scatter.scatter_add`

**What it exports**: `GINBackbone` class

**Key class**:
```python
class GINBackbone(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, n_layers, dropout):
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = ModuleList([GINEConv(...) for _ in range(n_layers)])
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))
    
    def forward(self, data, virtual_state=None):
        x = self.atom_encoder(data.x)
        for conv, norm, v_mlp in zip(self.convs, self.norms, self.virtual_mlp):
            x = x + virtual_state[data.batch]  # Add virtual state
            x = conv(x, data.edge_index, data.edge_attr)
            virtual_state = v_mlp(scatter_add(x, data.batch))
        g = global_add_pool(x, data.batch)
        return g, virtual_state
```

**What happens if removed**: **PolyChain has no graph encoder**

---

### `models/polychain/hamf.py` — Multi-Scale Fusion

**Purpose**: Fuses monomer/dimer/trimer embeddings via cross-attention

**When it runs**: During PolyChain forward pass

**Who calls it**: `polychain_model.py`

**What it imports**:
- `torch`, `torch.nn`, `math`

**What it exports**: `HAMF` class

**Key class**:
```python
class HAMF(nn.Module):
    def __init__(self, in_dim, out_dim, n_scales=3, n_layers=2, n_heads=4):
        self.proj = nn.Linear(in_dim, out_dim)
        self.scale_pe = nn.Parameter(torch.randn(n_scales, out_dim) * 0.02)
        self.blocks = ModuleList([HAMFBlock(...) for _ in range(n_layers)])
    
    def forward(self, scale_embeddings):
        x = torch.stack([self.proj(e) for e in scale_embeddings], dim=1)
        x = x + self.scale_pe.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return x.flatten(start_dim=1)  # (B, 3*dim)
```

**What happens if removed**: **No multi-scale fusion**

---

### `models/polychain/pecgn.py` — Periodic Boundary

**Purpose**: Adds learned periodic boundary operator

**When it runs**: During PolyChain forward pass

**Who calls it**: `polychain_model.py`

**What it imports**:
- `torch`, `torch.nn`

**What it exports**: `PECGN` class

**Key class**:
```python
class PECGN(nn.Module):
    def __init__(self, dim, cst_dim, init_alpha=0.05, max_alpha=0.3):
        self.boundary = BoundaryOp(dim, cst_dim)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, h_trimer, cst):
        B = h_trimer.size(0)
        dir_left = torch.zeros(B, 1, device=h_trimer.device)
        dir_right = torch.ones(B, 1, device=h_trimer.device)
        bL = self.boundary(h_trimer, cst, dir_left)
        bR = self.boundary(h_trimer, cst, dir_right)
        boundary = 0.5 * (bL + bR)
        alpha = torch.clamp(self.alpha, max=self.max_alpha)
        return h_trimer + alpha * boundary
```

**What happens if removed**: **No periodic equivariance**

---

### `models/polychain/cst.py` — Chain Statistics Token

**Purpose**: Computes polymer-specific features from SMILES

**When it runs**: During PolyChain forward pass

**Who calls it**: `polychain_model.py`, `training/train.py`

**What it imports**:
- `features.custom_polymer` — all feature functions
- `torch`, `torch.nn`

**What it exports**: `compute_cst()`, `compute_cst_batch()`, `CSTNormalizer`, `CST_DIM`

**Key function**:
```python
def compute_cst(smiles):
    base = {
        "n_asterisks": asterisks_count(smiles),
        "repeat_length": repeat_unit_length(smiles),
        "is_branched": branching_indicator(smiles),
        # ... 33 features total
    }
    return np.array([base.get(n, 0.0) for n in CST_BASE_FEATURE_NAMES])
```

**What happens if removed**: **No chain statistics**

---

## Feature Files

### `features/graphs.py` — Graph Construction

**Purpose**: Builds molecular graphs from SMILES

**When it runs**: During feature engineering and training

**Who calls it**: `features/graph_utils.py`, `training/train.py`, `inference/predictor.py`

**What it imports**:
- `torch`, `torch_geometric.data.Data`
- `rdkit.Chem`

**What it exports**: `smiles_to_graph()`, `kmer_graph()`, `periodic_graph()`, `atom_features()`, `bond_features()`

**Key functions**:
```python
def smiles_to_graph(smiles, y=None):
    """Convert SMILES to a monomer graph."""
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()])
    # ... build edge_index and edge_attr
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def kmer_graph(smiles, k=2, y=None):
    """Build a k-mer graph by concatenating k copies."""
    # ... concatenate k copies of the repeat unit

def periodic_graph(smiles, k=1, y=None):
    """Build a periodic graph by closing the k-mer chain."""
    km = kmer_graph(smiles, k=k)
    # ... add edge between the two * atoms
```

**What happens if removed**: **No graph data for any GNN**

---

### `features/fingerprints.py` — Molecular Fingerprints

**Purpose**: Computes molecular fingerprints

**When it runs**: During feature engineering

**Who calls it**: `features/build_features.py`, `reports/generate_reports.py`

**What it imports**:
- `numpy`
- `rdkit.Chem`, `rdkit.Chem.AllChem`, `rdkit.Chem.MACCSkeys`

**What it exports**: `morgan_fingerprints()`, `maccs_fingerprints()`, `atom_pair_fingerprints()`, `topological_torsion_fingerprints()`, `all_fingerprints()`

**Key function**:
```python
def morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = gen.GetFingerprint(mol)
        out[i] = np.array(fp, dtype=np.uint8)
    return out
```

**What happens if removed**: **No fingerprint features**

---

### `features/custom_polymer.py` — Polymer-Specific Features

**Purpose**: Computes polymer-specific features

**When it runs**: During feature engineering

**Who calls it**: `features/build_features.py`, `models/polychain/cst.py`, `inference/chat_interface.py`

**What it imports**:
- `numpy`, `pandas`
- `rdkit.Chem`, `rdkit.Chem.rdMolDescriptors`

**What it exports**: `compute_all_custom_features()`, `asterisks_count()`, `repeat_unit_length()`, `branching_indicator()`, `ring_statistics()`, `hbond_donor_acceptor()`, `rotatable_bonds()`, `end_group_counts()`, `rigidity_index()`, `hbond_density()`, `molecular_weight_monomer()`

**Key function**:
```python
def compute_all_custom_features(smiles_list):
    rows = []
    for smi in smiles_list:
        row = {
            "n_asterisks": asterisks_count(smi),
            "repeat_length": repeat_unit_length(smi),
            "is_branched": branching_indicator(smi),
            # ... 30+ features
        }
        rows.append(row)
    return pd.DataFrame(rows)
```

**What happens if removed**: **No polymer-specific features**

---

## Ensemble Files

### `ensemble/build_ensemble.py` — Ensemble Builder

**Purpose**: Combines OOF predictions into final submission

**When it runs**: Step 4 of the pipeline

**Who calls it**: `generate_all.py`

**What it imports**:
- `ensemble.weight_optimizer`
- `numpy`, `pandas`, `pickle`

**What it exports**: Nothing (saves `submission.csv`)

**Key functions**:
```python
def load_predictions(pred_dir):
    """Load all .pkl files and concatenate."""
    rows = []
    for pkl_file in pred_dir.glob("*.pkl"):
        data = pickle.load(open(pkl_file, "rb"))
        # ... extract predictions
    return pd.DataFrame(rows)

def build_oof_matrix(df):
    """Pivot to (n_samples, n_models) matrix."""
    grouped = df.groupby(["idx", "model_type"])["pred"].mean().unstack()
    return grouped.values, y.values, list(grouped.columns)
```

**What happens if removed**: **No submission.csv generated**

---

### `ensemble/weight_optimizer.py` — Weight Optimization

**Purpose**: Optimizes ensemble weights

**When it runs**: Called by `build_ensemble.py`

**Who calls it**: `build_ensemble.py`

**What it imports**:
- `numpy`, `scipy.optimize.minimize`, `sklearn.linear_model.Ridge`

**What it exports**: `get_weights()`, `inverse_rmse_weights()`, `nelder_mead_weights()`, `stacking_ridge()`

**Key functions**:
```python
def inverse_rmse_weights(oof_preds, y_true):
    """Weights proportional to 1/RMSE."""
    rmses = np.sqrt(np.mean((oof_preds - y_true[:, None]) ** 2, axis=0))
    inv = 1.0 / (rmses + 1e-8)
    return inv / inv.sum()

def nelder_mead_weights(oof_preds, y_true):
    """Optimize weights to minimize RMSE."""
    def loss(w):
        w = np.clip(w, 0, None)
        w = w / w.sum()
        return np.sqrt(np.mean((oof_preds @ w - y_true) ** 2))
    res = minimize(loss, x0, method="Nelder-Mead")
    return np.clip(res.x, 0, 1) / np.clip(res.x, 0, 1).sum()
```

**What happens if removed**: **No smart weight selection**

---

## Inference Files

### `inference/predictor.py` — Prediction Engine

**Purpose**: Loads trained model and predicts on new SMILES

**When it runs**: During inference

**Who calls it**: `chat_interface.py`, user code

**What it imports**:
- `features.graph_utils.build_multiscale`, `collate_multiscale`
- `models.polychain.PolyChain`
- `models.polychain.cst.compute_cst_batch`

**What it exports**: `PolymerPredictor` class

**Key class**:
```python
class PolymerPredictor:
    def __init__(self, checkpoint_path, device="cpu"):
        self.model = self._load_model(checkpoint_path)
    
    def predict(self, smiles_list):
        samples = [build_multiscale(s) for s in smiles_list]
        batch = collate_multiscale(samples)
        batch["cst"] = torch.tensor(compute_cst_batch(smiles))
        return self.model(batch).cpu().numpy()
```

**What happens if removed**: **Cannot make predictions**

---

## Config Files

### `config.yaml` — Global Configuration

**Purpose**: Project-wide settings

**When it runs**: Loaded at startup by `generate_all.py`, `training/train.py`

**Who uses it**: All scripts

**Key settings**:
```yaml
project:
  name: "polymer_competition"
  version: "1.0.0"

paths:
  data_dir: "data/"
  features_dir: "features/"
  models_dir: "models/"
  predictions_dir: "predictions/"
  outputs_dir: "outputs/"

seed: 42
deterministic: true

cv:
  n_folds: 5
  split_type: "group"

target:
  column: "property"
  task: "regression"

device:
  use_cuda: true
```

**What happens if removed**: **Uses default values, may break**

---

### `models/polychain/configs/finetune.yaml` — PolyChain Fine-Tuning

**Purpose**: PolyChain hyperparameters for fine-tuning

**When it runs**: During PolyChain training

**Who uses it**: `training/train.py`

**Key settings**:
```yaml
finetuning:
  task: "regression"
  target: "property"
  optimizer:
    type: "adamw"
    lr: 1.0e-4
    weight_decay: 1.0e-5
  scheduler:
    type: "cosine"
    warmup_epochs: 5
  epochs: 200
  batch_size: 32
  early_stopping:
    enabled: true
    patience: 30
```

**What happens if removed**: **Uses default hyperparameters**

---

## Test Files

### `tests/test_polychain.py` — PolyChain Tests

**Purpose**: Tests PolyChain forward pass and invariance

**When it runs**: `pytest tests/test_polychain.py`

**Who calls it**: Developer, CI/CD

**What it imports**:
- `features.graph_utils.build_multiscale`, `collate_multiscale`
- `models.polychain.PolyChain`
- `models.polychain.cst.compute_cst_batch`, `CST_DIM`

**Key tests**:
```python
def test_polychain_forward():
    """Test that PolyChain produces correct output shape."""
    batch, in_dim, edge_dim = _make_batch(SAMPLE_SMILES)
    model = PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim)
    out = model(batch)
    assert out.shape == (len(SAMPLE_SMILES),)

def test_polychain_translation_invariance():
    """Test that *CCO* and *COC* produce similar predictions."""
    out1 = model(batch1).item()
    out2 = model(batch2).item()
    assert abs(out1 - out2) < 1.0
```

**What happens if removed**: **No regression tests**

---

## Summary

| File | Purpose | When Runs | Who Calls |
|------|---------|-----------|-----------|
| `generate_all.py` | Pipeline orchestrator | User runs it | User |
| `training/train.py` | Model training | Step 3 | `generate_all.py` |
| `inference/chat_interface.py` | Web UI | User runs it | User |
| `models/polychain/polychain_model.py` | PolyChain model | Training/Inference | `train.py`, `predictor.py` |
| `features/graphs.py` | Graph construction | Feature engineering | Multiple |
| `ensemble/build_ensemble.py` | Ensemble blending | Step 4 | `generate_all.py` |
| `config.yaml` | Global config | Startup | All scripts |

---

## Key Takeaways

- Entry points are run directly by the user
- Model files define architectures
- Feature files convert SMILES to numbers
- Config files control behavior
- Test files verify correctness
- Each file has clear responsibilities and dependencies
