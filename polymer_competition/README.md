# PolyChain: Hierarchical Periodic Transformer for Polymer Property Prediction

A novel deep learning architecture for predicting polymer properties from SMILES strings. PolyChain introduces two novel components: **HAMF** (Hierarchy-Aware Multi-Scale Fusion) and **PECGN** (Periodic Equivariant Chain-Growth Network).

## Architecture

```
SMILES -> Multi-Scale Graphs (monomer/dimer/trimer/periodic)
       -> GIN-S Backbone (per scale)
       -> HAMF (cross-scale attention fusion)
       -> PECGN (periodic boundary injection)
       -> CST (chain statistics token)
       -> MLP Head -> Property Prediction
```

## Quick Start

### Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/PolyChain_Colab.ipynb)

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone and install
!git clone https://github.com/NotShubham1112/Poly.git
%cd Poly/polymer_competition
!pip install -r requirements.txt

# 3. Train
!python generate_all.py --config config.yaml
```

### Local Installation
```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python generate_all.py
```

## Project Structure

```
polymer_competition/
├── config.yaml                 # Global configuration
├── generate_all.py             # Master pipeline orchestrator
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── train.csv               # Training data (SMILES + target)
│   ├── test.csv                # Test data
│   ├── generate_splits.py      # CV split generation
│   └── download_polymer_data.py # Public dataset downloader
│
├── features/
│   ├── graphs.py               # Graph construction (monomer/dimer/trimer/periodic)
│   ├── graph_utils.py          # Multi-scale graph utilities
│   ├── build_features.py       # Feature matrix generation
│   ├── fingerprints.py         # Morgan/MACCS fingerprints
│   ├── descriptors.py          # RDKit descriptors
│   └── custom_polymer.py       # Polymer-specific features
│
├── models/
│   ├── polychain/              # PolyChain architecture
│   │   ├── polychain_model.py  # End-to-end model
│   │   ├── backbone.py         # GIN-S backbone
│   │   ├── hamf.py             # Hierarchy-Aware Multi-Scale Fusion
│   │   ├── pecgn.py            # Periodic Equivariant Chain-Growth Network
│   │   ├── cst.py              # Chain Statistics Token
│   │   └── configs/            # Model configs
│   ├── gnn.py                  # GCN, GAT, MPNN baselines
│   ├── mlp.py                  # MLP heads
│   ├── tree_models.py          # XGBoost, LightGBM, CatBoost, RF
│   └── baselines.py            # Ridge, Lasso
│
├── training/
│   ├── train.py                # Training entry point
│   ├── train_utils.py          # Metrics, checkpointing, seeding
│   ├── run_all_folds.py        # 5-fold CV runner
│   └── run_ablation.py         # PolyChain ablation study
│
├── inference/
│   ├── predictor.py            # PolymerPredictor class
│   └── chat_interface.py       # Streamlit chat UI
│
├── ensemble/
│   ├── build_ensemble.py       # Ensemble blend
│   └── weight_optimizer.py     # Weight optimization
│
├── reports/
│   ├── generate_reports.py     # Report generation
│   └── visualizations.py       # Matplotlib plot generation (8 plot types)
│
├── notebooks/
│   └── PolyChain_Colab.ipynb   # Google Colab notebook
│
├── tests/
│   ├── test_graphs.py          # Graph construction tests
│   ├── test_polychain.py       # PolyChain model tests
│   └── smoke_test.py           # End-to-end smoke test
│
└── docs/
    ├── architecture_overview.md
    └── polychain_whitepaper.md
```

## Model Comparison

| Model | Type | RMSE | R2 | Spearman |
|-------|------|------|----|----------|
| PolyChain | Graph (novel) | 0.554 | 0.921 | 0.960 |
| GAT | Graph | 0.695 | 0.875 | 0.937 |
| GCN | Graph | 0.736 | 0.860 | 0.929 |
| XGBoost | Tree | — | — | — |
| Ridge | Linear | — | — | — |

## Key Components

### HAMF (Hierarchy-Aware Multi-Scale Fusion)
Cross-attention mechanism that fuses monomer, dimer, and trimer scale embeddings. Each scale captures different levels of polymer structure.

### PECGN (Periodic Equivariant Chain-Growth Network)
Learned boundary operator that injects periodic boundary conditions. The closing edge connects the right connection point of the last repeat to the left connection point of the first repeat.

### CST (Chain Statistics Token)
SMILES-derived feature vector encoding polymer-specific properties: repeat unit length, branching, end groups, ring statistics, and molecular weight.

## Training Commands

```bash
# Single model, single fold
python -m training.train --model_type polychain --fold 0 --person team

# All models, all folds
python -m training.run_all_folds --models ridge,xgb,gcn,gat,polychain

# Ablation study
python -m training.run_ablation --fold 0 --epochs 50

# Resume training
python -m training.train --model_type polychain --fold 0 --person team --resume

# Generate reports
python reports/generate_reports.py --config config.yaml
```

## Visualization

Generate 8 types of publication-quality plots:
1. Training curves
2. Actual vs Predicted scatter
3. Residual plot
4. Cross-validation RMSE
5. Model comparison
6. Ablation study
7. Target distribution
8. HAMF attention heatmap

```python
from reports.visualizations import ReportGenerator
gen = ReportGenerator("reports/plots")
gen.plot_pred_vs_actual(y_true, y_pred, model_name="PolyChain")
```

## Checkpoint Format

All checkpoints contain:
```python
{
    "model_type": str,
    "epoch": int,
    "fold": int,
    "val_rmse": float,
    "config": dict,
    "model_state": dict,  # or sklearn model object
    "cst_mean": list,     # Polychain only
    "cst_std": list,      # Polychain only
}
```

## Dataset

The pipeline supports any CSV with columns: `id`, `SMILES`, `property`.

For testing, use the built-in polymer Tg dataset:
```bash
python data/download_polymer_data.py --dataset polymer_tg
```

## License

Academic use only. Part of IIT Madras Polymer Competition.
