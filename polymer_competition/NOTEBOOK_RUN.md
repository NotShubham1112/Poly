# PolyChain Colab Run Guide

Complete instructions for running PolyChain on Google Colab.

## Prerequisites
- Google account with Google Drive enabled
- Colab access (free tier works, GPU recommended)

## Step 1: Open Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File** -> **Open notebook**
3. Click **GitHub** tab
4. Paste: `https://github.com/NotShubham1112/Poly/blob/main/notebooks/PolyChain_Colab.ipynb`
5. Click **Open**

Alternatively, upload the notebook manually:
1. Download `PolyChain_Colab.ipynb` from the repository
2. In Colab: **File** -> **Upload notebook** -> select the file

## Step 2: Enable GPU
1. Click **Runtime** -> **Change runtime type**
2. Set **Hardware accelerator** to **GPU (T4)**
3. Click **Save**

## Step 3: Run All Cells
The notebook is designed to run top-to-bottom. Each section:

### Section 1: Setup (~3 min)
- Mounts Google Drive for checkpoint persistence
- Clones the repository
- Installs all dependencies
- Verifies CUDA, PyTorch, RDKit, PyG

### Section 2: Dataset (~1 min)
- Loads or downloads polymer dataset
- Visualizes target distribution
- Generates CV splits and features

### Section 3: PolyChain Smoke Test (~5 min)
- Trains Polychain on fold 0
- Verifies checkpoint save/load
- Tests inference on sample polymers
- Copies checkpoint to Drive

### Section 4: Full 5-Fold CV (~30-60 min)
- Trains selected models across all folds
- Generates results summary
- Creates model comparison plots

### Section 5: Ablation Study (~10 min)
- Trains Backbone-only, +HAMF, +PECGN, Full variants
- Generates ablation plot

### Section 6: Reports (~2 min)
- Generates error analysis plots
- Creates model summary CSV

### Section 7: Save to Drive (~1 min)
- Syncs all results to Google Drive
- Checkpoints survive Colab disconnections

## Step 4: Resume After Disconnection
If Colab disconnects:
1. Reopen the notebook
2. Run Section 1 (Setup)
3. Run Section 8 (Resume) — this copies checkpoints back from Drive
4. Continue from where you left off

## Estimated Training Times

| Task | Time (T4 GPU) |
|------|---------------|
| Setup + install | ~3 min |
| Smoke test (1 fold) | ~5 min |
| Full 5-fold CV (all models) | ~30-60 min |
| Ablation study | ~10 min |
| Report generation | ~2 min |
| **Total** | **~50-80 min** |

## Troubleshooting

### CUDA out of memory
- Reduce batch size in config.yaml: `batch_size: 16`
- Use fewer PolyChain layers: `n_backbone_layers: 2`

### Training is slow
- Ensure GPU is enabled (Runtime -> Change runtime type)
- Reduce epochs: `--epochs 50`
- Use fewer folds: `--folds 0,1`

### Import errors
- Re-run the install cell (Section 1.3)
- Check that all dependencies installed successfully

### Checkpoint not found
- Run Section 3.1 to train a model
- Check `outputs/checkpoints/` for saved files

### Drive mount fails
- Ensure Google Drive is enabled
- Try `drive.mount('/content/drive', force_remount=True)`

## Custom Dataset

To use your own data:
1. Upload `train.csv` to `data/` in the Colab file browser
2. CSV must have columns: `id`, `SMILES`, `property`
3. Run the notebook normally

## Model Selection

In Section 4.1, modify the `MODELS` variable:
```
MODELS = "ridge,xgb,polychain"  # fast
MODELS = "ridge,rf,xgb,lgb,catboost,mlp,gcn,gat,polychain"  # all
MODELS = "polychain"  # PolyChain only
```

## Output Structure

After running all sections:
```
outputs/checkpoints/     # Model weights
predictions/             # OOF predictions
results/                 # Metrics CSVs
reports/plots/           # Generated plots
reports/                 # Error analysis
```

All of the above are automatically synced to Google Drive at:
```
/content/drive/MyDrive/PolyChain/
```
