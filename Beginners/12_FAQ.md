# Chapter 12: Frequently Asked Questions

## Introduction

This chapter answers common questions about the PolyChain project. If you have a question, check here first — someone probably had the same question before!

---

## General Questions

### Q: What is PolyChain?

**A**: PolyChain is a machine learning project that predicts polymer properties (like Tg, Tm, density) from SMILES strings. It uses 11 different models, with a novel "PolyChain" architecture as the centerpiece.

---

### Q: Why are there 11 models?

**A**: Each model has different strengths:
- **Tree models** (XGBoost, LightGBM, CatBoost) are fast and handle tabular data well
- **GNN models** (GCN, GAT, MPNN) understand molecular structure
- **PolyChain** is the novel architecture that combines multi-scale graph reasoning

By combining all models (ensembling), we get better predictions than any single model.

---

### Q: What is the novel contribution?

**A**: PolyChain introduces two innovations:
1. **HAMF** (Hierarchy-Aware Multi-Scale Fusion): Fuses information from monomer, dimer, and trimer scales using cross-attention
2. **PECGN** (Periodic Equivariant Chain-Growth Network): Adds learned periodic boundary operator that is invariant to where you "cut" the polymer chain

---

### Q: How accurate is PolyChain?

**A**: Expected accuracy varies by property:
- **Tg**: R² ≈ 0.93-0.95 (very good)
- **Density**: R² ≈ 0.86-0.89 (good)
- **Tm**: R² ≈ 0.70-0.76 (moderate)

The largest gains are on properties where chain-scale phenomena matter most (Tm, density).

---

## Setup Questions

### Q: What Python version do I need?

**A**: Python 3.10 or higher. Check with `python --version`.

---

### Q: Do I need a GPU?

**A**: No, but it's recommended. CPU training works but is much slower (hours vs. minutes). GPU is especially important for PolyChain and GNN models.

---

### Q: How do I install RDKit?

**A**: Run `pip install rdkit-pypi`. If that fails, try `conda install -c conda-forge rdkit`.

---

### Q: Why is installation taking so long?

**A**: Some packages (PyTorch, RDKit) are large and take time to compile. Be patient — it should finish in 5-10 minutes.

---

### Q: Can I run this on Google Colab?

**A**: Yes! See Chapter 9 (Local Setup Guide) for Colab instructions. Free GPUs are available.

---

## Data Questions

### Q: Where do I get the data?

**A**: The competition data (train.csv, test.csv) should be placed in the `data/` folder. For testing, you can use sample data:
```bash
python data/download_sample_data.py
```

---

### Q: What format should the data be in?

**A**: CSV files with columns:
- `id`: Unique identifier
- `SMILES`: Polymer SMILES string with `*` connection points
- `property`: Target value (only in train.csv)

Example:
```csv
id,SMILES,property
0,*CCO*,350.2
1,*c1ccc(*)cc1*,420.5
```

---

### Q: What does `*` mean in SMILES?

**A**: `*` represents connection points in the polymer chain. For example, `*CCO*` means the repeat unit is "CCO" and connects to adjacent units at both ends.

---

### Q: How many samples do I need?

**A**: More is better. The project works with 100+ samples, but 1000+ is recommended for good performance.

---

## Training Questions

### Q: How long does training take?

**A**: Depends on hardware and model:
- **Tree models** (XGBoost, etc.): 1-5 minutes per fold
- **GNN models** (GCN, etc.): 10-30 minutes per fold
- **PolyChain**: 30-60 minutes per fold
- **Full pipeline** (all 11 models, 5 folds): 2-8 hours

---

### Q: Can I train only specific models?

**A**: Yes! Use the `--models` flag:
```bash
python generate_all.py --steps 3 --models xgb,lgb,polychain
```

---

### Q: Can I train only specific folds?

**A**: Yes! Use the `--fold` flag:
```bash
python -m training.train --model_type polychain --fold 0
```

---

### Q: How do I know which model is best?

**A**: Check `reports/model_summary.csv` after training. It contains RMSE, MAE, and R² for each model.

---

### Q: What is cross-validation?

**A**: Cross-validation splits the data into multiple "folds" (e.g., 5). Each fold is used once for validation while the others are used for training. This gives a more reliable estimate of model performance.

---

## Ensemble Questions

### Q: What is an ensemble?

**A**: An ensemble combines predictions from multiple models. Instead of relying on one model, we average predictions from all models, weighted by their performance.

---

### Q: How are weights determined?

**A**: Three strategies are available:
1. **Inverse RMSE**: Weight ∝ 1/RMSE (better models get higher weight)
2. **Nelder-Mead**: Optimization to minimize RMSE
3. **Stacking**: Ridge regression meta-learner

---

### Q: Can I add my own model to the ensemble?

**A**: Yes! See Chapter 11 (Common Modifications) for instructions on adding a new model.

---

## PolyChain Questions

### Q: What is HAMF?

**A**: HAMF (Hierarchy-Aware Multi-Scale Fusion) is a module that fuses information from monomer, dimer, and trimer scales using cross-attention. It treats the three scale embeddings as a sequence and applies chain-structured transformer.

---

### Q: What is PECGN?

**A**: PECGN (Periodic Equivariant Chain-Growth Network) adds a learned periodic boundary operator. It makes predictions invariant to where you "cut" the polymer chain.

---

### Q: What is CST?

**A**: CST (Chain Statistics Token) is a fixed-dimensional vector computed from SMILES alone. It contains polymer-specific features like repeat length, branching, end-groups, and ring statistics.

---

### Q: Why use multi-scale graphs?

**A**: Different scales capture different information:
- **Monomer**: Local chemical environment
- **Dimer**: How repeat units interact
- **Trimer**: Longer-range patterns
- **Periodic**: Infinite chain behavior

Fusing all scales gives a more complete picture.

---

### Q: What is asterisk-mask reconstruction?

**A**: A self-supervised pretraining task where `*` connection points are randomly masked, and the model must predict their type and neighbor atom. This teaches the model about polymer periodicity.

---

## Inference Questions

### Q: How do I make predictions?

**A**: Use the `PolymerPredictor` class:
```python
from inference.predictor import PolymerPredictor

pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")
result = pred.predict(["*CCO*", "*c1ccc(*)cc1*"])
```

---

### Q: How do I use the web interface?

**A**: Run:
```bash
streamlit run inference/chat_interface.py
```
Then open `http://localhost:8501` in your browser.

---

### Q: Can I use the web interface without training?

**A**: No, you need a trained model checkpoint. Train first with:
```bash
python -m training.train --model_type polychain --fold 0
```

---

## Troubleshooting Questions

### Q: I get "ModuleNotFoundError" — what do I do?

**A**: Install the missing package:
```bash
pip install <package_name>
```

---

### Q: I get "CUDA out of memory" — what do I do?

**A**: Reduce batch size in config or use CPU:
```yaml
# config.yaml
device:
  use_cuda: false
```

---

### Q: I get "FileNotFoundError" — what do I do?

**A**: Check that:
1. You're running from `polymer_competition/`
2. Data files exist in `data/`
3. Config paths are correct

---

### Q: Training is too slow — what do I do?

**A**: Options:
1. Use GPU if available
2. Reduce number of epochs
3. Reduce model size (hidden_dim, n_layers)
4. Train fewer models

---

### Q: Predictions are bad — what do I do?

**A**: Check:
1. Data quality (missing values, outliers)
2. Feature engineering (are features informative?)
3. Model configuration (hyperparameters)
4. Ensemble strategy (are weights optimal?)

---

## Advanced Questions

### Q: How do I add a new property to predict?

**A**: Change `target.column` in `config.yaml`:
```yaml
target:
  column: "Tm"  # Or any column in your data
```

---

### Q: How do I use transfer learning?

**A**: Use a pre-trained checkpoint:
```python
# Load pre-trained model
ckpt = torch.load("pretrained_checkpoint.pt")

# Fine-tune on your data
model.load_state_dict(ckpt["model_state"])
```

---

### Q: How do I deploy this to production?

**A**: Options:
1. **Streamlit app**: For internal use
2. **REST API**: Wrap `PolymerPredictor` in FastAPI/Flask
3. **Docker**: Containerize the application

---

### Q: Can I use this for copolymers?

**A**: Yes, but with limitations. The CST includes `n_monomers_copolymer` for copolymer support. However, sequence modeling is not implemented (only set-pooling of monomers).

---

## Still Have Questions?

If your question isn't answered here:
1. Check the error message carefully (Chapter 10)
2. Search the codebase for relevant keywords
3. Open an issue on GitHub
4. Ask the development team

---

## Key Takeaways

- Most questions have simple answers: check data, check paths, check dependencies
- Training time depends on hardware and model choice
- Ensemble improves predictions by combining multiple models
- PolyChain's innovations (HAMF, PECGN, CST) are the novel contributions
- Always check `reports/model_summary.csv` for model performance
