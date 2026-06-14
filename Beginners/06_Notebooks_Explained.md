# Chapter 6: Notebooks Explained

## Introduction

This chapter explains the analysis scripts and notebooks in the project. Notebooks are like "lab notebooks" — they let you explore data interactively and see results immediately.

---

## Core Concepts

### What is a Notebook?

A **notebook** is a document that mixes:
- **Code cells**: Python code that runs when you execute the cell
- **Markdown cells**: Explanatory text (like this document)
- **Output cells**: Results of running the code (charts, tables, text)

Think of it as an interactive Python script where you can see results after each step.

### What is EDA?

**EDA** (Exploratory Data Analysis) is the process of:
- Looking at your data before building models
- Finding patterns, outliers, and errors
- Understanding distributions and relationships

**Analogy**: Before cooking a meal, you check what ingredients you have, their freshness, and quantity. EDA does the same for data.

---

## The EDA Report Script

### `notebooks/eda_report.py`

**Purpose**: Automatically generates an EDA report for the polymer competition data

**When it runs**: Step 5 of the pipeline, or standalone

**Who calls it**: `generate_all.py`, or the user directly

**What it imports**:
- `pandas` — data manipulation
- `numpy` — numerical computing
- `matplotlib` — plotting
- `features.fingerprints` — for drift analysis
- `sklearn.decomposition.PCA` — for dimensionality reduction
- `scipy.stats.wasserstein_distance` — for distribution comparison

**What it exports**: Returns a summary dictionary, saves plots to `reports/`

---

### Major Sections

#### Section 1: Target Distribution

**What it does**:
- Computes statistics: mean, std, min, max, skewness, kurtosis
- Creates a histogram and box plot
- Saves to `reports/eda_target_distribution.png`

**Example output**:
```
Target Distribution:
  count: 1000
  mean: 350.2
  std: 45.3
  min: 200.1
  25%: 320.5
  50%: 348.7
  75%: 378.2
  max: 500.3
  skewness: 0.23
  kurtosis: -0.15
  n_missing: 0
```

**What a beginner should learn**:
- How to check if data is balanced
- What skewness means (positive = tail to the right)
- What kurtosis means (negative = lighter tails than normal)

#### Section 2: Missing Values

**What it does**:
- Counts missing values in each column
- Shows percentage of missing values
- Identifies columns with high missing rates

**Example output**:
```
Missing Values:
          missing    pct
MolWt          5   0.5%
TPSA           3   0.3%
```

**What a beginner should learn**:
- Missing values can break models
- Different columns may have different missing rates
- Strategies: drop, impute (fill with mean/median), or use models that handle missing values

#### Section 3: Duplicate SMILES

**What it does**:
- Counts duplicate SMILES strings
- Checks if duplicates have different target values (high variance)

**Example output**:
```
Duplicate SMILES:
  Total SMILES:  1000
  Unique SMILES: 950
  Duplicates:    50 (5.0%)
  ⚠ 3 duplicate groups have high target variance (>p90).
```

**What a beginner should learn**:
- Duplicates can bias the model
- High variance in duplicates suggests measurement error
- Should we keep or remove duplicates? Depends on context.

#### Section 4: Train/Test Distribution Drift

**What it does**:
- Computes Morgan fingerprints for train and test sets
- Uses PCA to reduce to 2D
- Computes Wasserstein distance between distributions
- Saves scatter plot to `reports/eda_train_test_drift.png`

**Example output**:
```
Train/Test Distribution:
  SMILES overlap: 0 molecules
  FP-PCA drift (Wasserstein): PC1=0.12, PC2=0.08
```

**What a beginner should learn**:
- **Distribution drift**: When train and test data come from different distributions
- **Wasserstein distance**: Measures how different two distributions are
- **PCA**: Reduces high-dimensional data to 2D for visualization
- If drift is high, models may not generalize well

#### Section 5: Leakage Checks

**What it does**:
- Checks if ID column correlates with target (possible leakage)
- Checks if SMILES length correlates with target

**Example output**:
```
Leakage Checks:
  id ↔ target correlation: 0.02
  ✓ No obvious leakage from ID.
  SMILES length ↔ target correlation: 0.15
```

**What a beginner should learn**:
- **Leakage**: When the model accidentally sees the answer during training
- If ID correlates with target, the model might memorize IDs instead of learning patterns
- SMILES length correlation is expected (longer molecules = different properties)

---

### How to Run

```bash
# Run from command line
python notebooks/eda_report.py --config config.yaml

# Run with custom paths
python notebooks/eda_report.py --train data/train.csv --test data/test.csv

# Use in Python code
from notebooks.eda_report import run_eda
results = run_eda("data/train.csv", "data/test.csv", target_col="property")
```

---

### How to Use in Jupyter

```python
# In a Jupyter notebook cell
from notebooks.eda_report import run_eda

# Run full EDA
results = run_eda("data/train.csv", "data/test.csv", target_col="property")

# Access results
print(results["target_stats"])
print(results["drift"])
```

---

## Planned Notebooks

The `notebooks/README.md` mentions planned notebooks:

### 1. `eda_report.ipynb` (Planned)

**Purpose**: Interactive EDA with explanations

**What it would contain**:
- Markdown explanations of each analysis step
- Interactive plots that you can hover over
- Ability to modify parameters and see results

### 2. `polychain_analysis.ipynb` (Planned)

**Purpose**: Analyze PolyChain's behavior

**What it would contain**:
- Visualize multi-scale embeddings
- Compare monomer/dimer/trimer representations
- Analyze HAMF attention weights
- Study PECGN boundary operator

### 3. `ablation_plots.ipynb` (Planned)

**Purpose**: Visualize ablation study results

**What it would contain**:
- Compare model performance with/without components
- Plot RMSE vs. number of scales
- Visualize CST feature importance

---

## Related Reports

### `reports/generate_reports.py`

**Purpose**: Generates model evaluation reports

**When it runs**: Step 5 of the pipeline

**What it generates**:

1. **`reports/model_summary.csv`** — Per-model metrics
   ```csv
   model,rmse,mae,r2,n_folds,weight
   ridge,45.2,32.1,0.85,5,0.05
   xgb,38.5,27.3,0.90,5,0.15
   polychain,32.1,22.5,0.93,5,0.30
   BLEND,28.5,19.8,0.95,5,1.00
   ```

2. **`reports/error_analysis.png`** — Error analysis plots
   - Residual distribution
   - Predicted vs. actual scatter
   - Per-model RMSE bar chart
   - Top 20 worst predictions

3. **`reports/shap_summary.png`** — SHAP feature importance
   - Shows which features contribute most to predictions
   - Uses tree-based model (XGBoost) for SHAP analysis

---

## Examples

### Example: Custom EDA

```python
import pandas as pd
from notebooks.eda_report import run_eda

# Load data
train = pd.read_csv("data/train.csv")

# Quick stats
print(train.describe())

# Target distribution
import matplotlib.pyplot as plt
plt.hist(train["property"], bins=50)
plt.xlabel("Property Value")
plt.ylabel("Count")
plt.title("Target Distribution")
plt.show()

# SMILES length vs target
train["smi_len"] = train["SMILES"].str.len()
plt.scatter(train["smi_len"], train["property"], alpha=0.3)
plt.xlabel("SMILES Length")
plt.ylabel("Property")
plt.show()
```

### Example: Analyzing Errors

```python
import pandas as pd
import pickle

# Load predictions
with open("predictions/team_xgb_fold0.pkl", "rb") as f:
    data = pickle.load(f)

# Compute residuals
residuals = data["y"] - data["pred"]

# Find worst predictions
worst_idx = residuals.abs().nlargest(10).index
print("Worst predictions:")
print(train.iloc[worst_idx][["SMILES", "property"]])
print("Predictions:", data["pred"][worst_idx])
```

---

## Common Mistakes

1. **Not looking at data before training**: EDA can reveal issues before they become problems
2. **Ignoring missing values**: Models may crash or produce wrong results
3. **Not checking for leakage**: If the model sees the answer, it will score well but fail in production
4. **Over-interpreting small datasets**: With few samples, patterns may be noise

---

## Summary

- `notebooks/eda_report.py` generates automated EDA reports
- Reports include: target distribution, missing values, duplicates, drift, leakage
- `reports/generate_reports.py` generates model evaluation reports
- Planned notebooks would provide interactive analysis

---

## Key Takeaways

- EDA helps you understand your data before building models
- The EDA script checks for common issues: missing values, duplicates, drift, leakage
- Reports help you compare models and identify weaknesses
- Always run EDA before training — it saves time and prevents mistakes
