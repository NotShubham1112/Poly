# Chapter 9: Local Setup Guide

## Introduction

This chapter explains how to set up and run the PolyChain project on your local machine. Follow these steps exactly, and you'll have the project running in minutes.

---

## Core Concepts

### What is a Virtual Environment?

A **virtual environment** is an isolated Python installation for a project. It prevents conflicts between different projects' dependencies.

**Analogy**: Like having a separate toolbox for each project — you don't want to mix your woodworking tools with your plumbing tools.

### What are Dependencies?

**Dependencies** are Python packages that your project needs to run. They're listed in `requirements.txt`.

---

## Prerequisites

### Required Software

| Software | Version | Purpose | How to Check |
|----------|---------|---------|--------------|
| Python | 3.10+ | Programming language | `python --version` |
| pip | Latest | Package installer | `pip --version` |
| Git | Latest | Version control | `git --version` |

### Optional Software

| Software | Purpose | When Needed |
|----------|---------|-------------|
| CUDA | GPU acceleration | Training deep learning models |
| Conda | Environment management | Alternative to venv |

---

## Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/NotShubham1112/Poly.git

# Navigate to the project
cd Poly/polymer_competition
```

### Step 2: Create Virtual Environment

**Using venv (recommended)**:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**Using Conda**:
```bash
# Create conda environment
conda create -n polychain python=3.10

# Activate it
conda activate polychain
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet speed.

### Step 4: Verify Installation

```bash
# Check Python version
python --version

# Check key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
```

### Step 5: Add Competition Data

```bash
# Ensure data files exist
ls data/train.csv
ls data/test.csv
```

If data files are missing, download them from the competition or use sample data:
```bash
# Download sample data (ESOL dataset)
python data/download_sample_data.py
```

---

## Running the Project

### Quick Start (Full Pipeline)

```bash
# Run the entire pipeline
python generate_all.py
```

This will:
1. Generate CV splits
2. Build feature matrix
3. Train all 11 models
4. Build ensemble
5. Generate reports

**Time**: 2-8 hours depending on hardware

### Running Specific Steps

```bash
# Step 1: Generate splits only
python -m data.generate_splits

# Step 2: Build features only
python -m features.build_features

# Step 3: Train specific models
python generate_all.py --steps 3 --models xgb,lgb

# Step 4: Build ensemble only
python -m ensemble.build_ensemble

# Step 5: Generate reports only
python reports/generate_reports.py
```

### Training a Single Model

```bash
# Train XGBoost on fold 0
python -m training.train --model_type xgb --fold 0

# Train PolyChain on fold 0
python -m training.train --model_type polychain --fold 0 --person myname
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_features.py

# Run with verbose output
pytest tests/ -v
```

### Starting the Web Interface

```bash
# Start Streamlit app
streamlit run inference/chat_interface.py
# or
streamlit run demo/app.py
```

Opens at: `http://localhost:8501`

---

## Hardware Requirements

### Minimum Requirements

| Resource | Specification |
|----------|---------------|
| CPU | 4+ cores |
| RAM | 8 GB |
| GPU | None (CPU only, slow) |
| Storage | 5 GB free space |

### Recommended Requirements

| Resource | Specification |
|----------|---------------|
| CPU | 8+ cores |
| RAM | 32 GB |
| GPU | NVIDIA GPU with 16+ GB VRAM |
| Storage | 20 GB free space |

### GPU Setup (Optional)

If you have an NVIDIA GPU:

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If True, GPU training is automatic
# If False, install CUDA toolkit from NVIDIA website
```

---

## Environment Variables

### No Environment Variables Required

The project uses `config.yaml` for configuration, not environment variables.

### Optional Environment Variables

```bash
# Set CUDA device (if you have multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Set random seed
export PYTHONHASHSEED=42
```

---

## Kaggle/Colab Setup

### Google Colab

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to project
%cd /content/drive/MyDrive/polymer_competition

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Run pipeline
!python generate_all.py
```

### Kaggle

```bash
# 1. Upload code as dataset
# 2. Copy to working directory
!cp -r /kaggle/input/polychain-source-code/* /kaggle/working/
%cd /kaggle/working/

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Run pipeline
!python generate_all.py
```

---

## Development Workflow

### 1. Make Changes

Edit files in your favorite editor (VS Code, PyCharm, etc.)

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Train Model

```bash
python -m training.train --model_type polychain --fold 0
```

### 4. Check Results

```bash
# View predictions
ls predictions/

# View reports
ls reports/
```

### 5. Iterate

Repeat steps 1-4 until satisfied.

---

## Common Setup Issues

### Issue 1: "ModuleNotFoundError: No module named 'rdkit'"

**Solution**:
```bash
pip install rdkit-pypi
```

### Issue 2: "CUDA out of memory"

**Solution**:
```bash
# Reduce batch size in config
# Or use CPU
python -m training.train --model_type polychain --fold 0
# Set device.use_cuda: false in config.yaml
```

### Issue 3: "FileNotFoundError: data/train.csv"

**Solution**:
```bash
# Download sample data
python data/download_sample_data.py

# Or place your data files in data/
```

### Issue 4: "ModuleNotFoundError: No module named 'torch_geometric'"

**Solution**:
```bash
pip install torch-geometric
# Or for specific CUDA version:
pip install torch-geometric -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html
```

### Issue 5: "PermissionError: [Errno 13] Permission denied"

**Solution**:
```bash
# On Windows, run as administrator
# On Mac/Linux, use sudo (not recommended)
# Better: fix file permissions
chmod +x venv/bin/activate
```

---

## Examples

### Example: Complete Setup from Scratch

```bash
# 1. Clone repo
git clone https://github.com/NotShubham1112/Poly.git
cd Poly/polymer_competition

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download sample data
python data/download_sample_data.py

# 5. Run quick test
python generate_all.py --steps 1,2 --models ridge

# 6. Check results
ls predictions/
```

### Example: Training Only PolyChain

```bash
# 1. Generate splits
python -m data.generate_splits

# 2. Build features
python -m features.build_features

# 3. Train PolyChain on fold 0
python -m training.train --model_type polychain --fold 0 --person test

# 4. Check prediction
ls predictions/test_polychain_fold0.pkl
```

---

## Summary

- Install Python 3.10+, create virtual environment, install dependencies
- Add competition data to `data/`
- Run `python generate_all.py` for full pipeline
- Use `--steps` and `--models` flags for partial runs
- GPU is optional but recommended for deep learning models

---

## Key Takeaways

- Always use a virtual environment
- Install dependencies with `pip install -r requirements.txt`
- Run from `polymer_competition/` directory
- GPU is optional but speeds up training significantly
- Check `requirements.txt` for all dependencies
