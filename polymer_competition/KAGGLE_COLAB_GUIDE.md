# PolyChain: Kaggle, Colab, and Local Training Guide

This guide explains how to take this codebase, set it up in a cloud notebook environment (Kaggle or Google Colab), and train your models.

---

## 1. Google Colab Setup

Google Colab is one of the easiest ways to get a free GPU to train PolyChain. 

### Option A: Via Google Drive (Recommended)
This method ensures your checkpoints and predictions are saved even if the Colab runtime disconnects.

1. Zip your local `polymer_competition` folder and upload it to your Google Drive.
2. Open a new Google Colab notebook.
3. Mount your Google Drive by running this cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Unzip the folder to your Colab working directory or navigate directly to it:
   ```bash
   %cd /content/drive/MyDrive/polymer_competition
   ```
5. Install the required dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
6. Run the master pipeline:
   ```bash
   !python generate_all.py --config config.yaml
   ```

### Option B: Via GitHub Clone
1. In Colab, clone the repository. If your repository is **private**, you will need to use a GitHub Personal Access Token (PAT):
   ```bash
   # If the repository is PUBLIC:
   !git clone https://github.com/NotShubham1112/Poly.git
   
   # If the repository is PRIVATE, use your username and PAT instead:
   # !git clone https://<USERNAME>:<YOUR_PAT>@github.com/NotShubham1112/Poly.git
   
   %cd Poly/polymer_competition
   ```
2. **IMPORTANT: Add Competition Data**
   Because competition data is private/large, it is NOT stored in the GitHub repository. You must upload `train.csv` and `test.csv` into the `data/` directory. You can do this by dragging and dropping them into the Colab file explorer under `Poly/polymer_competition/data/`.

3. Install requirements and train:
   ```bash
   !pip install -r requirements.txt
   !python generate_all.py
   ```
   *(Note: Remember to download your `outputs/` and `predictions/` folders before the runtime is recycled!)*

---

## 2. Kaggle Notebook Setup

Kaggle notebooks are structured differently. You cannot easily push code via Git, so the standard approach is to upload your codebase as a **Kaggle Dataset**.

### Step 1: Create a Kaggle Dataset
1. Zip this entire `polymer_competition` folder (exclude `.git`, `__pycache__`, and large virtual environment folders).
2. Go to Kaggle -> **Datasets** -> **New Dataset**.
3. Upload the zip file and name it something like `polychain-source-code`.
4. Click **Create**.

### Step 2: Attach to your Notebook
1. Open your competition Notebook on Kaggle.
2. Click **Add Data** (top right) -> **Your Datasets** -> select `polychain-source-code`.
3. The dataset will be mounted at `/kaggle/input/polychain-source-code/`.

### Step 3: Copy Code to Working Directory
Kaggle inputs are read-only. To allow the code to save checkpoints and predictions, copy it to the writable `/kaggle/working/` directory:

```bash
# In a Kaggle Notebook cell
!cp -r /kaggle/input/polychain-source-code/* /kaggle/working/
%cd /kaggle/working/
```

### Step 4: Install Dependencies & Train
Kaggle already has PyTorch and XGBoost installed, but you need to install RDKit and PyTorch Geometric:

```bash
!pip install -r requirements.txt
```

Train the models:
```bash
!python generate_all.py --config config.yaml
```

Once the run completes, the `submission.csv` will be generated in `/kaggle/working/`, which you can directly submit to the competition!

---

## 3. Training Locally (Your PC)

If you have a local GPU (or just want to test the baseline models on your CPU), you can run the code directly.

### Step 1: Set up a Virtual Environment
It is highly recommended to use a virtual environment (like `venv` or `conda`):

**Using Conda (Recommended for PyTorch):**
```bash
conda create -n polychain python=3.10
conda activate polychain
```

**Using venv:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Pipeline
You can run the entire orchestrator script to automatically process data, train all 11 model types, and build the final ensemble:

```bash
python generate_all.py
```

### Step 4: Run Specific Steps (Optional)
If you only want to train a specific model (e.g., PolyChain) on fold 0 for debugging:

```bash
# 1. Generate cross-validation splits
python -m data.generate_splits

# 2. Extract features/graphs
python -m features.build_features

# 3. Train only PolyChain on Fold 0
python -m training.train --model_type polychain --fold 0 --person local_test
```

### Step 5: Start the Chat UI Demo
To interact with your trained model naturally, run the Streamlit app:

```bash
streamlit run demo/app.py
```
This will open a browser window at `localhost:8501` where you can type polymer SMILES strings and get predictions!
