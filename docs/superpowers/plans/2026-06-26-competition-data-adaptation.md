# Competition Data Adaptation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt the existing polymer property prediction pipeline from the 903-row placeholder (single `property` column) to the official 6171-row competition dataset with two targets (Tg, Egc).

**Architecture:** Shared feature extraction on all SMILES → split by `target_type` → independent per-target CV/training/ensemble → merged submission. One feature cache per dataset (train/test) keyed by canonical SMILES.

**Tech Stack:** Python 3.10+, RDKit 2024.03+, PyTorch 2.1+, scikit-learn 1.4+, XGBoost/CatBoost/LightGBM

## Global Constraints

- All random seeds defined centrally in `config.yaml` (global=42, numpy=42, torch=42, xgboost=42, catboost=42, lightgbm=42)
- Column name normalization: `smiles` (lowercase input) → `SMILES` (uppercase internal) at load time
- SMILES canonicalization via RDKit before caching; invalid SMILES logged with configurable fail-policy
- Feature cache stores one vector per unique canonical SMILES; duplicates reference via lookup
- All submission artifacts must comply with competition `id,target` format
- Pipeline must be runnable in a Kaggle notebook (single execution, no external data, no pretrained weights)

---
### Task 1: Config & Data Foundation

**Files:**
- Modify: `polymer_competition/config.yaml`
- Create: `polymer_competition/data/split_by_target.py`
- Create: `polymer_competition/data/__init__.py`
- Test: `polymer_competition/tests/test_split_by_target.py`

**Interfaces:**
- Consumes: None (first task)
- Produces:
  - `config.yaml` — dual target structure with seeds and experiment version
  - `split_by_target(train_csv, test_csv, output_dir)` → `None` (writes per-target CSVs)
  - `data/__init__.py` — empty, makes `data` a package

- [ ] **Step 1: Write failing tests for split_by_target**

```python
"""tests/test_split_by_target.py"""
import tempfile
from pathlib import Path
import pandas as pd
from data.split_by_target import split_by_target


def test_split_by_target_creates_four_files():
    train = pd.DataFrame({
        "smiles": ["CCO", "CCC"],
        "target": [100.0, 5.0],
        "target_type": ["tg", "egc"],
    })
    test = pd.DataFrame({
        "id": [1, 2],
        "smiles": ["CCO", "CCC"],
        "target_type": ["tg", "egc"],
    })
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.csv"
        test_path = Path(tmp) / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        split_by_target(train_path, test_path, Path(tmp))
        assert (Path(tmp) / "tg" / "train.csv").exists()
        assert (Path(tmp) / "tg" / "test.csv").exists()
        assert (Path(tmp) / "egc" / "train.csv").exists()
        assert (Path(tmp) / "egc" / "test.csv").exists()
        tg_train = pd.read_csv(Path(tmp) / "tg" / "train.csv")
        assert len(tg_train) == 1
        assert tg_train.iloc[0]["target"] == 100.0


def test_split_by_target_preserves_id():
    train = pd.DataFrame({
        "smiles": ["CCO", "CCC"],
        "target": [100.0, 5.0],
        "target_type": ["tg", "egc"],
    })
    test = pd.DataFrame({
        "id": [10, 20],
        "smiles": ["CCO", "CCC"],
        "target_type": ["tg", "egc"],
    })
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.csv"
        test_path = Path(tmp) / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        split_by_target(train_path, test_path, Path(tmp))
        tg_test = pd.read_csv(Path(tmp) / "tg" / "test.csv")
        assert tg_test.iloc[0]["id"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd polymer_competition && python -m pytest tests/test_split_by_target.py -v`
Expected: ImportError for `split_by_target`

- [ ] **Step 3: Write split_by_target.py**

```python
"""data/split_by_target.py

Split train/test CSVs by target_type into per-target subdirectories.

Usage:
    python -m data.split_by_target --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def split_by_target(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    targets: list[str] | None = None,
) -> None:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    available = train["target_type"].unique().tolist()
    if targets is None:
        targets = available
    for t in targets:
        t_dir = Path(output_dir) / t
        t_dir.mkdir(parents=True, exist_ok=True)
        train_subset = train[train["target_type"] == t].copy()
        test_subset = test[test["target_type"] == t].copy()
        train_subset.to_csv(t_dir / "train.csv", index=False)
        test_subset.to_csv(t_dir / "test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--targets", default=None, help="Comma-separated target types")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    targets = args.targets.split(",") if args.targets else None
    split_by_target(
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir,
        targets=targets,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd polymer_competition && python -m pytest tests/test_split_by_target.py -v`
Expected: 2 passed

- [ ] **Step 5: Create __init__.py**

```python
# data/__init__.py
```

- [ ] **Step 6: Update config.yaml**

New config structure:
```yaml
project:
  name: "polymer_competition"
  version: "1.1.0"

paths:
  data_dir: "data/"
  features_dir: "features/"
  models_dir: "models/"
  training_dir: "training/"
  predictions_dir: "predictions/"
  outputs_dir: "outputs/"
  checkpoints_dir: "outputs/checkpoints/"
  logs_dir: "outputs/logs/"
  submissions_dir: "outputs/submissions/"
  reports_dir: "reports/"

seed:
  global: 42
  numpy: 42
  torch: 42
  python: 42
  xgboost: 42
  catboost: 42
  lightgbm: 42

deterministic: true

data:
  smiles_col: "smiles"
  id_col: "id"
  target_col: "target"
  target_type: "target_type"

targets:
  tg:
    type: "tg"
  egc:
    type: "egc"

cv:
  n_folds: 5
  split_type: "group"
  group_strategy: "scaffold"

device:
  use_cuda: true
  gpu_id: 0

experiment:
  version: "v1"

ensemble:
  strategy: "inverse_rmse"
  min_weight: 0.0
  max_weight: 1.0

logging:
  level: "INFO"
  tensorboard: false
  csv_log: true
```

- [ ] **Step 7: Copy competition data into pipeline**

```bash
Copy-Item "D:\Parth\Poly\kaggle data\train.csv" "D:\Parth\Poly\polymer_competition\data\train.csv" -Force
Copy-Item "D:\Parth\Poly\kaggle data\test.csv" "D:\Parth\Poly\polymer_competition\data\test.csv" -Force
```

- [ ] **Step 8: Run split_by_target to verify it works on real data**

Run: `cd polymer_competition && python -m data.split_by_target`
Expected: creates `data/tg/train.csv`, `data/tg/test.csv`, `data/egc/train.csv`, `data/egc/test.csv`

- [ ] **Step 9: Commit**

```bash
git add polymer_competition/config.yaml polymer_competition/data/split_by_target.py polymer_competition/data/__init__.py polymer_competition/tests/test_split_by_target.py polymer_competition/data/train.csv polymer_competition/data/test.csv
git commit -m "feat: add dual-target config and split_by_target script"
```

---
### Task 2: Feature Pipeline with Cache

**Files:**
- Modify: `polymer_competition/features/build_features.py`
- Test: `polymer_competition/tests/test_features.py` (extend existing)

**Interfaces:**
- Consumes:
  - `config.yaml` — data/smiles_col, targets, experiment/version
  - `data/train.csv` and `data/test.csv` (original, unsplit)
- Produces:
  - `data/processed/features_train.parquet` — train feature cache
  - `data/processed/features_test.parquet` — test feature cache
  - `data/processed/metadata.yaml` — cache metadata (version, commit, rdkit version)

- [ ] **Step 1: Write failing test for feature cache versioning**

Add to `tests/test_features.py`:
```python
def test_feature_cache_metadata():
    """After building features, metadata file exists with version info."""
    from pathlib import Path
    import yaml
    ROOT = Path(__file__).resolve().parent.parent
    meta_path = ROOT / "data" / "processed" / "metadata.yaml"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert "feature_version" in meta
    assert "git_commit" in meta
    assert "rdkit_version" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_features.py::test_feature_cache_metadata -v`
Expected: FAIL

- [ ] **Step 3: Rewrite build_features.py**

```python
"""features/build_features.py

Phased feature pipeline:

1. Load train.csv + test.csv, normalize column names (smiles -> SMILES)
2. Canonicalize SMILES via RDKit; report failures
3. Build feature cache (fingerprints + descriptors + custom) separated by dataset
4. Deduplicate: one feature vector per unique canonical SMILES
5. Split train cache by target_type for downstream consumption
6. Save splits.pkl per target
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit import __version__ as rdkit_version
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from .fingerprints import all_fingerprints
from .descriptors import compute_descriptors, select_descriptors_by_variance
from .custom_polymer import compute_all_custom_features


def canonicalize(smiles_list: list[str]) -> list[str | None]:
    """Canonicalize via RDKit; return None for invalid."""
    results = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            results.append(None)
        else:
            results.append(Chem.MolToSmiles(mol, canonical=True))
    return results


def load_and_normalize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        if c.lower() == "smiles":
            col_map[c] = "SMILES"
        elif c.lower() == "id":
            col_map[c] = "id"
    df = df.rename(columns=col_map)
    return df


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_config_hash(cfg: dict) -> str:
    raw = yaml.dump(cfg, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_features(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles_col = cfg.get("data", {}).get("smiles_col", "smiles")

    # Load and normalize
    train = load_and_normalize(data_dir / "train.csv")
    test = load_and_normalize(data_dir / "test.csv")
    print(f"Loaded train={len(train)} test={len(test)}")

    # Canonicalize
    all_smiles = train["SMILES"].tolist() + test["SMILES"].tolist()
    canon = canonicalize(all_smiles)
    n_invalid = sum(1 for c in canon if c is None)
    if n_invalid:
        print(f"WARNING: {n_invalid} SMILES failed canonicalization")
    train["canon_smiles"] = canon[: len(train)]
    test["canon_smiles"] = canon[len(train):]

    # Build features on deduplicated canonical SMILES
    unique_smiles = list(set(s for s in canon if s is not None))
    print(f"Building features on {len(unique_smiles)} unique canonical SMILES")
    fps = all_fingerprints(unique_smiles)
    desc = compute_descriptors(unique_smiles)
    desc = select_descriptors_by_variance(desc)
    cust = compute_all_custom_features(unique_smiles)

    fp_dfs = {}
    for name, arr in fps.items():
        cols = [f"{name}_{i}" for i in range(arr.shape[1])]
        fp_dfs[name] = pd.DataFrame(arr, columns=cols)
    desc_df = desc.drop(columns=["SMILES"], errors="ignore")
    cust_df = cust.drop(columns=["SMILES"], errors="ignore")

    cache_df = pd.concat(
        [pd.DataFrame({"canon_smiles": unique_smiles})]
        + list(fp_dfs.values())
        + [desc_df.reset_index(drop=True)]
        + [cust_df.reset_index(drop=True)],
        axis=1,
    )

    # Imputation
    num_cols = [c for c in cache_df.columns if c != "canon_smiles" and cache_df[c].dtype != object]
    imputer = SimpleImputer(strategy="median")
    cache_df[num_cols] = imputer.fit_transform(cache_df[num_cols])

    # Merge back to original rows via lookup
    train_idx_map = train["canon_smiles"].values
    test_idx_map = test["canon_smiles"].values
    canon_to_idx = {s: i for i, s in enumerate(cache_df["canon_smiles"].values)}

    def lookup_features(smiles_list, id_vals, id_col="id"):
        rows = []
        for smi, id_val in zip(smiles_list, id_vals):
            if smi is None or smi not in canon_to_idx:
                continue
            idx = canon_to_idx[smi]
            row = cache_df.iloc[idx].to_dict()
            row["SMILES"] = smi
            row[id_col] = id_val
            rows.append(row)
        return pd.DataFrame(rows)

    train_feat = lookup_features(
        train["canon_smiles"].values,
        train["id"].values if "id" in train.columns else range(len(train)),
    )
    test_feat = lookup_features(
        test["canon_smiles"].values,
        test["id"].values,
    )

    train_feat.to_parquet(out_dir / "features_train.parquet", index=False)
    test_feat.to_parquet(out_dir / "features_test.parquet", index=False)
    print(f"Train features: {train_feat.shape}, Test features: {test_feat.shape}")

    # Save metadata
    meta = {
        "feature_version": cfg.get("experiment", {}).get("version", "v1"),
        "git_commit": get_git_commit(),
        "config_hash": get_config_hash(cfg),
        "rdkit_version": rdkit_version,
        "n_unique_smiles": len(unique_smiles),
        "n_invalid_smiles": n_invalid,
        "n_train_rows": len(train_feat),
        "n_test_rows": len(test_feat),
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)
    print(f"Cache metadata -> {out_dir / 'metadata.yaml'}")

    # Per-target splits
    from data.split_by_target import split_by_target as _split_ds
    _split_ds(data_dir / "train.csv", data_dir / "test.csv", data_dir,
              targets=list(cfg["targets"].keys()))

    # CV splits per target
    for t_name, t_cfg in cfg["targets"].items():
        t_dir = data_dir / t_name
        t_train = pd.read_csv(t_dir / "train.csv")
        t_train = t_train.rename(columns={smiles_col: "SMILES"})
        scaffolds = t_train["SMILES"].apply(_smiles_scaffold).values
        gkf = GroupKFold(n_splits=cfg["cv"]["n_folds"])
        splits = {}
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(t_train, groups=scaffolds)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
        with open(data_dir / f"splits_{t_name}.pkl", "wb") as f:
            pickle.dump(splits, f)
        print(f"splits_{t_name}.pkl: {len(splits)} folds")


def _smiles_scaffold(s: str) -> str:
    s = s.replace("*", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return s[:20]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    build_features(args.config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd polymer_competition && python -m pytest tests/test_features.py -v`
Expected: all pass (including new metadata test)

- [ ] **Step 5: Smoke-test build_features on real data**

Run: `cd polymer_competition && python -m features.build_features`
Expected: train/test feature parquets created, metadata written, splits_{tg,egc}.pkl created

- [ ] **Step 6: Commit**

```bash
git add polymer_competition/features/build_features.py polymer_competition/tests/test_features.py polymer_competition/data/processed/
git commit -m "feat: feature pipeline with canonicalization, deduped cache, per-target splits"
```

---
### Task 3: Per-Target CV Splits

**Files:**
- Modify: `polymer_competition/data/generate_splits.py`

**Interfaces:**
- Consumes:
  - `config.yaml` — targets, cv/n_folds, seed/global, data/smiles_col
  - Per-target train.csv from `data/<target>/train.csv`
- Produces: `data/splits_{target}.pkl` (already handled in Task 2 Step 5, but make standalone script too)

- [ ] **Step 1: Write failing test for generate_splits with target param**

```python
def test_generate_splits_with_target():
    """generate_splits accepts target param and writes target-specific splits."""
    import tempfile
    import pandas as pd
    from data.generate_splits import generate_splits
    with tempfile.TemporaryDirectory() as tmp:
        train = pd.DataFrame({
            "SMILES": ["CCO", "CCC", "C=O", "CCO", "CCC", "C=O"],
            "target": [100, 5, 200, 105, 6, 195],
        })
        train.to_csv(Path(tmp) / "tg" / "train.csv", index=False)
        splits = generate_splits(
            Path(tmp) / "tg" / "train.csv",
            Path(tmp) / "splits_tg.pkl",
            n_folds=2,
            smiles_col="SMILES",
            target_col="target",
        )
        assert len(splits) == 2
        for fold_id, idx in splits.items():
            assert "train" in idx and "val" in idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_generate_splits.py -v`
Expected: FAIL (function signature doesn't match)

- [ ] **Step 3: Update generate_splits.py**

Update `generate_splits()` to accept `smiles_col` parameter and `target` name:

```python
def generate_splits(
    train_path: str | Path,
    output_path: str | Path,
    n_folds: int = 5,
    seed: int = 42,
    target_col: str = "target",
    smiles_col: str = "SMILES",
    strategy: str = "group",
) -> dict:
    train = pd.read_csv(train_path)
    # Normalize column names if needed
    if smiles_col not in train.columns:
        for c in train.columns:
            if c.lower() == smiles_col.lower():
                smiles_col = c
                break
    print(f"Loaded {len(train)} rows from {train_path}")
    splits = {}
    if strategy == "group":
        from sklearn.model_selection import GroupKFold
        scaffolds = train[smiles_col].apply(_smiles_scaffold).values
        gkf = GroupKFold(n_splits=n_folds)
        y = train[target_col].values if target_col in train.columns else np.zeros(len(train))
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups=scaffolds)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {n_folds}-fold splits -> {output_path}")
    for fold_id, idx in splits.items():
        print(f"  Fold {fold_id}: train={len(idx['train'])}, val={len(idx['val'])}")
    return splits
```

Update `main()` to iterate over targets:
```python
def main():
    parser = argparse.ArgumentParser(description="Generate CV splits")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strategy", default=None, choices=["group", "random"])
    parser.add_argument("--target", default=None, help="Target name (tg/egc). If omitted, runs for all.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    n_folds = cfg.get("cv", {}).get("n_folds", 5)
    seed = cfg.get("seed", {}).get("global", 42)
    strategy = args.strategy or cfg.get("cv", {}).get("split_type", "group")

    targets = [args.target] if args.target else list(cfg.get("targets", {"tg": {}}).keys())
    for t in targets:
        train_path = data_dir / t / "train.csv"
        output_path = data_dir / f"splits_{t}.pkl"
        if not train_path.exists():
            print(f"  Skipping {t}: {train_path} not found")
            continue
        generate_splits(
            train_path, output_path,
            n_folds=n_folds, seed=seed,
            target_col=cfg["data"]["target_col"],
            smiles_col=cfg["data"].get("smiles_col", "smiles"),
            strategy=strategy,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd polymer_competition && python -m pytest tests/test_generate_splits.py -v`
Expected: PASS

- [ ] **Step 5: Validate on real data**

Run: `cd polymer_competition && python -m data.generate_splits`
Expected: creates/confirms `data/splits_tg.pkl` and `data/splits_egc.pkl`

- [ ] **Step 6: Commit**

```bash
git add polymer_competition/data/generate_splits.py
git commit -m "feat: generate_splits supports per-target operation and configurable smiles_col"
```

---
### Task 4: Per-Target Training

**Files:**
- Modify: `polymer_competition/training/train.py`

**Interfaces:**
- Consumes:
  - `--target` argument (tg/egc)
  - Per-target feature parquet + splits_{target}.pkl
  - `config.yaml` — data/smiles_col, data/target_col
- Produces: `predictions/v1_{target}_{model}_fold{fold}.pkl`

- [ ] **Step 1: Write failing test for training with target**

```python
def test_train_accepts_target_arg():
    """train.py --target tg should load tg features and splits."""
    import subprocess, sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "training.train",
         "--model_type", "ridge", "--fold", "0",
         "--target", "tg", "--max_samples", "50"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    assert "tg" in result.stdout.lower() or "Fold 0" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd polymer_competition && python -m pytest tests/test_train_target.py -v`
Expected: FAIL (train.py doesn't accept --target yet)

- [ ] **Step 3: Update train.py with target support**

Key changes to `training/train.py`:

Add `--target` argument to `main()` parser:
```python
parser.add_argument("--target", default=None,
                    help="Target name (tg/egc). Loads target-specific features and splits.")
```

After parsing config, resolve target-specific paths:
```python
target = args.target or list(cfg.get("targets", {"tg": {}}).keys())[0]
target_cfg = cfg["targets"].get(target, {})

data_dir = Path(cfg["paths"]["data_dir"])

if args.target:
    # Load per-target features
    train = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
    # Filter by target type
    target_train_csv = data_dir / target / "train.csv"
    target_ids = set(pd.read_csv(target_train_csv)["id"].values)
    train = train[train["id"].isin(target_ids)].reset_index(drop=True)
    splits_path = data_dir / f"splits_{target}.pkl"
else:
    train = pd.read_parquet(data_dir / "processed" / "train_features.parquet")
    splits_path = data_dir / "splits.pkl"

target_col = cfg["data"]["target_col"]
feature_cols = [c for c in train.columns
                if c not in ("SMILES", "id", "canon_smiles", target_col)]
```

Update output filenames to include target:
```python
person = args.person or "anon"
ckpt_tag = f"{cfg['experiment']['version']}_{target}_{args.model_type}_fold{args.fold}"
```

For the target column in data loading:
```python
y_tr = tr_df[target_col].values
y_va = va_df[target_col].values
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd polymer_competition && python -m pytest tests/test_train_target.py -v`
Expected: PASS (trains ridge on 50 samples of tg)

- [ ] **Step 5: Validate on real data**

Run: `cd polymer_competition && python -m training.train --model_type ridge --fold 0 --target tg --max_samples 100`
Expected: Training completes, prediction file saved to `predictions/v1_tg_ridge_fold0.pkl`

- [ ] **Step 6: Commit**

```bash
git add polymer_competition/training/train.py
git commit -m "feat: train.py supports --target argument for per-target training"
```

---
### Task 4b: Per-Target Test Inference

**Files:**
- Modify: `polymer_competition/training/train.py` (add test inference after training)

**Interfaces:**
- Consumes: trained model checkpoint + test features
- Produces: `predictions/v1_{target}_{model}_fold{fold}_test.pkl`

- [ ] **Step 1: Add inference to train.py**

After the training and OOF prediction saving in `train.py`, add test inference:

```python
# Test inference (only if test features exist and --skip-inference not set)
if not getattr(args, "skip_inference", False):
    test_feat = pd.read_parquet(data_dir / "processed" / "features_test.parquet")
    # Filter to this target's test IDs
    target_test_csv = data_dir / target / "test.csv"
    test_ids = pd.read_csv(target_test_csv)["id"].tolist()
    test_feat = test_feat[test_feat["id"].isin(test_ids)].reset_index(drop=True)
    test_feat_cols = [c for c in test_feat.columns if c in feature_cols]
    X_test = test_feat[test_feat_cols].values

    if args.model_type in ("ridge", "xgb", "lgb", "catboost", "rf"):
        if scaler is not None:
            X_test = scaler.transform(X_test)
        test_preds = model.predict(X_test)
    elif args.model_type in ("mlp",):
        if scaler is not None:
            X_test = scaler.transform(X_test).astype(np.float32)
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.from_numpy(X_test).to(device)).squeeze(-1).cpu().numpy()
    elif args.model_type == "polychain":
        from features.graph_utils import build_multiscale
        from models.polychain.cst import compute_cst_batch
        test_samples = [build_multiscale(s) for s in test_feat["SMILES"].tolist()]
        test_samples = [s for s in test_samples if s is not None]
        def collate(samples):
            from features.graph_utils import collate_multiscale
            batch = collate_multiscale(samples)
            batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]), dtype=torch.float)
            return batch
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_samples, batch_size=64, shuffle=False, collate_fn=collate)
        model.eval()
        test_preds = []
        with torch.no_grad():
            for batch_dict in test_loader:
                batch_dict = move_to_device(batch_dict, device)
                pred = model(batch_dict)
                test_preds.append(pred.cpu().numpy())
        test_preds = np.concatenate(test_preds)
    else:
        # GNN models
        from features.graphs import smiles_to_graph
        test_graphs = [smiles_to_graph(s) for s in test_feat["SMILES"].tolist()]
        test_graphs = [g for g in test_graphs if g is not None]
        from torch_geometric.loader import DataLoader as PyGDL
        test_loader = PyGDL(test_graphs, batch_size=64, shuffle=False)
        model.eval()
        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                test_preds.append(pred.cpu().numpy())
        test_preds = np.concatenate(test_preds)

    # Save test predictions
    test_out = pred_dir / f"{ckpt_tag}_test.pkl"
    with open(test_out, "wb") as f:
        pickle.dump({
            "id": test_feat["id"].values.tolist(),
            "pred": test_preds.tolist(),
            "model_type": args.model_type,
            "fold": args.fold,
            "target": target,
        }, f)
    print(f"Test predictions saved -> {test_out}")
```

- [ ] **Step 2: Validate inference**

Run: `cd polymer_competition && python -m training.train --model_type ridge --fold 0 --target tg --max_samples 100`
Check: `predictions/v1_tg_ridge_fold0_test.pkl` created with correct IDs

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/training/train.py
git commit -m "feat: test inference after training, saves per-target test predictions"
```

---
### Task 5: Master Orchestration Update

**Files:**
- Modify: `polymer_competition/generate_all.py`

**Interfaces:**
- Consumes: All prior tasks (config, features, splits, training)
- Produces: Orchestrated execution of full pipeline per target

- [ ] **Step 1: Update generate_all.py**

Key changes:

```python
ALL_TARGETS = ["tg", "egc"]

def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--targets", default="tg,egc",
                        help="Comma-separated targets")
    # ... existing args
    args = parser.parse_args()

    targets = args.targets.split(",") if args.targets else ALL_TARGETS
    steps = set(int(s.strip()) for s in args.steps.split(","))

    # Step 1: Splits (runs for all targets)
    if 1 in steps:
        for target in targets:
            run_cmd(
                [sys.executable, "-m", "data.generate_splits",
                 "--config", args.config, "--target", target],
                desc=f"Step 1: Generate CV Splits for {target}",
            )

    # Step 2: Features (one pass)
    if 2 in steps:
        run_cmd(
            [sys.executable, "-m", "features.build_features", "--config", args.config],
            desc="Step 2: Build Feature Matrix",
        )

    # Step 3: Train (targets × models × folds)
    if 3 in steps:
        for target in targets:
            for model_type in models:
                for fold in range(args.n_folds):
                    exp_ver = cfg.get("experiment", {}).get("version", "v1")
                    pred_file = PROJECT_ROOT / "predictions" / f"{exp_ver}_{target}_{model_type}_fold{fold}.pkl"
                    if pred_file.exists():
                        print(f"  ✓ {pred_file.name} already exists — skipping.")
                        continue
                    cmd = [
                        sys.executable, "-m", "training.train",
                        "--model_type", model_type,
                        "--fold", str(fold),
                        "--config", args.config,
                        "--person", args.person,
                        "--target", target,
                    ]
                    # ... model config logic ...
                    run_cmd(cmd, desc=f"Step 3: Train {model_type} {target} (fold {fold})")

    # Step 4: Ensemble per target
    if 4 in steps:
        for target in targets:
            run_cmd(
                [sys.executable, "-m", "ensemble.build_ensemble",
                 "--config", args.config, "--target", target],
                desc=f"Step 4: Build Ensemble for {target}",
            )
        # Merge submissions
        run_cmd(
            [sys.executable, "-m", "data.merge_submissions",
             "--config", args.config],
            desc="Step 4b: Merge submissions",
        )

    # Step 5: Reports
    if 5 in steps:
        run_cmd(
            [sys.executable, "reports/generate_reports.py", "--config", args.config],
            desc="Step 5: Generate Reports",
        )
```

- [ ] **Step 2: Validate orchestration**

Run: `cd polymer_competition && python generate_all.py --steps 1,2 --targets tg`
Expected: Splits generated for tg, features built

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/generate_all.py
git commit -m "feat: orchestration supports --targets with hierarchical loop"
```

---
### Task 6: Per-Target Ensemble & Submission Merge

**Files:**
- Modify: `polymer_competition/ensemble/build_ensemble.py`
- Create: `polymer_competition/data/merge_submissions.py`
- Test: `polymer_competition/tests/test_merge_submissions.py`

**Interfaces:**
- Consumes:
  - `--target` argument
  - `predictions/v1_{target}_*.pkl` files
  - `data/<target>/test.csv` with original IDs
- Produces:
  - `ensembles/v1_{target}_weights.json`
  - `outputs/submissions/submission.csv` (final merged)

- [ ] **Step 1: Write failing test for merge_submissions**

```python
"""tests/test_merge_submissions.py"""
import tempfile
from pathlib import Path
import pandas as pd
from data.merge_submissions import merge_submissions


def test_merge_submissions_basic():
    tg = pd.DataFrame({"id": [1, 2], "target": [100.0, 200.0]})
    egc = pd.DataFrame({"id": [3, 4], "target": [5.0, 6.0]})
    with tempfile.TemporaryDirectory() as tmp:
        tg.to_csv(Path(tmp) / "tg_preds.csv", index=False)
        egc.to_csv(Path(tmp) / "egc_preds.csv", index=False)
        merge_submissions(Path(tmp) / "tg_preds.csv", Path(tmp) / "egc_preds.csv", Path(tmp) / "submission.csv")
        sub = pd.read_csv(Path(tmp) / "submission.csv")
        assert list(sub.columns) == ["id", "target"]
        assert len(sub) == 4
        assert sub["id"].tolist() == [1, 2, 3, 4]


def test_merge_submissions_sorts_by_id():
    tg = pd.DataFrame({"id": [10, 5], "target": [100.0, 200.0]})
    egc = pd.DataFrame({"id": [3, 1], "target": [5.0, 6.0]})
    with tempfile.TemporaryDirectory() as tmp:
        tg.to_csv(Path(tmp) / "tg_preds.csv", index=False)
        egc.to_csv(Path(tmp) / "egc_preds.csv", index=False)
        merge_submissions(Path(tmp) / "tg_preds.csv", Path(tmp) / "egc_preds.csv", Path(tmp) / "submission.csv")
        sub = pd.read_csv(Path(tmp) / "submission.csv")
        assert sub["id"].tolist() == [1, 3, 5, 10]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd polymer_competition && python -m pytest tests/test_merge_submissions.py -v`
Expected: ImportError

- [ ] **Step 3: Create merge_submissions.py**

```python
"""data/merge_submissions.py

Merge per-target prediction CSVs into final competition submission.

Usage:
    python -m data.merge_submissions --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def merge_submissions(
    tg_csv: str | Path,
    egc_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    tg = pd.read_csv(tg_csv)
    egc = pd.read_csv(egc_csv)
    combined = pd.concat([tg, egc], axis=0)
    combined = combined.sort_values("id").reset_index(drop=True)
    combined = combined[["id", "target"]]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"Merged submission ({len(combined)} rows) -> {output_csv}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tg", default=None)
    parser.add_argument("--egc", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sub_dir = Path(cfg["paths"].get("submissions_dir", "outputs/submissions/"))
    tg_path = args.tg or sub_dir / "tg_preds.csv"
    egc_path = args.egc or sub_dir / "egc_preds.csv"
    out_path = args.output or sub_dir / "submission.csv"
    merge_submissions(tg_path, egc_path, out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd polymer_competition && python -m pytest tests/test_merge_submissions.py -v`
Expected: 2 passed

- [ ] **Step 5: Update ensemble/build_ensemble.py with target support**

Add `--target` argument and filter prediction files by target prefix:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", required=True, help="Target name (tg/egc)")
    parser.add_argument("--strategy", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    sub_dir = Path(cfg["paths"]["submissions_dir"])
    sub_dir.mkdir(parents=True, exist_ok=True)

    exp = cfg.get("experiment", {}).get("version", "v1")
    target = args.target

    # Load OOF predictions matching this target (exclude _test.pkl files)
    rows = []
    for pkl_file in pred_dir.glob(f"{exp}_{target}_*.pkl"):
        if pkl_file.stem.endswith("_test"):
            continue
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        val_idx = np.asarray(data["val_idx"])
        preds = np.asarray(data["pred"])
        y = np.asarray(data["y"])
        for i, (idx, p, t) in enumerate(zip(val_idx, preds, y)):
            rows.append({
                "idx": int(idx),
                "y": float(t),
                "pred": float(p),
                "model_type": data.get("model_type", "unknown"),
                "fold": int(data.get("fold", 0)),
            })
    df = pd.DataFrame(rows)

    # Build OOF matrix + optimize weights
    grouped = df.groupby(["idx", "model_type"])["pred"].mean().unstack()
    y_true = df.groupby("idx")["y"].first().reindex(grouped.index)
    oof = grouped.values
    y = y_true.values
    model_names = list(grouped.columns)

    w = get_weights(args.strategy or cfg["ensemble"]["strategy"], oof, y)
    print(f"Weights ({target}): {dict(zip(model_names, w.round(4)))}")

    # Save weights
    weight_dir = Path("ensembles")
    weight_dir.mkdir(exist_ok=True)
    weight_path = weight_dir / f"{exp}_{target}_weights.json"
    with open(weight_path, "w") as f:
        json.dump({
            "experiment": exp,
            "target": target,
            "strategy": args.strategy or cfg["ensemble"]["strategy"],
            "weights": dict(zip(model_names, w.round(4))),
            "cv_score": float(np.sqrt(np.mean((oof @ w - y) ** 2))),
        }, f, indent=2)

    # Load test predictions (*_test.pkl suffixed files for this target)
    test_rows = []
    for pkl_file in pred_dir.glob(f"{exp}_{target}_*_test.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for i, p in enumerate(np.asarray(data["pred"])):
            test_rows.append({
                "id": int(np.asarray(data["id"])[i]),
                "pred": float(p),
            })
    test_df = pd.DataFrame(test_rows)
    test_blend = test_df.groupby("id")["pred"].mean().values
    submission = pd.DataFrame({
        "id": test_df["id"].unique(),
        "target": test_blend,
    })
    sub_path = sub_dir / f"{target}_preds.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission for {target} -> {sub_path}")
```

- [ ] **Step 6: Validate on real data**

Run: `cd polymer_competition && python -m ensemble.build_ensemble --target tg`
Expected: Weight file saved, tg_preds.csv created

- [ ] **Step 7: Commit**

```bash
git add polymer_competition/ensemble/build_ensemble.py polymer_competition/data/merge_submissions.py polymer_competition/tests/test_merge_submissions.py
git commit -m "feat: per-target ensemble with weight files and submission merge"
```

---
### Task 7: Experiment Tracking

**Files:**
- Create: `polymer_competition/experiments/__init__.py`
- Create: `polymer_competition/experiments/manifest.py`

**Interfaces:**
- Consumes: config, model predictions
- Produces: `polymer_competition/experiments/manifest.json`

- [ ] **Step 1: Create experiments/manifest.py**

```python
"""experiments/manifest.py

Lightweight experiment tracking: records each training run in manifest.json.
"""
from __future__ import annotations

import json
import time
import subprocess
import hashlib
from pathlib import Path

import yaml


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_config_hash(cfg: dict) -> str:
    raw = yaml.dump(cfg, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def record_run(
    experiment: str,
    target: str,
    model_type: str,
    fold: int,
    score: float | None = None,
    checkpoint: str | None = None,
    duration_sec: int = 0,
    seed: int = 42,
    config_path: str = "config.yaml",
) -> None:
    manifest = load_manifest()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    record = {
        "experiment": experiment,
        "target": target,
        "model": model_type,
        "fold": fold,
        "status": "completed" if score is not None else "failed",
        "score": score,
        "checkpoint": checkpoint or "",
        "duration_sec": duration_sec,
        "seed": seed,
        "git_commit": get_git_commit(),
        "config_hash": get_config_hash(cfg),
        "environment": str(Path("experiments") / "environment.txt"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest.append(record)
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
```

- [ ] **Step 2: Integrate manifest calls into training loop**

In `training/train.py`, add at end of successful training:
```python
from experiments.manifest import record_run
record_run(
    experiment=cfg.get("experiment", {}).get("version", "v1"),
    target=target,
    model_type=args.model_type,
    fold=args.fold,
    score=metrics.get("r2"),
    checkpoint=str(ckpt_dir / f"{ckpt_tag}_best.pt"),
    duration_sec=int(time.time() - t_start),
    seed=cfg.get("seed", {}).get("global", 42),
    config_path=args.config,
)
```

- [ ] **Step 3: Validate**

Run: `cd polymer_competition && python -m training.train --model_type ridge --fold 0 --target tg --max_samples 100`
Check: `experiments/manifest.json` exists and has a record

- [ ] **Step 4: Commit**

```bash
git add polymer_competition/experiments/
git commit -m "feat: experiment manifest tracking for training runs"
```

---
### Task 8: Kaggle Notebook

**Files:**
- Create: `polymer_competition/notebooks/kaggle_pipeline.ipynb`

**Interfaces:**
- Consumes: all pipeline modules
- Produces: `submission.csv` at end of notebook run

- [ ] **Step 1: Create kaggle_pipeline.ipynb**

The notebook should:
1. Install dependencies (`!pip install rdkit-pypi torch torch_geometric ...`)
2. Copy data from Kaggle input (`../input/aisehack-2-0/train.csv`, etc.)
3. Run `split_by_target`
4. Run `build_features` (or load cache)
5. Run `generate_splits` per target
6. Train selected models per target
7. Build ensembles per target
8. Merge submissions
9. Save `submission.csv` to Kaggle output directory

Key cells:
```python
# Cell 1: Install deps
!pip install rdkit-pypi pandas numpy scikit-learn pyyaml torch torch_geometric xgboost catboost lightgbm

# Cell 2: Imports and config
import sys, os
sys.path.insert(0, "/kaggle/working/polymer_competition")
# ... module imports

# Cell 3: Copy competition data
import shutil
shutil.copy("../input/aisehack-2-0/train.csv", "data/train.csv")
shutil.copy("../input/aisehack-2-0/test.csv", "data/test.csv")

# Cell 4: Pipeline steps
from data.split_by_target import split_by_target
split_by_target("data/train.csv", "data/test.csv", "data/")

from features.build_features import build_features
build_features()

from data.generate_splits import generate_splits
for target in ["tg", "egc"]:
    generate_splits(f"data/{target}/train.csv", f"data/splits_{target}.pkl")

# ... training loops per target ...

# Final cell: Save submission
shutil.copy("outputs/submissions/submission.csv", "../working/submission.csv")
```

- [ ] **Step 2: Validate notebook structure**

Notebook runs end-to-end in local environment (smoke test with `--max_samples 50 --n-folds 2 --models ridge,xgb`)

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/notebooks/kaggle_pipeline.ipynb
git commit -m "feat: Kaggle notebook with full pipeline orchestration"
```
