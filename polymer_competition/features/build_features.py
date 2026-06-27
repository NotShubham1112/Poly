from __future__ import annotations

import argparse
import gc
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
from .polymer_descriptors import compute_polymer_descriptors
from data.split_by_target import split_by_target


def canonicalize(smiles_list: list[str]) -> list[str | None]:
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


def _smiles_scaffold(smiles: str) -> str:
    s = smiles.replace("*", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return s[:20]


def build_features(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles_col = cfg.get("data", {}).get("smiles_col", "smiles")

    train = load_and_normalize(data_dir / "train.csv")
    test = load_and_normalize(data_dir / "test.csv")
    print(f"Loaded train={len(train)} test={len(test)}")

    all_smiles = train["SMILES"].tolist() + test["SMILES"].tolist()
    canon = canonicalize(all_smiles)
    n_invalid = sum(1 for c in canon if c is None)
    fail_policy = cfg.get("data", {}).get("fail_policy", "warn")
    if n_invalid:
        msg = f"{n_invalid} SMILES failed canonicalization"
        if fail_policy == "raise":
            raise ValueError(msg)
        elif fail_policy == "warn":
            print(f"WARNING: {msg}")
    train["canon_smiles"] = canon[: len(train)]
    test["canon_smiles"] = canon[len(train):]

    unique_smiles = sorted(set(s for s in canon if s is not None))
    print(f"Building features on {len(unique_smiles)} unique canonical SMILES")
    fps = all_fingerprints(unique_smiles)
    desc = compute_descriptors(unique_smiles)
    desc = select_descriptors_by_variance(desc)
    cust = compute_all_custom_features(unique_smiles)
    poly_desc = [compute_polymer_descriptors(s) for s in unique_smiles]
    poly_df = pd.DataFrame(poly_desc).add_prefix("polymer_")

    fp_dfs = {}
    for name, arr in fps.items():
        cols = [f"{name}_{i}" for i in range(arr.shape[1])]
        fp_dfs[name] = pd.DataFrame(arr, columns=cols)
    desc_df = desc.drop(columns=["SMILES"], errors="ignore")
    cust_df = cust.drop(columns=["SMILES"], errors="ignore")

    # Free intermediate DataFrames before building the large cache
    del fps, desc, cust
    gc.collect()

    cache_df = pd.concat(
        [pd.DataFrame({"canon_smiles": unique_smiles})]
        + [df.astype(np.float32) for df in fp_dfs.values()]
        + [desc_df.reset_index(drop=True).astype(np.float32)]
        + [cust_df.reset_index(drop=True).astype(np.float32)]
        + [poly_df.astype(np.float32)],
        axis=1,
    )

    del fp_dfs, desc_df, cust_df, poly_df
    gc.collect()

    num_cols = []
    for c in cache_df.columns:
        if c == "canon_smiles":
            continue
        col = cache_df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        if col.dtype != object:
            num_cols.append(c)
    # Replace inf with NaN before imputation (inf can't be cast to float32)
    for col in num_cols:
        col_vals = cache_df[col].values
        if np.isinf(col_vals).any():
            cache_df[col] = np.where(np.isinf(col_vals), np.nan, col_vals)
    imputer = SimpleImputer(strategy="median")
    cache_df[num_cols] = imputer.fit_transform(cache_df[num_cols]).astype(np.float32)

    canon_to_idx = {s: i for i, s in enumerate(cache_df["canon_smiles"].values)}

    def lookup_features(smiles_list, id_vals, id_col="id"):
        indices = []
        valid_smiles = []
        valid_ids = []
        for smi, id_val in zip(smiles_list, id_vals):
            if smi is None or smi not in canon_to_idx:
                continue
            indices.append(canon_to_idx[smi])
            valid_smiles.append(smi)
            valid_ids.append(id_val)
        result = cache_df.iloc[indices].copy()
        result["SMILES"] = valid_smiles
        result[id_col] = valid_ids
        return result

    train_feat = lookup_features(
        train["canon_smiles"].values,
        train["id"].values if "id" in train.columns else range(len(train)),
    )
    test_feat = lookup_features(
        test["canon_smiles"].values,
        test["id"].values,
    )

    # Convert to float32 before saving (halves parquet size and downstream memory)
    for col in train_feat.select_dtypes(include=["float64"]).columns:
        train_feat[col] = train_feat[col].astype(np.float32)
    for col in test_feat.select_dtypes(include=["float64"]).columns:
        test_feat[col] = test_feat[col].astype(np.float32)
    train_feat.to_parquet(out_dir / "features_train.parquet", index=False)
    test_feat.to_parquet(out_dir / "features_test.parquet", index=False)
    print(f"Train features: {train_feat.shape}, Test features: {test_feat.shape}")

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

    split_by_target(data_dir / "train.csv", data_dir / "test.csv", data_dir,
              targets=list(cfg["targets"].keys()))

    for t_name, t_cfg in cfg["targets"].items():
        scaffold_path = data_dir / f"splits_{t_name}_scaffold.pkl"
        if scaffold_path.exists():
            print(f"Using scaffold-aware splits for {t_name} from training/splits.py")
            continue

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    build_features(args.config)
