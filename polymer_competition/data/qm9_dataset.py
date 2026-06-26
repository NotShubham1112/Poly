"""
data/qm9_dataset.py

QM9 dataset loading, preprocessing, and PyTorch Dataset.

Usage:
    from data.qm9_dataset import QM9Dataset, prepare_qm9_data

    datasets, scaler = prepare_qm9_data(tokenizer, build_graphs=False)
    train_ds, val_ds, test_ds = datasets

    batch = train_ds[0]
    print(batch["property"].shape)   # (13,)
    print(batch["smiles"])
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset

from data.download_qm9 import ALL_TARGET_COLUMNS, PROPERTY_COLUMNS

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Property scaler – z-score normalisation (fit on train only)
# ---------------------------------------------------------------------------
class QM9PropertyScaler:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        self.mean = np.mean(values, axis=0).astype(np.float32)
        self.std = np.std(values, axis=0).astype(np.float32) + 1e-8

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


# ---------------------------------------------------------------------------
# SMILES validation
# ---------------------------------------------------------------------------
def validate_smiles(smiles: str) -> tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "MolFromSmiles returned None"
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, f"sanitize failed: {e}"
    return True, "valid"


# ---------------------------------------------------------------------------
# QM9 preprocessing pipeline
# ---------------------------------------------------------------------------
def load_qm9_raw(data_dir: str | Path = "data/") -> pd.DataFrame:
    path = Path(data_dir) / "qm9_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"QM9 raw data not found at {path}. Run `python -m data.download_qm9` first."
        )
    return pd.read_parquet(path)


def preprocess_qm9(
    df: pd.DataFrame,
    smiles_column: str = "canonical_smiles",
    target_columns: list[str] | None = None,
    min_atoms: int = 1,
    max_atoms: int = 50,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    if target_columns is None:
        target_columns = PROPERTY_COLUMNS

    rng = np.random.RandomState(seed)

    # --- SMILES validation via RDKit ---
    valid_mask = []
    invalid_reasons = []
    for smi in df[smiles_column].tolist():
        ok, reason = validate_smiles(smi)
        valid_mask.append(ok)
        if not ok:
            invalid_reasons.append(reason)

    n_total = len(df)
    n_valid = sum(valid_mask)
    n_invalid = n_total - n_valid
    log.info("SMILES validation: %d / %d valid (%.1f%%)",
             n_valid, n_total, 100.0 * n_valid / n_total)
    if n_invalid > 0:
        from collections import Counter
        top = Counter(invalid_reasons).most_common(5)
        log.info("Top-5 invalid reasons: %s", top)

    if n_valid == 0:
        raise ValueError("No valid molecules found in QM9 dataset.")

    df_valid = df[valid_mask].reset_index(drop=True).copy()

    # --- Filter by atom count ---
    if min_atoms > 1 or max_atoms < 50:
        mol_sizes = df_valid[smiles_column].apply(
            lambda s: Chem.MolFromSmiles(s).GetNumAtoms() if Chem.MolFromSmiles(s) else 0
        )
        mask = (mol_sizes >= min_atoms) & (mol_sizes <= max_atoms)
        df_valid = df_valid[mask].reset_index(drop=True)
        log.info("After atom-count filter [%d-%d]: %d molecules",
                 min_atoms, max_atoms, len(df_valid))

    # --- Shuffle + split (fixed seed, no leakage) ---
    indices = np.arange(len(df_valid))
    rng.shuffle(indices)
    df_shuffled = df_valid.iloc[indices].reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    splits = {
        "train": df_shuffled.iloc[:n_train].reset_index(drop=True),
        "val": df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True),
        "test": df_shuffled.iloc[n_train + n_val:].reset_index(drop=True),
    }
    log.info("Splits: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    # --- Property normalisation (fit on TRAIN only) ---
    target_cols = [c for c in target_columns if c in df_shuffled.columns]
    missing = [c for c in target_columns if c not in df_shuffled.columns]
    if missing:
        log.warning("Missing target columns (skipped): %s", missing)

    train_props = splits["train"][target_cols].values.astype(np.float32)
    scaler = QM9PropertyScaler()
    scaler.fit(train_props)

    log.info("Scaler mean (first 5): %s", np.round(scaler.mean[:5], 4))
    log.info("Scaler std  (first 5): %s", np.round(scaler.std[:5], 4))

    prop_stats = {}
    for col in target_cols:
        vals = df_shuffled[col].values
        prop_stats[col] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }

    # Log distribution stats
    log.info("Property distribution (full dataset):")
    for col in target_cols[:5]:
        s = prop_stats[col]
        log.info("  %-6s  min=%10.4f  max=%10.4f  mean=%10.4f  std=%10.4f",
                 col, s["min"], s["max"], s["mean"], s["std"])

    return {
        "splits": splits,
        "scaler": scaler,
        "target_columns": target_cols,
        "smiles_column": smiles_column,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "prop_stats": prop_stats,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class QM9Dataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: Callable,
        properties: np.ndarray | None = None,
        property_scaler: QM9PropertyScaler | None = None,
        max_len: int = 256,
        build_graphs: bool = False,
        dataset_name: str = "",
    ):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.records: list[dict] = []

        n_total = len(smiles_list)
        skip_token, skip_graph = 0, 0

        for i, smi in enumerate(smiles_list):
            tokens = tokenizer.try_encode(smi) if hasattr(tokenizer, "try_encode") else None
            if tokens is None:
                try:
                    tokens = tokenizer.encode(smi)
                except Exception as e:
                    log.warning("Failed to encode SMILES '%s': %s", smi, e)
                    skip_token += 1
                    continue

            if len(tokens) > max_len:
                tokens = tokens[:max_len - 1]
                tokens = torch.cat([tokens, torch.LongTensor([tokenizer.eos_token_id])])

            rec = {
                "input_ids": tokens,
                "target_ids": tokens.clone(),
                "smiles": smi.strip(),
            }

            if properties is not None:
                prop_val = properties[i]
                if property_scaler is not None:
                    prop_val = property_scaler.transform(np.array([prop_val]))[0]
                rec["property"] = torch.tensor(prop_val, dtype=torch.float)

            if build_graphs:
                from features.graph_utils import build_multiscale
                g = build_multiscale(smi.strip())
                if g is None:
                    skip_graph += 1
                    continue
                rec["graph_sample"] = g

            self.records.append(rec)

        n_kept = len(self.records)
        n_skipped = n_total - n_kept
        log.info(
            "QM9Dataset[%s]: %d / %d kept (tokenizer skip=%d, graph skip=%d)",
            dataset_name, n_kept, n_total, skip_token, skip_graph,
        )

        if not self.records:
            fallback = (
                tokenizer.try_encode("CCO")
                if hasattr(tokenizer, "try_encode")
                else tokenizer.encode("CCO")
            )
            if fallback is not None:
                self.records.append({
                    "input_ids": fallback,
                    "target_ids": fallback.clone(),
                    "smiles": "CCO",
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ---------------------------------------------------------------------------
# One-call convenience
# ---------------------------------------------------------------------------
def prepare_qm9_data(
    tokenizer: Callable,
    data_dir: str | Path = "data/",
    target_columns: list[str] | None = None,
    smiles_column: str = "canonical_smiles",
    build_graphs: bool = False,
    max_len: int = 256,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    log_level: int = logging.INFO,
) -> tuple[list[QM9Dataset], QM9PropertyScaler]:
    logging.basicConfig(level=log_level, format="%(levelname)s | %(message)s")

    log.info("Loading QM9 raw data from %s", data_dir)
    df = load_qm9_raw(data_dir)
    log.info("Loaded %d molecules", len(df))

    result = preprocess_qm9(
        df,
        smiles_column=smiles_column,
        target_columns=target_columns,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    splits = result["splits"]
    scaler = result["scaler"]
    target_cols = result["target_columns"]
    smi_col = result["smiles_column"]

    datasets = []
    for split_name in ["train", "val", "test"]:
        df_split = splits[split_name]
        smiles = df_split[smi_col].tolist()
        props = df_split[target_cols].values.astype(np.float32) if target_cols else None
        ds = QM9Dataset(
            smiles_list=smiles,
            tokenizer=tokenizer,
            properties=props,
            property_scaler=scaler,  # same scaler (fit on train) applied to all splits
            max_len=max_len,
            build_graphs=build_graphs,
            dataset_name=split_name,
        )
        datasets.append(ds)

    log.info("")
    log.info("=" * 50)
    log.info("QM9 dataset ready")
    log.info("  Train : %d samples", len(datasets[0]))
    log.info("  Val   : %d samples", len(datasets[1]))
    log.info("  Test  : %d samples", len(datasets[2]))
    log.info("  Targets (%d): %s", len(target_cols), target_cols)
    log.info("=" * 50)

    return datasets, scaler
