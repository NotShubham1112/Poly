"""
training.train_utils.py
Checkpoint save/load, auto-resume, metric tracking, logging.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Save a training checkpoint atomically with PID-unique temp file."""
    import os as _os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{_os.getpid()}.tmp"
    torch.save(state, tmp)
    if path.exists():
        path.unlink()
    tmp.rename(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a training checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)


class MetricTracker:
    """Track per-epoch metrics and emit CSV/JSON logs."""

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def log(self, epoch: int, metrics: dict) -> None:
        self.history.append({"epoch": epoch, **metrics})
        csv_path = self.save_dir / "metrics.csv"
        is_new = not csv_path.exists()
        import pandas as pd
        record = {"epoch": epoch, **metrics}
        pd.DataFrame([record]).to_csv(
            csv_path, index=False, mode="a" if not is_new else "w",
            header=is_new,
        )
        with open(self.save_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(y_true, y_pred).correlation)
