"""
inference.predictor.py

PolymerPredictor – load a trained PolyChain checkpoint and predict
properties for new SMILES strings.

Usage:
    from inference.predictor import PolymerPredictor
    pred = PolymerPredictor("outputs/checkpoints/polychain_best.pt")
    print(pred.predict(["*CCO*", "*c1ccc(*)cc1*"]))
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
logging.getLogger("rdkit").setLevel(logging.ERROR)

from features.graph_utils import build_multiscale, collate_multiscale
from models.polychain import PolyChain
from models.polychain.cst import compute_cst_batch


class PolymerPredictor:
    """Load a trained PolyChain checkpoint and predict on demand."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.target_mean: float | None = None
        self.target_std: float | None = None
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, path: str | Path) -> PolyChain:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        model = PolyChain(
            in_atom_dim=cfg["in_atom_dim"],
            in_edge_dim=cfg["in_edge_dim"],
            hidden_dim=cfg.get("hidden_dim", 256),
            n_backbone_layers=cfg.get("n_backbone_layers", 4),
            n_hamf_layers=cfg.get("n_hamf_layers", 2),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        if "cst_mean" in ckpt and "cst_std" in ckpt:
            model.cst_norm.mean.data = torch.tensor(ckpt["cst_mean"], dtype=torch.float)
            model.cst_norm.std.data = torch.tensor(ckpt["cst_std"], dtype=torch.float)
        if "target_mean" in ckpt:
            self.target_mean = ckpt["target_mean"]
            self.target_std = ckpt.get("target_std", None)
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def predict(self, smiles_list: Iterable[str]) -> np.ndarray:
        smiles_list = list(smiles_list)
        samples = [build_multiscale(s) for s in smiles_list]
        samples = [s for s in samples if s is not None]
        if not samples:
            return np.array([])
        batch = collate_multiscale(samples)
        batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]),
                                    dtype=torch.float)
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else
                     v.to(self.device) if hasattr(v, "to") else v)
                 for k, v in batch.items()}
        out = self.model(batch).cpu().numpy().ravel()
        if self.target_mean is not None:
            out = out * self.target_std + self.target_mean if self.target_std else out + self.target_mean
        out = np.clip(out, -100.0, 100.0)
        return out
