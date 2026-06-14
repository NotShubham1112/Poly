"""
training/run_ablation.py

Run PolyChain ablation study: Backbone only, +HAMF, +PECGN, Full.

Produces:
    results/ablation_results.csv
    reports/plots/ablation.png

Usage:
    python -m training.run_ablation --config config.yaml
    python -m training.run_ablation --fold 0 --epochs 50
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_utils import set_seed, save_checkpoint, load_checkpoint, rmse, mae, r2_score, spearman
from features.graph_utils import build_multiscale, collate_multiscale
from models.polychain.cst import compute_cst_batch
from torch.utils.data import DataLoader


class BackboneOnly(nn.Module):
    """PolyChain with only the GIN backbone (no HAMF, no PECGN)."""

    def __init__(self, in_atom_dim, in_edge_dim, hidden_dim=128, n_layers=2, cst_dim=32, dropout=0.2):
        super().__init__()
        from models.polychain.backbone import GINBackbone
        self.backbone = GINBackbone(in_atom_dim, in_edge_dim, hidden_dim, n_layers, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch_dict):
        g, _ = self.backbone(batch_dict["monomer"])
        return self.head(g).squeeze(-1)


class BackbonePlusHAMF(nn.Module):
    """PolyChain with backbone + HAMF (no PECGN)."""

    def __init__(self, in_atom_dim, in_edge_dim, hidden_dim=128, n_layers=2,
                 n_hamf_layers=1, cst_dim=32, dropout=0.2):
        super().__init__()
        from models.polychain.backbone import GINBackbone
        from models.polychain.hamf import HAMF
        self.backbone = GINBackbone(in_atom_dim, in_edge_dim, hidden_dim, n_layers, dropout)
        self.hamf = HAMF(hidden_dim, hidden_dim, n_scales=3, n_layers=n_hamf_layers, n_heads=4, dropout=dropout)
        self.head = nn.Linear(3 * hidden_dim, 1)

    def forward(self, batch_dict):
        h1 = self._encode(batch_dict["monomer"])
        h2 = self._encode(batch_dict["dimer"])
        h3 = self._encode(batch_dict["trimer"])
        fused = self.hamf([h1, h2, h3])
        return self.head(fused).squeeze(-1)

    def _encode(self, data):
        g, _ = self.backbone(data)
        return g


class BackbonePlusPECGN(nn.Module):
    """PolyChain with backbone + PECGN (no HAMF — uses monomer only)."""

    def __init__(self, in_atom_dim, in_edge_dim, hidden_dim=128, n_layers=2,
                 cst_dim=32, dropout=0.2):
        super().__init__()
        from models.polychain.backbone import GINBackbone
        from models.polychain.pecgn import PECGN
        self.backbone = GINBackbone(in_atom_dim, in_edge_dim, hidden_dim, n_layers, dropout)
        self.cst_norm = nn.Linear(cst_dim, hidden_dim)
        self.pecgn = PECGN(dim=hidden_dim, cst_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch_dict):
        g, _ = self.backbone(batch_dict["monomer"])
        cst_emb = self.cst_norm(batch_dict["cst"])
        periodic_emb = self.pecgn(g, cst_emb)
        return self.head(periodic_emb).squeeze(-1)


def train_variant(model, train_loader, val_loader, cfg, device, n_epochs=100):
    """Generic training loop for ablation variants."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4),
                            weight_decay=cfg.get("weight_decay", 1e-5))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience = cfg.get("early_stopping", {}).get("patience", 20)
    bad = 0
    train_losses = []
    val_rmses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                     v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            pred = model(batch)
            y = batch["y"].view(-1)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        train_losses.append(total_loss / max(n_batches, 1))

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                         v.to(device) if hasattr(v, "to") else v
                         for k, v in batch.items()}
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                gts.append(batch["y"].view(-1).cpu().numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        val_rmse = float(np.sqrt(np.mean((gts - preds) ** 2)))
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                     v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            pred = model(batch)
            preds.append(pred.cpu().numpy())
            gts.append(batch["y"].view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    metrics = {
        "rmse": rmse(gts, preds),
        "mae": mae(gts, preds),
        "r2": r2_score(gts, preds),
        "spearman": spearman(gts, preds),
    }

    return metrics, train_losses, val_rmses, preds, gts


def main():
    parser = argparse.ArgumentParser(description="PolyChain ablation study")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data_dir = Path(cfg["paths"]["data_dir"])
    train = pd.read_parquet(data_dir / "processed" / "train_features.parquet")
    with open(data_dir / "splits.pkl", "rb") as f:
        splits = pickle.load(f)

    target = cfg["target"]["column"]
    fold = splits[args.fold]
    tr_df = train.iloc[fold["train"]].reset_index(drop=True)
    va_df = train.iloc[fold["val"]].reset_index(drop=True)

    # Build graphs
    train_samples = [build_multiscale(s, y=y) for s, y in zip(tr_df["SMILES"], tr_df[target])]
    val_samples = [build_multiscale(s, y=y) for s, y in zip(va_df["SMILES"], va_df[target])]
    train_samples = [s for s in train_samples if s is not None]
    val_samples = [s for s in val_samples if s is not None]

    # CST calibration
    cst_train = compute_cst_batch([s.smiles for s in train_samples])
    cst_mean = cst_train.mean(axis=0)
    cst_std = cst_train.std(axis=0) + 1e-6

    def collate(samples):
        batch = collate_multiscale(samples)
        batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]), dtype=torch.float)
        return batch

    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    first = next(iter(train_loader))
    in_dim = first["monomer"].x.size(1)
    edge_dim = first["monomer"].edge_attr.size(1)
    cst_dim = first["cst"].size(1)

    # Define variants
    variants = {
        "Backbone": lambda: BackboneOnly(in_dim, edge_dim, args.hidden_dim, n_layers=2, cst_dim=cst_dim),
        "+HAMF": lambda: BackbonePlusHAMF(in_dim, edge_dim, args.hidden_dim, n_layers=2,
                                           n_hamf_layers=1, cst_dim=cst_dim),
        "+PECGN": lambda: BackbonePlusPECGN(in_dim, edge_dim, args.hidden_dim, n_layers=2,
                                             cst_dim=cst_dim),
    }

    # Full PolyChain
    from models.polychain import PolyChain
    variants["Full"] = lambda: PolyChain(
        in_atom_dim=in_dim, in_edge_dim=edge_dim, hidden_dim=args.hidden_dim,
        n_backbone_layers=2, n_hamf_layers=1, cst_dim=cst_dim, dropout=0.2,
    )

    # Inject CST stats into full model
    full_model_factory = variants["Full"]
    def make_full():
        m = full_model_factory()
        m.cst_norm.mean.data = torch.tensor(cst_mean, dtype=torch.float)
        m.cst_norm.std.data = torch.tensor(cst_std, dtype=torch.float)
        return m
    variants["Full"] = make_full

    # Run ablation
    results = {}
    for name, model_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {name}")
        print(f"{'='*60}")

        model = model_fn().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        metrics, train_losses, val_rmses, preds, gts = train_variant(
            model, train_loader, val_loader, cfg.get("model", {}), device, n_epochs=args.epochs
        )

        metrics["variant"] = name
        metrics["n_params"] = n_params
        results[name] = metrics
        print(f"  RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ablation_df = pd.DataFrame(list(results.values()))
    ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)
    print(f"\nAblation results -> {results_dir / 'ablation_results.csv'}")
    print(ablation_df.to_string(index=False))

    # Generate ablation plot
    try:
        from reports.visualizations import ReportGenerator
        gen = ReportGenerator(PROJECT_ROOT / "reports" / "plots")
        variant_rmse = {name: m["rmse"] for name, m in results.items()}
        gen.plot_ablation(variant_rmse, save_name="ablation")
    except Exception as e:
        print(f"Warning: Could not generate ablation plot: {e}")

    print("\n=== Ablation study complete ===")


if __name__ == "__main__":
    main()
