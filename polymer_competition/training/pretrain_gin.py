"""
training/pretrain_gin.py
Self-supervised GIN pretraining via masked atom prediction.
Uses ALL unique SMILES from train + test (no label leakage).

After pretraining, the encoder can be loaded into GINRegressor
for supervised fine-tuning on TG / EGC.

Usage:
    python -m training.pretrain_gin --epochs 200 --batch_size 32
    python -m training.pretrain_gin --resume outputs/pretrain_gin/checkpoint.pt
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from features.graphs import smiles_to_graph
from models.gnn import GINPretrainEncoder, mask_atom_types
from training.train import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PretrainDS(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_pretrain(batch):
    return PyGBatch.from_data_list(batch)


def build_all_graphs(smiles_list):
    graphs = []
    skipped = 0
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1
    if skipped:
        print(f"  Warning: skipped {skipped}/{len(smiles_list)} invalid SMILES")
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/pretrain_gin")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)
    print(f"Device: {DEVICE}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all unique SMILES from train + test (no label leakage)
    tr = pd.read_csv(data_dir / "train.csv")
    te = pd.read_csv(data_dir / "test.csv")
    all_smiles = np.unique(np.concatenate([tr["smiles"].values, te["smiles"].values]))
    print(f"Total unique SMILES: {len(all_smiles)}")

    # Build graphs
    print("Building graphs...")
    graphs = build_all_graphs(all_smiles)
    print(f"  Valid graphs: {len(graphs)}")

    ds = PretrainDS(graphs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_pretrain)

    # Determine input dims from first graph
    sample = graphs[0]
    in_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    print(f"  in_dim={in_dim}, edge_dim={edge_dim}")

    # Model
    model = GINPretrainEncoder(
        in_dim=in_dim, edge_dim=edge_dim,
        num_atom_types=13,
        hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
        n_layers=args.n_layers, dropout=0.1,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch} (best_loss={best_loss:.4f})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(DEVICE)
            x_masked, labels = mask_atom_types(batch, mask_ratio=args.mask_ratio)
            batch.x = x_masked
            loss, _ = model(batch, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_cur = optimizer.param_groups[0]["lr"]

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | best={best_loss:.4f} | lr={lr_cur:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"checkpoint_epoch{epoch}.pt")
            # Also save encoder-only weights for easy loading into regressor
            torch.save(model.encoder.state_dict(), out_dir / f"encoder_epoch{epoch}.pt")
            print(f"  Saved checkpoint -> epoch{epoch}")

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Final encoder saved: {out_dir / f'encoder_epoch{args.epochs}.pt'}")


if __name__ == "__main__":
    main()
