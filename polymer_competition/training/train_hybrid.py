"""
training/train_hybrid.py
HybridNet training: GIN on graphs + MLP on tabular features → fusion → prediction.
Requires precomputed features (features_train.parquet / features_test.parquet).

Usage:
    python -m training.train_hybrid --fold 0 --target tg --epochs 100
    python -m training.train_hybrid --target tg --all_folds
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from features.graphs import smiles_to_graph
from models.hybrid import HybridNet
from training.train import set_seed
from training.train_utils import rmse, r2_score
from torch_geometric.data import Batch as PyGBatch


class HybridDataset(Dataset):
    """Returns (graph, tab_features, y) tuples."""
    def __init__(self, graphs, tab_features, y=None):
        self.graphs = graphs
        self.tab = tab_features
        self.y = y

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        t = torch.from_numpy(self.tab[idx]).float()
        if self.y is not None:
            return g, t, self.y[idx]
        return g, t


def collate_hybrid(batch):
    if len(batch[0]) == 3:
        graphs = [b[0] for b in batch]
        tabs = torch.stack([b[1] for b in batch])
        ys = torch.tensor([b[2] for b in batch], dtype=torch.float)
        return PyGBatch.from_data_list(graphs), tabs, ys
    else:
        graphs = [b[0] for b in batch]
        tabs = torch.stack([b[1] for b in batch])
        return PyGBatch.from_data_list(graphs), tabs


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is None:
            continue
        graphs.append(g)
    return graphs


def load_features(data_dir, target, is_train=True):
    """Load precomputed tabular features aligned by target type.

    features_train.parquet has 6171 rows (same order as train.csv).
    We filter by target_type to get target-specific rows.
    """
    suffix = "train" if is_train else "test"
    feat = pd.read_parquet(data_dir / "processed" / f"features_{suffix}.parquet")
    data = pd.read_csv(data_dir / f"{'train' if is_train else 'test'}.csv")

    # Feature columns: everything except metadata
    exclude = {"id", "canon_smiles", "SMILES"}
    feat_cols = [c for c in feat.columns if c not in exclude]
    feat_arr = feat[feat_cols].values.astype(np.float32)

    target_mask = data["target_type"].values == target
    feat_arr = feat_arr[target_mask]
    data = data[target_mask].reset_index(drop=True)
    return feat_arr, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--target", default="tg")
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--graph_hidden", type=int, default=256)
    parser.add_argument("--graph_embed", type=int, default=128)
    parser.add_argument("--tab_hidden", type=int, default=1024)
    parser.add_argument("--tab_embed", type=int, default=512)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/hybrid")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load pre-saved checkpoints and evaluate")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and cfg.get("device", {}).get("use_cuda", True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tabular features
    print("Loading tabular features...")
    tab_train, train_data = load_features(data_dir, args.target, is_train=True)
    tab_test, test_data = load_features(data_dir, args.target, is_train=False)
    print(f"  Tabular: {tab_train.shape}, {tab_test.shape}")
    print(f"  Train samples: {len(train_data)}, Test: {len(test_data)}")

    # Build graphs
    print("Building graphs...")
    graphs_train = build_graphs(train_data["smiles"].tolist())
    graphs_test = build_graphs(test_data["smiles"].tolist())
    print(f"  Graphs: {len(graphs_train)}, {len(graphs_test)}")

    n_features = tab_train.shape[1]
    in_dim = graphs_train[0].x.size(1)
    edge_dim = graphs_train[0].edge_attr.size(1)

    # Standardize tabular features
    scaler = StandardScaler()
    tab_train = scaler.fit_transform(tab_train).astype(np.float32)
    tab_test = scaler.transform(tab_test).astype(np.float32)

    # Load splits
    splits_path = data_dir / f"splits_{args.target}.pkl"
    if not splits_path.exists():
        from data.generate_splits import generate_splits as _make_splits
        splits = _make_splits(
            str(splits_path), str(splits_path),
            n_folds=cfg["cv"]["n_folds"], seed=args.seed,
            target_col=cfg["data"]["target_col"],
            smiles_col=cfg["data"]["smiles_col"],
        )
    else:
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)

    y_train = train_data["target"].values.astype(np.float32)

    folds_to_run = range(cfg["cv"]["n_folds"]) if args.all_folds else [args.fold]
    all_oof = np.zeros(len(y_train))
    all_oof_count = np.zeros(len(y_train))
    test_preds_list = []

    for fold in folds_to_run:
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")

        tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]

        # Ensure indices are within bounds
        tr_idx = [i for i in tr_idx if i < len(graphs_train)]
        va_idx = [i for i in va_idx if i < len(graphs_train)]

        train_ds = HybridDataset(
            [graphs_train[i] for i in tr_idx],
            tab_train[tr_idx],
            y_train[tr_idx],
        )
        val_ds = HybridDataset(
            [graphs_train[i] for i in va_idx],
            tab_train[va_idx],
            y_train[va_idx],
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_hybrid,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_hybrid,
        )

        model = HybridNet(
            in_dim=in_dim, edge_dim=edge_dim,
            n_features=n_features,
            graph_hidden=args.graph_hidden,
            graph_embed=args.graph_embed,
            tab_hidden=args.tab_hidden,
            tab_embed=args.tab_embed,
            n_layers=3, dropout=0.2,
        ).to(device)

        # Two-stage training: freeze GIN initially so the tabular
        # branch can learn without corrupting the well-performing graph encoder.
        stage1_epochs = min(30, args.epochs // 3)
        model.freeze_graph_encoder()

        tab_params = list(model.tab_encoder.parameters()) + list(model.fusion.parameters())
        optimizer = torch.optim.AdamW(
            tab_params, lr=args.lr, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 20
        bad = 0

        for epoch in range(1, args.epochs + 1):
            # Stage 1 → Stage 2 transition: unfreeze GIN at epoch stage1_epochs
            if epoch == stage1_epochs + 1 and model.graph_encoder_is_frozen():
                model.unfreeze_graph_encoder()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr / 10, weight_decay=1e-5,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - stage1_epochs,
                )

            model.train()
            epoch_loss = 0.0
            for graph_batch, tab_batch, y_batch in train_loader:
                graph_batch = graph_batch.to(device)
                tab_batch = tab_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                pred = model(graph_batch, tab_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph_batch, tab_batch, y_batch in val_loader:
                    graph_batch = graph_batch.to(device)
                    tab_batch = tab_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(graph_batch, tab_batch)
                    val_loss += criterion(pred, y_batch).item()

            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                train_r2 = 1 - 2 * epoch_loss / max(len(train_loader), 1)
                print(f"  Epoch {epoch:3d}: train_loss={epoch_loss/len(train_loader):.4f}, "
                      f"val_loss={val_loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": best_state, "fold": fold, "val_rmse": float(np.sqrt(best_val_loss))},
                       ckpt_dir / f"hybrid_fold{fold}_best.pt")
        model.eval()

        # OOF predictions
        val_preds = []
        y_vals = []
        with torch.no_grad():
            for graph_batch, tab_batch, y_batch in val_loader:
                graph_batch = graph_batch.to(device)
                tab_batch = tab_batch.to(device)
                pred = model(graph_batch, tab_batch)
                val_preds.append(pred.cpu().numpy())
                y_vals.append(y_batch.numpy())
        val_preds = np.concatenate(val_preds)
        y_vals = np.concatenate(y_vals)

        val_r2 = r2_score(y_vals, val_preds)
        val_rmse_val = np.sqrt(np.mean((y_vals - val_preds) ** 2))
        print(f"  Val R²={val_r2:.4f}, RMSE={val_rmse_val:.4f}")

        all_oof[va_idx] = val_preds[:len(va_idx)]
        all_oof_count[va_idx] += 1

        # Test predictions
        test_ds = HybridDataset(graphs_test, tab_test)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_hybrid,
        )
        test_preds = []
        with torch.no_grad():
            for graph_batch, tab_batch in test_loader:
                graph_batch = graph_batch.to(device)
                tab_batch = tab_batch.to(device)
                pred = model(graph_batch, tab_batch)
                test_preds.append(pred.cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_preds_list.append(test_preds)

        # Save embeddings if requested
        if args.save_embeddings:
            emb_dir = out_dir / "embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)
            all_embs = []
            for graph_batch, tab_batch in DataLoader(
                HybridDataset(graphs_train, tab_train),
                batch_size=args.batch_size, shuffle=False,
                num_workers=0, collate_fn=collate_hybrid,
            ):
                graph_batch = graph_batch.to(device)
                tab_batch = tab_batch.to(device)
                ge = model.get_graph_embedding(graph_batch)
                te = model.get_tab_embedding(tab_batch)
                all_embs.append(torch.cat([ge, te], dim=1).cpu().numpy())
            all_embs = np.concatenate(all_embs)
            np.save(emb_dir / f"fold{fold}_hybrid_emb.npy", all_embs)

    # Aggregate OOF
    mask = all_oof_count > 0
    oof_r2 = r2_score(y_train[mask], all_oof[mask])
    print(f"\n{'='*50}")
    print(f"Overall OOF R²: {oof_r2:.4f}")
    print(f"{'='*50}")

    # Save OOF
    oof_data = {
        "val_idx": list(np.where(mask)[0]),
        "pred": all_oof[mask],
        "y": y_train[mask],
        "model_type": "hybrid",
        "target": args.target,
    }
    with open(out_dir / f"oof_{args.target}_hybrid.pkl", "wb") as f:
        pickle.dump(oof_data, f)
    print(f"Saved OOF -> {out_dir / f'oof_{args.target}_hybrid.pkl'}")

    # Average test predictions
    test_preds_avg = np.mean(test_preds_list, axis=0)
    test_out = test_data[["id"]].copy()
    test_out["target"] = test_preds_avg
    test_out.to_csv(out_dir / f"submission_hybrid_{args.target}.csv", index=False)
    print(f"Saved test -> {out_dir / f'submission_hybrid_{args.target}.csv'}")


if __name__ == "__main__":
    main()
