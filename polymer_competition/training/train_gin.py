"""
training/train_gin.py

Standalone GIN baseline training.
Trains a GIN on monomer graphs, computes OOF + test predictions,
and saves graph embeddings.

Usage:
    python -m training.train_gin --fold 0 --target tg --epochs 100
    python -m training.train_gin --target tg --all_folds
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader

from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from training.train import train_graph, set_seed
from training.train_utils import rmse, r2_score


def build_graphs(smiles_list, y_list=None):
    graphs = []
    valid_y = []
    for i, s in enumerate(smiles_list):
        g = smiles_to_graph(s)
        if g is None:
            continue
        if y_list is not None:
            g.y = torch.tensor([y_list[i]], dtype=torch.float)
            valid_y.append(y_list[i])
        graphs.append(g)
    if y_list is None:
        return graphs
    return graphs, np.array(valid_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--target", default="tg")
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/gin")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save GIN embeddings for downstream tree models")
    parser.add_argument("--pretrained_encoder", default=None,
                        help="Path to pretrained encoder .pt weights to load before fine-tuning")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder during fine-tuning (only train head)")
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

    # Load data
    print("Loading data...")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train_target = train[train["target_type"] == args.target].reset_index(drop=True)
    test_target = test[test["target_type"] == args.target].reset_index(drop=True)
    print(f"  {args.target}: {len(train_target)} train, {len(test_target)} test")

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
    print(f"  Loaded {len(splits)} folds")

    # Build full graph dataset
    print("Building graphs...")
    full_graphs, full_y = build_graphs(train_target["smiles"].tolist(),
                                       train_target["target"].values)
    test_graphs = build_graphs(test_target["smiles"].tolist())
    print(f"  {len(full_graphs)} train graphs, {len(test_graphs)} test graphs")

    in_dim = full_graphs[0].x.size(1)
    edge_dim = full_graphs[0].edge_attr.size(1)
    print(f"  in_dim={in_dim}, edge_dim={edge_dim}")

    folds_to_run = range(cfg["cv"]["n_folds"]) if args.all_folds else [args.fold]
    all_oof = np.zeros(len(full_graphs))
    all_oof_count = np.zeros(len(full_graphs))
    test_preds_list = []

    for fold in folds_to_run:
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")

        tr_idx = splits[fold]["train"]
        va_idx = splits[fold]["val"]

        train_graphs = [full_graphs[i] for i in tr_idx]
        val_graphs = [full_graphs[i] for i in va_idx]
        y_tr = full_y[tr_idx]
        y_va = full_y[va_idx]

        train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size,
                                     shuffle=True, num_workers=0)
        val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)

        model = GINRegressor(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
            n_layers=3, dropout=0.2,
        ).to(device)

        if args.pretrained_encoder:
            state = torch.load(args.pretrained_encoder, map_location=device, weights_only=True)
            # Handle both full checkpoint and encoder-only weights
            if "encoder." in list(state.keys())[0]:
                # Full state_dict from GINPretrainEncoder — extract encoder keys
                enc_state = {k.replace("encoder.", ""): v
                             for k, v in state.items() if k.startswith("encoder.")}
            else:
                # Already encoder-only weights
                enc_state = state
            missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
            if missing:
                print(f"  WARNING: missing keys in encoder: {missing}")
            if unexpected:
                print(f"  WARNING: unexpected keys in encoder: {unexpected}")
            print(f"  Loaded pretrained encoder from {args.pretrained_encoder}")

        if args.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("  Encoder frozen (only training head)")

        train_cfg = {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "amp": use_cuda,
            "swa": True,
            "swa_lr": 1e-5,
            "early_stopping": {"patience": 30},
            "auto_save_every": 0,  # disable periodic recovery checkpoints
        }

        model, best_val_rmse = train_graph(
            model, train_loader, val_loader, train_cfg, device,
            model_type="gin",
            ckpt_dir=out_dir / "checkpoints",
            fold=fold, person="gin",
            full_cfg=train_cfg,
        )

        # Validation predictions
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_preds.append(pred.cpu().numpy())
        val_preds = np.concatenate(val_preds)

        val_r2 = r2_score(y_va, val_preds)
        val_rmse_val = rmse(y_va, val_preds)
        print(f"  Val R²={val_r2:.4f}, RMSE={val_rmse_val:.4f}")

        all_oof[va_idx] += val_preds
        all_oof_count[va_idx] += 1

        # Test predictions
        test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size,
                                    shuffle=False, num_workers=0)
        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                test_preds.append(pred.cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_preds_list.append(test_preds)

        # Save embeddings if requested
        if args.save_embeddings:
            emb_dir = out_dir / "embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)
            all_embs = []
            for batch in PyGDataLoader(full_graphs, batch_size=args.batch_size,
                                        shuffle=False, num_workers=0):
                batch = batch.to(device)
                emb = model.get_embedding(batch)
                all_embs.append(emb.cpu().numpy())
            all_embs = np.concatenate(all_embs)
            np.save(emb_dir / f"fold{fold}_train_emb.npy", all_embs)

            test_embs = []
            for batch in PyGDataLoader(test_graphs, batch_size=args.batch_size,
                                        shuffle=False, num_workers=0):
                batch = batch.to(device)
                emb = model.get_embedding(batch)
                test_embs.append(emb.cpu().numpy())
            test_embs = np.concatenate(test_embs)
            np.save(emb_dir / f"fold{fold}_test_emb.npy", test_embs)

    # Aggregate OOF
    all_oof = all_oof / np.maximum(all_oof_count, 1)
    oof_r2 = r2_score(full_y, all_oof)
    print(f"\n{'='*50}")
    print(f"Overall OOF R²: {oof_r2:.4f}")
    print(f"{'='*50}")

    # Save OOF predictions
    oof_data = {
        "val_idx": list(range(len(full_y))),
        "pred": all_oof,
        "y": full_y,
        "model_type": "gin",
        "target": args.target,
    }
    with open(out_dir / f"oof_{args.target}_gin.pkl", "wb") as f:
        pickle.dump(oof_data, f)
    print(f"Saved OOF -> {out_dir / f'oof_{args.target}_gin.pkl'}")

    # Average test predictions across folds
    test_preds_avg = np.mean(test_preds_list, axis=0)
    test_out = test_target[["id"]].copy()
    test_out["target"] = test_preds_avg
    test_out.to_csv(out_dir / f"submission_gin_{args.target}.csv", index=False)
    print(f"Saved test -> {out_dir / f'submission_gin_{args.target}.csv'}")


if __name__ == "__main__":
    main()
