"""
training/train_multitask_gin.py
Multi-task GIN: shared encoder + separate TG / EGC heads.

Trains with alternating batches from both tasks. The shared encoder learns
polymer representations that serve both properties, providing regularization.

Usage:
    python -m training.train_multitask_gin --fold 0 --epochs 100
    python -m training.train_multitask_gin --all_folds --epochs 100
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
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader as PyGDataLoader

from features.graphs import smiles_to_graph
from models.gnn import MultiTaskGIN
from training.train import set_seed


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/multitask_gin")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and cfg.get("device", {}).get("use_cuda", True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    tr = pd.read_csv(data_dir / "train.csv")
    te = pd.read_csv(data_dir / "test.csv")

    tg_mask = tr["target_type"].values == "tg"
    egc_mask = tr["target_type"].values == "egc"
    y_tg = tr["target"].values[tg_mask].astype(np.float32)
    y_egc = tr["target"].values[egc_mask].astype(np.float32)
    smiles_tg = tr["smiles"].values[tg_mask]
    smiles_egc = tr["smiles"].values[egc_mask]

    te_tg_mask = te["target_type"].values == "tg"
    te_egc_mask = te["target_type"].values == "egc"
    te_ids_tg = te["id"].values[te_tg_mask]
    te_ids_egc = te["id"].values[te_egc_mask]
    te_smiles_tg = te["smiles"].values[te_tg_mask]
    te_smiles_egc = te["smiles"].values[te_egc_mask]

    print(f"TG: {len(y_tg)} train, {len(te_ids_tg)} test")
    print(f"EGC: {len(y_egc)} train, {len(te_ids_egc)} test")

    # Build graphs
    print("Building graphs...")
    tg_graphs = build_graphs(smiles_tg)
    egc_graphs = build_graphs(smiles_egc)
    te_tg_graphs = build_graphs(te_smiles_tg)
    te_egc_graphs = build_graphs(te_smiles_egc)
    print(f"  TG: {len(tg_graphs)} train, {len(te_tg_graphs)} test")
    print(f"  EGC: {len(egc_graphs)} train, {len(te_egc_graphs)} test")

    # Load splits
    with open(data_dir / "splits_tg.pkl", "rb") as f:
        tg_splits = pickle.load(f)
    with open(data_dir / "splits_egc.pkl", "rb") as f:
        egc_splits = pickle.load(f)

    in_dim = tg_graphs[0].x.size(1)
    edge_dim = tg_graphs[0].edge_attr.size(1)

    folds_to_run = range(5) if args.all_folds else [args.fold]
    tg_oof = np.zeros(len(y_tg), dtype=np.float32)
    egc_oof = np.zeros(len(y_egc), dtype=np.float32)
    tg_te_list, egc_te_list = [], []

    for fold in folds_to_run:
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")

        # Get indices
        tg_tr = [i for i in tg_splits[fold]["train"] if i < len(y_tg)]
        tg_va = [i for i in tg_splits[fold]["val"] if i < len(y_tg)]
        egc_tr = [i for i in egc_splits[fold]["train"] if i < len(y_egc)]
        egc_va = [i for i in egc_splits[fold]["val"] if i < len(y_egc)]

        # Build datasets: store task target in data.y
        for i in tg_tr:
            tg_graphs[i].y = torch.tensor([y_tg[i]], dtype=torch.float)
        for i in tg_va:
            tg_graphs[i].y = torch.tensor([y_tg[i]], dtype=torch.float)
        for i in egc_tr:
            egc_graphs[i].y = torch.tensor([y_egc[i]], dtype=torch.float)
        for i in egc_va:
            egc_graphs[i].y = torch.tensor([y_egc[i]], dtype=torch.float)

        tg_train_loader = PyGDataLoader(
            [tg_graphs[i] for i in tg_tr],
            batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        egc_train_loader = PyGDataLoader(
            [egc_graphs[i] for i in egc_tr],
            batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        tg_val_loader = PyGDataLoader(
            [tg_graphs[i] for i in tg_va],
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        egc_val_loader = PyGDataLoader(
            [egc_graphs[i] for i in egc_va],
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

        model = MultiTaskGIN(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
            n_layers=3, dropout=0.2,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 25
        bad = 0
        stage1 = min(30, args.epochs // 3)

        for epoch in range(1, args.epochs + 1):
            if epoch == stage1 + 1:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - stage1,
                )

            model.train()
            epoch_loss = 0.0
            tgl, egcl = 0.0, 0.0
            tgn, egcn = 0, 0

            # Interleave TG and EGC batches
            iter_tg = iter(tg_train_loader)
            iter_egc = iter(egc_train_loader)
            tg_done, egc_done = False, False

            while not (tg_done and egc_done):
                if not tg_done:
                    try:
                        batch = next(iter_tg)
                        batch = batch.to(device)
                        pred = model(batch, task="tg")
                        loss = criterion(pred, batch.y.squeeze())
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        tgl += loss.item()
                        tgn += 1
                    except StopIteration:
                        tg_done = True

                if not egc_done:
                    try:
                        batch = next(iter_egc)
                        batch = batch.to(device)
                        pred = model(batch, task="egc")
                        loss = criterion(pred, batch.y.squeeze())
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        egcl += loss.item()
                        egcn += 1
                    except StopIteration:
                        egc_done = True

            scheduler.step()

            # Validation
            model.eval()
            vl_tg, vl_egc = 0.0, 0.0
            with torch.no_grad():
                for batch in tg_val_loader:
                    batch = batch.to(device)
                    pred = model(batch, task="tg")
                    vl_tg += criterion(pred, batch.y.squeeze()).item()
                for batch in egc_val_loader:
                    batch = batch.to(device)
                    pred = model(batch, task="egc")
                    vl_egc += criterion(pred, batch.y.squeeze()).item()

            vl_tg /= max(len(tg_val_loader), 1)
            vl_egc /= max(len(egc_val_loader), 1)
            val_loss = vl_tg + vl_egc

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

            if epoch % 10 == 0 or epoch == 1:
                # Compute val R2
                tg_vp, tg_vy = [], []
                egc_vp, egc_vy = [], []
                with torch.no_grad():
                    for batch in tg_val_loader:
                        batch = batch.to(device)
                        tg_vp.append(model(batch, task="tg").cpu().numpy())
                        tg_vy.append(batch.y.squeeze().cpu().numpy())
                    for batch in egc_val_loader:
                        batch = batch.to(device)
                        egc_vp.append(model(batch, task="egc").cpu().numpy())
                        egc_vy.append(batch.y.squeeze().cpu().numpy())
                tg_r2 = r2_score(np.concatenate(tg_vy), np.concatenate(tg_vp)) if tg_vp else 0
                egc_r2 = r2_score(np.concatenate(egc_vy), np.concatenate(egc_vp)) if egc_vp else 0
                lr_cur = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:3d}: loss={epoch_loss/max(tgn+egcn,1):.3f}, "
                      f"TG_R2={tg_r2:.4f}, EGC_R2={egc_r2:.4f}, lr={lr_cur:.6f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # OOF predictions
        tg_vp, tg_vy = [], []
        egc_vp, egc_vy = [], []
        with torch.no_grad():
            for batch in tg_val_loader:
                batch = batch.to(device)
                tg_vp.append(model(batch, task="tg").cpu().numpy())
                tg_vy.append(batch.y.squeeze().cpu().numpy())
            for batch in egc_val_loader:
                batch = batch.to(device)
                egc_vp.append(model(batch, task="egc").cpu().numpy())
                egc_vy.append(batch.y.squeeze().cpu().numpy())
        tg_r2 = r2_score(np.concatenate(tg_vy), np.concatenate(tg_vp))
        egc_r2 = r2_score(np.concatenate(egc_vy), np.concatenate(egc_vp))
        print(f"  Fold {fold}: TG_R2={tg_r2:.4f}, EGC_R2={egc_r2:.4f}")

        tg_oof[tg_va] = np.concatenate(tg_vp)
        egc_oof[egc_va] = np.concatenate(egc_vp)

        # Test predictions
        te_tg_loader = PyGDataLoader(te_tg_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0)
        te_egc_loader = PyGDataLoader(te_egc_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0)
        tg_tp = []
        with torch.no_grad():
            for g in te_tg_loader:
                g = g.to(device)
                tg_tp.append(model(g, task="tg").cpu().numpy())
        egc_tp = []
        with torch.no_grad():
            for g in te_egc_loader:
                g = g.to(device)
                egc_tp.append(model(g, task="egc").cpu().numpy())
        tg_te_list.append(np.concatenate(tg_tp))
        egc_te_list.append(np.concatenate(egc_tp))

        # Save per-fold
        torch.save(best_state, out_dir / f"checkpoint_fold{fold}.pt")

    # Overall OOF
    tg_oof_r2 = r2_score(y_tg, tg_oof)
    egc_oof_r2 = r2_score(y_egc, egc_oof)
    mean_r2 = (tg_oof_r2 + egc_oof_r2) / 2
    print(f"\n{'='*50}")
    print(f"TG OOF R2: {tg_oof_r2:.4f}")
    print(f"EGC OOF R2: {egc_oof_r2:.4f}")
    print(f"Mean OOF R2: {mean_r2:.4f}")
    print(f"{'='*50}")

    # Save OOF
    for name, pred, y_true in [("tg", tg_oof, y_tg), ("egc", egc_oof, y_egc)]:
        data = {"val_idx": list(range(len(y_true))), "pred": pred, "y": y_true,
                "model_type": "multitask_gin", "target": name}
        with open(out_dir / f"oof_{name}.pkl", "wb") as f:
            pickle.dump(data, f)

    # Save test predictions
    tg_te_avg = np.mean(tg_te_list, axis=0)
    egc_te_avg = np.mean(egc_te_list, axis=0)
    sub = pd.concat([
        pd.DataFrame({"id": te_ids_tg, "target": tg_te_avg}),
        pd.DataFrame({"id": te_ids_egc, "target": egc_te_avg}),
    ])
    sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"Saved: {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
