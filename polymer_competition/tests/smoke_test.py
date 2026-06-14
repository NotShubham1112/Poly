"""PolyChain end-to-end smoke test: graph build -> train -> checkpoint -> reload -> predict."""
import pickle
import pandas as pd
import yaml
import torch
import numpy as np
from pathlib import Path

from features.graph_utils import build_multiscale, collate_multiscale
from models.polychain import PolyChain
from models.polychain.cst import compute_cst_batch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg["paths"]["data_dir"])
    train = pd.read_parquet(data_dir / "processed" / "train_features.parquet")
    with open(data_dir / "splits.pkl", "rb") as f:
        splits = pickle.load(f)

    target = cfg["target"]["column"]
    fold = splits[0]
    tr_df = train.iloc[fold["train"][:100]].reset_index(drop=True)
    va_df = train.iloc[fold["val"][:30]].reset_index(drop=True)
    print(f"Train: {len(tr_df)}, Val: {len(va_df)}")

    train_samples = [build_multiscale(s, y=y) for s, y in zip(tr_df["SMILES"], tr_df[target])]
    val_samples = [build_multiscale(s, y=y) for s, y in zip(va_df["SMILES"], va_df[target])]
    train_samples = [s for s in train_samples if s is not None]
    val_samples = [s for s in val_samples if s is not None]
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    cst_train = compute_cst_batch([s.smiles for s in train_samples])
    cst_mean = cst_train.mean(axis=0)
    cst_std = cst_train.std(axis=0) + 1e-6

    def collate(samples):
        batch = collate_multiscale(samples)
        batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]), dtype=torch.float)
        return batch

    train_loader = DataLoader(train_samples, batch_size=4, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_samples, batch_size=4, shuffle=False, collate_fn=collate)

    first = next(iter(train_loader))
    in_dim = first["monomer"].x.size(1)
    edge_dim = first["monomer"].edge_attr.size(1)
    cst_dim = first["cst"].size(1)
    print(f"in_dim={in_dim}, edge_dim={edge_dim}, cst_dim={cst_dim}")

    model = PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim, hidden_dim=64,
                      n_backbone_layers=2, n_hamf_layers=1, cst_dim=cst_dim, dropout=0.0)
    model.cst_norm.mean.data = torch.tensor(cst_mean, dtype=torch.float)
    model.cst_norm.std.data = torch.tensor(cst_std, dtype=torch.float)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.MSELoss()

    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch)
            y = batch["y"].view(-1)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                gts.append(batch["y"].view(-1).cpu().numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        val_rmse = float(np.sqrt(np.mean((gts - preds) ** 2)))
        print(f"Epoch {epoch}: loss={total_loss / len(train_loader):.4f}, val_rmse={val_rmse:.4f}")

    # Save checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "model_type": "polychain",
        "config": {
            "in_atom_dim": in_dim,
            "in_edge_dim": edge_dim,
            "hidden_dim": 64,
            "n_backbone_layers": 2,
            "n_hamf_layers": 1,
        },
        "cst_mean": cst_mean.tolist(),
        "cst_std": cst_std.tolist(),
        "epoch": 3,
        "val_rmse": val_rmse,
    }
    ckpt_dir = ROOT / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_dir / "polychain_smoke_test.pt")
    print("Checkpoint saved!")

    # Reload and verify
    loaded = torch.load(ckpt_dir / "polychain_smoke_test.pt", weights_only=False)
    print("Loaded keys:", list(loaded.keys()))
    print("Config:", loaded["config"])

    model2 = PolyChain(
        in_atom_dim=loaded["config"]["in_atom_dim"],
        in_edge_dim=loaded["config"]["in_edge_dim"],
        hidden_dim=loaded["config"].get("hidden_dim", 64),
        n_backbone_layers=loaded["config"].get("n_backbone_layers", 2),
        n_hamf_layers=loaded["config"].get("n_hamf_layers", 1),
        dropout=0.0,
    )
    model2.load_state_dict(loaded["model_state"])
    model2.cst_norm.mean.data = torch.tensor(loaded["cst_mean"], dtype=torch.float)
    model2.cst_norm.std.data = torch.tensor(loaded["cst_std"], dtype=torch.float)
    model2.eval()

    # Predict with both models on same input
    model.eval()
    with torch.no_grad():
        out1 = model(first).numpy()
        out2 = model2(first).numpy()
    match = np.allclose(out1, out2, atol=1e-5)
    print(f"Original output:  {out1.flatten().tolist()}")
    print(f"Reloaded output:  {out2.flatten().tolist()}")
    print(f"Outputs match: {match}")

    assert match, "Reloaded model produces different outputs!"
    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
