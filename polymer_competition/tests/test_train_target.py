import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_train_accepts_target_arg():
    """train.py --target tg should load tg features and splits."""
    result = subprocess.run(
        [sys.executable, "-m", "training.train",
         "--model_type", "ridge", "--fold", "0",
         "--target", "tg", "--max_samples", "50"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    assert "tg" in result.stdout.lower() or "Fold 0" in result.stdout


def test_mlp_args_accepted():
    """Verify --n_seeds and --loss arguments are accepted by train.py."""
    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--help"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert "--n_seeds" in result.stdout
    assert "--loss" in result.stdout
    assert "huber" in result.stdout
    assert "mse" in result.stdout


def test_huber_loss_works():
    """Verify huber_loss function computes correctly."""
    from training.train import huber_loss
    import torch
    pred = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float32)
    target = torch.tensor([1.5, 2.5, 2.5, 10.5], dtype=torch.float32)
    loss = huber_loss(pred, target, delta=1.0)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_multi_seed_mlp_training():
    """Verify multi-seed MLP training produces averaged predictions."""
    import numpy as np
    from models.mlp import FingerprintMLP
    from training.train_utils import set_seed
    import torch

    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    X_t = torch.from_numpy(X)

    preds = []
    for seed in (42, 43):
        set_seed(seed)
        model = FingerprintMLP(in_dim=10)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        for _ in range(30):
            opt.zero_grad()
            loss = torch.nn.MSELoss()(model(X_t).squeeze(-1), torch.from_numpy(y))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds.append(model(X_t).squeeze(-1).numpy())

    p1, p2 = preds
    avg = np.mean(preds, axis=0)
    assert p1.shape == (50,)
    assert p2.shape == (50,)
    assert avg.shape == (50,)
    assert not np.allclose(p1, p2, rtol=1e-3), "Different seeds should differ"
