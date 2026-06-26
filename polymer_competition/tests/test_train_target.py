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
