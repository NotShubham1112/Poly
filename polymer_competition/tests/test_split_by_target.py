"""tests/test_split_by_target.py"""
import tempfile
from pathlib import Path
import pandas as pd
from data.split_by_target import split_by_target


def test_split_by_target_creates_four_files():
    train = pd.DataFrame({
        "smiles": ["CCO", "CCC"],
        "target": [100.0, 5.0],
        "target_type": ["tg", "egc"],
    })
    test = pd.DataFrame({
        "id": [1, 2],
        "smiles": ["CCO", "CCC"],
        "target_type": ["tg", "egc"],
    })
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.csv"
        test_path = Path(tmp) / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        split_by_target(train_path, test_path, Path(tmp))
        assert (Path(tmp) / "tg" / "train.csv").exists()
        assert (Path(tmp) / "tg" / "test.csv").exists()
        assert (Path(tmp) / "egc" / "train.csv").exists()
        assert (Path(tmp) / "egc" / "test.csv").exists()
        tg_train = pd.read_csv(Path(tmp) / "tg" / "train.csv")
        assert len(tg_train) == 1
        assert tg_train.iloc[0]["target"] == 100.0


def test_split_by_target_preserves_id():
    train = pd.DataFrame({
        "smiles": ["CCO", "CCC"],
        "target": [100.0, 5.0],
        "target_type": ["tg", "egc"],
    })
    test = pd.DataFrame({
        "id": [10, 20],
        "smiles": ["CCO", "CCC"],
        "target_type": ["tg", "egc"],
    })
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.csv"
        test_path = Path(tmp) / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        split_by_target(train_path, test_path, Path(tmp))
        tg_test = pd.read_csv(Path(tmp) / "tg" / "test.csv")
        assert tg_test.iloc[0]["id"] == 10
