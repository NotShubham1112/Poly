import tempfile
from pathlib import Path
import pandas as pd
from data.generate_splits import generate_splits


def test_generate_splits_with_target():
    with tempfile.TemporaryDirectory() as tmp:
        tg_dir = Path(tmp) / "tg"
        tg_dir.mkdir(parents=True, exist_ok=True)
        train = pd.DataFrame({
            "SMILES": ["CCO", "CCC", "C=O", "CCO", "CCC", "C=O"],
            "target": [100, 5, 200, 105, 6, 195],
        })
        train.to_csv(tg_dir / "train.csv", index=False)
        splits = generate_splits(
            tg_dir / "train.csv",
            Path(tmp) / "splits_tg.pkl",
            n_folds=2,
            smiles_col="SMILES",
            target_col="target",
        )
        assert len(splits) == 2
        for fold_id, idx in splits.items():
            assert "train" in idx and "val" in idx
