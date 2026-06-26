import tempfile
from pathlib import Path
import pandas as pd
from data.merge_submissions import merge_submissions


def test_merge_submissions_basic():
    tg = pd.DataFrame({"id": [1, 2], "target": [100.0, 200.0]})
    egc = pd.DataFrame({"id": [3, 4], "target": [5.0, 6.0]})
    with tempfile.TemporaryDirectory() as tmp:
        tg.to_csv(Path(tmp) / "tg_preds.csv", index=False)
        egc.to_csv(Path(tmp) / "egc_preds.csv", index=False)
        merge_submissions(Path(tmp) / "tg_preds.csv", Path(tmp) / "egc_preds.csv", Path(tmp) / "submission.csv")
        sub = pd.read_csv(Path(tmp) / "submission.csv")
        assert list(sub.columns) == ["id", "target"]
        assert len(sub) == 4
        assert sub["id"].tolist() == [1, 2, 3, 4]


def test_merge_submissions_sorts_by_id():
    tg = pd.DataFrame({"id": [10, 5], "target": [100.0, 200.0]})
    egc = pd.DataFrame({"id": [3, 1], "target": [5.0, 6.0]})
    with tempfile.TemporaryDirectory() as tmp:
        tg.to_csv(Path(tmp) / "tg_preds.csv", index=False)
        egc.to_csv(Path(tmp) / "egc_preds.csv", index=False)
        merge_submissions(Path(tmp) / "tg_preds.csv", Path(tmp) / "egc_preds.csv", Path(tmp) / "submission.csv")
        sub = pd.read_csv(Path(tmp) / "submission.csv")
        assert sub["id"].tolist() == [1, 3, 5, 10]
