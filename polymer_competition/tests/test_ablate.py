"""Tests for ablate.py."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ensemble import weight_optimizer as wo
from polymer_competition import ablate as ab  # type: ignore  # noqa


@pytest.fixture
def synthetic_oof(tmp_path: Path) -> tuple[Path, str]:
    """Build a synthetic OOF dir with 3 models × 5 folds. Models have
    deliberately different quality so dropping the good one hurts."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    # Single-token exp_ver: the weight_optimizer loader splits filenames on '_'
    # and treats parts[2] as the model name. Underscores in exp_ver would shift
    # the columns and break parsing.
    exp_ver = "vtest"
    rng = np.random.default_rng(0)

    n_folds = 5
    val_per_fold = 30
    models = ["good", "medium", "noisy"]
    # 'good' is nearly perfect, 'medium' is OK, 'noisy' is barely signal
    qualities = {"good": 0.1, "medium": 0.5, "noisy": 2.0}

    for target in ["tg"]:
        for model, noise in qualities.items():
            for f in range(n_folds):
                y_val = rng.normal(size=val_per_fold)
                if model == "good":
                    preds_val = y_val + rng.normal(scale=noise, size=val_per_fold)
                else:
                    preds_val = y_val + rng.normal(scale=noise, size=val_per_fold)
                p = pred_dir / f"{exp_ver}_{target}_{model}_fold{f}.pkl"
                with open(p, "wb") as fh:
                    pickle.dump({"val_idx": list(range(f * val_per_fold, (f + 1) * val_per_fold)),
                                 "pred": preds_val, "y": y_val,
                                 "model_type": model, "fold": f, "target": target}, fh)
    return pred_dir, exp_ver


def test_ablate_target_returns_dataframe(synthetic_oof):
    pred_dir, exp_ver = synthetic_oof
    oof = wo.load_oof_predictions(pred_dir, "tg", exp_ver=exp_ver)
    df = ab.ablate_target(oof, n_folds=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) >= {"model", "r2_full", "r2_without", "delta_r2"}


def test_dropping_good_model_hurts_most(synthetic_oof):
    """Dropping the 'good' model should drop R² more than dropping 'noisy'."""
    pred_dir, exp_ver = synthetic_oof
    oof = wo.load_oof_predictions(pred_dir, "tg", exp_ver=exp_ver)
    df = ab.ablate_target(oof, n_folds=5)
    good = df[df["model"] == "good"]["delta_r2"].iloc[0]
    noisy = df[df["model"] == "noisy"]["delta_r2"].iloc[0]
    assert good < noisy, (
        f"Dropping the good model should hurt R² more than dropping noisy. "
        f"good={good}, noisy={noisy}"
    )


def test_ablation_sorted_descending(synthetic_oof):
    """ablate_target should sort by delta_r2 descending (largest positive at top)."""
    pred_dir, exp_ver = synthetic_oof
    oof = wo.load_oof_predictions(pred_dir, "tg", exp_ver=exp_ver)
    df = ab.ablate_target(oof, n_folds=5)
    deltas = df["delta_r2"].dropna().tolist()
    assert deltas == sorted(deltas, reverse=True)


def test_r2_subset_handles_too_few_models(synthetic_oof):
    """_r2_for_subset should return None when fewer than 2 models are available."""
    pred_dir, exp_ver = synthetic_oof
    oof = wo.load_oof_predictions(pred_dir, "tg", exp_ver=exp_ver)
    # Only one model → optimizer can't fit a meaningful blend
    r2 = ab._r2_for_subset(oof, ["good"], n_folds=5)
    assert r2 is None


def test_main_writes_csv(synthetic_oof, tmp_path: Path, monkeypatch):
    """main() should write reports/ablation_v28.csv (or per-target variant)."""
    pred_dir, exp_ver = synthetic_oof
    # Write a tiny config.yaml pointing at the synthetic pred dir
    cfg = (
        "paths:\n"
        "  predictions_dir: {pd}\n"
        "experiment:\n"
        "  version: {ev}\n"
    ).format(pd=str(pred_dir), ev=exp_ver)
    cfg_path = pred_dir.parent / "config_synth.yaml"
    cfg_path.write_text(cfg)

    # Patch CONFIG_PATH in ablate
    monkeypatch.setattr(ab, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(ab, "REPO_ROOT", pred_dir.parent)

    out_path = tmp_path / "ablation.csv"
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["ablate", "--target", "tg", "--exp_ver", exp_ver,
                    "--out", str(out_path)]
        ab.main()
    finally:
        sys.argv = orig_argv

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 3
    assert "model" in df.columns
    assert "delta_r2" in df.columns
