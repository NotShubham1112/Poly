"""
reports/run_all_visuals.py

Generate all evaluation visualizations for documentation:
  - SHAP feature importance, error analysis, model summary CSV
  - ReportGenerator plots (pred vs actual, residuals, CV RMSE, model comparison, etc.)

Usage:
    python -m reports.run_all_visuals --config config.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_all_oof_predictions(pred_dir: Path, exp: str) -> pd.DataFrame:
    rows = []
    pattern = f"{exp}_*_fold*.pkl"
    for pkl_file in sorted(pred_dir.glob(pattern)):
        if "_test" in pkl_file.name or "summary" in pkl_file.name:
            continue
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        val_idx = np.asarray(data["val_idx"])
        preds = np.asarray(data["pred"])
        y = np.asarray(data["y"])
        model_type = data.get("model_type", "unknown")
        fold = int(data.get("fold", 0))
        target = data.get("target", "unknown")
        for i, (idx, p, t) in enumerate(zip(val_idx, preds, y)):
            rows.append({
                "idx": int(idx),
                "y_true": float(t),
                "y_pred": float(p),
                "model": model_type,
                "fold": fold,
                "target": target,
            })
    return pd.DataFrame(rows)


def build_fold_metrics(df: pd.DataFrame) -> dict:
    result = {}
    for (model, target), grp in df.groupby(["model", "target"]):
        folds = []
        for fold, sub in grp.groupby("fold"):
            y_true = sub["y_true"].values
            y_pred = sub["y_pred"].values
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            folds.append({"fold": fold, "rmse": rmse, "mae": mae, "r2": r2})
        result.setdefault(target, {})[model] = folds
    return result


def generate_reportgenerator_plots(df: pd.DataFrame, output_dir: Path):
    from reports.visualizations import ReportGenerator

    gen = ReportGenerator(output_dir)
    models = df["model"].unique()
    targets = df["target"].unique()

    for target in targets:
        tdf = df[df["target"] == target]
        y_all = tdf.groupby("idx")["y_true"].first().values
        gen.plot_target_distribution(y_all, target_name=f"target_{target}",
                                     save_name=f"target_distribution_{target}")

    for model in models:
        for target in targets:
            sub = df[(df["model"] == model) & (df["target"] == target)]
            if len(sub) < 5:
                continue
            y_true = sub["y_true"].values
            y_pred = sub["y_pred"].values
            tag = f"{model}_{target}"
            gen.plot_pred_vs_actual(y_true, y_pred, model_name=tag,
                                    save_name=f"pred_vs_actual_{tag}")
            gen.plot_residuals(y_true, y_pred, model_name=tag,
                               save_name=f"residuals_{tag}")
            gen.plot_residual_distribution(y_true, y_pred, model_name=tag,
                                           save_name=f"residual_dist_{tag}")

    all_models_rmse = {}
    for model in models:
        sub = df[df["model"] == model]
        rmse = float(np.sqrt(np.mean((sub["y_true"] - sub["y_pred"]) ** 2)))
        all_models_rmse[model] = rmse
    if all_models_rmse:
        gen.plot_model_comparison(all_models_rmse, metric_name="rmse",
                                  save_name="model_comparison_rmse")
        r2_scores = {}
        for model in models:
            sub = df[df["model"] == model]
            yt, yp = sub["y_true"].values, sub["y_pred"].values
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2_scores[model] = 1 - ss_res / (ss_tot + 1e-12)
        gen.plot_model_comparison(r2_scores, metric_name="r2",
                                  save_name="model_comparison_r2")

    fold_metrics = build_fold_metrics(df)
    for target, models_dict in fold_metrics.items():
        for model, folds in models_dict.items():
            if len(folds) > 1:
                gen.plot_cv_rmse(folds, model_name=f"{model}_{target}",
                                 save_name=f"cv_rmse_{model}_{target}")

    print(f"  ReportGenerator plots saved to {output_dir}/")


def generate_shap_and_error_analysis(pred_dir: Path, data_dir: Path, output_dir: Path,
                                     target_col: str = "target", smiles_col: str = "smiles"):
    from reports.generate_reports import (
        generate_error_analysis,
        generate_model_summary,
        generate_shap_summary,
    )
    print("\n  --- SHAP Feature Importance ---")
    generate_shap_summary(pred_dir, data_dir, output_dir, target_col, smiles_col)
    print("\n  --- Error Analysis ---")
    generate_error_analysis(pred_dir, output_dir)
    print("\n  --- Model Summary CSV ---")
    generate_model_summary(pred_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate all evaluation visuals")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    exp = cfg.get("experiment", {}).get("version", "v1")
    data_cfg = cfg.get("data", {})
    target_col = data_cfg.get("target_col", "target")
    smiles_col = data_cfg.get("smiles_col", "smiles")
    reports_dir = Path("reports")
    plots_dir = reports_dir / "plots"

    print("=" * 60)
    print("  GENERATING ALL EVALUATION VISUALIZATIONS")
    print("=" * 60)

    print("\n[1/3] Loading OOF predictions...")
    df = load_all_oof_predictions(pred_dir, exp)
    if len(df) == 0:
        print(f"  No predictions found for {exp}_*_fold*.pkl in {pred_dir}")
    else:
        print(f"  Loaded {len(df)} rows, {df['model'].nunique()} models, {df['target'].nunique()} targets")

    print("\n[2/3] ReportGenerator plots (pred vs actual, residuals, CV, model comparison)...")
    if len(df) > 0:
        generate_reportgenerator_plots(df, plots_dir)
    else:
        print("  Skipped — no predictions loaded.")

    print("\n[3/3] SHAP, error analysis, model summary...")
    generate_shap_and_error_analysis(pred_dir, data_dir, reports_dir, target_col, smiles_col)

    print("\n" + "=" * 60)
    print("  ALL VISUALIZATIONS COMPLETE")
    print(f"  Plots:  {plots_dir}/")
    print(f"  Reports: {reports_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
