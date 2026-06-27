"""
reports/generate_reports.py

Generate model evaluation reports:
    1. SHAP feature importance summary (shap_summary.png)
    2. Error analysis: residual plots, worst predictions (error_analysis.png)
    3. Model summary CSV (model_summary.csv)

Usage:
    python -m reports.generate_reports --config config.yaml
    python reports/generate_reports.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_all_predictions(pred_dir: Path) -> pd.DataFrame:
    """Load all OOF .pkl files into a single DataFrame. Skips _test.pkl files."""
    rows = []
    for pkl in sorted(pred_dir.glob("*.pkl")):
        if pkl.stem.endswith("_test"):
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        val_idx = np.asarray(data.get("val_idx", []))
        preds = np.asarray(data.get("pred", []))
        y = np.asarray(data.get("y", []))
        if len(val_idx) == 0 or len(preds) == 0:
            continue
        for idx, p, t in zip(val_idx, preds, y):
            rows.append({
                "idx": int(idx),
                "y_true": float(t),
                "y_pred": float(p),
                "model": data.get("model_type", "unknown"),
                "fold": int(data.get("fold", 0)),
                "person": data.get("person", "anon"),
                "file": pkl.name,
            })
    return pd.DataFrame(rows)


def generate_shap_summary(pred_dir: Path, data_dir: Path, output_dir: Path,
                          target_col: str = "property"):
    """Generate SHAP feature importance using the best tree model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return

    # Try to load a trained XGBoost model from checkpoints
    ckpt_dir = Path("outputs/checkpoints")
    xgb_ckpts = list(ckpt_dir.glob("*xgb*.pkl")) if ckpt_dir.exists() else []

    if not xgb_ckpts:
        print("No XGBoost checkpoint found. Generating placeholder SHAP report.")
        # Generate a placeholder from feature data if available
        feat_path = data_dir / "processed" / "train_features.parquet"
        if not feat_path.exists():
            print(f"  No feature file at {feat_path}. Skipping SHAP.")
            return

        df = pd.read_parquet(feat_path)
        feature_cols = [c for c in df.columns if c not in ("SMILES", "id", target_col)]

        # Train a quick model for SHAP
        from xgboost import XGBRegressor
        X = df[feature_cols].values[:500]  # subset for speed
        y = df[target_col].values[:500] if target_col in df.columns else np.zeros(min(500, len(df)))

        model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:100])

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X[:100], feature_names=feature_cols,
                          show=False, max_display=30)
        plt.title("SHAP Feature Importance (Top 30)")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP summary -> {output_dir / 'shap_summary.png'}")
    else:
        print(f"Using checkpoint: {xgb_ckpts[0]}")
        # Load the actual trained model
        with open(xgb_ckpts[0], "rb") as f:
            model_data = pickle.load(f)
        # Implementation depends on checkpoint format


def generate_error_analysis(pred_dir: Path, output_dir: Path):
    """Generate residual plots and worst-prediction analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed. Skipping error analysis.")
        return

    df = load_all_predictions(pred_dir)
    if df.empty:
        print("No predictions found. Skipping error analysis.")
        return

    # Aggregate by (idx, model) → mean prediction
    models = df["model"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residual distribution (all models combined)
    ax = axes[0, 0]
    residuals = df["y_true"] - df["y_pred"]
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="#4C78A8")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution (All Models)")

    # 2. Predicted vs Actual scatter
    ax = axes[0, 1]
    for model in models[:6]:  # Top 6 models
        subset = df[df["model"] == model]
        ax.scatter(subset["y_true"], subset["y_pred"], alpha=0.3, s=10, label=model)
    lims = [df["y_true"].min(), df["y_true"].max()]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=7, loc="upper left")

    # 3. Per-model RMSE bar chart
    ax = axes[1, 0]
    per_model_rmse = {}
    for model in models:
        subset = df[df["model"] == model]
        rmse = np.sqrt(np.mean((subset["y_true"] - subset["y_pred"]) ** 2))
        per_model_rmse[model] = rmse
    sorted_models = sorted(per_model_rmse.items(), key=lambda x: x[1])
    names, values = zip(*sorted_models) if sorted_models else ([], [])
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("RMSE")
    ax.set_title("Per-Model RMSE")

    # 4. Worst predictions
    ax = axes[1, 1]
    agg = df.groupby("idx").agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
    ).reset_index()
    agg["abs_error"] = np.abs(agg["y_true"] - agg["y_pred"])
    worst = agg.nlargest(20, "abs_error")
    ax.barh(range(len(worst)), worst["abs_error"].values, color="#E45756")
    ax.set_yticks(range(len(worst)))
    ax.set_yticklabels([f"idx={i}" for i in worst["idx"]], fontsize=7)
    ax.set_xlabel("Absolute Error")
    ax.set_title("Top 20 Worst Predictions")

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Error analysis -> {output_dir / 'error_analysis.png'}")


def generate_model_summary(pred_dir: Path, output_dir: Path):
    """Generate a CSV summary of all models with OOF metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_predictions(pred_dir)
    if df.empty:
        print("No predictions found. Skipping model summary.")
        return

    rows = []
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        y_true = subset["y_true"].values
        y_pred = subset["y_pred"].values
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        n_folds = subset["fold"].nunique()
        persons = ", ".join(sorted(subset["person"].unique()))
        rows.append({
            "model": model,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
            "n_folds": n_folds,
            "persons": persons,
            "n_predictions": len(subset),
        })

    summary = pd.DataFrame(rows).sort_values("rmse")
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    print(f"  Model summary -> {output_dir / 'model_summary.csv'}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP computation (slow)")
    args = parser.parse_args()

    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}

    pred_dir = Path(cfg.get("paths", {}).get("predictions_dir", "predictions/"))
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    output_dir = Path("reports/")
    target_col = cfg.get("target", {}).get("column", "property")

    print("=" * 60)
    print("Generating evaluation reports")
    print("=" * 60)

    # 1. Model summary
    print("\n[1/3] Model Summary")
    generate_model_summary(pred_dir, output_dir)

    # 2. Error analysis
    print("\n[2/3] Error Analysis")
    generate_error_analysis(pred_dir, output_dir)

    # 3. SHAP
    if not args.skip_shap:
        print("\n[3/3] SHAP Feature Importance")
        generate_shap_summary(pred_dir, data_dir, output_dir, target_col)
    else:
        print("\n[3/3] SHAP — skipped")

    print("\n✓ Done. Reports saved to reports/")


if __name__ == "__main__":
    main()
