"""Train trees on original features + MTL embeddings, optimize ensemble, build submission.

Usage:
    python -m training.run_mtl_augmented
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
PROC_DIR = DATA_DIR / "processed"
PRED_DIR = Path("predictions")
ENS_DIR = Path("ensembles")
SUB_DIR = Path("outputs") / "submissions"
EXP_VER = "aug"

AUX_COLS = [
    "sp3_c_frac", "rotatable_bonds", "ring_count", "aromatic_rings",
    "hbd", "hba", "polymer_mw", "polymer_logp", "polymer_tpsa",
    "chain_flexibility", "hansen_dp", "hansen_dP", "hansen_dH",
]

TREE_PARAMS = {
    "xgb": {
        "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
        "tree_method": "hist", "device": "cuda",
        "random_state": 42, "eval_metric": "rmse",
        "early_stopping_rounds": 50, "verbose": False,
    },
    "lgb": {
        "n_estimators": 2000, "learning_rate": 0.05,
        "num_leaves": 31, "max_depth": -1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.0, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    },
    "catboost": {
        "iterations": 2000, "learning_rate": 0.05, "depth": 6,
        "l2_leaf_reg": 3.0, "random_seed": 42,
        "loss_function": "RMSE", "task_type": "CPU", "verbose": False,
	"early_stopping_rounds": 100,
    },
}


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def train_tree_model(model_type, X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(TREE_PARAMS[model_type])
    params["random_state"] = seed
    if model_type == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model_type == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return m.predict(X_va)


def train_full_tree_model(model_type, X, y, seed=42):
    params = dict(TREE_PARAMS[model_type])
    params["random_state"] = seed
    if model_type in ("xgb",):
        params.pop("early_stopping_rounds", None)
    if model_type == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(**params)
        m.fit(X, y, verbose=False)
    elif model_type == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(**params)
        m.fit(X, y)
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(**params)
        m.fit(X, y, verbose=False)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return m


def load_target_data(target):
    feat = pd.read_parquet(PROC_DIR / "features_train.parquet")
    raw = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat) == len(raw):
        feat["target"] = raw["target"].values
        feat["target_type"] = raw["target_type"].values
    feat = feat[feat["target_type"] == target].reset_index(drop=True)

    mtl = pd.read_parquet(PROC_DIR / f"mtl_embeddings_{target}_train.parquet")
    mtl_cols = [c for c in mtl.columns if c.startswith("MTL_")]

    feature_cols = [c for c in feat.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")]

    X_base = feat[feature_cols].values.astype(np.float32)
    X_mtl = mtl[mtl_cols].values.astype(np.float32)
    X_aug = np.column_stack([X_base, X_mtl])
    y = feat["target"].values.astype(np.float32)

    print(f"  {target}: {len(feat)} samples, base={X_base.shape[1]} feats, mtl={X_mtl.shape[1]} feats, total={X_aug.shape[1]}")

    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    return X_aug, y, splits, feat


def train_target(target, models):
    X_aug, y, splits, feat = load_target_data(target)

    results = {}
    for mt in models:
        print(f"\n  --- {target} / {mt} ---")

        oof_preds = np.full(len(feat), np.nan)
        oof_ys = np.full(len(feat), np.nan)
        fold_r2s = []

        for fold in range(5):
            tr_idx = splits[fold]["train"]
            va_idx = splits[fold]["val"]

            X_tr, y_tr = X_aug[tr_idx], y[tr_idx]
            X_va, y_va = X_aug[va_idx], y[va_idx]
            pred = train_tree_model(mt, X_tr, y_tr, X_va, y_va)
            r2 = r2_score(y_va, pred)
            fold_r2s.append(r2)

            oof_preds[va_idx] = pred
            oof_ys[va_idx] = y_va

            out = {
                "val_idx": va_idx, "pred": pred, "y": y_va,
                "model_type": mt, "fold": fold,
                "metrics": {"r2": r2},
            }
            with open(PRED_DIR / f"{EXP_VER}_{target}_{mt}_fold{fold}.pkl", "wb") as f:
                pickle.dump(out, f)
            print(f"    Fold {fold}: R²={r2:.4f}")

        valid = ~np.isnan(oof_preds)
        oof_r2 = r2_score(oof_ys[valid], oof_preds[valid])
        mean_fold_r2 = np.mean(fold_r2s)
        print(f"    OOF R²={oof_r2:.4f} (mean fold={mean_fold_r2:.4f})")
        results[mt] = {"oof_r2": oof_r2, "fold_r2s": fold_r2s, "oof_preds": oof_preds}

        # Full-data model for test predictions
        full_model = train_full_tree_model(mt, X_aug, y)
        feat_test = pd.read_parquet(PROC_DIR / "features_test.parquet")
        mtl_test = pd.read_parquet(PROC_DIR / f"mtl_embeddings_{target}_test.parquet")
        mtl_cols = [c for c in mtl_test.columns if c.startswith("MTL_")]
        feature_cols = [c for c in feat_test.columns
                        if c not in ("id", "SMILES", "canon_smiles")]
        X_test_base = feat_test[feature_cols].values.astype(np.float32)
        X_test_mtl = mtl_test[mtl_cols].values.astype(np.float32)
        X_test_aug = np.column_stack([X_test_base, X_test_mtl])
        test_pred = full_model.predict(X_test_aug)

        test_out = {
            "pred": test_pred,
            "id": feat_test["id"].values,
            "model_type": mt,
        }
        with open(PRED_DIR / f"{EXP_VER}_{target}_{mt}_fold0_test.pkl", "wb") as f:
            pickle.dump(test_out, f)
        print(f"    Test predictions saved ({len(test_pred)} rows)")

    return results


def optimize_and_submit(target, results, models):
    """Optimize ensemble weights and build submission."""
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE OPTIMIZATION: {target.upper()}")
    print(f"{'='*60}")

    # Build OOF matrix
    oof_preds = np.column_stack([results[mt]["oof_preds"] for mt in models])
    X_aug, y, splits, _ = load_target_data(target)

    # Remove NaN rows
    valid = ~np.any(np.isnan(oof_preds), axis=1)
    oof_clean = oof_preds[valid]
    y_clean = y[valid]
    print(f"  OOF matrix: {oof_clean.shape[1]} models, {valid.sum()} valid samples")

    # NNLS weights
    from scipy.optimize import nnls
    weights, _ = nnls(oof_clean, y_clean)
    weights = weights / weights.sum()
    blended = oof_clean @ weights
    oof_r2 = r2_score(y_clean, blended)
    print(f"  Weights: {dict(zip(models, np.round(weights, 4)))}")
    print(f"  Blended OOF R²: {oof_r2:.4f}")

    # Save weights
    ENS_DIR.mkdir(exist_ok=True)
    weight_path = ENS_DIR / f"{EXP_VER}_{target}_weights.json"
    with open(weight_path, "w") as f:
        json.dump({
            "experiment": EXP_VER, "target": target,
            "weights": dict(zip(models, np.round(weights, 4))),
            "cv_score": float(oof_r2),
        }, f, indent=2)
    print(f"  Weights saved -> {weight_path}")

    # Build test blend
    test_rows = []
    for mt in models:
        p = pickle.load(open(PRED_DIR / f"{EXP_VER}_{target}_{mt}_fold0_test.pkl", "rb"))
        for i, pred_i in enumerate(p["pred"]):
            test_rows.append({"id": int(p["id"][i]), "pred": float(pred_i), "model_type": mt})
    test_df = pd.DataFrame(test_rows)
    test_pivot = test_df.pivot_table(index="id", columns="model_type", values="pred")
    test_pivot = test_pivot[models]
    test_blend = test_pivot.values @ weights

    submission = pd.DataFrame({"id": test_pivot.index, "target": test_blend})
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUB_DIR / f"{target}_preds_{EXP_VER}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Submission -> {sub_path}")

    return weights, oof_r2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="xgb,lgb,catboost")
    args = parser.parse_args()
    models = args.models.split(",")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    all_info = {}
    for target in ["tg", "egc"]:
        print(f"\n{'#'*60}")
        print(f"  TRAINING: {target.upper()}")
        print(f"{'#'*60}")
        results = train_target(target, models)
        weights, oof_r2 = optimize_and_submit(target, results, models)
        all_info[target] = {
            "models": {mt: results[mt]["oof_r2"] for mt in models},
            "ensemble_weights": dict(zip(models, np.round(weights, 4))),
            "ensemble_oof_r2": oof_r2,
        }

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for target in ["tg", "egc"]:
        info = all_info[target]
        print(f"\n  {target.upper()}:")
        for mt, r2 in info["models"].items():
            print(f"    {mt}: {r2:.4f}")
        print(f"    Ensemble: {info['ensemble_oof_r2']:.4f}")
        print(f"    Weights: {info['ensemble_weights']}")

    # Combine TG + EGC into final submission
    tg_sub = pd.read_csv(SUB_DIR / f"tg_preds_{EXP_VER}.csv")
    egc_sub = pd.read_csv(SUB_DIR / f"egc_preds_{EXP_VER}.csv")
    combined = pd.concat([tg_sub, egc_sub], ignore_index=True)
    combined = combined.sort_values("id").reset_index(drop=True)
    combined.to_csv(SUB_DIR / f"submission_{EXP_VER}.csv", index=False)
    print(f"\n  Combined submission: {SUB_DIR / f'submission_{EXP_VER}.csv'} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
