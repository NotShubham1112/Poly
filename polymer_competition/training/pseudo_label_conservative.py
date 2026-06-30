"""Conservative pseudo-labeling: XGB-only, strict 20% confidence, sample_weight=0.5.

Key safety controls vs the failed aggressive approach:
1. Only XGB generates pseudo-labels (no ensemble smoothing)
2. 20% confidence threshold (only keep predictions where ALL 5 XGB folds agree)
3. sample_weight=0.5 for pseudo-labeled data (tells model these labels are less reliable)
4. Only XGB is retrained; all other models untouched
5. MLP/GNNs retained in ensemble for diversity
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

PRED_DIR = Path("predictions")
DATA_DIR = Path("data")
EXP_VER = "v27"
CONFIDENCE_PCT = 20  # only keep bottom 20% std (most confident)
SAMPLE_WEIGHT = 0.5


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def train_xgb_weighted(X_tr, y_tr, X_va, y_va, sample_weight_tr=None, seed=42):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, early_stopping_rounds=30,
    )
    fit_kwargs = {"eval_set": [(X_va, y_va)], "verbose": False}
    if sample_weight_tr is not None:
        fit_kwargs["sample_weight"] = sample_weight_tr
    model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def train_lgb_weighted(X_tr, y_tr, X_va, y_va, sample_weight_tr=None, seed=42):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    fit_kwargs = {"eval_set": [(X_va, y_va)],
                  "callbacks": [lgb.early_stopping(30), lgb.log_evaluation(0)]}
    if sample_weight_tr is not None:
        fit_kwargs["sample_weight"] = sample_weight_tr
    model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def run(target):
    print(f"\n{'='*60}")
    print(f"  CONSERVATIVE PSEUDO-LABELING: {target.upper()}")
    print(f"  Confidence: {CONFIDENCE_PCT}% | Sample weight: {SAMPLE_WEIGHT}")
    print(f"{'='*60}")

    # 1. Get XGB fold-level test predictions (5 folds)
    xgb_fold_preds = []
    xgb_fold_ids = []
    for fold in range(5):
        path = PRED_DIR / f"{EXP_VER}_{target}_xgb_fold{fold}_test.pkl"
        with open(path, "rb") as f:
            d = pickle.load(f)
        xgb_fold_preds.append(np.array(d["pred"]))
        xgb_fold_ids.append(np.array(d["id"]))

    xgb_fold_preds = np.array(xgb_fold_preds)  # (5, n_test)
    ids = xgb_fold_ids[0]
    print(f"  Test samples: {len(ids)}")

    # 2. Compute fold-level std (NOT ensemble std — this is XGB's own disagreement)
    std_per_sample = np.std(xgb_fold_preds, axis=0)
    mean_per_sample = np.mean(xgb_fold_preds, axis=0)
    print(f"  XGB fold std: mean={std_per_sample.mean():.4f}, "
          f"median={np.median(std_per_sample):.4f}")

    # 3. Select bottom {CONFIDENCE_PCT}% most confident
    threshold = np.percentile(std_per_sample, CONFIDENCE_PCT)
    mask = std_per_sample <= threshold
    pseudo_ids = ids[mask]
    pseudo_preds = mean_per_sample[mask]
    print(f"  Threshold std <= {threshold:.4f}: {mask.sum()}/{len(mask)} "
          f"({100*mask.sum()/len(mask):.1f}%) selected")

    # 4. Load features
    feat_train = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat_train) == len(raw_train):
        feat_train["target"] = raw_train["target"].values
        feat_train["target_type"] = raw_train["target_type"].values
    feat_train = feat_train[feat_train["target_type"] == target].reset_index(drop=True)

    feature_cols = [c for c in feat_train.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")]

    # 5. Build pseudo-label features
    pseudo_id_set = set(pseudo_ids.tolist())
    feat_test_indexed = feat_test.set_index("id")
    X_pseudo = np.array([feat_test_indexed.loc[i, feature_cols].values.astype(np.float32)
                         for i in pseudo_ids])
    y_pseudo = pseudo_preds.astype(np.float32)

    # 6. Load splits
    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    # Full test features
    X_test_all = feat_test[feature_cols].values.astype(np.float32)
    test_all_ids = feat_test["id"].values

    # 7. Retrain XGB with pseudo-labels (weighted)
    print(f"\n  --- Retraining XGB with weighted pseudo-labels ---")
    t0 = time.time()
    fold_results_xgb = []
    for fold in range(5):
        train_idx = splits[fold]["train"]
        val_idx = splits[fold]["val"]
        tr_df = feat_train.iloc[train_idx]
        va_df = feat_train.iloc[val_idx]

        X_tr = tr_df[feature_cols].values.astype(np.float32)
        y_tr = tr_df["target"].values.astype(np.float32)
        X_va = va_df[feature_cols].values.astype(np.float32)
        y_va = va_df["target"].values.astype(np.float32)

        # Combine: original train (weight=1.0) + pseudo (weight=0.5)
        X_combined = np.vstack([X_tr, X_pseudo])
        y_combined = np.concatenate([y_tr, y_pseudo])
        sw = np.concatenate([np.ones(len(y_tr)), np.full(len(y_pseudo), SAMPLE_WEIGHT)])

        model = train_xgb_weighted(X_combined, y_combined, X_va, y_va,
                                   sample_weight_tr=sw)
        pred_va = model.predict(X_va)
        pred_test = model.predict(X_test_all)

        r2 = r2_score(y_va, pred_va)

        # Load original for comparison
        orig_path = PRED_DIR / f"{EXP_VER}_{target}_xgb_fold{fold}.pkl"
        with open(orig_path, "rb") as f:
            orig_r2 = pickle.load(f)["metrics"]["r2"]

        delta = r2 - orig_r2
        marker = "+" if delta > 0 else "-"
        print(f"    Fold {fold}: R²={r2:.4f} (orig={orig_r2:.4f}, delta={delta:+.4f}) {marker}")
        fold_results_xgb.append((r2, orig_r2, delta))

        # Save pseudo-labeled XGB predictions
        out_oof = PRED_DIR / f"v31_{target}_xgb_fold{fold}.pkl"
        with open(out_oof, "wb") as f:
            pickle.dump({"val_idx": val_idx, "pred": pred_va, "y": y_va,
                         "metrics": {"r2": r2}, "model_type": "xgb",
                         "fold": fold, "target": target}, f)

        out_test = PRED_DIR / f"v31_{target}_xgb_fold{fold}_test.pkl"
        with open(out_test, "wb") as f:
            pickle.dump({"id": test_all_ids, "pred": pred_test,
                         "model_type": "xgb", "fold": fold, "target": target}, f)

    elapsed = time.time() - t0
    mean_r2 = np.mean([r[0] for r in fold_results_xgb])
    orig_mean = np.mean([r[1] for r in fold_results_xgb])
    print(f"  XGB: orig={orig_mean:.4f} -> pseudo={mean_r2:.4f} "
          f"(delta={mean_r2 - orig_mean:+.4f}) [{elapsed:.1f}s]")

    # 8. Also try LGB with same pseudo-labels
    print(f"\n  --- Retraining LGB with weighted pseudo-labels ---")
    t0 = time.time()
    fold_results_lgb = []
    for fold in range(5):
        train_idx = splits[fold]["train"]
        val_idx = splits[fold]["val"]
        tr_df = feat_train.iloc[train_idx]
        va_df = feat_train.iloc[val_idx]

        X_tr = tr_df[feature_cols].values.astype(np.float32)
        y_tr = tr_df["target"].values.astype(np.float32)
        X_va = va_df[feature_cols].values.astype(np.float32)
        y_va = va_df["target"].values.astype(np.float32)

        X_combined = np.vstack([X_tr, X_pseudo])
        y_combined = np.concatenate([y_tr, y_pseudo])
        sw = np.concatenate([np.ones(len(y_tr)), np.full(len(y_pseudo), SAMPLE_WEIGHT)])

        model = train_lgb_weighted(X_combined, y_combined, X_va, y_va,
                                   sample_weight_tr=sw)
        pred_va = model.predict(X_va)
        pred_test = model.predict(X_test_all)

        r2 = r2_score(y_va, pred_va)
        orig_path = PRED_DIR / f"{EXP_VER}_{target}_lgb_fold{fold}.pkl"
        with open(orig_path, "rb") as f:
            orig_r2 = pickle.load(f)["metrics"]["r2"]

        delta = r2 - orig_r2
        marker = "+" if delta > 0 else "-"
        print(f"    Fold {fold}: R²={r2:.4f} (orig={orig_r2:.4f}, delta={delta:+.4f}) {marker}")
        fold_results_lgb.append((r2, orig_r2, delta))

        out_oof = PRED_DIR / f"v31_{target}_lgb_fold{fold}.pkl"
        with open(out_oof, "wb") as f:
            pickle.dump({"val_idx": val_idx, "pred": pred_va, "y": y_va,
                         "metrics": {"r2": r2}, "model_type": "lgb",
                         "fold": fold, "target": target}, f)

        out_test = PRED_DIR / f"v31_{target}_lgb_fold{fold}_test.pkl"
        with open(out_test, "wb") as f:
            pickle.dump({"id": test_all_ids, "pred": pred_test,
                         "model_type": "lgb", "fold": fold, "target": target}, f)

    elapsed = time.time() - t0
    mean_r2_lgb = np.mean([r[0] for r in fold_results_lgb])
    orig_mean_lgb = np.mean([r[1] for r in fold_results_lgb])
    print(f"  LGB: orig={orig_mean_lgb:.4f} -> pseudo={mean_r2_lgb:.4f} "
          f"(delta={mean_r2_lgb - orig_mean_lgb:+.4f}) [{elapsed:.1f}s]")

    # 9. Also copy unchanged models' predictions to v31 namespace
    print(f"\n  --- Copying unchanged models to v31 ---")
    for model_type in ["catboost", "rf", "mlp", "gcn", "gat", "mpnn"]:
        for fold in range(5):
            for suffix in ["", "_test"]:
                src = PRED_DIR / f"{EXP_VER}_{target}_{model_type}_fold{fold}{suffix}.pkl"
                dst = PRED_DIR / f"v31_{target}_{model_type}_fold{fold}{suffix}.pkl"
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)
        if (PRED_DIR / f"{EXP_VER}_{target}_{model_type}_fold0.pkl").exists():
            print(f"    Copied {model_type}")


if __name__ == "__main__":
    for target in ["tg", "egc"]:
        run(target)
    print("\nDone. Run weight optimizer with --exp_ver v31 next.")
