"""Distribution Alignment: adversarial reweighting to match train/test distributions.

Pipeline:
1. Train a binary classifier to distinguish train vs test samples.
2. If AUC > 0.55, there's distribution shift.
3. Compute sample weights to up-weight train samples that look like test.
4. Retrain XGB/LGB with sample weights.
5. Evaluate OOF on original validation splits.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path("data")
PRED_DIR = Path("predictions")
EXP_VER = "v27"


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def run(target, weight_strategy="inverse_prob", clf_threshold=0.55):
    print(f"\n{'='*60}")
    print(f"  DISTRIBUTION ALIGNMENT: {target.upper()}")
    print(f"  Strategy: {weight_strategy}")
    print(f"{'='*60}")

    # Load features
    feat_train = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat_train) == len(raw_train):
        feat_train["target"] = raw_train["target"].values
        feat_train["target_type"] = raw_train["target_type"].values
    feat_train = feat_train[feat_train["target_type"] == target].reset_index(drop=True)

    feature_cols = [c for c in feat_train.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")]

    X_train = feat_train[feature_cols].values.astype(np.float32)
    X_test = feat_test[feature_cols].values.astype(np.float32)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")

    # Step 1: Train discriminator
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X_combined = np.vstack([X_train, X_test])
    y_combined = np.array([0] * len(X_train) + [1] * len(X_test))

    clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    clf.fit(X_combined, y_combined)

    train_probs = clf.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_combined, clf.predict_proba(X_combined)[:, 1])
    print(f"  Discriminator AUC: {auc:.4f}")

    if auc < clf_threshold:
        print(f"  AUC < {clf_threshold}: No significant distribution shift detected.")
        print(f"  Skipping reweighting.")
        return None

    # Step 2: Compute sample weights
    if weight_strategy == "inverse_prob":
        # Inverse probability weighting (stabilized)
        eps = 0.01
        train_probs_clipped = np.clip(train_probs, eps, 1 - eps)
        # Weight = P(test) / P(train) for train samples
        # High weight = sample looks like test data
        sample_weights = train_probs_clipped / (1 - train_probs_clipped)
        # Normalize to mean = 1
        sample_weights = sample_weights / sample_weights.mean()
    elif weight_strategy == "binary":
        # Simple binary: weight=2 if looks like test, weight=1 otherwise
        sample_weights = np.where(train_probs > 0.6, 2.0, 1.0)
    elif weight_strategy == "smooth":
        # Smooth weighting: weight = 1 + alpha * P(test)
        sample_weights = 1.0 + 5.0 * train_probs
        sample_weights = sample_weights / sample_weights.mean()
    else:
        sample_weights = np.ones(len(X_train))

    print(f"  Weight stats: mean={sample_weights.mean():.4f}, "
          f"std={sample_weights.std():.4f}, "
          f"max={sample_weights.max():.4f}, "
          f"min={sample_weights.min():.4f}")
    print(f"  High-weight samples (w>2): {(sample_weights > 2).sum()}")
    print(f"  Low-weight samples (w<0.5): {(sample_weights < 0.5).sum()}")

    # Step 3: Retrain XGB/LGB with sample weights
    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    results = {}
    for model_type in ["xgb", "lgb"]:
        print(f"\n  --- {model_type.upper()} with reweighting ---")
        fold_r2s = []
        t0 = time.time()

        for fold in range(5):
            train_idx = splits[fold]["train"]
            val_idx = splits[fold]["val"]

            X_tr = X_train[train_idx]
            y_tr = feat_train.iloc[train_idx]["target"].values.astype(np.float32)
            X_va = X_train[val_idx]
            y_va = feat_train.iloc[val_idx]["target"].values.astype(np.float32)
            w_tr = sample_weights[train_idx]

            if model_type == "xgb":
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1, early_stopping_rounds=30,
                )
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)], verbose=False)
            elif model_type == "lgb":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            pred_va = model.predict(X_va)
            r2 = r2_score(y_va, pred_va)

            # Load original for comparison
            orig_path = PRED_DIR / f"{EXP_VER}_{target}_{model_type}_fold{fold}.pkl"
            with open(orig_path, "rb") as f:
                orig_r2 = pickle.load(f)["metrics"]["r2"]

            delta = r2 - orig_r2
            marker = "+" if delta > 0 else "-"
            print(f"    Fold {fold}: R²={r2:.4f} (orig={orig_r2:.4f}, delta={delta:+.4f}) {marker}")

            fold_r2s.append(r2)

            # Save prediction
            val_idx_arr = splits[fold]["val"]
            out_path = PRED_DIR / f"v32_{target}_{model_type}_fold{fold}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump({
                    "val_idx": val_idx_arr, "pred": pred_va, "y": y_va,
                    "metrics": {"r2": r2}, "model_type": model_type,
                    "fold": fold, "target": target,
                }, f)

            # Save test prediction
            pred_test = model.predict(X_test)
            test_ids = feat_test["id"].values
            out_test = PRED_DIR / f"v32_{target}_{model_type}_fold{fold}_test.pkl"
            with open(out_test, "wb") as f:
                pickle.dump({
                    "id": test_ids, "pred": pred_test,
                    "model_type": model_type, "fold": fold, "target": target,
                }, f)

        elapsed = time.time() - t0
        mean_r2 = np.mean(fold_r2s)

        # Get original mean
        orig_r2s = []
        for fold in range(5):
            orig_path = PRED_DIR / f"{EXP_VER}_{target}_{model_type}_fold{fold}.pkl"
            with open(orig_path, "rb") as f:
                orig_r2s.append(pickle.load(f)["metrics"]["r2"])
        orig_mean = np.mean(orig_r2s)

        delta = mean_r2 - orig_mean
        print(f"  {model_type.upper()}: orig={orig_mean:.4f} -> aligned={mean_r2:.4f} "
              f"(delta={delta:+.4f}) [{elapsed:.1f}s]")
        results[model_type] = {"orig": orig_mean, "aligned": mean_r2, "delta": delta}

    return results


if __name__ == "__main__":
    all_results = {}
    for weight_strategy in ["inverse_prob", "smooth", "binary"]:
        print(f"\n{'#'*60}")
        print(f"  STRATEGY: {weight_strategy}")
        print(f"{'#'*60}")
        for target in ["tg", "egc"]:
            results = run(target, weight_strategy=weight_strategy)
            if results:
                all_results[f"{target}_{weight_strategy}"] = results

    print(f"\n{'='*60}")
    print("  ALL RESULTS")
    print(f"{'='*60}")
    for key, res in all_results.items():
        for model_type, r in res.items():
            print(f"  {key} {model_type}: orig={r['orig']:.4f} -> {r['aligned']:.4f} ({r['delta']:+.4f})")
