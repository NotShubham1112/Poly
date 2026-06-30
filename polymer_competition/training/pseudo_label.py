"""Pseudo-labeling pipeline: expand training set with high-confidence test predictions.

Strategy:
  1. Average test predictions across folds per model.
  2. Compute ensemble std across models → low std = high confidence.
  3. Select top-K% most confident test samples as pseudo-labels.
  4. Retrain tree models on (original train + pseudo-labeled test).
  5. Evaluate OOF on original validation splits (no leakage).

CRITICAL: Pseudo-labeled data is ONLY used for training, never for validation.
The OOF evaluation uses the original validation indices from the v27 splits.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

PRED_DIR = Path("predictions")
DATA_DIR = Path("data")
TREE_MODELS = ["xgb", "lgb", "catboost", "rf"]
CORE_TREE_MODELS = ["xgb", "lgb", "catboost", "rf"]  # exclude MLP per decision


def load_test_predictions(target, exp_ver="v27", models=None):
    """Load test predictions, average across folds per model.
    Returns: dict[model_name] -> np.ndarray of length n_test
    """
    if models is None:
        models = TREE_MODELS + ["mlp"]
    result = {}
    for model in models:
        fold_preds = []
        fold_ids = []
        for fold in range(5):
            path = PRED_DIR / f"{exp_ver}_{target}_{model}_fold{fold}_test.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as f:
                d = pickle.load(f)
            fold_preds.append(np.array(d["pred"]))
            fold_ids.append(np.array(d["id"]))
        if not fold_preds:
            continue
        # Average across folds
        result[model] = {
            "ids": fold_ids[0],  # all folds have same test ids
            "preds": np.mean(fold_preds, axis=0),
            "n_folds": len(fold_preds),
        }
    return result


def compute_ensemble_stats(test_preds, models=None):
    """Compute ensemble mean and std per test sample across models.
    Returns: ids, mean_preds, std_preds (aligned by test id)
    """
    if models is None:
        models = list(test_preds.keys())
    models = [m for m in models if m in test_preds]
    if not models:
        return None, None, None

    ids = test_preds[models[0]]["ids"]
    pred_matrix = np.column_stack([test_preds[m]["preds"] for m in models])
    mean_preds = pred_matrix.mean(axis=1)
    std_preds = pred_matrix.std(axis=1)
    return ids, mean_preds, std_preds


def select_pseudo_labels(ids, mean_preds, std_preds, confidence_pct=70):
    """Select test samples with std below the given percentile.
    Returns: selected_ids, selected_preds
    """
    threshold = np.percentile(std_preds, confidence_pct)
    mask = std_preds <= threshold
    selected_ids = ids[mask]
    selected_preds = mean_preds[mask]
    print(f"  Confidence threshold (std <= {threshold:.4f}): "
          f"{mask.sum()}/{len(mask)} test samples selected "
          f"({100*mask.sum()/len(mask):.1f}%)")
    return selected_ids, selected_preds, mask


def load_oof_predictions(target, exp_ver="v27", models=None):
    """Load OOF predictions for validation evaluation."""
    if models is None:
        models = TREE_MODELS + ["mlp"]
    result = {}
    for model in models:
        for fold in range(5):
            path = PRED_DIR / f"{exp_ver}_{target}_{model}_fold{fold}.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as f:
                d = pickle.load(f)
            if model not in result:
                result[model] = {}
            result[model][fold] = {
                "val_idx": np.array(d["val_idx"]),
                "pred": np.array(d["pred"]),
                "y": np.array(d["y"]),
            }
    return result


def build_combined_features(target, pseudo_ids, pseudo_preds, feat_test, feature_cols):
    """Build feature matrix for pseudo-labeled test samples.
    Returns: X_pseudo (n_pseudo, n_features), y_pseudo (n_pseudo,)
    """
    pseudo_mask = feat_test["id"].isin(pseudo_ids)
    X_pseudo = feat_test.loc[pseudo_mask, feature_cols].values.astype(np.float32)
    id_to_pred = dict(zip(pseudo_ids, pseudo_preds))
    y_pseudo = np.array([id_to_pred[i] for i in feat_test.loc[pseudo_mask, "id"].values],
                        dtype=np.float32)
    return X_pseudo, y_pseudo


def train_tree_with_pseudo(target, model_type, fold, X_tr, y_tr, X_va, y_va,
                           X_pseudo, y_pseudo, seed=42):
    """Train a tree model on combined (original + pseudo) data.
    Returns OOF predictions on the original validation set.
    """
    # Combine original training data with pseudo-labeled test data
    X_combined = np.vstack([X_tr, X_pseudo])
    y_combined = np.concatenate([y_tr, y_pseudo])

    if model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, early_stopping_rounds=30,
        )
        model.fit(X_combined, y_combined, eval_set=[(X_va, y_va)], verbose=False)
    elif model_type == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(X_combined, y_combined, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=0, early_stopping_rounds=30,
        )
        model.fit(X_combined, y_combined, eval_set=(X_va, y_va))
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
        model.fit(X_combined, y_combined)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pred_va = model.predict(X_va)
    return pred_va, model


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def run_pseudo_labeling(target, confidence_pct=70, exp_ver="v27"):
    """Full pseudo-labeling pipeline for one target."""
    print(f"\n{'='*60}")
    print(f"  PSEUDO-LABELING: {target.upper()} (confidence={confidence_pct}%)")
    print(f"{'='*60}")

    # 1. Load test predictions and compute ensemble stats
    test_preds = load_test_predictions(target, exp_ver)
    print(f"  Test prediction models: {list(test_preds.keys())}")

    # Use only core tree models for pseudo-labeling confidence
    ids, mean_preds, std_preds = compute_ensemble_stats(test_preds, CORE_TREE_MODELS)
    if ids is None:
        print("  ERROR: No test predictions available")
        return None

    print(f"  Test samples: {len(ids)}")
    print(f"  Ensemble std: mean={std_preds.mean():.4f}, median={np.median(std_preds):.4f}")

    # 2. Select high-confidence pseudo-labels
    pseudo_ids, pseudo_preds, mask = select_pseudo_labels(
        ids, mean_preds, std_preds, confidence_pct
    )

    # 3. Load features
    feat_train = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")

    # Load raw data to get targets (positional alignment like train.py)
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    target_col = "target"
    if len(feat_train) == len(raw_train):
        feat_train[target_col] = raw_train[target_col].values
        feat_train["target_type"] = raw_train["target_type"].values
    else:
        raise ValueError(f"Row count mismatch: feat_train={len(feat_train)}, raw_train={len(raw_train)}")

    # Filter to the correct target type
    feat_train = feat_train[feat_train["target_type"] == target].reset_index(drop=True)

    feature_cols = [c for c in feat_train.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")]

    # 4. Load splits
    splits_path = DATA_DIR / f"splits_{target}.pkl"
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    # 5. For each fold, retrain with pseudo-labels and evaluate OOF
    results = {}
    for fold in range(5):
        print(f"\n  --- Fold {fold} ---")
        train_idx = splits[fold]["train"]
        val_idx = splits[fold]["val"]

        tr_df = feat_train.iloc[train_idx]
        va_df = feat_train.iloc[val_idx]

        X_tr = tr_df[feature_cols].values.astype(np.float32)
        y_tr = tr_df["target"].values.astype(np.float32)
        X_va = va_df[feature_cols].values.astype(np.float32)
        y_va = va_df["target"].values.astype(np.float32)

        # Build pseudo-label features from test set
        X_pseudo, y_pseudo = build_combined_features(
            target, pseudo_ids, pseudo_preds, feat_test, feature_cols
        )
        print(f"  Original train: {len(X_tr)}, Pseudo: {len(X_pseudo)}, "
              f"Val: {len(X_va)}")

        fold_results = {}
        for model_type in CORE_TREE_MODELS:
            t0 = time.time()
            pred_va, model = train_tree_with_pseudo(
                target, model_type, fold, X_tr, y_tr, X_va, y_va,
                X_pseudo, y_pseudo
            )
            elapsed = time.time() - t0

            r2 = r2_score(y_va, pred_va)

            # Also load original OOF prediction for comparison
            orig_path = PRED_DIR / f"{exp_ver}_{target}_{model_type}_fold{fold}.pkl"
            with open(orig_path, "rb") as f:
                orig_d = pickle.load(f)
            orig_r2 = orig_d["metrics"]["r2"]
            delta = r2 - orig_r2

            marker = " +" if delta > 0 else " -" if delta < 0 else " ="
            print(f"    {model_type:10s}: R²={r2:.4f} (orig={orig_r2:.4f}, "
                  f"delta={delta:+.4f}){marker} [{elapsed:.1f}s]")

            fold_results[model_type] = {
                "r2": r2, "orig_r2": orig_r2, "delta": delta,
                "pred_va": pred_va, "y_va": y_va, "val_idx": val_idx,
            }
        results[fold] = fold_results

    # 6. Summary
    print(f"\n  === SUMMARY ({target.upper()}) ===")
    for model_type in CORE_TREE_MODELS:
        r2s = [results[f][model_type]["r2"] for f in range(5)]
        orig_r2s = [results[f][model_type]["orig_r2"] for f in range(5)]
        mean_r2 = np.mean(r2s)
        mean_orig = np.mean(orig_r2s)
        mean_delta = mean_r2 - mean_orig
        print(f"    {model_type:10s}: orig={mean_orig:.4f}, pseudo={mean_r2:.4f}, "
              f"delta={mean_delta:+.4f}")

    return results


if __name__ == "__main__":
    import sys
    conf_pct = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    which_target = sys.argv[2] if len(sys.argv) > 2 else "all"

    all_results = {}
    targets = ["tg", "egc"] if which_target == "all" else [which_target]
    for target in targets:
        results = run_pseudo_labeling(target, confidence_pct=conf_pct)
        if results:
            all_results[target] = results

    # Save results
    save_data = {}
    for target, results in all_results.items():
        save_data[target] = {}
        for fold, fold_results in results.items():
            save_data[target][fold] = {}
            for model_type, mr in fold_results.items():
                save_data[target][fold][model_type] = {
                    "r2": mr["r2"], "orig_r2": mr["orig_r2"], "delta": mr["delta"],
                }
    with open("pseudo_label_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved results -> pseudo_label_results.json")
