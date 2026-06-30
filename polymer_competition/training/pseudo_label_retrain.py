"""Pseudo-label retraining: retrain tree models with pseudo-labeled test data,
save OOF + test predictions, then re-optimize ensemble and build submission.

Usage:
    python -m training.pseudo_label_retrain --confidence_pct 70
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
import yaml

PRED_DIR = Path("predictions")
DATA_DIR = Path("data")
TREE_MODELS = ["xgb", "lgb", "catboost", "rf"]
EXP_VER = "v27"
OUT_EXP_VER = "v30"  # version tag for pseudo-labeled predictions


def load_test_predictions(target, exp_ver=EXP_VER, models=None):
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
        result[model] = {
            "ids": fold_ids[0],
            "preds": np.mean(fold_preds, axis=0),
        }
    return result


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def train_and_predict(target, model_type, fold, X_tr, y_tr, X_va, y_va, X_test,
                      test_ids, seed=42):
    """Train tree model on (train + pseudo), return OOF pred, test pred, model."""
    # Use ensemble stats from CORE models to get pseudo-labels
    # (test_ids and X_test are already the pseudo-labeled subset)

    if model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model_type == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=0, early_stopping_rounds=30,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

    pred_va = model.predict(X_va)
    pred_test = model.predict(X_test) if X_test is not None else None
    return pred_va, pred_test


def run(target, confidence_pct=70):
    print(f"\n{'='*60}")
    print(f"  PSEUDO-LABEL RETRAIN: {target.upper()} ({confidence_pct}%)")
    print(f"{'='*60}")

    # 1. Load test predictions for ensemble stats
    test_preds = load_test_predictions(target, EXP_VER)
    models_for_stats = ["xgb", "lgb", "catboost", "rf"]
    models_available = [m for m in models_for_stats if m in test_preds]
    if not models_available:
        print("ERROR: No test predictions"); return

    ids = test_preds[models_available[0]]["ids"]
    pred_matrix = np.column_stack([test_preds[m]["preds"] for m in models_available])
    mean_preds = pred_matrix.mean(axis=1)
    std_preds = pred_matrix.std(axis=1)

    # 2. Select high-confidence pseudo-labels
    threshold = np.percentile(std_preds, confidence_pct)
    mask = std_preds <= threshold
    pseudo_ids = ids[mask]
    pseudo_mean = mean_preds[mask]
    print(f"  Pseudo-labels: {mask.sum()}/{len(mask)} ({100*mask.sum()/len(mask):.1f}%)")

    # 3. Load features
    feat_train = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")

    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat_train) == len(raw_train):
        feat_train["target"] = raw_train["target"].values
        feat_train["target_type"] = raw_train["target_type"].values
    feat_train = feat_train[feat_train["target_type"] == target].reset_index(drop=True)

    feature_cols = [c for c in feat_train.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")]

    # 4. Get test features for pseudo-labeled samples
    pseudo_id_set = set(pseudo_ids.tolist())
    test_mask = feat_test["id"].isin(pseudo_id_set)
    X_pseudo_all = feat_test.loc[test_mask, feature_cols].values.astype(np.float32)
    # Match order
    id_to_idx = {id_val: idx for idx, id_val in enumerate(feat_test["id"].values)}
    pseudo_order = [id_to_idx[i] for i in pseudo_ids]
    X_pseudo = X_pseudo_all[np.isin(feat_test.loc[test_mask, "id"].values, pseudo_ids)]

    # Rebuild with correct ordering
    pseudo_feat_df = feat_test[feat_test["id"].isin(pseudo_id_set)].set_index("id")
    X_pseudo = np.array([pseudo_feat_df.loc[i, feature_cols].values.astype(np.float32)
                         for i in pseudo_ids])
    y_pseudo = pseudo_mean

    # 5. Load splits
    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    # Full test features for ALL test samples
    test_feat_cols = [c for c in feat_test.columns if c in feature_cols]
    X_test_all = feat_test[test_feat_cols].values.astype(np.float32)
    test_all_ids = feat_test["id"].values

    # 6. Retrain each model with pseudo-labels
    out_dir = PRED_DIR
    all_r2 = {}
    for model_type in TREE_MODELS:
        fold_r2 = []
        t0 = time.time()
        for fold in range(5):
            train_idx = splits[fold]["train"]
            val_idx = splits[fold]["val"]

            tr_df = feat_train.iloc[train_idx]
            va_df = feat_train.iloc[val_idx]

            X_tr = tr_df[feature_cols].values.astype(np.float32)
            y_tr = tr_df["target"].values.astype(np.float32)
            X_va = va_df[feature_cols].values.astype(np.float32)
            y_va = va_df["target"].values.astype(np.float32)

            # Combine original train with pseudo-labeled test
            X_combined = np.vstack([X_tr, X_pseudo])
            y_combined = np.concatenate([y_tr, y_pseudo])

            # Train on combined data, evaluate on original val
            pred_va, pred_test = train_and_predict(
                target, model_type, fold,
                X_combined, y_combined, X_va, y_va,
                X_test_all, test_all_ids
            )
            r2 = r2_score(y_va, pred_va)
            fold_r2.append(r2)

            # Save OOF prediction
            out_file = out_dir / f"{OUT_EXP_VER}_{target}_{model_type}_fold{fold}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump({
                    "val_idx": val_idx, "pred": pred_va, "y": y_va,
                    "metrics": {"r2": r2},
                    "model_type": model_type, "fold": fold, "target": target,
                }, f)

            # Save test prediction
            if pred_test is not None:
                out_test = out_dir / f"{OUT_EXP_VER}_{target}_{model_type}_fold{fold}_test.pkl"
                with open(out_test, "wb") as f:
                    pickle.dump({
                        "id": test_all_ids, "pred": pred_test,
                        "model_type": model_type, "fold": fold, "target": target,
                    }, f)

        elapsed = time.time() - t0
        mean_r2 = np.mean(fold_r2)
        # Load original R² for comparison
        orig_r2s = []
        for fold in range(5):
            orig_path = PRED_DIR / f"{EXP_VER}_{target}_{model_type}_fold{fold}.pkl"
            with open(orig_path, "rb") as f:
                orig_r2s.append(pickle.load(f)["metrics"]["r2"])
        orig_mean = np.mean(orig_r2s)

        all_r2[model_type] = {"pseudo": mean_r2, "orig": orig_mean}
        delta = mean_r2 - orig_mean
        print(f"  {model_type:10s}: orig={orig_mean:.4f} -> pseudo={mean_r2:.4f} "
              f"(delta={delta:+.4f}) [{elapsed:.1f}s]")

    return all_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence_pct", type=int, default=70)
    args = parser.parse_args()

    all_results = {}
    for target in ["tg", "egc"]:
        results = run(target, args.confidence_pct)
        if results:
            all_results[target] = results

    # Summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    for target in ["tg", "egc"]:
        if target not in all_results:
            continue
        print(f"\n  {target.upper()}:")
        for model_type in TREE_MODELS:
            r = all_results[target][model_type]
            delta = r["pseudo"] - r["orig"]
            print(f"    {model_type:10s}: {r['orig']:.4f} -> {r['pseudo']:.4f} ({delta:+.4f})")

        # Compute ensemble OOF
        from ensemble.weight_optimizer import load_oof_predictions, _stack_oof, get_weights, r2_score as wo_r2
        oof = load_oof_predictions(str(PRED_DIR), target, OUT_EXP_VER)
        models = [m for m in TREE_MODELS if m in oof]
        all_preds, all_y, active = _stack_oof(oof, models, n_folds=5)
        if all_preds is not None:
            w = get_weights("optimize", all_preds, all_y)
            ens_r2 = wo_r2(all_y, all_preds @ w)
            print(f"\n    ENSEMBLE (pseudo-labeled): R² = {ens_r2:.4f}")
            for m, ww in sorted(zip(active, w), key=lambda x: -x[1]):
                print(f"      {m}: {ww:.3f}")

    with open("pseudo_retrain_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved -> pseudo_retrain_results.json")


if __name__ == "__main__":
    main()
