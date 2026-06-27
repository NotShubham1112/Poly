import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load_oof_predictions(pred_dir, target, exp_ver="v1"):
    import pickle
    pred_dir = Path(pred_dir)
    models = []
    oof_dict = {}
    for pkl_path in sorted(pred_dir.glob(f"{exp_ver}_{target}_*_fold*.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        parts = pkl_path.stem.split("_")
        model = parts[2]
        fold = int(parts[3].replace("fold", ""))
        if "pred_va" not in data or "y_va" not in data:
            continue
        if model not in oof_dict:
            oof_dict[model] = {"preds": {}, "targets": {}}
        oof_dict[model]["preds"][fold] = np.array(data["pred_va"])
        oof_dict[model]["targets"][fold] = np.array(data["y_va"])
    return oof_dict


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def objective(weights, preds_matrix, y_true):
    blended = preds_matrix @ weights
    return -r2_score(y_true, blended)


def optimize_weights(oof_dict, n_folds=5):
    models = list(oof_dict.keys())
    n_models = len(models)
    if n_models == 0:
        return None, None
    all_preds = []
    all_y = []
    for fold in range(n_folds):
        fold_preds = []
        for model in models:
            if fold not in oof_dict[model]["preds"]:
                break
            fold_preds.append(oof_dict[model]["preds"][fold])
        if len(fold_preds) != n_models:
            continue
        fold_preds = np.column_stack(fold_preds)
        all_preds.append(fold_preds)
        all_y.append(oof_dict[models[0]]["targets"][fold])
    all_preds = np.vstack(all_preds)
    all_y = np.concatenate(all_y)
    n = all_preds.shape[1]
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(objective, x0, args=(all_preds, all_y),
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-8})
    best_r2 = -result.fun
    weights = result.x
    return dict(zip(models, [float(w) for w in weights])), float(f"{best_r2:.4f}")


def save_weights(target, weights, score, out_dir="ensembles"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weights_{target}.json"
    payload = {"target": target, "weights": weights, "val_r2": score}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved weights -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--exp_ver", default="v1")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    targets = [args.target] if args.target else (["tg", "egc"] if args.all else ["tg", "egc"])
    for target in targets:
        print(f"\n=== Optimizing weights for {target} ===")
        oof = load_oof_predictions(args.predictions_dir, target, args.exp_ver)
        if not oof:
            print(f"  No OOF predictions found for {target}")
            continue
        print(f"  Models found: {list(oof.keys())}")
        weights, score = optimize_weights(oof, args.n_folds)
        if weights is None:
            print(f"  Could not optimize (insufficient data)")
            continue
        print(f"  Best R²: {score}")
        for model, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {model}: {w:.3f}")
        save_weights(target, weights, score)


if __name__ == "__main__":
    main()
