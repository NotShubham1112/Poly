"""Hybrid ensemble analysis: compare v27/v28 model combinations for EGC."""
import pickle
import numpy as np
from ensemble.weight_optimizer import r2_score, objective
from scipy.optimize import minimize


def load_oof(pred_dir, target, exp_ver, model):
    preds_folds = {}
    for fold in range(5):
        path = f"{pred_dir}/{exp_ver}_{target}_{model}_fold{fold}.pkl"
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            preds_folds[fold] = {
                "pred": np.array(d["pred"]),
                "y": np.array(d["y"]),
            }
        except FileNotFoundError:
            pass
    return preds_folds


def stack_oof_models(model_oofs, n_folds=5):
    all_preds = []
    all_y = []
    for fold in range(n_folds):
        fold_preds = []
        fold_y = None
        ok = True
        for name, oof_dict in model_oofs.items():
            if fold not in oof_dict:
                ok = False
                break
            fold_preds.append(oof_dict[fold]["pred"])
            if fold_y is None:
                fold_y = oof_dict[fold]["y"]
            elif len(fold_y) != len(oof_dict[fold]["y"]):
                ok = False
                break
        if ok and fold_y is not None and len(fold_preds) == len(model_oofs):
            all_preds.append(np.column_stack(fold_preds))
            all_y.append(fold_y)
    if all_preds:
        return np.vstack(all_preds), np.concatenate(all_y)
    return None, None


def optimize_and_score(all_preds, all_y):
    n = all_preds.shape[1]
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(objective, x0, args=(all_preds, all_y),
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-8})
    w = result.x
    r2 = r2_score(all_y, all_preds @ w)
    return w, r2


def main():
    pred_dir = "predictions"

    v27_gnns = {}
    for m in ["gcn", "gat", "mpnn"]:
        v27_gnns[m] = load_oof(pred_dir, "egc", "v27", m)

    v28_trees = {}
    for m in ["xgb", "lgb", "catboost", "rf", "mlp"]:
        v28_trees[m] = load_oof(pred_dir, "egc", "v28", m)

    v27_trees = {}
    for m in ["xgb", "lgb", "catboost", "rf", "mlp"]:
        v27_trees[m] = load_oof(pred_dir, "egc", "v27", m)

    print("=== EGC Model Inventory ===")
    print(f"  v27 GNNs: {list(v27_gnns.keys())} ({sum(len(v) for v in v27_gnns.values())} fold-files)")
    print(f"  v28 trees: {list(v28_trees.keys())} ({sum(len(v) for v in v28_trees.values())} fold-files)")
    print(f"  v27 trees: {list(v27_trees.keys())} ({sum(len(v) for v in v27_trees.values())} fold-files)")
    print()

    combos = [
        ("v27 GNNs only", v27_gnns),
        ("v28 trees only", v28_trees),
        ("v27 trees only", v27_trees),
        ("v27 GNNs + v28 trees", {**v27_gnns, **v28_trees}),
        ("v27 GNNs + v27 trees", {**v27_gnns, **v27_trees}),
        ("ALL (v27 GNNs + v27 trees + v28 trees)", {**v27_gnns, **v27_trees, **v28_trees}),
    ]

    best_name = None
    best_r2 = -999

    print("=== ENSEMBLE COMPARISON ===")
    for name, model_dict in combos:
        all_preds, all_y = stack_oof_models(model_dict)
        if all_preds is None:
            print(f"{name}: COULD NOT STACK\n")
            continue

        w, r2 = optimize_and_score(all_preds, all_y)
        print(f"{name}:")
        print(f"  R2 = {r2:.4f}  (n_models={all_preds.shape[1]})")
        model_names = list(model_dict.keys())
        for mn, wi in zip(model_names, w):
            if wi > 0.001:
                print(f"    {mn}: {wi:.4f}")
        print()

        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    print(f"=== BEST: {best_name} with R2 = {best_r2:.4f} ===")


if __name__ == "__main__":
    main()
