"""Compare all ensemble strategies: SLSQP, Ridge stacking, LGBM stacking, CatBoost stacking.
Also tests model subset combinations."""
import pickle
import numpy as np
from pathlib import Path
from ensemble.weight_optimizer import (
    load_oof_predictions, _stack_oof, get_weights,
    stacking_meta_learner, r2_score
)

def try_combination(oof_dict, models_subset, target, n_folds=5, method="slsqp"):
    """Try a specific model subset with a specific method."""
    subset_dict = {m: oof_dict[m] for m in models_subset if m in oof_dict}
    if not subset_dict:
        return None, None

    models = list(subset_dict.keys())
    all_preds, all_y, active = _stack_oof(subset_dict, models, n_folds)
    if all_preds is None or len(active) < 2:
        return None, None

    n = all_preds.shape[1]

    if method == "slsqp":
        w = get_weights("optimize", all_preds, all_y)
        score = r2_score(all_y, all_preds @ w)
        return dict(zip(active, [float(x) for x in w])), float(f"{score:.4f}")

    elif method == "slsqp_bias":
        w = get_weights("optimize", all_preds, all_y, include_bias=True)
        n_models = len(active)
        model_w = w[:n_models]
        bias = w[n_models]
        blended = all_preds @ model_w + bias
        score = r2_score(all_y, blended)
        return dict(zip(active, [float(x) for x in model_w])), float(f"{score:.4f}")

    elif method.startswith("ridge"):
        alpha = float(method.split("_")[1]) if "_" in method else 1.0
        w = stacking_meta_learner(all_preds, all_y, learner="ridge", alpha=alpha)
        score = r2_score(all_y, all_preds @ w)
        return dict(zip(active, [float(x) for x in w])), float(f"{score:.4f}")

    elif method.startswith("lgbm"):
        w = stacking_meta_learner(all_preds, all_y, learner="lgbm")
        score = r2_score(all_y, all_preds @ w)
        return dict(zip(active, [float(x) for x in w])), float(f"{score:.4f}")

    elif method.startswith("catboost"):
        w = stacking_meta_learner(all_preds, all_y, learner="catboost")
        score = r2_score(all_y, all_preds @ w)
        return dict(zip(active, [float(x) for x in w])), float(f"{score:.4f}")

    elif method == "linear_combo":
        # Unconstrained linear combination (allows negative weights)
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=0.01, fit_intercept=True)
        m.fit(all_preds, all_y)
        blended = m.predict(all_preds)
        score = r2_score(all_y, blended)
        weights = {a: float(m.coef_[i]) for i, a in enumerate(active)}
        weights["__bias__"] = float(m.intercept_)
        return weights, float(f"{score:.4f}")

    return None, None


if __name__ == "__main__":
    for target in ["tg", "egc"]:
        print(f"\n{'='*60}")
        print(f"  {target.upper()} ENSEMBLE COMPARISON")
        print(f"{'='*60}")

        oof = load_oof_predictions("predictions", target, "v27")
        print(f"Models available: {list(oof.keys())}")

        # Core tree models (known good)
        core_trees = ["xgb", "lgb", "catboost", "rf"]
        core_plus_mlp = core_trees + ["mlp"]

        # All GNNs
        gnns = ["gcn", "gat", "mpnn"]

        # Model subsets to try
        subsets = {
            "trees_only": core_trees,
            "trees+mlp": core_plus_mlp,
            "trees+mlp+mpnn": core_plus_mlp + ["mpnn"],
            "all_v27": [m for m in oof.keys() if m not in ("10seed", "mlp_10seed")],
        }
        # Only include subsets where all models exist in oof
        subsets = {k: v for k, v in subsets.items() if all(m in oof for m in v)}

        # Methods to try
        methods = [
            "slsqp", "slsqp_bias",
            "ridge_0.01", "ridge_0.1", "ridge_1.0", "ridge_10.0",
            "lgbm",
            "linear_combo",
        ]

        best_score = -999
        best_config = None

        for sub_name, sub_models in subsets.items():
            for method in methods:
                weights, score = try_combination(oof, sub_models, target, method=method)
                if score is not None and score > best_score:
                    best_score = score
                    best_config = (sub_name, method, weights)
                    marker = " *** NEW BEST ***"
                else:
                    marker = ""
                if score is not None:
                    print(f"  {sub_name:20s} + {method:15s} => R²={score:.4f}{marker}")

        if best_config:
            sub_name, method, weights = best_config
            print(f"\n  BEST: {sub_name} + {method} => R²={best_score:.4f}")
            for m, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
                print(f"    {m}: {w:.4f}")
