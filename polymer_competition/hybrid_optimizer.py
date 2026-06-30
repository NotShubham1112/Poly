"""Hybrid ensemble: align v27/v28 OOF predictions by val_idx, then optimize."""
import pickle
import numpy as np
from ensemble.weight_optimizer import r2_score, objective
from scipy.optimize import minimize


def load_oof_with_idx(pred_dir, target, exp_ver, model):
    """Load OOF predictions and return dict keyed by (fold, val_idx)."""
    data = {}
    for fold in range(5):
        path = f"{pred_dir}/{exp_ver}_{target}_{model}_fold{fold}.pkl"
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            preds = np.array(d["pred"])
            y = np.array(d["y"])
            val_idx = np.array(d.get("val_idx", list(range(len(preds)))))
            data[fold] = {"pred": preds, "y": y, "val_idx": val_idx}
        except FileNotFoundError:
            pass
    return data


def align_and_stack(all_model_data):
    """Align all models by val_idx per fold, then stack.
    
    Returns (all_preds_matrix, all_y_vector, common_val_indices) or (None, None, None).
    """
    all_preds = []
    all_y = []
    all_val_idx = []
    
    for fold in range(5):
        # Find common val_idx across all models for this fold
        common_idx = None
        for name, model_data in all_model_data.items():
            if fold not in model_data:
                return None, None, None
            idx = set(model_data[fold]["val_idx"].tolist())
            if common_idx is None:
                common_idx = idx
            else:
                common_idx = common_idx & idx
        
        if not common_idx:
            return None, None, None
        
        common_idx = sorted(common_idx)
        
        # Build prediction matrix for common samples
        fold_preds = []
        fold_y = None
        for name, model_data in all_model_data.items():
            val_idx = model_data[fold]["val_idx"]
            pred = model_data[fold]["pred"]
            y = model_data[fold]["y"]
            
            # Map val_idx to position
            idx_to_pos = {int(v): i for i, v in enumerate(val_idx)}
            positions = [idx_to_pos[int(c)] for c in common_idx]
            
            fold_preds.append(pred[positions])
            if fold_y is None:
                fold_y = y[positions]
        
        all_preds.append(np.column_stack(fold_preds))
        all_y.append(fold_y)
        all_val_idx.extend(common_idx)
    
    return np.vstack(all_preds), np.concatenate(all_y), all_val_idx


def main():
    pred_dir = "predictions"
    
    # Load all models
    all_models = {}
    
    # v27 GNNs
    for m in ["gcn", "gat", "mpnn"]:
        all_models[f"v27_{m}"] = load_oof_with_idx(pred_dir, "egc", "v27", m)
    
    # v28 trees
    for m in ["xgb", "lgb", "catboost", "rf", "mlp"]:
        all_models[f"v28_{m}"] = load_oof_with_idx(pred_dir, "egc", "v28", m)
    
    # v27 trees
    for m in ["xgb", "lgb", "catboost", "rf", "mlp"]:
        all_models[f"v27_tree_{m}"] = load_oof_with_idx(pred_dir, "egc", "v27", m)
    
    print("=== EGC Hybrid Ensemble (Aligned by val_idx) ===\n")
    
    combos = [
        ("v27 trees only", {k: v for k, v in all_models.items() if k.startswith("v27_tree_")}),
        ("v28 trees only", {k: v for k, v in all_models.items() if k.startswith("v28_")}),
        ("v27 GNNs only", {k: v for k, v in all_models.items() if k.startswith("v27_") and "tree" not in k}),
        ("v27 GNNs + v28 trees", {k: v for k, v in all_models.items() if k.startswith("v27_") and "tree" not in k or k.startswith("v28_")}),
        ("v27 GNNs + v27 trees", {k: v for k, v in all_models.items() if k.startswith("v27_")}),
        ("v27 GNNs + v27 trees + v28 trees (FULL HYBRID)", all_models),
    ]
    
    best_name = None
    best_r2 = -999
    best_w = None
    best_names = None
    
    for name, model_dict in combos:
        all_preds, all_y, _ = align_and_stack(model_dict)
        if all_preds is None:
            print(f"{name}: COULD NOT ALIGN\n")
            continue
        
        n = all_preds.shape[1]
        x0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(objective, x0, args=(all_preds, all_y),
                          method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-8})
        w = result.x
        r2 = r2_score(all_y, all_preds @ w)
        
        print(f"{name}:")
        print(f"  R2 = {r2:.4f}  (n_models={n}, n_samples={len(all_y)})")
        model_names = list(model_dict.keys())
        for mn, wi in zip(model_names, w):
            if wi > 0.001:
                print(f"    {mn}: {wi:.4f}")
        print()
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_w = w
            best_names = list(model_dict.keys())
    
    print(f"=== BEST: {best_name} with R2 = {best_r2:.4f} ===")
    
    # Save best weights for use in submission
    if best_w is not None:
        import json
        weights_dict = {mn: float(wi) for mn, wi in zip(best_names, best_w) if wi > 0.001}
        with open("outputs/hybrid_egc_weights.json", "w") as f:
            json.dump({"name": best_name, "r2": best_r2, "weights": weights_dict}, f, indent=2)
        print(f"\nSaved hybrid weights to outputs/hybrid_egc_weights.json")


if __name__ == "__main__":
    main()
