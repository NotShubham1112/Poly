import pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from scipy.stats import pearsonr

TARGETS = ["tg", "egc"]

# All model OOF predictions
# For TG: we have XGB, GIN, Hybrid
# For EGC: we have XGB, GIN, Hybrid
# Also check if tree baselines exist

data_dir = Path("data")

for target in TARGETS:
    print(f"\n{'='*60}")
    print(f"  {target.upper()}")
    print(f"{'='*60}")
    
    tr = pd.read_csv(data_dir / "train.csv")
    mask = tr["target_type"].values == target
    y = tr["target"].values[mask].astype(np.float32)
    
    models = {}
    
    # Load GIN OOF
    gin_path = f"outputs/gin/{target}/oof_{target}_gin.pkl"
    if Path(gin_path).exists():
        d = pickle.load(open(gin_path, "rb"))
        models["GIN"] = d["pred"]
    
    # Load XGB OOF (compute on the fly)
    # Use the gin_xgb directory
    xgb_path = f"outputs/gin_xgb/{target}/oof_{target}_xgb.pkl"
    if Path(xgb_path).exists():
        d = pickle.load(open(xgb_path, "rb"))
        models["XGB"] = d["pred"]
    
    # Load Hybrid OOF
    hybrid_path = f"outputs/hybrid/{target}/oof_{target}_hybrid.pkl"
    if Path(hybrid_path).exists():
        d = pickle.load(open(hybrid_path, "rb"))
        models["Hybrid"] = d["pred"]
    
    print(f"\nModels found: {list(models.keys())}")
    
    # Individual performance
    for name, pred in models.items():
        r2 = r2_score(y, pred)
        print(f"  {name:10s}: R2={r2:.4f}")
    
    # Correlations
    model_names = list(models.keys())
    if len(model_names) >= 2:
        print("\n  Pairwise correlations:")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                n1, n2 = model_names[i], model_names[j]
                c, _ = pearsonr(models[n1], models[n2])
                print(f"    {n1} vs {n2}: r={c:.4f}")
    
    # 2-way blends
    if len(model_names) >= 2:
        print("\n  Best 2-way blends:")
        best_r2 = -1
        best_pair = None
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                n1, n2 = model_names[i], model_names[j]
                def nr2(w):
                    return -r2_score(y, w * models[n1] + (1-w) * models[n2])
                res = minimize(lambda w: nr2(w[0]), [0.5], bounds=[(0,1)], method="L-BFGS-B")
                w_opt = res.x[0]
                bl = w_opt * models[n1] + (1-w_opt) * models[n2]
                br2 = -res.fun
                print(f"    {n1}({w_opt:.3f}) + {n2}({1-w_opt:.3f}): R2={br2:.4f}")
                if br2 > best_r2:
                    best_r2 = br2
                    best_pair = (n1, n2, w_opt)
    
    # 3-way blend
    if len(model_names) >= 3:
        print("\n  Best 3-way blend:")
        def nr3(w):
            w = np.clip(w, 0, 1)
            w = w / w.sum()
            pred = sum(w[i] * models[model_names[i]] for i in range(len(model_names)))
            return -r2_score(y, pred)
        
        res = minimize(nr3, np.ones(len(model_names))/len(model_names),
                      bounds=[(0,1)]*len(model_names), method="L-BFGS-B")
        w_opt = res.x / res.x.sum()
        parts = " + ".join(f"{model_names[i]}({w_opt[i]:.3f})" for i in range(len(model_names)))
        print(f"    {parts}: R2={-res.fun:.4f}")

# Mean R2 across targets for the best blend
# First make sure we have the OOF files properly saved
print("\nPlease ensure all OOF files exist before running this.")
