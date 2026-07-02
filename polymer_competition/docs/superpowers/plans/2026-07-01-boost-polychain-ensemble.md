# PolyChain Ensemble Boost — 0.896 → 0.912+ Implementation Plan

> **For agentic workers:** Execute tasks sequentially inline. Each task is independently testable.

**Goal:** Boost PolyChain ensemble from 0.896 to 0.912+ mean R² using multi-seed training, SWA, optimised hyperparameters, and XGBoost inclusion.

**Architecture:** Train 5 seeds of PolyChain with `hidden_dim=384`, `dropout=0.15`, `n_hamf_layers=3`, `n_hamf_heads=8`, each with SWA over final 10 epochs, plus 1 XGBoost model on fingerprint features — all combined in a weighted ensemble.

**Tech Stack:** PyTorch, SWA (`torch.optim.swa_utils`), XGBoost, scikit-learn

## Global Constraints

- All changes must maintain backward compatibility with existing training pipeline
- Prediction pickle files must include seed info in model_type field for ensemble disambiguation
- SWA must not break existing checkpointing / resume logic
- Config changes must use existing YAML override pattern (model_types in config.yaml)

---

### Task 1: Create Optimised PolyChain Config Variant (`polychain_boosted`)

**Files:**
- Modify: `polymer_competition/config.yaml:110-128` (add `polychain_boosted` entry)

**Interfaces:**
- Consumes: existing `config.yaml` model_types section pattern
- Produces: `polychain_boosted` config variant with `hidden_dim=384`, `dropout=0.15`, `n_hamf_heads=8`, `n_hamf_layers=3`

- [ ] **Step 1: Add `polychain_boosted` to `config.yaml`**

Add after the `polychain_wide` entry in config.yaml:

```yaml
  polychain_boosted:
    extends: "models/polychain/configs/base.yaml"
    overrides:
      hidden_dim: 384
      n_backbone_layers: 6
      n_hamf_layers: 3
      n_hamf_heads: 8
      dropout: 0.15
      weight_decay: 1.0e-5
```

- [ ] **Step 2: Verify config loads**

Run: `python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(list(c['model_types'].keys()))"`
Expected: output includes `polychain_boosted`

---

### Task 2: Add `--seed` flag and Seed-Aware Predictions to `train.py`

**Files:**
- Modify: `polymer_competition/training/train.py:984-1012` (argparse), `train.py:1058-1068` (seed usage), `train.py:1335-1501` (PolyChain training + prediction save), `train.py:1497-1501` (prediction filename)

**Interfaces:**
- Consumes: existing CLI args + new `--seed` (optional int, overrides config seed)
- Produces: prediction pickles with seed suffix in filename and model_type

- [ ] **Step 1: Add `--seed` argument**

In the argparse section, add:
```python
parser.add_argument("--seed", type=int, default=None,
                    help="Override global seed (for multi-seed training)")
```

- [ ] **Step 2: Use the seed if provided**

After `seed = cfg.get("seed", {}).get("global", 42)` ~line 1066, add:
```python
if args.seed is not None:
    seed = args.seed
```

- [ ] **Step 3: Update PolyChain prediction filename to include seed**

Around line 1401 (`ckpt_tag` definition within the polychain block), change to:
```python
seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
ckpt_tag = f"{exp_ver}_{target}_{args.model_type}{seed_suffix}_fold{args.fold}"
```

Do the same for the prediction file around line 1497 (out_file construction) and around line 1590 (test predictions).

For the model_type stored in prediction pickle (so ensemble can distinguish seeds), change the `model_type` field in the pickle dicts to include the seed:
```python
model_type_key = f"{args.model_type}_s{args.seed}" if args.seed is not None else args.model_type
```

- [ ] **Step 4: Verify with quick import test**

Run: `python -c "from training.train import main; print('import ok')"`
Expected: import succeeds

---

### Task 3: Add SWA to `train_graph()`

**Files:**
- Modify: `polymer_competition/training/train.py:804-957` (train_graph function)

**Interfaces:**
- Consumes: existing train_graph parameters + new `use_swa` from cfg (default True)
- Produces: model loaded with SWA-averaged weights if swa enabled

- [ ] **Step 1: Add SWA imports at top of train_graph section**

Inside `train_graph`, after the criterion definition, add:
```python
# SWA
use_swa = cfg.get("swa", True)
swa_start = cfg.get("swa_start", 0.75)  # fraction of epochs to start SWA
swa_lr = cfg.get("swa_lr", 1e-5)
```

- [ ] **Step 2: Initialize SWA model**

After `sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)`, add:
```python
swa_model = None
if use_swa:
    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=swa_lr)
    swa_epoch_start = int(epochs * swa_start)
```

- [ ] **Step 3: Update training loop to include SWA update**

Inside the for-epoch loop, after `sched.step()`, add:
```python
# SWA: start averaging after swa_epoch_start
if use_swa and epoch >= swa_epoch_start:
    swa_model.update_parameters(model)
    swa_scheduler.step()
else:
    sched.step()
```

Note: This means we need to move `sched.step()` to be conditional (inside the else branch or alongside SWA). Let me restructure:

```python
# Inside train_graph loop, after gradient updates:
# Replace: sched.step()
# With:
if use_swa and epoch >= swa_epoch_start:
    swa_model.update_parameters(model)
    swa_scheduler.step()
else:
    sched.step()
```

- [ ] **Step 4: Save SWA model as "best" if it surpasses standard model**

On validation, evaluate both models. In the validation block, after computing `preds`:
```python
if use_swa and swa_model is not None:
    swa_model.eval()
    swa_preds, swa_gts = [], []
    with torch.no_grad():
        for batch_dict in val_loader:
            if model_type == "polychain":
                batch_dict = move_to_device(batch_dict, device)
                pred = swa_model(batch_dict)
            else:
                batch = batch_dict.to(device)
                pred = swa_model(batch)
            swa_preds.append(pred.cpu().numpy())
            swa_gts.append(y.view(-1).cpu().numpy())
    swa_preds = np.concatenate(swa_preds)
    swa_gts = np.concatenate(swa_gts)
    swa_val_rmse = rmse(swa_gts, swa_preds)
    if swa_val_rmse < best_val_rmse:
        best_val_rmse = swa_val_rmse
        best_state = {k: v.detach().clone() for k, v in swa_model.module.state_dict().items()}
```

- [ ] **Step 5: Verify import**

Run: `python -c "from training.train import train_graph; print('import ok')"`
Expected: import succeeds

---

### Task 4: Create `run_seeded_ensemble.py`

**Files:**
- Create: `polymer_competition/training/run_seeded_ensemble.py`

**Interfaces:**
- Consumes: config.yaml, seeds list, model types
- Produces: trained models, OOF predictions, ensemble weights, final submission

- [ ] **Step 1: Create script skeleton**

```python
"""
training/run_seeded_ensemble.py

Train PolyChain with N seeds + XGBoost, build weighted ensemble.

Usage:
    python -m training.run_seeded_ensemble --config config.yaml --n_seeds 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 123, 456, 789, 101112]


def run_cmd(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  WARNING: Command failed with exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--targets", default="tg,egc",
                        help="Comma-separated target types")
    parser.add_argument("--skip_xgb", action="store_true",
                        help="Skip XGBoost training")
    parser.add_argument("--skip_polychain", action="store_true",
                        help="Skip PolyChain training")
    parser.add_argument("--skip_ensemble", action="store_true",
                        help="Skip ensemble building")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    exp_ver = cfg.get("experiment", {}).get("version", "v28")
    targets = args.targets.split(",") if args.targets else ["tg", "egc"]

    seeds = SEEDS[:args.n_seeds]
    print(f"Using seeds: {seeds}")
    print(f"Targets: {targets}")
    print(f"Folds: {args.folds}")

    if not args.skip_polychain:
        for target in targets:
            for seed in seeds:
                for fold in range(args.folds):
                    cmd = [
                        sys.executable, "-m", "training.train",
                        "--model_type", "polychain_boosted",
                        "--fold", str(fold),
                        "--config", args.config,
                        "--target", target,
                        "--seed", str(seed),
                        "--epochs", "200",
                        "--person", f"boosted_s{seed}",
                    ]
                    rc = run_cmd(
                        cmd,
                        desc=f"PolyChain boosted seed={seed} {target} (fold {fold}/{args.folds - 1})",
                    )
                    if rc != 0:
                        print(f"  FAILED: seed={seed} {target} fold {fold}")

    if not args.skip_xgb:
        for target in targets:
            for fold in range(args.folds):
                cmd = [
                    sys.executable, "-m", "training.train",
                    "--model_type", "xgb",
                    "--fold", str(fold),
                    "--config", args.config,
                    "--target", target,
                    "--person", "boosted_xgb",
                ]
                rc = run_cmd(
                    cmd,
                    desc=f"XGBoost {target} (fold {fold}/{args.folds - 1})",
                )
                if rc != 0:
                    print(f"  FAILED: xgb {target} fold {fold}")

    if not args.skip_ensemble:
        for target in targets:
            cmd = [
                sys.executable, "-m", "ensemble.build_ensemble",
                "--config", args.config,
                "--target", target,
            ]
            run_cmd(cmd, desc=f"Build Ensemble for {target}")

        # Merge submissions
        run_cmd(
            [sys.executable, "-m", "data.merge_submissions",
             "--config", args.config],
            desc="Merge tg + egc submissions",
        )

    print("\n=== Seeded ensemble complete ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

Run: `python -c "from training.run_seeded_ensemble import main; print('ok')"`
Expected: import succeeds

---

### Task 5: Update Ensemble Loading for Seed-Aware Model Types

**Files:**
- Modify: `polymer_competition/ensemble/weight_optimizer.py:34-49` (load_oof_predictions)

**Interfaces:**
- Consumes: prediction pickle files with seed-aware names
- Produces: model_type with seed suffix parsed correctly

- [ ] **Step 1: Update `load_oof_predictions` to handle seed-aware filenames**

The current code at line 34-49 does:
```python
parts = pkl_path.stem.split("_")
model = parts[2]
```

For `v28_tg_polychain_boosted_s42_fold0.pkl` → parts = `['v28', 'tg', 'polychain', 'boosted', 's42', 'fold0']`, model = `parts[2]` = `polychain`. That's wrong - it should be `polychain_boosted_s42`.

Actually wait. Let me check the prediction file naming. Looking at config.yaml:
```yaml
  polychain_boosted:
    extends: "models/polychain/configs/base.yaml"
```

When this is used, `args.model_type` = `polychain_boosted`. The prediction filename is:
```
{exp_ver}_{target}_{model_type}_{seed_suffix}fold{fold}.pkl
```
Which would be: `v28_tg_polychain_boosted_s42_fold0.pkl`

Parts = `['v28', 'tg', 'polychain', 'boosted', 's42', 'fold0']`
parts[2] = `polychain` - that's wrong.

I need to fix `load_oof_predictions` to handle this. The issue is that `polychain_boosted` has an underscore in it, so splitting on underscore breaks.

The simplest fix is to look at the pattern more carefully:
- Format: `{exp_ver}_{target}_{model_type}_{seed_suffix}fold{fold}.pkl`
- The fold tag is the last part and always starts with `fold`
- The seed suffix (if present) is `s{seed}`
- So model_type is parts between target and seed_suffix/fold

Better approach: extract model_type from the pickle data (like `build_ensemble.py` does), not from parsing the filename.

Let me update `load_oof_predictions` to read model_type from the pickle data.

- [ ] **Step 2: Modify `load_oof_predictions`**

Replace the filename parsing approach with reading from pickle data:

```python
def load_oof_predictions(pred_dir, target, exp_ver="v1"):
    import pickle
    pred_dir = Path(pred_dir)
    models = []
    oof_dict = {}
    for pkl_path in sorted(pred_dir.glob(f"{exp_ver}_{target}_*_fold*.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if pkl_path.stem.endswith("_test"):
            continue
        if "pred" not in data or "y" not in data:
            continue
        model = data.get("model_type", "unknown")
        fold = data.get("fold", 0)
        if model not in oof_dict:
            oof_dict[model] = {"preds": {}, "targets": {}}
        oof_dict[model]["preds"][fold] = np.array(data["pred"])
        oof_dict[model]["targets"][fold] = np.array(data["y"])
    return oof_dict
```

- [ ] **Step 3: Verify**

Run: `python -c "from ensemble.weight_optimizer import load_oof_predictions; print('ok')"`
Expected: import succeeds

---

### Task 6: Smoke Test

- [ ] **Step 1: Run a single-fold test**

Run: `python -m training.run_seeded_ensemble --config config.yaml --n_seeds 2 --folds 1 --targets tg --skip_ensemble`
Expected: Runs PolyChain boosted with 2 seeds and XGBoost for fold 0, target tg

- [ ] **Step 2: Verify predictions exist**

Run: `Get-ChildItem -Path predictions/ -Filter "v28_tg_polychain_boosted_s*" | Select-Object Name`
Expected: Shows prediction files with seed info

- [ ] **Step 3: Build ensemble**

Run: `python -m ensemble.build_ensemble --config config.yaml --target tg`
Expected: Loads predictions, prints weights, saves submission
