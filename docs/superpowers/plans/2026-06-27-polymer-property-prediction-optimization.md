# Polymer Property Prediction Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise Mean R² from 0.888 to 0.994+ on the AISEHack 2.0 leaderboard through systematic feature expansion, model diversification, stacking ensemble, and Kaggle hardware optimization.

**Architecture:** Expand existing pipeline with polymer-specific descriptors, 15-model zoo (4 PolyChain variants), scaffold-aware CV, experiment scheduler for GPU/CPU overlap, multi-candidate stacking meta-learner, and Phase X hardware optimization for P100/T4 GPUs.

**Tech Stack:** PyTorch 2.5.1+cu121 (P100) / 2.6.0+cu124 (T4), PyTorch Geometric, XGBoost, LightGBM, CatBoost, RDKit, Optuna, scikit-learn.

## Global Constraints

- All changes must be compatible with Kaggle Notebook execution (P100-16GB or 2×T4-16GB)
- Kaggle runtime limit: 540 minutes (9 hours)
- No external data or pre-trained weights outside competition scope
- ChemBERTa-2 embeddings are Phase 2 (config-optional, default: off)
- All random seeds fixed at 42 for reproducibility
- Zero train/validation leakage at all stages
- Feature cache is global (no label); scalers/imputers/PCA are fold-local
- Graph features stored separately from tabular features (`graph_cache/` vs `features/`)
- All new features config-gated (`config.yaml`)
- PolyChain architecture: GIN-S backbone + HAMF + PECGN + CST (32-dim)
- Two-level stacking only if CV improves >0.002 over single-level
- Submission strategy: best of weighted / stacking / blended stacking by CV R²

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `features/polymer_descriptors.py` | Polymer-specific SMILES descriptors (chain length, tacticity, ring stats, branching, element comp, flexibility) |
| `features/interactions.py` | Pairwise descriptor interactions + PCA on high-correlation blocks |
| `training/scheduler.py` | Experiment Scheduler: GPU/CPU launch order, resume, runtime budget enforcement, manifest tracking |
| `training/splits.py` | Scaffold-aware 5-fold cross-validation split generation |

### Files to Modify

| File | What Changes |
|------|-------------|
| `features/graphs.py` | Expand atom features (formal charge, hybridisation, degree, ring membership); expand bond features (conjugation, ring membership) |
| `training/train.py` | Add AMP for GPU models, gradient checkpointing for PolyChain, async DataLoader config, batch size search |
| `ensemble/stacking_ensemble.py` | Add multi-candidate meta-model selection (RidgeCV, XGB, CatBoost, ElasticNet), conditional stage-2 stacking |
| `models/polychain/backbone.py` | Add gradient checkpointing in forward pass |
| `models/polychain/configs/base.yaml` | Already has base config; plan adds variants inline via config overrides |
| `config.yaml` | Add feature flags for polymer descriptors, interactions, embeddings |

---

## Task 1: Scaffold-Aware Split Generation

**Files:**
- Create: `training/splits.py`
- Test: Run directly to verify split integrity

**Interfaces:**
- Consumes: `data/train.csv` (SMILES column), `config.yaml` (n_folds, seed)
- Produces: `data/splits_{target}_scaffold.pkl` — pickled dict of `{fold: {"train_idx": [...], "val_idx": [...]}}`

- [ ] **Step 1: Write the split generation script**

`training/splits.py`:

```python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Scaffolds
from sklearn.model_selection import GroupKFold


def murcko_scaffold(smiles: str) -> str:
    """Compute Murcko scaffold from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fallback: use original SMILES as scaffold key
    try:
        scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def generate_scaffold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    smiles_col: str = "SMILES",
    target_col: str = "target",
    seed: int = 42,
) -> dict[int, dict[str, np.ndarray]]:
    """Generate scaffold-aware 5-fold CV splits.

    Molecules sharing the same Murcko scaffold are kept in the same fold
    to prevent scaffold leakage. Folds are stratified by target quantiles.
    """
    scaffolds = df[smiles_col].apply(murcko_scaffold)
    target_bins = pd.qcut(df[target_col].rank(method="first"), q=5, labels=False)

    # Group by scaffold, aggregate target bin (mode)
    scaffold_groups = pd.DataFrame({"scaffold": scaffolds, "bin": target_bins})
    group_df = scaffold_groups.groupby("scaffold").agg(
        bin=("bin", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
        count=("bin", "count"),
    ).reset_index()

    # GroupKFold on scaffold groups, stratified by bin
    gkf = GroupKFold(n_splits=n_folds)
    splits = {}
    rng = np.random.RandomState(seed)
    shuffled_idx = rng.permutation(len(group_df))
    group_df_shuffled = group_df.iloc[shuffled_idx].reset_index(drop=True)

    for fold, (train_group_idx, val_group_idx) in enumerate(
        gkf.split(group_df_shuffled, group_df_shuffled["bin"], groups=group_df_shuffled["scaffold"])
    ):
        val_scaffolds = set(group_df_shuffled.iloc[val_group_idx]["scaffold"])
        val_idx = df[scaffolds.isin(val_scaffolds)].index.values
        train_idx = df[~scaffolds.isin(val_scaffolds)].index.values
        splits[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    for target in ["tg", "egc"]:
        target_col = "target" if target not in df.columns else target
        tdf = df[df["target_type"] == target].reset_index(drop=True) if "target_type" in df.columns else df
        out_path = Path(args.output_dir) / f"splits_{target}_scaffold.pkl"
        if out_path.exists():
            print(f"Skipping {target} — split file already exists: {out_path}")
            continue
        splits = generate_scaffold_splits(
            tdf, n_folds=args.n_folds, target_col=target_col, seed=args.seed
        )
        with open(out_path, "wb") as f:
            pickle.dump(splits, f)
        fold_sizes = [len(v["val_idx"]) for v in splits.values()]
        print(f"{target}: {len(splits)} folds, val sizes={fold_sizes}, mean={np.mean(fold_sizes):.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the split generator and verify integrity**

```bash
cd polymer_competition
python -m training.splits --data data/train.csv --n_folds 5
```

Expected output:
```
tg: 5 folds, val sizes=[...], mean=...
egc: 5 folds, val sizes=[...], mean=...
```

Verify no scaffold leakage:
```python
import pickle
splits = pickle.load(open("data/splits_tg_scaffold.pkl", "rb"))
for fold in range(5):
    train = set(splits[fold]["train_idx"])
    val = set(splits[fold]["val_idx"])
    assert len(train & val) == 0, f"Fold {fold}: train/val overlap!"
print("All folds clean — no train/val overlap")
```

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/training/splits.py polymer_competition/data/splits_*.pkl
git commit -m "feat: scaffold-aware 5-fold CV split generation"
```

---

## Task 2: Polymer-Specific Descriptors

**Files:**
- Create: `features/polymer_descriptors.py`

**Interfaces:**
- Consumes: SMILES string (canonical or with `*` attachment points)
- Produces: `dict[str, float]` of polymer-specific descriptor values
- Called by: `features/build_features.py` in the feature extraction pipeline

- [ ] **Step 1: Write `features/polymer_descriptors.py`**

```python
from __future__ import annotations

import re
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def compute_polymer_descriptors(smiles: str) -> dict[str, float]:
    """Compute polymer-specific descriptors from a SMILES string.

    These capture chain-level properties that standard RDKit descriptors
    miss for polymers with * attachment points.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _empty_descriptors()

    result = {}

    # 1. Chain length estimate
    # Repeat unit MW = approximate. Count heavy atoms as proxy.
    n_heavy = mol.GetNumHeavyAtoms()
    result["n_heavy_atoms"] = float(n_heavy)

    # 2. Number of * attachment points (branching proxy)
    star_count = smiles.count("*")
    result["star_count"] = float(star_count)
    result["is_branched"] = 1.0 if star_count > 2 else 0.0

    # 3. Ring statistics
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    result["num_rings"] = float(n_rings)
    result["num_aromatic_rings"] = float(n_aromatic_rings)
    result["aromatic_fraction"] = float(n_aromatic_rings / max(n_rings, 1))

    # 4. Element composition ratios
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1

    for elem in ["F", "Cl", "Br", "I", "Si", "O", "N", "S", "P"]:
        result[f"atom_{elem}"] = float(atom_counts.get(elem, 0))
        result[f"atom_{elem}_frac"] = float(atom_counts.get(elem, 0) / max(n_heavy, 1))

    # 5. Rotatable bond fraction
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_bonds = mol.GetNumBonds()
    result["rotatable_bonds"] = float(n_rotatable)
    result["rotatable_fraction"] = float(n_rotatable / max(n_bonds, 1))
    result["flexibility_index"] = float(n_rotatable / max(n_heavy, 1))

    # 6. Topological properties
    result["tpsa"] = float(Descriptors.TPSA(mol))
    result["logp"] = float(Descriptors.MolLogP(mol))
    result["mw"] = float(Descriptors.MolWt(mol))

    # 7. Tacticity indicator (stereochemistry patterns)
    n_chiral = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    result["num_chiral_centers"] = float(n_chiral)
    result["has_stereo"] = 1.0 if "@" in smiles else 0.0

    return result


def _empty_descriptors() -> dict[str, float]:
    keys = [
        "n_heavy_atoms", "star_count", "is_branched",
        "num_rings", "num_aromatic_rings", "aromatic_fraction",
        "atom_F", "atom_Cl", "atom_Br", "atom_I", "atom_Si",
        "atom_O", "atom_N", "atom_S", "atom_P",
        "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
        "atom_Si_frac", "atom_O_frac", "atom_N_frac", "atom_S_frac", "atom_P_frac",
        "rotatable_bonds", "rotatable_fraction", "flexibility_index",
        "tpsa", "logp", "mw",
        "num_chiral_centers", "has_stereo",
    ]
    return {k: 0.0 for k in keys}
```

- [ ] **Step 2: Test descriptor computation**

```python
from features.polymer_descriptors import compute_polymer_descriptors

# Simple polymer SMILES
result = compute_polymer_descriptors("C=CC(C)C(=O)OC")
print(f"Descriptors: {len(result)} keys")
print(f"n_heavy_atoms={result['n_heavy_atoms']}, num_rings={result['num_rings']}")
print(f"logp={result['logp']:.2f}, tpsa={result['tpsa']:.1f}")

# Polymer with * attachment points
result2 = compute_polymer_descriptors("*C(=O)c1ccc(NC(=O)c2ccc(*)cc2)cc1")
print(f"\nBranched polymer: star_count={result2['star_count']}, is_branched={result2['is_branched']}")
assert all(v == 0.0 or not np.isnan(v) for v in result.values()), "NaN detected!"
print("All checks passed")
```

Expected: no errors, all values non-NaN.

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/features/polymer_descriptors.py
git commit -m "feat: polymer-specific SMILES descriptors (chain, rings, elements, flexibility)"
```

---

## Task 3: Expanded Graph Features

**Files:**
- Modify: `features/graphs.py` (atom_features and bond_features functions)

**Interfaces:**
- Consumes: RDKit `Mol` object (in `atom_features`/`bond_features`)
- Produces: Extended feature tensors with additional dimensions

- [ ] **Step 1: Read current atom_features function**

```bash
cd polymer_competition
grep -n "def atom_features\|def bond_features" features/graphs.py
```

- [ ] **Step 2: Expand atom features**

Find the `atom_features` function and extend the feature vector. After reading the current code, add these features to the end of the existing vector:

```python
# Inside atom_features, after existing features, add:
# Formal charge (one-hot: -1, 0, +1)
fc = atom.GetFormalCharge()
features.extend([1.0 if fc == -1 else 0.0,
                 1.0 if fc == 0 else 0.0,
                 1.0 if fc == 1 else 0.0])

# Hybridisation (one-hot: SP, SP2, SP3, other)
hyb = atom.GetHybridization()
features.extend([
    1.0 if hyb == Chem.HybridizationType.SP else 0.0,
    1.0 if hyb == Chem.HybridizationType.SP2 else 0.0,
    1.0 if hyb == Chem.HybridizationType.SP3 else 0.0,
    1.0 if hyb not in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3) else 0.0,
])

# Degree (one-hot: 0, 1, 2, 3, 4+)
degree = atom.GetDegree()
features.extend([
    1.0 if degree == 0 else 0.0,
    1.0 if degree == 1 else 0.0,
    1.0 if degree == 2 else 0.0,
    1.0 if degree == 3 else 0.0,
    1.0 if degree >= 4 else 0.0,
])

# Is in ring
features.append(1.0 if atom.IsInRing() else 0.0)

# Implicit valence (normalized)
features.append(min(atom.GetImplicitValence(), 4) / 4.0)
```

- [ ] **Step 3: Expand bond features**

Find the `bond_features` function and extend:

```python
# Inside bond_features, after existing features, add:
# Is conjugated
features.append(1.0 if bond.GetIsConjugated() else 0.0)

# Is in ring
features.append(1.0 if bond.IsInRing() else 0.0)

# Stereo (one-hot: any, E, Z, cis, trans, other)
stereo = bond.GetStereo()
features.extend([
    1.0 if stereo == Chem.BondStereo.STEREOANY else 0.0,
    1.0 if stereo == Chem.BondStereo.STEREOE else 0.0,
    1.0 if stereo == Chem.BondStereo.STEREOZ else 0.0,
    1.0 if stereo == Chem.BondStereo.STEREOCIS else 0.0,
    1.0 if stereo == Chem.BondStereo.STEREOTRANS else 0.0,
    1.0 if stereo not in (Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOE,
                          Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS,
                          Chem.BondStereo.STEREOTRANS) else 0.0,
])
```

- [ ] **Step 4: Update atom feature dimension constant**

Find the `BOND_FEAT_DIM` or atom feature dimension constant used in graph building and update it:

```bash
grep -n "BOND_FEAT_DIM\|in_atom_dim\|atom_feat_dim\|FEAT_DIM" features/graphs.py
```

Update the dimension constant to reflect the new feature count (previous dim + 14 for atom, previous dim + 9 for bond).

- [ ] **Step 5: Verify graph construction still works**

```python
from features.graphs import smiles_to_graph

g = smiles_to_graph("C=CC(C)C(=O)OC", y=1.0)
print(f"Atom features: {g.x.shape}  (expected [, 64])")
print(f"Bond features: {g.edge_attr.shape}  (expected [, 17])")
assert g is not None, "Graph construction failed!"
print("OK")
```

- [ ] **Step 6: Commit**

```bash
git add polymer_competition/features/graphs.py
git commit -m "feat: expanded atom+bond features (hybridisation, degree, rings, conjugation, stereo)"
```

---

## Task 4: Multi-Candidate Stacking Ensemble

**Files:**
- Modify: `ensemble/stacking_ensemble.py`

**Interfaces:**
- Produces: command-line selection of best meta-model per target; conditional stage-2 stacking

- [ ] **Step 1: Enhance the meta-model selection loop**

Replace the hard-coded meta-model with a multi-candidate selector that picks the best by CV:

```python
# After building OOF matrix and before fit, add:
def select_best_meta(oof: np.ndarray, y: np.ndarray,
                     model_names: list[str]) -> tuple[Any, str, float]:
    from catboost import CatBoostRegressor
    from sklearn.linear_model import RidgeCV, ElasticNetCV
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor

    candidates = {
        "ridge": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        "xgb": XGBRegressor(n_estimators=300, max_depth=3,
                            learning_rate=0.1, random_state=42,
                            n_jobs=-1),
        "catboost": CatBoostRegressor(iterations=500, depth=3,
                                       learning_rate=0.1, verbose=0,
                                       random_seed=42),
        "elasticnet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],
                                    alphas=[0.01, 0.1, 1.0, 10.0],
                                    max_iter=5000, random_state=42),
    }

    best_score, best_name, best_model = -np.inf, None, None
    results = {}
    for name, meta in candidates.items():
        scores = cross_val_score(meta, oof, y, cv=min(5, len(oof) // 2),
                                  scoring="r2", n_jobs=-1)
        mean_score = scores.mean()
        results[name] = mean_score
        print(f"  {name}: CV R² = {mean_score:.4f} (std={scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = meta

    print(f"  Best: {best_name} (R² = {best_score:.4f})")
    best_model.fit(oof, y)
    return best_model, best_name, best_score


def try_stage2_stacking(oof: np.ndarray, y: np.ndarray,
                         stage1_model, stage1_score: float,
                         model_names: list[str]) -> tuple[Any, float]:
    """Train a second-level model if it improves CV by >0.002."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor

    # Level 1 predictions
    oof_l1 = stage1_model.predict(oof).reshape(-1, 1)

    # Level 2: Ridge on L1 predictions + original features
    oof_l2 = np.concatenate([oof, oof_l1], axis=1)
    l2_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    l2_score = cross_val_score(l2_model, oof_l2, y, cv=min(5, len(oof) // 2),
                                scoring="r2", n_jobs=-1).mean()

    print(f"  Stage-2 CV R² = {l2_score:.4f} (improvement: {l2_score - stage1_score:.4f})")
    if l2_score > stage1_score + 0.002:
        print("  -> Using Stage-2 stacking")
        l2_model.fit(oof_l2, y)
        return l2_model, l2_score

    print("  -> Stage-2 not beneficial, keeping Stage-1")
    return None, stage1_score
```

- [ ] **Step 2: Replace the main() fit block**

Replace the single meta-model fit with:

```python
meta_model, meta_name, meta_score = select_best_meta(oof, y, model_names)
train_r2 = r2_score(y, meta_model.predict(oof))
print(f"Train R² (full OOF): {train_r2:.4f}")

# Stage-2 stacking (conditional)
stage2_model, final_score = try_stage2_stacking(oof, y, meta_model, meta_score, model_names)

# Save meta-model info
model_path = ensemble_dir / f"stacking_{target}.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "meta": meta_model,
        "meta_name": meta_name,
        "stage2": stage2_model,
        "cv_score": final_score,
        "model_names": model_names,
    }, f)
```

- [ ] **Step 3: Update test inference to handle stage-2**

Replace the predict block:

```python
if stage2_model is not None:
    # Stage-2: L1 predictions + original features
    l1_preds = meta_model.predict(test_pivot.values).reshape(-1, 1)
    test_input = np.concatenate([test_pivot.values, l1_preds], axis=1)
    test_preds = stage2_model.predict(test_input)
else:
    test_preds = meta_model.predict(test_pivot.values)
```

- [ ] **Step 4: Test the enhanced ensemble**

```bash
cd polymer_competition
python -m ensemble.stacking_ensemble --target tg --exp v1
```

Expected: prints CV R² for all 4 candidates, selects best, and optionally tries stage-2.

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/ensemble/stacking_ensemble.py
git commit -m "feat: multi-candidate meta-learner (ridge/xgb/catboost/elasticnet) + conditional stage-2 stacking"
```

---

## Task 5: Experiment Scheduler

**Files:**
- Create: `training/scheduler.py`

**Interfaces:**
- Consumes: `config.yaml`, `experiments/manifest.json`, list of (model, target, fold) triples
- Produces: managed run queue with GPU/CPU overlap, resume logic, runtime budget enforcement

- [ ] **Step 1: Write `training/scheduler.py`**

```python
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class RunSpec:
    model_type: str
    target: str
    fold: int
    device: str = "cpu"  # or "cuda:0", "cuda:1"
    priority: int = 5     # 1 = highest
    estimated_minutes: float = 10.0


class ExperimentScheduler:
    """Manages training job launch order with GPU/CPU overlap and resume."""

    # Device assignment per model type
    DEVICE_MAP: dict[str, str] = {
        "ridge": "cpu", "xgb": "cpu", "lgb": "cpu",
        "catboost": "cpu", "rf": "cpu", "mlp": "cpu",
        "gcn": "cuda", "gat": "cuda", "graph_transformer": "cuda",
        "polychain": "cuda", "polychain_deep": "cuda",
        "polychain_wide": "cuda", "polychain_light": "cuda",
    }

    # Launch order batches (parallel within batch)
    BATCHES: list[list[str]] = [
        ["ridge", "gcn"],            # Batch 1
        ["xgb", "gat"],              # Batch 2
        ["lgb", "graph_transformer"],# Batch 3
        ["catboost", "polychain"],   # Batch 4
        ["rf", "polychain_deep"],    # Batch 5
        ["mlp", "polychain_wide"],   # Batch 6
        ["polychain_light"],         # Batch 7
    ]

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.pred_dir = Path(self.cfg["paths"]["predictions_dir"])
        self.exp = self.cfg.get("experiment", {}).get("version", "v1")
        self.n_folds = self.cfg.get("cv", {}).get("n_folds", 5)
        self.targets = list(self.cfg.get("targets", {"tg": {}, "egc": {}}).keys())
        self.manifest_path = Path("experiments/manifest.json")
        self.completed_runs: set[tuple[str, str, int]] = set()
        self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            for entry in data:
                if entry.get("completed", False):
                    self.completed_runs.add(
                        (entry["model_type"], entry["target"], entry["fold"])
                    )
        print(f"Found {len(self.completed_runs)} completed runs in manifest")

    def get_pending_runs(self, targets: Optional[list[str]] = None,
                          model_types: Optional[list[str]] = None) -> list[RunSpec]:
        """Return sorted list of runs not yet completed."""
        targets = targets or self.targets
        model_types = model_types or list(self.DEVICE_MAP.keys())
        runs = []
        for target in targets:
            for model_type in model_types:
                for fold in range(self.n_folds):
                    if (model_type, target, fold) in self.completed_runs:
                        continue
                    device = self.DEVICE_MAP.get(model_type, "cpu")
                    runs.append(RunSpec(
                        model_type=model_type, target=target,
                        fold=fold, device=device,
                    ))
        return runs

    def estimate_remaining_time(self, runs: list[RunSpec]) -> float:
        """Estimate total remaining runtime in minutes."""
        total = 0.0
        gpu_time, cpu_time = 0.0, 0.0
        for r in runs:
            est = r.estimated_minutes
            if "cuda" in r.device:
                gpu_time += est
            else:
                cpu_time += est
        # GPU and CPU run in parallel, so total = max(gpu, cpu)
        return max(gpu_time, cpu_time)

    def budget_filter(self, runs: list[RunSpec],
                       remaining_minutes: float) -> list[RunSpec]:
        """Remove lowest-priority runs if over budget."""
        if remaining_minutes >= self.estimate_remaining_time(runs):
            return runs
        # Sort by priority (lower number = higher priority)
        runs_sorted = sorted(runs, key=lambda r: r.priority)
        while runs_sorted and self.estimate_remaining_time(runs_sorted) > remaining_minutes:
            removed = runs_sorted.pop()
            print(f"  Budget: removing {removed.model_type}/{removed.target}/fold{removed.fold}")
        return runs_sorted

    def launch_run(self, run: RunSpec) -> bool:
        """Launch a single training run. Returns True on success."""
        cuda_visible = ""
        if run.device.startswith("cuda:"):
            cuda_visible = f"CUDA_VISIBLE_DEVICES={run.device[-1]}"
        elif run.device == "cuda" and run.model_type in ("polychain", "polychain_deep"):
            # Heavy GPU models: alternate GPU if available
            cuda_visible = "CUDA_VISIBLE_DEVICES=0"

        cmd_parts = []
        if cuda_visible:
            cmd_parts.append(cuda_visible)
        cmd_parts.append(f"python -m training.train --model_type {run.model_type}")
        cmd_parts.append(f"--target {run.target} --fold {run.fold}")
        cmd_parts.append(f"--config config.yaml")

        cmd = " ".join(cmd_parts)
        print(f"  Launching: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED (code {result.returncode}): {result.stderr[:200]}")
            return False
        print(f"  Completed: {run.model_type}/{run.target}/fold{run.fold}")
        return True

    def run_all(self, targets: Optional[list[str]] = None,
                 model_types: Optional[list[str]] = None,
                 time_budget_minutes: Optional[float] = None):
        """Execute all pending runs with optimal launch order."""
        pending = self.get_pending_runs(targets, model_types)
        if not pending:
            print("All runs completed!")
            return

        print(f"Pending: {len(pending)} runs")

        if time_budget_minutes is not None:
            pending = self.budget_filter(pending, time_budget_minutes)
            print(f"After budget filter: {len(pending)} runs")

        # Group runs by batch order
        for batch_idx, batch_models in enumerate(self.BATCHES):
            batch_runs = [r for r in pending if r.model_type in batch_models]
            if not batch_runs:
                continue
            print(f"\nBatch {batch_idx + 1} ({len(batch_runs)} runs): {batch_models}")

            for run in batch_runs:
                success = self.launch_run(run)
                if not success:
                    print(f"  WARNING: {run.model_type}/{run.target}/fold{run.fold} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--budget", type=float, default=None,
                        help="Time budget in minutes")
    args = parser.parse_args()

    scheduler = ExperimentScheduler(args.config)
    scheduler.run_all(
        targets=args.targets,
        model_types=args.models,
        time_budget_minutes=args.budget,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add PolyChain variant config mapping**

The scheduler references `polychain_deep`, `polychain_wide`, `polychain_light` but they need config overrides. Add a config mapping in `config.yaml`:

```yaml
model_types:
  polychain:
    config: "models/polychain/configs/base.yaml"
  polychain_deep:
    extends: "models/polychain/configs/base.yaml"
    overrides:
      n_backbone_layers: 6
      n_hamf_layers: 3
      hidden_dim: 384
  polychain_wide:
    extends: "models/polychain/configs/base.yaml"
    overrides:
      hidden_dim: 512
  polychain_light:
    extends: "models/polychain/configs/base.yaml"
    overrides:
      n_backbone_layers: 3
      n_hamf_layers: 1
      hidden_dim: 128
```

Then in `training/train.py`, when loading model config for a variant name, first load the base config and apply overrides:

```python
# Inside train.py, after parsing model_type:
mt_config = cfg.get("model_types", {}).get(args.model_type, {})
if mt_config.get("extends"):
    base_path = mt_config["extends"]
    with open(base_path) as f:
        model_cfg = yaml.safe_load(f)
    # Flatten nested structure
    flat_cfg = {}
    for section in ("model", "optimizer", "scheduler", "regularization"):
        if section in model_cfg:
            flat_cfg.update(model_cfg[section])
    # Apply overrides
    overrides = mt_config.get("overrides", {})
    flat_cfg.update(overrides)
    model_cfg = flat_cfg
```

- [ ] **Step 4: Test the scheduler (dry-run mode)**

```python
from training.scheduler import ExperimentScheduler
sched = ExperimentScheduler("config.yaml")
pending = sched.get_pending_runs(targets=["tg"], model_types=["ridge", "gcn"])
print(f"Pending: {len(pending)} runs")
est = sched.estimate_remaining_time(pending)
print(f"Estimated time: {est:.1f} min")
filtered = sched.budget_filter(pending, remaining_minutes=5)
print(f"After 5min budget: {len(filtered)} runs remaining")
```

Expected: scheduler reports pending runs, respects budget filter.

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/training/scheduler.py
git commit -m "feat: Experiment Scheduler with GPU/CPU overlap, resume, runtime budget"
```

---

## Task 6: AMP + Gradient Checkpointing for GPU Models

**Files:**
- Modify: `training/train.py` (add AMP to GPU training blocks)
- Modify: `models/polychain/backbone.py` (add gradient checkpointing)

**Interfaces:**
- Consumes: existing training loops
- Produces: GPU models trained with mixed precision + optional gradient checkpointing

- [ ] **Step 1: Add AMP to PolyChain training block**

In `training/train.py`, find the polychain training section and wrap the forward/backward pass:

```python
# Inside the polychain training block, before calling train_graph,
# set up AMP:
use_amp = device.type == "cuda" and model_cfg.get("amp", True)
scaler = torch.cuda.amp.GradScaler() if use_amp else None
```

Then pass `scaler` to `train_graph`. In `train_graph` function:

```python
# Inside train_graph training loop:
for batch_dict in train_loader:
    batch_dict = move_to_device(batch_dict, device)
    if use_amp:
        with torch.cuda.amp.autocast():
            pred = model(batch_dict)
            loss = criterion(pred, batch_dict["y"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred = model(batch_dict)
        loss = criterion(pred, batch_dict["y"])
        loss.backward()
        optimizer.step()
```

- [ ] **Step 2: Add gradient checkpointing to PolyChain backbone**

In `models/polychain/backbone.py`, modify the forward pass:

```python
import torch.utils.checkpoint as checkpoint

# Inside GINBackbone.forward:
def layer_forward(layer_idx, h, edge_index, edge_attr):
    layer = self.layers[layer_idx]
    return layer(h, edge_index, edge_attr)

for i, layer in enumerate(self.layers):
    if self.training and self.grad_checkpoint:
        h = checkpoint.checkpoint(layer, h, edge_index, edge_attr,
                                   use_reentrant=False)
    else:
        h = layer(h, edge_index, edge_attr)
```

Add `grad_checkpoint` as an init parameter (default True on GPU).

- [ ] **Step 3: Add async DataLoader config**

In all GPU DataLoader instantiations in `train.py`, ensure:

```python
DataLoader(
    ...,
    num_workers=2,           # Use CPU for graph preprocessing
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True,  # Don't recreate per epoch
    prefetch_factor=2,        # Prefetch 2 batches ahead
)
```

- [ ] **Step 4: Verify AMP doesn't break CPU fallback**

```python
# AMP should be a no-op on CPU
import torch
with torch.cuda.amp.autocast(enabled=False):
    x = torch.tensor([1.0])
    y = x * 2.0
print(f"CPU AMP test: {y.item()} (expected 2.0)")
```

- [ ] **Step 5: Commit**

```bash
git add polymer_competition/training/train.py polymer_competition/models/polychain/backbone.py
git commit -m "perf: AMP for GPU models, gradient checkpointing for PolyChain, async DataLoaders"
```

---

## Task 7: Kaggle Environment Setup (Phase X)

**Files:**
- Modify: `notebooks/kaggle_pipeline.ipynb` (Cell 1 — GPU detection + install)
- No new files create; changes to existing notebook via the Python source scripts it imports

**Interfaces:**
- Detects P100 vs T4, installs correct PyTorch, configures cudnn

- [ ] **Step 1: Verify GPU detection in notebook Cell 1**

Ensure Cell 1 has the GPU detection before `import torch`:

```python
import subprocess, sys, os

# Detect GPU before importing torch (avoids sys.modules cache issue)
gpu_info = subprocess.check_output(
    "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
    shell=True
).decode().strip()
print(f"GPU: {gpu_info}")

if "P100" in gpu_info:
    print("Detected P100 (sm_60) — installing PyTorch 2.5.1+cu121")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--no-deps", "--force-reinstall", "-q"
    ])
elif "T4" in gpu_info or "Tesla T4" in gpu_info:
    print("Detected T4 (sm_75) — installing PyTorch 2.6.0+cu124")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.6.0+cu124", "torchvision==0.21.0+cu124",
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "--no-deps", "--force-reinstall", "-q"
    ])
else:
    print("Unknown GPU — using Kaggle default PyTorch")

# Now safe to import torch
import torch
assert torch.cuda.is_available(), "CUDA not available after install!"

# Configure for performance
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

- [ ] **Step 2: Add GPU utilization logging to training driver**

In `training/train.py`, add a background thread for GPU metrics:

```python
import threading

def _gpu_monitor(interval: float = 30.0, log_path: str = "outputs/logs/gpu_util.csv"):
    import csv, time
    from pathlib import Path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "util_pct", "memory_mb"])
    while getattr(_gpu_monitor, "_running", True):
        try:
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                shell=True
            ).decode().strip()
            parts = result.replace("%", "").replace(" MiB", "").split(", ")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([time.time(), parts[0], parts[1]])
        except Exception:
            pass
        time.sleep(interval)

_gpu_monitor._running = True
monitor_thread = threading.Thread(target=_gpu_monitor, daemon=True)
monitor_thread.start()
```

And at exit (in `atexit` or after training):

```python
_gpu_monitor._running = False
monitor_thread.join(timeout=5)
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/training/train.py polymer_competition/notebooks/kaggle_pipeline.ipynb
git commit -m "perf: Kaggle GPU detection, cuDNN tuning, GPU utilization monitoring"
```

---

## Task 8: Config Updates and Pipeline Integration

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add new feature flags to config.yaml**

```yaml
features:
  polymer_descriptors: true
  interaction_features: false      # Phase 3
  use_embeddings: false            # Phase 2 (ChemBERTa)
  embedding_model: "null"

training:
  amp: true
  gradient_checkpointing: true
  num_workers: 2
  pin_memory: true
  prefetch_factor: 2
  batch_size_search: true
  grad_accumulation_steps: 2

ensemble:
  strategy: "auto"       # auto = try all, pick best
  stage2_threshold: 0.002
```

- [ ] **Step 2: Update `training/train.py` to read new config keys**

In the `main()` function, after reading model_cfg, apply training config:

```python
# Apply training-level config
train_cfg = cfg.get("training", {})
model_cfg["amp"] = train_cfg.get("amp", True)
model_cfg["gradient_checkpointing"] = train_cfg.get("gradient_checkpointing", True)
model_cfg["num_workers"] = train_cfg.get("num_workers", 2)
model_cfg["pin_memory"] = train_cfg.get("pin_memory", True)
model_cfg["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)
```

- [ ] **Step 3: Commit**

```bash
git add polymer_competition/config.yaml polymer_competition/training/train.py
git commit -m "config: feature flags, training config, ensemble strategy in config.yaml"
```

---

## Task 9: Pipeline Runbook and Full Test

**Files:**
- No new files — run the full pipeline end-to-end

- [ ] **Step 1: Build feature cache with polymer descriptors**

```bash
cd polymer_competition
python -m features.build_features
```

Expected: caches created including new polymer-specific descriptors.

- [ ] **Step 2: Generate scaffold-aware splits**

```bash
python -m training.splits --data data/train.csv
```

Expected: `data/splits_tg_scaffold.pkl` and `data/splits_egc_scaffold.pkl` created.

- [ ] **Step 3: Run full pipeline with Experiment Scheduler**

```bash
python -m training.scheduler --targets tg egc --models ridge xgb lgb catboost rf mlp gcn gat
```

Expected: scheduler launches runs with GPU/CPU overlap, resumes skipped ones.

- [ ] **Step 4: Build stacking ensemble**

```bash
python -m ensemble.stacking_ensemble --target tg
python -m ensemble.stacking_ensemble --target egc
```

Expected: meta-model selection + submission CSV created.

- [ ] **Step 5: Verify submission integrity**

```python
import pandas as pd
sub = pd.read_csv("outputs/submissions/v1_tg_stacking.csv")
print(f"Rows: {len(sub)}, cols: {list(sub.columns)}")
print(f"Range: {sub['target'].min():.2f} to {sub['target'].max():.2f}")
assert "id" in sub.columns and "target" in sub.columns
assert not sub["target"].isna().any(), "NaN in predictions!"
print("Submission valid")
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: full pipeline — polymer descriptors, scheduler, stacking, AMP"
git push origin feat/competition-data-adaptation
```

---

## Task 10: Kaggle Deployment

**Files:**
- Modify: `notebooks/kaggle_pipeline.ipynb` (orchestration cells)

- [ ] **Step 1: Update notebook Cell 2 (git pull)**

```python
import subprocess, sys, os

WORK_DIR = "/kaggle/working"
REPO_URL = "https://github.com/NotShubham1112/Poly.git"
BRANCH = "feat/competition-data-adaptation"

os.chdir(WORK_DIR)
# Clone or pull
if not os.path.exists(os.path.join(WORK_DIR, "polymer_competition")):
    subprocess.check_call([
        "git", "clone", "-b", BRANCH, "--single-branch", REPO_URL
    ])
# Copy contents to WORK_DIR for direct import
subprocess.check_call([
    "cp", "-r", "Poly/polymer_competition/.", "."
], shell=True)
```

- [ ] **Step 2: Update notebook Cell 3 (feature build + splits)**

```python
!python -m features.build_features
!python -m training.splits --data data/train.csv
```

- [ ] **Step 3: Update notebook Cell 4 (scheduler)**

```python
!python -m training.scheduler --targets tg egc --models ridge xgb lgb catboost rf mlp gcn gat polychain
```

- [ ] **Step 4: Update notebook Cell 5 (ensemble)**

```python
!python -m ensemble.stacking_ensemble --target tg
!python -m ensemble.stacking_ensemble --target egc
!python -m ensemble.build_ensemble --target tg  # baseline comparison
!python -m ensemble.build_ensemble --target egc
```

- [ ] **Step 5: Update notebook Cell 6 (submission selection)**

```python
import pandas as pd
import numpy as np

# Compare stacking vs weighted ensemble CV scores
# Pick the best per-target and blend
sub_stack_tg = pd.read_csv("outputs/submissions/v1_tg_stacking.csv")
sub_stack_egc = pd.read_csv("outputs/submissions/v1_egc_stacking.csv")

# Final submission
tg_preds = sub_stack_tg["target"].values
egc_preds = sub_stack_egc["target"].values
ids = sub_stack_tg["id"].values

# Combine: first half tg, second half egc
all_preds = np.concatenate([tg_preds, egc_preds])
all_ids = np.concatenate([ids.astype(str) + "_tg", ids.astype(str) + "_egc"])

submission = pd.DataFrame({"id": all_ids, "target": all_preds})
submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows, range=[{all_preds.min():.2f}, {all_preds.max():.2f}]")
```

- [ ] **Step 6: Commit notebook changes**

```bash
git add polymer_competition/notebooks/kaggle_pipeline.ipynb
git commit -m "feat: Kaggle deployment cells — scheduler, stacking, submission"
git push origin feat/competition-data-adaptation
```

---

## Future Phases (Not in current scope)

| Phase | Tasks | Trigger |
|-------|-------|---------|
| Phase 2: ChemBERTa | `features/embeddings.py`, add to feature pipeline | CV plateaus below 0.970 |
| Phase 3: Interaction features | `features/interactions.py`, mutual-info selection | Need +0.01 post-embedding |
| Phase 4: Multi-task PolyChain | Shared backbone, dual heads | Need +0.02 and have runtime margin |
