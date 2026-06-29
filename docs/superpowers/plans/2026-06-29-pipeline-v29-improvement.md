# Pipeline v29 Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift Mean R² from ~0.894 to ~0.905–0.910 by improving TG-specific models, completing multi-task training, and adding targeted features — all within competition rules.

**Architecture:** Add RankGauss/Yeo-Johnson transforms, topological graph invariants, and GNN embeddings as extra features; retrain all models 5-fold with multi-seed MLP ensemble; run full 80-epoch multi-task; stack with per-target meta-learners.

**Tech Stack:** Python 3.10+, PyTorch, RDKit, scikit-learn, XGBoost, LightGBM, CatBoost

## Global Constraints

- No external data — only competition-provided train.csv and test.csv
- All models must be trainable in a Kaggle notebook (no external APIs, no pretrained weights)
- Reproducible with fixed seeds (global 42)
- All changes must pass existing tests
- Must maintain backward compatibility with existing config.yaml

---

## File Structure

### Modified files:
- `features/target_transforms.py` — add yeo_johnson_transform, rank_gauss_transform
- `features/advanced_descriptors.py` — add topological graph invariants
- `models/gnn.py` — add get_embedding() to GCN, GAT, DMPNN
- `training/train.py` — add --n_seeds for MLP ensemble, add Huber loss option
- `features/build_features.py` — wire GNN embeddings into feature cache (minor)

### Created files:
- None (all changes are modifications to existing files)

### Unchanged:
- `ensemble/weight_optimizer.py` — already handles per-target
- `ensemble/build_ensemble.py` — already handles per-target
- `config.yaml` — minor additions only
- `run_multitask.py` — no changes needed

---

### Task 1: Add RankGauss and Yeo-Johnson transforms

**Files:**
- Modify: `features/target_transforms.py` (add functions, update `select_best_transform`)

**Interfaces:**
- Consumes: `select_best_transform(y)` (existing, needs updating)
- Produces: `yeo_johnson_transform(y)` returns `(transformed, inverse_fn)`, `rank_gauss_transform(y)` returns `(transformed, inverse_fn)`

- [ ] **Step 1: Read existing file**

```
Read features/target_transforms.py
```

- [ ] **Step 2: Add yeo_johnson_transform function after existing transform functions**

```python
def yeo_johnson_transform(y: np.ndarray) -> tuple:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed = pt.fit_transform(y.reshape(-1, 1)).ravel()
    def inverse_fn(y_pred):
        return pt.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    return transformed, inverse_fn
```

- [ ] **Step 3: Add rank_gauss_transform function**

```python
def rank_gauss_transform(y: np.ndarray) -> tuple:
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(y)))
    transformed = qt.fit_transform(y.reshape(-1, 1)).ravel()
    def inverse_fn(y_pred):
        return qt.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    return transformed, inverse_fn
```

- [ ] **Step 4: Update select_best_transform to include yeo-johnson**

Must maintain the existing 3-tuple return signature `(transformed, inv_func, name)`.

```python
def select_best_transform(y: np.ndarray) -> tuple:
    skew = abs(skewness(y))
    if skew > 2.0:
        candidates = [
            ("log", log_transform),
            ("boxcox", boxcox_transform),
            ("yeo_johnson", yeo_johnson_transform),
        ]
    elif skew > 1.0:
        candidates = [
            ("boxcox", boxcox_transform),
            ("yeo_johnson", yeo_johnson_transform),
        ]
    else:
        return quantile_transform(y)  # already returns 3-tuple
    best = min(
        ((name, abs(skewness(t)), t, inv) for name, fn in candidates for t, inv in [fn(y)]),
        key=lambda x: x[1]
    )
    return best[2], best[3], best[0]
```

- [ ] **Step 5: Add import for PowerTransformer and scipy.stats.skew**

```
Add: from sklearn.preprocessing import PowerTransformer, QuantileTransformer
Add: from scipy.stats import skew as skewness
```

- [ ] **Step 6: Write tests**

In `tests/test_target_transforms.py` (create if not exists):

```python
def test_yeo_johnson_roundtrip():
    y = np.random.randn(100) * 10 + 50
    y_trans, inverse = yeo_johnson_transform(y)
    y_back = inverse(y_trans)
    assert np.allclose(y, y_back, atol=1e-6)

def test_rank_gauss_roundtrip():
    y = np.random.randn(100) * 10 + 50
    y_trans, inverse = rank_gauss_transform(y)
    y_back = inverse(y_trans)
    assert np.allclose(y, y_back, atol=1e-6)

def test_select_best_includes_yeojohnson():
    y = np.random.exponential(scale=10, size=200)
    y_trans, inverse = select_best_transform(y)
    y_back = inverse(y_trans)
    assert np.allclose(y, y_back, atol=1e-6)
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_target_transforms.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add features/target_transforms.py tests/test_target_transforms.py
git commit -m "feat: add Yeo-Johnson and RankGauss target transforms"
```

---

### Task 2: Add topological graph invariants

**Files:**
- Modify: `features/advanced_descriptors.py`

**Interfaces:**
- Consumes: `compute_all_advanced_features(mol)` (existing, extend its return dict)
- Produces: Extended dict with BalabanJ, WienerIndex, Chi0n, Chi1n, Chi2n, Chi3n, Chi4n, Chi0v, Chi1v, Chi2v, Chi3v, Chi4v, Kappa1, Kappa2, Kappa3, HallKierAlpha

- [ ] **Step 1: Read existing file**

```
Read features/advanced_descriptors.py
```

- [ ] **Step 2: Add import for rdkit.Chem.Descriptors at top of file**

```python
from rdkit.Chem import Descriptors
```

- [ ] **Step 3: Add compute_topological_invariants function**

```python
def compute_topological_invariants(mol) -> dict:
    if mol is None:
        return {k: 0.0 for k in TOPOLOGICAL_KEYS}
    try:
        return {
            "balaban_j": Descriptors.BalabanJ(mol),
            "wiener_index": Descriptors.WienerIndex(mol),
            "chi0n": Descriptors.Chi0n(mol),
            "chi1n": Descriptors.Chi1n(mol),
            "chi2n": Descriptors.Chi2n(mol),
            "chi3n": Descriptors.Chi3n(mol),
            "chi4n": Descriptors.Chi4n(mol),
            "chi0v": Descriptors.Chi0v(mol),
            "chi1v": Descriptors.Chi1v(mol),
            "chi2v": Descriptors.Chi2v(mol),
            "chi3v": Descriptors.Chi3v(mol),
            "chi4v": Descriptors.Chi4v(mol),
            "kappa1": Descriptors.Kappa1(mol),
            "kappa2": Descriptors.Kappa2(mol),
            "kappa3": Descriptors.Kappa3(mol),
            "hall_kier_alpha": Descriptors.HallKierAlpha(mol),
        }
    except Exception:
        return {k: 0.0 for k in TOPOLOGICAL_KEYS}


TOPOLOGICAL_KEYS = list(compute_topological_invariants(Chem.MolFromSmiles("CCO")).keys())
```

- [ ] **Step 4: Integrate into compute_all_advanced_features**

Add at end of `compute_all_advanced_features(mol)`:
```python
topo_features = compute_topological_invariants(mol)
result.update(topo_features)
return result
```

- [ ] **Step 5: Write tests**

```python
def test_topological_invariants_produce_values():
    mol = Chem.MolFromSmiles("CCO")
    result = compute_topological_invariants(mol)
    assert result["balaban_j"] > 0
    assert result["kappa1"] > 0
    assert len(result) == len(TOPOLOGICAL_KEYS)

def test_topological_invariants_none_mol():
    result = compute_topological_invariants(None)
    assert all(v == 0.0 for v in result.values())
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_advanced_descriptors.py -v -x 2>$null; if ($?) { pytest tests/ -k "topological" -v }`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add features/advanced_descriptors.py tests/
git commit -m "feat: add topological graph invariants (Balaban, Wiener, Chi, Kappa)"
```

---

### Task 3: Add get_embedding() to GNN models

**Files:**
- Modify: `models/gnn.py`

**Interfaces:**
- Consumes: Existing GCNRegressor, GATRegressor, DMPNNRegressor
- Produces: Each class gains `get_embedding(data)` method returning pooled graph embedding tensor

- [ ] **Step 1: Read models/gnn.py**

```
Read models/gnn.py
```

- [ ] **Step 2: Add get_embedding to GCNRegressor**

In the `forward` method, the pooled embedding `g` is computed before `self.head(g)`. Refactor:

```python
def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    for conv in self.convs:
        x = conv(x, edge_index)
        x = self.activations[i](x)
        x = self.dropouts[i](x)
    g = global_add_pool(x, batch)
    return self.head(g).squeeze(-1)

def get_embedding(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    for conv in self.convs:
        x = conv(x, edge_index)
        x = self.activations[i](x)
        x = self.dropouts[i](x)
    g = global_add_pool(x, batch)
    return g  # shape: (batch_size, hidden_dim)
```

Need to store `self.activations` and `self.dropouts` as ModuleLists in `__init__`:

```python
self.convs = nn.ModuleList(conv_layers)
self.activations = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])
self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
```

**Important:** Refactor so `forward` uses the stored `self.activations` and `self.dropouts` via index rather than inline ReLU/Dropout calls. Both GATRegressor and DMPNNRegressor need the same treatment.

- [ ] **Step 3: Apply same pattern to GATRegressor**

Same refactoring: store activations/dropouts as ModuleLists, add `get_embedding`.

- [ ] **Step 4: Apply same pattern to DMPNNRegressor**

Same refactoring: store activations/dropouts as ModuleLists, add `get_embedding`.

- [ ] **Step 5: Write tests**

```python
def test_gcn_get_embedding_shape():
    model = GCNRegressor(node_feat_dim=9, hidden_dim=64, n_layers=3, dropout=0.1)
    data = next(iter(test_loader))
    emb = model.get_embedding(data)
    assert emb.shape == (data.num_graphs, 64)
```

- [ ] **Step 6: Run existing tests**

Run: `pytest tests/test_gnn.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add models/gnn.py tests/
git commit -m "feat: add get_embedding() to GCN, GAT, DMPNN"
```

---

### Task 4: Add multi-seed MLP ensemble + Huber loss

**Files:**
- Modify: `training/train.py`

**Interfaces:**
- Consumes: `--model_type mlp`, `--fold N`, existing CLI
- Produces: `--n_seeds K` (default 5) trains K MLPs with seeds 42..42+K-1, averages OOF and test predictions; `--loss huber` selects Huber loss instead of MSE

- [ ] **Step 1: Read relevant sections of training/train.py**

```
Read training/train.py (focus on MLP training section ~lines 670-709 and argparse ~lines 1200+)
```

- [ ] **Step 2: Add argparse arguments**

```python
parser.add_argument("--n_seeds", type=int, default=1, help="Number of MLP seeds for ensemble (default: 1, no ensembling)")
parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"], help="Loss function for neural models")
```

- [ ] **Step 3: Add Huber loss option**

```python
def huber_loss(pred, target, delta=1.0):
    return nn.functional.huber_loss(pred, target, delta=delta)
```

When `args.loss == "huber"`, use `huber_loss(pred, target)` instead of `nn.MSELoss()(pred, target)` in the training loop.

- [ ] **Step 4: Implement multi-seed training loop**

Wrap the existing MLP single-seed logic:

```python
if model_type == "mlp" and args.n_seeds > 1:
    oof_preds = []
    test_preds = []
    base_seed = args.seed
    for seed_offset in range(args.n_seeds):
        current_seed = base_seed + seed_offset
        set_seed(current_seed)
        # ... existing single-run MLP training code ...
        oof_preds.append(fold_oof)
        test_preds.append(fold_test)
    fold_oof = np.mean(oof_preds, axis=0)
    fold_test = np.mean(test_preds, axis=0)
```

- [ ] **Step 5: Ensure saved pickle names include seed info for multi-seed runs**

For `n_seeds > 1`, save OOF as `{exp_ver}_{target}_mlp_{n_seeds}seed_fold{k}.pkl` to distinguish from single-seed.

- [ ] **Step 6: Write tests**

```python
def test_mlp_multi_seed_produces_unique_predictions():
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    model = FingerprintMLP(in_dim=10, hidden_dims=[16, 8], out_dim=1, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(20):
        opt.zero_grad()
        loss = criterion(model(torch.from_numpy(X)), torch.from_numpy(y))
        loss.backward()
        opt.step()
    # Seed 42
    set_seed(42)
    p1 = model(torch.from_numpy(X)).detach().numpy()
    # Seed 43
    model.load_state_dict(torch.load("checkpoint_seed42.pt"))
    # Re-init with different seed
    set_seed(43)
    model2 = FingerprintMLP(in_dim=10, hidden_dims=[16, 8], out_dim=1, dropout=0.1)
    model2.load_state_dict(model.state_dict())  # same init
    # Verify different seeds produce different results
    p2 = model2(torch.from_numpy(X)).detach().numpy()
    assert p1.shape == (50, 1)
```

- [ ] **Step 7: Commit**

```bash
git add training/train.py
git commit -m "feat: add multi-seed MLP ensemble and Huber loss"
```

---

### Task 5: Wire GNN embeddings into feature pipeline

**Files:**
- Modify: `features/build_features.py` (minor addition), `training/train.py` (embedding extraction during GNN training)

**Interfaces:**
- During GNN retraining (Task 6), each fold saves `{exp_ver}_{target}_gnn_embeddings_fold{k}.pt` containing dict of `{id: embedding_vector}`
- `build_features.py` reads these saved embeddings and adds them as feature columns

- [ ] **Step 1: Add embedding extraction to GNN training loop in training/train.py**

After each GNN validation pass, call `model.get_embedding(val_data)` and store results. Add this as a function called after GNN test inference:

```python
def extract_and_save_gnn_embeddings(model, val_loader, test_loader, val_ids, test_ids, fold, exp_ver, target, device):
    model.eval()
    val_embeddings = {}
    test_embeddings = {}
    with torch.no_grad():
        for batch, id_batch in zip(val_loader, val_ids):
            batch = batch.to(device)
            emb = model.get_embedding(batch)
            for i, g_id in enumerate(id_batch):
                val_embeddings[g_id] = emb[i].cpu().numpy()
        for batch, id_batch in zip(test_loader, test_ids):
            batch = batch.to(device)
            emb = model.get_embedding(batch)
            for i, g_id in enumerate(id_batch):
                test_embeddings[g_id] = emb[i].cpu().numpy()
    out_dir = Path(f"features/embeddings/{exp_ver}_{target}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"fold{fold}_val.npy", val_embeddings)
    np.save(out_dir / f"fold{fold}_test.npy", test_embeddings)
```

Call this after the GNN test inference within each fold's training loop in `training/train.py`.

- [ ] **Step 2: Add embedding loading to build_features.py**

After existing feature computation, load saved embeddings and merge:
```python
def load_gnn_embeddings(exp_ver, target, n_folds, all_ids) -> pd.DataFrame:
    emb_dicts = []
    for fold in range(n_folds):
        path = Path(f"features/embeddings/{exp_ver}_{target}_gnn_embeddings_fold{fold}.npy")
        if path.exists():
            fold_embs = np.load(path, allow_pickle=True).item()
            emb_dicts.append(fold_embs)
    if not emb_dicts:
        return pd.DataFrame()
    # Align: for each unique id, average embeddings across folds
    all_ids_flat = [item for sublist in all_ids for item in (sublist if isinstance(sublist, list) else [sublist])]
    unique_ids = sorted(set(all_ids_flat))
    emb_dim = len(next(iter(emb_dicts[0].values()))) if emb_dicts[0] else 0
    emb_matrix = np.zeros((len(unique_ids), emb_dim))
    for fold_embs in emb_dicts:
        for i, uid in enumerate(unique_ids):
            if uid in fold_embs:
                emb_matrix[i] += fold_embs[uid]
    emb_matrix /= len(emb_dicts)
    emb_cols = [f"gnn_emb_{i}" for i in range(emb_dim)]
    return pd.DataFrame(emb_matrix, columns=emb_cols, index=unique_ids)
```

Then in `build_features()`, after periodic/advanced features, call this and append to the cache_df:
```python
gnn_emb_df = load_gnn_embeddings(exp_ver, "tg", cfg["cv"]["n_folds"], train["id"].tolist())
if not gnn_emb_df.empty:
    cache_df = pd.concat([cache_df, gnn_emb_df.astype(np.float32)], axis=1)
```

- [ ] **Step 3: Commit**

```bash
git add features/build_features.py training/train.py
git commit -m "feat: wire GNN embeddings into feature pipeline"
```

---

### Task 6: Run full pipeline

**Files:** None (execution only)

- [ ] **Step 1: Rebuild feature cache**

```bash
cd polymer_competition && python features/build_features.py --config config.yaml
```

- [ ] **Step 2: Retrain tree models (XGB, LGB, CatBoost, RF) 5-fold**

```bash
python -m training.run_all_folds --models xgb,lgb,catboost,rf --config config.yaml
```

- [ ] **Step 3: Retrain GNNs (GCN, GAT, MPNN) with embedding extraction**

```bash
python -m training.run_all_folds --models gcn,gat,mpnn --config config.yaml
```

- [ ] **Step 4: Retrain MLP ensemble (5 seeds) on new features + GNN embeddings**

```bash
python -m training.run_all_folds --models mlp --n_seeds 5 --config config.yaml
```

- [ ] **Step 5: Run full 80-epoch multi-task**

```bash
python run_multitask.py --epochs 80
```

- [ ] **Step 6: Build ensemble with per-target stacking**

```bash
python -m ensemble.build_ensemble --config config.yaml --target tg
python -m ensemble.build_ensemble --config config.yaml --target egc
```

- [ ] **Step 7: Generate submission**

```bash
python run_submission.py
```

- [ ] **Step 8: Commit results**

```bash
git add -A
git commit -m "feat: v29 pipeline - new features, MLP ensemble, full multi-task"
```
