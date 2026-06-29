from __future__ import annotations

import argparse
import gc
import hashlib
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit import __version__ as rdkit_version
from sklearn.model_selection import GroupKFold

from .fingerprints import all_fingerprints
from .descriptors import compute_descriptors, select_descriptors_by_variance
from .custom_polymer import compute_all_custom_features
from .polymer_descriptors import compute_polymer_descriptors
from .advanced_descriptors import compute_all_advanced_features
from .interactions import compute_fingerprint_descriptor_interactions, compute_descriptor_ratios
from .preprocessing import FeaturePreprocessor
from data.split_by_target import split_by_target


def canonicalize(smiles_list: list[str]) -> list[str | None]:
    results = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            results.append(None)
        else:
            results.append(Chem.MolToSmiles(mol, canonical=True))
    return results


def load_and_normalize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        if c.lower() == "smiles":
            col_map[c] = "SMILES"
        elif c.lower() == "id":
            col_map[c] = "id"
    df = df.rename(columns=col_map)
    return df


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_config_hash(cfg: dict) -> str:
    raw = yaml.dump(cfg, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _smiles_scaffold(smiles: str) -> str:
    s = smiles.replace("*", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return s[:20]


def build_periodic_graph_features(smiles_list, n_repeats=3):
    """Extract descriptors from periodic polymer graphs (Antoniuk et al. 2022).
    
    Uses oligomer SMILES to capture chain-level properties that monomer
    graphs cannot represent.
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from .periodic_polymer import generate_oligomer_smiles
    
    records = []
    for smi in smiles_list:
        try:
            oligomer_smi = generate_oligomer_smiles(smi, n_repeats=n_repeats)
            mol = Chem.MolFromSmiles(oligomer_smi)
            if mol is None:
                records.append(_empty_periodic_record())
                continue
            
            record = {
                'periodic_mw': Descriptors.MolWt(mol),
                'periodic_logp': Descriptors.MolLogP(mol),
                'periodic_tpsa': Descriptors.TPSA(mol),
                'periodic_hbd': Descriptors.NumHDonors(mol),
                'periodic_hba': Descriptors.NumHAcceptors(mol),
                'periodic_rotatable': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'periodic_rings': rdMolDescriptors.CalcNumRings(mol),
                'periodic_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'periodic_heavy_atoms': mol.GetNumHeavyAtoms(),
                'periodic_fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
                'periodic_chain_length': n_repeats,
                'periodic_mw_per_repeat': Descriptors.MolWt(mol) / n_repeats,
                'periodic_conjugation_ratio': sum(1 for b in mol.GetBonds() if b.GetIsConjugated()) / max(mol.GetNumBonds(), 1),
                'periodic_backbone_length': mol.GetNumHeavyAtoms() / n_repeats,
            }
            records.append(record)
        except Exception:
            records.append(_empty_periodic_record())
    
    return pd.DataFrame(records)


def _empty_periodic_record():
    return {k: 0.0 for k in [
        'periodic_mw', 'periodic_logp', 'periodic_tpsa', 'periodic_hbd',
        'periodic_hba', 'periodic_rotatable', 'periodic_rings', 'periodic_aromatic_rings',
        'periodic_heavy_atoms', 'periodic_fraction_csp3', 'periodic_chain_length',
        'periodic_mw_per_repeat', 'periodic_conjugation_ratio', 'periodic_backbone_length'
    ]}


def load_gnn_embeddings(exp_ver, target, n_folds, all_ids) -> pd.DataFrame:
    """Load GNN embeddings from saved .npy files.

    Only reads val embeddings (features/embeddings/*/fold*_val.npy) for
    training feature construction. Test embeddings (*_test.npy) are saved
    by train.py for downstream inference but not consumed here.
    """
    import numpy as np
    from pathlib import Path

    emb_dicts = []
    for fold in range(n_folds):
        path = Path(f"features/embeddings/{exp_ver}_{target}/fold{fold}_val.npy")
        if path.exists():
            fold_embs = np.load(path, allow_pickle=True).item()
            emb_dicts.append(fold_embs)
    if not emb_dicts:
        return pd.DataFrame()
    all_ids_flat = []
    for item in all_ids:
        if isinstance(item, list):
            all_ids_flat.extend(item)
        else:
            all_ids_flat.append(item)
    unique_ids = sorted(set(all_ids_flat))
    if not emb_dicts[0]:
        return pd.DataFrame()
    emb_dim = len(next(iter(emb_dicts[0].values())))
    emb_matrix = np.zeros((len(unique_ids), emb_dim))
    for fold_embs in emb_dicts:
        for i, uid in enumerate(unique_ids):
            if str(uid) in fold_embs:
                emb_matrix[i] += fold_embs[str(uid)]
    emb_matrix /= len(emb_dicts)
    emb_cols = [f"gnn_emb_{i}" for i in range(emb_dim)]
    result_df = pd.DataFrame(emb_matrix, columns=emb_cols)
    result_df["id"] = unique_ids
    return result_df


def build_features(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles_col = cfg.get("data", {}).get("smiles_col", "smiles")

    train = load_and_normalize(data_dir / "train.csv")
    test = load_and_normalize(data_dir / "test.csv")
    print(f"Loaded train={len(train)} test={len(test)}")

    all_smiles = train["SMILES"].tolist() + test["SMILES"].tolist()
    canon = canonicalize(all_smiles)
    n_invalid = sum(1 for c in canon if c is None)
    fail_policy = cfg.get("data", {}).get("fail_policy", "warn")
    if n_invalid:
        msg = f"{n_invalid} SMILES failed canonicalization"
        if fail_policy == "raise":
            raise ValueError(msg)
        elif fail_policy == "warn":
            print(f"WARNING: {msg}")
    train["canon_smiles"] = canon[: len(train)]
    test["canon_smiles"] = canon[len(train):]

    unique_smiles = sorted(set(s for s in canon if s is not None))
    print(f"Building features on {len(unique_smiles)} unique canonical SMILES")
    fps = all_fingerprints(unique_smiles)
    desc = compute_descriptors(unique_smiles)
    desc = select_descriptors_by_variance(desc)
    cust = compute_all_custom_features(unique_smiles)
    poly_desc = [compute_polymer_descriptors(s) for s in unique_smiles]
    poly_df = pd.DataFrame(poly_desc).add_prefix("polymer_")
    
    # Build periodic graph features
    print("Building periodic graph features...")
    periodic_features = build_periodic_graph_features(unique_smiles, n_repeats=3)
    print(f"  Added {periodic_features.shape[1]} periodic graph features")

    # Build advanced polymer features
    print("Building advanced polymer features...")
    advanced_features = []
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        feat = compute_all_advanced_features(mol)
        advanced_features.append(feat)
    advanced_df = pd.DataFrame(advanced_features)
    print(f"  Added {advanced_df.shape[1]} advanced polymer features")

    # Load GNN embeddings if available
    exp_ver = cfg.get("experiment", {}).get("version", "v1")
    gnn_emb_df = load_gnn_embeddings(exp_ver, "tg", cfg["cv"]["n_folds"], train["id"].tolist())
    if not gnn_emb_df.empty:
        id_to_canon = dict(zip(train["id"].astype(str), train["canon_smiles"]))
        gnn_emb_df["canon_smiles"] = gnn_emb_df["id"].astype(str).map(id_to_canon)
        gnn_emb_df = gnn_emb_df.dropna(subset=["canon_smiles"])
        gnn_cols = [c for c in gnn_emb_df.columns if c not in ("id", "canon_smiles")]
        gnn_emb_by_smiles = gnn_emb_df.set_index("canon_smiles").reindex(unique_smiles).fillna(0.0).reset_index(drop=True)
        print(f"  Added {len(gnn_cols)} GNN embedding features")
    else:
        gnn_emb_by_smiles = None

    fp_dfs = {}
    for name, arr in fps.items():
        cols = [f"{name}_{i}" for i in range(arr.shape[1])]
        fp_dfs[name] = pd.DataFrame(arr, columns=cols)
    desc_df = desc.drop(columns=["SMILES"], errors="ignore")
    cust_df = cust.drop(columns=["SMILES"], errors="ignore")

    # Build interaction features
    print("Building interaction features...")
    all_fp_df = pd.concat(fp_dfs.values(), axis=1).reset_index(drop=True)
    interactions_df = compute_fingerprint_descriptor_interactions(all_fp_df, desc_df.reset_index(drop=True), top_k=30)
    ratios_df = compute_descriptor_ratios(desc_df.reset_index(drop=True))
    print(f"  Added {interactions_df.shape[1]} interaction features, {ratios_df.shape[1]} descriptor ratios")

    # Free intermediate DataFrames before building the large cache
    del fps, desc, cust
    gc.collect()

    concat_parts = (
        [pd.DataFrame({"canon_smiles": unique_smiles})]
        + [df.astype(np.float32) for df in fp_dfs.values()]
        + [desc_df.reset_index(drop=True).astype(np.float32)]
        + [cust_df.reset_index(drop=True).astype(np.float32)]
        + [poly_df.astype(np.float32)]
        + [periodic_features.astype(np.float32)]
        + [advanced_df.astype(np.float32)]
        + [interactions_df.astype(np.float32)]
        + [ratios_df.astype(np.float32)]
    )
    if gnn_emb_by_smiles is not None:
        concat_parts.append(gnn_emb_by_smiles.astype(np.float32))
    cache_df = pd.concat(concat_parts, axis=1)

    del fp_dfs, desc_df, cust_df, poly_df
    gc.collect()

    num_cols = []
    for c in cache_df.columns:
        if c == "canon_smiles":
            continue
        col = cache_df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        if col.dtype != object:
            num_cols.append(c)
    # Replace inf with NaN before imputation (inf can't be cast to float32)
    for col in num_cols:
        col_vals = cache_df[col].values
        if np.isinf(col_vals).any():
            cache_df[col] = np.where(np.isinf(col_vals), np.nan, col_vals)
    # Apply FeaturePreprocessor for imputation, cleaning, and MI-based feature selection
    y_train_combined = train["target"].values
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(cache_df.iloc[:len(train)][num_cols], y=y_train_combined)
    cache_df[num_cols] = preprocessor.transform(cache_df[num_cols], scale=False)

    canon_to_idx = {s: i for i, s in enumerate(cache_df["canon_smiles"].values)}

    def lookup_features(smiles_list, id_vals, id_col="id"):
        indices = []
        valid_smiles = []
        valid_ids = []
        for smi, id_val in zip(smiles_list, id_vals):
            if smi is None or smi not in canon_to_idx:
                continue
            indices.append(canon_to_idx[smi])
            valid_smiles.append(smi)
            valid_ids.append(id_val)
        result = cache_df.iloc[indices].copy()
        result["SMILES"] = valid_smiles
        result[id_col] = valid_ids
        return result

    train_feat = lookup_features(
        train["canon_smiles"].values,
        train["id"].values if "id" in train.columns else range(len(train)),
    )
    test_feat = lookup_features(
        test["canon_smiles"].values,
        test["id"].values,
    )

    # Convert to float32 before saving (halves parquet size and downstream memory)
    for col in train_feat.select_dtypes(include=["float64"]).columns:
        train_feat[col] = train_feat[col].astype(np.float32)
    for col in test_feat.select_dtypes(include=["float64"]).columns:
        test_feat[col] = test_feat[col].astype(np.float32)
    train_feat.to_parquet(out_dir / "features_train.parquet", index=False)
    test_feat.to_parquet(out_dir / "features_test.parquet", index=False)
    print(f"Train features: {train_feat.shape}, Test features: {test_feat.shape}")

    meta = {
        "feature_version": cfg.get("experiment", {}).get("version", "v1"),
        "git_commit": get_git_commit(),
        "config_hash": get_config_hash(cfg),
        "rdkit_version": rdkit_version,
        "n_unique_smiles": len(unique_smiles),
        "n_invalid_smiles": n_invalid,
        "n_train_rows": len(train_feat),
        "n_test_rows": len(test_feat),
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)
    print(f"Cache metadata -> {out_dir / 'metadata.yaml'}")

    split_by_target(data_dir / "train.csv", data_dir / "test.csv", data_dir,
              targets=list(cfg["targets"].keys()))

    for t_name, t_cfg in cfg["targets"].items():
        scaffold_path = data_dir / f"splits_{t_name}_scaffold.pkl"
        if scaffold_path.exists():
            print(f"Using scaffold-aware splits for {t_name} from training/splits.py")
            continue

        t_dir = data_dir / t_name
        t_train = pd.read_csv(t_dir / "train.csv")
        t_train = t_train.rename(columns={smiles_col: "SMILES"})
        scaffolds = t_train["SMILES"].apply(_smiles_scaffold).values
        gkf = GroupKFold(n_splits=cfg["cv"]["n_folds"])
        splits = {}
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(t_train, groups=scaffolds)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
        with open(data_dir / f"splits_{t_name}.pkl", "wb") as f:
            pickle.dump(splits, f)
        print(f"splits_{t_name}.pkl: {len(splits)} folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    build_features(args.config)
