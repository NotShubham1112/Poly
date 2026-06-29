import pandas as pd
import numpy as np


def compute_fingerprint_descriptor_interactions(
    fp_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    top_k: int = 30,
) -> pd.DataFrame:
    """Compute top interactions between fingerprints and descriptors."""
    interactions = {}

    fp_cols = [c for c in fp_df.columns if fp_df[c].sum() > 0][:top_k]
    desc_cols = desc_df.columns[:top_k]

    for fp_col in fp_cols:
        for desc_col in desc_cols:
            if fp_df[fp_col].std() > 0 and desc_df[desc_col].std() > 0:
                name = f"fp_x_{desc_col}"
                if name not in interactions:
                    interactions[name] = fp_df[fp_col] * desc_df[desc_col]
                else:
                    interactions[name] = (
                        interactions[name] + fp_df[fp_col] * desc_df[desc_col]
                    )

    return pd.DataFrame(interactions, index=fp_df.index)


def compute_descriptor_ratios(desc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute meaningful descriptor ratios."""
    ratios = {}

    if "MolWt" in desc_df.columns and "HeavyAtomCount" in desc_df.columns:
        ratios["mw_per_atom"] = desc_df["MolWt"] / (desc_df["HeavyAtomCount"] + 1)

    if "LogP" in desc_df.columns and "TPSA" in desc_df.columns:
        ratios["logp_tpsa_ratio"] = desc_df["LogP"] / (desc_df["TPSA"] + 1)

    if "NumHDonors" in desc_df.columns and "NumHAcceptors" in desc_df.columns:
        ratios["hbd_hba_ratio"] = desc_df["NumHDonors"] / (desc_df["NumHAcceptors"] + 1)

    if "RingCount" in desc_df.columns and "FractionCSP3" in desc_df.columns:
        ratios["ring_sp3_ratio"] = desc_df["RingCount"] / (desc_df["FractionCSP3"] + 0.01)

    if "MolLogP" in desc_df.columns and "TPSA" in desc_df.columns:
        ratios["logp_tpsa_v2"] = desc_df["MolLogP"] / (desc_df["TPSA"] + 1)

    if "NumRotatableBonds" in desc_df.columns and "RingCount" in desc_df.columns:
        ratios["rotatable_ring_ratio"] = desc_df["NumRotatableBonds"] / (
            desc_df["RingCount"] + 1
        )

    return pd.DataFrame(ratios, index=desc_df.index)
