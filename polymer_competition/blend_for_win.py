import pandas as pd
import pickle
import glob
import numpy as np
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ARCH_A_CSV = r"D:\Parth\Poly\submission (2).csv"
TEST_CSV = "data/test.csv"
PRED_DIR = "predictions"
OUTPUT = "submission_champion.csv"

WEIGHT_ARCH = 0.65
WEIGHT_POLY = 1.0 - WEIGHT_ARCH

# ---------------------------------------------------------------------------
# 1. Load test metadata (full 4115 rows with id + target_type)
# ---------------------------------------------------------------------------
test = pd.read_csv(TEST_CSV)
all_ids = test["id"].values
is_tg = (test["target_type"] == "tg").values
is_egc = (test["target_type"] == "egc").values
print(f"[1] Test set: {len(test)} total ({is_tg.sum()} TG, {is_egc.sum()} EGC)")

# ---------------------------------------------------------------------------
# 2. Average PolyChain TG predictions
# ---------------------------------------------------------------------------
tg_files = sorted(glob.glob(os.path.join(PRED_DIR, "v28_tg_polychain_boosted_s*_fold*_test.pkl")))
print(f"[2] TG files found: {len(tg_files)}")

if len(tg_files) == 0:
    raise FileNotFoundError("No TG PolyChain test files found. Run ensemble first.")

# Stack all predictions
tg_pred_list = []
tg_ids = None
for f in tg_files:
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    if tg_ids is None:
        tg_ids = d["id"]
    else:
        assert list(d["id"]) == list(tg_ids), f"ID mismatch in {f}"
    tg_pred_list.append(np.array(d["pred"]))

avg_tg = np.mean(tg_pred_list, axis=0)
print(f"    Averaged {len(tg_pred_list)} TG predictions, shape={avg_tg.shape}")

# ---------------------------------------------------------------------------
# 3. Average PolyChain EGC predictions (may be incomplete)
# ---------------------------------------------------------------------------
egc_files = sorted(glob.glob(os.path.join(PRED_DIR, "v28_egc_polychain_boosted_s*_fold*_test.pkl")))
print(f"[3] EGC files found: {len(egc_files)} / 25 expected")

if len(egc_files) == 25:
    egc_pred_list = []
    egc_ids = None
    for f in egc_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        if egc_ids is None:
            egc_ids = d["id"]
        else:
            assert list(d["id"]) == list(egc_ids), f"ID mismatch in {f}"
        egc_pred_list.append(np.array(d["pred"]))
    avg_egc = np.mean(egc_pred_list, axis=0)
    print(f"    Averaged {len(egc_pred_list)} EGC predictions, shape={avg_egc.shape}")
else:
    print("    WARNING: EGC ensemble not complete. EGC predictions will be from Arch A only.")
    avg_egc = None

# ---------------------------------------------------------------------------
# 4. Build full PolyChain submission (4115 rows)
# ---------------------------------------------------------------------------
poly_full = np.full(len(all_ids), np.nan)

# Map TG predictions
tg_id_to_idx = {tid: i for i, tid in enumerate(tg_ids)}
for i, pid in enumerate(all_ids):
    if is_tg[i] and pid in tg_id_to_idx:
        poly_full[i] = avg_tg[tg_id_to_idx[pid]]

# Map EGC predictions
if avg_egc is not None:
    egc_id_to_idx = {eid: i for i, eid in enumerate(egc_ids)}
    for i, pid in enumerate(all_ids):
        if is_egc[i] and pid in egc_id_to_idx:
            poly_full[i] = avg_egc[egc_id_to_idx[pid]]

# Check for unmapped
n_missing = int(np.isnan(poly_full).sum())
if n_missing > 0:
    print(f"    WARNING: {n_missing} test rows have no PolyChain prediction (falling back to Arch A)")

# Fill missing with Arch A values later
# ---------------------------------------------------------------------------
# 5. Load Arch A
# ---------------------------------------------------------------------------
archA = pd.read_csv(ARCH_A_CSV)
arch_preds = archA["target"].values
print(f"[4] Arch A loaded: {len(archA)} predictions")
assert len(arch_preds) == len(all_ids), f"Arch A count ({len(arch_preds)}) != test count ({len(all_ids)})"

# ---------------------------------------------------------------------------
# 6. Blend
# ---------------------------------------------------------------------------
blended = WEIGHT_ARCH * arch_preds + WEIGHT_POLY * poly_full

# Fill any NaN positions with Arch A prediction
nan_mask = np.isnan(blended)
if nan_mask.any():
    print(f"    Filling {nan_mask.sum()} NaN positions with Arch A predictions")
    blended[nan_mask] = arch_preds[nan_mask]

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
final = pd.DataFrame({"id": all_ids, "target": blended})
final.to_csv(OUTPUT, index=False)
print(f"[5] Saved: {OUTPUT}")
print(f"    Blend weights: Arch A = {WEIGHT_ARCH:.2f}, PolyChain = {WEIGHT_POLY:.2f}")
print(f"    PolyChain TG rows used: {len(tg_pred_list)} files")
print(f"    PolyChain EGC rows used: {len(egc_files)} files")
print(f"    NaN values after blend: {nan_mask.sum()}")
