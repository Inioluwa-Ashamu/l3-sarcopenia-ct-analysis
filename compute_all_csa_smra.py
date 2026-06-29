# -*- coding: utf-8 -*-
"""
Compute L3 Skeletal Muscle CSA (cm2) and SMRA (HU) for Manual, DL, and TS masks
in one pass, writing a single comparison CSV.

- Finds L3_index from each patient's metadata.txt
- Reads CT (original.nii.gz)
- Loads masks for each source:
    MANUAL: from MANUAL_DIR (may be single-slice or 3D)
    DL:     from DL_DIR (full 3D with only L3 slice non-zero, as saved earlier)
    TS:     from patient/.../segmentations2_tissue/*skeletal*muscle*.nii*
- Handles shape mismatches via 2D NN resampling (masks only)
- HU window for SMRA: [-29, 150]

Output: <ATTDS_DATA_ROOT>/dl_runs/comparison_all.csv
Columns:
patient_id, L3_index, px_area_cm2,
CSA_manual, SMRA_manual, npx_manual,
CSA_DL, SMRA_DL, npx_DL,
CSA_TS, SMRA_TS, npx_TS,
notes_manual, notes_DL, notes_TS
"""

import os, re, csv, json, math
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# ----------------- CONFIG -----------------
ROOT       = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))          # dataset root
MANUAL_DIR = Path(os.environ.get("ATTDS_MANUAL_DIR", "masks"))      # folder containing manual L3 masks
DL_DIR     = ROOT / "dl_preds"                    # folder with DL predicted masks
TS_DIRNAME = "segmentations2_tissue"              # subfolder inside each patient for TS masks
OUT_CSV    = ROOT / "dl_runs" / "comparison_all.csv"

HU_MIN, HU_MAX = -29, 150                         # standard muscle attenuation window
# ------------------------------------------

def extract_pid(name: str):
    m = re.search(r"(patient\s*\d+)", str(name), flags=re.IGNORECASE)
    return m.group(1).lower().replace(" ", "") if m else None

def read_l3_index(pid: str):
    meta = ROOT / pid / "metadata.txt"
    if not meta.exists(): return None
    s = meta.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"L3_index:\s*([0-9]+)", s)
    return int(m.group(1)) if m else None

def read_ct(pid: str):
    p = ROOT / pid / "original.nii.gz"
    if not p.exists():
        return None, None
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).astype(np.int16)  # (Z,Y,X)
    return arr, img

def get_spacing_mm(pid: str, ct_img: sitk.Image):
    # Prefer metadata.json if present
    meta_json = ROOT / pid / "metadata.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            sx, sy = float(meta["spacing"][0]), float(meta["spacing"][1])
            return sx, sy
        except Exception:
            pass
    # Fallback to CT header
    sx, sy = ct_img.GetSpacing()[0], ct_img.GetSpacing()[1]
    return sx, sy

def nn_resample_mask_2d(mask2d: np.ndarray, target_shape):
    """Nearest-neighbour 2D resample to target (H,W) using SimpleITK to preserve labels."""
    if mask2d.shape == target_shape:
        return (mask2d > 0).astype(np.uint8)
    itk = sitk.GetImageFromArray((mask2d > 0).astype(np.uint8))
    ref = sitk.Image(int(target_shape[1]), int(target_shape[0]), sitk.sitkUInt8)  # (X,Y)
    ref.SetOrigin((0.0, 0.0)); ref.SetSpacing((1.0, 1.0)); ref.SetDirection((1,0,0,1))
    out = sitk.Resample(itk, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out).astype(np.uint8)

def mask_slice_from_path(mask_path: Path, L: int):
    """
    Returns (mask2d, note)
    - If 3D: return slice L if in range else first slice (note logged)
    - If 2D or single-slice: return that slice
    """
    note = "ok"
    itk = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(itk)  # (Z,Y,X) or (Y,X)
    if arr.ndim == 2:
        return (arr > 0).astype(np.uint8), note
    # 3D
    Z, Y, X = arr.shape
    if 0 <= L < Z:
        return (arr[L] > 0).astype(np.uint8), note
    note = f"used_slice0_Z={Z}"
    return (arr[0] > 0).astype(np.uint8), note

def compute_csa_smra(ct2d: np.ndarray, mask2d_bin: np.ndarray, px_area_cm2: float):
    npx = int(mask2d_bin.sum())
    csa = npx * px_area_cm2
    # SMRA: mean HU within both mask and HU window
    if npx == 0:
        return csa, math.nan, npx
    hu_vals = ct2d[(mask2d_bin > 0) & (ct2d >= HU_MIN) & (ct2d <= HU_MAX)]
    smra = float(hu_vals.mean()) if hu_vals.size > 0 else math.nan
    return csa, smra, npx

def find_manual_mask(pid: str):
    # Try common patterns; override here if you keep a mapping CSV
    pats = [f"{pid}*.nii*", f"*{pid}*.nii*"]
    for pat in pats:
        hits = list(MANUAL_DIR.glob(pat))
        if hits:
            return hits[0]
    return None

def find_dl_mask(pid: str):
    pats = [f"{pid}_L3_skm_pred.nii*", f"{pid}*pred*.nii*"]
    for pat in pats:
        hits = list(DL_DIR.glob(pat))
        if hits:
            return hits[0]
    return None

def find_ts_mask(pid: str):
    d = ROOT / pid / TS_DIRNAME
    if not d.exists(): return None
    pats = ["*skeletal*muscle*.nii*", "*muscle*.nii*"]
    for pat in pats:
        hits = list(d.glob(pat))
        if hits:
            return hits[0]
    return None

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    patients = sorted([p.name for p in ROOT.iterdir() if p.is_dir() and p.name.lower().startswith("patient")])

    header = [
        "patient_id","L3_index","px_area_cm2",
        "CSA_manual","SMRA_manual","npx_manual","notes_manual",
        "CSA_DL","SMRA_DL","npx_DL","notes_DL",
        "CSA_TS","SMRA_TS","npx_TS","notes_TS"
    ]
    rows = [header]

    for pid in patients:
        L = read_l3_index(pid)
        if L is None:
            # Skip TBD/duds
            continue

        ct3d, ct_img = read_ct(pid)
        if ct3d is None:
            continue
        if not (0 <= L < ct3d.shape[0]):
            continue

        ct2d = ct3d[L]  # (Y,X)
        sx_mm, sy_mm = get_spacing_mm(pid, ct_img)
        px_area_cm2 = (sx_mm * sy_mm) / 100.0  # mm2 -> cm2

        # --- Manual ---
        m_mask_p = find_manual_mask(pid)
        CSA_m = SMRA_m = math.nan; npx_m = 0; note_m = "not_found"
        if m_mask_p:
            m2d, note_m = mask_slice_from_path(m_mask_p, L)
            if m2d.shape != ct2d.shape:
                m2d = nn_resample_mask_2d(m2d, ct2d.shape)
            CSA_m, SMRA_m, npx_m = compute_csa_smra(ct2d, (m2d > 0).astype(np.uint8), px_area_cm2)
            if npx_m == 0: note_m = (note_m + "|manual_empty") if note_m!="ok" else "manual_empty"

        # --- DL ---
        d_mask_p = find_dl_mask(pid)
        CSA_d = SMRA_d = math.nan; npx_d = 0; note_d = "not_found"
        if d_mask_p:
            d2d, note_d = mask_slice_from_path(d_mask_p, L)
            if d2d.shape != ct2d.shape:
                d2d = nn_resample_mask_2d(d2d, ct2d.shape)
            CSA_d, SMRA_d, npx_d = compute_csa_smra(ct2d, (d2d > 0).astype(np.uint8), px_area_cm2)
            if npx_d == 0: note_d = (note_d + "|dl_empty") if note_d!="ok" else "dl_empty"

        # --- TS ---
        t_mask_p = find_ts_mask(pid)
        CSA_t = SMRA_t = math.nan; npx_t = 0; note_t = "not_found"
        if t_mask_p:
            t2d, note_t = mask_slice_from_path(t_mask_p, L)
            if t2d.shape != ct2d.shape:
                t2d = nn_resample_mask_2d(t2d, ct2d.shape)
            CSA_t, SMRA_t, npx_t = compute_csa_smra(ct2d, (t2d > 0).astype(np.uint8), px_area_cm2)
            if npx_t == 0: note_t = (note_t + "|ts_empty") if note_t!="ok" else "ts_empty"

        rows.append([
            pid, L, f"{px_area_cm2:.5f}",
            f"{CSA_m:.2f}" if not math.isnan(CSA_m) else "", f"{SMRA_m:.1f}" if not math.isnan(SMRA_m) else "", npx_m, note_m,
            f"{CSA_d:.2f}" if not math.isnan(CSA_d) else "", f"{SMRA_d:.1f}" if not math.isnan(SMRA_d) else "", npx_d, note_d,
            f"{CSA_t:.2f}" if not math.isnan(CSA_t) else "", f"{SMRA_t:.1f}" if not math.isnan(SMRA_t) else "", npx_t, note_t
        ])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("Wrote", OUT_CSV)

if __name__ == "__main__":
    main()
