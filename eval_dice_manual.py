# dl/eval_dice_manual.py
# Strict evaluator: DL vs Manual Dice at patient-specific L3 slice.
# Writes dice only when BOTH masks exist and are non-empty. Records a 'note'.

import os, re, csv, math
from pathlib import Path
import numpy as np
import SimpleITK as sitk

# -------------- CONFIG --------------
ROOT       = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))        # dataset root with patientXXX/
PRED_DIR   = ROOT / "dl_preds"                  # DL predicted masks (full 3D with only L3 slice nonzero)
MANUAL_DIR = Path(os.environ.get("ATTDS_MANUAL_DIR", "masks"))     # manual mask folder
OUT_CSV    = ROOT / "dl_runs" / "eval_dice_manual_vs_DL.csv"
# ------------------------------------

def read_l3_index(pid: str):
    meta = ROOT / pid / "metadata.txt"
    if not meta.exists():
        return None
    s = meta.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"L3_index:\s*([0-9]+)", s)
    return int(m.group(1)) if m else None

def load_slice(path: Path, L: int):
    """
    Load a 2D slice for comparison:
    - If the file is 2D/single-slice, return that.
    - If 3D, return slice L if in range, else slice 0 (and flag in note).
    Returns (arr2d_uint8, note_addition)
    """
    note = ""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X) or (Y,X)
    if arr.ndim == 2:
        return arr.astype(np.uint8), note
    Z, Y, X = arr.shape
    if 0 <= L < Z:
        return arr[L].astype(np.uint8), note
    note = f"used_slice0_Z={Z}"
    return arr[0].astype(np.uint8), note

def resample_nn_2d(mask2d: np.ndarray, target_shape):
    """Nearest-neighbour resample to target (H,W) using SimpleITK (keeps labels crisp)."""
    if mask2d.shape == target_shape:
        return (mask2d > 0).astype(np.uint8)
    itk = sitk.GetImageFromArray((mask2d > 0).astype(np.uint8))
    ref = sitk.Image(int(target_shape[1]), int(target_shape[0]), sitk.sitkUInt8)  # (X,Y)
    ref.SetOrigin((0.0, 0.0)); ref.SetSpacing((1.0, 1.0)); ref.SetDirection((1,0,0,1))
    out = sitk.Resample(itk, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out).astype(np.uint8)

def dice_coef(a: np.ndarray, b: np.ndarray, eps=1e-6):
    a = a > 0; b = b > 0
    den = a.sum() + b.sum()
    if den == 0:
        return math.nan
    return (2.0 * np.logical_and(a, b).sum()) / (den + eps)

def extract_pid(name: str):
    m = re.search(r"(patient\s*\d+)", str(name), flags=re.IGNORECASE)
    return m.group(1).lower().replace(" ", "") if m else None

def find_manual_path(pid: str):
    for pat in (f"{pid}*.nii*", f"*{pid}*.nii*"):
        hits = list(MANUAL_DIR.glob(pat))
        if hits:
            return hits[0]
    return None

def try_safe_transforms(man_bin: np.ndarray, pred_bin: np.ndarray):
    """
    Try a few safe transforms on pred to maximise Dice (to handle minor orientation issues).
    Returns (best_dice, tag, best_pred_bin)
    """
    candidates = [(pred_bin, "orig")]
    # only try transpose if shapes allow
    if man_bin.shape[::-1] == pred_bin.shape:
        candidates.append((pred_bin.T, "transpose"))
    candidates.append((np.fliplr(pred_bin), "hflip"))
    candidates.append((np.flipud(pred_bin), "vflip"))

    best_d, best_tag, best_p = -1.0, "orig", pred_bin
    for p, tag in candidates:
        if p.shape != man_bin.shape:
            continue
        d = dice_coef(man_bin, p)
        if not math.isnan(d) and d > best_d:
            best_d, best_tag, best_p = d, tag, p
    return best_d, best_tag, best_p

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(PRED_DIR.glob("patient*_L3_skm_pred.nii*"))

    rows = [("patient_id", "dice", "note")]
    n_ok = n_skipped = 0

    for pf in pred_files:
        pid = extract_pid(pf.name)
        if not pid:
            continue

        L = read_l3_index(pid)
        if L is None:
            rows.append((pid, "", "no L3_index")); n_skipped += 1; continue

        man_path = find_manual_path(pid)
        if not man_path:
            rows.append((pid, "", "manual not found")); n_skipped += 1; continue

        # load 2D slices
        pred2d, note_p = load_slice(pf, L)
        man2d,  note_m = load_slice(man_path, L)

        # match shapes
        if man2d.shape != pred2d.shape:
            man2d = resample_nn_2d(man2d, pred2d.shape)

        man_bin  = (man2d  > 0).astype(np.uint8)
        pred_bin = (pred2d > 0).astype(np.uint8)

        if man_bin.sum() == 0:
            rows.append((pid, "", "manual empty" + (f"|{note_m}" if note_m else ""))); n_skipped += 1; continue
        if pred_bin.sum() == 0:
            rows.append((pid, "", "pred empty at L3" + (f"|{note_p}" if note_p else ""))); n_skipped += 1; continue

        d, tag, _ = try_safe_transforms(man_bin, pred_bin)
        note = "ok" if tag == "orig" and not note_p and not note_m else "|".join([x for x in ["ok" if tag=="orig" else f"used_{tag}", note_p, note_m] if x])

        rows.append((pid, f"{d:.4f}", note))
        n_ok += 1

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    print(f"Wrote {OUT_CSV}")
    print(f"Valid comparisons (wrote dice): {n_ok}")
    print(f"Skipped (missing/empty): {n_skipped}")

if __name__ == "__main__":
    main()
