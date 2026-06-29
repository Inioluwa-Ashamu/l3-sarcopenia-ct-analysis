import os, re, glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
# Point these at your folders
L3_SLICES_DIR   = os.environ.get("ATTDS_L3_SLICES_DIR", "annotations/nii_for_snap")
MANUAL_MASKS_DIR= os.environ.get("ATTDS_MANUAL_DIR", "masks")

# OPTIONAL: add more mask sources for comparison contours
TS_MASKS_DIR    = None  # e.g., "anon_dig/patientXXX/segmentations2_tissue/skeletal_muscle.nii.gz"
DL_MASKS_DIR    = None  # e.g., "anon_dig/dl_preds"

OUT_DIR         = os.environ.get("ATTDS_OVERLAY_DIR", "overlays_contour")

# Filename patterns (glob). We'll match by patientXXX in the name.
SLICE_PATS = ["*L3slice*.nii*", "*.nii", "*.nii.gz"]
MAN_PATS   = ["*manual*.nii*", "patient*.nii*", "*.nii", "*.nii.gz"]
TS_PATS    = ["*skeletal*muscle*.nii*", "*muscle*.nii*", "*.nii", "*.nii.gz"]
DL_PATS    = ["*L3_skm_pred*.nii*", "*.nii", "*.nii.gz"]

HU_MIN, HU_MAX = -200, 300
FIGSIZE = (6, 6)
DPI = 160
# ---------------------------

def find_files(folder, patterns):
    if not folder: return []
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(Path(folder)/pat)))
    # unique preserve order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            out.append(f); seen.add(f)
    return out

def extract_pid(name):
    m = re.search(r"(patient\s*\d+)", str(name), flags=re.IGNORECASE)
    return m.group(1).lower().replace(" ", "") if m else None

def read_nii_2d(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X) or (Y,X)
    if arr.ndim == 3:
        arr = arr[0]  # take first slice (your L3slice.nii is already single-slice)
    return arr, img

def resample_mask_to_shape(mask_arr, target_shape):
    if mask_arr.shape == target_shape:
        return (mask_arr > 0).astype(np.uint8)
    m_itk = sitk.GetImageFromArray((mask_arr>0).astype(np.uint8))
    ref = sitk.Image(target_shape[1], target_shape[0], sitk.sitkUInt8)
    ref.SetOrigin((0.0, 0.0)); ref.SetSpacing((1.0, 1.0)); ref.SetDirection((1,0,0,1))
    res = sitk.Resample(m_itk, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(res).astype(np.uint8)

def window_ct_hu(ct, lo=HU_MIN, hi=HU_MAX):
    ct = np.clip(ct, lo, hi)
    return (ct - lo) / max(1e-6, (hi - lo))

def contour(ax, mask_bin, color="r", lw=1.8, ls="-"):
    # Use matplotlib contour at 0.5 to draw crisp outlines
    ax.contour(mask_bin.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls, antialiased=True)

def index_by_pid(files):
    m = {}
    for f in files:
        pid = extract_pid(f)
        if pid and pid not in m:
            m[pid] = f
    return m

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    slices = find_files(L3_SLICES_DIR, SLICE_PATS)
    mans   = find_files(MANUAL_MASKS_DIR, MAN_PATS)
    tss    = find_files(TS_MASKS_DIR, TS_PATS) if TS_MASKS_DIR else []
    dls    = find_files(DL_MASKS_DIR, DL_PATS) if DL_MASKS_DIR else []

    slice_map = index_by_pid(slices)
    man_map   = index_by_pid(mans)
    ts_map    = index_by_pid(tss) if tss else {}
    dl_map    = index_by_pid(dls) if dls else {}

    pids = sorted(set(slice_map) & set(man_map))  # require at least manual + slice
    print(f"Matched {len(pids)} patients with L3slice + MANUAL.")

    for pid in pids:
        sl_path = slice_map[pid]
        man_path= man_map[pid]
        ts_path = ts_map.get(pid)
        dl_path = dl_map.get(pid)

        ct2d, _ = read_nii_2d(sl_path)
        man2d,_ = read_nii_2d(man_path)
        man2d = resample_mask_to_shape(man2d, ct2d.shape)

        # Optional other contours
        ts2d = None
        if ts_path:
            ts2d,_ = read_nii_2d(ts_path)
            ts2d = resample_mask_to_shape(ts2d, ct2d.shape)

        dl2d = None
        if dl_path:
            dl2d,_ = read_nii_2d(dl_path)
            dl2d = resample_mask_to_shape(dl2d, ct2d.shape)

        if man2d.sum() == 0:
            print(f"[WARN] {pid}: manual mask empty; skipping.")
            continue

        ct_norm = window_ct_hu(ct2d)

        # Draw single image with contour(s)
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        ax = plt.axes([0,0,1,1])
        ax.imshow(ct_norm, cmap="gray", interpolation="nearest")
        # MANUAL = solid red
        contour(ax, man2d, color="r", lw=2.0, ls="-")
        # TS = dashed green
        if ts2d is not None and ts2d.sum()>0:
            contour(ax, ts2d, color="g", lw=1.6, ls="--")
        # DL = dotted blue
        if dl2d is not None and dl2d.sum()>0:
            contour(ax, dl2d, color="b", lw=1.6, ls=":")
        ax.set_axis_off()

        title = f"{pid}  (Manual=red" + (", TS=green" if ts2d is not None else "") + (", DL=blue" if dl2d is not None else "") + ")"
        plt.title(title, fontsize=9)

        out_png = str(Path(OUT_DIR)/f"{pid}_L3_contours.png")
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[OK] {pid}: {Path(out_png).name}")

    # Report missing
    missing_man = sorted(set(slice_map) - set(man_map))
    if missing_man:
        print(f"\nNo MANUAL found for: {', '.join(missing_man[:10])}{' ...' if len(missing_man)>10 else ''}")

if __name__ == "__main__":
    main()
