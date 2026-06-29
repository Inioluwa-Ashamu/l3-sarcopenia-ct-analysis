# file: apply_tissue_masks_at_L3.py
# CSA-first variant: runs with or without heights.csv.
# If height/sex are missing, SMI and sarcopenia label are left blank.

import os, re, csv, json, logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
ROOT = os.environ.get("ATTDS_DATA_ROOT", "anon_dig")
DIR_L3 = "segmentations2_L3"
DIR_TISSUE = "segmentations2_tissue"
HEIGHTS_CSV = os.path.join(ROOT, "heights.csv")   # optional; if missing, SMI is skipped
LOG_LEVEL = "INFO"

# HU filter to tighten muscle area inside mask (recommended)
HU_MIN, HU_MAX = -29, 150
SAVE_OVERLAYS = True
OVERLAY_MUSCLE = "overlay_L3_skeletal_muscle.png"
OVERLAY_ALL    = "overlay_L3_tissues.png"

# SMI cutoffs (used only when height+sex available)
CUTOFFS = {"male": 44.6, "female": 34.0}

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- HELPERS -------------------
def read_spacing(pdir):
    with open(os.path.join(pdir, "metadata.json"), "r") as f:
        m = json.load(f)
    return float(m["spacing_x_mm"]), float(m["spacing_y_mm"])

def get_l3_index_from_metadata(pdir):
    idx = None
    meta_txt = os.path.join(pdir, "metadata.txt")
    if os.path.exists(meta_txt):
        with open(meta_txt, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "L3_index" in line:
                    nums = re.findall(r"\b\d+\b", line)
                    if nums:
                        idx = int(nums[-1])
    return idx

def get_l3_index_from_mask(pdir):
    path = os.path.join(pdir, DIR_L3, "vertebrae_L3.nii.gz")
    if not os.path.exists(path):
        return None
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    areas = [int((arr[z] > 0).sum()) for z in range(arr.shape[0])]
    return int(np.argmax(areas)) if max(areas) > 0 else None

def read_slice(path, z):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if not (0 <= z < arr.shape[0]):
        raise IndexError(f"Slice {z} out of bounds for {os.path.basename(path)} (depth={arr.shape[0]})")
    return arr[z]

def first_existing(tissue_dir, names):
    for n in names:
        p = os.path.join(tissue_dir, n)
        if os.path.exists(p):
            return p
    return None

def load_heights(csv_path):
    heights, sex = {}, {}
    if not os.path.exists(csv_path): return heights, sex
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            pid = (r.get("patient_id") or "").strip()
            h = (r.get("height_m") or "").strip()
            s = (r.get("sex") or "").strip().lower()
            if pid and h:
                try: heights[pid] = float(h)
                except: pass
            if pid and s:
                sex[pid] = s
    return heights, sex

def edge_from_binary(mask):
    m = mask.astype(np.uint8)
    up, down = np.roll(m, -1, 0), np.roll(m, 1, 0)
    left, right = np.roll(m, 1, 1), np.roll(m, -1, 1)
    neigh_min = np.minimum.reduce([up, down, left, right])
    return (m > 0) & (neigh_min == 0)

def save_overlay(ct_slice_hu, layers, out_path):
    plt.figure(figsize=(5,5))
    plt.imshow(ct_slice_hu, cmap='gray')
    H, W = ct_slice_hu.shape
    over = np.zeros((H, W, 4), dtype=float)
    for m, rgba in layers:
        edges = edge_from_binary(m)
        over[edges] = rgba
    plt.imshow(over)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# -------------------- MAIN ----------------------
def main():
    heights, sexmap = load_heights(HEIGHTS_CSV)
    has_heights = bool(heights)
    if not has_heights:
        logging.info("No heights.csv found; will output CSA only (SMI/labels blank).")

    out_csv = os.path.join(ROOT, "l3_tissue_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow([
            "patient_id","L3_index","spacing_x_mm","spacing_y_mm",
            "CSA_skeletal_muscle_cm2","CSA_subcutaneous_fat_cm2",
            "CSA_visceral_fat_cm2","CSA_intermuscular_fat_cm2",
            "height_m","SMI_cm2_m2","sex","sarcopenia_label","cutoff_used"
        ])

        for patient_id in sorted(d for d in os.listdir(ROOT) if d.lower().startswith("patient")):
            pdir = os.path.join(ROOT, patient_id)
            if not os.path.isdir(pdir):
                continue

            ct_path = os.path.join(pdir, "original.nii.gz")
            l3_mask_path = os.path.join(pdir, DIR_L3, "vertebrae_L3.nii.gz")
            tissue_dir = os.path.join(pdir, DIR_TISSUE)
            if not (os.path.exists(ct_path) and os.path.exists(l3_mask_path) and os.path.isdir(tissue_dir)):
                continue

            # 1) L3 index (metadata first, fallback to mask)
            l3 = get_l3_index_from_metadata(pdir)
            if l3 is None:
                l3 = get_l3_index_from_mask(pdir)
                if l3 is None:
                    logging.warning("%s: cannot determine L3 index; skipping.", patient_id)
                    continue

            # 2) spacing & pixel area
            try:
                sx_mm, sy_mm = read_spacing(pdir)
            except Exception as e:
                logging.warning("%s: missing spacing; skipping. (%s)", patient_id, e)
                continue
            px_cm2 = (sx_mm/10.0) * (sy_mm/10.0)

            # 3) tissue files
            sm_path  = first_existing(tissue_dir, ["skeletal_muscle.nii.gz","muscle.nii.gz","skeletal_muscles.nii.gz"])
            sat_path = first_existing(tissue_dir, ["subcutaneous_fat.nii.gz"])
            vat_path = first_existing(tissue_dir, ["torso_fat.nii.gz","visceral_fat.nii.gz","intra_abdominal_fat.nii.gz"])
            imat_path= first_existing(tissue_dir, ["intermuscular_fat.nii.gz"])
            if not sm_path:
                logging.warning("%s: no skeletal_muscle mask; skipping.", patient_id)
                continue

            # 4) slice extraction
            ct_slice_hu = read_slice(ct_path, l3)
            sm_slice    = read_slice(sm_path,  l3) > 0
            sat_slice   = read_slice(sat_path, l3) > 0 if sat_path else None
            vat_slice   = read_slice(vat_path, l3) > 0 if vat_path else None
            imat_slice  = read_slice(imat_path,l3) > 0 if imat_path else None

            # 5) HU tightening inside skeletal muscle mask
            hu_ok = (ct_slice_hu >= HU_MIN) & (ct_slice_hu <= HU_MAX)
            sm_slice = sm_slice & hu_ok

            # 6) areas
            a_sm  = float(sm_slice.sum()) * px_cm2
            a_sat = float(sat_slice.sum()) * px_cm2 if sat_slice is not None else None
            a_vat = float(vat_slice.sum()) * px_cm2 if vat_slice is not None else None
            a_imat= float(imat_slice.sum()) * px_cm2 if imat_slice is not None else None

            # 7) SMI + label (only if height+sex available)
            h = heights.get(patient_id) if has_heights else None
            sex = sexmap.get(patient_id, "") if has_heights else ""
            smi = (a_sm/(h*h)) if (h and h > 0) else None
            cutoff = CUTOFFS.get(sex) if sex else None
            label = None
            if smi is not None and cutoff is not None:
                label = "sarcopenic" if smi < cutoff else "non_sarcopenic"

            # 8) write outputs
            wr.writerow([
                patient_id, l3, sx_mm, sy_mm,
                f"{a_sm:.2f}",
                (f"{a_sat:.2f}" if a_sat is not None else ""),
                (f"{a_vat:.2f}" if a_vat is not None else ""),
                (f"{a_imat:.2f}" if a_imat is not None else ""),
                (h if h else ""),
                (f"{smi:.2f}" if smi is not None else ""),
                sex, (label or ""), (cutoff if cutoff is not None else "")
            ])

            with open(os.path.join(pdir, "metrics.txt"), "a", encoding="utf-8") as mf:
                mf.write(f"SMA_L3_CSA_cm2={a_sm:.2f}\n")
                if smi is not None:
                    mf.write(f"SMA_L3_SMI_cm2_m2={smi:.2f}\n")
                if label:
                    mf.write(f"Sarcopenia_label={label} (sex={sex}, cutoff={cutoff})\n")

            # 9) overlays
            if SAVE_OVERLAYS:
                # skeletal-only
                try:
                    layers = [(sm_slice,  (1, 0, 0, 0.85))]
                    save_overlay(ct_slice_hu, layers, os.path.join(pdir, OVERLAY_MUSCLE))
                except Exception as e:
                    logging.warning("%s: muscle overlay failed (%s)", patient_id, e)
                # all tissues (for QC)
                try:
                    layers = [(sm_slice, (1, 0, 0, 0.85))]
                    if sat_slice is not None: layers.append((sat_slice, (1, 1, 0, 0.75)))
                    if vat_slice is not None: layers.append((vat_slice, (0, 0.6, 1, 0.75)))
                    if imat_slice is not None: layers.append((imat_slice,(1, 0, 1, 0.75)))
                    save_overlay(ct_slice_hu, layers, os.path.join(pdir, OVERLAY_ALL))
                except Exception as e:
                    logging.warning("%s: all-tissues overlay failed (%s)", patient_id, e)

            logging.info("%s: L3=%d | SM_CSA=%.2f cm%s",
                         patient_id, l3, a_sm, f", SMI={smi:.2f}" if smi is not None else "")

    logging.info("Saved cohort CSV: %s", out_csv)

if __name__ == "__main__":
    main()
