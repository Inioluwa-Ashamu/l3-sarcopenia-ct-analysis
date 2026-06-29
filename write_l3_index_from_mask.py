# file: write_l3_index_from_mask.py
# Purpose: derive L3 slice index from vertebrae_L3.nii.gz and persist to metadata.txt.
# Works with your run_all_segmentation.py outputs (no prior extract_* scripts needed).

import os
import re
import sys
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# --------- CONFIG (edit if paths differ) ----------
ROOT = os.environ.get("ATTDS_DATA_ROOT", "anon_dig")        # same root used in run_all_segmentation.py
L3_DIRNAME = "segmentations2_L3"
MASK_NAME = "vertebrae_L3.nii.gz"
LOG_LEVEL = "INFO"
SAVE_OVERLAY = True   # set False to skip overlay
OVERLAY_NAME = "overlay_L3_vertebra.png"

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")

def read_mask(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)   # shape: (Z, Y, X)
    return arr, img

def find_max_area_slice(mask_arr):
    # area per axial slice (count >0)
    areas = np.asarray([(mask_arr[z] > 0).sum() for z in range(mask_arr.shape[0])], dtype=np.int64)
    if areas.size == 0 or areas.max() == 0:
        return None
    return int(areas.argmax())

def write_or_update_l3_index(meta_txt_path, l3_index):
    """Add or update a single 'L3_index: N' line in metadata.txt (idempotent)."""
    line_new = f"L3_index: {l3_index}\n"
    if not os.path.exists(meta_txt_path):
        with open(meta_txt_path, "w", encoding="utf-8") as f:
            f.write(line_new)
        return True

    with open(meta_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Update if any line contains 'L3_index'
    updated = False
    for i, line in enumerate(lines):
        if "L3_index" in line:
            lines[i] = line_new
            updated = True
            break
    if not updated:
        lines.append(line_new)

    with open(meta_txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True

def save_overlay(patient_dir, ct_path, mask_arr, z, out_name):
    """Save a quick PNG showing the L3 vertebra mask edges on top of the HU slice."""
    try:
        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        if not (0 <= z < ct.shape[0]):  # guard
            return
        ct_slice = ct[z]

        # derive edges from mask
        m = (mask_arr[z] > 0).astype(np.uint8)
        # simple 4-neighbour edge
        edges = (m - np.minimum.reduce([np.roll(m,1,0), np.roll(m,-1,0), np.roll(m,1,1), np.roll(m,-1,1)])) > 0

        plt.figure(figsize=(5,5))
        plt.imshow(ct_slice, cmap="gray")
        # red edges with alpha
        overlay = np.zeros((*edges.shape, 4), dtype=float)
        overlay[edges] = [1, 0, 0, 0.9]
        plt.imshow(overlay)
        plt.axis("off")
        out_path = os.path.join(patient_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        logging.info("Saved overlay: %s (z=%d)", out_path, z)
    except Exception as e:
        logging.warning("Overlay failed: %s", e)

def process_patient(patient_id, patient_dir):
    ct_path = os.path.join(patient_dir, "original.nii.gz")
    l3_mask_path = os.path.join(patient_dir, L3_DIRNAME, MASK_NAME)
    if not os.path.exists(ct_path):
        logging.warning("%s: missing original.nii.gz - skip", patient_id)
        return
    if not os.path.exists(l3_mask_path):
        logging.warning("%s: missing %s - skip", patient_id, l3_mask_path)
        return

    # derive L3 index from mask
    mask_arr, _ = read_mask(l3_mask_path)
    z = find_max_area_slice(mask_arr)
    if z is None:
        logging.warning("%s: L3 mask has no positive voxels - skip", patient_id)
        return

    # write/update metadata.txt
    meta_txt = os.path.join(patient_dir, "metadata.txt")
    write_or_update_l3_index(meta_txt, z)
    logging.info("%s: L3_index = %d written to metadata.txt", patient_id, z)

    # optional overlay
    if SAVE_OVERLAY:
        save_overlay(patient_dir, ct_path, mask_arr, z, OVERLAY_NAME)

def main(root):
    patients = [d for d in sorted(os.listdir(root)) if d.lower().startswith("patient")]
    if not patients:
        logging.error("No patient* directories found under %s", root)
        return
    for pid in patients:
        pdir = os.path.join(root, pid)
        if os.path.isdir(pdir):
            process_patient(pid, pdir)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1]:
        ROOT = sys.argv[1]
    main(ROOT)
