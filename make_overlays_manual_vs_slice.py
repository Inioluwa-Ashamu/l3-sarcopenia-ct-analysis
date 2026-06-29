import os, re, glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
L3_SLICES_DIR = os.environ.get("ATTDS_L3_SLICES_DIR", "annotations/nii_for_snap")
MANUAL_MASKS_DIR = os.environ.get("ATTDS_MANUAL_DIR", "masks")
OUT_DIR = os.environ.get("ATTDS_OVERLAY_DIR", "overlays_manual")

# Filename patterns (glob). Keep broad; we match patient IDs by regex.
SLICE_PATTERNS = ["*.nii", "*.nii.gz", "*L3slice*.nii*", "*_L3slice.nii*"]
MASK_PATTERNS  = ["*.nii", "*.nii.gz", "*manual*.nii*", "patient*.nii*"]

# HU window and overlay settings
HU_MIN, HU_MAX = -200, 300
ALPHA_FILL = 0.35
LINEWIDTH = 1.8
FIGSIZE = (6, 6)  # inches
DPI = 160
# ---------------------------

def find_files(folder, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(Path(folder)/pat)))
    # unique-preserving order
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            uniq.append(f); seen.add(f)
    return uniq

def extract_pid(name):
    """
    Extract 'patientXXX' from a filename or path (case-insensitive).
    Returns lowercase like 'patient012' or None.
    """
    m = re.search(r"(patient\s*\d+)", name, flags=re.IGNORECASE)
    return m.group(1).lower().replace(" ", "") if m else None

def read_nii_2d(path):
    """
    Reads a NIfTI. If 3D with depth==1, squeezes to 2D.
    If 3D depth>1, takes the first slice [0].
    Returns np.ndarray (H,W) and the sitk.Image (for spacing if needed).
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X) or (Y,X)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            # If user accidentally saved a 3D stack, take the first slice
            arr = arr[0]
    return arr, img

def resample_mask_to_image(mask_arr, target_shape):
    """
    Nearest-neighbour resize if shapes differ. Uses SimpleITK for correctness.
    """
    if mask_arr.shape == target_shape:
        return mask_arr.astype(np.uint8)

    # Convert back to sitk, then resample to target shape using nearest neighbour
    mask_itk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    # Create a reference image with the desired size (target_shape in Y,X)
    ref = sitk.Image(target_shape[1], target_shape[0], sitk.sitkUInt8)  # note X, Y
    ref.SetOrigin((0.0, 0.0))
    ref.SetSpacing((1.0, 1.0))
    ref.SetDirection((1,0,0,1))  # identity 2D

    res = sitk.Resample(
        mask_itk,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )
    return sitk.GetArrayFromImage(res).astype(np.uint8)

def window_ct_hu(ct, lo=HU_MIN, hi=HU_MAX):
    ct = np.clip(ct, lo, hi)
    # scale to [0,1] for display
    return (ct - lo) / max(1e-6, (hi - lo))

def plot_overlay(ct_norm, mask_bin, title, out_png):
    """
    Saves two images:
      - *_overlay_fill.png  : translucent filled mask
      - *_overlay_contour.png : mask contour
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # 1) Filled overlay
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = plt.axes([0,0,1,1])  # tight
    ax.imshow(ct_norm, cmap="gray", interpolation="nearest")
    ax.imshow(np.ma.masked_where(mask_bin == 0, mask_bin), alpha=ALPHA_FILL, interpolation="nearest")
    ax.set_axis_off()
    plt.title(title, fontsize=10)
    fill_path = out_png.replace(".png", "_overlay_fill.png")
    plt.savefig(fill_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 2) Contour overlay
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = plt.axes([0,0,1,1])
    ax.imshow(ct_norm, cmap="gray", interpolation="nearest")
    # draw contours (vectorized): find edges by erosion XOR
    from scipy.ndimage import binary_erosion
    edges = mask_bin.astype(bool) ^ binary_erosion(mask_bin.astype(bool), iterations=1)
    # paint edges as max intensity (white) on top
    edge_img = np.dstack([ct_norm]*3)  # RGB
    edge_img[edges] = [1.0, 0.0, 0.0]  # red edges; Matplotlib default color set will render red
    ax.imshow(edge_img, interpolation="nearest")
    ax.set_axis_off()
    plt.title(title + " (contour)", fontsize=10)
    cnt_path = out_png.replace(".png", "_overlay_contour.png")
    plt.savefig(cnt_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return fill_path, cnt_path

def main():
    slices = find_files(L3_SLICES_DIR, SLICE_PATTERNS)
    masks  = find_files(MANUAL_MASKS_DIR, MASK_PATTERNS)

    # Index by patient id
    slice_map = {}
    for s in slices:
        pid = extract_pid(s)
        if pid: slice_map[pid] = s

    mask_map = {}
    for m in masks:
        pid = extract_pid(m)
        if pid and pid not in mask_map:
            mask_map[pid] = m  # first match wins; adjust if you have multiples

    # Pair and render
    out_root = Path(OUT_DIR)
    paired = sorted(set(slice_map.keys()) & set(mask_map.keys()))

    print(f"Found {len(paired)} matched patient IDs.")

    for pid in paired:
        s_path = slice_map[pid]
        m_path = mask_map[pid]

        # Read slice & mask
        ct2d, _ = read_nii_2d(s_path)
        mask2d, _ = read_nii_2d(m_path)

        # Ensure mask matches slice shape
        if mask2d.shape != ct2d.shape:
            mask2d = resample_mask_to_image(mask2d, ct2d.shape)

        mask_bin = (mask2d > 0).astype(np.uint8)
        if mask_bin.sum() == 0:
            print(f"[WARN] {pid}: manual mask empty (no positive voxels); skipping overlay.")
            continue

        # Window CT for display
        ct_norm = window_ct_hu(ct2d)

        # Save overlays
        out_png = str(out_root / f"{pid}_manual_on_L3slice.png")
        fill_path, cnt_path = plot_overlay(ct_norm, mask_bin, f"{pid} manual vs L3slice", out_png)
        print(f"[OK] {pid}: {Path(fill_path).name}, {Path(cnt_path).name}")

    # Report missing pairs
    missing_masks = sorted(set(slice_map.keys()) - set(mask_map.keys()))
    missing_slices = sorted(set(mask_map.keys()) - set(slice_map.keys()))
    if missing_masks:
        print(f"\nNo MANUAL mask for: {', '.join(missing_masks[:10])}{' ...' if len(missing_masks)>10 else ''}")
    if missing_slices:
        print(f"No L3 SLICE for: {', '.join(missing_slices[:10])}{' ...' if len(missing_slices)>10 else ''}")

if __name__ == "__main__":
    main()
