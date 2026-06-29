import math

import numpy as np


HU_MIN = -29
HU_MAX = 150


def compute_csa_smra(ct2d: np.ndarray, mask2d_bin: np.ndarray, px_area_cm2: float):
    """Return CSA, SMRA, and mask pixel count for a 2D CT slice and binary mask."""
    npx = int((mask2d_bin > 0).sum())
    csa = npx * px_area_cm2
    if npx == 0:
        return csa, math.nan, npx

    hu_vals = ct2d[(mask2d_bin > 0) & (ct2d >= HU_MIN) & (ct2d <= HU_MAX)]
    smra = float(hu_vals.mean()) if hu_vals.size > 0 else math.nan
    return csa, smra, npx


def dice_coef(a: np.ndarray, b: np.ndarray, eps=1e-6):
    """Return Dice similarity for two binary masks."""
    a = a > 0
    b = b > 0
    den = a.sum() + b.sum()
    if den == 0:
        return math.nan
    return (2.0 * np.logical_and(a, b).sum()) / (den + eps)


def find_max_area_slice(mask_arr: np.ndarray):
    """Return the axial index with the largest positive mask area."""
    areas = np.asarray([(mask_arr[z] > 0).sum() for z in range(mask_arr.shape[0])], dtype=np.int64)
    if areas.size == 0 or areas.max() == 0:
        return None
    return int(areas.argmax())
