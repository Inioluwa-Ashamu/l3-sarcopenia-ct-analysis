import os, re, glob, json, numpy as np, SimpleITK as sitk
from pathlib import Path

ROOT = os.environ.get("ATTDS_DATA_ROOT", "anon_dig")
OUT  = Path(ROOT) / "dl_data" / "train_npz"
OUT.mkdir(parents=True, exist_ok=True)

def read_nii(p):
    img = sitk.ReadImage(str(p)); arr = sitk.GetArrayFromImage(img)  # z,y,x
    return arr, img

def find_l3_index(meta_txt):
    with open(meta_txt, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    m = re.search(r"L3_index:\s*([0-9]+)", s)
    return int(m.group(1)) if m else None

def hu_normalize(arr):  # scale HU to [-1,1] around typical range
    arr = np.clip(arr, -200, 300)  # gentle clamp for abdomen
    return (arr - arr.min()) / (arr.max() - arr.min()) * 2 - 1

def find_skm_mask_dir(pid):
    cand = [
        Path(ROOT)/pid/"segmentations2_tissue",
        Path(ROOT)/pid/"segmentations_tissue",
        Path(ROOT)/pid/"segmentations"/"tissue",
    ]
    for c in cand:
        if c.exists(): return c
    return None

def find_binary_muscle_mask(mask_dir):
    pats = ["*skeletal*muscle*.nii*", "*muscle*.nii*"]
    for p in pats:
        hits = list(mask_dir.glob(p))
        if hits: return hits[0]
    return None

def save_pair(pid):
    pat_dir = Path(ROOT)/pid
    meta = pat_dir/"metadata.txt"
    nii  = pat_dir/"original.nii.gz"
    if not (meta.exists() and nii.exists()): return False

    l3 = find_l3_index(meta)
    if l3 is None: return False  # no index or TBD

    img3d, img_itk = read_nii(nii)
    if l3<0 or l3>=img3d.shape[0]: return False
    ct = img3d[l3].astype(np.int16)

    mask_dir = find_skm_mask_dir(pid)
    if not mask_dir: return False
    skm = find_binary_muscle_mask(mask_dir)
    if not skm: return False

    m3d, _ = read_nii(skm)
    if m3d.ndim != 3 or m3d.shape[0] != img3d.shape[0]: return False
    m = (m3d[l3] > 0).astype(np.uint8)

    x = hu_normalize(ct).astype(np.float32)
    if m.sum() == 0: return False  # skip silent duds

    np.savez_compressed(OUT/f"{pid}.npz", image=x, mask=m)
    return True

def main():
    pids = [p.name for p in Path(ROOT).iterdir() if p.is_dir() and p.name.startswith("patient")]
    ok = 0
    for pid in sorted(pids):
        try:
            ok += bool(save_pair(pid))
        except Exception as e:
            pass
    print(f"Saved {ok} training pairs to {OUT}")

if __name__ == "__main__":
    main()
