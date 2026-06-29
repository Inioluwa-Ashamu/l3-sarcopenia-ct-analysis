import os, re, csv
from pathlib import Path
import numpy as np, SimpleITK as sitk

ROOT = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))
TS_DIRNAME = "segmentations2_tissue"
MANUAL_DIR = Path(os.environ.get("ATTDS_MANUAL_DIR", "masks"))
OUT = ROOT/"dl_runs"/"eval_dice_manual_vs_TS.csv"

def read_l3_index(pid):
    s = (ROOT/pid/"metadata.txt").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"L3_index:\s*([0-9]+)", s); return int(m.group(1)) if m else None

def to2d(p, L):
    itk = sitk.ReadImage(str(p)); arr = sitk.GetArrayFromImage(itk)
    if arr.ndim == 2: return arr
    Z,Y,X = arr.shape; return arr[L] if 0<=L<Z else arr[0]

def bin(a): return (a>0).astype(np.uint8)
def dice(a,b,eps=1e-6):
    a,b = a>0, b>0
    return (2*(a&b).sum()) / (a.sum()+b.sum()+eps)

def resample(mask, target_shape):
    if mask.shape == target_shape: return bin(mask)
    itk = sitk.GetImageFromArray(bin(mask))
    ref = sitk.Image(target_shape[1], target_shape[0], sitk.sitkUInt8)
    ref.SetOrigin((0.0,0.0)); ref.SetSpacing((1.0,1.0)); ref.SetDirection((1,0,0,1))
    out = sitk.Resample(itk, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out).astype(np.uint8)

def main():
    rows=[("patient_id","dice")]
    for p in sorted([x.name for x in ROOT.iterdir() if x.is_dir() and x.name.startswith("patient")]):
        L = read_l3_index(p);
        if L is None: continue
        ts_dir = ROOT/p/TS_DIRNAME
        if not ts_dir.exists(): continue
        hits = list(ts_dir.glob("*skeletal*muscle*.nii*")) or list(ts_dir.glob("*muscle*.nii*"))
        if not hits: continue
        ts2d = to2d(hits[0], L)

        man = None
        for pat in (f"{p}*.nii*", f"*{p}*.nii*"):
            got = list(MANUAL_DIR.glob(pat))
            if got: man = got[0]; break
        if not man: continue
        m2d = to2d(man, L)
        m2d = resample(m2d, ts2d.shape)
        rows.append((p, f"{dice(m2d, ts2d):.4f}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT,"w",newline="") as f: csv.writer(f).writerows(rows)
    print("Wrote", OUT)

if __name__=="__main__":
    main()
