# dl/predict_l3_unet.py
import os, re, json, numpy as np, torch, SimpleITK as sitk
from pathlib import Path
from train_l3_unet import UNet
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))
WEIGHTS = Path(os.environ.get("ATTDS_MODEL_WEIGHTS", str(ROOT/"dl_runs"/"l3_unet_best.pt")))
OUTDIR  = Path(os.environ.get("ATTDS_DL_DIR", str(ROOT/"dl_preds"))); OUTDIR.mkdir(exist_ok=True, parents=True)

def hu_norm(arr):
    arr = np.clip(arr, -200, 300)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 2 - 1

def l3_index(meta_txt):
    s = open(meta_txt,"r",encoding="utf-8",errors="ignore").read()
    m = re.search(r"L3_index:\s*([0-9]+)", s)
    return int(m.group(1)) if m else None

def main():
    net = UNet(ch=16).to(DEVICE)
    net.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    net.eval()

    pids = [p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("patient")]
    for pid in sorted(pids):
        meta = ROOT/pid/"metadata.txt"
        nii  = ROOT/pid/"original.nii.gz"
        if not (meta.exists() and nii.exists()): continue
        L = l3_index(meta)
        if L is None: continue  # skip TBD

        img = sitk.ReadImage(str(nii))
        arr = sitk.GetArrayFromImage(img)  # z,y,x
        if not (0 <= L < arr.shape[0]): continue

        sl = arr[L].astype(np.int16)
        x = hu_norm(sl)[None, None, ...].astype(np.float32)

        with torch.no_grad():
            pr = torch.sigmoid(net(torch.from_numpy(x).to(DEVICE))).cpu().numpy()[0,0]

        # build 3D volume mask with same size as CT
        vol = np.zeros_like(arr, dtype=np.uint8)
        vol[L] = (pr >= 0.5).astype(np.uint8)

        out_itk = sitk.GetImageFromArray(vol)      # (z,y,x)
        out_itk.CopyInformation(img)               # sizes match -> OK
        sitk.WriteImage(out_itk, str(OUTDIR/f"{pid}_L3_skm_pred.nii.gz"))

if __name__ == "__main__":
    main()
