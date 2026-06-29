import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))
DATA = Path(os.environ.get("ATTDS_TRAIN_NPZ_DIR", str(ROOT / "dl_data" / "train_npz")))
OUT  = Path(os.environ.get("ATTDS_RUNS_DIR", str(ROOT / "dl_runs"))); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15; BATCH = 16; LR = 1e-3

class NPZSet(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        z = np.load(self.files[i])
        x = z["image"][None, ...]          # 1,H,W
        y = z["mask"][None, ...].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def split(files, val_frac=0.2):
    random.shuffle(files)
    n = int(len(files)*val_frac)
    return files[n:], files[:n]

# --- small UNet ---
def conv(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        nn.Conv2d(ch_out, ch_out, 3, padding=1), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.c1 = conv(1, ch);     self.p1 = nn.MaxPool2d(2)
        self.c2 = conv(ch, ch*2);  self.p2 = nn.MaxPool2d(2)
        self.c3 = conv(ch*2, ch*4);self.p3 = nn.MaxPool2d(2)
        self.c4 = conv(ch*4, ch*8)

        self.u3 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2)
        self.c5 = conv(ch*8, ch*4)
        self.u2 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2)
        self.c6 = conv(ch*4, ch*2)
        self.u1 = nn.ConvTranspose2d(ch*2, ch, 2, 2)
        self.c7 = conv(ch*2, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        c1 = self.c1(x); x = self.p1(c1)
        c2 = self.c2(x); x = self.p2(c2)
        c3 = self.c3(x); x = self.p3(c3)
        c4 = self.c4(x)
        x = self.u3(c4); x = torch.cat([x, c3], 1); x = self.c5(x)
        x = self.u2(x);  x = torch.cat([x, c2], 1); x = self.c6(x)
        x = self.u1(x);  x = torch.cat([x, c1], 1); x = self.c7(x)
        return self.out(x)

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2*(pred*target).sum(dim=(2,3))
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def run():
    files = [str(p) for p in DATA.glob("*.npz")]
    if len(files) < 20:
        print("Not enough pairs to train."); return
    tr, va = split(files, 0.2)
    tr_dl = DataLoader(NPZSet(tr), batch_size=BATCH, shuffle=True, num_workers=0)
    va_dl = DataLoader(NPZSet(va), batch_size=BATCH, shuffle=False, num_workers=0)

    net = UNet(ch=16).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    best = 1e9; best_pth = OUT/"l3_unet_best.pt"
    for ep in range(1, EPOCHS+1):
        net.train(); tl=0
        for x,y in tr_dl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            pred = net(x)
            loss = 0.5*bce(pred,y) + 0.5*dice_loss(pred,y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x.size(0)
        net.eval(); vl=0
        with torch.no_grad():
            for x,y in va_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = net(x)
                loss = 0.5*bce(pred,y) + 0.5*dice_loss(pred,y)
                vl += loss.item()*x.size(0)
        tl/=len(tr_dl.dataset); vl/=len(va_dl.dataset)
        print(f"Epoch {ep}: train {tl:.4f}  val {vl:.4f}")
        if vl < best:
            best = vl; torch.save(net.state_dict(), best_pth)
    print("Saved:", best_pth)

if __name__ == "__main__":
    run()
