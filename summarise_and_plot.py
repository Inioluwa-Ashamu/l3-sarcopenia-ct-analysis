# dl/summarise_and_plot.py
import csv, math, os, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))
RUNS = ROOT / "dl_runs"
CMP  = RUNS / "comparison_all.csv"           # Manual/DL/TS CSA & SMRA
DICE = RUNS / "eval_dice_manual_vs_DL.csv"   # DL vs Manual Dice
OUTD = RUNS / "figs"; OUTD.mkdir(parents=True, exist_ok=True)
SUMT = RUNS / "summary.txt"

def to_float(x):
    try: return float(x)
    except: return math.nan

# ------ read combined metrics ------
rows=[]
with open(CMP, newline="", encoding="utf-8") as fh:
    rdr = csv.DictReader(fh)
    for r in rdr:
        rows.append(r)

# ------ read dice ------
dice_map={}
if DICE.exists():
    with open(DICE, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            pid = r.get("patient_id")
            dval = to_float(r.get("dice", ""))
            if pid and not math.isnan(dval):
                dice_map[pid] = dval

# Vectors
def col(name):
    out=[]
    for r in rows:
        x = r.get(name, "")
        out.append(to_float(x) if x!="" else math.nan)
    return np.array(out, dtype=float)

pid   = [r["patient_id"] for r in rows]
cma   = col("CSA_manual")
cdl   = col("CSA_DL")
cts   = col("CSA_TS")
sma   = col("SMRA_manual")
sdl   = col("SMRA_DL")
sts   = col("SMRA_TS")

# restrict to cases with manual present
mask_manual = ~np.isnan(cma)
pid_m = [p for p,m in zip(pid,mask_manual) if m]
cma_m = cma[mask_manual]
cdl_m = cdl[mask_manual]
cts_m = cts[mask_manual]

# diffs vs manual
dl_diff = cdl_m - cma_m
ts_diff = cts_m - cma_m

# dice list aligned to manual pids
dice_list = [dice_map.get(p, math.nan) for p in pid_m]
dice_arr  = np.array(dice_list, dtype=float)
dice_arr  = dice_arr[~np.isnan(dice_arr)]

def med_iqr(x):
    x = x[~np.isnan(x)]
    if x.size == 0: return (math.nan, math.nan, math.nan, 0)
    return (float(np.median(x)),
            float(np.percentile(x,25)),
            float(np.percentile(x,75)),
            int(x.size))

# ---- summary text ----
md, q1d, q3d, nd = med_iqr(dice_arr)
mdl, q1l, q3l, nl = med_iqr(dl_diff)
mts, q1t, q3t, nt = med_iqr(ts_diff)

with open(SUMT, "w", encoding="utf-8") as fh:
    fh.write("=== SUMMARY ===\n")
    fh.write(f"N valid rows (comparison_all.csv): {len(rows)}\n")
    fh.write(f"N manual cases (CSA_manual present): {len(cma_m)}\n")
    fh.write(f"N Dice (DL vs Manual): {nd}\n\n")
    fh.write(f"Dice (DL vs Manual): median={md:.3f}  IQR={q1d:.3f}-{q3d:.3f}\n")
    fh.write(f"CSA diff (DL - Manual) [cm2]: median={mdl:.2f}  IQR={q1l:.2f}-{q3l:.2f}  (N={nl})\n")
    fh.write(f"CSA diff (TS - Manual) [cm2]: median={mts:.2f}  IQR={q1t:.2f}-{q3t:.2f}  (N={nt})\n")

print("Wrote", SUMT)

# ---- plots ----
# 1) CSA histogram (Manual vs DL)
plt.figure()
plt.hist(cma_m[~np.isnan(cma_m)], bins=15, alpha=0.6, label="Manual")
plt.hist(cdl_m[~np.isnan(cdl_m)], bins=15, alpha=0.6, label="DL")
plt.xlabel("L3 Skeletal Muscle CSA (cm2)"); plt.ylabel("Count"); plt.legend()
plt.title("CSA at L3: Manual vs DL"); plt.savefig(OUTD/"hist_csa_manual_vs_dl.png", bbox_inches="tight"); plt.close()

# 2) Scatter DL vs Manual
valid = ~np.isnan(cma_m) & ~np.isnan(cdl_m)
x, y = cma_m[valid], cdl_m[valid]
plt.figure()
plt.scatter(x, y, s=14)
if x.size and y.size:
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = (hi-lo)*0.05
    plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--')
plt.xlabel("CSA Manual (cm2)"); plt.ylabel("CSA DL (cm2)")
plt.title("Agreement: DL vs Manual"); plt.savefig(OUTD/"scatter_csa_dl_vs_manual.png", bbox_inches="tight"); plt.close()

# 3) Dice boxplot
if dice_arr.size>0:
    plt.figure()
    plt.boxplot([dice_arr], labels=["DL vs Manual"])
    plt.ylabel("Dice similarity")
    plt.title("Segmentation overlap at L3"); plt.ylim(0,1.0)
    plt.savefig(OUTD/"box_dice_dl_manual.png", bbox_inches="tight"); plt.close()

print("Wrote figs to", OUTD)
