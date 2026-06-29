# -*- coding: utf-8 -*-
"""
Generate all dissertation-ready plots from:
 - comparison_all.csv  (Manual/DL/TS CSA & SMRA)
 - eval_dice_manual_vs_DL.csv
 - (optional) eval_dice_manual_vs_TS.csv

Outputs PNGs into: ROOT/dl_runs/figs/
Each plot is a single figure (no subplots), matplotlib default colors only.

Author: You, 2025
"""

import csv, math, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, faster, no Qt warnings
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT = Path(os.environ.get("ATTDS_DATA_ROOT", "anon_dig"))
RUNS = ROOT / "dl_runs"
CMP  = RUNS / "comparison_all.csv"
DICE_DL = RUNS / "eval_dice_manual_vs_DL.csv"       # required (your DL vs manual)
DICE_TS = RUNS / "eval_dice_manual_vs_TS.csv"       # optional; skip if missing
FIGDIR = RUNS / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------

def fnum(x):
    try: return float(x)
    except: return math.nan

def read_csv_dict(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def nanfilter(*arrays):
    """Return mask where all arrays are finite numbers (no NaNs)."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask

def med_iqr(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (math.nan, math.nan, math.nan, 0)
    return (float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
            int(arr.size))

def save_hist(data, bins, xlabel, title, fname, labels=None, alpha=0.65):
    plt.figure()
    if isinstance(data, (list, tuple)) and labels:
        for d, lab in zip(data, labels):
            d = np.array(d); d = d[np.isfinite(d)]
            if d.size: plt.hist(d, bins=bins, alpha=alpha, label=lab)
        plt.legend()
    else:
        d = np.array(data); d = d[np.isfinite(d)]
        plt.hist(d, bins=bins)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title)
    plt.savefig(FIGDIR/fname, bbox_inches="tight"); plt.close()

def save_box(data, labels, ylabel, title, fname, ylim=None):
    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.ylabel(ylabel); plt.title(title)
    if ylim: plt.ylim(*ylim)
    plt.savefig(FIGDIR/fname, bbox_inches="tight"); plt.close()

def save_scatter(x, y, xlabel, ylabel, title, fname, draw_identity=True, fit_line=True):
    x = np.array(x); y = np.array(y)
    m = nanfilter(x, y)
    x, y = x[m], y[m]
    plt.figure()
    plt.scatter(x, y, s=14)
    if draw_identity and x.size and y.size:
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--')
    if fit_line and x.size > 1:
        # simple linear fit (least squares)
        A = np.vstack([x, np.ones_like(x)]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        xx = np.linspace(x.min(), x.max(), 100)
        plt.plot(xx, k*xx + b, '-')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.savefig(FIGDIR/fname, bbox_inches="tight"); plt.close()

def save_bland_altman(a, b, label_a, label_b, title, fname):
    a = np.array(a); b = np.array(b)
    m = nanfilter(a, b)
    a, b = a[m], b[m]
    plt.figure()
    mean = (a + b) / 2.0
    diff = b - a
    mu = diff.mean()
    sd = diff.std(ddof=1) if diff.size > 1 else 0.0
    plt.scatter(mean, diff, s=14)
    plt.axhline(mu, linestyle='-')
    plt.axhline(mu + 1.96*sd, linestyle='--')
    plt.axhline(mu - 1.96*sd, linestyle='--')
    plt.xlabel(f"Mean of {label_a} and {label_b}")
    plt.ylabel(f"{label_b} - {label_a}")
    plt.title(title)
    plt.savefig(FIGDIR/fname, bbox_inches="tight"); plt.close()

def main():
    if not CMP.exists():
        print("comparison_all.csv not found:", CMP); return

    rows = read_csv_dict(CMP)

    pids   = [r["patient_id"] for r in rows]
    L3idx  = np.array([fnum(r.get("L3_index","")) for r in rows], dtype=float)

    # Metrics
    CSA_m  = np.array([fnum(r.get("CSA_manual","")) for r in rows], dtype=float)
    CSA_d  = np.array([fnum(r.get("CSA_DL",""))     for r in rows], dtype=float)
    CSA_t  = np.array([fnum(r.get("CSA_TS",""))     for r in rows], dtype=float)

    SMRA_m = np.array([fnum(r.get("SMRA_manual","")) for r in rows], dtype=float)
    SMRA_d = np.array([fnum(r.get("SMRA_DL",""))     for r in rows], dtype=float)
    SMRA_t = np.array([fnum(r.get("SMRA_TS",""))     for r in rows], dtype=float)

    # Notes (for audits)
    notes_m = [r.get("notes_manual","") for r in rows]
    notes_d = [r.get("notes_DL","") for r in rows]
    notes_t = [r.get("notes_TS","") for r in rows]

    # Dice DL vs Manual (required)
    dice_map = {}
    if DICE_DL.exists():
        for r in read_csv_dict(DICE_DL):
            if r.get("patient_id") and r.get("dice"):
                dice_map[r["patient_id"]] = fnum(r["dice"])
    dice_arr = np.array([dice_map.get(pid, math.nan) for pid in pids], dtype=float)

    # Dice TS vs Manual (optional)
    dice_ts_map = {}
    if DICE_TS.exists():
        for r in read_csv_dict(DICE_TS):
            if r.get("patient_id") and r.get("dice"):
                dice_ts_map[r["patient_id"]] = fnum(r["dice"])
    dice_ts_arr = np.array([dice_ts_map.get(pid, math.nan) for pid in pids], dtype=float)

    # =================== DISTRIBUTIONS ===================
    # CSA histograms
    save_hist([CSA_m, CSA_d, CSA_t], bins=20,
              xlabel="L3 Skeletal Muscle CSA (cm2)",
              title="CSA at L3: Manual vs DL vs TS",
              fname="hist_csa_manual_dl_ts.png",
              labels=["Manual","DL","TS"])

    # SMRA histograms
    save_hist([SMRA_m, SMRA_d, SMRA_t], bins=20,
              xlabel="Skeletal Muscle HU (SMRA)",
              title="SMRA at L3: Manual vs DL vs TS",
              fname="hist_smra_manual_dl_ts.png",
              labels=["Manual","DL","TS"])

    # CSA boxplots (for manual subset)
    mask_has_manual = np.isfinite(CSA_m)
    save_box(
        [CSA_m[mask_has_manual], CSA_d[mask_has_manual], CSA_t[mask_has_manual]],
        labels=["Manual","DL","TS"],
        ylabel="CSA (cm2)",
        title="CSA at L3 (Manual subset)",
        fname="box_csa_manual_subset.png"
    )

    # SMRA boxplots (manual subset)
    save_box(
        [SMRA_m[mask_has_manual], SMRA_d[mask_has_manual], SMRA_t[mask_has_manual]],
        labels=["Manual","DL","TS"],
        ylabel="SMRA (HU)",
        title="SMRA at L3 (Manual subset)",
        fname="box_smra_manual_subset.png"
    )

    # =================== AGREEMENT / CORRELATION ===================
    # DL vs Manual CSA scatter
    save_scatter(CSA_m, CSA_d,
                 xlabel="CSA Manual (cm2)", ylabel="CSA DL (cm2)",
                 title="Agreement: CSA DL vs Manual",
                 fname="scatter_csa_dl_vs_manual.png")

    # TS vs Manual CSA scatter
    save_scatter(CSA_m, CSA_t,
                 xlabel="CSA Manual (cm2)", ylabel="CSA TS (cm2)",
                 title="Agreement: CSA TS vs Manual",
                 fname="scatter_csa_ts_vs_manual.png")

    # SMRA DL vs Manual scatter
    save_scatter(SMRA_m, SMRA_d,
                 xlabel="SMRA Manual (HU)", ylabel="SMRA DL (HU)",
                 title="Agreement: SMRA DL vs Manual",
                 fname="scatter_smra_dl_vs_manual.png")

    # SMRA TS vs Manual scatter
    save_scatter(SMRA_m, SMRA_t,
                 xlabel="SMRA Manual (HU)", ylabel="SMRA TS (HU)",
                 title="Agreement: SMRA TS vs Manual",
                 fname="scatter_smra_ts_vs_manual.png")

    # Bland-Altman (CSA)
    save_bland_altman(CSA_m, CSA_d,
                      label_a="CSA Manual", label_b="CSA DL",
                      title="Bland-Altman: CSA (DL - Manual) vs mean",
                      fname="ba_csa_dl_vs_manual.png")
    save_bland_altman(CSA_m, CSA_t,
                      label_a="CSA Manual", label_b="CSA TS",
                      title="Bland-Altman: CSA (TS - Manual) vs mean",
                      fname="ba_csa_ts_vs_manual.png")

    # Bland-Altman (SMRA)
    save_bland_altman(SMRA_m, SMRA_d,
                      label_a="SMRA Manual", label_b="SMRA DL",
                      title="Bland-Altman: SMRA (DL - Manual) vs mean",
                      fname="ba_smra_dl_vs_manual.png")
    save_bland_altman(SMRA_m, SMRA_t,
                      label_a="SMRA Manual", label_b="SMRA TS",
                      title="Bland-Altman: SMRA (TS - Manual) vs mean",
                      fname="ba_smra_ts_vs_manual.png")

    # =================== DICE PLOTS ===================
    # Dice histogram/boxplot (DL vs Manual)
    if np.isfinite(dice_arr).any():
        save_hist(dice_arr, bins=np.linspace(0,1,21),
                  xlabel="Dice (DL vs Manual)", title="Dice Distribution (DL vs Manual)",
                  fname="hist_dice_dl_manual.png")
        save_box([dice_arr], labels=["DL vs Manual"], ylabel="Dice",
                 title="Dice overlap at L3", fname="box_dice_dl_manual.png", ylim=(0,1.0))

    # Dice TS vs Manual (if available)
    if np.isfinite(dice_ts_arr).any():
        save_hist(dice_ts_arr, bins=np.linspace(0,1,21),
                  xlabel="Dice (TS vs Manual)", title="Dice Distribution (TS vs Manual)",
                  fname="hist_dice_ts_manual.png")
        save_box([dice_ts_arr], labels=["TS vs Manual"], ylabel="Dice",
                 title="Dice overlap at L3 (TS)", fname="box_dice_ts_manual.png", ylim=(0,1.0))

    # Dice vs CSA (manual) scatter - does overlap worsen on extremes?
    save_scatter([dice_map.get(pid, math.nan) for pid in pids], CSA_m,
                 xlabel="Dice (DL vs Manual)", ylabel="CSA Manual (cm2)",
                 title="Dice vs Manual CSA", fname="scatter_dice_vs_csa_manual.png",
                 draw_identity=False, fit_line=True)

    # Dice vs |CSA diff| scatter
    abs_diff_dl = np.abs(CSA_d - CSA_m)
    save_scatter([dice_map.get(pid, math.nan) for pid in pids], abs_diff_dl,
                 xlabel="Dice (DL vs Manual)", ylabel="|CSA DL - Manual| (cm2)",
                 title="Dice vs Absolute CSA Error (DL)", fname="scatter_dice_vs_absdiff_dl.png",
                 draw_identity=False, fit_line=True)

    # =================== NOTES / DATA QUALITY AUDITS ===================
    def count_notes(notes):
        from collections import Counter
        buckets = []
        for s in notes:
            if not s: continue
            parts = [p.strip() for p in s.split("|") if p.strip()]
            buckets.extend(parts if parts else ["ok"])
        return dict(Counter(buckets)) if buckets else {}

    notesM = count_notes(notes_m)
    notesD = count_notes(notes_d)
    notesT = count_notes(notes_t)

    def bar_from_dict(d, title, fname):
        if not d: return
        keys = list(d.keys()); vals = [d[k] for k in keys]
        plt.figure()
        plt.bar(range(len(keys)), vals)
        plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
        plt.ylabel("Count"); plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGDIR/fname, bbox_inches="tight"); plt.close()

    bar_from_dict(notesM, "Manual notes (quality/missing)", "bar_notes_manual.png")
    bar_from_dict(notesD, "DL notes (quality/missing)", "bar_notes_dl.png")
    bar_from_dict(notesT, "TS notes (quality/missing)", "bar_notes_ts.png")

    # =================== CDFs (nice summary view) ===================
    def save_cdf(data, label, xlabel, title, fname):
        d = np.array(data); d = d[np.isfinite(d)]
        if d.size == 0: return
        d = np.sort(d)
        y = np.linspace(0, 1, d.size)
        plt.figure()
        plt.plot(d, y)
        plt.xlabel(xlabel); plt.ylabel("Cumulative proportion")
        plt.title(title)
        plt.savefig(FIGDIR/fname, bbox_inches="tight"); plt.close()

    save_cdf(CSA_m, "Manual", "CSA (cm2)", "CDF: CSA Manual", "cdf_csa_manual.png")
    save_cdf(CSA_d, "DL",     "CSA (cm2)", "CDF: CSA DL",     "cdf_csa_dl.png")
    save_cdf(CSA_t, "TS",     "CSA (cm2)", "CDF: CSA TS",     "cdf_csa_ts.png")

    # =================== PRINT SUMMARY ===================
    md, q1d, q3d, nd = med_iqr(dice_arr)
    mdl, q1l, q3l, nl = med_iqr(CSA_d - CSA_m)
    mts, q1t, q3t, nt = med_iqr(CSA_t - CSA_m)

    print("Figures saved to:", FIGDIR)
    print(f"Dice (DL vs Manual) median={md:.3f} IQR={q1d:.3f}-{q3d:.3f} N={nd}")
    print(f"CSA diff (DL-Manual) median={mdl:.2f} IQR={q1l:.2f}-{q3l:.2f} N={nl}")
    print(f"CSA diff (TS-Manual) median={mts:.2f} IQR={q1t:.2f}-{q3t:.2f} N={nt}")

if __name__ == "__main__":
    main()
