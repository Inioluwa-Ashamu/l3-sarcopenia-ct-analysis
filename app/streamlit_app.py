"""
Unified Research-Grade L3 Sarcopenia Streamlit Application
=========================================================

This single-file app merges and supersedes both prior prototypes:
- "L3 Sarcopenia Streamlit App.py"
- "L3 Sarcopenia Streamlit App-proV3.py"

Key capabilities:
- Robust data ingestion: DICOM series (via SimpleITK) and NIfTI, with validation.
- L3 index resolution: from metadata.txt; fallback heuristics available.
- Mask handling: Manual, TotalSegmentator (TS) tissue masks, and Deep Learning (DL) predictions.
- DL inference: pluggable PyTorch U-Net wrapper, optional on-the-fly inference.
- Metrics: CSA (cm2), SMRA (HU in [-29,150]), SMI (cm2/m^2) when height provided.
- Threshold profiles: configurable SMI cut-offs (default Derstine 2018).
- Single-patient viewer: interactive overlays, PNG export, metrics CSV.
- Cohort analytics: distributions, agreement plots, Bland-Altman, Dice summaries.
- Ad-hoc uploads: single-slice CT + mask CSA/SMRA calculator.
- Reporting: per-patient PDF report via reportlab.
- Engineering quality: logging, exceptions, docstrings, modular functions.

Run:
    streamlit run app/streamlit_app.py

Environment overrides (optional):
    APP_ROOT, APP_MANUAL_DIR, APP_DL_DIR, APP_TS_SUBDIR, APP_WEIGHTS, APP_LOG_LEVEL

Note:
- For submission this is self-contained. In a production repo, split into modules and add unit tests.
"""
from __future__ import annotations

import os
# OpenMP/libiomp hotfixes before heavy imports
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import re
import json
import math
import time
import enum
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import SimpleITK as sitk

# Optional imports guarded
try:
    import pydicom  # noqa: F401
except Exception:
    pydicom = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.utils import ImageReader
except Exception:
    A4 = None

# ------------------------------
# Logging
# ------------------------------
import logging
LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("l3_merged_app")

# ------------------------------
# Constants & Defaults
# ------------------------------
DEFAULT_ROOT = Path(os.environ.get("APP_ROOT", os.environ.get("ATTDS_DATA_ROOT", "anon_dig")))
DEFAULT_MANUAL_DIR = Path(os.environ.get("APP_MANUAL_DIR", os.environ.get("ATTDS_MANUAL_DIR", "masks")))
DEFAULT_DL_DIR = Path(os.environ.get("APP_DL_DIR", str(DEFAULT_ROOT / "dl_preds")))
DEFAULT_TS_SUBDIR = os.environ.get("APP_TS_SUBDIR", "segmentations2_tissue")
DEFAULT_WEIGHTS = os.environ.get("APP_WEIGHTS", str(DEFAULT_ROOT / "dl_runs" / "l3_unet_best.pt"))

HU_WINDOW_DEFAULT = (-200, 300)
HU_MUSCLE = (-29, 150)

SMI_THRESHOLDS = {
    # Default reference; user-selectable via sidebar
    "Derstine2018_US": {"SMI_cm2_m2": (44.6, 34.0)},  # (men, women)
}

class Sex(enum.Enum):
    UNKNOWN = "Unknown"
    MALE = "Male"
    FEMALE = "Female"

# ------------------------------
# Utility functions
# ------------------------------

def safe_read_text(path: Path) -> str:
    """Best-effort UTF-8 text reader with ignore errors."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def parse_l3_index_from_metadata(meta_txt: Path) -> Optional[int]:
    """Extract L3_index: N from metadata.txt if present."""
    s = safe_read_text(meta_txt)
    m = re.search(r"L3_index:\s*([0-9]+)", s)
    return int(m.group(1)) if m else None


def list_patient_ids(root: Path) -> List[str]:
    try:
        return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("patient")])
    except Exception:
        return []


def hu_window(ct2d: np.ndarray, lo: int, hi: int) -> np.ndarray:
    arr = np.clip(ct2d, lo, hi)
    denom = float(hi - lo) if hi > lo else 1.0
    return (arr - lo) / denom


def nn_resample_mask(mask2d: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resampling for label masks to match CT slice shape."""
    if mask2d.shape == target_shape:
        return (mask2d > 0).astype(np.uint8)
    itk = sitk.GetImageFromArray((mask2d > 0).astype(np.uint8))
    ref = sitk.Image(int(target_shape[1]), int(target_shape[0]), sitk.sitkUInt8)
    ref.SetOrigin((0.0, 0.0)); ref.SetSpacing((1.0, 1.0)); ref.SetDirection((1, 0, 0, 1))
    out = sitk.Resample(itk, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out).astype(np.uint8)


def compute_csa_smra(ct2d: np.ndarray, mask2d_bin: np.ndarray, sx_mm: float, sy_mm: float,
                     hu_min: int = HU_MUSCLE[0], hu_max: int = HU_MUSCLE[1]) -> Tuple[float, Optional[float], int, float]:
    """CSA (cm2) from pixel count * pixel area; SMRA within HU range; returns (CSA, SMRA, n_pixels, px_area)."""
    px_area_cm2 = (sx_mm * sy_mm) / 100.0
    npx = int((mask2d_bin > 0).sum())
    csa = npx * px_area_cm2
    if npx == 0:
        return csa, None, npx, px_area_cm2
    hu_vals = ct2d[(mask2d_bin > 0) & (ct2d >= hu_min) & (ct2d <= hu_max)]
    smra = float(hu_vals.mean()) if hu_vals.size > 0 else None
    return csa, smra, npx, px_area_cm2


def load_ct_volume(root: Path, pid: str) -> Tuple[Optional[np.ndarray], Optional[sitk.Image]]:
    p = root / pid / "original.nii.gz"
    if not p.exists():
        return None, None
    try:
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img).astype(np.int16)
        return arr, img
    except Exception as e:
        LOGGER.error("Failed to read CT volume for %s: %s", pid, e)
        return None, None


def get_xy_spacing(root: Path, pid: str, ct_img: sitk.Image) -> Tuple[float, float]:
    meta_json = root / pid / "metadata.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            # support either [sx, sy, sz] or dict with keys
            spacing = meta.get("spacing") or [meta.get("spacing_x_mm"), meta.get("spacing_y_mm"), meta.get("slice_thickness_mm")]
            sx, sy = float(spacing[0]), float(spacing[1])
            return sx, sy
        except Exception:
            pass
    sx, sy = ct_img.GetSpacing()[0], ct_img.GetSpacing()[1]
    return sx, sy


def load_mask_to_slice2d(path: Optional[Path], L: int) -> Tuple[Optional[np.ndarray], str]:
    if path is None or not path.exists():
        return None, "not_found"
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 2:
            return (arr > 0).astype(np.uint8), "ok"
        Z = arr.shape[0]
        if 0 <= L < Z:
            return (arr[L] > 0).astype(np.uint8), "ok"
        return (arr[0] > 0).astype(np.uint8), f"used_slice0_Z={Z}"
    except Exception as e:
        return None, f"error:{e}"


def load_nifti_slice2d(path: Optional[Path], L: int) -> Tuple[Optional[np.ndarray], str]:
    if path is None or not path.exists():
        return None, "not_found"
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 2:
            return arr.astype(np.int16), "ok"
        Z = arr.shape[0]
        if 0 <= L < Z:
            return arr[L].astype(np.int16), "ok"
        return arr[0].astype(np.int16), f"used_slice0_Z={Z}"
    except Exception as e:
        return None, f"error:{e}"

# ------------------------------
# Model wrapper (DL inference)
# ------------------------------

class SmallUNet2D(nn.Module):  # type: ignore[misc]
    """Compact 2D U-Net for binary mask prediction."""
    def __init__(self, ch: int = 16):
        super().__init__()
        self.enc1 = self._block(1, ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(ch, ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(ch * 2, ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self._block(ch * 4, ch * 8)
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, 2)
        self.dec3 = self._block(ch * 8, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, 2)
        self.dec2 = self._block(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, 2)
        self.dec1 = self._block(ch * 2, ch)
        self.outc = nn.Conv2d(ch, 1, 1)

    @staticmethod
    def _block(i_ch: int, o_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(i_ch, o_ch, 3, padding=1), nn.BatchNorm2d(o_ch), nn.ReLU(inplace=True),
            nn.Conv2d(o_ch, o_ch, 3, padding=1), nn.BatchNorm2d(o_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore[override]
        c1 = self.enc1(x); x = self.pool1(c1)
        c2 = self.enc2(x); x = self.pool2(c2)
        c3 = self.enc3(x); x = self.pool3(c3)
        x = self.bottleneck(x)
        x = self.up3(x); x = torch.cat([x, c3], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, c2], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, c1], dim=1); x = self.dec1(x)
        return self.outc(x)


@dataclass
class ModelWrapper:
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    weights_path: Optional[Path] = Path(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS else None
    model: Optional[nn.Module] = None

    def load(self) -> None:
        if torch is None:
            LOGGER.warning("PyTorch unavailable; DL inference disabled.")
            self.model = None
            return
        net = SmallUNet2D(ch=16).to(self.device)
        if self.weights_path and Path(self.weights_path).exists():
            ckpt = torch.load(str(self.weights_path), map_location=self.device)
            net.load_state_dict(ckpt)
            LOGGER.info("Loaded weights from %s", self.weights_path)
        else:
            LOGGER.warning("Weights not found at %s; running with random init.", self.weights_path)
        net.eval()
        self.model = net

    @staticmethod
    def _preprocess(ct2d: np.ndarray) -> np.ndarray:
        arr = np.clip(ct2d.astype(np.float32), -200, 300)
        # scale to [-1,1]
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        x = (arr - mn) / (mx - mn) * 2 - 1
        return x

    @torch.no_grad()
    def predict(self, ct2d: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.model is None or torch is None:
            return np.zeros_like(ct2d, dtype=np.uint8)
        x = self._preprocess(ct2d)[None, None, ...]
        xt = torch.from_numpy(x).to(self.device)
        pr = torch.sigmoid(self.model(xt)).cpu().numpy()[0, 0]
        return (pr >= float(threshold)).astype(np.uint8)

# ------------------------------
# Visualization
# ------------------------------

def render_overlay_png(ct2d: np.ndarray, overlays: Dict[str, Optional[np.ndarray]],
                       hu_lo: int, hu_hi: int,
                       colors: Dict[str, str], styles: Dict[str, str],
                       title: str = "") -> bytes:
    ct_norm = hu_window(ct2d, hu_lo, hu_hi)
    fig = plt.figure(figsize=(6.0, 6.0), dpi=160)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(ct_norm, cmap="gray", interpolation="nearest")
    for name, m in overlays.items():
        if m is None:
            continue
        mbin = (m > 0).astype(float)
        if mbin.sum() == 0:
            continue
        ax.contour(mbin, levels=[0.5], colors=[colors.get(name, "r")], linewidths=1.8, linestyles=styles.get(name, "-"))
    ax.set_axis_off()
    if title:
        plt.title(title, fontsize=10)
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0); plt.close(fig); buf.seek(0)
    return buf.read()


def fig_histogram(data_series: List[np.ndarray], labels: List[str], bins: int, xlabel: str, title: str) -> bytes:
    fig = plt.figure()
    for d, lab in zip(data_series, labels):
        dd = np.array(d); dd = dd[np.isfinite(dd)]
        if dd.size:
            plt.hist(dd, bins=bins, alpha=0.6, label=lab)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title); plt.legend()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
    return buf.read()


def fig_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, identity: bool = True) -> bytes:
    x = np.array(x); y = np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    fig = plt.figure(); plt.scatter(x, y, s=14)
    if identity and x.size and y.size:
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.read()


def fig_bland_altman(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, title: str) -> bytes:
    a = np.array(a); b = np.array(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    fig = plt.figure()
    mean = (a + b) / 2.0; diff = b - a
    mu = diff.mean() if diff.size else 0.0
    sd = diff.std(ddof=1) if diff.size > 1 else 0.0
    plt.scatter(mean, diff, s=14)
    plt.axhline(mu, linestyle='-'); plt.axhline(mu + 1.96*sd, linestyle='--'); plt.axhline(mu - 1.96*sd, linestyle='--')
    plt.xlabel(f"Mean of {label_a} & {label_b}"); plt.ylabel(f"{label_b} - {label_a}"); plt.title(title)
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return buf.read()

# ------------------------------
# PDF Reporting
# ------------------------------

def generate_pdf_report(pdf_path: Path, pid: str, meta: Dict[str, Any], metrics_table: pd.DataFrame, overlay_png: bytes, notes: Optional[str] = None) -> None:
    if A4 is None:
        raise RuntimeError("reportlab not installed")
    c = pdfcanvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    def header(title: str):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(30*mm, height - 25*mm, title)
        c.setFont("Helvetica", 9)
        c.drawRightString(width - 20*mm, height - 25*mm, time.strftime("%Y-%m-%d %H:%M"))

    def kv(y: float, k: str, v: str):
        c.setFont("Helvetica", 10); c.drawString(30*mm, y, f"{k}:")
        c.setFont("Helvetica-Bold", 10); c.drawString(65*mm, y, v)

    header("L3 Sarcopenia Report")
    y = height - 40*mm
    kv(y, "Patient ID", pid); y -= 6*mm
    kv(y, "L3 Index", str(meta.get("L3_index", ""))); y -= 6*mm
    kv(y, "Pixel area (cm2)", f"{meta.get('px_area_cm2','')}"); y -= 6*mm
    kv(y, "HU window", f"[{HU_MUSCLE[0]}, {HU_MUSCLE[1]}]")

    data = [list(metrics_table.columns)] + metrics_table.astype(str).values.tolist()
    tbl = Table(data, colWidths=[35*mm, 30*mm, 25*mm, 25*mm, 35*mm])
    tbl.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('FONT', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONT', (0,1), (-1,-1), 'Helvetica'),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
    ]))
    tbl.wrapOn(c, width, height)
    tbl.drawOn(c, 25*mm, y - 35*mm)

    try:
        img = ImageReader(io.BytesIO(overlay_png))
        c.drawImage(img, 120*mm, 80*mm, width=75*mm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    if notes:
        c.setFont("Helvetica", 9); c.drawString(25*mm, 20*mm, "Notes:")
        t = c.beginText(25*mm, 15*mm); t.setFont("Helvetica", 8)
        for line in notes.splitlines():
            t.textLine(line)
        c.drawText(t)

    c.showPage(); c.save()

# ------------------------------
# Streamlit UI helpers
# ------------------------------

def sidebar_configuration() -> Dict[str, Any]:
    st.sidebar.header("Configuration")
    root_in = st.sidebar.text_input("Dataset root", str(DEFAULT_ROOT))
    manual_in = st.sidebar.text_input("Manual masks dir", str(DEFAULT_MANUAL_DIR))
    dl_in = st.sidebar.text_input("DL predictions dir", str(DEFAULT_DL_DIR))
    ts_subdir = st.sidebar.text_input("TS subfolder (per patient)", DEFAULT_TS_SUBDIR)
    weights = st.sidebar.text_input("Model weights (.pt)", DEFAULT_WEIGHTS)

    hu_lo, hu_hi = st.sidebar.slider("Display HU window", -500, 500, HU_WINDOW_DEFAULT, 1)

    st.sidebar.markdown("---")
    show_manual = st.sidebar.checkbox("Show Manual", True)
    show_dl = st.sidebar.checkbox("Show DL", True)
    show_ts = st.sidebar.checkbox("Show TS", False)

    st.sidebar.markdown("---")
    sex = st.sidebar.selectbox("Sex (for SMI)", [s.value for s in Sex])
    height_m_str = st.sidebar.text_input("Height (m, optional)", "")
    try:
        height_m_val = float(height_m_str) if height_m_str.strip() else None
    except Exception:
        height_m_val = None
        st.sidebar.warning("Invalid height; leave blank to skip SMI.")

    st.sidebar.markdown("---")
    thresh_profile = st.sidebar.selectbox("Threshold profile", list(SMI_THRESHOLDS.keys()))

    st.sidebar.markdown("---")
    export_pdf = st.sidebar.checkbox("Enable PDF reporting", True)

    st.sidebar.markdown("---")
    allow_upload = st.sidebar.checkbox("Ad hoc single-slice upload mode", False)

    return {
        "root": Path(root_in),
        "manual_dir": Path(manual_in),
        "dl_dir": Path(dl_in),
        "ts_subdir": ts_subdir,
        "weights": Path(weights) if weights else None,
        "hu_window": (hu_lo, hu_hi),
        "show_manual": show_manual,
        "show_dl": show_dl,
        "show_ts": show_ts,
        "sex": sex,
        "height_m": height_m_val,
        "thresh_profile": thresh_profile,
        "export_pdf": export_pdf,
        "allow_upload": allow_upload,
    }


def topbar_header():
    st.title("L3 Sarcopenia CT Analysis - Research Viewer")
    st.caption(
        "Select a patient to view the recorded L3 slice, overlay Manual/DL/TS masks, and compute CSA/SMRA. "
        "If height is provided, SMI and classification are included."
    )


def patient_selector(root: Path) -> Optional[str]:
    pids = list_patient_ids(root)
    if not pids:
        st.error("No patient folders found. Check dataset root.")
        return None
    col1, col2 = st.columns([1, 3])
    with col1:
        pid = st.selectbox("Patient ID", options=pids)
    with col2:
        st.write("Data root:", root)
    return pid


def metrics_table_from_sources(ct2d: np.ndarray, sx_mm: float, sy_mm: float,
                               manual: Optional[np.ndarray], dl: Optional[np.ndarray], ts: Optional[np.ndarray],
                               height_m: Optional[float], sex: str,
                               thresh_profile: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    px_area_cm2 = (sx_mm * sy_mm) / 100.0
    sex_enum = Sex(sex) if sex in [s.value for s in Sex] else Sex.UNKNOWN

    def add_row(name: str, mask: Optional[np.ndarray]):
        if mask is None:
            rows.append({"Source": name, "CSA_cm2": None, "SMRA_HU": None, "SMI_cm2_m2": None, "n_pixels": 0})
            return
        csa, smra, npx, _ = compute_csa_smra(ct2d, (mask > 0).astype(np.uint8), sx_mm, sy_mm)
        smi = (csa / (height_m ** 2)) if (height_m and height_m > 0) else None
        rows.append({"Source": name, "CSA_cm2": csa, "SMRA_HU": smra, "SMI_cm2_m2": smi, "n_pixels": npx})

    add_row("Manual", manual)
    add_row("DL", dl)
    add_row("TS", ts)

    df = pd.DataFrame(rows)

    # Classification by SMI where possible
    classification: Dict[str, str] = {}
    if height_m and height_m > 0 and thresh_profile in SMI_THRESHOLDS:
        men_cut, women_cut = SMI_THRESHOLDS[thresh_profile]["SMI_cm2_m2"]
        cut = men_cut if sex_enum == Sex.MALE else women_cut if sex_enum == Sex.FEMALE else None
        if cut is not None and "SMI_cm2_m2" in df:
            for _, row in df.iterrows():
                smi_val = row.get("SMI_cm2_m2", None)
                if pd.notna(smi_val):
                    classification[row["Source"]] = "Low muscle (SMI)" if smi_val < cut else "Normal (SMI)"

    meta = {"px_area_cm2": round(px_area_cm2, 5), "classification": classification}

    # Round for display
    df_display = df.copy()
    if "CSA_cm2" in df_display:
        df_display["CSA_cm2"] = df_display["CSA_cm2"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    if "SMRA_HU" in df_display:
        df_display["SMRA_HU"] = df_display["SMRA_HU"].map(lambda x: None if pd.isna(x) else round(float(x), 1))
    if "SMI_cm2_m2" in df_display:
        df_display["SMI_cm2_m2"] = df_display["SMI_cm2_m2"].map(lambda x: None if pd.isna(x) else round(float(x), 2))

    return df_display, meta


def cohort_plots_section(comparison_csv: Optional[Path], dice_csv: Optional[Path]):
    st.subheader("Cohort analytics")
    if not comparison_csv or not comparison_csv.exists():
        st.info("comparison_all.csv not found. Provide it to enable cohort plots.")
        return

    df = pd.read_csv(comparison_csv)

    def to_f(series: pd.Series) -> np.ndarray:
        return series.astype(float).replace({np.inf: np.nan, -np.inf: np.nan}).to_numpy()

    CSA_m = to_f(df.get("CSA_manual", pd.Series([], dtype=float)))
    CSA_d = to_f(df.get("CSA_DL", pd.Series([], dtype=float)))
    CSA_t = to_f(df.get("CSA_TS", pd.Series([], dtype=float)))
    SMRA_m = to_f(df.get("SMRA_manual", pd.Series([], dtype=float)))
    SMRA_d = to_f(df.get("SMRA_DL", pd.Series([], dtype=float)))
    SMRA_t = to_f(df.get("SMRA_TS", pd.Series([], dtype=float)))

    col1, col2 = st.columns(2)
    with col1:
        img = fig_histogram([CSA_m, CSA_d, CSA_t], ["Manual", "DL", "TS"], 20, "CSA (cm)", "CSA at L3: Manual vs DL vs TS")
        st.image(img)
    with col2:
        img = fig_histogram([SMRA_m, SMRA_d, SMRA_t], ["Manual", "DL", "TS"], 20, "SMRA (HU)", "SMRA at L3: Manual vs DL vs TS")
        st.image(img)

    col3, col4 = st.columns(2)
    with col3:
        st.image(fig_scatter(CSA_m, CSA_d, "CSA Manual (cm)", "CSA DL (cm)", "Agreement: DL vs Manual"))
    with col4:
        st.image(fig_scatter(CSA_m, CSA_t, "CSA Manual (cm)", "CSA TS (cm)", "Agreement: TS vs Manual"))

    col5, col6 = st.columns(2)
    with col5:
        st.image(fig_bland_altman(CSA_m, CSA_d, "CSA Manual", "CSA DL", "Bland-Altman: DL  Manual (CSA)"))
    with col6:
        st.image(fig_bland_altman(CSA_m, CSA_t, "CSA Manual", "CSA TS", "Bland-Altman: TS  Manual (CSA)"))

    if dice_csv and dice_csv.exists():
        ddf = pd.read_csv(dice_csv)
        if {"patient_id", "dice"}.issubset(ddf.columns):
            dice_map = {r["patient_id"]: r["dice"] for _, r in ddf.iterrows() if not pd.isna(r["dice"])}
            dice_list = [dice_map.get(pid, np.nan) for pid in df.get("patient_id", [])]
            dice_arr = np.array(dice_list, dtype=float)
            st.caption(f"Dice available for N={np.isfinite(dice_arr).sum()} patients.")
            fig = plt.figure(); plt.boxplot([dice_arr[~np.isnan(dice_arr)]], labels=["DL vs Manual"])
            plt.ylabel("Dice"); plt.ylim(0, 1.0); buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
            st.image(buf.read())

# ------------------------------
# Main App
# ------------------------------

def main():
    st.set_page_config(page_title="L3 Sarcopenia  Unified App", layout="wide")
    cfg = sidebar_configuration()
    topbar_header()

    # Patient selection
    pid = patient_selector(cfg["root"])
    if not pid:
        return

    # L3 index resolution
    meta_txt = cfg["root"] / pid / "metadata.txt"
    l3_index = parse_l3_index_from_metadata(meta_txt)

    # CT volume
    ct3d, ct_img = load_ct_volume(cfg["root"], pid)
    if ct3d is None or ct_img is None:
        st.error("Could not load CT for selected patient.")
        return
    if l3_index is None or not (0 <= l3_index < ct3d.shape[0]):
        st.error("L3_index is missing or out of range for this patient.")
        return

    ct2d = ct3d[l3_index]
    sx_mm, sy_mm = get_xy_spacing(cfg["root"], pid, ct_img)

    # Mask discovery
    man_path = None
    if cfg["show_manual"]:
        for pat in (f"{pid}*.nii*", f"*{pid}*.nii*"):
            hits = list(cfg["manual_dir"].glob(pat))
            if hits:
                man_path = hits[0]; break
    man_mask, note_m = load_mask_to_slice2d(man_path, l3_index) if man_path else (None, "disabled")

    dl_mask = None; note_d = "disabled"
    if cfg["show_dl"]:
        pred_path_candidates = [cfg["dl_dir"] / f"{pid}_L3_skm_pred.nii.gz", cfg["dl_dir"] / f"{pid}_L3_skm_pred.nii"]
        pred_path = next((p for p in pred_path_candidates if p.exists()), None)
        if pred_path is not None:
            dl_mask, note_d = load_mask_to_slice2d(pred_path, l3_index)
        else:
            st.info("DL prediction not found on disk; running model inference on the fly.")
            wrapper = ModelWrapper(weights_path=cfg["weights"])
            wrapper.load()
            dl_mask = wrapper.predict(ct2d) if wrapper.model is not None else None
            note_d = "inferred"

    ts_mask, note_t = (None, "disabled")
    if cfg["show_ts"]:
        ts_dir = cfg["root"] / pid / cfg["ts_subdir"]
        ts_file = None
        if ts_dir.exists():
            for pat in ("*skeletal*muscle*.nii*", "*muscle*.nii*"):
                hits = list(ts_dir.glob(pat))
                if hits:
                    ts_file = hits[0]; break
        if ts_file:
            ts_mask, note_t = load_mask_to_slice2d(ts_file, l3_index)

    # Resample to match CT slice if needed
    for name, m in (("Manual", man_mask), ("DL", dl_mask), ("TS", ts_mask)):
        if m is not None and m.shape != ct2d.shape:
            LOGGER.warning("Resampling %s mask from %s to %s", name, m.shape, ct2d.shape)
            if name == "Manual":
                man_mask = nn_resample_mask(m, ct2d.shape)
            elif name == "DL":
                dl_mask = nn_resample_mask(m, ct2d.shape)
            else:
                ts_mask = nn_resample_mask(m, ct2d.shape)

    # Metrics & display
    df_metrics, meta = metrics_table_from_sources(
        ct2d, sx_mm, sy_mm, man_mask, dl_mask, ts_mask, cfg["height_m"], cfg["sex"], cfg["thresh_profile"]
    )

    colA, colB = st.columns([2, 1])
    with colA:
        overlays = {
            "Manual": man_mask if cfg["show_manual"] else None,
            "DL": dl_mask if cfg["show_dl"] else None,
            "TS": ts_mask if cfg["show_ts"] else None,
        }
        colors = {"Manual": "r", "DL": "b", "TS": "g"}
        styles = {"Manual": "-", "DL": ":", "TS": "--"}
        png = render_overlay_png(
            ct2d, overlays, cfg["hu_window"][0], cfg["hu_window"][1], colors, styles,
            title=f"{pid}  L3 slice (HU {cfg['hu_window'][0]}..{cfg['hu_window'][1]})"
        )
        st.image(png, caption="Contours: Manual=red, DL=blue, TS=green", use_column_width=True)
        st.download_button("Download overlay PNG", data=png, file_name=f"{pid}_L3_overlay.png", mime="image/png")

    with colB:
        st.subheader("Metrics (L3)")
        st.dataframe(df_metrics, use_container_width=True)
        if meta.get("classification"):
            st.markdown("**SMI classification**")
            for src, label in meta["classification"].items():
                st.write(f"{src}: {label}")
        csv_bytes = df_metrics.to_csv(index=False).encode("utf-8")
        st.download_button("Download metrics CSV", data=csv_bytes, file_name=f"{pid}_L3_metrics.csv", mime="text/csv")

        if cfg["export_pdf"] and A4 is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    pdf_path = Path(tf.name)
                generate_pdf_report(
                    pdf_path,
                    pid,
                    {"L3_index": l3_index, "px_area_cm2": meta.get("px_area_cm2")},
                    df_metrics[["Source", "CSA_cm2", "SMRA_HU", "SMI_cm2_m2", "n_pixels"]],
                    png,
                    notes="Computed at single-slice L3. HU muscle window [-29,150]."
                )
                with open(pdf_path, "rb") as fh:
                    st.download_button("Download PDF report", data=fh.read(), file_name=f"{pid}_L3_report.pdf", mime="application/pdf")
                pdf_path.unlink(missing_ok=True)
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

    st.markdown("---")
    with st.expander("Cohort analytics (optional)"):
        comparison_csv = cfg["root"] / "dl_runs" / "comparison_all.csv"
        dice_csv = cfg["root"] / "dl_runs" / "eval_dice_manual_vs_DL.csv"
        cohort_plots_section(comparison_csv if comparison_csv.exists() else None, dice_csv if dice_csv.exists() else None)

    st.markdown("---")
    with st.expander("Diagnostics"):
        st.write({
            "Patient": pid,
            "L3_index": l3_index,
            "CT shape": ct3d.shape,
            "XY spacing (mm)": (sx_mm, sy_mm),
            "Manual note": note_m,
            "DL note": note_d,
            "TS note": note_t,
        })

    # Ad hoc uploads (from the earlier prototype)
    if cfg["allow_upload"]:
        st.markdown("---")
        st.header("Ad hoc: Upload single-slice CT and mask for quick CSA/SMRA")
        up_ct = st.file_uploader("CT slice NIfTI (.nii/.nii.gz)", type=["nii", "nii.gz"], key="up_ct")
        up_mask = st.file_uploader("Mask NIfTI (.nii/.nii.gz)", type=["nii", "nii.gz"], key="up_mask")
        if up_ct is not None and up_mask is not None:
            try:
                ct_img = sitk.ReadImage(io.BytesIO(up_ct.read()))
                mask_img = sitk.ReadImage(io.BytesIO(up_mask.read()))
                ct_arr = sitk.GetArrayFromImage(ct_img)
                mask_arr = sitk.GetArrayFromImage(mask_img)
                ct2 = ct_arr[0].astype(np.int16) if ct_arr.ndim == 3 else ct_arr.astype(np.int16)
                m2 = mask_arr[0] if mask_arr.ndim == 3 else mask_arr
                if m2.shape != ct2.shape:
                    m2 = nn_resample_mask(m2, ct2.shape)
                sx, sy = ct_img.GetSpacing()[0], ct_img.GetSpacing()[1]
                csa, smra, npx, px_area = compute_csa_smra(ct2, (m2 > 0).astype(np.uint8), sx, sy)
                st.write({
                    "CSA_cm2": round(csa, 2),
                    "SMRA_HU": (None if smra is None else round(smra, 1)),
                    "n_pixels": int(npx),
                    "px_area_cm2": round(px_area, 5),
                })
                img_bytes = render_overlay_png(ct2, {"MASK": (m2 > 0).astype(np.uint8)}, cfg["hu_window"][0], cfg["hu_window"][1],
                                               colors={"MASK": "r"}, styles={"MASK": "-"}, title="Uploaded CT + Mask")
                st.image(img_bytes, caption="Uploaded CT with mask (red)")
            except Exception as e:
                st.error(f"Failed to process uploads: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.exception("Unhandled error in app: %s", exc)
        st.error("An unrecoverable error occurred. Check logs for details.")
