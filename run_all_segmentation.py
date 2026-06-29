# file: run_all_segmentation.py
# Purpose: Run all required TotalSegmentator segmentations reproducibly per patient.
# Segs: L3 vertebrae (for indexing), tissue types (skeletal_muscle / fat compartments),
#       optional muscles subset. Designed for Windows-friendly Unicode handling.

import os
import sys
import csv
import time
import json
import logging
import subprocess
from datetime import datetime

# ----------------------- CONFIG (edit as needed) -----------------------
DIGITIZED_ROOT = os.environ.get("ATTDS_DATA_ROOT", "anon_dig")        # where patientXXX folders live
TS_EXE         = os.environ.get("ATTDS_TS_EXE", "TotalSegmentator")
TIMEOUT_SEC    = 36000                         # per-case timeout
LOG_LEVEL      = "INFO"

# Which segmentations to run
RUN_L3        = True
RUN_TISSUE    = True
RUN_MUSCLES   = False   # optional; useful if your TS build supports a dedicated "muscles" subset

# Output folder names (created inside each patient folder)
OUT_L3_NAME      = "segmentations2_L3"
OUT_TISSUE_NAME  = "segmentations2_tissue"
OUT_MUSCLE_NAME  = "segmentations2_muscles"

# ----------------------- LOGGING -----------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------- UTILS -------------------------
def is_patient_dir(name: str) -> bool:
    return name.lower().startswith("patient")

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def ts_run(cmd, timeout=TIMEOUT_SEC):
    """
    Run a subprocess with robust text decoding on Windows.
    Returns (ok: bool, stdout: str, stderr: str).
    """
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout
        )
        return True, res.stdout, res.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired as e:
        return False, "", f"TimeoutExpired after {timeout}s"

def write_manifest_row(manifest_path, row):
    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(["timestamp_iso","patient_id","task","output_dir","status","note"])
        wr.writerow(row)

def get_ts_version(ts_exe: str) -> str:
    ok, out, err = ts_run([ts_exe, "--version"], timeout=60)
    return (out or err or "unknown").strip()

def already_has_files(out_dir: str) -> bool:
    return os.path.isdir(out_dir) and any(fn.endswith(".nii.gz") for fn in os.listdir(out_dir))

# ----------------------- PER-TASK RUNNERS -----------------------
def run_l3(patient_id: str, patient_dir: str, ts_exe: str, manifest_path: str):
    input_nii = os.path.join(patient_dir, "original.nii.gz")
    out_dir   = ensure_dir(os.path.join(patient_dir, OUT_L3_NAME))
    expected_mask = os.path.join(out_dir, "vertebrae_L3.nii.gz")

    if not os.path.exists(input_nii):
        logging.warning("%s: no original.nii.gz; skipping L3.", patient_id)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "L3", out_dir, "skipped", "missing original.nii.gz"])
        return

    if os.path.exists(expected_mask):
        logging.info("%s: L3 already present -> %s", patient_id, expected_mask)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "L3", out_dir, "ok", "exists"])
        return

    logging.info("%s: running L3 vertebra segmentation ...", patient_id)
    cmd = [ts_exe, "-i", input_nii, "-o", out_dir, "--roi_subset", "vertebrae_L3"]
    ok, out, err = ts_run(cmd)
    if ok and os.path.exists(expected_mask):
        logging.info(" %s: L3 done.", patient_id)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "L3", out_dir, "ok", ""])
    else:
        logging.error("%s: L3 failed.\n%s\n%s", patient_id, out, err)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "L3", out_dir, "failed", (err or out)[:3000]])

def run_tissue(patient_id: str, patient_dir: str, ts_exe: str, manifest_path: str):
    """
    Prefer 'tissue_4_types' (skeletal_muscle, subcutaneous_fat, torso_fat, intermuscular_fat).
    Fallback to 'tissue_types'. Fallback to full labels (heavier).
    """
    input_nii = os.path.join(patient_dir, "original.nii.gz")
    out_dir   = ensure_dir(os.path.join(patient_dir, OUT_TISSUE_NAME))

    if not os.path.exists(input_nii):
        logging.warning("%s: no original.nii.gz; skipping tissue.", patient_id)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "tissue", out_dir, "skipped", "missing original.nii.gz"])
        return

    # consider at least skeletal_muscle as evidence of completion
    expected_any = [ "skeletal_muscle.nii.gz", "muscle.nii.gz", "skeletal_muscles.nii.gz" ]
    if os.path.isdir(out_dir) and any(os.path.exists(os.path.join(out_dir, f)) for f in expected_any):
        logging.info("%s: tissue types already present -> %s", patient_id, out_dir)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "tissue", out_dir, "ok", "exists"])
        return

    tried = []
    for task in ["tissue_4_types", "tissue_types", None]:  # None => full labels
        if task:
            cmd = [ts_exe, "-i", input_nii, "-o", out_dir, "--task", task]
            tried.append(" ".join(cmd))
            logging.info("%s: running TS task=%s ...", patient_id, task)
        else:
            cmd = [ts_exe, "-i", input_nii, "-o", out_dir]
            tried.append(" ".join(cmd))
            logging.info("%s: running TS full labels (fallback) ...", patient_id)

        ok, out, err = ts_run(cmd)
        if ok:
            # check presence of at least one expected output
            if any(os.path.exists(os.path.join(out_dir, f)) for f in expected_any) or any(fn.endswith(".nii.gz") for fn in os.listdir(out_dir)):
                logging.info("%s: tissue segmentation done (mode=%s).", patient_id, task or "full")
                write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, f"tissue({task or 'full'})", out_dir, "ok", ""])
                return
        logging.warning("%s: attempt failed for %s.\n%s\n%s", patient_id, task or "full", out, err)

    # If we get here, all attempts failed
    logging.error("%s: all tissue attempts failed.\nTried:\n%s", patient_id, "\n".join(tried))
    write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "tissue", out_dir, "failed", "all attempts"])

def run_muscles(patient_id: str, patient_dir: str, ts_exe: str, manifest_path: str):
    """
    Optional dedicated 'muscles' subset (if supported by your TS build).
    """
    input_nii = os.path.join(patient_dir, "original.nii.gz")
    out_dir   = ensure_dir(os.path.join(patient_dir, OUT_MUSCLE_NAME))

    if not os.path.exists(input_nii):
        logging.warning("%s: no original.nii.gz; skipping muscles.", patient_id)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "muscles", out_dir, "skipped", "missing original.nii.gz"])
        return

    if already_has_files(out_dir):
        logging.info("%s: muscles already present -> %s", patient_id, out_dir)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "muscles", out_dir, "ok", "exists"])
        return

    logging.info("%s: running TS --roi_subset muscles ...", patient_id)
    cmd = [ts_exe, "-i", input_nii, "-o", out_dir, "--roi_subset", "muscles"]
    ok, out, err = ts_run(cmd)
    if ok and already_has_files(out_dir):
        logging.info("%s: muscles done.", patient_id)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "muscles", out_dir, "ok", ""])
    else:
        logging.warning("%s: muscles subset not supported or empty; stderr:\n%s", patient_id, err or out)
        write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "muscles", out_dir, "failed", "subset unsupported or empty"])

# ----------------------- MAIN -----------------------
def main():
    # Provenance capture (version file at root)
    prov_dir = ensure_dir(os.path.join(DIGITIZED_ROOT, "_provenance"))
    with open(os.path.join(prov_dir, "totalsegmentator_version.txt"), "w", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\n{get_ts_version(TS_EXE)}\n")

    manifest_path = os.path.join(DIGITIZED_ROOT, "segmentation_manifest.csv")

    # Iterate patients
    for name in sorted(os.listdir(DIGITIZED_ROOT)):
        if not is_patient_dir(name):
            continue
        patient_id = name
        patient_dir = os.path.join(DIGITIZED_ROOT, name)
        input_nii = os.path.join(patient_dir, "original.nii.gz")

        if not os.path.isdir(patient_dir):
            continue
        if not os.path.exists(input_nii):
            logging.warning("%s: missing original.nii.gz; skipping all.", patient_id)
            write_manifest_row(manifest_path, [datetime.now().isoformat(), patient_id, "all", "", "skipped", "missing original.nii.gz"])
            continue

        # L3 vertebra
        if RUN_L3:
            run_l3(patient_id, patient_dir, TS_EXE, manifest_path)

        # Tissue types
        if RUN_TISSUE:
            run_tissue(patient_id, patient_dir, TS_EXE, manifest_path)

        # Optional: muscles subset
        if RUN_MUSCLES:
            run_muscles(patient_id, patient_dir, TS_EXE, manifest_path)

    logging.info("All done. Manifest: %s", manifest_path)

if __name__ == "__main__":
    # Allow overriding TS_EXE or ROOT via CLI (optional)
    # Example: python run_all_segmentation.py "TotalSegmentator" "anon_dig"
    if len(sys.argv) >= 2 and sys.argv[1]:
        TS_EXE = sys.argv[1]
    if len(sys.argv) >= 3 and sys.argv[2]:
        DIGITIZED_ROOT = sys.argv[2]
    main()
