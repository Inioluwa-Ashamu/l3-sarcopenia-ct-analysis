# Automated L3 Sarcopenia Analysis from CT Imaging

Research prototype for automated skeletal muscle analysis at the L3 vertebral level using CT imaging, TotalSegmentator, a compact 2D U-Net, and quantitative body-composition metrics.

> Research use only. This project is not clinically validated, not regulatory approved, and must not be used for diagnosis or treatment decisions.

## Overview

Sarcopenia, the loss of skeletal muscle mass and function, is associated with frailty, poor surgical outcomes, oncology prognosis, and reduced physiological reserve. Abdominal CT scans often contain enough information to estimate skeletal muscle area at the L3 vertebral level, a commonly used imaging landmark for body-composition research.

This repository implements an end-to-end research workflow for:

- converting CT DICOM series to NIfTI,
- identifying the L3 slice,
- segmenting skeletal muscle using TotalSegmentator and a 2D U-Net,
- calculating CSA, SMRA, and optional SMI,
- comparing automated masks with manual annotations,
- generating evaluation figures,
- reviewing outputs in a Streamlit app.

## Features

- DICOM discovery, series grouping, CT conversion, and metadata extraction.
- L3 slice selection from TotalSegmentator vertebra output.
- Skeletal muscle and tissue-mask processing at L3.
- Compact PyTorch 2D U-Net for L3 skeletal muscle segmentation.
- CSA and SMRA calculation using CT spacing and HU filtering.
- Optional SMI calculation where height is available.
- Dice, agreement, Bland-Altman, histogram, and boxplot reporting.
- Streamlit interface for patient review, overlays, CSV export, and PDF reports.

## Architecture

```text
DICOM CT series
  -> NIfTI conversion and metadata extraction
  -> TotalSegmentator L3 vertebra segmentation
  -> L3 slice index selection
  -> TotalSegmentator tissue masks and/or U-Net prediction
  -> L3 skeletal muscle mask
  -> CSA, SMRA, optional SMI
  -> CSV metrics, plots, overlays, Streamlit review
```

## Repository Structure

```text
.
├── app/
│   └── streamlit_app.py
├── docs/
│   ├── data_privacy.md
│   ├── methodology.md
│   └── model_card.md
├── tests/
│   └── test_metrics.py
├── digitize_dicom.py
├── run_all_segmentation.py
├── write_l3_index_from_mask.py
├── export_l3_pairs.py
├── train_l3_unet.py
├── predict_l3_unet.py
├── compute_all_csa_smra.py
├── eval_dice_manual.py
├── eval_dice_manual_vs_TS.py
├── make_all_plots.py
├── requirements.txt
└── example_config.yaml
```

## Installation

```bash
git clone https://github.com/Inioluwa-Ashamu/Automated-Tool-To-Detect-Sarcopenia.git
cd Automated-Tool-To-Detect-Sarcopenia
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU inference or training, install the appropriate PyTorch build for your CUDA version. TotalSegmentator may require additional setup depending on platform and model-cache configuration.

## Configuration

Most scripts can be configured using environment variables. Copy `.env.example` and adapt the paths for your local machine.

```powershell
$env:ATTDS_DATA_ROOT="C:\path\to\anon_dig"
$env:ATTDS_RAW_ROOT="C:\path\to\raw_dicom"
$env:ATTDS_MANUAL_DIR="C:\path\to\manual_masks"
$env:ATTDS_TS_EXE="TotalSegmentator"
```

The scripts still support direct editing for quick dissertation-style runs, but environment variables are preferred for reproducibility and GitHub presentation.

## Usage

```bash
python digitize_dicom.py
python run_all_segmentation.py
python write_l3_index_from_mask.py
python export_l3_pairs.py
python train_l3_unet.py
python predict_l3_unet.py
python compute_all_csa_smra.py
python eval_dice_manual.py
python eval_dice_manual_vs_TS.py
python make_all_plots.py
streamlit run app/streamlit_app.py
```

## Example Workflow

1. Place anonymised CT DICOM folders under a dataset root.
2. Convert each CT series into `original.nii.gz`.
3. Run TotalSegmentator to produce `vertebrae_L3.nii.gz` and tissue masks.
4. Write the L3 slice index into each patient's metadata.
5. Generate L3 skeletal muscle predictions using the U-Net or use TotalSegmentator masks.
6. Calculate CSA and SMRA at L3.
7. Compare automated masks with manual masks where available.
8. Review overlays and metrics in Streamlit.

## Outputs

- Per-patient NIfTI CT volumes and metadata.
- L3 vertebra masks and tissue masks.
- U-Net skeletal muscle prediction masks.
- `comparison_all.csv` with CSA and SMRA across mask sources.
- Dice evaluation CSVs.
- Histograms, scatter plots, Bland-Altman plots, boxplots, and CDFs.
- PNG overlays and optional PDF reports.

## Evaluation

The project evaluates segmentation agreement using Dice similarity against manual masks where available. It also compares derived biomarkers such as CSA and SMRA across manual, U-Net, and TotalSegmentator outputs. Agreement is visualised using scatter plots and Bland-Altman plots.

Performance values should be reported only after running the evaluation pipeline on a clearly described dataset.

## Limitations

- Research prototype only; not a medical device.
- No clinical validation or regulatory approval.
- Dataset size and cohort characteristics are not included in this repository.
- No public demo CT imaging data is currently provided.
- Current U-Net operates on selected 2D L3 slices rather than performing full 3D localisation.
- L3 localisation currently depends on TotalSegmentator output.
- DICOM anonymisation support is limited and should not be treated as a complete de-identification pipeline.

## Future Work

- Refactor scripts into a tested Python package.
- Add CLI argument parsing for every workflow stage.
- Add synthetic imaging fixtures and broader unit tests.
- Include screenshots and demo outputs generated from non-patient synthetic data.
- Validate on an independent dataset with documented cohort details.
- Compare against additional segmentation baselines.
- Improve DICOM de-identification and audit logging.
- Containerise the app for reproducible deployment.

## Ethical and Privacy Considerations

Medical imaging data is sensitive and must be handled under appropriate governance, consent, anonymisation, and access-control procedures. This repository should not include identifiable DICOM files or patient-derived images unless they are fully approved for public release. Outputs should be interpreted as research measurements, not clinical decisions.

## Technologies

Python, PyTorch, SimpleITK, pydicom, NumPy, pandas, matplotlib, nibabel, TotalSegmentator, Streamlit, reportlab.

## Citation

Ashamu, I. (2025). *Automated Tool to Detect Sarcopenia from CT Scans*. MSc Artificial Intelligence Dissertation, Manchester Metropolitan University.
