# Automated Tool to Detect Sarcopenia - Portfolio Review

## 1. End-to-end project summary

**What the project does**
This project builds a Python pipeline for opportunistic sarcopenia analysis from abdominal CT imaging. It converts CT DICOM series into NIfTI volumes, identifies the L3 vertebral level, extracts skeletal muscle masks, calculates body-composition biomarkers, compares automated methods against manual masks, generates evaluation plots, and provides a Streamlit viewer for reviewing patient-level outputs.

**Problem solved**
Sarcopenia is associated with frailty, poor surgical outcomes, oncology prognosis, and reduced resilience in older or clinically vulnerable patients. Manual assessment on CT is time-consuming and requires specialist imaging workflows. This project explores whether routine CT scans can be processed into reproducible L3 muscle measurements that could support research or future clinical decision-support workflows.

**Why L3 matters**
The cross-sectional skeletal muscle area at the third lumbar vertebra is widely used as a proxy for whole-body muscle mass. At L3, skeletal muscle cross-sectional area (CSA), skeletal muscle radiation attenuation (SMRA), and, where height is available, skeletal muscle index (SMI) can be derived from CT imaging.

**How the system works technically**
The current workflow is script-based:

1. `digitize_dicom.py` discovers CT DICOM files, groups them by SeriesInstanceUID, selects the largest CT series, converts it to `original.nii.gz`, saves a normalised NIfTI volume, and writes spacing metadata.
2. `run_all_segmentation.py` runs TotalSegmentator for the L3 vertebra and tissue masks.
3. `write_l3_index_from_mask.py` finds the axial slice with the largest `vertebrae_L3` mask area and writes it to `metadata.txt`.
4. `export_l3_pairs.py` exports L3 CT slices and skeletal muscle masks into `.npz` training pairs.
5. `train_l3_unet.py` trains a compact 2D U-Net on those L3 slice/mask pairs.
6. `predict_l3_unet.py` produces L3 skeletal muscle predictions as 3D NIfTI masks with only the L3 slice populated.
7. `compute_all_csa_smra.py`, `eval_dice_manual.py`, and `eval_dice_manual_vs_TS.py` compute CSA, SMRA, and Dice agreement.
8. `make_all_plots.py` generates histograms, scatter plots, Bland-Altman plots, boxplots, CDFs, and quality-note summaries.
9. `L3 Sarcopenia Streamlit App-proV5.py` provides a single-patient and cohort-review interface.

**Inputs and outputs**

Inputs:
- CT DICOM folders.
- Manual L3 skeletal muscle masks where available.
- TotalSegmentator outputs.
- Optional height and sex CSV for SMI and threshold-based labels.

Outputs:
- `original.nii.gz`, `normalized.nii.gz`, `metadata.json`, `metadata.txt`, and preview PNGs per patient.
- L3 vertebra and tissue segmentation masks.
- L3 training pairs in `.npz` format.
- U-Net weights in `dl_runs/l3_unet_best.pt`.
- DL prediction masks in `dl_preds`.
- `comparison_all.csv`, Dice CSVs, summary text, and plots.
- Streamlit visual overlays, downloadable CSV metrics, PNG overlays, and PDF reports.

## 2. Technical architecture breakdown

**Data ingestion**
Implemented in `digitize_dicom.py`. The script recursively finds CT DICOM files, groups by series UID, selects the largest series, sorts slices by z-position, converts the series using SimpleITK, and stores metadata. It also attempts to anonymise selected DICOM tags. This is a useful start, but anonymisation is destructive because it edits source files in place, and it only removes a small PHI subset.

**DICOM/NIfTI processing**
The repo uses `pydicom`, `SimpleITK`, and NIfTI outputs. CT volumes are represented as `(Z, Y, X)` arrays. Pixel spacing is used to convert mask pixel counts into area. Some metadata handling is inconsistent: `metadata.json` is written with `spacing_x_mm` and `spacing_y_mm`, while one script first looks for a `spacing` list and then falls back to the CT header.

**L3 slice identification**
L3 is not learned directly by the U-Net. The implemented L3 localisation uses TotalSegmentator's `vertebrae_L3.nii.gz`, then selects the axial slice with maximum positive L3 mask area. The chosen index is written to `metadata.txt`.

**Segmentation approach**
There are three mask sources:
- Manual masks: used as reference annotations where available.
- TotalSegmentator tissue masks: used to extract skeletal muscle and fat compartments.
- Custom 2D U-Net: trained on exported L3 CT slice and skeletal muscle mask pairs.

**U-Net / TotalSegmentator usage**
The U-Net is a compact 2D binary segmentation model with BCE plus Dice loss. It predicts skeletal muscle on the already selected L3 slice. TotalSegmentator is used both for L3 vertebra detection and tissue segmentation. In practice, the custom model appears to be trained from TotalSegmentator-derived skeletal muscle masks unless manual-mask training pairs are substituted.

**Biomarker calculation**
CSA is calculated as:

`positive_mask_pixels * pixel_area_cm2`

SMRA is calculated as mean CT attenuation within the mask after applying the common skeletal muscle HU window:

`-29 HU to 150 HU`

SMI is optionally calculated when height is available:

`SMI = CSA_cm2 / height_m^2`

The Streamlit app includes threshold-based SMI classification using a configurable profile.

**Evaluation and reporting pipeline**
The repo evaluates overlap using Dice scores for DL vs manual and TS vs manual. It also compares CSA and SMRA across manual, DL, and TS masks. `make_all_plots.py` generates distribution plots, agreement plots, Bland-Altman plots, Dice plots, CDFs, and data-quality note counts.

**Streamlit app functionality**
The app supports patient selection, L3 CT display, manual/DL/TS contour overlays, CSA/SMRA/SMI calculation, metrics CSV export, PNG overlay export, PDF report generation, cohort analytics from CSV outputs, and diagnostic metadata display. It is strong as a dissertation prototype, but should be split into modules before being presented as production-style software.

## 3. Repo improvement plan

### Recommended structure

```text
sarcopenia-l3-analysis/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  .env.example
  LICENSE
  src/
    sarcopenia_l3/
      __init__.py
      config.py
      io/
        dicom.py
        nifti.py
      preprocessing/
        digitise.py
        l3_index.py
      segmentation/
        totalsegmentator_runner.py
        unet.py
        predict.py
      metrics/
        body_composition.py
        dice.py
      reporting/
        plots.py
        pdf.py
  app/
    streamlit_app.py
  scripts/
    digitise_dataset.py
    run_totalsegmentator.py
    export_training_pairs.py
    train_unet.py
    predict_unet.py
    evaluate.py
    make_plots.py
  tests/
    test_metrics.py
    test_l3_index.py
    test_dice.py
    test_metadata.py
  docs/
    methodology.md
    data_schema.md
    limitations.md
    model_card.md
    screenshots/
  examples/
    sample_metadata.json
    sample_metrics.csv
    demo_outputs/
```

### Rename and reorganise

- Rename `L3 Sarcopenia Streamlit App-proV5.py` to `app/streamlit_app.py`.
- Rename `digitize_dicom.py` to `scripts/digitise_dataset.py` or `scripts/digitize_dataset.py` and move core functions into `src/sarcopenia_l3/io/`.
- Move U-Net architecture into `src/sarcopenia_l3/segmentation/unet.py`.
- Move metric logic into `src/sarcopenia_l3/metrics/body_composition.py`.
- Move plotting functions into `src/sarcopenia_l3/reporting/plots.py`.
- Replace hard-coded paths with CLI arguments and/or environment variables.
- Use one spelling convention: either UK `digitise` or US `digitize`, not both.

### Add

- `pyproject.toml` or `setup.cfg` for package metadata and tooling.
- `.env.example` showing `DATA_ROOT`, `MANUAL_MASK_DIR`, `TS_EXE`, and `MODEL_WEIGHTS`.
- `.gitignore` for medical images, model weights, outputs, and local environment files.
- Small synthetic test fixtures that do not contain patient data.
- Unit tests for CSA, SMRA, Dice, L3 index extraction, metadata parsing, and mask resampling.
- A model card describing task, intended use, inputs, outputs, limitations, and non-clinical status.
- A data card explaining that real CT data cannot be included.
- Screenshots or synthetic/demo outputs for the README.
- A clear pipeline diagram.
- A `Makefile` or `justfile` for common commands.

### Clean up

- Remove personal local paths such as `C:\Users\sophi\Downloads\ATTDS` and `E:\ATTDS`.
- Fix mojibake/encoding issues in README and comments.
- Remove duplicate plotting scripts or clearly mark them as legacy.
- Avoid in-place anonymisation of raw DICOM files.
- Add structured logging and fail-fast validation for missing data.
- Do not claim deterministic reproducibility unless random seeds and environment pinning are added.

## 4. Improved README

```markdown
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

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU inference or training, install the appropriate PyTorch build for your CUDA version. TotalSegmentator may require additional setup depending on platform and model-cache configuration.

## Usage

Set paths in a config file or environment variables, then run the workflow:

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
streamlit run "L3 Sarcopenia Streamlit App-proV5.py"
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
- No public demo imaging data is currently provided.
- Scripts contain hard-coded local paths and should be parameterised.
- Current U-Net operates on selected 2D L3 slices rather than performing full 3D localisation.
- L3 localisation currently depends on TotalSegmentator output.
- Anonymisation logic is limited and should not be treated as a complete DICOM de-identification pipeline.

## Future Work

- Refactor into a tested Python package.
- Add CLI configuration and reproducible experiment tracking.
- Add synthetic fixtures and unit tests.
- Include a model card, data card, and screenshots.
- Validate on an independent dataset with documented cohort details.
- Compare against additional segmentation baselines.
- Improve DICOM de-identification and audit logging.
- Containerise the app for reproducible deployment.

## Ethical and Privacy Considerations

Medical imaging data is sensitive and must be handled under appropriate governance, consent, anonymisation, and access-control procedures. This repository should not include identifiable DICOM files or patient-derived images unless they are fully approved for public release. Outputs should be interpreted as research measurements, not clinical decisions.

## Technologies

Python, PyTorch, SimpleITK, pydicom, NumPy, pandas, matplotlib, nibabel, TotalSegmentator, Streamlit, reportlab.
```

## 5. CV content

**Short 2-line version**
Built a Python research pipeline for L3 sarcopenia analysis from CT scans, combining DICOM/NIfTI processing, TotalSegmentator, a 2D U-Net, CSA/SMRA biomarker calculation, evaluation plots, and a Streamlit review app.
Developed as an MSc AI dissertation project focused on medical imaging automation, reproducible analysis, and clinical-AI workflow design.

**Stronger project-entry version**
Automated L3 Sarcopenia Analysis from CT Imaging - MSc AI Dissertation Project
Designed and implemented an end-to-end medical imaging pipeline to process CT DICOM scans, identify the L3 vertebral level, segment skeletal muscle using TotalSegmentator and a compact PyTorch U-Net, calculate CSA/SMRA/optional SMI biomarkers, evaluate automated masks against manual annotations, and present patient-level results through an interactive Streamlit application.

**NHS/data/AI bullets**

- Built a reproducible Python workflow for CT image processing, L3 slice selection, segmentation, biomarker calculation, evaluation, and reporting.
- Converted CT DICOM series into NIfTI volumes using SimpleITK and extracted image-spacing metadata for quantitative analysis.
- Used TotalSegmentator to support vertebral landmarking and tissue segmentation, then calculated L3 skeletal muscle CSA and SMRA.
- Implemented a compact PyTorch 2D U-Net for L3 skeletal muscle segmentation using exported CT/mask training pairs.
- Compared manual, deep-learning, and TotalSegmentator-derived masks using Dice scores and CSA/SMRA agreement analysis.
- Generated audit-friendly CSV outputs and research plots including histograms, scatter plots, Bland-Altman plots, boxplots, and CDFs.
- Developed a Streamlit app for patient-level image review, segmentation overlays, metric export, cohort analytics, and PDF reporting.
- Documented limitations around clinical validation, data privacy, deployment readiness, and research-only use.

**Healthtech/medical-imaging version**
Developed an end-to-end CT body-composition analysis prototype for L3 sarcopenia assessment, integrating DICOM ingestion, NIfTI conversion, TotalSegmentator-based vertebral localisation, 2D U-Net skeletal muscle segmentation, CSA/SMRA biomarker extraction, manual-mask comparison, and a Streamlit-based clinical review interface. The project demonstrates practical understanding of medical image formats, pixel-spacing-aware measurement, segmentation evaluation, and the governance constraints of clinical AI.

**Junior AI/data science version**
Built a medical imaging machine-learning project using Python, PyTorch, NumPy, pandas, SimpleITK, and Streamlit. The pipeline prepares CT image data, trains a small U-Net segmentation model, calculates quantitative features from model outputs, compares predictions with reference masks, and produces visual reports for analysis. The project strengthened my skills in data preprocessing, model training, evaluation, visualisation, and communicating limitations responsibly.

## 6. LinkedIn and portfolio content

**LinkedIn project description**
I built an MSc AI dissertation project exploring automated sarcopenia analysis from CT scans. The system processes CT DICOM data, converts scans to NIfTI, identifies the L3 vertebral level using TotalSegmentator output, segments skeletal muscle using both TotalSegmentator-derived masks and a compact PyTorch U-Net, then calculates CSA and SMRA biomarkers. I also developed evaluation scripts for Dice and agreement analysis, generated Bland-Altman and distribution plots, and built a Streamlit app for reviewing overlays and exporting patient-level metrics. The project is research-only and not clinically validated, but it gave me hands-on experience with medical imaging workflows, clinical AI limitations, and reproducible analysis.

**GitHub pinned-project description**
Research prototype for L3 sarcopenia analysis from CT scans using DICOM/NIfTI processing, TotalSegmentator, PyTorch U-Net segmentation, CSA/SMRA biomarker calculation, evaluation plots, and a Streamlit review app.

**Portfolio case study structure**

1. Problem: why sarcopenia matters and why CT-based L3 analysis is useful.
2. Dataset and constraints: anonymised CT data, manual masks where available, no public patient data.
3. Workflow: DICOM to NIfTI, L3 identification, segmentation, metrics, evaluation, app.
4. Technical implementation: SimpleITK, pydicom, TotalSegmentator, PyTorch U-Net, Streamlit.
5. Quantitative outputs: CSA, SMRA, optional SMI, Dice, agreement plots.
6. Interface: screenshots of overlays, metrics table, CSV/PDF export.
7. Limitations: no clinical validation, no regulatory approval, local paths, limited tests.
8. Improvements: package refactor, tests, config, model card, synthetic demo data.
9. What I learned: medical image handling, metric calculation, model evaluation, responsible clinical-AI communication.

**Short interview explanation**
This was my MSc AI dissertation project. I built a research pipeline that takes CT imaging data, converts DICOM into NIfTI, identifies the L3 slice using a vertebra segmentation from TotalSegmentator, and then measures skeletal muscle at that level. I implemented a compact 2D U-Net for skeletal muscle segmentation, compared outputs with manual masks and TotalSegmentator masks, calculated CSA and SMRA, and generated evaluation plots and a Streamlit viewer. The key thing I would stress is that it is a research prototype, not a clinically validated tool, but it shows practical end-to-end medical imaging work from data ingestion through model evaluation and user-facing reporting.

## 7. Job application positioning

**NHS data analyst roles**
Emphasise data quality, reproducible pipelines, CSV reporting, governance awareness, and clinical context. Do not over-focus on deep learning. Position it as evidence that you can work with healthcare data, cleanly document outputs, and understand the importance of auditability and privacy.

**Junior data scientist roles**
Emphasise preprocessing, model training, evaluation metrics, visualisation, and limitations. Explain the feature/metric extraction clearly: image segmentation produces masks, masks produce quantitative biomarkers, biomarkers are compared across methods.

**AI engineer roles**
Emphasise PyTorch model implementation, inference workflow, model-output handling, modularisation opportunities, and deployment via Streamlit. Be honest that the current repo needs engineering hardening before production.

**Healthtech roles**
Emphasise the full workflow: medical data ingestion, clinical motivation, interpretable outputs, clinician-facing interface, privacy considerations, and safe framing. Healthtech employers will value your ability to avoid overclaiming.

**Medical imaging / clinical AI roles**
Emphasise DICOM/NIfTI handling, SimpleITK, segmentation masks, pixel spacing, HU windows, Dice, Bland-Altman analysis, TotalSegmentator, and L3 body-composition methodology. Be precise that L3 localisation is based on TotalSegmentator rather than learned from scratch.

## 8. Honest limitations

Do not claim unless proven and documented:

- Do not claim clinical validation.
- Do not claim NHS deployment.
- Do not claim regulatory approval or medical-device readiness.
- Do not claim diagnostic accuracy.
- Do not claim large-scale performance unless dataset size and cohort details are reported.
- Do not claim the U-Net identifies L3 automatically; the repo uses TotalSegmentator L3 vertebra output for L3 slice selection.
- Do not claim robust anonymisation; current anonymisation removes only selected tags and edits source files in place.
- Do not claim production quality; scripts contain hard-coded paths and lack tests.
- Do not claim public reproducibility without demo data, config, seeds, and environment pinning.
- Do not claim superiority over TotalSegmentator unless evaluation results support it.

What you can fairly claim:

- You built an end-to-end research prototype.
- You worked with DICOM, NIfTI, CT HU values, spacing-aware measurements, and segmentation masks.
- You implemented a PyTorch U-Net segmentation workflow.
- You integrated TotalSegmentator into a body-composition pipeline.
- You calculated CSA, SMRA, and optional SMI.
- You evaluated automated masks against manual annotations where available.
- You built a Streamlit viewer for overlays, metrics, and reporting.

## 9. Priority action list: 1-3 day application-ready plan

1. Fix the README encoding issues and replace it with the improved README above.
2. Remove or parameterise all hard-coded local paths using CLI arguments, `.env`, or a config file.
3. Rename the Streamlit file to `app/streamlit_app.py`.
4. Add `.gitignore`, `.env.example`, and a short `docs/data_privacy.md`.
5. Add screenshots or synthetic/demo output images for the README.
6. Add a `docs/model_card.md` with intended use, inputs, outputs, limitations, and research-only status.
7. Add unit tests for `compute_csa_smra`, Dice calculation, L3 index extraction, and metadata parsing.
8. Consolidate duplicate plotting/evaluation scripts or mark older scripts as legacy.
9. Add a clear `example_config.yaml` and show one command sequence that runs the workflow.
10. Add a pipeline diagram to the README.
11. Create a `docs/methodology.md` explaining L3 selection, HU windows, CSA, SMRA, SMI, and evaluation.
12. Add a short `results.md` only if you can include real evaluated numbers with dataset size and cohort description.

## Overall assessment

This is a strong portfolio foundation because it combines medical imaging, AI segmentation, quantitative biomarker calculation, evaluation, visualisation, and an interactive app. The strongest parts are the end-to-end thinking, practical image-processing workflow, use of established tools, and awareness of clinical measurement outputs.

The weakest parts are repository engineering: hard-coded paths, no tests, no packaged structure, no demo data, no model card, no documented dataset characteristics, and README encoding problems. For applications, the project should be presented as a serious research prototype with responsible limitations, not as a deployable clinical tool.
