# Model Card

## Model

Compact 2D U-Net implemented in PyTorch for binary skeletal muscle segmentation on selected L3 CT slices.

## Intended Use

Research exploration of automated L3 skeletal muscle segmentation and biomarker extraction from CT images.

## Not Intended For

- Clinical diagnosis.
- Treatment decisions.
- Autonomous screening.
- NHS deployment without validation, governance review, and regulatory assessment.

## Inputs

- A single axial CT slice at the L3 vertebral level.
- CT values clipped and normalised for model inference.

## Outputs

- Binary skeletal muscle mask for the L3 slice.
- Downstream CSA, SMRA, and optional SMI metrics when processed by the metric scripts.

## Training Data

Training pairs are exported by `export_l3_pairs.py` from CT volumes and available skeletal muscle masks. The public repository does not include patient data, cohort details, or trained weights.

## Evaluation

The project includes scripts for Dice overlap and CSA/SMRA agreement analysis against manual masks where available. Any reported performance should include dataset size, cohort details, annotation source, and evaluation split.

## Limitations

- 2D L3 slice segmentation only.
- L3 localisation depends on TotalSegmentator.
- No public external validation documented in this repository.
- No clinical validation or regulatory approval.
- Performance may vary with scanner protocol, anatomy, pathology, slice thickness, and annotation conventions.
