# Methodology

## Clinical Rationale

Skeletal muscle area at the L3 vertebral level is commonly used in CT body-composition research as a proxy for whole-body muscle mass. This project explores an automated workflow for extracting L3 skeletal muscle measurements from CT imaging.

## Workflow

1. CT DICOM series are discovered, grouped by `SeriesInstanceUID`, sorted by z-position, and converted to NIfTI.
2. TotalSegmentator is used to segment the L3 vertebra.
3. The L3 axial slice is selected as the slice with the largest positive `vertebrae_L3` mask area.
4. Skeletal muscle masks are obtained from manual annotations, TotalSegmentator tissue outputs, or a compact 2D U-Net.
5. CSA and SMRA are calculated from the L3 skeletal muscle mask and CT Hounsfield units.
6. Automated masks are compared with manual masks where available using Dice and agreement plots.

## Biomarkers

Cross-sectional area:

```text
CSA_cm2 = positive_mask_pixels * pixel_area_cm2
```

Skeletal muscle radiation attenuation:

```text
SMRA_HU = mean CT HU inside skeletal muscle mask within -29 to 150 HU
```

Skeletal muscle index, when height is available:

```text
SMI_cm2_m2 = CSA_cm2 / height_m^2
```

## Evaluation

The repository supports:

- Dice similarity for segmentation overlap.
- CSA and SMRA comparison across manual, U-Net, and TotalSegmentator outputs.
- Scatter plots and Bland-Altman plots for agreement analysis.
- Distribution plots and quality-note summaries.

## Key Technical Assumption

The custom U-Net segments skeletal muscle on an already selected 2D L3 slice. It does not perform full-volume L3 localisation. L3 localisation currently depends on TotalSegmentator vertebra segmentation.
