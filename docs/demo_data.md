# Demo Data

The assets in `docs/assets/` and the CSV in `examples/` are synthetic portfolio examples. They are designed to show the repository workflow without exposing patient imaging or patient-derived outputs.

These files should not be used as evidence of model performance. Real evaluation results should only be reported with a documented dataset, annotation protocol, train/test split, and governance approval for any images or derived outputs.

The project intentionally excludes:

- raw DICOM data,
- NIfTI volumes,
- trained model weights,
- patient-level overlays,
- generated reports that could contain clinical context.

Use the synthetic files for README illustrations and smoke-testing downstream plotting code only.
