# Data Privacy and Governance

This project is designed for research and portfolio presentation. It should not contain identifiable medical images, DICOM metadata, patient names, hospital identifiers, dates of birth, accession numbers, or derived images that could identify a person.

## Current Privacy Position

- The repository does not include the original CT dataset.
- Large imaging files and generated patient artefacts are excluded by `.gitignore`.
- The DICOM digitisation script removes a small set of common identifying tags, but this is not a complete de-identification workflow.

## Important Limitation

`digitize_dicom.py` should not be treated as a production anonymisation tool. Medical-image de-identification requires a formal policy, DICOM tag review, burned-in pixel review, audit logging, and local information-governance approval.

## Recommended Practice

- Keep raw DICOM data outside the repository.
- Use approved anonymisation tooling before research use.
- Do not commit patient-derived screenshots or overlays unless explicitly approved for public release.
- Use synthetic or fully approved demo data for GitHub screenshots.
- Document dataset access, consent basis, and governance approvals in private project records.
