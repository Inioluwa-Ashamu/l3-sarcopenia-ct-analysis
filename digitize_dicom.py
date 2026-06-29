import os
import json
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import logging
import stat

# ----------------------------- Settings -----------------------------
RAW_ROOT = os.environ.get("ATTDS_RAW_ROOT", "raw_data")
OUTPUT_ROOT = os.environ.get("ATTDS_DATA_ROOT", "anon_dig")
ANONYMIZE = True  # set to False to keep PHI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------- Core Functions -----------------------------

def extract_all_dicom_files(directory):
    dicom_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                if ds.Modality == "CT":
                    dicom_paths.append(path)
            except:
                continue
    return dicom_paths

def group_by_series_uid(dicom_paths):
    series_dict = {}
    for path in dicom_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            uid = ds.SeriesInstanceUID
            series_dict.setdefault(uid, []).append(path)
        except:
            continue
    return series_dict

def select_largest_series(series_dict):
    return max(series_dict.items(), key=lambda item: len(item[1]), default=(None, []))[1]

def ensure_sorted_by_position(dicom_paths):
    slices = []
    for f in dicom_paths:
        try:
            ds = pydicom.dcmread(f)
            slices.append((f, float(ds.ImagePositionPatient[2])))
        except:
            continue
    slices.sort(key=lambda x: x[1])
    return [f for f, _ in slices]

def convert_to_numpy(sorted_dicom_paths):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_dicom_paths)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    return array, image

def normalize_hu(image_array, min_hu=-1024, max_hu=400):
    image_array = (image_array - min_hu) / (max_hu - min_hu)
    return np.clip(image_array, 0, 1)

def extract_pixel_spacing(image):
    spacing = image.GetSpacing()
    return {
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "slice_thickness_mm": spacing[2]
    }

def save_metadata_json(metadata, output_path):
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info("Metadata saved.")

def save_slice_positions(image, output_path):
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    depth = image.GetSize()[2]
    positions = [origin[2] + i * spacing[2] for i in range(depth)]
    np.save(os.path.join(output_path, "z_positions.npy"), positions)
    logging.info("Z-slice positions saved.")

def visualize_middle_slice(original, normalized, output_path):
    mid = original.shape[0] // 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original[mid], cmap='gray')
    axs[0].set_title("Original HU")
    axs[0].axis('off')
    axs[1].imshow(normalized[mid], cmap='gray')
    axs[1].set_title("Normalized HU")
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Middle slice preview saved: {output_path}")

def save_as_nifti(array, sitk_image, output_path):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(sitk_image)
    sitk.WriteImage(image, output_path)
    logging.info(f"NIfTI saved: {output_path}")

def anonymize_dicom_file(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, force=True)
        for tag in ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "OtherPatientIDs"]:
            if tag in ds:
                del ds[tag]
        os.chmod(dicom_path, stat.S_IWRITE)
        ds.save_as(dicom_path)
    except Exception as e:
        logging.warning(f"Failed to anonymize {dicom_path}: {e}")

def save_patient_data(np_array, norm_array, sitk_image, output_path):
    os.makedirs(output_path, exist_ok=True)
    # np.save(os.path.join(output_path, "original.npy"), np_array)
    # np.save(os.path.join(output_path, "normalized.npy"), norm_array)
    save_as_nifti(np_array, sitk_image, os.path.join(output_path, "original.nii.gz"))
    save_as_nifti(norm_array, sitk_image, os.path.join(output_path, "normalized.nii.gz"))
    visualize_middle_slice(np_array, norm_array, os.path.join(output_path, "middle_slice.png"))
    save_metadata_json(extract_pixel_spacing(sitk_image), output_path)
    save_slice_positions(sitk_image, output_path)
    with open(os.path.join(output_path, "metadata.txt"), "w") as f:
        f.write(f"Slices: {np_array.shape[0]}\n")
        f.write(f"Shape: {np_array.shape[1]} x {np_array.shape[2]}\n")
        f.write("L3_index: TBD\n")
    logging.info(f"All patient data saved to: {output_path}")

# ----------------------------- Main Runner -----------------------------
if __name__ == "__main__":
    if len(os.sys.argv) >= 2 and os.sys.argv[1]:
        RAW_ROOT = os.sys.argv[1]
    if len(os.sys.argv) >= 3 and os.sys.argv[2]:
        OUTPUT_ROOT = os.sys.argv[2]
    logging.info(" Starting digitization...")

    patient_dirs = [os.path.join(RAW_ROOT, d) for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))]

    for patient_path in sorted(patient_dirs):
        patient_id = os.path.basename(patient_path)
        output_path = os.path.join(OUTPUT_ROOT, patient_id)

        dicom_paths = extract_all_dicom_files(patient_path)
        logging.info(f"{patient_id}: {len(dicom_paths)} DICOM files")

        if not dicom_paths:
            logging.error(f"{patient_id}: No DICOMs found")
            continue

        if ANONYMIZE:
            for p in dicom_paths:
                anonymize_dicom_file(p)

        series_dict = group_by_series_uid(dicom_paths)
        largest_series = select_largest_series(series_dict)

        if not largest_series:
            logging.error(f"{patient_id}: No CT series found")
            continue

        sorted_paths = ensure_sorted_by_position(largest_series)

        try:
            np_array, sitk_image = convert_to_numpy(sorted_paths)
            norm_array = normalize_hu(np_array)
            save_patient_data(np_array, norm_array, sitk_image, output_path)
            logging.info(f"Done: {patient_id}")
        except Exception as e:
            logging.error(f"Failed on {patient_id}: {e}")
