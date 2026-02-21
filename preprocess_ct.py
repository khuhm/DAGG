import os
import glob
import shutil
import json
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Configuration ---
INPUT_ROOT = "/path/to/input/crop_patches/processed_output_all_test_merged"
OUTPUT_ROOT = "/path/to/output/crop_patches/npy_output_lobes_all_test_merged"

# Multiprocessing Settings
NUM_WORKERS = 8  # Set to 8 workers as requested

# Resampling Target
TARGET_SPACING = (0.703125, 0.703125, 1.0) # (x, y, z)

# Normalization Parameters
CLIP_MIN = -999.0
CLIP_MAX = 205.0
NORM_MEAN = -549.9546508789062
NORM_STD = 312.59503173828125

def resample_volume(image, target_spacing, is_label=False):
    """
    Resample the volume using SimpleITK.
    - image: sitk image object
    - target_spacing: (x, y, z) tuple
    - is_label: Use NearestNeighbor if True, else use Linear interpolation
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size (maintaining physical size)
    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        
    return resampler.Execute(image)

def process_case(case_path):
    """
    Process a single case folder (e.g., train_6_a_2).
    """
    case_name = os.path.basename(case_path)
    output_case_dir = os.path.join(OUTPUT_ROOT, case_name)
    os.makedirs(output_case_dir, exist_ok=True)
    
    # 1. Copy Metadata
    meta_src = os.path.join(case_path, "metadata.json")
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(output_case_dir, "metadata.json"))
        
    # 2. Find image files (search only for original images first, not prediction or gt)
    all_nii = glob.glob(os.path.join(case_path, "*.nii.gz"))
    image_files = [f for f in all_nii if "_pred.nii.gz" not in f and "_gt.nii.gz" not in f]
    
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(".nii.gz", "")
        
        # Define the paired file paths
        pred_path = os.path.join(case_path, f"{base_name}_pred.nii.gz")
        gt_path = os.path.join(case_path, f"{base_name}_gt.nii.gz")
        
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            # Skip if pairs do not match (print if logging is needed)
            continue

        try:
            # --- Load Data ---
            sitk_img = sitk.ReadImage(img_path)
            sitk_pred = sitk.ReadImage(pred_path)
            sitk_gt = sitk.ReadImage(gt_path)

            # --- Resample ---
            # Image: Linear / Mask: Nearest Neighbor
            sitk_img_res = resample_volume(sitk_img, TARGET_SPACING, is_label=False)
            sitk_pred_res = resample_volume(sitk_pred, TARGET_SPACING, is_label=True)
            sitk_gt_res = resample_volume(sitk_gt, TARGET_SPACING, is_label=True)
            
            # --- Convert to Numpy (z, y, x) ---
            arr_img = sitk.GetArrayFromImage(sitk_img_res)
            arr_pred = sitk.GetArrayFromImage(sitk_pred_res)
            arr_gt = sitk.GetArrayFromImage(sitk_gt_res)
            
            # --- Preprocess Image (Clip & Normalize) ---
            arr_img = np.clip(arr_img, CLIP_MIN, CLIP_MAX)
            arr_img = (arr_img - NORM_MEAN) / NORM_STD
            
            # --- Transpose to (h, w, d) i.e., (y, x, z) ---
            # sitk (z, y, x) -> (y, x, z)
            # arr_img = arr_img.transpose(1, 2, 0)
            # arr_pred = arr_pred.transpose(1, 2, 0)
            # arr_gt = arr_gt.transpose(1, 2, 0)
            
            # --- Save as .npy ---
            np.save(os.path.join(output_case_dir, f"{base_name}.npy"), arr_img.astype(np.float32))
            np.save(os.path.join(output_case_dir, f"{base_name}_pred.npy"), arr_pred.astype(np.uint8))
            np.save(os.path.join(output_case_dir, f"{base_name}_gt.npy"), arr_gt.astype(np.uint8))
            
        except Exception as e:
            print(f"[Error] Failed processing {base_name} in {case_name}: {e}")

def main():
    # Search by folder
    case_dirs = [d for d in glob.glob(os.path.join(INPUT_ROOT, "*")) if os.path.isdir(d)]
    
    # case_dirs = [case_dirs[42]]

    print(f"Target: {len(case_dirs)} cases.")
    print(f"Processing with {NUM_WORKERS} workers...")
    
    # Apply max_workers=8
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_case, case_dirs), total=len(case_dirs)))

    print("All processing completed.")

if __name__ == "__main__":
    main()