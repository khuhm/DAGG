import os
import json
import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import concurrent.futures
from functools import partial

# =================CONFIGURATION=================
# [Modified] Mode setting (can be changed to 'train' or 'test')
MODE = "test_merged"

BASE_PATHS = {
    "pred_dir": "/path/to/pred_dir/fold_0",
    "image_root": "/path/to/image_root",
    "gt_root": "/path/to/gt_root",
    "ts_root": "/path/to/ts_root",
    "dataset_json": "/path/to/dataset.json",
    "output_dir": "/path/to/output_dir"
}

MIN_PHYSICAL_SIZE_MM = (56.25, 56.25, 48.0)
MARGIN_RATIO = 0.3
LOBE_LABELS = [10, 11, 12, 13, 14]
LOBE_NAMES = ["LUL", "LLL", "RUL", "RML", "RLL"]
NUM_WORKERS = 8

# =================HELPER FUNCTIONS=================

def load_and_flatten_json(path):
    """
    Loads JSON and converts all cases into a {"filename": metadata} dictionary 
    without distinguishing keys like 'train', 'val', etc. 
    Improves search speed (O(1)) and removes key dependency.
    """
    with open(path, 'r') as f:
        raw_data = json.load(f)
    
    lookup_table = {}
    
    # Handle both cases where raw_data is a list or a dictionary
    if isinstance(raw_data, dict):
        for key, item_list in raw_data.items():
            # Structure where lists exist under keys like "train", "val", etc.
            if isinstance(item_list, list):
                for item in item_list:
                    if "name" in item:
                        lookup_table[item["name"]] = item
    elif isinstance(raw_data, list):
        # Case where the top level is a list
        for item in raw_data:
            if "name" in item:
                lookup_table[item["name"]] = item
                
    return lookup_table

def find_file_recursive(root_dir, filename):
    for dirpath, _, filenames in os.walk(root_dir):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def get_voxel_spacing(nifti_img):
    return nifti_img.header.get_zooms()

def calculate_crop_slice(center_idx, bbox_size, min_size, img_dim, margin_ratio):
    margin_size = bbox_size * (1 + margin_ratio)
    final_size = max(margin_size, min_size)
    radius = final_size / 2.0
    
    start = int(np.round(center_idx - radius))
    end = int(np.round(center_idx + radius))
    
    current_len = end - start
    if current_len < int(np.ceil(final_size)):
        end += 1
        
    return start, end

def pad_and_crop(data, slices, pad_value=0):
    is_4d = (data.ndim == 4)
    if is_4d:
        spatial_shape = data.shape[1:]
        data_spatial_dims = 3
    else:
        spatial_shape = data.shape
        data_spatial_dims = 3

    pads = []
    final_slices = []
    
    for i in range(data_spatial_dims):
        sl = slices[i]
        start, end = sl.start, sl.stop
        dim_len = spatial_shape[i]
        
        pad_before = max(0, -start)
        pad_after = max(0, end - dim_len)
        pads.append((pad_before, pad_after))
        
        valid_start = max(0, start)
        valid_end = min(dim_len, end)
        final_slices.append(slice(valid_start, valid_end))
        
    if is_4d:
        cropped = data[:, final_slices[0], final_slices[1], final_slices[2]]
        full_pads = [(0, 0)] + pads
    else:
        cropped = data[final_slices[0], final_slices[1], final_slices[2]]
        full_pads = pads

    if sum(sum(p) for p in full_pads) > 0:
        return np.pad(cropped, full_pads, mode='constant', constant_values=pad_value)
    else:
        return cropped

def process_case(pred_path, dataset_lookup, output_root):
    """
    dataset_lookup: Dictionary in the form of {"filename": {meta...}, ...}
    """
    try:
        filename = os.path.basename(pred_path)
        case_name = filename.replace('.nii.gz', '')
        
        # 1. Find files
        img_path = find_file_recursive(BASE_PATHS['image_root'], filename)
        gt_path = os.path.join(BASE_PATHS['gt_root'], filename)
        ts_path = find_file_recursive(BASE_PATHS['ts_root'], filename)
        
        if not (img_path and os.path.exists(gt_path) and ts_path):
            return f"[Skip] Missing files for {case_name}"

        # 2. Load data
        img_nii = nib.load(img_path)
        gt_nii = nib.load(gt_path)
        ts_nii = nib.load(ts_path)
        pred_nii = nib.load(pred_path)
        
        img_data = img_nii.get_fdata()
        gt_data = gt_nii.get_fdata()
        ts_data = ts_nii.get_fdata()
        pred_data = pred_nii.get_fdata()
        
        spacing = get_voxel_spacing(img_nii)
        min_voxels = np.array([m / s for m, s in zip(MIN_PHYSICAL_SIZE_MM, spacing)])
        
        # JSON matching (O(1) Lookup)
        case_meta = dataset_lookup.get(filename)
        
        if case_meta is None:
            # If metadata is completely missing, consider it a data issue and raise an error
            raise ValueError(f"Metadata not found for file: {filename}")
            
        case_out_dir = os.path.join(output_root, case_name)
        os.makedirs(case_out_dir, exist_ok=True)
        
        metadata_list = []
        
        # 3. Lung BBox
        lung_mask = np.isin(ts_data, LOBE_LABELS)
        if not np.any(lung_mask):
            lung_bbox_min = np.array([0, 0, 0])
            lung_bbox_size = np.array(img_data.shape)
        else:
            lung_locs = ndimage.find_objects(lung_mask.astype(int))[0]
            lung_bbox_min = np.array([s.start for s in lung_locs])
            lung_bbox_max = np.array([s.stop for s in lung_locs])
            lung_bbox_size = lung_bbox_max - lung_bbox_min

        # 4. Separate Instances
        labeled_pred, num_features = ndimage.label(pred_data)
        if num_features == 0:
            return f"[Skip] No instances in {case_name}"

        objects = ndimage.find_objects(labeled_pred)
        
        for idx, sl in enumerate(objects):
            if sl is None: continue
            instance_id = idx + 1
            
            # Instance Mask
            current_instance_mask = (labeled_pred == instance_id).astype(np.uint8)
            
            # BBox & Crop Slices
            starts = np.array([s.start for s in sl])
            stops = np.array([s.stop for s in sl])
            sizes = stops - starts
            centers = starts + sizes / 2.0
            
            crop_slices = []
            for dim in range(3):
                s, e = calculate_crop_slice(centers[dim], sizes[dim], min_voxels[dim], img_data.shape[dim], MARGIN_RATIO)
                crop_slices.append(slice(s, e))
            
            # 5. Crop
            cropped_img = pad_and_crop(img_data, crop_slices)
            cropped_inst_mask = pad_and_crop(current_instance_mask, crop_slices)
            cropped_ts = pad_and_crop(ts_data, crop_slices)
            cropped_gt = pad_and_crop(gt_data, crop_slices)
            
            # 6. GT Overlap & Merge
            matched_findings = []
            merged_gt_mask = np.zeros_like(cropped_inst_mask)
            
            for f_idx in range(cropped_gt.shape[0]):
                gt_finding_vol = cropped_gt[f_idx]
                if np.sum(gt_finding_vol) == 0: continue

                gt_labels = np.unique(gt_finding_vol)
                gt_labels = gt_labels[gt_labels > 0]

                finding_has_overlap = False
                for gt_lab in gt_labels:
                    gt_instance_mask = (gt_finding_vol == gt_lab).astype(np.uint8)
                    overlap = np.sum(cropped_inst_mask * gt_instance_mask)
                    if overlap > 0:
                        merged_gt_mask = np.maximum(merged_gt_mask, gt_instance_mask)
                        finding_has_overlap = True
                
                if finding_has_overlap:
                    # [Modified] Raise an error instead of Unknown (Strict Mode)
                    f_key = str(f_idx)
                    if "findings" not in case_meta:
                         raise ValueError(f"[Data Error] 'findings' key missing in metadata for {filename}")
                         
                    finding_text = case_meta["findings"].get(f_key)
                    
                    if finding_text is None:
                        # Data integrity issue: No text corresponding to the matched index -> Terminate immediately
                        raise ValueError(f"[Data Error] Finding index '{f_key}' not found in 'findings' for {filename}")
                    
                    matched_findings.append({
                        "finding_idx": int(f_idx),
                        "text": finding_text
                    })

            # 7. Lobe & Rel Coords
            lobe_counts = {lbl: 0 for lbl in LOBE_LABELS}
            total_voxels_in_crop = float(cropped_ts.size)
            for lbl in LOBE_LABELS:
                lobe_counts[lbl] = np.sum(cropped_ts == lbl)
            lobe_vector = [float(lobe_counts[lbl]) / total_voxels_in_crop if total_voxels_in_crop > 0 else 0.0 for lbl in LOBE_LABELS]
            
            max_idx = np.argmax(lobe_vector)
            dominant_lobe = LOBE_NAMES[max_idx] if lobe_vector[max_idx] > 0 else "None"
            
            crop_center_global = np.array([(sl.start + sl.stop)/2 for sl in crop_slices])
            lung_center_global = lung_bbox_min + lung_bbox_size / 2.0
            diff_from_center = crop_center_global - lung_center_global
            rel_coords = diff_from_center / np.maximum(lung_bbox_size, 1e-6)

            # 8. Save
            save_name_base = f"{case_name}_{instance_id}"
            
            nib.save(nib.Nifti1Image(cropped_img, img_nii.affine, img_nii.header), 
                     os.path.join(case_out_dir, f"{save_name_base}.nii.gz"))
            nib.save(nib.Nifti1Image(cropped_inst_mask.astype(np.uint8), img_nii.affine, img_nii.header), 
                     os.path.join(case_out_dir, f"{save_name_base}_pred.nii.gz"))
            nib.save(nib.Nifti1Image(merged_gt_mask.astype(np.uint8), img_nii.affine, img_nii.header), 
                     os.path.join(case_out_dir, f"{save_name_base}_gt.nii.gz"))

            inst_meta = {
                "instance_id": int(instance_id),
                "filename_img": f"{save_name_base}.nii.gz",
                "filename_pred": f"{save_name_base}_pred.nii.gz",
                "filename_gt": f"{save_name_base}_gt.nii.gz",
                "matched_findings": matched_findings,
                "crop_center_global_xyz": [float(x) for x in crop_center_global],
                "relative_coords_xyz": [float(x) for x in rel_coords],
                "lobe_vector": lobe_vector,
                "dominant_lobe": dominant_lobe
            }
            metadata_list.append(inst_meta)
            
        with open(os.path.join(case_out_dir, "metadata.json"), 'w') as f:
            json.dump(metadata_list, f, indent=4)
            
        return f"[Done] {case_name}"

    except Exception as e:
        # Upon error, throw the error object to the caller (main) or return a clear message
        # Raise to catch the exception in concurrent.futures
        raise e 

def main():
    # 1. Load JSON and Flatten (Create Lookup Table)
    print("Loading and indexing dataset JSON...")
    ds_lookup = load_and_flatten_json(BASE_PATHS['dataset_json'])
    print(f"Loaded metadata for {len(ds_lookup)} cases.")
    
    target_dir = os.path.join(BASE_PATHS['pred_dir'], MODE)
    if not os.path.exists(target_dir):
        # If pred_dir itself already ends with 'train' or 'test', use that path
        if os.path.basename(BASE_PATHS['pred_dir']) == MODE:
            target_dir = BASE_PATHS['pred_dir']
        else:
            print(f"Error: Target directory for mode '{MODE}' does not exist at:")
            print(f" -> {target_dir}")
            return

    final_output_dir = f"{BASE_PATHS['output_dir']}_{MODE}"
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir, exist_ok=True)
        
    print(f"Mode: {MODE}")
    print(f"Input Directory: {target_dir}")
    print(f"Output Directory: {final_output_dir}")

    pred_files = []
    for f in os.listdir(target_dir):
        if f.endswith('.nii.gz'):
            pred_files.append(os.path.join(target_dir, f))
    
    print(f"Found {len(pred_files)} predicted mask files in {target_dir}")
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    
    # Partial function binding
    process_func = partial(process_case, dataset_lookup=ds_lookup, output_root=final_output_dir)

    # Execute parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Store future objects in a list
        futures = {executor.submit(process_func, f): f for f in pred_files}
        
        # Use as_completed to process as they finish, implement logic to terminate immediately upon error
        try:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(pred_files)):
                try:
                    result = future.result() # Re-raise if there is an Exception here
                    # Print normal completion log if necessary
                except Exception as exc:
                    print(f"\n[CRITICAL ERROR] Process failed. Terminating immediately.")
                    print(f"Error details: {exc}")
                    
                    # Forcefully terminate all running processes
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc # Terminate main process

        except Exception as e:
            print("Program terminated due to data integrity error.")
            exit(1)
            
    print("All processing completed successfully.")

if __name__ == "__main__":
    main()