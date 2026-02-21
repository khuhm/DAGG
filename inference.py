import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import random 

# ==========================================
# 1. Configuration (Must be modified according to the user environment)
# ==========================================
# Path settings
INFERENCE_MODE = 'test' # val / test
ENABLE_STITCHING = True

BASE_DIR = "/path/to/ReXGroundingCT"

NPY_OUTPUT_DIR = os.path.join(BASE_DIR, "crop_patches/npy_output_lobes_all_train_merged")
PROCESSED_DIR = "/path/to/crop_patches/processed_output_all_train_merged"

TEST_NPY_DIR = os.path.join(BASE_DIR, "crop_patches/npy_output_lobes_all_test_merged")
PROCESSED_TEST_DIR = "/path/to/crop_patches/processed_output_all_test_merged"

SEGMENTATION_REF_DIR = "/path/to/dataset/ReXGroundingCT/segmentations"

# [Modified] Dynamic generation of OUTPUT_DIR path (ID + "_" + Mode)
EXP_NAME = "TrainVal_EXP_ID" # Experiment ID (Timestamp, etc.)
INF_RESULT_ROOT = "/path/to/sigmoid_contrastive_loss/inference_results"

# Final path example: .../TrainVal_EXP_ID_validation
OUTPUT_DIR = os.path.join(INF_RESULT_ROOT, f"{EXP_NAME}_{INFERENCE_MODE}")
MODEL_CHECKPOINT = f"/path/to/sigmoid_contrastive_loss/experiments/{EXP_NAME}/models/best_model.pth"
DATASET_JSON_PATH = "/path/to/dataset/ReXGroundingCT/dataset.json"
TEXT_EMB_DIR = os.path.join(BASE_DIR, "text_embeddings/Qwen3-Embedding-4B")

# [Dataset Split Settings] - Set identical to the training code
NUM_TRAIN_CASES = 2692
NUM_VAL_CASES = 300
SHUFFLE_CASES = True
SHUFFLE_SEED = 42

# Model hyperparameters (must be identical to training settings)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (48, 80, 80) # Patch Size
TEXT_INPUT_DIM = 2560        # Qwen Embedding Dim
OUTPUT_DIM = 2560
LAYERS_PER_SCALE = [1, 3, 4, 6, 6] # Example setting (needs confirmation with training code)
BASE_CHANNELS = 32
USE_RESIDUAL_BLOCK = True
USE_METADATA_FUSION = True    # Appears as True in the code
USE_TEXT_PROJ = True
THRESHOLD = 0.5            # Positive classification threshold

LEARNABLE_SCALE = False       # True: Learnable (nn.Parameter), False: Fixed
LEARNABLE_BIAS = False        # True: Learnable (nn.Parameter), False: Fixed

INIT_SCALE_FACTOR = 1.0      # Initialized with np.log(INIT_SCALE_FACTOR). (e.g., 1/0.07 ≈ 14.28)
INIT_BIAS_VALUE = 0.0  

USE_DROPOUT = False        # True: Use Dropout (Prevent Overfitting)
DROPOUT_RATE = 0.5        # 0.3 ~ 0.5 recommended
FC_DROPOUT_RATE = 0.5     # Dropout for Fully Connected Layer (Usually set higher than Conv)
USE_CONV_POOLING = False   # True: Use Conv Pooling, False: Use existing Pooling
USE_MLP_PROJECTOR = False  # True: Use MLP Projector, False: Use Linear Projector

# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TotalSegmentator Lobe Labels
LOBE_LABELS = {
    'left_upper': 10,
    'left_lower': 11,
    'right_upper': 12,
    'right_middle': 13,
    'right_lower': 14
}

# Map for converting Metadata String (e.g., "RML") to ID
LOBE_STR_TO_ID = {
    "LUL": 10, "LLL": 11, 
    "RUL": 12, "RML": 13, "RLL": 14
}

def parse_findings_location(text):
    return None

def get_inference_config(mode):
    """
    Return (case list, data path, prediction result path) depending on mode
    """
    if mode == 'val':
        # (Maintain existing code) For Validation during training (last N cases)
        print(f">>> Mode: VALIDATION (Using last {NUM_VAL_CASES} cases from Train set)")
        print(f"Loading cases from: {NPY_OUTPUT_DIR}")
        
        all_cases = [
            d for d in os.listdir(NPY_OUTPUT_DIR) 
            if os.path.isdir(os.path.join(NPY_OUTPUT_DIR, d))
        ]
        
        if SHUFFLE_CASES:
            random.Random(SHUFFLE_SEED).shuffle(all_cases)
        else:
            all_cases.sort()
            
        target_cases = sorted(all_cases[-NUM_VAL_CASES:])
        return target_cases, NPY_OUTPUT_DIR, PROCESSED_DIR

    elif mode == 'test':
        # [Modified] Use "val" split from dataset.json as Test target
        print(f">>> Mode: TEST (Using 'val' split from dataset.json)")
        
        if not os.path.exists(DATASET_JSON_PATH):
            raise FileNotFoundError(f"Dataset JSON file not found: {DATASET_JSON_PATH}")
            
        with open(DATASET_JSON_PATH, 'r') as f:
            ds_raw = json.load(f)
            
        if "val" not in ds_raw:
            raise KeyError("Key 'val' not found in dataset.json")
            
        # Extract names from JSON "val" list and remove .nii.gz
        target_cases = []
        for item in ds_raw["val"]:
            case_name = item["name"].replace(".nii.gz", "")
            target_cases.append(case_name)
            
        print(f"Loaded {len(target_cases)} cases from dataset.json ['val']")
        
        # Use existing TEST_NPY_DIR for data path (Assuming patch data exists in that path)
        return sorted(target_cases), TEST_NPY_DIR, PROCESSED_TEST_DIR

    else:
        raise ValueError(f"Invalid INFERENCE_MODE: {mode}. Use 'val' or 'test'.")
    
# [Helper Function] Define Center Crop function (located at the top of the code or inside the function)
def center_crop_3d_numpy(img, mask, target_size=(48, 80, 80)):
    """
    Input: (D, H, W) numpy array
    Output: Center cropped (target_size) array
    """
    d, h, w = img.shape
    td, th, tw = target_size
    
    # If the input is smaller than the target, it should be left as is or padded to prevent errors,
    # But here we assume the input is larger and only perform Crop.
    if d < td or h < th or w < tw:
        # If the input is smaller, padding logic is needed, 
        # Since the user's case (81->80) is cropping, the logic below is sufficient.
        # Use min as a safety measure
        pass 

    # Calculate start index (center alignment)
    d_s = max(0, (d - td) // 2)
    h_s = max(0, (h - th) // 2)
    w_s = max(0, (w - tw) // 2)
    
    d_e = min(d, d_s + td)
    h_e = min(h, h_s + th)
    w_e = min(w, w_s + tw)
    
    # Perform Crop
    img_cropped = img[d_s:d_e, h_s:h_e, w_s:w_e]
    
    mask_cropped = None
    if mask is not None:
        mask_cropped = mask[d_s:d_e, h_s:h_e, w_s:w_e]
        
    return img_cropped, mask_cropped

# ==========================================
# 2. Model Architecture (Restore provided code)
# ==========================================
class ResidualBlock3d(nn.Module):
    def __init__(self, channels, use_instance_norm=True):
        super(ResidualBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(channels, affine=True) if use_instance_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(channels, affine=True) if use_instance_norm else nn.Identity()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ContrastiveNet(nn.Module):
    def __init__(self, 
                 text_input_dim=2560,     # [Added] Original text dimension
                 output_dim=512, 
                 layers_per_scale=[1, 1], 
                 base_channels=32,
                 use_instance_norm=False,
                 use_map=False,
                 mask_input=False,
                 mask_margin=1,
                 use_text_proj=False,
                 use_text_proj_mlp=False,
                 use_metadata_fusion=False,
                 use_metadata_mlp=False,
                 meta_dim=0,
                 meta_embed_dim=64,
                 use_residual=False): 
        super(ContrastiveNet, self).__init__()
        
        self.use_text_proj = use_text_proj
        self.use_map = use_map
        self.mask_input = mask_input
        self.use_residual = use_residual
        self.use_conv_pooling = USE_CONV_POOLING
        self.use_mlp_projector = USE_MLP_PROJECTOR

        if isinstance(mask_margin, int):
            self.mask_margin = (mask_margin, mask_margin, mask_margin)
        else:
            # Assume (D, H, W) order
            self.mask_margin = tuple(mask_margin)

        self.use_metadata_fusion = use_metadata_fusion
        self.use_metadata_mlp = use_metadata_mlp

        self.num_scales = len(layers_per_scale)
        
        layers = []
        in_channels = 1
        current_channels = base_channels
        
        for scale_idx, num_layers in enumerate(layers_per_scale):
            # 1. Downsampling (Applied after Scale 0)
            if scale_idx > 0:
                layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, stride=2, padding=1))
                if use_instance_norm:
                    layers.append(nn.InstanceNorm3d(current_channels, affine=True))
                layers.append(nn.ReLU())
                # [NEW] Apply Dropout3d after Downsampling
                if USE_DROPOUT:
                    layers.append(nn.Dropout3d(p=DROPOUT_RATE))
                in_channels = current_channels

            if USE_RESIDUAL_BLOCK and in_channels != current_channels:
                layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, padding=1))
                if use_instance_norm:
                    layers.append(nn.InstanceNorm3d(current_channels, affine=True))
                layers.append(nn.ReLU(inplace=True))
                if USE_DROPOUT:
                        layers.append(nn.Dropout3d(p=DROPOUT_RATE))
                in_channels = current_channels

            # ------------------------------------------------------------------
            # 2. Main Processing Blocks
            # ------------------------------------------------------------------
            if USE_RESIDUAL_BLOCK:
                for i in range(num_layers):
                    layers.append(ResidualBlock3d(current_channels, use_instance_norm))
            else:
                for i in range(num_layers):
                    layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, padding=1))
                    if use_instance_norm:
                        layers.append(nn.InstanceNorm3d(current_channels, affine=True))
                    layers.append(nn.ReLU(inplace=True))
                    if USE_DROPOUT:
                        layers.append(nn.Dropout3d(p=DROPOUT_RATE))
                    in_channels = current_channels
            
            # Expand channel only when it's not the last scale
            if scale_idx < self.num_scales - 1:
                current_channels *= 2 

        self.features = nn.Sequential(*layers)

        if USE_CONV_POOLING:
            # 1. Adjust to 3x5x5 regardless of incoming size (safety measure & compression)
            self.pooling_adapter = nn.AdaptiveAvgPool3d((3, 5, 5))
            
            # 2. Conv that reads the entire 3x5x5 at once and vectorizes it
            #    (Effectively the same as a Dense Layer with unshared parameters)
            self.final_pool_conv = nn.Sequential(
                nn.Conv3d(current_channels, current_channels, 
                          kernel_size=(3, 5, 5), bias=False), # bias=False if using Norm
                nn.LayerNorm([current_channels, 1, 1, 1]),    # Perform channel-wise normalization even if it's 1x1x1
                nn.ReLU(inplace=True) 
            )

            # Initialization (Xavier)
            for m in self.final_pool_conv.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight)

        if self.use_metadata_fusion:
            if self.use_metadata_mlp:
                # [Option A] Use MLP Encoding
                # Expand dimension and add nonlinearity from (Raw Dim -> Embed Dim)
                self.meta_encoder = nn.Sequential(
                    nn.Linear(meta_dim, meta_embed_dim),
                    nn.LayerNorm(meta_embed_dim),
                    nn.ReLU(),
                    nn.Linear(meta_embed_dim, meta_embed_dim)
                )
                added_dim = meta_embed_dim
            else:
                # [Option B] Raw Concatenation
                # No separate Encoder, use Raw Dimension as is
                self.meta_encoder = nn.Identity()
                added_dim = meta_dim
            
            # Confirm the dimension to be added to image features
            proj_input_dim = current_channels + added_dim
        else:
            # Metadata not used
            proj_input_dim = current_channels

        if self.use_mlp_projector:
            # MLP Style: Linear -> LayerNorm -> ReLU -> Linear
            # It is common to keep the Hidden Dimension identical to the input dimension
            hidden_dim = output_dim
            self.img_projector = nn.Sequential(
                nn.Linear(proj_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), # [Key Point] Use LN instead of BN for Small Batch
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            # *Note: Dropout is usually not applied to the Projection Head.
        else:
            # Simple Linear Style
            self.img_projector = nn.Linear(proj_input_dim, output_dim)
    
        # --- [MODIFIED] Text Projector Initialization ---
        if self.use_text_proj:
            if use_text_proj_mlp:
                # [NEW] MLP structure: Linear -> ReLU -> Linear
                # Maintain text_input_dim for Hidden Layer dimension to minimize information loss
                self.text_projector = nn.Sequential(
                    nn.Linear(text_input_dim, text_input_dim), 
                    nn.ReLU(), # Or nn.GELU()
                    nn.Linear(text_input_dim, output_dim)
                )
            else:
                # Existing method: Simple Linear
                self.text_projector = nn.Linear(text_input_dim, output_dim)

        if self.use_text_proj:
            self.text_projector = nn.Linear(text_input_dim, output_dim)

        init_scale_val = np.log(INIT_SCALE_FACTOR)
        if LEARNABLE_SCALE:
            self.logit_scale = nn.Parameter(torch.ones([]) * init_scale_val)
        else:
            # Register as buffer so it is saved in state_dict and follows device movement even if not learned
            self.register_buffer("logit_scale", torch.ones([]) * init_scale_val)

        # 2. Bias
        # Initial value: INIT_BIAS_VALUE
        if LEARNABLE_BIAS:
            self.bias = nn.Parameter(torch.ones([]) * INIT_BIAS_VALUE)
        else:
            self.register_buffer("bias", torch.ones([]) * INIT_BIAS_VALUE)

    def forward(self, x, mask=None, coords=None, lobe_vecs=None):
        # [Added] 0. Input Masking: Start by masking the input
        if self.mask_input and mask is not None:
            # Perform Dilation if any margin is greater than 1
            # (No change if kernel_size is 1)
            if any(m > 1 for m in self.mask_margin):
                k_d, k_h, k_w = self.mask_margin
                
                # Padding calculation: Half of Kernel size (floor) -> Same Padding effect (odd kernel recommended)
                pad_d, pad_h, pad_w = k_d // 2, k_h // 2, k_w // 2
                
                # Apply Anisotropic Max Pooling
                applied_mask = F.max_pool3d(
                    mask, 
                    kernel_size=(k_d, k_h, k_w), 
                    stride=1, 
                    padding=(pad_d, pad_h, pad_w)
                )
            else:
                applied_mask = mask
                
            x = x * applied_mask

        # 1. Feature Extraction
        feat = self.features(x)
        
        # 2. Pooling Logic (Modified with Macro)
        if USE_CONV_POOLING:
            # [CHANGED] Conv Pooling Path
            # (1) Forcibly adjust to 3x5x5 size
            # feat_adapted = self.pooling_adapter(feat) 
            
            # (2) Apply 3x5x5 Conv -> (B, C, 1, 1, 1)
            feat_pooled = self.final_pool_conv(feat)
            
            # (3) Flatten -> (B, C)
            avg_feat = feat_pooled.flatten(1)
            
        elif self.use_map and mask is not None:
            # Existing Masked Average Pooling Path
            target_size = feat.shape[2:]
            resized_mask = F.adaptive_avg_pool3d(mask, target_size)
            sum_feat = torch.sum(feat * resized_mask, dim=(2, 3, 4))
            sum_mask = torch.sum(resized_mask, dim=(2, 3, 4))
            avg_feat = sum_feat / (sum_mask + 1e-6)
        else:
            # Existing Global Average Pooling Path
            avg_feat = torch.mean(feat, dim=(2, 3, 4))
        
        # 2. Metadata Fusion Branch
        if self.use_metadata_fusion:
            if coords is None or lobe_vecs is None:
                raise ValueError("Metadata required when use_metadata_fusion is True")
            
            # Combine (B, meta_dim)
            raw_meta = torch.cat([coords, lobe_vecs], dim=1)
            
            if self.use_metadata_mlp:
                # [Option A] Pass through MLP: (B, 64)
                meta_feat = self.meta_encoder(raw_meta)
            else:
                # [Option B] Use as is: (B, 9)
                meta_feat = raw_meta
            
            # (B, Image_Ch + Meta_Feat)
            avg_feat = torch.cat([avg_feat, meta_feat], dim=1)
            
        return self.img_projector(avg_feat) # Return Image embedding

    def encode_text(self, text_emb):
        # --- [Added] Text Forward ---
        if self.use_text_proj:
            return self.text_projector(text_emb)
        return text_emb # Return as is if not used


# class ContrastiveNet(nn.Module):
#     def __init__(self, 
#                  text_input_dim=2560, output_dim=512, layers_per_scale=[1, 1], 
#                  base_channels=32, use_instance_norm=False, use_map=False,
#                  mask_input=False, mask_margin=1, use_text_proj=False,
#                  use_text_proj_mlp=False, use_metadata_fusion=True,
#                  use_metadata_mlp=False, meta_dim=8, meta_embed_dim=64, # meta_dim=8 (assuming coord 3 + lobe 5)
#                  use_residual=False): 
#         super(ContrastiveNet, self).__init__()
        
#         self.use_text_proj = use_text_proj
#         self.use_map = use_map
#         self.mask_input = mask_input
#         self.use_residual = use_residual
#         self.mask_margin = (mask_margin, mask_margin, mask_margin) if isinstance(mask_margin, int) else tuple(mask_margin)
#         self.use_metadata_fusion = use_metadata_fusion
#         self.use_metadata_mlp = use_metadata_mlp
#         self.num_scales = len(layers_per_scales := layers_per_scale)
        
#         layers = []
#         in_channels = 1
#         current_channels = base_channels
        
#         for scale_idx, num_layers in enumerate(layers_per_scale):
#             if scale_idx > 0:
#                 layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, stride=2, padding=1))
#                 if use_instance_norm: layers.append(nn.InstanceNorm3d(current_channels, affine=True))
#                 layers.append(nn.ReLU())
#                 in_channels = current_channels

#             if use_residual and in_channels != current_channels:
#                 layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, padding=1))
#                 if use_instance_norm: layers.append(nn.InstanceNorm3d(current_channels, affine=True))
#                 layers.append(nn.ReLU(inplace=True))
#                 in_channels = current_channels

#             if use_residual:
#                 for i in range(num_layers): layers.append(ResidualBlock3d(current_channels, use_instance_norm))
#             else:
#                 for i in range(num_layers):
#                     layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, padding=1))
#                     if use_instance_norm: layers.append(nn.InstanceNorm3d(current_channels, affine=True))
#                     layers.append(nn.ReLU(inplace=True))
#                     in_channels = current_channels
            
#             if scale_idx < self.num_scales - 1:
#                 current_channels *= 2 

#         self.features = nn.Sequential(*layers)

#         if self.use_metadata_fusion:
#             if self.use_metadata_mlp:
#                 self.meta_encoder = nn.Sequential(
#                     nn.Linear(meta_dim, meta_embed_dim), nn.LayerNorm(meta_embed_dim), nn.ReLU(),
#                     nn.Linear(meta_embed_dim, meta_embed_dim)
#                 )
#                 added_dim = meta_embed_dim
#             else:
#                 self.meta_encoder = nn.Identity()
#                 added_dim = meta_dim
#             proj_input_dim = current_channels + added_dim
#         else:
#             proj_input_dim = current_channels

#         self.img_projector = nn.Linear(proj_input_dim, output_dim)
    
#         if self.use_text_proj:
#             if use_text_proj_mlp:
#                 self.text_projector = nn.Sequential(
#                     nn.Linear(text_input_dim, text_input_dim), nn.ReLU(), nn.Linear(text_input_dim, output_dim)
#                 )
#             else:
#                 self.text_projector = nn.Linear(text_input_dim, output_dim)

#         # Scale & Bias (Register as buffer to correspond with state_dict load)
#         self.register_buffer("logit_scale", torch.ones([]) * np.log(14.28)) # Arbitrary initial setting (overwritten upon load)
#         self.register_buffer("bias", torch.zeros([]))

#     def forward(self, x, mask=None, coords=None, lobe_vecs=None):
#         if self.mask_input and mask is not None:
#              x = x * mask # Simplified masking logic for inference
        
#         feat = self.features(x)
        
#         if self.use_map and mask is not None:
#             target_size = feat.shape[2:]
#             resized_mask = F.adaptive_avg_pool3d(mask, target_size)
#             sum_feat = torch.sum(feat * resized_mask, dim=(2, 3, 4))
#             sum_mask = torch.sum(resized_mask, dim=(2, 3, 4))
#             avg_feat = sum_feat / (sum_mask + 1e-6)
#         else:
#             avg_feat = torch.mean(feat, dim=(2, 3, 4))
        
#         if self.use_metadata_fusion:
#             raw_meta = torch.cat([coords, lobe_vecs], dim=1)
#             meta_feat = self.meta_encoder(raw_meta)
#             avg_feat = torch.cat([avg_feat, meta_feat], dim=1)
            
#         return self.img_projector(avg_feat)

#     def encode_text(self, text_emb):
#         if self.use_text_proj:
#             return self.text_projector(text_emb)
#         return text_emb

# ==========================================
# 3. Helper Functions
# ==========================================

def load_text_embeddings(case_name):
    """ Load case_name.npy file (Shape: [Num_Findings, Embed_Dim]) """
    path = os.path.join(TEXT_EMB_DIR, f"{case_name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

def get_matched_gt_indices(metadata_instance):
    """ Extract GT Finding Index list for the instance """
    return [mf['finding_idx'] for mf in metadata_instance.get('matched_findings', [])]

# ==========================================
# 4. Inference Logic
# ==========================================

def run_inference():
    print(">>> Initialize Model...")
    # Initialize model (Pass parameters according to Reference Config)
    model = ContrastiveNet(
        text_input_dim=TEXT_INPUT_DIM, 
        output_dim=OUTPUT_DIM,
        layers_per_scale=LAYERS_PER_SCALE,
        base_channels=BASE_CHANNELS,
        use_instance_norm=True,
        use_residual=USE_RESIDUAL_BLOCK,
        use_metadata_fusion=USE_METADATA_FUSION,
        meta_dim=8, 
        use_text_proj=USE_TEXT_PROJ,
    ).to(DEVICE)

    # Load weights (weights_only=False)
    print(f"Loading checkpoint: {MODEL_CHECKPOINT}")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # List for saving evaluation metrics
    total_labels = []
    total_preds = []

    # [Modified] Get settings according to mode
    target_cases, data_root_dir, pred_root_dir = get_inference_config(INFERENCE_MODE)
    
    print(f">>> Start Inference on {len(target_cases)} cases...")
    print(f"    - Data Dir: {data_root_dir}")
    print(f"    - Pred Dir: {pred_root_dir}")
    
    # [Added] Load Dataset JSON and create findings mapping per case (Optimize Text Search)
    print("Loading Dataset Info for Text Parsing...")
    case_findings_map = {} # case_id -> { "0": text, "1": text ... }
    if os.path.exists(DATASET_JSON_PATH):
        with open(DATASET_JSON_PATH, 'r') as f:
            ds_raw = json.load(f)
            # Iterate over all splits like train, val, etc.
            for split_key in ds_raw:
                if isinstance(ds_raw[split_key], list):
                    for item in ds_raw[split_key]:
                        # "train_123.nii.gz" -> "train_123"
                        c_name = item['name'].replace('.nii.gz', '')
                        case_findings_map[c_name] = item.get('findings', {})

    for case_id in tqdm(target_cases):
        case_path = os.path.join(data_root_dir, case_id)

        # [Modified] Handle cases where case folder (patch data) is missing
        if not os.path.isdir(case_path):
            # Even if there's no folder, if in Stitching mode, generate and save an empty Volume(0)
            if ENABLE_STITCHING:
                ref_nii_path = os.path.join(SEGMENTATION_REF_DIR, f"{case_id}.nii.gz")
                if os.path.exists(ref_nii_path):
                    ref_img = nib.load(ref_nii_path)
                    F_dim, H, W, D = ref_img.shape
                    
                    # Create empty Volume with all zeros
                    pred_volume = np.zeros((F_dim, H, W, D), dtype=np.uint8)
                    
                    # Save result
                    save_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")
                    new_img = nib.Nifti1Image(pred_volume, ref_img.affine, ref_img.header)
                    nib.save(new_img, save_path)
                    # print(f"Saved empty volume for missing case: {case_id}")
            
            # Processing complete, move to next case
            continue

        if ENABLE_STITCHING:
            ref_nii_path = os.path.join(SEGMENTATION_REF_DIR, f"{case_id}.nii.gz")
            if not os.path.exists(ref_nii_path):
                print(f"Skipping {case_id}: Ref volume not found.")
                continue
            
            ref_img = nib.load(ref_nii_path)
            F_dim, H, W, D = ref_img.shape
            pred_volume = np.zeros((F_dim, H, W, D), dtype=np.uint8)
        else:
            # If not Stitching, skip Ref load (Speed up)
            pass

        # ---------------------------
        # B. Text Embeddings Load
        # ---------------------------
        # (N_findings, Dim)
        text_embs_raw = load_text_embeddings(case_id) 
        if text_embs_raw is None:
            print(f"Skipping {case_id}: Text embeddings not found.")
            continue
        
        # ---------------------------
        # C. Process Instances
        # ---------------------------
        metadata_path = os.path.join(case_path, "metadata.json")
        if not os.path.exists(metadata_path): continue
        
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
            
        # Instance-wise loop
        for inst_meta in metadata_list:
            instance_id = inst_meta['instance_id']
            npy_path = os.path.join(case_path, f"{case_id}_{instance_id}.npy") # e.g., train_..._1.npy. Pattern checking required
            
            # Check with glob as file names might differ
            if not os.path.exists(npy_path):
                # Try pattern: train_12991_a_1_1.npy
                pat = os.path.join(case_path, f"*_{instance_id}.npy")
                candidates = glob.glob(pat)
                # Exclude _mask.npy or _pred.npy
                candidates = [c for c in candidates if "_mask" not in c and "_pred" not in c and "_gt" not in c]
                if not candidates: continue
                npy_path = candidates[0]

            # 1. Image Load & Preprocess
            img_arr = np.load(npy_path) # Shape: (D, H, W) or similar
            mask_path = npy_path.replace(".npy", "_mask.npy")
            
            if os.path.exists(mask_path):
                mask_arr = np.load(mask_path)
            else:
                # Generate with 0 if mask is missing
                mask_arr = np.zeros_like(img_arr)

            # [Modified] Apply Center Crop (48, 80, 80)
            img_arr, mask_arr = center_crop_3d_numpy(img_arr, mask_arr, INPUT_SHAPE)

            # Model Inputs
            img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, D, H, W)
            
            # Mask (for input masking)
            mask_tensor = torch.from_numpy(mask_arr).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Metadata Inputs
            raw_coords = inst_meta.get('relative_coords_xyz', [0.0, 0.0, 0.0])
            coords_zyx = [raw_coords[2], raw_coords[1], raw_coords[0]]
            rel_coords = torch.tensor(coords_zyx).float().unsqueeze(0).to(DEVICE)
            lobe_vec = torch.tensor(inst_meta.get('lobe_vector', [0,0,0,0,0])).float().unsqueeze(0).to(DEVICE)
            
            # 2. Model Prediction
            with torch.no_grad():
                # Image Embedding
                img_emb = model(img_tensor, mask=mask_tensor, coords=rel_coords, lobe_vecs=lobe_vec) # (1, Out_Dim)
                img_emb = F.normalize(img_emb, dim=-1)
                
                # Text Embedding (Batch Processing)
                txt_tensor = torch.from_numpy(text_embs_raw).float().to(DEVICE) # (N_find, In_Dim)
                txt_emb = model.encode_text(txt_tensor) # (N_find, Out_Dim)
                txt_emb = F.normalize(txt_emb, dim=-1)
                
                # Similarity & Prob
                # (1, Dim) @ (Dim, N_find) -> (1, N_find)
                logits = (img_emb @ txt_emb.T) * model.logit_scale.exp() + model.bias
                # probs = torch.sigmoid(logits).cpu().numpy().flatten() # (N_find,)
                probs = logits.cpu().numpy().flatten() # (N_find,)
                

            # 3. Metric Calculation Setup
            gt_indices = get_matched_gt_indices(inst_meta) # List of int
            
            current_case_texts = case_findings_map.get(case_id, {})

            # candidate_indices = set([int(idx) for idx in inst_meta.get('candidate_findings', [])])

            for f_idx, prob in enumerate(probs):
                # if f_idx not in candidate_indices:
                #     continue

                label = 1 if f_idx in gt_indices else 0
                pred = 1 if prob >= THRESHOLD else 0
                # pred = label 

                total_labels.append(label)
                total_preds.append(pred)
                
                # 4. Stitching (If Predicted Positive)
                if pred == 1 and ENABLE_STITCHING:
                    # [NEW] Lobe Consistency Check
                    should_stitch = True

                    if should_stitch:
                        pred_nii_name = f"{case_id}_{instance_id}_pred.nii.gz"
                        pred_nii_path = os.path.join(pred_root_dir, case_id, pred_nii_name)
                        # pred_nii_path = os.path.join(PROCESSED_DIR, case_id, pred_nii_name)

                        if os.path.exists(pred_nii_path):

                            pred_patch_nii = nib.load(pred_nii_path)
                            pred_patch_arr = pred_patch_nii.get_fdata() # Shape: (Axis0, Axis1, Axis2)
                            
                            ph, pw, pd = pred_patch_arr.shape 

                            # ----------------------------------------------------
                            # Coordinate Logic: Direct Mapping (Index 0->0, 1->1, 2->2)
                            # ----------------------------------------------------
                            # crop_center_global_xyz: [x, y, z]
                            # Target Volume Shape: (F, H, W, D) -> Spatial Indices: 1, 2, 3
                            # User Request: x->h, y->w, z->d
                            
                            center_xyz = inst_meta['crop_center_global_xyz']
                            c_h = int(center_xyz[0]) # x -> Match to Axis 0 (H)
                            c_w = int(center_xyz[1]) # y -> Match to Axis 1 (W)
                            c_d = int(center_xyz[2]) # z -> Match to Axis 2 (D)

                            # --- 1. H range (Axis 0) ---
                            h_start = c_h - ph // 2
                            h_end = h_start + ph
                            
                            # --- 2. W range (Axis 1) ---
                            w_start = c_w - pw // 2
                            w_end = w_start + pw
                            
                            # --- 3. D range (Axis 2) ---
                            d_start = c_d - pd // 2
                            d_end = d_start + pd
                            
                            # ----------------------------------------------------
                            # Boundary Clipping & Slicing
                            # ----------------------------------------------------
                            # Target Volume Bounds (H, W, D are from ref_img.shape[1:])
                            ts_h, te_h = max(0, h_start), min(H, h_end)
                            ts_w, te_w = max(0, w_start), min(W, w_end)
                            ts_d, te_d = max(0, d_start), min(D, d_end)
                            
                            # Patch Bounds (Local) - Global 
                            ls_h = ts_h - h_start
                            le_h = ls_h + (te_h - ts_h)
                            
                            ls_w = ts_w - w_start
                            le_w = ls_w + (te_w - ts_w)
                            
                            ls_d = ts_d - d_start
                            le_d = ls_d + (te_d - ts_d)
                            
                            # 
                            if (te_h > ts_h) and (te_w > ts_w) and (te_d > ts_d):
                                # Patch Crop (Direct Slicing, No Transpose)
                                # patch_arr shape: (ph, pw, pd) -> Slicing corresponding axes
                                patch_crop = pred_patch_arr[ls_h:le_h, ls_w:le_w, ls_d:le_d]
                                
                                # Target Volume Update
                                # pred_volume shape: (F, H, W, D)
                                existing = pred_volume[f_idx, ts_h:te_h, ts_w:te_w, ts_d:te_d]
                                
                                # Max operation to merge overlaps
                                pred_volume[f_idx, ts_h:te_h, ts_w:te_w, ts_d:te_d] = np.maximum(existing, patch_crop)
                                
                        else:
                            print(f"Warning: Prediction file not found: {pred_nii_path}")

        # ---------------------------
        # D. Save Result
        # ---------------------------
        if ENABLE_STITCHING:
            save_path = os.path.join(OUTPUT_DIR, f"{case_id}.nii.gz")
            new_img = nib.Nifti1Image(pred_volume.astype(np.uint8), ref_img.affine, ref_img.header)
            nib.save(new_img, save_path)

    # ==========================================
    # 5. Global Metrics Output
    # ==========================================
    if total_labels:
        # Confusion Matrix
        cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # 1. Recall (Sensitivity) = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # 2. Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # 3. Balanced Accuracy = (Recall + Specificity) / 2
        balanced_acc = (recall + specificity) / 2.0

        # ( Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn)

        result_str = (
            "\n=== Final Evaluation Metrics ===\n"
            f"Recall (Sensitivity): {recall:.4f}\n"
            f"Specificity         : {specificity:.4f}\n"
            f"Balanced Accuracy   : {balanced_acc:.4f}\n"
            f"--------------------------------\n"
            f"Total Accuracy      : {acc:.4f}\n"
            f"Confusion Matrix    :\n{cm}\n"
            f"(TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp})\n"
        )
        
        # 1. Console Print
        print(result_str)

        # 2. Save to TXT
        txt_save_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
        with open(txt_save_path, "w") as f:
            f.write(result_str)
        print(f"[Saved] Metrics text: {txt_save_path}")

        # 3. Save to JPG (Confusion Matrix)
        jpg_save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.jpg")
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"Balanced Acc: {balanced_acc:.4f}")
        plt.savefig(jpg_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Saved] Confusion Matrix image: {jpg_save_path}")
    else:
        print("No valid instances processed.")

if __name__ == "__main__":
    run_inference()