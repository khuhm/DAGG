import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import random
from datetime import datetime
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

# ==========================================
# 1. Configuration & Case Discovery
# ==========================================
BASE_DIR = "/home/user/ReXGroundingCT"
NPY_OUTPUT_DIR = os.path.join(BASE_DIR, "crop_patches/npy_output_lobes_all_train_merged")
TEXT_EMB_DIR = os.path.join(BASE_DIR, "text_embeddings/Qwen3-Embedding-4B") # BioViL-T / Qwen3-Embedding-4B
DATASET_JSON_PATH = "/home/user/dataset/ReXGroundingCT/dataset.json"

ALL_CATEGORIES = [
    '1a', '1b', '1c', '1d', '1e', '1f', 
    '2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h'
]
CAT_TO_IDX = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}


NUM_TRAIN_CASES = 2692 
NUM_VAL_CASES = 300    
SHUFFLE_CASES = True 
SHUFFLE_SEED = 42
VALID_INTERVAL = 1  

USE_RESIDUAL_BLOCK = True
LAYERS_PER_SCALE = [1, 3, 4, 6, 6]
USE_CONV_POOLING = False
USE_MLP_PROJECTOR = False

# LAYERS_PER_SCALE = [1, 1, 1, 1, 1]
BASE_CHANNELS = 32
USE_MAP = False
USE_INSTANCE_NORM = True
MASK_INPUT = False
MASK_MARGIN = [1, 1, 1]
USE_TEXT_PROJECTION = True  #
USE_TEXT_PROJECTION_MLP = False
USE_ROW_LEVEL_BALANCE = False   
USE_CATEGORY_MASKING = False
MASK_INVALID_BG_REGIONS = False
IGNORE_BG_SAMPLES = False
EXPAND_POSITIVE_TO_SAME_CATEGORY = False
ONLY_POSITIVE_LOSS = False
INCLUDE_VALID_BG_REGIONS = False
INCLUDE_ALL_BG_REGIONS = False

USE_COSINE_SCHEDULER = False

USE_DROPOUT = False       
DROPOUT_RATE = 0.5        
FC_DROPOUT_RATE = 0.5     

USE_MASK_AREA_WEIGHTING = False

USE_COSINE_LOSS = True  
COSINE_MARGIN = 0.0     

USE_VALID_MASK_FOR_LOSS = True  
USE_BALANCED_LOSS = True  
USE_DATA_AUGMENTATION = False
DISABLE_MIRROR_AUGMENTATION = False

LEARNABLE_SCALE = False       
LEARNABLE_BIAS = False        

INIT_SCALE_FACTOR = 1.0      
INIT_BIAS_VALUE = 0.0        

USE_LOGIT_CLAMP = False      
MAX_LOGIT_SCALE = 100.0     

alpha = 1
LOSS_WEIGHT_POS = alpha / (alpha + 1)  # 3/4 = 0.75
LOSS_WEIGHT_NEG = 1.0 / (alpha + 1)    # 1/4 = 0.25
print(f"Positive Weight: {LOSS_WEIGHT_POS:.4f}") # 0.7500
print(f"Negative Weight: {LOSS_WEIGHT_NEG:.4f}") # 0.2500

USE_METADATA_FUSION = True 
USE_METADATA_MLP = False
COORD_DIM = 3      # (z, y, x)
LOBE_DIM = 5       # 
META_EMBED_DIM = 64

PRED_THRESHOLD = 0.5

all_cases = [
    d for d in os.listdir(NPY_OUTPUT_DIR) 
    if os.path.isdir(os.path.join(NPY_OUTPUT_DIR, d))
]

if SHUFFLE_CASES:
    random.Random(SHUFFLE_SEED).shuffle(all_cases)
else:
    all_cases.sort()

# Case Split
TARGET_CASES = all_cases
train_cases = all_cases[:NUM_TRAIN_CASES]
val_cases = all_cases[-NUM_VAL_CASES:]

print(f"Total Cases: {len(TARGET_CASES)}")
print(f"Train Cases: {len(train_cases)} | Val Cases: {len(val_cases)}")

WORK_DIR = os.path.join(BASE_DIR, "sigmoid_contrastive_loss")
EXPERIMENT_NAME = f"TrainVal_{len(train_cases)}_{len(val_cases)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(WORK_DIR, "experiments", EXPERIMENT_NAME)

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models") 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10000
VISUALIZE_INTERVAL = 1000
CROP_SIZE = (48, 80, 80)
TEXT_DIM = 2560
RAW_TEXT_DIM = 2560     # 
PROJECTED_DIM = 2560     # 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Dataset & Preprocessing 
# ==========================================
def random_crop_3d(img, mask, target_size):
    d, h, w = img.shape
    td, th, tw = target_size
    pd, ph, pw = max(0, (td-d)//2), max(0, (th-h)//2), max(0, (tw-w)//2)
    
    if pd>0 or ph>0 or pw>0:
        pad_width = ((pd,pd), (ph,ph), (pw,pw))
        img = np.pad(img, pad_width, mode='constant', constant_values=0)
        mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    
    d_new, h_new, w_new = img.shape
    d_max, h_max, w_max = max(0, d_new-td), max(0, h_new-th), max(0, w_new-tw)
    
    sd = random.randint(0, d_max)
    sh = random.randint(0, h_max)
    sw = random.randint(0, w_max)
    
    return img[sd:sd+td, sh:sh+th, sw:sw+tw], mask[sd:sd+td, sh:sh+th, sw:sw+tw]

class CenterCropTransform(BasicTransform):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get("image")
        seg = data_dict.get("seg")
        
        # data shape: (B, C, D, H, W)
        shape = data.shape[1:] 
        crop_size = self.crop_size


        center_d = shape[0] // 2
        center_h = shape[1] // 2
        center_w = shape[2] // 2
        

        d_start = max(0, center_d - crop_size[0] // 2)
        h_start = max(0, center_h - crop_size[1] // 2)
        w_start = max(0, center_w - crop_size[2] // 2)
        
        d_end = min(shape[0], d_start + crop_size[0])
        h_end = min(shape[1], h_start + crop_size[1])
        w_end = min(shape[2], w_start + crop_size[2])
        
        # Slicing
        data_dict["image"] = data[:, d_start:d_end, h_start:h_end, w_start:w_end]
        if seg is not None:
            data_dict["seg"] = seg[:, d_start:d_end, h_start:h_end, w_start:w_end]
            
        return data_dict
    
def get_train_transforms(patch_size):
    transforms = []
    
    dist_from_border = [i // 2 for i in patch_size]
    
    rotation_deg = 30.  
    scale_range = (0.7, 1.4) 

    transforms.append(
        SpatialTransform(
            patch_size=patch_size, 
            patch_center_dist_from_border=dist_from_border, 
            random_crop=True,              
            p_elastic_deform=0,             
            p_rotation=0.2,
            rotation=(-rotation_deg / 360 * 2. * np.pi, rotation_deg / 360 * 2. * np.pi), 
            p_scaling=0.2, 
            scaling=scale_range,           
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False
        )
    )

    if not DISABLE_MIRROR_AUGMENTATION:
        transforms.append(MirrorTransform(allowed_axes=(0, 1, 2)))

    # 2. Gaussian Noise
    transforms.append(RandomTransform(
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_channel=1,
            synchronize_channels=True
        ), apply_probability=0.1
    ))

    # 3. Gaussian Blur
    transforms.append(RandomTransform(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.),
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=0.5
        ), apply_probability=0.2
    ))

    # 4. Brightness (Multiplicative)
    transforms.append(RandomTransform(
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast((0.75, 1.25)),
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))

    # 5. Contrast
    transforms.append(RandomTransform(
        ContrastTransform(
            contrast_range=BGContrast((0.75, 1.25)),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))

    # 6. Low Resolution Simulation
    transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes = None,
            p_per_channel=0.5
        ), apply_probability=0.25
    ))

    # 7. Gamma Correction
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.1
    ))

    return ComposeTransforms(transforms)

def get_val_transforms(patch_size):
    return CenterCropTransform(patch_size)

class MultiCaseCTDataset(Dataset):
    def __init__(self, base_dir, case_names, crop_size, is_train=True):
        self.base_dir = base_dir
        self.crop_size = crop_size
        self.case_names = case_names
        self.is_train = is_train

        if self.is_train:
            if USE_DATA_AUGMENTATION:
                self.transforms = get_train_transforms(crop_size)
            else:
                dist_from_border = [i // 2 for i in crop_size]
                self.transforms = SpatialTransform(crop_size, patch_center_dist_from_border=dist_from_border, random_crop=True, 
                                                   p_rotation=0, p_scaling=0, p_elastic_deform=0)
        else:
            self.transforms = get_val_transforms(crop_size)

        self.samples = []
        self.global_text_embs = []
        self.global_text_cats = []
        
        self.case_category_map = {} # case_name -> {finding_idx(str) -> category(str)}

        if os.path.exists(DATASET_JSON_PATH):
            with open(DATASET_JSON_PATH, 'r') as f:
                raw_data = json.load(f)
                
            for split in ['train', 'val']:
                if split in raw_data:
                    for item in raw_data[split]:
                        case_key = item['name'].replace('.nii.gz', '')
                        if 'categories' in item:
                            self.case_category_map[case_key] = item['categories']
        else:
            print(f"Warning: {DATASET_JSON_PATH} not found.")

        # ----------------------------------------------------

        current_offset = 0

        for case in case_names:
            case_data_dir = os.path.join(NPY_OUTPUT_DIR, case)
            metadata_path = os.path.join(case_data_dir, "metadata.json")
            text_emb_path = os.path.join(TEXT_EMB_DIR, f"{case}.npy")
            
            if not os.path.exists(text_emb_path) or not os.path.exists(metadata_path):
                continue

            emb = np.load(text_emb_path)
            self.global_text_embs.append(emb)
            num_local_findings = emb.shape[0]
            
            curr_case_cats_map = self.case_category_map.get(case, {})
            curr_cats_list = []

            for f_idx in range(num_local_findings):
                cat_str = curr_case_cats_map.get(str(f_idx), 'Unknown') 
                curr_cats_list.append(cat_str)
            
            self.global_text_cats.extend(curr_cats_list)

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            curr_case_cats = self.case_category_map.get(case, {})
            
            for inst in metadata:
                iid = inst['instance_id']
                if os.path.exists(os.path.join(case_data_dir, f"{case}_{iid}.npy")):

                    matched_cats = []
                    matched_findings = inst.get('matched_findings', [])

                    candidate_findings_str = inst.get('candidate_findings', [])
                    candidate_indices = [int(c) for c in candidate_findings_str]

                    for mf in matched_findings:
                        f_idx = str(mf['finding_idx']) 
                        if f_idx in curr_case_cats:
                            matched_cats.append(curr_case_cats[f_idx])

                    relative_coords = inst.get('relative_coords_xyz', [0.0, 0.0, 0.0])
                    lobe_vector = inst.get('lobe_vector', [0.0] * 5) 

                    self.samples.append({
                        'case_name': case,
                        'instance_id': iid,
                        'data_dir': case_data_dir,
                        'matched_findings': inst.get('matched_findings', []),
                        'candidate_findings': candidate_indices,
                        'offset': current_offset,
                        'num_local': num_local_findings,
                        'categories': matched_cats,
                        'relative_coords': relative_coords,
                        'lobe_vector': lobe_vector
                    })
            current_offset += num_local_findings

        if self.global_text_embs:
            self.global_text_embs = np.concatenate(self.global_text_embs, axis=0)
        self.total_findings = self.global_text_embs.shape[0]
        
        assert len(self.global_text_cats) == self.total_findings, \
            f"Mismatch: Embs {self.total_findings} vs Cats {len(self.global_text_cats)}"
        
    def get_text_embeddings(self):
        return self.global_text_embs
    
    def get_text_categories(self):
        return self.global_text_cats
    
    def get_category_stats(self):
        stats = {cat: 0 for cat in ALL_CATEGORIES}
        total_instances = len(self.samples)
        
        for samp in self.samples:
            for c in samp['categories']:
                if c in stats:
                    stats[c] += 1
        return stats, total_instances
    
    def get_category_relationship_matrix(self):
        all_cats = self.global_text_cats # list of strings (len = N)
        n_findings = len(all_cats)
        
        unique_cats = list(set(all_cats))
        cat_to_int = {c: i for i, c in enumerate(unique_cats)}
        
        cat_indices = np.array([cat_to_int[c] for c in all_cats]) # (N,)
        cat_tensor = torch.from_numpy(cat_indices).long() # (N,)

        relationship_matrix = (cat_tensor.unsqueeze(1) == cat_tensor.unsqueeze(0)).float()
        
        return relationship_matrix
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        info = self.samples[idx]
        img_path = os.path.join(info['data_dir'], f"{info['case_name']}_{info['instance_id']}.npy")
        mask_path = os.path.join(info['data_dir'], f"{info['case_name']}_{info['instance_id']}_pred.npy")
        
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        data_dict = {
            'image': img, 
        }
        
        data_dict = self.transforms(**data_dict)
        
        # (1, 1, D, H, W) -> (D, H, W)
        img = data_dict['image']

        patch_size = self.crop_size
        _, d, h, w = mask.shape
        
        cd, ch, cw = d // 2, h // 2, w // 2
        
        sd = max(0, cd - patch_size[0] // 2)
        sh = max(0, ch - patch_size[1] // 2)
        sw = max(0, cw - patch_size[2] // 2)
        
        ed = min(d, sd + patch_size[0])
        eh = min(h, sh + patch_size[1])
        ew = min(w, sw + patch_size[2])
        
        # Mask Slicing
        mask = mask[:, sd:ed, sh:eh, sw:ew]

        # -------------------------------------------------------

        # img = np.ascontiguousarray(img)
        # mask = np.ascontiguousarray(mask)

        # img = torch.from_numpy(img).unsqueeze(0)   # (1, D, H, W)
        # mask = torch.from_numpy(mask).unsqueeze(0) # (1, D, H, W)
        # img, mask = random_crop_3d(img, mask, self.crop_size)
        
        label = torch.zeros(self.total_findings)
        valid_mask = torch.zeros(self.total_findings)
        
        offset = info['offset']
        num_local = info['num_local']
        
        valid_mask[offset : offset + num_local] = 1.0
        # for cand_idx in info['candidate_findings']:
        #     if cand_idx < num_local: # Index Range Check
        #         valid_mask[offset + cand_idx] = 1.0

        for finding in info['matched_findings']:
            local_idx = finding['finding_idx']
            if local_idx < num_local:
                label[offset + local_idx] = 1.0
        
        # 2. Category Labels (Multi-hot encoding, size=14)
        cat_label = torch.zeros(len(ALL_CATEGORIES), dtype=torch.float32)
        for cat_str in info['categories']:
            if cat_str in CAT_TO_IDX:
                cat_label[CAT_TO_IDX[cat_str]] = 1.0

        raw_coords = info['relative_coords'] # [x, y, z]
        coords_zyx = [raw_coords[2], raw_coords[1], raw_coords[0]] 
        relative_coords = torch.tensor(coords_zyx, dtype=torch.float32)
        lobe_vector = torch.tensor(info['lobe_vector'], dtype=torch.float32)

        return img, mask, label, valid_mask, cat_label, info['instance_id'], relative_coords, lobe_vector

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
                 text_input_dim=2560,    
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
            self.mask_margin = tuple(mask_margin)

        self.use_metadata_fusion = use_metadata_fusion
        self.use_metadata_mlp = use_metadata_mlp

        self.num_scales = len(layers_per_scale)
        
        layers = []
        in_channels = 1
        current_channels = base_channels
        
        for scale_idx, num_layers in enumerate(layers_per_scale):
            if scale_idx > 0:
                layers.append(nn.Conv3d(in_channels, current_channels, kernel_size=3, stride=2, padding=1))
                if use_instance_norm:
                    layers.append(nn.InstanceNorm3d(current_channels, affine=True))
                layers.append(nn.ReLU())
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
            
            if scale_idx < self.num_scales - 1:
                current_channels *= 2 

        self.features = nn.Sequential(*layers)

        if USE_CONV_POOLING:
            self.pooling_adapter = nn.AdaptiveAvgPool3d((3, 5, 5))

            self.final_pool_conv = nn.Sequential(
                nn.Conv3d(current_channels, current_channels, 
                          kernel_size=(3, 5, 5), bias=False), 
                nn.LayerNorm([current_channels, 1, 1, 1]),   
                nn.ReLU(inplace=True) 
            )

            for m in self.final_pool_conv.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight)

        if self.use_metadata_fusion:
            if self.use_metadata_mlp:
                self.meta_encoder = nn.Sequential(
                    nn.Linear(meta_dim, meta_embed_dim),
                    nn.LayerNorm(meta_embed_dim),
                    nn.ReLU(),
                    nn.Linear(meta_embed_dim, meta_embed_dim)
                )
                added_dim = meta_embed_dim
            else:
                self.meta_encoder = nn.Identity()
                added_dim = meta_dim
            
            proj_input_dim = current_channels + added_dim
        else:
            proj_input_dim = current_channels

        if self.use_mlp_projector:
            hidden_dim = output_dim
            self.img_projector = nn.Sequential(
                nn.Linear(proj_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Simple Linear Style
            self.img_projector = nn.Linear(proj_input_dim, output_dim)
    
        # --- [MODIFIED] Text Projector Initialization ---
        if self.use_text_proj:
            if use_text_proj_mlp:
                self.text_projector = nn.Sequential(
                    nn.Linear(text_input_dim, text_input_dim), 
                    nn.ReLU(), 
                    nn.Linear(text_input_dim, output_dim)
                )
            else:
                self.text_projector = nn.Linear(text_input_dim, output_dim)

        if self.use_text_proj:
            self.text_projector = nn.Linear(text_input_dim, output_dim)

        init_scale_val = np.log(INIT_SCALE_FACTOR)
        if LEARNABLE_SCALE:
            self.logit_scale = nn.Parameter(torch.ones([]) * init_scale_val)
        else:
            self.register_buffer("logit_scale", torch.ones([]) * init_scale_val)

        # 2. Bias
        if LEARNABLE_BIAS:
            self.bias = nn.Parameter(torch.ones([]) * INIT_BIAS_VALUE)
        else:
            self.register_buffer("bias", torch.ones([]) * INIT_BIAS_VALUE)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1))
        # self.bias = nn.Parameter(torch.zeros([]))
        # self.logit_scale = torch.ones([]) * np.log(1)
        # self.bias = torch.zeros([])

    def forward(self, x, mask=None, coords=None, lobe_vecs=None):
        if self.mask_input and mask is not None:
            if any(m > 1 for m in self.mask_margin):
                k_d, k_h, k_w = self.mask_margin
                
                pad_d, pad_h, pad_w = k_d // 2, k_h // 2, k_w // 2
                
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
            # feat_adapted = self.pooling_adapter(feat) 
            
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
            
            raw_meta = torch.cat([coords, lobe_vecs], dim=1)
            
            if self.use_metadata_mlp:
                # [Option A] MLP : (B, 64)
                meta_feat = self.meta_encoder(raw_meta)
            else:
                # [Option B] : (B, 9)
                meta_feat = raw_meta
            
            # (B, Image_Ch + Meta_Feat)
            avg_feat = torch.cat([avg_feat, meta_feat], dim=1)
            
        return self.img_projector(avg_feat) # Image embedding 

    def encode_text(self, text_emb):
        # --- [] Text Forward ---
        if self.use_text_proj:
            return self.text_projector(text_emb)
        return text_emb # 

# ==========================================
# 4. Visualization Logic 
# ==========================================
def save_image_tsne(img_embs, labels_list, epoch, save_dir, prefix=""):
    plt.figure(figsize=(10, 8))
    n_samples = img_embs.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(img_embs)
    
    primary_labels = []
    for l_vec in labels_list:
        indices = np.where(l_vec == 1)[0]
        primary_labels.append(indices[0] if len(indices) > 0 else -1)
        
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=primary_labels, cmap='nipy_spectral', 
                         marker='o', s=60, edgecolors='k', alpha=0.8)
    plt.colorbar(scatter, label="Global Finding ID")
    plt.title(f"{prefix} Image Embeddings t-SNE (Epoch {epoch})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{prefix}_tsne_IMG_epoch_{epoch:04d}.png"))
    plt.close()

def save_text_tsne(text_embs, epoch, save_dir):
    plt.figure(figsize=(10, 8))
    n_samples = text_embs.shape[0]
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(text_embs)
    indices = np.arange(n_samples)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=indices, cmap='nipy_spectral', marker='x', s=80, alpha=0.8)
    plt.title("Text Embeddings t-SNE (Reference)")
    plt.savefig(os.path.join(save_dir, f"tsne_TEXT_epoch_{epoch:04d}.png"))
    plt.close()

# ==========================================
# 5. Training & Validation Loop
# ==========================================
def main():
    train_ds = MultiCaseCTDataset(BASE_DIR, train_cases, CROP_SIZE)
    val_ds = MultiCaseCTDataset(BASE_DIR, val_cases, CROP_SIZE, is_train=False)

    print(f"Total Unique Findings (Labels per case): {train_ds.total_findings}")

    # [NEW] 
    # shape: (Num_Findings, Num_Findings), 1.0 if same category
    cat_relation_mat = train_ds.get_category_relationship_matrix().to(device)

    # 
    print("="*50)
    print(" >>> Train Dataset Category Statistics <<<")
    train_stats, train_total = train_ds.get_category_stats()
    print(f" Total Samples: {train_total}")
    for cat, count in train_stats.items():
        if count > 0:
            print(f"   Category {cat}: {count}")
            
    print("-" * 30)
    print(" >>> Val Dataset Category Statistics <<<")
    val_stats, val_total = val_ds.get_category_stats()
    print(f" Total Samples: {val_total}")
    for cat, count in val_stats.items():
        if count > 0:
            print(f"   Category {cat}: {count}")
    print("="*50)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 1. 
    raw_train_text = torch.tensor(train_ds.get_text_embeddings(), dtype=torch.float32).to(device)
    raw_val_text = torch.tensor(val_ds.get_text_embeddings(), dtype=torch.float32).to(device)
    
    # 
    if not USE_TEXT_PROJECTION:
        raw_train_text = F.normalize(raw_train_text, p=2, dim=1)
        raw_val_text = F.normalize(raw_val_text, p=2, dim=1)
    
    model = ContrastiveNet(text_input_dim=RAW_TEXT_DIM,output_dim=PROJECTED_DIM, layers_per_scale=LAYERS_PER_SCALE, base_channels=BASE_CHANNELS, use_instance_norm=USE_INSTANCE_NORM, use_map=USE_MAP, mask_input=MASK_INPUT,
        mask_margin=MASK_MARGIN, use_text_proj=USE_TEXT_PROJECTION, use_text_proj_mlp=USE_TEXT_PROJECTION_MLP, 
        use_metadata_fusion=USE_METADATA_FUSION, use_metadata_mlp = USE_METADATA_MLP, meta_dim=(COORD_DIM + LOBE_DIM) if USE_METADATA_FUSION else 0, meta_embed_dim = META_EMBED_DIM,
        use_residual=USE_RESIDUAL_BLOCK).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = None
    if USE_COSINE_SCHEDULER:
        # T_max: 
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        print(f" >>> Cosine Annealing Scheduler Activated (T_max={NUM_EPOCHS})")
    else:
        print(" >>> Using Fixed Learning Rate")

    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        t_metrics = {
            'loss': [], 
            'pos_acc': [], 'neg_acc': [], 
            'pos_sim': [], 'neg_sim': [], 
            'pos_prob': [], 'neg_prob': [],
            'neg_sim_weighted': [], 'neg_prob_weighted': [] # [NEW]
        }
        for imgs, masks, labels, v_masks, cat_labels, _, relative_coords, lobe_vectors in train_loader:
            imgs, masks, labels, v_masks, cat_labels, relative_coords, lobe_vectors = imgs.to(device), masks.to(device), labels.to(device), v_masks.to(device), cat_labels.to(device), relative_coords.to(device), lobe_vectors.to(device)
            
            if USE_MASK_AREA_WEIGHTING:
                # 
                sample_weights = masks.reshape(masks.size(0), -1).sum(dim=1)
                # 
            else:
                sample_weights = torch.ones(imgs.size(0)).to(device)
            
            optimizer.zero_grad()
            
            if USE_METADATA_FUSION:
                img_embs = model(imgs, masks, relative_coords, lobe_vectors)
            else:
                img_embs = model(imgs, masks)
            img_embs = F.normalize(img_embs, p=2, dim=1)

            # 2. 
            if USE_TEXT_PROJECTION:
                # (All Findings, Raw Dim) -> (All Findings, Proj Dim)
                curr_text_embs = model.encode_text(raw_train_text) 
                curr_text_embs = F.normalize(curr_text_embs, p=2, dim=1)
            else:
                curr_text_embs = raw_train_text # 

            sim = torch.matmul(img_embs, curr_text_embs.T)
            scale = model.logit_scale.exp()
            if USE_LOGIT_CLAMP:
                # Scale
                scale = torch.clamp(scale, max=MAX_LOGIT_SCALE)
            logits = (sim * scale) + model.bias
            
            targets = labels.clone()

            if EXPAND_POSITIVE_TO_SAME_CATEGORY:
                # 1. (B, N) x (N, N) -> (B, N)
                pos_related_map = torch.matmul(labels, cat_relation_mat)
                
                targets = ((targets == 1) | (pos_related_map > 0)).float()

            # --- [MODIFIED] Loss Calculation Logic ---
            if USE_COSINE_LOSS:
                pos_loss = (1.0 - sim) * targets
                neg_sim_margin = torch.clamp(sim - COSINE_MARGIN, min=0)
                neg_loss = neg_sim_margin * (1.0 - targets)
                raw_loss = pos_loss + neg_loss
            else:
                raw_loss = criterion(logits, targets)

            final_loss_mask = torch.ones_like(targets).to(device)
            is_bg_row = (targets.sum(dim=1) == 0)

            if USE_CATEGORY_MASKING:
                pos_related_map = torch.matmul(targets, cat_relation_mat)
                ignore_condition = (targets == 0) & (pos_related_map > 0)
                final_loss_mask = final_loss_mask * (~ignore_condition).float()

            if IGNORE_BG_SAMPLES:
                # is_bg_row (B,) -> (B, 1) -> Broadcasting
                final_loss_mask[is_bg_row, :] = 0.0
        
                
                bg_rows_expanded = is_bg_row.unsqueeze(1).expand_as(targets)
                mask_logic = (~bg_rows_expanded) | (v_masks == 1)
                
                final_loss_mask = final_loss_mask * mask_logic.float()

            if USE_VALID_MASK_FOR_LOSS:
                final_loss_mask = final_loss_mask * v_masks

            if ONLY_POSITIVE_LOSS:
                final_loss_mask = final_loss_mask * targets

            if INCLUDE_VALID_BG_REGIONS:
                bg_rows_expanded = is_bg_row.unsqueeze(1).expand_as(final_loss_mask)
                valid_bg_mask = bg_rows_expanded & (v_masks == 1)
                final_loss_mask = torch.max(final_loss_mask, valid_bg_mask.float())

            if INCLUDE_ALL_BG_REGIONS:
                bg_rows_expanded = is_bg_row.unsqueeze(1).expand_as(final_loss_mask)
                final_loss_mask = torch.max(final_loss_mask, bg_rows_expanded.float())

            weight_map = sample_weights.unsqueeze(1).expand_as(raw_loss)

            # === [NEW LOSS LOGIC] Row-level Balancing ===
            if USE_ROW_LEVEL_BALANCE:
                is_pos_row = ~is_bg_row # labels.sum > 0
                is_neg_row = is_bg_row  # labels.sum == 0

                pos_mask_2d = is_pos_row.unsqueeze(1).expand_as(raw_loss)
                neg_mask_2d = is_neg_row.unsqueeze(1).expand_as(raw_loss)

                pos_mask_final = pos_mask_2d & (final_loss_mask == 1)
                neg_mask_final = neg_mask_2d & (final_loss_mask == 1)

                if pos_mask_final.sum() > 0:
                    pos_loss = (raw_loss * pos_mask_final).sum() / (pos_mask_final.sum() + 1e-6)
                else:
                    pos_loss = 0.0
                
                if neg_mask_final.sum() > 0:
                    neg_loss = (raw_loss * neg_mask_final).sum() / (neg_mask_final.sum() + 1e-6)
                else:
                    neg_loss = 0.0

                if pos_mask_final.sum() > 0 and neg_mask_final.sum() > 0:
                    loss = (pos_loss + neg_loss) / 2.0
                elif pos_mask_final.sum() > 0:
                    loss = pos_loss
                else:
                    loss = neg_loss

            elif USE_BALANCED_LOSS:
                # Element-wise Balancing
                pos_mask = (targets == 1) & (final_loss_mask == 1)
                neg_mask = (targets == 0) & (final_loss_mask == 1)
                
                if pos_mask.sum() > 0:
                    pos_loss = (raw_loss * pos_mask).sum() / (pos_mask.sum() + 1e-6)
                else:
                    pos_loss = 0.0

                if neg_mask.sum() > 0:
                    if USE_MASK_AREA_WEIGHTING:
                        # Loss = Sum(Loss * Weight) / Sum(Weight)
                        weighted_neg_loss = raw_loss * weight_map * neg_mask
                        weighted_norm = weight_map * neg_mask
                        neg_loss = weighted_neg_loss.sum() / (weighted_norm.sum() + 1e-6)
                    else:
                        neg_loss = (raw_loss * neg_mask).sum() / (neg_mask.sum() + 1e-6)
                else:
                    neg_loss = 0.0

                loss = (pos_loss * LOSS_WEIGHT_POS) + (neg_loss * LOSS_WEIGHT_NEG)
            else:
                # Simple Average
                loss = (raw_loss * final_loss_mask).sum() / (final_loss_mask.sum() + 1e-6)

            # loss = (criterion(logits, labels) * v_masks).sum() / (v_masks.sum() + 1e-6)
            # loss = criterion(logits, labels).mean()
            loss.backward(); optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits) 
                preds = (probs > PRED_THRESHOLD).float()
                pos_m, neg_m = (labels == 1) & (v_masks == 1), (labels == 0) & (v_masks == 1)
                t_metrics['loss'].append(loss.item())
                if pos_m.sum() > 0:
                    t_metrics['pos_acc'].append((preds[pos_m] == 1).float().mean().item())
                    t_metrics['pos_sim'].append(sim[pos_m].mean().item())
                    t_metrics['pos_prob'].append(probs[pos_m].mean().item())
                    if USE_MASK_AREA_WEIGHTING:
                        # neg_m: (B, N) Boolean
                        # weight_map: (B, N) Float (Expanded Area)
                        
                        w_selected = weight_map[neg_m]      # (K,)
                        sim_selected = sim[neg_m]           # (K,)
                        prob_selected = probs[neg_m]        # (K,)
                        
                        w_sum = w_selected.sum() + 1e-6
                        
                        # Weighted Mean: Sum(Val * W) / Sum(W)
                        w_sim_val = (sim_selected * w_selected).sum() / w_sum
                        w_prob_val = (prob_selected * w_selected).sum() / w_sum

                        t_metrics['neg_sim_weighted'].append(w_sim_val.item())
                        t_metrics['neg_prob_weighted'].append(w_prob_val.item())
                    
                if neg_m.sum() > 0:
                    t_metrics['neg_acc'].append((preds[neg_m] == 0).float().mean().item())
                    t_metrics['neg_sim'].append(sim[neg_m].mean().item())
                    t_metrics['neg_prob'].append(probs[neg_m].mean().item())

        if USE_COSINE_SCHEDULER and scheduler is not None:
            scheduler.step()

        # --- LOGGING ---
        # Train Scalars
        avg_t_pos, avg_t_neg = np.mean(t_metrics['pos_acc']), np.mean(t_metrics['neg_acc'])
        writer.add_scalar('Loss/train', np.mean(t_metrics['loss']), epoch)
        writer.add_scalar('Accuracy/Positive_Recall', avg_t_pos, epoch)
        writer.add_scalar('Accuracy/Negative_Specificity', avg_t_neg, epoch)
        writer.add_scalar('Accuracy/Balanced', (avg_t_pos + avg_t_neg)/2, epoch)
        writer.add_scalar('Similarity/Positive', np.mean(t_metrics['pos_sim']), epoch)
        writer.add_scalar('Similarity/Negative', np.mean(t_metrics['neg_sim']), epoch)
        writer.add_scalar('Similarity/Gap', np.mean(t_metrics['pos_sim']) - np.mean(t_metrics['neg_sim']), epoch)
        writer.add_scalar('Probability/Positive', np.mean(t_metrics['pos_prob']), epoch)
        writer.add_scalar('Probability/Negative', np.mean(t_metrics['neg_prob']), epoch)
        if t_metrics['neg_sim_weighted']:
            writer.add_scalar('Similarity/Negative_Weighted', np.mean(t_metrics['neg_sim_weighted']), epoch)
            writer.add_scalar('Similarity/Gap_Weighted', np.mean(t_metrics['pos_sim']) - np.mean(t_metrics['neg_sim_weighted']), epoch)
        
        if t_metrics['neg_prob_weighted']:
            writer.add_scalar('Probability/Negative_Weighted', np.mean(t_metrics['neg_prob_weighted']), epoch)

        # --- VALIDATION ---
        if epoch == 1 or epoch % VALID_INTERVAL == 0:
            model.eval()
            # v_metrics = {'loss': [], 'pos_acc': [], 'neg_acc': [], 'pos_sim': [], 'neg_sim': []}
            val_stats = {
                'loss_sum': 0.0, 'loss_count': 0,       
                'pos_correct': 0, 'pos_total': 0,       
                'neg_correct': 0, 'neg_total': 0,       
                'pos_sim_sum': 0.0, 'neg_sim_sum': 0.0, 
                'pos_prob_sum': 0.0, 'neg_prob_sum': 0.0,
                
                # [NEW] Weighted Metrics Aggregation
                'neg_sim_weighted_sum': 0.0, 
                'neg_prob_weighted_sum': 0.0,
                'neg_weight_total': 0.0 
            }

            val_vis_embs, val_vis_labels = [], []
            with torch.no_grad():
                if USE_TEXT_PROJECTION:
                    curr_val_text_embs = F.normalize(model.encode_text(raw_val_text), p=2, dim=1)
                else:
                    curr_val_text_embs = raw_val_text

                for imgs, masks, labels, v_masks, cat_labels, _, relative_coords, lobe_vectors in val_loader:
                    imgs, masks, labels, v_masks, cat_labels, relative_coords, lobe_vectors = imgs.to(device), masks.to(device), labels.to(device), v_masks.to(device), cat_labels.to(device), relative_coords.to(device), lobe_vectors.to(device)
                    
                    # --- [NEW] Calculate Mask Weights (Per Sample) ---
                    if USE_MASK_AREA_WEIGHTING:
                        sample_weights = masks.reshape(masks.size(0), -1).sum(dim=1)
                    else:
                        sample_weights = torch.ones(imgs.size(0)).to(device)

                    if USE_METADATA_FUSION:
                        img_embs = model(imgs, masks, relative_coords, lobe_vectors)
                    else:
                        img_embs = model(imgs, masks)
                    embs = F.normalize(img_embs, p=2, dim=1)
                    sim = torch.matmul(embs, curr_val_text_embs.T)
                    scale = model.logit_scale.exp()
                    if USE_LOGIT_CLAMP:
                        scale = torch.clamp(scale, max=MAX_LOGIT_SCALE)
                    logits = (sim * scale) + model.bias

                    targets = labels.clone()

                    if EXPAND_POSITIVE_TO_SAME_CATEGORY:
                        # 1. (B, N) x (N, N) -> (B, N)
                        pos_related_map = torch.matmul(labels, cat_relation_mat)
                        targets = ((targets == 1) | (pos_related_map > 0)).float()

                    if USE_COSINE_LOSS:
                        pos_loss = (1.0 - sim) * targets
                        neg_sim_margin = torch.clamp(sim - COSINE_MARGIN, min=0)
                        neg_loss = neg_sim_margin * (1.0 - targets)
                        raw_loss = pos_loss + neg_loss
                    else:
                        raw_loss = criterion(logits, targets)

                    if USE_MASK_AREA_WEIGHTING:
                        weight_map = sample_weights.unsqueeze(1).expand_as(raw_loss)
                    final_loss_mask = torch.ones_like(targets).to(device)
                    is_bg_row = (targets.sum(dim=1) == 0)

                    if USE_CATEGORY_MASKING:
                        pos_related_map = torch.matmul(targets, cat_relation_mat)
                        ignore_condition = (targets == 0) & (pos_related_map > 0)
                        final_loss_mask = final_loss_mask * (~ignore_condition).float()

                    if IGNORE_BG_SAMPLES:
                        # is_bg_row (B,) -> (B, 1) -> Broadcasting
                        final_loss_mask[is_bg_row, :] = 0.0
                    
                    # 4. [NEW] Mask Invalid Regions ONLY in BG Samples
                    if MASK_INVALID_BG_REGIONS:
                        
                        bg_rows_expanded = is_bg_row.unsqueeze(1).expand_as(targets)
                        mask_logic = (~bg_rows_expanded) | (v_masks == 1)
                        
                        final_loss_mask = final_loss_mask * mask_logic.float()
                    if USE_VALID_MASK_FOR_LOSS:
                        final_loss_mask = final_loss_mask * v_masks

                    if ONLY_POSITIVE_LOSS:
                        final_loss_mask = final_loss_mask * targets

                    if INCLUDE_VALID_BG_REGIONS:
                        bg_rows_expanded = is_bg_row.unsqueeze(1).expand_as(final_loss_mask)
                        valid_bg_mask = bg_rows_expanded & (v_masks == 1)
                        final_loss_mask = torch.max(final_loss_mask, valid_bg_mask.float())

                    if INCLUDE_ALL_BG_REGIONS:
                        bg_rows_expanded = is_bg_row.unsqueeze(1).expand_as(final_loss_mask)
                        final_loss_mask = torch.max(final_loss_mask, bg_rows_expanded.float())

                    # === [NEW LOSS LOGIC] Row-level Balancing ===
                    if USE_ROW_LEVEL_BALANCE:
                        is_pos_row = ~is_bg_row # labels.sum > 0
                        is_neg_row = is_bg_row  # labels.sum == 0

               
                        pos_mask_2d = is_pos_row.unsqueeze(1).expand_as(raw_loss)
                        neg_mask_2d = is_neg_row.unsqueeze(1).expand_as(raw_loss)

                   
                        pos_mask_final = pos_mask_2d & (final_loss_mask == 1)
                        neg_mask_final = neg_mask_2d & (final_loss_mask == 1)

                  
                        if pos_mask_final.sum() > 0:
                            pos_loss = (raw_loss * pos_mask_final).sum() / (pos_mask_final.sum() + 1e-6)
                        else:
                            pos_loss = 0.0
                        
                        if neg_mask_final.sum() > 0:
                            neg_loss = (raw_loss * neg_mask_final).sum() / (neg_mask_final.sum() + 1e-6)
                        else:
                            neg_loss = 0.0

                        if pos_mask_final.sum() > 0 and neg_mask_final.sum() > 0:
                            loss = (pos_loss + neg_loss) / 2.0
                        elif pos_mask_final.sum() > 0:
                            loss = pos_loss
                        else:
                            loss = neg_loss

                    elif USE_BALANCED_LOSS:
                        # Element-wise Balancing
                        pos_mask = (targets == 1) & (final_loss_mask == 1)
                        neg_mask = (targets == 0) & (final_loss_mask == 1)
                        
                        # Positive Loss
                        if pos_mask.sum() > 0:
                            pos_loss = (raw_loss * pos_mask).sum() / (pos_mask.sum() + 1e-6)
                        else:
                            pos_loss = 0.0

                        # Negative Loss (Weighted)
                        if neg_mask.sum() > 0:
                            if USE_MASK_AREA_WEIGHTING:
                               
                                weighted_neg_loss = raw_loss * weight_map * neg_mask
                                weighted_norm = weight_map * neg_mask
                                neg_loss = weighted_neg_loss.sum() / (weighted_norm.sum() + 1e-6)
                            else:
                                
                                neg_loss = (raw_loss * neg_mask).sum() / (neg_mask.sum() + 1e-6)
                        else:
                            neg_loss = 0.0

                        loss = (pos_loss * LOSS_WEIGHT_POS) + (neg_loss * LOSS_WEIGHT_NEG)
                    else:
                        # Simple Average
                        loss = (raw_loss * final_loss_mask).sum() / (final_loss_mask.sum() + 1e-6)

                    #
                    val_stats['loss_sum'] += loss.item()
                    val_stats['loss_count'] += 1
                    
                    #
                    probs = torch.sigmoid(logits) 
                    preds = (probs > PRED_THRESHOLD).float()
                    pos_m, neg_m = (labels == 1) & (v_masks == 1), (labels == 0) & (v_masks == 1)

                    # Positive Metrics Accumulation
                    n_pos = pos_m.sum().item()
                    if n_pos > 0:
                        val_stats['pos_correct'] += (preds[pos_m] == 1).float().sum().item()
                        val_stats['pos_sim_sum'] += sim[pos_m].sum().item()
                        val_stats['pos_prob_sum'] += probs[pos_m].sum().item()
                        val_stats['pos_total'] += n_pos

                    # Negative Metrics Accumulation
                    n_neg = neg_m.sum().item()
                    if n_neg > 0:
                        val_stats['neg_correct'] += (preds[neg_m] == 0).float().sum().item()
                        val_stats['neg_sim_sum'] += sim[neg_m].sum().item()
                        val_stats['neg_prob_sum'] += probs[neg_m].sum().item()
                        val_stats['neg_total'] += n_neg

                        # [NEW] Weighted Negative Metrics Accumulation
                        if USE_MASK_AREA_WEIGHTING:
                            # (Batch,) -> (Batch, Findings)
                            weight_map = sample_weights.unsqueeze(1).expand_as(sim)
                            
                            w_selected = weight_map[neg_m]       # (K,)
                            sim_selected = sim[neg_m]            # (K,)
                            prob_selected = probs[neg_m]         # (K,)
                            
                            # 
                            val_stats['neg_sim_weighted_sum'] += (sim_selected * w_selected).sum().item()
                            val_stats['neg_prob_weighted_sum'] += (prob_selected * w_selected).sum().item()
                            val_stats['neg_weight_total'] += w_selected.sum().item()

            # ZeroDivisionError 
            avg_v_loss = val_stats['loss_sum'] / max(val_stats['loss_count'], 1)
            
            avg_v_pos_acc = val_stats['pos_correct'] / max(val_stats['pos_total'], 1)
            avg_v_neg_acc = val_stats['neg_correct'] / max(val_stats['neg_total'], 1)
            
            avg_v_pos_sim = val_stats['pos_sim_sum'] / max(val_stats['pos_total'], 1)
            avg_v_neg_sim = val_stats['neg_sim_sum'] / max(val_stats['neg_total'], 1)

            avg_v_pos_prob = val_stats['pos_prob_sum'] / max(val_stats['pos_total'], 1)
            avg_v_neg_prob = val_stats['neg_prob_sum'] / max(val_stats['neg_total'], 1)

            # [NEW] Weighted Averages
            if USE_MASK_AREA_WEIGHTING and val_stats['neg_weight_total'] > 0:
                avg_v_neg_sim_weighted = val_stats['neg_sim_weighted_sum'] / val_stats['neg_weight_total']
                avg_v_neg_prob_weighted = val_stats['neg_prob_weighted_sum'] / val_stats['neg_weight_total']

            # Writer Logging 
            writer.add_scalar('Loss/Val', avg_v_loss, epoch)
            writer.add_scalar('Accuracy/Val_Positive_Recall', avg_v_pos_acc, epoch)
            writer.add_scalar('Accuracy/Val_Negative_Specificity', avg_v_neg_acc, epoch)
            writer.add_scalar('Accuracy/Val_Balanced', (avg_v_pos_acc + avg_v_neg_acc) / 2.0, epoch)
            writer.add_scalar('Similarity/Val_Positive', avg_v_pos_sim, epoch)
            writer.add_scalar('Similarity/Val_Negative', avg_v_neg_sim, epoch)
            writer.add_scalar('Similarity/Val_Gap', avg_v_pos_sim - avg_v_neg_sim, epoch)

            writer.add_scalar('Probability/Val_Positive', avg_v_pos_prob, epoch)
            writer.add_scalar('Probability/Val_Negative', avg_v_neg_prob, epoch)

            # [NEW] Weighted Metrics Logging
            if USE_MASK_AREA_WEIGHTING:
                writer.add_scalar('Similarity/Val_Negative_Weighted', avg_v_neg_sim_weighted, epoch)
                writer.add_scalar('Similarity/Val_Gap_Weighted', avg_v_pos_sim - avg_v_neg_sim_weighted, epoch)
                writer.add_scalar('Probability/Val_Negative_Weighted', avg_v_neg_prob_weighted, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Val] Epoch {epoch}: Loss {avg_v_loss:.4f} | "
                f"PosAcc {avg_v_pos_acc:.4f} | NegAcc {avg_v_neg_acc:.4f} | "
                f"SimGap {(avg_v_pos_sim - avg_v_neg_sim):.4f} | " 
                f"LR {current_lr:.8f}") 
        # Parameter Logging (Scale & Bias )
        writer.add_scalar('Params/scale', model.logit_scale.exp().item(), epoch)
        writer.add_scalar('Params/bias', model.bias.item(), epoch)

        # Visualization
        if epoch % VISUALIZE_INTERVAL == 0 and val_vis_embs:
            save_image_tsne(np.concatenate(val_vis_embs), np.concatenate(val_vis_labels), epoch, PLOT_DIR, prefix="VAL")

        # --- [
        if epoch % VALID_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_v_loss, # Validation Loss 
            }
            
            # 1.  (latest)
            latest_path = os.path.join(MODEL_DIR, "latest_model.pth")
            torch.save(checkpoint, latest_path)

            # 2. Best Model (Lowest Validation Loss )
            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                best_path = os.path.join(MODEL_DIR, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f" [Save] Best Model Updated (Val Loss: {best_val_loss:.4f})")

    writer.close()

if __name__ == "__main__":
    main()