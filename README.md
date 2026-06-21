# DAGG: Decoupled Anatomically-Guided Grounding

This repository contains the official implementation of the **DAGG (Decoupled Anatomically-Guided Grounding)** framework.

## 📋 Prerequisites

Before running the DAGG pipeline, ensure you have the following external frameworks installed:
* **[nnUNet](https://github.com/MIC-DKFZ/nnUNet):** Required for Stage 1 class-agnostic lesion segmentation.
* **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator):** Required for extracting lung lobe masks.

---

## 🚀 Pipeline & Usage

The DAGG framework consists of a two-stage process. Follow the steps below to train and evaluate the model.

### Step 1: Stage 1 Training & Mask Extraction
1.  **Class-Agnostic Lesion Segmentation:** Train the Stage 1 segmentation network using the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework.
2.  **Lobe Mask Extraction:** Use [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) to extract the anatomical lobe masks from the CT volumes.

### Step 2: Data Preprocessing for Stage 2
Once Stage 1 is complete, prepare the data for Stage 2 training by running the following scripts:

1.  **Extract Candidate Subvolumes:**
    Generate candidate lesion crops in parallel.
    ```bash
    python generate_lesion_crops_parallel.py
    ```
2.  **Preprocess CT Data:**
    Process the CT volumes and save them as `.npy` files for efficient loading.
    ```bash
    python preprocess_ct.py
    ```
3.  **Process Text Embeddings:**
    Extract text embeddings and save them as `.npy` files.
    ```bash
    python process_embeddings.py
    ```

### Step 3: Stage 2 Training
Train the Stage 2 anatomically-guided grounding network. This step updates all learnable parameters in the module.

The Stage 2 image encoder utilizes a truncated lightweight 3D CNN (dropping the deepest layer to form a `32->64->128->256->320` channel progression). This reduces the encoder's parameter count to approximately ~50M, compared to the ~102M parameters of the full 3D U-Net used in Stage 1.

```bash
python train.py
```

### Step 4: Inference
After training is complete, run the inference script to obtain the final full-volume grounding masks.

```bash
python inference.py
```
