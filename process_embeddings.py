import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
from torch import Tensor

# ================= CONFIGURATION =================
# Dataset path
DATASET_PATH = "/path/to/dataset/dataset.json"
# Output directory path
OUTPUT_DIR = "/path/to/output/text_embeddings/Qwen3-Embedding-4B"

# Select model ('biovil' or 'qwen')
MODEL_TYPE = "qwen" 

# HuggingFace model path (modify as needed)
MODEL_PATHS = {
    # "biovil": "microsoft/BiomedVLP-CXR-BERT-specialized", # Example: BioViL
    "biovil": "microsoft/BiomedVLP-BioViL-T",               # Example: BioViL
    "qwen": "Qwen/Qwen3-Embedding-4B"                       # Example: Qwen
}

# GPU configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


class TextEncoder:
    def __init__(self, model_type, model_path, device):
        self.model_type = model_type
        self.device = device
        print(f"Loading {model_type} model from {model_path}...")
        
        if model_type == "biovil":
            # trust_remote_code=True is required to load BioViL's custom method (get_projected_text_embeddings).
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        elif model_type == "qwen":
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_path, 
            #     device_map="auto", 
            #     torch_dtype=torch.float16,
            #     trust_remote_code=True
            # )
            # if self.tokenizer.pad_token is None:
            #     self.tokenizer.pad_token = self.tokenizer.eos_token

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
        
        self.model.eval()

    def get_embeddings(self, texts):
        """Takes a list of texts and returns an embedding tensor"""
        if not texts:
            return None

        # Preprocessing and encoding according to BioViL official usage
        if self.model_type == "biovil":
            with torch.no_grad():
                # Use batch_encode_plus as in the official example (or calling tokenizer works similarly)
                tokenizer_output = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=texts,
                    add_special_tokens=True,
                    padding='longest', # Pad to the longest sequence in the batch
                    return_tensors='pt'
                ).to(self.device)

                # ★ Key modification: Use the method from the official usage
                # This method returns normalized embeddings passed through the projection layer.
                embeddings = self.model.get_projected_text_embeddings(
                    input_ids=tokenizer_output.input_ids,
                    attention_mask=tokenizer_output.attention_mask
                )
                
                # The result is already a tensor, so move to CPU and convert to numpy
                return embeddings.cpu().numpy()

        elif self.model_type == "qwen":
            # 1. Define Task Instruction
            task = 'Given a chest CT finding, identify the pathological abnormality, precise anatomical location, and detailed radiological characteristics including type, severity, and distribution'

            # 2. Apply Instruction to all inputs
            # (Treat all finding texts as queries and attach instructions)
            input_texts = [get_detailed_instruct(task, t) for t in texts]
            
            # 3. Tokenize
            batch_dict = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(self.device)

            # 4. Model Forward & Pooling
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # (Optional) Retrieval models usually perform normalization. 
                # It's not in the official code snippet, but adding the line below is recommended for retrieval performance.
                # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()
        
            # Qwen logic (keep existing)
            # inputs = self.tokenizer(
            #     texts, 
            #     padding=True, 
            #     truncation=True, 
            #     return_tensors="pt", 
            #     max_length=512
            # ).to(self.device)

            # with torch.no_grad():
            #     outputs = self.model(**inputs)
            #     # Last token hidden state extraction
            #     hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
            #     # Find the last actual token position considering the attention mask
            #     sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
            #     embeddings = hidden_states[torch.arange(hidden_states.size(0), device=self.device), sequence_lengths]
            
            max_length = 8192

            # Tokenize the input texts
            batch_dict = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict.to(self.model.device)
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            return embeddings.cpu().numpy()

def main():
    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load model
    encoder = TextEncoder(MODEL_TYPE, MODEL_PATHS[MODEL_TYPE], DEVICE)

    # 3. Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 4. Process data (process all lists regardless of keys like train, val, etc.)
    total_cases = 0
    for split_key, case_list in data.items():
        print(f"Processing split: {split_key} ({len(case_list)} cases)")
        
        for case in tqdm(case_list, desc=f"Encoding {split_key}"):
            case_name = case.get("name")
            findings_dict = case.get("findings", {})
            
            if not case_name or not findings_dict:
                continue

            # Ensure findings order (sort by dictionary keys)
            sorted_keys = sorted(findings_dict.keys(), key=lambda x: int(x))
            texts = [findings_dict[k] for k in sorted_keys]
            
            # Extract embeddings
            embeddings = encoder.get_embeddings(texts) # Shape: (Num_findings, Embedding_dim)

            if embeddings is None:
                continue

            base_name = case_name.replace(".nii.gz", "").replace(".nii", "")
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
            np.save(save_path, embeddings)

            total_cases += 1

    print(f"Done! Processed {total_cases} cases. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()