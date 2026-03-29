import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import snapshot_download, HfApi

# --- FETCH HF TOKEN ---
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    print("Successfully loaded HF_TOKEN from Kaggle Secrets.")
except ImportError:
    HF_TOKEN = os.getenv("HF_TOKEN")
    print("Kaggle Secrets not found. Falling back to environment variables.")

# --- CONFIGURATION ---
HF_DATASET_REPO = "hannusia123123/cossacks-lora"
LOCAL_DIR = "/kaggle/working/cossacks_data"
TRIGGER_WORD = "kzk style"
MODEL_ID = "Salesforce/blip-image-captioning-base"

def download_dataset():

    print(f"Downloading dataset from HF: {HF_DATASET_REPO}...")
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        token=HF_TOKEN
    )
    print(f"Dataset downloaded to {LOCAL_DIR}")

def run_captioning():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

    image_files = [f for f in os.listdir(LOCAL_DIR) if f.lower().endswith(('.png', '.jpg'))]
    print(f"Found {len(image_files)} images. Starting annotation...")

    for filename in tqdm(image_files):
        img_path = os.path.join(LOCAL_DIR, filename)
        try:
            raw_image = Image.open(img_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{TRIGGER_WORD}, {caption}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Annotation completed! .txt files are saved next to the images.")

def upload_annotations():

    print("Uploading annotations to Hugging Face...")
    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set. Cannot upload.")
        return

    api = HfApi()
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns="*.txt",
        commit_message="Add BLIP annotations with trigger word"
    )
    print(f"Annotations successfully uploaded to https://huggingface.co/datasets/{HF_DATASET_REPO}")

if __name__ == "__main__":
    download_dataset()
    run_captioning()
    upload_annotations()