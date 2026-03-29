import os
from PIL import Image
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
HF_DATASET_REPO = "hannusia123123/cossacks-lora"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- PATHS SETUP ---
RAW_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'

os.makedirs(PROCESSED_PATH, exist_ok=True)

TARGET_SIZE = (1024, 1024)

def process_images():

    valid_files = [f for f in os.listdir(RAW_PATH) if f.lower().endswith(('.png', '.jpg'))]
    
    for idx, filename in enumerate(sorted(valid_files), start=1):
        with Image.open(os.path.join(RAW_PATH, filename)) as img:
            img = img.convert("RGB")
            
            width, height = img.size
            aspect_ratio = width / height
            
            if aspect_ratio > 1:
                new_width = int(TARGET_SIZE[1] * aspect_ratio)
                img = img.resize((new_width, TARGET_SIZE[1]), Image.LANCZOS)
                left = (new_width - TARGET_SIZE[0]) / 2
                img = img.crop((left, 0, left + TARGET_SIZE[0], TARGET_SIZE[1]))
            else:
                new_height = int(TARGET_SIZE[0] / aspect_ratio)
                img = img.resize((TARGET_SIZE[0], new_height), Image.LANCZOS)
                top = (new_height - TARGET_SIZE[1]) / 2
                img = img.crop((0, top, TARGET_SIZE[0], top + TARGET_SIZE[1]))
            
            clean_name = f"cossack_{idx:03d}.jpg"
            img.save(os.path.join(PROCESSED_PATH, clean_name), "JPEG", quality=95)
            
    print(f"Image processing completed. Processed and renamed {len(valid_files)} images.")

def upload_to_hf():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in environment variables.")
        return

    api = HfApi()
    
    try:
        api.create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            exist_ok=True
        )
        print(f"Repository {HF_DATASET_REPO} is ready.")
    except Exception as e:
        print(f"An error occurred while creating the repository: {e}")
        return

    api.upload_folder(
        folder_path=PROCESSED_PATH,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"The dataset is uploaded to https://huggingface.co/datasets/{HF_DATASET_REPO}")
if __name__ == "__main__":
    process_images()
    upload_to_hf()