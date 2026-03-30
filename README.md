# 🎨 SDXL Style LoRA Training — "Cossacks" 

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/hannusia123123/cossacks-models)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo%20Space-blue)](https://huggingface.co/spaces/hannusia123123/cossacks-lora-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Logging-orange)](https://api.wandb.ai/links/anyknysh2000-/byrdyz1b)

**Objective:** Train a Stable Diffusion XL (SDXL) model to generate images in the specific 2D cel-shaded style of the classic Ukrainian animated film "How the Cossacks..." using Low-Rank Adaptation (LoRA).

---

## 📂 1. Input Data and Annotations
* **Dataset Size:** 43 images.
* **Format:** Image-text pairs (`.jpg` + `.txt` with detailed captions).
* **Dataset Link:** [https://huggingface.co/datasets/hannusia123123/cossacks-lora/tree/main]
* **Annotation Strategy:** The dataset was carefully curated to cover three main categories: characters, architecture/landscapes, and objects. The trigger token `kzk style` was used in combination with highly descriptive scene captions to help the model disentangle the specific 2D style from the underlying content.

---

## 💻 2. Model Training Code & Setup
* **Environment:** Kaggle (2x T4 GPU)
* **Training Code:** [https://github.com/Hannysia/cossacks-lora/blob/master/notebooks/02-training-and-validation.ipynb]
* **Training Logs & Metrics (W&B):** [https://api.wandb.ai/links/anyknysh2000-/byrdyz1b]


**Key Hyperparameters & Architectural Decisions:**
* **Network Dim / Alpha:** `32 / 32`. Chosen as the optimal balance for a style LoRA on SDXL, providing enough capacity to learn the 2D lineart and flat colors without severe overfitting.
* **Optimizer:** `Prodigy`. This adaptive optimizer was selected to eliminate the need for manual learning rate tuning. Given the limited training time and dataset size, Prodigy automatically discovered the optimal learning rate, ensuring fast convergence. The argument d_coef=2 was applied to maintain a healthy learning speed throughout the session, allowing the model to effectively capture the stylized "Cossack" aesthetic.
* **Gradient Accumulation:** `8` (with `train_batch_size=1`). This created an effective batch size of 16 across two T4 GPUs, which significantly smoothed and stabilized the Loss curve.

---

## 🖼️ 3. Evaluation & Inference
After comparing checkpoints from various stages (Epochs 30, 35, and 40), **Epoch 35** was identified as the sweet spot. It provides the most authentic 2D aesthetic with bold outlines while maintaining structural integrity.

* **Interactive Gallery & Code:** [https://github.com/Hannysia/cossacks-lora/blob/master/notebooks/03-generating-test-results.ipynb]

  *(The notebook contains 10 high-resolution generations with full prompt/seed transparency)*
* **Model Weights:** [https://huggingface.co/hannusia123123/cossacks-models/tree/main/experiment_2]

---

## 🧠 4. Discussion & Challenges

### **1. Micro-anatomy & Facial Features**
Due to the small dataset size (43 samples), the base SDXL model occasionally conflicted with the LoRA when rendering small facial details on wide-angle shots. 
* **Solution:** Implemented **Prompt Engineering** and aggressive **Negative Prompts** (`samurai, realistic, 3d, blurry eyes`) to force the model into the correct 2D proportions. For production environments, an Inpainting/ADetailer pass is recommended.

### **2. Cultural Overlap (The "Samurai" Effect)**
During early tests, the model sometimes confused the cossack "oseledets" (forelock) with traditional Asian hairstyles. 
* **Solution:** Refined the training captions to emphasize specific Ukrainian attributes and utilized Negative Prompts to steer the model away from "Samurai" or "Shogun" aesthetics.

---

## 🛠️ Installation & Local Execution

### 1. Prerequisites
* **Python 3.10+** (tested on 3.11)
* **CUDA-enabled GPU** (at least 12GB VRAM recommended for SDXL)
* **Windows/Linux/MacOS**

### 2. Setup Environment
Clone the repository and create a virtual environment:
```bash
git clone https://huggingface.co/hannusia123123/cossacks-models
cd cossacks-models
python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\activate
# Activate on Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
I have pinned the exact versions that were used during development to ensure 100% reproducibility:

```bash
pip install -r requirements.txt
```

### 4. Launch the App
Run the following command to start the Gradio interface:

```bash
python app.py
```
Note: On the first run, the script will download the base SDXL model (~14GB) from the Hugging Face Hub. This will only happen once.


### 🌐 Live Web Demo (Hugging Face Spaces)

If you prefer not to install anything locally, a live demo is available here:

👉 [https://huggingface.co/spaces/hannusia123123/cossacks-lora-demo]

⚠️ Important Performance Note: The Space is currently running on a Free CPU Tier.

Model Loading: Very fast.

Inference Speed: Extremeley slow (~30-40 minutes per image). This is purely due to hardware limitations on the free hosting. For a standard evaluation experience, please refer to the Local Execution steps above.