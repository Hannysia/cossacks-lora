import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
import os

# --- Configuration ---
REPO_ID = "hannusia123123/cossacks-models"
SUBFOLDER = "experiment_2"
WEIGHT_NAME = "cossacks_exp2-000035.safetensors"

def load_pipeline():
    print("🚀 Loading SDXL Base Model from Hugging Face...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None, 
        use_safetensors=True
    )
    
    pipe.to(device)
    if device == "cuda":
        print("✅ Running on GPU (CUDA)")
    else:
        print("⚠️ CUDA not found, running on CPU (Expect very slow performance)")

    print(f"📦 Loading LoRA Weights: {WEIGHT_NAME}...")
    pipe.load_lora_weights(REPO_ID, weight_name=WEIGHT_NAME, subfolder=SUBFOLDER)
    
    if device == "cuda":
        pipe.enable_vae_slicing()
    
    return pipe

pipeline = load_pipeline()

def generate_image(prompt, neg_prompt, steps, cfg, seed):
    if seed == -1 or seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    print(f"🎨 Generating: {prompt} (Seed: {seed})")
    
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=int(steps),
            guidance_scale=cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator
        ).images[0]
    
    return image, seed

# --- Gradio Interface ---
with gr.Blocks(title="Cossacks Generator") as demo:
    gr.Markdown("# 🎨 Cossacks Animation Style Generator")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Positive Prompt (Trigger: kzk style)", 
                value="kzk style, close-up portrait of a joyful cossack, expressive friendly smile, thick black mustache",
                lines=3
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="blurry eyes, distorted, realistic, 3d, samurai, elf, photography, asian features"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    steps = gr.Slider(10, 50, value=50, step=1, label="Inference Steps")
                    cfg = gr.Slider(1, 15, value=5.0, step=0.5, label="Guidance Scale")
            
            seed = gr.Number(label="Seed (-1 for random)", value=-1)
            run_btn = gr.Button("Generate ✨", variant="primary")
            
        with gr.Column():
            result_img = gr.Image(label="Result", type="pil")
            result_seed = gr.Number(label="Used Seed")

    run_btn.click(
        fn=generate_image, 
        inputs=[prompt, neg_prompt, steps, cfg, seed], 
        outputs=[result_img, result_seed]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())