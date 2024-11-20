from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import torch
from PIL import Image

base_model_path = "models/sar/sar.safetensors"
# Load the pipeline and replace the scheduler with DDIM
pipeline = StableDiffusionPipeline.from_single_file(
                base_model_path, 
                torch_dtype=torch.float16
                )
pipeline.to("cuda")
scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler
model = pipeline.unet

# Initialize noisy latents
latents = torch.randn((1, model.config.in_channels, 64, 64), dtype=torch.float16).to(model.device)
prompt1 = "portrait, a man"
prompt2 = "portrait, a man in front of a baloon"
negative_prompt = "blurry, low resolution, bad composition, deformed buildings"

with torch.no_grad():
    text_embeddings1 = pipeline.text_encoder(pipeline.tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda"))[0]
    text_embeddings1 = text_embeddings1.to(torch.float16)
    text_embeddings2 = pipeline.text_encoder(pipeline.tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda"))[0]
    text_embeddings2 = text_embeddings2.to(torch.float16)
    negative_embeddings = pipeline.text_encoder(
        pipeline.tokenizer(negative_prompt, return_tensors="pt").input_ids.to("cuda")
    )[0]
    negative_embeddings = negative_embeddings.to(torch.float16)

guidance_scale = 6
single_mix_coeff = 0.2
# Perform DDIM denoising
scheduler.set_timesteps(50)  # Number of DDIM steps
for i, t in enumerate(scheduler.timesteps):
    with torch.no_grad():
        noise_pred1 = model(latents, t, text_embeddings1)["sample"]
        noise_pred2 = model(latents, t, text_embeddings2)["sample"]
        noise_pred_negative = model(latents, t, negative_embeddings)["sample"]
        noise_pred = noise_pred1 * single_mix_coeff + noise_pred2 * (1 - single_mix_coeff)
        noise_pred = noise_pred + guidance_scale * (noise_pred - noise_pred_negative)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if i % 10 == 9:
        with torch.no_grad():
            image = pipeline.decode_latents(latents)
            final_image = Image.fromarray((image[0] * 255).astype("uint8"))
            final_image.save(f"output-mix{single_mix_coeff}-{i+1}.png")
