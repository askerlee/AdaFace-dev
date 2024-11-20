from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from PIL import Image

base_model_path = "models/sar/sar.safetensors"
# Load the pipeline and replace the scheduler with DDIM
pipeline = StableDiffusionPipeline.from_single_file(
                base_model_path, 
                torch_dtype=torch.float16
                )
scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler
model = pipeline.unet

# Initialize noisy latents
latents = torch.randn((1, model.config.in_channels, 64, 64)).to(model.device)

# Perform DDIM denoising
scheduler.set_timesteps(50)  # Number of DDIM steps
for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(latents, t)["sample"]
    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode latents into an image
image = pipeline.decode_latents(latents)
final_image = Image.fromarray((image[0] * 255).astype("uint8"))
final_image.show()
