from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from adaface.diffusers_attn_lora_capture import UNetMidBlock2D_forward_capture
from einops import rearrange
import argparse
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="Input image path")
parser.add_argument("--output", type=str, default="Output image path")
args = parser.parse_args()

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
prompt2 = "portrait, a man in a park"
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

scaling_factor = pipeline.vae.config.scaling_factor

for i, t in enumerate(scheduler.timesteps):
    with torch.no_grad():
        #noise_pred1 = model(latents, t, text_embeddings1)["sample"]
        noise_pred2 = model(latents, t, text_embeddings2)["sample"]
        noise_pred_negative = model(latents, t, negative_embeddings)["sample"]
    
        #noise_pred1 = noise_pred1 + guidance_scale * (noise_pred1 - noise_pred_negative)
        noise_pred2 = noise_pred2 + guidance_scale * (noise_pred2 - noise_pred_negative)
        #latents1 = scheduler.step(noise_pred1, t, latents).prev_sample
        latents2 = scheduler.step(noise_pred2, t, latents).prev_sample
        latents  = latents2

        if False:
            # scaled_latents1 = latents1 / scaling_factor
            #image1 = pipeline.vae.decode(scaled_latents1, return_dict=False)[0]
            #pipeline.vae.decoder.mid_block.encoder_hidden_states = \
            #    [ rearrange(hs, 'b n y x -> b (y x) n').contiguous() for hs in pipeline.vae.decoder.mid_block.hidden_states ]
            scaled_latents2 = latents2 / scaling_factor
            image2 = pipeline.vae.decode(scaled_latents2, return_dict=False)[0]
            latents = pipeline.vae.encode(image2).latent_dist.mode()
            latents = scaling_factor * latents

    if i % 10 == 9:
        with torch.no_grad():
            mid_block_forward = pipeline.vae.decoder.mid_block.forward
            pipeline.vae.decoder.mid_block.forward = UNetMidBlock2D_forward_capture.__get__(pipeline.vae.decoder.mid_block)
            image = pipeline.decode_latents(latents)
            final_image = Image.fromarray((image[0] * 255).astype("uint8"))
            final_image.save(f"output-mix{single_mix_coeff}-{i+1}.png")
            pipeline.vae.decoder.mid_block.forward = mid_block_forward
            image2 = pipeline.decode_latents(latents)
            final_image2 = Image.fromarray((image2[0] * 255).astype("uint8"))
            final_image2.save(f"orig-output-mix{single_mix_coeff}-{i+1}.png")

'''
input_image = Image.open(args.input).convert("RGB")
input_image = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).half().to("cuda")
input_image = input_image.unsqueeze(0) / 255.0 * 2 - 1
input_image = F.interpolate(input_image, (512, 512), mode="bilinear")
latents = pipeline.vae.encode(input_image).latent_dist.mode()
with torch.no_grad():
    recon_image = pipeline.vae.decode(latents, return_dict=False)[0]
    recon_image = ((recon_image[0].detach().cpu() + 1) / 2 * 255).numpy().astype("uint8")
    recon_image = np.transpose(recon_image, (1, 2, 0))
    recon_image = Image.fromarray(recon_image)
    recon_image.save(args.output)
    exit(0)
'''
