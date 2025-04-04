from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import numpy as np
from PIL import Image
import argparse
from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="Input image path")
parser.add_argument("--output", type=str, default="Output image path")
parser.add_argument("--comp", type=str, default="in a park")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--test_clsneg", type=str2bool, const=True, default=False, nargs="?")
args = parser.parse_args()

if args.seed > 0:
    seed_everything(args.seed)

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

def test_enc_dec(pipeline, input, output):
    input_image = Image.open(input).convert("RGB")
    input_image = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).half().to("cuda")
    input_image = input_image.unsqueeze(0) / 255.0 * 2 - 1
    input_image = F.interpolate(input_image, (512, 512), mode="bilinear")
    latents = pipeline.vae.encode(input_image).latent_dist.mode()
    with torch.no_grad():
        recon_image = pipeline.vae.decode(latents, return_dict=False)[0]
        recon_image = torch.clamp(recon_image, -1, 1)
        recon_image = ((recon_image[0].detach().cpu() + 1) / 2 * 255).numpy().astype("uint8")
        recon_image = np.transpose(recon_image, (1, 2, 0))
        recon_image = Image.fromarray(recon_image)
        recon_image.save(output)

def test_dec_enc(pipeline, latents):
    scaling_factor = pipeline.vae.config.scaling_factor
    scaled_latents = latents / scaling_factor
    with torch.no_grad():
        image = pipeline.vae.decode(scaled_latents, return_dict=False)[0]
        latents = pipeline.vae.encode(image).latent_dist.mode()

    latents = scaling_factor * latents

# Initialize noisy latents
latents0 = torch.randn((1, model.config.in_channels, 64, 64), dtype=torch.float16).to(model.device)
prompt0          = "portrait, taylor swift"
prompt_cls0      = "portrait, a woman"
negative_prompt0 = "blurry, low resolution, bad composition, deformed buildings"

guidance_scale = 6
# Perform DDIM denoising
scheduler.set_timesteps(50)  # Number of DDIM steps
# N = 2: noise_pred = noise_pred * (guidance_scale - 0.05) - noise_pred_negative * (guidance_scale - 1)
# The diffusion model is robust enough to handle this level of dampening.
cfg_dampens = [0, 0, 0.05, 0]

if args.test_clsneg:
    for N in range(4):
        if N == 0:
            # The normal CFG prompts
            prompt = ", ".join([prompt0, args.comp])
            negative_prompt = negative_prompt0
        elif N == 1 or N == 2:
            # The CLSNEG prompts
            prompt = ", ".join([prompt0, args.comp])
            negative_prompt = ", ".join([prompt_cls0, args.comp, negative_prompt0])
        elif N == 3:
            # The non-compositional prompts
            prompt = prompt0
            negative_prompt = negative_prompt0

        cfg_dampen = cfg_dampens[N]

        print(f"Prompt: {prompt}")
        print(f"Negative Prompt: {negative_prompt}")
        
        with torch.no_grad():
            text_embeddings = pipeline.text_encoder(pipeline.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda"))[0]
            text_embeddings = text_embeddings.to(torch.float16)

            negative_embeddings = pipeline.text_encoder(
                pipeline.tokenizer(negative_prompt, return_tensors="pt").input_ids.to("cuda")
            )[0]
            negative_embeddings = negative_embeddings.to(torch.float16)

        latents = latents0.clone()
        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                noise_pred          = model(latents, t, text_embeddings)["sample"]
                noise_pred_negative = model(latents, t, negative_embeddings)["sample"]
            
                noise_pred = noise_pred * (guidance_scale - cfg_dampen) - noise_pred_negative * (guidance_scale - 1)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            if i % 10 == 9:
                with torch.no_grad():
                    image = pipeline.decode_latents(latents)
                    out_image = Image.fromarray((image[0] * 255).astype("uint8"))
                    if N == 0:
                        out_filepath = f"{N}-normal-{i+1}.png"
                    elif N == 1 or N == 2:
                        out_filepath = f"{N}-clsneg-{i+1}.png"
                    elif N == 3:
                        out_filepath = f"{N}-nocomp-{i+1}.png"

                    out_image.save(out_filepath)
                    print(f"Saved {out_filepath}")
                    
