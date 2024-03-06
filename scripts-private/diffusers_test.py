import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import sys
sys.path.append("/home/shaohua/ip_adapter")
import os, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="subjects-celebrity/taylorswift")
# "a man in superman costume"
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image_folder = args.subject
subject_name = os.path.basename(image_folder)
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
image_count = 0
faceid_embeds = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    faces = app.get(image)
    faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
    image_count += 1

print(f"Extracted ID embeddings from {image_count} images in {image_folder}")
faceid_embeds = torch.cat(faceid_embeds, dim=1)

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "models/ip-adapter/ip-adapter-faceid-portrait_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

# generate image
prompt = args.prompt 
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=512, height=512, num_inference_steps=30, seed=2023
)
save_dir = "samples-ip"
os.makedirs(save_dir, exist_ok=True)
# Save 4 images as a grid image in save_dir
grid_image = Image.new('RGB', (512 * 2, 512 * 2))
for i, image in enumerate(images):
    image = image.resize((512, 512))
    grid_image.paste(image, (512 * (i % 2), 512 * (i // 2)))

prompt_sig = prompt.replace(" ", "_").replace(",", "_")
grid_path = os.path.join(save_dir, f"{subject_name}-{prompt_sig}.png")
grid_image.save(grid_path)
print(f"Saved to {grid_path}")
