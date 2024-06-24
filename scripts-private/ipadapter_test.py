import cv2
from insightface.app import FaceAnalysis
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import sys
#sys.path.append("/home/shaohua/ip_adapter")
import os, glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="subjects-celebrity/taylorswift")
parser.add_argument("--image_count", type=int, default=5, help="Number of images to use")
# "a man in superman costume"
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--noise", type=float, default=0)
args = parser.parse_args()

from ip_adapter.ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
# FaceAnalysis will try to find the ckpt in: models/arc2face/models/antelopev2. 
# Note the second "model" in the path.
app = FaceAnalysis(name="buffalo_l", root='models/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

image_folder = args.subject
if image_folder.endswith("/"):
    image_folder = image_folder[:-1]

if os.path.isfile(image_folder):
    image_paths = [image_folder]
    # Get the second to the last part of the path
    subject_name = os.path.basename(os.path.dirname(image_folder))

else:
    subject_name = os.path.basename(image_folder)
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

image_count = 0
faceid_embeds = []
for image_path in image_paths:
    image_np = cv2.imread(image_path)
    #image_np = np.array(Image.open(image_path))
    #image_np2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    face_infos = app.get(image_np)
    print(image_path, len(face_infos))
    if len(face_infos) == 0:
        continue
    # only use the maximum face
    face_info = sorted(face_infos, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
    # Each faceid_embed: [1, 1, 512]
    faceid_embeds.append(torch.from_numpy(face_info.normed_embedding).unsqueeze(0).unsqueeze(0))
    image_count += 1
    if image_count >= args.image_count:
        break

print(f"Extracted ID embeddings from {image_count} images in {image_folder}")
if len(faceid_embeds) == 0:
    print("No face detected")
    sys.exit(0)

# faceid_embeds: [1, 10, 512]
faceid_embeds = torch.cat(faceid_embeds, dim=1)
faceid_embeds += torch.randn_like(faceid_embeds) * args.noise
# faceid_embeds: [1, 1, 512]. If we don't keepdim, then it's [1, 512], 
# and the resulted prompt embeddings are the same.
faceid_embeds = faceid_embeds.mean(dim=1, keepdim=True)
#print(faceid_embeds.norm(dim=-1))
#faceid_embeds = F.normalize(faceid_embeds, p=2, dim=-1)
n_cond = faceid_embeds.shape[1]

base_model_path = "runwayml/stable-diffusion-v1-5" #"SG161222/Realistic_Vision_V4.0_noVAE"
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
# num_tokens * n_cond = 80 is the maximum number of image tokens that can be used in the model.
# If 10 face images are provided, 160 image tokens will be generated, 
# but only the first 80 will be effective.
# If we take the average of the face embeddings (as above), then only 16 image tokens will be generated,
# and all of them will be effective.
ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, 
                           n_cond=n_cond, 
                           ablation_no_attn_proc=False,
                           ablation_no_image_proj=False)

# generate image
prompt = args.prompt 
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

subj_embeds, uncond_embeds = ip_model.get_image_embeds(faceid_embeds)

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
grid_filepath = os.path.join(save_dir, f"{subject_name}-{prompt_sig}-noise{args.noise}.png")
if os.path.exists(grid_filepath):
    grid_count = 2
    grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{args.noise}-{grid_count}.jpg')
    while os.path.exists(grid_filepath):
        grid_count += 1
        grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{args.noise}-{grid_count}.jpg')

grid_image.save(grid_filepath)
print(f"Saved to {grid_filepath}")
