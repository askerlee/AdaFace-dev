import diffusers

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel
from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import os, argparse, sys, glob, cv2

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default="/home/shaohua/adaprompt/subjects-private/xiuchao")
parser.add_argument("--image_count", type=int, default=5, help="Number of images to use")
# "a man in superman costume"
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--noise", type=float, default=0)
parser.add_argument("--randface", action="store_true")

args = parser.parse_args()

base_model = 'runwayml/stable-diffusion-v1-5'

text_encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

orig_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda") 

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=text_encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipeline.scheduler = noise_scheduler
pipeline = pipeline.to('cuda')

app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

if not args.randface:
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
        # Each faceid_embed: [1, 512]
        faceid_embeds.append(torch.from_numpy(face_info.normed_embedding).unsqueeze(0))
        image_count += 1
        if image_count >= args.image_count:
            break

    print(f"Extracted ID embeddings from {image_count} images in {image_folder}")
    if len(faceid_embeds) == 0:
        print("No face detected")
        sys.exit(0)

    # faceid_embeds: [10, 512]
    faceid_embeds = torch.cat(faceid_embeds, dim=0)
    faceid_embeds += torch.randn_like(faceid_embeds) * args.noise
    # faceid_embeds: [1, 512]. 
    # and the resulted prompt embeddings are the same.
    faceid_embeds = faceid_embeds.mean(dim=0, keepdim=True).to(torch.float16).to('cuda')

else:
    faceid_embeds = torch.randn(1, 512).to(torch.float16).to('cuda')
    subject_name = "randface"

faceid_embeds = F.normalize(faceid_embeds, p=2, dim=-1)

# id_prompt_emb: [1, 77, 768]
id_prompt_emb = project_face_embs(pipeline, faceid_embeds)    # pass through the encoder
neg_id_prompt_emb = project_face_embs(pipeline, torch.zeros_like(faceid_embeds))    # pass through the encoder
pipeline.text_encoder = orig_text_encoder
num_images = 4

comp_prompt = args.prompt 

if len(comp_prompt) > 0:
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    # prompt_embeds_, negative_prompt_embeds_: [4, 77, 768]
    prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(comp_prompt, device='cuda', num_images_per_prompt = num_images,
                                                                    do_classifier_free_guidance=True, negative_prompt=negative_prompt)
    pipeline.text_encoder = text_encoder
    prompt_emb = torch.cat([id_prompt_emb.repeat(num_images, 1, 1), prompt_embeds_], dim=1)
    neg_prompt_emb = torch.cat([neg_id_prompt_emb.repeat(num_images, 1, 1), negative_prompt_embeds_], dim=1)
else:
    prompt_emb = id_prompt_emb
    neg_prompt_emb = neg_id_prompt_emb
    
images = pipeline(prompt_embeds=prompt_emb, 
                  negative_prompt_embeds=neg_prompt_emb,
                  num_inference_steps=40, 
                  guidance_scale=3.0, 
                  num_images_per_prompt=num_images).images


save_dir = "samples"
os.makedirs(save_dir, exist_ok=True)
# Save 4 images as a grid image in save_dir
grid_image = Image.new('RGB', (512 * 2, 512 * 2))
for i, image in enumerate(images):
    image = image.resize((512, 512))
    grid_image.paste(image, (512 * (i % 2), 512 * (i // 2)))

prompt_sig = comp_prompt.replace(" ", "_").replace(",", "_")
grid_filepath = os.path.join(save_dir, f"{subject_name}-{prompt_sig}-noise{args.noise}.png")
if os.path.exists(grid_filepath):
    grid_count = 2
    grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{args.noise}-{grid_count}.jpg')
    while os.path.exists(grid_filepath):
        grid_count += 1
        grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{args.noise}-{grid_count}.jpg')

grid_image.save(grid_filepath)
print(f"Saved to {grid_filepath}")
