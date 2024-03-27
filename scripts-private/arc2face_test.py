import diffusers

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from arc2face.arc2face import CLIPTextModelWrapper
from insightface.app import FaceAnalysis

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse, sys, glob, cv2
from ldm.util import get_arc2face_id_prompt_embs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="/home/shaohua/adaprompt/subjects-private/xiuchao")
    parser.add_argument("--example_image_count", type=int, default=5, help="Number of example images to use")
    parser.add_argument("--out_image_count",     type=int, default=4, help="Number of images to generate")
    # "a man in superman costume"
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--randface", action="store_true")

    args = parser.parse_args()

    base_model = 'runwayml/stable-diffusion-v1-5'

    text_encoder = CLIPTextModelWrapper.from_pretrained(
        'arc2face/models', subfolder="encoder", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    orig_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda") 

    unet = UNet2DConditionModel.from_pretrained(
        'arc2face/models', subfolder="arc2face", torch_dtype=torch.float16
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

    if not args.randface:
        image_folder = args.subject
        if image_folder.endswith("/"):
            image_folder = image_folder[:-1]

        if os.path.isfile(image_folder):
            # Get the second to the last part of the path
            subject_name = os.path.basename(os.path.dirname(image_folder))
            image_paths = [image_folder]

        else:
            subject_name = os.path.basename(image_folder)
            image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    else:
        subject_name = "randface"
        image_paths = None

    face_app = FaceAnalysis(name='antelopev2', root='arc2face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(512, 512))

    faceid_embeds, id_prompt_emb, neg_id_prompt_emb \
        = get_arc2face_id_prompt_embs(face_app, tokenizer, text_encoder,
                                      image_folder, image_paths, 
                                      images_np=None,
                                      example_image_count=args.example_image_count, 
                                      out_image_count=args.out_image_count,
                                      device='cuda',
                                      rand_face=args.randface, 
                                      noise_level=args.noise,
                                      verbose=True)


    pipeline.text_encoder = orig_text_encoder
    num_images = args.out_image_count

    comp_prompt = args.prompt 

    if len(comp_prompt) > 0:
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        # prompt_embeds_, negative_prompt_embeds_: [4, 77, 768]
        prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(comp_prompt, device='cuda', num_images_per_prompt = num_images,
                                                                         do_classifier_free_guidance=True, negative_prompt=negative_prompt)
        pipeline.text_encoder = text_encoder
        pos_prompt_emb  = torch.cat([id_prompt_emb,     prompt_embeds_], dim=1)
        neg_prompt_emb  = torch.cat([neg_id_prompt_emb, negative_prompt_embeds_], dim=1)
    else:
        pos_prompt_emb = id_prompt_emb
        neg_prompt_emb = neg_id_prompt_emb
        
    images = pipeline(prompt_embeds=pos_prompt_emb, 
                      negative_prompt_embeds=neg_prompt_emb,
                      num_inference_steps=40, 
                      guidance_scale=3.0, 
                      num_images_per_prompt=num_images).images

    save_dir = "samples-ada"
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
