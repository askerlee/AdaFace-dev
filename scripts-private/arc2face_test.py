from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from ldm.modules.arc2face_models import CLIPTextModelWrapper
from insightface.app import FaceAnalysis

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse, sys, glob, cv2
from ldm.util import get_arc2face_id_prompt_embs

    
def save_images(images, subject_name, prompt, noise_level, save_dir = "samples-ada"):
    
    os.makedirs(save_dir, exist_ok=True)
    # Save 4 images as a grid image in save_dir
    grid_image = Image.new('RGB', (512 * 2, 512 * 2))
    for i, image in enumerate(images):
        image = image.resize((512, 512))
        grid_image.paste(image, (512 * (i % 2), 512 * (i // 2)))

    prompt_sig = prompt.replace(" ", "_").replace(",", "_")
    grid_filepath = os.path.join(save_dir, f"{subject_name}-{prompt_sig}-noise{noise_level:.02f}.png")
    if os.path.exists(grid_filepath):
        grid_count = 2
        grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{noise_level:.02f}-{grid_count}.jpg')
        while os.path.exists(grid_filepath):
            grid_count += 1
            grid_filepath = os.path.join(save_dir, f'{subject_name}-{prompt_sig}-noise{noise_level:.02f}-{grid_count}.jpg')

    grid_image.save(grid_filepath)
    print(f"Saved to {grid_filepath}")

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
        'models/arc2face', subfolder="encoder", torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    orig_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda") 

    unet = UNet2DConditionModel.from_pretrained(
        'models/arc2face', subfolder="arc2face", torch_dtype=torch.float16
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
            image_types = ["*.jpg", "*.png", "*.jpeg"]
            alltype_image_paths = []
            for image_type in image_types:
                # glob returns the full path.
                image_paths = glob.glob(os.path.join(image_folder, image_type))
                if len(image_paths) > 0:
                    alltype_image_paths.extend(image_paths)
            # image_paths contain at most args.example_image_count full image paths.
            image_paths = alltype_image_paths[:args.example_image_count]
    else:
        subject_name = None
        image_paths = None
        image_folder = None

    # FaceAnalysis will try to find the ckpt in: models/arc2face/models/antelopev2. 
    # Note the second "model" in the path.
    face_app = FaceAnalysis(name='antelopev2', root='models/arc2face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(512, 512))
    
    subject_name = "randface-" + str(torch.seed()) if args.randface else subject_name
    rand_face_embs=torch.randn(1, 512)
    id_batch_size = args.out_image_count

    input_max_length = 22

    # Noise level is the *relative* std of the noise added to the face embeddings.
    # A noise level of 0.08 could change gender, but 0.06 is usually safe.
    for noise_level in (0, 0.03, 0.06):
        pre_face_embs = rand_face_embs if args.randface else None

        # id_prompt_emb, neg_id_prompt_emb are in the image prompt space.
        faceid_embeds, id_prompt_emb, neg_id_prompt_emb \
            = get_arc2face_id_prompt_embs(face_app, tokenizer, text_encoder,
                                            extract_faceid_embeds=not args.randface,
                                            pre_face_embs=pre_face_embs,
                                            image_folder=image_folder, image_paths=image_paths,
                                            images_np=None,
                                            id_batch_size=id_batch_size,
                                            device='cuda',
                                            input_max_length=input_max_length,
                                            noise_level=noise_level,
                                            gen_neg_prompt=True, 
                                            verbose=True)

        if args.randface:
            id_prompt_emb = id_prompt_emb.repeat(args.out_image_count, 1, 1)
            neg_id_prompt_emb = neg_id_prompt_emb.repeat(args.out_image_count, 1, 1)

        pipeline.text_encoder = orig_text_encoder

        filler_prompt = "photo of a id person"
        comp_prompt = args.prompt 
        test_core_embs = False
        if test_core_embs:
            comp_prompt = filler_prompt

        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        # prompt_embeds_, negative_prompt_embeds_: [4, 77, 768]
        prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(comp_prompt, device='cuda', num_images_per_prompt = args.out_image_count,
                                                                         do_classifier_free_guidance=True, negative_prompt=negative_prompt)
        pipeline.text_encoder = text_encoder

        if test_core_embs:
            # By replacing comp_prompt with filler_prompt, and replacing prompt_embeds_ 4:20 with id_prompt_emb 4:20,
            # the resulting images are quite similar to those generated with id_prompt_emb. 
            # This shows id_prompt_emb 4:20 contains the ID of the person.
            prompt_embeds_[:, 4:20] = id_prompt_emb[:, 4:20]
            id_prompt_emb = prompt_embeds_
            negative_prompt_embeds_[:, :neg_id_prompt_emb.shape[1]] = neg_id_prompt_emb
            
        if len(comp_prompt) > 0:
            pos_prompt_emb  = torch.cat([id_prompt_emb,     prompt_embeds_], dim=1)
            neg_prompt_emb  = torch.cat([neg_id_prompt_emb, negative_prompt_embeds_], dim=1)
        else:
            pos_prompt_emb = id_prompt_emb
            neg_prompt_emb = neg_id_prompt_emb

        noise = torch.randn(args.out_image_count, 4, 64, 64).cuda()
        negative_prompt_embeds_ = negative_prompt_embeds_[:, :neg_id_prompt_emb.shape[1]]

        for guidance_scale in [2, 4]:
            images = pipeline(image=noise,
                            prompt_embeds=pos_prompt_emb, 
                            negative_prompt_embeds=negative_prompt_embeds_, 
                            num_inference_steps=40, 
                            guidance_scale=guidance_scale, 
                            num_images_per_prompt=1).images

            save_images(images, subject_name, f"guide{guidance_scale}-origneg", noise_level)

            images2 = pipeline(image=noise,
                                prompt_embeds=pos_prompt_emb, 
                                negative_prompt_embeds=neg_prompt_emb,
                                num_inference_steps=40, 
                                guidance_scale=guidance_scale, 
                                num_images_per_prompt=1).images
            save_images(images2, subject_name, f"guide{guidance_scale}-arcneg", noise_level)
