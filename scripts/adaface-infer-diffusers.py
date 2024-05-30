from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from ldm.modules.arc2face_models import CLIPTextModelWrapper
from ldm.modules.subj_basis_generator import SubjBasisGenerator
from insightface.app import FaceAnalysis

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, argparse, sys, glob, cv2, re
from ldm.util import get_arc2face_id_prompt_embs

    
def save_images(images, num_images_per_row, subject_name, prompt, noise_level, save_dir = "samples-ada"):
    os.makedirs(save_dir, exist_ok=True)
    num_columns = int(np.ceil(len(images) / num_images_per_row))
    # Save 4 images as a grid image in save_dir
    grid_image = Image.new('RGB', (512 * num_images_per_row, 512 * num_columns))
    for i, image in enumerate(images):
        image = image.resize((512, 512))
        grid_image.paste(image, (512 * (i % num_images_per_row), 512 * (i // num_images_per_row)))

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

def load_subj_basis_generators(ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt["string_to_subj_basis_generator_dict"]

def extend_tokenizer_and_text_encoder(pipeline, num_vectors, subject_string):
    if num_vectors < 1:
        raise ValueError(f"num_vectors has to be larger or equal to 1, but is {num_vectors}")

    tokenizer = pipeline.tokenizer
    # Add z0, z1, z2, ..., z15.
    placeholder_tokens = []
    for i in range(0, num_vectors):
        placeholder_tokens.append(f"{subject_string}_{i}")

    # Add the new tokens to the tokenizer.
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {subject_string}. Please pass a different"
            " `subject_string` that is not already in the tokenizer.")

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # print(placeholder_token_ids)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    pipeline.text_encoder.resize_token_embeddings(len(tokenizer))
    return placeholder_tokens, placeholder_token_ids

def update_text_encoder_subj_embs(pipeline, subj_embs, placeholder_token_ids):
    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            token_embeds[token_id] = subj_embs[0, i]
        print(f"Updated {len(placeholder_token_ids)} tokens in the text encoder.")

def update_prompt(prompt, placeholder_tokens, subject_string):
    placeholder_tokens_str = " ".join(placeholder_tokens)

    if re.search(r'\b' + subject_string + r'\b', prompt) is None:
        print(f"Subject string '{subject_string}' not found in the prompt. Adding it.")
        comp_prompt = placeholder_tokens_str + " " + prompt
    else:
        # Replace the subject string with the placeholder tokens.
        comp_prompt = re.sub(r'\b' + subject_string + r'\b', placeholder_tokens_str, prompt)
    return comp_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='runwayml/stable-diffusion-v1-5', 
                        help="Type of checkpoints to use (default: SD 1.5)")
    parser.add_argument("--embman_ckpt", type=str, required=True,
                        help="Path to the checkpoint of the embedding manager")
    parser.add_argument("--subject", type=str, default="/home/shaohua/adaprompt/subjects-private/xiuchao")
    parser.add_argument("--example_image_count", type=int, default=3, help="Number of example images to use")
    parser.add_argument("--out_image_count",     type=int, default=4, help="Number of images to generate")
    parser.add_argument("--prompt", type=str, default="a woman z in superman costume")
    parser.add_argument("--noise",  type=float, default=0)
    parser.add_argument("--randface", action="store_true")
    parser.add_argument("--scale", dest='guidance_scale', type=float, default=4, 
                        help="Guidance scale for the diffusion model")
    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")
    parser.add_argument("--num_vectors", type=int, default=16,
                        help="Number of vectors used to represent the subject.")
    parser.add_argument("--num_images_per_row", type=int, default=4,
                        help="Number of images to display in a row in the output grid image.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of DDIM inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    args = parser.parse_args()
    
    if re.match(r"^\d+$", args.device):
        args.device = f"cuda:{args.device}"

    string_to_subj_basis_generator_dict = load_subj_basis_generators(args.embman_ckpt, args.device)
    if args.subject_string not in string_to_subj_basis_generator_dict:
        print(f"Subject '{args.subject_string}' not found in the embedding manager.")
        sys.exit(1)

    subj_basis_generator = string_to_subj_basis_generator_dict[args.subject_string]
    # In the original ckpt, num_out_layers is 16 for layerwise embeddings. 
    # But we don't do layerwise embeddings here, so we set it to 1.
    subj_basis_generator.num_out_layers = 1
    print(f"Loaded subject basis generator for '{args.subject_string}'.")
    print(repr(subj_basis_generator))

    base_model = args.base_model

    # arc2face_text_encoder maps the face analysis embedding to 16 face embeddings 
    # in the UNet image space.
    arc2face_text_encoder = CLIPTextModelWrapper.from_pretrained(
        'arc2face/models', subfolder="encoder", torch_dtype=torch.float16
    )
    arc2face_text_encoder = arc2face_text_encoder.to(args.device)

    pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
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
    pipeline = pipeline.to(args.device)
    placeholder_tokens, placeholder_token_ids = extend_tokenizer_and_text_encoder(pipeline, args.num_vectors, args.subject_string)

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
            if args.example_image_count > 0:
                image_paths = alltype_image_paths[:args.example_image_count]
            else:
                image_paths = alltype_image_paths
    else:
        subject_name = None
        image_paths = None
        image_folder = None

    face_app = FaceAnalysis(name='antelopev2', root='arc2face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(512, 512))

    subject_name = "randface-" + str(torch.seed()) if args.randface else subject_name
    rand_face_embs = torch.randn(1, 512)

    # Noise level is the *relative* std of the noise added to the face embeddings.
    # A noise level of 0.08 could change gender, but 0.06 is usually safe.
    noise_level = args.noise
    if args.randface and noise_level == 0:
        noise_level = 0.04

    pre_face_embs = rand_face_embs if args.randface else None

    # faceid_embeds is a batch of extracted face analysis embeddings (BS * 512).
    # If extract_faceid_embeds is True, faceid_embeds is an embedding repeated by BS times.
    # Otherwise, faceid_embeds is a batch of out_image_count random embeddings, different from each other.
    # The same applies to id_prompt_emb.
    # id_prompt_emb is in the image prompt space.
    # faceid_embeds: [1, 512]
    # id_prompt_emb: [1, 16, 768]. 
    # Since return_core_id_embs is True, id_prompt_emb is only the 16 core ID embeddings.
    # arc2face prompt template: "photo of a id person"
    # ID embeddings start from "id person ...". So there are 3 template tokens before the 16 ID embeddings.
    faceid_embeds, id_prompt_emb \
        = get_arc2face_id_prompt_embs(face_app, pipeline.tokenizer, arc2face_text_encoder,
                                      extract_faceid_embeds=not args.randface,
                                      pre_face_embs=pre_face_embs,
                                      # image_folder is passed only for logging purpose. 
                                      # image_paths contains the paths of the images.
                                      image_folder=image_folder, image_paths=image_paths,
                                      images_np=None,
                                      id_batch_size=1,
                                      device=args.device,
                                      # input_max_length == 22: only keep the first 22 tokens, 
                                      # including 3 template tokens and 16 ID tokens, and BOS and EOS tokens.
                                      input_max_length=22,
                                      noise_level=noise_level,
                                      return_core_id_embs=True,
                                      gen_neg_prompt=False, 
                                      verbose=True)
    
    # adaface_subj_embs: [1, 1, 16, 768]. adaface_prompt_embs: [1, 77, 768]
    adaface_subj_embs, adaface_prompt_embs = \
        subj_basis_generator(id_prompt_emb, None, None,
                             is_face=True, is_training=False,
                             adaface_prompt_embs_inf_type='full_half_pad')
    # adaface_subj_embs: [1, 16, 768]
    adaface_subj_embs = adaface_subj_embs.squeeze(1)
    # Extend pipeline.text_encoder with the adaface subject emeddings.

    update_text_encoder_subj_embs(pipeline, adaface_subj_embs, placeholder_token_ids)
    comp_prompt = update_prompt(args.prompt, placeholder_tokens, args.subject_string)
    print(f"Prompt: {comp_prompt}")

    negative_prompt = "flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, " \
                      "mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, " \
                      "mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, " \
                      "nude, naked, nsfw, topless, bare breasts"
    
    # prompt_embeds_, negative_prompt_embeds_: [4, 77, 768]
    prompt_embeds_, negative_prompt_embeds_ = pipeline.encode_prompt(comp_prompt, device=args.device, num_images_per_prompt = args.out_image_count,
                                                                     do_classifier_free_guidance=True, negative_prompt=negative_prompt)

    noise = torch.randn(args.out_image_count, 4, 64, 64).cuda()

    images = pipeline(image=noise,
                      prompt_embeds=prompt_embeds_, 
                      negative_prompt_embeds=negative_prompt_embeds_, 
                      num_inference_steps=args.num_inference_steps, 
                      guidance_scale=args.guidance_scale, 
                      num_images_per_prompt=1).images

    save_images(images, args.num_images_per_row, subject_name, f"guide{args.guidance_scale}", noise_level)
