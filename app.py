import sys
sys.path.append('./')

from adaface.adaface_wrapper import AdaFaceWrapper
import torch
import numpy as np
import random
import os, re
import time
import gradio as gr
import spaces

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--adaface_encoder_types", type=str, nargs="+", default=["consistentID", "arc2face"],
                    choices=["arc2face", "consistentID"], help="Type(s) of the ID2Ada prompt encoders")
parser.add_argument('--adaface_ckpt_path', type=str, default='models/adaface/VGGface2_HQ_masks2024-10-14T16-09-24_zero3-ada-3500.pt',
                    help="Path to the checkpoint of the ID2Ada prompt encoders")
# If adaface_encoder_cfg_scales is not specified, the weights will be set to 6.0 (consistentID) and 1.0 (arc2face).
parser.add_argument('--adaface_encoder_cfg_scales', type=float, nargs="+", default=[6.0, 1.0],    
                    help="Scales for the ID2Ada prompt encoders")
parser.add_argument("--enabled_encoders", type=str, nargs="+", default=None,
                    choices=["arc2face", "consistentID"], 
                    help="List of enabled encoders (among the list of adaface_encoder_types). Default: None (all enabled)")
parser.add_argument('--model_style_type', type=str, default='photorealistic',
                    choices=["realistic", "anime", "photorealistic"], help="Type of the base model")
parser.add_argument("--guidance_scale", type=float, default=5.0,
                    help="The guidance scale for the diffusion model. Default: 5.0")
parser.add_argument("--unet_uses_attn_lora", type=str2bool, nargs="?", const=True, default=False,
                    help="Whether to use LoRA in the Diffusers UNet model")
# --attn_lora_layer_names and --q_lora_updates_query are only effective
# when --unet_uses_attn_lora is set to True.
parser.add_argument("--attn_lora_layer_names", type=str, nargs="*", default=['q', 'k', 'v', 'out'],
                    choices=['q', 'k', 'v', 'out'], help="Names of the cross-attn components to apply LoRA on")
parser.add_argument("--q_lora_updates_query", type=str2bool, nargs="?", const=True, default=True,
                    help="Whether the q LoRA updates the query in the Diffusers UNet model. "
                         "If False, the q lora only updates query2.")

parser.add_argument('--extra_save_dir', type=str, default=None, help="Directory to save the generated images")
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--ip', type=str, default="0.0.0.0")
args = parser.parse_args()

model_style_type2base_model_path = {
    "realistic": "models/rv51/realisticVisionV51_v51VAE_dste8.safetensors",
    "anime": "models/aingdiffusion/aingdiffusion_v170_ar.safetensors",
    "photorealistic": "models/sar/sar.safetensors" # LDM format. Needs to be converted.
}
base_model_path = model_style_type2base_model_path[args.model_style_type]

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if args.gpu is None else f"cuda:{args.gpu}"
print(f"Device: {device}")

global adaface
adaface = AdaFaceWrapper(pipeline_name="text2img", base_model_path=base_model_path,
                         adaface_encoder_types=args.adaface_encoder_types, 
                         adaface_ckpt_paths=args.adaface_ckpt_path, 
                         adaface_encoder_cfg_scales=args.adaface_encoder_cfg_scales,
                         enabled_encoders=args.enabled_encoders,
                         unet_types=None, extra_unet_dirpaths=None, unet_weights_in_ensemble=None, 
                         unet_uses_attn_lora=args.unet_uses_attn_lora,
                         attn_lora_layer_names=args.attn_lora_layer_names,
                         shrink_subj_attn=False,
                         q_lora_updates_query=args.q_lora_updates_query,
                         device='cpu')

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def swap_to_gallery(images):
    # Update uploaded_files_gallery, show files, hide clear_button_column
    # Or:
    # Update uploaded_init_img_gallery, show init_img_files, hide init_clear_button_column
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(value=images, visible=False)

def remove_back_to_files():
    # Hide uploaded_files_gallery,    show clear_button_column,      hide files,           reset init_img_selected_idx
    # Or:
    # Hide uploaded_init_img_gallery, hide init_clear_button_column, show init_img_files,  reset init_img_selected_idx
    return gr.update(visible=False), gr.update(visible=False), gr.update(value=None, visible=True), gr.update(value="")

@spaces.GPU
def generate_image(image_paths, image_paths2, guidance_scale, perturb_std,
                   num_images, prompt, negative_prompt, enhance_face, enhance_composition, 
                   seed, subj_name_sig, progress=gr.Progress(track_tqdm=True)):

    global adaface

    adaface.to(device)

    if image_paths is None or len(image_paths) == 0:
        raise gr.Error(f"Cannot find any input face image! Please upload a face image.")
    
    if image_paths2 is not None and len(image_paths2) > 0:
        image_paths = image_paths + image_paths2

    if prompt is None:
        prompt = ""

    adaface_subj_embs = \
        adaface.prepare_adaface_embeddings(image_paths=image_paths, face_id_embs=None, 
                                           avg_at_stage='id_emb',
                                           perturb_at_stage='img_prompt_emb',
                                           perturb_std=perturb_std, update_text_encoder=True)
    
    if adaface_subj_embs is None:
        raise gr.Error(f"Failed to detect any faces! Please try with other images")

    # Sometimes the pipeline is on CPU, although we've put it on CUDA (due to some offloading mechanism).
    # Therefore we set the generator to the correct device.
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Manual seed: {seed}.")
    # Generate two images each time for the user to select from.
    noise = torch.randn(num_images, 3, 512, 512, device=device, generator=generator)
    #print(noise.abs().sum())
    # samples: A list of PIL Image instances.
    if enhance_face and "face portrait" not in prompt:
        if "portrait" in prompt:
            # Enhance the face features by replacing "portrait" with "face portrait".
            prompt = prompt.replace("portrait", "face portrait")
        else:
            prompt = "face portrait, " + prompt

    generator = torch.Generator(device=adaface.pipeline._execution_device).manual_seed(seed)
    samples = adaface(noise, prompt, negative_prompt=negative_prompt, 
                      guidance_scale=guidance_scale, 
                      out_image_count=num_images, generator=generator, 
                      repeat_prompt_for_each_encoder=enhance_composition,
                      verbose=True)

    session_signature = ",".join(image_paths + [prompt, str(seed)])
    temp_folder = os.path.join("/tmp/gradio", f"{hash(session_signature)}")
    os.makedirs(temp_folder, exist_ok=True)

    saved_image_paths = []
    # adaface_ckpt_path = "VGGface2_HQ_masks2024-11-28T13-13-20_zero3-ada/checkpoints/embeddings_gs-2000.pt"
    matches = re.search(r"\d{4}-(\d{2})-(\d{2})T(\d{2})-\d{2}-\d{2}_zero3-ada/checkpoints/embeddings_gs-(\d+).pt", args.adaface_ckpt_path)
    # Extract the checkpoint signature as 112813-2000
    ckpt_sig = f"{matches.group(1)}{matches.group(2)}{matches.group(3)}-{matches.group(4)}"

    prompt_keywords     = ['armor', 'beach', 'chef', 'dancing', 'iron man', 'jedi', 
                           'street', 'guitar', 'reading', 'running', 'superman', 'new year']
    keywords_reduction  = { 'iron man': 'ironman', 'dancing': 'dance', 
                            'running':  'run',     'reading': 'read', 'new year': 'newyear' }

    prompt_sig = None
    for keyword in prompt_keywords:
        if keyword in prompt.lower():
            prompt_sig = keywords_reduction.get(keyword, keyword)
            break

    if prompt_sig is None:
        # Use the last word of the prompt as the signature.
        prompt_sig = prompt.lower().replace(",", " ").split()[-1]

    if len(prompt_sig) > 0:
        prompt_sig = "-" + prompt_sig

    extra_save_dir = args.extra_save_dir
    if extra_save_dir is not None:
        os.makedirs(extra_save_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        filename = f"adaface{ckpt_sig}{prompt_sig}-{i+1}.png"
        if len(subj_name_sig) > 0:
            filename = f"{subj_name_sig.lower()}-{filename}"
        filepath = os.path.join(temp_folder, filename)
        # Save the image
        sample.save(filepath)  # Adjust to your image saving method
        saved_image_paths.append(filepath)

        if extra_save_dir is not None:
            extra_filepath = os.path.join(extra_save_dir, filename)
            sample.save(extra_filepath)
            print(extra_filepath)
    
    # Solution suggested by o1 to force the client browser to reload images 
    # when we change guidance scales only.
    saved_image_paths = [f"{url}?t={int(time.time())}" for url in saved_image_paths]

    return saved_image_paths

def check_prompt_and_model_type(prompt, model_style_type, adaface_encoder_cfg_scale1):
    global adaface

    model_style_type = model_style_type.lower()
    # If the base model type is changed, reload the model.
    if model_style_type != args.model_style_type or adaface_encoder_cfg_scale1 != args.adaface_encoder_cfg_scales[0]:
        if model_style_type != args.model_style_type:
            # Update base model type.
            args.model_style_type = model_style_type
            print(f"Switching to the base model type: {model_style_type}.")

            adaface = AdaFaceWrapper(pipeline_name="text2img", base_model_path=model_style_type2base_model_path[model_style_type],
                                    adaface_encoder_types=args.adaface_encoder_types,
                                    adaface_ckpt_paths=args.adaface_ckpt_path,                          
                                    adaface_encoder_cfg_scales=args.adaface_encoder_cfg_scales,
                                    enabled_encoders=args.enabled_encoders,
                                    unet_types=None, extra_unet_dirpaths=None, unet_weights_in_ensemble=None, 
                                    unet_uses_attn_lora=args.unet_uses_attn_lora,
                                    attn_lora_layer_names=args.attn_lora_layer_names,
                                    shrink_subj_attn=False,
                                    q_lora_updates_query=args.q_lora_updates_query,
                                    device='cpu')

    if adaface_encoder_cfg_scale1 != args.adaface_encoder_cfg_scales[0]:
        args.adaface_encoder_cfg_scales[0] = adaface_encoder_cfg_scale1
        adaface.set_adaface_encoder_cfg_scales(args.adaface_encoder_cfg_scales)
        print(f"Updating the scale for consistentID encoder to {adaface_encoder_cfg_scale1}.")

    if not prompt:
        raise gr.Error("Prompt cannot be blank")

### Description
title = r"""
<h1>AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization</h1>
"""

description = r"""
<b>Official demo</b> for our working paper <b>AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization</b>.<br>

❗️**What's New**❗️
- Support switching between two model styles: **Realistic** and **Anime**.
- If you just changed the model style, the first image/video generation will take extra 20~30 seconds for loading new model weight.

❗️**Tips**❗️
1. Upload one or more images of a person. If multiple faces are detected, we use the largest one. 
2. Check "Enhance Face" to highlight fine facial features.
3. If the face dominates the image, try increasing 'Weight of ID prompt in the negative prompt'.
4. AdaFace Text-to-Video: <a href="https://huggingface.co/spaces/adaface-neurips/adaface-animate" style="display: inline-flex; align-items: center;">
  AdaFace-Animate 
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces" style="margin-left: 5px;">
</a>

**TODO:**
- ControlNet integration.
"""

css = '''
.gradio-container {width: 95% !important}
.custom-gallery { 
    height: 800px !important; 
    width: 100%; 
    margin: 10px auto; 
    padding: 10px; 
    overflow-y: auto !important; 
}
'''
with gr.Blocks(css=css, theme=gr.themes.Origin()) as demo:

    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            
            # upload face image
            # img_file = gr.Image(label="Upload a photo with a face", type="filepath")
            img_files = gr.File(
                        label="Drag / Select 1 or more photos of a person's face",
                        file_types=["image"],
                        file_count="multiple"
                    )
            # When files are uploaded, show the images in the gallery and hide the file uploader.
            uploaded_files_gallery  = gr.Gallery(label="Subject images", visible=False, columns=3, rows=1, height=300)
            with gr.Column(visible=False) as clear_button_column:
                remove_and_reupload = gr.ClearButton(value="Remove and upload subject images", 
                                                     components=img_files, size="sm")

            img_files2 = gr.File(
                        label="Drag / Select 1 or more photos of another person's face (optional)",
                        file_types=["image"],
                        file_count="multiple"
                    )
            
            uploaded_files_gallery2 = gr.Gallery(label="2nd Subject images (optional)", visible=False, columns=3, rows=1, height=300)
            with gr.Column(visible=False) as clear_button_column2:
                remove_and_reupload2 = gr.ClearButton(value="Remove and upload 2nd Subject images", 
                                                      components=img_files2, size="sm")

            prompt = gr.Dropdown(label="Prompt",
                       info="Try something like 'walking on the beach'. If the face is not in focus, try checking 'enhance face'.",
                       value="portrait, highlighted hair, futuristic silver armor suit, confident stance, living room, smiling, head tilted, perfect smooth skin",
                       allow_custom_value=True,
                       filterable=False,
                       choices=[
                            "portrait, highlighted hair, futuristic silver armor suit, confident stance, living room, smiling, head tilted, perfect smooth skin",
                            "portrait, walking on the beach, sunset, orange sky",
                            "portrait, in a white apron and chef hat, garnishing a gourmet dish",
                            "portrait, dancing pose among folks in a park, waving hands",
                            "portrait, in iron man costume, the sky ablaze with hues of orange and purple",
                            "portrait, jedi wielding a lightsaber, star wars, eye level shot",
                            "portrait, night view of tokyo street, neon light",
                            "portrait, playing guitar on a boat, ocean waves",
                            "portrait, with a passion for reading, curled up with a book in a cozy nook near a window",
                            "portrait, celebrating chinese new year, fireworks",
                            "portrait, running pose in a park, eye level shot",
                            "portrait, in superman costume, the sky ablaze with hues of orange and purple"
                       ])
            
            enhance_face = gr.Checkbox(label="Enhance face", value=False, 
                                       info="Enhance the face features by prepending 'face portrait' to the prompt")
            enhance_composition = \
                gr.Checkbox(label="Enhance composition", value=False, 
                            info="Enhance the overall composition by repeating the compositional part of the prompt")

            subj_name_sig = gr.Textbox(
                label="Nickname of Subject (optional for saving images)", 
                value="",
            )
            subj_name_sig2 = gr.Textbox(
                label="Nickname of 2nd Subject (optional for saving images)", 
                value="",
                visible=False,
            )

            submit = gr.Button("Submit", variant="primary")

            negative_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, nude, naked, nsfw, topless, bare breasts",
            )

            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=1.0,
                maximum=12.0,
                step=1.0,
                value=args.guidance_scale,
            )

            adaface_encoder_cfg_scale1 = gr.Slider(
                label="Scale for consistentID encoder",
                minimum=1.0,
                maximum=12.0,
                step=1.0,
                value=args.adaface_encoder_cfg_scales[0],
                visible=False,
            )

            model_style_type = gr.Dropdown(
                label="Base Model Style Type",
                info="Switching the base model type will take 10~20 seconds to reload the model",
                value=args.model_style_type.capitalize(),
                choices=["Realistic", "Anime", "Photorealistic"],
                allow_custom_value=False,
                filterable=False,
            )

            perturb_std = gr.Slider(
                label="Std of noise added to input (may help stablize face embeddings)",
                minimum=0.0,
                maximum=0.05,
                step=0.025,
                value=0.0,
                visible=False,
            )
            num_images = gr.Slider(
                label="Number of output images",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, info="Uncheck for reproducible results")

        with gr.Column():
            out_gallery = gr.Gallery(label="Generated Images", interactive=False, columns=2, rows=4, height=800,
                                     elem_classes="custom-gallery")

        img_files.upload(fn=swap_to_gallery,  inputs=img_files,  outputs=[uploaded_files_gallery,  clear_button_column,  img_files])
        img_files2.upload(fn=swap_to_gallery, inputs=img_files2, outputs=[uploaded_files_gallery2, clear_button_column2, img_files2])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files_gallery, clear_button_column, 
                                                                    img_files, subj_name_sig])
        remove_and_reupload2.click(fn=remove_back_to_files, outputs=[uploaded_files_gallery2, clear_button_column2, 
                                                                    img_files2, subj_name_sig2])

        check_prompt_and_model_type_call_dict = {
            'fn': check_prompt_and_model_type,
            'inputs': [prompt, model_style_type, adaface_encoder_cfg_scale1],
            'outputs': None
        }
        randomize_seed_fn_call_dict = {
            'fn': randomize_seed_fn,
            'inputs': [seed, randomize_seed],
            'outputs': seed
        }
        generate_image_call_dict = {
            'fn': generate_image,
            'inputs': [img_files, img_files2, guidance_scale, perturb_std, num_images, prompt, 
                       negative_prompt, enhance_face, enhance_composition, seed, subj_name_sig],
            'outputs': [out_gallery]
        }
        submit.click(**check_prompt_and_model_type_call_dict).success(**randomize_seed_fn_call_dict).then(**generate_image_call_dict)
        subj_name_sig.submit(**check_prompt_and_model_type_call_dict).success(**randomize_seed_fn_call_dict).then(**generate_image_call_dict)
        
demo.launch(share=True, server_name=args.ip, ssl_verify=False)