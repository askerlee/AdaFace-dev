import sys
sys.path.append('./')

from adaface.adaface_wrapper import AdaFaceWrapper
import torch
import numpy as np
import random

import gradio as gr
import spaces
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--adaface_ckpt_paths', type=str, nargs="+", 
                    default=['models/adaface/subjects-celebrity2024-05-16T17-22-46_zero3-ada-30000.pt'])
parser.add_argument("--adaface_encoder_types", type=str, nargs="+", default=["arc2face"],
                    choices=["arc2face", "consistentID"], help="Type(s) of the ID2Ada prompt encoders")
# If adaface_encoder_scales is not specified, the weights will be set to all 6.0.
parser.add_argument('--adaface_encoder_scales', type=float, nargs="+", default=None,    
                    help="Weights for the ID2Ada prompt encoders")
parser.add_argument('--base_model_path', type=str, default='models/ensemble/sd15-dste8-vae.safetensors')
parser.add_argument('--extra_unet_paths', type=str, nargs="+", default=['models/ensemble/rv4-unet', 
                                                                        'models/ensemble/ar18-unet'], 
                    help="Extra paths to the checkpoints of the UNet models")
parser.add_argument('--unet_weights', type=float, nargs="+", default=[4, 2, 1], 
                    help="Weights for the UNet models")
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--ip', type=str, default="0.0.0.0")
args = parser.parse_args()

# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if args.gpu is None else f"cuda:{args.gpu}"
print(f"Device: {device}")

adaface = AdaFaceWrapper(pipeline_name="text2img", base_model_path=args.base_model_path,
                         adaface_encoder_types=args.adaface_encoder_types, 
                         adaface_ckpt_paths=args.adaface_ckpt_paths, 
                         adaface_encoder_scales=args.adaface_encoder_scales,
                         extra_unet_paths=args.extra_unet_paths, unet_weights=args.unet_weights,
                         device=device)

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
    return gr.update(visible=False), gr.update(visible=False), gr.update(value=None, visible=True)

def update_out_gallery(images):
    #rows = (len(images) + 1) // 2  # Calculate the number of rows needed
    return gr.update(height=800)

@spaces.GPU
def generate_image(image_paths, guidance_scale, adaface_id_cfg_scale, noise_std_to_input,
                   num_images, prompt, negative_prompt, enhance_face,
                   seed, progress=gr.Progress(track_tqdm=True)):

    if image_paths is None or len(image_paths) == 0:
        raise gr.Error(f"Cannot find any input face image! Please upload a face image.")
    
    if prompt is None:
        prompt = ""

    adaface_subj_embs, teacher_neg_id_prompt_embs = \
        adaface.prepare_adaface_embeddings(image_paths=image_paths, face_id_embs=None, 
                                           noise_level=noise_std_to_input, update_text_encoder=True)
    
    if adaface_subj_embs is None:
        raise gr.Error(f"Failed to detect any faces! Please try with other images")

    # Sometimes the pipeline is on CPU, although we've put it on CUDA (due to some offloading mechanism).
    # Therefore we set the generator to the correct device.
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Manual seed: {seed}")
    # Generate two images each time for the user to select from.
    noise = torch.randn(num_images, 3, 512, 512, device=device, generator=generator)
    #print(noise.abs().sum())
    # samples: A list of PIL Image instances.
    if enhance_face and "face portrait" not in prompt:
        prompt = "face portrait, " + prompt

    generator = torch.Generator(device=adaface.pipeline._execution_device).manual_seed(seed)
    samples = adaface(noise, prompt, negative_prompt, guidance_scale=guidance_scale, 
                      out_image_count=num_images, generator=generator, verbose=True)
    return samples

### Description
title = r"""
<h1>AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization</h1>
"""

description = r"""
<b>Official demo</b> for our NeurIPS 2024 submission <b>AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization</b>.<br>

❗️**Tips**❗️
1. Upload one or more images of a person. If multiple faces are detected, we use the largest one. 
2. Increase <b>AdaFace CFG Scale</b> (preferred) or <b>Guidance scale</b> and/or to highlight fine facial features.
3. AdaFace Text-to-Video: <a href="https://huggingface.co/spaces/adaface-neurips/adaface-animate" style="display: inline-flex; align-items: center;">
  AdaFace-Animate 
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces" style="margin-left: 5px;">
</a>

**TODO:**
- ControlNet integration.
"""

css = '''
.gradio-container {width: 95% !important},
.custom-gallery { 
    height: 800px; 
    width: 100%; 
    margin: 10px auto; 
    padding: 10px; 
    overflow-y: auto; 
}
'''
with gr.Blocks(css=css) as demo:

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
            uploaded_files_gallery = gr.Gallery(label="Subject images", visible=False, columns=3, rows=1, height=300)
            with gr.Column(visible=False) as clear_button_column:
                remove_and_reupload = gr.ClearButton(value="Remove and upload subject images", components=img_files, size="sm")

            prompt = gr.Dropdown(label="Prompt",
                       info="Try something like 'walking on the beach'. If the face is not in focus, try checking 'enhance_face'.",
                       value=None,
                       allow_custom_value=True,
                       filterable=False,
                       choices=[
                            "((best quality)), ((masterpiece)), ((realistic)), highlighted hair, futuristic silver armor suit, confident stance, high-resolution, living room, smiling, head tilted, perfect smooth skin",
                            "walking on the beach, sunset, orange sky",
                            "in a white apron and chef hat, garnishing a gourmet dish",
                            "dancing pose among folks in a park, waving hands",
                            "in iron man costume, the sky ablaze with hues of orange and purple",
                            "jedi wielding a lightsaber, star wars, eye level shot",
                            "playing guitar on a boat, ocean waves",
                            "with a passion for reading, curled up with a book in a cozy nook near a window",
                            "running pose in a park, eye level shot",
                            "in superman costume, the sky ablaze with hues of orange and purple"
                       ])
            
            enhance_face = gr.Checkbox(label="Enhance face", value=True, info="Enhance the face features by prepending 'face portrait' to the prompt")

            submit = gr.Button("Submit", variant="primary")

            negative_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, nude, naked, nsfw, topless, bare breasts",
            )
                        
            adaface_id_cfg_scale = gr.Slider(
                    label="AdaFace CFG Scale",
                    info="The CFG scale of the AdaFace ID embeddings (influencing fine facial features)",
                    minimum=1,
                    maximum=12.0,
                    step=1,
                    value=6.0,
                )
            
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=1.0,
                maximum=16.0,
                step=1.0,
                value=8.0,
            )

            noise_std_to_input = gr.Slider(
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
                maximum=6,
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
            out_gallery = gr.Gallery(label="Generated Images", interactive=False, columns=2, rows=2, height=800,
                                     elem_classes="custom-gallery")

        img_files.upload(fn=swap_to_gallery, inputs=img_files, outputs=[uploaded_files_gallery, clear_button_column, img_files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files_gallery, clear_button_column, img_files])

        submit.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[img_files, guidance_scale, adaface_id_cfg_scale, noise_std_to_input, num_images, 
                    prompt, negative_prompt, enhance_face, seed],
            outputs=[out_gallery]
        ).then(
            fn=update_out_gallery,
            inputs=[out_gallery],
            outputs=[out_gallery]
        )

demo.launch(share=True, server_name=args.ip, ssl_verify=False)