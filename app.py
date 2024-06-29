import sys
sys.path.append('./')

from adaface.adaface_wrapper import AdaFaceWrapper
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random

import gradio as gr
import spaces
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--adaface_ckpt_path', type=str, 
                    default='models/adaface/subjects-celebrity2024-05-16T17-22-46_zero3-ada-30000.pt')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--ip', type=str, default="0.0.0.0")
args = parser.parse_args()

# global variable
MAX_SEED = np.iinfo(np.int32).max
if torch.cuda.is_available():
    device = "cuda" if args.gpu is None else f"cuda:{args.gpu}"
else:
    device = "cpu"
dtype = torch.float16

# base_model_path is only used for initialization, not really used in the inference.
adaface = AdaFaceWrapper(pipeline_name="text2img", base_model_path="models/sar/sar.safetensors",
                         adaface_ckpt_path=args.adaface_ckpt_path, device=device)

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
    return gr.update(height=600)

@spaces.GPU
def generate_image(image_paths, guidance_scale, adaface_id_cfg_scale,
                   num_images, prompt, negative_prompt, seed, progress=gr.Progress(track_tqdm=True)):

    if image_paths is None or len(image_paths) == 0:
        raise gr.Error(f"Cannot find any input face image! Please upload a face image.")
    
    if prompt is None:
        prompt = ""

    adaface_subj_embs = \
        adaface.generate_adaface_embeddings(image_folder=None, image_paths=image_paths,
                                            out_id_embs_scale=adaface_id_cfg_scale, update_text_encoder=True)
    
    if adaface_subj_embs is None:
        raise gr.Error(f"Failed to detect any faces! Please try with other images")

    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Manual seed: {seed}")
    # Generate two images each time for the user to select from.
    noise = torch.randn(num_images, 3, 512, 512, device=device, generator=generator)
    #print(noise.abs().sum())
    # samples: A list of PIL Image instances.
    samples = adaface(noise, prompt, negative_prompt, guidance_scale=guidance_scale, out_image_count=num_images, generator=generator, verbose=True)
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

**TODO**
- ControlNet integration.
"""

css = '''
.gradio-container {width: 85% !important}
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
                       info="Try something like 'man/woman walking on the beach'. If the face is not in focus, try adding 'face portrait of' at the beginning.",
                       value=None,
                       allow_custom_value=True,
                       filterable=False,
                       choices=[
                            "woman ((best quality)), ((masterpiece)), ((realistic)), long highlighted hair, futuristic silver armor suit, confident stance, high-resolution, living room, smiling, head tilted, perfect smooth skin",
                            "woman walking on the beach, sunset, orange sky",
                            "woman in a white apron and chef hat, garnishing a gourmet dish, full body view, long shot",
                            "woman dancing pose among folks in a park, waving hands",
                            "woman in iron man costume flying pose, the sky ablaze with hues of orange and purple, full body view, long shot",
                            "woman jedi wielding a lightsaber, star wars, full body view, eye level shot",
                            "woman playing guitar on a boat, ocean waves",
                            "woman with a passion for reading, curled up with a book in a cozy nook near a window",
                            "woman running pose in a park, eye level shot",
                            "woman in superman costume flying pose, the sky ablaze with hues of orange and purple, full body view, long shot"
                       ])
            
            submit = gr.Button("Submit", variant="primary")

            negative_prompt = gr.Textbox(
                label="Negative Prompt", 
                value="flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, nude, naked, nsfw, topless, bare breasts",
            )
                        
            adaface_id_cfg_scale = gr.Slider(
                    label="AdaFace CFG Scale",
                    info="The CFG scale of the AdaFace ID embeddings (influencing fine facial features)",
                    minimum=0.5,
                    maximum=8.0,
                    step=0.5,
                    value=4.0,
                )
            
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.5,
                maximum=8.0,
                step=0.5,
                value=4.0,
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
            out_gallery = gr.Gallery(label="Generated Images", columns=2, rows=2, height=600)

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
            inputs=[img_files, guidance_scale, adaface_id_cfg_scale, num_images, prompt, negative_prompt, seed],
            outputs=[out_gallery]
        ).then(
            fn=update_out_gallery,
            inputs=[out_gallery],
            outputs=[out_gallery]
        )

demo.launch(share=True, server_name=args.ip, ssl_verify=False)