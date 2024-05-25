from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch
import argparse
parser = argparse.ArgumentParser()
# --cond_image
parser.add_argument("--cond_image", type=str, help="path to the conditional image")
# --text_prompt_input
parser.add_argument("--text_prompt_input", type=str, help="text prompt input")
# --cond_subject
parser.add_argument("--cond_subject", type=str, help="conditional subject type")
# --out_image
parser.add_argument("--out_image", type=str, help="path to the output image")
args = parser.parse_args()

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16
).to("cuda")

cond_subject = args.cond_subject
tgt_subject = args.cond_subject
text_prompt_input = args.text_prompt_input
cond_image = load_image(args.cond_image)

iter_seed = 88888
guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

output = blip_diffusion_pipe(
    text_prompt_input,
    cond_image,
    cond_subject,
    tgt_subject,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
).images
output[0].save(args.out_image)
print(f"Saved image to {args.out_image}")
