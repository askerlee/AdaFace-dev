import torch

from PIL import Image
from lavis.models import load_model_and_preprocess
import os

model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device="cuda", is_eval=True)
finetuned_ckpt = "/path/to/adaface/blip-diffusion-models/checkpoint_40.pth"
model.load_checkpoint(finetuned_ckpt)
cond_subject = "dog"
tgt_subject = "dog"
text_prompt = "a marble sculpture"
# prompt = "in oil painting"

cond_subjects = [txt_preprocess["eval"](cond_subject)]
tgt_subjects = [txt_preprocess["eval"](tgt_subject)]
text_prompt = [txt_preprocess["eval"](text_prompt)]
samples = {
    "cond_images": None,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
}
text_prompt_nospace = text_prompt[0].replace(" ", "-")
num_output = 4

iter_seed = 8888
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

os.makedirs(f"samples-dbeval-blip/{text_prompt_nospace}", exist_ok=True)

for i in range(num_output):
    output = model.generate(
        samples,
        seed=iter_seed + i,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )

    savepath = f"samples-dbeval-blip/{text_prompt_nospace}/{i}.png"
    output[0].save(savepath)
    print(f"Saved {savepath}")



