from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler, DDIMScheduler
import torch

from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from compel import Compel

torch.set_printoptions(precision=4, sci_mode=False)

model_sig = 'sd15'

if model_sig == 'sd21':
    model_id = "stabilityai/stable-diffusion-2-1"
elif model_sig == 'sd15':
    model_id = "runwayml/stable-diffusion-v1-5"
elif model_sig == 'turbo':
    model_id = "stabilityai/sd-turbo"
elif model_sig == 'xl':
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

def dummy(images, **kwargs):
    return images, [False] * len(images)

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
if model_sig != 'turbo':
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config #, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )

# print(type(pipe))
# pipe: diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline
pipe.to("cuda")

pipe.safety_checker = dummy

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

#image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
#prompt = "a portrait of a frog with golden hair and a crown"
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe"

prompt_embeds = compel_proc(prompt)
empty_prompt = ""
empty_prompt_embeds = compel_proc(empty_prompt)
word_weight = 1.1 ** 2
prompt_weighted_embeds = empty_prompt_embeds + (prompt_embeds - empty_prompt_embeds) * word_weight

generator = torch.manual_seed(0)
if model_sig == 'turbo':    
    image = pipe(
        prompt=prompt, num_inference_steps=1, generator=generator, guidance_scale=0.0
    ).images[0]
else:
    image = pipe(prompt_embeds=prompt_weighted_embeds, guidance_scale=5).images[0]

image.save(f"test-{model_sig}.png")
# breakpoint()
