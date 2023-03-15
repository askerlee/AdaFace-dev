from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe
import os

model_id = "runwayml/stable-diffusion-v1-5"

# prompt = "style of <s1><s2>, baby lion"
torch.manual_seed(0)
subjects = [ "alexachung", "caradelevingne", "corgi", "donnieyen", "gabrielleunion", 
             "ianarmitage", "jaychou", "jenniferlawrence", "jiffpom", "keanureeves", 
             "lilbub", "lisa", "masatosakai", "michelleyeoh", "princessmonstertruck", 
             "ryangosling", "sandraoh", "selenagomez", "smritimandhana", "spikelee", 
             "stephenchow", "taylorswift", "timotheechalamet", "tomholland", "zendaya" ]

for subject in subjects:
    print("Subject {}:".format(subject))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        "cuda"
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if os.path.exists("exps/{}/final_lora.safetensors".format(subject)):
        print("Loading from exps/{}/final_lora.safetensors".format(subject))
    else:
        print("ERROR: No exps/{}/final_lora.safetensors".format(subject))
        continue    

    patch_pipe(
        pipe,
        "exps/{}/final_lora.safetensors".format(subject),
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    tune_lora_scale(pipe.unet, 1.00)
    tune_lora_scale(pipe.text_encoder, 1.00)

    for i in range(8):
        torch.manual_seed(i)
        prompt = "<s1>"
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
        # create folder f"samples-lora/{subject}" if it doesn't exist
        os.makedirs(f"samples-lora/{subject}", exist_ok=True)
        image.save(f"samples-lora/{subject}/{i:05d}.jpg")
