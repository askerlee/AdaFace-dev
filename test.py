from diffusers import DiffusionPipeline

model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)
ldm.to("cuda")
# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of an animal not a squirrel eating a burger"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

# save images
for idx, image in enumerate(images):
    image.save(f"not-squirrel-{idx}.png")
    