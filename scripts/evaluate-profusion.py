import torch
import argparse, os
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from diffusers import StableDiffusionPromptNetPipeline, StableDiffusionInpaintPipeline
from transformers import AutoProcessor, CLIPModel
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from accelerate import Accelerator
import torchvision.transforms as T
import random

BICUBIC = InterpolationMode.BICUBIC
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, DDPMScheduler
torch.manual_seed(0)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_dir",
        type=str,
        help="if specified, load prompts from this file",
        default = './prompts.txt'
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        help="load images from this file for inpainting",
        default = "./input"
    )
    parser.add_argument(
        "--inpainting_repeat",
        type=float,
        help="Number of inpainting for each image",
        default=8
    )
    parser.add_argument(
        "--gpu",
        type=float,
        help="gpu id",
        default=0
    )
    args = parser.parse_args()
    return args


# This examples was implemented on A6000

def sampling_kwargs(step=50, prompt="in Ghibli style", cfg=7.0, ref_cfg=7.0, residual=0.0, fusion=True, 
                    refine_step=5, refine_eta=1., refine_emb_scale=0.7, refine_cfg=5.0):
    kwargs = {}
    kwargs["num_inference_steps"] = step 
    # This is for simplicity, revise it if you want something else
    kwargs["prompt"] = "a holder " + prompt 
    kwargs["guidance_scale"] = cfg 
    kwargs["res_prompt_scale"] = residual
    if fusion: # if we use a reference prompt for structure information fusion
        kwargs["ref_prompt"] = "a person  " + prompt
        kwargs["guidance_scale_ref"] = ref_cfg  # also can use different scale
        kwargs["refine_step"] = refine_step  # when refine_step == 0, it means we assume conditions are independent (which leads to worse results)
        kwargs["refine_eta"] = refine_eta
        kwargs["refine_emb_scale"] = refine_emb_scale 
        kwargs["refine_guidance_scale"] = refine_cfg            
    else:
        kwargs["ref_prompt"] = None
        kwargs["guidance_scale_ref"] = 0.
        kwargs["refine_step"] = 0
    return kwargs


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def process_img(img_file, random=False):
    if type(img_file) == str :
        img_file = [img_file]
        
    input_img = []
    for img in img_file:
        image = Image.open( img).convert('RGB')
        w, h = image.size
        crop = min(w, h)
        if random:
            image = T.Resize(560, interpolation=T.InterpolationMode.BILINEAR)(image)
            image = T.RandomCrop(512)(image)
            image = T.RandomHorizontalFlip()(image)
        else:
            image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
        input_img_ = image = image.resize((512, 512), Image.LANCZOS)
        input_img.append(ToTensor()(image).unsqueeze(0))
    input_img = torch.cat(input_img).to("cuda").to(vae.dtype)
    img_latents = vae.encode(input_img * 2.0 - 1.0).latent_dist.sample()
    img_latents = img_latents * vae.config.scaling_factor

    img_4_clip = processor(input_img)
    vision_embeds = openclip.vision_model(img_4_clip, output_hidden_states=True)
    vision_hidden_states = vision_embeds.last_hidden_state
    return img_latents, vision_hidden_states, input_img_

opt = parse_arg()
processor = Compose([
    Resize(224, interpolation=BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

model_path = "./identity_small"
# model_path = "./pretrained"
use_fp16 = True
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")  # must use DDIM when refine_step > 0

if use_fp16:
    pipe = StableDiffusionPromptNetPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    weight_dtype = torch.float16
else:
    pipe = StableDiffusionPromptNetPipeline.from_pretrained(model_path, scheduler=scheduler)
    weight_dtype = torch.float32

torch.cuda.set_device(opt.gpu)
device = torch.device(f"cuda:{opt.gpu}")
# pipe.to("cuda")
pipe.to(device)
vae = pipe.vae
tokenizer = pipe.tokenizer
openclip = pipe.openclip
text_encoder = openclip.text_model
vision_encoder = openclip.vision_model
promptnet = pipe.promptnet
unet = pipe.unet

# Freeze vae and text_encoder
vae.requires_grad_(False)
openclip.requires_grad_(False)
unet.requires_grad_(False)

# print(f"Model {model_path} has been loaded")

# # pretrained model may NOT satisfy our requirements
# # especially on image which is very different from FFHQ

test_img = opt.images_dir
test_imgs = [opt.images_dir + "/" +  img for img in os.listdir(test_img)]
# prompt = "by ilya kuvshinov, clear face, cloudy sky background lush landscape illustration concept art anime key visual by makoto shinkai, sharp focus"
with open(opt.prompts_dir) as f:
    prompts = [line.rstrip() for line in f]
print(f"prompt: {prompts}")


## inference before fine-tuning

# prompt = prompts[0]
gt_latents, vision_hidden_states, input_img_ = process_img(test_imgs[0])
# print(f'the photo used is {test_imgs[0]}')
# kwargs = sampling_kwargs(prompt = prompt,
#                          step = 50,
#                          cfg = 7.0,
#                          fusion = True,
#                         )
# image = pipe(ref_image_latent=gt_latents, ref_image_embed=vision_hidden_states, **kwargs).images[0]
# image_pretrained_model = get_concat_h(input_img_, image)
# print("Results before fine-tuning")
# image_pretrained_model.show()
# image_pretrained_model.save("./results/pretrained_model.jpg")


# setting a proper mask

mask = torch.zeros((1, 3, 512, 512)).cuda()
mask[:, :, 30:460, 100:400] += 1
train_transforms = T.Compose(
    [
        T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(512),
        T.RandomHorizontalFlip(),
    ]
)
image = ToTensor()(train_transforms(Image.open(test_imgs[0]).convert('RGB'))).unsqueeze(0).cuda()
T.ToPILImage()((image*mask).squeeze()).show()
T.ToPILImage()((image*mask).squeeze()).save("./results/mask.jpg")


# # Prepare a mini dataset
# # You can also use real images instead of augmentation

if os.path.exists('./mini/'):
    print('./mini path exists')
else:
    print('./mini path does not exists')
    idx = 0
    print(f'number of image: {len(test_imgs) * opt.inpainting_repeat}')
    for i in range(len(test_imgs)):
        data_augmentation = True # use data augmentation means that we will augment the input image to be a mini dataset
        augmentation_scale = (0.6, 1.0) # scale the original image, sometimes this influences the results on some images
        num = opt.inpainting_repeat # number of augmented images
        os.makedirs('./mini', exist_ok=True)
        train_imgs = []
        if data_augmentation:
            # prepare a mini-dataset with the target single image
            train_transforms = T.Compose(
                [
                    T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                    T.CenterCrop(512),
                    T.RandomHorizontalFlip(),
                ]
            )
            inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float16,)
            inpaint_pipe.to("cuda")
            prompt = "a photo of a girl, by ilya kuvshinov, clear face, cloudy sky background lush landscape illustration concept art anime key visual by makoto shinkai, sharp focus"
            negative_prompt = "magzine, frame, tiled, repeated, multiple people, multiple faces, group of people, split frame, multiple panel, split image, watermark, boarder, diptych, triptych, nudes, big breast"
            to_show_img = input_img_
            with torch.no_grad():
                for j in range(num):
                    image = ToTensor()(train_transforms(Image.open(test_imgs[idx]).convert('RGB'))).unsqueeze(0).cuda() + 1e-5
                    image *= mask
                    image = T.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=augmentation_scale)(image)
                    mask_image = 1.0 - image.clone().to(dtype=torch.bool).to(dtype=torch.int8)
                    image = T.ToPILImage()(image.squeeze())
                    mask_image = T.ToPILImage()(mask_image.squeeze())
                    # image and mask_image should be PIL images.
                    # The mask structure is white for inpainting and black for keeping as is
                    image = inpaint_pipe(prompt=prompt, num_inference_steps=50, image=image, mask_image=mask_image, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]
                    image.show()
                    image.save(f'./mini/{j+idx}.jpg')
                    train_imgs.append(image)
                    to_show_img = get_concat_h(to_show_img, image)
            # del inpaint_pipe
            inpaint_pipe.to("cpu")   
        else:
            for j in range(num):
                train_imgs.append(input_img_)
                input_img_.save(f'./mini/{j+idx}.jpg')
        # input_img_.save(f'./mini/{idx+num}.jpg')
        idx +=1

# Load mini dataset, you can delete some poor quality images
mini_dataset = './mini'
mini_fnames = [os.path.join(r, f) for r, d, fs in os.walk(mini_dataset) for f in fs]
latents_ = []
vision_hidden_states_batch_ = []
print(mini_fnames)
for img_file in mini_fnames:
    if '.ipynb_checkpoints/' not in img_file:
        new_latents, new_vision_hidden_states, new_input_img = process_img(img_file)
        latents_.append(new_latents)
        vision_hidden_states_batch_.append(new_vision_hidden_states)
    
latents_ = torch.cat(latents_).to(gt_latents.device)
vision_hidden_states_batch_ = torch.cat(vision_hidden_states_batch_).to(gt_latents.device)
print(latents_.shape, vision_hidden_states_batch_.shape)

# finetune a model
noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
iterations = 300# 300 more iterations can lead to better results, needed iterations can be very different on different images
batch_size = 3 # 2 choose batch size based on your device, you can use a larger batch size and less iterations
finetune_unet = True 
save_path = './saved_model'
# Load mini dataset
mini_dataset = './mini'
mini_fnames = [os.path.join(r, f) for r, d, fs in os.walk(mini_dataset) for f in fs]

# assert latents_.shape[0] >= batch_size

params_to_optimize = list(promptnet.parameters())
if finetune_unet:
    for (name, param) in unet.named_parameters():
        if 'to_q' in name or 'to_k' in name or 'to_v' in name:
            param.requires_grad = True
            params_to_optimize.append(param)


promptnet.to(dtype=torch.float32)
if finetune_unet:
    unet.to(dtype=torch.float32)

optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=2e-5, 
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)

accelerator = Accelerator(mixed_precision="fp16")

# promptnet, optimizer = accelerator.prepare(promptnet, optimizer)
if finetune_unet:
    promptnet, unet, optimizer = accelerator.prepare(promptnet,unet, optimizer)
else:
    promptnet, optimizer = accelerator.prepare(promptnet, optimizer)

openclip.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
if not finetune_unet:
    unet.to(accelerator.device, dtype=weight_dtype)
else:
    unet.train()
promptnet.train()

for epoch in tqdm(range(iterations)):
    # latents_, vision_hidden_states_batch_, _ = process_img(mini_fnames, True)

    idx = torch.randperm(latents_.shape[0])
    ref_latents = latents_[idx][:batch_size]
    vision_hidden_states_batch = vision_hidden_states_batch_[idx][:batch_size]
    idx_2 = torch.randperm(latents_.shape[0])
    latents = latents_[idx_2][:batch_size]
    # print('line 316',latents.shape,vision_hidden_states_batch.shape)
    
    placeholder_pre_prompt_ids = tokenizer("a photo of ", padding=True, return_tensors="pt")["input_ids"]
    placeholder_pre_prompt_ids = placeholder_pre_prompt_ids.reshape(-1)

    noise = torch.randn_like(latents)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    bsz = latents.shape[0]

    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    pseudo_prompt, _, _ = promptnet(sample=ref_latents, timestep=timesteps, encoder_hidden_states=vision_hidden_states_batch, promptnet_cond=noisy_latents, return_dict=False, )
    
    placeholder_prompt_ids = torch.cat([placeholder_pre_prompt_ids[:-1].to(latents.device), torch.tensor([0] * pseudo_prompt.shape[1]).to(latents.device), placeholder_pre_prompt_ids[-1:].to(latents.device)], dim=-1)
    
    pseudo_hidden_states = text_encoder.embeddings(placeholder_prompt_ids)
    pseudo_hidden_states = pseudo_hidden_states.repeat((pseudo_prompt.shape[0], 1, 1))
    pseudo_hidden_states[:, len(placeholder_pre_prompt_ids) - 1: pseudo_prompt.shape[1] + len(placeholder_pre_prompt_ids) - 1, :] = pseudo_prompt 
    causal_attention_mask = text_encoder._build_causal_attention_mask(pseudo_hidden_states.shape[0], pseudo_hidden_states.shape[1], pseudo_hidden_states.dtype).to(pseudo_hidden_states.device)
    encoder_outputs = text_encoder.encoder(pseudo_hidden_states, causal_attention_mask=causal_attention_mask, output_hidden_states=True)
    encoder_hidden_states = text_encoder.final_layer_norm(encoder_outputs.hidden_states[-2]).to(dtype=latents.dtype)

    outputs_ = unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=None, mid_block_additional_residual=None, res_scale=0.0)

    loss = ((outputs_.sample.float() - target.float()) ** 2).mean((1, 2, 3)).mean()
    accelerator.backward(loss)
    # if accelerator.sync_gradients:
        # accelerator.clip_grad_norm_(promptnet.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()
    
# save the model you just trained
pipeline = StableDiffusionPromptNetPipeline.from_pretrained(
    model_path,
    openclip=openclip,
    vae=vae,
    unet=accelerator.unwrap_model(unet),
    promptnet=accelerator.unwrap_model(promptnet),
    tokenizer=tokenizer,
)
pipeline.save_pretrained(save_path)
print(f"The model has been saved to {save_path}")











#load the trained model
# save_path = './saved_model_messi'
scheduler = DDIMScheduler.from_pretrained(save_path, subfolder="scheduler")  # must use DDIM when refine_step > 0
if use_fp16:
    pipe = StableDiffusionPromptNetPipeline.from_pretrained(save_path, scheduler=scheduler, torch_dtype=torch.float16)
    weight_dtype = torch.float16
else:
    pipe = StableDiffusionPromptNetPipeline.from_pretrained(save_path, scheduler=scheduler)
    weight_dtype = torch.float32
torch.cuda.set_device(opt.gpu)
device = torch.device(f"cuda:{opt.gpu}")
# pipe.to("cuda")
pipe.to(device)
print(f"Model loaded from {save_path}")

# test the model you just trained
torch.manual_seed(0)

# prompt = "in Ghibli anime style, trending on pixiv fanbox"


# gt_latents, vision_hidden_states, input_img_ = process_img(test_imgs[0])
# kwargs = sampling_kwargs(prompt = prompt,
#                         step = 50,
#                         cfg = 5.0,
#                         fusion = False,
#                         )
# image = pipe(ref_image_latent=gt_latents, ref_image_embed=vision_hidden_states, **kwargs).images[0]
# image_finetuned_model = get_concat_h(input_img_, image)

# print("Results before fine-tuning.")
# image_pretrained_model.show()
# print("Results after fine-tuning, WITHOUT fusion sampling")
# image_finetuned_model.show()

# proposed fusion sampling
for j in range(len(prompts)):
    kwargs = sampling_kwargs(prompt = prompts[j],
                            step = 50, # sampling steps
                            cfg = 7.0, # increase this if you want more information from the input image. decrease this when you find information from image is too strong (fails to generate according to text)
                            ref_cfg = 5.0, # increase this if you want more information from the prompt 
                            fusion = True, # use fusion sampling or not
                            refine_step = 1, # when fusion=True, refine_step=0 means we consider conditions to be independent, refine_step>0 means we consider them as dependent
                            refine_emb_scale = 0.6, # increase this if you want some more information from input image, decrease if text information is not correctly generated. Normally 0.4~0.9 should work.
                            refine_cfg=7.0, # guidance for fusion step sampling
                            )
    gt_latents, vision_hidden_states, input_img_ = process_img(test_imgs[2])



    print("Results after fine-tuning, WITH fusion sampling")
    output_img = input_img_
    for k in range(4):
        image = pipe(ref_image_latent=gt_latents, ref_image_embed=vision_hidden_states, 
                    **kwargs).images[0]
        output_img = get_concat_h(output_img, image)
    output_img.save(f'result{j}.jpg')


# if __name__ == "__main__":
#     opt = parse_arg()
#     main(opt)