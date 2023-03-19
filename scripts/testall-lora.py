from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe, image_grid
import os
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_comp", dest="gen_single", action="store_false", 
                        help="Generate composition images (default: False, generating single images)")
    parser.add_argument("--guidance_scale", type=float, default=-1, help="guidance scale")
    parser.add_argument("--model", dest="model_suffix", type=str, default="", 
                        help="model suffix")
    parser.add_argument("--num_samples", type=int, default=8, help="number of samples")
    # composition case file path
    parser.add_argument("--case_file", type=str, default="scripts/compositon-cases.sh", 
                        help="case script file")
    # range of subjects to generate
    parser.add_argument("--range", type=str, default=None, 
                        help="Range of subjects to generate (Index starts from 1 and is inclusive, e.g., 1-25)")
    args = parser.parse_args()
    return args

def dummy_nsfw_filter(images, **kwargs):
    return images, False

def parse_case_file(case_file_path):
    cases = []
    with open(case_file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0 and line[0] != "#" and re.search(r"^set [-la\s]*cases", line) is not None ]
        for line in lines:
            # Clear all previous cases. This is to simulate the behavior of a shell script.
            if re.match(r"^set cases", line) is not None:
                cases = []
                continue
            # set -a cases "jiffpom | wearing superman's uniform, cute face | jiffpom-superman | pom dog, instagram"
            mat = re.search(r"^set [-la]+\s+cases\s+[\'\"]([^\'\"]+)[\'\"]", line)
            if mat is not None:
                case = mat.group(1).split(" | ")
                cases.append(case)
            else:
                breakpoint()
    
    return cases

args = parse_args()
model_id = "runwayml/stable-diffusion-v1-5"
args.ckpt_folder = "exps" + args.model_suffix
args.out_folder = "samples-lora" + args.model_suffix

# prompt = "style of <s1><s2>, baby lion"
torch.manual_seed(0)

if args.guidance_scale == -1:
    args.guidance_scale = 5 if args.gen_single else 10

if args.gen_single:
    #                  1              2             3          4             5
    subjects = [ "alexachung", "caradelevingne", "corgi", "donnieyen", "gabrielleunion", 
    #                  6              7             8          9             10
                 "iainarmitage", "jaychou", "jenniferlawrence", "jiffpom", "keanureeves", 
    #                  11             12            13         14            15
                 "lilbub", "lisa", "masatosakai", "michelleyeoh", "princessmonstertruck", 
    #                  16             17            18         19            20
                 "ryangosling", "sandraoh", "selenagomez", "smritimandhana", "spikelee", 
    #                  21             22            23         24            25
                 "stephenchow", "taylorswift", "timotheechalamet", "tomholland", "zendaya" ]

    if args.range is not None:
        range_strs = args.range.split("-")
        # low is 1-indexed, converted to 0-indexed by -1.
        # high is inclusive, converted to exclusive without adding offset.
        low, high = int(range_strs[0]) - 1, int(range_strs[1])
        subjects = subjects[low:high]

    # For plain subject generation.
    prompts = [ "" for i in range(len(subjects))]
    subj_folders = list(subjects)

else:
    # For composition case generation.
    cases = parse_case_file(args.case_file)
    subjects = []
    prompts = []
    subj_folders = []
    for case in cases:
        subjects.append(case[0])
        prompts.append(case[1])
        subj_folders.append(case[2])

for subject, prompt, subj_folder in list(zip(subjects, prompts, subj_folders)):
    print("Subject {}:".format(subject))
    ckpt_path = f"{args.ckpt_folder}/{subject}/final_lora.safetensors"
    if os.path.exists(ckpt_path):
        print("Loading from", ckpt_path)
    else:
        print("ERROR: No ", ckpt_path)
        continue    

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        "cuda"
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # Disable NSFW check
    pipe.safety_checker = dummy_nsfw_filter

    patch_pipe(
        pipe, ckpt_path, patch_text=True, patch_ti=True, patch_unet=True,
    )

    tune_lora_scale(pipe.unet, 1.00)
    tune_lora_scale(pipe.text_encoder, 1.00)

    imgs = []
    # create folder f"samples-lora/{subject}" if it doesn't exist
    os.makedirs(f"{args.out_folder}/{subj_folder}", exist_ok=True)

    if "{}" in prompt:
        prompt2 = prompt.format("<s1>")
    else:
        prompt2 = "<s1> " + prompt
        prompt2 = prompt2.strip()
    print("Prompt:", prompt2)

    for i in range(8):
        torch.manual_seed(i)
        image = pipe(prompt2, num_inference_steps=50, guidance_scale=args.guidance_scale).images[0]
        imgs.append(image)
        image.save(f"{args.out_folder}/{subj_folder}/{i:05d}.jpg")

    imgs = image_grid(imgs, 1, 8)
    prompt_sig = re.sub("<|>", "", prompt2).replace(" ", "-")[:40]
    if len(prompt_sig) == 0:
        prompt_sig = "plain"

    filepath = f"{args.out_folder}/{subject}-lora-{prompt_sig}.jpg"
    suffix = 1
    while os.path.exists(filepath):
        suffix += 1
        filepath = f"{args.out_folder}/{subject}-lora-{prompt_sig}-{suffix}.jpg"

    imgs.save(filepath)
    print("Saved to {}".format(filepath))

