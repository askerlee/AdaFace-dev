import argparse
from scripts.ckpt_lib import save_ckpt, load_ckpt
import torch
from diffusers import StableDiffusionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--in_ckpt",  type=str, required=True, help="Path to the input checkpoint")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

pipeline = StableDiffusionPipeline.from_single_file(args.in_ckpt, torch_dtype=torch.float16)
unet = pipeline.unet
state_dict = unet.state_dict()
save_ckpt(None, state_dict, args.out_ckpt)
