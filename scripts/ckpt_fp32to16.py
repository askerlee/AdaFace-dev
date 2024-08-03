import argparse
from scripts.ckpt_lib import save_ckpt, load_ckpt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--in_ckpt",  type=str, required=True, help="Path to the input checkpoint")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

# Load the checkpoint
_, state_dict = load_ckpt(args.in_ckpt)
state_dict2 = {}

for k in state_dict:
    # Skip ema weights
    if k.startswith("model_ema."):
        continue    
    if state_dict[k].dtype == torch.float32:
        state_dict2[k] = state_dict[k].half()
    else:
        state_dict2[k] = state_dict[k]

save_ckpt(None, state_dict2, args.out_ckpt)
