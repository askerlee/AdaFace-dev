import argparse
from scripts.ckpt_lib import save_ckpt, load_ckpt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--in_ckpt",  type=str, required=True, help="Path to the input checkpoint")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

# Load the checkpoint
_, state_dict = load_ckpt(args.in_ckpt)

for k in state_dict:
    if state_dict[k].dtype == torch.float32:
        state_dict[k] = state_dict[k].half()

save_ckpt(None, state_dict, args.out_ckpt)
