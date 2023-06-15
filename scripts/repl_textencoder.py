import torch
from safetensors.torch import load_file as safetensors_load_file
import sys, argparse

def load_two_models(ckpt_filepath, safetensor_filepath):
    print(f"Loading model from {ckpt_filepath}")
    sd_ckpt = torch.load(ckpt_filepath, map_location="cpu")
    print(f"Loading model from {safetensor_filepath}")
    safe_state_dict = safetensors_load_file(safetensor_filepath, device="cpu")
    return sd_ckpt, safe_state_dict

parser = argparse.ArgumentParser()
parser.add_argument("--sd_ckpt", type=str, required=True, help="Path to the stable diffusion checkpoint")
parser.add_argument("--safetensor", type=str, required=True, help="Path to the 3rd party safetensor checkpoint")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

sd_ckpt, safe_state_dict = load_two_models(args.sd_ckpt, args.safetensor)
sd_state_dict = sd_ckpt["state_dict"]

repl_count = 0

for k in sd_state_dict:
    if k not in safe_state_dict:
        print(f"'{k}' not in safetensor checkpoint")
        continue
    if sd_state_dict[k].shape != safe_state_dict[k].shape:
        print(f"'{k}' shape mismatch: {sd_state_dict[k].shape} vs {safe_state_dict[k].shape}")
        breakpoint()
    if k.startswith("cond_stage_model."):
        sd_state_dict[k] = safe_state_dict[k]
        repl_count += 1

print(f"Replaced {repl_count} parameters")
if repl_count > 0:
    torch.save(sd_ckpt, args.out_ckpt)
    print(f"Saved to {args.out_ckpt}")



