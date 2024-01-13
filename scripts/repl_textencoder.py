import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
import sys, argparse

def load_ckpt(ckpt_filepath):
    print(f"Loading model from {ckpt_filepath}")
    if ckpt_filepath.endswith(".safetensors"):
        state_dict = safetensors_load_file(ckpt_filepath, device="cpu")
        ckpt = None
    else:
        ckpt = torch.load(ckpt_filepath, map_location="cpu")
        state_dict = ckpt["state_dict"]
    return ckpt, state_dict

def save_ckpt(ckpt, ckpt_state_dict, ckpt_filepath):
    if ckpt_filepath.endswith(".safetensors"):
        safetensors_save_file(ckpt_state_dict, ckpt_filepath)
    else:
        if ckpt is not None:
            torch.save(ckpt, ckpt_filepath)
        else:
            torch.save(ckpt_state_dict, ckpt_filepath)

    print(f"Saved to {ckpt_filepath}")

def load_two_models(base_ckpt_filepath, te_ckpt_filepath):
    base_ckpt, base_state_dict = load_ckpt(base_ckpt_filepath)
    _, te_state_dict = load_ckpt(te_ckpt_filepath)
    # Other fields in sd_ckpt are also needed when saving the checkpoint. 
    # So return the whole sd_ckpt.
    return base_ckpt, base_state_dict, te_state_dict

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", type=str, required=True, help="Path to the base checkpoint")
parser.add_argument("--te_ckpt", type=str, required=True, help="Path to the checkpoint providing text encoder")
parser.add_argument("--out_ckpt", type=str, required=True, help="Path to the output checkpoint")
args = parser.parse_args()

base_ckpt, base_state_dict, te_state_dict = load_two_models(args.base_ckpt, args.te_ckpt)
# base_state_dict = sd_ckpt["state_dict"]

repl_count = 0

for k in base_state_dict:
    if k.startswith("cond_stage_model."):
        if k not in te_state_dict:
            print(f"!!!! '{k}' not in TE checkpoint")
            continue
        if base_state_dict[k].shape != te_state_dict[k].shape:
            print(f"!!!! '{k}' shape mismatch: {base_state_dict[k].shape} vs {te_state_dict[k].shape} !!!!")
            continue
        print(k)
        base_state_dict[k] = te_state_dict[k]
        repl_count += 1

if repl_count > 0:
    print(f"{repl_count} parameters replaced")
    save_ckpt(base_ckpt, base_state_dict, args.out_ckpt)
else:
    print("ERROR: No parameter replaced")
