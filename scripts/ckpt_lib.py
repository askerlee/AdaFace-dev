import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def load_ckpt(ckpt_filepath):
    print(f"Loading model from {ckpt_filepath}")
    if ckpt_filepath.endswith(".safetensors"):
        state_dict = safetensors_load_file(ckpt_filepath, device="cpu")
        ckpt = None
    else:
        ckpt = torch.load(ckpt_filepath, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
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

def load_two_models(base_ckpt_filepath, donor_ckpt_filepath):
    base_ckpt, base_state_dict = load_ckpt(base_ckpt_filepath)
    _, donor_state_dict = load_ckpt(donor_ckpt_filepath)
    # Other fields in sd_ckpt are also needed when saving the checkpoint. 
    # So return the whole sd_ckpt.
    return base_ckpt, base_state_dict, donor_state_dict
