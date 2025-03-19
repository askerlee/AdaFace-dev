import torch
import argparse
import re
from safetensors.torch import load_file as safetensors_load_file

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt1", type=str, required=True, help="Path to the first  model checkpoint")
parser.add_argument("--ckpt2", type=str, required=True, help="Path to the second model checkpoint")
parser.add_argument("--keypat", type=str, default=None, help="Key pattern to filter the parameters")
parser.add_argument("--is_ada", action="store_true", help="Whether the checkpoint is of an AdaFace model")
args = parser.parse_args()

def load_ckpt(ckpt_path):
    try:
        return torch.load(ckpt_path, map_location="cpu")
    except:
        return safetensors_load_file(ckpt_path)
    
def find_top_k_most_different_parameter(model1, model2, keypat, topk=100):
    name2diff = {}
    if isinstance(model1, dict):
        model1_named_params = model1.items()
    else:
        model1_named_params = model1.named_parameters()
    if isinstance(model2, dict):
        model2_named_params = model2.items()
    else:
        model2_named_params = model2.named_parameters()

    for (name1, param1), (name2, param2) in zip(model1_named_params, model2_named_params):
        if name1 != name2:
            breakpoint()
        if keypat is not None and re.search(keypat, name1) is None:
            continue

        if param1.dtype == torch.long or param1.dtype == torch.int:
            continue

        # Compute the absolute difference
        diff = (param1 - param2).abs().mean()
        
        # Compute the relative difference
        # Avoid division by zero by adding a small constant (e.g., 1e-8)
        relative_diff = diff / ((param1.abs().mean() + param2.abs().mean()) / 2 + 1e-8)
        name2diff[name1] = relative_diff

    # Sort the parameters by relative difference
    name2diff = sorted(name2diff.items(), key=lambda x: x[1], reverse=True)
    for i in range(topk):
        name, diff = name2diff[i]
        if diff == 0:
            break
        print(f"{name}: {diff:.3f}")

    print(f"{i} parameters with the highest relative difference were output")

'''
from diffusers import UNet2DConditionModel

arc2face = UNet2DConditionModel.from_pretrained(
    'models/arc2face', subfolder="arc2face", torch_dtype=torch.float32
)

unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder="unet", torch_dtype=torch.float32
)

find_top_k_most_different_parameter(arc2face, unet)
'''

model1 = load_ckpt(args.ckpt1)
model2 = load_ckpt(args.ckpt2)
if args.is_ada:
    model1 = model1['string_to_subj_basis_generator_dict']['z']
    model2 = model2['string_to_subj_basis_generator_dict']['z']
    if isinstance(model1, torch.nn.ModuleList):
        models1 = [ m.prompt2token_proj.text_model for m in model1 ]
        models2 = [ m.prompt2token_proj.text_model for m in model2 ]
    else:
        models1 = [model1.prompt2token_proj.text_model]
        models2 = [model2.prompt2token_proj.text_model]
else:
    models1 = [model1]
    models2 = [model2]

for i, (model1, model2) in enumerate(zip(models1, models2)):
    print(f"Model {i+1}")
    find_top_k_most_different_parameter(model1, model2, args.keypat)
    print()
