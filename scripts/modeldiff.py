import torch
import argparse

def find_top_k_most_different_parameter(model1, model2, topk=100):
    name2diff = {}

    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            breakpoint()
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
        print(f"{name}: {diff:.3f}")


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

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt1", type=str, required=True, help="Path to the first  model checkpoint")
parser.add_argument("--ckpt2", type=str, required=True, help="Path to the second model checkpoint")
parser.add_argument("--is_ada", action="store_true", help="Whether the checkpoint is of an AdaFace model")
args = parser.parse_args()

model1 = torch.load(args.ckpt1)
model2 = torch.load(args.ckpt2)
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
    find_top_k_most_different_parameter(model1, model2)
