import torch
from diffusers import UNet2DConditionModel

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

arc2face = UNet2DConditionModel.from_pretrained(
    'arc2face/models', subfolder="arc2face", torch_dtype=torch.float32
)

unet = UNet2DConditionModel.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder="unet", torch_dtype=torch.float32
)

find_top_k_most_different_parameter(arc2face, unet)
