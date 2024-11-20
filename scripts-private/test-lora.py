from torchvision.models import resnet50
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from torch import nn

model = resnet50(pretrained=True)
lora_rank  = 128
lora_alpha = 32
peft_config = LoraConfig(inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0,
                         target_modules="layer4.2.conv1")
model2 = get_peft_model(model, peft_config)

lora_modules = {}
for name, module in model2.named_modules():
    if hasattr(module, "lora_alpha"):
        # ModuleDict doesn't allow "." in the key.
        name = name.replace(".", "_")
        lora_modules[name] = module
        # lora_alpha is applied through the scaling dict.
        # So we back up the original scaling dict as scaling_.
        module.scaling_      = module.scaling
        module.zero_scaling_ = { k: 0 for k in module.scaling.keys() }

lora_modules = nn.ModuleDict(lora_modules)
lora_params = []
for par_module in lora_modules.values():
    for module in (par_module.lora_A, par_module.lora_B):
        for param in module.parameters():
            lora_params.append(param)

model.eval()
model2.eval()

input = torch.randn(1, 3, 224, 224)
a = model(input)
b = model2(input)
for param in lora_params:
    param.data = torch.randn_like(param)
c = model2(input)

for module in lora_modules.values():
    module.scaling = module.zero_scaling_

d = model2(input)

for module in lora_modules.values():
    module.scaling = module.scaling_

with model2.disable_adapter():
    e = model2(input)

breakpoint()

