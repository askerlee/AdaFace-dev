from torchvision.models import resnet50
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from torch import nn
from peft.tuners.lora import LoraLayer, Conv2d
import copy

orig_model = resnet50(pretrained=True)
model = copy.deepcopy(orig_model)

lora_rank  = 128
lora_alpha = 32
peft_config = LoraConfig(inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0,
                         target_modules="layer4.2.conv1")
model2 = get_peft_model(orig_model, peft_config)

lora_modules = {}

layer4_2_conv1 = Conv2d(model.layer4[2].conv1, adapter_name='default', r=128, lora_alpha=lora_alpha)
model.layer4[2].conv1 = layer4_2_conv1
lora_modules['direct'] = layer4_2_conv1
layer4_2_conv1.scaling_     = layer4_2_conv1.scaling
layer4_2_conv1.zero_scaling_ = { k: 0 for k in layer4_2_conv1.scaling.keys() }

for name, module in model2.named_modules():
    if isinstance(module, LoraLayer):
        # ModuleDict doesn't allow "." in the key.
        name = name.replace(".", "_")
        lora_modules[name] = module
        # lora_alpha is applied through the scaling dict.
        # So we back up the original scaling dict as scaling_.
        module.scaling_      = module.scaling
        module.zero_scaling_ = { k: 0 for k in module.scaling.keys() }
# base_model_model_layer4_2_conv1 <= model.layer4[2].conv1        
print(f"{len(lora_modules)} LoraLayers: {list(lora_modules.keys())}")

lora_modules = nn.ModuleDict(lora_modules)
lora_params = []
for par_module in lora_modules.values():
    for module in (par_module.lora_A, par_module.lora_B):
        for param in module.parameters():
            lora_params.append(param)

orig_model.eval()
model.eval()
model2.eval()

input = torch.randn(1, 3, 224, 224)
a0 = orig_model(input)
a = model(input)
b = model2(input)
for param in lora_params:
    param.data = torch.randn_like(param)
c = model2(input)
d = model(input)

for module in lora_modules.values():
    module.scaling = module.zero_scaling_

c2 = model2(input)
d2 = model(input)
breakpoint()

