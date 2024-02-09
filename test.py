import torch
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor
from ldm.modules.subj_basis_generator import CLIPVisionModelWithMask

model = CLIPVisionModelWithMask.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image: PIL.Image.Image. np.array(image): (480, 640, 3).
image = Image.open(requests.get(url, stream=True).raw)
# (480, 640) -> (224, 224).
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    # [1, 257, 1280]
    image_embeds = model(**inputs, output_hidden_states=True).hidden_states[-2]
