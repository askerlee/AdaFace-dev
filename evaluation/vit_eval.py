import clip
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from transformers.image_utils  import ImageInput

class ViTEvaluator(object):
    def __init__(self, device, vit_model='facebook/dino-vits16') -> None:
        self.device = device
        self.vit_model = vit_model
        self.model = ViTModel.from_pretrained(self.vit_model)
        self.preprocess = ViTFeatureExtractor.from_pretrained(self.vit_model)

    @torch.no_grad()
    def encode_images(self, images: ImageInput, cls_only: bool = True) -> torch.Tensor:
        # If not cls_only, spatial location of the subject will impact matching similarity, 
        # which seems unreasonable? So by default, we only use the first token (CLS) as the image features.
        inputs  = self.preprocess(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        # last_hidden_states: [B, 197, 384]
        last_hidden_states = outputs.last_hidden_state

        if cls_only:
            # [B, 384]
            return last_hidden_states[:, 0]
        else:
            # img_to_img_similarity() requires [B, D]-shaped features. 
            # TODO: flatten the image features to 1D. But this seems unreasonable. 
            # So just leave the buggy code here.
            # [B, 197, 384]
            return last_hidden_states[:]

    def get_image_features(self, img: ImageInput, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        # [B1, 384]
        src_img_features = self.get_image_features(src_images)
        # [B2, 384]
        gen_img_features = self.get_image_features(generated_images)
        # ([B1, B2]).mean()
        return (src_img_features @ gen_img_features.T).mean()
