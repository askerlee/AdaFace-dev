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
        self.model.to(self.device)
        self.model.eval()
    '''
ViTImageProcessor {
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.485,
    0.456,
    0.406
  ],
  "image_processor_type": "ViTImageProcessor",
  "image_std": [
    0.229,
    0.224,
    0.225
  ],
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 224,
    "width": 224
  }
}
    if do_normalize, then normalize the image with image_mean and image_std.
    [1, 1, 1] => [[2.249, 2.429, 2.64]]
    '''

    @torch.no_grad()
    def encode_images(self, images: ImageInput) -> torch.Tensor:
        inputs  = self.preprocess(images=images, return_tensors="pt")
        inputs  = inputs.to(self.device)
        outputs = self.model(**inputs)
        # last_hidden_states: [B, 197, 384]
        last_hidden_states = outputs.last_hidden_state

        # We only use CLS token's features, so that the spatial location of the subject will not impact matching. 
        # [B, 384]
        return last_hidden_states[:, 0]

    def get_image_features(self, img: ImageInput, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def image_pairwise_similarity(self, images1, images2, reduction='mean'):
        # [B1, 512]
        images1_features = self.get_image_features(images1)
        # [B2, 512]
        images2_features = self.get_image_features(images2)

        sim_scores = images1_features @ images2_features.T

        if reduction == 'mean':
            return sim_scores.mean()
        elif reduction == 'diag':
            assert len(images1) == len(images2), f"Number of images1 {len(images1)} != Number of images2 {len(images2)}"
            return sim_scores.diag()
        elif reduction == 'diagmean':
            assert len(images1) == len(images2), f"Number of images1 {len(images1)} != Number of images {len(images2)}"
            return sim_scores.diag().mean()
        elif reduction == 'none':
            return sim_scores
        else:
            raise NotImplementedError
                