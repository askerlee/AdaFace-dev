# import clip
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms

class CLIPEvaluator(nn.Module):
    def __init__(self, device, torch_dtype=torch.float16, 
                 clip_model_name='openai/clip-vit-base-patch32') -> None:
        super().__init__()
        self.device = device
        # Load the CLIP model and processor from Hugging Face
        self.model = CLIPModel.from_pretrained(clip_model_name, torch_dtype=torch_dtype).to(device)
        # The CLIPProcessor wraps CLIPImageProcessor and CLIPTokenizer 
        # into a single instance to both encode the text and prepare the images. 
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Define custom preprocessing pipeline to match the given transforms
        self.preprocess = transforms.Compose([
            # Diffusion decoded output images are already with mean 0 and std 1. 
            # No need to do normalization.
            # transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0]), 
            transforms.Resize(224),  # Resize to 224x224
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))  # Normalization to match CLIP's input
        ])

    def _apply(self, fn):
        super()._apply(fn)  # Call the parent _apply to handle parameters and buffers
        # A dirty hack to get the device of the model, passed from 
        # parent.model.to(self.root_device) => parent._apply(convert) => module._apply(fn)
        test_tensor = torch.zeros(1)  # Create a test tensor
        transformed_tensor = fn(test_tensor)  # Apply `fn()` to test it
        device = transformed_tensor.device  # Get the device of the transformed tensor
        # No need to reload face_app on the same device.
        if device == self.device:
            return
        self.device = device
        self.model = self.model.to(device)

    def tokenize(self, strings: list) -> torch.Tensor:
        inputs = self.processor(text=strings, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True).to(self.device)
        return inputs.input_ids

    @torch.no_grad()
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        text_features = self.model.get_text_features(input_ids=tokens)
        return text_features

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)  # Apply custom preprocessing
        # pixel_values = images.unsqueeze(0)  # Ensure the image is in the right format (batch size, channels, height, width)
        image_features = self.model.get_image_features(pixel_values=images)
        return image_features

    def get_text_features(self, text: str, norm: bool = True, get_token_emb: bool = False) -> torch.Tensor:
        tokens = self.tokenize([text])  # Tokenize the input text

        if get_token_emb:
            # Token embeddings are not exposed directly in Hugging Face's CLIP, so this part would need to be implemented differently.
            raise NotImplementedError("Token embeddings are not exposed directly in Hugging Face's CLIPModel.")
        else:
            text_features = self.encode_text(tokens).detach()

        if norm:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, images: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(images)

        if norm:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    # images1 and images2 should be tensors, not PIL images or numpy arrays,
    # as transforms.Normalize() in self.preprocess() expects tensors.
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

    def text_pairwise_similarity(self, textset1, textset2, reduction='mean', get_token_emb=False):
        textset1_features = [ self.get_text_features(t, get_token_emb=get_token_emb) for t in textset1 ]
        textset2_features = [ self.get_text_features(t, get_token_emb=get_token_emb) for t in textset2 ]
        textset1_features = torch.cat(textset1_features, dim=0)
        textset2_features = torch.cat(textset2_features, dim=0)

        sim_scores = textset1_features @ textset2_features.T
        if reduction == 'mean':
            return sim_scores.mean()
        elif reduction == 'diag':
            assert len(textset1) == len(textset2), f"Number of text set 1 {len(textset1)} != Number of text set 2 {len(textset2)}"
            return sim_scores.diag()
        elif reduction == 'diagmean':
            assert len(textset1) == len(textset2), f"Number of text set 1 {len(textset1)} != Number of text set 2 {len(textset2)}"
            return sim_scores.diag().mean()
        elif reduction == 'none':
            return sim_scores
        else:
            raise NotImplementedError        

    # The gradient is cut to prevent from going back to get_text_features(). 
    # So if text-image similarity is used as loss,
    # the text embedding from the compared input text will not be updated. 
    # The generated images and their conditioning text embeddings will be updated.
    # txt_to_img_similarity() assumes images are tensors with mean=[0, 0, 0], std=[1, 1, 1].
    def txt_to_img_similarity(self, text, images, reduction='mean'):
        # text: a string. 
        # text_features: [1, 512], gen_img_features: [4, 512]
        text_features   = self.get_text_features(text).to(self.device)
        img_features    = self.get_image_features(images)

        # sim_scores: [1, 4], example: [[0.2788, 0.2612, 0.2561, 0.2800]]
        sim_scores = text_features @ img_features.T

        if reduction == 'mean':
            return sim_scores.mean()
        elif reduction == 'diag':
            assert len(text) == len(images), f"Number of prompts {len(text)} != Number of images {len(images)}"
            return sim_scores.diag()
        elif reduction == 'diagmean':
            assert len(text) == len(images), f"Number of prompts {len(text)} != Number of images {len(images)}"
            return sim_scores.diag().mean()
        elif reduction == 'none':
            return sim_scores
        else:
            raise NotImplementedError

class CLIPImagesEvaluator(CLIPEvaluator):
    def __init__(self, device) -> None:
        super().__init__(device)

    def evaluate(self, gen_samples, ref_images, target_text):

        sim_samples_to_img  = self.image_pairwise_similarity(ref_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text, gen_samples)

        return sim_samples_to_img, sim_samples_to_text