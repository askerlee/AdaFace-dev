import clip
import torch
import torch.nn.functional as F
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device='cpu')
        # First put both the text encoder and visual encoder on CPU.
        # Then put the visual encoder on GPU. In effect, 
        # it puts the text encoder on CPU and the visual encoder on GPU, to avoid OOM.
        # We assume self.device is a GPU from the beginning, and won't be updated through .cuda().
        self.model.visual.to(device)

        self.clip_preprocess = clip_preprocess

        # preprocessing: (input_image + 1) / 2
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        '''
        preprocess:
        Compose(
            Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
            Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
            CenterCrop(size=(224, 224))
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        )
        [[1, 1, 1]] => [[1.93, 2.07, 2.15]]
        '''

    def tokenize(self, strings: list):
        return clip.tokenize(strings, truncate=True).to('cpu')
        
    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True, get_token_emb: bool = False) -> torch.Tensor:

        tokens = clip.tokenize(text, truncate=True).to('cpu')

        if get_token_emb:
            text_features = self.model.token_embedding(tokens).detach()
            # tokens, is_valid_text: [1, 77].
            is_valid_text = (tokens != 0) & (tokens != 49406) & (tokens != 49407)
            # text_features: [1, 77, 768] => [n, 768]
            text_features = text_features[is_valid_text]
        else:
            # tokens: [1, 77]. text_features: [1, 768].
            text_features = self.encode_text(tokens).detach()
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

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
    # txt_to_img_similarity() assumes images are tensors with mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0].
    def txt_to_img_similarity(self, text, images, reduction='mean'):
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

class LDMCLIPEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def gen_and_evaluate(self, ldm_model, ref_images, target_text, n_samples=64, n_steps=50):
        
        sampler = DDIMSampler(ldm_model)

        samples_per_batch = 8
        n_batches         = n_samples // samples_per_batch

        # generate samples
        all_samples=list()
        with torch.no_grad():
            with ldm_model.ema_scope():                
                uc = ldm_model.get_learned_conditioning(samples_per_batch * [""])

                for batch in range(n_batches):
                    c = ldm_model.get_learned_conditioning(samples_per_batch * [target_text])
                    shape = [4, 256//8, 256//8]
                    samples_ddim, _ = sampler.sample(S=n_steps,
                                                    conditioning=c,
                                                    batch_size=samples_per_batch,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=5.0,
                                                    unconditional_conditioning=uc,
                                                    eta=0.0)

                    x_samples_ddim = ldm_model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)

                    all_samples.append(x_samples_ddim)
        
        all_samples = torch.cat(all_samples, axis=0)

        sim_samples_to_img  = self.image_pairwise_similarity(ref_images, all_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), all_samples)

        return sim_samples_to_img, sim_samples_to_text


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, ref_images, target_text):

        sim_samples_to_img  = self.image_pairwise_similarity(ref_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text, gen_samples)

        return sim_samples_to_img, sim_samples_to_text
