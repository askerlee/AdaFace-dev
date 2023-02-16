import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .compositions import sample_compositions
import random
import torch

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

# Should never use per_image_tokens.
class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 # min_rand_scaling: -1 (disabled) or a float 
                 # that specifies the minimum scaling factor.
                 min_rand_scaling=-1,
                 set="train",
                 placeholder_token="*",
                 cls_token="person",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 
        self.placeholder_token = placeholder_token
        self.cls_token = cls_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        if min_rand_scaling > 0:
            # RandomResizedCrop only enlarges (a crop of) the image, so we use RandomAffine instead.
            self.random_scaler = transforms.Compose([
                                    transforms.RandomAffine(degrees=0, shear=0, scale=(min_rand_scaling, 1)),
                                    transforms.Resize(size),
                                ])
            print(f"{set} images will be randomly scaled with range ({min_rand_scaling}, 1)")
        else:
            self.random_scaler = None

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        # Should never use per_image_tokens. So we don't modify the corresponding branch.
        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            subj_prompt_single = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            template = random.choice(imagenet_templates_small)
            subj_prompt_single  = template.format(placeholder_string)
            cls_prompt_single   = template.format(self.cls_token)
            composition_partial = sample_compositions(1)[0]
            subj_prompt_comp    = subj_prompt_single + " " + composition_partial
            cls_prompt_comp     = cls_prompt_single  + " " + composition_partial

        example["caption"]              = subj_prompt_single
        example["subj_prompt_comp"]     = subj_prompt_comp
        example["cls_prompt_comp"]      = cls_prompt_comp
        example["cls_prompt_single"]    = cls_prompt_single

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        if self.random_scaler:
            image_tensor = torch.tensor(np.array(image).astype(np.uint8)).permute(2, 0, 1)
            mask = torch.ones_like(image_tensor[0:1])
            image_ext = torch.cat([image_tensor, mask], dim=0)
            image_ext = self.random_scaler(image_ext)
            image_tensor = image_ext[:3].permute(1, 2, 0).numpy().astype(np.uint8)
            mask = image_ext[3].numpy().astype(np.uint8)
            # mask[mask > 0] = 1. No need to do thresholding, as mask is uint8.
            example["mask"]  = mask

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    