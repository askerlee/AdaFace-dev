import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .compositions import sample_compositions
import random
import torch
import re

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

# "bike", "person", "ball" are a commonly seen "object" that can appear in various contexts. 
# So it's a good template object for computing the delta loss.
# "person" is also used for animals, as their dynamic compositions are highly similar.
# "mickey", "snoopy", "pikachu" are common cartoon characters.
default_cls_delta_tokens = [ [ "bike", "person", "ball" ], 
                             [ "person", "dog" ],
                             [ "mickey", "snoopy", "pikachu" ] ]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 # rand_scaling_range: None (disabled) or a tuple of floats
                 # that specifies the (minimum, maximum) scaling factors.
                 rand_scaling_range=None,
                 set="train",
                 placeholder_token="z",
                 # cls token used to compute the delta loss.
                 cls_delta_token=None,  
                 cls_distill_token=None,
                 center_crop=False,
                 num_compositions_per_image=1,
                 broad_class=1,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in sorted(os.listdir(self.data_root))]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 
        self.placeholder_token  = placeholder_token
        if cls_delta_token is None:
            self.cls_delta_token = None
            self.use_default_cls_delta_token = True
        else:
            self.cls_delta_token = cls_delta_token
            self.use_default_cls_delta_token = False

        self.cls_distill_token = cls_distill_token

        self.center_crop = center_crop

        if set == "train":
            self.is_training = True
            self._length = self.num_images * repeats
        else:
            self.is_training = False

        self.size = size
        self.interpolation = {"linear":   PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic":  PIL.Image.BICUBIC,
                              "lanczos":  PIL.Image.LANCZOS,
                              }[interpolation]
        
        if self.is_training:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)
            if rand_scaling_range is not None:
                # If the scale factor > 1, RandomAffine will crop the scaled image to the original size.
                # If the scale factor < 1, RandomAffine will pad the scaled image to the original size.
                self.random_scaler = transforms.Compose([
                                        transforms.RandomAffine(degrees=0, shear=0, scale=rand_scaling_range),
                                        transforms.Resize(size),
                                    ])
                print(f"{set} images will be randomly scaled in range {rand_scaling_range}")
        else:
            self.random_scaler = None
            self.flip = None

        self.num_compositions_per_image = num_compositions_per_image
        self.broad_class = broad_class
        # cartoon characters are usually depicted as human-like, so is_animal is True.
        self.is_animal = (broad_class == 1 or broad_class == 2)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_path = self.image_paths[index % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.use_default_cls_delta_token:
            cls_delta_token = random.choice(default_cls_delta_tokens[self.broad_class])
        else:
            cls_delta_token = self.cls_delta_token

        if self.cls_distill_token is None:
            cls_distill_token = cls_delta_token
        else:
            cls_distill_token = self.cls_distill_token

        template = random.choice(imagenet_templates_small)
        subj_prompt_single  = template.format(placeholder_string)
        cls_prompt_single   = template.format(cls_distill_token)

        # "face portrait" trick for humans/animals.
        if self.broad_class == 1:
            fp_trick_use_diff_templates = False
            faceportrait_template = "a face portrait of a {}"
            if fp_trick_use_diff_templates:
                # subj_prompt_fp_template intentionally doesn't use the "face portrait" template.
                # Most tokens are still aligned.
                # It slightly improves compositionality at the cost of reducing face similarity.
                subj_prompt_fp_template = "a good photo of a {}"
            else:
                subj_prompt_fp_template = faceportrait_template
            subj_prompt_single_fp = subj_prompt_fp_template.format(placeholder_string)
            cls_prompt_single_fp  = faceportrait_template.format(cls_distill_token)
            subj_prompt_comps_fp  = []
            cls_prompt_comps_fp   = []

        subj_prompt_comps = []
        cls_prompt_comps  = []

        for _ in range(self.num_compositions_per_image):
            composition_partial = sample_compositions(1, self.is_animal, is_training=True)[0]
            subj_prompt_comp    = subj_prompt_single + " " + composition_partial
            cls_prompt_comp     = cls_prompt_single  + " " + composition_partial
            subj_prompt_comps.append(subj_prompt_comp)
            cls_prompt_comps.append(cls_prompt_comp)
            if self.broad_class == 1:
                subj_prompt_comp_fp = subj_prompt_single_fp + " " + composition_partial
                cls_prompt_comp_fp  = cls_prompt_single_fp  + " " + composition_partial
                subj_prompt_comps_fp.append(subj_prompt_comp_fp)
                cls_prompt_comps_fp.append(cls_prompt_comp_fp)

        # Will split by "|" in the ddpm trainer.
        subj_prompt_comp = "|".join(subj_prompt_comps)
        cls_prompt_comp  = "|".join(cls_prompt_comps)
        example["cls_prompt_single"]    = cls_prompt_single
        example["subj_prompt_comp"]     = subj_prompt_comp
        example["cls_prompt_comp"]      = cls_prompt_comp
        if self.broad_class == 1:
            subj_prompt_comp_fp = "|".join(subj_prompt_comps_fp)
            cls_prompt_comp_fp  = "|".join(cls_prompt_comps_fp)
            # Delta loss requires subj_prompt_single/cls_prompt_single to be token-wise aligned
            # with subj_prompt_comp/cls_prompt_comp, so we need to specify them in the dataloader as well.
            example["subj_prompt_single_fp"] = subj_prompt_single_fp
            example["cls_prompt_single_fp"]  = cls_prompt_single_fp
            example["subj_prompt_comp_fp"]   = subj_prompt_comp_fp
            example["cls_prompt_comp_fp"]    = cls_prompt_comp_fp

        #print(f"subj_prompt_comp: {subj_prompt_comp}")
        #print(f"cls_prompt_comp: {cls_prompt_comp}")

        example["caption"]              = subj_prompt_single

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:        # default: False
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.flip:
            image = self.flip(image)
            
        scale_p = 0.5
        # Do random scaling with 50% chance. Not to do it all the time, 
        # as it seems to hurt (maybe introduced domain gap between training and inference?)
        if self.random_scaler:
            if random.random() < scale_p:
                image_tensor = torch.tensor(np.array(image).astype(np.uint8)).permute(2, 0, 1)
                mask = torch.ones_like(image_tensor[0:1])
                image_ext = torch.cat([image_tensor, mask], dim=0)
                # image_ext: [4, 512, 512]
                image_ext = self.random_scaler(image_ext)
                # After random scaling, the valid area is only a sub-region at the center of the image.
                # NOTE: random shifting DISABLED, as it seems to hurt.
                # ??% chance to randomly roll towards right and bottom (), 
                # and ??% chance to keep the valid area at the center.
                shift_p = 0     #0.5
                if random.random() < shift_p:
                    # count number of empty lines at the left, right, top, bottom edges of image_ext.
                    # mask = image_ext[3] is uint8, so all pixels >= 0. 
                    # sum(dim=1).cumsum() == 0 means this is an empty row at the top of the image.
                    top0     = (image_ext[3].sum(dim=1).cumsum(dim=0) == 0).sum().item()
                    # flip first then cumsum(), so that cumsum() is from bottom to top.
                    bottom0  = (image_ext[3].sum(dim=1).flip(dims=(0,)).cumsum(dim=0) == 0).sum().item()
                    # sum(dim=0).cumsum() == 0 means this is an empty column at the left of the image.
                    left0    = (image_ext[3].sum(dim=0).cumsum(dim=0) == 0).sum().item()
                    # flip first then cumsum(), so that cumsum() is from right to left.
                    right0   = (image_ext[3].sum(dim=0).flip(dims=(0,)).cumsum(dim=0) == 0).sum().item()
                    # Randomly roll towards right and bottom, within the empty edges.
                    # randint() includes both end points, so max(dy) =  top0 + bottom.
                    # MG: margin at each side, i.e., after rolling, 
                    # there are still at least 12 empty lines at each side.
                    # The average scaling is (1+0.7)/2 = 0.85, i.e., 7.5% empty lines at each side.
                    # 7.5% * 512 = 38.4 pixels, so set the margin to be around 1/3 of that.
                    MG = 12     
                    if top0 + bottom0 > 2*MG:
                        dy = random.randint(0, top0 + bottom0 - 2*MG)
                        # Shift up instead (negative dy)
                        if dy > bottom0 - MG:
                            dy = -(dy - bottom0 + MG)
                    else:
                        dy = 0
                    if left0 + right0 > 2*MG:
                        dx = random.randint(0, left0 + right0 - 2*MG)
                        # Shift left instead (negative dx)
                        if dx > right0 - MG:
                            dx = -(dx - right0 + MG)
                    else:
                        dx = 0
                    image_ext = torch.roll(image_ext, shifts=(dy, dx), dims=(1, 2))

                image_tensor = image_ext[:3].permute(1, 2, 0).numpy().astype(np.uint8)
                mask = image_ext[3].numpy().astype(np.uint8)
            else:
                # No scaling happens, so the whole image is masked.
                mask = np.ones_like(np.array(image).astype(np.uint8)[:, :, 0])

            # mask[mask > 0] = 1. No need to do thresholding, as mask is uint8.
            example["mask"]  = mask

        # Also return the unnormalized numpy array image.
        image = np.array(image).astype(np.uint8)
        # example["image_unnorm"]: [0, 255]
        example["image_unnorm"] = image
        # example["image"]: [-1, 1]
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
