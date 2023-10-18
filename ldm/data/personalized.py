import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from torchvision.transforms import InterpolationMode
from .compositions import sample_compositions
import random
import torch
import re
import webdataset as wds


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
                 flip_p=0.5,
                 # rand_scale_range: None (disabled) or a tuple of floats
                 # that specifies the (minimum, maximum) scaling factors.
                 rand_scale_range=None,
                 set="train",
                 placeholder_string="z",
                 background_string=None,
                 # placeholder_prefix for all types of prompts. Could be a list of strings, separated by ",".
                 common_placeholder_prefix=None,   
                 # placeholder_prefix for compositional prompts. Could be a list of strings, separated by ",".
                 compos_placeholder_prefix=None,   
                 # cls token used to compute the delta loss.
                 cls_delta_token=None,  
                 # background tokens used to compute the delta loss.
                 cls_bg_delta_tokens=None,
                # num_vectors_per_token: how many vectors in each layer are allocated to model 
                # the subject. If num_vectors_per_token > 1, pad with "," in the prompts to leave
                # room for those extra vectors.
                 num_vectors_per_token=1,
                 num_vectors_per_bg_token=1,
                 center_crop=False,
                 num_compositions_per_image=1,
                 broad_class=1,
                 comp_wds_path=None,    # Path to the composition webdatabase .tar file
                 verbose=False,
                 ):

        self.data_root = data_root
        
        # image_paths and mask_paths are full paths.
        image_paths         = [os.path.join(self.data_root, file_path) for file_path in sorted(os.listdir(self.data_root))]
        self.image_paths    = list(filter(lambda x: "_mask" not in x and os.path.splitext(x)[1].lower() != '.txt', image_paths))
        fg_mask_paths       = [ os.path.splitext(x)[0] + "_mask.png" for x in self.image_paths ]
        self.fg_mask_paths  = list(map(lambda x: x if x in image_paths else None, fg_mask_paths))
        num_valid_fg_masks  = sum([ 1 if x is not None else 0 for x in self.fg_mask_paths ])
        annot_paths         = [ os.path.splitext(x)[0] + ".txt" for x in self.image_paths ]
        self.annot_paths    = list(map(lambda x: x if x in image_paths else None, annot_paths))
        num_valid_annots    = sum([ 1 if x is not None else 0 for x in self.annot_paths ])

        if verbose:
            print("{} images, {} fg masks, {} annotations found in '{}'".format( \
                  len(self.image_paths), num_valid_fg_masks, num_valid_annots, self.data_root))
            if num_valid_fg_masks > 0 and num_valid_fg_masks < len(self.image_paths):
                print("WARNING: {} fg masks are missing!".format(len(self.image_paths) - num_valid_fg_masks))
            if num_valid_annots > 0 and num_valid_annots < len(self.image_paths):
                print("WARNING: {} annotations are missing!".format(len(self.image_paths) - num_valid_annots))

        self.num_images = len(self.image_paths)
        if set == "train":
            self.is_training = True
            self._length = self.num_images * repeats
        else:
            self.is_training = False
            self._length = self.num_images 

        self.comp_wds_path = comp_wds_path
        # wds composition is enabled if there are fg masks.
        if self.is_training and (self.comp_wds_path is not None) and (num_valid_fg_masks > 0):
            self.comp_wds = wds.WebDataset(self.comp_wds_path).shuffle(100).decode("pil").to_tuple("jpg;png", "json")
            self.comp_wds_iter = iter(self.comp_wds)
            self.p_wds_comp = 0.25
            print(f"Composition webdataset {self.comp_wds_path} is enabled with prob {self.p_wds_comp}")
        else:
            self.comp_wds = None
            self.comp_wds_iter = None
            self.p_wds_comp = 0.0

        self.placeholder_string  = placeholder_string
        self.background_string   = background_string
        self.broad_class = broad_class

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.placeholder_token = None
        self.background_token  = None

        # placeholder_prefix could be a list of strings, separated by ",".
        if common_placeholder_prefix is not None:
            self.common_placeholder_prefixes   = re.split("\s*,\s*", common_placeholder_prefix)
        else:
            self.common_placeholder_prefixes   = None
        if compos_placeholder_prefix is not None:
            self.compos_placeholder_prefixes   = re.split("\s*,\s*", compos_placeholder_prefix)
        else:
            self.compos_placeholder_prefixes   = None

        if cls_delta_token is None:
            if verbose:
                print("WARNING: default cls_delta_tokens are used!")
            self.cls_delta_tokens = default_cls_delta_tokens[self.broad_class]
        else:
            self.cls_delta_tokens = [ cls_delta_token ]

        self.cls_bg_delta_tokens = cls_bg_delta_tokens

        self.num_vectors_per_token    = num_vectors_per_token
        self.num_vectors_per_bg_token = num_vectors_per_bg_token
        self.center_crop = center_crop

        self.size = size
        interpolation_scheme = "nearest"
        self.interpolation = {"nearest":  PIL.Image.NEAREST,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic":  PIL.Image.BICUBIC,
                              "lanczos":  PIL.Image.LANCZOS,
                              }[interpolation_scheme]
        
        if self.is_training:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)
            if rand_scale_range is not None:
                # If the scale factor > 1, RandomAffine will crop the scaled image to the original size.
                # If the scale factor < 1, RandomAffine will pad the scaled image to the original size.
                # Because the scaled "images" are extended with two mask channels, which contain many zeros,
                # We have to use interpolation=InterpolationMode.NEAREST, otherwise the zeros in the mask will 
                # be interpolated to image pixel values.
                # RandomAffine() doesn't change the input image size, 
                # so transforms.Resize() is redundant. Anyway, just keep it here.
                self.random_scaler = transforms.Compose([
                                        transforms.RandomAffine(degrees=0, shear=0, scale=rand_scale_range,
                                                                interpolation=InterpolationMode.NEAREST),
                                        transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
                                     ])
                print(f"{set} images will be randomly scaled in range {rand_scale_range}")
            
            if self.p_wds_comp > 0:
                # rand_scale_range is (0.7, 1.0) by default. Here we use a smaller range, 
                # i.e., more aggressive scaling.
                self.random_small_scaler = transforms.Compose([
                                            transforms.RandomAffine(degrees=0, shear=0, scale=(0.5, 0.8),
                                                                    interpolation=InterpolationMode.NEAREST),
                                            transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
                                           ])
                print(f"{set} fg will be randomly scaled to (0.5, 0.8) before overlaying to bg images")

        else:
            self.random_scaler = None
            self.flip = None

        self.num_compositions_per_image = num_compositions_per_image
        # cartoon characters are usually depicted as human-like, so is_animal is True.
        self.is_animal = (broad_class == 1 or broad_class == 2)
   
    def __len__(self):
        return self._length

    def __getitem__(self, index):        
        example = {}
        image_path = self.image_paths[index % self.num_images]
        fg_mask_path  = self.fg_mask_paths[index % self.num_images] 
        annot_path    = self.annot_paths[index % self.num_images]

        image_obj = Image.open(image_path)
        if not image_obj.mode == "RGB":
            image_obj = image_obj.convert("RGB")

        # default to score-sde preprocessing -- i don't understand what this comment means, but keep it here. 
        image       = np.array(image_obj).astype(np.uint8)
        # Convert back to Image object, so that image_obj is made sure to be uint8.
        image_obj   = Image.fromarray(image)
       
        if fg_mask_path is not None:
            # mask is 8-bit grayscale, with same size as image. E.g., image is of [1282, 1282, 3],
            # then mask is of [1282, 1282], with values True or False. 
            # After converting to "L", mask is still [1282, 1282],
            # but pixel values change from True (1) to 255.
            fg_mask_obj = Image.open(fg_mask_path).convert("L")
            has_fg_mask = True
        else:
            # This mask is created to avoid multiple None-checking later. But it's not passed to the trainer.
            # image_obj has been converted to "RGB" if it doesn't have 3 channels. 
            # So mask is of [1282, 1282, 3] as well.
            # To conform with the case where fg_mask_path is not None, we make the fg_mask 
            # pixel values to be either 0 or 255.
            fg_mask_obj = Image.fromarray(np.ones_like(image[:, :, 0]) * 255)
            has_fg_mask = False

        # image is made sure to be uint8. So fg_mask is also uint8.
        fg_mask     = np.array(fg_mask_obj).astype(np.uint8)
        # Concatenate fg_mask to the last channel of image, so that we don't need to transform fg_mask separately.
        # image_mask: (1282, 1282, 4)
        image_mask  = np.concatenate([image, fg_mask[:, :, np.newaxis]], axis=2)
        image_mask_obj  = Image.fromarray(image_mask)

        mask_fg_percent = fg_mask.astype(float).sum() / (255 * fg_mask.size)
        # print(f"mask_fg_percent: {mask_fg_percent}")

        #print(f"subj_prompt_comp: {subj_prompt_comp}")
        #print(f"cls_prompt_comp: {cls_prompt_comp}")

        if self.center_crop:        # default: False
            crop = min(image.shape[0], image.shape[1])
            h, w, = image.shape[0], image.shape[1]
            image_mask = image_mask[(h - crop) // 2:(h + crop) // 2,
                         (w - crop) // 2:(w + crop) // 2]
            
            image_mask_obj  = Image.fromarray(image_mask)

        if self.size is not None:
            # Because image_mask_obj has an extra channel of mask, which contains many zeros.
            # Using resampling schemes other than 'NEAREST' will introduce many zeros at the border. 
            # Therefore, we fix the resampling/interpolation scheme to be 'NEAREST'.
            #image_mask_obj_old = image_mask_obj
            image_mask_obj = image_mask_obj.resize((self.size, self.size), resample=self.interpolation)
            #breakpoint()

        if self.flip:
            image_mask_obj = self.flip(image_mask_obj)
        
        image_mask = np.array(image_mask_obj)
        random_scaler = self.random_scaler
        if random_scaler is not None:
            scale_p = 0.5
        else:
            scale_p = 0

        if has_fg_mask and self.p_wds_comp > 0 and random.random() < self.p_wds_comp:
            self.do_wds_comp = True
            # If do_wds_comp, and fg areas are large enough, then we do more aggressive scaling to fg,
            # so that fg won't dominate the whole image, which may help learning composition.
            random_scaler = self.random_small_scaler if mask_fg_percent > 0.1 else self.random_scaler
            scale_p = 1
        else:
            self.do_wds_comp = False

        # Do random scaling with 50% chance. Not to do it all the time, 
        # as it seems to hurt (maybe introduced domain gap between training and inference?)
        if scale_p > 0 and random.random() < scale_p:
                image_tensor = torch.tensor(image_mask).permute(2, 0, 1)
                # aug_mask doesn't have to take {0, 255}. Since some inaccuracy around the boundary
                # doesn't really matter. But fg_mask has to take {0, 255}, otherwise after scaling,
                # some foreground pixels will become 0, and when the foreground area is small, 
                # such loss is significant.
                aug_mask    = torch.ones_like(image_tensor[0:1])
                image_ext   = torch.cat([image_tensor, aug_mask], dim=0)
                # image_ext: [4, 512, 512]
                image_ext   = random_scaler(image_ext)
                # After random scaling, the valid area is only a sub-region at the center of the image.
                # NOTE: random shifting DISABLED, as it seems to hurt.
                # ??% chance to randomly roll towards right and bottom (), 
                # and ??% chance to keep the valid area at the center.
                shift_p = 0.5
                if random.random() < shift_p:
                    # count number of empty lines at the left, right, top, bottom edges of image_ext.
                    # aug_mask = image_ext[3] is uint8, so all pixels >= 0. 
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
                    # Because the image border is padded with zeros, we can simply roll() 
                    # without worrying about the border values.
                    image_ext = torch.roll(image_ext, shifts=(dy, dx), dims=(1, 2))

                # image_mask is a concatenation of image and mask, not a mask for the image.
                image_mask  = image_ext[:4].permute(1, 2, 0).numpy().astype(np.uint8)
                # aug_mask: [512, 512].
                aug_mask    = image_ext[4].numpy().astype(np.uint8)

                # Sanity check.
                if np.any(image_mask * aug_mask[:, :, np.newaxis] != image_mask):
                    breakpoint()
        else:
            # If random scaling or wds composition is enabled, then even if no scaling happens
            # or no overlay_mask is generated, we still need to put a all-1 'aug_mask' into the example.
            # 'aug_mask' has to be present in all examples, otherwise collation will encounter exceptions.
            if self.random_scaler or self.p_wds_comp > 0:
                aug_mask = np.ones_like(image_mask[:, :, 0])
            else:
                # aug_mask will not be present in any examples, so set it to None.
                aug_mask = None

        image       = image_mask[:, :, :3]
        # fg_mask is a 1-channel mask.
        fg_mask     = image_mask[:, :, 3]
        # Scale and round fg_mask to 0 or 1.
        fg_mask     = (fg_mask  / 255).astype(np.uint8)
        # No need to scale aug_mask, as it's already 0 or 1.
        if aug_mask is not None:
            aug_mask    = aug_mask.astype(np.uint8)

        example["has_fg_mask"]  = has_fg_mask
        # If no fg_mask is loaded from file. 'fg_mask' is all-1, and 'has_fg_mask' is set to False.
        # 'fg_mask' has to be present in all examples, otherwise collation will cause exceptions.
        example["fg_mask"]      = fg_mask

        if self.do_wds_comp:
            Found = False
            while not Found:
                try:
                    bg_img, bg_json = next(self.comp_wds_iter)
                except:
                    self.comp_wds_iter = iter(self.comp_wds)
                    bg_img, bg_json = next(self.comp_wds_iter)

                bg_prompt = bg_json['caption']
                bg_prompt_tokens = self.tokenizer(bg_prompt)['input_ids']
                if self.placeholder_token is None:
                    self.placeholder_token = self.tokenizer(self.placeholder_string)['input_ids'][1]
                if self.background_token is None:
                    self.background_token = self.tokenizer(self.background_string)['input_ids'][1]

                # Skip those image/prompt pairs that will cause parsing errors.
                Found = self.placeholder_token not in bg_prompt_tokens \
                          and self.background_token not in bg_prompt_tokens

            # bg_img is PIL Image -> np.array (512, 512, 3)
            bg_img = np.array(bg_img).astype(np.uint8)
            orig_h, orig_w = bg_json['original_height'], bg_json['original_width']
            min_height, min_width = self.size, self.size
            scale = min(min_height / orig_h, min_width / orig_w)
            bg_h, bg_w   = int(orig_h * scale), int(orig_w * scale)
            overlay_mask = np.ones((bg_h, bg_w), dtype=np.uint8)

            if bg_h < min_height:
                h_pad_top = int((min_height - bg_h) / 2.0)
                h_pad_bottom = min_height - bg_h - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0

            if bg_w < min_width:
                w_pad_left = int((min_width - bg_w) / 2.0)
                w_pad_right = min_width - bg_w - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0

            overlay_mask = np.pad(overlay_mask, ((h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right)), mode='constant', constant_values=0)
            assert overlay_mask.shape[0] == min_height and overlay_mask.shape[1] == min_width

            # Replace the original aug_mask to fg_mask + overlay_mask as the valid image area.
            aug_mask = np.logical_or(fg_mask, overlay_mask).astype(np.uint8)
            # Blend fg area with bg_img. fg_mask is 2D, so add 1D channel.
            image = np.where(fg_mask[:, :, None] > 0, image, bg_img)

            DEBUG_OVERLAY = False
            if DEBUG_OVERLAY:
                self.overlay_dir = "overlay-samples"
                os.makedirs(self.overlay_dir, exist_ok=True)
                overlay_sample_count = len(os.listdir(self.overlay_dir))
                overlay_sample_filepath = os.path.join(self.overlay_dir, f'{overlay_sample_count:04}.jpg')
                if os.path.exists(overlay_sample_filepath):
                    while os.path.exists(overlay_sample_filepath):
                        overlay_sample_count += 1
                        overlay_sample_filepath = os.path.join(self.overlay_dir, f'{overlay_sample_count:04}.jpg')
                Image.fromarray(image).save(overlay_sample_filepath)
                print("Saved overlay sample to {}".format(overlay_sample_filepath))

        example["aug_mask"]  = aug_mask

        # Also return the unnormalized numpy array image.
        # example["image_unnorm"]: [0, 255]
        example["image_unnorm"] = image
        # example["image"]: [0, 255] -> [-1, 1]
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        self.generate_prompts(example)
        if self.do_wds_comp:
            # common_placeholder_prefix is prepended to caption and caption_bg.
            # compos_placeholder_prefix is prepended to subj_prompt_single, subj_prompt_comps,
            # cls_prompt_single, cls_prompt_comps.
            example["caption"]              = example["caption"]    + ", " + bg_prompt
            example["caption_bg"]           = example["caption_bg"] + ", " + bg_prompt
        
        example["do_wds_comp"]          = self.do_wds_comp

        return example

    def generate_prompts(self, example):
        placeholder_string = self.placeholder_string

        background_string  = self.background_string
        cls_delta_token = random.choice(self.cls_delta_tokens)
        # If background_string is None, then cls_bg_delta_tokens should be None as well, 
        # and cls_bg_delta_token is None.
        cls_bg_delta_token = random.choice(self.cls_bg_delta_tokens) if self.cls_bg_delta_tokens is not None \
                               else self.background_string

        # If num_vectors_per_token == 3:
        # "z"    => "z, , "
        # "girl" => "girl, , "
        # Need to leave a space between multiple ",,", otherwise they are treated as one token.
        if self.num_vectors_per_token > 1:
            placeholder_string += ", " * (self.num_vectors_per_token - 1)
            cls_delta_token    += ", " * (self.num_vectors_per_token - 1)
        if self.num_vectors_per_bg_token > 1 and background_string is not None:
            background_string += ", " * (self.num_vectors_per_bg_token - 1)
            cls_bg_delta_token += ", " * (self.num_vectors_per_bg_token - 1)

        if self.common_placeholder_prefixes is not None:
            common_placeholder_prefix = random.choice(self.common_placeholder_prefixes)
            placeholder_string = common_placeholder_prefix + " " + placeholder_string
            cls_delta_token    = common_placeholder_prefix + " " + cls_delta_token
        # common_placeholder_prefixes are specified for red_cartoon.
        # compos_placeholder_prefixes are specified for fixhand.
        # They usually won't be used together. 
        # If both common_placeholder_prefixes and compos_placeholder_prefixes happen to be specified,
        # then the prompt is like "a photo of a compos_placeholder_prefix common_placeholder_prefix z ...".
        if self.compos_placeholder_prefixes is not None:
            compos_placeholder_prefix = random.choice(self.compos_placeholder_prefixes)
            compos_placeholder_string = compos_placeholder_prefix + " " + placeholder_string
            compos_cls_delta_token    = compos_placeholder_prefix + " " + cls_delta_token
        else:
            compos_placeholder_string = placeholder_string
            compos_cls_delta_token    = cls_delta_token

        template = random.choice(imagenet_templates_small)

        subj_prompt_single  = template
        cls_prompt_single   = template

        bg_suffix = " with background {}".format(background_string) if background_string is not None else ""
        # If background_string is None, then cls_bg_delta_token is None as well, thus cls_bg_suffix is "".
        cls_bg_suffix = " with background {}".format(cls_bg_delta_token) if cls_bg_delta_token is not None else ""
        # bug_suffix: " with background y". cls_bg_suffix: " with background grass/rock".
        placeholder_string_with_bg          = placeholder_string        + bg_suffix
        cls_delta_token_with_bg             = cls_delta_token           + cls_bg_suffix
        compos_placeholder_string_with_bg   = compos_placeholder_string + bg_suffix
        compos_cls_delta_token_with_bg      = compos_cls_delta_token    + cls_bg_suffix

        # "face portrait" trick for humans/animals.
        if self.broad_class == 1:
            fp_prompt_template = "a face portrait of a {}"
            subj_prompt_single_fp = fp_prompt_template
            cls_prompt_single_fp  = fp_prompt_template
            subj_prompt_comps_fp  = []
            cls_prompt_comps_fp   = []

        subj_prompt_comps = []
        cls_prompt_comps  = []
        comps_are_appearances = []

        if self.is_animal:
            subj_type = "animal" 
        else:
            subj_type = "object"

        for _ in range(self.num_compositions_per_image):
            compositions_partial, are_appearances = sample_compositions(1, subj_type, is_training=True)
            # Only sampled one instance, so compositions_partial and are_appearances are lists of length 1.
            composition_partial = compositions_partial[0]
            comp_is_appearance  = are_appearances[0]
            subj_prompt_comp    = subj_prompt_single + " " + composition_partial
            cls_prompt_comp     = cls_prompt_single  + " " + composition_partial
            subj_prompt_comps.append(subj_prompt_comp)
            cls_prompt_comps.append(cls_prompt_comp)
            comps_are_appearances.append(comp_is_appearance)

            if self.broad_class == 1:
                subj_prompt_comp_fp = subj_prompt_single_fp + " " + composition_partial
                cls_prompt_comp_fp  = cls_prompt_single_fp  + " " + composition_partial
                subj_prompt_comps_fp.append(subj_prompt_comp_fp)
                cls_prompt_comps_fp.append(cls_prompt_comp_fp)

        # NOTE: "caption" and "caption_bg" are only for image reconstruction iterations.
        # But subj_prompt_single must align with cls_prompt_single, subj_prompt_comp, cls_prompt_comp.
        # So they are different when compos_placeholder_prefix is specified.
        # Only "caption" and "caption_bg" are formatted with placeholder_string and placeholder_string_with_bg.
        # Other 4 types of prompts are formatted with compos_placeholder_string and compos_cls_delta_token.
        example["caption"]              = subj_prompt_single.format(placeholder_string)
        example["caption_bg"]           = subj_prompt_single.format(placeholder_string_with_bg)

        example["subj_prompt_single"]   = subj_prompt_single.format(compos_placeholder_string)
        example["cls_prompt_single"]    = cls_prompt_single.format(compos_cls_delta_token)
        # Will be split by "|" in the ddpm trainer.
        subj_prompt_comp = "|".join([ subj_prompt_comp.format(compos_placeholder_string) for subj_prompt_comp in subj_prompt_comps])
        cls_prompt_comp  = "|".join([ cls_prompt_comp.format(compos_cls_delta_token)     for cls_prompt_comp  in cls_prompt_comps])
        example["subj_prompt_comp"]     = subj_prompt_comp
        example["cls_prompt_comp"]      = cls_prompt_comp

        if bg_suffix:
            example["subj_prompt_single_bg"] = subj_prompt_single.format(compos_placeholder_string_with_bg)
            example["cls_prompt_single_bg"]  = cls_prompt_single.format(compos_cls_delta_token_with_bg)
            # *_comp_bg prompts are for static delta loss on training images.
            example["subj_prompt_comp_bg"]   = "|".join([ subj_prompt_comp.format(compos_placeholder_string_with_bg) for subj_prompt_comp in subj_prompt_comps])
            example["cls_prompt_comp_bg"]    = "|".join([ cls_prompt_comp.format(compos_cls_delta_token_with_bg)     for cls_prompt_comp  in cls_prompt_comps])

        if self.broad_class == 1:
            # Delta loss requires subj_prompt_single/cls_prompt_single to be token-wise aligned
            # with subj_prompt_comp/cls_prompt_comp, so we need to specify them in the dataloader as well.
            example["subj_prompt_single_fp"] = subj_prompt_single_fp.format(compos_placeholder_string)
            example["cls_prompt_single_fp"]  = cls_prompt_single_fp.format(compos_cls_delta_token)
            example["subj_prompt_comp_fp"]   = "|".join([ subj_prompt_comp_fp.format(compos_placeholder_string) for subj_prompt_comp_fp in subj_prompt_comps_fp]) 
            example["cls_prompt_comp_fp"]    = "|".join([ cls_prompt_comp_fp.format(compos_cls_delta_token)     for cls_prompt_comp_fp  in cls_prompt_comps_fp])

            if bg_suffix:
                example["subj_prompt_single_fp_bg"] = subj_prompt_single_fp.format(compos_placeholder_string_with_bg)
                example["cls_prompt_single_fp_bg"]  = cls_prompt_single_fp.format(compos_cls_delta_token_with_bg)
                # *_comp_bg prompts are for static delta loss on training images.
                example["subj_prompt_comp_fp_bg"]   = "|".join([ subj_prompt_comp_fp.format(compos_placeholder_string_with_bg) for subj_prompt_comp_fp in subj_prompt_comps_fp])
                example["cls_prompt_comp_fp_bg"]    = "|".join([ cls_prompt_comp_fp.format(compos_cls_delta_token_with_bg)     for cls_prompt_comp_fp  in cls_prompt_comps_fp])

        # comps_are_appearances is a list of length num_compositions_per_image. So in the collated batch, 
        # "comps_are_appearances" is a list of lists and needs concat.
        example["comps_are_appearances"]    = comps_are_appearances
