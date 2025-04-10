import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .compositions import sample_compositions
import random
import torch
import torch.distributed as dist
from queue import Queue
import json

# We've made sure that for each template, the number of the words is the same as the number of the tokens.
# So when we do prompt alignment, we can simply use the number of the words to align the tokens.
base_templates = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close up photo of the {}',
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
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'an illustration of the {}',
    'an illustration of one {}',
    'an illustration of the clean {}',
    'an illustration of a nice {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a depiction of the {}',
    'a depiction of one {}',
    'a depiction of the clean {}',
    'a depiction of a nice {}',
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

single_human_pat = "man|woman|person|boy|girl|child|kid|baby|adult|guy|lady|gentleman|lady|male|female|human"
single_role_pat  = "cook|chef|waiter|waitress|doctor|nurse|policeman|policewoman|fireman|firewoman|firefighter|teacher|student|professor|driver|pilot|farmer|worker|artist|painter|photographer|dancer|singer|musician|player|athlete|player|biker|cyclist|bicyclist"
plural_human_pat = "men|women|people|boys|girls|children|kids|babies|adults|guys|ladies|gentlemen|ladies|males|females|humans"
plural_role_pat  = "cooks|chefs|waiters|waitresses|doctors|nurses|policemen|policewomen|firemen|firewomen|firefighters|teachers|students|professors|drivers|pilots|farmers|workers|artists|painters|photographers|dancers|singers|musicians|players|athletes|players|bikers|cyclists|bicyclists"
animal_pat       = "cat|cats|dog|dogs"
human_animal_pat = "|".join([single_human_pat, single_role_pat, plural_human_pat, plural_role_pat, animal_pat])

def filter_non_image(x):
    exclusion_pats = [ "_mask.png", ".pt", ".json" ]
    return not any([ pat in x for pat in exclusion_pats ])

class PersonalizedBase(Dataset):
    def __init__(self,
                 # a list of folders containing subfolders, each subfolder containing images of a subject.
                 # data_roots can also be a list or a single path, each pointing to a single image file.
                 data_roots,
                 # a list of folders containing no subfolders, but only images of multiple subjects.
                 # Their cls_delta_string are all set to the default 'person', as their genders are unknown.
                 mix_subj_data_roots=None,
                 size=512,
                 repeats=100,
                 max_num_subjects_per_base_folder=-1,    # Set to -1 to load all subjects in each base folder.
                 max_num_images_per_subject=20,          # Set to -1 to load all images in each subject folder.
                 flip_p=0.5,
                 # rand_scale_range: None (disabled) or a tuple of floats
                 # that specifies the (minimum, maximum) scaling factors.
                 rand_scale_range=[0.4, 1.0],
                 set_name="train",
                 subject_string="z",
                 # cls string used to compute the delta loss.
                 # default_cls_delta_string is the same as subj init string.
                 default_cls_delta_string=None,  
                # num_vectors_per_subj_token: how many vectors in each layer are allocated to model 
                # the subject. If num_vectors_per_subj_token > 1, pad with "," in the prompts to leave
                # room for those extra vectors.
                 num_vectors_per_subj_token=1,
                 load_meta_subj2person_type_cache_path=None,
                 save_meta_subj2person_type_cache_path=None,
                 verbose=False, 
                 ):

        # If data_roots is a single string, convert it to a list of strings.
        # Otherwise, data_roots is already a list of strings.

        if data_roots is None:
            data_roots = []
        elif isinstance(data_roots, str):
            data_roots = [ data_roots ]

        if isinstance(mix_subj_data_roots, str):
            mix_subj_data_roots = [ mix_subj_data_roots ]
        # Now mix_subj_data_roots is either None or a list of folders.
                
        subj_roots = []
        # Map base_folder to a boolean value, indicating whether the subjects in the base_folder are mixed.
        base_folders_are_mix_subj = {}

        for base_folder in data_roots:
            if not os.path.isdir(base_folder):
                print(f"WARNING: {base_folder} is not a valid folder, skip")
                continue
            subfolders = [ f.path for f in os.scandir(base_folder) if f.is_dir() ]
            # If base_folder contains subfolders, then expand them.
            if len(subfolders) > 0:
                # Limit the number of subjects from each base_folder to 1000, to speed up loading.
                if max_num_subjects_per_base_folder > 0:
                    subj_roots.extend(subfolders[:max_num_subjects_per_base_folder])
                else:
                    # Load all subjects in each base folder.
                    subj_roots.extend(subfolders)
            else:
                # Remove the trailing "/" or "\" in the folder name. Otherwise basename() will return "".
                if base_folder.endswith("/") or base_folder.endswith("\\"):
                    base_folder = base_folder[:-1]
                # base_folder is a single folder containing images of a subject. No need to expand its subfolders.
                subj_roots.append(base_folder)

        for base_folder in subj_roots:
            base_folders_are_mix_subj[base_folder] = False

        if mix_subj_data_roots is not None:
            for base_folder in mix_subj_data_roots:
                # Remove the trailing "/" or "\" in the folder name. Otherwise basename() will return "".
                if base_folder.endswith("/") or base_folder.endswith("\\"):
                    base_folder = base_folder[:-1]
                    
                subj_roots.append(base_folder)
                base_folders_are_mix_subj[base_folder] = True

        # Sort subj_roots, so that the order of subjects is consistent.
        self.subj_roots = sorted(subj_roots)
        # subject_names: sorted ascendingly for subjects within the same folder.
        self.subject_names = [] #[ os.path.basename(subj_root) for subj_root in self.subj_roots ]
        # are_mix_subj_folders: a list of boolean values, indicating whether the subjects in the 
        # base_folder are mixed, indexed by subject_idx in __getitem__().
        self.are_mix_subj_folders = []
        # assert len(self.subj_roots) > 0, f"No data found in data_roots={data_roots}!"

        self.image_paths_by_subj    = []
        self.image_count_by_subj    = []
        self.fg_mask_paths_by_subj  = []
        self.caption_paths_by_subj  = []
        total_num_valid_fg_masks    = 0
        total_num_valid_captions    = 0
        if load_meta_subj2person_type_cache_path is not None:
            try:
                meta_subj2person_type = json.load(open(load_meta_subj2person_type_cache_path, "r"))
                print(f"Loaded meta_subj2person_type from {load_meta_subj2person_type_cache_path}")
            except:
                print(f"Failed to load meta_subj2person_type from {load_meta_subj2person_type_cache_path}, ignore")
                meta_subj2person_type = {}
        else:
            meta_subj2person_type = {}

        for subj_root in self.subj_roots:
            subject_name = os.path.basename(subj_root)
            base_folder_is_mix_subj = base_folders_are_mix_subj[subj_root]
            all_filenames = os.listdir(subj_root)
            # If the base folder is mixed, it contains maybe 100K+ images, so we don't sort them.
            if not base_folder_is_mix_subj:
                all_filenames = sorted(all_filenames)

            # image_paths and mask_paths are full paths.
            all_file_paths      = [os.path.join(subj_root, file_path) for file_path in all_filenames]
            all_file_path_set   = set(all_file_paths)
            image_paths         = list(filter(lambda x: filter_non_image(x) and os.path.splitext(x)[1].lower() != '.txt', all_file_paths))
            # Limit the number of images for each subject to 100, to speed up loading.
            if (not base_folder_is_mix_subj) and max_num_images_per_subject > 0:
                image_paths = image_paths[:max_num_images_per_subject]

            mix_sig = 'mix' if base_folder_is_mix_subj else 'single'
            if len(image_paths) == 0:
                print(f"No images found in {mix_sig} '{subj_root}', skip")
                continue

            fg_mask_paths       = [ os.path.splitext(x)[0] + "_mask.png" for x in image_paths ]
            fg_mask_paths       = list(map(lambda x: x if x in all_file_path_set else None, fg_mask_paths))
            num_valid_fg_masks  = sum([ 1 if x is not None else 0 for x in fg_mask_paths ])
            caption_paths       = [ os.path.splitext(x)[0] + ".txt" for x in image_paths ]
            caption_paths       = list(map(lambda x: x if x in all_file_path_set else None, caption_paths))
            num_valid_captions  = sum([ 1 if x is not None else 0 for x in caption_paths ])

            self.subject_names.append(subject_name)
            self.image_paths_by_subj.append(image_paths)
            self.fg_mask_paths_by_subj.append(fg_mask_paths)
            self.caption_paths_by_subj.append(caption_paths)
            self.image_count_by_subj.append(len(image_paths))
            self.are_mix_subj_folders.append(base_folder_is_mix_subj)

            if verbose:
                print(f"Found {len(image_paths)} images in {mix_sig} '{subj_root}'")

            # Only load metainfo.json if the person type is not in the cache.
            if subject_name not in meta_subj2person_type:
                if 'metainfo.json' in all_filenames:
                    metainfo_path = os.path.join(subj_root, 'metainfo.json')
                    metainfo = json.load(open(metainfo_path, "r"))
                    if 'person_type' in metainfo:
                        meta_subj2person_type[subject_name] = metainfo['person_type']
                    else:
                        meta_subj2person_type[subject_name] = default_cls_delta_string

            total_num_valid_fg_masks += num_valid_fg_masks
            total_num_valid_captions += num_valid_captions

            if (len(self.subj_roots) < 80 or self.are_mix_subj_folders[-1]):
                print("{} images, {} fg masks, {} captions found in '{}'".format( \
                    len(image_paths), num_valid_fg_masks, num_valid_captions, subj_root))
                if num_valid_fg_masks > 0 and num_valid_fg_masks < len(image_paths):
                    print("WARNING: {} fg masks are missing!".format(len(image_paths) - num_valid_fg_masks))
                if num_valid_captions > 0 and num_valid_captions < len(image_paths):
                    print("WARNING: {} captions are missing!".format(len(image_paths) - num_valid_captions))

        # self.image_paths, ...         are for the one-level indexing, i.e., directly indexing into a particular image.
        # self.image_paths_by_subj, ... are for the two-level indexing, i.e., first indexing into a subject, then
        # indexing into an image within that subject.
        self.image_paths   = sum(self.image_paths_by_subj, [])
        self.fg_mask_paths = sum(self.fg_mask_paths_by_subj, [])
        self.caption_paths = sum(self.caption_paths_by_subj, [])

        # During evaluation, some paths in data_roots may be image files, not folders.
        # We append them to self.image_paths.
        for base_folder in data_roots:
            if not os.path.isdir(base_folder):
                self.image_paths.append(base_folder)
                self.fg_mask_paths.append(None)
                self.caption_paths.append(None)
                # Full path is used as the subject name.
                self.subject_names.append(base_folder)

        self.num_subjects = len(self.subject_names)

        print(f"Found {len(self.image_paths)} images in {len(self.subj_roots)} folders, {total_num_valid_fg_masks} fg masks, " \
              f"{total_num_valid_captions} captions")
  
        if save_meta_subj2person_type_cache_path is not None:
            json.dump(meta_subj2person_type, open(save_meta_subj2person_type_cache_path, "w"))
            print(f"Saved meta_subj2person_type to {save_meta_subj2person_type_cache_path}")

        self.num_images = len(self.image_paths)
        self.set_name = set_name
        if set_name == "train":
            self.is_training = True
            self._length = self.num_images * repeats
        else:
            self.is_training = False
            self._length = self.num_images 

        subj2attr = {}
        # NOTE: if do_zero_shot, all subjects share the same subject placeholders and embedders.
        self.subject_strings = [ subject_string ] * self.num_subjects

        self.cls_delta_strings = []
        self.subjects_are_faces = []
        
        for subject_name in self.subject_names:
            # Set the cls_delta_string for each subject.
            if 'cls_delta_strings' in subj2attr and subject_name in subj2attr['cls_delta_strings']:
                cls_delta_string = subj2attr['cls_delta_strings'][subject_name]
            elif subject_name in meta_subj2person_type:
                cls_delta_string = meta_subj2person_type[subject_name]
            elif default_cls_delta_string is not None:
                cls_delta_string = default_cls_delta_string
            else:
                cls_delta_string = "person"

            self.cls_delta_strings.append(cls_delta_string)

            if 'are_faces' in subj2attr and subject_name in subj2attr['are_faces']:
                is_face = subj2attr['are_faces'][subject_name]
            else:
                # By default, assume all subjects are faces.
                is_face = True

            self.subjects_are_faces.append(is_face)

        self.num_vectors_per_subj_token = num_vectors_per_subj_token

        self.size = size
        interpolation_scheme = "nearest"
        self.interpolation = { "nearest":  PIL.Image.NEAREST,
                               "bilinear": PIL.Image.BILINEAR,
                               "bicubic":  PIL.Image.BICUBIC,
                               "lanczos":  PIL.Image.LANCZOS,
                              }[interpolation_scheme]
        
        if self.is_training:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)
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
            print(f"{set_name} images will be randomly scaled in range {rand_scale_range}")
        else:
            self.flip = None
            self.random_scaler = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # If is_subject_idx == True, it means index is the index of a subject, not a particular image.
        # i.e., we proceed to select an image from the image set of the subject (indexed by index).
        # Otherwise, index is the index of a global image set.
        if isinstance(index, tuple):
            index, is_subject_idx = index
        else:
            is_subject_idx = False

        example = {}
        if is_subject_idx:
            # Draw a random image from the image set corresponding to the subject index.
            # The subject index could also point to a mixed subject folder, which contains 
            # thousands of images of different subjects.
            image_paths     = self.image_paths_by_subj[index]
            for trial in range(10):
                # Draw a random image from the subject dataset indexed by index.
                image_idx       = random.randint(0, len(image_paths) - 1)
                image_path      = image_paths[image_idx]
                # Sometimes we remove some images during the training process, 
                # so we need to check if the image exists.
                if os.path.exists(image_path):
                    break
                else:
                    print(f"WARNING: {image_path} doesn't exist!")

            if not os.path.exists(image_path):
                print(f"ERROR: {image_path} still doesn't exist after 10 trials!")
                return None                
            
            fg_mask_path    = self.fg_mask_paths_by_subj[index][image_idx]
            caption_path    = self.caption_paths_by_subj[index][image_idx]
            subject_idx     = index
        else:
            image_path    = self.image_paths[index % self.num_images]
            fg_mask_path  = self.fg_mask_paths[index % self.num_images] 
            caption_path  = self.caption_paths[index % self.num_images]
            subject_idx   = 0

        image_obj = Image.open(image_path)
        if image_obj.mode != "RGB":
            image_obj = image_obj.convert("RGB")

        # default to score-sde preprocessing -- i don't understand what this comment means, but keep it here. 
        image       = np.array(image_obj).astype(np.uint8)
        # Convert back to Image object, so that image_obj is made sure to be uint8.
        image_obj   = Image.fromarray(image)
       
        if fg_mask_path is not None:
            # mask is 8-bit grayscale, with same size as image. E.g., image is of [1282, 1282, 3],
            # then mask is of [1282, 1282], with values True or False, converting to 0/1, which is wrong.
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
        fg_mask         = np.array(fg_mask_obj).astype(np.uint8)
        # Concatenate fg_mask to the last channel of image, so that we don't need to transform fg_mask separately.
        # image_mask: (1282, 1282, 4)
        image_mask      = np.concatenate([image, fg_mask[:, :, np.newaxis]], axis=2)
        image_mask_obj  = Image.fromarray(image_mask)

        #mask_fg_percent = fg_mask.astype(float).sum() / (255 * fg_mask.size)
        # print(f"mask_fg_percent: {mask_fg_percent}")

        #print(f"subj_comp_prompt: {subj_comp_prompt}")
        #print(f"cls_comp_prompt: {cls_comp_prompt}")

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

        shift_p = 1

        if self.is_training:
            # Do random scaling with 50% chance. Not to do it all the time, 
            # as it seems to hurt (maybe introduced domain gap between training and inference?)
            image_tensor = torch.tensor(image_mask).permute(2, 0, 1)
            # aug_mask doesn't have to take {0, 255}. Since some inaccuracy around the boundary
            # doesn't really matter. But fg_mask has to take {0, 255}, otherwise after scaling,
            # some foreground pixels will become 0, and when the foreground area is small, 
            # such loss is significant.
            aug_mask    = torch.ones_like(image_tensor[0:1])
            image_ext   = torch.cat([image_tensor, aug_mask], dim=0)
            # image_ext: [4, 512, 512]
            image_ext   = self.random_scaler(image_ext)
            # After random scaling, the valid area is only a sub-region at the center of the image.
            # NOTE: random shifting DISABLED, as it seems to hurt.
            # ??% chance to randomly roll towards right and bottom (), 
            # and ??% chance to keep the valid area at the center.
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

            # Sanity check to make sure image_mask is a subset of aug_mask.
            if np.any(image_mask * aug_mask[:, :, np.newaxis] != image_mask):
                breakpoint()
        else:
            aug_mask = None
            
        image       = image_mask[:, :, :3]
        # fg_mask is a 1-channel mask.
        fg_mask     = image_mask[:, :, 3]
        # Scale and round fg_mask to 0 or 1.
        fg_mask     = (fg_mask  / 255).astype(np.uint8)
        # No need to scale aug_mask, as it's already 0 or 1.
        if aug_mask is not None:
            aug_mask    = aug_mask.astype(np.uint8)

        example["image_path"]   = image_path
        example["has_fg_mask"]  = has_fg_mask
        # If no fg_mask is loaded from file. 'fg_mask' is all-1, and 'has_fg_mask' is set to False.
        # 'fg_mask' has to be present in all examples, otherwise collation will cause exceptions.
        example["fg_mask"]      = fg_mask
        example["aug_mask"]     = aug_mask

        # Also return the unnormalized numpy array image.
        # example["image_unnorm"]: [0, 255]
        example["image_unnorm"] = image
        # example["image"]: [0, 255] -> [-1, 1]
        example["image"]        = (image / 127.5 - 1.0).astype(np.float32)

        self.generate_prompts(example, subject_idx)

        if is_subject_idx:
            is_in_mix_subj_folder = self.are_mix_subj_folders[subject_idx]
        else:
            # If the image is from a global image set, then there's only a single subject.
            # So is_in_mix_subj_folder is False.
            is_in_mix_subj_folder = False
        example["is_in_mix_subj_folder"] = is_in_mix_subj_folder

        return example

    def generate_prompts(self, example, subject_idx):
        # If there are multiple subjects, then subject_string is like: 'z0', 'z1', ....
        # Otherwise, subject_string is simply 'z'.
        # subject_name is unique across different subjects, but subject_string is the same when do_zero_shot.
        subject_name        = self.subject_names[subject_idx]
        subject_string      = self.subject_strings[subject_idx]
        cls_delta_string    = self.cls_delta_strings[subject_idx]
        is_animal           = True

        example["subject_name"] = subject_name

        # If num_vectors_per_subj_token == 3:
        # "z"    => "z, , "
        # "girl" => "girl, , "
        # Need to leave a space between multiple ",,", otherwise they are treated as one token.
        if self.num_vectors_per_subj_token > 1:
            subject_string      += ", " * (self.num_vectors_per_subj_token - 1)
            cls_delta_string    += ", " * (self.num_vectors_per_subj_token - 1)

        if is_animal:
            subj_type = "animal" 
        else:
            subj_type = "object"

        compos_partials, modifiers = sample_compositions(1, subj_type)
        compos_partial      = compos_partials[0]
        modifier            = modifiers[0]
        # First modifier, then compos_partial, to align with single_mod_prompt_tmpl,
        # which ends with the modifier.
        mod_compos_partial  = modifier + ", " + compos_partial

        base_template = random.choice(base_templates)
        # "face portrait" trick for humans/animals.
        # Note in subj_comp_fp_prompt_tmpl, "face portrait" is replaced by "a portrait",
        # to avoid the face being too dominant in the image.
        # In cls_comp_fp_prompt_tmpl, "face portrait" is kept, 
        # since the face tends to be small in class images without it.
        # NOTE: We've made sure that the number of words in each base_template is the same as the number of tokens.
        # So we can use the number of tokens to calculate the number of extra tokens in the base_template 
        # than in single_fp_prompt_tmpl.
        base_fp_template = "a portrait of {}"
        base_tmpl_num_extra_tokens = len(base_template.split()) - len(base_fp_template.split())
        # Append ", " to the base_template, so that compos_partial tokens are aligned between fp and non-fp prompts.
        # We align fp prompts with non-fp prompts, because we always use fp prompts for the cls comp prompts,
        # but sometimes we use non-fp prompts for the subj comp prompts.
        cls_base_fp_template  = "a portrait of {}" + ', ' * base_tmpl_num_extra_tokens
        subj_base_fp_template = "a portrait of {}" + ', ' * base_tmpl_num_extra_tokens

        example["subj_single_prompt"]       = base_template.format(subject_string)
        example["subj_comp_prompt"]         = base_template.format(subject_string)   + ", " + compos_partial
        example["cls_single_prompt"]        = base_template.format(cls_delta_string)
        example["cls_comp_prompt"]          = base_template.format(cls_delta_string) + ", " + compos_partial

        # Prompt delta loss requires subj_single_prompt/cls_single_prompt to be token-wise aligned
        # with subj_comp_prompt/cls_comp_prompt, so we need to specify them in the dataloader as well.
        example["subj_single_prompt_fp"]    = subj_base_fp_template.format(subject_string)
        example["subj_comp_prompt_fp"]      = subj_base_fp_template.format(subject_string)   + ", " + compos_partial
        example["cls_single_prompt_fp"]     = cls_base_fp_template.format( cls_delta_string)
        example["cls_comp_prompt_fp"]       = cls_base_fp_template.format( cls_delta_string) + ", " + compos_partial

        example["subj_single_mod_prompt"]   = base_template.format(subject_string)     + ", " + modifier
        example["cls_single_mod_prompt"]    = base_template.format(cls_delta_string)   + ", " + modifier
        example["subj_comp_mod_prompt"]     = base_template.format(  subject_string)   + ", " + mod_compos_partial
        example["cls_comp_mod_prompt"]      = base_template.format(  cls_delta_string) + ", " + mod_compos_partial

        example["subj_single_mod_prompt_fp"] = subj_base_fp_template.format(subject_string)   + ", " + modifier
        example["cls_single_mod_prompt_fp"]  = cls_base_fp_template.format( cls_delta_string) + ", " + modifier
        example["subj_comp_mod_prompt_fp"]   = subj_base_fp_template.format(subject_string)   + ", " + mod_compos_partial
        example["cls_comp_mod_prompt_fp"]    = cls_base_fp_template.format( cls_delta_string) + ", " + mod_compos_partial
        
        example["mod_compos_partial_prompt"] = mod_compos_partial
        example["compos_partial_prompt"]     = compos_partial
        example["prompt_modifier"]           = modifier

# SubjectSampler randomly samples a subject/mix-subject-folder index.
# This subject index will be used by an PersonalizedBase instance to draw random images.
# num_batches: total number of batches of this training.
# In a multi-GPU training, we haven't done anything to seed each sampler differently.
# In the first few iterations, they will sample the same subjects, but 
# due to randomness in the DDPM model (?), soon the sampled subjects will be different on different GPUs.
# subject_names: a list of subject names, each name is indexed by subj_idx (for debugging).
# subjects_are_faces: a list of boolean values, indicating whether the subject(s) is a face, indexed by subj_idx.
class SubjectSampler(Sampler):
    def __init__(self, num_subjects, subject_names, subjects_are_faces, 
                 image_count_by_subj, num_batches, batch_size, skip_non_faces=True, debug=False):

        # If do_zero_shot, then skip non-faces in the dataset. Otherwise, non-face subjects (dogs, cats)
        # will disrupt the model update.
        self.subjects_are_faces = subjects_are_faces
        self.skip_non_faces     = skip_non_faces        
        self.batch_size         = batch_size
        self.num_batches        = num_batches
        self.num_subjects       = num_subjects
        self.subject_names      = subject_names
        image_count_by_subj     = np.array(image_count_by_subj)

        '''
        (Pdb) self.subj_weights[-10:]
        array([0.00001 , 0.000039, 0.000039, 0.000039, 0.020065, 0.012911,
            0.208715, 0.096761, 0.09661 , 0.096963])      
        (Pdb) self.subj_weights[-10:].sum()
        0.5321516276579649              
        '''
        
        self.subj_weights       = image_count_by_subj / image_count_by_subj.sum()

        assert self.num_subjects > 0, "FATAL: no subjects are found in the dataset!"
        self.rank = dist.get_rank()
        print(f"SubjectSampler rank {self.rank}, initialized on {self.num_subjects} subjects, "
              f"batches: {self.batch_size}*{self.num_batches}")
        self.curr_subj_idx = 0

    def __len__(self):
        return self.num_batches * self.batch_size
    
    def next_subject(self):
        while True:
            # np.random.choice() returns an array, even if the size is 1.
            subj_idx = np.random.choice(self.num_subjects, 1, p=self.subj_weights)[0]
            if not self.skip_non_faces or self.subjects_are_faces[subj_idx]:
                break

        return subj_idx

    def __iter__(self):
        for i in range(self.num_batches * self.batch_size):
            self.curr_subj_idx   = self.next_subject()
            yield self.curr_subj_idx, True
