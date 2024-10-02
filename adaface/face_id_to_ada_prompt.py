import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor
from .arc2face_models import CLIPTextModelWrapper
from ConsistentID.lib.pipeline_ConsistentID import ConsistentIDPipeline 
from .util import perturb_tensor, pad_image_obj_to_square, \
                  calc_stats, patch_clip_image_encoder_with_mask, CLIPVisionModelWithMask
from adaface.subj_basis_generator import SubjBasisGenerator
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import os
from omegaconf.listconfig import ListConfig

# adaface_encoder_types can be a list of one or more encoder types.
# adaface_ckpt_paths can be one or a list of ckpt paths.
# adaface_encoder_cfg_scales is None, or a list of scales for the adaface encoder types.
def create_id2ada_prompt_encoder(adaface_encoder_types, adaface_ckpt_paths=None, 
                                 adaface_encoder_cfg_scales=None, enabled_encoders=None,
                                 *args, **kwargs):
    if len(adaface_encoder_types) == 1:
        adaface_encoder_type = adaface_encoder_types[0]
        adaface_ckpt_path = adaface_ckpt_paths[0] if adaface_ckpt_paths is not None else None
        if adaface_encoder_type == 'arc2face':
            id2ada_prompt_encoder = \
                Arc2Face_ID2AdaPrompt(adaface_ckpt_path=adaface_ckpt_path, 
                                    *args, **kwargs)
        elif adaface_encoder_type == 'consistentID':
            id2ada_prompt_encoder = \
                ConsistentID_ID2AdaPrompt(pipe=None,
                                        adaface_ckpt_path=adaface_ckpt_path, 
                                        *args, **kwargs)
    else:
        id2ada_prompt_encoder = Joint_FaceID2AdaPrompt(adaface_encoder_types, adaface_ckpt_paths, 
                                                       adaface_encoder_cfg_scales, enabled_encoders,
                                                       *args, **kwargs)
    
    return id2ada_prompt_encoder

class FaceID2AdaPrompt(nn.Module):
    # To be initialized in derived classes.
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize model components.
        # These components of ConsistentID_ID2AdaPrompt will be shared with the teacher model.
        # So we don't initialize them in the ctor(), but borrow them from the teacher model.
        # These components of Arc2Face_ID2AdaPrompt will be initialized in its ctor().
        self.clip_image_encoder             = None
        self.clip_preprocessor              = None
        self.face_app                       = None
        self.text_to_image_prompt_encoder   = None
        self.tokenizer                      = None
        self.dtype                          = kwargs.get('dtype', torch.float16)

        # Load Img2Ada SubjectBasisGenerator.
        self.subject_string                 = kwargs.get('subject_string', 'z')
        self.adaface_ckpt_path              = kwargs.get('adaface_ckpt_path', None)
        self.subj_basis_generator           = None
        # -1: use the default scale for the adaface encoder type.
        # i.e., 6 for arc2face and 1 for consistentID.
        self.out_id_embs_cfg_scale          = kwargs.get('out_id_embs_cfg_scale', -1)
        self.is_training                    = kwargs.get('is_training', False)
        # extend_prompt2token_proj_attention_multiplier is an integer >= 1.
        # TODO: extend_prompt2token_proj_attention_multiplier should be a list of integers.        
        self.extend_prompt2token_proj_attention_multiplier = kwargs.get('extend_prompt2token_proj_attention_multiplier', 1)
        self.prompt2token_proj_ext_attention_perturb_ratio = kwargs.get('prompt2token_proj_ext_attention_perturb_ratio', 0.1)
        
        # Set model behavior configurations.
        self.gen_neg_img_prompt             = False
        self.clip_neg_features              = None

        self.use_clip_embs                          = False
        self.do_contrast_clip_embs_on_bg_features   = False
        # num_id_vecs is the output embeddings of the ID2ImgPrompt module.
        # If there's no static image suffix embeddings, then num_id_vecs is also
        # the number of ada embeddings returned by the subject basis generator.
        # num_id_vecs will be set in each derived class.
        self.num_static_img_suffix_embs     = kwargs.get('num_static_img_suffix_embs', 0)
        print(f'{self.name} Adaface uses {self.num_id_vecs} ID image embeddings and {self.num_static_img_suffix_embs} fixed image embeddings as input.')

        self.id_img_prompt_max_length       = 77
        self.face_id_dim                    = 512
        # clip_embedding_dim: by default it's the OpenAI CLIP embedding dim.
        # Could be overridden by derived classes.
        self.clip_embedding_dim             = 1024
        self.output_dim                     = 768

    def get_id2img_learnable_modules(self):
        raise NotImplementedError
    
    def load_id2img_learnable_modules(self, id2img_learnable_modules_state_dict_list):
        id2img_prompt_encoder_learnable_modules = self.get_id2img_learnable_modules()
        for module, state_dict in zip(id2img_prompt_encoder_learnable_modules, id2img_learnable_modules_state_dict_list):
            module.load_state_dict(state_dict)
        print(f'{len(id2img_prompt_encoder_learnable_modules)} ID2ImgPrompt encoder modules loaded.')
    
    # init_subj_basis_generator() can only be called after the derived class is initialized,
    # when self.num_id_vecs, self.num_static_img_suffix_embs and self.clip_embedding_dim have been set.
    def init_subj_basis_generator(self):
        self.subj_basis_generator = \
            SubjBasisGenerator(num_id_vecs = self.num_id_vecs,
                               num_static_img_suffix_embs = self.num_static_img_suffix_embs,
                               bg_image_embedding_dim = self.clip_embedding_dim, 
                               output_dim = self.output_dim,
                               placeholder_is_bg = False,
                               prompt2token_proj_grad_scale = 1,
                               bg_prompt_translator_has_to_out_proj=False)

    def load_adaface_ckpt(self, adaface_ckpt_path):
        ckpt = torch.load(adaface_ckpt_path, map_location='cpu')
        string_to_subj_basis_generator_dict = ckpt["string_to_subj_basis_generator_dict"]
        if self.subject_string not in string_to_subj_basis_generator_dict:
            print(f"Subject '{self.subject_string}' not found in the embedding manager.")
            breakpoint()

        ckpt_subj_basis_generator = string_to_subj_basis_generator_dict[self.subject_string]
        ckpt_subj_basis_generator.N_ID              = self.num_id_vecs
        # Since we directly use the subject basis generator object from the ckpt,
        # fixing the number of static image suffix embeddings is much simpler.
        # Otherwise if we want to load the subject basis generator from its state_dict, 
        # things are more complicated, see embedding manager's load().
        ckpt_subj_basis_generator.N_SFX             = self.num_static_img_suffix_embs
        # obj_proj_in and pos_embs are for non-faces. So they are useless for human faces.
        ckpt_subj_basis_generator.obj_proj_in       = None
        ckpt_subj_basis_generator.pos_embs          = None     
        # Handle differences in num_static_img_suffix_embs between the current model and the ckpt.
        ckpt_subj_basis_generator.initialize_static_img_suffix_embs(self.num_static_img_suffix_embs, img_prompt_dim=self.output_dim)
        # Fix missing variables in old ckpt.
        ckpt_subj_basis_generator.patch_old_subj_basis_generator_ckpt()

        self.subj_basis_generator.extend_prompt2token_proj_attention(\
            ckpt_subj_basis_generator.prompt2token_proj_attention_multipliers, -1, -1, 1, perturb_std=0)
        ret = self.subj_basis_generator.load_state_dict(ckpt_subj_basis_generator.state_dict(), strict=False)
        print(f"{adaface_ckpt_path}: subject basis generator loaded for '{self.name}'.")
        print(repr(ckpt_subj_basis_generator))

        if ret is not None and len(ret.missing_keys) > 0:
            print(f"Missing keys: {ret.missing_keys}")
        if ret is not None and len(ret.unexpected_keys) > 0:
            print(f"Unexpected keys: {ret.unexpected_keys}")

        # extend_prompt2token_proj_attention_multiplier is an integer >= 1.
        # TODO: extend_prompt2token_proj_attention_multiplier should be a list of integers.        
        # If extend_prompt2token_proj_attention_multiplier > 1, then after loading state_dict, 
        # extend subj_basis_generator again.
        if self.extend_prompt2token_proj_attention_multiplier > 1:
            # During this extension, the added noise does change the extra copies of attention weights, since they are not in the ckpt.
            # During training,  prompt2token_proj_ext_attention_perturb_ratio == 0.1.
            # During inference, prompt2token_proj_ext_attention_perturb_ratio == 0.
            self.subj_basis_generator.extend_prompt2token_proj_attention(\
                None, -1, -1, self.extend_prompt2token_proj_attention_multiplier,
                perturb_std=self.prompt2token_proj_ext_attention_perturb_ratio)

        self.subj_basis_generator.freeze_prompt2token_proj()

    @torch.no_grad()
    def get_clip_neg_features(self, BS):
        if self.clip_neg_features is None:
            # neg_pixel_values: [1, 3, 224, 224]. clip_neg_features is invariant to the actual image.
            neg_pixel_values = torch.zeros([1, 3, 224, 224], device=self.clip_image_encoder.device, dtype=self.dtype)
            # Precompute CLIP negative features for the negative image prompt.
            self.clip_neg_features = self.clip_image_encoder(neg_pixel_values, attn_mask=None, output_hidden_states=True).hidden_states[-2]
        
        clip_neg_features = self.clip_neg_features.repeat(BS, 1, 1)
        return clip_neg_features
    
    # image_objs: a list of np array / tensor / Image objects of different sizes [Hi, Wi].
    # If image_objs is a list of tensors, then each tensor should be [3, Hi, Wi].
    # If image_objs is None, then image_paths should be provided, 
    # and image_objs will be loaded from image_paths.
    # fg_masks: None, or a list of [Hi, Wi].
    def extract_init_id_embeds_from_images(self, image_objs, image_paths, fg_masks=None, 
                                           size=(512, 512), calc_avg=False, 
                                           skip_non_faces=True, return_clip_embs=None, 
                                           do_contrast_clip_embs_on_bg_features=None, 
                                           verbose=False):
        # If return_clip_embs or do_contrast_clip_embs_on_bg_features is not provided, 
        # then use their default values.
        if return_clip_embs is None:
            return_clip_embs = self.use_clip_embs
        if do_contrast_clip_embs_on_bg_features is None:
            do_contrast_clip_embs_on_bg_features = self.do_contrast_clip_embs_on_bg_features

        # clip_image_encoder should be already put on GPU. 
        # So its .device is the device of its parameters.
        device = self.clip_image_encoder.device

        image_pixel_values  = []
        all_id_embs         = []
        faceless_img_count  = 0

        if image_objs is None and image_paths is not None:
            image_objs = []
            for image_path in image_paths:
                image_obj = Image.open(image_path)
                image_objs.append(image_obj)
            print(f'Loaded {len(image_objs)} images from {image_paths[0]}...')

        # image_objs could be a batch of images that have been collated into a tensor or np array.
        # image_objs can also be a list of images.
        # The code below that processes them one by one can be applied in both cases.
        # If image_objs are a collated batch, processing them one by one will not add much overhead.
        for idx, image_obj in enumerate(image_objs):
            if return_clip_embs:
                # input to clip_preprocessor: an image or a batch of images, each being PIL.Image.Image, numpy.ndarray, 
                # torch.Tensor, tf.Tensor or jax.ndarray.
                # Different sizes of images are standardized to the same size 224*224.
                clip_image_pixel_values = self.clip_preprocessor(images=image_obj, return_tensors="pt").pixel_values
                image_pixel_values.append(clip_image_pixel_values)

            # Convert tensor to numpy array.
            if isinstance(image_obj, torch.Tensor):
                image_obj = image_obj.cpu().numpy().transpose(1, 2, 0)
            if isinstance(image_obj, np.ndarray):
                image_obj = Image.fromarray(image_obj)
            # Resize image_obj to (512, 512). The scheme is Image.NEAREST, to be consistent with 
            # PersonalizedBase dataset class.
            image_obj, _, _ = pad_image_obj_to_square(image_obj)
            image_np = np.array(image_obj.resize(size, Image.NEAREST))
            face_info = self.face_app.get(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if len(face_info) > 0:
                face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
                # id_emb: [512,]
                id_emb = torch.from_numpy(face_info.normed_embedding)
            else:
                faceless_img_count += 1
                print(f'No face detected in {image_paths[idx]}.', end=' ')
                if not skip_non_faces:
                    print('Replace with random face embedding.')
                    # During training, use a random tensor as the face embedding.
                    id_emb = torch.randn(512)
                else:
                    print(f'Skip.')
                    continue

            all_id_embs.append(id_emb)

        if verbose:
            print(f'{len(all_id_embs)} face images identified, {faceless_img_count} faceless images.')
        
        # No face is detected in the input images.
        if len(all_id_embs) == 0:
            return faceless_img_count, None, None
        
        # all_id_embs: [BS, 512].
        all_id_embs = torch.stack(all_id_embs, dim=0).to(device=device, dtype=torch.float16)

        if return_clip_embs:
            # image_pixel_values: [BS, 3, 224, 224]
            image_pixel_values = torch.cat(image_pixel_values, dim=0)
            image_pixel_values = image_pixel_values.to(device=device, dtype=torch.float16)

            if fg_masks is not None:
                assert len(fg_masks) == len(image_objs)
                # fg_masks is a list of masks.
                if isinstance(fg_masks, (list, tuple)):
                    fg_masks2 = []
                    for fg_mask in fg_masks:
                        # fg_mask: [Hi, Wi]
                        # BUG: clip_preprocessor will do central crop on images. But fg_mask is not central cropped.
                        # If the ref image is not square, then the fg_mask will not match the image.
                        # TODO: crop fg_mask and images to square before calling extract_init_id_embeds_from_images().
                        # fg_mask2: [Hi, Wi] -> [1, 1, 224, 224]            
                        fg_mask2 = torch.tensor(fg_mask, device=device, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
                        fg_mask2 = F.interpolate(fg_mask2, size=image_pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                        fg_masks2.append(fg_mask2)
                    # fg_masks2: [BS, 224, 224]
                    fg_masks2 = torch.cat(fg_masks2, dim=0).squeeze(1)
                else:
                    # fg_masks is a collated batch of masks.
                    # The actual size doesn't matter, 
                    # as fg_mask2 will be resized to the same size as image features 
                    # (much smaller than image_pixel_values).            
                    fg_masks2 = fg_masks.to(device=device, dtype=torch.float16).unsqueeze(1)
                    # F.interpolate() always return a copy, even if scale_factor=1. So we don't need to clone fg_masks2.
                    fg_masks2 = F.interpolate(fg_masks2, size=image_pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                    fg_masks2 = fg_masks2.squeeze(1)
            else:
                # fg_mask2: [BS, 224, 224]. 
                fg_masks2 = torch.ones_like(image_pixel_values[:, 0, :, :], device=device, dtype=torch.float16)

            clip_neg_features = self.get_clip_neg_features(BS=image_pixel_values.shape[0])

            with torch.no_grad():
                # image_fg_features: [BS, 257, 1280]. 257: 16*16 (patch_embeds) + 1 (class_embeds).
                image_fg_dict  = self.clip_image_encoder(image_pixel_values, attn_mask=fg_masks2, output_hidden_states=True)
                # attn_mask: [BS, 1, 257]
                image_fg_features = image_fg_dict.hidden_states[-2]
                if image_fg_dict.attn_mask is not None:
                    image_fg_features = image_fg_features * image_fg_dict.attn_mask

                # A negative mask is used to extract the background features.
                # If fg_masks is None, then fg_masks2 is all ones, and bg masks is all zeros.
                # Therefore, all pixels are masked. The extracted image_bg_features will be 
                # meaningless in this case.
                image_bg_dict  = self.clip_image_encoder(image_pixel_values, attn_mask=1-fg_masks2, output_hidden_states=True)
                image_bg_features = image_bg_dict.hidden_states[-2]
                # Subtract the feature bias (null features) from the bg features, to highlight the useful bg features.
                if do_contrast_clip_embs_on_bg_features:
                    image_bg_features = image_bg_features - clip_neg_features                
                if image_bg_dict.attn_mask is not None:
                    image_bg_features = image_bg_features * image_bg_dict.attn_mask        

            # clip_fgbg_features: [BS, 514, 1280]. 514 = 257*2.
            # all_id_embs:   [BS, 512].
            clip_fgbg_features = torch.cat([image_fg_features, image_bg_features], dim=1)
        else:
            clip_fgbg_features = None
            clip_neg_features  = None

        if calc_avg:
            if return_clip_embs:
                # clip_fgbg_features: [BS, 514, 1280] -> [1, 514, 1280].
                # all_id_embs:       [BS, 512]       -> [1, 512].
                clip_fgbg_features = clip_fgbg_features.mean(dim=0, keepdim=True)
                clip_neg_features  = clip_neg_features.mean(dim=0, keepdim=True)

            debug = False
            if debug and all_id_embs is not None:
                print(image_paths)                    
                calc_stats('all_id_embs', all_id_embs)
                # Compute pairwise similarities of the embeddings.
                all_id_embs = F.normalize(all_id_embs, p=2, dim=1)
                pairwise_sim = torch.matmul(all_id_embs, all_id_embs.t())
                print('pairwise_sim:', pairwise_sim)
                top_dir = os.path.dirname(image_paths[0]) 
                mean_emb_path = os.path.join(top_dir, "mean_emb.pt")
                if os.path.exists(mean_emb_path):
                    mean_emb = torch.load(mean_emb_path)
                    sim_to_mean = torch.matmul(all_id_embs, mean_emb.t())
                    print('sim_to_mean:', sim_to_mean)

            if all_id_embs is not None:
                id_embs = all_id_embs.mean(dim=0, keepdim=True)
                # Without normalization, id_embs.norm(dim=1) is ~0.9. So normalization doesn't have much effect.
                id_embs = F.normalize(id_embs, p=2, dim=-1)
            # id_embs is None only if insightface_app is None, i.e., disabled by the user.
        else:
            # Don't do average of all_id_embs.
            id_embs = all_id_embs
                    
        return faceless_img_count, id_embs, clip_fgbg_features

    # This function should be implemented in derived classes.
    # We don't plan to fine-tune the ID2ImgPrompt module. So disable the gradient computation.
    def map_init_id_to_img_prompt_embs(self, init_id_embs, 
                                       clip_features=None,
                                       called_for_neg_img_prompt=False):
        raise NotImplementedError
        
    # If init_id_embs/pre_clip_features is provided, then use the provided face embeddings.
    # Otherwise, if image_paths/image_objs are provided, extract face embeddings from the images.
    # Otherwise, we generate random face embeddings [id_batch_size, 512].
    def get_img_prompt_embs(self, init_id_embs, pre_clip_features, image_paths, image_objs,
                            id_batch_size, 
                            skip_non_faces=True, 
                            avg_at_stage=None,     # id_emb, img_prompt_emb, or None.
                            perturb_at_stage=None, # id_emb, img_prompt_emb, or None.
                            perturb_std=0.0, 
                            verbose=False):
        face_image_count = 0
        device = self.clip_image_encoder.device
        clip_neg_features = self.get_clip_neg_features(BS=id_batch_size)

        if init_id_embs is None:
            # Input images are not provided. Generate random face embeddings.
            if image_paths is None and image_objs is None:
                faceid_embeds_from_images = False
                # Use random face embeddings as faceid_embeds. [BS, 512].
                faceid_embeds       = torch.randn(id_batch_size, 512).to(device=device, dtype=torch.float16)
                # Since it's a batch of random IDs, the CLIP features are all zeros as a placeholder.
                # Only ConsistentID_ID2AdaPrompt will use clip_fgbg_features and clip_neg_features.
                # Experiments show that using random clip features yields much better images than using zeros.
                clip_fgbg_features  = torch.randn(id_batch_size, 514, 1280).to(device=device, dtype=torch.float16) \
                                        if self.use_clip_embs else None
            else:
                # Extract face ID embeddings and CLIP features from the images.
                faceid_embeds_from_images = True
                faceless_img_count, faceid_embeds, clip_fgbg_features \
                    = self.extract_init_id_embeds_from_images( \
                        image_objs, image_paths=image_paths, size=(512, 512), 
                        calc_avg=(avg_at_stage == 'id_emb'), 
                        skip_non_faces=skip_non_faces, 
                        verbose=verbose)

                if image_paths is not None:
                    face_image_count = len(image_paths) - faceless_img_count
                else:
                    face_image_count = len(image_objs)  - faceless_img_count
        else:
            faceid_embeds_from_images = False
            # Use the provided init_id_embs as faceid_embeds.
            faceid_embeds = init_id_embs
            if pre_clip_features is not None:
                clip_fgbg_features = pre_clip_features
            else:
                clip_fgbg_features = None

            if faceid_embeds.shape[0] == 1:
                faceid_embeds = faceid_embeds.repeat(id_batch_size, 1)
                if clip_fgbg_features is not None:
                    clip_fgbg_features = clip_fgbg_features.repeat(id_batch_size, 1, 1)

        # If skip_non_faces, then faceid_embeds won't be None.
        # Otherwise, if faceid_embeds_from_images, and no face images are detected,
        # then we return Nones.
        if faceid_embeds is None:
            return face_image_count, None, None, None
        
        if perturb_at_stage == 'id_emb' and perturb_std > 0:
            # If id_batch_size > 1, after adding noises, the id_batch_size embeddings will be different.
            faceid_embeds = perturb_tensor(faceid_embeds, perturb_std, perturb_std_is_relative=True, keep_norm=True)
            if self.name == 'consistentID' or self.name == 'jointIDs':
                clip_fgbg_features = perturb_tensor(clip_fgbg_features, perturb_std, perturb_std_is_relative=True, keep_norm=True)

        faceid_embeds = F.normalize(faceid_embeds, p=2, dim=-1)

        # pos_prompt_embs, neg_prompt_embs: [BS, 77, 768] or [BS, 22, 768].
        with torch.no_grad():
            pos_prompt_embs  = \
                self.map_init_id_to_img_prompt_embs(faceid_embeds, clip_fgbg_features,
                                                    called_for_neg_img_prompt=False)
        
        if avg_at_stage == 'img_prompt_emb':
            pos_prompt_embs     = pos_prompt_embs.mean(dim=0, keepdim=True)
            faceid_embeds       = faceid_embeds.mean(dim=0, keepdim=True)
            if clip_fgbg_features is not None:
                clip_fgbg_features = clip_fgbg_features.mean(dim=0, keepdim=True)

        if perturb_at_stage == 'img_prompt_emb' and perturb_std > 0:
            # NOTE: for simplicity, pos_prompt_embs and pos_core_prompt_emb are perturbed independently.
            # This could cause inconsistency between pos_prompt_embs and pos_core_prompt_emb.
            # But in practice, unless we use both pos_prompt_embs and pos_core_prompt_emb
            # this is not an issue. But we rarely use pos_prompt_embs and pos_core_prompt_emb together.
            pos_prompt_embs     = perturb_tensor(pos_prompt_embs,     perturb_std, perturb_std_is_relative=True, keep_norm=True)

        # If faceid_embeds_from_images, and the prompt embeddings are already averaged, then 
        # we assume all images are from the same subject, and the batch dim of faceid_embeds is 1. 
        # So we need to repeat faceid_embeds.
        if faceid_embeds_from_images and avg_at_stage is not None:
            faceid_embeds   = faceid_embeds.repeat(id_batch_size, 1)
            pos_prompt_embs = pos_prompt_embs.repeat(id_batch_size, 1, 1)
            if clip_fgbg_features is not None:
                clip_fgbg_features = clip_fgbg_features.repeat(id_batch_size, 1, 1)

        if self.gen_neg_img_prompt:
            # Never perturb the negative prompt embeddings.
            with torch.no_grad():
                neg_prompt_embs = \
                    self.map_init_id_to_img_prompt_embs(torch.zeros_like(faceid_embeds),
                                                        clip_neg_features,
                                                        called_for_neg_img_prompt=True)
    
            return face_image_count, faceid_embeds, pos_prompt_embs, neg_prompt_embs
        else:
            return face_image_count, faceid_embeds, pos_prompt_embs, None

    # get_batched_img_prompt_embs() is a wrapper of get_img_prompt_embs() 
    # which is convenient for batched training.
    # NOTE: get_batched_img_prompt_embs() should only be called during training.
    # It is a wrapper of get_img_prompt_embs() which is convenient for batched training.
    # If init_id_embs is None, generate random face embeddings [BS, 512].
    # Returns faceid_embeds, id2img_prompt_emb.
    def get_batched_img_prompt_embs(self, batch_size, init_id_embs, pre_clip_features):
        # pos_prompt_embs, neg_prompt_embs are generated without gradient computation.
        # So we don't need to worry that the teacher model weights are updated.
        return self.get_img_prompt_embs(init_id_embs=init_id_embs,
                                        pre_clip_features=pre_clip_features,
                                        image_paths=None,
                                        image_objs=None, 
                                        id_batch_size=batch_size,
                                        # During training, don't skip non-face images. Instead, 
                                        # setting skip_non_faces=False will replace them by random face embeddings.
                                        skip_non_faces=False,
                                        # We always assume the instances belong to different subjects. 
                                        # So never average the embeddings across instances. 
                                        avg_at_stage=None, 
                                        verbose=False)

    # If img_prompt_embs is provided, we use it directly.
    # Otherwise, if face_id_embs is provided, we use it to generate img_prompt_embs.
    # Otherwise, if image_paths is provided, we extract face_id_embs from the images.
    # image_paths: a list of image paths. image_folder: the parent folder name.
    # avg_at_stage: 'id_emb', 'img_prompt_emb', or None.
    # avg_at_stage == ada_prompt_emb usually produces the worst results.
    # avg_at_stage == id_emb is slightly better than img_prompt_emb, but sometimes img_prompt_emb is better.
    # p_dropout and return_zero_embs_for_dropped_encoders are only used by Joint_FaceID2AdaPrompt.
    def generate_adaface_embeddings(self, image_paths, face_id_embs=None, img_prompt_embs=None,
                                    p_dropout=0,
                                    return_zero_embs_for_dropped_encoders=True,
                                    avg_at_stage='id_emb', # id_emb, img_prompt_emb, or None.
                                    perturb_at_stage=None, # id_emb, img_prompt_emb, or None.
                                    perturb_std=0, enable_static_img_suffix_embs=False):
        if (avg_at_stage is None) or avg_at_stage.lower() == 'none':
            img_prompt_avg_at_stage = None
        else:
            img_prompt_avg_at_stage = avg_at_stage

        if img_prompt_embs is None:
            # Do averaging. So id_batch_size becomes 1 after averaging.
            if img_prompt_avg_at_stage is not None:
                id_batch_size = 1
            else:
                if face_id_embs is not None:
                    id_batch_size = face_id_embs.shape[0]
                elif image_paths is not None:
                    id_batch_size = len(image_paths)
                else:
                    id_batch_size = 1
                
            # faceid_embeds: [BS, 512] is a batch of extracted face analysis embeddings. NOT used later.
            # NOTE: If face_id_embs, image_paths and image_objs are all None, 
            # then get_img_prompt_embs() generates random faceid_embeds/img_prompt_embs, 
            # and each instance is different.
            # Otherwise, if face_id_embs is provided, it's used.
            # If not, image_paths/image_objs are used to extract face embeddings.
            # img_prompt_embs is in the image prompt space.
            # img_prompt_embs: [BS, 16/4, 768].
            face_image_count, faceid_embeds, img_prompt_embs, neg_img_prompt_embs \
                = self.get_img_prompt_embs(\
                    init_id_embs=face_id_embs,
                    pre_clip_features=None,
                    # image_folder is passed only for logging purpose. 
                    # image_paths contains the paths of the images.
                    image_paths=image_paths, image_objs=None,
                    id_batch_size=id_batch_size, 
                    perturb_at_stage=perturb_at_stage,
                    perturb_std=perturb_std, 
                    avg_at_stage=img_prompt_avg_at_stage,
                    verbose=True)
            
            if face_image_count == 0:
                return None
        
        # No matter whether avg_at_stage is id_emb or img_prompt_emb, we average img_prompt_embs.
        elif avg_at_stage is not None and avg_at_stage.lower() != 'none':
            # img_prompt_embs: [BS, 16/4, 768] -> [1, 16/4, 768].
            img_prompt_embs = img_prompt_embs.mean(dim=0, keepdim=True)
            
        # adaface_subj_embs: [BS, 16/4, 768]. 
        adaface_subj_embs = \
            self.subj_basis_generator(img_prompt_embs, clip_features=None, raw_id_embs=None, 
                                      out_id_embs_cfg_scale=self.out_id_embs_cfg_scale,
                                      is_face=True, 
                                      enable_static_img_suffix_embs=enable_static_img_suffix_embs)
        # During training,  img_prompt_avg_at_stage is None, and BS >= 1.
        # During inference, img_prompt_avg_at_stage is 'id_emb' or 'img_prompt_emb', and BS == 1.
        if img_prompt_avg_at_stage is not None:
            # adaface_subj_embs: [1, 16, 768] -> [16, 768]
            adaface_subj_embs = adaface_subj_embs.squeeze(0)

        return adaface_subj_embs

class Arc2Face_ID2AdaPrompt(FaceID2AdaPrompt):
    def __init__(self, *args, **kwargs):
        self.name = 'arc2face'
        self.num_id_vecs = 16

        super().__init__(*args, **kwargs)

        self.clip_image_encoder = CLIPVisionModelWithMask.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_preprocessor  = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_image_encoder.eval()
        if self.dtype == torch.float16:
            self.clip_image_encoder.half()
        print(f'CLIP image encoder loaded.')

        '''
        {'landmark_3d_68': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc190>, 
        'landmark_2d_106': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc2b0>, 
        'detection': <insightface.model_zoo.retinaface.RetinaFace object at 0x7f8e3f0cc100>, 
        'genderage': <insightface.model_zoo.attribute.Attribute object at 0x7f8e3f0cc1f0>, 
        'recognition': <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x7f8e3f0cc0d0>}
        '''
        # Use the same model as ID2AdaPrompt does.
        # FaceAnalysis will try to find the ckpt in: models/insightface/models/antelopev2. 
        # Note there's a second "model" in the path.        
        # Note DON'T use CUDAExecutionProvider, as it will hang DDP training. 
        # Seems when loading insightface onto the GPU, it will only reside on the first GPU. 
        # Then the process on the second GPU has issue to communicate with insightface on the first GPU, causing hanging.
        self.face_app = FaceAnalysis(name='antelopev2', root='models/insightface', 
                                            providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(512, 512))
        print(f'Face encoder loaded on CPU.')

        self.text_to_image_prompt_encoder = CLIPTextModelWrapper.from_pretrained(
                                                'models/arc2face', subfolder="encoder", 
                                                torch_dtype=self.dtype
                                            )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        if self.out_id_embs_cfg_scale == -1:
            self.out_id_embs_cfg_scale = 1
        #### Arc2Face pipeline specific configs ####
        self.gen_neg_img_prompt             = False
        # bg CLIP features are used by the bg subject basis generator.
        self.use_clip_embs                  = True
        self.do_contrast_clip_embs_on_bg_features   = True
        # self.num_static_img_suffix_embs is initialized in the parent class.
        self.id_img_prompt_max_length       = 22
        self.clip_embedding_dim             = 1024

        self.init_subj_basis_generator()
        if self.adaface_ckpt_path is not None:
            self.load_adaface_ckpt(self.adaface_ckpt_path)

        print(f"{self.name} ada prompt encoder initialized, "
              f"ID vecs: {self.num_id_vecs}, static suffix: {self.num_static_img_suffix_embs}.")

    # Arc2Face_ID2AdaPrompt never uses clip_features or called_for_neg_img_prompt.
    def map_init_id_to_img_prompt_embs(self, init_id_embs, 
                                       clip_features=None,
                                       called_for_neg_img_prompt=False):

        '''
        self.text_to_image_prompt_encoder: arc2face_models.py:CLIPTextModelWrapper instance.
        init_id_embs: (N, 512) normalized Face ID embeddings.
        '''

        # arcface_token_id: 1014
        arcface_token_id = self.tokenizer.encode("id", add_special_tokens=False)[0]

        # This step should be quite fast, and there's no need to cache the input_ids.
        input_ids = self.tokenizer(
                "photo of a id person",
                truncation=True,
                padding="max_length",
                # In Arc2Face_ID2AdaPrompt, id_img_prompt_max_length is 22.
                # Arc2Face's image prompt is meanlingless in tokens other than ID tokens.
                max_length=self.id_img_prompt_max_length, 
                return_tensors="pt",
            ).input_ids.to(init_id_embs.device)
        # input_ids: [1, 22] or [3, 22] (during training).
        input_ids = input_ids.repeat(len(init_id_embs), 1)
        init_id_embs = init_id_embs.to(self.dtype)
        # face_embs_padded: [1, 512] -> [1, 768].
        face_embs_padded = F.pad(init_id_embs, (0, self.text_to_image_prompt_encoder.config.hidden_size - init_id_embs.shape[-1]), "constant", 0)
        # self.text_to_image_prompt_encoder(input_ids=input_ids, ...) is called twice. The first is only to get the token embeddings (the shallowest mapping).
        # The second call does the ordinary CLIP text encoding pass.
        token_embs = self.text_to_image_prompt_encoder(input_ids=input_ids, return_token_embs=True)
        token_embs[input_ids==arcface_token_id] = face_embs_padded

        prompt_embeds = self.text_to_image_prompt_encoder(
            input_ids=input_ids,
            input_token_embs=token_embs,
            return_token_embs=False
        )[0]

        # Restore the original dtype of prompt_embeds: float16 -> float32.
        prompt_embeds = prompt_embeds.to(self.dtype)

        # token 4: 'id' in "photo of a id person". 
        # 4:20 are the most important 16 embeddings that contain the subject's identity.
        # [N, 22, 768] -> [N, 16, 768]
        return prompt_embeds[:, 4:20]

    def get_id2img_learnable_modules(self):
        return [ self.text_to_image_prompt_encoder ]
    
# ConsistentID_ID2AdaPrompt is just a wrapper of ConsistentIDPipeline, so it's not an nn.Module.
class ConsistentID_ID2AdaPrompt(FaceID2AdaPrompt):
    def __init__(self, pipe=None, base_model_path="models/ensemble/sd15-dste8-vae.safetensors", 
                 *args, **kwargs):
        self.name = 'consistentID'
        self.num_id_vecs = 4
        
        super().__init__(*args, **kwargs)
        if pipe is None:
            # The base_model_path is kind of arbitrary, as the UNet and VAE in the model 
            # are not used and will be released soon.
            # Only the consistentID modules and bise_net are used.
            assert base_model_path is not None, "base_model_path should be provided."
            pipe = ConsistentIDPipeline.from_single_file(
                base_model_path, 
                torch_dtype=self.dtype
            )
            pipe.load_ConsistentID_model(consistentID_weight_path="./models/ConsistentID/ConsistentID-v1.bin",
                                         bise_net_weight_path="./models/ConsistentID/BiSeNet_pretrained_for_ConsistentID.pth")
            # Since the passed-in pipe is None, this should be called during inference,
            # when the teacher ConsistentIDPipeline is not initialized. 
            # Therefore, we release VAE, UNet and text_encoder to save memory.
            pipe.release_components(["unet", "vae"])

        # Otherwise, we share the pipeline with the teacher. 
        # So we don't release the components.
        self.pipe                           = pipe
        self.face_app                       = pipe.face_app
        # ConsistentID uses 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'.
        self.clip_image_encoder             = patch_clip_image_encoder_with_mask(pipe.clip_encoder)
        self.clip_preprocessor              = pipe.clip_preprocessor
        self.text_to_image_prompt_encoder   = pipe.text_encoder
        self.tokenizer                      = pipe.tokenizer
        self.image_proj_model               = pipe.image_proj_model

        self.clip_image_encoder.eval()                
        self.image_proj_model.eval()
        if self.dtype == torch.float16:
            self.clip_image_encoder.half()
            self.image_proj_model.half()

        if self.out_id_embs_cfg_scale == -1:
            self.out_id_embs_cfg_scale = 6
        #### ConsistentID pipeline specific configs ####
        # self.num_static_img_suffix_embs is initialized in the parent class.
        self.gen_neg_img_prompt             = True
        self.use_clip_embs                  = True
        self.do_contrast_clip_embs_on_bg_features   = True
        self.clip_embedding_dim             = 1280
        self.s_scale                        = 1.0
        self.shortcut                       = False

        self.init_subj_basis_generator()
        if self.adaface_ckpt_path is not None:
            self.load_adaface_ckpt(self.adaface_ckpt_path)

        print(f"{self.name} ada prompt encoder initialized, "
              f"ID vecs: {self.num_id_vecs}, static suffix: {self.num_static_img_suffix_embs}.")

    def map_init_id_to_img_prompt_embs(self, init_id_embs, 
                                       clip_features=None,
                                       called_for_neg_img_prompt=False):
        assert init_id_embs is not None, "init_id_embs should be provided."

        init_id_embs  = init_id_embs.to(self.dtype)
        clip_features = clip_features.to(self.dtype)

        if not called_for_neg_img_prompt:
            # clip_features: [BS, 514, 1280].
            # clip_features is provided when the function is called within 
            # ConsistentID_ID2AdaPrompt:extract_init_id_embeds_from_images(), which is
            # image_fg_features and image_bg_features concatenated at dim=1. 
            # Therefore, we split clip_image_double_embeds into image_fg_features and image_bg_features.
            # image_bg_features is not used in ConsistentID_ID2AdaPrompt.
            image_fg_features, image_bg_features = clip_features.chunk(2, dim=1)
            # clip_image_embeds: [BS, 257, 1280].
            clip_image_embeds = image_fg_features
        else:
            # clip_features is the negative image features. So we don't need to split it.
            clip_image_embeds = clip_features
            init_id_embs = torch.zeros_like(init_id_embs)

        faceid_embeds = init_id_embs
        # image_proj_model maps 1280-dim OpenCLIP embeddings to 768-dim face prompt embeddings.
        # clip_image_embeds are used as queries to transform faceid_embeds.
        # faceid_embeds -> kv, clip_image_embeds -> q
        if faceid_embeds.shape[0] != clip_image_embeds.shape[0]:
            breakpoint()

        try:
            global_id_embeds = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=self.shortcut, scale=self.s_scale)
        except:
            breakpoint()
        
        return global_id_embeds

    def get_id2img_learnable_modules(self):
        return [ self.image_proj_model ]

# A wrapper for combining multiple FaceID2AdaPrompt instances.
class Joint_FaceID2AdaPrompt(FaceID2AdaPrompt):
    def __init__(self, adaface_encoder_types, adaface_ckpt_paths, 
                 out_id_embs_cfg_scales=None, enabled_encoders=None,
                 *args, **kwargs):
        self.name = 'jointIDs'        
        assert len(adaface_encoder_types) > 0, "adaface_encoder_types should not be empty."
        adaface_encoder_types2num_id_vecs = { 'arc2face': 16, 'consistentID': 4 }
        self.encoders_num_id_vecs = [ adaface_encoder_types2num_id_vecs[encoder_type] \
                                      for encoder_type in adaface_encoder_types ]
        self.num_id_vecs = sum(self.encoders_num_id_vecs)
        super().__init__(*args, **kwargs)
        
        self.num_sub_encoders = len(adaface_encoder_types)
        self.id2ada_prompt_encoders = nn.ModuleList()
        self.encoders_num_static_img_suffix_embs = []

        # TODO: apply adaface_encoder_cfg_scales to influence the final prompt embeddings.
        # Now they are just placeholders.
        if out_id_embs_cfg_scales is None:
            # -1: use the default scale for the adaface encoder type.
            # i.e., 6 for arc2face and 1 for consistentID.
            self.out_id_embs_cfg_scales = [-1] * self.num_sub_encoders
        else:
            # Do not normalize the weights, and just use them as is.
            self.out_id_embs_cfg_scales = out_id_embs_cfg_scales

        # Note we don't pass the adaface_ckpt_paths to the base class, but instead,
        # we load them once and for all in self.load_adaface_ckpt().
        for i, encoder_type in enumerate(adaface_encoder_types):
            kwargs['out_id_embs_cfg_scale'] = self.out_id_embs_cfg_scales[i]
            if encoder_type == 'arc2face':
                encoder = Arc2Face_ID2AdaPrompt(*args, **kwargs)
            elif encoder_type == 'consistentID':
                encoder = ConsistentID_ID2AdaPrompt(*args, **kwargs)
            else:
                breakpoint()
            self.id2ada_prompt_encoders.append(encoder)
            self.encoders_num_static_img_suffix_embs.append(encoder.num_static_img_suffix_embs)

        self.num_static_img_suffix_embs     = sum(self.encoders_num_static_img_suffix_embs)
        # No need to set gen_neg_img_prompt, as we don't access it in this class, but rather
        # in the derived classes.
        # self.gen_neg_img_prompt           = True
        # self.use_clip_embs                = True
        # self.do_contrast_clip_embs_on_bg_features   = True
        self.face_id_dims                   = [encoder.face_id_dim for encoder in self.id2ada_prompt_encoders]
        self.face_id_dim                    = sum(self.face_id_dims)
        # Different adaface encoders may have different clip_embedding_dim.
        # clip_embedding_dim is only used for bg subject basis generator.
        # Here we use the joint clip embeddings of both OpenAI CLIP and laion CLIP.
        # Therefore, the clip_embedding_dim is the sum of the clip_embedding_dims of all adaface encoders.
        self.clip_embedding_dims            = [encoder.clip_embedding_dim for encoder in self.id2ada_prompt_encoders]
        self.clip_embedding_dim             = sum(self.clip_embedding_dims)
        # The ctors of the derived classes have already initialized encoder.subj_basis_generator.
        # If subj_basis_generator expansion params are specified, they are equally applied to all adaface encoders.
        # This self.subj_basis_generator is not meant to be called as self.subj_basis_generator(), but instead,
        # it's used as a unified interface to save/load the subj_basis_generator of all adaface encoders.
        self.subj_basis_generator           = \
            nn.ModuleList( [encoder.subj_basis_generator for encoder \
                            in self.id2ada_prompt_encoders] )
        
        if adaface_ckpt_paths is not None:
            self.load_adaface_ckpt(adaface_ckpt_paths)

        print(f"{self.name} ada prompt encoder initialized with {self.num_sub_encoders} sub-encoders. "
              f"ID vecs: {self.num_id_vecs}, static suffix embs: {self.num_static_img_suffix_embs}.")
        
        if enabled_encoders is not None:
            self.are_encoders_enabled = \
                torch.tensor([True if encoder_type in enabled_encoders else False \
                              for encoder_type in adaface_encoder_types])
            if not self.are_encoders_enabled.any():
                print(f"All encoders are disabled, which shoudn't happen.")
                breakpoint()
            if self.are_encoders_enabled.sum() < self.num_sub_encoders:
                disabled_encoders = [ encoder_type for i, encoder_type in enumerate(adaface_encoder_types) \
                                        if not self.are_encoders_enabled[i] ]
                print(f"{len(disabled_encoders)} encoders are disabled: {disabled_encoders}.")
        else:
            self.are_encoders_enabled = \
                torch.tensor([True] * self.num_sub_encoders)

    def load_adaface_ckpt(self, adaface_ckpt_paths):
        # If only one adaface ckpt path is provided, then we assume it's the ckpt of the Joint_FaceID2AdaPrompt,
        # so we dereference the list to get the actual path and load the subj_basis_generators of all adaface encoders.
        if isinstance(adaface_ckpt_paths, (list, tuple, ListConfig)):
            if len(adaface_ckpt_paths) == 1 and self.num_sub_encoders > 1:
                adaface_ckpt_paths = adaface_ckpt_paths[0]

        if isinstance(adaface_ckpt_paths, str):
            # This is only applicable to newest ckpts of Joint_FaceID2AdaPrompt, where 
            # the ckpt_subj_basis_generator is an nn.ModuleList of multiple subj_basis_generators. 
            # Therefore, no need to patch missing variables. 
            ckpt = torch.load(adaface_ckpt_paths, map_location='cpu')
            string_to_subj_basis_generator_dict = ckpt["string_to_subj_basis_generator_dict"]
            if self.subject_string not in string_to_subj_basis_generator_dict:
                print(f"Subject '{self.subject_string}' not found in the embedding manager.")
                breakpoint()

            ckpt_subj_basis_generators = string_to_subj_basis_generator_dict[self.subject_string]
            for i, subj_basis_generator in enumerate(self.subj_basis_generator):
                ckpt_subj_basis_generator = ckpt_subj_basis_generators[i]
                # Handle differences in num_static_img_suffix_embs between the current model and the ckpt.
                ckpt_subj_basis_generator.initialize_static_img_suffix_embs(self.encoders_num_static_img_suffix_embs[i], 
                                                                            img_prompt_dim=self.output_dim)

                subj_basis_generator.extend_prompt2token_proj_attention(\
                    ckpt_subj_basis_generator.prompt2token_proj_attention_multipliers, -1, -1, 1, perturb_std=0)                
                subj_basis_generator.load_state_dict(ckpt_subj_basis_generator.state_dict())

                # extend_prompt2token_proj_attention_multiplier is an integer >= 1.
                # TODO: extend_prompt2token_proj_attention_multiplier should be a list of integers.        
                # If extend_prompt2token_proj_attention_multiplier > 1, then after loading state_dict, 
                # extend subj_basis_generator again.
                if self.extend_prompt2token_proj_attention_multiplier > 1:
                    # During this extension, the added noise does change the extra copies of attention weights, since they are not in the ckpt.
                    # During training,  prompt2token_proj_ext_attention_perturb_ratio == 0.1.
                    # During inference, prompt2token_proj_ext_attention_perturb_ratio == 0.
                    subj_basis_generator.extend_prompt2token_proj_attention(\
                        None, -1, -1, self.extend_prompt2token_proj_attention_multiplier,
                        perturb_std=self.prompt2token_proj_ext_attention_perturb_ratio)

                subj_basis_generator.freeze_prompt2token_proj()

            print(f"{adaface_ckpt_paths}: {len(self.subj_basis_generator)} subj_basis_generators loaded for {self.name}.")

        elif isinstance(adaface_ckpt_paths, (list, tuple, ListConfig)):
            for i, ckpt_path in enumerate(adaface_ckpt_paths):
                self.id2ada_prompt_encoders[i].load_adaface_ckpt(ckpt_path)
        else:
            breakpoint()

    def extract_init_id_embeds_from_images(self, *args, **kwargs):
        total_faceless_img_count = 0
        all_id_embs = []
        all_clip_fgbg_features = []
        id_embs_shape = None
        clip_fgbg_features_shape = None
        # clip_image_encoder should be already put on GPU. 
        # So its .device is the device of its parameters.
        device = self.id2ada_prompt_encoders[0].clip_image_encoder.device

        for i, id2ada_prompt_encoder in enumerate(self.id2ada_prompt_encoders):
            faceless_img_count, id_embs, clip_fgbg_features = \
                id2ada_prompt_encoder.extract_init_id_embeds_from_images(*args, **kwargs)
            total_faceless_img_count += faceless_img_count
            # id_embs: [BS, 512] or [1, 512] (if calc_avg == True), or None.
            # id_embs has the same shape across all id2ada_prompt_encoders.
            all_id_embs.append(id_embs)
            # clip_fgbg_features: [BS, 514, 1280/1024] or [1, 514, 1280/1024] (if calc_avg == True), or None.
            # clip_fgbg_features has the same shape except for the last dimension across all id2ada_prompt_encoders.
            all_clip_fgbg_features.append(clip_fgbg_features)
            if id_embs is not None:
                id_embs_shape = id_embs.shape
            if clip_fgbg_features is not None:
                clip_fgbg_features_shape = clip_fgbg_features.shape
        
        num_extracted_id_embs = 0
        for i in range(len(all_id_embs)):
            if all_id_embs[i] is not None:
                # As calc_avg is the same for all id2ada_prompt_encoders, 
                # each id_embs and clip_fgbg_features should have the same shape, if they are not None.
                if all_id_embs[i].shape != id_embs_shape:
                    print("Inconsistent ID embedding shapes.")
                    breakpoint()
                else:
                    num_extracted_id_embs += 1
            else:
                all_id_embs[i] = torch.zeros(id_embs_shape, dtype=torch.float16, device=device)

            clip_fgbg_features_shape2 = torch.Size(clip_fgbg_features_shape[:-1] + (self.clip_embedding_dims[i],))
            if all_clip_fgbg_features[i] is not None:
                if all_clip_fgbg_features[i].shape != clip_fgbg_features_shape2:
                    print("Inconsistent clip features shapes.")
                    breakpoint()
            else:
                all_clip_fgbg_features[i] = torch.zeros(clip_fgbg_features_shape2, 
                                                        dtype=torch.float16, device=device)

        # If at least one face encoder detects faces, then return the embeddings.
        # Otherwise return None embeddings.
        # It's possible that some face encoders detect faces, while others don't,
        # since different face encoders use different face detection models.
        if num_extracted_id_embs == 0:
            return 0, None, None
    
        all_id_embs = torch.cat(all_id_embs, dim=1)
        # clip_fgbg_features: [BS, 514, 1280] or [BS, 514, 1024]. So we concatenate them along dim=2.
        all_clip_fgbg_features = torch.cat(all_clip_fgbg_features, dim=2)
        return total_faceless_img_count, all_id_embs, all_clip_fgbg_features

    # init_id_embs, clip_features are never None.
    def map_init_id_to_img_prompt_embs(self, init_id_embs, 
                                       clip_features=None,
                                       called_for_neg_img_prompt=False):
        if init_id_embs is None or clip_features is None:
            breakpoint()

        # each id_embs and clip_fgbg_features should have the same shape.
        # If some of them were None, they have been replaced by zero embeddings.        
        all_init_id_embs  = init_id_embs.split(self.face_id_dims,         dim=1)
        all_clip_features = clip_features.split(self.clip_embedding_dims, dim=2)
        all_img_prompt_embs = []

        for i, id2ada_prompt_encoder in enumerate(self.id2ada_prompt_encoders):
            img_prompt_embs = id2ada_prompt_encoder.map_init_id_to_img_prompt_embs(
                all_init_id_embs[i], clip_features=all_clip_features[i],
                called_for_neg_img_prompt=called_for_neg_img_prompt,
            )
            all_img_prompt_embs.append(img_prompt_embs)

        all_img_prompt_embs = torch.cat(all_img_prompt_embs, dim=1)
        return all_img_prompt_embs

    # If init_id_embs/pre_clip_features is provided, then use the provided face embeddings.
    # Otherwise, if image_paths/image_objs are provided, extract face embeddings from the images.
    # Otherwise, we generate random face embeddings [id_batch_size, 512].
    def get_img_prompt_embs(self, init_id_embs, pre_clip_features, *args, **kwargs):
        face_image_counts = []
        all_faceid_embeds = []
        all_pos_prompt_embs = []
        all_neg_prompt_embs = []
        faceid_embeds_shape = None
        # clip_image_encoder should be already put on GPU. 
        # So its .device is the device of its parameters.
        device = self.id2ada_prompt_encoders[0].clip_image_encoder.device

        # init_id_embs, pre_clip_features could be None. If they are None,
        # we split them into individual vectors for each id2ada_prompt_encoder.
        if init_id_embs is not None:
            all_init_id_embs = init_id_embs.split(self.face_id_dims, dim=1)
        else:
            all_init_id_embs = [None] * self.num_sub_encoders
        if pre_clip_features is not None:
            all_pre_clip_features = pre_clip_features.split(self.clip_embedding_dims, dim=2)
        else:
            all_pre_clip_features = [None] * self.num_sub_encoders

        faceid_embeds_shape = None
        for i, id2ada_prompt_encoder in enumerate(self.id2ada_prompt_encoders):
            face_image_count, faceid_embeds, pos_prompt_embs, neg_prompt_embs = \
                id2ada_prompt_encoder.get_img_prompt_embs(all_init_id_embs[i], all_pre_clip_features[i], 
                                                          *args, **kwargs)
            face_image_counts.append(face_image_count)
            all_faceid_embeds.append(faceid_embeds)
            all_pos_prompt_embs.append(pos_prompt_embs)
            all_neg_prompt_embs.append(neg_prompt_embs)
            # all faceid_embeds have the same shape across all id2ada_prompt_encoders.
            # But pos_prompt_embs and neg_prompt_embs may have different number of ID embeddings.
            if faceid_embeds is not None:
                faceid_embeds_shape = faceid_embeds.shape

        if faceid_embeds_shape is None:
            return 0, None, None, None

        # We take the maximum face_image_count among all adaface encoders.
        face_image_count = max(face_image_counts)
        BS = faceid_embeds.shape[0]

        for i in range(len(all_faceid_embeds)):
            if all_faceid_embeds[i] is not None:
                if all_faceid_embeds[i].shape != faceid_embeds_shape:
                    print("Inconsistent face embedding shapes.")
                    breakpoint()
            else:
                all_faceid_embeds[i] = torch.zeros(faceid_embeds_shape, dtype=torch.float16, device=device)

            N_ID = self.encoders_num_id_vecs[i]
            if all_pos_prompt_embs[i] is None:
                # Both pos_prompt_embs and neg_prompt_embs have N_ID == num_id_vecs embeddings.
                all_pos_prompt_embs[i] = torch.zeros((BS, N_ID, 768), dtype=torch.float16, device=device)
            if all_neg_prompt_embs[i] is None:
                all_neg_prompt_embs[i] = torch.zeros((BS, N_ID, 768), dtype=torch.float16, device=device)

        all_faceid_embeds   = torch.cat(all_faceid_embeds,   dim=1)
        all_pos_prompt_embs = torch.cat(all_pos_prompt_embs, dim=1)
        all_neg_prompt_embs = torch.cat(all_neg_prompt_embs, dim=1)

        return face_image_count, all_faceid_embeds, all_pos_prompt_embs, all_neg_prompt_embs
    
    # We don't need to implement get_batched_img_prompt_embs() since the interface
    # is fully compatible with FaceID2AdaPrompt.get_batched_img_prompt_embs().

    def generate_adaface_embeddings(self, image_paths, face_id_embs=None, 
                                    img_prompt_embs=None, p_dropout=0, 
                                    return_zero_embs_for_dropped_encoders=True,
                                    *args, **kwargs): 
        # clip_image_encoder should be already put on GPU. 
        # So its .device is the device of its parameters.
        device = self.id2ada_prompt_encoders[0].clip_image_encoder.device
        is_emb_averaged = kwargs.get('avg_at_stage', None) is not None
        BS = -1

        if face_id_embs is not None:
            BS = face_id_embs.shape[0]
            all_face_id_embs = face_id_embs.split(self.face_id_dims, dim=1)
        else:
            all_face_id_embs = [None] * self.num_sub_encoders
        if img_prompt_embs is not None:
            BS = img_prompt_embs.shape[0] if BS == -1 else BS
            if img_prompt_embs.shape[1] != self.num_id_vecs:
                breakpoint()
            all_img_prompt_embs = img_prompt_embs.split(self.encoders_num_id_vecs, dim=1)
        else:
            all_img_prompt_embs = [None] * self.num_sub_encoders
        if image_paths is not None:
            BS = len(image_paths) if BS == -1 else BS
        if BS == -1:
            breakpoint()

        # During training, p_dropout is 0.1. During inference, p_dropout is 0.
        # When there are two sub-encoders, the prob of one encoder being dropped is 
        # p_dropout * 2 - p_dropout^2 = 0.18.
        if p_dropout > 0:
            # self.are_encoders_enabled is a global mask.
            # are_encoders_enabled is a local mask for each batch.
            are_encoders_enabled = torch.rand(self.num_sub_encoders) < p_dropout
            are_encoders_enabled = are_encoders_enabled & self.are_encoders_enabled
            # We should at least enable one encoder.
            if not are_encoders_enabled.any():
                # Randomly enable an encoder with self.are_encoders_enabled[i] == True.
                enabled_indices = torch.nonzero(self.are_encoders_enabled).squeeze(1)
                sel_idx = torch.randint(0, len(enabled_indices), (1,)).item()
                are_encoders_enabled[enabled_indices[sel_idx]] = True
        else:
            are_encoders_enabled = self.are_encoders_enabled

        all_adaface_subj_embs = []
        num_available_id_vecs = 0

        for i, id2ada_prompt_encoder in enumerate(self.id2ada_prompt_encoders):
            if not are_encoders_enabled[i]:
                adaface_subj_embs = None
                print(f"Encoder {id2ada_prompt_encoder.name} is dropped.")
            else:
                # ddpm.embedding_manager.train() -> id2ada_prompt_encoder.train() -> each sub-enconder's train().
                # -> each sub-enconder's subj_basis_generator.train(). 
                # Therefore grad for the following call is enabled.
                adaface_subj_embs = \
                    id2ada_prompt_encoder.generate_adaface_embeddings(image_paths,
                                                                      all_face_id_embs[i],
                                                                      all_img_prompt_embs[i],
                                                                      *args, **kwargs)
            
            # adaface_subj_embs: [16, 768] or [4, 768].
            N_ID = self.encoders_num_id_vecs[i]
            if adaface_subj_embs is None:
                if not return_zero_embs_for_dropped_encoders:
                    continue
                else:
                    subj_emb_shape = (N_ID, 768) if is_emb_averaged else (BS, N_ID, 768)
                    # adaface_subj_embs is zero-filled. So N_ID is not counted as available subject embeddings.
                    adaface_subj_embs = torch.zeros(subj_emb_shape, dtype=torch.float16, device=device)
                    all_adaface_subj_embs.append(adaface_subj_embs)
            else:
                all_adaface_subj_embs.append(adaface_subj_embs)
                num_available_id_vecs += N_ID
                
        # No faces are found in the images, so return None embeddings.
        # We don't want to return an all-zero embedding, which is useless.
        if num_available_id_vecs == 0:
            return None
        
        # If id2ada_prompt_encoders are ["arc2face", "consistentID"], then 
        # during inference, we average across the batch dim.
        # all_adaface_subj_embs[0]: [4, 768]. all_adaface_subj_embs[1]: [16, 768].
        # all_adaface_subj_embs: [20, 768].
        # during training, we don't average across the batch dim.
        # all_adaface_subj_embs[0]: [BS, 4, 768]. all_adaface_subj_embs[1]: [BS, 16, 768].
        # all_adaface_subj_embs: [BS, 20, 768].
        all_adaface_subj_embs = torch.cat(all_adaface_subj_embs, dim=-2)
        return all_adaface_subj_embs


'''
# For ip-adapter distillation on objects. Strictly speaking, it's not face-to-image prompts, but
# CLIP/DINO visual features to image prompts.
class Objects_Vis2ImgPrompt(nn.Module):
    def __init__(self):
        self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16')
        self.dino_encoder.eval()
        self.dino_encoder.half()
        self.dino_preprocess = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
        print(f'DINO encoder loaded.')

'''
