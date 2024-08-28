import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor
from .arc2face_models import CLIPTextModelWrapper
from ConsistentID.lib.pipline_ConsistentID import ConsistentIDPipeline 
from .util import add_noise_to_tensor, add_noise_to_np_array, pad_image_obj_to_square, \
                  calc_stats, CLIPVisionModelWithMask, patch_clip_image_encoder_with_mask
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import os

class FaceID2ImgPrompt(nn.Module):
    # To be initialized in derived classes.
    def __init__(self):
        super().__init__()
        self.clip_image_encoder             = None
        self.clip_preprocessor              = None
        self.face_app                       = None
        self.text_to_image_prompt_encoder   = None
        self.tokenizer                      = None

    # images: numpy.ndarray or torch.Tensor.
    # images: a list of np array / tensor / Image objects of different sizes [Hi, Wi].
    # If images is a list of tensors, then each tensor should be [3, Hi, Wi].
    # If images is None, then image_paths should be provided, 
    # and images will be loaded from image_paths.
    # fg_masks: None, or a list of [Hi, Wi].
    def extract_init_id_embeds_from_images(self, images, image_paths, fg_masks=None, 
                                           size=(512, 512), calc_avg=False, 
                                           skip_non_faces=True, return_clip_embs=True, 
                                           verbose=False):
        # clip_image_encoder should be already put on GPU. 
        # So its .device is the device of its parameters.
        device = self.clip_image_encoder.device

        image_pixel_values  = []
        all_id_embs         = []
        faceless_img_count  = 0

        if images is None and image_paths is not None:
            image_objs = []
            for image_path in image_paths:
                image_obj = Image.open(image_path)
                image_objs.append(image_obj)
            images = image_objs
            print(f'Loaded {len(images)} images from {image_paths[0]}...')

        # images could be a batch of images that have been collated into a tensor or np array.
        # images can also be a list of images.
        # The code below that processes them one by one can be applied in both cases.
        # If images are a collated batch, processing them one by one will not add much overhead.
        for idx, image in enumerate(images):
            if return_clip_embs:
                # input to clip_preprocessor: an image or a batch of images, each being PIL.Image.Image, numpy.ndarray, 
                # torch.Tensor, tf.Tensor or jax.ndarray.
                # Different sizes of images are standardized to the same size 224*224.
                clip_image_pixel_values = self.clip_preprocessor(images=image, return_tensors="pt").pixel_values
                image_pixel_values.append(clip_image_pixel_values)

            # Convert tensor to numpy array.
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy().transpose(1, 2, 0)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # Resize image to (512, 512). The scheme is Image.NEAREST, to be consistent with 
            # PersonalizedBase dataset class.
            image_obj, _, _ = pad_image_obj_to_square(image)
            image_np = np.array(image_obj.resize(size, Image.NEAREST))
            face_info = self.face_app.get(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if len(face_info) == 0 and not skip_non_faces:
                print(f'No face detected in {image_paths[idx]}. Use random face embedding.')
                # If no face is detected (e.g. animals or bad images), then use a random tensor as the face embedding.
                id_emb = torch.randn(512, device=device)
                faceless_img_count += 1
            elif len(face_info) > 0:
                face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
                # id_emb: [512,]
                id_emb = torch.from_numpy(face_info.normed_embedding).to(device)
            else:
                # len(face_info) == 0 and skip_non_faces.
                # Skip images without faces.
                print(f'Skip image with no face: {image_paths[idx]}')
                continue

            all_id_embs.append(id_emb)

        if verbose:
            print(f'{len(all_id_embs)} face images identified, {faceless_img_count} faceless images.')
        # all_id_embs: [BS, 512].
        all_id_embs = torch.stack(all_id_embs, dim=0).to(device)

        if return_clip_embs:
            # image_pixel_values: [BS, 3, 224, 224]
            image_pixel_values = torch.cat(image_pixel_values, dim=0)
            image_pixel_values = image_pixel_values.to(device)

            if fg_masks is not None:
                assert len(fg_masks) == len(images)
                # fg_masks is a list of masks.
                if isinstance(fg_masks, (list, tuple)):
                    fg_masks2 = []
                    for fg_mask in fg_masks:
                        # fg_mask: [Hi, Wi]
                        # BUG: clip_preprocessor will do central crop on images. But fg_mask is not central cropped.
                        # If the ref image is not square, then the fg_mask will not match the image.
                        # TODO: crop fg_mask and images to square before calling extract_init_id_embeds_from_images().
                        # fg_mask2: [Hi, Wi] -> [1, 1, 224, 224]            
                        fg_mask2 = torch.tensor(fg_mask, device=device).float().unsqueeze(0).unsqueeze(0)
                        fg_mask2 = F.interpolate(fg_mask2, size=image_pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                        fg_masks2.append(fg_mask2)
                    # fg_masks2: [BS, 224, 224]
                    fg_masks2 = torch.cat(fg_masks2, dim=0).squeeze(1)
                else:
                    # fg_masks is a collated batch of masks.
                    # The actual size doesn't matter, 
                    # as fg_mask2 will be resized to the same size as image features 
                    # (much smaller than image_pixel_values).            
                    fg_masks2 = fg_masks.to(device=device).float().unsqueeze(1)
                    # F.interpolate() always return a copy, even if scale_factor=1. So we don't need to clone fg_masks2.
                    fg_masks2 = F.interpolate(fg_masks2, size=image_pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                    fg_masks2 = fg_masks2.squeeze(1)
            else:
                # fg_mask2: [BS, 224, 224]. 
                fg_masks2 = torch.ones_like(image_pixel_values[:, 0, :, :], device=device)

            with torch.no_grad():
                # neg_pixel_values: [1, 3, 224, 224]
                neg_pixel_values = torch.zeros_like(image_pixel_values[:1], device=device)
                neg_image_features = self.clip_image_encoder(neg_pixel_values.half(), attn_mask=None, output_hidden_states=True).hidden_states[-2]

                # image_fg_features: [BS, 257, 1280]. 257: 16*16 (patch_embeds) + 1 (class_embeds).
                image_fg_dict  = self.clip_image_encoder(image_pixel_values.half(), attn_mask=fg_masks2.half(), output_hidden_states=True)
                # attn_mask: [BS, 1, 257]
                image_fg_features = image_fg_dict.hidden_states[-2] - neg_image_features
                if image_fg_dict.attn_mask is not None:
                    image_fg_features = image_fg_features * image_fg_dict.attn_mask

                # A negative mask is used to extract the background features.
                image_bg_dict  = self.clip_image_encoder(image_pixel_values.half(), attn_mask=1-fg_masks2.half(), output_hidden_states=True)
                image_bg_features = image_bg_dict.hidden_states[-2] - neg_image_features
                if image_bg_dict.attn_mask is not None:
                    image_bg_features = image_bg_features * image_bg_dict.attn_mask        

            # clip_features: [BS, 514, 1280].
            # all_id_embs:   [BS, 512].
            clip_features = torch.cat([image_fg_features, image_bg_features], dim=1)
            clip_features = clip_features.to(image_pixel_values.dtype)
        else:
            clip_features = None

        if calc_avg:
            if return_clip_embs:
                # clip_features: [BS, 514, 1280] -> [1, 514, 1280].
                # all_id_embs:       [BS, 512]       -> [1, 512].
                clip_features = clip_features.mean(dim=0, keepdim=True)

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
                    
        return faceless_img_count, id_embs, clip_features

    # This function should be implemented in derived classes.
    def conv_init_id_to_img_prompt_embs(faceid_embeds, input_max_length=77,
                                        return_full_and_core_embs=True):
        raise NotImplementedError
        
    # If init_id_embs / image_paths / image_objs are all None,
    # we generate random face embeddings [BS, 512].
    # Otherwise we extract face ID embeds from image_paths or image_objs.
    # We don't plan to fine-tune Arc2Face. So disable the gradient computation.
    @torch.no_grad()
    def get_img_prompt_embs(self, init_id_embs, image_paths, image_objs,
                            id_batch_size, input_max_length=77, 
                            noise_level=0.0, 
                            return_core_id_embs_only=True,
                            avg_at_stage=None,  # id_emb, prompt_emb, or None.
                            gen_neg_prompt=False, verbose=False):
        face_image_count = 0
        device = self.clip_image_encoder.device

        if init_id_embs is None:
            faceid_embeds_from_images = True
            if image_paths is None and image_objs is None:
                # Use random face embeddings as faceid_embeds. [BS, 512].
                faceid_embeds = torch.randn(id_batch_size, 512)
            else:
                faceless_img_count, faceid_embeds, _ = self.extract_init_id_embeds_from_images( \
                    image_objs, image_paths=image_paths, size=(512, 512), 
                    calc_avg=(avg_at_stage == 'id_emb'), 
                    skip_non_faces=True, return_clip_embs=False, verbose=verbose)
                if image_paths is not None:
                    face_image_count = len(image_paths) - faceless_img_count
                else:
                    face_image_count = len(image_objs) - faceless_img_count
        else:
            faceid_embeds_from_images = False
            # Use the provided init_id_embs as faceid_embeds.
            faceid_embeds = init_id_embs
            if init_id_embs.shape[0] == 1:
                faceid_embeds = faceid_embeds.repeat(id_batch_size, 1)

        faceid_embeds = faceid_embeds.to(device=device, dtype=torch.float16)

        if noise_level > 0:
            # If id_batch_size > 1, after adding noises, the id_batch_size embeddings will be different.
            faceid_embeds = add_noise_to_tensor(faceid_embeds, noise_level, noise_std_is_relative=True, keep_norm=True)

        faceid_embeds = F.normalize(faceid_embeds, p=2, dim=-1)

        # arc2face_pos_prompt_emb, arc2face_neg_prompt_emb: [BS, 77, 768]
        with torch.no_grad():
            arc2face_pos_prompt_emb, arc2face_pos_core_prompt_emb  = \
                self.conv_init_id_to_img_prompt_embs(faceid_embeds, input_max_length=input_max_length,
                                                     return_full_and_core_embs=True)
        
        if avg_at_stage == 'prompt_emb':
            arc2face_pos_prompt_emb      = arc2face_pos_prompt_emb.mean(dim=0, keepdim=True)
            arc2face_pos_core_prompt_emb = arc2face_pos_core_prompt_emb.mean(dim=0, keepdim=True)

        if return_core_id_embs_only:
            arc2face_pos_prompt_emb = arc2face_pos_core_prompt_emb

        # If faceid_embeds_from_images, and the prompt embeddings are already averaged, then 
        # we assume all images are from the same subject, and the batch dim of faceid_embeds is 1. 
        # So we need to repeat faceid_embeds.
        if faceid_embeds_from_images and avg_at_stage is not None:
            faceid_embeds = faceid_embeds.repeat(id_batch_size, 1)
            arc2face_pos_prompt_emb = arc2face_pos_prompt_emb.repeat(id_batch_size, 1, 1)

        if gen_neg_prompt:
            with torch.no_grad():
                arc2face_neg_prompt_emb, arc2face_neg_core_prompt_emb = \
                    self.conv_init_id_to_img_prompt_embs(torch.zeros_like(faceid_embeds),
                                                         input_max_length=input_max_length,
                                                         return_full_and_core_embs=True)
                if return_core_id_embs_only:
                    arc2face_neg_prompt_emb = arc2face_neg_core_prompt_emb

            return face_image_count, faceid_embeds, arc2face_pos_prompt_emb, arc2face_neg_prompt_emb
        else:
            return face_image_count, faceid_embeds, arc2face_pos_prompt_emb

    # get_batched_img_prompt_embs() is a wrapper of get_img_prompt_embs() 
    # which is convenient for batched training.
    # If init_id_embs is None, generate random face embeddings [BS, 512].
    # Returns faceid_embeds, id2img_prompt_emb.
    def get_batched_img_prompt_embs(self, batch_size, init_id_embs=None):
        return self.get_img_prompt_embs(init_id_embs=init_id_embs,
                                        image_paths=None,
                                        image_objs=None, 
                                        id_batch_size=batch_size,
                                        input_max_length=22, # Remove all paddings.
                                        return_core_id_embs_only=True, 
                                        avg_at_stage=None, 
                                        gen_neg_prompt=False,                                       
                                        verbose=False)

class Arc2Face_ID2ImgPrompt(FaceID2ImgPrompt):
    def __init__(self):
        super().__init__()

        self.clip_image_encoder = CLIPVisionModelWithMask.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_preprocessor  = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_image_encoder.eval()
        self.clip_image_encoder.half()
        print(f'CLIP image encoder loaded.')

        '''
        {'landmark_3d_68': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc190>, 
        'landmark_2d_106': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc2b0>, 
        'detection': <insightface.model_zoo.retinaface.RetinaFace object at 0x7f8e3f0cc100>, 
        'genderage': <insightface.model_zoo.attribute.Attribute object at 0x7f8e3f0cc1f0>, 
        'recognition': <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x7f8e3f0cc0d0>}
        '''
        # Use the same model as ID2ImgPrompt does.
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
                                                torch_dtype=torch.float16
                                            )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        print(f'Arc2Face text-to-image prompt encoder loaded.')

    # In AdaFaceWrapper, input_max_length is 22.
    def conv_init_id_to_img_prompt_embs(self, init_id_embs, input_max_length=77, 
                                        return_full_and_core_embs=True):

        '''
        self.text_to_image_prompt_encoder: arc2face_models.py:CLIPTextModelWrapper instance.
        init_id_embs: (N, 512) normalized Face ID embeddings.
        return_full_and_core_embs: Return both the full prompt embeddings and the core embeddings. 
                                If False, return only the core embeddings.

        '''

        # arcface_token_id: 1014
        arcface_token_id = self.tokenizer.encode("id", add_special_tokens=False)[0]

        # This step should be quite fast, and there's no need to cache the input_ids.
        input_ids = self.tokenizer(
                "photo of a id person",
                truncation=True,
                padding="max_length",
                max_length=input_max_length, 
                return_tensors="pt",
            ).input_ids.to(init_id_embs.device)
        # input_ids: [1, 77] or [3, 77] (during training).
        input_ids = input_ids.repeat(len(init_id_embs), 1)
        face_embs_dtype = init_id_embs.dtype
        init_id_embs = init_id_embs.to(self.text_to_image_prompt_encoder.dtype)
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
        prompt_embeds = prompt_embeds.to(face_embs_dtype)

        if return_full_and_core_embs:
            # token 4: 'id' in "photo of a id person". 
            # 4:20 are the most important 16 embeddings that contain the subject's identity.
            # [N, 77, 768] -> [N, 16, 768]
            return prompt_embeds, prompt_embeds[:, 4:20]
        else:
            # [N, 16, 768]
            return prompt_embeds[:, 4:20]

# ConsistentID_ID2ImgPrompt is just a wrapper of ConsistentIDPipeline, so it's not an nn.Module.
class ConsistentID_ID2ImgPrompt(FaceID2ImgPrompt):
    def __init__(self, pipe=None, base_model_path=None):
        if pipe is None:
            assert base_model_path is not None, "base_model_path should be provided."
            pipe = ConsistentIDPipeline.from_single_file(
                base_model_path, 
                torch_dtype=torch.float16
            )
            pipe.load_ConsistentID_model(consistentID_weight_path="./models/ConsistentID/ConsistentID-v1.bin",
                                         bise_net_weight_path="./models/ConsistentID/BiSeNet_pretrained_for_ConsistentID.pth")
            # Since pipe is None, this should be called during inference,
            # when the teacher ConsistentIDPipeline is not initialized. 
            # Therefore, we release VAE, UNet and text_encoder to save memory.
            pipe.release_components(["unet", "vae"])

        # Otherwise, we share the pipeline with the teacher. 
        # So we don't release the components.
        self.pipe                           = pipe
        self.face_app                       = pipe.face_app
        self.clip_image_encoder             = patch_clip_image_encoder_with_mask(pipe.clip_encoder)
        self.clip_preprocessor              = pipe.clip_preprocessor
        self.text_to_image_prompt_encoder   = pipe.text_encoder
        self.tokenizer                      = pipe.tokenizer

    # In AdaFaceWrapper, input_max_length is 22.
    def conv_init_id_to_img_prompt_embs(self, init_id_embs, input_max_length=77, return_full_and_core_embs=True):

        '''
        self.text_to_image_prompt_encoder: arc2face_models.py:CLIPTextModelWrapper instance.
        init_id_embs: (N, 512) normalized Face ID embeddings.
        return_full_and_core_embs: Return both the full prompt embeddings and the core embeddings. 
                                If False, return only the core embeddings.

        '''

        # arcface_token_id: 1014
        arcface_token_id = self.tokenizer.encode("id", add_special_tokens=False)[0]

        # This step should be quite fast, and there's no need to cache the input_ids.
        input_ids = self.tokenizer(
                "photo of a id person",
                truncation=True,
                padding="max_length",
                max_length=input_max_length, 
                return_tensors="pt",
            ).input_ids.to(init_id_embs.device)
        # input_ids: [1, 77] or [3, 77] (during training).
        input_ids = input_ids.repeat(len(init_id_embs), 1)
        face_embs_dtype = init_id_embs.dtype
        init_id_embs = init_id_embs.to(self.text_to_image_prompt_encoder.dtype)
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
        prompt_embeds = prompt_embeds.to(face_embs_dtype)

        if return_full_and_core_embs:
            # token 4: 'id' in "photo of a id person". 
            # 4:20 are the most important 16 embeddings that contain the subject's identity.
            # [N, 77, 768] -> [N, 16, 768]
            return prompt_embeds, prompt_embeds[:, 4:20]
        else:
            # [N, 16, 768]
            return prompt_embeds[:, 4:20]



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
