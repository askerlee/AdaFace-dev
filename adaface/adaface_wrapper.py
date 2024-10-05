import torch
import torch.nn as nn
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from adaface.util import UNetEnsemble
from adaface.face_id_to_ada_prompt import create_id2ada_prompt_encoder
from safetensors.torch import load_file as safetensors_load_file
import re, os
import numpy as np
import sys

class AdaFaceWrapper(nn.Module):
    def __init__(self, pipeline_name, base_model_path, adaface_encoder_types, 
                 adaface_ckpt_paths, adaface_encoder_cfg_scales=None, 
                 enabled_encoders=None,
                 subject_string='z', num_inference_steps=50, negative_prompt=None,
                 use_840k_vae=False, use_ds_text_encoder=False, 
                 main_unet_filepath=None, unet_types=None, extra_unet_dirpaths=None, unet_weights=None,
                 device='cuda', is_training=False):
        '''
        pipeline_name: "text2img", "img2img", "text2img3", "flux", or None. 
        If None, it's used only as a face encoder, and the unet and vae are
        removed from the pipeline to release RAM.
        '''
        super().__init__()
        self.pipeline_name = pipeline_name
        self.base_model_path = base_model_path
        self.adaface_encoder_types = adaface_encoder_types

        self.adaface_ckpt_paths = adaface_ckpt_paths
        self.adaface_encoder_cfg_scales = adaface_encoder_cfg_scales
        self.enabled_encoders = enabled_encoders
        self.subject_string = subject_string

        self.num_inference_steps = num_inference_steps
        self.use_840k_vae = use_840k_vae
        self.use_ds_text_encoder = use_ds_text_encoder
        self.main_unet_filepath = main_unet_filepath
        self.unet_types = unet_types
        self.extra_unet_dirpaths = extra_unet_dirpaths
        self.unet_weights = unet_weights
        self.device = device
        self.is_training = is_training

        if negative_prompt is None:
            self.negative_prompt = \
            "flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, " \
            "mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, " \
            "mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, " \
            "nude, naked, nsfw, topless, bare breasts"
        else:
            self.negative_prompt = negative_prompt

        self.initialize_pipeline()
        # During inference, we never use static image suffix embeddings. 
        # So num_id_vecs is the length of the returned adaface embeddings for each encoder.
        self.encoders_num_id_vecs = self.id2ada_prompt_encoder.encoders_num_id_vecs
        self.extend_tokenizer_and_text_encoder()

    def initialize_pipeline(self):
        self.id2ada_prompt_encoder = create_id2ada_prompt_encoder(self.adaface_encoder_types,
                                                                  self.adaface_ckpt_paths,
                                                                  self.adaface_encoder_cfg_scales,
                                                                  self.enabled_encoders)

        self.id2ada_prompt_encoder.to(self.device)
        print(f"adaface_encoder_cfg_scales: {self.adaface_encoder_cfg_scales}")

        if self.use_840k_vae:
            # The 840000-step vae model is slightly better in face details than the original vae model.
            # https://huggingface.co/stabilityai/sd-vae-ft-mse-original
            vae = AutoencoderKL.from_single_file("models/diffusers/sd-vae-ft-mse-original/vae-ft-mse-840000-ema-pruned.ckpt", 
                                                 torch_dtype=torch.float16)
        else:
            vae = None

        if self.use_ds_text_encoder:
            # The dreamshaper v7 finetuned text encoder follows the prompt slightly better than the original text encoder.
            # https://huggingface.co/Lykon/DreamShaper/tree/main/text_encoder
            text_encoder = CLIPTextModel.from_pretrained("models/diffusers/ds_text_encoder", 
                                                         torch_dtype=torch.float16)
        else:
            text_encoder = None

        remove_unet = False

        if self.pipeline_name == "img2img":
            PipelineClass = StableDiffusionImg2ImgPipeline
        elif self.pipeline_name == "text2img":
            PipelineClass = StableDiffusionPipeline
        elif self.pipeline_name == "text2img3":
            PipelineClass = StableDiffusion3Pipeline
        elif self.pipeline_name == "flux":
            PipelineClass = FluxPipeline
        # pipeline_name is None means only use this instance to generate adaface embeddings, not to generate images.
        elif self.pipeline_name is None:
            PipelineClass = StableDiffusionPipeline
            remove_unet = True
        else:
            raise ValueError(f"Unknown pipeline name: {self.pipeline_name}")
        
        if self.base_model_path is None:
            base_model_path_dict = { 
                'text2img':  'runwayml/stable-diffusion-v1-5',
                'text2img3': 'stabilityai/stable-diffusion-3-medium-diffusers',
                'flux':      'black-forest-labs/FLUX.1-schnell',
            }
            self.base_model_path = base_model_path_dict[self.pipeline_name]

        if os.path.isfile(self.base_model_path):
            pipeline = PipelineClass.from_single_file(
                self.base_model_path, 
                torch_dtype=torch.float16
                )
        else:
            pipeline = PipelineClass.from_pretrained(
                    self.base_model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None
                )
        
        if self.main_unet_filepath is not None:
            print(f"Replacing the UNet with the UNet from {self.main_unet_filepath}.")
            ret = pipeline.unet.load_state_dict(self.load_unet_from_file(self.main_unet_filepath, device='cpu'))
            if len(ret.missing_keys) > 0:
                print(f"Missing keys: {ret.missing_keys}")
            if len(ret.unexpected_keys) > 0:
                print(f"Unexpected keys: {ret.unexpected_keys}")

        if (self.unet_types is not None and len(self.unet_types) > 0) \
          or (self.extra_unet_dirpaths is not None and len(self.extra_unet_dirpaths) > 0):
            unet_ensemble = UNetEnsemble([pipeline.unet], self.unet_types, self.extra_unet_dirpaths, self.unet_weights,
                                         device=self.device, torch_dtype=torch.float16)
            pipeline.unet = unet_ensemble

        print(f"Loaded pipeline from {self.base_model_path}.")
        
        if self.use_840k_vae:
            pipeline.vae = vae
            print("Replaced the VAE with the 840k-step VAE.")
            
        if self.use_ds_text_encoder:
            pipeline.text_encoder = text_encoder
            print("Replaced the text encoder with the DreamShaper text encoder.")

        if remove_unet:
            # Remove unet and vae to release RAM. Only keep tokenizer and text_encoder.
            pipeline.unet = None
            pipeline.vae  = None
            print("Removed UNet and VAE from the pipeline.")

        if self.pipeline_name not in ["text2img3", "flux"]:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
            pipeline.scheduler = noise_scheduler
        # Otherwise, pipeline.scheduler == FlowMatchEulerDiscreteScheduler
        self.pipeline = pipeline.to(self.device)

    def load_unet_from_file(self, unet_path, device=None):
        if os.path.isfile(unet_path):
            if unet_path.endswith(".safetensors"):
                unet_state_dict = safetensors_load_file(unet_path, device=device)
            else:
                unet_state_dict = torch.load(unet_path, map_location=device)

            key0 = list(unet_state_dict.keys())[0]
            if key0.startswith("model.diffusion_model"):
                key_prefix = ""
                is_ldm_unet = True
            elif key0.startswith("diffusion_model"):
                key_prefix = "model."
                is_ldm_unet = True
            else:
                is_ldm_unet = False

            if is_ldm_unet:
                unet_state_dict2 = {}
                for key, value in unet_state_dict.items():
                    key2 = key_prefix + key
                    unet_state_dict2[key2] = value
                print(f"LDM UNet detected. Convert to diffusers")
                ldm_unet_config = { 'layers_per_block': 2 }
                unet_state_dict = convert_ldm_unet_checkpoint(unet_state_dict2, ldm_unet_config)
        else:
            raise ValueError(f"UNet path {unet_path} is not a file.")
        return unet_state_dict
        
    def extend_tokenizer_and_text_encoder(self):
        if np.sum(self.encoders_num_id_vecs) < 1:
            raise ValueError(f"encoders_num_id_vecs has to be larger or equal to 1, but is {self.encoders_num_id_vecs}")

        tokenizer = self.pipeline.tokenizer
        # If adaface_encoder_types is ["arc2face", "consistentID"], then total_num_id_vecs = 20.
        # We add z_0_0, z_0_1, z_0_2, ..., z_0_15, z_1_0, z_1_1, z_1_2, z_1_3 to the tokenizer.
        self.all_placeholder_tokens = []
        self.placeholder_tokens_strs = []
        for i in range(len(self.adaface_encoder_types)):
            placeholder_tokens = []
            for j in range(self.encoders_num_id_vecs[i]):
                placeholder_tokens.append(f"{self.subject_string}_{i}_{j}")
                placeholder_tokens_str = " ".join(placeholder_tokens)

            self.all_placeholder_tokens.extend(placeholder_tokens)
            self.placeholder_tokens_strs.append(placeholder_tokens_str)

        self.all_placeholder_tokens_str = " ".join(self.placeholder_tokens_strs)

        # Add the new tokens to the tokenizer.
        num_added_tokens = tokenizer.add_tokens(self.all_placeholder_tokens)
        if num_added_tokens != np.sum(self.encoders_num_id_vecs):
            raise ValueError(
                f"The tokenizer already contains some of the tokens {self.all_placeholder_tokens_str}. Please pass a different"
                " `subject_string` that is not already in the tokenizer.")

        print(f"Added {num_added_tokens} tokens ({self.all_placeholder_tokens_str}) to the tokenizer.")
        
        # placeholder_token_ids: [49408, ..., 49423].
        self.placeholder_token_ids = tokenizer.convert_tokens_to_ids(self.all_placeholder_tokens)
        #print("New tokens:", self.placeholder_token_ids)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        old_weight_shape = self.pipeline.text_encoder.get_input_embeddings().weight.shape
        self.pipeline.text_encoder.resize_token_embeddings(len(tokenizer))
        new_weight = self.pipeline.text_encoder.get_input_embeddings().weight
        print(f"Resized text encoder token embeddings from {old_weight_shape} to {new_weight.shape} on {new_weight.device}.")

    # Extend pipeline.text_encoder with the adaface subject emeddings.
    # subj_embs: [16, 768].
    def update_text_encoder_subj_embeddings(self, subj_embs):
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        # token_embeds: [49412, 768]
        token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for i, token_id in enumerate(self.placeholder_token_ids):
                token_embeds[token_id] = subj_embs[i]
            print(f"Updated {len(self.placeholder_token_ids)} tokens ({self.all_placeholder_tokens_str}) in the text encoder.")

    def update_prompt(self, prompt):
        if prompt is None:
            prompt = ""

        # Delete the subject_string from the prompt.
        re.sub(r'\b(a|an|the)\s+' + self.subject_string + r'\b,?', "", prompt)
        re.sub(r'\b' + self.subject_string + r'\b,?',              "", prompt)
        # Prevously, arc2face ada prompts work better if they are prepended to the prompt,
        # and consistentID ada prompts work better if they are appended to the prompt.
        # When we do joint training, seems both work better if they are appended to the prompt.
        # Therefore we simply appended all placeholder_tokens_str's to the prompt.
        # NOTE: Prepending them hurts compositional prompts.
        prompt = prompt + " " + self.all_placeholder_tokens_str

        return prompt

    # avg_at_stage: 'id_emb', 'img_prompt_emb', or None.
    # avg_at_stage == ada_prompt_emb usually produces the worst results.
    # id_emb is slightly better than img_prompt_emb, but sometimes img_prompt_emb is better.
    def prepare_adaface_embeddings(self, image_paths, face_id_embs=None, 
                                   avg_at_stage='id_emb', # id_emb, img_prompt_emb, ada_prompt_emb, or None.
                                   perturb_at_stage=None, # id_emb, img_prompt_emb, or None.
                                   perturb_std=0, update_text_encoder=True):

        all_adaface_subj_embs = \
            self.id2ada_prompt_encoder.generate_adaface_embeddings(\
                    image_paths, face_id_embs=face_id_embs, 
                    img_prompt_embs=None, 
                    avg_at_stage=avg_at_stage,
                    perturb_at_stage=perturb_at_stage,
                    perturb_std=perturb_std,
                    enable_static_img_suffix_embs=False)
        
        if all_adaface_subj_embs is None:
            return None
        
        # [1, 1, 16, 768] -> [16, 768]
        all_adaface_subj_embs = all_adaface_subj_embs.squeeze(0).squeeze(0)

        if update_text_encoder:
            self.update_text_encoder_subj_embeddings(all_adaface_subj_embs)
        return all_adaface_subj_embs

    def encode_prompt(self, prompt, negative_prompt=None, device=None, verbose=False):
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        
        if device is None:
            device = self.device
        
        prompt = self.update_prompt(prompt)
        if verbose:
            print(f"Subject prompt: {prompt}")

        # For some unknown reason, the text_encoder is still on CPU after self.pipeline.to(self.device).
        # So we manually move it to GPU here.
        self.pipeline.text_encoder.to(device)
        # pooled_prompt_embeds_, negative_pooled_prompt_embeds_ are used by text2img3 and flux.
        pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = None, None

        # Compatible with older versions of diffusers.
        if not hasattr(self.pipeline, "encode_prompt"):
            # prompt_embeds_, negative_prompt_embeds_: [77, 768] -> [1, 77, 768].
            prompt_embeds_, negative_prompt_embeds_ = \
                self.pipeline._encode_prompt(prompt, device=device, num_images_per_prompt=1,
                                             do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            prompt_embeds_ = prompt_embeds_.unsqueeze(0)
            negative_prompt_embeds_ = negative_prompt_embeds_.unsqueeze(0)
        else:
            if self.pipeline_name in ["text2img3", "flux"]:
                # prompt_embeds_, negative_prompt_embeds_: [1, 333, 4096]
                # pooled_prompt_embeds_, negative_pooled_prompt_embeds_: [1, 2048]
                # CLIP Text Encoder prompt uses a maximum sequence length of 77.
                # T5 Text Encoder prompt uses a maximum sequence length of 256.
                # 333 = 256 + 77.
                prompt_t5 = prompt + "".join([", "] * 256)
                if self.pipeline_name == "text2img3":
                    prompt_embeds_, negative_prompt_embeds_, \
                    pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = \
                        self.pipeline.encode_prompt(prompt, prompt, prompt_t5, device=device, 
                                                    num_images_per_prompt=1, 
                                                    do_classifier_free_guidance=True,
                                                    negative_prompt=negative_prompt)
                elif self.pipeline_name == "flux":
                    # prompt_embeds_: [1, 512, 4096]
                    # pooled_prompt_embeds_: [1, 768]
                    prompt_embeds_, pooled_prompt_embeds_, text_ids = \
                        self.pipeline.encode_prompt(prompt, prompt_t5, device=device, 
                                                    num_images_per_prompt=1)
                    negative_prompt_embeds_ = negative_pooled_prompt_embeds_ = None
                else:
                    breakpoint()
            else:
                # prompt_embeds_, negative_prompt_embeds_: [1, 77, 768]
                prompt_embeds_, negative_prompt_embeds_ = \
                    self.pipeline.encode_prompt(prompt, device=device, 
                                                num_images_per_prompt=1, 
                                                do_classifier_free_guidance=True,
                                                negative_prompt=negative_prompt)
                
        return prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds_, negative_pooled_prompt_embeds_
    
    # ref_img_strength is used only in the img2img pipeline.
    def forward(self, noise, prompt, negative_prompt=None, guidance_scale=4.0, 
                out_image_count=4, ref_img_strength=0.8, generator=None, verbose=False):
        noise = noise.to(device=self.device, dtype=torch.float16)

        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        # prompt_embeds_, negative_prompt_embeds_: [1, 77, 768]
        prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds_, \
            negative_pooled_prompt_embeds_ = \
                self.encode_prompt(prompt, negative_prompt, device=self.device, verbose=verbose)
        # Repeat the prompt embeddings for all images in the batch.
        prompt_embeds_ = prompt_embeds_.repeat(out_image_count, 1, 1)
        if negative_prompt_embeds_ is not None:
            negative_prompt_embeds_ = negative_prompt_embeds_.repeat(out_image_count, 1, 1)

        if self.pipeline_name == "text2img3":
            pooled_prompt_embeds_           = pooled_prompt_embeds_.repeat(out_image_count, 1)
            negative_pooled_prompt_embeds_  = negative_pooled_prompt_embeds_.repeat(out_image_count, 1)

            # noise: [BS, 4, 64, 64]
            # When the pipeline is text2img, strength is ignored.
            images = self.pipeline(prompt_embeds=prompt_embeds_, 
                                   negative_prompt_embeds=negative_prompt_embeds_, 
                                   pooled_prompt_embeds=pooled_prompt_embeds_,
                                   negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_,
                                   num_inference_steps=self.num_inference_steps, 
                                   guidance_scale=guidance_scale, 
                                   num_images_per_prompt=1,
                                   generator=generator).images
        elif self.pipeline_name == "flux":
            images = self.pipeline(prompt_embeds=prompt_embeds_, 
                                   pooled_prompt_embeds=pooled_prompt_embeds_, 
                                   num_inference_steps=4, 
                                   guidance_scale=guidance_scale, 
                                   num_images_per_prompt=1,
                                   generator=generator).images
        else:
            # When the pipeline is text2img, noise: [BS, 4, 64, 64], and strength is ignored.
            # When the pipeline is img2img,  noise is an initiali image of [BS, 3, 512, 512],
            # whose pixels are normalized to [0, 1].
            images = self.pipeline(image=noise,
                                   prompt_embeds=prompt_embeds_, 
                                   negative_prompt_embeds=negative_prompt_embeds_, 
                                   num_inference_steps=self.num_inference_steps, 
                                   guidance_scale=guidance_scale, 
                                   num_images_per_prompt=1,
                                   strength=ref_img_strength,
                                   generator=generator).images
        # images: [BS, 3, 512, 512]
        return images
    