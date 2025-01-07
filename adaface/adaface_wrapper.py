import torch
import torch.nn as nn
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    #FluxPipeline,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverSinglestepScheduler,
    AutoencoderKL,
    LCMScheduler,
)
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from adaface.util import UNetEnsemble
from adaface.face_id_to_ada_prompt import create_id2ada_prompt_encoder
from adaface.diffusers_attn_lora_capture import set_up_attn_processors, set_up_ffn_loras, set_lora_and_capture_flags
from safetensors.torch import load_file as safetensors_load_file
import re, os
import numpy as np
from peft.utils.constants import DUMMY_TARGET_MODULES

class AdaFaceWrapper(nn.Module):
    def __init__(self, pipeline_name, base_model_path, adaface_encoder_types, 
                 adaface_ckpt_paths, adaface_encoder_cfg_scales=None, 
                 enabled_encoders=None, use_lcm=False, default_scheduler_name='ddim',
                 num_inference_steps=50, subject_string='z', negative_prompt=None,
                 use_840k_vae=False, use_ds_text_encoder=False, 
                 main_unet_filepath=None, unet_types=None, extra_unet_dirpaths=None, unet_weights_in_ensemble=None,
                 enable_static_img_suffix_embs=None, unet_uses_attn_lora=False,
                 suppress_sc_subj_attn=False, device='cuda', is_training=False):
        '''
        pipeline_name: "text2img", "text2imgxl", "img2img", "text2img3", "flux", or None. 
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
        self.enable_static_img_suffix_embs = enable_static_img_suffix_embs
        self.unet_uses_attn_lora = unet_uses_attn_lora
        self.use_lcm = use_lcm
        self.subject_string = subject_string
        self.suppress_sc_subj_attn = suppress_sc_subj_attn

        self.default_scheduler_name = default_scheduler_name
        self.num_inference_steps = num_inference_steps if not use_lcm else 4
        self.use_840k_vae = use_840k_vae
        self.use_ds_text_encoder = use_ds_text_encoder
        self.main_unet_filepath = main_unet_filepath
        self.unet_types = unet_types
        self.extra_unet_dirpaths = extra_unet_dirpaths
        self.unet_weights_in_ensemble = unet_weights_in_ensemble
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
        self.encoders_num_id_vecs = np.array(self.id2ada_prompt_encoder.encoders_num_id_vecs)
        self.encoders_num_static_img_suffix_embs = np.array(self.id2ada_prompt_encoder.encoders_num_static_img_suffix_embs)
        self.encoders_num_id_vecs += self.encoders_num_static_img_suffix_embs
        self.extend_tokenizer_and_text_encoder()

    def to(self, device):
        self.device = device
        self.id2ada_prompt_encoder.to(device)
        self.pipeline.to(device)
        print(f"Moved AdaFaceWrapper to {device}.")
        return self
    
    def initialize_pipeline(self):
        self.id2ada_prompt_encoder = create_id2ada_prompt_encoder(self.adaface_encoder_types,
                                                                  self.adaface_ckpt_paths,
                                                                  self.adaface_encoder_cfg_scales,
                                                                  self.enabled_encoders,
                                                                  num_static_img_suffix_embs=4)

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
        elif self.pipeline_name == "text2imgxl":
            PipelineClass = StableDiffusionXLPipeline
        elif self.pipeline_name == "text2img3":
            PipelineClass = StableDiffusion3Pipeline
        #elif self.pipeline_name == "flux":
        #    PipelineClass = FluxPipeline
        # pipeline_name is None means only use this instance to generate adaface embeddings, not to generate images.
        elif self.pipeline_name is None:
            PipelineClass = StableDiffusionPipeline
            remove_unet = True
        else:
            raise ValueError(f"Unknown pipeline name: {self.pipeline_name}")
        
        if self.base_model_path is None:
            base_model_path_dict = { 
                'text2img':     'models/sd15-dste8-vae.safetensors',
                'text2imgxl':   'stabilityai/stable-diffusion-xl-base-1.0',
                'text2img3':    'stabilityai/stable-diffusion-3-medium-diffusers',
                'flux':         'black-forest-labs/FLUX.1-schnell',
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
        
        if self.use_lcm:
            lcm_path_dict = {
                'text2img':     'latent-consistency/lcm-lora-sdv1-5',
                'text2imgxl':   'latent-consistency/lcm-lora-sdxl',
            }
            if self.pipeline_name not in lcm_path_dict:
                raise ValueError(f"Pipeline {self.pipeline_name} does not support LCM.")
            
            lcm_path = lcm_path_dict[self.pipeline_name]
            pipeline.load_lora_weights(lcm_path)
            pipeline.fuse_lora()
            print(f"Loaded LCM weights from {lcm_path}.")
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

        if self.main_unet_filepath is not None:
            print(f"Replacing the UNet with the UNet from {self.main_unet_filepath}.")
            ret = pipeline.unet.load_state_dict(self.load_unet_from_file(self.main_unet_filepath, device='cpu'))
            if len(ret.missing_keys) > 0:
                print(f"Missing keys: {ret.missing_keys}")
            if len(ret.unexpected_keys) > 0:
                print(f"Unexpected keys: {ret.unexpected_keys}")

        if (self.unet_types is not None and len(self.unet_types) > 0) \
          or (self.extra_unet_dirpaths is not None and len(self.extra_unet_dirpaths) > 0):
            unet_ensemble = UNetEnsemble([pipeline.unet], self.unet_types, self.extra_unet_dirpaths, self.unet_weights_in_ensemble,
                                         device=self.device, torch_dtype=torch.float16)
            pipeline.unet = unet_ensemble

        print(f"Loaded pipeline from {self.base_model_path}.")
        if not remove_unet and (self.unet_uses_attn_lora or self.suppress_sc_subj_attn):
            unet2 = self.load_unet_lora_weights(pipeline.unet, use_attn_lora=self.unet_uses_attn_lora,
                                                suppress_sc_subj_attn=self.suppress_sc_subj_attn)
                                                
            pipeline.unet = unet2

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

        if self.pipeline_name not in ["text2imgxl", "text2img3", "flux"] and not self.use_lcm:
            if self.default_scheduler_name == 'ddim':
                noise_scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False,
                    steps_offset=1,
                    timestep_spacing="leading",
                    rescale_betas_zero_snr=False,
                )
            elif self.default_scheduler_name == 'pndm':
                noise_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    set_alpha_to_one=False,
                    steps_offset=1,
                    timestep_spacing="leading",
                    skip_prk_steps=True,
                )
            elif self.default_scheduler_name == 'dpm++':
                noise_scheduler = DPMSolverSinglestepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    prediction_type="epsilon",
                    num_train_timesteps=1000,
                    trained_betas=None,
                    thresholding=False,
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    lower_order_final=True,
                    use_karras_sigmas=True,
                )                
            else:
                breakpoint()                

            pipeline.scheduler = noise_scheduler
        # Otherwise, if not use_lcm, pipeline.scheduler == FlowMatchEulerDiscreteScheduler
        #            if     use_lcm, pipeline.scheduler == LCMScheduler
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

    # Adapted from ConsistentIDPipeline:set_ip_adapter().
    def load_unet_loras(self, unet, unet_lora_modules_state_dict, 
                        use_attn_lora=True, use_ffn_lora=False, suppress_sc_subj_attn=False):
        # We don't have to specify subj_attn_var_shrink_factor for set_up_attn_processors(),
        # as we'll load the learned subj_attn_var_shrink_factor from ckpt.
        attn_capture_procs, attn_opt_modules = \
            set_up_attn_processors(unet, enable_lora=True, lora_layer_names=['q'],
                                   lora_rank=192, lora_scale_down=16)
        # up_blocks.3.resnets.[1~2].conv1, conv2, conv_shortcut. [12] matches 1 or 2.
        if use_ffn_lora:
            target_modules_pat = 'up_blocks.3.resnets.[12].conv.+'
        else:
            # A special pattern, "dummy-target-modules" tells PEFT to add loras on NONE of the layers.
            # We couldn't simply skip PEFT initialization (converting unet to a PEFT model),
            # otherwise the attn lora layers will cause nan quickly during a fp16 training.
            target_modules_pat = DUMMY_TARGET_MODULES

        unet, ffn_lora_layers, ffn_opt_modules = \
            set_up_ffn_loras(unet, target_modules_pat=target_modules_pat, lora_uses_dora=True)

        # self.attn_capture_procs and ffn_lora_layers will be used in set_lora_and_capture_flags().
        self.attn_capture_procs = list(attn_capture_procs.values())
        self.ffn_lora_layers    = list(ffn_lora_layers.values())
        # Combine attn_opt_modules and ffn_opt_modules into unet_lora_modules.
        # unet_lora_modules is for optimization and loading/saving.
        unet_lora_modules = {}
        # attn_opt_modules and ffn_opt_modules have different depths of keys.
        # attn_opt_modules:
        # up_blocks_3_attentions_1_transformer_blocks_0_attn2_processor_std_shrink_factor,
        # up_blocks_3_attentions_1_transformer_blocks_0_attn2_processor_to_q_lora_lora_A, ...
        # ffn_opt_modules:
        # base_model_model_up_blocks_3_resnets_1_conv1_lora_A, ...
        # with the prefix 'base_model_model_'. Because ffn_opt_modules are extracted from the peft-wrapped model,
        # and attn_opt_modules are extracted from the original unet model.
        # To be compatible with old param keys, we append 'base_model_model_' to the keys of attn_opt_modules.
        unet_lora_modules.update({ f'base_model_model_{k}': v for k, v in attn_opt_modules.items() })
        unet_lora_modules.update(ffn_opt_modules)
        self.unet_lora_modules  = torch.nn.ParameterDict(unet_lora_modules)

        missing, unexpected = self.unet_lora_modules.load_state_dict(unet_lora_modules_state_dict, strict=False)
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

        print(f"Loaded {len(unet_lora_modules_state_dict)} LoRA weights on the UNet:\n{unet_lora_modules.keys()}")
        self.outfeat_capture_blocks.append(unet.up_blocks[3])

        # If suppress_sc_subj_attn is True and use_attn_lora is False, we load all these params from ckpt,
        # but since we set use_attn_lora to False, attn loras won't be used during inference nonetheless.
        set_lora_and_capture_flags(self.attn_capture_procs, self.outfeat_capture_blocks, self.ffn_lora_layers, 
                                   use_attn_lora, use_ffn_lora, capture_ca_activations=False, 
                                   suppress_subj_attn=suppress_sc_subj_attn)

        return unet

    def load_unet_lora_weights(self, unet, use_attn_lora=True, suppress_sc_subj_attn=False):
        unet_lora_weight_found = False
        if isinstance(self.adaface_ckpt_paths, str):
            adaface_ckpt_paths = [self.adaface_ckpt_paths]
        else:
            adaface_ckpt_paths = self.adaface_ckpt_paths

        for adaface_ckpt_path in adaface_ckpt_paths:
            ckpt_dict = torch.load(adaface_ckpt_path, map_location='cpu')
            if 'unet_lora_modules' in ckpt_dict:
                unet_lora_modules_state_dict = ckpt_dict['unet_lora_modules']                
                print(f"{len(unet_lora_modules_state_dict)} LoRA weights found in {adaface_ckpt_path}.")
                unet_lora_weight_found = True
                break

        # Since unet lora weights are not found in the adaface ckpt, we give up on loading unet attn processors.
        if not unet_lora_weight_found:
            print(f"LoRA weights not found in {self.adaface_ckpt_paths}.")
            return unet
        
        self.outfeat_capture_blocks = []

        if isinstance(unet, UNetEnsemble):
            for i, unet_ in enumerate(unet.unets):
                unet_ = self.load_unet_loras(unet_, unet_lora_modules_state_dict, 
                                             use_attn_lora=use_attn_lora, 
                                             suppress_sc_subj_attn=suppress_sc_subj_attn)
                unet.unets[i] = unet_
            print(f"Loaded LoRA processors on UNetEnsemble of {len(unet.unets)} UNets.")
        else:
            unet = self.load_unet_loras(unet, unet_lora_modules_state_dict, 
                                        use_attn_lora=use_attn_lora, 
                                        suppress_sc_subj_attn=suppress_sc_subj_attn)

        return unet

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
        self.updated_tokens_str = self.all_placeholder_tokens_str
        # all_null_placeholder_tokens_str: ", , , , ..." (20 times).
        # It just contains the commas and spaces with the same length, but no actual tokens.
        self.all_null_placeholder_tokens_str = " ".join([", "] * len(self.all_placeholder_tokens))

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
    def update_text_encoder_subj_embeddings(self, subj_embs, lens_subj_emb_segments):
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        # token_embeds: [49412, 768]
        token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
        all_encoders_updated_tokens = []
        all_encoders_updated_token_strs = []
        idx = 0
        
        with torch.no_grad():
            # sum of lens_subj_emb_segments are probably shorter than self.placeholder_token_ids,
            # when some static_img_suffix_embs are disabled.
            for i, encoder_type in enumerate(self.adaface_encoder_types):
                encoder_updated_tokens = []
                if (self.enabled_encoders is not None) and (encoder_type not in self.enabled_encoders):
                    idx += lens_subj_emb_segments[i]
                    continue
                for j in range(lens_subj_emb_segments[i]):
                    placeholder_token = f"{self.subject_string}_{i}_{j}"
                    token_id = self.pipeline.tokenizer.convert_tokens_to_ids(placeholder_token)
                    token_embeds[token_id] = subj_embs[idx]
                    encoder_updated_tokens.append(placeholder_token)
                    idx += 1

                all_encoders_updated_tokens.extend(encoder_updated_tokens)
                all_encoders_updated_token_strs.append(" ".join(encoder_updated_tokens))

            self.updated_tokens_str = " ".join(all_encoders_updated_token_strs)
            self.all_encoders_updated_token_strs = all_encoders_updated_token_strs
            print(f"Updated {len(all_encoders_updated_tokens)} tokens ({self.updated_tokens_str}) in the text encoder.")

    def update_prompt(self, prompt, placeholder_tokens_pos='append',
                      repeat_prompt_for_each_encoder=True,
                      use_null_placeholders=False):
        if prompt is None:
            prompt = ""

        if use_null_placeholders:
            all_placeholder_tokens_str = self.all_null_placeholder_tokens_str
            if not re.search(r"\b(man|woman|person|child|girl|boy)\b", prompt.lower()):
                all_placeholder_tokens_str = "person " + all_placeholder_tokens_str
            repeat_prompt_for_each_encoder = False
        else:
            all_placeholder_tokens_str = self.updated_tokens_str

        # Delete the subject_string from the prompt.
        prompt = re.sub(r'\b(a|an|the)\s+' + self.subject_string + r'\b,?', "", prompt)
        prompt = re.sub(r'\b' + self.subject_string + r'\b,?',              "", prompt)
        # Prevously, arc2face ada prompts work better if they are prepended to the prompt,
        # and consistentID ada prompts work better if they are appended to the prompt.
        # When we do joint training, seems both work better if they are appended to the prompt.
        # Therefore we simply appended all placeholder_tokens_str's to the prompt.
        # NOTE: Prepending them hurts compositional prompts.
        if repeat_prompt_for_each_encoder:
            encoder_prompts = []
            for encoder_updated_token_strs in self.all_encoders_updated_token_strs:
                if placeholder_tokens_pos == 'prepend':
                    encoder_prompt = encoder_updated_token_strs + " " + prompt
                elif placeholder_tokens_pos == 'append':
                    encoder_prompt = prompt + " " + encoder_updated_token_strs
                else:
                    breakpoint()
                encoder_prompts.append(encoder_prompt)
            prompt = ", ".join(encoder_prompts)
        else:
            if placeholder_tokens_pos == 'prepend':
                prompt = all_placeholder_tokens_str + " " + prompt
            elif placeholder_tokens_pos == 'append':
                prompt = prompt + " " + all_placeholder_tokens_str
            else:
                breakpoint()

        return prompt

    # If face_id_embs is None, then it extracts face_id_embs from the images,
    # then map them to ada prompt embeddings.
    # avg_at_stage: 'id_emb', 'img_prompt_emb', or None.
    # avg_at_stage == ada_prompt_emb usually produces the worst results.
    # id_emb is slightly better than img_prompt_emb, but sometimes img_prompt_emb is better.
    def prepare_adaface_embeddings(self, image_paths, face_id_embs=None, 
                                   avg_at_stage='id_emb', # id_emb, img_prompt_emb, ada_prompt_emb, or None.
                                   perturb_at_stage=None, # id_emb, img_prompt_emb, or None.
                                   perturb_std=0, update_text_encoder=True):

        all_adaface_subj_embs, lens_subj_emb_segments = \
            self.id2ada_prompt_encoder.generate_adaface_embeddings(\
                    image_paths, face_id_embs=face_id_embs, 
                    img_prompt_embs=None, 
                    avg_at_stage=avg_at_stage,
                    perturb_at_stage=perturb_at_stage,
                    perturb_std=perturb_std,
                    enable_static_img_suffix_embs=self.enable_static_img_suffix_embs)
        
        if all_adaface_subj_embs is None:
            return None
        
        if all_adaface_subj_embs.ndim == 4:
            # [1, 1, 20, 768] -> [20, 768]
            all_adaface_subj_embs = all_adaface_subj_embs.squeeze(0).squeeze(0)
        elif all_adaface_subj_embs.ndim == 3:
            # [1, 20, 768] -> [20, 768]
            all_adaface_subj_embs = all_adaface_subj_embs.squeeze(0)

        if update_text_encoder:
            self.update_text_encoder_subj_embeddings(all_adaface_subj_embs, lens_subj_emb_segments)
        return all_adaface_subj_embs

    def diffusers_encode_prompts(self, prompt, plain_prompt, negative_prompt, device):
        # pooled_prompt_embeds_, negative_pooled_prompt_embeds_ are used by text2img3 and flux.
        pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = None, None

        # Compatible with older versions of diffusers.
        if not hasattr(self.pipeline, "encode_prompt"):
            # prompt_embeds_, negative_prompt_embeds_: [77, 768] -> [1, 77, 768].
            prompt_embeds_, negative_prompt_embeds_ = \
                self.pipeline._encode_prompt(prompt, device=device, num_images_per_prompt=1,
                                             do_classifier_free_guidance=True, 
                                             negative_prompt=negative_prompt)
            prompt_embeds_ = prompt_embeds_.unsqueeze(0)
            negative_prompt_embeds_ = negative_prompt_embeds_.unsqueeze(0)
        else:
            if self.pipeline_name in ["text2imgxl", "text2img3", "flux"]:
                prompt_2 = plain_prompt
                # CLIP Text Encoder prompt uses a maximum sequence length of 77.
                # T5 Text Encoder prompt uses a maximum sequence length of 256.
                # 333 = 256 + 77.
                prompt_t5 = prompt + "".join([", "] * 256)

                # prompt_embeds_, negative_prompt_embeds_: [1, 333, 4096]
                # pooled_prompt_embeds_, negative_pooled_prompt_embeds_: [1, 2048]
                if self.pipeline_name == "text2imgxl":
                    prompt_embeds_, negative_prompt_embeds_, \
                    pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = \
                        self.pipeline.encode_prompt(prompt, prompt_2, device=device, 
                                                    num_images_per_prompt=1, 
                                                    do_classifier_free_guidance=True,
                                                    negative_prompt=negative_prompt)        
                elif self.pipeline_name == "text2img3":
                    prompt_embeds_, negative_prompt_embeds_, \
                    pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = \
                        self.pipeline.encode_prompt(prompt, prompt_2, prompt_t5, device=device, 
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
                # "text2img" and "img2img" pipelines.
                # prompt_embeds_, negative_prompt_embeds_: [1, 77, 768]
                prompt_embeds_, negative_prompt_embeds_ = \
                    self.pipeline.encode_prompt(prompt, device=device, 
                                                num_images_per_prompt=1, 
                                                do_classifier_free_guidance=True,
                                                negative_prompt=negative_prompt)
    
        return prompt_embeds_, negative_prompt_embeds_, \
               pooled_prompt_embeds_, negative_pooled_prompt_embeds_
    
    def encode_prompt(self, prompt, negative_prompt=None, 
                      placeholder_tokens_pos='append',
                      ablate_prompt_only_placeholders=False,
                      ablate_prompt_no_placeholders=False,
                      repeat_prompt_for_each_encoder=True,
                      device=None, verbose=False):
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        
        if device is None:
            device = self.device
        
        plain_prompt = prompt
        if ablate_prompt_only_placeholders:
            prompt = self.updated_tokens_str
        else:
            prompt = self.update_prompt(prompt, placeholder_tokens_pos=placeholder_tokens_pos,
                                        repeat_prompt_for_each_encoder=repeat_prompt_for_each_encoder,
                                        use_null_placeholders=ablate_prompt_no_placeholders)

        if verbose:
            print(f"Subject prompt:\n{prompt}")

        # For some unknown reason, the text_encoder is still on CPU after self.pipeline.to(self.device).
        # So we manually move it to GPU here.
        self.pipeline.text_encoder.to(device)

        prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = \
            self.diffusers_encode_prompts(prompt, plain_prompt, negative_prompt, device)

        return prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds_, negative_pooled_prompt_embeds_
    
    # ref_img_strength is used only in the img2img pipeline.
    def forward(self, noise, prompt, prompt_embeds=None, negative_prompt=None, 
                placeholder_tokens_pos='append',
                guidance_scale=6.0, out_image_count=4, 
                ref_img_strength=0.8, generator=None, 
                ablate_prompt_only_placeholders=False,
                ablate_prompt_no_placeholders=False,
                repeat_prompt_for_each_encoder=True,                
                verbose=False):
        noise = noise.to(device=self.device, dtype=torch.float16)
        if self.use_lcm:
            guidance_scale = 0

        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        # prompt_embeds_, negative_prompt_embeds_: [1, 77, 768]
        if prompt_embeds is None:
            prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds_, \
                negative_pooled_prompt_embeds_ = \
                    self.encode_prompt(prompt, negative_prompt, 
                                       placeholder_tokens_pos=placeholder_tokens_pos,
                                       ablate_prompt_only_placeholders=ablate_prompt_only_placeholders,
                                       ablate_prompt_no_placeholders=ablate_prompt_no_placeholders,
                                       repeat_prompt_for_each_encoder=repeat_prompt_for_each_encoder,
                                       device=self.device, 
                                       verbose=verbose)
        else:   
            if len(prompt_embeds) == 2:
                prompt_embeds_, negative_prompt_embeds_ = prompt_embeds
                pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = None, None
            elif len(prompt_embeds) == 4:
                prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds_, \
                    negative_pooled_prompt_embeds_ = prompt_embeds
            else:
                breakpoint()

        # Repeat the prompt embeddings for all images in the batch.
        prompt_embeds_ = prompt_embeds_.repeat(out_image_count, 1, 1)

        if negative_prompt_embeds_ is not None:
            negative_prompt_embeds_ = negative_prompt_embeds_.repeat(out_image_count, 1, 1)

        if self.pipeline_name in ["text2imgxl", "text2img3"]:
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
    