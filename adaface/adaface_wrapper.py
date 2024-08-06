import torch
import torch.nn as nn
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusion3Pipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from insightface.app import FaceAnalysis
from adaface.arc2face_models import CLIPTextModelWrapper
from adaface.util import get_arc2face_id_prompt_embs, UNetEnsemble
import re, os
import sys
sys.modules['ldm'] = sys.modules['adaface']

class AdaFaceWrapper(nn.Module):
    def __init__(self, pipeline_name, base_model_path, adaface_ckpt_path, device,
                 subject_string='z', num_vectors=16, 
                 num_inference_steps=50, negative_prompt=None,
                 use_840k_vae=False, use_ds_text_encoder=False, 
                 extra_unet_paths=None, unet_weights=None,
                 is_training=False):
        '''
        pipeline_name: "text2img" or "img2img" or None. If None, the unet and vae are
        removed from the pipeline to release RAM.
        '''
        super().__init__()
        self.pipeline_name = pipeline_name
        self.base_model_path = base_model_path
        self.adaface_ckpt_path = adaface_ckpt_path
        self.use_840k_vae = use_840k_vae
        self.use_ds_text_encoder = use_ds_text_encoder
        self.subject_string = subject_string
        self.num_vectors = num_vectors
        self.num_inference_steps = num_inference_steps
        self.extra_unet_paths = extra_unet_paths
        self.unet_weights = unet_weights
        self.device = device
        self.is_training = is_training
        self.initialize_pipeline()
        self.extend_tokenizer_and_text_encoder()
        if negative_prompt is None:
            self.negative_prompt = \
            "flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, " \
            "mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, " \
            "mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, " \
            "nude, naked, nsfw, topless, bare breasts"
        else:
            self.negative_prompt = negative_prompt

    def load_subj_basis_generator(self, adaface_ckpt_path):
        ckpt = torch.load(adaface_ckpt_path, map_location='cpu')
        string_to_subj_basis_generator_dict = ckpt["string_to_subj_basis_generator_dict"]
        if self.subject_string not in string_to_subj_basis_generator_dict:
            print(f"Subject '{self.subject_string}' not found in the embedding manager.")
            breakpoint()

        self.subj_basis_generator = string_to_subj_basis_generator_dict[self.subject_string]
        # In the original ckpt, num_out_layers is 16 for layerwise embeddings. 
        # But we don't do layerwise embeddings here, so we set it to 1.
        self.subj_basis_generator.num_out_layers = 1
        print(f"Loaded subject basis generator for '{self.subject_string}'.")
        print(repr(self.subj_basis_generator))
        self.subj_basis_generator.to(self.device)
        if self.is_training:
            self.subj_basis_generator.train()
        else:
            self.subj_basis_generator.eval()

    def initialize_pipeline(self):
        self.load_subj_basis_generator(self.adaface_ckpt_path)

        # arc2face_text_encoder maps the face analysis embedding to 16 face embeddings 
        # in the UNet image space.
        # arc2face_text_encoder always uses float16.
        arc2face_text_encoder = CLIPTextModelWrapper.from_pretrained(
            'models/arc2face', subfolder="encoder", torch_dtype=torch.float16
        )
        self.arc2face_text_encoder = arc2face_text_encoder.to(self.device)

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
        # pipeline_name is None means only use this instance to generate adaface embeddings, not to generate images.
        elif self.pipeline_name is None:
            PipelineClass = StableDiffusionPipeline
            remove_unet = True
        else:
            raise ValueError(f"Unknown pipeline name: {self.pipeline_name}")
        
        if self.base_model_path is None:
            base_model_path_dict = { 
                'text2img':  'runwayml/stable-diffusion-v1-5',
                'text2img3': 'stabilityai/stable-diffusion-3-medium-diffusers' 
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
        
        if self.extra_unet_paths is not None and len(self.extra_unet_paths) > 0:
            unet_ensemble = UNetEnsemble(pipeline.unet, self.extra_unet_paths, self.unet_weights,
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

        if self.pipeline_name != "text2img3":
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )

            pipeline.scheduler = noise_scheduler

        self.pipeline = pipeline.to(self.device)
        # FaceAnalysis will try to find the ckpt in: models/insightface/models/antelopev2. 
        # Note there's a second "model" in the path.
        self.face_app = FaceAnalysis(name='antelopev2', root='models/insightface', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(512, 512))
        # Patch the missing tokenizer in the subj_basis_generator.
        if not hasattr(self.subj_basis_generator, 'clip_tokenizer'):
            self.subj_basis_generator.clip_tokenizer = self.pipeline.tokenizer
            print("Patched the missing tokenizer in the subj_basis_generator.")

    def extend_tokenizer_and_text_encoder(self):
        if self.num_vectors < 1:
            raise ValueError(f"num_vectors has to be larger or equal to 1, but is {self.num_vectors}")

        tokenizer = self.pipeline.tokenizer
        # Add z0, z1, z2, ..., z15.
        self.placeholder_tokens = []
        for i in range(0, self.num_vectors):
            self.placeholder_tokens.append(f"{self.subject_string}_{i}")

        self.placeholder_tokens_str = " ".join(self.placeholder_tokens)

        # Add the new tokens to the tokenizer.
        num_added_tokens = tokenizer.add_tokens(self.placeholder_tokens)
        if num_added_tokens != self.num_vectors:
            raise ValueError(
                f"The tokenizer already contains the token {self.subject_string}. Please pass a different"
                " `subject_string` that is not already in the tokenizer.")

        print(f"Added {num_added_tokens} tokens ({self.placeholder_tokens_str}) to the tokenizer.")
        
        # placeholder_token_ids: [49408, ..., 49423].
        self.placeholder_token_ids = tokenizer.convert_tokens_to_ids(self.placeholder_tokens)
        # print(self.placeholder_token_ids)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        old_weight = self.pipeline.text_encoder.get_input_embeddings().weight
        self.pipeline.text_encoder.resize_token_embeddings(len(tokenizer))
        new_weight = self.pipeline.text_encoder.get_input_embeddings().weight
        print(f"Resized text encoder token embeddings from {old_weight.shape} to {new_weight.shape} on {new_weight.device}.")

    # Extend pipeline.text_encoder with the adaface subject emeddings.
    # subj_embs: [16, 768].
    def update_text_encoder_subj_embs(self, subj_embs):
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for i, token_id in enumerate(self.placeholder_token_ids):
                token_embeds[token_id] = subj_embs[i]
            print(f"Updated {len(self.placeholder_token_ids)} tokens ({self.placeholder_tokens_str}) in the text encoder.")

    def update_prompt(self, prompt):
        if prompt is None:
            prompt = ""
            
        # If the placeholder tokens are already in the prompt, then return the prompt as is.
        if self.placeholder_tokens_str in prompt:
            return prompt
        
        # If the subject string 'z' is not in the prompt, then simply prepend the placeholder tokens to the prompt.
        if re.search(r'\b' + self.subject_string + r'\b', prompt) is None:
            print(f"Subject string '{self.subject_string}' not found in the prompt. Adding it.")
            comp_prompt = self.placeholder_tokens_str + " " + prompt
        else:
            # Replace the subject string 'z' with the placeholder tokens.
            comp_prompt = re.sub(r'\b' + self.subject_string + r'\b', self.placeholder_tokens_str, prompt)
        return comp_prompt

    # image_paths: a list of image paths. image_folder: the parent folder name.
    def generate_adaface_embeddings(self, image_paths, image_folder=None, 
                                    pre_face_embs=None, gen_rand_face=False, 
                                    out_id_embs_scale=1., noise_level=0, noise_std_to_input=0, 
                                    update_text_encoder=True):
        # faceid_embeds is a batch of extracted face analysis embeddings (BS * 512 = id_batch_size * 512).
        # If extract_faceid_embeds is True, faceid_embeds is *the same* embedding repeated by id_batch_size times.
        # Otherwise, faceid_embeds is a batch of random embeddings, each instance is different.
        # The same applies to id_prompt_emb.
        # faceid_embeds is in the face analysis embeddings. id_prompt_emb is in the image prompt space.
        # Here id_batch_size = 1, so
        # faceid_embeds: [1, 512]. NOT used later.
        # id_prompt_emb: [1, 16, 768]. 
        # NOTE: Since return_core_id_embs is True, id_prompt_emb is only the 16 core ID embeddings.
        # arc2face prompt template: "photo of a id person"
        # ID embeddings start from "id person ...". So there are 3 template tokens before the 16 ID embeddings.
        face_image_count, faceid_embeds, id_prompt_emb \
            = get_arc2face_id_prompt_embs(self.face_app, self.pipeline.tokenizer, self.arc2face_text_encoder,
                                          extract_faceid_embeds=not gen_rand_face,
                                          pre_face_embs=pre_face_embs,
                                          # image_folder is passed only for logging purpose. 
                                          # image_paths contains the paths of the images.
                                          image_folder=image_folder, image_paths=image_paths,
                                          images_np=None,
                                          id_batch_size=1,
                                          device=self.device,
                                          # input_max_length == 22: only keep the first 22 tokens, 
                                          # including 3 template tokens and 16 ID tokens, and BOS and EOS tokens.
                                          # The results are indistinguishable from input_max_length=77.
                                          input_max_length=22,
                                          noise_level=noise_level,
                                          noise_std_to_input=noise_std_to_input,
                                          return_core_id_embs=True,
                                          avg_at_stage=None,
                                          gen_neg_prompt=False, 
                                          verbose=True)
        
        if face_image_count == 0:
            return None
        
        # adaface_subj_embs: [1, 1, 16, 768]. 
        # adaface_prompt_embs: [1, 77, 768] (not used).
        adaface_subj_embs, adaface_prompt_embs = \
            self.subj_basis_generator(id_prompt_emb, None, None, 
                                      out_id_embs_scale=out_id_embs_scale,
                                      is_face=True, is_training=False,
                                      adaface_prompt_embs_inf_type='full_half_pad')
        # adaface_subj_embs: [N, 16, 768] -> [16, 768]
        adaface_subj_embs = adaface_subj_embs.mean(dim=0).squeeze(0)
        if update_text_encoder:
            self.update_text_encoder_subj_embs(adaface_subj_embs)
        return adaface_subj_embs

    def encode_prompt(self, prompt, negative_prompt=None, device=None, verbose=False):
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        
        if device is None:
            device = self.device
        
        prompt_ = prompt
        prompt = self.update_prompt(prompt)
        if verbose:
            print(f"Prompt: {prompt}")

        # For some unknown reason, the text_encoder is still on CPU after self.pipeline.to(self.device).
        # So we manually move it to GPU here.
        self.pipeline.text_encoder.to(device)
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
            if self.pipeline_name == "text2img3":
                # prompt_embeds_, negative_prompt_embeds_: [1, 333, 4096]
                # pooled_prompt_embeds_, negative_pooled_prompt_embeds_: [1, 2048]
                # CLIP Text Encoder prompt uses a maximum sequence length of 77.
                # T5 Text Encoder prompt uses a maximum sequence length of 256.
                # 333 = 256 + 77.
                prompt_t5 = prompt_ + "".join([", "] * 256)
                prompt_embeds_, negative_prompt_embeds_, \
                pooled_prompt_embeds_, negative_pooled_prompt_embeds_ = \
                    self.pipeline.encode_prompt(prompt, prompt_, prompt_t5, device=device, 
                                                num_images_per_prompt=1, 
                                                do_classifier_free_guidance=True,
                                                negative_prompt=negative_prompt)
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
        prompt_embeds_          = prompt_embeds_.repeat(out_image_count, 1, 1)
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
        else:
            # noise: [BS, 4, 64, 64]
            # When the pipeline is text2img, strength is ignored.
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
    