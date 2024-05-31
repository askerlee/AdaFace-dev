import torch
import torch.nn as nn
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    AutoencoderKL,
)
from insightface.app import FaceAnalysis
from ldm.modules.arc2face_models import CLIPTextModelWrapper
import re, os
from ldm.util import get_arc2face_id_prompt_embs

class AdaFaceWrapper(nn.Module):
    def __init__(self, base_model_path, embman_ckpt, subject_string, num_vectors, device, 
                 num_inference_steps=50, guidance_scale=4.0, negative_prompt=None):
        super().__init__()
        self.base_model_path = base_model_path
        self.embman_ckpt = embman_ckpt
        self.subject_string = subject_string
        self.num_vectors = num_vectors
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
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

    def initialize_pipeline(self):
        ckpt = torch.load(self.embman_ckpt, map_location=self.device)
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

        # arc2face_text_encoder maps the face analysis embedding to 16 face embeddings 
        # in the UNet image space.
        arc2face_text_encoder = CLIPTextModelWrapper.from_pretrained(
            'models/arc2face', subfolder="encoder", torch_dtype=torch.float16
        )
        self.arc2face_text_encoder = arc2face_text_encoder.to(self.device)

        # The 840000-step vae model is slightly better in face details than the original vae model.
        # https://huggingface.co/stabilityai/sd-vae-ft-mse-original
        vae = AutoencoderKL.from_single_file("models/diffusers/sd-vae-ft-mse-original/vae-ft-mse-840000-ema-pruned.ckpt", torch_dtype=torch.float16)
        # The dreamshaper v7 finetuned text encoder follows the prompt slightly better than the original text encoder.
        # https://huggingface.co/Lykon/DreamShaper/tree/main/text_encoder
        text_encoder = CLIPTextModel.from_pretrained("models/diffusers/ds_text_encoder", torch_dtype=torch.float16)
        if os.path.isfile(self.base_model_path):
            pipeline = StableDiffusionPipeline.from_single_file(
                self.base_model_path, 
                text_encoder=text_encoder, 
                vae=vae, 
                torch_dtype=torch.float16
                )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                    self.base_model_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    torch_dtype=torch.float16,
                    safety_checker=None
                )

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
        # FaceAnalysis will try to find the ckpt in: models/arc2face/models/antelopev2. 
        # Note the second "model" in the path.
        self.face_app = FaceAnalysis(name='antelopev2', root='models/arc2face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(512, 512))
        # Patch the missing tokenizer in the subj_basis_generator.
        if not hasattr(self.subj_basis_generator, 'clip_tokenizer'):
            self.subj_basis_generator.clip_tokenizer = pipeline.tokenizer
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

        self.placeholder_token_ids = tokenizer.convert_tokens_to_ids(self.placeholder_tokens)
        # print(self.placeholder_token_ids)
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.pipeline.text_encoder.resize_token_embeddings(len(tokenizer))

    # Extend pipeline.text_encoder with the adaface subject emeddings.
    # subj_embs: [16, 768].
    def update_text_encoder_subj_embs(self, subj_embs):
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.pipeline.text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for i, token_id in enumerate(self.placeholder_token_ids):
                token_embeds[token_id] = subj_embs[i]
            print(f"Updated {len(self.placeholder_token_ids)} tokens in the text encoder.")

    def update_prompt(self, prompt):
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

    def generate_adaface_embeddings(self, image_folder, image_paths, pre_face_embs, gen_rand_face, 
                                    noise_level, update_text_encoder=True):
        # faceid_embeds is a batch of extracted face analysis embeddings (BS * 512).
        # If extract_faceid_embeds is True, faceid_embeds is an embedding repeated by BS times.
        # Otherwise, faceid_embeds is a batch of out_image_count random embeddings, different from each other.
        # The same applies to id_prompt_emb.
        # id_prompt_emb is in the image prompt space.
        # faceid_embeds: [1, 512]. NOT used later.
        # id_prompt_emb: [1, 16, 768]. 
        # Since return_core_id_embs is True, id_prompt_emb is only the 16 core ID embeddings.
        # arc2face prompt template: "photo of a id person"
        # ID embeddings start from "id person ...". So there are 3 template tokens before the 16 ID embeddings.
        faceid_embeds, id_prompt_emb \
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
                                        input_max_length=22,
                                        noise_level=noise_level,
                                        return_core_id_embs=True,
                                        gen_neg_prompt=False, 
                                        verbose=True)
        
        # adaface_subj_embs: [1, 1, 16, 768]. 
        # adaface_prompt_embs: [1, 77, 768] (not used).
        adaface_subj_embs, adaface_prompt_embs = \
            self.subj_basis_generator(id_prompt_emb, None, None,
                                      is_face=True, is_training=False,
                                      adaface_prompt_embs_inf_type='full_half_pad')
        # adaface_subj_embs: [16, 768]
        adaface_subj_embs = adaface_subj_embs.squeeze()
        if update_text_encoder:
            self.update_text_encoder_subj_embs(adaface_subj_embs)
        return adaface_subj_embs

    def forward(self, noise, prompt, out_image_count=4, verbose=True):
        prompt = self.update_prompt(prompt)
        if verbose:
            print(f"Prompt: {prompt}")

        # noise: [BS, 4, 64, 64]
        # prompt_embeds_, negative_prompt_embeds_: [4, 77, 768]
        prompt_embeds_, negative_prompt_embeds_ = \
            self.pipeline.encode_prompt(prompt, device=self.device, num_images_per_prompt = out_image_count,
                                        do_classifier_free_guidance=True, negative_prompt=self.negative_prompt)

        images = self.pipeline(image=noise,
                               prompt_embeds=prompt_embeds_, 
                               negative_prompt_embeds=negative_prompt_embeds_, 
                               num_inference_steps=self.num_inference_steps, 
                               guidance_scale=self.guidance_scale, 
                               num_images_per_prompt=1).images
        # images: [BS, 3, 512, 512]
        return images
    