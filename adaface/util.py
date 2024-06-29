import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

# add_noise_to_tensor() adds a fixed amount of noise to the tensor.
def add_noise_to_tensor(ts, noise_std, noise_std_is_relative=True, keep_norm=False,
                        std_dim=-1, norm_dim=-1):
    if noise_std_is_relative:
        ts_std_mean = ts.std(dim=std_dim).mean().detach()
        noise_std *= ts_std_mean

    noise = torch.randn_like(ts) * noise_std
    if keep_norm:
        orig_norm = ts.norm(dim=norm_dim, keepdim=True)
        ts = ts + noise
        new_norm  = ts.norm(dim=norm_dim, keepdim=True).detach()
        ts = ts * orig_norm / (new_norm + 1e-8)
    else:
        ts = ts + noise
        
    return ts


# Revised from RevGrad, by removing the grad negation.
class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_, debug=False):
        ctx.save_for_backward(alpha_, debug)
        output = input_
        if debug:
            print(f"input: {input_.abs().mean().item()}")
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        # saved_tensors returns a tuple of tensors.
        alpha_, debug = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_output2 = grad_output * alpha_
            if debug:
                print(f"grad_output2: {grad_output2.abs().mean().item()}")
        else:
            grad_output2 = None
        return grad_output2, None, None

class GradientScaler(nn.Module):
    def __init__(self, alpha=1., debug=False, *args, **kwargs):
        """
        A gradient scaling layer.
        This layer has no parameters, and simply scales the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)
        self._debug = torch.tensor(debug, requires_grad=False)

    def forward(self, input_):
        _debug = self._debug if hasattr(self, '_debug') else False
        return ScaleGrad.apply(input_, self._alpha.to(input_.device), _debug)

def gen_gradient_scaler(alpha, debug=False):
    if alpha == 1:
        return nn.Identity()
    if alpha > 0:
        return GradientScaler(alpha, debug=debug)
    else:
        assert alpha == 0
        # Don't use lambda function here, otherwise the object can't be pickled.
        return torch.detach

#@torch.autocast(device_type="cuda")
# In AdaFaceWrapper, input_max_length is 22.
def arc2face_forward_face_embs(tokenizer, arc2face_text_encoder, face_embs, 
                               input_max_length=77, return_full_and_core_embs=True):

    '''
    arc2face_text_encoder: arc2face_models.py CLIPTextModelWrapper instance.
    face_embs: (N, 512) normalized ArcFace embeddings.
    return_full_and_core_embs: Return both the full prompt embeddings and the core embeddings. 
                               If False, return only the core embeddings.

    '''

    # arcface_token_id: 1014
    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]

    # This step should be quite fast, and there's no need to cache the input_ids.
    input_ids = tokenizer(
            "photo of a id person",
            truncation=True,
            padding="max_length",
            max_length=input_max_length, #tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(face_embs.device)
    # input_ids: [1, 77] or [3, 77] (during training).
    input_ids = input_ids.repeat(len(face_embs), 1)
    face_embs_dtype = face_embs.dtype
    face_embs = face_embs.to(arc2face_text_encoder.dtype)
    # face_embs_padded: [1, 512] -> [1, 768].
    face_embs_padded = F.pad(face_embs, (0, arc2face_text_encoder.config.hidden_size - face_embs.shape[-1]), "constant", 0)
    # arc2face_text_encoder(input_ids=input_ids, ...) is called twice. The first is only to get the token embeddings (the shallowest mapping).
    # The second call does the ordinary CLIP text encoding pass.
    token_embs = arc2face_text_encoder(input_ids=input_ids, return_token_embs=True)
    token_embs[input_ids==arcface_token_id] = face_embs_padded

    prompt_embeds = arc2face_text_encoder(
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

def get_b_core_e_embeddings(prompt_embeds, length=22):
    b_core_e_embs = torch.cat([ prompt_embeds[:, :length], prompt_embeds[:, [-1]] ], dim=1)
    return b_core_e_embs

# return_emb_types: a list of strings, each string is among ['full', 'core', 'full_zeroed_extra', 'b_core_e'].
def arc2face_inverse_face_prompt_embs(clip_tokenizer, inverse_text_encoder, face_prompt_embs, list_extra_words,
                                      return_emb_types, pad_embeddings, hidden_state_layer_weights=None, 
                                      input_max_length=77, zs_extra_words_scale=0.5):

    '''
    inverse_text_encoder: arc2face_models.py CLIPTextModelWrapper instance with **custom weights**.
    inverse_text_encoder is NOT the original arc2face text encoder, but retrained to do inverse mapping.
    face_prompt_embs: (BS, 16, 768). Only the core embeddings, no paddings.
    list_extra_words: [s_1, ..., s_BS], each s_i is a list of extra words to be added to the prompt.
    return_full_and_core_embs: Return both the full prompt embeddings and the core embeddings. 
                               If False, return only the core embeddings.
    '''

    if list_extra_words is not None:
        if len(list_extra_words) != len(face_prompt_embs):
            if len(face_prompt_embs) > 1:
                print("Warn: list_extra_words has different length as face_prompt_embs.")
                if len(list_extra_words) == 1:
                    list_extra_words = list_extra_words * len(face_prompt_embs)
                else:
                    breakpoint()
            else:
                # len(face_prompt_embs) == 1, this occurs when same_subject_in_batch == True, e.g. in do_mix_prompt_distillation.
                # But list_extra_words always corresponds to the actual batch size. So we only take the first element.
                list_extra_words = list_extra_words[:1]
                
        for extra_words in list_extra_words:
            assert len(extra_words.split()) <= 2, "Each extra_words string should consist of at most 2 words."
        # 16 ", " are placeholders for face_prompt_embs.
        prompt_templates = [ "photo of a " + ", " * 16 + list_extra_words[i] for i in range(len(list_extra_words)) ]
    else:
        # 16 ", " are placeholders for face_prompt_embs.
        # No extra words are added to the prompt.
        prompt_templates = [ "photo of a " + ", " * 16 for _ in range(len(face_prompt_embs)) ]

    # This step should be quite fast, and there's no need to cache the input_ids.
    # input_ids: [BS, 77].
    input_ids = clip_tokenizer(
            prompt_templates,
            truncation=True,
            padding="max_length",
            max_length=input_max_length,
            return_tensors="pt",
        ).input_ids.to(face_prompt_embs.device)

    face_prompt_embs_dtype  = face_prompt_embs.dtype
    face_prompt_embs        = face_prompt_embs.to(inverse_text_encoder.dtype)

    # token_embs: [1, 77, 768]. This call is only to get the template token embeddings (the shallowest mapping).
    token_embs = inverse_text_encoder(input_ids=input_ids, return_token_embs=True)
    # token 4: first ", " in the template prompt.
    # Replace embeddings of 16 placeholder ", " with face_prompt_embs.
    token_embs[:, 4:20] = face_prompt_embs

    # This call does the ordinary CLIP text encoding pass.
    prompt_embeds = inverse_text_encoder(
        input_ids=input_ids,
        input_token_embs=token_embs,
        hidden_state_layer_weights=hidden_state_layer_weights,
        return_token_embs=False
    )[0]

    # Restore the original dtype of prompt_embeds: float16 -> float32.
    prompt_embeds = prompt_embeds.to(face_prompt_embs_dtype)
    # token 4: first ", " in the template prompt.
    # 4:20 are the most important 16 embeddings that contain the subject's identity.
    # 20:22 are embeddings of the (at most) two extra words.
    # [N, 77, 768] -> [N, 16, 768]
    core_prompt_embs = prompt_embeds[:, 4:20]
    if list_extra_words is not None:
        # [N, 16, 768] -> [N, 18, 768]
        extra_words_embs = prompt_embeds[:, 20:22] * zs_extra_words_scale
        core_prompt_embs = torch.cat([core_prompt_embs, extra_words_embs], dim=1)

    return_prompts = []
    for emb_type in return_emb_types:
        if emb_type == 'full':
            return_prompts.append(prompt_embeds)
        elif emb_type == 'full_half_pad':
            prompt_embeds2 = prompt_embeds.clone()
            PADS  = prompt_embeds2.shape[1] - 23
            if PADS >= 2:
                # Fill half of the remaining embeddings with pad embeddings.
                prompt_embeds2[:, 22:22+PADS//2] = pad_embeddings[22:22+PADS//2]
            return_prompts.append(prompt_embeds2)
        elif emb_type == 'full_pad':
            prompt_embeds2 = prompt_embeds.clone()
            # Fill the 22nd to the second last embeddings with pad embeddings.
            prompt_embeds2[:, 22:-1] = pad_embeddings[22:-1]
            return_prompts.append(prompt_embeds2)
        elif emb_type == 'core':
            return_prompts.append(core_prompt_embs)
        elif emb_type == 'full_zeroed_extra':
            prompt_embeds2 = prompt_embeds.clone()
            # Only add two pad embeddings. The remaining embeddings are set to 0.
            # Make the positional embeddings align with the actual positions.
            prompt_embeds2[:, 22:24] = pad_embeddings[22:24]
            prompt_embeds2[:, 24:-1] = 0
            return_prompts.append(prompt_embeds2)
        elif emb_type == 'b_core_e':
            # The first 22 embeddings, plus the last EOS embedding.
            b_core_e_embs = get_b_core_e_embeddings(prompt_embeds, length=22)
            return_prompts.append(b_core_e_embs)
        else:
            breakpoint()

    return return_prompts

# if pre_face_embs is None, generate random face embeddings [BS, 512].
# image_folder is passed only for logging purpose. image_paths contains the paths of the images.
def get_arc2face_id_prompt_embs(face_app, clip_tokenizer, arc2face_text_encoder, 
                                extract_faceid_embeds, pre_face_embs, 
                                image_folder, image_paths, images_np,
                                id_batch_size, device, 
                                input_max_length=77, noise_level=0.0, 
                                return_core_id_embs=False,
                                gen_neg_prompt=False, verbose=False):
    face_image_count = 0

    if extract_faceid_embeds:
        faceid_embeds = []
        if image_paths is not None:
            images_np = []
            for image_path in image_paths:
                image_np = np.array(Image.open(image_path))
                images_np.append(image_np)

        for i, image_np in enumerate(images_np):
            image_obj = Image.fromarray(image_np).resize((512, 512), Image.NEAREST)
            # Remove alpha channel if it exists.
            if image_obj.mode == 'RGBA':
                image_obj = image_obj.convert('RGB')
            # This seems NOT a bug. The input image should be in BGR format, as per 
            # https://github.com/deepinsight/insightface/issues/524
            image_np = cv2.cvtColor(np.array(image_obj), cv2.COLOR_RGB2BGR)
            image_np = np.array(image_obj)

            face_infos = face_app.get(image_np)
            if verbose and image_paths is not None:
                print(image_paths[i], len(face_infos))
            # Assume all images belong to the same subject. Therefore, we can skip the images with no face detected.
            if len(face_infos) == 0:
                continue
            # only use the maximum face
            face_info = sorted(face_infos, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
            # Each faceid_embed: [1, 512]
            faceid_embeds.append(torch.from_numpy(face_info.normed_embedding).unsqueeze(0))
            face_image_count += 1

        if verbose:
            if image_folder is not None:
                print(f"Extracted ID embeddings from {face_image_count} images in {image_folder}")
            else:
                print(f"Extracted ID embeddings from {face_image_count} images")

        if len(faceid_embeds) == 0:
            print("No face detected. Use a random face instead.")
            faceid_embeds = torch.randn(id_batch_size, 512).to(device=device, dtype=torch.float16)
        else:
            # faceid_embeds: [10, 512]
            faceid_embeds = torch.cat(faceid_embeds, dim=0)
            # faceid_embeds: [10, 512] -> [1, 512].
            # and the resulted prompt embeddings are the same.
            faceid_embeds = faceid_embeds.mean(dim=0, keepdim=True).to(device=device, dtype=torch.float16)
    else:
        # Random face embeddings. faceid_embeds: [BS, 512].
        if pre_face_embs is None:
            faceid_embeds = torch.randn(id_batch_size, 512)
        else:
            faceid_embeds = pre_face_embs
            if pre_face_embs.shape[0] == 1:
                faceid_embeds = faceid_embeds.repeat(id_batch_size, 1)

        faceid_embeds = faceid_embeds.to(device=device, dtype=torch.float16)

    if noise_level > 0:
        # If id_batch_size > 1, after adding noises, the id_batch_size embeddings will be different.
        faceid_embeds = add_noise_to_tensor(faceid_embeds, noise_level, noise_std_is_relative=True, keep_norm=True)

    faceid_embeds = F.normalize(faceid_embeds, p=2, dim=-1)

    # arc2face_pos_prompt_emb, arc2face_neg_prompt_emb: [BS, 77, 768]
    with torch.no_grad():
        arc2face_pos_prompt_emb, arc2face_pos_core_prompt_emb  = \
             arc2face_forward_face_embs(clip_tokenizer, arc2face_text_encoder, 
                                        faceid_embeds, input_max_length=input_max_length,
                                        return_full_and_core_embs=True)
        if return_core_id_embs:
            arc2face_pos_prompt_emb = arc2face_pos_core_prompt_emb
    # If extract_faceid_embeds, we assume all images are from the same subject, and the batch dim of faceid_embeds is 1. 
    # So we need to repeat faceid_embeds.
    if extract_faceid_embeds:
        faceid_embeds = faceid_embeds.repeat(id_batch_size, 1)
        arc2face_pos_prompt_emb = arc2face_pos_prompt_emb.repeat(id_batch_size, 1, 1)

    if gen_neg_prompt:
        with torch.no_grad():
            arc2face_neg_prompt_emb, arc2face_neg_core_prompt_emb = \
                arc2face_forward_face_embs(clip_tokenizer, arc2face_text_encoder, 
                                           torch.zeros_like(faceid_embeds),
                                           input_max_length=input_max_length,
                                           return_full_and_core_embs=True)
            if return_core_id_embs:
                arc2face_neg_prompt_emb = arc2face_neg_core_prompt_emb

        #if extract_faceid_embeds:
        #    arc2face_neg_prompt_emb = arc2face_neg_prompt_emb.repeat(id_batch_size, 1, 1)
        return face_image_count, faceid_embeds, arc2face_pos_prompt_emb, arc2face_neg_prompt_emb
    else:
        return face_image_count, faceid_embeds, arc2face_pos_prompt_emb
    