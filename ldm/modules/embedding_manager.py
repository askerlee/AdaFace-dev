import torch
from torch import nn
import torch.distributed as dist
from einops import rearrange
from adaface.subj_basis_generator import SubjBasisGenerator
import sys
#sys.modules['ldm.modules']                      = sys.modules['adaface']
sys.modules['ldm.modules.subj_basis_generator'] = sys.modules['adaface.subj_basis_generator']
sys.modules['ldm.modules.arc2face_models']      = sys.modules['adaface.arc2face_models']
from adaface.face_id_to_ada_prompt import create_id2ada_prompt_encoder

import torch.nn.functional as F
import numpy as np

from ldm.util import extract_first_index_in_each_instance, \
                     anneal_perturb_embedding, \
                     get_clip_tokens_for_string, get_embeddings_for_clip_tokens, \
                     scan_cls_delta_strings, calc_init_word_embeddings
                     
from functools import partial
from collections import OrderedDict
import copy

# When debugging, make the printed tensors less messy.
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

def reg_loss(x, loss_type='l2', selector=None):
    if selector is not None:
        # If selector(x) is False, the gradient flow is cut off.
        x = x * selector(x).float()
    if loss_type == 'l1':
        return x.abs().mean()
    elif loss_type == 'l2':
        return (x * x).mean()
    else:
        breakpoint()

# text_embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
# = LatentDiffusion.cond_stage_model
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            text_embedder,   
            subject_strings,
            # If background_strings are specified, they are part of the list placeholder_strings.
            background_strings=None,
            subj_name_to_cls_delta_string=None,
            # token2num_vectors: how many vectors in each layer are allocated to model 
            # the subject (represented as the subject token) and the background. 
            # token2num_vectors is a dict.
            token2num_vectors={},
            skip_loading_token2num_vectors=False,
            use_layerwise_embedding=True,
            out_emb_dim=768,
            num_unet_ca_layers=16,
            layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            training_begin_perturb_std_range=None,
            training_end_perturb_std_range=None,
            training_perturb_prob=None,
            do_zero_shot=True,
            id2ada_prompt_encoder_type='arc2face',
            id2img_prompt_encoder_trainable=False,
            to_load_id2img_learnable_modules=False,
            subj_name_to_being_faces=None,   # subj_name_to_being_faces: a dict that maps subject names to is_face.
            zs_cls_delta_string='person',
            zs_cls_delta_token_weights=None,
            zs_prompt2token_proj_grad_scale=1,
            zs_extra_words_scale=0.5,
            # During inference, zs_prompt2token_proj_ext_attention_perturb_ratio is not specified. 
            # Therefore no perturbation during inference.
            zs_prompt2token_proj_ext_attention_perturb_ratio=0, 
            adaface_ckpt_path=None,
            extend_prompt2token_proj_attention_multiplier=1,
            to_load_old_adaface_ckpt=False,
            num_static_img_suffix_embs=0,
    ):
        super().__init__()

        self.rank = -1
        self.do_zero_shot = do_zero_shot

        self.string_to_token_dict = OrderedDict()
        
        self.string_to_subj_basis_generator_dict = nn.ModuleDict()
        self.placeholder_to_emb_cache            = nn.ParameterDict() # These should not be optimized
        self.use_layerwise_embedding = use_layerwise_embedding
        self.num_unet_ca_layers = num_unet_ca_layers
        if self.use_layerwise_embedding:
            # self.num_layers_per_embedder specifies the total layers of embeddings for each embedder.
            # There could be multiple embeddings for each layer.
            self.num_layers_per_embedder = num_unet_ca_layers
        else:
            self.num_layers_per_embedder = 1

        self.subject_strings = subject_strings
        if background_strings is not None:
            self.background_strings = list(background_strings)
        else:
            self.background_strings = []

        self.background_string_dict = { s: True for s in self.background_strings }
        self.placeholder_strings    = list(subject_strings) + self.background_strings
        self.subject_string_dict    = { s: True for s in self.subject_strings }

        self.set_training_perturb_specs(training_begin_perturb_std_range, 
                                        training_end_perturb_std_range,
                                        training_perturb_prob)

        self.layer_idx2ca_layer_idx = layer_idx2ca_layer_idx
        self.ca_layer_idx2layer_idx = { v: k for k, v in layer_idx2ca_layer_idx.items() }

        #                        1     2     4    5     7     8     12    16    
        self.ca_infeat_dims =  [ 320,  320,  640, 640, 1280, 1280, 1280, 1280, 
        #                        17    18    19   20    21    22    23    24                       
                                 1280, 1280, 640, 640, 640,  320,  320,  320 ]
        # ca_outfeat_dims are the same as ca_infeat_dims.

        # token2num_vectors: a dict. How many vectors in each layer 
        # are allocated to model the subject (represented as the subject token).        
        # token2num_vectors[S*] > 1:
        # *multi-vector subject embeddings*. In this space, S* is embedded into multiple 
        # learned embeddings, an approach that is equivalent to describing
        # the concept through multiple learned pseudo-words. 
        # This setting was proposed in the TI paper,
        # and AdaFace also supports it for more expressive modeling.
        self.skip_loading_token2num_vectors = skip_loading_token2num_vectors
        self.set_num_vectors_per_subj_token(token2num_vectors)
        self.out_emb_dim = out_emb_dim

        # Save this function to be used in load() when doing placeholder substitution.
        self.get_tokens_for_string       = partial(get_clip_tokens_for_string,       text_embedder.tokenizer)
        self.get_embeddings_for_tokens   = partial(get_embeddings_for_clip_tokens,   text_embedder.transformer.text_model.embeddings)

        # "," -> 267, "z": 345, "y": 344.
        self.subj_idx_to_cls_delta_tokens   = {}
        self.subj_idx_to_cls_delta_token_weights  = {}
        self.placeholder_token_to_idx       = {}

        if self.do_zero_shot:
            # No matter whether using layerwise embeddings, the basis vecs of either static or ada embedders are always layerwise_lora_rank,
            # as the basis vecs are shared across all CA layers.
            # But different vectors of the same embeddings are combinations of different basis vecs.
            # Therefore, num_total_basis_vecs is multiplied by num_vectors_each_subj_bg_pair.
            # num_vectors_each_subj_bg_pair: the number of vectors per (subj, bg) placeholder pair.
            # It's implied that all subj placeholders have the same number of vectors,
            # and all bg placeholders have the same number of vectors.
            self.number_vectors_each_subj = self.token2num_vectors.get(self.subject_strings[0])
            if len(self.background_strings) > 0:
                self.num_vectors_each_bg = self.token2num_vectors.get(self.background_strings[0], 4)
            else:
                self.num_vectors_each_bg = 0

            self.zs_cls_delta_string  = zs_cls_delta_string
            self.zs_prompt2token_proj_grad_scale = zs_prompt2token_proj_grad_scale
            self.zs_extra_words_scale = zs_extra_words_scale

            if zs_prompt2token_proj_grad_scale == 0:
                print("Warning: prompt2token_proj is frozen, so don't add noise to it.")
                self.zs_prompt2token_proj_ext_attention_perturb_ratio = 0
            else:
                self.zs_prompt2token_proj_ext_attention_perturb_ratio = zs_prompt2token_proj_ext_attention_perturb_ratio
            self.id2ada_prompt_encoder = create_id2ada_prompt_encoder(id2ada_prompt_encoder_type)
            self.id2img_prompt_encoder_trainable    = id2img_prompt_encoder_trainable
            self.to_load_id2img_learnable_modules   = to_load_id2img_learnable_modules

            if self.zs_cls_delta_string is not None:
                self.zs_cls_delta_tokens = self.get_tokens_for_string(zs_cls_delta_string)
                if zs_cls_delta_token_weights is None:
                    self.zs_cls_delta_token_weights = torch.ones(len(self.zs_cls_delta_tokens))
                    self.zs_cls_delta_token_weights[-1] = 2
                else:
                    self.zs_cls_delta_token_weights = torch.tensor(zs_cls_delta_token_weights, dtype=float)
                # The last word is the main word "man, woman, boy, girl" whose weight will be normalized to 1;
                # if there are any words before this word, their weights will be normalized to 0.25.
                self.zs_cls_delta_token_weights **= 2
                self.zs_cls_delta_token_weights /= self.zs_cls_delta_token_weights.max()
            else:
                self.zs_cls_delta_tokens = None
                self.zs_cls_delta_token_weights = None

        for placeholder_idx, placeholder_string in enumerate(self.placeholder_strings):
            placeholder_is_bg =  (placeholder_string in self.background_string_dict)
            # get_tokens_for_string <= get_clip_tokens_for_string.
            # force_single_token = True, as there should be only one token in placeholder_string.
            placeholder_token = self.get_tokens_for_string(placeholder_string, force_single_token=True)[0].item()
            self.string_to_token_dict[placeholder_string] = placeholder_token

            if self.do_zero_shot:
                # num_out_embs_per_layer: 16 if fg or 4 if bg. 
                num_out_embs_per_layer = self.number_vectors_each_subj if not placeholder_is_bg else self.num_vectors_each_bg

                subj_basis_generator = SubjBasisGenerator(num_out_embs_per_layer = num_out_embs_per_layer,
                                                          num_out_layers = self.num_unet_ca_layers,
                                                          # bg_image_embedding_dim: laion: 1280, openai: 1024.
                                                          # OpenAI CLIP output dim is 768, but the dim of the second last layer is 1024.
                                                          bg_image_embedding_dim = self.id2ada_prompt_encoder.clip_embedding_dim, 
                                                          output_dim = out_emb_dim,
                                                          placeholder_is_bg = placeholder_is_bg,
                                                          prompt2token_proj_grad_scale = self.zs_prompt2token_proj_grad_scale,
                                                          bg_prompt_translator_has_to_out_proj=False,
                                                          num_static_img_suffix_embs = num_static_img_suffix_embs,
                                                          zs_extra_words_scale = self.zs_extra_words_scale)

                self.string_to_subj_basis_generator_dict[placeholder_string] = subj_basis_generator

        # Initialize self.subj_name_to_cls_delta_tokens.
        self.init_cls_delta_tokens(self.get_tokens_for_string, self.get_embeddings_for_tokens, 
                                   subj_name_to_cls_delta_string, zs_cls_delta_string)
        self.init_subj_name_to_being_faces(subj_name_to_being_faces)

        self.layer_idx = -1
        self.static_subj_embs_dict = {}   
        self.clear_prompt_adhoc_info()
        # 'recon_iter', 'compos_distill_iter', 'empty'.
        self.iter_type = None       
        if self.do_zero_shot:
            self.set_curr_batch_subject_names(["zs_default"], 'recon_iter')
        else:
            self.curr_batch_subj_names = []
            self.current_subj_name_to_cls_delta_tokens = {}
            self.cls_delta_strings = None

        self.loss_call_count = 0
        self.training_percent = 0
        self.emb_global_scales_dict = None
        # ca_q_bns and ca_outfeat_lns are used to normalize the q/out features
        # in loss computation in ddpm.py, and not used in this script.
        ca_q_bns = {}
        ca_outfeat_lns = {}
        for ca_layer_idx in range(self.num_unet_ca_layers):
            layer_idx = self.ca_layer_idx2layer_idx[ca_layer_idx]
            ca_q_bns[str(layer_idx)]       = nn.BatchNorm2d(self.ca_infeat_dims[ca_layer_idx], affine=False)
            ca_outfeat_lns[str(layer_idx)] = nn.LayerNorm(self.ca_infeat_dims[ca_layer_idx], elementwise_affine=False)
            #print(layer_idx, self.ca_infeat_dims[ca_layer_idx])

        self.ca_q_bns       = nn.ModuleDict(ca_q_bns)
        self.ca_outfeat_lns = nn.ModuleDict(ca_outfeat_lns)

        # zs_image_prompt_dict have two keys: 'subj', 'bg'.
        # zs_image_prompt_dict['subj'] is the image prompt embs for the subject, which is ready to be used.
        # zs_image_prompt_dict['bg'] is the clip features for the background, 
        # which needs to be translated by a bg SubjBasisGenerator.
        self.zs_image_prompt_dict = {}

        print("EmbeddingManager on subj={}, bg={} init with {} vec(s)".format(
               self.subject_strings, self.background_strings, self.token2num_vectors))
        
        # Add the search span by 1, just to be safe.
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN += 1
        print(f"CLS_DELTA_STRING_MAX_SEARCH_SPAN={self.CLS_DELTA_STRING_MAX_SEARCH_SPAN}")

        if adaface_ckpt_path is not None:
            self.load(adaface_ckpt_path,
                      extend_prompt2token_proj_attention_multiplier,
                      to_load_old_adaface_ckpt)

    def init_cls_delta_tokens(self, get_tokens_for_string, get_embeddings_for_tokens, 
                              subj_name_to_cls_delta_string, 
                              zs_cls_delta_string=None):
        if subj_name_to_cls_delta_string is None:
            subj_name_to_cls_delta_string = {}
        if zs_cls_delta_string is not None:
            # During inference, subj_name_to_cls_delta_string contains 'zs_default' as the subject name, and maps
            # to zs_cls_delta_string, the default class delta string.
            subj_name_to_cls_delta_string['zs_default'] = zs_cls_delta_string

        # We don't know the gender of a rand_id_to_img_prompt subject.
        subj_name_to_cls_delta_string['rand_id_to_img_prompt'] = 'person'

        self.subj_name_to_cls_delta_string  = subj_name_to_cls_delta_string
        self.subj_name_to_cls_delta_tokens  = {}
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN = 0

        for subj_name in self.subj_name_to_cls_delta_string:
            cls_delta_string = self.subj_name_to_cls_delta_string[subj_name]
            cls_delta_tokens, cls_delta_token_weights, _, _ = \
                calc_init_word_embeddings(get_tokens_for_string, get_embeddings_for_tokens,
                                          cls_delta_string, None)

            num_cls_delta_tokens = len(cls_delta_tokens)

            # subj_idx_to_cls_delta_tokens is used to examine class prompts, 
            # to see if there are subsequences of cls_delta_tokens.
            # If there are, the embeddings of init_word_tokens should be combined through weighted sum.
            self.subj_name_to_cls_delta_tokens[subj_name]        = cls_delta_tokens

            # CLS_DELTA_STRING_MAX_SEARCH_SPAN should be the max number of extra tokens
            # (all excluding the first of the init word tokens; the first corresponds to the subject token).
            # If multiple subject strings appear in the same prompt, then CLS_DELTA_STRING_MAX_SEARCH_SPAN 
            # should be multiplied by the number of subject strings. Currently not implemented.
            if num_cls_delta_tokens - 1 > self.CLS_DELTA_STRING_MAX_SEARCH_SPAN:
                self.CLS_DELTA_STRING_MAX_SEARCH_SPAN = num_cls_delta_tokens - 1

    def init_subj_name_to_being_faces(self, subj_name_to_being_faces):
        # subj_name_to_being_faces: a dict that maps subject names to is_face.
        # subj_name_to_being_faces is used in ddpm.py and not here.
        self.subj_name_to_being_faces = subj_name_to_being_faces if subj_name_to_being_faces is not None \
                                            else {subj_name: True for subj_name in self.subject_strings}
        self.subj_name_to_being_faces['rand_id_to_img_prompt'] = True
        self.subj_name_to_being_faces['zs_default']            = True

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # If self.use_layerwise_embedding, then each token expands to num_unet_ca_layers = 16 
    # layerwise embeddings.
    def forward(
            self,
            tokenized_text,         # [B, N]. 
            embedded_text,          # [B, N, 768]. 
    ):
        # When delta loss is used, B is not batch_size, but batch_size * 4 * num_compositions_per_image.
        # If bs=2, num_compositions_per_image=2, then B=16.
        # In the iterations when ada delta loss is enabled, in effect num_compositions_per_image is 1, 
        # even if it's specified as 2, so B=8.
        B, N, device = *tokenized_text.shape, tokenized_text.device

        # placeholder_indices will be regenerated with update_placeholder_indices() 
        # within get_static_embedding().
        self.clear_prompt_adhoc_info()
        
        # We need to clone embedded_text, as sometimes (when it's not layerwise, such as TI) 
        # the modification in get_static_embedding() is in-place. 
        # The keys of static_subj_embs_dict are the placeholder strings.
        static_embeded_text, tokenized_text_repeated, static_subj_embs_dict = \
                        self.get_static_embedding(tokenized_text, embedded_text.clone(), 
                                                  B, N, self.num_unet_ca_layers)
        # Cache the static embeddings to be used in ada embedding computation and
        # embedding orthogonal loss later.
        self.static_subj_embs_dict = {}

        for k in static_subj_embs_dict:
            self.static_subj_embs_dict[k] = static_subj_embs_dict[k]

        # Update the prompt token embedding mask.
        # tokenized_text_repeated is repeated 16 times along the batch dimension.
        self.update_prompt_masks(tokenized_text, tokenized_text_repeated)

        return static_embeded_text
    
    # N: length of sequence (including padding).
    def get_static_embedding(self, tokenized_text, embedded_text, BS, N, num_unet_ca_layers):
        
        # Put dist.get_rank() here. We couldn't get the rank in __init__(), as the default process group has not been initialized 
        # at that time.
        if self.rank == -1:
            try:
                # During inference, dist.get_rank() will raise an exception.
                self.rank = dist.get_rank()
            except:
                self.rank = 0

        orig_tokenized_text = tokenized_text
        static_subj_embs_dict = {}
        self.cls_delta_string_indices = []

        if self.use_layerwise_embedding:
            # embedded_text: [B, N, 768] => [B, 16, N, 768] => [16*B, N, 768].
            # "Tuck" the layer dimension into the batch dimension, 
            # to keep embedded_text in 3D, same as the input.
            # After repeat, the same instance is repeated 16 times, which are adjacent 
            # to each other across the batch dim:
            # [b1_l1, ..., b1_l16, b2_l1, ..., b2_l16, ..., bB_l1, ..., bB_l16].
            # {________b1________} {_______b2_______}  ...  {_______bB________}
            embedded_text = embedded_text.unsqueeze(1).repeat(1, num_unet_ca_layers, 1, 1).view(BS * num_unet_ca_layers, N, -1)
            # tokenized_text: [B, 16, N] => [16*B, N]
            # tokenized_text has to be repeated along the layer dimension as well, so that 
            # placeholder_indices can index the embedding at each layer in the batch.
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1, num_unet_ca_layers, 1).view(BS * num_unet_ca_layers, N)
            # mirror-reflect the embedding along the layer dimension, to make it symmetric 
            # in the encoder & decoder.

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            # If there's only one vector per token, we can do a simple replacement
            placeholder_indices = torch.where(tokenized_text == placeholder_token)
            # No placeholder token is found in the current batch.
            if placeholder_indices[0].numel() == 0:
                continue
            
            placeholder_is_bg = (placeholder_string in self.background_string_dict)
            # If multiple occurrences are found in a prompt, only keep the first as the subject.
            # Other occurrences are treated as part of the background prompt (this may happen if
            # composition image overlay is used).
            placeholder_indices_1st = extract_first_index_in_each_instance(placeholder_indices)
            # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
            # Non-layerwise: embedded_text[placeholder_indices]: [2, 768].  subj_static_embedding: [1, K, 768].
            # layerwise: placeholder_indices =  
            # (tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 
            #  tensor([ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]))
            # embedded_text[placeholder_indices]: [32, 768]. subj_static_embedding: [16, K, 768].
            # The first 16 elements (0-8) in embedded_text[placeholder_indices] correspond to the 16 layers of the 
            # first instance in the batch.
            # 16 layers of subj_static_embedding are repeated REAL_OCCURS_IN_BATCH times.
            # subj_static_embedding: [16, 768] repeat=> [32, 768]
            # LINK #init_embed

            # REAL_OCCURS_IN_BATCH: the real number of occurrences of the placeholder in the current batch,
            # not repetitively counting the occurrences in the embedded_text repeated for M layers.
            REAL_OCCURS_IN_BATCH = placeholder_indices_1st[0].numel() // self.num_layers_per_embedder
            # Some prompts don't contain the placeholder token. This could happen in a compositional 
            # distillation iteration, or an inference iteration. 
            # Anyway, we need to check whether the cls_delta_string
            # occurs in the prompts without the placeholder token. If so, we need to merge 
            # their embeddings to one (the first) embedding, and delete the 2nd to the last embeddings,
            # using merge_cls_token_embeddings().
            # current_subj_name_to_cls_delta_tokens only contains the cls_delta_tokens of the current batch.
            if REAL_OCCURS_IN_BATCH < BS and self.CLS_DELTA_STRING_MAX_SEARCH_SPAN > 0 \
              and len(self.current_subj_name_to_cls_delta_tokens) > 0:
                cls_delta_string_indices = scan_cls_delta_strings(tokenized_text,
                                                                  placeholder_indices_1st,
                                                                  self.current_subj_name_to_cls_delta_tokens,
                                                                  self.CLS_DELTA_STRING_MAX_SEARCH_SPAN)
                # cls_delta_string_indices is a list of tuples, each tuple is 
                # (batch_i, start_N, num_cls_delta_tokens, placeholder_token).
                self.cls_delta_string_indices += cls_delta_string_indices

            # Generate the actual subj_static_embedding on the fly.
            # The 16 static subject embeddings are formed by linearly combining the basis vectors.
            # The matrix operations are done on the fly.
            # subj_static_embedding: [16, K, 768].
            if self.do_zero_shot:
                if placeholder_is_bg:
                    id2img_prompt_embs  = None
                    zs_clip_features    = self.zs_image_prompt_dict['bg']
                else:
                    # id2img_embs (ID embeddings only): [BS, 16, 768] or [BS, 4, 768].
                    id2img_prompt_embs  = self.zs_image_prompt_dict['subj'] if self.curr_subj_is_face else None
                    zs_clip_features    = None

                subj_basis_generator = self.string_to_subj_basis_generator_dict[placeholder_string]
                    
                # zs_clip_features: [BS, 257, 1280]
                # adaface_subj_embs:   [BS, 16, 16, 768] if fg, or [BS,  16, 4, 768] if bg.
                adaface_subj_embs, placeholder_adaface_prompt_embs = \
                        subj_basis_generator(id2img_prompt_embs,
                                             zs_clip_features, None, 
                                             out_id_embs_cfg_scale=1, is_face=self.curr_subj_is_face,
                                             is_training=self.training,
                                             adaface_prompt_embs_inf_type='full_half_pad')
                # In a mix prompt batch (either compos_distill_iter or recon_iter with delta loss), 
                # REAL_OCCURS_IN_BATCH counts the number of subject-single and subject-comp instances.
                # But adaface_subj_embs is generated from the subject-single instance only.
                # Repeating at dim 0 is correct even if adaface_subj_embs has a batch size > 1:
                # If the subject-single batch is like [s1, s2], then the repeated batch is [s1, s2, s1, s2], 
                # matching the batch structure of (subject-single, subject-single, ...).
                if adaface_subj_embs.shape[0] < REAL_OCCURS_IN_BATCH:
                    adaface_subj_embs = adaface_subj_embs.repeat(REAL_OCCURS_IN_BATCH // adaface_subj_embs.shape[0], 1, 1, 1)

                if self.do_zero_shot and not placeholder_is_bg:
                    # id2img_prompt_embs: [BS, 16, 768] or [BS, 4, 768] is the ID2ImgPrompt embeddings. 
                    self.id2img_embs = id2img_prompt_embs
                    if self.iter_type in ['compos_distill_iter']:
                        assert placeholder_adaface_prompt_embs is not None

                # NOTE: the condition iter_type == 'compos_distill_iter' is vital, as a recon_iter with delta loss 
                # also has the 4-type prompt structure.
                # But we should NEVER replace the subject-single embeddings with the frozen ones, 
                # because these embeddings are used to reconstruct the noisy image, and if replaced,
                # the model will learn nothing from the recon loss.
                # One potential issue is the delta loss may slowly degrade the identity information in the embeddings.
                # So we will replace the subject-single embeddings when computing the delta loss in ddpm.py later.
                if self.training and not placeholder_is_bg and self.iter_type in ['compos_distill_iter']: #, 'recon_iter']:
                    # compos_distill_iter always has same_subject_in_batch == True. 
                    # So id2img_prompt_embs: [1, 512].
                    if id2img_prompt_embs.shape[0] != 1:
                        breakpoint()
                    subj_basis_generator0 = self.frozen_string_to_subj_basis_generator_dict[placeholder_string]
                    with torch.no_grad():
                        # adaface_subj_embs0: ID embeddings from the frozen subj_basis_generator.
                        # This is to reduce overfitting of subj_basis_generator after it's been finetuned.
                        adaface_subj_embs0, placeholder_adaface_prompt_embs0 = \
                                subj_basis_generator0(id2img_prompt_embs, zs_clip_features, None, 
                                                      out_id_embs_cfg_scale=1, is_face=self.curr_subj_is_face,
                                                      is_training=self.training,
                                                      adaface_prompt_embs_inf_type='full_half_pad')
                        
                    # adaface_subj_embs0: [1, 16, 16, 768] -> [2, 16, 16, 768].
                    adaface_subj_embs0 = adaface_subj_embs0.repeat(REAL_OCCURS_IN_BATCH // 2, 1, 1, 1)
                    # adaface_subj_embs0: [2, 16, 16, 768] -> [32, 16, 768].
                    self.adaface_subj_embs0 = rearrange(adaface_subj_embs0, 'b l k d -> (b l) k d').contiguous()
                    # Only replace the subject-single embeddings in the compos_distill_iter.
                    # Replace the the subj-single embeddings with frozen subject embeddings, which is the first 1/4
                    # of the whole batch, i.e., the first REAL_OCCURS_IN_BATCH // 2 embeddings.
                    if self.iter_type == 'compos_distill_iter':
                        NUM_HALF_SUBJS = REAL_OCCURS_IN_BATCH // 2
                        # Still allow a small inference from the updated subj-single embeddings, 
                        # maybe this will make the images more natural?
                        adaface_subj_embs[:NUM_HALF_SUBJS] = adaface_subj_embs0.to(adaface_subj_embs.dtype) * 0.9 \
                                                            + adaface_subj_embs[:NUM_HALF_SUBJS] * 0.1
                        
                        if self.rank == 0:
                            print(f"compos_distill_iter. Replace the first {REAL_OCCURS_IN_BATCH // 2} embeddings with the frozen embeddings.")
            else:
                adaface_subj_embs = None

            # subj_static_embedding is adaface_subj_embs reshaped.
            subj_static_embedding = rearrange(adaface_subj_embs, 'b l k d -> (b l) k d').contiguous()
            subj_static_embedding = subj_static_embedding.to(embedded_text.dtype)
            static_subj_embs_dict[placeholder_string] = subj_static_embedding

            for k in range(self.token2num_vectors[placeholder_string]):
                # embedded_text is repeated 16 times along the layer dimension, with size of dim 0 = 16 * BS.
                # The result of the repeat is: the same instance is repeated 16 times, which are adjacent 
                # to each other across the batch dim:
                # [b1_l1, ..., b1_l16, b2_l1, ..., b2_l16, ..., bB_l1, ..., bB_l16].
                # {________b1________} {_______b2_______}  ...  {_______bB________}
                # The first dim of subj_static_embedding is the layer dim (size = 16). 
                # So we repeat the 16 layers of the k-th embedding, subj_static_embedding[:, k], 
                # REAL_OCCURS_IN_BATCH times, to match 16*REAL_OCCURS_IN_BATCH.
                # After repeat, the RHS is
                # [ek_l1, ..., ek_l16, ek_l1, ..., ek_l16, ..., ek_l1, ..., ek_l16].
                # {________b1________} {_______b2_______}  ...  {_______bB________}
                # During inference, BS = 1, subj_static_embedding_k: [16, 768]
                subj_static_embedding_k = subj_static_embedding[:, k]
                
                if self.training and self.training_begin_perturb_std_range is not None:
                    # The std of subj_static_embedding is around 0.07, times training_end_perturb_std_range
                    # (0.02 ~ 0.04) is very small. Therefore, it won't hurt the subject identity encoded
                    # in the embeddings.
                    subj_static_embedding_k = \
                        anneal_perturb_embedding(subj_static_embedding_k, 
                                                 self.training_percent,
                                                 self.training_begin_perturb_std_range,
                                                 self.training_end_perturb_std_range,
                                                 self.training_perturb_prob[self.iter_type],
                                                 perturb_std_is_relative=True, keep_norm=False,
                                                 verbose=False)

                # Training with delta loss. Each subject only appears once in subj_static_embedding, 
                # but twice in the prompts (subject single and subject comp), so we need to repeat it twice.
                if REAL_OCCURS_IN_BATCH == BS // 2 and subj_static_embedding_k.shape[0] == REAL_OCCURS_IN_BATCH // 2 * num_unet_ca_layers:
                    # subj_static_embedding_k: [48, 768] => [48*2, 768]
                    subj_static_embedding_k = subj_static_embedding_k.repeat(2, 1)
                # Single-subject batch. It's either during inference, or during training with same_subject_in_batch=True.
                # BS = 1, subj_static_embedding_k: [16, 768]
                # Each subject only appears once in subj_static_embedding, but BS == REAL_OCCURS_IN_BATCH
                # times in the prompts. Therefore, it's repeated REAL_OCCURS_IN_BATCH times.
                elif subj_static_embedding_k.shape[0] == num_unet_ca_layers:
                    # subj_static_embedding_k: [16, 768] => [16*REAL_OCCURS_IN_BATCH, 768]
                    subj_static_embedding_k = subj_static_embedding_k.repeat(REAL_OCCURS_IN_BATCH, 1)
                elif subj_static_embedding_k.shape[0] != num_unet_ca_layers * REAL_OCCURS_IN_BATCH:
                    breakpoint()
                # Otherwise, subj_static_embedding_k.shape[0] == num_unet_ca_layers * REAL_OCCURS_IN_BATCH,
                # i.e., the left and right sides will have the same number of identity embeddings, and we don't need to do anything.

                # Assign the k-th token embedding (along the text dim).
                placeholder_indices_k = (placeholder_indices_1st[0], placeholder_indices_1st[1] + k)
                embedded_text[placeholder_indices_k] = subj_static_embedding_k

            # Cache the placeholder indices for mix prompt distillation.
            # Note placeholder_indices are recomputed in update_placeholder_indices(), 
            # we don't simply cache placeholder_indices here as they are repeated 16 times 
            # to replace in 16 layers. 
            # But we need them without repetitions for mix prompt distillation.
            # If num_vectors_per_subj_token > 1, then repeat the indices and add to offsets.
            # If background_strings is None, then always update the indices. Otherwise, 
            # skip updating placeholder indices of the background string.
            self.update_placeholder_indices(orig_tokenized_text, placeholder_string, placeholder_token, 
                                            self.token2num_vectors[placeholder_string],
                                            placeholder_is_bg=placeholder_is_bg)
  
        return embedded_text, tokenized_text, static_subj_embs_dict

    # extend_placeholders() should only be called during inference, 
    # or after the model has been initialized.
    def extend_placeholders(self, new_subject_strings, new_background_strings, 
                            num_vectors_per_subj_token, num_vectors_per_bg_token):
        added_placeholders = []

        if new_subject_strings is not None:
            for k in new_subject_strings:
                if k is None or k in self.subject_strings:
                    continue
                self.subject_strings.append(k)
                self.subject_string_dict[k] = True
                self.placeholder_strings.append(k)
                self.token2num_vectors[k] = num_vectors_per_subj_token
                added_placeholders.append(k)
                print(f"Added new subject string: {k}->num_vectors_per_subj_token={num_vectors_per_subj_token}.")

        if new_background_strings is not None:
            for k in new_background_strings:
                if k is None or k in self.background_strings:
                    continue
                self.background_strings.append(k)
                self.background_string_dict[k] = True
                self.placeholder_strings.append(k)
                self.token2num_vectors[k] = num_vectors_per_bg_token
                added_placeholders.append(k)
                print(f"Added new background string: {k}->num_vectors_per_subj_token={num_vectors_per_bg_token}.")
    
    # Update prompt_emb_mask.
    # tokenized_text: [B, N] = [2/4, 77].
    # If 'validation' is present in the config file,
    # DDPM.validation_step() -> LatentDiffusion.shared_step() -> .forward()
    # -> .get_text_conditioning() -> .cond_stage_model.encode()
    # -> EmbeddingManager.forward() -> here.
    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # Occasionally, image_logger is called, which calls LatentDiffusion.log_images ->
    # .get_text_conditioning() -> ... -> here.
    # Such prompt_emb_mask won't be used in calc_prompt_emb_delta_loss() and won't be cleared.
    # prompt_emb_mask: [B, N, 1], where N=77 is the prompt length after padding.
    def update_prompt_masks(self, tokenized_text, tokenized_text_repeated=False):
        # Exclude the starting (49406) and padding tokens (49047) from delta loss.
        prompt_emb_mask  = (tokenized_text != 49406 ) & (tokenized_text != 49407)
        # [B, N] => [B, N, 1]
        self.prompt_emb_mask  = prompt_emb_mask.float().unsqueeze(2)

    def clear_prompt_adhoc_info(self):
        self.placeholder2indices    = {}
        self.prompt_emb_mask        = None

    # Set ad-hoc data structures for computing placeholder embeddings and various losses.
    def set_prompt_adhoc_info(self, prompt_adhoc_info):
        self.placeholder2indices    = prompt_adhoc_info['placeholder2indices']
        self.prompt_emb_mask        = prompt_adhoc_info['prompt_emb_mask']
    
    # During training, set_curr_batch_subject_names() is called in ddpm.py.
    # During inference, set_curr_batch_subject_names() is called by the embedding manager.
    def set_curr_batch_subject_names(self, subj_names, text_conditioning_iter_type):
        self.curr_batch_subj_names = subj_names
        # During inference, as self.curr_batch_subj_names is not set, the three dicts are empty.
        self.current_subj_name_to_cls_delta_tokens = { subj_name: self.subj_name_to_cls_delta_tokens[subj_name] \
                                                       for subj_name in self.curr_batch_subj_names }
                
        # During training, we get the current subject name from self.curr_batch_subj_names, then map to 
        # curr_subj_is_face. 
        # During inference, curr_batch_subj_names = ['zs_default'], which maps to True in subj_name_to_being_faces,
        # so curr_subj_is_face == True.
        # BUG: if there are multiple subjects in the same batch, then is_face is only 
        # about the first subject. But now we only support one subject in a batch.
        if len(self.curr_batch_subj_names) > 0:
            self.curr_subj_is_face = self.subj_name_to_being_faces[self.curr_batch_subj_names[0]]

        if len(self.current_subj_name_to_cls_delta_tokens) > 0:
            self.cls_delta_strings       = [ self.subj_name_to_cls_delta_string[subj_name] \
                                             for subj_name in self.curr_batch_subj_names ]
        else:
            self.cls_delta_strings = None

        self.set_curr_iter_type(text_conditioning_iter_type)
        if True: #cls_delta_strings is not None and 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
            print(f"{self.rank} subjects: {self.curr_batch_subj_names}, cls_delta_strings: {self.cls_delta_strings}")

    def set_curr_iter_type(self, text_conditioning_iter_type):
        self.iter_type = text_conditioning_iter_type
        # In a compos_distill_iter, all subjects are the same. So we only keep the first cls_delta_string.
        if self.cls_delta_strings is not None and self.iter_type == 'compos_distill_iter':
            self.cls_delta_strings = self.cls_delta_strings[:1]
        
    def update_placeholder_indices(self, tokenized_text, placeholder_string, placeholder_token, num_vectors_per_subj_token, placeholder_is_bg):
        placeholder_indices = torch.where(tokenized_text == placeholder_token)
        placeholder_indices_B, placeholder_indices_N = extract_first_index_in_each_instance(placeholder_indices)

        if len(placeholder_indices_B) == 0:
            self.placeholder2indices[placeholder_string] = None
            return

        if num_vectors_per_subj_token > 1:
            BS = placeholder_indices_B.shape[0]
            # unsqueeze(1) -> [B, 1] => [B, num_vectors_per_subj_token] => [B * num_vectors_per_subj_token].
            # Make sure the embedding indices of the same instance are grouped together.
            # [b1_v1, b1_v2, b1_v3, b2_v1, b2_v2, b2_v3, ...].
            # Then we can easily get the indices of a certain sub-batch.
            placeholder_indices_B = placeholder_indices_B.unsqueeze(1).repeat(1, num_vectors_per_subj_token).view(-1)
            placeholder_indices_N = placeholder_indices_N.unsqueeze(1).repeat(1, num_vectors_per_subj_token).view(-1)
            # Add offsets to the indices of the pseudo-tokens.
            placeholder_indices_N_off = placeholder_indices_N + torch.arange(num_vectors_per_subj_token, device=tokenized_text.device).repeat(BS)
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N_off)
        else:
            # placeholder_indices contains the indices of all placeholder embeddings.
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N)
        
        self.placeholder2indices[placeholder_string] = placeholder_indices

    def set_training_perturb_specs(self, training_begin_perturb_std_range, 
                                   training_end_perturb_std_range,
                                   training_perturb_prob):
        self.training_begin_perturb_std_range = training_begin_perturb_std_range
        self.training_end_perturb_std_range   = training_end_perturb_std_range
        self.training_perturb_prob            = training_perturb_prob

        if training_begin_perturb_std_range is None and training_end_perturb_std_range is None:
            print(f"Disable training perturbance")
        else:
            print(f"training perturbance std range: {training_begin_perturb_std_range}-{training_end_perturb_std_range}"
                  f", with prob = {training_perturb_prob}")

    def set_zs_image_prompts_and_features(self, id2img_prompt_embs, zs_clip_bg_features):
        # id2img_prompt_embs: [1, 16, 768] or [1, 4, 768].
        # zs_clip_bg_features: [1, 257, 1280].

        self.zs_image_prompt_dict = { 'subj': id2img_prompt_embs,
                                      'bg':   zs_clip_bg_features,
                                    }

        # Clear the saved subj_embs.
        self.static_subj_embs_dict = {}

    def set_num_vectors_per_subj_token(self, token2num_vectors):
        self.token2num_vectors = token2num_vectors
        print(f"Set token2num_vectors: {self.token2num_vectors}")

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, adaface_ckpt_path):
        saved_dict = { "string_to_subj_basis_generator_dict":   self.string_to_subj_basis_generator_dict,
                        "token2num_vectors":                    self.token2num_vectors,
                        "placeholder_strings":                  self.placeholder_strings,
                        "subject_strings":                      self.subject_strings,
                        "background_strings":                   self.background_strings,
                        # Used to normalize attention features for calc_comp_fg_bg_preserve_loss() during training.
                        "ca_q_bns":                             self.ca_q_bns,
                        "ca_outfeat_lns":                       self.ca_outfeat_lns,
                        "do_zero_shot":                         self.do_zero_shot,
                     }
        
        if self.id2img_prompt_encoder_trainable:
            id2img_learnable_modules = self.id2ada_prompt_encoder.get_id2img_learnable_modules()
            saved_dict["id2img_prompt_encoder_learnable_modules"] = [ module.state_dict() for module in id2img_learnable_modules ]

        torch.save(saved_dict, adaface_ckpt_path)

    # Load custom tokens and their learned embeddings from "embeddings_gs-4500.pt".
    def load(self, adaface_ckpt_paths, extend_prompt2token_proj_attention_multiplier=1, to_load_old_adaface_ckpt=False):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        token2num_vectors                   = {}
        self.string_to_token_dict           = {}

        self.subject_strings                = []
        self.background_strings             = []

        if isinstance(adaface_ckpt_paths, str):
            adaface_ckpt_paths = [adaface_ckpt_paths]

        for adaface_ckpt_path in adaface_ckpt_paths:
            ckpt_path_parts = adaface_ckpt_path.split(":")
            adaface_ckpt_path = ckpt_path_parts[0]
            if len(ckpt_path_parts) == 2:
                placeholder_mapper = {}
                for placeholder_mapping in ckpt_path_parts[1].split(","):
                    from_, to_ = placeholder_mapping.split("-")
                    placeholder_mapper[from_] = to_
            else:
                placeholder_mapper = None

            ckpt = torch.load(adaface_ckpt_path, map_location='cpu')

            if "background_strings" in ckpt:
                ckpt_background_strings = ckpt["background_strings"]
            else:
                ckpt_background_strings = []

            if "ca_q_bns" in ckpt:
                self.ca_q_bns = ckpt["ca_q_bns"]
            if "ca_outfeat_lns" in ckpt:
                self.ca_outfeat_lns = ckpt["ca_outfeat_lns"]

            # Only load subj_basis_generator from ckpt if the ckpt is set with the same do_zero_shot.
            if "do_zero_shot" in ckpt and self.do_zero_shot == ckpt["do_zero_shot"]:
                for km, ckpt_subj_basis_generator in ckpt["string_to_subj_basis_generator_dict"].items():
                    ret = None
                    # repr(ckpt_subj_basis_generator) will assign missing variables to ckpt_subj_basis_generator.
                    if to_load_old_adaface_ckpt:
                        print(f"Loading ckpt_subj_basis_generator {km}")
                    else:
                        print(f"Loading {repr(ckpt_subj_basis_generator)}")

                    # self.string_to_subj_basis_generator_dict[km] is either not initialized, or initialized with a smaller depth.
                    # Then replace it with the one in ckpt.
                    # print(f"Overwrite {repr(self.string_to_subj_basis_generator_dict[km])}")
                    ckpt_subj_basis_generator.face_proj_in = None
                    # If old ckpts don't have num_static_img_suffix_embs, set it to 0.
                    if not hasattr(ckpt_subj_basis_generator, "num_static_img_suffix_embs"):
                        ckpt_subj_basis_generator.num_static_img_suffix_embs = 0
                        ckpt_subj_basis_generator.static_img_suffix_embs = None

                    curr_num_static_img_suffix_embs = self.string_to_subj_basis_generator_dict[km].num_static_img_suffix_embs

                    if to_load_old_adaface_ckpt:
                        # Delete lora2hira, latent_queries and layers, as lora2hira and layers belong to 
                        # old ckpts, and latent_queries has different shapes.
                        ckpt_subj_basis_generator.lora2hira = None
                        ckpt_subj_basis_generator.latent_queries = None
                        ckpt_subj_basis_generator.latent_query_lns = None
                        ckpt_subj_basis_generator.layers = None
                        ckpt_subj_basis_generator.obj_proj_in = None
                        ckpt_subj_basis_generator.proj_in = None
                        ckpt_subj_basis_generator.pos_embs = None
                        ckpt_subj_basis_generator.num_static_img_suffix_embs = curr_num_static_img_suffix_embs

                        self.string_to_subj_basis_generator_dict[km] = ckpt_subj_basis_generator
                        # If curr_num_static_img_suffix_embs is different with num_static_img_suffix_embs in the ckpt,
                        # ckpt_subj_basis_generator.static_img_suffix_embs will be adjusted to the correct size 
                        # in patch_old_subj_basis_generator_ckpt().
                        self.string_to_subj_basis_generator_dict[km].patch_old_subj_basis_generator_ckpt()
                        self.string_to_subj_basis_generator_dict[km].freeze_prompt2token_proj()
                        continue

                    # Compatible with older ckpts which only have per-layer hidden_state_layer_weights.
                    if (not ckpt_subj_basis_generator.placeholder_is_bg) \
                      and ckpt_subj_basis_generator.hidden_state_layer_weights.shape[-1] != self.string_to_subj_basis_generator_dict[km].hidden_state_layer_weights.shape[-1]:
                        if self.string_to_subj_basis_generator_dict[km].hidden_state_layer_weights.shape[-1] == 1:
                            # hidden_state_layer_weights: [3, 768] -> [3, 1]
                            ckpt_subj_basis_generator.hidden_state_layer_weights = nn.Parameter(ckpt_subj_basis_generator.hidden_state_layer_weights.mean(dim=1, keepdim=True))
                            print(f"Average along features: hidden_state_layer_weights -> {ckpt_subj_basis_generator.hidden_state_layer_weights.shape}")
                        else:
                            # hidden_state_layer_weights: [3, 1] -> [3, 768]
                            ckpt_subj_basis_generator.hidden_state_layer_weights = nn.Parameter(ckpt_subj_basis_generator.hidden_state_layer_weights.repeat(1, 768))
                            print(f"Expand along features:  hidden_state_layer_weights -> {ckpt_subj_basis_generator.hidden_state_layer_weights.shape}")

                    # ckpt_subj_basis_generator.prompt2token_proj hasn't been extended. 
                    # So only extend self.string_to_subj_basis_generator_dict[km] after loading the state_dict.
                    # This should happen only during training, not inference. 
                    # Therefore, whether perturb_std is 0 or not doesn't really matter the inference result.
                    if ckpt_subj_basis_generator.placeholder_is_bg:
                        ret = self.string_to_subj_basis_generator_dict[km].load_state_dict(ckpt_subj_basis_generator.state_dict(), strict=False)
                    else:
                        ckpt_prompt2token_proj_attention_multipliers = ckpt_subj_basis_generator.prompt2token_proj_attention_multipliers
                        # Fix old ckpt bug of having negative attention multipliers.
                        ckpt_prompt2token_proj_attention_multipliers = [ m if m > 0 else 1 for m in ckpt_prompt2token_proj_attention_multipliers ]
                        self.string_to_subj_basis_generator_dict[km].extend_prompt2token_proj_attention(\
                            ckpt_prompt2token_proj_attention_multipliers, -1, -1, 1, perturb_std=0)
                        # Handle differences in num_static_img_suffix_embs between the current model and the ckpt.
                        if curr_num_static_img_suffix_embs != ckpt_subj_basis_generator.num_static_img_suffix_embs:
                            if curr_num_static_img_suffix_embs == 0 or curr_num_static_img_suffix_embs > ckpt_subj_basis_generator.num_static_img_suffix_embs:
                                # The current model has more static_img_suffix_embs than the old ckpt, or has no static_img_suffix_embs.
                                # So we skip loading the static_img_suffix_embs from the ckpt.
                                ckpt_subj_basis_generator.static_img_suffix_embs = None
                            else:
                                # The old ckpt has more static_img_suffix_embs than the current model.
                                # So we need to remove the extra static_img_suffix_embs from the ckpt before loading.
                                ckpt_subj_basis_generator.static_img_suffix_embs = ckpt_subj_basis_generator.static_img_suffix_embs[:, :curr_num_static_img_suffix_embs]

                        ret = self.string_to_subj_basis_generator_dict[km].load_state_dict(ckpt_subj_basis_generator.state_dict(), strict=False)
                        # If extend_prompt2token_proj_attention_multiplier > 1, then after loading state_dict, extend the prompt2token_proj.
                        if extend_prompt2token_proj_attention_multiplier > 1:
                            # During this extension, the added noise does change the extra copies of attention weights, since they are not in the ckpt.
                            # During training,  zs_prompt2token_proj_ext_attention_perturb_ratio == 0.1.
                            # During inference, zs_prompt2token_proj_ext_attention_perturb_ratio == 0.
                            # All CLIP encoder layers are 0-11. 
                            # 0, 6: extend the first 6 layers 0-5 (not including layer 6).
                            # 0, 3: extend the first 3 layers 0-2 (not including layer 3).
                            self.string_to_subj_basis_generator_dict[km].extend_prompt2token_proj_attention(\
                                None, -1, -1, extend_prompt2token_proj_attention_multiplier,
                                perturb_std=self.zs_prompt2token_proj_ext_attention_perturb_ratio)

                    if ret is not None and len(ret.missing_keys) > 0:
                        print(f"Missing keys: {ret.missing_keys}")
                    if ret is not None and len(ret.unexpected_keys) > 0:
                        print(f"Unexpected keys: {ret.unexpected_keys}")

                    # Fix missing variables in the old ckpt.
                    self.string_to_subj_basis_generator_dict[km].patch_old_subj_basis_generator_ckpt()
                    self.string_to_subj_basis_generator_dict[km].freeze_prompt2token_proj()

                if "id2img_prompt_encoder_learnable_modules" in ckpt:
                    if self.to_load_id2img_learnable_modules:
                        self.id2ada_prompt_encoder.load_id2img_learnable_modules(ckpt["id2img_prompt_encoder_learnable_modules"])
                    else:
                        print(f'ID2ImgPrompt encoder learnable modules in {adaface_ckpt_path} are not loaded.')

                if self.do_zero_shot and self.training:
                    # make_frozen_copy_of_subj_basis_generators() make a frozen copy of the original subj_basis_generators, 
                    # which is used to generate the subject embeddings for subject-single prompts.
                    self.make_frozen_copy_of_subj_basis_generators()

            else:
                print(f"Skipping loading subj_basis_generator from {adaface_ckpt_path}")

            for token_idx, km in enumerate(ckpt["placeholder_strings"]):
                # Mapped from km in ckpt to km2 in the current session. Partial matching is allowed.
                if (placeholder_mapper is not None) and (km in placeholder_mapper):
                    km2 = placeholder_mapper[km]
                else:
                    km2 = km

                try:
                    k2_token = self.get_tokens_for_string(km2, force_single_token=True)[0]
                except:
                    breakpoint()
                if km2 in self.string_to_token_dict:
                    if km2 in self.background_strings:
                        print(f"Duplicate key {km}->{km2} in {adaface_ckpt_path}. Ignored.")
                        continue

                    raise ValueError(f"Duplicate key {km}->{km2} in {adaface_ckpt_path}")

                # Merge the (possibly substituted) subject strings from the ckpt with 
                # self.subject_strings and self.background_strings.
                if km in ckpt_background_strings:
                    self.background_strings = list(set(self.background_strings + [km2]))
                    print("Add background string", km2)
                elif km not in self.background_strings:
                    # Add km2 to self.subject_strings, even if it's not in ckpt["subject_strings"].
                    # This is to be compatible with older ckpts which don't save ckpt["subject_strings"].
                    self.subject_strings = list(set(self.subject_strings + [km2]))
                    print("Add subject string", km2)

                # The mapping in string_to_token_dict is determined by the tokenizer. 
                # Shouldn't do the km->km2 mapping on string_to_token_dict.
                self.string_to_token_dict[km2] = k2_token

                if km in ckpt["token2num_vectors"]:
                    token2num_vectors[km2] = ckpt["token2num_vectors"][km]

                print(f"Loaded {km}->{km2} from {adaface_ckpt_path}")
                
            if "token2num_vectors" in ckpt and not self.skip_loading_token2num_vectors:
                self.set_num_vectors_per_subj_token(token2num_vectors)

        self.placeholder_strings = self.subject_strings + self.background_strings

        # Regenerate subject_string_dict, background_string_dict 
        # in case subject_strings or background_strings have been changed.
        self.subject_string_dict    = { s: True for s in self.subject_strings }
        self.background_string_dict = { s: True for s in self.background_strings }

    # make_frozen_copy_of_subj_basis_generators() is only used during training, to generate the subject embeddings for subject-single prompts.
    def make_frozen_copy_of_subj_basis_generators(self, dtype=torch.float16):
        # frozen_string_to_subj_basis_generator_dict won't be returned by optimized_parameters(),
        # so it won't be updated.
        self.frozen_string_to_subj_basis_generator_dict = copy.deepcopy(self.string_to_subj_basis_generator_dict)
        # Convert the frozen copy of subj_basis_generators to float16 to save RAM.
        self.frozen_string_to_subj_basis_generator_dict.to(dtype=dtype)
        print("Made a frozen copy of subj_basis_generators")

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def optimized_parameters(self):
        subj_basis_generator_param_list0 = list(self.string_to_subj_basis_generator_dict.parameters())
        subj_basis_generator_param_list = [ p for p in subj_basis_generator_param_list0 if p.requires_grad ]
        num_no_grad_params  = len(subj_basis_generator_param_list0) - len(subj_basis_generator_param_list)
        num_total_params    = len(subj_basis_generator_param_list0)
        print(f"Filtered out {num_no_grad_params} no-grad / {num_total_params} total parameters in subj_basis_generator_param_list0.")

        return subj_basis_generator_param_list

    def embedding_attractor_loss(self):
        loss = 0.
        num_placeholders = len(self.placeholder_strings)

        for key in self.placeholder_strings:
            optimized = self.static_subj_embs_dict[key]
            coarse = 0 #self.initial_embeddings[key]
            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_placeholders

        return loss

    def zs_subject_embedding_norm_loss(self):
        loss_static = 0.
        euc_loss_type   = 'l2'       # l1, l2. l2 is recommended.

        zs_subj_emb_bias_reg_weight = 0.1
        static_l2_loss_boost        = 5
        num_out_embeddings          = 0

        for key in self.placeholder_strings:
            subj_embeddings = self.static_subj_embs_dict[key]
            loss_subjemb_bias   = reg_loss(subj_embeddings, loss_type=euc_loss_type)
            num_out_embeddings += subj_embeddings.shape[1]
            curr_loss   = loss_subjemb_bias * zs_subj_emb_bias_reg_weight                
            loss_static = loss_static + curr_loss * static_l2_loss_boost

        loss_static /= num_out_embeddings

        return loss_static

    def embedding_reg_loss(self):
        self.loss_call_count += 1
        if self.do_zero_shot:
            return self.zs_subject_embedding_norm_loss()
        else:
            return self.embedding_attractor_loss()
