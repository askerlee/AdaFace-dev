import torch
from torch import nn
import torch.distributed as dist
import sys
#sys.modules['ldm.modules']                      = sys.modules['adaface']
#sys.modules['ldm.modules.subj_basis_generator'] = sys.modules['adaface.subj_basis_generator']
#sys.modules['ldm.modules.arc2face_models']      = sys.modules['adaface.arc2face_models']
from adaface.face_id_to_ada_prompt import create_id2ada_prompt_encoder
import numpy as np

from ldm.util import extract_first_index_in_each_instance, anneal_perturb_embedding, \
                     get_clip_tokens_for_string, get_embeddings_for_clip_tokens, \
                     scan_cls_delta_strings
                     
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
            subj_name_to_cls_delta_string=None,
            out_emb_dim=768,
            num_unet_ca_layers=16,
            layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            training_perturb_std_range=None,
            training_perturb_prob=None,
            subj_name_to_being_faces=None,   # subj_name_to_being_faces: a dict that maps subject names to is_face.
            cls_delta_string='person',
            cls_delta_token_weights=None,
            # During training,  prompt2token_proj_ext_attention_perturb_ratio is 0.1.
            # During inference, prompt2token_proj_ext_attention_perturb_ratio is 0. 
            prompt2token_proj_ext_attention_perturb_ratio=0, 
            # The following arguments are to be set in main.py or stable_txt2img.py.
            adaface_ckpt_paths=None,
            adaface_encoder_types=None, 
            enabled_encoders=None,
            extend_prompt2token_proj_attention_multiplier=1,
            num_static_img_suffix_embs=0,
            p_encoder_dropout=0,
            gen_ss_from_frozen_subj_basis_generator=False,
            multi_token_filler=',',
            unet_lora_modules=None,
            load_unet_attn_lora_from_ckpt=True,
            load_unet_ffn_lora_from_ckpt=True,
    ):
        super().__init__()

        self.rank = -1

        self.string_to_token_dict = OrderedDict()
        
        self.string_to_subj_basis_generator_dict = nn.ModuleDict()
        self.placeholder_to_emb_cache            = nn.ParameterDict() # These should not be optimized
        self.num_unet_ca_layers                  = num_unet_ca_layers

        self.subject_strings = subject_strings

        # subject_string_dict is used in ddpm.py.
        self.subject_string_dict    = { s: True for s in self.subject_strings }
        self.placeholder_strings    = list(subject_strings)

        self.set_training_perturb_specs(training_perturb_std_range, training_perturb_prob)

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
        self.token2num_vectors = {}
        self.out_emb_dim = out_emb_dim
        self.p_encoder_dropout = p_encoder_dropout

        # Save this function to be used in load() when doing placeholder substitution.
        self.get_tokens_for_string       = partial(get_clip_tokens_for_string,       text_embedder.tokenizer)
        self.get_embeddings_for_tokens   = partial(get_embeddings_for_clip_tokens,   text_embedder.transformer.text_model.embeddings)

        # "," -> 267, "z": 345, "y": 344.
        self.subj_idx_to_cls_delta_tokens   = {}
        self.subj_idx_to_cls_delta_token_weights  = {}
        self.placeholder_token_to_idx       = {}

        self.cls_delta_string  = cls_delta_string
        self.prompt2token_proj_ext_attention_perturb_ratio = prompt2token_proj_ext_attention_perturb_ratio

        self.adaface_encoder_types = adaface_encoder_types
        self.enabled_encoders = enabled_encoders        
        # The embedding manager is primarily used during training, so we set is_training=True.
        self.id2ada_prompt_encoder = \
            create_id2ada_prompt_encoder(adaface_encoder_types, 
                                         num_static_img_suffix_embs=num_static_img_suffix_embs,
                                         enabled_encoders=enabled_encoders,
                                         extend_prompt2token_proj_attention_multiplier=extend_prompt2token_proj_attention_multiplier,
                                         prompt2token_proj_ext_attention_perturb_ratio=prompt2token_proj_ext_attention_perturb_ratio,
                                         is_training=True, img2txt_dtype=torch.float32)
        
        # gen_ss_from_frozen_subj_basis_generator: generate subject-single ada embeddings from a frozen encoder,
        # to avoid the encoder from slowly degenerating during training.
        self.gen_ss_from_frozen_subj_basis_generator = gen_ss_from_frozen_subj_basis_generator
        # No need to explicity set the params of subj_basis_generator_frozen to be requires_grad = False,
        # as it's not included in the param list returned by .optimized_parameters().
        if self.gen_ss_from_frozen_subj_basis_generator:
            self.subj_basis_generator_frozen = copy.deepcopy(self.id2ada_prompt_encoder.subj_basis_generator)
        else:
            self.subj_basis_generator_frozen = None

        if self.cls_delta_string is not None:
            self.cls_delta_tokens = self.get_tokens_for_string(cls_delta_string)
            if cls_delta_token_weights is None:
                self.cls_delta_token_weights = torch.ones(len(self.cls_delta_tokens))
                self.cls_delta_token_weights[-1] = 2
            else:
                self.cls_delta_token_weights = torch.tensor(cls_delta_token_weights, dtype=float)
            # The last word is the main word "man, woman, boy, girl" whose weight will be normalized to 1;
            # if there are any words before this word, their weights will be normalized to 0.25.
            self.cls_delta_token_weights **= 2
            self.cls_delta_token_weights /= self.cls_delta_token_weights.max()
        else:
            self.cls_delta_tokens = None
            self.cls_delta_token_weights = None

        for placeholder_idx, placeholder_string in enumerate(self.placeholder_strings):
            # get_tokens_for_string <= get_clip_tokens_for_string.
            # force_single_token = True, as there should be only one token in placeholder_string.
            placeholder_token = self.get_tokens_for_string(placeholder_string, force_single_token=True)[0].item()
            self.string_to_token_dict[placeholder_string] = placeholder_token

            # The subject SubjBasisGenerator will be initialized in self.id2ada_prompt_encoder.
            # If adaface_encoder_types contains multiple encoders (i.e, an actual type of 'jointIDs')
            # then .subj_basis_generator is not a SubjBasisGenerator instance,
            # but rather an nn.ModuleList of SubjBasisGenerator instances.
            # Such an instance should not be called to get the adaface_subj_embs, but only be used for save/load.
            subj_basis_generator = self.id2ada_prompt_encoder.subj_basis_generator

            self.string_to_subj_basis_generator_dict[placeholder_string] = subj_basis_generator

            self.token2num_vectors[placeholder_string] = self.id2ada_prompt_encoder.num_id_vecs 
            if num_static_img_suffix_embs > 0:
                # num_static_img_suffix_embs is the total number of multiple static_img_suffix_embs for multiple encoders.
                self.token2num_vectors[placeholder_string] += self.id2ada_prompt_encoder.num_static_img_suffix_embs

        # ',' is used as filler tokens.
        self.multi_token_filler = multi_token_filler
        self.string_to_token_dict[self.multi_token_filler] = \
            self.get_tokens_for_string(self.multi_token_filler, force_single_token=True)[0].item()

        self.unet_lora_modules = unet_lora_modules
        # self.load() loads subj SubjBasisGenerators.
        # The FaceID2AdaPrompt.load_adaface_ckpt() only loads fg SubjBasisGenerators.
        # So we don't pass adafaec_ckpt_paths to create_id2ada_prompt_encoder(), 
        # but instead use self.load().
        if adaface_ckpt_paths is not None:
            self.load(adaface_ckpt_paths, load_unet_attn_lora_from_ckpt, load_unet_ffn_lora_from_ckpt)

        # Initialize self.subj_name_to_cls_delta_tokens.
        self.init_cls_delta_tokens(self.get_tokens_for_string, subj_name_to_cls_delta_string, cls_delta_string)
        self.init_subj_name_to_being_faces(subj_name_to_being_faces)

        self.layer_idx = -1
        self.clear_prompt_adhoc_info()
        # 'recon_iter', 'unet_distill_iter', 'compos_distill_iter', 'plain_text_iter'.
        self.iter_type = None       
        self.set_curr_batch_subject_names(["default"])
        # real_batch_size = 10000: an arbitrary large number. 
        # Will be updated through set_image_prompts_and_iter_type() during training.
        self.set_image_prompts_and_iter_type(None, None, 'plain_text_iter', real_batch_size=10000)

        self.loss_call_count = 0

        # image_prompt_dict have two keys: 'subj', 'bg'.
        # image_prompt_dict['subj'] is the image prompt embs for the subject, which is ready to be used.
        # image_prompt_dict['bg'] is the clip features for the background, 
        # which needs to be translated by a bg SubjBasisGenerator.
        self.image_prompt_dict = {}

        print("EmbeddingManager on subj={}, init with {} vec(s)".format(
               self.subject_strings, self.token2num_vectors))
        
        # Add the search span by 1, just to be safe.
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN += 1
        print(f"CLS_DELTA_STRING_MAX_SEARCH_SPAN={self.CLS_DELTA_STRING_MAX_SEARCH_SPAN}")
    
    def init_cls_delta_tokens(self, get_tokens_for_string, subj_name_to_cls_delta_string, 
                              cls_delta_string=None):
        if subj_name_to_cls_delta_string is None:
            subj_name_to_cls_delta_string = {}
        if cls_delta_string is not None:
            # During inference, subj_name_to_cls_delta_string contains 'default' as the subject name, and maps
            # to cls_delta_string, the default class delta string.
            subj_name_to_cls_delta_string['default'] = cls_delta_string

        # We don't know the gender of a rand_id_to_img_prompt subject.
        subj_name_to_cls_delta_string['rand_id_to_img_prompt'] = 'person'

        self.subj_name_to_cls_delta_string  = subj_name_to_cls_delta_string
        self.subj_name_to_cls_delta_tokens  = {}
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN = 0

        for subj_name in self.subj_name_to_cls_delta_string:
            cls_delta_string = self.subj_name_to_cls_delta_string[subj_name]
            cls_delta_tokens = get_tokens_for_string(cls_delta_string)
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
        self.subj_name_to_being_faces['default']            = True

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    def forward(
            self,
            tokenized_text,         # [B, N]. 
            embedded_text,          # [B, N, 768]. 
    ):
        # placeholder_indices will be regenerated with update_placeholder_indices() 
        # within update_text_embeddings().
        self.clear_prompt_adhoc_info()
        
        # We need to clone embedded_text, as the modification in update_text_embeddings() 
        # will be in-place. 
        static_embeded_text, tokenized_text_repeated = \
                self.update_text_embeddings(tokenized_text, embedded_text.clone())

        # Update the prompt token embedding mask.
        self.update_prompt_masks(tokenized_text, tokenized_text_repeated)

        return static_embeded_text
    
    def update_text_embeddings(self, tokenized_text, embedded_text):
        BS = tokenized_text.shape[0]
        N  = tokenized_text.shape[1]

        # Put dist.get_rank() here. We couldn't get the rank in __init__(), as the default process group has not been initialized 
        # at that time.
        if self.rank == -1:
            try:
                # During inference, dist.get_rank() will raise an exception.
                self.rank = dist.get_rank()
            except:
                self.rank = 0

        orig_tokenized_text = tokenized_text
        self.cls_delta_string_indices = []

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            if placeholder_string == self.multi_token_filler:
                continue

            # If there's only one vector per token, we can do a simple replacement
            placeholder_indices = torch.where(tokenized_text == placeholder_token)
            # No placeholder token is found in the current batch.
            if placeholder_indices[0].numel() == 0:
                continue
            
            # If multiple occurrences are found in a prompt, only keep the first as the subject.
            # Other occurrences are treated as part of the background prompt (this may happen if
            # composition image overlay is used).
            placeholder_indices_1st = extract_first_index_in_each_instance(placeholder_indices)
            # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
            # embedded_text[placeholder_indices]: [2, 768].  adaface_subj_embs: [1, K, 768].

            # REAL_OCCURS_IN_BATCH: the real number of occurrences of the placeholder in the current batch,
            # not repetitively counting the occurrences in the embedded_text repeated for M layers.
            REAL_OCCURS_IN_BATCH = placeholder_indices_1st[0].numel()
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

            # Generate the actual adaface_subj_embs on the fly.
            # adaface_subj_embs: [BS, K, 768] or [1, K, 768].

            # id2img_embs (ID embeddings only): [BS, 16, 768] or [BS, 4, 768].
            id2img_prompt_embs  = self.image_prompt_dict['subj'] if self.curr_subj_is_face else None
            if self.iter_type == 'compos_distill_iter':
                # Only use the first instance in the batch to generate the adaface_subj_embs,
                # as the whole batch is of the same subject.
                id2img_prompt_embs = id2img_prompt_embs[:1]

            # static_img_suffix_embs are supposed to narrow the domain gap between the teacher and student models.
            # So we don't use them in compos_distill_iter or recon_iter.
            enable_static_img_suffix_embs = (self.iter_type == 'unet_distill_iter')

            #for subj_basis_generator in self.id2ada_prompt_encoder.subj_basis_generator:
            #    subj_basis_generator.prompt2token_proj.to(torch.float16)
                
            # lens_subj_emb_segments: th length of subject embeddings in each encoder.
            # enable_static_img_suffix_embs=None: use default settings, i.e.,
            # consistentID enables  static_img_suffix_embs, 
            # arc2face     disables static_img_suffix_embs.
            adaface_subj_embs, lens_subj_emb_segments = \
                self.id2ada_prompt_encoder.generate_adaface_embeddings(
                        image_paths=None, face_id_embs=None,
                        img_prompt_embs=id2img_prompt_embs,
                        p_dropout=self.p_encoder_dropout if self.training else 0,
                        # If an encoder is dropped out, then the corresponding adaface_subj_embs
                        # will not be included in the returned adaface_subj_embs. 
                        # So the returned adaface_subj_embs only contain valid embeddings and 
                        # could be shorter than self.token2num_vectors[placeholder_string].
                        return_zero_embs_for_dropped_encoders=False,
                        avg_at_stage=None,
                        enable_static_img_suffix_embs=enable_static_img_suffix_embs,
                    )
            # adaface_subj_embs should never be None, since we have made sure that not all encoders are dropped out,
            # and the passed in id2img_prompt_embs are always valid (even if no faces are detected in the input image,
            # we fill in random values). The following is just in case.
            if adaface_subj_embs is None:
                adaface_subj_embs = torch.zeros(REAL_OCCURS_IN_BATCH, self.out_emb_dim, device=embedded_text.device)
                breakpoint()

            # adaface_subj_embs: [BS, K, 768].
            # In a mix prompt batch (either compos_distill_iter or recon_iter with delta loss), 
            # REAL_OCCURS_IN_BATCH counts the number of subject-single and subject-comp instances.
            # But adaface_subj_embs is generated from the subject-single instance only.
            # Repeating at dim 0 is correct even if adaface_subj_embs has a batch size > 1:
            # If the subject-single batch is like [s1, s2], then the repeated batch is [s1, s2, s1, s2], 
            # matching the batch structure of (subject-single, subject-single, ...).
            if adaface_subj_embs.shape[0] < REAL_OCCURS_IN_BATCH:
                # adaface_subj_embs.shape[0] < self.real_batch_size should only happen in a compos_distill_iter, 
                # in which all the subjects are the same and adaface_subj_embs.shape[0] == 1.
                # In a recon_iter, even if adaface_subj_embs.shape[0] > 0, it may still < REAL_OCCURS_IN_BATCH,
                # when the prompts are a batch of the 4-type prompts.
                # (in that case, adaface_subj_embs.shape[0] == self.real_batch_size == REAL_OCCURS_IN_BATCH / 2)
                # So we check self.real_batch_size instead of REAL_OCCURS_IN_BATCH.
                if adaface_subj_embs.shape[0] < self.real_batch_size and self.iter_type != 'compos_distill_iter':
                    breakpoint()
                adaface_subj_embs = adaface_subj_embs.repeat(REAL_OCCURS_IN_BATCH // adaface_subj_embs.shape[0], 1, 1)

            # Replace the first adaface_subj_embs with adaface_subj_embs0 from a frozen encoder.
            if self.iter_type == 'compos_distill_iter' and self.gen_ss_from_frozen_subj_basis_generator:
                subj_basis_generator = self.id2ada_prompt_encoder.subj_basis_generator
                self.id2ada_prompt_encoder.subj_basis_generator = self.subj_basis_generator_frozen
                # In a compos_distill_iter, all subjects are the same. As implemented above, 
                # we only use the first instance in the batch to generate the adaface_subj_embs,
                # as the whole batch is of the same subject. Therefore, adaface_subj_embs0 
                # is only the embeddings of the first instance.
                # Since subj_basis_generator_frozen is frozen, there's no point to enable gradient flow.
                with torch.no_grad():
                    adaface_subj_embs0, lens_subj_emb_segments = \
                        self.id2ada_prompt_encoder.generate_adaface_embeddings(
                            image_paths=None, face_id_embs=None,
                            img_prompt_embs=id2img_prompt_embs,
                            p_dropout=self.p_encoder_dropout if self.training else 0,
                            # If an encoder is dropped out, then the corresponding adaface_subj_embs
                            # will not be included in the returned adaface_subj_embs. 
                            # So the returned adaface_subj_embs only contain valid embeddings and 
                            # could be shorter than self.token2num_vectors[placeholder_string].
                            return_zero_embs_for_dropped_encoders=False,
                            avg_at_stage=None,
                            enable_static_img_suffix_embs=enable_static_img_suffix_embs,
                        )
                self.id2ada_prompt_encoder.subj_basis_generator = subj_basis_generator

                # Replace the first adaface_subj_embs with adaface_subj_embs0.
                # adaface_subj_embs0: [1, 16, 768]. adaface_subj_embs: [2, 16, 768], 
                # which has been repeated twice from [1, 16, 768] above.
                # Still allow a small gradient into the unfrozen subj-single embeddings, 
                # maybe this will make the images more natural?
                adaface_subj_embs[:1] = adaface_subj_embs[:1] * 0.1 + adaface_subj_embs0 * 0.9

            adaface_subj_embs = adaface_subj_embs.to(embedded_text.dtype)
            # adaface_subj_embs could be shorter than self.token2num_vectors[placeholder_string], but never longer.
            if adaface_subj_embs.shape[1] > self.token2num_vectors[placeholder_string]:
                breakpoint()

            # adaface_subj_embs could be shorter than self.token2num_vectors[placeholder_string],
            # due to random dropout of some adaface encoders.
            # So we always replace the first adaface_subj_embs.shape[1] embeddings.
            k2 = 0
            for k in range(adaface_subj_embs.shape[1]):
                adaface_subj_emb_k = adaface_subj_embs[:, k]
                
                if self.training and self.training_perturb_std_range is not None:
                    # The std of adaface_subj_embs is around 0.07, times training_perturb_std_range
                    # (0.05 ~ 0.1) is very small. Therefore, it won't hurt the subject identity encoded
                    # in the embeddings.
                    adaface_subj_emb_k = \
                        anneal_perturb_embedding(adaface_subj_emb_k, 0, 
                                                 self.training_perturb_std_range, None,
                                                 self.training_perturb_prob[self.iter_type],
                                                 perturb_std_is_relative=True, keep_norm=False,
                                                 verbose=False)
                    
                if adaface_subj_emb_k.shape[0] != REAL_OCCURS_IN_BATCH:
                    breakpoint()

                # placeholder_indices_k is the indices of the k-th placeholder token of the whole batch.
                placeholder_indices_k = (placeholder_indices_1st[0], placeholder_indices_1st[1] + k2)
                # There could be a gap between "z , , , ..." and ", , , ...". For example, during inference,
                # the original prompt is repeated twice for the two encoders. The first segment contains 3 ", ",
                # and the second segment contains 15 ", ". The second copy of the prompt is between the two segments.
                while k2 < N and \
                  ((tokenized_text[placeholder_indices_k] != placeholder_token) & \
                   (tokenized_text[placeholder_indices_k] != self.string_to_token_dict[self.multi_token_filler])).any():
                    k2 += 1
                    # Check the k2-th token.
                    placeholder_indices_k = (placeholder_indices_1st[0], placeholder_indices_1st[1] + k2)

                if k2 >= N:
                    breakpoint()

                # Assign the k-th token embedding (along the text dim).
                # Note adaface_subj_emb_k is a batch of the k-th ID embeddings (of possibly different subjects), 
                # not a single embedding.
                embedded_text[placeholder_indices_k] = adaface_subj_emb_k
                k2 += 1
                
            # Cache the placeholder indices for mix prompt distillation.
            # Note placeholder_indices are recomputed in update_placeholder_indices(), 
            # But we need them without repetitions for mix prompt distillation.
            # If num_vectors_per_subj_token > 1, then repeat the indices and add to offsets.
            self.update_placeholder_indices(orig_tokenized_text, placeholder_string, placeholder_token, 
                                            adaface_subj_embs.shape[1])

        return embedded_text, tokenized_text

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

    # clear_prompt_adhoc_info() is called in ddpm.py:guided_denoise() when switching prompts.
    def clear_prompt_adhoc_info(self):
        self.placeholder2indices    = {}
        self.prompt_emb_mask        = None

    # During training,  set_curr_batch_subject_names() is called in ddpm.py.
    # During inference, set_curr_batch_subject_names() is called by the embedding manager.
    def set_curr_batch_subject_names(self, subj_names):
        self.curr_batch_subj_names = subj_names
        # During inference, as self.curr_batch_subj_names is not set, the three dicts are empty.
        self.current_subj_name_to_cls_delta_tokens = { subj_name: self.subj_name_to_cls_delta_tokens[subj_name] \
                                                       for subj_name in self.curr_batch_subj_names }
                
        # During training, we get the current subject name from self.curr_batch_subj_names, then map to 
        # curr_subj_is_face. 
        # During inference, curr_batch_subj_names = ['default'], which maps to True in subj_name_to_being_faces,
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

        print(f"{self.rank} subjects: {self.curr_batch_subj_names}, cls_delta_strings: {self.cls_delta_strings}")

    def update_placeholder_indices(self, tokenized_text, placeholder_string, placeholder_token, num_vectors_per_subj_token):
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

    def set_training_perturb_specs(self, training_perturb_std_range, training_perturb_prob):
        self.training_perturb_std_range = training_perturb_std_range
        self.training_perturb_prob      = training_perturb_prob

        if training_perturb_std_range is None:
            print(f"Disable training perturbance")
        else:
            print(f"training perturbance std range: {training_perturb_std_range}"
                  f", with prob = {training_perturb_prob}")

    def set_image_prompts_and_iter_type(self, id2img_prompt_embs, clip_bg_features, 
                                        iter_type, real_batch_size):
        #                     consistentID       arc2face          jointIDs
        # id2img_prompt_embs: [1, 16, 768]   or [1, 4, 768]    or [1, 20, 768].
        # clip_bg_features:   [1, 257, 1280] or [1, 257, 1024] or [1, 257, 1280+1024].
        self.image_prompt_dict = { 'subj': id2img_prompt_embs,
                                   'bg':   clip_bg_features,
                                 }
        self.iter_type = iter_type
        self.real_batch_size = real_batch_size
        # In a compos_distill_iter, all subjects are the same. So we only keep the first cls_delta_string.
        if self.cls_delta_strings is not None and self.iter_type == 'compos_distill_iter':
            self.cls_delta_strings = self.cls_delta_strings[:1]

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, adaface_ckpt_path):
        saved_dict = {  "string_to_subj_basis_generator_dict":  self.string_to_subj_basis_generator_dict,
                        "placeholder_strings":                  self.placeholder_strings,
                        "subject_strings":                      self.subject_strings,
                     }
        
        if self.unet_lora_modules is not None:
            saved_dict["unet_lora_modules"] = self.unet_lora_modules.state_dict()

        torch.save(saved_dict, adaface_ckpt_path)

    # Load custom tokens and their learned embeddings from "embeddings_gs-4500.pt".
    def load(self, adaface_ckpt_paths, load_unet_attn_lora_from_ckpt=True, load_unet_ffn_lora_from_ckpt=True):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        self.string_to_token_dict   = {}
        self.subject_strings        = []

        # Load adaface_ckpt_paths[0] into id2ada_prompt_encoder.subj_basis_generator.
        self.id2ada_prompt_encoder.load_adaface_ckpt(adaface_ckpt_paths[0])
        print(f"Loaded {adaface_ckpt_paths[0]} into trainable subj_basis_generator")

        if self.gen_ss_from_frozen_subj_basis_generator:
            if len(adaface_ckpt_paths) >= 2:
                frozen_ckpt_path = adaface_ckpt_paths[1]
            else:
                # Use the same ckpt as the frozen subj_basis_generator.
                frozen_ckpt_path = adaface_ckpt_paths[0]

            # Back up id2ada_prompt_encoder.subj_basis_generator.
            subj_basis_generator = self.id2ada_prompt_encoder.subj_basis_generator
            # Temporarily replace id2ada_prompt_encoder.subj_basis_generator with subj_basis_generator_frozen.
            self.id2ada_prompt_encoder.subj_basis_generator = self.subj_basis_generator_frozen
            # Load frozen_ckpt_path into id2ada_prompt_encoder.subj_basis_generator.
            self.id2ada_prompt_encoder.load_adaface_ckpt(frozen_ckpt_path)
            print(f"Loaded {frozen_ckpt_path} into trainable subj_basis_generator")
            # Restore id2ada_prompt_encoder.subj_basis_generator.
            self.id2ada_prompt_encoder.subj_basis_generator = subj_basis_generator

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

            if "placeholder_strings" in ckpt:
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
                        print(f"Duplicate key {km}->{km2} in {adaface_ckpt_path}. Ignored.")
                        continue

                    # Add km2 to self.subject_strings, even if it's not in ckpt["subject_strings"].
                    # This is to be compatible with older ckpts which don't save ckpt["subject_strings"].
                    self.subject_strings = list(set(self.subject_strings + [km2]))
                    print("Add subject string", km2)

                    # The mapping in string_to_token_dict is determined by the tokenizer. 
                    # Shouldn't do the km->km2 mapping on string_to_token_dict.
                    self.string_to_token_dict[km2] = k2_token
                    print(f"Loaded {km}->{km2} from {adaface_ckpt_path}")
                
            if self.unet_lora_modules is not None and 'unet_lora_modules' in ckpt \
              and (load_unet_attn_lora_from_ckpt or load_unet_ffn_lora_from_ckpt):
                unet_lora_modules = ckpt['unet_lora_modules']
                if not load_unet_attn_lora_from_ckpt:
                    # Filter out the attention LoRA modules.
                    unet_lora_modules = { k: v for k, v in unet_lora_modules.items() if 'attn2_processor' not in k }
                if not load_unet_ffn_lora_from_ckpt:
                    # Filter out the FFN LoRA modules.
                    unet_lora_modules = { k: v for k, v in unet_lora_modules.items() if 'resnets' not in k }

                ret = self.unet_lora_modules.load_state_dict(unet_lora_modules, strict=False)
                print(f"Loaded {len(unet_lora_modules)} LoRA weights")

                if ret is not None and len(ret.missing_keys) > 0:
                    print(f"Missing keys: {ret.missing_keys}")
                if ret is not None and len(ret.unexpected_keys) > 0:
                    print(f"Unexpected keys: {ret.unexpected_keys}")

            elif self.unet_lora_modules is None:
                # unet unet_lora_modules are not enabled.
                continue
            elif 'unet_lora_modules' not in ckpt:
                print(f"'unet_lora_modules' not found in {adaface_ckpt_path}")
            elif not load_unet_attn_lora_from_ckpt and not load_unet_ffn_lora_from_ckpt:
                print(f"Instructed to skip loading unet_lora_modules from {adaface_ckpt_path}")

        # ',' is used as filler tokens.
        self.string_to_token_dict[self.multi_token_filler] = \
            self.get_tokens_for_string(self.multi_token_filler, force_single_token=True)[0].item()
        self.placeholder_strings = self.subject_strings
        # Regenerate subject_string_dict, in case subject_strings have been changed.
        # subject_string_dict is used in ddpm.py.
        self.subject_string_dict = { s: True for s in self.subject_strings }

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def optimized_parameters(self):
        subj_basis_generator_param_list0 = list(self.string_to_subj_basis_generator_dict.parameters())
        subj_basis_generator_param_list = [ p for p in subj_basis_generator_param_list0 if p.requires_grad ]
        num_no_grad_params  = len(subj_basis_generator_param_list0) - len(subj_basis_generator_param_list)
        num_total_params    = len(subj_basis_generator_param_list0)
        print(f"Filtered out {num_no_grad_params} no-grad / {num_total_params} total parameters in subj_basis_generator_param_list0.")

        if self.unet_lora_modules is not None:
            unet_loras_param_list = list(self.unet_lora_modules.parameters())
            for param in unet_loras_param_list:
                param.requires_grad = True
            print(f"Total parameters in unet_lora_modules: {len(unet_loras_param_list)}")
        else:
            unet_loras_param_list = []

        return subj_basis_generator_param_list + unet_loras_param_list
