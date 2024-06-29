# Borrowed from ip-adapter resampler.py.
# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import CLIPVisionModel, CLIPTokenizer

import numpy as np
from torch import einsum
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput
from adaface.util import arc2face_inverse_face_prompt_embs, gen_gradient_scaler
from adaface.arc2face_models import CLIPTextModelWrapper
import sys
sys.modules['ldm'] = sys.modules['adaface']

def reshape_tensor(x, num_heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, num_heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, num_heads, length, -1)
    return x

# FFN. Added a Dropout layer at the end, so that it can still load the old ckpt.
def FeedForward(dim, mult=4, p_dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
        nn.Dropout(p_dropout),
    )

# IP-Adapter FaceID class. Only used in knn-faces.py.
# From: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter_faceid_separate.py
class IP_MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x
    
# group_dim: the tensor dimension that corresponds to the multiple groups.
class LearnedSoftAggregate(nn.Module):
    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim  = group_dim
        # num_feat = 1: element-wise score function & softmax.
        # num_feat > 1: the linear score function is applied to the last dim (features) of the input tensor. 
        self.num_feat   = num_feat
        self.feat2score = nn.Linear(num_feat, 1, bias=False)
        self.keepdim    = keepdim

    def forward(self, x, score_basis=None):
        # If there's only one mode, do nothing.
        if x.shape[self.group_dim] == 1:
            if self.keepdim:
                return x
            else:
                return x.squeeze(self.group_dim)
            
        # Assume the last dim of x is the feature dim.
        if score_basis is None:
            score_basis = x
        
        if self.num_feat == 1:
            mode_scores = self.feat2score(score_basis.unsqueeze(-1)).squeeze(-1)
        else:
            mode_scores = self.feat2score(score_basis)
        attn_probs  = mode_scores.softmax(dim=self.group_dim)
        x_aggr      = (x * attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
        return x_aggr
    
def LoRA_ExpandEmbs(input_dim, lora_rank, output_dim, num_modes, 
                    num_output_vecs, elementwise_affine=True, p_dropout=0.1):
    return nn.Sequential(
        # Project to [BS, lora_rank * output_dim * num_modes].
        # It takes a huge param size. 512 * 32 * 768 * 4 = 6,291,456.
        nn.Linear(input_dim, lora_rank * output_dim * num_modes, bias=False),
        # Reshape to [BS, lora_rank, output_dim].
        Rearrange('b (m q d) -> b m q d', q=lora_rank, m=num_modes, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        # Aggregate [BS, num_modes, loar_rank, output_dim] -> [BS, lora_rank, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False) if num_modes > 1 \
            else Rearrange('b () q d -> b q d'),
        nn.Dropout(p_dropout),
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, num_output_vecs].
        nn.Linear(lora_rank, num_output_vecs, bias=False),
        # Permute to [BS, num_output_vecs, output_dim].
        Rearrange('b d q -> b q d'),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(p_dropout),
    )

def ExpandEmbs(input_dim, output_dim, expansion_ratio, elementwise_affine=True, p_dropout=0.1):
    return nn.Sequential(
        # Project to [BS, num_output_vecs * output_dim].
        nn.Linear(input_dim, expansion_ratio * output_dim, bias=False),
        # Reshape to [BS, num_output_vecs, output_dim].
        Rearrange('b (e d) -> b e d', e=expansion_ratio, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(p_dropout),
    )

# Input: [BS, N, D].
def MultimodeProjection(input_dim, output_dim=-1, num_modes=4, elementwise_affine=True, p_dropout=0.1):
    if output_dim == -1:
        output_dim = input_dim

    return nn.Sequential(
            nn.Linear(input_dim, output_dim * num_modes, bias=False),
            # Reshape to [BS, num_output_vecs, output_dim].
            Rearrange('b n (m d) -> b n m d', m=num_modes, d=output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
            # If num_modes == 1, then simply remove the mode dim. Otherwise, aggregate the modes.
            LearnedSoftAggregate(num_feat=output_dim, group_dim=2, keepdim=False) if num_modes > 1 \
                else Rearrange('b n () d -> b n d'),
            nn.Dropout(p_dropout),
    )

# Low-rank to high-rank transformation.
def Lora2Hira(lora_rank, hira_rank, output_dim, num_modes, elementwise_affine=True, p_dropout=0.1):
    return nn.Sequential(        
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, hira_rank].
        nn.Linear(lora_rank, hira_rank * num_modes, bias=False),
        # Reshape and permute to [BS, num_modes, num_output_vecs, output_dim].
        Rearrange('b d (m q) -> b m q d', m=num_modes, q=hira_rank),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        # Aggregate [BS, num_modes, hira_rank, output_dim] -> [BS, hira_rank, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False) if num_modes > 1 \
            else Rearrange('b () q d -> b q d'),       
        nn.Dropout(p_dropout),    
    )

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, num_heads=8, elementwise_affine=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)

        self.to_q   = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv  = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latent_queries):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latent_queries = self.norm2(latent_queries)

        b, l, _ = latent_queries.shape

        q = self.to_q(latent_queries)
        kv_input = torch.cat((x, latent_queries), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.num_heads)
        k = reshape_tensor(k, self.num_heads)
        v = reshape_tensor(v, self.num_heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        attn = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class CrossAttention(nn.Module):
    # output_dim is always the same as input_dim.
    # num_q only matters when q_aware_to_v is True. 
    # If q_aware_to_v is False, query x in forward() is still usable.
    def __init__(self, input_dim, num_heads=6, p_dropout=0.05, 
                 identity_to_q=False, identity_to_k=False, identity_to_v=False, v_has_skip=True,
                 q_aware_to_v=True, num_q=416, v_repeat=4, q_aware_to_v_lora_rank=64,
                 identity_to_out=False, out_has_skip=False):
        super().__init__()
        dim_head  = input_dim // num_heads
        inner_dim = dim_head   * num_heads

        self.num_heads      = num_heads
        self.q_aware_to_v   = q_aware_to_v
        self.v_has_skip     = v_has_skip
        self.to_q = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=True) 
                    ) if not identity_to_q else nn.Identity()
        self.to_k = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=True) 
                    ) if not identity_to_k else nn.Identity()
        
        self.v_repeat = v_repeat
        self.num_q_group = num_q_group = num_q // v_repeat      # 416 / 4 = 104.

        # If q_aware_to_v is True, then self.to_v consists of num_q projections of input_dim to inner_dim.
        # Otherwise, self.to_v consists of a single projection of input_dim to inner_dim.
        if q_aware_to_v:
            # all_q_mid: 104 * 64 = 6656.
            all_q_mid = num_q_group * q_aware_to_v_lora_rank
            self.to_v = nn.Sequential(
                # number of params: 768 * 6656 = 5,111,808.
                # Input:  [BS, 16, 768]. Output: [BS, 16, 104*64] = [BS, 16, 6656].
                # Each 768-dim vec is dispersed into 104 64-dim vecs.
                nn.Linear(input_dim, all_q_mid, bias=False),
                nn.LayerNorm(all_q_mid, elementwise_affine=True),
                # Change the dim of the tensor to [BS, 6656, 16], as Conv1d transforms dim 1.
                Rearrange('b n q -> b q n', q=all_q_mid),
                # Each q_aware_to_v projection has its own linear layer.
                # The total number of parameters will be 6656*768 = 5,111,808.
                # Output: [BS, 104*768, 16]. Each 64 dim feature is expanded to 768 dim.
                nn.Conv1d(
                    in_channels=all_q_mid,
                    out_channels=num_q_group * input_dim,
                    kernel_size=1,
                    groups=num_q_group,
                    bias=False,
                ),
                # Output: [BS, 104, 16, 768].
                Rearrange('b (q d) n -> b q n d', q=num_q_group, d=input_dim),
                nn.LayerNorm(input_dim, elementwise_affine=True),
            )
        else:
            self.to_v = nn.Sequential(
                            nn.Linear(input_dim, inner_dim, bias=False),
                            nn.LayerNorm(inner_dim, elementwise_affine=True) 
                        ) if not identity_to_v else nn.Identity()

        if identity_to_out:
            assert not out_has_skip, "identity_to_out=True, then out_has_skip has to be False."

        if identity_to_out:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=False),
                nn.Dropout(p_dropout),
                nn.LayerNorm(inner_dim, elementwise_affine=True)
            )

        self.out_has_skip = out_has_skip
        self.attn_drop = nn.Dropout(p_dropout)

    def forward(self, x, context=None, attn_mat=None, return_attn=False):
        h = self.num_heads

        if context is None:
            context = x

        if attn_mat is None:
            # q: [BS, Q, D] -> [BS, Q, D].
            q = self.to_q(x)
            # k: [BS, L, D] -> [BS, L, D].
            k = self.to_k(context)
            # q: [6, 512, 128], k: [6, 17, 128].
            q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))

        if self.q_aware_to_v:
            # context: [BS, L, D]. v: [BS, Q, L, D].
            # There are effectively Q to_v projections.
            v = self.to_v(context)
            if self.v_has_skip:
                v = v + context.unsqueeze(1)
        else:
            # v: [BS, L, D].
            v = self.to_v(context)
            if self.v_has_skip:
                v = v + context

        #print(v.shape)

        if self.q_aware_to_v:
            # v: [6, 64, 17, 128].
            # v is query-specific, so there's an extra dim for the query.
            v = rearrange(v, 'b q n (h d) -> (b h) q n d', h=h)
            # Each v is for a query group with 512/64 = 8 queries.
            # So each v is repeated 8 times to match the number of queries.
            # v: [6, 64, 17, 128] -> [6, 512, 17, 128].
            v = v.repeat(1, self.v_repeat, 1, 1)
        else:
            v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        if attn_mat is None:
            scale = q.size(-1) ** -0.25
            sim = einsum('b i d, b j d -> b i j', q * scale, k * scale)
            # sim: [6, 64, 17]. 6: bs 1 * h 6.
            # attention, what we cannot get enough of
            # NOTE: the normalization is done across tokens, not across pixels.
            # So for each pixel, the sum of attention scores across tokens is 1.
            attn = sim.softmax(dim=-1)
            attn = self.attn_drop(attn)
            #print(attn.std())
        else:
            attn = attn_mat

        if self.q_aware_to_v:
            # attn: [6, 32, 17]. v: [6, 32, 17, 128]. 128: dim of each head. out: [6, 32, 128].
            # out is combined with different attn weights and v for different queries.
            out = einsum('b i j, b i j d -> b i d', attn, v)
        else:
            # v: [6, 17, 128]. out: [6, 32, 128].
            out = einsum('b i j, b j d -> b i d',   attn, v)

        # [6, 32, 128] -> [1, 32, 768].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if self.out_has_skip:
            out = self.to_out(out) + out
        else:
            out = self.to_out(out)

        if return_attn:
            return out, attn
        else:
            return out

class SubjBasisGenerator(nn.Module):
    def __init__(
        self,
        # number of cross-attention heads. Half of the number of heads 12 of OpenAI clip-vit-large-patch14:
        # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
        num_heads=6,                       
        num_id_vecs={ 'subj': 77, 'bg': 257 }, # number of identity vectors. 18: 16 face tokens + 2 extra tokens. 257: 257 CLIP tokens.
        num_out_embs_per_layer=4,             # num_out_embs. subj: 16. bg: 4.
        num_out_layers=16,                    # number of layers of output embeddings.
        image_embedding_dim=768,              # CLIP image feature dimension, as per config.json above.
        # DINO vits16 has 6 attention heads:
        # https://huggingface.co/facebook/dino-vits16/blob/main/config.json
        dino_embedding_dim=384,             # DINO object feature dimension for objects.
        output_dim=768,                     # CLIP text embedding input dimension.
        placeholder_is_bg: bool = False,    # Whether the placeholder is for the image background.
        prompt2token_proj_grad_scale: float = 0.4,  # Gradient scale for prompt2token_proj.
        zs_extra_words_scale: float = 0.5,     # Scale for extra words in the prompt2token_proj.
        learnable_hidden_state_weights_scheme: str = 'per-layer',  # none, per-layer.
        bg_prompt_translator_has_to_out_proj: bool = False,  # Whether the prompt_trans_layers have a to_out projection.
    ):
        super().__init__()

        self.placeholder_is_bg      = placeholder_is_bg
        self.num_out_layers         = num_out_layers
        self.num_out_embs_per_layer = num_out_embs_per_layer
        # subj: 64, bg: 32.
        self.num_out_embs           = num_out_layers * num_out_embs_per_layer
        self.output_dim             = output_dim
        # num_id_vecs should be the number of core ID embs, 16.
        # However, in such case, pos_embs is not used. So it doesn't matter if it's wrongly set.
        self.num_id_vecs = num_id_vecs['bg'] if placeholder_is_bg else num_id_vecs['subj']
        self.pos_embs    = nn.Parameter(torch.randn(1, self.num_id_vecs, output_dim))
        self.pos_embs_ln = nn.LayerNorm(output_dim)
        self.zs_extra_words_scale = zs_extra_words_scale
        self.output_scale           = output_dim ** -0.5
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        if not self.placeholder_is_bg:
            # [1, 384] -> [1, 16, 768].
            # TODO: use CLIPTextModelWrapper as obj_proj_in.
            self.obj_proj_in = ExpandEmbs(dino_embedding_dim, output_dim, expansion_ratio=self.num_id_vecs)

            # self.prompt2token_proj: [1, 16, 768] -> [1, 77, 768] (with paddings).
            # If self.placeholder_is_bg: prompt2token_proj is set to None.
            self.prompt2token_proj  = CLIPTextModelWrapper.from_pretrained('openai/clip-vit-large-patch14')
            self.prompt2token_proj_grad_scale = prompt2token_proj_grad_scale
            self.prompt2token_proj_grad_scaler = gen_gradient_scaler(prompt2token_proj_grad_scale)
            print(f"Subj prompt2token_proj initialized with grad scale of {prompt2token_proj_grad_scale}.")            
            # Freeze prompt2token_proj if prompt2token_proj_grad_scale is 0.
            # Set requires_grad to False for all parameters in prompt2token_proj, to save memory taken by the optimizer.
            if prompt2token_proj_grad_scale == 0:
                self.freeze_prompt2token_proj()

            self.prompt2token_proj_attention_multiplier = -1
            self.initialize_hidden_state_layer_weights(learnable_hidden_state_weights_scheme, 'cpu')
            self.pad_embeddings = None
            self.bg_proj_in = None
        else:
            # For background placeholders, face and object embeddings are not used as they are foreground.
            self.obj_proj_in  = None
            self.prompt2token_proj = None
            print("Bg prompt2token_proj is set to None.")

            self.bg_proj_in = nn.Sequential(
                nn.Linear(image_embedding_dim, output_dim, bias=False),
                nn.LayerNorm(output_dim),
            )

            self.latent_queries     = nn.Parameter(torch.randn(1, self.num_out_embs, output_dim))
            self.latent_queries_ln  = nn.LayerNorm(output_dim)

            self.bg_prompt_translator_has_to_out_proj = bg_prompt_translator_has_to_out_proj
            identity_to_v   = False
            v_has_skip      = not identity_to_v                         # True
            identity_to_out = not bg_prompt_translator_has_to_out_proj  # True
            out_has_skip    = not identity_to_out                       # False
            # prompt_translator has a to_v projection with skip connection, and doesn't have a to_out projection.
            # dim=768, num_heads=6.
            self.prompt_translator = \
                CrossAttention(input_dim=output_dim, num_heads=num_heads, p_dropout=0.05,
                                identity_to_q=False, identity_to_k=False, identity_to_v=identity_to_v,
                                q_aware_to_v=False,  v_has_skip=v_has_skip,
                                num_q=0, # When not q_aware_to_v, num_q is not referenced.
                                identity_to_out=identity_to_out,
                                out_has_skip=out_has_skip)
            ''' 
            prompt_translator: CLIPEncoder
            # https://github.com/huggingface/transformers/blob/1872bde7fc6a5d6796bd742bc2dc38eaf8069c5d/src/transformers/models/clip/modeling_clip.py#L566
            # CLIPEncoder.layers: 12 layers of CLIPEncoderLayer, each being
                (0): CLIPEncoderLayer(
                    (self_attn): CLIPAttention(
                        (k_proj): Linear(in_features=768, out_features=768, bias=True)
                        (v_proj): Linear(in_features=768, out_features=768, bias=True)
                        (q_proj): Linear(in_features=768, out_features=768, bias=True)
                        (out_proj): Linear(in_features=768, out_features=768, bias=True)
                    )
                    (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (mlp): CLIPMLP(
                        (activation_fn): QuickGELUActivation()
                        (fc1): Linear(in_features=768, out_features=3072, bias=True)
                        (fc2): Linear(in_features=3072, out_features=768, bias=True)
                    )
                    (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                )
            '''

        print(repr(self))

    # raw_id_embs: ArcFace embeddings for faces (not used since we have arc2face_id_embs), 
    # or DINO embeddings for objects.
    # arc2face_id_embs: [BS, 16, 768], the core identity embeddings generated by Arc2Face.
    def forward(self, arc2face_id_embs, clip_features=None, raw_id_embs=None, out_id_embs_scale=1.0,
                is_face=True, is_training=False, adaface_prompt_embs_inf_type='full_half_pad'):    
        
        if not self.placeholder_is_bg:
            BS = arc2face_id_embs.shape[0]
        else:
            # If bg, then arc2face_id_embs is set to None, but clip_features is not None.
            BS = clip_features.shape[0]

        adaface_prompt_embs = None
        if not hasattr(self, 'clip_tokenizer'):
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # No need to use raw_id_embs if placeholder_is_bg.
        if not self.placeholder_is_bg:
            if is_face:
                assert arc2face_id_embs is not None
                # arc2face_embs has been projected to the (modified) prompt embedding space 
                # by arc2face_forward_face_embs. This prompt embedding space is modified because Arc2Face finetuned
                # the text encoder and the U-Net.
                # in embedding_manager: [BS, 16, 768] -> [BS, 77, 768].
                # arc2face_id_embs is part of arc2face_embs: [BS, 77, 768] -> [BS, 16, 768].
                # adaface_prompt_embs is projected to the prompt embedding spaces. This is the 
                # original U-Net prompt embedding space.

                # hidden_state_layer_weights: [[0.9163], [0.9483], [2.0762]]
                hidden_state_layer_weights = self.hidden_state_layer_weights_grad_scaler(self.hidden_state_layer_weights)
                # return_emb_types: a list of strings, each string is among 
                # ['full', 'core', 'full_pad', 'full_half_pad', 'full_zeroed_extra', 'b_core_e'].
                # Using b_core_e is more computationally efficient than using full_zeroed_extra. 
                # But there is an unknow BUG that causes crash when using b_core_e. 
                if is_training:
                    return_emb_types = ['full_pad', 'core']
                else:
                    # adaface_prompt_embs_inf_type: default is full_half_pad, same as training.
                    return_emb_types = [adaface_prompt_embs_inf_type, 'core']

                if self.pad_embeddings is None:
                    self.generate_pad_embeddings()
                else:
                    self.pad_embeddings = self.pad_embeddings.to(arc2face_id_embs.device)

                with torch.set_grad_enabled(self.training and self.prompt2token_proj_grad_scale != 0):
                    # If list_extra_words is not None, then core_id_embs: [BS, 18, 768], three leading words, the 16 identity tokens 
                    # and (at most) two extra words in full_prompt_embs, without BOS and EOS.
                    # If list_extra_words is None, then core_id_embs: [BS, 16, 768], the 16 identity tokens in full_prompt_embs.
                    # hidden_state_layer_weights: [[0.9163], [0.9483], [2.0762]]
                    # zs_extra_words_scale is only effective when list_extra_words is not None.
                    # adaface_prompt_embs: [BS, 77, 768], core_id_embs: [BS, 16, 768].
                    adaface_prompt_embs, core_id_embs = \
                        arc2face_inverse_face_prompt_embs(self.clip_tokenizer, 
                                                          self.prompt2token_proj, 
                                                          arc2face_id_embs, 
                                                          list_extra_words=None,
                                                          return_emb_types=return_emb_types, 
                                                          pad_embeddings=self.pad_embeddings,
                                                          hidden_state_layer_weights=hidden_state_layer_weights,
                                                          input_max_length=77, zs_extra_words_scale=self.zs_extra_words_scale)
                # Reduce the update rate to prompt2token_proj.
                adaface_prompt_embs = self.prompt2token_proj_grad_scaler(adaface_prompt_embs)
                core_id_embs = self.prompt2token_proj_grad_scaler(core_id_embs)
            elif raw_id_embs is not None:
                # id_embs: [BS, 384] -> [BS, 18, 768].
                # obj_proj_in is expected to project the DINO object features to 
                # the token embedding space. So no need to use prompt2token_proj.
                id_embs = self.obj_proj_in(raw_id_embs)
            else:
                breakpoint()
        else:
            # Otherwise, context is the ad-hoc CLIP image features.
            # id_embs: [BS, 257, 768].
            id_embs = self.bg_proj_in(clip_features)

        if self.placeholder_is_bg:
            id_embs = id_embs + self.pos_embs_ln(self.pos_embs)
            latent_queries = self.latent_queries_ln(self.latent_queries).repeat(BS, 1, 1)
            # If bg, we don't have to use a specific attn layer for each 4-vec set. Instead, one attn layer can generate 257 embs, 
            # and we take the first 16*4=64.             
            # Output of prompt_translator is exactly num_out_embs == 64 tokens. id_embs_out: [BS, 64, 768].
            # prompt_translator: better named as bg_prompt_translator. It maps the bg features 
            # to bg prompt embeddings.
            with torch.set_grad_enabled(self.training):
                id_embs_out = self.prompt_translator(latent_queries, id_embs)
            # [BS, 64, 768] -> [BS, 16, 4, 768]
            id_embs_out = id_embs_out.reshape(BS, self.num_out_layers, -1, self.output_dim)
            adaface_subj_embs = id_embs_out * self.output_scale    # * 0.036
        else:
            # adaface_subj_embs: [BS, 16, 768] -> [BS, 1, 16, 768] -> [BS, 16, 16, 768]
            adaface_subj_embs = core_id_embs.unsqueeze(1).repeat(1, self.num_out_layers, 1, 1)
        
        # If out_id_embs_scale < 1, adaface_subj_embs is a mix of adaface_subj_embs and pad_embeddings.
        if out_id_embs_scale != 1:
            # pad_embeddings: [77, 768] -> [16, 768] -> [1, 1, 16, 768].
            pad_embeddings = self.pad_embeddings[4:4+self.num_out_embs_per_layer].unsqueeze(0).unsqueeze(0)
            adaface_subj_embs =   adaface_subj_embs * out_id_embs_scale \
                                + pad_embeddings    * (1 - out_id_embs_scale)
        
        return adaface_subj_embs, adaface_prompt_embs

    def initialize_hidden_state_layer_weights(self, learnable_hidden_state_weights_scheme, device):
        if learnable_hidden_state_weights_scheme == 'none':
            self.hidden_state_layer_weights = None
            # A grad scaler with alpha =1 is nn.Identity(), which outputs None given None as input.
            self.hidden_state_layer_weights_grad_scaler = gen_gradient_scaler(1)
            print("hidden_state_layer_weights is set to None.")

        elif learnable_hidden_state_weights_scheme == 'per-layer':
            # Learnable weights of the last 3 layers, initialized to putting more focus on the last layer.
            # 'per-layer': Different weights for different layers, but the same for different channels.
            # hidden_state_layer_weights: [3, 1].
            self.hidden_state_layer_weights = nn.Parameter(torch.tensor([[1.0], [2.0], [4.0]], device=device),
                                                            requires_grad=True)
            self.hidden_state_layer_weights_grad_scaler = gen_gradient_scaler(5)
            print("hidden_state_layer_weights initialized as per-layer [1, 2, 4], with grad scaler 5.")
        else:
            breakpoint()

    def generate_pad_embeddings(self):
        # clip_embeddings: CLIPTextEmbeddings instance. pad_embeddings is generated after 
        # prompt2token_proj is loaded from the finetuned weight. It seems such pad embeddings perform 
        # slightly better than the original pad embeddings.
        clip_embeddings = self.prompt2token_proj.text_model.embeddings
        # clip_embeddings() and clip_embeddings.token_embedding() differ in that 
        # clip_embeddings() adds positional embeddings, while clip_embeddings.token_embedding() doesn't.
        # Adding positional embeddings seems to help somewhat.
        # pad_tokens: pad_token_id 49407 repeated 77 times.
        # pad_token_id is the EOS token. But BOS is 49406.
        pad_tokens = torch.tensor([self.clip_tokenizer.pad_token_id]).to(clip_embeddings.token_embedding.weight.device).repeat(77)
        # pad_embeddings: [77, 768]. 
        pad_embeddings = clip_embeddings(pad_tokens)[0]
        # We don't allow face recon to influence the pad embeddings. 
        # Otherwise, face identity will leak into the pad embeddings.
        self.pad_embeddings = pad_embeddings.detach()

    def extend_prompt2token_proj_attention(self, begin_layer_idx=-1, end_layer_idx=-1, multiplier=2, noise_std=0.1):
        if multiplier > 1:
            num_extended_layers = self.prompt2token_proj.extend_clip_attention_MKV_multiplier(begin_layer_idx, end_layer_idx, multiplier, noise_std)
            self.prompt2token_proj_attention_multiplier = multiplier
            print(f"{num_extended_layers} layers in prompt2token_proj_attention are x{multiplier}")

    def freeze_prompt2token_proj(self):
        # If bg, then prompt2token_proj is set to None. Therefore no need to freeze it.
        # Then we don't have to check whether it's for subj or bg.
        if self.prompt2token_proj is not None:
            frozen_param_names = []
            for param_name, param in self.prompt2token_proj.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_param_names.append(param_name)
                # If param is already frozen, then no need to freeze it again.
            print(f"{len(frozen_param_names)} params in Subj prompt2token_proj is frozen.")
            #print(f"Frozen parameters:\n{frozen_param_names}")

    def __repr__(self):
        type_sig = 'subj' if not self.placeholder_is_bg else 'bg'
        # Fix compatability with the previous version.
        if not hasattr(self, 'bg_prompt_translator_has_to_out_proj'):
            self.bg_prompt_translator_has_to_out_proj = False
        if not hasattr(self, 'num_out_embs'):
            self.num_out_embs = -1
        return f"{type_sig} SubjBasisGenerator: num_out_embs={self.num_out_embs}, " \
               f"bg_prompt_translator_has_to_out_proj={self.bg_prompt_translator_has_to_out_proj}"
    
@dataclass
class BaseModelOutputWithPooling2(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attn_mask: Optional[torch.FloatTensor] = None

# Revised from CLIPVisionTransformer to support attention mask. 
# self: a CLIPVisionTransformer instance.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L821
# pixel_values: preprocessed B*C*H*W images. [BS, 3, 224, 224]
# attn_mask: B*H*W attention mask.
def CLIPVisionTransformer_forward(self, pixel_values = None, attn_mask=None, 
                                  output_attentions = None,
                                  output_hidden_states = None, return_dict = None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Visual tokens are flattended in embeddings().
        # self.embeddings: CLIPVisionEmbeddings.
        # hidden_states: [BS, 257, 1280]. 257: 16*16 (patch_embeds) + 1 (class_embeds).
        # 16*16 is output from Conv2d(3, 1280, kernel_size=(14, 14), stride=(14, 14), bias=False).
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        
        if attn_mask is not None:
            # feat_edge_size: 16.
            feat_edge_size = np.sqrt(hidden_states.shape[1] - 1).astype(int)
            # attn_mask: [BS, 512, 512] -> [BS, 1, 16, 16].
            attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(feat_edge_size, feat_edge_size), mode='nearest')
            # Flatten the mask: [BS, 1, 16, 16] => [BS, 1, 256].
            attn_mask = attn_mask.flatten(2)
            # Prepend 1 to the mask: [BS, 1, 256] => [BS, 1, 257]. 
            # This 1 corresponds to class_embeds, which is always attended to.
            attn_mask = torch.cat([torch.ones_like(attn_mask[:, :, :1]), attn_mask], dim=-1)
            attn_mask_pairs = torch.matmul(attn_mask.transpose(-1, -2), attn_mask).unsqueeze(1)
        else:
            attn_mask_pairs = None

        # encoder: CLIPEncoder.
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # New feature: (***The official documentation is wrong***)
            # attention_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length, sequence_length)`, *optional*):
            #                 Mask to avoid performing attention on pairs of token. Mask values selected in `[0, 1]`:
            #                 - 1 for pairs that are **not masked**,
            #                 - 0 for pairs that are **masked**.    
            # attention_mask is eventually used by CLIPEncoderLayer:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L370
            attention_mask=attn_mask_pairs,
            output_attentions=output_attentions,        # False
            output_hidden_states=output_hidden_states,  # True
            return_dict=return_dict,                    # True
        )

        # last_hidden_state: [BS, 257, 1280]
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # return_dict is True.
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling2(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            # Newly added: return resized flattened attention mask.
            # [BS, 1, 257] -> [BS, 257, 1]
            attn_mask=attn_mask.permute(0, 2, 1) if attn_mask is not None else None
        )


class CLIPVisionModelWithMask(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace vision_model.forward() with the new one that supports mask.
        self.vision_model.forward = CLIPVisionTransformer_forward.__get__(self.vision_model)
    
    def forward(self, pixel_values = None, attn_mask = None, output_attentions = None,
                output_hidden_states = None, return_dict = None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
    
