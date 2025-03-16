import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import logging, is_torch_version, deprecate
from diffusers.utils.torch_utils import fourier_filter
# UNet is a diffusers PeftAdapterMixin instance.
from diffusers.loaders.peft import PeftAdapterMixin
from peft import LoraConfig, get_peft_model
import peft.tuners.lora as peft_lora
from peft.tuners.lora.dora import DoraLinearLayer
from einops import rearrange
import math, re
import numpy as np
from peft.tuners.tuners_utils import BaseTunerLayer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def dummy_func(*args, **kwargs):
    pass

def split_indices_by_instance(indices, as_dict=False):
    indices_B, indices_N = indices
    unique_indices_B = torch.unique(indices_B)
    if not as_dict:
        indices_by_instance = [ (indices_B[indices_B == uib], indices_N[indices_B == uib]) for uib in unique_indices_B ]
    else:
        indices_by_instance = { uib.item(): indices_N[indices_B == uib] for uib in unique_indices_B }
    return indices_by_instance

# If do_sum, returned emb_attns is 3D. Otherwise 4D.
# indices are applied on the first 2 dims of attn_mat.
def sel_emb_attns_by_indices(attn_mat, indices, all_token_weights=None, do_sum=True, do_mean=False):
    indices_by_instance = split_indices_by_instance(indices)
    
    # emb_attns[0]: [1, 9, 8, 64]
    # 8: 8 attention heads. Last dim 64: number of image tokens.
    emb_attns   = [ attn_mat[inst_indices].unsqueeze(0) for inst_indices in indices_by_instance ]
    if all_token_weights is not None:
        # all_token_weights: [4, 77].
        # token_weights_by_instance[0]: [1, 9, 1, 1].
        token_weights = [ all_token_weights[inst_indices].reshape(1, -1, 1, 1) for inst_indices in indices_by_instance ]
    else:
        token_weights = [ 1 ] * len(indices_by_instance)

    # Apply token weights.
    emb_attns = [ emb_attns[i] * token_weights[i] for i in range(len(indices_by_instance)) ]

    # sum among K_subj_i subj embeddings -> [1, 8, 64]
    if do_sum:
        emb_attns   = [ emb_attns[i].sum(dim=1) for i in range(len(indices_by_instance)) ]
    elif do_mean:
        emb_attns   = [ emb_attns[i].mean(dim=1) for i in range(len(indices_by_instance)) ]

    emb_attns = torch.cat(emb_attns, dim=0)
    return emb_attns

def gaussian_pdf_2d(x, y, x0, y0, std_x, std_y):
    """
    Evaluate the uncorrelated 2D Gaussian PDF at coordinates (x, y).
    x, y are torch tensors of the same shape.
    We don't need to normalize the PDF in the traditional way, as we'll normalize the center to 1,
    and the traditional normalization factor will be canceled out.
    """
    return torch.exp(-(((x - x0)**2)/(2*std_x**2) + ((y - y0)**2)/(2*std_y**2)))

@torch.compile
# Return a scale matrix, in which most elements are 1, and at the subject-embedding rows,
# the scales decrease from 1 at the Gaussian center to something around 0.3 at the image boundary.
# This scale matrix is intended to multiple with the attention prob matrix, so that the subject
# embeddings won't dominate the whole image.
def calc_subj_attn_scales(attn_score, subj_indices, subj_attn_var_shrink_factor):
    # attn_score: [1, 8, 4096, 77]
    # subj_indices: 
    '''
    (tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:1'), tensor([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23], device='cuda:1'))    
    '''
    if subj_attn_var_shrink_factor <= 1:
        breakpoint()

    # attn_score2: [1, 8, 4096, 77] -> [1, 77, 8, 4096]
    attn_score2 = attn_score.permute(0, 3, 1, 2)
    # subj_attn: [1, 20, 8, 4096]. 
    # do_sum=True: the returned subj_attn is summed over subject embeddings.
    subj_attn = sel_emb_attns_by_indices(attn_score2, subj_indices, do_sum=False)
    # N_SUB: Number of subject instances
    N_SUB, N_EMB = subj_attn.size(0), subj_attn.size(1)
    # Average over the heads. subj_attn: [1, 20, 4096]
    subj_attn = subj_attn.mean(dim=2)
    # 1) Normalize subj_attn as a 2D probability distribution.
    # To be precise, we should consider other tokens than the subject embeddings when computing subj_prob.
    # However, after normalization, the subj_prob below should only differ from the precise subj_pro
    # by a constant factor, which doesn't affect our estimation of the subject center and variances.
    subj_prob = subj_attn.softmax(dim=2)
    H = W = int(np.sqrt(attn_score2.size(-1)))
    # subj_attn: [1, 20, 4096] -> [1, 20, 64, 64]
    subj_prob_3d = subj_prob.view(N_SUB, N_EMB, H, W)
    # 2) Compute y_center and x_center
    # ys: [1, 64, 1]. xs: [1, 1, 64]
    ys = torch.arange(H, device=subj_attn.device).view(1, H, 1)
    xs = torch.arange(W, device=subj_attn.device).view(1, 1, W)
    y_center = (subj_prob_3d * ys).sum(dim=(2,3))  # [N_SUB, N_EMB]
    x_center = (subj_prob_3d * xs).sum(dim=(2,3))  # [N_SUB, N_EMB]
    # 3) Variances
    y_sq = (subj_prob_3d * ys**2).sum(dim=(2,3))   # E[Y^2]
    x_sq = (subj_prob_3d * xs**2).sum(dim=(2,3))   # E[X^2]

    var_y = y_sq - y_center**2
    var_x = x_sq - x_center**2

    # 4) 2D std
    # y_center, x_center: [35.1875, 32.9062].  
    # std_x is often slightly smaller than std_y, meaning the face is slightly taller than wider.
    # std_x, std_y: [N_SUB, N_EMB]. The stds across embeddings are quite similar. 
    # So using embedding-specific stds may not make much difference. But who knows?
    '''
    std_x: [[18.9531, 18.1719, 19.0000, 19.2031, 19.0469, 19.0469, 18.2969, 18.2969,
             18.7969, 18.5000, 18.5000, 18.2031, 18.2031, 18.2031, 18.2031, 18.2031,
             18.2031, 18.2344, 19.2812, 18.5781]]    
    std_y: [[18.8750, 17.9375, 18.7188, 18.9062, 18.7344, 18.7344, 18.1719, 18.1875,
             18.7344, 18.4219, 18.3906, 18.2188, 18.1875, 18.1875, 18.2188, 18.1875,
             18.2188, 18.2188, 19.0000, 18.5469]]
    '''
    std_x  = var_x.sqrt()
    std_y  = var_y.sqrt()

    # BUG: the multiplication of the original subj attn probs with pdf_grid is incorrect.
    # In particular, if subj_attn_var_shrink_factor = 2, then the new var_x or var_y is 
    # 1/2 of the original, which is correct.
    # But for other values, it doesn't lead to a new var of var_x / subj_attn_var_shrink_factor or 
    # std_y / subj_attn_var_shrink_factor.
    # Nonetheless, it does lead to smaller var_x and var_y, and 
    # as subj_attn_var_shrink_factor increases,
    # the new var_x or var_y decreases. So we'll still use this equation.
    shrinker_std_x = std_x * 1 / (subj_attn_var_shrink_factor - 1)
    shrinker_std_y = std_y * 1 / (subj_attn_var_shrink_factor - 1)
    # Create 1D coordinate arrays (the range can be chosen as needed)
    # Make a meshgrid. X: [64, 64, 1, 1], Y: [64, 64, 1, 1]
    X, Y = torch.meshgrid(torch.arange(W, device=subj_attn.device), 
                          torch.arange(H, device=subj_attn.device))
    X = X.to(attn_score.dtype).unsqueeze(-1).unsqueeze(-1)
    Y = Y.to(attn_score.dtype).unsqueeze(-1).unsqueeze(-1)
    # shrinker_grid: [64, 64, 1, 20]
    shrinker_grid = gaussian_pdf_2d(X, Y, x_center, y_center, shrinker_std_x, shrinker_std_y)
    # Normalize shrinker_grid, so that the maximum value (at the point nearest to the center) 
    # is always 1, i.e., the subject activation at this point is not scaled down.
    # NOTE: shrinker_grid values are in the log scale. So most of them are negative.
    # shrinker_grid.mean(): 0.2~0.4.
    shrinker_grid = shrinker_grid / shrinker_grid.max().detach()
    # shrinker_grid: [1, 20, X=64, Y=64] -> [1, 20, Y=64, X=64] -> [1, 20, 1, 4096]
    shrinker_grid = shrinker_grid.permute(2, 3, 0, 1).reshape(N_SUB, N_EMB, 1, -1)
    # subj_attn_scales: [1, 77, 1, 4096]
    subj_attn_scales = torch.ones_like(attn_score2[:, :, :1])
    # subj_attn_scales[subj_indices]: [20, 1, 4096]. shrinker_grid will be broadcasted to them.
    subj_attn_scales[subj_indices] = shrinker_grid
    # subj_attn_scales: [1, 1, 4096, 77]
    subj_attn_scales = subj_attn_scales.permute(0, 2, 3, 1)
    # print(f"shrinker_grid mean: {shrinker_grid.mean().item():.4f}")
    return subj_attn_scales

# Slow implementation equivalent to F.scaled_dot_product_attention.
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 shrink_subj_attn=False, subj_indices=None, subj_attn_var_shrink_factor=2., 
                                 is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    B, L, S = query.size(0), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # 1: head (to be broadcasted). L: query length. S: key length.
    attn_bias = torch.zeros(B, 1, L, S, device=query.device, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(B, 1, L, S, device=query.device, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if shrink_subj_attn and subj_indices is not None:
        # subj_attn_scales: [1, 1, 4096, 77]. At the last dim, 1 everywhere except 
        # for the subject embeddings indexed by subj_indices.
        subj_attn_scales = calc_subj_attn_scales(attn_weight, subj_indices, subj_attn_var_shrink_factor)
    else:
        subj_attn_scales = 1

    # attn_bias: [1, 1, 4096, 77], the same size as a single-head attn_weight.
    attn_weight += attn_bias
    attn_score = attn_weight
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # subj_attn_scales: [1, 1, 4096, 77]. At the last dim, 1 everywhere except 
    # for the subject embeddings indexed by subj_indices.
    # NOTE: After scaling, the "probabilities" of the subject embeddings will sum to < 1.
    # But this is intended, as we want to scale down the impact of the subject embeddings 
    # in the computed attention output tensors.
    attn_weight = attn_weight * subj_attn_scales
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    output = attn_weight @ value
    return output, attn_score, attn_weight

# All layers share the same attention processor instance.
class AttnProcessor_LoRA_Capture(nn.Module):
    r"""
    Revised from AttnProcessor2_0
    """
    # lora_proj_layers is a dict of lora_layer_name -> lora_proj_layer.
    def __init__(self, capture_ca_activations: bool = False, enable_lora: bool = False, 
                 lora_uses_dora=True, lora_proj_layers=None, 
                 lora_rank: int = 192, lora_alpha: float = 16,
                 subj_attn_var_shrink_factor: float = 2.,
                 q_lora_updates_query=False, attn_proc_idx=-1):
        super().__init__()

        self.global_enable_lora = enable_lora
        self.attn_proc_idx = attn_proc_idx
        # reset_attn_cache_and_flags() sets the local (call-specific) self.enable_lora flag.
        # By default, shrink_subj_attn is False. Later in layers 22, 23, 24 it will be set to True.
        self.reset_attn_cache_and_flags(capture_ca_activations, False, enable_lora)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = self.lora_alpha / self.lora_rank
        self.subj_attn_var_shrink_factor = subj_attn_var_shrink_factor
        self.q_lora_updates_query = q_lora_updates_query

        self.to_q_lora = self.to_k_lora = self.to_v_lora = self.to_out_lora = None
        if self.global_enable_lora:
            for lora_layer_name, lora_proj_layer in lora_proj_layers.items():
                if lora_layer_name == 'q':
                    self.to_q_lora   = peft_lora.Linear(lora_proj_layer, 'default', r=lora_rank, lora_alpha=lora_alpha, 
                                                        use_dora=lora_uses_dora, lora_dropout=0.1)
                elif lora_layer_name == 'k':
                    self.to_k_lora   = peft_lora.Linear(lora_proj_layer, 'default', r=lora_rank, lora_alpha=lora_alpha, 
                                                        use_dora=lora_uses_dora, lora_dropout=0.1)
                elif lora_layer_name == 'v':
                    self.to_v_lora   = peft_lora.Linear(lora_proj_layer, 'default', r=lora_rank, lora_alpha=lora_alpha, 
                                                        use_dora=lora_uses_dora, lora_dropout=0.1)
                elif lora_layer_name == 'out':
                    self.to_out_lora = peft_lora.Linear(lora_proj_layer, 'default', r=lora_rank, lora_alpha=lora_alpha, 
                                                        use_dora=lora_uses_dora, lora_dropout=0.1)

    # LoRA layers can be enabled/disabled dynamically.
    def reset_attn_cache_and_flags(self, capture_ca_activations, shrink_subj_attn, enable_lora):
        self.capture_ca_activations = capture_ca_activations
        self.shrink_subj_attn     = shrink_subj_attn
        self.cached_activations     = {}
        # Only enable LoRA for the next call(s) if global_enable_lora is set to True.
        self.enable_lora = enable_lora and self.global_enable_lora

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
        subj_indices: Optional[Tuple[torch.IntTensor, torch.IntTensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        # hidden_states: [1, 4096, 320]
        residual = hidden_states
        # attn.spatial_norm is None.
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # Collapse the spatial dimensions to a single token dimension.
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        # NOTE: there's a inconsistency between q lora and k, v loras. 
        # k, v loras are directly applied to key and value (currently k, v loras are never enabled), 
        # while q lora is applied to query2, and we keep the query unchanged. 
        if self.enable_lora and self.to_q_lora is not None:
            # query2 will be used in ldm/util.py:calc_elastic_matching_loss() to get more accurate
            # cross attention scores between the latent images of the sc and mc instances.
            query2 = self.to_q_lora(hidden_states)
            # If not q_lora_updates_query, only query2 will be impacted by the LoRA layer.
            # The query, and thus the attention score and attn_out, will be the same 
            # as the original ones. 
            if self.q_lora_updates_query:
                query = query2
        else:
            query2 = query

        scale = 1 / math.sqrt(query.size(-1))

        is_cross_attn = (encoder_hidden_states is not None)
        if (not is_cross_attn) and (img_mask is not None):
            # NOTE: we assume the image is square. But this will fail if the image is not square.
            # hidden_states: [BS, 4096, 320]. img_mask: [BS, 1, 64, 64]
            # Scale the mask to the same size as hidden_states.
            mask_size = int(math.sqrt(hidden_states.shape[-2]))
            img_mask = F.interpolate(img_mask, size=(mask_size, mask_size), mode='nearest')
            if (img_mask.sum(dim=(2, 3)) == 0).any():
                img_mask = None
            else:
                # img_mask: [2, 1, 64, 64] -> [2, 4096]
                img_mask = rearrange(img_mask, 'b ... -> b (...)').contiguous()
                # max_neg_value = -torch.finfo(hidden_states.dtype).max
                # img_mask: [2, 4096] -> [2, 1, 1, 4096]
                img_mask = rearrange(img_mask.bool(), 'b j -> b () () j')
                # attn_score: [16, 4096, 4096]. img_mask will be broadcasted to [16, 4096, 4096].
                # So some rows in dim 1 (e.g. [0, :, 4095]) of attn_score will be masked out (all elements in [0, :, 4095] is -inf).
                # But not all elements in [0, 4095, :] is -inf. Since the softmax is done along dim 2, this is fine.
                # attn_score.masked_fill_(~img_mask, max_neg_value)
                # NOTE: If there's an attention mask, it will be replaced by img_mask.
                attention_mask = img_mask

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.enable_lora and self.to_k_lora is not None:
            key = self.to_k_lora(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)

        if self.enable_lora and self.to_v_lora is not None:
            value = self.to_v_lora(encoder_hidden_states)
        else:
            value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            query2 = attn.norm_q(query2)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query  = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        query2 = query2.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key   = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        if is_cross_attn and (self.capture_ca_activations or self.shrink_subj_attn):
            hidden_states, attn_score, attn_prob = \
                scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, 
                                             dropout_p=0.0, shrink_subj_attn=self.shrink_subj_attn,
                                             subj_indices=subj_indices,
                                             subj_attn_var_shrink_factor=self.subj_attn_var_shrink_factor)
        else:
            # Use the faster implementation of scaled_dot_product_attention 
            # when not capturing the activations or suppressing the subject attention.
            hidden_states = \
                F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            attn_prob = attn_score = None

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        if self.enable_lora and self.to_out_lora is not None:
            hidden_states = self.to_out_lora(hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if is_cross_attn and self.capture_ca_activations:
            # cached q will be used in ddpm.py:calc_comp_fg_bg_preserve_loss(), in which two qs will multiply each other.
            # So sqrt(scale) will scale the product of two qs by scale.
            # ANCHOR[id=attention_caching]
            # query: [2, 8, 4096, 40] -> [2, 320, 4096]
            self.cached_activations['q'] = \
                rearrange(query, 'b h n d -> b (h d) n').contiguous() * math.sqrt(scale)
            self.cached_activations['q2'] = \
                rearrange(query2, 'b h n d -> b (h d) n').contiguous() * math.sqrt(scale)
            self.cached_activations['k'] = \
                rearrange(key, 'b h n d -> b (h d) n').contiguous() * math.sqrt(scale)
            # attn_prob, attn_score: [2, 8, 4096, 77]
            self.cached_activations['attn'] = attn_prob
            self.cached_activations['attnscore'] = attn_score
            # attn_out: [b, n, h * d] -> [b, h * d, n]
            # [2, 4096, 320] -> [2, 320, 4096].
            self.cached_activations['attn_out'] = hidden_states.permute(0, 2, 1).contiguous()

        return hidden_states

def CrossAttnUpBlock2D_forward_capture(
    self,
    hidden_states: torch.Tensor,
    res_hidden_states_tuple: Tuple[torch.Tensor, ...],
    temb: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    self.cached_outfeats = {}
    res_hidden_states_stopgrad  = getattr(self, "res_hidden_states_stopgrad", False)
    capture_outfeats            = getattr(self, "capture_outfeats",           False)
    layer_idx = 0

    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        if res_hidden_states_stopgrad:
            res_hidden_states = res_hidden_states.detach()

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            # resnet: ResnetBlock2D instance.
            #LINK diffusers.models.resnet.ResnetBlock2D
            # up_blocks.3.resnets.2.conv_shortcut is a module within ResnetBlock2D,
            # it's not transforming the UNet shortcut features.
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        if capture_outfeats:
            self.cached_outfeats[layer_idx] = hidden_states
            layer_idx += 1

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states


# Adapted from ConsistentIDPipeline:set_ip_adapter().
# attn_lora_layer_names: candidates are subsets of ['q', 'k', 'v', 'out'].
def set_up_attn_processors(unet, use_attn_lora, attn_lora_layer_names=['q', 'k', 'v', 'out'], 
                           lora_rank=192, lora_scale_down=8, subj_attn_var_shrink_factor=2.,
                           q_lora_updates_query=False):
    attn_procs = {}
    attn_capture_procs = {}
    unet_modules = dict(unet.named_modules())
    attn_opt_modules = {}

    attn_proc_idx = 0

    for name, attn_proc in unet.attn_processors.items():
        # Only capture the activations of the last 3 CA layers.
        if not name.startswith("up_blocks.3"):
            # Not the last 3 CA layers. Don't enable LoRA or capture activations. 
            # Then the layer falls back to the original attention mechanism.
            # We still use AttnProcessor_LoRA_Capture, as it can handle img_mask.
            attn_procs[name] = AttnProcessor_LoRA_Capture(
                capture_ca_activations=False, enable_lora=False, attn_proc_idx=-1)
            continue
        # cross_attention_dim: 768.
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None:
            # Self attention. Don't enable LoRA or capture activations.
            # We replace the default attn_proc with AttnProcessor_LoRA_Capture, 
            # so that it can incorporate img_mask into self-attention.
            attn_procs[name] = AttnProcessor_LoRA_Capture(
                capture_ca_activations=False, enable_lora=False, attn_proc_idx=-1)            
            continue

        # block_id = 3
        # hidden_size: 320
        # hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        # 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor' ->
        # 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q'
        lora_layer_dict = {}
        lora_layer_dict['q']    = unet_modules[name[:-9] + "to_q"]
        lora_layer_dict['k']    = unet_modules[name[:-9] + "to_k"]
        lora_layer_dict['v']    = unet_modules[name[:-9] + "to_v"]
        # to_out is a ModuleList(Linear, Dropout).
        lora_layer_dict['out']  = unet_modules[name[:-9] + "to_out"][0]

        lora_proj_layers = {}
        # Only apply LoRA to the specified layers.
        for lora_layer_name in attn_lora_layer_names:
            lora_proj_layers[lora_layer_name] = lora_layer_dict[lora_layer_name]

        attn_capture_proc = AttnProcessor_LoRA_Capture(
            capture_ca_activations=True, enable_lora=use_attn_lora, 
            lora_uses_dora=True, lora_proj_layers=lora_proj_layers,
            # LoRA up is initialized to 0. So no need to worry that the LoRA output may be too large.
            lora_rank=lora_rank, lora_alpha=lora_rank // lora_scale_down,
            subj_attn_var_shrink_factor=subj_attn_var_shrink_factor,
            q_lora_updates_query=q_lora_updates_query, attn_proc_idx=attn_proc_idx)
        
        attn_proc_idx += 1
        # attn_procs has to use the original names.
        attn_procs[name] = attn_capture_proc
        # ModuleDict doesn't allow "." in the key.
        name = name.replace(".", "_")
        attn_capture_procs[name] = attn_capture_proc
    
        if use_attn_lora:
            for subname, module in attn_capture_proc.named_modules():
                if isinstance(module, peft_lora.LoraLayer):
                    # ModuleDict doesn't allow "." in the key.
                    lora_path = name + "_" + subname.replace(".", "_")
                    attn_opt_modules[lora_path + "_lora_A"] = module.lora_A
                    attn_opt_modules[lora_path + "_lora_B"] = module.lora_B
                    # lora_uses_dora is always True, so we don't check it here.
                    attn_opt_modules[lora_path + "_lora_magnitude_vector"] = module.lora_magnitude_vector
                    # We will manage attn adapters directly. By default, LoraLayer is an instance of BaseTunerLayer,
                    # so according to the code logic in diffusers/loaders/peft.py,
                    # they will be managed by the diffusers PeftAdapterMixin instance, through the
                    # enable_adapters(), and set_adapter() methods.
                    # Therefore, we disable these calls on module. 
                    # disable_adapters() is a property and changing it will cause exceptions.
                    module.enable_adapters  = dummy_func
                    module.set_adapter      = dummy_func

    unet.set_attn_processor(attn_procs)

    print(f"Set up {len(attn_capture_procs)} CrossAttn processors on {attn_capture_procs.keys()}.")
    print(f"Set up {len(attn_opt_modules)} attn LoRA params: {attn_opt_modules.keys()}.")
    return attn_capture_procs, attn_opt_modules

# NOTE: cross-attn layers are included in the returned lora_modules.
def set_up_ffn_loras(unet, target_modules_pat, lora_uses_dora=False, lora_rank=192, lora_alpha=16):
    # target_modules_pat = 'up_blocks.3.resnets.[12].conv[a-z0-9_]+'
    # up_blocks.3.resnets.[1~2].conv1, conv2, conv_shortcut
    # Cannot set to conv.+ as it will match added adapter module names, including
    # up_blocks.3.resnets.1.conv1.base_layer, up_blocks.3.resnets.1.conv1.lora_dropout
    if target_modules_pat is not None:
        peft_config = LoraConfig(use_dora=lora_uses_dora, inference_mode=False, r=lora_rank, 
                                 lora_alpha=lora_alpha, lora_dropout=0.1,
                                 target_modules=target_modules_pat)

        # UNet is a diffusers PeftAdapterMixin instance. Using get_peft_model on it will
        # cause weird errors. Instead, we directly use diffusers peft adapter methods.
        unet.add_adapter(peft_config, "recon_loss")
        unet.add_adapter(peft_config, "unet_distill")
        unet.add_adapter(peft_config, "comp_distill")
        unet.enable_adapters()

    # lora_layers contain both the LoRA A and B matrices, as well as the original layers.
    # lora_layers are used to set the flag, not used for optimization.
    # lora_modules contain only the LoRA A and B matrices, so they are used for optimization.
    # NOTE: lora_modules contain both ffn and cross-attn lora modules.
    ffn_lora_layers = {}
    ffn_opt_modules = {}
    for name, module in unet.named_modules():
        if isinstance(module, peft_lora.LoraLayer):
            # We don't want to include cross-attn layers in ffn_lora_layers.
            if target_modules_pat is not None and re.search(target_modules_pat, name):
                ffn_lora_layers[name] = module
                # ModuleDict doesn't allow "." in the key.
                name = name.replace(".", "_")
                # Since ModuleDict doesn't allow "." in the key, we manually collect
                # the LoRA matrices in each module.
                # NOTE: We cannot put every sub-module of module into lora_modules,
                # as base_layer is also a sub-module of module, which we shouldn't optimize.
                # Each value in ffn_opt_modules is a ModuleDict:
                '''
                    (Pdb) ffn_opt_modules['up_blocks_3_resnets_1_conv1_lora_A']
                    ModuleDict(
                    (unet_distill): Conv2d(640, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (recon_loss): Conv2d(640, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    )                
                '''
                ffn_opt_modules[name + "_lora_A"] = module.lora_A
                ffn_opt_modules[name + "_lora_B"] = module.lora_B
                if lora_uses_dora:
                    ffn_opt_modules[name + "_lora_magnitude_vector"] = module.lora_magnitude_vector

    print(f"Set up {len(ffn_lora_layers)} FFN LoRA layers: {ffn_lora_layers.keys()}.")
    print(f"Set up {len(ffn_opt_modules)} FFN LoRA params: {ffn_opt_modules.keys()}.")

    return ffn_lora_layers, ffn_opt_modules

def set_lora_and_capture_flags(unet, unet_lora_modules, attn_capture_procs, 
                               outfeat_capture_blocks, res_hidden_states_stopgrad_blocks,
                               use_attn_lora, use_ffn_lora, ffn_lora_adapter_name, capture_ca_activations, 
                               shrink_subj_attn, res_hidden_states_stopgrad):
    # For attn capture procs, capture_ca_activations and use_attn_lora are set in reset_attn_cache_and_flags().
    for attn_capture_proc in attn_capture_procs:
        attn_capture_proc.reset_attn_cache_and_flags(capture_ca_activations, shrink_subj_attn, enable_lora=use_attn_lora)
    # outfeat_capture_blocks only contains the last up block, up_blocks[3].
    # It contains 3 FFN layers. We want to capture their output features.
    for block in outfeat_capture_blocks:
        block.capture_outfeats           = capture_ca_activations

    for block in res_hidden_states_stopgrad_blocks:
        block.res_hidden_states_stopgrad = res_hidden_states_stopgrad

    if not use_ffn_lora:
        unet.disable_adapters()
    else:
        # ffn_lora_adapter_name: 'recon_loss', 'unet_distill', 'comp_distill'.
        if ffn_lora_adapter_name is not None:
            unet.set_adapter(ffn_lora_adapter_name)
            # NOTE: Don't forget to enable_adapters(). 
            # The adapters are not enabled by default after set_adapter().
            unet.enable_adapters()
        else:
            breakpoint()

    # During training, disable_adapters() and set_adapter() will set all/inactive adapters with requires_grad=False, 
    # which might cause issues during DDP training.
    # So we restore them to requires_grad=True.
    # During test, unet_lora_modules will be passed as None, so this block will be skipped.
    if unet_lora_modules is not None:
        for param in unet_lora_modules.parameters():
            param.requires_grad = True

def get_captured_activations(capture_ca_activations, attn_capture_procs, outfeat_capture_blocks, 
                             captured_layer_indices=[22, 23, 24], out_dtype=torch.float32):
    captured_activations = { k: {} for k in ('outfeat', 'attn', 'attnscore', 'q', 'q2', 'k', 'attn_out') }

    if not capture_ca_activations:
        return captured_activations
    
    all_cached_outfeats = []
    for block in outfeat_capture_blocks:
        all_cached_outfeats.append(block.cached_outfeats)
        # Clear the capture flag and cached outfeats.
        block.cached_outfeats = {}
        block.capture_outfeats = False

    for layer_idx in captured_layer_indices:
        # Subtract 22 to ca_layer_idx to match the layer index in up_blocks[3].cached_outfeats.
        # 23, 24 -> 1, 2 (!! not 0, 1 !!)
        internal_idx = layer_idx - 22
        for k in captured_activations.keys():
            if k == 'outfeat':
                # Currently we only capture one block, up_blocks.3. So we hard-code the index 0.
                captured_activations['outfeat'][layer_idx] = all_cached_outfeats[0][internal_idx].to(out_dtype)
            else:
                # internal_idx is the index of layers in up_blocks.3. 
                # Layers 22, 23 and 24 map to 0, 1 and 2.
                cached_activations = attn_capture_procs[internal_idx].cached_activations
                captured_activations[k][layer_idx] = cached_activations[k].to(out_dtype)

    return captured_activations
