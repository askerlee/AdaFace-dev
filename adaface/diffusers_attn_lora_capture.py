import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import logging, is_torch_version, deprecate
from peft import LoraConfig, get_peft_model
import peft.tuners.lora as peft_lora
from peft.tuners.lora.dora import DoraLinearLayer
from ldm.util import sel_emb_attns_by_indices
from einops import rearrange
import math, re
import numpy as np

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def log_gaussian_2d(x, y, x0, y0, std_x, std_y):
    """
    Evaluate the uncorrelated 2D Gaussian PDF at coordinates (x, y).
    x, y can be scalars, or NumPy arrays of the same shape.
    """
    return -(((x - x0)**2)/(2*std_x**2) + ((y - y0)**2)/(2*std_y**2))

def calc_subj_attn_bias(attn_score, subj_indices, subj_attn_var_shrink_factor):
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
    # subj_attn: [1, 8, 4096]. Already summed over subject embeddings.
    subj_attn = sel_emb_attns_by_indices(attn_score2, subj_indices, do_sum=True)
    SN = subj_attn.size(0)
    # Average over the heads. subj_attn: [1, 4096]
    subj_attn = subj_attn.mean(dim=1)
    # 1) Normalize subj_attn as a 2D probability distribution.
    subj_prob = subj_attn.softmax(dim=1)
    H = W = int(np.sqrt(attn_score2.size(-1)))
    # subj_attn: [1, 4096] -> [1, 64, 64]
    subj_prob_3d = subj_prob.view(SN, H, W)
    # 2) Compute y_center and x_center
    # ys: [1, 64, 1]. xs: [1, 1, 64]
    ys = torch.arange(H, device=subj_attn.device).view(1, H, 1)
    xs = torch.arange(W, device=subj_attn.device).view(1, 1, W)
    y_center = (subj_prob_3d * ys).sum(dim=(1,2))  # [N]
    x_center = (subj_prob_3d * xs).sum(dim=(1,2))  # [N]
    # 3) Variances
    y_sq = (subj_prob_3d * ys**2).sum(dim=(1,2))   # E[Y^2]
    x_sq = (subj_prob_3d * xs**2).sum(dim=(1,2))   # E[X^2]

    var_y = y_sq - y_center**2
    var_x = x_sq - x_center**2

    # 4) 2D std
    # y_center, x_center: [35.1875, 32.9062], std_2d: [26.8281].  
    #var_2d = var_y + var_x
    #std_2d = var_2d.sqrt() 
    # std_x is often slightly smaller than std_y, meaning the face is slightly taller than wider.
    # std_x: [16.0312], std_y: [17.7500].
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
    # Make a meshgrid. X: [1, 64, 64], Y: [1, 64, 64]
    X, Y = torch.meshgrid(torch.arange(W, device=subj_attn.device), 
                          torch.arange(H, device=subj_attn.device))
    X = X.to(attn_score.dtype).unsqueeze(0)
    Y = Y.to(attn_score.dtype).unsqueeze(0)
    shrinker_grid = log_gaussian_2d(X, Y, x_center, y_center, shrinker_std_x, shrinker_std_y)
    # Normalize shrinker_grid, so that the maximum value (at the point nearest to the center) 
    # is always 0, i.e., the subject activation at this point is not scaled down.
    # NOTE: shrinker_grid values are in the log scale. So most of them are negative.
    shrinker_grid = shrinker_grid - shrinker_grid.max().detach()
    # shrinker_grid: [1, X=64, Y=64] -> [1, Y=64, X=64] -> [1, 1, 4096]
    shrinker_grid = shrinker_grid.permute(0, 2, 1).reshape(SN, 1, -1)
    # subj_attn_bias: [1, 77, 1, 4096]
    subj_attn_bias = torch.zeros_like(attn_score2[:, :, :1])
    # subj_attn_bias[subj_indices]: [20, 1, 4096]. shrinker_grid will be broadcasted to them.
    subj_attn_bias[subj_indices] = shrinker_grid
    # subj_attn_bias: [1, 1, 4096, 77]
    subj_attn_bias = subj_attn_bias.permute(0, 2, 3, 1)
    # print(f"std_2d: {std_2d.item():.4f}, std_x: {std_x.mean().item():.4f}, std_y: {std_y.mean().item():.4f}")
    return subj_attn_bias

# Slow implementation equivalent to F.scaled_dot_product_attention.
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 subj_indices=None, subj_attn_var_shrink_factor=2., is_causal=False, 
                                 scale=None, enable_gqa=False) -> torch.Tensor:
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

    if subj_indices is not None:
        subj_attn_bias = calc_subj_attn_bias(attn_weight, subj_indices, subj_attn_var_shrink_factor)
    else:
        subj_attn_bias = 0

    # attn_bias: [1, 1, 4096, 77], the same size as a single-head attn_weight.
    # subj_attn_bias: [1, 1, 4096, 77]. At the last dim, 0 everywhere except 
    # for the subject embeddings indexed by subj_indices.
    attn_weight += attn_bias + subj_attn_bias
    attn_score = attn_weight
    attn_weight = torch.softmax(attn_weight, dim=-1)
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
                 hidden_size: int = -1, cross_attention_dim: int = 768, 
                 lora_rank: int = 192, lora_alpha: float = 16,
                 subj_attn_var_shrink_factor: float = 2.):
        super().__init__()

        self.global_enable_lora = enable_lora
        # reset_attn_cache_and_flags() sets the local (call-specific) self.enable_lora flag.
        # By default, suppress_subj_attn is False. Later in layers 22, 23, 24 it will be set to True.
        self.reset_attn_cache_and_flags(capture_ca_activations, False, enable_lora)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = self.lora_alpha / self.lora_rank
        self.subj_attn_var_shrink_factor = nn.Parameter(torch.tensor(subj_attn_var_shrink_factor, dtype=float), 
                                                        requires_grad=True)

        '''
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning        
        '''
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
    def reset_attn_cache_and_flags(self, capture_ca_activations, suppress_subj_attn, enable_lora):
        self.capture_ca_activations = capture_ca_activations
        self.suppress_subj_attn     = suppress_subj_attn
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
        
        if (subj_indices is not None) and not self.suppress_subj_attn:
            subj_indices = None
        
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
        # k, v loras are applied to key and value (currently k, v loras are never enabled), 
        # while q lora is applied to query2, and we keep query unchanged. 
        if self.enable_lora and self.to_q_lora is not None:
            query2 = self.to_q_lora(hidden_states)
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
        if is_cross_attn and (self.capture_ca_activations or self.suppress_subj_attn):
            hidden_states, attn_score, attn_prob = \
                scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, 
                                             dropout_p=0.0, subj_indices=subj_indices,
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
    if hasattr(self, "capture_outfeats"):
        capture_outfeats = self.capture_outfeats
    else:
        capture_outfeats = False

    layer_idx = 0

    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

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


def UNetMidBlock2D_forward_capture(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
    hidden_states = self.resnets[0](hidden_states, temb)

    attn_i = 0
    self.hidden_states = []
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        if attn is not None:
            # self.hidden_states stores the pre-attention hidden states,
            # which will be used in cross-attentions.
            self.hidden_states.append(hidden_states)
            # encoder_hidden_states = all_encoder_hidden_states[attn_i]
            #cross_hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, 
            #                           temb=temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=None, temb=temb)

        hidden_states = resnet(hidden_states, temb)
        attn_i += 1

    return hidden_states


# Adapted from ConsistentIDPipeline:set_ip_adapter().
# attn_lora_layer_names: candidates are subsets of ['q', 'k', 'v', 'out'].
def set_up_attn_processors(unet, use_attn_lora, attn_lora_layer_names=['q'], 
                           lora_rank=192, lora_scale_down=8, subj_attn_var_shrink_factor=2.):
    attn_procs = {}
    attn_capture_procs = {}
    unet_modules = dict(unet.named_modules())
    attn_opt_modules = {}

    for name, attn_proc in unet.attn_processors.items():
        # Only capture the activations of the last 3 CA layers.
        if not name.startswith("up_blocks.3"):
            # Not the last 3 CA layers. Don't enable LoRA or capture activations. 
            # Then the layer falls back to the original attention mechanism.
            # We still use AttnProcessor_LoRA_Capture, as it can handle img_mask.
            attn_procs[name] = AttnProcessor_LoRA_Capture(
                capture_ca_activations=False, enable_lora=False)
            continue
        # cross_attention_dim: 768.
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # Self attention. Don't enable LoRA or capture activations.
        if cross_attention_dim is None or (name.startswith("up_blocks.3.attentions.0")):
            # We replace the default attn_proc with AttnProcessor_LoRA_Capture, 
            # as it can handle img_mask.
            attn_procs[name] = AttnProcessor_LoRA_Capture(
                capture_ca_activations=False, enable_lora=False)
            continue

        block_id = 3
        # hidden_size: 320
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
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
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, 
            # LoRA up is initialized to 0. So no need to worry that the LoRA output may be too large.
            lora_rank=lora_rank, lora_alpha=lora_rank // lora_scale_down,
            subj_attn_var_shrink_factor=subj_attn_var_shrink_factor)
        
        # attn_procs has to use the original names.
        attn_procs[name] = attn_capture_proc
        # ModuleDict doesn't allow "." in the key.
        name = name.replace(".", "_")
        attn_capture_procs[name] = attn_capture_proc
        attn_opt_modules[name + "_subj_attn_var_shrink_factor"] = attn_capture_proc.subj_attn_var_shrink_factor
    
        if use_attn_lora:
            for subname, module in attn_capture_proc.named_modules():
                if isinstance(module, peft_lora.LoraLayer):
                    # ModuleDict doesn't allow "." in the key.
                    lora_path = name + "_" + subname.replace(".", "_")
                    attn_opt_modules[lora_path + "_lora_A"] = module.lora_A
                    attn_opt_modules[lora_path + "_lora_B"] = module.lora_B
                    # lora_uses_dora is always True, so we don't check it here.
                    attn_opt_modules[lora_path + "_lora_magnitude_vector"] = module.lora_magnitude_vector

    unet.set_attn_processor(attn_procs)
    print(f"Set up {len(attn_capture_procs)} CrossAttn processors on {attn_capture_procs.keys()}.")
    print(f"Set up {len(attn_opt_modules)} attn LoRA params: {attn_opt_modules.keys()}.")
    return attn_capture_procs, attn_opt_modules

# NOTE: cross-attn layers are included in the returned lora_modules.
def set_up_ffn_loras(unet, target_modules_pat, lora_uses_dora=False, lora_rank=192, lora_alpha=16):
    # up_blocks.3.resnets.[1~2].conv1, conv2, conv_shortcut
    if target_modules_pat is not None:
        peft_config = LoraConfig(use_dora=lora_uses_dora, inference_mode=False, r=lora_rank, 
                                 lora_alpha=lora_alpha, lora_dropout=0.1,
                                 target_modules=target_modules_pat)
        unet = get_peft_model(unet, peft_config)

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
                ffn_opt_modules[name + "_lora_A"] = module.lora_A
                ffn_opt_modules[name + "_lora_B"] = module.lora_B
                if lora_uses_dora:
                    ffn_opt_modules[name + "_lora_magnitude_vector"] = module.lora_magnitude_vector

    print(f"Set up {len(ffn_lora_layers)} FFN LoRA layers: {ffn_lora_layers.keys()}.")
    print(f"Set up {len(ffn_opt_modules)} FFN LoRA params: {ffn_opt_modules.keys()}.")
    if target_modules_pat is not None:
        unet.print_trainable_parameters()
    return unet, ffn_lora_layers, ffn_opt_modules

def set_lora_and_capture_flags(attn_capture_procs, outfeat_capture_blocks, ffn_lora_layers, 
                               use_attn_lora, use_ffn_lora, capture_ca_activations, suppress_subj_attn=False):
    # For attn capture procs, capture_ca_activations and use_attn_lora are set in reset_attn_cache_and_flags().
    for attn_capture_proc in attn_capture_procs:
        attn_capture_proc.reset_attn_cache_and_flags(capture_ca_activations, suppress_subj_attn, enable_lora=use_attn_lora)
    # outfeat_capture_blocks only contains the last up block, up_blocks[3].
    # It contains 3 FFN layers. We want to capture their output features.
    for block in outfeat_capture_blocks:
        block.capture_outfeats = capture_ca_activations
    # We no longer manipulate the lora_layer.scaling to disable a LoRA.
    # That method is slow and seems LoRA params are still optimized.
    # Instead we directly set the disable_adapters_ flag in the LoRA layers.
    # If not use_ffn_lora, ffn_lora_layers is an empty ModuleDict.
    for lora_layer in ffn_lora_layers:
        lora_layer.disable_adapters_ = not use_ffn_lora

def get_captured_activations(capture_ca_activations, attn_capture_procs, outfeat_capture_blocks, 
                             captured_layer_indices=[23, 24], out_dtype=torch.float32):
    captured_activations = { k: {} for k in ('outfeat', 'attn', 'attnscore', 'q', 'q2', 'attn_out') }

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
                # internal_idx is the index of layers in up_blocks.3. Layers 23 and 24 map to 1 and 2.
                # But layers in attn_capture_procs correspond to up_blocks.3.attentions[1:].
                # Therefore, we need to subtract 1 from internal_idx to match the index in attn_capture_procs.
                # NOTE: Layer 22 capturing is not supported, as layer 22 has internal_idx 0, and -1 maps
                # to the last layer in attn_capture_procs, which is layer 24.
                cached_activations = attn_capture_procs[internal_idx - 1].cached_activations
                captured_activations[k][layer_idx] = cached_activations[k].to(out_dtype)

    return captured_activations
