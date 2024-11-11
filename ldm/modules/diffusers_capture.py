import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging, is_torch_version, deprecate
from diffusers.utils.torch_utils import apply_freeu
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D

from einops import rearrange
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Revised from pytorch/tests/test_transformers.py: sdp_ref().
# This may be slower than F.scaled_dot_product_attention, 
# but we need attn_score and attn.
def sdp_slow(q, k, v, attn_mask=None, dropout_p=0.0):
    E = q.size(-1)
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    attn_score = attn
    attn = torch.nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn_score, attn

# All layers share the same attention processor instance.
class AttnProcessor_Capture:
    r"""
    Revised from AttnProcessor2_0
    """

    def __init__(self, capture_ca_activations: bool = False):
        self.capture_ca_activations = capture_ca_activations
        self.clear_attn_cache()

    def clear_attn_cache(self):
        self.cached_activations = {}
        for k in ['q', 'attn', 'attnscore', 'attn_out']:
            self.cached_activations[k] = []

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
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
        scale = 1 / math.sqrt(query.size(-1))

        is_cross_attn = (encoder_hidden_states is not None)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        '''
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        '''

        query, key, value = map(lambda t: rearrange(t, 'b h n d -> (b h) n d').contiguous(), (query, key, value))
        hidden_states, attn_score, attn_prob  = sdp_slow(query, key, value, attn_mask=attention_mask, dropout_p=0.0)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
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
            self.cached_activations['q'].append(
                rearrange(query,   '(b h) n d -> b (h d) n', h=attn.heads).contiguous() * math.sqrt(scale) )
            self.cached_activations['attn'].append(
                rearrange(attn_prob, '(b h) i j -> b h i j', h=attn.heads).contiguous() )
            self.cached_activations['attnscore'].append(
                rearrange(attn_score, '(b h) i j -> b h i j', h=attn.heads).contiguous() )
            # attn_out: [b, n, h * d] -> [b, h * d, n]
            self.cached_activations['attn_out'].append(
                hidden_states.permute(0, 2, 1).contiguous() )

        return hidden_states

def CrossAttnUpBlock2D_capture_forward(
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

    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )

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

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

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
