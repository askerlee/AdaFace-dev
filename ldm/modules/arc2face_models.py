import torch
import torch.nn as nn
from transformers import CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPAttention
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
# from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
_make_causal_mask = AttentionMaskConverter._make_causal_mask
_expand_mask = AttentionMaskConverter._expand_mask

from ldm.util import add_noise_to_tensor

# Extend CLIPAttention by using multiple k_proj and v_proj in each head.
# To avoid too much increase of computation, we don't extend q_proj.
class CLIPAttentionMKV(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, multiplier=2):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.multiplier  = multiplier

        self.k_proj   = nn.Linear(self.embed_dim, self.embed_dim * self.multiplier)
        self.v_proj   = nn.Linear(self.embed_dim, self.embed_dim * self.multiplier)
        self.q_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def extend_weights(self, clip_attn_layer, noise_std=0.1, 
                       noise_std_is_relative=True, keep_norm=False):
        # q_proj and out_proj are the same as the original CLIPAttention.
        self.q_proj.weight.data   = clip_attn_layer.q_proj.weight.data.clone()
        self.q_proj.bias.data     = clip_attn_layer.q_proj.bias.data.clone()
        self.out_proj.weight.data = clip_attn_layer.out_proj.weight.data.clone()
        self.out_proj.bias.data   = clip_attn_layer.out_proj.bias.data.clone()

        # bias doesn't need noise perturbation, as after the weights are noised, 
        # different copies of the weight/bias will receive different gradients, 
        # making the bias terms diverge and identifiable after training.
        self.v_proj.bias.data     = clip_attn_layer.v_proj.bias.data.repeat(self.multiplier)
        self.k_proj.bias.data     = clip_attn_layer.k_proj.bias.data.repeat(self.multiplier)
        ORIG_V_SHAPE    = clip_attn_layer.v_proj.weight.shape
        ORIG_V_SHAPE_D0 = ORIG_V_SHAPE[0]
        self.v_proj.weight.data   = clip_attn_layer.v_proj.weight.data.repeat(self.multiplier, 1)
        # Adding noise to the extra copies of the weights.
        self.v_proj.weight.data[ORIG_V_SHAPE_D0:] = \
            add_noise_to_tensor(self.v_proj.weight.data[ORIG_V_SHAPE_D0:], 
                                noise_std, noise_std_is_relative, keep_norm)
        NEW_V_SHAPE     = self.v_proj.weight.shape
        NOISED_V_SHAPE  = self.v_proj.weight.data[ORIG_V_SHAPE_D0:].shape
        print(f"{NOISED_V_SHAPE} in {NEW_V_SHAPE} of v_proj is added with noise")

        ORIG_K_SHAPE    = clip_attn_layer.k_proj.weight.shape
        ORIG_K_SHAPE_D0 = ORIG_K_SHAPE[0]
        self.k_proj.weight.data   = clip_attn_layer.k_proj.weight.data.repeat(self.multiplier, 1)
        # Adding noise to the extra copies of the weights.
        self.k_proj.weight.data[ORIG_K_SHAPE_D0:] = \
            add_noise_to_tensor(self.k_proj.weight.data[ORIG_K_SHAPE_D0:], 
                                noise_std, noise_std_is_relative, keep_norm)
        NEW_K_SHAPE     = self.k_proj.weight.shape
        NOISED_K_SHAPE  = self.k_proj.weight.data[ORIG_K_SHAPE_D0:].shape
        print(f"{NOISED_K_SHAPE} in {NEW_K_SHAPE} of k_proj is added with noise")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        # For key_states and value_states, the multiplier is absorbed into the seq_len (dim 1, shape specified as -1).
        key_states   = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states   = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        # src_len0 is the original src_len without the multiplier.
        src_len0 = src_len // self.multiplier
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len0):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len0)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, self.multiplier, src_len0) + causal_attention_mask.unsqueeze(3)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len0):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len0)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, self.multiplier, src_len0) + attention_mask.unsqueeze(3)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
    
class CLIPTextModelWrapper(CLIPTextModel):
    # Adapted from https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/clip/modeling_clip.py#L812
    # Modified to accept precomputed token embeddings "input_token_embs" as input or calculate them from input_ids and return them.
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_token_embs: Optional[torch.Tensor] = None,
        hidden_state_layer_weights: Optional[torch.Tensor] = None,
        return_token_embs: Optional[bool] = False,
    ) -> Union[Tuple, torch.Tensor, BaseModelOutputWithPooling]:

        if return_token_embs:
            return self.text_model.embeddings.token_embedding(input_ids)
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        output_attentions = output_attentions if output_attentions is not None else self.text_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.text_model.config.output_hidden_states
        )
        if hidden_state_layer_weights is not None:
            output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.text_model.config.use_return_dict
    
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
    
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    
        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=input_token_embs)
    
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
    
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            # output_hidden_states is False by default, and only True if hidden_state_layer_weights is provided.
            output_hidden_states=output_hidden_states,      
            return_dict=return_dict,
        )
    
        # If output_hidden_states is True, then encoder_outputs[0] is last_hidden_state [1, 22, 768].
        # encoder_outputs[1] is hidden_states, which is a tuple of 13 hidden states, each being [1, 22, 768].
        # encoder_outputs[0] == encoder_outputs[1][12].
        if hidden_state_layer_weights is None:
            last_hidden_state = encoder_outputs[0]
        else:
            num_hidden_state_layers = len(hidden_state_layer_weights)
            last_hidden_states = encoder_outputs[1][-num_hidden_state_layers:]
            hidden_state_layer_weights = hidden_state_layer_weights.to(last_hidden_states[0].dtype)
            # Normalize the weights of to sum to 1 across layers.
            # hidden_state_layer_weights: [3, 1] or [3, 768].
            hidden_state_layer_weights = hidden_state_layer_weights / hidden_state_layer_weights.sum(dim=0, keepdim=True)
            # [3, 1/768] -> [3, 1, 1, 1/768]
            hidden_state_layer_weights = hidden_state_layer_weights.unsqueeze(1).unsqueeze(1)
            # A weighted sum of last_hidden_states.
            # [3, 1, 22, 768] * [3, 1, 1, 1/768] -> [3, 1, 22, 768] -> [1, 22, 768]
            last_hidden_state = (torch.stack(last_hidden_states, dim=0) * hidden_state_layer_weights).sum(dim=0)
            
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # self.text_model.eos_token_id == 2 is True.
        if self.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
    
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def extend_clip_attention_MKV_multiplier(self, multiplier=2, noise_std=0.1):
        num_extended_layers = 0

        for layer in self.text_model.encoder.layers:
            # This shouldn't happen, unless self_attn has already been extended as CLIPAttentionMKV.
            if not isinstance(layer.self_attn, CLIPAttention):
                breakpoint()
            old_attn_layer = layer.self_attn
            layer.self_attn = CLIPAttentionMKV(old_attn_layer.config, multiplier)
            layer.self_attn.extend_weights(old_attn_layer, noise_std)
            num_extended_layers += 1
    
        return num_extended_layers
    
