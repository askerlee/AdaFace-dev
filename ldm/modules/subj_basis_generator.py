# Borrowed from ip-adapter resampler.py.
# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import CLIPVisionModel
import numpy as np
from torch import einsum
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

def reshape_tensor(x, num_heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, num_heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, num_heads, length, -1)
    return x


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


# FFN
def FeedForward(dim, mult=4, elementwise_affine=True):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
        nn.LayerNorm(dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),
    )

# group_dim: the tensor dimension that corresponds to the multiple groups.
class LearnedSoftAggregate(nn.Module):
    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim  = group_dim
        # num_feat = 1: element-wise score function & softmax.
        # num_feat > 1: the linear score function is applied to the last dim (features) of the input tensor. 
        self.num_feat   = num_feat
        self.feat2score = nn.Linear(num_feat, 1)
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
    
def LoRA_Emb2Queries(input_dim, lora_rank, output_dim, num_modes, 
                     num_output_queries, elementwise_affine=True):
    return nn.Sequential(
        # Project to [BS, lora_rank * output_dim * num_modes].
        # It takes a huge param size. 512 * 32 * 768 * 4 = 6,291,456.
        nn.Linear(input_dim, lora_rank * output_dim * num_modes, bias=False),
        # Reshape to [BS, lora_rank, output_dim].
        Rearrange('b (m q d) -> b m q d', q=lora_rank, m=num_modes, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        # Aggregate [BS, num_modes, loar_rank, output_dim] -> [BS, lora_rank, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False),
        nn.Dropout(0.1),
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, num_output_queries].
        nn.Linear(lora_rank, num_output_queries, bias=False),
        # Permute to [BS, num_output_queries, output_dim].
        Rearrange('b d q -> b q d'),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),
    )

def Emb2Queries(input_dim, output_dim, num_output_queries, elementwise_affine=True):
    return nn.Sequential(
        # Project to [BS, num_output_queries * output_dim * num_modes].
        # It takes a huge param size. 512 * 32 * 768 * 4 = 6,291,456.
        nn.Linear(input_dim, num_output_queries * output_dim, bias=False),
        # Reshape to [BS, num_output_queries, output_dim].
        Rearrange('b (q d) -> b q d', q=num_output_queries, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),
    )

# Low-rank to high-rank transformation.
def Lora2Hira(lora_rank, output_dim, num_modes, num_output_queries, elementwise_affine=True):
    return nn.Sequential(        
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, num_output_queries].
        nn.Linear(lora_rank, num_output_queries * num_modes, bias=False),
        # Reshape and permute to [BS, num_modes, num_output_queries, output_dim].
        Rearrange('b d (m q) -> b m q d', m=num_modes, q=num_output_queries),
        # Aggregate [BS, num_modes, num_output_queries, output_dim] -> [BS, num_output_queries, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False),        
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),    
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
    def __init__(self, input_dim, context_dim, num_heads, dropout=0.1, attn_polarity=5, elementwise_affine=True, 
                 identity_to_q=False, identity_to_v=False, identity_to_out=False):
        super().__init__()
        dim_head  = input_dim // num_heads
        inner_dim = dim_head   * num_heads

        self.num_heads = num_heads
        self.attn_polarity = attn_polarity

        # To increase stability,, we add layernorm to q,k,v projections.
        self.to_q = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False) if not identity_to_q else nn.Identity(),
                        nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine))
        
        self.to_k = nn.Sequential(
                        # If context_dim != input_dim, then we need to project context_dim to input_dim 
                        # for similarity computation.
                        nn.Linear(context_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine))
        
        self.to_v = nn.Sequential(
                        nn.Linear(context_dim, context_dim, bias=False) if not identity_to_v else nn.Identity(),
                        nn.LayerNorm(context_dim, elementwise_affine=elementwise_affine))

        self.to_out = nn.Sequential(
            nn.Linear(context_dim, context_dim, bias=False) if not identity_to_out else nn.Identity(),
            nn.LayerNorm(context_dim, elementwise_affine=elementwise_affine),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        h = self.num_heads

        q = self.to_q(x)
        if context is None:
            context = x

        k = self.to_k(context)            
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # * self.attn_polarity to increase the polarity of the attention scores. 
        # Otherwise the variance is too small and the attention is too uniform.
        '''
        for N in (768, 7680, 76800, 768000):
            ln = nn.LayerNorm(N)
            a = torch.randn(100, N)
            b = torch.randn(100, N)
            x=(ln(a) * ln(b)).sum(dim=1) * 5 / (N**0.5)
            print(x.std(dim=0))
            
        tensor(4.6600, grad_fn=<StdBackward0>)
        tensor(5.3220, grad_fn=<StdBackward0>)
        tensor(4.8963, grad_fn=<StdBackward0>)
        tensor(5.0100, grad_fn=<StdBackward0>)        
        '''        
        scale = q.size(-1) ** -0.25

        sim = einsum('b i d, b j d -> b i j', q * scale, k * scale) * self.attn_polarity
        # sim: [16, 378, 257]. 16: bs 1 * h 16.
        # attention, what we cannot get enough of
        # NOTE: the normalization is done across tokens, not across pixels.
        # So for each pixel, the sum of attention scores across tokens is 1.
        attn = sim.softmax(dim=-1)
        #breakpoint()

        # v: [16, 257, 48]. 48: dim of each head. out: [16, 378, 48].
        out = einsum('b i j, b j d -> b i d', attn, v)
        # [16, 378, 48] -> [1, 378, 768].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        out = self.to_out(out)

        return out

class SubjBasisGenerator(nn.Module):
    def __init__(
        self,
        depth=1,                            # number of (CrossAttention, FeedForward) layers.     
        # number of cross-attention heads.
        num_heads=1,
        # num_out_queries: number of output queries.
        # 2 * Static/Ada layerwise_lora_rank. * 2 to generate both static and ada bases.
        # Two different SubjBasisGenerator instances are used to generate subj and bg embedder bases.
        num_out_queries=234,                # fg: 234 = 9 * 26. bg: 104 = 4 * 26.
        image_embedding_dim=1280,           # CLIP image feature dimension, as per config.json above.
        face_embedding_dim=512,             # insightface face feature dimension for humans.
        dino_embedding_dim=384,             # DINO object feature dimension for objects.
        num_latent_queries=64,              # number of low-rank latent queries.
        latent_query_dim=64,                # Latent query dimension.
        num_lora2hira_modes=4,              # number of modes for Lora2Hira.  
        output_dim=768,                     # CLIP text embedding input dimension.
        max_seq_len: int = 257,             # [CLS token, image tokens]
        apply_pos_emb: bool = True,         # Newer IP Adapter uses positional embeddings.
        elementwise_affine: bool = True,    # Whether to use elementwise affine in LayerNorms.
        codebook_size: int = 2340,          # Size of the codebook, 50 * num_out_queries.
        use_FFN: bool = True,               # Whether to use FeedForward layer after cross-attention.
        placeholder_is_bg: bool = False,    # Whether the placeholder is for the image background.
    ):
        super().__init__()
        assert depth > 0, "depth must be > 0."
        self.depth = depth

        self.proj_in = nn.Sequential(
            nn.Linear(image_embedding_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        )

        self.placeholder_is_bg = placeholder_is_bg
        self.num_latent_queries  = num_latent_queries
        if not self.placeholder_is_bg:
            self.face_proj_in = Emb2Queries(face_embedding_dim, latent_query_dim, num_output_queries=num_latent_queries,
                                            elementwise_affine=elementwise_affine)
            self.obj_proj_in  = Emb2Queries(dino_embedding_dim, latent_query_dim, num_output_queries=num_latent_queries,
                                            elementwise_affine=elementwise_affine)
            # If it's a subject placeholder, then latent_queries 
            # are generated from face or object embeddings. No static_latent_queries.
            self.static_latent_queries = None            
        else:
            # For background placeholders, face and object embeddings are not used as they are foreground.
            self.face_proj_in = None
            self.obj_proj_in  = None
            # These static set of latent queries are used to attend to the ad-hoc CLIP image features.
            self.static_latent_queries = nn.Parameter(torch.randn(1, num_latent_queries, latent_query_dim) / latent_query_dim**0.5)

        # If placeholder_is_bg, then don't use codebook, instead draw features 
        # from the ad-hoc CLIP image features.
        self.use_codebook = not placeholder_is_bg
        # If use_codebook, then no point to use positional embeddings.
        if self.use_codebook:
            apply_pos_emb = False
            # Evenly distribute the codebooks across the depth.
            self.codebook_size = codebook_size // self.depth
        else:
            self.codebook_size = -1

        if apply_pos_emb:
            self.pos_emb    = nn.Embedding(max_seq_len, output_dim)                             
            self.pos_emb_ln = nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine)  
        else:
            self.pos_emb    = None
            self.pos_emb_ln = None

        self.lq_ln    = nn.LayerNorm(latent_query_dim, elementwise_affine=elementwise_affine)

        self.num_out_queries        = num_out_queries
        self.num_lora2hira_modes    = num_lora2hira_modes
        self.elementwise_affine     = elementwise_affine
        self.output_scale           = output_dim ** -0.5
        self.lora2hira = Lora2Hira(lora_rank=num_latent_queries, output_dim=output_dim, num_modes=num_lora2hira_modes,
                                   num_output_queries=num_out_queries, elementwise_affine=elementwise_affine)
        
        self.layers    = nn.ModuleList([])
        self.codebooks = nn.ParameterList([]) if self.use_codebook else None
        self.use_FFN   = use_FFN

        for dep in range(depth):
            if dep == 0:
                input_dim = latent_query_dim
            else:
                input_dim = output_dim

            self.layers.append(
                nn.ModuleList(
                    [
                        # dim=768, num_heads=16.
                        # Should we disable elementwise_affine in CrossAttention layernorms? I'm not sure.
                        # Currently it's the only place where elementwise_affine is used.
                        CrossAttention(input_dim=input_dim, context_dim=output_dim, num_heads=num_heads, dropout=0.1,
                                       identity_to_q=True, identity_to_v=True, identity_to_out=True),
                        # FeedForward: 2-layer MLP with GELU activation.
                        # LayerNorm -> Linear -> GELU -> Linear.
                        FeedForward(dim=output_dim, mult=1, elementwise_affine=elementwise_affine) \
                            if self.use_FFN else nn.Identity(),
                    ]
                )
            )
            if self.use_codebook:
                layer_codebook = nn.Parameter(torch.randn(1, self.codebook_size, output_dim) / output_dim**0.5)
                self.codebooks.append(layer_codebook)

        print(repr(self))

    def forward(self, clip_features, id_embs, is_face):     
        if not self.use_codebook:
            x = self.proj_in(clip_features)
            if self.pos_emb is not None:
                n, device = x.shape[1], x.device
                pos_emb = self.pos_emb(torch.arange(n, device=device))
                # Downscale positional embeddings to reduce its impact.
                x = x + self.pos_emb_ln(pos_emb) * 0.5

        # No need to use id_embs if placeholder_is_bg.
        if (not self.placeholder_is_bg) and (id_embs is not None):
            # id_embs: [1, 512].
            # latent_queries: [1, 64, 64].
            if is_face:
                latent_queries = self.face_proj_in(id_embs)
            else:
                latent_queries = self.obj_proj_in(id_embs)
        else:
            latent_queries = self.static_latent_queries
    
        latent_queries = self.lq_ln(latent_queries.repeat(clip_features.size(0), 1, 1))

        for i, (attn, ff) in enumerate(self.layers):
            if self.use_codebook:
                context = self.codebooks[i]
            else:
                # Otherwise, context is the ad-hoc CLIP image features.
                context = x
            
            # No residual connection at the first layer.
            if i == 0:
                latent_queries = attn(latent_queries, context)
            else:
                latent_queries = attn(latent_queries, context) + latent_queries

            if self.use_FFN:
                latent_queries = ff(latent_queries) + latent_queries
        
        latent_queries = self.lora2hira(latent_queries)
        return latent_queries * self.output_scale

    def __repr__(self):
        type_sig = 'subj' if not self.placeholder_is_bg else 'bg'
        return f"{type_sig} SubjBasisGenerator: depth={self.depth}, use_FFN={self.use_FFN}, num_out_queries={self.num_out_queries}, num_latent_queries={self.num_latent_queries}," \
               f"num_lora2hira_modes={self.num_lora2hira_modes}, elementwise_affine={self.elementwise_affine}, codebook_size={self.codebook_size}"
    
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

# Revised from CLIPVisionTransformer. self: a CLIPVisionTransformer instance.
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
            attn_mask = torch.cat([torch.ones(*attn_mask.shape[:2], 1).to(attn_mask.device), attn_mask], dim=-1)
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
    
