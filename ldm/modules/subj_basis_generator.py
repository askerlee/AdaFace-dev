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
from transformers.modeling_outputs import BaseModelOutputWithPooling
import numpy as np
from torch import einsum

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim, elementwise_affine=False),
        nn.Linear(dim, inner_dim, bias=False),
        nn.LayerNorm(dim, elementwise_affine=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, elementwise_affine=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

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
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        attn = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


# All CrossAttention layers have 8 heads.
class CrossAttention(nn.Module):
    def __init__(self, input_dim, heads=8, dropout=0.2):
        super().__init__()
        dim_head = input_dim // heads
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.25
        self.heads = heads

        # To increase stability,, we add layernorm to q,k,v projections.
        self.to_q = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=False))
        
        self.to_k = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=False))
        
        self.to_v = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=False))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.LayerNorm(inner_dim, elementwise_affine=False),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        if context is None:
            context = x

        k = self.to_k(context)            
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # * 5 to increase the polarity of the attention scores. Otherwise the variance is too small
        # and the attention is too uniform.
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
        scale = q.size(-1) ** -0.5
        sim = einsum('b i d, b j d -> b i j', q * scale, k * scale) * 5
        # sim: [16, 378, 257]. 16: bs 1 * h 16.
        # attention, what we cannot get enough of
        # NOTE: the normalization is done across tokens, not across pixels.
        # So for each pixel, the sum of attention scores across tokens is 1.
        attn = sim.softmax(dim=-1)

        # v: [16, 257, 48]. 48: dim of each head. out: [16, 378, 48].
        out = einsum('b i j, b j d -> b i d', attn, v)
        # [16, 378, 48] -> [1, 378, 768].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        out = self.to_out(out)

        return out

# Simplified from IP-Adapter Resampler.
class SubjBasisGenerator(nn.Module):
    def __init__(
        self,
        dim=768,                            # Internal feature dimension. Same as output_dim.
        depth=1,                            # number of (PerceiverAttention, FeedForward) layers.     
        # number of heads as per https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/config.json        
        heads=16,       
        # num_subj_queries: number of subject latent_queries.
        # num_bg_queries:   number of background latent_queries.
        # 2 * Static/Ada layerwise_lora_rank. * 2 to generate both static and ada bases.
        # The same SubjBasisGenerator instance is used to generate both subj and bg embedder bases, 
        # provided different input features in two different passes.
        num_subj_queries=378,               # 378 = 9 * 42.
        num_bg_queries=168,                 # 168 = 4 * 42.
        image_embedding_dim=1280,           # CLIP image feature dimension, as per config.json above.
        output_dim=768,                     # CLIP text embedding input dimension.
        ff_mult=1,                          # FF inner_dim = dim * ff_mult. Set to 1 to reduce the number of parameters.
        max_seq_len: int = 257,             # [CLS token, image tokens]
        apply_pos_emb: bool = True,         # Newer IP Adapter uses positional embeddings.
        # number of latent_queries derived from mean pooled representation of the image.
        ## 6 = 3*2 (3 among 10 static bases and 3 among 10 ada bases).
        num_latents_mean_pooled: int = 0,   # Disabled.
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(image_embedding_dim, dim),
            nn.LayerNorm(dim, elementwise_affine=False),
        )
        self.pos_emb    = nn.Embedding(max_seq_len, dim)                if apply_pos_emb else None
        self.pos_emb_ln = nn.LayerNorm(dim, elementwise_affine=False)   if apply_pos_emb else None

        self.latent_subj_queries = nn.Parameter(torch.randn(1, num_subj_queries, dim) / dim**0.5)
        self.latent_bg_queries   = nn.Parameter(torch.randn(1, num_bg_queries, dim)   / dim**0.5)
        self.lq_ln          = nn.LayerNorm(dim, elementwise_affine=False)

        # Remove proj_out to reduce the number of parameters, since image_embedding_dim = output_dim = 768.
        self.proj_out = nn.Identity() #nn.Linear(dim, output_dim)
        # NOTE: norm_out is the only LayerNorm with elementwise_affine=True.
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim, elementwise_affine=False),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # dim=768, heads=16.
                        CrossAttention(input_dim=dim, heads=heads, dropout=0.2),
                        # FeedForward: 2-layer MLP with GELU activation.
                        # LayerNorm -> Linear -> GELU -> Linear.
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, placeholder_is_bg=False):        
        x = self.proj_in(x)
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            # Downscale positional embeddings to reduce its impact.
            x = x + self.pos_emb_ln(pos_emb) * 0.5

        # placeholder_is_bg determines whether to select latent_bg_queries or latent_subj_queries,
        # which then extract either background or subject bases.
        latent_queries = self.latent_bg_queries if placeholder_is_bg else self.latent_subj_queries
        latent_queries = self.lq_ln(latent_queries.repeat(x.size(0), 1, 1))

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latent_queries = torch.cat((meanpooled_latents, latent_queries), dim=-2)

        for attn, ff in self.layers:
            latent_queries = attn(latent_queries, x) + latent_queries
            latent_queries = ff(latent_queries)      + latent_queries

        latent_queries = self.proj_out(latent_queries)
        return self.norm_out(latent_queries)

# Revised from CLIPVisionTransformer. self: a CLIPVisionTransformer instance.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L821
# pixel_values: preprocessed B*C*H*W images. [1, 3, 224, 224]
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
        # hidden_states: [1, 257, 1280]. 257: 16*16 (patch_embeds) + 1 (class_embeds).
        # 16*16 is output from Conv2d(3, 1280, kernel_size=(14, 14), stride=(14, 14), bias=False).
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        
        if attn_mask is not None:
            feat_edge_size = np.sqrt(hidden_states.shape[1] - 1).astype(int)
            attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(feat_edge_size, feat_edge_size))
            # Flatten the mask: [1, 1, 16, 16] => [1, 1, 256].
            attn_mask = attn_mask.reshape(*attn_mask.shape[:2], -1)
            # Prepend 1 to the mask: [1, 1, 256] => [1, 1, 257]. 
            # This 1 corresponds to class_embeds, which is always attended to.
            attn_mask = torch.cat([torch.ones(*attn_mask.shape[:2], 1).to(attn_mask.device), attn_mask], dim=-1)
            attn_mask_pairs = torch.matmul(attn_mask.transpose(-1, -2), attn_mask).bool().unsqueeze(1)
        else:
            attn_mask_pairs = None

        # encoder: CLIPEncoder.
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # New feature: (***The official documentation is wrong***)
            # attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
            #                 Mask to avoid performing attention on pairs of token. Mask values selected in `[0, 1]`:
            #                 - 1 for pairs that are **not masked**,
            #                 - 0 for pairs that are **masked**.            
            attention_mask=attn_mask_pairs,
            output_attentions=output_attentions,        # False
            output_hidden_states=output_hidden_states,  # True
            return_dict=return_dict,                    # True
        )

        # last_hidden_state: [1, 257, 1280]
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
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
    
