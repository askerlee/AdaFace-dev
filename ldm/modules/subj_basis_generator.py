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
from ldm.util import anneal_value

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
                     num_output_vecs, elementwise_affine=True):
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
        # Project to [BS, output_dim, num_output_vecs].
        nn.Linear(lora_rank, num_output_vecs, bias=False),
        # Permute to [BS, num_output_vecs, output_dim].
        Rearrange('b d q -> b q d'),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),
    )

def ExpandEmbs(input_dim, output_dim, expansion_ratio, elementwise_affine=True):
    return nn.Sequential(
        # Project to [BS, num_output_vecs * output_dim].
        nn.Linear(input_dim, expansion_ratio * output_dim, bias=False),
        # Reshape to [BS, num_output_vecs, output_dim].
        Rearrange('b (e d) -> b e d', e=expansion_ratio, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(0.1),
    )

# Input: [BS, N, D].
def MultimodeProjection(input_dim, output_dim=-1, num_modes=4, elementwise_affine=True):
    if output_dim == -1:
        output_dim = input_dim

    return nn.Sequential(
            nn.Linear(input_dim, output_dim * num_modes, bias=False),
            # Reshape to [BS, num_output_vecs, output_dim].
            Rearrange('b n (m d) -> b n m d', m=num_modes, d=output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
            LearnedSoftAggregate(num_feat=output_dim, group_dim=2, keepdim=False),
            nn.Dropout(0.1),
    )

# Low-rank to high-rank transformation.
def Lora2Hira(lora_rank, hira_rank, output_dim, num_modes, elementwise_affine=True):
    return nn.Sequential(        
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, hira_rank].
        nn.Linear(lora_rank, hira_rank * num_modes, bias=False),
        # Reshape and permute to [BS, num_modes, num_output_vecs, output_dim].
        Rearrange('b d (m q) -> b m q d', m=num_modes, q=hira_rank),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        # Aggregate [BS, num_modes, hira_rank, output_dim] -> [BS, hira_rank, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False),        
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
    def __init__(self, input_dim, num_heads=6, dropout=0.1, 
                 identity_to_q=False, identity_to_k=False, identity_to_v=False, dynamic_to_v=True, 
                 identity_to_out=False, out_has_skip=False):
        super().__init__()
        dim_head  = input_dim // num_heads
        inner_dim = dim_head   * num_heads

        self.num_heads = num_heads
        self.dynamic_to_v = dynamic_to_v
        self.to_q = nn.Linear(input_dim, inner_dim, bias=False) if not identity_to_q else nn.Identity()
        self.to_k = nn.Linear(input_dim, inner_dim, bias=False) if not identity_to_k else nn.Identity()
        # If dynamic_to_v is True, then self.to_v is applied on x to get the actual to_v, 
        # which is then applied on context.
        # Otherwise, self.to_v is applied on context.
        if dynamic_to_v:
            self.to_v = nn.Linear(input_dim, input_dim * input_dim, bias=False)
        else:
            self.to_v = nn.Linear(input_dim, inner_dim, bias=False) if not identity_to_v else nn.Identity()

        assert not (identity_to_out and out_has_skip), "identity_to_out and out_has_skip cannot be both True."

        self.to_out = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False) if not identity_to_out else nn.Identity(),
            nn.Dropout(dropout)
        )
        self.out_has_skip = out_has_skip
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, context=None):
        h = self.num_heads

        # q: [BS, Q, D] -> [BS, Q, D].
        q = self.to_q(x)
        if context is None:
            context = x

        # k: [BS, L, D] -> [BS, L, D].
        k = self.to_k(context)

        # Compatible with old code.
        if not hasattr(self, 'dynamic_to_v'):
            self.dynamic_to_v = False

        if self.dynamic_to_v:
            # x: [BS, Q, D] -> [BS, Q, D, D]. There are effectively Q to_v projections.
            # The BS dim is repeated from a batch size of 1.
            to_v = self.to_v(x).reshape(-1, x.shape[1], x.shape[2], x.shape[2])
            # context: [BS, L, D] -> [BS, 1, L, D].
            # to_v: [BS, Q, D, D]
            # v: [BS, Q, L, D].
            v = torch.matmul(context.unsqueeze(1), to_v)
        else:
            # v: [BS, L, D].
            v = self.to_v(context)

        # q: [6, 32, 128], k: [6, 17, 128].
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))
        if self.dynamic_to_v:
            # v: [6, 32, 17, 128].
            v = rearrange(v, 'b q n (h d) -> (b h) q n d', h=h)
        else:
            v = rearrange(v, 'b n (h d) -> (b h) n d', h=h)

        scale = q.size(-1) ** -0.25

        sim = einsum('b i d, b j d -> b i j', q * scale, k * scale)
        # sim: [6, 64, 17]. 6: bs 1 * h 6.
        # attention, what we cannot get enough of
        # NOTE: the normalization is done across tokens, not across pixels.
        # So for each pixel, the sum of attention scores across tokens is 1.
        attn = sim.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #print(attn.std())

        if self.dynamic_to_v:
            # attn: [6, 32, 17]. v: [6, 32, 17, 128]. 128: dim of each head. out: [6, 32, 128].
            out = einsum('b i j, b i j d -> b i d', attn, v)
        else:
            # v: [6, 17, 128]. out: [6, 32, 128].
            out = einsum('b i j, b j d -> b i d', attn, v)

        # [6, 32, 128] -> [1, 32, 768].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if self.out_has_skip:
            out = self.to_out(out) + out
        else:
            out = self.to_out(out)

        return out

class SubjBasisGenerator(nn.Module):
    def __init__(
        self,
        depth=2,                            # number of (CrossAttention, FeedForward) layers.     
        # number of cross-attention heads. Half of the number of heads 12 of OpenAI clip-vit-large-patch14:
        # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
        num_heads=6,                       
        # num_out_queries: number of output queries.
        # 2 * Static/Ada layerwise_lora_rank. * 2 to generate both static and ada bases.
        # Two different SubjBasisGenerator instances are used to generate subj and bg embedder bases.
        num_id_vecs=16,                     # number of identity vectors.
        num_out_queries=234,                # fg: 234 = 9 * 26. bg: 104 = 4 * 26.
        image_embedding_dim=768,            # CLIP image feature dimension, as per config.json above.
        # DINO vits16 has 6 attention heads:
        # https://huggingface.co/facebook/dino-vits16/blob/main/config.json
        dino_embedding_dim=384,             # DINO object feature dimension for objects.
        # Number of low-rank latent queries. If num_latent_queries = 1, 
        # then basically all output queries are the same.
        num_latent_queries=32,               
        num_prompt2token_emb_modes=4,       # number of modes for prompt2token_emb.
        num_lora2hira_modes=4,              # number of modes for Lora2Hira.  
        output_dim=768,                     # CLIP text embedding input dimension.
        elementwise_affine: bool = True,    # Whether to use elementwise affine in LayerNorms.
        use_FFN: bool = False,              # Whether to use FeedForward layer after cross-attention.
        placeholder_is_bg: bool = False,    # Whether the placeholder is for the image background.
        ip_model_ckpt_path: str = None,     # Path to the IP-Adapter model checkpoint.
        mean_face_proj_emb_path: str = None, # Path to the mean face projection embedding.
        use_dynamic_to_v: bool = False,      # Whether to use dynamic to_v in CrossAttention.
    ):
        super().__init__()
        assert depth > 0, "depth must be > 0."
        self.depth = depth

        self.proj_in = nn.Sequential(
            nn.Linear(image_embedding_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        )

        self.placeholder_is_bg   = placeholder_is_bg
        self.num_latent_queries  = num_latent_queries
        self.latent_query_dim    = output_dim
        self.mean_face_proj_emb  = 0

        # If not self.placeholder_is_bg:
        # The dimension of IP-Adapter face features for humans is the same as output_dim = latent_query_dim.   
        # self.face_proj_in: [1, 512] -> [1, 16, 768].
        # If self.placeholder_is_bg: face_proj_in is set to None.
        self.init_face_proj_in(output_dim, ip_model_ckpt_path, mean_face_proj_emb_path, device='cpu')

        if not self.placeholder_is_bg:
            # [1, 384] -> [1, 16, 768].
            self.obj_proj_in  = ExpandEmbs(dino_embedding_dim, output_dim, expansion_ratio=num_id_vecs,
                                           elementwise_affine=elementwise_affine)
            self.prompt2token_emb_proj = MultimodeProjection(input_dim=output_dim, 
                                                             num_modes=num_prompt2token_emb_modes,
                                                             elementwise_affine=elementwise_affine)
        else:
            # For background placeholders, face and object embeddings are not used as they are foreground.
            self.face_proj_in = None
            self.obj_proj_in  = None
            self.prompt2token_emb_proj = None

        self.num_out_queries        = num_out_queries
        self.num_lora2hira_modes    = num_lora2hira_modes
        self.elementwise_affine     = elementwise_affine
        self.output_scale           = output_dim ** -0.5
        # Linearly combine the latent queries to generate the output queries.
        self.lora2hira = Lora2Hira(lora_rank=num_latent_queries, hira_rank=num_out_queries, 
                                   output_dim=output_dim, num_modes=num_lora2hira_modes,
                                   elementwise_affine=elementwise_affine)
        
        self.layers             = nn.ModuleList([])
        self.latent_queries     = nn.ParameterList([])
        self.latent_query_lns   = nn.ModuleList([])
        self.use_FFN            = use_FFN

        for dep in range(depth):
            dynamic_to_v = use_dynamic_to_v and not self.placeholder_is_bg
            identity_to_v   = not dynamic_to_v
            identity_to_out = dynamic_to_v
            out_has_skip = not identity_to_out

            self.layers.append(
                nn.ModuleList(
                    [
                        # dim=768, num_heads=6.
                        CrossAttention(input_dim=output_dim, num_heads=num_heads, dropout=0.1,
                                       identity_to_q=True, identity_to_k=True, identity_to_v=identity_to_v,
                                       dynamic_to_v=dynamic_to_v, 
                                       identity_to_out=identity_to_out,
                                       out_has_skip=out_has_skip),
                        # FeedForward: 2-layer MLP with GELU activation.
                        # LayerNorm -> Linear -> GELU -> Linear.
                        # Only use FFN in the first layer.
                        FeedForward(dim=output_dim, mult=1, elementwise_affine=elementwise_affine) \
                            if self.use_FFN else nn.Identity(),
                    ]
                )
            )
            # These sets of latent queries are used to attend to the ad-hoc identity features.
            layer_latent_queries = nn.Parameter(torch.randn(1, self.num_latent_queries, output_dim) / output_dim**0.5)
            self.latent_queries.append(layer_latent_queries)
            self.latent_query_lns.append(nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine))

        print(repr(self))

    def forward(self, clip_features, id_embs, extra_token_embs, is_face, training_percent=0):    
        BS = clip_features.shape[0]
        # No need to use id_embs if placeholder_is_bg.
        if (not self.placeholder_is_bg) and (id_embs is not None):
            if is_face:
                # id_embs: [BS, 512] -> [BS, 16, 768].
                if self.freeze_face_proj_in:
                    # Loaded pretrained IP-Adapter model weight. No need to update face_proj_in.
                    with torch.no_grad():
                        id_embs0 = self.face_proj_in(id_embs)
                else:
                    # face_proj_in will be updated during training.
                    id_embs0 = self.face_proj_in(id_embs)

                id_embs = F.normalize(id_embs0, p=2, dim=2) - self.mean_face_proj_emb
                # id_embs is projected to the token embedding space.
                id_embs = self.prompt2token_emb_proj(id_embs)
            else:
                # id_embs: [BS, 384] -> [BS, 16, 768].
                # obj_proj_in is expected to project the DINO object features to 
                # the token embedding space. So no need to use prompt2token_emb_proj.
                id_embs = self.obj_proj_in(id_embs)

            if extra_token_embs is not None:
                # extra_token_embs: [BS, 768] -> [BS, 1, 768].
                # extra_token_embs should have been layer-normalized before being passed to the model.
                # Otherwise, the default magnitude of the extra_token_embs is much smaller than id_embs.
                # If applicable, extra_token_embs should have been token-wise weighted before 
                # being passed to the model.
                # Although id_embs have been L2-normalized, later it has passed through self.prompt2token_emb_proj
                # which layer-normalizes the output. So it's not L2-normalized anymore.
                # Therefore, we don't need to L2-normalize extra_token_embs.
                extra_token_embs = extra_token_embs.unsqueeze(1)
                #extra_token_embs = F.normalize(extra_token_embs, p=2, dim=2)
                id_embs = torch.cat([id_embs, extra_token_embs], dim=1)
        else:
            # Otherwise, context is the ad-hoc CLIP image features.
            # id_embs: [BS, 257, 768].
            id_embs = self.proj_in(clip_features)

        # context is already in the token embedding space.
        context = id_embs

        for i, (attn, ff) in enumerate(self.layers):
            latent_queries = self.latent_query_lns[i](self.latent_queries[i])
            latent_queries = latent_queries.repeat(BS, 1, 1)
            context = attn(latent_queries, context)

            # Gradually reduce the drop path rate from 0.4 to 0.1 (average: 0.3).
            # The ratio in dinov2's paper is 0.3 or 0.4. 
            # https://github.com/huggingface/pytorch-image-models/issues/1836
            p_drop_path = anneal_value(training_percent, 1, (0.4, 0.2)) if self.training else 0
            # ff is either nn.Identity() or nn.Sequential. If it's nn.Sequential, it implies self.use_FFN is True.
            # (torch.rand(1) > self.p_drop_path) is evaluated to [True] or [False], which is equivalent to True or False.
            if isinstance(ff, nn.Sequential) and (torch.rand(1) > p_drop_path):
                context = ff(context) + context
        
        output_queries = self.lora2hira(context)
        return output_queries * self.output_scale

    def init_face_proj_in(self, output_dim=768, ip_model_ckpt_path=None, 
                          mean_face_proj_emb_path=None,
                          device='cpu'):
        if self.placeholder_is_bg:
            self.face_proj_in = None
            self.freeze_face_proj_in = False
            print("Bg face_proj_in is set to None")
            return
        
        self.face_proj_in = IP_MLPProjModel(cross_attention_dim=output_dim, 
                                            id_embeddings_dim=512, num_tokens=16)
        self.face_proj_in.to(device)

        if ip_model_ckpt_path is not None:
            ip_model_ckpt = torch.load(ip_model_ckpt_path, map_location=device)
            self.face_proj_in.load_state_dict(ip_model_ckpt['image_proj'])
            self.freeze_face_proj_in = True
            print(f"Subj face_proj_in is loaded from {ip_model_ckpt_path}")
        else:
            self.freeze_face_proj_in = False
            print("Subj face_proj_in is randomly initialized")

        if mean_face_proj_emb_path is not None:
            self.mean_face_proj_emb = torch.load(mean_face_proj_emb_path)
            # Wrap mean_face_proj_emb with nn.Parameter, so that it's put on the GPU automatically.
            self.mean_face_proj_emb = nn.Parameter(self.mean_face_proj_emb, requires_grad=False)
            print(f"mean_face_proj_emb ({list(self.mean_face_proj_emb.shape)}) is loaded from {mean_face_proj_emb_path}")

    def __repr__(self):
        type_sig = 'subj' if not self.placeholder_is_bg else 'bg'
        return f"{type_sig} SubjBasisGenerator: depth={self.depth}, use_FFN={self.use_FFN}, latent_queries={self.num_latent_queries}*{self.latent_query_dim}, num_out_queries={self.num_out_queries}, " \
               f"num_lora2hira_modes={self.num_lora2hira_modes}, elementwise_affine={self.elementwise_affine}"
    
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
    
