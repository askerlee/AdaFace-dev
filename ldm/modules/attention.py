from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint
from ldm.util import replace_rows_by_conv_attn, normalize_attn_at_indices, \
                     replace_rows_of_copycat_embs, contrast_fgbg_attns_in_attn_mat

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# LinearAttention is not used in Stable Diffusion.
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

# All CrossAttention layers have 8 heads.
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # print(dim_head, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.save_attn_vars = False
        self.use_conv_attn_kernel_size = -1
        self.shift_attn_maps_for_diff_embs = True
        self.infeat_size            = None
        self.conv_attn_layer_scale  = 1.0
        self.normalize_subj_attn    = False
        self.attn_copycat_emb_mod   = -1
        self.contrast_fgbg_coeff    = 0
        self.is_training            = True
        self.bg_attn_behavior_in_inference = 'zero'  # 'zero', 'copy_fg', 'contrast_fg'

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # x, q: [4, 4096, 320]
        q = self.to_q(x)

        if exists(context):
            context_provided = True
        else:
            context_provided = False
            context = x

        if callable(context):
            # Pass (x, ...) to context() to get the real context.
            # If uncond (null condition) is active, then returned subj_indices = None.
            # Don't do conv attn if uncond is active.
            layer_attn_components = { 'x': x, 'q': q, 'to_k': self.to_k, 
                                      'infeat_size': self.infeat_size, 'scale': self.scale }
            context, subj_indices, bg_indices = context(layer_attn_components)
        else:
            subj_indices = bg_indices = None

        if type(context) == tuple:
            v_context, k_context  = context
        else:
            v_context = k_context = context

        k = self.to_k(k_context)            
        v = self.to_v(v_context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
                
        # In-place replacement of the row(s) in the attention matrix sim corresponding to the subject tokens, 
        # by attention scores computed with a convolutional attention mechanism.
        # If uncond (null condition) is active, then returned subj_indices = None.
        # Don't do conv attn if uncond is active.
        # abs(self.conv_attn_layer_scale) >= 1e-6: 
        # Sometimes conv_attn_layer_scale is a tiny negative number, and checking for equality with 0.0
        # will fail.
        if context_provided and subj_indices is not None:
            if self.use_conv_attn_kernel_size > 0 and abs(self.conv_attn_layer_scale) >= 1e-6:
                # infeat_size is set in SpatialTransformer.forward().
                # conv_attn_mix_weight=1: weight to mix conv attn with point-wise attn. 
                # Setting to 1 disables point-wise attn.
                sim = replace_rows_by_conv_attn(sim, q, k, subj_indices, self.infeat_size, 
                                                self.use_conv_attn_kernel_size,
                                                h, self.scale, self.conv_attn_layer_scale, 
                                                conv_attn_mix_weight=1,
                                                shift_attn_maps_for_diff_embs=self.shift_attn_maps_for_diff_embs)

            if self.attn_copycat_emb_mod > 0:
                sim = replace_rows_of_copycat_embs(sim, subj_indices, self.attn_copycat_emb_mod, h)
            
            if self.contrast_fgbg_coeff > 0 and bg_indices is not None:
                # During inference, if bg tokens are included in the prompt, 
                # we set bg attn to 0 to prevent it from affecting the generation.
                if self.bg_attn_behavior_in_inference is not None and (not self.is_training):
                    bg_attn_behavior = self.bg_attn_behavior_in_inference
                else:
                    bg_attn_behavior = 'contrast_fg'

                sim = contrast_fgbg_attns_in_attn_mat(sim, subj_indices, bg_indices, h,
                                                       bg_attn_behavior=bg_attn_behavior,
                                                       contrast_coeff=self.contrast_fgbg_coeff)

            if self.normalize_subj_attn:
                sim = normalize_attn_at_indices(sim, subj_indices, h)
            
        # if context_provided (cross attn with text prompt), then sim: [16, 4096, 77]. 
        # Otherwise, it's self attention, sim: [16, 4096, 4096].
        # img_mask should only be provided and applied if not context_provided. 
        # Otherwise, the whole column of 77 sim values will all be max_neg_value, 
        # which lead to nan after softmax.
        if exists(mask):
            # mask: [2, 1, 64, 64] -> [16, 1, 4096]
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask.bool(), 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # sim: [64, 4096, 77]. 64: bs * h.
        # attention, what we cannot get enough of
        # NOTE: the normalization is done across tokens, not across pixels.
        # So for each pixel, the sum of attention scores across tokens is 1.
        attn = sim.softmax(dim=-1)

        # v: [64, 77, 40]. 40: dim of each head. out: [64, 4096, 40].
        out = einsum('b i j, b j d -> b i d', attn, v)
        # [64, 4096, 40] -> [8, 4096, 320].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)

        if self.save_attn_vars:
            self.cached_attn_mat    = rearrange(attn, '(b h) i j -> b h i j', h=h)
            self.cached_attn_scores = rearrange(sim,  '(b h) i j -> b h i j', h=h)
            # cached_k will be used in ddpm.py:calc_fg_bg_key_ortho_loss(), in which two ks will multiply each other.
            # So sqrt(self.scale) will scale the product of two ks by self.scale.
            self.cached_k           = rearrange(k,    '(b h) n d -> b h n d', h=h) * math.sqrt(self.scale)
            self.cached_v           = rearrange(v,    '(b h) n d -> b h n d', h=h) * math.sqrt(self.scale)
            #self.cached_out      = out
            #self.cached_infeat_size = self.infeat_size
            #breakpoint()

        return out


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        return checkpoint(self._forward, (x, context, mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x1 = self.attn1(self.norm1(x), mask=mask) + x
        # The mask is of the key (context). 
        # If key is text prompt, then we shouldn't provide img_mask.
        # Otherwise nan will occur.
        x_ca = self.attn2(self.norm2(x1), context=context)
        x2 = x1 + x_ca

        x3 = self.ff(self.norm3(x2)) + x2

        return x3

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        self.save_feat = False

    def forward(self, x, context=None, mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # Each block is BasicTransformerBlock.
        for block in self.transformer_blocks:
            block.attn2.infeat_size = (h, w)
            # mask: [2, 1, 64, 64]
            mask2 = F.interpolate(mask, size=(h, w), mode='nearest') if exists(mask) else None
            x = block(x, context=context, mask=mask2)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        if self.save_feat:
            self.feat = x

        x = self.proj_out(x)
        return x + x_in