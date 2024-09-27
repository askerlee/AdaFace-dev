from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

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
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3).contiguous()
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w).contiguous()
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
        q = rearrange(q, 'b c h w -> b (h w) c').contiguous()
        k = rearrange(k, 'b c h w -> b c (h w)').contiguous()
        w_ = torch.einsum('bij,bjk->bik', q, k).contiguous()

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)').contiguous()
        w_ = rearrange(w_, 'b i j -> b j i').contiguous()
        h_ = torch.einsum('bij,bjk->bik', v, w_).contiguous()
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h).contiguous()
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

        # always query_dim == inner_dim.
        self.to_q = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # print(dim_head, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.save_attn_vars     = False
        self.cached_activations = None
        self.infeat_size            = None
        
    def forward(self, x, context=None, mask=None):
        h = self.heads

        # x, q: [4, 4096, 320]
        q = self.to_q(x)

        if not exists(context):
            context = x

        if callable(context):
            # Call context() to get the real context.
            context = context()

        k = self.to_k(context)            
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h).contiguous(), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # Otherwise, the whole column of 77 sim values will all be max_neg_value, 
        # which lead to nan after softmax.
        if exists(mask):
            # mask: [2, 1, 64, 64] -> [2, 4096]
            mask = rearrange(mask, 'b ... -> b (...)').contiguous()
            max_neg_value = -torch.finfo(sim.dtype).max
            # mask: [2, 4096] -> [16, 1, 4096]
            mask = repeat(mask.bool(), 'b j -> (b h) () j', h=h)
            # sim: [16, 4096, 4096]. mask will be broadcasted to [16, 4096, 4096].
            # So some rows in dim 1 (e.g. [0, :, 4095]) of sim will be masked out (all elements in [0, :, 4095] is -inf).
            # But not all elements in [0, 4095, :] is -inf. Since the softmax is done along dim 2, this is fine.
            sim.masked_fill_(~mask, max_neg_value)

        # sim: [64, 4096, 77]. 64: bs * h.
        # attention, what we cannot get enough of
        # NOTE: the normalization is done across tokens, not across pixels.
        # So for each pixel, the sum of attention scores across tokens is 1.
        attn = sim.softmax(dim=-1)
        # v: [64, 77, 40]. 40: dim of each head. out: [64, 4096, 40].
        out = einsum('b i j, b j d -> b i d', attn, v)
        # [64, 4096, 40] -> [8, 4096, 320].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()
        out = self.to_out(out)

        if self.save_attn_vars:
            self.cached_activations = {}
            # cached q will be used in ddpm.py:calc_comp_fg_bg_preserve_loss(), in which two qs will multiply each other.
            # So sqrt(self.scale) will scale the product of two qs by self.scale.
            self.cached_activations['q'] = rearrange(q,    '(b h) n d -> b h n d', h=h).contiguous() * math.sqrt(self.scale)
            # cached k, v will be used in ddpm.py:calc_subj_comp_ortho_loss(), in which two ks will multiply each other.
            # So sqrt(self.scale) will scale the product of two ks/vs by self.scale.
            #self.cached_activations['k'] = rearrange(k,    '(b h) n d -> b h n d', h=h).contiguous() * math.sqrt(self.scale)
            #self.cached_activations['v'] = rearrange(v,    '(b h) n d -> b h n d', h=h).contiguous() * math.sqrt(self.scale)
            self.cached_activations['attn'] = rearrange(attn, '(b h) i j -> b h i j', h=h).contiguous()
            self.cached_activations['attnscore'] = rearrange(sim,  '(b h) i j -> b h i j', h=h).contiguous()

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
        # x1 dim  ==  x dim  ==  attn1 dim  ==  attn2 query_dim.
        x1 = self.attn1(self.norm1(x), mask=mask) + x
        # The mask is of the key (context). 
        # TODO: If key is text prompt, and we provide img_mask, then nan will occur.
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
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # Each block is BasicTransformerBlock.
        for block in self.transformer_blocks:
            block.attn2.infeat_size = (h, w)
            # mask: [2, 1, 64, 64]
            mask2 = F.interpolate(mask, size=(h, w), mode='nearest') if exists(mask) else None
            x = block(x, context=context, mask=mask2)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        
        if self.save_feat:
            self.feat = x

        x = self.proj_out(x)
        return x + x_in
