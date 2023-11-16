from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
from ldm.util import distribute_embedding_to_M_tokens

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, mask=None):
        for layer in self:
            # TimestepBlock: often ResBlock layers, which take time embedding as an input.
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # SpatialTransformer layers take text embedding as an input.
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, mask=mask)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,     # 2
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.debug_attn = False

        self.backup_vars = { 
                            'use_conv_attn_kernel_size:layerwise':      [-1] * 16,
                            'attn_copycat_emb_range':                   None,
                            'contrast_fg_bg_attns':                     False,
                            'bg_attn_behavior_in_inference':            'zero',
                            'conv_attn_layer_scale:layerwise':          None,
                            'save_attn_vars':                           False,
                            'normalize_subj_attn':                      False,
                            'is_training':                              True,
                           }

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    # set_cross_attn_flags: Set one or more flags for all or a subset of cross-attention layers.
    # If ca_layer_indices is None, then set the flags for all cross-attention layers.
    def set_cross_attn_flags(self, ca_flag_dict=None,   ca_layer_indices=None,
                             trans_flag_dict=None,      trans_layer_indices=None):
        if ca_flag_dict is None and trans_flag_dict is None:
            return None, None
        
        all_ca_layer_indices = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # l2ca: short for layer_idx2ca_layer_idx.
        l2ca = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                 17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 }
        
        if ca_layer_indices is None:
            ca_layer_indices = all_ca_layer_indices
        if trans_layer_indices is None:
            trans_layer_indices = all_ca_layer_indices

        if (ca_flag_dict is not None) and len(ca_layer_indices) > 0:            
            old_ca_flag_dict    = {}
            for k, v in ca_flag_dict.items():
                old_ca_flag_dict[k] = self.backup_vars[k]
                self.backup_vars[k] = v

                if k.endswith(":layerwise"):
                    k = k[:-len(":layerwise")]
                    v_is_layer_specific = (v is not None)
                else:
                    v_is_layer_specific = False

                layer_idx = 0
                for module in self.input_blocks:
                    if layer_idx in ca_layer_indices:
                        # module: SpatialTransformer.
                        # module.transformer_blocks: contains only 1 BasicTransformerBlock 
                        # that does cross-attention with layer_context in attn2 only.      
                        v2 = v[l2ca[layer_idx]] if v_is_layer_specific else v               
                        module[1].transformer_blocks[0].attn2.__dict__[k] = v2
                    layer_idx += 1

                if layer_idx in ca_layer_indices:
                    v2 = v[l2ca[layer_idx]] if v_is_layer_specific else v
                    self.middle_block[1].transformer_blocks[0].attn2.__dict__[k] = v2
                layer_idx += 1

                for module in self.output_blocks:
                    if layer_idx in ca_layer_indices:
                        v2 = v[l2ca[layer_idx]] if v_is_layer_specific else v
                        module[1].transformer_blocks[0].attn2.__dict__[k] = v2
                    layer_idx += 1
        else:
            old_ca_flag_dict = None

        # trans_flag_dict is optional.
        if (trans_flag_dict is not None) and len(trans_layer_indices) > 0:  
            old_trans_flag_dict = {}
            for k, v in trans_flag_dict.items():
                old_trans_flag_dict[k] = self.backup_vars[k]
                self.backup_vars[k] = v

                if k.endswith(":layerwise"):
                    k = k[:-len(":layerwise")]
                    v_is_layer_specific = (v is not None)
                else:
                    v_is_layer_specific = False

                layer_idx = 0
                for module in self.input_blocks:
                    if layer_idx in trans_layer_indices:
                        # module: SpatialTransformer.
                        # module.transformer_blocks: contains only 1 BasicTransformerBlock 
                        # that does cross-attention with layer_context in attn2 only.   
                        v2 = v[l2ca[layer_idx]] if v_is_layer_specific else v
                        module[1].transformer_blocks[0].__dict__[k] = v2
                    layer_idx += 1

                if layer_idx in trans_layer_indices:
                    v2 = v[l2ca[layer_idx]] if v_is_layer_specific else v
                    self.middle_block[1].transformer_blocks[0].__dict__[k] = v2
                layer_idx += 1

                for module in self.output_blocks:
                    if layer_idx in trans_layer_indices:
                        v2 = v[l2ca[layer_idx]] if v_is_layer_specific else v
                        module[1].transformer_blocks[0].__dict__[k] = v2
                    layer_idx += 1
        else:
            old_trans_flag_dict = None

        return old_ca_flag_dict, old_trans_flag_dict
    
    def forward(self, x, timesteps=None, context=None, y=None, 
                context_in=None, extra_info=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        
        hs = []

        distill_feats = {}
        distill_attns = {}
        distill_attnscores = {}
        distill_ks      = {}
        distill_vs      = {}

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        use_layerwise_context = extra_info.get('use_layerwise_context', False) if extra_info is not None else False
        use_ada_context       = extra_info.get('use_ada_context', False)       if extra_info is not None else False
        iter_type             = extra_info.get('iter_type', 'normal_recon')    if extra_info is not None else 'normal_recon'
        is_training           = extra_info.get('is_training', True)            if extra_info is not None else True
        capture_distill_attn  = extra_info.get('capture_distill_attn', False)  if extra_info is not None else False
        use_conv_attn_kernel_size    = extra_info.get('use_conv_attn_kernel_size',  None)   if extra_info is not None else None
        attn_copycat_emb_range       = extra_info.get('attn_copycat_emb_range',  None)  if extra_info is not None else None
        contrast_fg_bg_attns         = extra_info.get('contrast_fg_bg_attns',    False) if extra_info is not None else False
        bg_attn_behavior_in_inference = extra_info.get('bg_attn_behavior_in_inference', 'zero') if extra_info is not None else 'zero'
        conv_attn_layerwise_scales   = extra_info.get('conv_attn_layerwise_scales', None) if extra_info is not None else None
        subj_indices          = extra_info.get('subj_indices', None)           if extra_info is not None else None
        bg_indices            = extra_info.get('bg_indices', None)             if extra_info is not None else None
        normalize_subj_attn   = extra_info.get('normalize_subj_attn', False)   if extra_info is not None else False
        img_mask              = extra_info.get('img_mask', None)               if extra_info is not None else None
        emb_v_mixer           = extra_info.get('emb_v_mixer', None)            if extra_info is not None else None
        emb_k_mixer           = extra_info.get('emb_k_mixer', None)            if extra_info is not None else None
        emb_v_layers_cls_mix_scales = extra_info.get('emb_v_layers_cls_mix_scales', None)   if extra_info is not None else None
        emb_k_layers_cls_mix_scales = extra_info.get('emb_k_layers_cls_mix_scales', None)   if extra_info is not None else None
        debug_attn            = extra_info.get('debug_attn', self.debug_attn)  if extra_info is not None else self.debug_attn

        # If uncond (null) condition is active, then subj_indices = None.
        subj_indices_B, subj_indices_N = subj_indices if subj_indices is not None else (None, None)

        if use_layerwise_context:
            B = x.shape[0]
            # If use_layerwise_context, then context is static layerwise embeddings.
            # context: [16*B, N, 768] reshape => [B, 16, N, 768] permute => [16, B, N, 768]
            context = context.reshape(B, 16, -1, context.shape[-1]).permute(1, 0, 2, 3)

        def get_layer_context(layer_idx, layer_attn_components):
            # print(h.shape)
            if not use_layerwise_context:
                return context, None

            # skipped_layers: 0, 3, 6, 9, 10, 11, 13, 14, 15
            # 25 layers, among which 16 layers are conditioned.
            layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 }
            # Simply return None, as the context is not used anyway.
            if layer_idx not in layer_idx2ca_layer_idx:
                return None, None, None
            
            emb_idx = layer_idx2ca_layer_idx[layer_idx]
            layer_static_context = context[emb_idx]

            if iter_type.startswith("mix_"):
                # layer_static_context is v, k concatenated. Separate it into v and k.
                layer_static_context_v, layer_static_context_k = \
                            layer_static_context.chunk(2, dim=1)
                if layer_static_context_v.shape[1] < 77:
                    breakpoint()
            else:
                layer_static_context_v = layer_static_context

            if use_ada_context:
                ada_embedder   = extra_info['ada_embedder']
                layer_attn_components['layer_static_prompt_embs'] = layer_static_context_v
                # emb: time embedding. h: features from the previous layer.
                # context_in: ['an illustration of a dirty z, , ,  swimming in the ocean, with backlight', 
                #              'an illustration of a dirty z, , ,  swimming in the ocean, with backlight', 
                #              'an illustration of a dirty cat, , ,  swimming in the ocean, with backlight', 
                #              'an illustration of a dirty cat, , ,  swimming in the ocean, with backlight']
                # The 1st and 2nd, and 3rd and 4th prompts are the same, if it's a do_teacher_filter iteration.
                layer_ada_context, ada_emb_weight \
                    = ada_embedder(context_in, layer_idx, layer_attn_components, emb)
                static_emb_weight = 1 - ada_emb_weight

                # If static context is expanded by doing prompt mixing,
                # we need to duplicate layer_ada_context along dim 1 (tokens dim) to match the token number.
                # 'mix_' in iter_type: could be "mix_hijk" (training or inference).
                if iter_type == 'mix_hijk':
                    if layer_ada_context.shape[1] != layer_static_context.shape[1] // 2:
                        breakpoint()
                    # iter_type == 'mix_hijk'. layer_static_context has been split into v and k above.
                    # layer_static_context_v, layer_static_context_k = layer_static_context.chunk(2, dim=1)
                    # The second half of a mix_hijk batch is always the mix instances,
                    # even for twin comp sets.
                    subj_layer_ada_context, cls_layer_ada_context = layer_ada_context.chunk(2)
                    # In ddpm, distribute_embedding_to_M_tokens() is applied on a text embedding whose 1st dim is the 16 layers.
                    # Here, the 1st dim of cls_layer_ada_context is the batch.
                    # But we can still use distribute_embedding_to_M_tokens() without special processing, since
                    # in both cases, the 2nd dim is the token dim, so distribute_embedding_to_M_tokens() works in both cases.
                    # subj_indices_N:      subject token indices within the subject single prompt (BS=1).
                    # len(subj_indices_N): embedding number of the subject token.
                    # cls_layer_ada_context: [2, 77, 768]. subj_indices_N: [6, 7, 8, 9, 6, 7, 8, 9]. 
                    # Four embeddings (6,7,8,9) for each token.
                    cls_layer_ada_context = distribute_embedding_to_M_tokens(cls_layer_ada_context, subj_indices_N)
                    if emb_v_mixer is not None:
                        # Mix subj ada emb into mix ada emb, in the same way as to static embeddings.
                        # emb_v_cls_mix_scale: [2, 1]
                        emb_v_cls_mix_scale   = emb_v_layers_cls_mix_scales[:, [emb_idx]]
                        # subj_layer_ada_context, cls_layer_ada_context: [2, 77, 768]
                        # emb_v_mixer is a partial function that implies mix_indices=subj_indices_1b.                            
                        mix_layer_ada_context_v = emb_v_mixer(cls_layer_ada_context, subj_layer_ada_context, 
                                                              c1_mix_scale=emb_v_cls_mix_scale)
                    else:
                        mix_layer_ada_context_v = cls_layer_ada_context
                        emb_v_cls_mix_scale = 0
                            
                    layer_ada_context_v = th.cat([subj_layer_ada_context, mix_layer_ada_context_v], dim=0)
                    # layer_static_context_v, layer_ada_context_v: [2, 77, 768]
                    # layer_hyb_context_v: hybrid (static and ada) layer context 
                    # fed to the current UNet layer, [2, 77, 768]
                    layer_hyb_context_v = layer_static_context_v * static_emb_weight \
                                           + layer_ada_context_v * ada_emb_weight

                    if emb_k_mixer is not None:
                        # Mix subj ada emb into mix ada emb, in the same way as to static embeddings.
                        # emb_v_cls_mix_scale: [2, 1]
                        emb_k_cls_mix_scale   = emb_k_layers_cls_mix_scales[:, [emb_idx]]
                        # subj_layer_ada_context, cls_layer_ada_context: [2, 77, 768].
                        # emb_v_mixer is a partial function that implies mix_indices=subj_indices_1b.
                        mix_layer_ada_context_k = emb_k_mixer(cls_layer_ada_context, subj_layer_ada_context, 
                                                              c1_mix_scale=emb_k_cls_mix_scale)
                    else:
                        mix_layer_ada_context_k = cls_layer_ada_context 
                        emb_k_cls_mix_scale = 0

                    layer_ada_context_k = th.cat([subj_layer_ada_context, mix_layer_ada_context_k], dim=0)

                    # Replace layer_static_context with layer_static_context_k.
                    layer_hyb_context_k = layer_static_context_k * static_emb_weight \
                                           + layer_ada_context_k * ada_emb_weight
                    
                    # Pass both embeddings for hijacking the key of layer_hyb_context_v by layer_context_k.
                    layer_context = (layer_hyb_context_v, layer_hyb_context_k)
                else:
                    # normal_recon iters. Only one copy of k/v.
                    layer_hyb_context = layer_static_context * static_emb_weight \
                                         + layer_ada_context * ada_emb_weight

                    # Both the key and the value are layer_hyb_context. Only provide one.
                    layer_context = layer_hyb_context

            else:
                layer_context = layer_static_context

            # subj_indices is passed from extra_info, which was obtained when generating static embeddings.
            # Return subj_indices to cross attention layers for conv attn computation.
            return layer_context, subj_indices, bg_indices

        # conv_attn_layerwise_scales are not specified. So use the default value 0.5.
        # Here conv_attn_layerwise_scales is a list of scalars, not a learnable tensor.
        if conv_attn_layerwise_scales is None:
            # 1, 2, 4, 5, 7, 8           feature maps: 64, 64, 32, 32, 16, 16.
            # 0~5  (1, 2, 4, 5, 7, 8):                      weight 0.5.
            # 12, 16, 17, 18, 19, 20, 21 feature maps: 8, 16, 16, 16, 32, 32, 32.
            # 6~12 (12, 16, 17, 18, 19, 20, 21):            weight 0.5. #0.8.
            # 22, 23, 24                 feature maps: 64, 64, 64.
            # 13~15 (22, 23, 24):                           weight 0.5. #0.8.
            # This setting is based on the empirical observations of 
            # the learned conv_attn_layerwise_scales.      
            conv_attn_layerwise_scales = [1] * 16

        use_conv_attn_kernel_sizes = np.ones(16) * use_conv_attn_kernel_size
        # Most layers use use_conv_attn_kernel_size as the conv attn kernel size.
        # But disable conv attn on layers 6-10, i.e., 12, 16, 17, 18, 19. 
        # Based on the learned conv_attn_layerwise_scales, 
        # these layers don't like 3x3 conv attn (conv_attn_layerwise_scales are driven to 0.5~0.7).
        # Probably because the feature maps are too small (8x8 - 32x32), and a 3x3 conv attn head
        # takes up too much space.
        use_conv_attn_kernel_sizes[6:11] = -1

        # ca_layer_indices = None: Apply conv attn on all layers. 
        # Although layer 12 has small 8x8 feature maps, since we linearly combine 
        # pointwise attn with conv attn, we still apply conv attn (3x3) on it.
        ca_flags_stack = []
        old_ca_flags, _ = \
            self.set_cross_attn_flags( ca_flag_dict   = { 'use_conv_attn_kernel_size:layerwise': use_conv_attn_kernel_sizes,
                                                          'attn_copycat_emb_range':    attn_copycat_emb_range,
                                                          'contrast_fg_bg_attns':      contrast_fg_bg_attns,
                                                          'bg_attn_behavior_in_inference': 
                                                             bg_attn_behavior_in_inference,
                                                          'conv_attn_layer_scale:layerwise':
                                                             conv_attn_layerwise_scales,
                                                          'normalize_subj_attn': normalize_subj_attn,
                                                          'is_training': is_training },
                                       ca_layer_indices = None )
            
        # ca_flags_stack: each is (old_ca_flags, ca_layer_indices, old_trans_flags, trans_layer_indices).
        # None here means ca_flags have been applied to all layers.
        ca_flags_stack.append([ old_ca_flags, None, None, None ])

        if iter_type.startswith("mix_") or capture_distill_attn or debug_attn:
            # If iter_type == 'mix_hijk', save attention matrices and output features for distillation.
            distill_layer_indices = [7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            distill_old_ca_flags, _ = self.set_cross_attn_flags(ca_flag_dict = {'save_attn_vars': True}, 
                                                                ca_layer_indices = distill_layer_indices)
            ca_flags_stack.append([ distill_old_ca_flags, distill_layer_indices, None, None ])
        else:
            distill_layer_indices = []

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        
        # 0  input h:   [2, 4,    64, 64]
        # 1             [2, 320,  64, 64]
        # 2             [2, 320,  64, 64]
        # 3             [2, 320,  64, 64]
        # 4             [2, 320,  32, 32]
        # 5             [2, 640,  32, 32]
        # 6             [2, 640,  32, 32]
        # 7             [2, 640,  16, 16]
        # 8             [2, 1280, 16, 16]
        # 9             [2, 1280, 16, 16]
        # 10            [2, 1280, 8,  8]
        # 11            [2, 1280, 8,  8]
        # 12            [2, 1280, 8,  8]
        layer_idx = 0

        for module in self.input_blocks:
            get_layer_idx_context = partial(get_layer_context, layer_idx)
            # layer_context: [2, 77, 768], conditioning embedding.
            # emb: [2, 1280], time embedding.
            h = module(h, emb, get_layer_idx_context, mask=img_mask)
            hs.append(h)

            if layer_idx in distill_layer_indices:
                    distill_attns[layer_idx]        = module[1].transformer_blocks[0].attn2.cached_attn_mat 
                    distill_attnscores[layer_idx]   = module[1].transformer_blocks[0].attn2.cached_attn_scores
                    distill_ks[layer_idx]           = module[1].transformer_blocks[0].attn2.cached_k
                    distill_vs[layer_idx]           = module[1].transformer_blocks[0].attn2.cached_v
                    distill_feats[layer_idx]        = h

            layer_idx += 1
        
        get_layer_idx_context = partial(get_layer_context, layer_idx)

        # 13 [2, 1280, 8, 8]
        h = self.middle_block(h, emb, get_layer_idx_context, mask=img_mask)
        if layer_idx in distill_layer_indices:
                distill_attns[layer_idx]        = self.middle_block[1].transformer_blocks[0].attn2.cached_attn_mat 
                distill_attnscores[layer_idx]   = self.middle_block[1].transformer_blocks[0].attn2.cached_attn_scores
                distill_ks[layer_idx]           = self.middle_block[1].transformer_blocks[0].attn2.cached_k
                distill_vs[layer_idx]           = self.middle_block[1].transformer_blocks[0].attn2.cached_v
                distill_feats[layer_idx]        = h

        layer_idx += 1

        # 14 [2, 1280, 8,  8]
        # 15 [2, 1280, 8,  8]
        # 16 [2, 1280, 16, 16]
        # 17 [2, 1280, 16, 16]
        # 18 [2, 1280, 16, 16]
        # 19 [2, 1280, 32, 32]
        # 20 [2, 640,  32, 32]
        # 21 [2, 640,  32, 32]
        # 22 [2, 640,  64, 64]
        # 23 [2, 320,  64, 64]
        # 24 [2, 320,  64, 64]
        
        for module in self.output_blocks:
            get_layer_idx_context = partial(get_layer_context, layer_idx)
            skip_h = hs.pop()
            h = th.cat([h, skip_h], dim=1)

            # layer_context: [2, 77, 768], emb: [2, 1280].
            h = module(h, emb, get_layer_idx_context, mask=img_mask)
            if layer_idx in distill_layer_indices:
                    distill_attns[layer_idx]        = module[1].transformer_blocks[0].attn2.cached_attn_mat 
                    distill_attnscores[layer_idx]   = module[1].transformer_blocks[0].attn2.cached_attn_scores
                    distill_ks[layer_idx]           = module[1].transformer_blocks[0].attn2.cached_k
                    distill_vs[layer_idx]           = module[1].transformer_blocks[0].attn2.cached_v
                    distill_feats[layer_idx]        = h

            layer_idx += 1

        extra_info['unet_feats']        = distill_feats
        extra_info['unet_attns']        = distill_attns
        extra_info['unet_attnscores']   = distill_attnscores
        extra_info['unet_ks']           = distill_ks
        extra_info['unet_vs']           = distill_vs

        if debug_attn:
            breakpoint()
        
        # Restore the original flags in cross-attention layers, in reverse order.
        for old_ca_flags, ca_layer_indices, old_trans_flags, trans_layer_indices in reversed(ca_flags_stack):
            self.set_cross_attn_flags(ca_flag_dict=old_ca_flags, 
                                      ca_layer_indices=ca_layer_indices,
                                      trans_flag_dict=old_trans_flags,
                                      trans_layer_indices=trans_layer_indices)

        # [2, 320, 64, 64]
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

