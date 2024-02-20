import torch
from torch import nn, einsum
from einops import rearrange, repeat
from ldm.modules.ema import LitEma
from ldm.modules.subj_basis_generator import SubjBasisGenerator

import torch.nn.functional as F
import numpy as np

from ldm.util import masked_mean, gen_gradient_scaler, extract_first_index_in_each_instance, \
                     add_noise_to_embedding, calc_ref_cosine_loss, \
                     get_clip_tokens_for_string, get_embeddings_for_clip_tokens, \
                     scan_cls_delta_strings, torch_uniform, extend_indices_N_by_n_times, \
                     extend_clip_text_embedder, calc_init_word_embeddings, calc_stats

from functools import partial
from collections import OrderedDict
import random
import copy
import glob

# When debugging, make the printed tensors less messy.
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

def reg_loss(x, loss_type='l2', selector=None):
    if selector is not None:
        # If selector(x) is False, the gradient flow is cut off.
        x = x * selector(x).float()
    if loss_type == 'l1':
        return x.abs().mean()
    elif loss_type == 'l2':
        return (x * x).mean()
    else:
        breakpoint()

# LNCat3 can be used on 2 or 3 input tensors.
class LNCat3(nn.Module):
    def __init__(self, chan1, chan2, chan3=0, dim=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(chan1, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(chan2, elementwise_affine=False)

        if chan3 > 0:
            self.ln3 = nn.LayerNorm(chan3, elementwise_affine=False)
        else:
            self.ln3 = None

        self.dim = dim

    def forward(self, x1, x2, x3=None):
        x1 = self.ln1(x1)
        x2 = self.ln2(x2)
        if (x3 is None) or (self.ln3 is None):
            return torch.cat([x1, x2], dim=self.dim)
        
        x3 = self.ln3(x3)
        return torch.cat([x1, x2, x3], dim=self.dim)

class MaskedAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    # x: [N, C, H, W], mask: [N, 1, H0, W0] of 1s (keep) or 0s (discard).
    # H, W: feature map size, H0, W0: original image size.
    # Return: [N, C]
    def forward(self, x, k=None, mask=None):
        if mask is None:
            return self.avgpool(x).view(x.shape[0], -1)
        
        mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        x = x * mask
        x = x.sum(dim=(2,3)) / mask.sum(dim=(2,3))
        return x

class MaskedAvgPool1d(nn.Module):
    def __init__(self, dim=1, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    #    x: [N, L, C], mask: [N, L, 1].
    # or x: [N, C, L], mask: [N, 1, L].
    # mask: values are 1s (keep) or 0s (discard).
    # L: number of patches.
    # Return: [N, C] (if keepdim=False) or [N, 1, C] (if keepdim=True and dim=1),
    # or [N, C, 1] (if keepdim=False and dim=2).
    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=self.dim, keepdim=self.keepdim)
        
        x = x * mask
        x = x.sum(dim=self.dim, keepdim=self.keepdim) / mask.sum(dim=self.dim, keepdim=self.keepdim)
        return x

# Set infeat_grad_scale < 1 to reduce the gradient flow into the UNet.
# feat_dims (ca_infeat_dims) = [ 320,  320,  640, 640, 1280, 1280, 1280, 1280, 
#                                1280, 1280, 640, 640, 640,  320,  320,  320 ]
class AttentionalPooler(nn.Module):
    def __init__(self, layer_idx, feat_dim, feat_reduction_ratio=8,
                 infeat_grad_scale=0.5):
        super().__init__()
        # Set to the same number of heads as the CrossAttention layers.
        # All CrossAttention layers in UNet have 8 heads.
        self.n_heads = 8    
        self.layer_inner_dim = feat_dim
        self.lora_dim = int(feat_dim / feat_reduction_ratio)
        #self.lora_attn_score_scale = self.lora_dim ** -0.5

        self.lora_fg_q_ln  = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)
        self.lora_bg_q_ln  = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)
        self.lora_k_ln     = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)

        # lora: Do dimension reduction to each head individually.
        # Most compute is spent on lora_to_k, as it's applied to all image tokens.
        # lora_to_k takes 8*(1280/8 * 1280/8/8) * N = 1280*160 = 25600*N flops, where N is the number of patches.
        # Without lora_to_k, most compute is spent on q*k, whose total flops is 
        # 8*(1280/8 * 1280/8) * N * 2 = 409600*N flops. So it's roughly a 16x reduction.
        # If layer_inner_dim == 1280, lora_dim == 160, n_heads == 8, 
        # then lora_to_k.weight is [160, 160, 1]. It's actually 8 groups, each (1280/8 * 160/8).
        # The to_q, to_k, to_v projections in the cross-attnion layer don't have bias. 
        # So we don't use bias here.
        self.lora_to_k     = nn.Conv1d(self.layer_inner_dim, self.lora_dim, kernel_size=1, groups=self.n_heads, bias=False)
        self.lora_to_fg_q  = nn.Conv1d(self.layer_inner_dim, self.lora_dim, kernel_size=1, groups=self.n_heads, bias=False)
        self.lora_to_bg_q  = nn.Conv1d(self.layer_inner_dim, self.lora_dim, kernel_size=1, groups=self.n_heads, bias=False)
        # Conv1d weight is normalized as U(-sqrt(k), sqrt(k)), where k = groups / (C_in * kernel_size).
        # But later einsum(...) uses all 8 groups, making the sum ~8 times larger. Therefore, we scale down
        # lora_q and lor_k by sqrt(groups), to cancel the magnitude mismatch.
        self.conv1d_extra_scale = self.n_heads ** -0.5

        # layer_idx is recorded for debugging purpose.
        self.layer_idx = layer_idx

        self.infeat_grad_scale = infeat_grad_scale      # Default 0.5

        self.is_fgbg_competitive = True
        self.attn_drop = nn.Dropout(0.1)
        self.out_drop  = nn.Dropout(0.1)

    # k: query in the UNet attention layer. Used as key here.
    # fg_q_emb: [768,] static subject embedding of this layer. Used as query here.
    # Use UNet feature query as k, and subject token as q, to compute the attention, 
    # which aggregates the original UNet layer input features x.
    # bg_q_emb: [N, 768].
    def forward(self, layer_attn_components, fg_q_emb, bg_q_emb, img_mask=None, debug=False):
        # x and q have the same shape.
        ca_x, ca_q, ca_to_k, ca_x_size \
                = [ layer_attn_components[key] for key in ('x', 'q', 'to_k', 'infeat_size') ]

        infeat_grad_scaler = gen_gradient_scaler(self.infeat_grad_scale)
        # By default, infeat_grad_scaler does 0.5 gs.
        ca_x_gs = infeat_grad_scaler(ca_x)
        ca_q_gs = infeat_grad_scaler(ca_q)

        x = ca_x_gs
        k = ca_q_gs
        # k: query (i.e., projected image features) from the UNet cross-attention layer. 
        # Repurposed as key here.
        k_ln    = self.lora_k_ln(k)
        # x is x1 in BasicTransformerBlock, which will be added with x_ca output from the cross-attn layer.
        # cross-attn v is the projection of the prompt embedding. So in order to yield proper x_ca,
        # we add x to k, so x is part of the v of the attentional pooler (whose output is the 
        # input features to the ada embedder).
        # On the other hand, k is cross-attn q, which is multiplied with the 
        # cross-attn k (projection of the prompt embedding). In order to provide proper cross-attn k,
        # we include k as the input to the attentional pooler.
        # Therefore, v = x + k. We can also concat(x, k), but it will double the feature dimension.
        #calc_stats(f"{self.layer_idx}-x",       x,      mean_dim=-1)
        #calc_stats(f"{self.layer_idx}-k_ln",    k_ln,   mean_dim=-1)
        # NOTE: The magnitude of k_ln is roughly 2x of x. So k_ln dominates v. 
        # But adding x to k_ln may enrich the features slightly and improve the performance slightly.
        # the to_q() of the UNet cross-attention layer doesn't change the dimension of k,
        # i.e., k dim = x dim. Therefore, x + k_ln is legal.
        # k_ln is roughly sqrt(n_heads) times of the scale of x.
        # So we have to normalize v by sqrt(n_heads) to match the scale of x. 
        # TODO: find out why the scale of x is sqrt(n_heads) smaller than that of lora_k_ln(k)?
        v = (x + k_ln) * (self.n_heads ** -0.5)
        # Use v as k. Originally we use normalized k_ln as k. But since v is enriched than k_ln,
        # we use v as k.
        k = v

        # Use to_k of the UNet attention layer as to_q here, 
        # as the subject embedding is used as the key in UNet.
        to_q = ca_to_k
        # fg_q_emb: [768] => [1, 768].
        fg_q_emb = fg_q_emb.unsqueeze(0)

        # to_q is actually to_k in the UNet attention layer, 
        # as the subject embedding is used as the key in UNet.
        # After applying to_q on fg_q_emb, fg_q consists of 8 heads.
        # fg_q: [1, 768] -> [1, 320].
        fg_q = to_q(fg_q_emb)
        # fg_q: [1, 320] -> [N, 1, 320]
        try:
            fg_q = repeat(fg_q, 'n d -> b n d', b=x.shape[0])
        except:
            breakpoint()

        # bg_q_emb: [N, 768] -> [N, 1, 768].
        bg_q_emb = bg_q_emb.unsqueeze(1)
        # bg_q: [N, 1, 768] -> [N, 1, 320].
        bg_q = to_q(bg_q_emb)

        # fg_q: [B, 1, 320], k: [B, 4096, 320], v: [B, 4096, 320]. 
        # The 320 dims of q,k consist of 8 heads, each head having 40 dims.
        #breakpoint()
        fg_q_ln = self.lora_fg_q_ln(fg_q)
        bg_q_ln = self.lora_bg_q_ln(bg_q)
        # q: [B, 1, 320]    -> [B, 320, 1]
        # k: [B, 4096, 320] -> [B, 320, 4096]
        # Permute dims to match the dim order of nn.Conv1d.
        fg_q_ln, bg_q_ln, k = map(lambda t: t.permute(0, 2, 1), (fg_q_ln, bg_q_ln, k))

        # NOTE: 320 and 64 are multi-head concatenated, 8*40 and 8*8.
        # lora_to_fg_q, lora_to_bg_q: nn.Conv1d of 8 groups, each group corresponding to a head.
        # In each group, reduce dimension 40 -> 5. 
        # So the overall computation complexity is 8*40*8=320*8, instead of 320*64.
        lora_fg_q = self.lora_to_fg_q(fg_q_ln)
        # During training, at 20% of the chance, bg_q_emb is dropped out to be 0. 
        # For these cases, lora_bg_q is 0 => sim_scores[:, 1] = 0, i.e., bg attn is uniform.
        lora_bg_q = self.lora_to_bg_q(bg_q_ln)
        # To be compatible with older ckpts.
        self.conv1d_extra_scale = self.n_heads ** -0.5

        # lora_k: [B, 320, 4096] -> [B, 64, 4096].
        lora_k = self.lora_to_k(k) * self.conv1d_extra_scale
        # lora_fg_q, lora_bg_q: [B, 64, 1]    -> [B, 1, 64]
        # lora_k:               [8, 64, 4096] -> [8, 4096, 64]
        lora_fg_q, lora_bg_q, lora_k = map(lambda t: t.permute(0, 2, 1), (lora_fg_q, lora_bg_q, lora_k))
        # lora_fg_q, lora_bg_q are two artificial tokens, each with 64 dims.
        # lora_q: [B, 2, 64]. 
        # (lora_fg_q, lora_bg_q) have std 1, but lora_k has std 0.37. So we scale down lora_fg_q, lora_bg_q, so that
        # the overall scale of (lora_fg_q, lora_bg_q) and lora_k are roughly the same.
        lora_q = torch.cat([lora_fg_q, lora_bg_q], dim=1) * (self.n_heads ** -0.5) * self.conv1d_extra_scale

        # Tuck the 8 heads into the batch dim.
        lora_q, lora_k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.n_heads), (lora_q, lora_k, v))

        # Dot product of the last dim. sim_scores: [B, 2, 4096].
        # The sim_scores are too large. So we scale them down by lora_attn_score_scale.
        # The root cause why the sim_scores are too large is due to the different initialization 
        # of nn.Conv1d (compared with nn.Linear).
        # lora_q and lora_k are both torch.float16. But lora_q is often larger.
        # If we do einsum(...) * self.lora_attn_score_scale, sometimes inf will occur, causing nan.
        # Therefore we do lora_q * self.lora_attn_score_scale first to avoid nan errors. 
        sim_scores = einsum('b i d, b j d -> b i j', lora_q, lora_k)
        #print(sim_scores.std())

        # Average sim scores across the heads. avg_sim_scores: [B, 2, 4096].
        avg_sim_scores = rearrange(sim_scores, '(b h) i j -> b h i j', h=self.n_heads).mean(dim=1, keepdim=True)
        avg_sim_scores = repeat(avg_sim_scores, 'b 1 i j -> b h i j', h=self.n_heads)
        avg_sim_scores = rearrange(avg_sim_scores, 'b h i j -> (b h) i j', h=self.n_heads)

        # Use avg_sim_scores to smooth sim_scores. 
        # Otherwise, sim_scores of some particular heads may have too large variances.
        sim_scores = sim_scores * 0.5 + avg_sim_scores * 0.5

        if img_mask is not None:
            # img_mask: [B, 1, 64, 64] 
            img_mask = F.interpolate(img_mask, size=ca_x_size, mode='nearest')
            # N, 1, H, W -> N, 1, L=H*W
            # -> [B, 8, 4096]
            # float_tensor.bool() converts 0.1/0.2... to True.
            img_mask = rearrange(img_mask, 'b ... -> b 1 (...)')
            img_mask = repeat(img_mask.bool(), 'b 1 j -> (b h) () j', h=self.n_heads)
            max_neg_value = -torch.finfo(sim_scores.dtype).max
            # masked_fill_() will broadcast img_mask to sim_scores's shape [B, 2, 4096].
            sim_scores.masked_fill_(~img_mask, max_neg_value)

        # attn: [B, 2, 4096]. 2: fg/bg, 4096: image patches.
        # ** If only normalizing across the token (2) dimension, the performance is poor. **
        if self.is_fgbg_competitive:
            # Attention probs are normalized across the joint space of fg/bg (2) and image patches (4096).
            # This means the fg attention on patch p_0 is competitive with all other patches {p_i}
            # in the image, as well as with bg on p_0. 
            # Although after ln_fg_out, the overall scales of fg and bg, 
            # are normalized (removed), respectively, it still gives more flexibility 
            # to each individual patch how much attention it will receive from an fg embedding 
            # (relative to bg embeddings).
            sim_scores_shape = sim_scores.shape
            # softmax() is applied on the joint space of fg/bg (2) and image patches (4096).
            # Therefore, when some pixels are masked, i.e., both fg and bg scores are masked at certain pixels,
            # doing so won't lead to 0.5/0.5 probs at these pixels.
            attn = sim_scores.reshape(sim_scores.shape[0], -1).softmax(dim=1)
            attn = attn.reshape(sim_scores_shape)
        else:
            # Attention probs are normalized across the image patches (4096) dimension.
            # fg attn and bg attn are independent of each other.
            attn = sim_scores.softmax(dim=-1)

        if torch.isnan(attn).any():
            print(f"AttentionalPooler: attn has NaN: {attn}")
            breakpoint()

        attn = self.attn_drop(attn)
        # attn_fg, attn_bg: [B*h, 1, 4096].
        attn_fg, attn_bg = attn.split(1, dim=1)
        attn_fg_sum = attn_fg.sum(dim=(1,2))
        if torch.any(attn_fg_sum == 0):
            if debug:
                print(f"AttentionalPooler: attn_fg_sum is 0: {attn_fg_sum}")
                breakpoint()

        # Do attentional feature pooling on v.
        # fg_out: [B, 1, 320]. 320: feature dimension. 
        fg_out = einsum('b i j, b j d -> b i d', attn_fg, v)
        # fg_out = self.ln_fg_out(fg_out)
        fg_out  = self.out_drop(fg_out)

        # bg_out: [B, 1, 320], similarly computed as fg_out.
        bg_out = einsum('b i j, b j d -> b i d', attn_bg, v)
        # bg_out = self.ln_bg_out(bg_out)
        bg_out  = self.out_drop(bg_out)

        fg_out, bg_out = map(lambda out: rearrange(out, '(b h) n d -> b n (h d)', h=self.n_heads), (fg_out, bg_out))

        # out: N, 1, D -> N, D, i.e., ([2, 768], [2, 768]).
        # Make the output shape consistent with MaskedAvgPool2d.
        return { 'fg_out': fg_out.squeeze(1), 'bg_out': bg_out.squeeze(1), 
                 'attn_fg': attn_fg, 'attn_bg': attn_bg }

# init_embedding: [L, M, 768].
class Embedding3d(nn.Module):
    def __init__(self, num_layers=16, num_vectors_per_subj_token=9, 
                 out_emb_dim=768, init_embedding=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_vectors_per_subj_token = num_vectors_per_subj_token
        self.out_emb_dim = out_emb_dim
        # self.embedding: [16, K, 768]
        self.embedding = nn.Parameter(torch.zeros(num_layers, num_vectors_per_subj_token, out_emb_dim), requires_grad=True)
        if init_embedding is not None:
            # init_embedding: [1, 768] => [16, 9, 768].
            self.embedding.data = init_embedding.clone()

        self.reset_cached_layer_tracker()

    def forward(self, layer_idx, token_idx=None):
        if token_idx is None:
            return self.embedding[layer_idx]
        else:
            return self.embedding[layer_idx, token_idx]
    
    # For EMA embeddings, the computation graph of new_embedding is disconnected.
    # Otherwise the computation graph from the previous iteration will be retained, causing OOM.
    def cache_layer(self, layer_idx, new_embedding, has_grad=False):
        if has_grad:
            embedding2 = self.embedding.clone()
            embedding2[layer_idx] = new_embedding
            self.embedding = nn.Parameter(embedding2, requires_grad=True)
        else:
            self.embedding.data[layer_idx] = new_embedding

        self.cached_layers[layer_idx] = 1

    def reset_cached_layer_tracker(self):
        self.cached_layers = {}

class StaticLayerwiseEmbedding(nn.Module):
    # num_layers: 16 (9 layers out of 25 of UNet are skipped), out_emb_dim: 768, 
    # r: rank of basis vectors. Usually set as (num of init words * ratio), ratio=2 or 3.
    # num of init words is usually 2. So r is usually 4~6.
    # If using init_vecs, init_noise_stds are applied to basis_rand_weights. 
    # Otherwise, init_up_noise_stds has no effect.
    # init_vec_weights: weights of init_vecs, not "init weights of vecs".
    # Assume init_vec_weights is already normalized, i.e., init_vec_weights.sum() = 1.
    # init_vecs_repeat: repeat init_vecs for multiple times as part of the initial basis vectors.
    # This is to help preserve more information of init_vecs during optimization with more redundancy, 
    # in case one copy is destroyed.
    # Extra copies of init_vecs are added with random noises to avoid the non-identifiability issue.
    # init_up_noise_stds[1] << init_up_noise_stds[0].
    # has_bias: if enabled, the output vectors will be dominated by self.bias.
    def __init__(self, num_layers=16, num_vectors_per_subj_token=1, 
                 out_emb_dim=768, r=6, init_noise_stds=(0.1, 0.04), 
                 init_string=None, init_vecs=None, init_vec_weights=None, 
                 has_bias=True, token_string="", 
                 # If do_zero_shot, then the basis vectors are generated by it 
                 # in a zero-shot fashion, instead of being learned from the data.                 
                 do_zero_shot=False,
                 device_type="cuda"):
        super().__init__()

        self.do_zero_shot = do_zero_shot

        self.token_string = token_string
        self.num_layers = num_layers
        self.K = num_vectors_per_subj_token
        self.out_emb_dim = out_emb_dim
        self.r = r

        if r > min(num_layers, out_emb_dim):
            raise ValueError(
                f"StaticLayerwiseEmbedding LoRA rank {r} must be less or equal than {min(num_layers, out_emb_dim)}"
            )

        if init_vecs is not None:
            if init_vecs.shape[1] != out_emb_dim or init_vecs.shape[0] > num_layers:
                raise ValueError(
                    f"StaticLayerwiseEmbedding init vectors shape {init_vecs.shape} must be (<={num_layers}, {out_emb_dim})"
                )

            N = self.N = init_vecs.shape[0]
            # pre_vecs: basis vectors that are initialized by prespecified init_vecs, updated through BP.
            # pre_vecs: [K, N, 768].

            if not self.do_zero_shot:
                self.pre_vecs = nn.Parameter(init_vecs.unsqueeze(0).repeat(self.K, 1, 1), requires_grad=True)
            else:
                self.pre_vecs = None
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs.data, dim=-1)
        else:
            N = self.N = 0
            self.pre_vecs = None

        # basis_rand_weights: 16 * K * r, basis_vecs: K * r * 768. 
        # output embeddings: basis_rand_weights * basis_vecs = 16 * K * 768.
        self.basis_rand_weights = nn.Parameter(torch.randn(num_layers, self.K, r), requires_grad=True)
        # basis_vecs: [K, r-N, 768], K sets, each consisting of r-N randomly initialized basis vectors. 
        # Will be updated through BP.
        # Each embedding of the K embeddings has its own basis_vecs and pre_vecs.
        # Separate pre_vecs and basis_vecs, to apply different regularizations on them.
        if not self.do_zero_shot:
            self.basis_vecs = nn.Parameter(torch.randn(self.K, r - N, out_emb_dim), requires_grad=True)
            # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
            self.basis_vecs.data = F.normalize(self.basis_vecs, dim=-1) / 4.
            # Always set the last basis vector to 0.
            self.basis_vecs.data[-1] = 0
        else:
            self.basis_vecs = None

        self.has_bias    = has_bias
        self.device_type = device_type
        # basis_comm_weights: [1, K, r]. Initialized as equal weights, then tuned through BP.
        # basis_comm_weights is added with basis_rand_weights. 
        # basis_rand_weights is a random component of the actual weights.
        # So equal weights here won't cause non-identifiable parameters and equal graidents.
        self.basis_comm_weights = nn.Parameter(torch.ones(1, self.K, r) / r, requires_grad=True)

        # N: number of init vectors.
        # init_up_noise_stds is only applied when init_vecs is passed in.
        if init_vecs is not None:
            self.basis_comm_weights.data.fill_(1. / N)
            # In basis_comm_weights, 
            # [:, :, N:] corresponds to randomly initialized self.basis_vecs.
            # [:, :, :N] corresponds to self.pre_vecs.
            # Lower the weights of the remaining r-N random vectors, to prevent the model  
            # from going too far away from the subspace spanned with init_vecs.
            # By default, these weights are 1/6*0.4 = 0.067.
            self.basis_comm_weights.data[:, :, N:] *= 0.4

            # basis_comm_weights: [1, K, r]
            if init_vec_weights is not None:
                assert len(init_vec_weights) == len(init_vecs), f"init_vec_weights must have length {len(init_vecs)}"
                # Assume init_vec_weights is already normalized, i.e., init_vec_weights.sum() = 1.
                self.basis_comm_weights.data[:, :, :N] = init_vec_weights.unsqueeze(0).unsqueeze(0)

            # After scaled down by init_up_noise_stds[0] (default 0.1) and 
            # init_up_noise_stds[1] (default 0.01), self.basis_rand_weights contains only small noise.
            # The common weight matrix (broadcasted from self.basis_comm_weights)             
            #                       [w_0, ..., w_N, 0, ..., 0,
            #                        w_0, ..., w_N, 0, ..., 0,
            #                                 ...                         
            #                        w_0, ..., w_N, 0, ..., 0.]
            # will be added to self.basis_rand_weights in forward().
            # First N columns are coefficients of init_vecs (or basis_vecs), 
            # these noises will be added to the common weight matrix, 
            # so make the noises smaller (init_noise_stds[1]=0.04).
            # As a result, the component from basis_vecs has less randomness, 
            # and is more constant across rows.
            self.basis_rand_weights.data[:, :, :N]    *= init_noise_stds[1]
            # The last num_layers-N block are coefficients of the extra learned vectors.
            # We don't want the result embeddings to be confined 
            # in the subspace of basis_vecs. So we let the extra learned vectors play a bigger role,
            # by making the noises larger (init_noise_stds[0]=0.1).
            self.basis_rand_weights.data[:, :, N:]    *= init_noise_stds[0]
        else:
            self.N = 0

        if self.has_bias and not self.do_zero_shot:
            # self.bias: 16 * K * 768.
            self.bias = nn.Parameter(torch.zeros(num_layers, self.K, out_emb_dim), requires_grad=True)
        else:
            self.bias = 0

        layers_out_lns = []
        for i in range(num_layers):
            # A specific LayerNorm is applied on each of the K embeddings in each layer.
            layer_out_lns = nn.ModuleList( [ nn.LayerNorm(out_emb_dim, elementwise_affine=False) for k in range(self.K) ] )
            layers_out_lns.append(layer_out_lns)
        self.layers_out_lns = nn.ModuleList(layers_out_lns)

        zero_shot_sig = "Zero-shot" if self.do_zero_shot else "Slow"
        print(f"{zero_shot_sig} StaticLayerwiseEmbedding {token_string} initialized with {self.K} total embs, {self.N} init vectors ({init_string}), {self.r} basis vectors")

    # Return static embeddings of all layers together.
    # static_zs_embs: [16, K, 768]. 
    # 16: number of layers. K: number of vectors per token. 
    def forward(self, static_zs_embs=None):
        with torch.autocast(device_type=self.device_type, enabled=False):
            # self.basis_comm_weights: [1, K, r] broadcasted to [16, K, r].
            basis_weights   = self.basis_rand_weights   + self.basis_comm_weights
            # torch.matmul: matrix multiplication.
            # torch.matmul(lora_up, basis_vecs): 16 * 768.

            if self.do_zero_shot:
                # Copy to bias, so that static_zs_embs is regularized by layerwise_embedding_norm_loss().
                self.bias = static_zs_embs
                return static_zs_embs
            
            # self.N: number of pre_vecs.
            if self.N > 0:
                # basis_vecs: [K, r, 768].
                basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=1)
            else:
                basis_vecs = self.basis_vecs

            # For each k < K, [16, 1, r]_k * [r, 768]_k -> [16, 1, 768].
            out_vecs_unnorm = [ torch.matmul(basis_weights[:, k], basis_vecs[k]) for k in range(self.K) ]
            # Apply layer-and-embedding specific layer normalization.
            # Note the order of for i ... for k ... in the list comprehension is the same as 
            # a conventional for loop: 1 for i ...
            #                          2     for k ...
            # Otherwise embeddings in out_vecs_ln will be in wrong order.
            out_vecs_ln = [ self.layers_out_lns[i][k](out_vecs_unnorm[k][i]) for i in range(self.num_layers) for k in range(self.K) ]
            out_vecs0 = torch.stack(out_vecs_ln, dim=0).reshape(self.num_layers, self.K, -1) / np.sqrt(self.out_emb_dim)

            # Different layers and embeddings have different biases.
            # self.bias: [16, K, 768].
            out_vecs = out_vecs0 + self.bias
            # Return static embeddings of all layers together: [16, K, 768].
            return out_vecs

class AdaEmbedding(nn.Module):
    # num_layers: 16 (9 layers out of 25 of UNet are skipped).
    # out_emb_dim: 768, r: 12.
    # infeat_dims: a list of 25 integers, each is the dimension of 
    # the input feature from the respective layer. 9 of them are skipped.
    # infeat_dims are (almost) reflective around the middle layer, except for the first and last layers.
    # Layer indices absent in layer_idx2ca_layer_idx are skipped layers.
    def __init__(self, num_layers=16, num_vectors_per_subj_token=1, 
                 fg_emb_count=1, bg_emb_count=0, use_cached_bg=False,
                 out_emb_dim=768, r=12, 
                 init_string=None, init_vecs=None, 
                 # 16 cross-attention layers.
                 ca_infeat_dims = [ 320,  320,  640, 640, 1280, 1280, 1280, 1280, 
                                    1280, 1280, 640, 640, 640,  320,  320,  320 ],
                 # skipped_layers = [0, 3, 6, 9, 10, 11, 13, 14, 15],
                 layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                            17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },
                 has_bias=True, use_attn_pooler=True,
                 attn_pooler_feat_reduction_ratio=8,
                 token_string="", placeholder_is_bg=False, 
                 # If do_zero_shot, then the basis vectors are generated by it 
                 # in a zero-shot fashion, instead of being learned from the data.
                 do_zero_shot=False, 
                 device_type="cuda"):
        super().__init__()

        self.do_zero_shot = do_zero_shot

        self.token_string = token_string
        assert num_layers == len(layer_idx2ca_layer_idx), f"num_layers={num_layers} != len(layer_idx2ca_layer_idx)={len(layer_idx2ca_layer_idx)}"
        self.num_layers = num_layers
        self.out_emb_dim = out_emb_dim
        self.K = num_vectors_per_subj_token
        self.fg_emb_count = fg_emb_count
        self.bg_emb_count = bg_emb_count
        assert fg_emb_count + bg_emb_count <= self.K, \
            f"fg_emb_count={fg_emb_count} + bg_emb_count={bg_emb_count} > num_vectors_per_subj_token={self.K}"

        # placeholder_is_bg: is this token trying to model the background?
        self.placeholder_is_bg    = placeholder_is_bg
        self.use_cached_bg  = use_cached_bg
        if self.use_cached_bg:
            self.cached_bg_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # Only takes bg or fg features but not both.
        self.is_fg_only = (fg_emb_count == self.K) 
        self.is_bg_only = (bg_emb_count == self.K)
        self.is_one_stream_only = self.is_fg_only or self.is_bg_only

        self.r = r
        self.use_attn_pooler = use_attn_pooler
        self.attn_pooler_feat_reduction_ratio = attn_pooler_feat_reduction_ratio

        # emb_infeat_types: 0 = fg, 1 = bg, 2 = fg_bg. 
        # Usually there are no type-2 (fg_bg) embeddings.
        self.emb_infeat_types = [ 0 ] * self.fg_emb_count + [ 1 ] * self.bg_emb_count \
                                + [ 2 ] * (self.K - self.fg_emb_count - self.bg_emb_count)

        self.device_type = device_type
        self.layer_idx2ca_layer_idx = layer_idx2ca_layer_idx
        # ca_layer_idx2layer_idx: Reverse mapping of layer_idx2ca_layer_idx.
        self.ca_layer_idx2layer_idx = { v: k for k, v in layer_idx2ca_layer_idx.items() }

        # Unless is_one_stream_only, then always use both fg and bg features as the input to the Linear, 
        # therefore the input feature dim is doubled.
        # But a particular Linear may only use either the fg or bg half of the features according to emb_infeat_types, 
        # so the other half of the weights will be masked.
        # If is_one_stream_only, then only use fg or bg features as the input to the Linear, and no weight masking is needed.
        self.H = H  = 1 if self.is_one_stream_only else 2
        # If all Linears only use half of the input features (effectively H=1, although nominally H=2), 
        # then only use 1/4 of the time embeddings. Otherwise, use 1/2 of the time embeddings.
        TIME_Hs = [ 2 if (self.use_attn_pooler and emb_infeat_type == 2) else 1 for emb_infeat_type in self.emb_infeat_types ]
        TIME_H = max(TIME_Hs)
        # The dimension of the time embeddings used will be 
        # the first TD_frac dimensions of image features.
        # Most infeat_dims: 1280, *0.25 = 320. 
        # 320 dim should contain enough info, and avoid
        # overweighing the image features.
        self.TD_frac = 0.25 * TIME_H

        if init_vecs is not None:
            if init_vecs.shape[1] != out_emb_dim or init_vecs.shape[0] > num_layers:
                raise ValueError(
                    f"AdaEmbedding LoRA init vectors shape {init_vecs.shape} must be (<={num_layers}, {out_emb_dim})"
                )

            # N: number of init vectors.
            N = self.N = init_vecs.shape[0]
            # pre_vecs: basis vectors that are initialized by prespecified init_vecs, updated through BP.
            # pre_vecs: [K, N, 768].
            if not self.do_zero_shot:
                self.pre_vecs = nn.Parameter(init_vecs.unsqueeze(0).repeat(self.K, 1, 1), requires_grad=True)
            else:
                self.pre_vecs = None
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined basis vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs, dim=-1)
        else:
            N = self.N = 0
            self.pre_vecs = None

        # basis_vecs: [K, r-N, 768], K sets, each consisting of r-N randomly initialized basis vectors. 
        # Will be updated through BP.
        # Each embedding of the K embeddings has its own basis_vecs and pre_vecs.
        # Separate pre_vecs and basis_vecs, to apply different regularizations on them.
        if not self.do_zero_shot:
            self.basis_vecs = nn.Parameter(torch.randn(self.K, r - N, out_emb_dim), requires_grad=True)
            # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
            self.basis_vecs.data = F.normalize(self.basis_vecs, dim=-1) / 4.
            # Always set the last basis vector to 0.
            self.basis_vecs.data[:, -1] = 0
        else:
            self.basis_vecs = None

        self.ca_infeat_dims = list(ca_infeat_dims)
        # self.infeat_dims = [ 320 for i in range(25) ]

        poolers = []
        for i in range(num_layers):
            infeat_dim = self.ca_infeat_dims[i]

            if self.use_attn_pooler:
                pooler = AttentionalPooler(i, infeat_dim, feat_reduction_ratio=self.attn_pooler_feat_reduction_ratio,
                                           infeat_grad_scale=1)
            else:
                pooler = MaskedAvgPool1d() #MaskedAvgPool2d()
            poolers.append(pooler)

        self.poolers = nn.ModuleList(poolers)

        layer_coeff_maps = []
        layers_out_lns  = []
        # LNCat3s: multiple cat(LN(infeat), LN(time_emb), LN(static_emb)).
        # If self.in_static_emb_dim == 0, then static_emb is not used as input to the Linear.
        layer_lncat3s = []
        self.TDs = []

        for i in range(num_layers):
            # Time embedding dimension is the same as the input feature dimension.
            # So TD = TD_frac * ca_infeat_dims[i].
            TD = int(self.TD_frac * self.ca_infeat_dims[i])
            self.TDs.append(TD)

            # input  dim: self.ca_infeat_dims[i] + TD, since first TD dims of time_emb is part of the input features.
            # output dim: r * K, will be reshaped to [K, r].
            # This Linear outputs K sets of r-dim vectors, 
            # each set being the coefficients of the r basis vectors. 
            layer_coeff_maps.append( nn.Linear(self.ca_infeat_dims[i] * H + TD, 
                                                r * self.K, bias=True) )
            layer_lncat3s.append(LNCat3(self.ca_infeat_dims[i] * H, TD))
            # A specific LayerNorm is applied on each of the K embeddings in each layer.
            layer_out_lns = nn.ModuleList( [ nn.LayerNorm(out_emb_dim, elementwise_affine=False) for k in range(self.K) ] )
            layers_out_lns.append(layer_out_lns)

        self.layer_coeff_maps   = nn.ModuleList(layer_coeff_maps)
        self.layers_out_lns     = nn.ModuleList(layers_out_lns)
        self.layer_lncat3s      = nn.ModuleList(layer_lncat3s)

        self.reduce_fg_bg_cross_weights()

        self.has_bias = has_bias
        if has_bias and not self.do_zero_shot:
            # self.bias: [16, K, 768].
            self.bias = nn.Parameter(torch.zeros(num_layers, self.K, out_emb_dim), requires_grad=True)
        else:
            self.bias = 0

        zero_shot_sig = "Zero-shot" if self.do_zero_shot else "Slow"
        print(f"{zero_shot_sig} AdaEmbedding {token_string} initialized with {fg_emb_count}/{bg_emb_count}/{self.K} fg/bg/total embs, {self.N} init vectors ({init_string}), {self.r} basis vectors")

        self.call_count = 0
        self.debug = False

    # If reduced_layer_idx is specified, then only mask for one layer. Otherwise, mask for all layers.
    # When generating ada embeddings for each layer in turn, only masking one layer will reduce processing time.
    def reduce_fg_bg_cross_weights(self, reduced_layer_idx=None):
        # If placeholder_is_bg: 
        # its "fg infeat" is the attn pooled infeat using the main embedding, in this case, the bg embedding.
        # Therefore, "fg infeat" is of the background.
        # "bg infeat" is the cached bg infeat produced by the previous fg embedder, so it's also bg infeat.
        # Therefore, no need to scale the weights.        
        #if self.placeholder_is_bg:
        #    return
        
        # Currently only supports H = 1 or 2.
        # Skip masking if is_one_stream_only.
        if self.H == 1:
            return

        assert self.H == 2

        layer_range = range(self.num_layers) if reduced_layer_idx is None else [reduced_layer_idx]
        cross_weight_max_ratio = 0.01

        for layer_idx in layer_range:
            SINGLE_D = self.ca_infeat_dims[layer_idx]
            TD       = self.TDs[layer_idx]
            assert self.layer_coeff_maps[layer_idx].in_features == SINGLE_D * 2 + TD
            layer_coeff_map_weight = self.layer_coeff_maps[layer_idx].weight.data
            # The weight of Linear has shape [out_features, in_features]. 
            # Split the first dim, out_features => [K, r].
            # layer_coeff_map_weight_embs: [K, r, in_features].
            layer_coeff_map_weight_embs = layer_coeff_map_weight.view(self.K, self.r, -1)

            for emb_idx in range(self.K):
                # layer_coeff_map_weight_emb: [r, in_features].
                layer_coeff_map_weight_emb = layer_coeff_map_weight_embs[emb_idx]
                # self.emb_infeat_types: [0, 0, 0, 0, 1, 1, 1, 1, 2]. 0 = fg, 1 = bg, 2 = fg_bg
                emb_infeat_type = self.emb_infeat_types[emb_idx]
                if emb_infeat_type == 0:
                    # layer_coeff_map_weight_emb: [r, in_features]. 
                    # fg_to_fg_coeff_mean_weight: the average weight of the mapping from fg infeat 
                    # to the r basis vector coeffs.
                    fg_to_fg_coeff_mean_weight  = layer_coeff_map_weight_emb[:, :SINGLE_D].abs().mean().item()
                    # bg_to_fg_coeff_mean_weight: the average weight of the mapping from bg infeat to 
                    # the r basis vector coeffs.
                    bg_to_fg_coeff_mean_weight  = layer_coeff_map_weight_emb[:, SINGLE_D:SINGLE_D*2].abs().mean().item()
                    # Scale down the weights from bg infeat to fg coeffs, so that this mean weight
                    # is at most cross_weight_max_ratio of the mean weight from fg infeat to fg coeffs.
                    b2f_down_scale  = min(1, cross_weight_max_ratio * fg_to_fg_coeff_mean_weight / (bg_to_fg_coeff_mean_weight + 1e-6))
                    layer_coeff_map_weight_emb[:, SINGLE_D:SINGLE_D*2] *= b2f_down_scale
                    #print(f"Layer {layer_idx} emb {emb_idx} fg_to_fg_coeff_mean_weight {fg_to_fg_coeff_mean_weight:.3f} bg_to_fg_coeff_mean_weight {bg_to_fg_coeff_mean_weight:.3f} b2f_down_scale {b2f_down_scale:.3f}")
                elif emb_infeat_type == 1:
                    # bg embeddings. Take full bg infeat and 0.3 of fg infeat as input.
                    bg_to_bg_coeff_mean_weight  = layer_coeff_map_weight_emb[:, SINGLE_D:SINGLE_D*2].abs().mean().item()
                    fg_to_bg_coeff_mean_weight  = layer_coeff_map_weight_emb[:, :SINGLE_D].abs().mean().item()
                    f2b_down_scale      = min(1, cross_weight_max_ratio * bg_to_bg_coeff_mean_weight / (fg_to_bg_coeff_mean_weight + 1e-6))
                    layer_coeff_map_weight_emb[:, :SINGLE_D] *= f2b_down_scale
                    #print(f"Layer {layer_idx} emb {emb_idx} bg_to_bg_coeff_mean_weight {bg_to_bg_coeff_mean_weight:.3f} fg_to_bg_coeff_mean_weight {fg_to_bg_coeff_mean_weight:.3f} f2b_down_scale {f2b_down_scale:.3f}")
                # Otherwise, emb_infeat_type == 2, no scaling is needed.

    # ca_infeat: 4D image feature tensor [B, C, H, W]. C: 320.
    # layer_idx: 0 ~ 24. ca_layer_idx: 0 ~ 15.
    # time_emb: [B, 1280].
    # zs_basis_vecs: [K, r, 768]. K: number of vectors per token. r: number of basis vectors for each vector.                    
    def forward(self, layer_idx, layer_attn_components, time_emb, 
                layer_subj_emb_probe, layer_static_extra_emb_mean, 
                img_mask=None, cached_pooler_bg_out=None, 
                zs_basis_vecs=None, zs_bias=None):
        ca_layer_idx = self.layer_idx2ca_layer_idx[layer_idx]
        pooler  = self.poolers[ca_layer_idx]
        ## Some Linears mainly use either fg or bg features. So we reduce cross weights.
        #if self.training:
        #    self.reduce_fg_bg_cross_weights(ca_layer_idx)

        if not self.is_fg_only and self.use_cached_bg:
            # cached_pooler_bg_out must be provided when use_cached_bg.
            if cached_pooler_bg_out is None:
                breakpoint()
        
        if self.debug:
            breakpoint()

        with torch.autocast(device_type=self.device_type, enabled=True):
            # Even if ca_infeat is grad-scaled, the pooler still receives the full gradient.
            # But the grad is scaled when it's further passed to the UNet.

            # bg token and use_cached_bg=True.
            # But by default, the bg token uses 2/3 of the bg features, and 1/3 of the fg features. 
            # But if use_cached_bg=True, then the 1/3 of the fg features are replaced 
            # by the cached bg features.
            cached_bg_used = False
            if self.use_attn_pooler and self.use_cached_bg:
                infeat_bg     = cached_pooler_bg_out
                cached_bg_used = True
            if not (self.is_bg_only and cached_bg_used):
                # Either cached_bg_used, or not is_bg_only.
                # In either case, we need to get features using pooler.
                # layer_subj_emb_probe should be quite similar to the ada embedding at this layer.
                # So we use layer_subj_emb_probe as an approximate query to do the attention-based pooling.
                # layer_subj_emb_probe: [768]. layer_static_extra_emb_mean: [2, 768].
                infeat_pooled_dict  = pooler(layer_attn_components, 
                                             fg_q_emb=layer_subj_emb_probe, 
                                             bg_q_emb=layer_static_extra_emb_mean,
                                             img_mask=img_mask,
                                             debug=self.debug)
 
            if self.use_attn_pooler:
                # infeat_fg, infeat_bg: [2, 320]
                infeat_fg = infeat_pooled_dict['fg_out']
                if not cached_bg_used:
                    infeat_bg = infeat_pooled_dict['bg_out']
                # infeat_fg won't be used in future calls, so it is released now.
                # But infeat_pooled_dict['bg_out'] may be cached and used in future calls.
                del infeat_pooled_dict['fg_out']

                # infeat_fg_bg: [2, 640]
                infeat_fg_bg = torch.cat([infeat_fg, infeat_bg], dim=-1)
            else:
                # Since not use_attn_pooler, always cached_bg_used = False. 
                # So we always use freshly pooled features. 
                # In this case, infeat_pooled_dict is a tensor.
                infeat_fg_bg = infeat_pooled_dict

            # time_emb has a fixed dimension of 1280. But infeat has variable dimensions.
            # Only use the first TD dimensions of the time embedding, 
            # as the time embedding is highly redundant, and the first TD dimensions are sufficient
            # to capture the temporal information.
            # Note to take the first TD dimensions, instead of the last TD dimensions,
            # as the leading dimensions are most sensitive to time change, 
            # and the last dimensions tend to be the same for all time steps.
            # TD is typically C_layer/4, so that the time embeddings won't dominate 
            # the image features infeat_fg_bg.
            TD = self.TDs[ca_layer_idx]
            
            time_feat = time_emb[:, :TD]
            ablate_time = False
            if ablate_time:
                time_feat = torch.zeros_like(time_feat)

            # infeat_time_emb: cat(ln(infeat_fg_bg), ln(time_emb)) as the input features.
            infeat_time_emb    = self.layer_lncat3s[ca_layer_idx](infeat_fg_bg, time_feat)
            # basis_dyn_coeffs: [BS, r*K] => [BS, K, r].
            # Consider the last dim. 
            basis_dyn_coeffs = self.layer_coeff_maps[ca_layer_idx](infeat_time_emb).reshape(-1, self.K, self.r)

            if self.do_zero_shot:
                assert zs_basis_vecs is not None, "Zero-shot AdaEmbedding requires zs_basis_vecs"
                basis_vecs = zs_basis_vecs
                # Copy to basis_vecs, so that zs_basis_vecs will be regularized in layerwise_embedding_norm_loss().
                # Although self.basis_vecs is supposed to have different sizes from basis_vecs (due to pre_vecs),
                # since we can't regularize pre_vecs in zero-shot setting, we copy the whole zs_basis_vecs to self.basis_vecs,
                # so that all vectors in zs_basis_vecs will be regularized.
                self.basis_vecs = zs_basis_vecs
                if self.has_bias:
                    assert zs_bias is not None,   "Zero-shot AdaEmbedding requires zs_bias"
                    # Copy to bias, so that zs_bias will be regularized in layerwise_embedding_norm_loss().
                    self.bias = zs_bias                
            else:
                # self.N: number of pre_vecs.
                if self.N > 0:
                    # pre_vecs:   [K, N, 768], basis_vecs: [K, r - N, 768]. 
                    # basis_vecs: [K, r, 768].
                    basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=1)
                else:
                    basis_vecs = self.basis_vecs

            out_lns = self.layers_out_lns[ca_layer_idx]
            # out_vecs_unnorm: K elements, each is [BS, 1, r] x [r, 768]_k = [BS, 1, 768].
            out_vecs_unnorm = [ torch.matmul(basis_dyn_coeffs[:, k], basis_vecs[k]) for k in range(self.K) ]

            out_vecs0 = torch.stack([ out_lns[k](out_vecs_unnorm[k]) for k in range(self.K) ], dim=1)
            # out_emb_dim: 768.
            out_vecs0 = out_vecs0 / np.sqrt(self.out_emb_dim)

            # bias: [1, K, 768]
            bias = self.bias[ca_layer_idx].unsqueeze(0)
            # [BS, K, 768] + [1, K, 768] = [BS, K, 768].
            out_vecs  = out_vecs0 + bias

            if 'call_count' not in self.__dict__:
                self.call_count = 0

            self.verbose = False
            if self.verbose and self.call_count % 10 == 0:
                calc_stats(f'{ca_layer_idx} time_emb', time_emb[:, :TD])
                calc_stats(f'{ca_layer_idx} infeat_fg_bg', infeat_fg_bg)
                calc_stats(f'{ca_layer_idx} basis_dyn_coeffs', basis_dyn_coeffs)
                calc_stats(f'{ca_layer_idx} out_vecs0', out_vecs0)
                calc_stats(f'{ca_layer_idx} bias', bias)

            if ca_layer_idx == 24:
                self.call_count += 1

        # Return infeat_pooled_dict to be used by another ada_embedder that specializes on the background.
        return out_vecs, infeat_pooled_dict


# Initialize static/ada embedders.
def create_static_ada_embedders(out_emb_dim, num_layers_per_embedder, num_vectors_per_subj_token, layerwise_lora_rank, 
                                initializer_string, init_word_embeddings, init_word_weights, 
                                placeholder_string, avg_init_word_embedding_3d, placeholder_is_bg, ada_uses_attn_pooler,
                                attn_pooler_feat_reduction_ratio, do_zero_shot):
    # A static/ada embedder can generate K embeddings.
    # layerwise_lora_rank > 0 implies use_layerwise_embedding.
    if layerwise_lora_rank > 0:
        # if self.emb_ema_as_pooling_probe_weight > 0, then calculate EMA of embeddings 
        # to be used in Ada attn pooler.
        # num_layers_per_embedder = num_unet_ca_layers
        token_static_embedder   = StaticLayerwiseEmbedding(num_layers_per_embedder, 
                                                            num_vectors_per_subj_token, 
                                                            out_emb_dim, 
                                                            layerwise_lora_rank, 
                                                            (0.1, 0.02), 
                                                            initializer_string,
                                                            init_word_embeddings, init_word_weights, 
                                                            token_string=placeholder_string,
                                                            do_zero_shot=do_zero_shot)
            
        # For subject embeddings:    2/3 of the embeddings are fg embeddings (focus on fg infeat), 
        # and 1/3 are bg embeddings (focus on bg infeat).
        # For background embeddings: 2/3 of the embeddings are bg embeddings (focus on bg infeat), 
        # and 1/3 are fg embeddings (focus on fg infeat).
        # Note fg embeddings still take 0.3 of bg infeat, and bg embeddings still take 0.3 of fg infeat.
        # No embeddings are fg-bg embeddings, which take fg and bg infeat with equal weights.
        # If num_vectors_per_subj_token == 1, then fg_emb_count = 1, bg_emb_count = 0.
        # If num_vectors_per_subj_token == 9, then fg_emb_count = 6, bg_emb_count = 3.
        if placeholder_is_bg:
            bg_emb_count = max(1, num_vectors_per_subj_token * 2 // 3)
            fg_emb_count = num_vectors_per_subj_token - bg_emb_count
        else:
            fg_emb_count = max(1, num_vectors_per_subj_token * 2 // 3)
            bg_emb_count = num_vectors_per_subj_token - fg_emb_count

        use_cached_bg = placeholder_is_bg

        token_ada_embedder  = AdaEmbedding(num_layers_per_embedder, 
                                            num_vectors_per_subj_token, 
                                            fg_emb_count, 
                                            bg_emb_count,
                                            use_cached_bg,
                                            out_emb_dim,                                                    
                                            layerwise_lora_rank, 
                                            initializer_string,
                                            init_word_embeddings,
                                            use_attn_pooler=ada_uses_attn_pooler,
                                            attn_pooler_feat_reduction_ratio=attn_pooler_feat_reduction_ratio,
                                            token_string=placeholder_string,
                                            placeholder_is_bg=placeholder_is_bg,
                                            do_zero_shot=do_zero_shot)
    else:
        # Degenerate to Textual Inversion. 
        # ANCHOR[id=init_embed] : 16*K vectors are initialized with the same embedding.
        token_static_embedder   = nn.Parameter(avg_init_word_embedding_3d, requires_grad=True)
        token_ada_embedder      = None
        print("Warning: Degenerate to Textual Inversion. No AdaEmbedding is created.")

    return token_static_embedder, token_ada_embedder

# text_embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
# = LatentDiffusion.cond_stage_model
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            text_embedder,              
            subject_strings,
            # If background_strings are specified, they are part of the list placeholder_strings.
            background_strings=None,
            initializer_strings=None,
            list_initializer_word_weights=None,
            subj_name_to_cls_delta_strings=None,
            subj_name_to_cls_delta_word_weights=None,
            # token2num_vectors: how many vectors in each layer are allocated to model 
            # the subject (represented as the subject token) and the background. 
            # token2num_vectors is a dict.
            token2num_vectors={},
            use_layerwise_embedding=True,
            out_emb_dim=768,
            num_unet_ca_layers=16,
            layerwise_lora_rank=10,
            layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            ada_emb_weight=0.5, 
            ada_uses_attn_pooler=True,
            attn_pooler_feat_reduction_ratio=8,
            emb_ema_as_pooling_probe_weight=0,
            training_begin_add_noise_std_range=None,
            training_end_add_noise_std_range=None,
            training_add_noise_prob=None,
            use_conv_attn_kernel_size=-1,
            conv_attn_layerwise_scale_learnable=False,
            prompt_embedding_clamp_value=-1,
            background_extra_global_scale=1.,
            emb_reg_loss_scale=1,
            shared_placeholder_set='subj,bg',
            shared_embedder_components='pooler',
            do_zero_shot=False,
            zs_image_emb_dim=1280,
            zs_use_face_embs=False,
            # A few args, like embedding_manager_ckpt, ckpt_params_perturb_ratio, 
            # are used in ddpm.py, but ignored here.
            **kwargs
    ):
        super().__init__()

        self.do_zero_shot = do_zero_shot

        self.string_to_token_dict = OrderedDict()
        
        self.string_to_static_embedder_dict = nn.ParameterDict()
        self.string_to_ada_embedder_dict    = nn.ModuleDict()
        self.string_to_emb_ema_dict         = nn.ModuleDict()
        self.initial_embeddings             = nn.ParameterDict() # These should not be optimized
        self.placeholder_to_emb_cache       = nn.ParameterDict() # These should not be optimized

        self.set_ada_emb_weight(ada_emb_weight, is_first_time_print=True)
        self.ada_uses_attn_pooler = ada_uses_attn_pooler
        # If self.do_zero_shot, then disable emb_ema_as_pooling_probe. 
        # Otherwise, emb_ema_as_pooling_probe_weight = 0.5.
        if self.do_zero_shot:
            emb_ema_as_pooling_probe_weight = 0
        self.set_emb_ema_as_pooling_probe_weight(emb_ema_as_pooling_probe_weight)

        self.emb_ema_grad_scale = 0.05
        self.emb_ema_grad_scaler = gen_gradient_scaler(self.emb_ema_grad_scale)

        self.use_layerwise_embedding = use_layerwise_embedding
        self.num_unet_ca_layers = num_unet_ca_layers
        if self.use_layerwise_embedding:
            # self.num_layers_per_embedder specifies the total layers of embeddings for each embedder.
            # There could be multiple embeddings for each layer.
            self.num_layers_per_embedder = num_unet_ca_layers
        else:
            self.num_layers_per_embedder = 1

        self.subject_strings = subject_strings
        if background_strings is not None:
            self.background_strings = list(background_strings)
        else:
            self.background_strings = []

        self.background_string_dict = { s: True for s in self.background_strings }
        self.placeholder_strings    = list(subject_strings) + self.background_strings
        self.subject_string_dict    = { s: True for s in self.subject_strings }

        # Model should be still on CPU. So no need to consider the device when extending the text embedder.
        # Extend CLIP text embedder with the subject strings / background strings / wds background strings.
        # In order to load the text embedder ckpt, the text_model.embeddings.token_embedding hasn't been
        # extended yet. So we  save extended_token_embeddings to be extended to text_model.embeddings.token_embedding
        # later in main.py.
        self.extended_token_embeddings = extend_clip_text_embedder(text_embedder, {}, self.placeholder_strings)


        # Each placeholder string has a corresponding emb_global_scale_score, 
        # converted to emb_global_scale.
        self.emb_global_scale_scores = nn.Parameter(torch.zeros(len(self.placeholder_strings)), 
                                                    requires_grad=True)
        self.subj2conv_attn_layerwise_scales = nn.ParameterDict()
        self.initialize_subj2conv_attn_layerwise_scales(1, learnable=conv_attn_layerwise_scale_learnable)
        
        self.set_training_add_noise_specs(training_begin_add_noise_std_range, 
                                          training_end_add_noise_std_range,
                                          training_add_noise_prob)
        self.set_embs_attn_tricks(use_conv_attn_kernel_size)
        self.set_attn_pooler_feat_reduction_ratio(attn_pooler_feat_reduction_ratio)

        self.layer_idx2ca_layer_idx = layer_idx2ca_layer_idx
        self.ca_layer_idx2layer_idx = { v: k for k, v in layer_idx2ca_layer_idx.items() }

        #                        1     2     4    5     7     8     12    16    
        self.ca_infeat_dims =  [ 320,  320,  640, 640, 1280, 1280, 1280, 1280, 
        #                        17    18    19   20    21    22    23    24                       
                                 1280, 1280, 640, 640, 640,  320,  320,  320 ]
        # ca_outfeat_dims are the same as ca_infeat_dims.

        # token2num_vectors: a dict. How many vectors in each layer 
        # are allocated to model the subject (represented as the subject token).        
        # token2num_vectors[S*] > 1:
        # *multi-vector subject embeddings*. In this space, S* is embedded into multiple 
        # learned embeddings, an approach that is equivalent to describing
        # the concept through multiple learned pseudo-words. 
        # This setting was proposed in the TI paper,
        # and AdaPrompt also supports it for more expressive modeling.
        self.set_num_vectors_per_subj_token(token2num_vectors)
        self.out_emb_dim = out_emb_dim

        # hasattr(text_embedder, 'tokenizer') -> True
        if hasattr(text_embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            get_tokens_for_string       = partial(get_clip_tokens_for_string,       text_embedder.tokenizer)
            get_embeddings_for_tokens   = partial(get_embeddings_for_clip_tokens,   text_embedder.transformer.text_model.embeddings)
        else:
            breakpoint()

        # Save this function to be used in load() when doing placeholder substitution.
        self.get_tokens_for_string = get_tokens_for_string
        str2lora_rank = {}
        # "," -> 267, "z": 345, "y": 344.
        self.subj_idx_to_cls_delta_tokens   = {}
        self.subj_idx_to_cls_delta_token_weights  = {}
        self.placeholder_token_to_idx       = {}

        assert initializer_strings is not None, "initializer_strings must be specified"
        if list_initializer_word_weights is None:
            list_initializer_word_weights = [ None ] * len(self.placeholder_strings)
        self.list_initializer_word_weights   = list_initializer_word_weights
        self.layerwise_lora_rank = layerwise_lora_rank

        for placeholder_idx, placeholder_string in enumerate(self.placeholder_strings):
            placeholder_is_bg =  (placeholder_string in self.background_string_dict)
            # get_tokens_for_string <= get_clip_tokens_for_string.
            # force_single_token = True, as there should be only one token in placeholder_string.
            placeholder_token = get_tokens_for_string(placeholder_string, force_single_token=True)[0].item()

            num_vectors_per_subj_token = self.token2num_vectors.get(placeholder_string, 1)
            initializer_string     = initializer_strings[placeholder_idx]
            initializer_word_weights   = list_initializer_word_weights[placeholder_idx]

            # The background token may not have initializer words. So its corresponding
            # init_word_embeddings, avg_init_word_embedding, init_word_weights are None.
            try:
                init_word_tokens, init_word_weights, init_word_embeddings, avg_init_word_embedding = \
                calc_init_word_embeddings(get_tokens_for_string, get_embeddings_for_tokens,
                                          initializer_string, initializer_word_weights)
            except:
                breakpoint()

            str2lora_rank[placeholder_string] = layerwise_lora_rank
            self.string_to_token_dict[placeholder_string] = placeholder_token
            # initial_embeddings are only used to compute the regularization loss.
            # Wrap with Parameter so that they will be saved to checkpoints.
            # avg_init_word_embedding_3d: [1, 768] => [16, 9, 768]
            if avg_init_word_embedding is not None:
                avg_init_word_embedding_3d = avg_init_word_embedding.unsqueeze(0).repeat(self.num_layers_per_embedder, num_vectors_per_subj_token, 1)
            else:
                # Use zero tensor as avg_init_word_embedding_3d.
                avg_init_word_embedding_3d = torch.zeros(self.num_layers_per_embedder, num_vectors_per_subj_token, 768)

            token_static_embedder, token_ada_embedder = \
                create_static_ada_embedders(out_emb_dim, self.num_unet_ca_layers, num_vectors_per_subj_token, layerwise_lora_rank, 
                                            initializer_string, init_word_embeddings, init_word_weights, 
                                            placeholder_string, avg_init_word_embedding_3d, placeholder_is_bg, ada_uses_attn_pooler,
                                            attn_pooler_feat_reduction_ratio, do_zero_shot)

            # avg_init_word_embedding_3d: [16, 9, 768]. 
            # All layers of all 9 embeddings are initialized as avg_init_word_embedding.            
            if self.emb_ema_as_pooling_probe_weight > 0:
                # token_emb is the embedding in the prompt embeddings, not the token embeddings
                # generated by StaticLayerwiseEmbedding / AdaEmbedding.
                # Cannot initialize token_emb with avg_init_word_embedding, as avg_init_word_embedding
                # is the token embedding, not the prompt embedding.
                token_emb_cache = Embedding3d(self.num_layers_per_embedder, num_vectors_per_subj_token, out_emb_dim, 
                                              init_embedding=None)
                self.placeholder_to_emb_cache[placeholder_string] = token_emb_cache

            # token_static_embedder: a StaticLayerwiseEmbedding object (when use_layerwise_embedding) or an embedding vector.
            # Pytorch >= 1.12.0 allows to put an nn.Module object into an nn.ParameterDict.
            self.string_to_static_embedder_dict[placeholder_string] = token_static_embedder
            self.string_to_ada_embedder_dict[placeholder_string]    = token_ada_embedder
            self.string_to_emb_ema_dict[placeholder_string]         = None

            if init_word_embeddings is not None:
                # initial_embeddings won't be optimized. Just used to compute the regularization loss.
                # Use nn.Parameter to put it on cuda. Not to use register_buffer(), since it's a dict.
                self.initial_embeddings[placeholder_string] = nn.Parameter(init_word_embeddings, requires_grad=False)
            else:
                self.initial_embeddings[placeholder_string] = None

        # Initialize self.subj_name_to_cls_delta_tokens and self.subj_name_to_cls_delta_token_weights.
        self.init_cls_delta_tokens(get_tokens_for_string, get_embeddings_for_tokens, 
                                   subj_name_to_cls_delta_strings, subj_name_to_cls_delta_word_weights)
        self.share_embedder_components(shared_placeholder_set, shared_embedder_components)

        self.layer_idx = -1
        self.static_subj_embs_dict = {}   
        self.ada_subj_embs_dict    = {}
        self.ada_subj_attn_dict    = {}
        self.clear_ada_layer_temp_info()
        self.clear_prompt_adhoc_info()
        self.curr_batch_subj_names = []
        self.img_mask = None
        self.loss_call_count = 0
        self.training_percent = 0
        # Store the text_embedder to compute the delta loss.
        self.text_embedder  = text_embedder
        self.ada_prompt_embeddings_cache    = {}
        self.ada_prompt_placeholder2indices_cache = {}
        self.emb_global_scales_dict = None
        self.iter_type = None       # 'recon_iter' or 'distill_iter'
        self.prompt_embedding_clamp_value  = prompt_embedding_clamp_value
        self.background_extra_global_scale = background_extra_global_scale
        self.emb_reg_loss_scale = emb_reg_loss_scale
        # ca_q_bns and ca_outfeat_lns are used to normalize the q/out features
        # in loss computation in ddpm.py, and not used in this script.
        ca_q_bns = {}
        ca_outfeat_lns = {}
        for ca_layer_idx in range(self.num_unet_ca_layers):
            layer_idx = self.ca_layer_idx2layer_idx[ca_layer_idx]
            ca_q_bns[str(layer_idx)]       = nn.BatchNorm2d(self.ca_infeat_dims[ca_layer_idx], affine=False)
            ca_outfeat_lns[str(layer_idx)] = nn.LayerNorm(self.ca_infeat_dims[ca_layer_idx], elementwise_affine=False)
            #print(layer_idx, self.ca_infeat_dims[ca_layer_idx])

        self.ca_q_bns       = nn.ModuleDict(ca_q_bns)
        self.ca_outfeat_lns = nn.ModuleDict(ca_outfeat_lns)

        # zs_image_feat_dict have three keys: 'subj', 'bg', 'face'.
        self.zs_image_feat_dict = {}
        if self.do_zero_shot:
            # No matter whether using layerwise embeddings, the basis vecs of either static or ada embedders are always layerwise_lora_rank,
            # as the basis vecs are shared across all CA layers.
            # But different vectors of the same embeddings are combinations of different basis vecs.
            # Therefore, num_total_basis_vecs is multiplied by num_vectors_each_subj_bg_pair.
            # num_vectors_each_subj_bg_pair: the number of vectors per (subj, bg) placeholder pair.
            # It's implied that all subj placeholders have the same number of vectors,
            # and all bg placeholders have the same number of vectors.
            self.number_vectors_each_subj = self.token2num_vectors.get(self.subject_strings[0], 9)
            if len(self.background_strings) > 0:
                self.num_vectors_each_bg = self.token2num_vectors.get(self.background_strings[0], 4)
            else:
                self.num_vectors_each_bg = 0

            # num_zs_vecs_per_token: 10 + 16 + 16 = 42.
            # 10: 10 basis vecs for ada embedder. 16: 16 layerwise biases for ada embedder. 
            # Another 16: 16 layerwise static embeddings.
            self.num_zs_vecs_per_token = layerwise_lora_rank + self.num_unet_ca_layers * 2
            # num_subj_queries: 9 * 42 = 378.
            self.num_zs_vecs_per_subj  = self.number_vectors_each_subj * self.num_zs_vecs_per_token
            # num_bg_queries: 4 * 42 = 168.
            self.num_zs_vecs_per_bg    = self.num_vectors_each_bg * self.num_zs_vecs_per_token
            self.subj_basis_generator = SubjBasisGenerator(num_subj_queries = self.num_zs_vecs_per_subj,
                                                           num_bg_queries   = self.num_zs_vecs_per_bg,
                                                           # zs_image_emb_dim: laion: 1280, openai: 768.
                                                           image_embedding_dim = zs_image_emb_dim, 
                                                           dim = out_emb_dim,
                                                           output_dim = out_emb_dim,
                                                           use_face_embs=zs_use_face_embs)

        else:
            self.subj_basis_generator = None

        print("EmbeddingManager on subj={}, bg={} init with {} vec(s), layerwise_lora_rank={}, ada_emb_weight={}".format(
               self.subject_strings, self.background_strings, self.token2num_vectors, str2lora_rank, 
               ada_emb_weight))
        
        # Add the search span by 1, just to be safe.
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN += 1
        print(f"CLS_DELTA_STRING_MAX_SEARCH_SPAN={self.CLS_DELTA_STRING_MAX_SEARCH_SPAN}")

    def init_cls_delta_tokens(self, get_tokens_for_string, get_embeddings_for_tokens, 
                              subj_name_to_cls_delta_strings, subj_name_to_cls_delta_word_weights):
        self.subj_name_to_cls_delta_tokens  = {}
        self.subj_name_to_cls_delta_token_weights = {}
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN = 0

        if subj_name_to_cls_delta_strings is None:
            return

        # subj_name_to_cls_delta_word_weights is of type omegaconf. If without convertion to dict,
        # "subj_name_to_cls_delta_token_weights[subj_name] = cls_delta_token_weights" will throw an error.
        self.subj_name_to_cls_delta_token_weights = dict(subj_name_to_cls_delta_word_weights)

        for subj_name in subj_name_to_cls_delta_strings:
            cls_delta_string = subj_name_to_cls_delta_strings[subj_name]
            cls_delta_token_weights = subj_name_to_cls_delta_word_weights[subj_name]
            cls_delta_tokens, cls_delta_token_weights, _, _ = \
                calc_init_word_embeddings(get_tokens_for_string, get_embeddings_for_tokens,
                                          cls_delta_string, cls_delta_token_weights)

            num_cls_delta_tokens = len(cls_delta_tokens)
            if len(cls_delta_token_weights) != num_cls_delta_tokens:
                # BUG: some rare words may be split into two tokens. But this should be extremely rare.
                # Any common words will be mapped to one token only.
                breakpoint()
            
            # subj_idx_to_cls_delta_tokens is used to examine class prompts, 
            # to see if there are subsequences of cls_delta_tokens.
            # If there are, the embeddings of init_word_tokens should be combined through weighted sum.
            self.subj_name_to_cls_delta_tokens[subj_name]        = cls_delta_tokens
            self.subj_name_to_cls_delta_token_weights[subj_name] = cls_delta_token_weights

            # CLS_DELTA_STRING_MAX_SEARCH_SPAN should be the max number of extra tokens
            # (all excluding the first of the init word tokens; the first corresponds to the subject token).
            # If multiple subject strings appear in the same prompt, then CLS_DELTA_STRING_MAX_SEARCH_SPAN 
            # should be multiplied by the number of subject strings. Currently not implemented.
            if num_cls_delta_tokens - 1 > self.CLS_DELTA_STRING_MAX_SEARCH_SPAN:
                self.CLS_DELTA_STRING_MAX_SEARCH_SPAN = num_cls_delta_tokens - 1

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # If self.use_layerwise_embedding, then each token expands to num_unet_ca_layers = 16 
    # layerwise embeddings.
    def forward(
            self,
            tokenized_text,         # [B, N]. 
            embedded_text,          # [B, N, 768]. 
    ):
        # When delta loss is used, B is not batch_size, but batch_size * 4 * num_compositions_per_image.
        # If bs=2, num_compositions_per_image=2, then B=16.
        # In the iterations when ada delta loss is enabled, in effect num_compositions_per_image is 1, 
        # even if it's specified as 2, so B=8.
        B, N, device = *tokenized_text.shape, tokenized_text.device

        # gen_ada_embedding is dynamically switched on/off by cache_layer_features_for_ada()/clear_ada_layer_temp_info().
        # No need to calculate prompt_emb_mask here, as the mask for ada embeddings is 
        # the same as for the static embeddings. 
        # AdaPrompt combines static and ada embeddings. So the static embedding replacement 
        # code below is always called, and prompt_emb_mask is always calculated.
        if self.gen_ada_embedding:
            # self.layer_idx, self.ca_infeat, self.time_emb were cached by 
            # a previous call of  cache_layer_features_for_ada() from UNet.
            ada_embedded_text, ada_subj_embs_dict, token2ada_attn = \
                self.get_ada_embedding(self.layer_idx, self.layer_attn_components, self.time_emb,
                                       tokenized_text, embedded_text)

            # Cache the ada embeddings to be used in embedding orthogonal loss later.
            for k in ada_subj_embs_dict:
                ca_layer_idx = self.layer_idx2ca_layer_idx[self.layer_idx]
                self.ada_subj_embs_dict[k].cache_layer(ca_layer_idx, 
                                                       ada_subj_embs_dict[k], 
                                                       has_grad=True)

                if k in token2ada_attn:
                    self.ada_subj_attn_dict[k] = token2ada_attn[k]

            # Release ada-specific intermediate variables.
            self.clear_ada_layer_temp_info()
                        
            # No prompt repeating happened in get_ada_embedding(), 
            # so pass the original tokenized_text as tokenized_text_repeated.
            self.update_prompt_masks(tokenized_text, tokenized_text)
            return ada_embedded_text

        else:
            # placeholder_indices will be regenerated with update_placeholder_indices() 
            # within get_static_embedding().
            self.clear_prompt_adhoc_info()
            
            # We need to clone embedded_text, as sometimes (when it's not layerwise, such as TI) 
            # the modification in get_static_embedding() is in-place. 
            # The keys of static_subj_embs_dict are the placeholder strings.
            static_embeded_text, tokenized_text_repeated, static_subj_embs_dict = \
                            self.get_static_embedding(tokenized_text, embedded_text.clone(), 
                                                      self.zs_image_feat_dict,
                                                      self.string_to_static_embedder_dict,
                                                      B, N, self.num_unet_ca_layers, device)
            # Cache the static embeddings to be used in ada embedding computation and
            # embedding orthogonal loss later.
            self.static_subj_embs_dict = {}
            self.ada_subj_embs_dict = {}
            self.ada_subj_attn_dict = {}

            for k in static_subj_embs_dict:
                self.static_subj_embs_dict[k] = static_subj_embs_dict[k]
                # Initialize the ada token embedding cache for each token.
                # k is the placeholder string.
                self.ada_subj_embs_dict[k]    = Embedding3d(self.num_unet_ca_layers, 
                                                            self.token2num_vectors[k], 
                                                            self.out_emb_dim)
                self.ada_subj_embs_dict[k].to(device)

            # Update the prompt token embedding mask.
            # tokenized_text_repeated is repeated 16 times along the batch dimension.
            self.update_prompt_masks(tokenized_text, tokenized_text_repeated)

            return static_embeded_text
    
    # N: length of sequence (including padding).
    def get_static_embedding(self, tokenized_text, embedded_text, zs_image_feat_dict, embedder_dict, 
                             BS, N, num_unet_ca_layers, device):
        orig_tokenized_text = tokenized_text
        static_subj_embs_dict = {}
        self.cls_delta_string_indices = []
        prompt_subj_name_to_cls_delta_tokens = { subj_name: self.subj_name_to_cls_delta_tokens[subj_name] \
                                                 for subj_name in self.curr_batch_subj_names }

        if self.use_layerwise_embedding:
            # embedded_text: [B, N, 768] => [B, 16, N, 768] => [16*B, N, 768].
            # "Tuck" the layer dimension into the batch dimension, 
            # to keep embedded_text in 3D, same as the input.
            # After repeat, the same instance is repeated 16 times, which are adjacent 
            # to each other across the batch dim:
            # [b1_l1, ..., b1_l16, b2_l1, ..., b2_l16, ..., bB_l1, ..., bB_l16].
            # {________b1________} {_______b2_______}  ...  {_______bB________}
            embedded_text = embedded_text.unsqueeze(1).repeat(1, num_unet_ca_layers, 1, 1).view(BS * num_unet_ca_layers, N, -1)
            # tokenized_text: [B, 16, N] => [16*B, N]
            # tokenized_text has to be repeated along the layer dimension as well, so that 
            # placeholder_indices can index the embedding at each layer in the batch.
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1, num_unet_ca_layers, 1).view(BS * num_unet_ca_layers, N)
            # mirror-reflect the embedding along the layer dimension, to make it symmetric 
            # in the encoder & decoder.

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            # If there's only one vector per token, we can do a simple replacement
            placeholder_indices = torch.where(tokenized_text == placeholder_token)
            # No placeholder token is found in the current batch.
            if placeholder_indices[0].numel() == 0:
                continue
            
            placeholder_is_bg = (placeholder_string in self.background_string_dict)
            # If multiple occurrences are found in a prompt, only keep the first as the subject.
            # Other occurrences are treated as part of the background prompt (this may happen if
            # composition image overlay is used).
            placeholder_indices_1st = extract_first_index_in_each_instance(placeholder_indices)
            # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
            # Non-layerwise: embedded_text[placeholder_indices]: [2, 768].  subj_static_embedding: [1, K, 768].
            # layerwise: placeholder_indices =  
            # (tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 
            #  tensor([ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]))
            # embedded_text[placeholder_indices]: [32, 768]. subj_static_embedding: [16, K, 768].
            # The first 16 elements (0-8) in embedded_text[placeholder_indices] correspond to the 16 layers of the 
            # first instance in the batch.
            # 16 layers of subj_static_embedding are repeated REAL_OCCURS_IN_BATCH times.
            # subj_static_embedding: [16, 768] repeat=> [32, 768]
            # LINK #init_embed

            # REAL_OCCURS_IN_BATCH: the real number of occurrences of the placeholder in the current batch,
            # not repetitively counting the occurrences in the embedded_text repeated for M layers.
            REAL_OCCURS_IN_BATCH = placeholder_indices_1st[0].numel() // self.num_layers_per_embedder
            # Some prompts don't contain the placeholder token. This could happen in a compositional 
            # distillation iteration, or an inference iteration. 
            # Anyway, we need to check whether the cls_delta_string
            # occurs in the prompts without the placeholder token. If so, we need to merge 
            # their embeddings to one (the first) embedding, and delete the 2nd to the last embeddings,
            # using merge_cls_token_embeddings().
            if REAL_OCCURS_IN_BATCH < BS and self.CLS_DELTA_STRING_MAX_SEARCH_SPAN > 0 \
              and len(prompt_subj_name_to_cls_delta_tokens) > 0:
                cls_delta_string_indices = scan_cls_delta_strings(tokenized_text,
                                                                  placeholder_indices_1st,
                                                                  prompt_subj_name_to_cls_delta_tokens,
                                                                  self.CLS_DELTA_STRING_MAX_SEARCH_SPAN)
                # cls_delta_string_indices is a list of tuples, each tuple is 
                # (batch_i, start_N, num_cls_delta_tokens, placeholder_token).
                self.cls_delta_string_indices += cls_delta_string_indices

            static_embedder = embedder_dict[placeholder_string].to(device)
            if isinstance(static_embedder, StaticLayerwiseEmbedding):
                # Generate the actual subj_static_embedding on the fly.
                # The 16 static subject embeddings are formed by linearly combining the basis vectors.
                # The matrix operations are done on the fly.
                # subj_static_embedding: [16, K, 768].
                if self.do_zero_shot:
                    if placeholder_is_bg:
                        zs_clip_features = zs_image_feat_dict['bg']
                        num_vectors_each_placeholder = self.num_vectors_each_bg
                    else:
                        zs_clip_features = zs_image_feat_dict['subj']
                        num_vectors_each_placeholder = self.number_vectors_each_subj

                    zs_face_embs = zs_image_feat_dict['face']

                    # zs_clip_features: [1, 257, 1280]
                    # zs_vecs_2sets: [1, 468, 768] -> [9, 52, 768]
                    zs_vecs_2sets = self.subj_basis_generator(zs_clip_features, zs_face_embs, placeholder_is_bg)
                    zs_vecs_2sets = zs_vecs_2sets.reshape(num_vectors_each_placeholder,
                                                          self.num_zs_vecs_per_token, -1)
                    # If subj:
                    # ada_zs_basis_vecs: [9, 10, 768], ada_zs_bias: [9, 16, 768], static_zs_embs: [9, 16, 768].
                    # If bg:
                    # ada_zs_basis_vecs: [4, 10, 768], ada_zs_bias: [4, 16, 768], static_zs_embs: [4, 16, 768].
                    ada_zs_basis_vecs, ada_zs_bias, static_zs_embs = \
                        zs_vecs_2sets[:, :self.layerwise_lora_rank], \
                        zs_vecs_2sets[:, self.layerwise_lora_rank:self.layerwise_lora_rank+self.num_unet_ca_layers], \
                        zs_vecs_2sets[:, self.layerwise_lora_rank+self.num_unet_ca_layers:]

                    #breakpoint()
                    self.subj2ada_zs_basis_vecs[placeholder_string] = ada_zs_basis_vecs
                    self.subj2ada_zs_bias[placeholder_string]       = ada_zs_bias.permute(1, 0, 2)
                    # subj_static_embedding: [9, 16, 768] => [16, 9, 768]
                    static_zs_embs = static_zs_embs.permute(1, 0, 2)
                else:
                    static_zs_embs = None

                subj_static_embedding = static_embedder(static_zs_embs)
            else:
                # static_embedder is already the embedding vectors.
                subj_static_embedding = static_embedder

            static_subj_embs_dict[placeholder_string] = subj_static_embedding

            for k in range(self.token2num_vectors[placeholder_string]):
                # Assign the k-th token embedding (along the text dim).
                placeholder_indices_k = (placeholder_indices_1st[0], placeholder_indices_1st[1] + k)
                # embedded_text is repeated 16 times along the layer dimension, with size of dim 0 = 16 * BS.
                # After repeat, the same instance is repeated 16 times, which are adjacent 
                # to each other across the batch dim:
                # [b1_l1, ..., b1_l16, b2_l1, ..., b2_l16, ..., bB_l1, ..., bB_l16].
                # {________b1________} {_______b2_______}  ...  {_______bB________}
                # The first dim of subj_static_embedding is the layer dim (size = 16). 
                # So we repeat the 16 layers of the k-th embedding, subj_static_embedding[:, k], 
                # REAL_OCCURS_IN_BATCH times, to match 16*REAL_OCCURS_IN_BATCH.
                # After repeat, the RHS is
                # [ek_l1, ..., ek_l16, ek_l1, ..., ek_l16, ..., ek_l1, ..., ek_l16].
                # {________b1________} {_______b2_______}  ...  {_______bB________}
                subj_static_embedding_k = subj_static_embedding[:, k]

                if self.training and self.training_begin_add_noise_std_range is not None:
                    subj_static_embedding_k = add_noise_to_embedding(subj_static_embedding_k, 
                                                                     self.training_percent,
                                                                     self.training_begin_add_noise_std_range,
                                                                     self.training_end_add_noise_std_range,
                                                                     self.training_add_noise_prob[self.iter_type])
                # subj_static_embedding_k: [16, 768] => [16*REAL_OCCURS_IN_BATCH, 768]
                embedded_text[placeholder_indices_k] = subj_static_embedding_k.repeat(REAL_OCCURS_IN_BATCH, 1)

            # Cache the placeholder indices for mix prompt distillation.
            # Note placeholder_indices are recomputed in update_placeholder_indices(), 
            # we don't simply cache placeholder_indices here as they are repeated 16 times 
            # to replace in 16 layers. 
            # But we need them without repetitions for mix prompt distillation.
            # If num_vectors_per_subj_token > 1, then repeat the indices and add to offsets.
            # If background_strings is None, then always update the indices. Otherwise, 
            # skip updating placeholder indices of the background string.
            self.update_placeholder_indices(orig_tokenized_text, placeholder_string, placeholder_token, 
                                            self.token2num_vectors[placeholder_string],
                                            placeholder_is_bg=placeholder_is_bg)
        
        #print(self.cls_delta_string_indices)

        return embedded_text, tokenized_text, static_subj_embs_dict

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # As the output embedding is only generated for a particular layer, 
    # no need to repeat num_unet_ca_layers times as replacing with the static embedding in forward().
    def get_ada_embedding(
            self,
            layer_idx,              # the index of the current layer in the UNet.
            layer_attn_components,  # intermediate features and projections of the UNet attention layer.
            time_emb,               # time embedding of the current iteration.
            tokenized_text,         # [B, N]. Identical B copies along the batch dimension.
            embedded_text,          # [B, N, 768]. Identical B copies along the batch dimension.
    ):
        BS, device = tokenized_text.shape[0], tokenized_text.device
        cached_pooler_bg_out = None
        ada_subj_embs_dict   = {}
        token2ada_attn       = {}
        ca_layer_idx = self.layer_idx2ca_layer_idx[layer_idx]
        
        assert self.use_layerwise_embedding, "Non-layerwise embedding cannot call get_ada_embedding()."
        layer_static_prompt_embs   = layer_attn_components['layer_static_prompt_embs']
        # Clamping here seems to reduce the performance.
        # layer_static_prompt_embs   = clamp_prompt_embedding(self.prompt_embedding_clamp_value, layer_static_prompt_embs)
        
        self.cls_delta_string_indices = []
        prompt_subj_name_to_cls_delta_tokens = { subj_name: self.subj_name_to_cls_delta_tokens[subj_name] \
                                                 for subj_name in self.curr_batch_subj_names }

        # string_to_token_dict is an OrderedDict, with subject tokens added first, and 
        # the background token last (order controlled in main.py). 
        # This order ensures that the background Ada embedder can always use 
        # cached_pooler_bg_out produced by the previous subject Ada embedder.
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_is_bg = (placeholder_string in self.background_string_dict)
            # There's only one vector per token, we can do a simple replacement
            # embedded_text: [B, N, 768].
            # tokenized_text: [B, N].
            placeholder_indices = torch.where(tokenized_text == placeholder_token)
            # Skip generating the ada embedding if the corresponding placeholder token is not in the batch.
            if placeholder_indices[0].numel() == 0:
                continue

            # extract_first_index_in_each_instance(): Get the index to the first token in each instance.
            placeholder_indices_1st = extract_first_index_in_each_instance(placeholder_indices)
            # Expand placeholder_indices to include commas, if K > 1.
            placeholder_indices_ext = extend_indices_N_by_n_times(placeholder_indices, 
                                                                  self.token2num_vectors[placeholder_string])

            # REAL_OCCURS_IN_BATCH: the real number of occurrences of the placeholder in the current batch.
            # As ada embeddings are generated layer by layer, there is no repetition in the batch dimension.
            REAL_OCCURS_IN_BATCH = placeholder_indices_1st[0].numel()
            # Some prompts don't contain the placeholder token. This could happen in a compositional 
            # distillation iteration, or an inference iteration. 
            # Anyway, we need to check whether the cls_delta_string
            # occurs in the prompts without the placeholder token. If so, we need to merge 
            # their embeddings to one (the first) embedding, and delete the 2nd to the last embeddings,
            # using merge_cls_token_embeddings().
            # During inference, cls delta strings are not used. So CLS_DELTA_STRING_MAX_SEARCH_SPAN = -1.
            if REAL_OCCURS_IN_BATCH < BS and self.CLS_DELTA_STRING_MAX_SEARCH_SPAN > 0 \
              and len(prompt_subj_name_to_cls_delta_tokens) > 0:
                cls_delta_string_indices = scan_cls_delta_strings(tokenized_text,
                                                                  placeholder_indices_1st,
                                                                  prompt_subj_name_to_cls_delta_tokens,
                                                                  self.CLS_DELTA_STRING_MAX_SEARCH_SPAN)
                
                # cls_delta_string_indices is a list of tuples, each tuple is 
                # (batch_i, start_N, num_cls_delta_tokens, placeholder_token).
                self.cls_delta_string_indices += cls_delta_string_indices

            # For fg (subject) tokens, exclude fg embeddings from computing layer_static_extra_emb_mean. 
            # For bg (junk) tokens,    exclude fg embeddings from computing layer_static_extra_emb_mean.
            if placeholder_is_bg:
                # Chance is bg embeddings may accidentally attent to fg areas,
                # then self-reinforce and contaminate the bg embeddings with fg features.
                # However, if we mask out bg embeddings from computing layer_static_extra_emb_mean,
                # the performance will drop a lot.
                list_of_indices_to_mask = [ self.placeholder2indices[k] for k in self.placeholder2indices if k in self.subject_strings ]
            else:
                ## Why not mask bg indices for fg ada? bg embeddings are supposed to be of a similar nature 
                ## as the extra compositional embeddings. Incorporating them in layer_static_extra_emb_mean
                ## will make fg and bg embeddings more orthogonal (i.e., attend to different areas).
                list_of_indices_to_mask = [ self.placeholder2indices[k] for k in self.placeholder2indices if k in self.subject_strings ]

            # layer_static_prompt_embs: [4, 77, 768]. 
            # prompt_emb_mask: [4, 77, 1], which excludes SOT and padding tokens.
            # list_of_indices_to_mask: extra tokens to exclude, esp. the fg tokens.
            layer_static_extra_emb_mean = \
                self.calc_layer_static_extra_emb_mean(layer_static_prompt_embs, self.prompt_emb_mask, 
                                                      list_of_indices_to_mask, dropout_prob=0.1)

            # Clear cached_pooler_bg_out before generating subject embedding(s) of a subject token.
            # If the next token is the background token, the keep the cache, so that the background
            # token will reuse the cached_pooler_bg_out computed by the previous subject token.
            if not placeholder_is_bg:
                cached_pooler_bg_out = None

            ada_embedder = self.string_to_ada_embedder_dict[placeholder_string].to(device)
            assert isinstance(ada_embedder, AdaEmbedding)

            # When it's the turn of the background Ada embedder, the cached_pooler_bg_out
            # should have been computed by the previous subject Ada embedder. 
            # Otherwise it's a bug.
            if placeholder_is_bg \
              and ada_embedder.use_cached_bg and cached_pooler_bg_out is None:
                breakpoint()

            # layer_static_prompt_embs[curr_subj_indices]: subj_static_embedding, [(BS/2)*K, 768].
            # BS/2: In distillation iterations, only half instances of the batch contain the subject token.
            # If (BS/2) > 1 or K > 1, then take the mean embedding of the K embeddings.
            # Even if BS/2 > 1, the K static embeddings of different instances are the same. 
            # So it's ok to take the mean of all of them.
            # layer_static_subj_emb, layer_subj_emb_probe: [768].
            # layer_subj_emb_probe will be repeated by BS times to match the batch size in attn pooler.
            curr_subj_indices = placeholder_indices_ext
            layer_static_subj_emb = layer_static_prompt_embs[curr_subj_indices].mean(dim=0)

            # Don't use emb_ema_as_pooling_probe for background Ada embedder.
            if self.emb_ema_as_pooling_probe_weight > 0 and not placeholder_is_bg:
                # Use the first embedding of K embeddings as the probe.
                token_emb_ema = self.string_to_emb_ema_dict[placeholder_string]
                if token_emb_ema is None:
                    # In the first iteration, the token EMA hasn't been initialized. 
                    # So we only use static embeddings as the probe, 
                    # by setting emb_ema_as_pooling_probe_weight = 0.
                    emb_ema_as_pooling_probe_weight = 0
                    layer_subj_emb_ema = 0
                else:
                    emb_ema_as_pooling_probe_weight = self.emb_ema_as_pooling_probe_weight
                    # static_embedder is an Embedding3d within a LitEma. Get the actual embedding.
                    # LitEma copies the member variables of the wrapped Embedding3d. 
                    # So it's static_embedder.embedding.
                    # Our modified LitEma allows to be updated by SGD. However, the update may be 
                    # too aggressive. So we scale down the gradient by a factor of 0.1.
                    token_emb_ema_embedding = token_emb_ema.embedding
                    # layer_subj_emb_probe: [16, 9, 768] => [9, 768] => [768].
                    # We only need one embedding of [768]. But
                    # the layer subj embedding is of [9, 768]. So we take the first embedding.
                    layer_subj_emb_ema = token_emb_ema_embedding[ca_layer_idx].mean(dim=0)
                    # emb_ema_grad_scale = 0.05. Avoid updating the EMA embedding too quickly.
                    layer_subj_emb_ema = self.emb_ema_grad_scaler(layer_subj_emb_ema)

                # Although layer_subj_emb_ema cannot be updated through SGD, 
                # layer_static_subj_emb is updateable. So the probe will still adapt to the learning objective.
                layer_subj_emb_probe = (1 - emb_ema_as_pooling_probe_weight) * layer_static_subj_emb \
                                        + emb_ema_as_pooling_probe_weight    * layer_subj_emb_ema
            else:
                layer_subj_emb_probe = layer_static_subj_emb

            # Fix scale of the subj embedding probe (mean subj embedding), since it's scaled down by 
            # sqrt(num_vectors_per_subj_token), now we scale it up by sqrt(num_vectors_per_subj_token).
            layer_subj_emb_probe = layer_subj_emb_probe * np.sqrt(self.token2num_vectors[placeholder_string])

            if self.do_zero_shot:
                ada_zs_basis_vecs = self.subj2ada_zs_basis_vecs[placeholder_string]
                ada_zs_bias       = self.subj2ada_zs_bias[placeholder_string]
            else:
                ada_zs_basis_vecs = None
                ada_zs_bias       = None

            # Generate the actual subj_ada_embedding on the fly.
            # subj_ada_embedding: [B, K, 768]. B: 2 or 4 (regularization batches).
            # Before this call, we assume static_subj_embs has been generated by 
            # a call to get_static_embedding(). 
            # The pipeline is generate static embeddings first, then generate the ada embeddings. 
            # So this assumption should always hold.
            # For background Ada embedder, cached_pooler_bg_out is only used when
            # use_cached_bg. Otherwise, it's ignored.
            subj_ada_embedding, infeat_pooled_dict = \
                        ada_embedder(layer_idx, layer_attn_components, time_emb,
                                     layer_subj_emb_probe,
                                     layer_static_extra_emb_mean, 
                                     self.img_mask, cached_pooler_bg_out,
                                     ada_zs_basis_vecs, ada_zs_bias)

            if self.img_mask is not None and self.img_mask.max() > 1:
                breakpoint()

            if not placeholder_is_bg:
                # Cache the bg infeat computed by the first (fg) ada embedder, 
                # to be used by the second Ada embedder and the background Ada embedder.
                # NOTE: this assumes the background token always appears after the subject tokens.
                # Otherwise the cached_pooler_bg_out is not available when the background Ada embedder accesses it.
                cached_pooler_bg_out    = infeat_pooled_dict['bg_out']
                # During training, emb man self.training is True. 
                # This is in contrast to the UNet, whose self.training is always False 
                # (therefore cannot be used as an indicator).
                token2ada_attn.setdefault(placeholder_string, {})

                if self.training:
                    # 'attn_fg' and 'attn_bg' are used to reweight the attention maps in the CA layers.
                    # No need to let the grad flow back to the pooler attn from the weights for the CA layers.
                    token2ada_attn[placeholder_string]['attn_fg'] = infeat_pooled_dict['attn_fg'].detach()
                    token2ada_attn[placeholder_string]['attn_bg'] = infeat_pooled_dict['attn_bg'].detach()
                else:
                    # During inference, the second half of the batch is the unconditional prompts 
                    # which don't contain the placeholder token.
                    # Only assign ada subj attn_fg to instances that contain the placeholder token.
                    # In other instances that don't contain the placeholder token, the attn_fg is all 1.
                    placeholder_indices_B   = placeholder_indices_1st[0]
                    token2ada_attn[placeholder_string]['attn_fg'] = torch.ones_like(infeat_pooled_dict['attn_fg'])
                    token2ada_attn[placeholder_string]['attn_bg'] = torch.ones_like(infeat_pooled_dict['attn_bg'])
                    token2ada_attn[placeholder_string]['attn_fg'][placeholder_indices_B] = infeat_pooled_dict['attn_fg'][placeholder_indices_B]
                    token2ada_attn[placeholder_string]['attn_bg'][placeholder_indices_B] = infeat_pooled_dict['attn_bg'][placeholder_indices_B]

            for k in range(self.token2num_vectors[placeholder_string]):
                # embedded_text[placeholder_indices_1st] indexes the embedding at each instance in the batch.
                # embedded_text[placeholder_indices_1st]: [2, 768].  subj_ada_embedding: [2, 768].
                # Sometimes (e.g. during inference, cond instances contain the placeholder token but
                # uncond instances don't), only half of the instances of tokenized_text contain the placeholder token.
                # But ca_infeat is still of the full batch size. So subj_ada_embedding has a batch size 
                # larger than the number of instances containing the placeholder token. We need to index
                # subj_ada_embedding with placeholder_indices_1st[0] to get the matching new subj_ada_embedding.
                # We can save some computation by generating embeddings only for the instances containing 
                # the placeholder token. But that requires complex processing so not pursued now.
                placeholder_indices_k = (placeholder_indices_1st[0], placeholder_indices_1st[1] + k)
                # subj_ada_embedding: [BS, K, 768]. BS: 2 or 4 (regularization batches).
                subj_ada_embedding_k = subj_ada_embedding[placeholder_indices_1st[0], k]

                if self.training and self.training_begin_add_noise_std_range is not None:
                    subj_ada_embedding_k = add_noise_to_embedding(subj_ada_embedding_k,  
                                                                  self.training_percent,
                                                                  self.training_begin_add_noise_std_range,
                                                                  self.training_end_add_noise_std_range,
                                                                  self.training_add_noise_prob[self.iter_type])
                    
                embedded_text[placeholder_indices_k] = subj_ada_embedding_k

            # Remove the batch dim.
            ada_subj_embs_dict[placeholder_string] = subj_ada_embedding.mean(dim=0)

        #print(self.cls_delta_string_indices)

        return embedded_text, ada_subj_embs_dict, token2ada_attn

    # extend_placeholders() should only be called during inference, 
    # or after the model has been initialized.
    def extend_placeholders(self, new_subject_strings, new_background_strings, 
                            num_vectors_per_subj_token, num_vectors_per_bg_token):
        added_placeholders = []

        if new_subject_strings is not None:
            for k in new_subject_strings:
                if k is None or k in self.subject_strings:
                    continue
                self.subject_strings.append(k)
                self.subject_string_dict[k] = True
                self.placeholder_strings.append(k)
                self.token2num_vectors[k] = num_vectors_per_subj_token
                added_placeholders.append(k)
                print(f"Added new subject string: {k}->num_vectors_per_subj_token={num_vectors_per_subj_token}.")

        if new_background_strings is not None:
            for k in new_background_strings:
                if k is None or k in self.background_strings:
                    continue
                self.background_strings.append(k)
                self.background_string_dict[k] = True
                self.placeholder_strings.append(k)
                self.token2num_vectors[k] = num_vectors_per_bg_token
                added_placeholders.append(k)
                print(f"Added new background string: {k}->num_vectors_per_subj_token={num_vectors_per_bg_token}.")

        if len(added_placeholders) > 0:
            extended_token_embeddings = extend_clip_text_embedder(self.text_embedder, {}, added_placeholders)
            if extended_token_embeddings is not None:
                # text_embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
                # = LatentDiffusion.cond_stage_model
                # Make sure text_embedder has been initialized. Otherwise when loading pretrained weights,
                # it will throw an exception, as now the vocab size of the text_embedder is different from
                # the pretrained weights.
                # Update self.extended_token_embeddings. These new token embeddings will be extended
                # to CLIP text embedder in either main.py or stable_txt2img.py later.
                if self.extended_token_embeddings is not None:
                    self.extended_token_embeddings = torch.cat([self.extended_token_embeddings, extended_token_embeddings], dim=0)
                else:
                    self.extended_token_embeddings = extended_token_embeddings
        return
    
    # Update prompt_emb_mask.
    # tokenized_text: [B, N] = [2/4, 77].
    # DDPM.validation_step() -> LatentDiffusion.shared_step() -> .forward()
    # -> .get_learned_conditioning() -> .cond_stage_model.encode()
    # -> EmbeddingManager.forward() -> here.
    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # Occasionally, image_logger is called, which calls LatentDiffusion.log_images ->
    # .get_learned_conditioning() -> ... -> here.
    # Such prompt_emb_mask won't be used in calc_prompt_emb_delta_loss() and won't be cleared.
    # prompt_emb_mask: [B, N, 1], where N=77 is the prompt length after padding.
    def update_prompt_masks(self, tokenized_text, tokenized_text_repeated=False):
        # Exclude the starting (49406) and padding tokens (49047) from delta loss.
        prompt_emb_mask  = (tokenized_text != 49406 ) & (tokenized_text != 49407)
        # [B, N] => [B, N, 1]
        self.prompt_emb_mask  = prompt_emb_mask.float().unsqueeze(2)
        padding_tokens_mask = (tokenized_text == 49407).float()
        B,  N  = tokenized_text.shape
        B2, N2 = tokenized_text_repeated.shape
        assert N == N2

        # Don't block the subj-padding interaction.
        # # Block the interaction between the subject and padding tokens with a probability of 0.5.
        p_block_subj_padding_interaction = 0 if self.training else 0

    # layer_static_prompt_embs: [4, 77, 768].
    # emb_mask:          [4, 77, 1]. Set to 1 to include / 0 to exclude from the mean.
    # (Note sometimes the batch size of emb_mask is different from layer_static_prompt_embs).
    def calc_layer_static_extra_emb_mean(self, layer_static_prompt_embs, emb_mask, 
                                         list_of_indices_to_mask, dropout_prob=0.1):
        emb_mask = emb_mask.clone()
        count_of_masked_indices = sum([ (indices_to_mask is not None) for indices_to_mask in list_of_indices_to_mask])
        assert count_of_masked_indices > 0

        if list_of_indices_to_mask is not None:
            for indices_to_mask in list_of_indices_to_mask:
                if indices_to_mask is not None:
                    # Mask out embeddings.
                    emb_mask[indices_to_mask] = 0

        if layer_static_prompt_embs.shape[0] < emb_mask.shape[0]:
            # In teacher filtering stage within prompt distillation iterations,
            # there are more prompts (BS) than prompts containing subject tokens (BS/2). 
            # So we take the first half instances of emb_mask 
            # (the second half instances of emb_mask are the same as the first half)
            emb_mask = emb_mask[:layer_static_prompt_embs.shape[0]]
        elif layer_static_prompt_embs.shape[0] > emb_mask.shape[0]:
            # This should only happen during inference, where the cond and uncond 
            # static embeddings are combined as input, but emb_mask is only the 
            # mask of the cond embeddings. 
            # The batch structure is [4*cond, 4*uncond]. So we append 
            # a zero tensor of size 4 to emb_mask to match the uncond embeddings.
            emb_mask = torch.cat([emb_mask, torch.zeros_like(emb_mask)], dim=0)
        
        layer_static_extra_emb_mean = masked_mean(layer_static_prompt_embs, emb_mask, dim=1)
        # Drop out the static embedding mean features with a probability of 0.2.
        if self.training and random.random() < dropout_prob:
            layer_static_extra_emb_mean.zero_()

        return layer_static_extra_emb_mean
    
    def clear_prompt_adhoc_info(self):
        self.placeholder2indices    = {}
        self.img_mask               = None
        self.prompt_emb_mask        = None

    # Set ad-hoc data structures for computing placeholder embeddings and various losses.
    def set_prompt_adhoc_info(self, prompt_adhoc_info):
        self.placeholder2indices    = prompt_adhoc_info['placeholder2indices']
        # There are image margins after the original image is scaled down, or 
        # after using wds overlay images as input.
        # When doing attentional pooling / average pooling of image features, 
        # the margin area contains no signal, so we use img_mask to mask it out. 
        # Each image has its own img_mask, so img_mask has a shape of [B, 1, H, W].
        self.img_mask               = prompt_adhoc_info['img_mask']
        self.prompt_emb_mask        = prompt_adhoc_info['prompt_emb_mask']
    
    def set_curr_batch_subject_names(self, subj_names):
        self.curr_batch_subj_names = subj_names

    # Cache features used to compute ada embeddings.
    def cache_layer_features_for_ada(self, layer_idx, layer_attn_components, time_emb):
        self.gen_ada_embedding      = True
        self.layer_idx              = layer_idx
        self.time_emb               = time_emb
        self.layer_attn_components  = layer_attn_components

    # Clear layer-specific intermediate variables. Also clear gen_ada_embedding,
    # which will be enabled again through cache_layer_features_for_ada() in ddpm.py.
    def clear_ada_layer_temp_info(self):
        self.gen_ada_embedding      = False
        self.layer_idx              = -1
        self.time_emb               = None
        self.layer_attn_components  = None

    def update_placeholder_indices(self, tokenized_text, placeholder_string, placeholder_token, num_vectors_per_subj_token, placeholder_is_bg):
        placeholder_indices = torch.where(tokenized_text == placeholder_token)
        placeholder_indices_B, placeholder_indices_N = extract_first_index_in_each_instance(placeholder_indices)

        if len(placeholder_indices_B) == 0:
            self.placeholder2indices[placeholder_string] = None
            return

        if num_vectors_per_subj_token > 1:
            BS = placeholder_indices_B.shape[0]
            # unsqueeze(1) -> [B, 1] => [B, num_vectors_per_subj_token] => [B * num_vectors_per_subj_token].
            # Make sure the embedding indices of the same instance are grouped together.
            # [b1_v1, b1_v2, b1_v3, b2_v1, b2_v2, b2_v3, ...].
            # Then we can easily get the indices of a certain sub-batch.
            placeholder_indices_B = placeholder_indices_B.unsqueeze(1).repeat(1, num_vectors_per_subj_token).view(-1)
            placeholder_indices_N = placeholder_indices_N.unsqueeze(1).repeat(1, num_vectors_per_subj_token).view(-1)
            # Add offsets to the indices of the pseudo-tokens.
            placeholder_indices_N_off = placeholder_indices_N + torch.arange(num_vectors_per_subj_token, device=tokenized_text.device).repeat(BS)
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N_off)
        else:
            # placeholder_indices contains the indices of all placeholder embeddings.
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N)
        
        self.placeholder2indices[placeholder_string] = placeholder_indices

    def get_ada_emb_weight(self, do_perturb=True):
        if self.training and do_perturb:
            # 0.5 -> uniform in [0.4, 0.7]. Inject randomness to reduce overfitting.
            ada_emb_weight = self.ada_emb_weight * np.random.uniform(0.8, 1.4)
        else:
            ada_emb_weight = self.ada_emb_weight        
        return ada_emb_weight
 
    def get_ada_subj_attn_dict(self):
        return self.ada_subj_attn_dict
    
    def set_training_add_noise_specs(self, training_begin_add_noise_std_range, 
                                     training_end_add_noise_std_range,
                                     training_add_noise_prob):
        self.training_begin_add_noise_std_range = training_begin_add_noise_std_range
        self.training_end_add_noise_std_range   = training_end_add_noise_std_range
        self.training_add_noise_prob      = training_add_noise_prob
        if training_begin_add_noise_std_range is None and training_end_add_noise_std_range is None:
            print(f"Disable training_add_noise")
        else:
            print(f"training add_noise std range: {training_begin_add_noise_std_range}-{training_end_add_noise_std_range}"
                  ", with prob = {training_add_noise_prob}")

    def initialize_subj2conv_attn_layerwise_scales(self, default_conv_attn_scale=1, 
                                                   subj2conv_attn_layerwise_scales=None,
                                                   learnable=False):
        self.conv_attn_layerwise_scale_learnable = learnable
        
        if subj2conv_attn_layerwise_scales is not None:
            for subj_string, conv_attn_layerwise_scales in subj2conv_attn_layerwise_scales.items():
                conv_attn_layerwise_scales = nn.Parameter(conv_attn_layerwise_scales,
                                                          requires_grad=learnable)            
                print(f"Change {subj_string}: conv_attn_layerwise_scales = {conv_attn_layerwise_scales}")
                self.subj2conv_attn_layerwise_scales[subj_string] = conv_attn_layerwise_scales
        else:
            if self.subj2conv_attn_layerwise_scales is None:
                # If subj2conv_attn_layerwise_scales have already been initialized, don't reinitialize.
                self.subj2conv_attn_layerwise_scales = nn.ParameterDict()

            for subj_string in self.subject_strings:
                # If subj2conv_attn_layerwise_scales[subj_string] have already been initialized, 
                # don't overwrite them.
                if subj_string not in self.subj2conv_attn_layerwise_scales:
                    conv_attn_layerwise_scales = nn.Parameter(torch.ones(self.num_layers_per_embedder) * default_conv_attn_scale, 
                                                              requires_grad=learnable)
                    self.subj2conv_attn_layerwise_scales[subj_string] = conv_attn_layerwise_scales
                print(f"Initialize {subj_string}: conv_attn_layerwise_scales = {conv_attn_layerwise_scales}")

    def get_subj2conv_attn_layer_scale(self):
        # Clip the scales to [0.1, 3].
        #for subj_string in self.subject_strings:
        #    self.subj2conv_attn_layerwise_scales[subj_string].data.clamp_(min=0.1, max=3)

        return self.subj2conv_attn_layerwise_scales
    
    def get_emb_global_scales_dict(self, regen, do_perturb=True):
        if not regen:
            if self.emb_global_scales_dict is None:
                regen = True
            else:
                # Reuse the cached global scales.
                emb_global_scales_dict = self.emb_global_scales_dict

        if regen:
            # emb_global_scale_score = 0  -> emb_global_scale = 1, 
            # emb_global_scale_score = 1  -> emb_global_scale = 1.23
            # emb_global_scale_score = -1 -> emb_global_scale = 0.77
            emb_global_scales = self.emb_global_scale_scores.sigmoid() + 0.5
            if self.training and do_perturb:
                perturbation = torch_uniform(0.8, 1.4, (emb_global_scales.shape[0],), device=emb_global_scales.device)
                # 1 -> uniform in [0.8, 1.4]. Inject randomness to reduce overfitting.
                emb_global_scales = emb_global_scales * perturbation

            emb_global_scales_dict = {}
            for i, placeholder in enumerate(self.string_to_token_dict.keys()):
                emb_global_scales_dict[placeholder] = emb_global_scales[i]

            # Cache the global scales to be reused by get_layer_ada_conditioning() for ada embeddings.
            self.emb_global_scales_dict = emb_global_scales_dict

        return emb_global_scales_dict
    
    def set_ada_emb_weight(self, ada_emb_weight, is_first_time_print=False):
        if is_first_time_print:
            print(f"Setting ada_emb_weight = {ada_emb_weight}")
        else:
            if self.ada_emb_weight != ada_emb_weight:
                print(f"ada_emb_weight: {self.ada_emb_weight} => {ada_emb_weight}")

        self.ada_emb_weight = ada_emb_weight

    def set_embs_attn_tricks(self, use_conv_attn_kernel_size=None):
        if use_conv_attn_kernel_size is not None:
            self.use_conv_attn_kernel_size = use_conv_attn_kernel_size
            extra_msg = ", DISABLED" if use_conv_attn_kernel_size is -1 else ""
            print(f"Setting use_conv_attn_kernel_size = {use_conv_attn_kernel_size}{extra_msg}")

    def set_emb_ema_as_pooling_probe_weight(self, emb_ema_as_pooling_probe_weight):
        self.emb_ema_as_pooling_probe_weight = emb_ema_as_pooling_probe_weight
        print(f"Setting emb_ema_as_pooling_probe_weight = {emb_ema_as_pooling_probe_weight}")

    def set_attn_pooler_feat_reduction_ratio(self, attn_pooler_feat_reduction_ratio):
        self.attn_pooler_feat_reduction_ratio = attn_pooler_feat_reduction_ratio
        print(f"Setting attn_pooler_feat_reduction_ratio = {attn_pooler_feat_reduction_ratio}")

    def set_zs_image_features(self, zs_clip_features, zs_face_embs):
        # zs_clip_features: [1, 514, 1280]
        # zs_clip_subj_features, zs_clip_bg_features: [1, 257, 1280].
        zs_clip_subj_features, zs_clip_bg_features = zs_clip_features.chunk(2, dim=1)
        #print(zs_clip_subj_features.mean(dim=1).squeeze(0)[:20])
        #print(zs_clip_bg_features.mean(dim=1).squeeze(0)[:20])

        self.zs_image_feat_dict = { 'subj': zs_clip_subj_features, 'bg': zs_clip_bg_features,
                                    'face': zs_face_embs }
        # Beginning of a new iteration, clear the cached ada_zs_basis_vecs and ada_zs_bias.
        self.subj2ada_zs_basis_vecs = {}
        self.subj2ada_zs_bias       = {}
        # Clear the basis_vecs and bias saved in embedders.
        for placeholder_string in self.placeholder_strings:
            self.string_to_static_embedder_dict[placeholder_string].basis_vecs = None
            self.string_to_static_embedder_dict[placeholder_string].bias       = None
            self.string_to_ada_embedder_dict[placeholder_string].basis_vecs    = None
            self.string_to_ada_embedder_dict[placeholder_string].bias          = None

    # NOTE: prompt embeddings are the embeddings of the whole prompt (including other tokens), 
    # not just the ada or static embeddings of the subject.
    def cache_ada_prompt_embedding(self, layer_idx, embedding):
        ca_layer_idx = self.layer_idx2ca_layer_idx[layer_idx]
        self.ada_prompt_embeddings_cache[ca_layer_idx] = embedding
        # If there are multiple layers, only the last placeholder_indices are cached.
        self.ada_prompt_placeholder2indices_cache = copy.copy(self.placeholder2indices)

    def get_cached_ada_prompt_embeddings_as_tensor(self):
        # No tokens appear in the current prompt. So the ada prompt embedding cache is empty.
        if len(self.ada_prompt_embeddings_cache) == 0:
            return None
        
        # ada_prompt_embeddings_cache is a list of tensors, not a tensor itself. 
        # So the length is the number of layers that have been cached.
        # Each element is [BS, 77, 768], the ada prompt embeddings (BS >=1) of a layer.
        if len(self.ada_prompt_embeddings_cache) != self.num_layers_per_embedder:
            breakpoint()
        # Stack the cached prompt embeddings of all layers into a tensor.
        ada_prompt_embeddings = [ self.ada_prompt_embeddings_cache[ca_layer_idx] 
                                  for ca_layer_idx in range(self.num_layers_per_embedder) ]
        # Each element in ada_prompt_embeddings_cache is [BS, 77, 768] =>
        # ada_prompt_embeddings: [BS, 16, 77, 768]. BS: batch size (2 or 4). 16: num layers.
        ada_prompt_embeddings = torch.stack(ada_prompt_embeddings, dim=1)
        return ada_prompt_embeddings

    # self.ada_prompt_embeddings_cache is a cache for the prompt embeddings of all layers, 
    # for computing the prompt delta loss.
    # NOTE: prompt embeddings are the embeddings of the whole prompt (including other tokens), 
    # not just the ada or static embeddings of the subject.
    def clear_ada_prompt_embeddings_cache(self):
        if len(self.ada_prompt_embeddings_cache) == self.num_layers_per_embedder:
            self.update_emb_ema(self.ada_prompt_placeholder2indices_cache)

        self.ada_prompt_embeddings_cache = {}

    def update_emb_ema(self, placeholder2indices):
        if self.training and self.emb_ema_as_pooling_probe_weight > 0:
            for k, token_emb_cache_obj in self.placeholder_to_emb_cache.items():
                # If all layers of ada embeddings have been cached in token_emb_cache_obj,
                # then it's time to update EMA embeddings.
                # This should happen after the previous training iteration finishes and 
                # before the current training iteration begins.
                # placeholder_to_emb_cache is used to compute EMA of ada embeddings.
                token_emb_cache_obj  = self.placeholder_to_emb_cache[k]

                if k not in placeholder2indices:
                    continue
                token_indices = placeholder2indices[k]
                
                # token k doesn't appear in this prompt (should be bg token).
                if token_indices is None:
                    continue

                for ca_layer_idx, layer_ada_prompt_embs in self.ada_prompt_embeddings_cache.items():
                    # token_indices may only contain the indices of some (not all) instances in the batch.
                    # layer_ada_prompt_embs: [2, 77, 768].
                    # layer_ada_prompt_embs[token_indices]: [18, 768] => [2, 9, 768].
                    VALID_BS = token_indices[0].unique().shape[0]
                    if VALID_BS > layer_ada_prompt_embs.shape[0]:
                        breakpoint()
                    # We don't update the whole batch of embeddings, just those selected by token_indices.
                    # For example, fg_indices only select the instances containing the subject token.
                    layer_token_prompt_embs = layer_ada_prompt_embs[token_indices].reshape(
                                                    VALID_BS, -1, layer_ada_prompt_embs.shape[2])
                    # LitEma requires an nn.Module to do updating. 
                    # So we use token_emb_cache_obj as a dummy Embedding3d to update the EMA embedding.
                    # We can update after the ada embeddings of all layers are cached into token_emb_cache_obj.
                    # layer_token_prompt_embs.mean(dim=0): [9, 768].
                    token_emb_cache_obj.cache_layer(ca_layer_idx, layer_token_prompt_embs.mean(dim=0))

                if len(token_emb_cache_obj.cached_layers) == token_emb_cache_obj.num_layers:
                    if self.string_to_emb_ema_dict[k] is None:
                        # First iteration, initialize the LitEma object.
                        print("Initializing LitEma for token", k)
                        # requires_grad=True, to allow EMA embeddings to be updated by SGD.
                        self.string_to_emb_ema_dict[k] = LitEma(token_emb_cache_obj, decay=0.998, requires_grad=True)
                        # Put the newly initialized LitEma object on CUDA.
                        self.string_to_emb_ema_dict[k].to(layer_token_prompt_embs.device)
                    else:
                        # Update EMA embeddings.
                        self.string_to_emb_ema_dict[k](token_emb_cache_obj)

                    token_emb_cache_obj.reset_cached_layer_tracker()

    def set_num_vectors_per_subj_token(self, token2num_vectors):
        self.token2num_vectors = token2num_vectors
        print(f"Set token2num_vectors: {self.token2num_vectors}")

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, ckpt_path):
        torch.save({ "string_to_token":                 self.string_to_token_dict,
                     "string_to_static_embedder":       self.string_to_static_embedder_dict,
                     "string_to_ada_embedder":          self.string_to_ada_embedder_dict,
                     "string_to_emb_ema_dict":          self.string_to_emb_ema_dict,
                     "token2num_vectors":               self.token2num_vectors,
                     "emb_global_scale_scores":         self.emb_global_scale_scores,
                     "ada_emb_weight":                  self.ada_emb_weight,  
                     "emb_ema_as_pooling_probe_weight": self.emb_ema_as_pooling_probe_weight,
                     "shared_placeholder_set":          self.shared_placeholder_set,
                     "shared_embedder_components":      self.shared_embedder_components,
                     # Learnable weights for scaling conv attns.
                     "subj2conv_attn_layerwise_scales":  self.subj2conv_attn_layerwise_scales,
                     "use_conv_attn_kernel_size":        self.use_conv_attn_kernel_size,
                     "attn_pooler_feat_reduction_ratio": self.attn_pooler_feat_reduction_ratio,
                     "placeholder_strings":              self.placeholder_strings,
                     "subject_strings":                  self.subject_strings,
                     "background_strings":               self.background_strings,
                     "ca_q_bns":                         self.ca_q_bns,
                     "ca_outfeat_lns":                   self.ca_outfeat_lns,
                     "do_zero_shot":                     self.do_zero_shot,
                     "subj_basis_generator":             self.subj_basis_generator,
                   }, 
                    ckpt_path)

    # load custom tokens and their learned embeddings from "embeddings_gs-4200.pt".
    # If src_placeholders = None, then load the whole embedding manager. Otherwise, src_placeholders should 
    # be two strings, either "subject_string,background_string", or "1,1" which means the first subject and
    # the first background string.
    def load(self, ckpt_paths, ckpt_params_perturb_ratio=0, src_placeholders=None, 
             loaded_embedder_components=None, frozen_placeholder_set=None, frozen_embedder_components=None):
        if src_placeholders is not None and loaded_embedder_components is not None:
            self.load_embedder_components(ckpt_paths, src_placeholders=src_placeholders, 
                                          loaded_embedder_components=loaded_embedder_components,
                                          frozen_placeholder_set=frozen_placeholder_set,
                                          frozen_embedder_components=frozen_embedder_components)
            return

        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        token2num_vectors                   = {}
        emb_global_scale_scores_dict        = {}
        subj2conv_attn_layerwise_scales     = {}
        
        self.string_to_token_dict           = {}
        self.string_to_emb_ema_dict         = nn.ModuleDict()
        self.string_to_static_embedder_dict = nn.ParameterDict()
        self.string_to_ada_embedder_dict    = nn.ModuleDict()

        self.subject_strings                = []
        self.background_strings             = []
        extended_token_embeddings                = []

        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]

        for ckpt_path in ckpt_paths:
            ckpt_path_parts = ckpt_path.split(":")
            ckpt_path = ckpt_path_parts[0]
            if len(ckpt_path_parts) == 2:
                placeholder_mapper = {}
                for placeholder_mapping in ckpt_path_parts[1].split(","):
                    from_, to_ = placeholder_mapping.split("-")
                    placeholder_mapper[from_] = to_
            else:
                placeholder_mapper = None

            ckpt = torch.load(ckpt_path, map_location='cpu')
            # If multiple checkpoints have different ada_emb_weight, the last one will be used.
            if "ada_emb_weight" in ckpt:
                self.set_ada_emb_weight(ckpt["ada_emb_weight"], is_first_time_print=False)

            if "emb_ema_as_pooling_probe_weight" in ckpt and not self.do_zero_shot:
                self.set_emb_ema_as_pooling_probe_weight(ckpt["emb_ema_as_pooling_probe_weight"])

            if "background_strings" in ckpt:
                ckpt_background_strings = ckpt["background_strings"]
            else:
                ckpt_background_strings = []

            use_conv_attn_kernel_size   = ckpt.get("use_conv_attn_kernel_size", None)
            self.set_embs_attn_tricks(use_conv_attn_kernel_size)

            if "attn_pooler_feat_reduction_ratio" in ckpt:
                # Setting attn_pooler_feat_reduction_ratio doesn't have much impact actually,
                # since the attn pooler is loaded from ckpt, whose feat_reduction_ratio has been
                # implicitly determined.
                self.set_attn_pooler_feat_reduction_ratio(ckpt["attn_pooler_feat_reduction_ratio"])

            if "ca_q_bns" in ckpt:
                self.ca_q_bns = ckpt["ca_q_bns"]
            if "ca_outfeat_lns" in ckpt:
                self.ca_outfeat_lns = ckpt["ca_outfeat_lns"]

            # Only load subj_basis_generator from ckpt if the ckpt is set with the same do_zero_shot.
            if "do_zero_shot" in ckpt and self.do_zero_shot == ckpt["do_zero_shot"]:
                self.subj_basis_generator   = ckpt["subj_basis_generator"]

            for token_idx, km in enumerate(ckpt["placeholder_strings"]):
                # Mapped from km in ckpt to km2 in the current session. Partial matching is allowed.
                if (placeholder_mapper is not None) and (km in placeholder_mapper):
                    km2 = placeholder_mapper[km]
                else:
                    km2 = km

                try:
                    k2_token = self.get_tokens_for_string(km2, force_single_token=True)[0]
                except:
                    k2_embedding = extend_clip_text_embedder(self.text_embedder, {}, [km2])
                    extended_token_embeddings.append(k2_embedding)
                    k2_token = self.get_tokens_for_string(km2, force_single_token=True)[0]
                    print
                if km2 in self.string_to_token_dict:
                    if km2 in self.background_strings:
                        print(f"Duplicate key {km}->{km2} in {ckpt_path}. Ignored.")
                        continue

                    raise ValueError(f"Duplicate key {km}->{km2} in {ckpt_path}")

                if "emb_global_scale_scores" in ckpt:
                    emb_global_scale_scores_dict[km2] = ckpt["emb_global_scale_scores"][token_idx]
                if "subj2conv_attn_layerwise_scales" in ckpt and km in ckpt["subj2conv_attn_layerwise_scales"]:
                    subj2conv_attn_layerwise_scales[km2] = ckpt["subj2conv_attn_layerwise_scales"][km]

                # Merge the (possibly substituted) subject strings from the ckpt with 
                # self.subject_strings and self.background_strings.
                if km in ckpt_background_strings:
                    self.background_strings = list(set(self.background_strings + [km2]))
                    print("Add background string", km2)
                elif km not in self.background_strings:
                    # Add km2 to self.subject_strings, even if it's not in ckpt["subject_strings"].
                    # This is to be compatible with older ckpts which don't save ckpt["subject_strings"].
                    self.subject_strings = list(set(self.subject_strings + [km2]))
                    print("Add subject string", km2)

                # The mapping in string_to_token_dict is determined by the tokenizer. 
                # Shouldn't do the km->km2 mapping on string_to_token_dict.
                self.string_to_token_dict[km2] = k2_token

                self.string_to_static_embedder_dict[km2] = ckpt["string_to_static_embedder"][km]
                self.string_to_ada_embedder_dict[km2]    = ckpt["string_to_ada_embedder"][km]
                if self.emb_ema_as_pooling_probe_weight > 0:
                    self.string_to_emb_ema_dict[km2]     = ckpt["string_to_emb_ema_dict"][km]
                    
                if km in ckpt["token2num_vectors"]:
                    token2num_vectors[km2] = ckpt["token2num_vectors"][km]

                print(f"Loaded {km}->{km2} from {ckpt_path}")
                
                static_embedder = self.string_to_static_embedder_dict[km2]
                # Make compatible with older ckpts which don't have do_zero_shot.
                if 'do_zero_shot' not in static_embedder.__dict__:
                    static_embedder.do_zero_shot = self.do_zero_shot

                ada_embedder = self.string_to_ada_embedder_dict[km2]
                # Make compatible with older ckpts which don't have do_zero_shot.
                if 'do_zero_shot' not in ada_embedder.__dict__:
                    ada_embedder.do_zero_shot = self.do_zero_shot

                print(f"{km2}: {ada_embedder.fg_emb_count}/{ada_embedder.bg_emb_count}/{ada_embedder.K} fg/bg/total embeddings")

            if "token2num_vectors" in ckpt:
                self.set_num_vectors_per_subj_token(token2num_vectors)

            # In theory, if some ckpt has shared_placeholder_set = True, and some has False,
            # then the ckpt attn poolers after the last True ckpt will not be shared.
            # But this shouldn't be a concern, as such scenarios should be very rare.
            if "shared_placeholder_set" in ckpt:
                self.share_embedder_components(ckpt["shared_placeholder_set"], ckpt["shared_embedder_components"])

        self.emb_global_scale_scores = nn.Parameter(torch.zeros(len(self.string_to_token_dict)), 
                                                    requires_grad=True)
        for token_idx, km in enumerate(self.string_to_token_dict):
            if km in emb_global_scale_scores_dict:
                self.emb_global_scale_scores.data[token_idx] = emb_global_scale_scores_dict[km]

        print(f"Set emb_global_scales = {self.get_emb_global_scales_dict(regen=True, do_perturb=False)}")

        if len(subj2conv_attn_layerwise_scales) > 0:
            self.initialize_subj2conv_attn_layerwise_scales(1, subj2conv_attn_layerwise_scales)

        self.placeholder_strings = self.subject_strings + self.background_strings
        if len(extended_token_embeddings) > 0:
            self.extended_token_embeddings = torch.cat(extended_token_embeddings, dim=0)
            print(f"Extended {len(extended_token_embeddings)} token embeddings")

        # When we resume training from a ckpt, sometimes we want to perturb the parameters
        # to reduce overfitting.
        if ckpt_params_perturb_ratio > 0:
            self.perturb_model_parameters(ckpt_params_perturb_ratio)

        # Regenerate subject_string_dict, background_string_dict 
        # in case subject_strings or background_strings have been changed.
        self.subject_string_dict    = { s: True for s in self.subject_strings }
        self.background_string_dict = { s: True for s in self.background_strings }

    # src_placeholders should be two strings, either "subject_string,background_string", 
    # or "1,1" which means the first subject and the first background string.
    def load_embedder_components(self, ckpt_paths, src_placeholders, 
                                 loaded_embedder_components, 
                                 frozen_placeholder_set, frozen_embedder_components):
        if isinstance(loaded_embedder_components, str):
            loaded_embedder_components = loaded_embedder_components.split(",")

        src_subj_string, src_bg_string = src_placeholders.split(",")
        
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]

        for ckpt_i, ckpt_path in enumerate(ckpt_paths):
            ckpt_path_parts = ckpt_path.split(":")
            ckpt_path = ckpt_path_parts[0]
            ckpt = torch.load(ckpt_path, map_location='cpu')

            if "attn_pooler_feat_reduction_ratio" in ckpt:
                # Setting attn_pooler_feat_reduction_ratio doesn't have much impact actually,
                # since the attn pooler is loaded from ckpt, whose feat_reduction_ratio has been
                # implicitly determined.
                # If loading multiple ckpts, and their attn_pooler_feat_reduction_ratio are different,
                # the last one will be kept. But this should happen extremely rare.
                self.set_attn_pooler_feat_reduction_ratio(ckpt["attn_pooler_feat_reduction_ratio"])

            # If src_subj_string/src_bg_string are "1", then replace them with the first subject/background string
            # of the first ckpt. 
            if ckpt_i == 0:
                if src_subj_string == "1":
                    src_subj_string = ckpt["subject_strings"][0]
                if src_bg_string == "1":
                    src_bg_string   = ckpt["background_strings"][0]

            for km in ckpt["string_to_ada_embedder"].keys():
                if src_subj_string == km:
                    for subj_string in self.subject_strings:
                        if 'pooler' in loaded_embedder_components:
                            # All subject strings share the same poolers loaded from the ckpt.
                            self.string_to_ada_embedder_dict[subj_string].poolers = ckpt["string_to_ada_embedder"][km].poolers
                        if 'layer_coeff_maps' in loaded_embedder_components:
                            # All subject strings share the same layer_coeff_maps loaded from the ckpt.
                            self.string_to_ada_embedder_dict[subj_string].layer_coeff_maps = ckpt["string_to_ada_embedder"][km].layer_coeff_maps
                        if 'basis_weights' in loaded_embedder_components:
                            self.string_to_static_embedder_dict[subj_string].basis_rand_weights = ckpt["string_to_static_embedder"][km].basis_rand_weights
                            self.string_to_static_embedder_dict[subj_string].basis_comm_weights = ckpt["string_to_static_embedder"][km].basis_comm_weights

                elif src_bg_string == km:
                    for bg_string in self.background_strings:
                        if 'pooler' in loaded_embedder_components:
                            # All background strings share the same poolers loaded from the ckpt.
                            self.string_to_ada_embedder_dict[bg_string].poolers = ckpt["string_to_ada_embedder"][km].poolers
                        if 'layer_coeff_maps' in loaded_embedder_components:
                            self.string_to_ada_embedder_dict[bg_string].layer_coeff_maps = ckpt["string_to_ada_embedder"][km].layer_coeff_maps
                        if 'basis_weights' in loaded_embedder_components:
                            self.string_to_static_embedder_dict[bg_string].basis_rand_weights = ckpt["string_to_static_embedder"][km].basis_rand_weights
                            self.string_to_static_embedder_dict[bg_string].basis_comm_weights = ckpt["string_to_static_embedder"][km].basis_comm_weights
                            print(f"Loaded basis_weights {km}->{src_bg_string} from {ckpt_path}")

            # No need to call share_embedder_components() here, as the poolers are already shared 
            # after the aasignment above.

        self.freeze_embedder_components(frozen_placeholder_set, frozen_embedder_components)

    def share_embedder_components(self, shared_placeholder_set, shared_embedder_components):
        if isinstance(shared_placeholder_set, str):
            shared_placeholder_set = shared_placeholder_set.split(",")
        if isinstance(shared_embedder_components, str):
            shared_embedder_components = shared_embedder_components.split(",")

        self.shared_placeholder_set      = shared_placeholder_set
        self.shared_embedder_components  = shared_embedder_components

        if shared_placeholder_set is not None and shared_placeholder_set is not None:
            first_subj_ada = self.string_to_ada_embedder_dict[self.subject_strings[0]]
            first_bg_ada   = self.string_to_ada_embedder_dict[self.background_strings[0]]
            first_subj_static_embedder = self.string_to_static_embedder_dict[self.subject_strings[0]]
            first_bg_static_embedder   = self.string_to_static_embedder_dict[self.background_strings[0]]

            subj_components_share_count = dict([(component, 0) for component in shared_embedder_components])
            bg_components_share_count   = dict([(component, 0) for component in shared_embedder_components])
            shared_placeholder_strings = []

            if 'subj' in shared_placeholder_set:
                shared_placeholder_strings += self.subject_strings
            if 'bg'   in shared_placeholder_set:
                shared_placeholder_strings += self.background_strings

            for placeholder_string in shared_placeholder_strings:
                    if placeholder_string in self.subject_strings:
                        if 'pooler' in shared_embedder_components:
                            self.string_to_ada_embedder_dict[placeholder_string].poolers = first_subj_ada.poolers
                            subj_components_share_count['pooler'] += 1
                        if 'layer_coeff_maps' in shared_embedder_components:
                            self.string_to_ada_embedder_dict[placeholder_string].layer_coeff_maps = first_subj_ada.layer_coeff_maps
                            subj_components_share_count['layer_coeff_maps'] += 1
                        if 'basis_weights' in shared_embedder_components:
                            self.string_to_static_embedder_dict[placeholder_string].basis_rand_weights = first_subj_static_embedder.basis_rand_weights
                            self.string_to_static_embedder_dict[placeholder_string].basis_comm_weights = first_subj_static_embedder.basis_comm_weights
                            subj_components_share_count['basis_weights'] += 1
                    else:
                        if 'pooler' in shared_embedder_components:
                            self.string_to_ada_embedder_dict[placeholder_string].poolers = first_bg_ada.poolers
                            bg_components_share_count['pooler'] += 1
                        if 'layer_coeff_maps' in shared_embedder_components:
                            self.string_to_ada_embedder_dict[placeholder_string].layer_coeff_maps = first_bg_ada.layer_coeff_maps
                            bg_components_share_count['layer_coeff_maps'] += 1
                        if 'basis_weights' in shared_embedder_components:
                            self.string_to_static_embedder_dict[placeholder_string].basis_rand_weights = first_bg_static_embedder.basis_rand_weights
                            self.string_to_static_embedder_dict[placeholder_string].basis_comm_weights = first_bg_static_embedder.basis_comm_weights
                            bg_components_share_count['basis_weights'] += 1

            for component in shared_embedder_components:
                print(f"Shared {component} for {subj_components_share_count[component]} subject tokens and {bg_components_share_count[component]} background tokens")
        else:
            print("Not sharing any embedder components")

    def freeze_embedder_components(self, frozen_placeholder_set, frozen_embedder_components):
        if isinstance(frozen_placeholder_set, str):
            frozen_placeholder_set = frozen_placeholder_set.split(",")
        if isinstance(frozen_embedder_components, str):
            frozen_embedder_components = frozen_embedder_components.split(",")

        self.frozen_placeholder_set      = frozen_placeholder_set
        self.frozen_embedder_components  = frozen_embedder_components

        if frozen_placeholder_set is not None and frozen_embedder_components is not None:
            components_frozen_count     = dict([(component, 0) for component in frozen_embedder_components])
            frozen_placeholder_strings  = []

            if 'subj' in frozen_placeholder_set:
                frozen_placeholder_strings += self.subject_strings
            if 'bg'   in frozen_placeholder_set:
                frozen_placeholder_strings += self.background_strings

            for placeholder_string in frozen_placeholder_strings:
                ada_embedder = self.string_to_ada_embedder_dict[placeholder_string]
                static_embedder = self.string_to_static_embedder_dict[placeholder_string]

                if 'pooler' in frozen_embedder_components:
                    for pooler in ada_embedder.poolers:
                        for param in pooler.parameters():
                            param.requires_grad = False
                        components_frozen_count['pooler'] += 1

                if 'layer_coeff_maps' in frozen_embedder_components:
                    for param in ada_embedder.layer_coeff_maps.parameters():
                        param.requires_grad = False
                    components_frozen_count['layer_coeff_maps'] += 1

                if 'basis_weights' in frozen_embedder_components:
                    static_embedder.basis_rand_weights.requires_grad = False
                    static_embedder.basis_comm_weights.requires_grad = False
                    components_frozen_count['basis_weights'] += 1

            for component in frozen_embedder_components:
                print(f"Froze {components_frozen_count[component]} {component}s for {frozen_placeholder_strings}")

        else:
            print("Not freezing any embedder components")

    def perturb_model_parameters(self, perturb_ratio=0.2):
        param_group_list = self.optimized_parameters()
        num_perturbed_params = 0
        for param_group in param_group_list:
            for param in param_group['params']:
                if param.requires_grad:
                    # 0.5 -> uniform in [0.4, 0.7]. Inject randomness to reduce overfitting.
                    perturbation = torch_uniform(1 - perturb_ratio, 1 + perturb_ratio, 
                                                 param.shape, device=param.device)
                    param.data = param.data * perturbation
                    num_perturbed_params += 1
        
        print(f"Perturbed {num_perturbed_params} parameters with range = ({1 - perturb_ratio}, {1 + perturb_ratio})")

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def optimized_parameters(self):
        shared_embedder_param_list = []

        if self.shared_embedder_components is not None:
            # Only get the parameters of the first subject and background embedders.
            placeholder_strings = [ self.subject_strings[0], self.background_strings[0] ]

            for placeholder_string in placeholder_strings:
                ada_embedder             = self.string_to_ada_embedder_dict[placeholder_string]
                static_embedder = self.string_to_static_embedder_dict[placeholder_string]

                if 'pooler' in self.shared_embedder_components:
                    shared_embedder_param_list += list(ada_embedder.poolers.parameters())
                if 'layer_coeff_maps' in self.shared_embedder_components:
                    shared_embedder_param_list += list(ada_embedder.layer_coeff_maps.parameters())
                if 'basis_weights' in self.shared_embedder_components:
                    shared_embedder_param_list += [ static_embedder.basis_rand_weights, static_embedder.basis_comm_weights ]

        shared_embedder_param_ids =  { id(p) for p in shared_embedder_param_list }

        # The LR of the poolers is ~ 1/sqrt(N), where N is the number of subjects.
        # This is to slow down pooler update when we do multi-subject training.
        # Only applicable to AdamW training, and not for Prodigy.
        shared_embedder_params = [ { 'params': shared_embedder_param_list, 'lr_ratio': 1. / np.sqrt(len(self.subject_strings)),
                                     'excluded_from_prodigy': False } ]

        private_embedder_params = [ p for p in self.string_to_static_embedder_dict.parameters() if id(p) not in shared_embedder_param_ids ] \
                                  + [ p for p in self.string_to_ada_embedder_dict.parameters() if id(p) not in shared_embedder_param_ids ]

        # self.initial_embeddings and self.placeholder_to_emb_cache are not included 
        # in the optimized parameters.
        private_params_list = private_embedder_params \
                             + list(self.string_to_emb_ema_dict.parameters()) 

        private_params  = [ { 'params': private_params_list, 'lr_ratio': 1, 
                              'excluded_from_prodigy': False } ]
        # For unknown reason, emb_global_scale_scores are not aggressively optimized by Prodigy.
        slow_params_incl_prodigy = [ { 'params': [ self.emb_global_scale_scores ],   'lr_ratio': 0.1,
                                       'excluded_from_prodigy': False } ]
        
        if self.conv_attn_layerwise_scale_learnable: 
            ## Prodigy is too aggressive on the conv attn layerwise scales. 
            ## So don't train them if using Prodigy.
            slow_params_excl_prodigy = [ { 'params': list(self.subj2conv_attn_layerwise_scales.parameters()),
                                           'lr_ratio': 0.1, 'excluded_from_prodigy': True } ]
        else:
            slow_params_excl_prodigy = []

        if self.do_zero_shot:
            subj_basis_generator_params = [ { 'params': list(self.subj_basis_generator.parameters()), 
                                              'lr_ratio': 1, 'excluded_from_prodigy': False } ]
        else:
            subj_basis_generator_params = []

        return shared_embedder_params + private_params + subj_basis_generator_params \
               + slow_params_incl_prodigy + slow_params_excl_prodigy

    def embedding_attractor_loss(self):
        loss = 0.
        num_placeholders = len(self.placeholder_strings)

        for key in self.placeholder_strings:
            optimized = self.string_to_static_embedder_dict[key]
            coarse = self.initial_embeddings[key]
            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_placeholders

        return loss

    def layerwise_embedding_norm_loss(self):
        loss_static = 0.
        loss_ada    = 0.
        euc_loss_type   = 'l2'       # l1, l2. l2 is recommended.

        # Update: revert the reduction of the reg weight of the global bias, although 
        # the common practice is no regularization on biases.
        # If bias_reg_weight_base = 0.01, bias (= static part in ada embeddings) 
        # becomes too large (1~2), and may overpower the dynamic part.
        bias_reg_weight_base        = 0.1
        basis_reg_weight_base       = 0.1
        # We've applied LayerNorm to zs basis_generator pos_emb, so we don't need to regularize it.
        zs_basis_gennerator_pos_emb_weight = 0 #0.2
        ada_maps_weight_reg_weight  = 0.05
        ada_maps_bias_reg_weight    = 0 #0.001   # 0.001 -> 0
        pre_vecs_reg_weight         = 0.05
        static_l2_loss_boost        = 5
        ada_static_loss_boost_ratio = 2
        ada_l2_loss_boost           = static_l2_loss_boost * ada_static_loss_boost_ratio

        ada_attn_poolers_reg_weight = 0.02

        # Dynamically adjust the regularization weights. The larger the norm, the larger the weight.
        # T: temperature. Larger T => when norm grows, the penalty is more severe.
        T = 1.5
        num_out_embeddings  = 0

        for key in self.placeholder_strings:
            for embobj in (self.string_to_static_embedder_dict[key], 
                           self.string_to_ada_embedder_dict[key]):
                # Skip non-layerwise embeddings.
                if not isinstance(embobj, (StaticLayerwiseEmbedding, AdaEmbedding)):
                    continue
                
                # init_vecs is used to regularize pre_vecs.
                # init_vecs could be scalar 0, if initial embeddings are not specified.
                # In this case, loss_pre_vecs becomes the zero-centered attractor loss.
                # init_vecs, init_word_embeddings: [N, 768].
                init_vecs = self.initial_embeddings[key]

                # bias, pre_vecs, basis_vecs exist in 
                # both StaticLayerwiseEmbedding and AdaEmbedding.
                # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
                # pre_vecs are updated through BP. Therefore, loss_pre_vecs prevents pre_vecs 
                # from drifting too far from init_vecs.
                # NOTE: If self.do_zero_shot, then AdaEmbedding.bias, AdaEmbedding.basis_vecs are
                # generated by the basis_generator, and are regularized by the losses below.
                # But StaticLayerwiseEmbedding.basis_vecs are absent and are not regularized.
                # StaticLayerwiseEmbedding.bias is the embeddings generated by the basis_generator,
                # and are regularized.
                if embobj.has_bias and isinstance(embobj.bias, (torch.Tensor, nn.Parameter)):
                    loss_bias        = reg_loss(embobj.bias, loss_type=euc_loss_type)
                    # bias_reg_weight is computed dynamically. But it's detached from the graph.
                    # embobj.bias: now [Layers, K, 768].
                    bias_reg_weight  = bias_reg_weight_base  * torch.norm(embobj.bias, dim=-1).mean().item() ** T
                else:
                    loss_bias        = 0.
                    bias_reg_weight  = 0.

                if embobj.basis_vecs is not None:
                    loss_basis       = reg_loss(embobj.basis_vecs, loss_type=euc_loss_type)
                    # Like bias_reg_weight, basis_reg_weight is also computed dynamically 
                    # and detached from the graph.
                    # basis_vecs: now [K, r-N, 768] for Ada embedder, or [r-N, 768] for Static embedder.
                    basis_reg_weight = basis_reg_weight_base * torch.norm(embobj.basis_vecs, dim=-1).mean().item() ** T
                else:
                    loss_basis       = 0.
                    basis_reg_weight = 0.

                # N: number of pre_vecs (init_vecs). N > 0 implies (init_vecs is not None).
                if not self.do_zero_shot and embobj.N > 0 and pre_vecs_reg_weight > 0:
                    # pre_vecs has a K dim: [K, N, 768].
                    # init_vecs: [N, 768] => [1, N, 768].
                    init_vecs = init_vecs.unsqueeze(0)
                    # Shape inconsistency happens when we load a checkpoint 
                    # that is trained for a different subject as initialization.
                    if embobj.pre_vecs.shape[1] != init_vecs.shape[1]:
                        min_N = min(embobj.pre_vecs.shape[1], init_vecs.shape[1])
                    else:
                        min_N = embobj.pre_vecs.shape[1]
                        
                    loss_pre_vecs = reg_loss(embobj.pre_vecs[:, :min_N] - init_vecs[:, :min_N], 
                                             loss_type=euc_loss_type)
                else:
                    loss_pre_vecs = 0.

                loss_ada_maps_weight = 0.
                loss_ada_maps_bias   = 0.
                loss_ada_attn_pooler = 0.
                loss_ada_chan_weights_map_weight = 0.
                loss_ada_chan_weights_map_bias = 0.

                if isinstance(embobj, AdaEmbedding):
                    for i, map in enumerate(embobj.layer_coeff_maps):
                        loss_ada_maps_weight += reg_loss(map.weight, loss_type=euc_loss_type)
                        loss_ada_maps_bias   += reg_loss(map.bias,   loss_type=euc_loss_type)
                    if self.ada_uses_attn_pooler:
                        for i, pooler in enumerate(embobj.poolers):
                            loss_ada_attn_pooler  += reg_loss(pooler.lora_to_k.weight, loss_type=euc_loss_type)
                            loss_ada_attn_pooler  += reg_loss(pooler.lora_to_fg_q.weight, loss_type=euc_loss_type)
                            loss_ada_attn_pooler  += reg_loss(pooler.lora_to_bg_q.weight, loss_type=euc_loss_type)

                if type(loss_bias) == int:
                    breakpoint()

                num_out_embeddings += embobj.K

                # The losses of most components are times embobj.K, except for ada attn pooler, 
                # whose parameters don't get inflated with K.
                curr_loss = (  loss_bias             * bias_reg_weight \
                             + loss_basis            * basis_reg_weight \
                             + loss_pre_vecs         * pre_vecs_reg_weight \
                             + loss_ada_maps_weight  * ada_maps_weight_reg_weight \
                             + loss_ada_maps_bias    * ada_maps_bias_reg_weight ) * embobj.K \
                             + loss_ada_attn_pooler  * ada_attn_poolers_reg_weight \
                             + loss_ada_chan_weights_map_weight * ada_maps_weight_reg_weight \
                             + loss_ada_chan_weights_map_bias   * ada_maps_bias_reg_weight
                
                debug = False
                if debug and self.loss_call_count % 100 == 0:
                    print_str = f'reg_bias={loss_bias.item():.4f}, ' \
                                f'reg_basis={loss_basis.item():.4f}, '
                    if embobj.N > 0:
                        print_str += f'reg_pre_vecs={loss_pre_vecs.item():.4f}, '

                    if isinstance(embobj, AdaEmbedding):
                        print_str += f'loss_ada_maps_weight={loss_ada_maps_weight.item():.4f}, ' \
                                     f'loss_ada_maps_bias={loss_ada_maps_bias.item():.4f}'

                    print(print_str)

                if isinstance(embobj, StaticLayerwiseEmbedding):
                    loss_static = loss_static + curr_loss * static_l2_loss_boost
                else:
                    loss_ada = loss_ada + curr_loss * ada_l2_loss_boost

        # emb_reg_loss_scale is set to 1 by default.
        # If loading a ckpt trained on a different subject, emb_reg_loss_scale is set to 0.1,
        # to avoid emb_reg_loss becoming too large.
        # The equation below is actually: loss_static * emb_reg_loss_scale / (num_out_embeddings / 2).
        # num_out_embeddings counts both the embeddings of the static part and the dynamic part.
        # It's the twice of the actual number of embeddings. So divide by 2.
        loss_static *= self.emb_reg_loss_scale * 2 / num_out_embeddings
        loss_ada    *= self.emb_reg_loss_scale * 2 / num_out_embeddings
        
        '''
        if self.do_zero_shot and self.subj_basis_generator.pos_emb is not None:
            # Without this reg loss, the pos_emb of the basis generator may become too large 
            # and dominate the generated bases, leading to overfitting.
            loss_zs_basis_gennerator_pos_emb = reg_loss(self.subj_basis_generator.pos_emb, loss_type=euc_loss_type)
            loss_static += loss_zs_basis_gennerator_pos_emb * zs_basis_gennerator_pos_emb_weight
        '''

        return loss_static, loss_ada

    def embedding_reg_loss(self):
        self.loss_call_count += 1
        if self.use_layerwise_embedding:
            return self.layerwise_embedding_norm_loss()
        else:
            return self.embedding_attractor_loss()

    def calc_fg_bg_token_embs_ortho_loss(self, fg_bg_string_lists=None, ada_grad_scale=0.1, fg_grad_scale=0.5):
        if fg_bg_string_lists is None:
            fg_bg_string_lists = list(filter(lambda k: k not in self.background_string_dict, self.static_subj_embs_dict)), \
                                 list(filter(lambda k: k in self.background_string_dict, self.static_subj_embs_dict))
            #print(fg_bg_string_lists)

        loss_fg_bg_token_emb_ortho = 0.
        num_fg_bg_pairs = 0
        # Smaller grad scale for ada embeddings, because they are volatile and the gradients are noisy.
        ada_grad_scaler = gen_gradient_scaler(ada_grad_scale)

        # Sum of pairwise fg/bg token embedding ortho losses.
        for fg_string in fg_bg_string_lists[0]:
            for bg_string in fg_bg_string_lists[1]:
                fg_static_token_emb         = self.static_subj_embs_dict[fg_string]
                bg_static_token_emb         = self.static_subj_embs_dict[bg_string]

                try:
                    # fg_static_token_emb: [16, 9, 768]. 16: num layers. 9: num of vectors.
                    # It's the static token embeddings (not static prompt embeddings) of fg_string.
                    # fg_ada_token_emb_cache_obj: an Embedding3d object that's primarily used 
                    # to compute EMA embeddings.
                    # It stores the ada token embeddings (not ada prompt embeddings) of fg_string.
                    fg_ada_token_emb_cache_obj  = self.ada_subj_embs_dict[fg_string]
                    bg_ada_token_emb_cache_obj  = self.ada_subj_embs_dict[bg_string]
                except KeyError:
                    continue

                if len(fg_ada_token_emb_cache_obj.cached_layers) == fg_ada_token_emb_cache_obj.num_layers:
                    # fg_ada_token_emb: [16, 9, 768]. 16: num layers. 9: num vectors.
                    fg_ada_token_emb = fg_ada_token_emb_cache_obj.embedding
                else:
                    # Shouldn't reach here. 
                    # ada_subj_embs_dict is only cleared and updated after the next call to gen_ada_embedding().
                    fg_ada_token_emb = 0
                    breakpoint()

                if len(bg_ada_token_emb_cache_obj.cached_layers) == bg_ada_token_emb_cache_obj.num_layers:
                    bg_ada_token_emb = bg_ada_token_emb_cache_obj.embedding
                else:
                    # Shouldn't reach here.
                    # ada_subj_embs_dict is only cleared and updated after the next call to gen_ada_embedding().
                    bg_ada_token_emb = 0
                    breakpoint()

                # fg_hybrid_token_emb: [16, 9, 768]. 16: num layers. 9: num vectors.
                # bg_hybrid_token_emb: [16, 4, 768]. 16: num layers. 4: num vectors.
                # fg_ada_token_emb/bg_ada_token_emb are volatile and the gradients are noisy. 
                # So we scale down their gradients to 0.1.
                fg_hybrid_token_emb = fg_static_token_emb * (1 - self.ada_emb_weight) \
                                        + ada_grad_scaler(fg_ada_token_emb) * self.ada_emb_weight
                bg_hybrid_token_emb = bg_static_token_emb * (1 - self.ada_emb_weight) \
                                        + ada_grad_scaler(bg_ada_token_emb)  * self.ada_emb_weight
                
                '''
                fg_hybrid_token_emb = fg_static_token_emb
                bg_hybrid_token_emb = bg_static_token_emb
                '''

                # The embeddings are token embeddings, not prompt embeddings. 
                # So clamp_prompt_embedding() is not applicable.
                #fg_hybrid_token_emb, bg_hybrid_token_emb = \
                #    clamp_prompt_embedding(self.prompt_embedding_clamp_value, fg_hybrid_token_emb, bg_hybrid_token_emb)
                
                # fg_hybrid_token_emb, bg_hybrid_token_emb: [16, 768]. 16: num layers.
                fg_hybrid_token_mean_emb = fg_hybrid_token_emb.mean(dim=1)
                bg_hybrid_token_mean_emb = bg_hybrid_token_emb.mean(dim=1)

                loss_fg_bg_pair_token_emb_ortho = \
                    calc_ref_cosine_loss(bg_hybrid_token_mean_emb, fg_hybrid_token_mean_emb, 
                                         exponent=2, do_demean_first=False,
                                         first_n_dims_to_flatten=1, 
                                         ref_grad_scale=fg_grad_scale,
                                         aim_to_align=False)
                
                loss_fg_bg_token_emb_ortho += loss_fg_bg_pair_token_emb_ortho
                num_fg_bg_pairs += 1
        
        if num_fg_bg_pairs == 0:
            return 0.
        else:
            return loss_fg_bg_token_emb_ortho / num_fg_bg_pairs
    
if __name__ == '__main__':
    # The example code below is obsolete.    
    attnpool = AttentionalPooler()
    x = torch.randn(2, 768, 16, 16)
    mask = torch.randint(2, size=(2, 1, 16, 16)).float()
    y = attnpool(x, x, mask)
    print(y.shape)
