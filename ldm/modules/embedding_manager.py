import torch
from torch import nn, einsum
import torch.distributed as dist
from einops import rearrange, repeat
from adaface.subj_basis_generator import SubjBasisGenerator
import sys
sys.modules['ldm.modules.subj_basis_generator'] = sys.modules['adaface.subj_basis_generator']
sys.modules['ldm.modules.arc2face_models']      = sys.modules['adaface.arc2face_models']
from adaface.util import arc2face_forward_face_embs

import torch.nn.functional as F
import numpy as np

from ldm.util import gen_gradient_scaler, extract_first_index_in_each_instance, \
                     anneal_add_noise_to_embedding, calc_ref_cosine_loss, \
                     get_clip_tokens_for_string, get_embeddings_for_clip_tokens, \
                     scan_cls_delta_strings, torch_uniform, \
                     extend_clip_text_embedder, calc_init_word_embeddings, calc_stats
                     
from functools import partial
from collections import OrderedDict
import copy

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
    # adaface_subj_embs: [BS, 16, K, 768]. 
    # 16: number of layers. K: number of vectors per token. 
    def forward(self, adaface_subj_embs=None):
        with torch.autocast(device_type=self.device_type, enabled=False):
            # self.basis_comm_weights: [1, K, r] broadcasted to [16, K, r].
            basis_weights   = self.basis_rand_weights   + self.basis_comm_weights
            # torch.matmul: matrix multiplication.
            # torch.matmul(lora_up, basis_vecs): 16 * 768.

            if self.do_zero_shot:
                # (i0, e0), (i0, e1), ..., (i0, e15), (i1, e0), (i1, e1), ..., (i1, e15), ..., (iB, e15).
                adaface_subj_embs = rearrange(adaface_subj_embs, 'b l k d -> (b l) k d')
                # Copy to bias, so that adaface_subj_embs is regularized by layerwise_embedding_norm_loss().
                self.bias = adaface_subj_embs
                # Make sure adaface_subj_embs is regularized by layerwise_embedding_norm_loss().
                self.has_bias = True
                return adaface_subj_embs
            
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
        if self.do_zero_shot:
            assert has_bias, "Zero-shot AdaEmbedding must set has_bias=True"

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
            # Scale down the gradient to basis_dyn_coeffs, i.e., update layer_coeff_maps more slowly.
            # Update: change the gradient scale to 1 so that disabling the scaling, 
            # as the gradient on basis_dyn_coeffs are only around 1e-7.
            self.basis_dyn_coeffs_scaler = gen_gradient_scaler(1, debug=False) #True)
            
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
            self.bias = None

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
    # zs_basis_vecs: [BS, K, r, 768]. K: number of vectors per token. r: number of basis vectors for each vector.                    
    def forward(self, layer_idx, layer_attn_components, time_emb, 
                layer_subj_emb_probe, layer_static_extra_emb_mean, 
                img_mask=None, cached_pooler_bg_out=None, 
                zs_basis_vecs=None):
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
                basis_dyn_coeffs = self.basis_dyn_coeffs_scaler(basis_dyn_coeffs)
            else:
                # self.N: number of pre_vecs.
                if self.N > 0:
                    # pre_vecs:   [K, N, 768], basis_vecs: [K, r - N, 768]. 
                    # basis_vecs: [K, r, 768].
                    basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=1)
                else:
                    basis_vecs = self.basis_vecs
                basis_vecs = basis_vecs.unsqueeze(0)

            out_lns = self.layers_out_lns[ca_layer_idx]
            # out_vecs_unnorm: K elements, each is [BS, 1, r] x [BS, r, 768]_k = [BS, 1, 768].
            out_vecs_unnorm = [ torch.matmul(basis_dyn_coeffs[:, k].unsqueeze(1), basis_vecs[:, k]) for k in range(self.K) ]            
            out_vecs0 = torch.cat([ out_lns[k](out_vecs_unnorm[k]) for k in range(self.K) ], dim=1)
            # out_emb_dim: 768.
            out_vecs0 = out_vecs0 / np.sqrt(self.out_emb_dim)

            if self.has_bias and not self.do_zero_shot:
                # bias: [1, K, 768]
                bias = self.bias[ca_layer_idx].unsqueeze(0)
            else:
                bias = 0

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

# Initialize static embedders.
def create_static_embedders(out_emb_dim, num_layers_per_embedder, num_vectors_per_subj_token, 
                            layerwise_lora_rank, initializer_string, init_word_embeddings, 
                            init_word_weights, placeholder_string, avg_init_word_embedding_3d, 
                            do_zero_shot):
    # A static embedder can generate K embeddings.
    # layerwise_lora_rank > 0 implies use_layerwise_embedding.
    if layerwise_lora_rank > 0:
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
    else:
        # Degenerate to Textual Inversion. 
        # ANCHOR[id=init_embed] : 16*K vectors are initialized with the same embedding.
        token_static_embedder   = nn.Parameter(avg_init_word_embedding_3d, requires_grad=True)
        print("Warning: Degenerate to Textual Inversion.")

    return token_static_embedder

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
            subj_name_to_cls_delta_string=None,
            subj_name_to_cls_delta_word_weights=None,
            # token2num_vectors: how many vectors in each layer are allocated to model 
            # the subject (represented as the subject token) and the background. 
            # token2num_vectors is a dict.
            token2num_vectors={},
            skip_loading_token2num_vectors=False,
            use_layerwise_embedding=True,
            out_emb_dim=768,
            num_unet_ca_layers=16,
            layerwise_lora_rank=10,
            layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            training_begin_add_noise_std_range=None,
            training_end_add_noise_std_range=None,
            training_add_noise_prob=None,
            use_conv_attn_kernel_size=-1,
            do_zero_shot=True,
            zs_image_emb_dim=1024,
            subj_name_to_being_faces=None,   # subj_name_to_being_faces: a dict that maps subject names to is_face.
            zs_cls_delta_string='person',
            zs_cls_delta_token_weights=None,
            zs_prompt2token_proj_grad_scale=0.4,
            zs_load_subj_basis_generators_from_ckpt=True,
            zs_extra_words_scale=0.5,
            # During inference, zs_prompt2token_proj_ext_attention_perturb_ratio is not specified. 
            # Therefore no perturbation during inference.
            zs_prompt2token_proj_ext_attention_perturb_ratio=0, 
            zs_adaface_prompt_embs_inf_type='full_half_pad',
            # A few args, like embedding_manager_ckpt, are used in ddpm.py, but ignored here.
            **kwargs
    ):
        super().__init__()

        self.do_zero_shot = do_zero_shot

        self.string_to_token_dict = OrderedDict()
        
        self.string_to_static_embedder_dict      = nn.ParameterDict()
        self.string_to_subj_basis_generator_dict = nn.ModuleDict()
        self.initial_embeddings                  = nn.ParameterDict() # These should not be optimized
        self.placeholder_to_emb_cache            = nn.ParameterDict() # These should not be optimized
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

        self.set_training_add_noise_specs(training_begin_add_noise_std_range, 
                                          training_end_add_noise_std_range,
                                          training_add_noise_prob)
        
        self.set_conv_attn_kernel_size(use_conv_attn_kernel_size)

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
        # and AdaFace also supports it for more expressive modeling.
        self.skip_loading_token2num_vectors = skip_loading_token2num_vectors
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
        self.get_embeddings_for_tokens = get_embeddings_for_tokens
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

            self.zs_cls_delta_string   = zs_cls_delta_string
            self.zs_prompt2token_proj_grad_scale = zs_prompt2token_proj_grad_scale
            self.zs_load_subj_basis_generators_from_ckpt = zs_load_subj_basis_generators_from_ckpt
            self.zs_extra_words_scale = zs_extra_words_scale

            self.arc2face_embs = None
            if zs_prompt2token_proj_grad_scale == 0:
                print("Warning: prompt2token_proj is frozen, so don't add noise to it.")
                self.zs_prompt2token_proj_ext_attention_perturb_ratio = 0
            else:
                self.zs_prompt2token_proj_ext_attention_perturb_ratio = zs_prompt2token_proj_ext_attention_perturb_ratio
            self.zs_adaface_prompt_embs_inf_type = zs_adaface_prompt_embs_inf_type
            # arc2face_text_encoder will be passed from ddpm.py after the Arc2FaceWrapper instance 
            # is initialized, so as to save some RAM.
            self.arc2face_text_encoder = None

            if self.zs_cls_delta_string is not None:
                self.zs_cls_delta_tokens   = get_tokens_for_string(zs_cls_delta_string)
                if zs_cls_delta_token_weights is None:
                    self.zs_cls_delta_token_weights = torch.ones(len(self.zs_cls_delta_tokens))
                    self.zs_cls_delta_token_weights[-1] = 2
                else:
                    self.zs_cls_delta_token_weights = torch.tensor(zs_cls_delta_token_weights, dtype=float)
                # The last word is the main word "man, woman, boy, girl" whose weight will be normalized to 1;
                # if there are any words before this word, their weights will be normalized to 0.25.
                self.zs_cls_delta_token_weights **= 2
                self.zs_cls_delta_token_weights /= self.zs_cls_delta_token_weights.max()
            else:
                self.zs_cls_delta_tokens = None
                self.zs_cls_delta_token_weights = None

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

            token_static_embedder = \
                create_static_embedders(out_emb_dim, self.num_unet_ca_layers, num_vectors_per_subj_token, layerwise_lora_rank, 
                                        initializer_string, init_word_embeddings, init_word_weights, 
                                        placeholder_string, avg_init_word_embedding_3d, do_zero_shot)

            # token_static_embedder: a StaticLayerwiseEmbedding object (when use_layerwise_embedding) or an embedding vector.
            # Pytorch >= 1.12.0 allows to put an nn.Module object into an nn.ParameterDict.
            self.string_to_static_embedder_dict[placeholder_string] = token_static_embedder

            if init_word_embeddings is not None:
                # initial_embeddings won't be optimized. Just used to compute the regularization loss.
                # Use nn.Parameter to put it on cuda. Not to use register_buffer(), since it's a dict.
                self.initial_embeddings[placeholder_string] = nn.Parameter(init_word_embeddings, requires_grad=False)
            else:
                self.initial_embeddings[placeholder_string] = None

            if self.do_zero_shot:
                # num_out_embs_per_layer: 16 if fg or 4 if bg. 
                num_out_embs_per_layer = self.number_vectors_each_subj if not placeholder_is_bg else self.num_vectors_each_bg

                subj_basis_generator = SubjBasisGenerator(num_out_embs_per_layer = num_out_embs_per_layer,
                                                          num_out_layers = self.num_unet_ca_layers,
                                                          # zs_image_emb_dim: laion: 1280, openai: 768.
                                                          image_embedding_dim = zs_image_emb_dim, 
                                                          output_dim = out_emb_dim,
                                                          placeholder_is_bg = placeholder_is_bg,
                                                          prompt2token_proj_grad_scale = self.zs_prompt2token_proj_grad_scale,
                                                          bg_prompt_translator_has_to_out_proj=False,
                                                          zs_extra_words_scale = self.zs_extra_words_scale)

                self.string_to_subj_basis_generator_dict[placeholder_string] = subj_basis_generator

        # Initialize self.subj_name_to_cls_delta_tokens and self.subj_name_to_cls_delta_token_weights.
        self.init_cls_delta_tokens(get_tokens_for_string, get_embeddings_for_tokens, 
                                   subj_name_to_cls_delta_string, subj_name_to_cls_delta_word_weights,
                                   zs_cls_delta_string)
        self.init_subj_name_to_being_faces(subj_name_to_being_faces)

        self.layer_idx = -1
        self.static_subj_embs_dict = {}   
        self.clear_prompt_adhoc_info()
        # 'recon_iter', 'compos_distill_iter', 'arc2face_inverse_clip_iter', 'arc2face_clip_iter', 'empty'.
        self.iter_type = None       
        if self.do_zero_shot:
            self.set_curr_batch_subject_names(["zs_default"], 'recon_iter')
        else:
            self.curr_batch_subj_names = []
            self.current_subj_name_to_cls_delta_tokens = {}
            self.cls_delta_strings = None

        self.img_mask = None
        self.loss_call_count = 0
        self.training_percent = 0
        # Store the text_embedder to compute the delta loss.
        self.text_embedder  = text_embedder
        self.tokenizer      = text_embedder.tokenizer
        self.emb_global_scales_dict = None
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

        # zs_image_feat_dict have three keys: 'subj', 'bg', 'id'.
        self.zs_image_feat_dict = {}
        self.zs_out_id_embs_scale_range = (1.0, 1.0)

        print("EmbeddingManager on subj={}, bg={} init with {} vec(s), layerwise_lora_rank={}".format(
               self.subject_strings, self.background_strings, self.token2num_vectors, str2lora_rank))
        
        # Add the search span by 1, just to be safe.
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN += 1
        print(f"CLS_DELTA_STRING_MAX_SEARCH_SPAN={self.CLS_DELTA_STRING_MAX_SEARCH_SPAN}")

    def init_cls_delta_tokens(self, get_tokens_for_string, get_embeddings_for_tokens, 
                              subj_name_to_cls_delta_string, subj_name_to_cls_delta_word_weights,
                              zs_cls_delta_string=None):
        if subj_name_to_cls_delta_string is None:
            subj_name_to_cls_delta_string = {}
        if subj_name_to_cls_delta_word_weights is None:
            subj_name_to_cls_delta_word_weights = {}
        if zs_cls_delta_string is not None:
            # During inference, subj_name_to_cls_delta_string contains 'zs_default' as the subject name, and maps
            # to zs_cls_delta_string, the default class delta string.
            subj_name_to_cls_delta_string['zs_default'] = zs_cls_delta_string
            subj_name_to_cls_delta_word_weights['zs_default'] = [1] * len(get_tokens_for_string(zs_cls_delta_string))

        # We don't know the gender of a random arc2face subject.
        subj_name_to_cls_delta_string['arc2face'] = 'person'
        subj_name_to_cls_delta_word_weights['arc2face'] = [1]

        self.subj_name_to_cls_delta_string  = subj_name_to_cls_delta_string
        self.subj_name_to_cls_delta_tokens  = {}
        self.subj_name_to_cls_delta_token_weights = {}
        self.CLS_DELTA_STRING_MAX_SEARCH_SPAN = 0

        # subj_name_to_cls_delta_word_weights is of type omegaconf. If without convertion to dict,
        # "subj_name_to_cls_delta_token_weights[subj_name] = cls_delta_token_weights" will throw an error.
        self.subj_name_to_cls_delta_token_weights = dict(subj_name_to_cls_delta_word_weights)

        for subj_name in self.subj_name_to_cls_delta_string:
            cls_delta_string = self.subj_name_to_cls_delta_string[subj_name]
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

    def init_subj_name_to_being_faces(self, subj_name_to_being_faces):
        # subj_name_to_being_faces: a dict that maps subject names to is_face.
        # subj_name_to_being_faces is used in ddpm.py and not here.
        self.subj_name_to_being_faces = subj_name_to_being_faces if subj_name_to_being_faces is not None \
                                            else {subj_name: True for subj_name in self.subject_strings}
        self.subj_name_to_being_faces['arc2face']   = True
        self.subj_name_to_being_faces['zs_default'] = True

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

        for k in static_subj_embs_dict:
            self.static_subj_embs_dict[k] = static_subj_embs_dict[k]

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

        adaface_prompt_embs = None
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

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
            # current_subj_name_to_cls_delta_tokens only contains the cls_delta_tokens of the current batch.
            if REAL_OCCURS_IN_BATCH < BS and self.CLS_DELTA_STRING_MAX_SEARCH_SPAN > 0 \
              and len(self.current_subj_name_to_cls_delta_tokens) > 0:
                cls_delta_string_indices = scan_cls_delta_strings(tokenized_text,
                                                                  placeholder_indices_1st,
                                                                  self.current_subj_name_to_cls_delta_tokens,
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
                    else:
                        zs_clip_features = zs_image_feat_dict['subj']

                    # zs_id_embs: [1, 512].
                    zs_id_embs = zs_image_feat_dict['id']
                    subj_basis_generator = self.string_to_subj_basis_generator_dict[placeholder_string]
                        
                    # Loaded pretrained IP-Adapter model weight. No need to update arc2face_text_encoder.
                    # So arc2face_text_encoder is frozen.
                    if self.do_zero_shot and not placeholder_is_bg and self.curr_subj_is_face:
                        self.check_arc2face_text_encoder(zs_id_embs.device)

                        with torch.no_grad():
                            # arc2face_embs: [BS, 77, 768]. arc2face_id_embs: [BS, 16, 768].
                            placeholder_arc2face_embs, arc2face_id_embs = \
                                    arc2face_forward_face_embs(self.tokenizer, self.arc2face_text_encoder, 
                                                               zs_id_embs, return_full_and_core_embs=True)
                    else:
                        placeholder_arc2face_embs = None
                        arc2face_id_embs = None

                    # zs_clip_features: [BS, 257, 1280]
                    # adaface_subj_embs:   [BS, 16, 16, 768] if fg, or [BS,  16, 4, 768] if bg.
                    # zs_id_embs: the low-level ID embeddings from FaceAnalysis. Not actually used.
                    adaface_subj_embs, placeholder_adaface_prompt_embs = \
                            subj_basis_generator(arc2face_id_embs,
                                                 zs_clip_features, zs_id_embs, 
                                                 # Only scale with the lower bound.
                                                 # TODO: more granular, layerwise scaling.
                                                 self.zs_out_id_embs_scale_range[0],
                                                 is_face=self.curr_subj_is_face,
                                                 is_training=self.training,
                                                 adaface_prompt_embs_inf_type=self.zs_adaface_prompt_embs_inf_type)
                    # In a mix prompt batch (either compos_distill_iter or recon_iter with delta loss), 
                    # REAL_OCCURS_IN_BATCH counts the number of subject-single and subject-comp instances.
                    # But adaface_subj_embs is generated from the subject-single instance only.
                    # Repeating at dim 0 is correct even if adaface_subj_embs has a batch size > 1:
                    # If the subject-single batch is like [s1, s2], then the repeated batch is [s1, s2, s1, s2], 
                    # matching the batch structure of (subject-single, subject-single, ...).
                    if adaface_subj_embs.shape[0] < REAL_OCCURS_IN_BATCH:
                        adaface_subj_embs_orig_bs = adaface_subj_embs.shape[0]
                        adaface_subj_embs = adaface_subj_embs.repeat(REAL_OCCURS_IN_BATCH // adaface_subj_embs.shape[0], 1, 1, 1)
                        #if rank == 0:
                        #    print(f"Repeat adaface_subj_embs from {adaface_subj_embs_orig_bs} to {REAL_OCCURS_IN_BATCH}.")

                    if self.do_zero_shot and not placeholder_is_bg:
                        if self.iter_type == 'arc2face_inverse_clip_iter' or self.iter_type == 'arc2face_clip_iter':
                            # NOTE: arc2face_embs is the Arc2Face forward embeddings, while 
                            # adaface_prompt_embs is the Arc2Face inverse embeddings.
                            # arc2face_embs: [BS, 77, 768].
                            self.arc2face_embs = placeholder_arc2face_embs
                        if self.iter_type == 'arc2face_inverse_clip_iter' or self.iter_type == 'compos_distill_iter':
                            assert placeholder_adaface_prompt_embs is not None
                            adaface_prompt_embs = placeholder_adaface_prompt_embs

                    # NOTE: the condition iter_type == 'compos_distill_iter' is vital, as a recon_iter with delta loss 
                    # also has the 4-type prompt structure.
                    # But we should NEVER replace the subject-single embeddings with the frozen ones, 
                    # because these embeddings are used to reconstruct the noisy image, and if replaced,
                    # the model will learn nothing from the recon loss.
                    # One potential issue is the delta loss may slowly degrade the identity information in the embeddings.
                    # So we will replace the subject-single embeddings when computing the delta loss in ddpm.py later.
                    if self.training and not placeholder_is_bg and self.iter_type in ['compos_distill_iter']: #, 'recon_iter']:
                        # compos_distill_iter is with same_subject_in_batch=True. 
                        # So zs_id_embs: [1, 512].
                        if zs_id_embs.shape[0] != 1:
                            breakpoint()
                        subj_basis_generator0 = self.frozen_string_to_subj_basis_generator_dict[placeholder_string]
                        with torch.no_grad():
                            # adaface_subj_embs0: ID embeddings from the frozen subj_basis_generator.
                            adaface_subj_embs0, placeholder_adaface_prompt_embs0 = \
                                    subj_basis_generator0(arc2face_id_embs, zs_clip_features, zs_id_embs, 
                                                          self.zs_out_id_embs_scale_range[0],
                                                          is_face=self.curr_subj_is_face,
                                                          is_training=self.training,
                                                          adaface_prompt_embs_inf_type=self.zs_adaface_prompt_embs_inf_type)
                            
                        # adaface_subj_embs0: [1, 16, 16, 768] -> [2, 16, 16, 768].
                        adaface_subj_embs0 = adaface_subj_embs0.repeat(REAL_OCCURS_IN_BATCH // 2, 1, 1, 1)
                        # adaface_subj_embs0: [2, 16, 16, 768] -> [32, 16, 768].
                        self.adaface_subj_embs0 = rearrange(adaface_subj_embs0, 'b l k d -> (b l) k d')
                        # Only replace the subject-single embeddings in the compos_distill_iter.
                        # Replace the the subj-single embeddings with frozen subject embeddings, which is the first 1/4
                        # of the whole batch, i.e., the first REAL_OCCURS_IN_BATCH // 2 embeddings.
                        if self.iter_type == 'compos_distill_iter':
                            NUM_HALF_SUBJS = REAL_OCCURS_IN_BATCH // 2
                            # Still allow a small inference from the updated subj-single embeddings, 
                            # maybe this will make the images more natural?
                            adaface_subj_embs[:NUM_HALF_SUBJS] = adaface_subj_embs0.to(adaface_subj_embs.dtype) * 0.9 \
                                                              + adaface_subj_embs[:NUM_HALF_SUBJS] * 0.1
                            
                            if rank == 0:
                                print(f"compos_distill_iter. Replace the first {REAL_OCCURS_IN_BATCH // 2} embeddings with the frozen embeddings.")
                else:
                    adaface_subj_embs = None

                # static_embedder essentially only does:
                # >> adaface_subj_embs = rearrange(adaface_subj_embs, 'b l k d -> (b l) k d')
                subj_static_embedding = static_embedder(adaface_subj_embs)
                subj_static_embedding = subj_static_embedding.to(embedded_text.dtype)
            else:
                # static_embedder is already the embedding vectors.
                subj_static_embedding = static_embedder

            static_subj_embs_dict[placeholder_string] = subj_static_embedding

            for k in range(self.token2num_vectors[placeholder_string]):
                # embedded_text is repeated 16 times along the layer dimension, with size of dim 0 = 16 * BS.
                # The result of the repeat is: the same instance is repeated 16 times, which are adjacent 
                # to each other across the batch dim:
                # [b1_l1, ..., b1_l16, b2_l1, ..., b2_l16, ..., bB_l1, ..., bB_l16].
                # {________b1________} {_______b2_______}  ...  {_______bB________}
                # The first dim of subj_static_embedding is the layer dim (size = 16). 
                # So we repeat the 16 layers of the k-th embedding, subj_static_embedding[:, k], 
                # REAL_OCCURS_IN_BATCH times, to match 16*REAL_OCCURS_IN_BATCH.
                # After repeat, the RHS is
                # [ek_l1, ..., ek_l16, ek_l1, ..., ek_l16, ..., ek_l1, ..., ek_l16].
                # {________b1________} {_______b2_______}  ...  {_______bB________}
                # During inference, BS = 1, subj_static_embedding_k: [16, 768]
                subj_static_embedding_k = subj_static_embedding[:, k]
                
                if self.training and self.training_begin_add_noise_std_range is not None:
                    # The std of subj_static_embedding is around 0.07, times training_end_add_noise_std_range
                    # (0.02 ~ 0.04) is very small. Therefore, it won't hurt the subject identity encoded
                    # in the embeddings.
                    subj_static_embedding_k = \
                        anneal_add_noise_to_embedding(subj_static_embedding_k, 
                                                      self.training_percent,
                                                      self.training_begin_add_noise_std_range,
                                                      self.training_end_add_noise_std_range,
                                                      self.training_add_noise_prob[self.iter_type],
                                                      noise_std_is_relative=True, keep_norm=False)

                # Training with delta loss. Each subject only appears once in subj_static_embedding, 
                # but twice in the prompts (subject single and subject comp), so we need to repeat it twice.
                if REAL_OCCURS_IN_BATCH == BS // 2 and subj_static_embedding_k.shape[0] == REAL_OCCURS_IN_BATCH // 2 * num_unet_ca_layers:
                    # subj_static_embedding_k: [48, 768] => [48*2, 768]
                    subj_static_embedding_k = subj_static_embedding_k.repeat(2, 1)
                # Single-subject batch. It's either during inference, or during training with same_subject_in_batch=True.
                # BS = 1, subj_static_embedding_k: [16, 768]
                # Each subject only appears once in subj_static_embedding, but BS == REAL_OCCURS_IN_BATCH
                # times in the prompts. Therefore, it's repeated REAL_OCCURS_IN_BATCH times.
                elif subj_static_embedding_k.shape[0] == num_unet_ca_layers:
                    # subj_static_embedding_k: [16, 768] => [16*REAL_OCCURS_IN_BATCH, 768]
                    subj_static_embedding_k = subj_static_embedding_k.repeat(REAL_OCCURS_IN_BATCH, 1)
                elif subj_static_embedding_k.shape[0] != num_unet_ca_layers * REAL_OCCURS_IN_BATCH:
                    breakpoint()
                # Otherwise, subj_static_embedding_k.shape[0] == num_unet_ca_layers * REAL_OCCURS_IN_BATCH,
                # i.e., the left and right sides will have the same number of identity embeddings, and we don't need to do anything.

                # Assign the k-th token embedding (along the text dim).
                placeholder_indices_k = (placeholder_indices_1st[0], placeholder_indices_1st[1] + k)
                embedded_text[placeholder_indices_k] = subj_static_embedding_k

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
        if self.do_zero_shot:
            if self.iter_type == 'arc2face_inverse_clip_iter':
                # In an arc2face_inverse_clip_iter, inversed arc2face prompt embeddings is used as the prompt embeddings.
                # The updated embedded_text above is ignored. But subj_static_embeddings is 
                # still involved in delta-loss computation.
                # embedded_text: [1, 77, 768]
                embedded_text = adaface_prompt_embs
            # NOTE: if self.iter_type == 'arc2face_clip_iter', we CANNOT return self.arc2face_embs
            # as the updated embedded_text, since the returned embedded_text will be encoded again by the text encoder.
            # Instead, we replace the prompt embeddings with the arc2face_embs in ddpm.py:get_learned_conditioning().

        return embedded_text, tokenized_text, static_subj_embs_dict

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
    
    # During training, set_curr_batch_subject_names() is called in ddpm.py.
    # During inference, set_curr_batch_subject_names() is called by the embedding manager.
    def set_curr_batch_subject_names(self, subj_names, embman_iter_type):
        self.curr_batch_subj_names = subj_names
        # During inference, as self.curr_batch_subj_names is not set, the three dicts are empty.
        self.current_subj_name_to_cls_delta_tokens = { subj_name: self.subj_name_to_cls_delta_tokens[subj_name] \
                                                       for subj_name in self.curr_batch_subj_names }
                
        # During training, we get the current subject name from self.curr_batch_subj_names, then map to 
        # curr_subj_is_face. 
        # During inference, curr_batch_subj_names = ['zs_default'], which maps to True in subj_name_to_being_faces,
        # so curr_subj_is_face == True.
        # BUG: if there are multiple subjects in the same batch, then is_face is only 
        # about the first subject. But now we only support one subject in a batch.
        if len(self.curr_batch_subj_names) > 0:
            self.curr_subj_is_face = self.subj_name_to_being_faces[self.curr_batch_subj_names[0]]

        if len(self.current_subj_name_to_cls_delta_tokens) > 0:
            self.cls_delta_strings       = [ self.subj_name_to_cls_delta_string[subj_name] \
                                             for subj_name in self.curr_batch_subj_names ]
        else:
            self.cls_delta_strings = None

        self.set_curr_iter_type(embman_iter_type)
        if True: #cls_delta_strings is not None and 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
            print(f"subjects:{self.curr_batch_subj_names}, cls_delta_strings: {self.cls_delta_strings}")

    def set_curr_iter_type(self, embman_iter_type):
        self.iter_type = embman_iter_type
        # In a compos_distill_iter, all subjects are the same. So we only keep the first cls_delta_string.
        if self.cls_delta_strings is not None and self.iter_type == 'compos_distill_iter':
            self.cls_delta_strings = self.cls_delta_strings[:1]
        
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

    def set_conv_attn_kernel_size(self, use_conv_attn_kernel_size=-1):
        # The first time use_conv_attn_kernel_size is set.
        if not hasattr(self, 'use_conv_attn_kernel_size'):
            self.use_conv_attn_kernel_size = use_conv_attn_kernel_size
        # use_conv_attn_kernel_size is the default value None.
        elif self.use_conv_attn_kernel_size is None:
            if use_conv_attn_kernel_size == -1:
                print("DISABLED: Setting use_conv_attn_kernel_size = -1")
                self.use_conv_attn_kernel_size = -1
            else:
                print(f"Setting use_conv_attn_kernel_size = {use_conv_attn_kernel_size}")
                self.use_conv_attn_kernel_size = use_conv_attn_kernel_size
        # use_conv_attn_kernel_size has been set before. Reject the new value.
        else:
            if self.use_conv_attn_kernel_size != use_conv_attn_kernel_size:
                print(f"use_conv_attn_kernel_size has been set to {self.use_conv_attn_kernel_size}, "
                       "refusing to change it to {use_conv_attn_kernel_size}")

    def check_arc2face_text_encoder(self, device):
        if self.arc2face_text_encoder is None:
            from adaface.arc2face_models import CLIPTextModelWrapper
            print("arc2face_text_encoder is None. Initialize it as a private copy.")
            self.arc2face_text_encoder = CLIPTextModelWrapper.from_pretrained(
                                            'models/arc2face', subfolder="encoder", torch_dtype=torch.float16
                                        )
            self.arc2face_text_encoder.to(device)
            
    def set_zs_image_features(self, zs_clip_features, zs_id_embs, zs_out_id_embs_scale_range=(1.0, 1.0), 
                              add_noise_to_zs_id_embs=False):
        # zs_clip_features: [1, 514, 1280]
        # zs_clip_subj_features, zs_clip_bg_features: [1, 257, 1280].
        # zs_id_embs: [1, 512]. 
        zs_clip_subj_features, zs_clip_bg_features = zs_clip_features.chunk(2, dim=1)

        # Add noise to all_id_embs during training with probability 0.5.
        # Noise level is gradually reduced from [0.01, 0.02] to [0.005, 0.01] during training.
        # Noise std is absolute, not relative (to the std of all_id_embs).
        # If add_noise_to_real_id_embs in ddpm.py, zs_id_embs has been added with noise of
        # std [0.02, 0.06]. Here we add noise again, but it's much smaller than the previous noise.
        # Therefore it doesn't matter.
        if self.training and add_noise_to_zs_id_embs:
            zs_id_embs = anneal_add_noise_to_embedding(zs_id_embs, self.training_percent,
                                                       begin_noise_std_range=[0.01, 0.02], 
                                                       end_noise_std_range  =[0.01, 0.02],
                                                       add_noise_prob=0.5, noise_std_is_relative=True,
                                                       keep_norm=True)

        self.zs_image_feat_dict = { 'subj': zs_clip_subj_features, 'bg': zs_clip_bg_features,
                                    'id':   zs_id_embs }
        self.zs_out_id_embs_scale_range = zs_out_id_embs_scale_range

        # Clear the basis_vecs and bias saved in embedders.
        for placeholder_string in self.placeholder_strings:
            self.string_to_static_embedder_dict[placeholder_string].basis_vecs = None
            self.string_to_static_embedder_dict[placeholder_string].bias       = None

    def set_num_vectors_per_subj_token(self, token2num_vectors):
        self.token2num_vectors = token2num_vectors
        print(f"Set token2num_vectors: {self.token2num_vectors}")

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, ckpt_path):
        torch.save({ "string_to_token":                  self.string_to_token_dict,
                     "string_to_static_embedder":        self.string_to_static_embedder_dict,
                     "string_to_subj_basis_generator_dict": self.string_to_subj_basis_generator_dict,
                     "token2num_vectors":                self.token2num_vectors,
                     "emb_global_scale_scores":          self.emb_global_scale_scores,
                     "use_conv_attn_kernel_size":        self.use_conv_attn_kernel_size,
                     "placeholder_strings":              self.placeholder_strings,
                     "subject_strings":                  self.subject_strings,
                     "background_strings":               self.background_strings,
                     "ca_q_bns":                         self.ca_q_bns,
                     "ca_outfeat_lns":                   self.ca_outfeat_lns,
                     "do_zero_shot":                     self.do_zero_shot,
                   }, 
                    ckpt_path)

    # Load custom tokens and their learned embeddings from "embeddings_gs-4500.pt".
    def load(self, ckpt_paths, extend_prompt2token_proj_attention_multiplier=-1, load_old_embman_ckpt=False):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        token2num_vectors                   = {}
        emb_global_scale_scores_dict        = {}        
        self.string_to_token_dict           = {}
        self.string_to_static_embedder_dict = nn.ParameterDict()

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

            if "background_strings" in ckpt:
                ckpt_background_strings = ckpt["background_strings"]
            else:
                ckpt_background_strings = []

            use_conv_attn_kernel_size   = ckpt.get("use_conv_attn_kernel_size", None)
            self.set_conv_attn_kernel_size(use_conv_attn_kernel_size)

            if "ca_q_bns" in ckpt:
                self.ca_q_bns = ckpt["ca_q_bns"]
            if "ca_outfeat_lns" in ckpt:
                self.ca_outfeat_lns = ckpt["ca_outfeat_lns"]

            # Only load subj_basis_generator from ckpt if the ckpt is set with the same do_zero_shot.
            if "do_zero_shot" in ckpt and self.do_zero_shot == ckpt["do_zero_shot"] and self.zs_load_subj_basis_generators_from_ckpt:
                for km, ckpt_subj_basis_generator in ckpt["string_to_subj_basis_generator_dict"].items():
                    # repr(ckpt_subj_basis_generator) will assign missing variables to ckpt_subj_basis_generator.
                    if load_old_embman_ckpt:
                        print(f"Loading ckpt_subj_basis_generator {km}")
                    else:
                        print(f"Loading {repr(ckpt_subj_basis_generator)}")

                    # self.string_to_subj_basis_generator_dict[km] is either not initialized, or initialized with a smaller depth.
                    # Then replace it with the one in ckpt.
                    # print(f"Overwrite {repr(self.string_to_subj_basis_generator_dict[km])}")
                    ckpt_subj_basis_generator.face_proj_in = None
                    
                    if load_old_embman_ckpt:
                        # Skip loadding lora2hira, latent_queries and layers, as the old ckpt has different shapes.
                        ckpt_subj_basis_generator.lora2hira = None
                        ckpt_subj_basis_generator.latent_queries = None
                        ckpt_subj_basis_generator.latent_query_lns = None
                        ckpt_subj_basis_generator.layers = None
                        ckpt_subj_basis_generator.obj_proj_in = None
                        ckpt_subj_basis_generator.proj_in = None
                        ckpt_subj_basis_generator.pos_embs = None
                        self.string_to_subj_basis_generator_dict[km].pos_embs.data.zero_()

                    # No extension. So we just assign the ckpt to the current subj_basis_generator.
                    # TODO: fix the logic below thoroughly in the future.
                    if extend_prompt2token_proj_attention_multiplier == -1:
                        self.string_to_subj_basis_generator_dict[km] = ckpt_subj_basis_generator
                        continue

                    # Compatible with older ckpts which only have per-layer hidden_state_layer_weights.
                    if (not ckpt_subj_basis_generator.placeholder_is_bg) \
                      and ckpt_subj_basis_generator.hidden_state_layer_weights.shape[-1] != self.string_to_subj_basis_generator_dict[km].hidden_state_layer_weights.shape[-1]:
                        if self.string_to_subj_basis_generator_dict[km].hidden_state_layer_weights.shape[-1] == 1:
                            # hidden_state_layer_weights: [3, 768] -> [3, 1]
                            ckpt_subj_basis_generator.hidden_state_layer_weights = nn.Parameter(ckpt_subj_basis_generator.hidden_state_layer_weights.mean(dim=1, keepdim=True))
                            print(f"Average along features: hidden_state_layer_weights -> {ckpt_subj_basis_generator.hidden_state_layer_weights.shape}")
                        else:
                            # hidden_state_layer_weights: [3, 1] -> [3, 768]
                            ckpt_subj_basis_generator.hidden_state_layer_weights = nn.Parameter(ckpt_subj_basis_generator.hidden_state_layer_weights.repeat(1, 768))
                            print(f"Expand along features:  hidden_state_layer_weights -> {ckpt_subj_basis_generator.hidden_state_layer_weights.shape}")

                    # ckpt_subj_basis_generator.prompt2token_proj hasn't been extended. 
                    # So only extend self.string_to_subj_basis_generator_dict[km] after loading the state_dict.
                    # This should happen only during training, not inference. 
                    # Therefore, whether noise_std is 0 or not doesn't really matter the inference result.
                    if ckpt_subj_basis_generator.placeholder_is_bg or (not hasattr(ckpt_subj_basis_generator, "prompt2token_proj_attention_multiplier")) \
                      or ckpt_subj_basis_generator.prompt2token_proj_attention_multiplier == -1:
                        # If extend_prompt2token_proj_attention_multiplier > 1, then after loading state_dict, extend the prompt2token_proj.
                        ret = self.string_to_subj_basis_generator_dict[km].load_state_dict(ckpt_subj_basis_generator.state_dict(), strict=False)
                        if not ckpt_subj_basis_generator.placeholder_is_bg and extend_prompt2token_proj_attention_multiplier > 1:
                            # -1, -1: extend all layers
                            self.string_to_subj_basis_generator_dict[km].extend_prompt2token_proj_attention(-1, -1,
                                                                                                            extend_prompt2token_proj_attention_multiplier,
                                                                                                            noise_std=self.zs_prompt2token_proj_ext_attention_perturb_ratio)

                    # placeholder is fg, and ckpt_subj_basis_generator.prompt2token_proj_attention_multiplier > 1,
                    # and extend_prompt2token_proj_attention_multiplier is either unspecified, or is multiple of ckpt.
                    elif extend_prompt2token_proj_attention_multiplier == -1 \
                      or extend_prompt2token_proj_attention_multiplier % ckpt_subj_basis_generator.prompt2token_proj_attention_multiplier == 0:
                        # Extend the CLIPAttention layers in the subj_basis_generator, before loading the state_dict.
                        # This means that during inference, we don't need to specify extend_prompt2token_proj_attention_multiplier.
                        # If the ckpt has an extended prompt2token_proj, then the subj_basis_generator's prompt2token_proj will be extended 
                        # before loading the state_dict.
                        # NOTE: This could happen either during training or inference. Since state_dict will be loaded,
                        # whether noise_std is 0 or not has no impact to the extended attention weights.
                        # -1, -1: extend all layers
                        self.string_to_subj_basis_generator_dict[km].extend_prompt2token_proj_attention(-1, -1,
                                                                                                        ckpt_subj_basis_generator.prompt2token_proj_attention_multiplier,
                                                                                                        noise_std=self.zs_prompt2token_proj_ext_attention_perturb_ratio)
                        ret = self.string_to_subj_basis_generator_dict[km].load_state_dict(ckpt_subj_basis_generator.state_dict(), strict=False)
                        if extend_prompt2token_proj_attention_multiplier > 0:
                            second_ext_multiplier = extend_prompt2token_proj_attention_multiplier // ckpt_subj_basis_generator.prompt2token_proj_attention_multiplier
                            # During this extension, the added noise does change the extra copies of attention weights, since they are not in the ckpt.
                            # During training,  zs_prompt2token_proj_ext_attention_perturb_ratio == 0.1.
                            # During inference, zs_prompt2token_proj_ext_attention_perturb_ratio == 0.
                            # All CLIP encoder layers are 0-11. 
                            # 0, 6: extend the first 6 layers 0-5 (not including layer 6).
                            # 0, 3: extend the first 3 layers 0-2 (not including layer 3).
                            self.string_to_subj_basis_generator_dict[km].extend_prompt2token_proj_attention(0, 3,
                                                                                                            second_ext_multiplier,
                                                                                                            noise_std=self.zs_prompt2token_proj_ext_attention_perturb_ratio)
                    # extend_prompt2token_proj_attention_multiplier is specified but inconsistent with ckpt, debug.
                    else:
                        breakpoint()

                    if len(ret.missing_keys) > 0:
                        print(f"Missing keys: {ret.missing_keys}")
                    if len(ret.unexpected_keys) > 0:
                        print(f"Unexpected keys: {ret.unexpected_keys}")

                    if self.zs_prompt2token_proj_grad_scale == 0:
                        # If it's for bg token, then freeze_prompt2token_proj() does nothing.
                        self.string_to_subj_basis_generator_dict[km].freeze_prompt2token_proj()

            else:
                print(f"Skipping loading subj_basis_generator from {ckpt_path}")

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

                if km in ckpt["token2num_vectors"]:
                    token2num_vectors[km2] = ckpt["token2num_vectors"][km]

                print(f"Loaded {km}->{km2} from {ckpt_path}")
                
                static_embedder = self.string_to_static_embedder_dict[km2]
                # Make compatible with older ckpts which don't have do_zero_shot.
                if 'do_zero_shot' not in static_embedder.__dict__:
                    static_embedder.do_zero_shot = self.do_zero_shot

            if "token2num_vectors" in ckpt and not self.skip_loading_token2num_vectors:
                self.set_num_vectors_per_subj_token(token2num_vectors)

        self.emb_global_scale_scores = nn.Parameter(torch.zeros(len(self.string_to_token_dict)), 
                                                    requires_grad=True)
        for token_idx, km in enumerate(self.string_to_token_dict):
            if km in emb_global_scale_scores_dict:
                self.emb_global_scale_scores.data[token_idx] = emb_global_scale_scores_dict[km]

        print(f"Set emb_global_scales = {self.get_emb_global_scales_dict(regen=True, do_perturb=False)}")

        self.placeholder_strings = self.subject_strings + self.background_strings
        if len(extended_token_embeddings) > 0:
            self.extended_token_embeddings = torch.cat(extended_token_embeddings, dim=0)
            print(f"Extended {len(extended_token_embeddings)} token embeddings")

        # Regenerate subject_string_dict, background_string_dict 
        # in case subject_strings or background_strings have been changed.
        self.subject_string_dict    = { s: True for s in self.subject_strings }
        self.background_string_dict = { s: True for s in self.background_strings }

    # make_frozen_copy_of_subj_basis_generators() is only used during training, to generate the subject embeddings for subject-single prompts.
    def make_frozen_copy_of_subj_basis_generators(self, dtype=torch.float16):
        # frozen_string_to_subj_basis_generator_dict won't be returned by optimized_parameters(),
        # so it won't be updated.
        self.frozen_string_to_subj_basis_generator_dict = copy.deepcopy(self.string_to_subj_basis_generator_dict)
        # Convert the frozen copy of subj_basis_generators to float16 to save RAM.
        self.frozen_string_to_subj_basis_generator_dict.to(dtype=dtype)
        print("Made a frozen copy of subj_basis_generators")

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
        # For unknown reason, emb_global_scale_scores are not aggressively optimized by Prodigy.
        slow_params_incl_prodigy = [ { 'params': [ self.emb_global_scale_scores ],   'lr_ratio': 0.1,
                                       'excluded_from_prodigy': False } ]
        slow_params_excl_prodigy = []

        if self.do_zero_shot:
            subj_basis_generator_param_list0 = list(self.string_to_subj_basis_generator_dict.parameters())
            subj_basis_generator_param_list = [ p for p in subj_basis_generator_param_list0 if p.requires_grad ]
            num_no_grad_params  = len(subj_basis_generator_param_list0) - len(subj_basis_generator_param_list)
            num_total_params    = len(subj_basis_generator_param_list0)
            print(f"Filtered out {num_no_grad_params} no-grad / {num_total_params} total parameters in subj_basis_generator_param_list0.")
            subj_basis_generator_params = [ { 'params': subj_basis_generator_param_list, 
                                              'lr_ratio': 1, 'excluded_from_prodigy': False } ]
        else:
            subj_basis_generator_params = []

        return subj_basis_generator_params + slow_params_incl_prodigy + slow_params_excl_prodigy

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
        euc_loss_type   = 'l2'       # l1, l2. l2 is recommended.

        # Update: revert the reduction of the reg weight of the global bias, although 
        # the common practice is no regularization on biases.
        # If bias_reg_weight_base = 0.01, bias becomes too large (1~2), 
        # and may overpower the dynamic part.
        bias_reg_weight             = 0.1
        basis_reg_weight            = 0.1
        pre_vecs_reg_weight         = 0.05
        static_l2_loss_boost        = 5
        num_out_embeddings          = 0

        for key in self.placeholder_strings:
            for embobj in (self.string_to_static_embedder_dict[key]):
                # Skip non-layerwise embeddings.
                if not isinstance(embobj, StaticLayerwiseEmbedding):
                    continue
                
                # init_vecs is used to regularize pre_vecs.
                # init_vecs could be scalar 0, if initial embeddings are not specified.
                # In this case, loss_pre_vecs becomes the zero-centered attractor loss.
                # init_vecs, init_word_embeddings: [N, 768].
                init_vecs = self.initial_embeddings[key]

                # bias, pre_vecs, basis_vecs exist in StaticLayerwiseEmbedding.
                # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
                # pre_vecs are updated through BP. Therefore, loss_pre_vecs prevents pre_vecs 
                # from drifting too far from init_vecs.
                # NOTE: If self.do_zero_shot, StaticLayerwiseEmbedding.basis_vecs are absent and are not regularized.
                # StaticLayerwiseEmbedding.bias is the embeddings generated by the basis_generator,
                # and are regularized.
                if embobj.has_bias and isinstance(embobj.bias, (torch.Tensor, nn.Parameter)):
                    # embobj.bias: now [Layers, K, 768].
                    loss_bias        = reg_loss(embobj.bias, loss_type=euc_loss_type)
                else:
                    loss_bias        = 0.

                if embobj.basis_vecs is not None:
                    # basis_vecs: [r-N, 768] for Static embedder.
                    loss_basis       = reg_loss(embobj.basis_vecs, loss_type=euc_loss_type)
                else:
                    loss_basis       = 0.

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

                num_out_embeddings += embobj.K

                curr_loss =    loss_bias             * bias_reg_weight \
                             + loss_basis            * basis_reg_weight \
                             + loss_pre_vecs         * pre_vecs_reg_weight 
                
                debug = False
                if debug and self.loss_call_count % 100 == 0:
                    print_str = f'reg_bias={loss_bias.item():.4f}, ' \
                                f'reg_basis={loss_basis.item():.4f}, '
                    if embobj.N > 0:
                        print_str += f'reg_pre_vecs={loss_pre_vecs.item():.4f}, '

                    print(print_str)

                loss_static = loss_static + curr_loss * static_l2_loss_boost

        loss_static /= num_out_embeddings

        return loss_static

    def embedding_reg_loss(self):
        self.loss_call_count += 1
        if self.use_layerwise_embedding:
            return self.layerwise_embedding_norm_loss()
        else:
            return self.embedding_attractor_loss()

    # NOTE: calc_fg_bg_token_embs_ortho_loss() is DISABLED if do_zero_shot but not same_subject_in_batch,
    # i.e., do_zero_shot and different subjects are sampled in the same batch,
    # because the fg/bg ortho loss with multiple subjects will be too complicated, esp. considering
    # that in a compositional iteration, only the first prompt (corresponding to the first subject) is active.
    def calc_fg_bg_token_embs_ortho_loss(self, fg_bg_string_lists=None, fg_grad_scale=0.5):
        if fg_bg_string_lists is None:
            fg_bg_string_lists = list(filter(lambda k: k in self.subject_string_dict,    self.static_subj_embs_dict)), \
                                 list(filter(lambda k: k in self.background_string_dict, self.static_subj_embs_dict))
            #print(fg_bg_string_lists)

        loss_fg_bg_token_emb_ortho = 0.
        num_fg_bg_pairs = 0

        # Sum of pairwise fg/bg token embedding ortho losses.
        for fg_string in fg_bg_string_lists[0]:
            for bg_string in fg_bg_string_lists[1]:
                fg_static_token_emb         = self.static_subj_embs_dict[fg_string]
                bg_static_token_emb         = self.static_subj_embs_dict[bg_string]

                # fg_hybrid_token_emb: [16, 9, 768]. 16: num layers. 9: num vectors.
                # bg_hybrid_token_emb: [16, 4, 768]. 16: num layers. 4: num vectors.
                fg_hybrid_token_emb = fg_static_token_emb
                bg_hybrid_token_emb = bg_static_token_emb
                
                '''
                fg_hybrid_token_emb = fg_static_token_emb
                bg_hybrid_token_emb = bg_static_token_emb
                '''

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
