import torch
from torch import nn, einsum
from einops import rearrange, repeat
from ldm.modules.ema import LitEma

import torch.nn.functional as F
import numpy as np

from ldm.util import masked_mean, gen_gradient_scaler, extract_first_index_in_each_instance, \
                     add_noise_to_embedding, calc_ref_cosine_loss

from functools import partial
from collections import OrderedDict
import random
import copy

# When debugging, make the printed tensors less messy.
torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

def get_clip_tokens_for_string(tokenizer, string, force_single_token=False):
    '''
    # If string is a new token, add it to the tokenizer.
    if string not in tokenizer.get_vocab():
        tokenizer.add_tokens([string])
        # tokenizer() returns [49406, 49408, 49407]. 
        # 49406: start of text, 49407: end of text, 49408: new token.
        new_token_id = tokenizer(string)["input_ids"][1]
        print("Added new token to tokenizer: {} -> {}".format(string, new_token_id))
    '''
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # [49406, 11781,  4668, 49407, 49407...]. 49406: start of text, 49407: end of text
    # 11781,  4668: tokens of "stuffed animal".
    token_count = torch.count_nonzero(tokens - 49407) - 1
    assert token_count >= 1, f"No token found in string '{string}'"
    if force_single_token:
        assert token_count == 1, f"String '{string}' maps to more than a single token. Please use another string"

    # Remove start and end tokens.
    return tokens[0, 1:1+token_count]

def get_bert_tokens_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embeddings_for_clip_tokens(embedder, tokens):
    # embedder(tokens): [1, N, 768]. N: number of tokens. 
    # RETURN: [N, 768]
    return embedder(tokens)[0]

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

def calc_stats(emb_name, embeddings):
    print("%s:" %emb_name)
    emb_mean = embeddings.mean(0, keepdim=True).repeat(embeddings.size(0), 1)
    l1_loss = F.l1_loss(embeddings, emb_mean)
    # F.l2_loss doesn't take sqrt. So the loss is very small. 
    # Compute it manually.
    l2_loss = ((embeddings - emb_mean) ** 2).mean().sqrt()
    norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
    print("L1: %.4f, L2: %.4f" %(l1_loss.item(), l2_loss.item()))
    print("Norms: min: %.4f, max: %.4f, mean: %.4f, std: %.4f" %(norms.min(), norms.max(), norms.mean(), norms.std()))

# LNCat3 can be used on 2 or 3 input tensors.
class LNCat3(nn.Module):
    def __init__(self, chan1, chan2, chan3=0, dim=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(chan1, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(chan2, elementwise_affine=True)

        if chan3 > 0:
            self.ln3 = nn.LayerNorm(chan3, elementwise_affine=True)
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
#                                  1280, 1280, 640, 640, 640,  320,  320,  320 ]
class AttentionalPooler(nn.Module):
    def __init__(self, layer_idx, feat_dim, feat_to_lora_dim_ratio=8,
                 infeat_grad_scale=0.5):
        super().__init__()
        # Set to the same number of heads as the CrossAttention layers.
        # All CrossAttention layers in UNet have 8 heads.
        self.n_heads = 8    
        self.layer_inner_dim = feat_dim
        self.lora_dim = feat_dim // feat_to_lora_dim_ratio
        # nn.Conv1d weights are initialized as Uniform(-..., sqrt( n_heads / fan_in )).
        # nn.Linear weights are initialized as Uniform(-..., sqrt( 1 / fan_in )).
        # So nn.Conv1d weights are sqrt(n_heads) times greater than those of nn.Linear.
        # Since lora_q and lora_k both go through nn.Conv1d, their product is 
        # n_heads greater than the product of the corresponding q and k in the cross-attn layer.
        # To match the sim scores from the cross-attn layer, 
        # we need to scale down the sim scores by n_heads. However, 
        # seems normalizing by n_heads is too aggressive and slows down the learning. 
        # So we take the sqrt(n_heads) instead.
        # self.lora_dim * self.n_heads is actually feat_dim. 
        # So lora_attn_score_scale is identical to the score scale used in the cross-attn layer.
        self.lora_attn_score_scale = (self.lora_dim * self.n_heads) ** -0.5

        self.lora_fg_q_ln  = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)
        self.lora_bg_q_ln  = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)
        self.lora_k_ln     = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)

        # The to_q, to_k, to_v projections in the cross-attnion layer don't have bias. 
        # So we don't use bias here.
        # If layer_inner_dim == 1280, lora_dim == 160, n_heads == 8, 
        # then lora_to_k.weight is [160, 160, 1]. It's actually 8 * (1280/8 * 160/8).
        self.lora_to_k     = nn.Conv1d(self.layer_inner_dim, self.lora_dim, kernel_size=1, groups=self.n_heads, bias=False)
        self.lora_to_fg_q  = nn.Conv1d(self.layer_inner_dim, self.lora_dim, kernel_size=1, groups=self.n_heads, bias=False)
        self.lora_to_bg_q  = nn.Conv1d(self.layer_inner_dim, self.lora_dim, kernel_size=1, groups=self.n_heads, bias=False)

        #self.ln_fg_out  = nn.LayerNorm(feat_dim, elementwise_affine=True)
        #self.ln_bg_out  = nn.LayerNorm(feat_dim, elementwise_affine=True)

        self.layer_idx = layer_idx

        self.infeat_grad_scale = infeat_grad_scale      # Default 0.5
        self.infeat_grad_scaler = gen_gradient_scaler(self.infeat_grad_scale)

        # v_pooler is used only if fg_q_emb is None.
        self.v_pooler = MaskedAvgPool1d(dim=1, keepdim=True)
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
                           
        # By default, infeat_grad_scaler does 0.5 gs.
        ca_x_gs = self.infeat_grad_scaler(ca_x)
        ca_q_gs = self.infeat_grad_scaler(ca_q)

        x = ca_x_gs
        k = ca_q_gs
        # k: query (i.e., projected image features) from the UNet cross-attention layer. 
        # Repurposed as key here.
        # the to_q() of the UNet cross-attention layer doesn't change the dimension of k,
        # i.e., k dim = x dim. Therefore, x + k_ln is legal.
        k_ln    = self.lora_k_ln(k)
        # x is x1 in BasicTransformerBlock, which will be added with x_ca output from the cross-attn layer.
        # cross-attn v is the projection of the prompt embedding. So in order to yield proper x_ca,
        # we add x to k, so x is part of the v of the attentional pooler (whose output is the 
        # input features to the ada embedder).
        # On the other hand, k is cross-attn q, which is multiplied with the 
        # cross-attn k (projection of the prompt embedding). In order to provide proper cross-attn k,
        # we include k as the input to the attentional pooler.
        # Therefore, v = x + k. We can also concat(x, k), but it will double the feature dimension.
        #calc_stats("x", x)
        #calc_stats("k_ln", k_ln)
        v = (x + k_ln) / 2

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
        fg_q_ln, bg_q_ln, k_ln = map(lambda t: t.permute(0, 2, 1), (fg_q_ln, bg_q_ln, k_ln))

        # NOTE: 320 and 64 are multi-head concatenated, 8*40 and 8*8.
        # lora_to_fg_q, lora_to_bg_q: nn.Conv1d of 8 groups, each group corresponding to a head.
        # In each group, reduce dimension 40 -> 5. 
        # So the overall computation complexity is 8*40*8=320*8, instead of 320*64.
        lora_fg_q = self.lora_to_fg_q(fg_q_ln)
        # During training, at 20% of the chance, bg_q_emb is dropped out to be 0. 
        # For these cases, lora_bg_q is 0 => sim_scores[:, 1] = 0, i.e., bg attn is uniform.
        lora_bg_q = self.lora_to_bg_q(bg_q_ln)
        # lora_k: [B, 320, 4096] -> [B, 64, 4096].
        lora_k = self.lora_to_k(k_ln)
        # lora_fg_q, lora_bg_q: [B, 64, 1]    -> [B, 1, 64]
        # lora_k:               [8, 64, 4096] -> [8, 4096, 64]
        lora_fg_q, lora_bg_q, lora_k = map(lambda t: t.permute(0, 2, 1), (lora_fg_q, lora_bg_q, lora_k))
        # lora_fg_q, lora_bg_q are two artificial tokens, each with 64 dims.
        # lora_q: [B, 2, 64]. 
        lora_q = torch.cat([lora_fg_q, lora_bg_q], dim=1)

        # Dot product of the last dim. sim_scores: [B, 2, 4096].
        # The sim_scores are too large. So we scale them down by lora_attn_score_scale.
        # The root cause why the sim_scores are too large is due to the different initialization 
        # of nn.Conv1d (compared with nn.Linear).
        sim_scores = einsum('b i d, b j d -> b i j', lora_q, lora_k) \
                        * self.lora_attn_score_scale

        if img_mask is not None:
            # img_mask: [B, 1, 64, 64] 
            img_mask = F.interpolate(img_mask, size=ca_x_size, mode='nearest')
            # N, 1, H, W -> N, 1, L=H*W
            # => [B, 1, 4096]
            img_mask = rearrange(img_mask, 'b ... -> b 1 (...)')
            # float_tensor.bool() converts 0.1/0.2... to True.
            img_mask = img_mask.bool()
            max_neg_value = -torch.finfo(sim_scores.dtype).max
            # masked_fill_() will broadcast img_mask to sim_scores's shape [B, 2, 4096].
            sim_scores.masked_fill_(~img_mask, max_neg_value)

            # Prepare to be used by v_pooler.
            img_mask = img_mask.permute(0, 2, 1)

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
            attn = sim_scores.reshape(sim_scores.shape[0], -1).softmax(dim=1)
            attn = attn.reshape(sim_scores_shape)
        else:
            # Attention probs are normalized across the image patches (4096) dimension.
            # fg attn and bg attn are independent of each other.
            attn = sim_scores.softmax(dim=-1)

        attn = self.attn_drop(attn)
        # attn_fg, attn_bg: [B, 1, 4096].
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

        # out: N, 1, D -> N, D, i.e., ([2, 768], [2, 768]).
        # Make the output shape consistent with MaskedAvgPool2d.
        return { 'fg_out': fg_out.squeeze(1), 'bg_out': bg_out.squeeze(1), 
                 'attn_fg': attn_fg }

# init_embedding: [L, M, 768].
class Embedding3d(nn.Module):
    def __init__(self, num_layers=16, num_vectors_per_token=9, 
                 out_emb_dim=768, init_embedding=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_vectors_per_token = num_vectors_per_token
        self.out_emb_dim = out_emb_dim
        # self.embedding: [16, K, 768]
        self.embedding = nn.Parameter(torch.zeros(num_layers, num_vectors_per_token, out_emb_dim), requires_grad=True)
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
    def __init__(self, num_layers=16, num_vectors_per_token=1, 
                 out_emb_dim=768, r=6, init_noise_stds=(0.1, 0.04), 
                 init_words=None, init_vecs=None, init_vec_weights=None, 
                 has_bias=True, device_type="cuda",
                 token_string=""):
        super().__init__()
        self.token_string = token_string

        self.num_layers = num_layers
        self.K = num_vectors_per_token
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
            self.pre_vecs = nn.Parameter(init_vecs.unsqueeze(0).repeat(self.K, 1, 1), requires_grad=True)
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs.data, dim=-1)
        else:
            N = self.N = 0
            self.pre_vecs = None

        # basis_rand_weights: 16 * K * r, basis_vecs: K * r * 768. 
        # output embeddings: basis_rand_weights * basis_vecs = 16 * K * 768.
        self.basis_rand_weights = nn.Parameter(torch.randn(num_layers, self.K, r))
        # basis_vecs: [K, r-N, 768], K sets, each consisting of r-N randomly initialized basis vectors. 
        # Will be updated through BP.
        # Each embedding of the K embeddings has its own basis_vecs and pre_vecs.
        # Separate pre_vecs and basis_vecs, to apply different regularizations on them.
        self.basis_vecs = nn.Parameter(torch.randn(self.K, r - N, out_emb_dim), requires_grad=True)
        # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
        self.basis_vecs.data = F.normalize(self.basis_vecs, dim=-1) / 4.
        # Always set the last basis vector to 0.
        self.basis_vecs.data[-1] = 0

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

        if self.has_bias:
            # bias: 16 * 768.
            self.bias = nn.Parameter(torch.zeros(num_layers, self.K, out_emb_dim), requires_grad=True)
        else:
            self.bias = 0

        layers_out_lns = []
        for i in range(num_layers):
            # A specific LayerNorm is applied on each of the K embeddings in each layer.
            layer_out_lns = nn.ModuleList( [ nn.LayerNorm(out_emb_dim, elementwise_affine=True) for k in range(self.K) ] )
            layers_out_lns.append(layer_out_lns)
        self.layers_out_lns = nn.ModuleList(layers_out_lns)

        print(f"StaticLayerwiseEmbedding {token_string} initialized with {self.K} total embs, {self.N} init vectors ({init_words}), {self.r} basis vectors")

    # Return static embeddings of all layers together.
    def forward(self):
        with torch.autocast(device_type=self.device_type, enabled=False):
            # self.basis_comm_weights: [1, K, r] broadcasted to [16, K, r].
            basis_weights   = self.basis_rand_weights   + self.basis_comm_weights
            # torch.matmul: matrix multiplication.
            # torch.matmul(lora_up, basis_vecs): 16 * 768.

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
    def __init__(self, num_layers=16, num_vectors_per_token=1, 
                 fg_emb_count=1, bg_emb_count=0, use_cached_bg=False,
                 out_emb_dim=768, r=12, 
                 init_words=None, init_vecs=None, 
                 # 16 cross-attention layers.
                 ca_infeat_dims = [ 320,  320,  640, 640, 1280, 1280, 1280, 1280, 
                                    1280, 1280, 640, 640, 640,  320,  320,  320 ],
                 # skipped_layers = [0, 3, 6, 9, 10, 11, 13, 14, 15],
                 layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                            17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },
                 has_bias=True, use_attn_pooler=True,
                 token_string="", token_is_bg=False, device_type="cuda"):
        super().__init__()
        self.token_string = token_string
        assert num_layers == len(layer_idx2ca_layer_idx), f"num_layers={num_layers} != len(layer_idx2ca_layer_idx)={len(layer_idx2ca_layer_idx)}"
        self.num_layers = num_layers
        self.out_emb_dim = out_emb_dim
        self.K = num_vectors_per_token
        self.fg_emb_count = fg_emb_count
        self.bg_emb_count = bg_emb_count
        assert fg_emb_count + bg_emb_count <= self.K, \
            f"fg_emb_count={fg_emb_count} + bg_emb_count={bg_emb_count} > num_vectors_per_token={self.K}"

        # token_is_bg: is this token trying to model the background?
        self.token_is_bg    = token_is_bg
        self.use_cached_bg  = use_cached_bg
        if self.use_cached_bg:
            self.cached_bg_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # Only takes bg or fg features but not both.
        self.is_fg_only = (fg_emb_count == self.K) 
        self.is_bg_only = (bg_emb_count == self.K)
        self.is_one_stream_only = self.is_fg_only or self.is_bg_only

        self.r = r
        self.use_attn_pooler = use_attn_pooler

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
            self.pre_vecs = nn.Parameter(init_vecs.unsqueeze(0).repeat(self.K, 1, 1), requires_grad=True)
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined basis vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs, dim=-1)
        else:
            N = self.N = 0
            self.pre_vecs = None

        # basis_vecs: [K, r-N, 768], K sets, each consisting of r-N randomly initialized basis vectors. 
        # Will be updated through BP.
        # Each embedding of the K embeddings has its own basis_vecs and pre_vecs.
        # Separate pre_vecs and basis_vecs, to apply different regularizations on them.
        self.basis_vecs = nn.Parameter(torch.randn(self.K, r - N, out_emb_dim), requires_grad=True)
        # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
        self.basis_vecs.data = F.normalize(self.basis_vecs, dim=-1) / 4.
        # Always set the last basis vector to 0.
        self.basis_vecs.data[:, -1] = 0

        self.ca_infeat_dims = list(ca_infeat_dims)
        # self.infeat_dims = [ 320 for i in range(25) ]

        poolers = []
        for i in range(num_layers):
            infeat_dim = self.ca_infeat_dims[i]

            if self.use_attn_pooler:
                pooler = AttentionalPooler(i, infeat_dim, infeat_grad_scale=0.5)
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
            layer_out_lns = nn.ModuleList( [ nn.LayerNorm(out_emb_dim, elementwise_affine=True) for k in range(self.K) ] )
            layers_out_lns.append(layer_out_lns)

        self.layer_coeff_maps   = nn.ModuleList(layer_coeff_maps)
        self.layers_out_lns     = nn.ModuleList(layers_out_lns)
        self.layer_lncat3s      = nn.ModuleList(layer_lncat3s)

        self.reduce_fg_bg_cross_weights()

        self.has_bias = has_bias
        if has_bias:
            # bias: [16, K, 768].
            self.bias = nn.Parameter(torch.zeros(num_layers, self.K, out_emb_dim), requires_grad=True)
        else:
            self.bias = 0

        print(f"AdaEmbedding {token_string} initialized with {fg_emb_count}/{bg_emb_count}/{self.K} fg/bg/total embs, {self.N} init vectors ({init_words}), {self.r} basis vectors")
        self.call_count = 0
        self.debug = False

    # If reduced_layer_idx is specified, then only mask for one layer. Otherwise, mask for all layers.
    # When generating ada embeddings for each layer in turn, only masking one layer will reduce processing time.
    def reduce_fg_bg_cross_weights(self, reduced_layer_idx=None):
        # If token_is_bg: 
        # its "fg infeat" is the attn pooled infeat using the main embedding, in this case, the bg embedding.
        # Therefore, "fg infeat" is of the background.
        # "bg infeat" is the cached bg infeat produced by the previous fg embedder, so it's also bg infeat.
        # Therefore, no need to scale the weights.        
        #if self.token_is_bg:
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
    def forward(self, layer_idx, layer_attn_components, time_emb, 
                layer_subj_emb_probe, layer_static_extra_emb_mean, 
                img_mask=None, cached_infeat_pooled=None):
        ca_layer_idx = self.layer_idx2ca_layer_idx[layer_idx]
        pooler  = self.poolers[ca_layer_idx]
        ## Some Linears mainly use either fg or bg features. So we reduce cross weights.
        #if self.training:
        #    self.reduce_fg_bg_cross_weights(ca_layer_idx)

        if not self.is_fg_only and self.use_cached_bg:
            # cached_infeat_pooled must be provided when use_cached_bg.
            if cached_infeat_pooled is None:
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
                infeat_bg     = cached_infeat_pooled['bg_out']
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

# text_embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
# = LatentDiffusion.cond_stage_model
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            text_embedder,              
            placeholder_strings=None,
            # If background_strings are specified, they are part of the list placeholder_strings.
            background_strings=None,
            initializer_words=None,
            initializer_weights=None,
            # num_vectors_per_token: how many vectors in each layer are allocated to model 
            # the subject (represented as the subject token) and the background. 
            # num_vectors_per_token is an int or a dict.
            num_vectors_per_token=None,
            use_layerwise_embedding=False,
            num_unet_ca_layers=16,
            # If two tokens, LoRA rank= 2 * layerwise_lora_rank_token_ratio = 4. That means,
            # compress 16 embeddings to the linear combination of 4 embeddings,
            # in which two are initialized as the two token embeddings, and two are learned through BP.
            layerwise_lora_rank_token_ratio=2,
            layerwise_lora_rank=-1,
            # If no initializer words are specified, then default LoRA rank = 2.
            layerwise_lora_default_rank=5,
            layer_idx2ca_layer_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            ada_emb_weight=0.5, 
            p_ada_emb_weight_01=0,
            ada_use_attn_pooler=True,
            emb_ema_as_pooling_probe_weight=0,
            training_add_noise_std_range=None,
            training_add_noise_prob=None,
            use_conv_attn_kernel_size=-1,
            conv_attn_layerwise_scale_learnable=False,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = OrderedDict()
        
        self.string_to_static_embedder_dict = nn.ParameterDict()
        self.string_to_ada_embedder_dict    = nn.ModuleDict()
        self.string_to_emb_ema_dict         = nn.ModuleDict()
        self.initial_embeddings  = nn.ParameterDict() # These should not be optimized
        self.token2emb_cache = nn.ParameterDict() # These should not be optimized

        self.set_ada_emb_weight(ada_emb_weight, p_ada_emb_weight_01, is_first_time_print=True)
        self.ada_use_attn_pooler = ada_use_attn_pooler
        self.emb_ema_as_pooling_probe_weight   = emb_ema_as_pooling_probe_weight
        self.emb_ema_grad_scale = 0.05
        self.emb_ema_grad_scaler = gen_gradient_scaler(self.emb_ema_grad_scale)

        self.use_layerwise_embedding = use_layerwise_embedding
        self.layerwise_lora_rank_token_ratio = layerwise_lora_rank_token_ratio
        self.num_unet_ca_layers = num_unet_ca_layers
        if self.use_layerwise_embedding:
            # self.num_layers_per_embedder specifies the total layers of embeddings for each embedder.
            # There could be multiple embeddings for each layer.
            self.num_layers_per_embedder = num_unet_ca_layers
        else:
            self.num_layers_per_embedder = 1

        self.emb_global_scale_score = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.initialize_conv_attn_layerwise_scales(1, learnable=conv_attn_layerwise_scale_learnable)
        
        self.set_training_add_noise_specs(training_add_noise_std_range, training_add_noise_prob)
        self.set_embs_attn_tricks(use_conv_attn_kernel_size)

        self.layer_idx2ca_layer_idx = layer_idx2ca_layer_idx

        # num_vectors_per_token: an int or a dict. How many vectors in each layer 
        # are allocated to model the subject (represented as the subject token).        
        # num_vectors_per_token > 1:
        # *multi-vector subject embeddings*. In this space, S* is embedded into multiple 
        # learned embeddings, an approach that is equivalent to describing
        # the concept through multiple learned pseudo-words. 
        # This setting was proposed in the TI paper,
        # and AdaPrompt also supports it for more expressive modeling.
        self.token2num_vectors = {}
        self.set_num_vectors_per_token(num_vectors_per_token, placeholder_strings)
        if background_strings is not None:
            self.background_strings = background_strings
        else:
            self.background_strings = []
        self.subject_strings = [ s for s in placeholder_strings if s not in self.background_strings ]

        # hasattr(text_embedder, 'tokenizer') -> True
        if hasattr(text_embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_tokens_for_string       = partial(get_clip_tokens_for_string,       text_embedder.tokenizer)
            get_embeddings_for_tokens   = partial(get_embeddings_for_clip_tokens,   text_embedder.transformer.text_model.embeddings)
            self.token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_tokens_for_string = partial(get_bert_tokens_for_string, text_embedder.tknz_fn)
            get_embeddings_for_tokens = text_embedder.transformer.token_emb
            self.token_dim = 1280

        # Save this function to be used in load() when doing placeholder substitution.
        self.get_tokens_for_string = get_tokens_for_string
        str2lora_rank = {}

        for idx, placeholder_string in enumerate(placeholder_strings):
            # get_token_for_string <= get_clip_token_for_string.
            # force_single_token = True, as there should be only one token in placeholder_string.
            tokens = get_tokens_for_string(placeholder_string, force_single_token=True)
            token = tokens[0]

            num_vectors_per_token = self.token2num_vectors[placeholder_string]

            if (initializer_words is not None) and idx < len(initializer_words):
                init_words = initializer_words[idx]
                init_word_tokens = get_tokens_for_string(init_words)
                N = len(init_word_tokens)
                if initializer_weights is not None and idx < len(initializer_weights):
                    init_word_weights = initializer_weights[idx]
                    init_word_weights = torch.tensor(init_word_weights, dtype=torch.float32)
                    init_word_weights = init_word_weights / init_word_weights.sum()
                else:
                    # Equal weights for all words.
                    init_word_weights = torch.ones(N, dtype=torch.float32) / N

                if layerwise_lora_rank <= 0:
                    layerwise_lora_rank = round(layerwise_lora_rank_token_ratio * N + 1) if self.use_layerwise_embedding else -1
                
                with torch.no_grad():
                    init_word_embeddings = get_embeddings_for_tokens(init_word_tokens.cpu())
                    # init_word_embeddings: [2, 768]. avg_init_word_embedding: [1, 768].
                    avg_init_word_embedding = (init_word_embeddings * init_word_weights.unsqueeze(1)).sum(dim=0, keepdim=True)

            else:
                # The background embedding is not initialized with any word embedding.
                layerwise_lora_rank  = layerwise_lora_default_rank
                init_word_embeddings = None
                init_word_weights    = None

            str2lora_rank[placeholder_string] = layerwise_lora_rank
            self.string_to_token_dict[placeholder_string] = token
            # initial_embeddings are only used to compute the regularization loss.
            # Wrap with Parameter so that they will be saved to checkpoints.
            # avg_init_word_embedding_3d: [1, 768] => [16, 9, 768]
            avg_init_word_embedding_3d = avg_init_word_embedding.unsqueeze(0).repeat(self.num_layers_per_embedder, num_vectors_per_token, 1)

            # Initialize static/ada embedders.
            # A static/ada embedder can generate K embeddings.
            # layerwise_lora_rank > 0 implies use_layerwise_embedding.
            if layerwise_lora_rank > 0:
                # if self.emb_ema_as_pooling_probe_weight > 0, then calculate EMA of embeddings 
                # to be used in Ada attn pooler.
                # num_layers_per_embedder = num_unet_ca_layers
                token_static_embedder   = StaticLayerwiseEmbedding(self.num_layers_per_embedder, 
                                                                    num_vectors_per_token, 
                                                                    self.token_dim, 
                                                                    layerwise_lora_rank, 
                                                                    (0.1, 0.02), 
                                                                    init_words,
                                                                    init_word_embeddings, init_word_weights, 
                                                                    token_string=placeholder_string)
                    
                # avg_init_word_embedding_3d: [16, 9, 768]. 
                # All layers of all 9 embeddings are initialized as avg_init_word_embedding.
                if self.emb_ema_as_pooling_probe_weight > 0:
                    # token_emb is the embedding in the prompt embeddings, not the token embeddings
                    # generated by StaticLayerwiseEmbedding / AdaEmbedding.
                    token_emb_cache = Embedding3d(self.num_layers_per_embedder, num_vectors_per_token, self.token_dim, 
                                                  init_embedding=None)
                    self.token2emb_cache[placeholder_string] = token_emb_cache

                token_is_bg =  (placeholder_string in self.background_strings)
                # For subject embeddings:    2/3 of the embeddings are fg embeddings (focus on fg infeat), 
                # and 1/3 are bg embeddings (focus on bg infeat).
                # For background embeddings: 2/3 of the embeddings are bg embeddings (focus on bg infeat), 
                # and 1/3 are fg embeddings (focus on fg infeat).
                # Note fg embeddings still take 0.3 of bg infeat, and bg embeddings still take 0.3 of fg infeat.
                # No embeddings are fg-bg embeddings, which take fg and bg infeat with equal weights.
                # If num_vectors_per_token == 1, then fg_emb_count = 1, bg_emb_count = 0.
                # If num_vectors_per_token == 9, then fg_emb_count = 6, bg_emb_count = 3.
                if token_is_bg:
                    bg_emb_count = max(1, num_vectors_per_token * 2 // 3)
                    fg_emb_count = num_vectors_per_token - bg_emb_count
                else:
                    fg_emb_count = max(1, num_vectors_per_token * 2 // 3)
                    bg_emb_count = num_vectors_per_token - fg_emb_count

                use_cached_bg = token_is_bg

                token_ada_embedder  = AdaEmbedding(self.num_layers_per_embedder, 
                                                   num_vectors_per_token, 
                                                   fg_emb_count, 
                                                   bg_emb_count,
                                                   use_cached_bg,
                                                   self.token_dim,                                                    
                                                   layerwise_lora_rank, 
                                                   init_words,
                                                   init_word_embeddings,
                                                   use_attn_pooler=ada_use_attn_pooler,
                                                   token_string=placeholder_string,
                                                   token_is_bg=token_is_bg)
            else:
                # Degenerate to Textual Inversion. 
                # ANCHOR[id=init_embed] : 16*K vectors are initialized with the same embedding.
                token_static_embedder   = nn.Parameter(avg_init_word_embedding_3d, requires_grad=True)
                token_ada_embedder      = None

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

        self.layer_idx = -1
        self.static_subj_embs_dict = {}   
        self.ada_subj_embs_dict    = {}
        self.ada_subj_attn_dict    = {}
        self.clear_ada_layer_temp_info()
        self.clear_placeholder_indices()
        self.clear_prompt_masks()
        self.img_mask = None
        self.loss_call_count = 0
        # Store the text_embedder to compute the delta loss.
        self.text_embedder  = text_embedder
        self.ada_prompt_embeddings_cache    = {}
        self.ada_prompt_token2indices_cache = {}
        self.iter_type = None       # 'recon_iter' or 'distill_iter'

        print("EmbeddingManager on subj={}, bg={} init with {} vec(s), layerwise_lora_rank={}, ada_emb_weight={}".format(
               self.subject_strings, self.background_strings, self.token2num_vectors, str2lora_rank, 
               ada_emb_weight))
            
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

        # gen_ada_embedding is dynamically switched on/off by  cache_layer_features_for_ada()/clear_ada_layer_temp_info().
        # No need to calculate prompt_emb_mask here, as the mask for ada embeddings is 
        # the same as for the static embeddings. 
        # AdaPrompt combines static and ada embeddings. So the static embedding replacement 
        # code below is always called, and prompt_emb_mask is always calculated.
        if self.gen_ada_embedding:
            # self.layer_idx, self.ca_infeat, self.time_emb were cached by 
            # a previous call of  cache_layer_features_for_ada() from UNet.
            ada_embedded_text, ada_subj_embs_dict, token2fg_attn = \
                self.get_ada_embedding(self.layer_idx, self.layer_attn_components, self.time_emb,
                                       tokenized_text, embedded_text)

            ca_layer_idx = self.layer_idx2ca_layer_idx[self.layer_idx]
            # Cache the ada embeddings to be used in embedding orthogonal loss later.
            for k in ada_subj_embs_dict:
                self.ada_subj_embs_dict[k].cache_layer(ca_layer_idx, 
                                                       ada_subj_embs_dict[k], 
                                                       has_grad=True)
                if k in token2fg_attn:
                    self.ada_subj_attn_dict[k] = token2fg_attn[k]

            # Release ada-specific intermediate variables.
            self.clear_ada_layer_temp_info()
                        
            # No prompt repeating happened in get_ada_embedding(), 
            # so pass the original tokenized_text as tokenized_text_repeated.
            self.update_prompt_masks(tokenized_text, tokenized_text)
            return ada_embedded_text

        else:
            self.clear_placeholder_indices()
            self.clear_prompt_masks()
            # We need to clone embedded_text, as sometimes (when it's not layerwise, such as TI) 
            # the modification in get_static_embedding() is in-place. 
            static_embeded_text, tokenized_text_repeated, static_subj_embs_dict = \
                            self.get_static_embedding(tokenized_text, embedded_text.clone(), 
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
                self.ada_subj_embs_dict[k]    = Embedding3d(self.num_unet_ca_layers, 
                                                            self.token2num_vectors[k], 
                                                            self.token_dim)
                self.ada_subj_embs_dict[k].to(device)

            # Update the prompt token embedding mask.
            # tokenized_text_repeated is repeated 16 times along the batch dimension.
            self.update_prompt_masks(tokenized_text, tokenized_text_repeated)

            return static_embeded_text
    
    # N: length of sequence (including padding).
    def get_static_embedding(self, tokenized_text, embedded_text, embedder_dict, 
                             B, N, num_unet_ca_layers, device):
        orig_tokenized_text = tokenized_text
        static_subj_embs_dict = {}

        if self.use_layerwise_embedding:
            # embedded_text: [B, N, 768] => [B, 16, N, 768] => [16*B, N, 768].
            # "Tuck" the layer dimension into the batch dimension, 
            # to keep embedded_text in 3D, same as the input.
            # After repeat, the same instance is repeated 16 times, which are adjacent 
            # to each other across the batch dim:
            # [b1_l1, ..., b1_l16, b2_l1, ..., b2_l16, ..., bB_l1, ..., bB_l16].
            # {________b1________} {_______b2_______}  ...  {_______bB________}
            embedded_text = embedded_text.unsqueeze(1).repeat(1, num_unet_ca_layers, 1, 1).view(B * num_unet_ca_layers, N, -1)
            # tokenized_text: [B, 16, N] => [16*B, N]
            # tokenized_text has to be repeated along the layer dimension as well, so that 
            # placeholder_indices can index the embedding at each layer in the batch.
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1, num_unet_ca_layers, 1).view(B * num_unet_ca_layers, N)
            # mirror-reflect the embedding along the layer dimension, to make it symmetric 
            # in the encoder & decoder.

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            # If there's only one vector per token, we can do a simple replacement
            placeholder_indices = torch.where(tokenized_text == placeholder_token.to(device))
            # No placeholder token is found in the current batch.
            if placeholder_indices[0].numel() == 0:
                continue
            
            token_is_bg = (placeholder_string in self.background_strings)
            # If multiple occurrences are found in a prompt, only keep the first as the subject.
            # Other occurrences are treated as part of the background prompt (this may happen if
            # composition image overlay is used).
            placeholder_indices = extract_first_index_in_each_instance(placeholder_indices)
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
            REAL_OCCURS_IN_BATCH = placeholder_indices[0].numel() // self.num_layers_per_embedder

            static_embedder = embedder_dict[placeholder_string].to(device)
            if isinstance(static_embedder, StaticLayerwiseEmbedding):
                # Generate the actual subj_static_embedding on the fly.
                # The 16 static subject embeddings are formed by linearly combining the basis vectors.
                # The matrix operations are done on the fly.
                # subj_static_embedding: [16, K, 768].
                subj_static_embedding = static_embedder()
            else:
                # static_embedder is already the embeddings.
                subj_static_embedding = static_embedder

            static_subj_embs_dict[placeholder_string] = subj_static_embedding

            for k in range(self.token2num_vectors[placeholder_string]):
                placeholder_indices_k = (placeholder_indices[0], placeholder_indices[1] + k)
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

                if self.training and self.training_add_noise_std_range is not None:
                    subj_static_embedding_k = add_noise_to_embedding(subj_static_embedding_k, 
                                                                     self.training_add_noise_std_range,
                                                                     self.training_add_noise_prob[self.iter_type])
                    
                embedded_text[placeholder_indices_k] = subj_static_embedding_k.repeat(REAL_OCCURS_IN_BATCH, 1)

            # Cache the placeholder indices for mix prompt distillation.
            # Note placeholder_indices are recomputed in update_placeholder_indices(), 
            # we don't simply cache placeholder_indices here as they are repeated 16 times 
            # to replace in 16 layers. 
            # But we need them without repetitions for mix prompt distillation.
            # If num_vectors_per_token > 1, then repeat the indices and add to offsets.
            # If background_strings is None, then always update the indices. Otherwise, 
            # skip updating placeholder indices of the background string.
            self.update_placeholder_indices(orig_tokenized_text, placeholder_string, placeholder_token, 
                                            self.token2num_vectors[placeholder_string],
                                            token_is_bg=token_is_bg)
        
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
        cached_infeat_pooled = None
        ada_subj_embs_dict  = {}
        token2fg_attn       = {}
        
        assert self.use_layerwise_embedding, "Non-layerwise embedding cannot call get_ada_embedding()."
        layer_static_prompt_embs   = layer_attn_components['layer_static_prompt_embs']

        # string_to_token_dict is an OrderedDict, with subject tokens added first, and 
        # the background token last (order controlled in main.py). 
        # This order ensures that the background Ada embedder can always use 
        # cached_infeat_pooled produced by the previous subject Ada embedder.
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            token_is_bg = (placeholder_string in self.background_strings)
            # There's only one vector per token, we can do a simple replacement
            # embedded_text: [B, N, 768].
            # tokenized_text: [B, N].
            placeholder_indices = torch.where(tokenized_text == placeholder_token.to(device))
            # Skip generating the ada embedding if the corresponding placeholder token is not in the batch.
            if placeholder_indices[0].numel() == 0:
                continue

            # extract_first_index_in_each_instance(): Get the index to the first token in each instance.
            placeholder_indices_1st = extract_first_index_in_each_instance(placeholder_indices)

            # For fg (subject) tokens, exclude fg embeddings from computing layer_static_extra_emb_mean. 
            # For bg (junk) tokens,    exclude fg embeddings from computing layer_static_extra_emb_mean.
            if token_is_bg:
                # Chance is bg embeddings may accidentally attent to fg areas,
                # then self-reinforce and contaminate the bg embeddings with fg features.
                # However, if we mask out bg embeddings from computing layer_static_extra_emb_mean,
                # the performance will drop a lot.
                list_of_indices_to_mask = [self.placeholder_indices_fg] #, self.placeholder_indices_bg]  
            else:
                ## Why not mask bg indices for fg ada? bg embeddings are supposed to be of a similar nature 
                ## as the extra compositional embeddings. Incorporating them in layer_static_extra_emb_mean
                ## will make fg and bg embeddings more orthogonal (i.e., attend to different areas).
                list_of_indices_to_mask = [self.placeholder_indices_fg]

            # layer_static_prompt_embs: [4, 77, 768]. 
            # prompt_emb_mask: [4, 77, 1], which excludes SOT and padding tokens.
            # list_of_indices_to_mask: extra tokens to exclude, esp. the fg tokens.
            layer_static_extra_emb_mean = \
                self.calc_layer_static_extra_emb_mean(layer_static_prompt_embs, self.prompt_emb_mask, 
                                                      list_of_indices_to_mask, dropout_prob=0.2)

            # Clear cached_infeat_pooled before generating subject embedding(s) of a subject token.
            # If the next token is the background token, the keep the cache, so that the background
            # token will reuse the cached_infeat_pooled computed by the previous subject token.
            if not token_is_bg:
                cached_infeat_pooled = None

            ada_embedder = self.string_to_ada_embedder_dict[placeholder_string].to(device)
            assert isinstance(ada_embedder, AdaEmbedding)

            # When it's the turn of the background Ada embedder, the cached_infeat_pooled
            # should have been computed by the previous subject Ada embedder. 
            # Otherwise it's a bug.
            if token_is_bg \
              and ada_embedder.use_cached_bg and cached_infeat_pooled is None:
                breakpoint()

            # placeholder_indices_1st obtained above is different from self.placeholder_indices_fg.
            # self.placeholder_indices_fg contains the indices for the succeeding commas.
            # placeholder_indices_1st contains the indices for the first subject token ("z") only.
            if not token_is_bg \
                   and (self.placeholder_indices_fg[0].unique().shape != placeholder_indices_1st[0].shape):
                breakpoint()

            # static_subj_embs_dict[placeholder_string]: static_subj_embeddings, [(BS/2)*K, 768].
            # BS/2: In distillation iterations, only half instances of the batch contain the subject token.
            # If (BS/2) > 1 or K > 1, then take the mean embedding of the K embeddings.
            # Even if BS/2 > 1, the K static embeddings of different instances are the same.
            # layer_subj_emb_probe: [768].
            curr_subj_indices = self.placeholder_indices_fg if not token_is_bg else self.placeholder_indices_bg
            layer_static_subj_emb = layer_static_prompt_embs[curr_subj_indices].mean(dim=0)

            # Don't use emb_ema_as_pooling_probe for background Ada embedder.
            if self.emb_ema_as_pooling_probe_weight > 0 and not token_is_bg:
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
                    ca_layer_idx = self.layer_idx2ca_layer_idx[layer_idx]
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
            # sqrt(num_vectors_per_token), now we scale it up by sqrt(num_vectors_per_token).
            layer_subj_emb_probe = layer_subj_emb_probe * np.sqrt(self.token2num_vectors[placeholder_string])

            # Generate the actual subj_ada_embedding on the fly.
            # subj_ada_embedding: [B, K, 768]. B: 2 or 4 (regularization batches).
            # Before this call, we assume static_subj_embs has been generated by 
            # a call to get_static_embedding(). 
            # The pipeline is generate static embeddings first, then generate the ada embeddings. 
            # So this assumption should always hold.
            # For background Ada embedder, cached_infeat_pooled is only used when
            # use_cached_bg. Otherwise, it's ignored.
            subj_ada_embedding, infeat_pooled_dict = \
                        ada_embedder(layer_idx, layer_attn_components, time_emb,
                                     layer_subj_emb_probe,
                                     layer_static_extra_emb_mean, 
                                     self.img_mask, cached_infeat_pooled)

            if self.img_mask is not None and self.img_mask.max() > 1:
                breakpoint()

            if not token_is_bg:
                # Cache the bg infeat computed by the first (fg) ada embedder, 
                # to be used by the second Ada embedder and the background Ada embedder.
                # NOTE: this assumes the background token always appears after the subject tokens.
                # Otherwise the cached_infeat_pooled is not available when the background Ada embedder accesses it.
                cached_infeat_pooled    = infeat_pooled_dict
                placeholder_indices_B   = placeholder_indices_1st[0]
                token2fg_attn[placeholder_string] = torch.ones_like(infeat_pooled_dict['attn_fg'])
                # Only assign ada subj attn_fg to instances that contain the placeholder token.
                # In other instances that don't contain the placeholder token, the attn_fg is all 1.
                token2fg_attn[placeholder_string][placeholder_indices_B] = infeat_pooled_dict['attn_fg'][placeholder_indices_B]

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

                if self.training and self.training_add_noise_std_range is not None:
                    subj_ada_embedding_k = add_noise_to_embedding(subj_ada_embedding_k, 
                                                                  self.training_add_noise_std_range,
                                                                  self.training_add_noise_prob[self.iter_type])
                    
                embedded_text[placeholder_indices_k] = subj_ada_embedding_k

            # Remove the batch dim.
            ada_subj_embs_dict[placeholder_string] = subj_ada_embedding.mean(dim=0)

        return embedded_text, ada_subj_embs_dict, token2fg_attn

    # Update prompt_emb_mask and prompt_token_attn_mask.
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
        # prompt_token_attn_mask: [B, N]
        padding_tokens_mask = (tokenized_text == 49407).float()
        B,  N  = tokenized_text.shape
        B2, N2 = tokenized_text_repeated.shape
        assert N == N2

        # Don't block the subj-padding interaction.
        # # Block the interaction between the subject and padding tokens with a probability of 0.5.
        p_block_subj_padding_interaction = 0 if self.training else 0

        # prompt_token_attn_mask: [B, N, N]
        if self.placeholder_indices_fg is not None and random.random() < p_block_subj_padding_interaction:
            # subj_tokens_mask: [B, N].
            # If zeros_like(tokenized_text, dtype=float), then subj_tokens_mask is of type float64, 
            # inconsistent with padding_tokens_mask.
            # If zeros_like(tokenized_text).float(), then subj_tokens_mask is of type float32.
            subj_tokens_mask = torch.zeros_like(tokenized_text).float()
            subj_tokens_mask[self.placeholder_indices_fg] = 1
            # A batch with 4 types of sub-blocks. Only the subject half is the subject prompts. 
            # The other half is the class prompts. But for better alignment, 
            # we still mask the class tokens.
            #if len(self.placeholder_indices_fg[0].unique()) == B // 2:
            #    cls_indices = (self.placeholder_indices_fg[0] + B // 2, self.placeholder_indices_fg[1])
            #    subj_tokens_mask[cls_indices] = 1

            # subj_tokens_mask: [B, N] => [B, N, 1]. 
            # padding_tokens_mask: [B, N] => [B, 1, N].
            # subj_padding_interact_mask: [B, N, N].
            subj_padding_interact_mask = torch.matmul(subj_tokens_mask.unsqueeze(2), padding_tokens_mask.unsqueeze(1))
            # Repeat subj_padding_interact_mask 16 times along the batch dimension: 
            # [B, N, N] => [B, 1, N, N] => [B, 16, N, N] => [16*B, N, N].
            if B2 > B:
                assert B2 % B == 0
                REPEAT = B2 // B
                subj_padding_interact_mask = subj_padding_interact_mask.unsqueeze(1).repeat(1, REPEAT, 1, 1).view(B2, N, N)

            self.prompt_token_attn_mask = subj_padding_interact_mask
        else:
            self.prompt_token_attn_mask = None
            
    # layer_static_prompt_embs: [4, 77, 768].
    # emb_mask:          [4, 77, 1]. Set to 1 to include / 0 to exclude from the mean.
    # (Note sometimes the batch size of emb_mask is different from layer_static_prompt_embs).
    def calc_layer_static_extra_emb_mean(self, layer_static_prompt_embs, emb_mask, 
                                         list_of_indices_to_mask, dropout_prob=0.2):
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
    
    def clear_placeholder_indices(self):
        self.placeholder_indices_fg = None
        self.placeholder_indices_bg = None
        self.token2indices = {}

    def clear_prompt_masks(self):
        self.prompt_emb_mask = None
        self.prompt_token_attn_mask = None

    def update_placeholder_indices(self, tokenized_text, placeholder_string, placeholder_token, num_vectors_per_token, token_is_bg):
        placeholder_indices = torch.where(tokenized_text == placeholder_token.to(tokenized_text.device))
        placeholder_indices_B, placeholder_indices_N = extract_first_index_in_each_instance(placeholder_indices)

        if len(placeholder_indices_B) == 0:
            if token_is_bg:
                self.placeholder_indices_bg = None
                self.token2indices[placeholder_string] = None
            return

        if num_vectors_per_token > 1:
            BS = placeholder_indices_B.shape[0]
            # unsqueeze(1) -> [B, 1] => [B, num_vectors_per_token] => [B * num_vectors_per_token].
            # Make sure the embedding indices of the same instance are grouped together.
            # [b1_v1, b1_v2, b1_v3, b2_v1, b2_v2, b2_v3, ...].
            # Then we can easily get the indices of a certain sub-batch.
            placeholder_indices_B = placeholder_indices_B.unsqueeze(1).repeat(1, num_vectors_per_token).view(-1)
            placeholder_indices_N = placeholder_indices_N.unsqueeze(1).repeat(1, num_vectors_per_token).view(-1)
            # Add offsets to the indices of the pseudo-tokens.
            placeholder_indices_N_off = placeholder_indices_N + torch.arange(num_vectors_per_token, device=tokenized_text.device).repeat(BS)
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N_off)
        else:
            # placeholder_indices contains the indices of all placeholder embeddings.
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N)
        
        if token_is_bg:
            self.placeholder_indices_bg = placeholder_indices
        else:
            self.placeholder_indices_fg = placeholder_indices
        
        self.token2indices[placeholder_string] = placeholder_indices

    def get_ada_emb_weight(self):
        if self.training:
            p01 = self.p_ada_emb_weight_01
            if p01 > 0:
                is_ada_emb_weight_01 = \
                    np.random.choice([0, 1, 2], p=(p01, p01, 1 - 2 * p01))
                if is_ada_emb_weight_01 == 0 or is_ada_emb_weight_01 == 1:
                    ada_emb_weight = is_ada_emb_weight_01
                else:
                    # 0.5 -> uniform in [0.4, 0.7]. Inject randomness to reduce overfitting.
                    ada_emb_weight = self.ada_emb_weight * np.random.uniform(0.8, 1.4)                    
            else:
                # 0.5 -> uniform in [0.4, 0.7]. Inject randomness to reduce overfitting.
                ada_emb_weight = self.ada_emb_weight * np.random.uniform(0.8, 1.4)
        else:
            ada_emb_weight = self.ada_emb_weight        
        return ada_emb_weight
 
    def get_ada_subj_attn_dict(self):
        return self.ada_subj_attn_dict
    
    def set_training_add_noise_specs(self, training_add_noise_std_range, training_add_noise_prob):
        self.training_add_noise_std_range = training_add_noise_std_range
        self.training_add_noise_prob      = training_add_noise_prob
        if training_add_noise_std_range is None:
            print(f"Disable training_add_noise")
        else:
            print(f"training_add_noise_std_range = {training_add_noise_std_range} with prob = {training_add_noise_prob}")

    def initialize_conv_attn_layerwise_scales(self, default_conv_attn_scale=1, 
                                              conv_attn_layerwise_scales=None,
                                              learnable=False):
        self.conv_attn_layerwise_scale_learnable = learnable
        
        if conv_attn_layerwise_scales is not None:
            self.conv_attn_layerwise_scales = nn.Parameter(conv_attn_layerwise_scales,
                                                           requires_grad=learnable)            
            print(f"Change conv_attn_layerwise_scales = {self.conv_attn_layerwise_scales}")

        else:
            self.conv_attn_layerwise_scales = \
                nn.Parameter(torch.ones(self.num_layers_per_embedder) * default_conv_attn_scale, 
                                        requires_grad=learnable)
            print(f"Initialize conv_attn_layerwise_scales = {self.conv_attn_layerwise_scales}")

    def get_conv_attn_layerwise_scales(self):
        # Clip the scales to [0.1, 3].
        self.conv_attn_layerwise_scales.data.clamp_(min=0.1, max=3)
        return self.conv_attn_layerwise_scales
    
    def get_emb_global_scale(self, do_perturb=True):
        # emb_global_scale_score = 0  -> emb_global_scale = 1, 
        # emb_global_scale_score = 1  -> emb_global_scale = 1.23
        # emb_global_scale_score = -1 -> emb_global_scale = 0.77
        emb_global_scale = self.emb_global_scale_score.sigmoid() + 0.5
        if self.training and do_perturb:
            # 1 -> uniform in [0.8, 1.4]. Inject randomness to reduce overfitting.
            emb_global_scale = emb_global_scale * np.random.uniform(0.8, 1.4)
        else:
            emb_global_scale = emb_global_scale        
        return emb_global_scale
    
    def set_ada_emb_weight(self, ada_emb_weight, p_ada_emb_weight_01=0, is_first_time_print=False):
        if is_first_time_print:
            print(f"Setting ada_emb_weight = {ada_emb_weight}, p_ada_emb_weight_01={p_ada_emb_weight_01}")
        else:
            if self.ada_emb_weight != ada_emb_weight:
                print(f"ada_emb_weight: {self.ada_emb_weight} => {ada_emb_weight}")
            if self.p_ada_emb_weight_01 != p_ada_emb_weight_01:
                print(f"p_ada_emb_weight_01: {self.p_ada_emb_weight_01} => {p_ada_emb_weight_01}")

        self.ada_emb_weight = ada_emb_weight
        self.p_ada_emb_weight_01 = p_ada_emb_weight_01

    def set_embs_attn_tricks(self, use_conv_attn_kernel_size=None):
        if use_conv_attn_kernel_size is not None:
            self.use_conv_attn_kernel_size = use_conv_attn_kernel_size
            extra_msg = ", DISABLED" if use_conv_attn_kernel_size is -1 else ""
            print(f"Setting use_conv_attn_kernel_size = {use_conv_attn_kernel_size}{extra_msg}")

    def set_emb_ema_as_pooling_probe_weight(self, emb_ema_as_pooling_probe_weight):
        self.emb_ema_as_pooling_probe_weight = emb_ema_as_pooling_probe_weight
        print(f"Setting emb_ema_as_pooling_probe_weight = {emb_ema_as_pooling_probe_weight}")

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

    # Set volatile data structures for computing ada embeddings.
    def set_volatile_ds(self, volatile_ds):
        self.placeholder_indices_fg = volatile_ds['subj_indices']
        self.placeholder_indices_bg = volatile_ds['bg_indices']
        self.token2indices          = volatile_ds['token2indices']
        # There are image margins after the original image is scaled down, or 
        # after using wds overlay images as input.
        # When doing attentional pooling / average pooling of image features, 
        # the margin area contains no signal, so we use img_mask to mask it out. 
        # Each image has its own img_mask, so img_mask has a shape of [B, 1, H, W].
        self.img_mask = volatile_ds['img_mask']
        self.prompt_emb_mask = volatile_ds['prompt_emb_mask']
        
    # NOTE: prompt embeddings are the embeddings of the whole prompt (including other tokens), 
    # not just the ada or static embeddings of the subject.
    def cache_ada_prompt_embedding(self, layer_idx, embedding):
        ca_layer_idx = self.layer_idx2ca_layer_idx[layer_idx]
        self.ada_prompt_embeddings_cache[ca_layer_idx] = embedding
        # If there are multiple layers, only the last placeholder_indices are cached.
        self.ada_prompt_token2indices_cache = copy.copy(self.token2indices)

    def get_cached_ada_prompt_embeddings_as_tensor(self):
        # No tokens appear in the current prompt. So the ada prompt embedding cache is empty.
        if len(self.ada_prompt_embeddings_cache) == 0:
            return None
        
        if len(self.ada_prompt_embeddings_cache) != self.num_layers_per_embedder:
            breakpoint()
        # Stack the cached prompt embeddings of all layers into a tensor.
        ada_prompt_embeddings = [ self.ada_prompt_embeddings_cache[ca_layer_idx] 
                                  for ca_layer_idx in range(self.num_layers_per_embedder) ]
        # ada_prompt_embeddings: [BS, 16, 77, 768]. BS: batch size (2 or 4). 16: num layers.
        ada_prompt_embeddings = torch.stack(ada_prompt_embeddings, dim=1)
        return ada_prompt_embeddings

    # self.ada_prompt_embeddings_cache is a cache for the prompt embeddings of all layers, 
    # for computing the prompt delta loss.
    # NOTE: prompt embeddings are the embeddings of the whole prompt (including other tokens), 
    # not just the ada or static embeddings of the subject.
    def clear_ada_prompt_embeddings_cache(self):
        if len(self.ada_prompt_embeddings_cache) == self.num_layers_per_embedder:
            self.update_emb_ema(self.ada_prompt_token2indices_cache)

        self.ada_prompt_embeddings_cache = {}

    def update_emb_ema(self, token2indices):
        if self.training and self.emb_ema_as_pooling_probe_weight > 0:
            for k, token_emb_cache_obj in self.token2emb_cache.items():
                # If all layers of ada embeddings have been cached in token_emb_cache_obj,
                # then it's time to update EMA embeddings.
                # This should happen after the previous training iteration finishes and 
                # before the current training iteration begins.
                # token2emb_cache is used to compute EMA of ada embeddings.
                token_emb_cache_obj  = self.token2emb_cache[k]

                if k not in token2indices:
                    continue
                token_indices = token2indices[k]
                
                # token k doesn't appear in this prompt (should be bg token).
                if token_indices is None:
                    continue

                for ca_layer_idx, ada_prompt_embs in self.ada_prompt_embeddings_cache.items():
                    # token_indices may only contain the indices of some (not all) instances in the batch.
                    # ada_prompt_embs: [2, 77, 768].
                    # ada_prompt_embs[token_indices]: [18, 768] => [2, 9, 768].
                    VALID_BS = token_indices[0].unique().shape[0]
                    if VALID_BS > ada_prompt_embs.shape[0]:
                        breakpoint()
                    token_prompt_embs = ada_prompt_embs[token_indices].reshape(
                                        VALID_BS, -1, ada_prompt_embs.shape[2])
                    # LitEma requires an nn.Module to do updating. 
                    # So we use token_emb_cache_obj as a dummy Embedding3d to update the EMA embedding.
                    # We can update after the ada embeddings of all layers are cached into token_emb_cache_obj.
                    # token_prompt_embs.mean(dim=0): [9, 768].
                    token_emb_cache_obj.cache_layer(ca_layer_idx, token_prompt_embs.mean(dim=0))

                if len(token_emb_cache_obj.cached_layers) == token_emb_cache_obj.num_layers:
                    if self.string_to_emb_ema_dict[k] is None:
                        # First iteration, initialize the LitEma object.
                        print("Initializing LitEma for token", k)
                        # requires_grad=True, to allow EMA embeddings to be updated by SGD.
                        self.string_to_emb_ema_dict[k] = LitEma(token_emb_cache_obj, decay=0.998, requires_grad=True)
                        # Put the newly initialized LitEma object on CUDA.
                        self.string_to_emb_ema_dict[k].to(token_prompt_embs.device)
                    else:
                        # Update EMA embeddings.
                        self.string_to_emb_ema_dict[k](token_emb_cache_obj)

                    token_emb_cache_obj.reset_cached_layer_tracker()

    def set_num_vectors_per_token(self, num_vectors_per_token, placeholder_strings=None):
        if num_vectors_per_token is None or type(num_vectors_per_token) == int:
            # If token2num_vectors is not specified, then set all tokens to have 1 vector.
            # If token2num_vectors is an int, then set all tokens to have 
            # token2num_vectors vectors.
            if num_vectors_per_token is None:
                num_vectors_per_token = 1

            # During inference, placeholder_strings might be None. 
            # In that case, we use string_to_token_dict loaded from the ckpt.
            if placeholder_strings is None:
                placeholder_strings = self.string_to_token_dict.keys()
            for k in placeholder_strings:
                self.token2num_vectors[k] = num_vectors_per_token
        else:
            self.token2num_vectors = num_vectors_per_token
        print(f"Set token2num_vectors: {self.token2num_vectors}")

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, ckpt_path):
        torch.save({ "string_to_token":                 self.string_to_token_dict,
                     "string_to_static_embedder":       self.string_to_static_embedder_dict,
                     "string_to_ada_embedder":          self.string_to_ada_embedder_dict,
                     "string_to_emb_ema_dict":          self.string_to_emb_ema_dict,
                     "token2num_vectors":               self.token2num_vectors,
                     "emb_global_scale_score":          self.emb_global_scale_score,
                     "ada_emb_weight":                  self.ada_emb_weight,  
                     "p_ada_emb_weight_01":             self.p_ada_emb_weight_01,
                     "emb_ema_as_pooling_probe_weight": self.emb_ema_as_pooling_probe_weight,
                     # Learnable weights for scaling conv attns.
                     "conv_attn_layerwise_scales":      self.conv_attn_layerwise_scales,
                     "use_conv_attn_kernel_size":       self.use_conv_attn_kernel_size,
                   }, 
                    ckpt_path)

    # load custom tokens and their learned embeddings from "embeddings_gs-4200.pt".
    def load(self, ckpt_paths):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        self.string_to_token_dict           = {}
        self.string_to_static_embedder_dict = nn.ParameterDict()
        self.string_to_ada_embedder_dict    = nn.ModuleDict()
        self.string_to_emb_ema_dict         = nn.ModuleDict()
        
        token2num_vectors                   = {}

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
                p_ada_emb_weight_01 = ckpt.get("p_ada_emb_weight_01", 0)
                self.set_ada_emb_weight(ckpt["ada_emb_weight"], p_ada_emb_weight_01, is_first_time_print=False)

            if "emb_global_scale_score" in ckpt:
                self.emb_global_scale_score = ckpt["emb_global_scale_score"]
                print(f"Set emb_global_scale = {self.get_emb_global_scale(do_perturb=False):.4f}")

            if "emb_ema_as_pooling_probe_weight" in ckpt:
                self.set_emb_ema_as_pooling_probe_weight(ckpt["emb_ema_as_pooling_probe_weight"])
            else:
                self.emb_ema_as_pooling_probe_weight = 0

            if "conv_attn_layerwise_scales" in ckpt:
                self.initialize_conv_attn_layerwise_scales(1, ckpt["conv_attn_layerwise_scales"])

            use_conv_attn_kernel_size   = ckpt.get("use_conv_attn_kernel_size", None)
            self.set_embs_attn_tricks(use_conv_attn_kernel_size)

            for k in ckpt["string_to_token"]:
                if (placeholder_mapper is not None) and (k in placeholder_mapper):
                    k2 = placeholder_mapper[k]
                    k2_token = self.get_tokens_for_string(k2)[0]
                else:
                    k2 = k
                    k2_token = ckpt["string_to_token"][k]

                if k2 in self.string_to_token_dict:
                    raise ValueError(f"Duplicate key {k}->{k2} in {ckpt_path}")

                # The mapping in string_to_token_dict is determined by the tokenizer. 
                # Shouldn't do the k->k2 mapping on string_to_token_dict.
                self.string_to_token_dict[k2]   = k2_token

                # Mapped from k in ckpt to k2 in the current session.
                for km in ckpt["string_to_static_embedder"].keys():
                    # If there are pseudo-tokens within multi-embedding tokens, load them as well.
                    if km.startswith(k):
                        km2 = km.replace(k, k2)
                        self.string_to_static_embedder_dict[km2] = ckpt["string_to_static_embedder"][km]
                        self.string_to_ada_embedder_dict[km2]    = ckpt["string_to_ada_embedder"][km]

                        if self.emb_ema_as_pooling_probe_weight > 0:
                            self.string_to_emb_ema_dict[km2]     = ckpt["string_to_emb_ema_dict"][km]
                        
                        if km in ckpt["token2num_vectors"]:
                            token2num_vectors[km2] = ckpt["token2num_vectors"][km]
                        print(f"Loaded {km}->{km2} from {ckpt_path}")
                        ada = self.string_to_ada_embedder_dict[km2]
                        print(f"{km2}: {ada.fg_emb_count}/{ada.bg_emb_count}/{ada.K} fg/bg/total embeddings")

            if "token2num_vectors" in ckpt:
                self.set_num_vectors_per_token(token2num_vectors)

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def optimized_parameters(self):
        normal_params = list(self.string_to_static_embedder_dict.parameters()) \
                        + list(self.string_to_ada_embedder_dict.parameters()) \
                        + list(self.string_to_emb_ema_dict.parameters()) \
                        + [ self.emb_global_scale_score ]

        params_with_lr_ratios = [ { 'params': normal_params, 'lr_ratio': 1 } ]

        if self.conv_attn_layerwise_scale_learnable:
            slow_params = [self.conv_attn_layerwise_scales]
            params_with_lr_ratios = params_with_lr_ratios + [ { 'params': slow_params,   'lr_ratio': 0.1 } ]

        return params_with_lr_ratios
        
    def embedding_attractor_loss(self):
        loss = 0.
        num_placeholders = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_static_embedder_dict[key]
            coarse = self.initial_embeddings[key]
            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_placeholders

        return loss

    # Do not use. Performs poorly. 
    def layerwise_embedding_attractor_loss(self):
        loss = 0.
        num_placeholders    = len(self.initial_embeddings)
        euc_loss_type       = 'l1'       # l1, l2
        euc_loss_weight     = 1.0
        cosine_loss_weight  = 1 - euc_loss_weight
        l2_norm_weight      = 0.00
        reg_center_type     = 'init'     # avg, init

        for key in self.initial_embeddings:
            embeddings = self.string_to_static_embedder_dict[key]
            # Generate the actual embeddings on the fly.
            if isinstance(embeddings, StaticLayerwiseEmbedding):
                embeddings = embeddings()

            if reg_center_type == 'init':
                # initial_embeddings[key] is already [L, 768]. No need to repeat().
                reg_center = self.initial_embeddings[key]
            else:
                # make avg_embedding the same shape as embeddings, 
                # to avoid F.*_loss() whining about broadcasting.
                avg_embedding = embeddings.mean(dim=0, keepdim=True).repeat(embeddings.shape[0], 1)
                reg_center = avg_embedding

            l2_norm_reg = torch.norm(embeddings, dim=1).mean()
            if euc_loss_type == 'l1':
                # Push the embedding of each layer towards the mean embedding averaged across layers.
                euc_loss = F.l1_loss(embeddings, reg_center)
            elif euc_loss_type == 'l2':
                # Push the embedding of each layer towards the mean embedding averaged across layers.
                euc_loss = F.mse_loss(embeddings, reg_center)

            if cosine_loss_weight > 0:
                cosine_mat = F.cosine_similarity(embeddings[:,:,None], embeddings.t()[None,:,:])
                # There are N*(N-1)/2 elements in torch.triu(cosine_mat, diagonal=1).
                cosine_loss = 1. - torch.triu(cosine_mat, diagonal=1).sum() * 2 / (embeddings.shape[0] * (embeddings.shape[0] - 1))
                # cosines = F.cosine_similarity(embeddings, reg_center)
                # cosine_loss = 1. - cosines.mean()
            else:
                cosine_loss = 0.

            loss = loss + euc_loss * euc_loss_weight \
                   + cosine_loss * cosine_loss_weight \
                   + l2_norm_reg * l2_norm_weight

        return loss / num_placeholders

    def layerwise_embedding_norm_loss(self):
        loss_static = 0.
        loss_ada    = 0.
        euc_loss_type     = 'l2'       # l1, l2. l2 is recommended.

        # Update: revert the reduction of the reg weight of the global bias, although 
        # the common practice is no regularization on biases.
        # If bias_reg_weight_base = 0.01, bias (= static part in ada embeddings) 
        # becomes too large (1~2), and may overpower the dynamic part.
        bias_reg_weight_base        = 0.1
        basis_reg_weight_base       = 0.1
        ada_maps_weight_reg_weight  = 1.
        # If ada_maps_bias_reg_weight = 0.02, map biases are usually very small (< 0.001)
        # If ada_maps_bias_reg_weight = 0.001, map biases are still very small. 
        # So this weight doesn't matter much.
        ada_maps_bias_reg_weight    = 0.001   # 0.02 -> 0.001
        pre_vecs_reg_weight         = 0.1
        static_l2_loss_boost        = 5
        ada_static_loss_boost_ratio = 2
        ada_l2_loss_boost           = static_l2_loss_boost * ada_static_loss_boost_ratio

        ada_attn_poolers_reg_weight = 0.2

        # Dynamically adjust the regularization weights. The larger the norm, the larger the weight.
        # T: temperature. Larger T => when norm grows, the penalty is more severe.
        T = 1.5
        num_out_embeddings  = 0

        for key in self.initial_embeddings:
            for embobj in (self.string_to_static_embedder_dict[key], 
                           self.string_to_ada_embedder_dict[key]):
                # Skip non-layerwise embeddings.
                if not isinstance(embobj, (StaticLayerwiseEmbedding, AdaEmbedding)):
                    continue
                
                # init_vecs is used to regularize pre_vecs.
                # init_vecs could be scalar 0, if initial embeddings are not specified.
                # In this case, loss_pre_vecs becomes the zero-centered attractor loss.
                init_vecs = self.initial_embeddings[key]

                # bias, pre_vecs, basis_vecs exist in 
                # both StaticLayerwiseEmbedding and AdaEmbedding.
                # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
                # pre_vecs are updated through BP. Therefore, loss_pre_vecs prevents pre_vecs 
                # from drifting too far from init_vecs.
                if embobj.has_bias:
                    loss_bias        = reg_loss(embobj.bias, loss_type=euc_loss_type)
                    # bias_reg_weight is computed dynamically. But it's detached from the graph.
                    # embobj.bias: now [Layers, K, 768].
                    bias_reg_weight  = bias_reg_weight_base  * torch.norm(embobj.bias, dim=-1).mean().item() ** T
                else:
                    loss_bias        = 0.
                    bias_reg_weight  = 0.

                loss_basis       = reg_loss(embobj.basis_vecs, loss_type=euc_loss_type)
                # Like bias_reg_weight, basis_reg_weight is also computed dynamically 
                # and detached from the graph.
                # basis_vecs: now [K, r-N, 768] for Ada embedder, or [r-N, 768] for Static embedder.
                basis_reg_weight = basis_reg_weight_base * torch.norm(embobj.basis_vecs, dim=-1).mean().item() ** T

                # N: number of pre_vecs (init_vecs).
                if embobj.N > 0:
                    # If pre_vecs has a K dim (shape [K, 1, 768]), then init_vecs is automatically broadcasted.
                    loss_pre_vecs = reg_loss(embobj.pre_vecs - init_vecs, loss_type=euc_loss_type)
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
                    if self.ada_use_attn_pooler:
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

        # num_out_embeddings counts both the embeddings of the static part and the dynamic part.
        # It's the twice of the actual number of embeddings.
        num_out_embeddings /= 2

        return loss_static / num_out_embeddings, loss_ada / num_out_embeddings

    def embedding_reg_loss(self):
        self.loss_call_count += 1
        if self.use_layerwise_embedding:
            return self.layerwise_embedding_norm_loss()
        else:
            return self.embedding_attractor_loss()

    def calc_fg_bg_token_embs_ortho_loss(self, fg_bg_token_lists=None, ada_grad_scale=0.1, fg_grad_scale=0.4):
        if fg_bg_token_lists is None:
            fg_bg_token_lists = [ self.subject_strings, self.background_strings ]
        
        loss_fg_bg_token_emb_ortho = 0.
        num_fg_bg_pairs = 0
        ada_grad_scaler = gen_gradient_scaler(ada_grad_scale)

        for fg_token in fg_bg_token_lists[0]:
            for bg_token in fg_bg_token_lists[1]:
                try:
                    fg_static_token_emb         = self.static_subj_embs_dict[fg_token]
                    fg_ada_token_emb_cache_obj  = self.ada_subj_embs_dict[fg_token]
                    # fg_ada_token_emb_ema_obj    = self.string_to_emb_ema_dict[fg_token]
                    bg_static_token_emb         = self.static_subj_embs_dict[bg_token]
                    bg_ada_token_emb_cache_obj  = self.ada_subj_embs_dict[bg_token]
                except KeyError:
                    continue

                if len(fg_ada_token_emb_cache_obj.cached_layers) == fg_ada_token_emb_cache_obj.num_layers:
                    fg_ada_token_emb = fg_ada_token_emb_cache_obj.embedding
                else:
                    fg_ada_token_emb = 0

                '''
                if fg_ada_token_emb_ema_obj is not None:
                    fg_ada_token_emb_ema = fg_ada_token_emb_ema_obj.embedding
                else:
                    fg_ada_token_emb_ema = 0
                '''

                if len(bg_ada_token_emb_cache_obj.cached_layers) == bg_ada_token_emb_cache_obj.num_layers:
                    bg_ada_token_emb = bg_ada_token_emb_cache_obj.embedding
                else:
                    bg_ada_token_emb = 0

                ada_emb_ema_weight  = self.emb_ema_as_pooling_probe_weight
                # fg_hybrid_token_emb: [16, 9, 768]. 16: num layers. 9: num vectors.
                # bg_hybrid_token_emb: [16, 4, 768]. 16: num layers. 4: num vectors.
                # fg_ada_token_emb/bg_ada_token_emb are volatile and the gradients are noisy. 
                # So we scale down their gradients to 0.1.
                fg_hybrid_token_emb = fg_static_token_emb * (1 - ada_emb_ema_weight) \
                                        + ada_grad_scaler(fg_ada_token_emb) * ada_emb_ema_weight
                bg_hybrid_token_emb = bg_static_token_emb * (1 - ada_emb_ema_weight) \
                                        + ada_grad_scaler(bg_ada_token_emb)  * ada_emb_ema_weight
                
                # fg_hybrid_token_emb, bg_hybrid_token_emb: [16, 768]. 16: num layers.
                fg_hybrid_token_mean_emb = fg_hybrid_token_emb.mean(dim=1)
                bg_hybrid_token_mean_emb = bg_hybrid_token_emb.mean(dim=1)

                loss_fg_bg_pair_token_emb_ortho = \
                    calc_ref_cosine_loss(bg_hybrid_token_mean_emb, fg_hybrid_token_mean_emb, 
                                         exponent=2, do_demean_first=False,
                                         first_n_dims_to_flatten=1, 
                                         ref_grad_scale=fg_grad_scale)
                
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
