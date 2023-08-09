import torch
from torch import nn, einsum
from einops import rearrange, repeat

import torch.nn.functional as F
import numpy as np

from ldm.util import ortho_subtract, calc_delta_loss, GradientScaler
from functools import partial
from collections import OrderedDict

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

def selective_reg_loss(x, loss_type='l2', selector=None):
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

class LNCat2(nn.Module):
    def __init__(self, chan1, chan2, dim=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(chan1, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(chan2, elementwise_affine=True)
        self.dim = dim

    def forward(self, x1, x2):
        x1 = self.ln1(x1)
        x2 = self.ln2(x2)
        return torch.cat([x1, x2], dim=self.dim)

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

class AvgPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    # x: [N, C, L], mask: [N, 1, L0] of 1s (keep) or 0s (discard).
    # L: feature map size, L0: original sequence length.
    # Return: [N, C]
    def forward(self, x, k=None, mask=None):
        #if mask is None:
        return x.mean(dim=1)
        
        x = x * mask
        x = x.sum(dim=2) / mask.sum(dim=2)
        return x

# There's almost no learnable parameters in AttentionalPooler, except elementwise affines in 3 LayerNorms.
class AttentionalPooler(nn.Module):
    def __init__(self, layer_idx, feat_dim, token_emb_dim, lora_dim=64,
                 infeat_grad_scale=0.5):
        super().__init__()
        self.n_heads = 1
        self.layer_inner_dim = feat_dim
        
        if lora_dim > 0:
            self.use_lora = True
            self.lora_attn_score_scale = lora_dim ** -0.5
            self.lora_ln_q  = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)
            self.lora_ln_k  = nn.LayerNorm(self.layer_inner_dim, elementwise_affine=False)
            self.lora_to_k  = nn.Linear(self.layer_inner_dim, lora_dim, bias=False)
            self.lora_to_q  = nn.Linear(self.layer_inner_dim, lora_dim, bias=False)
        else:
            self.use_lora = False

        self.ln_x = nn.LayerNorm(feat_dim,      elementwise_affine=True)
        self.ln_k = nn.LayerNorm(feat_dim,      elementwise_affine=True)
        self.ln_q = nn.LayerNorm(token_emb_dim, elementwise_affine=True)

        self.layer_idx = layer_idx

        # Set to < 1 to reduce the gradient flow into the UNet.
        self.infeat_grad_scale = infeat_grad_scale
        if self.infeat_grad_scale < 1:
            self.infeat_grad_scaler = GradientScaler(self.infeat_grad_scale)
        else:
            self.infeat_grad_scaler = None

    # k: query in the UNet attention layer. Used as key here.
    # token_q_emb: [768,] static subject embedding of this layer. Used as query here.
    # Use UNet feature query as k, and subject token as q, to compute the attention, 
    # which aggregates the original UNet layer input features x.
    def forward(self, layer_attn_components, token_q_emb, mask=None, bp_to_unet=False):
        h = self.n_heads
        layer_infeat, layer_inquery, layer_to_k, attn_score_scale = layer_attn_components

        if bp_to_unet:
            if self.infeat_grad_scale < 1:
                #grad_scaler = grad_scaler.cuda()
                layer_infeat_gradscaled  = self.infeat_grad_scaler(layer_infeat)
                layer_inquery_gradscaled = self.infeat_grad_scaler(layer_inquery)
            else:
                layer_infeat_gradscaled  = layer_infeat
                layer_inquery_gradscaled = layer_inquery
        else:
            # Ordinary image reconstruction iterations. No BP into the UNet.
            # But if attentional pooler is used, it will also not be BPed into and not updated.
            # When not bp_to_unet, completely cut off the gradient flow into the UNet.
            # bp_to_unet is enabled when doing composition regularization iterations. 
            layer_infeat_gradscaled  = layer_infeat.detach()
            layer_inquery_gradscaled = layer_inquery.detach()

        x = layer_infeat_gradscaled
        k = layer_inquery_gradscaled
        # Use to_k of the UNet attention layer as to_q here, 
        # as the subject embedding is used as the key in UNet.
        to_q = layer_to_k

        if x.ndim == 4:
            if mask is not None:
                mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
                # float_tensor.bool() converts to 0.1/0.2... to True.
                # N, 1, H, W -> N, 1, L=H*W
                mask = rearrange(mask, 'b ... -> b (...)')
                # N, 1, L -> N*Head, 1, L, i.e., [8, 1, 256].
                # einops.repeat() is more convenient than unsqueeze().repeat().squeeze().
                mask = repeat(mask, 'b j -> (b h) () j', h=h)

            # x: N, D, H, W -> N, D, S -> N, S, D
            x = x.flatten(2).permute(0, 2, 1)
        else:
            mask = None
            # x is already 3D.

        token_q_emb = token_q_emb.unsqueeze(0)

        # to_q is actually to_k in the UNet attention layer, 
        # as the subject embedding is used as the key in UNet.
        q = to_q(self.ln_q(token_q_emb))
        # q: [1, 320] -> [N, 1, 320]
        q = repeat(q, 'n d -> b n d', b=x.shape[0])

        # k: query from the UNet attention layer. Used as key here.
        # No need to go through to_k() here, as it's already the query from the UNet attention layer.
        k = self.ln_k(k)
        v = self.ln_x(x)

        # q: [8, 1, 32], k: [8, 256, 32], v: [8, 256, 192]. 8: B*Head = 2*4.
        # The 768 dims of v are split into 4 heads, each head has 192 dims.
        # This is kind of arbitrary, as there's no v projection that makes each head have congruent channels.
        # But introducing a v projection would greatly increase parameters.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if self.use_lora:
            # lora_q: [N, 1, 320] -> [N, 1, 64]
            lora_q = self.lora_to_q(self.lora_ln_q(q))
            # lora_k: [N, L, 64]
            lora_k = self.lora_to_k(self.lora_ln_k(k))
            # lora_scores: [8, 1, 256].
            lora_scores = einsum('b i d, b j d -> b i j', lora_q, lora_k) * self.lora_attn_score_scale
            sim_scores  = lora_scores

        else:
            # sim_scores: [8, 1, 256].
            sim_scores = einsum('b i d, b j d -> b i j', q, k) * attn_score_scale

        if mask is not None:
            mask = mask.bool()
            max_neg_value = -torch.finfo(sim_scores.dtype).max
            sim_scores.masked_fill_(~mask, max_neg_value)

        attn = sim_scores.softmax(dim=-1)

        # fg_out: [8, 1, 192].
        fg_out = einsum('b i j, b j d -> b i d', attn, v)
        # fg_out: [8, 1, 192] -> [2, 1, 4*192] = [2, 1, 768].
        fg_out = rearrange(fg_out, '(b h) n d -> b n (h d)', h=h)

        # The residual of the mean input features subtracted by fg_out.
        bg_out = v.mean(dim=1, keepdim=True) - fg_out

        # out: [2, 1, 768], [2, 1, 768] => [2, 1, 1536] => [2, 1536].
        # out = torch.cat([fg_out, bg_out], dim=-1)
        # out: N, 1, D -> N, D, i.e., ([2, 768], [2, 768]).
        # Make the output shape consistent with MaskedAvgPool2d.
        return fg_out.squeeze(1), bg_out.squeeze(1)

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
            # Only one vector is passed in.
            if init_vecs.ndim == 1:
                init_vecs = init_vecs.unsqueeze(0)

            if init_vecs.shape[1] != out_emb_dim or init_vecs.shape[0] > num_layers:
                raise ValueError(
                    f"StaticLayerwiseEmbedding init vectors shape {init_vecs.shape} must be (<={num_layers}, {out_emb_dim})"
                )

            N = self.N = init_vecs.shape[0]
            # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
            # pre_vecs are updated through BP.
            self.pre_vecs = nn.Parameter(init_vecs.clone(), requires_grad=True)
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs.data, dim=1)
        else:
            self.N = N = 0
            self.pre_vecs = None

        # basis_rand_weights: 16 * r, basis_vecs: r * 768. basis_rand_weights * basis_vecs: 16 * 768.
        self.basis_rand_weights    = nn.Parameter(torch.randn(num_layers, self.K, r))
        # basis_vecs consists of r-N randomly initialized basis vectors. 
        # Will be updated through BP.
        self.basis_vecs = nn.Parameter(torch.randn(r - N, out_emb_dim), requires_grad=True)
        # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
        self.basis_vecs.data = F.normalize(self.basis_vecs, dim=1) / 4.
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
    def forward(self, only_bias=False):
        with torch.autocast(device_type=self.device_type, enabled=False):
            if only_bias:
                return self.bias

            # self.basis_comm_weights: [1, K, r] broadcasted to [16, K, r].
            basis_weights   = self.basis_rand_weights   + self.basis_comm_weights
            # torch.matmul: matrix multiplication.
            # torch.matmul(lora_up, basis_vecs): 16 * 768.

            # self.N: number of pre_vecs.
            if self.N > 0:
                # basis_vecs: [r, 768].
                basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=0)
            else:
                basis_vecs = self.basis_vecs

            # [16, K, r] * [r, 768] -> [16, K, 768].
            out_vecs = torch.matmul(basis_weights, basis_vecs)
            # Apply layer-and-embedding specific layer normalization.
            # Note the order of for i ... for k ... in the list comprehension is the same as 
            # a conventional for loop: 1 for i ...
            #                          2     for k ...
            # Otherwise embeddings in out_vecs_ln will be in wrong order.
            out_vecs_ln = [ self.layers_out_lns[i][k](out_vecs[i,k]) for i in range(self.num_layers) for k in range(self.K) ]
            out_vecs_ln = torch.stack(out_vecs_ln, dim=0).reshape(self.num_layers, self.K, -1) / np.sqrt(self.out_emb_dim)

            # Different layers and embeddings have different biases.
            # self.bias: [16, K, 768].
            out_vecs_ln = out_vecs_ln + self.bias
            # Return static embeddings of all layers together: [16, K, 768].
            return out_vecs_ln

class AdaEmbedding(nn.Module):
    # num_layers: 16 (9 layers out of 25 of UNet are skipped).
    # out_emb_dim: 768, r: 12.
    # infeat_dims: a list of 25 integers, each is the dimension of 
    # the input feature from the respective layer. 9 of them are skipped.
    # infeat_dims are (almost) reflective around the middle layer, except for the first and last layers.
    # Layer indices absent in layer_idx2emb_idx are skipped layers.
    def __init__(self, num_layers=16, num_vectors_per_token=1, 
                 fg_emb_count=1, bg_emb_count=0,
                 out_emb_dim=768, r=12, 
                 init_words=None, init_vecs=None, 
                 attn_infeat_dims = [ 320, 320, 640, 640, 1280, 1280, 1280, 1280, 
                                      1280, 1280, 640, 640, 640, 320, 320, 320 ],
                 # skipped_layers = [0, 3, 6, 9, 10, 11, 13, 14, 15],
                 layer_idx2emb_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },
                 has_bias=True, use_attn_pooler=True,
                 device_type="cuda", token_string=""):
        super().__init__()
        self.token_string = token_string
        assert num_layers == len(layer_idx2emb_idx), f"num_layers={num_layers} != len(layer_idx2emb_idx)={len(layer_idx2emb_idx)}"
        self.num_layers = num_layers
        self.out_emb_dim = out_emb_dim
        self.K = num_vectors_per_token
        self.fg_emb_count = fg_emb_count
        self.bg_emb_count = bg_emb_count
        assert fg_emb_count + bg_emb_count <= self.K, \
            f"fg_emb_count={fg_emb_count} + bg_emb_count={bg_emb_count} > num_vectors_per_token={self.K}"

        # Only takes bg features. Could possibly use cached bg features.
        self.is_bg_only = (bg_emb_count == self.K)
        self.r = r
        self.use_attn_pooler = use_attn_pooler

        # emb_infeat_types: 0 = fg, 1 = bg, 2 = fg_bg. 
        # Usually there are no type-2 (fg_bg) embeddings.
        self.emb_infeat_types = [ 0 ] * self.fg_emb_count + [ 1 ] * self.bg_emb_count \
                                + [ 2 ] * (self.K - self.fg_emb_count - self.bg_emb_count)

        cand_basis_masks = [ torch.cat([ torch.ones(r),  torch.zeros(r) ]),
                             torch.cat([ torch.zeros(r), torch.ones(r) ]),
                             torch.ones(2 * r) ]
        
        if not self.is_bg_only:
            # basis_mask: [K, 2*r]. 
            # The purpose of basis_mask is to separate the basis vectors into fg and bg parts, so that
            # fg-only embeddings only use the fg part of the basis vectors, 
            # and bg-only embeddings only use the bg part.
            basis_mask = torch.stack([ cand_basis_masks[emb_infeat_type] for emb_infeat_type in self.emb_infeat_types ], dim=0)
            # basis_mask: [K, 2*r] -> [1, K, 2*r].
            basis_mask = basis_mask.unsqueeze(0)
            # basis_mask will be put on cuda automatically.
            self.register_buffer(name='basis_mask', tensor=basis_mask, persistent=False)
        else:
            self.basis_mask = 1

        self.device_type = device_type
        self.layer_idx2emb_idx = layer_idx2emb_idx
        # emb_idx2layer_idx: Reverse mapping of layer_idx2emb_idx.
        self.emb_idx2layer_idx = { v: k for k, v in layer_idx2emb_idx.items() }

        # Unless is_bg_only, then always use both fg and bg features as the input to the Linear, 
        # therefore the input feature dim is doubled.
        # But a particular Linear may only use either the fg or bg half of the features according to emb_infeat_types, 
        # so the other half of the weights will be masked.
        # If is_bg_only, then only use bg features as the input to the Linear, and no weight masking is needed.
        self.H = H  = 1 if self.is_bg_only else 2
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
            # Only one vector is passed in.
            if init_vecs.ndim == 1:
                init_vecs = init_vecs.unsqueeze(0)

            if init_vecs.shape[1] != out_emb_dim or init_vecs.shape[0] > num_layers:
                raise ValueError(
                    f"AdaEmbedding LoRA init vectors shape {init_vecs.shape} must be (<={num_layers}, {out_emb_dim})"
                )

            N = self.N = init_vecs.shape[0]
            # pre_vecs: [2, 1, 768].
            self.pre_vecs = nn.Parameter(init_vecs.unsqueeze(0).repeat(H, 1, 1), requires_grad=True)
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined basis vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs, dim=1)
        else:
            N = self.N = 0
            self.pre_vecs = None

        # Separate pre_vecs and basis_vecs, to apply different regularizations on them.
        # basis_vecs: [K, r-N, 768], consists of r-N basis vectors. Will be updated through BP.
        self.basis_vecs = nn.Parameter(torch.randn(H, r - N, out_emb_dim), requires_grad=True)
        # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
        self.basis_vecs.data = F.normalize(self.basis_vecs, dim=-1) / 4.
        # Always set the last basis vector to 0.
        self.basis_vecs.data[:, -1] = 0

        self.attn_infeat_dims = list(attn_infeat_dims)
        # self.infeat_dims = [ 320 for i in range(25) ]

        poolers = []
        for i in range(num_layers):
            infeat_dim = self.attn_infeat_dims[i]

            if self.use_attn_pooler:
                pooler = AttentionalPooler(i, infeat_dim, out_emb_dim, infeat_grad_scale=0.5)
            else:
                pooler = AvgPool1d() #MaskedAvgPool2d()
            poolers.append(pooler)

        self.poolers = nn.ModuleList(poolers)

        layer_maps = []
        layers_out_lns  = []
        # lncat2s: multiple cat(LN(infeat), LN(time_emb)).
        layer_lncat2s = []
        self.TDs = []

        for i in range(num_layers):
            # Time embedding dimension is the same as the input feature dimension.
            # So TD = TD_frac * attn_infeat_dims[i].
            TD = int(self.TD_frac * self.attn_infeat_dims[i])
            self.TDs.append(TD)

            # input  dim: self.attn_infeat_dims[i] + TD, since first TD dims of time_emb is part of the input features.
            # output dim: r * H * K, will be reshaped to [K, r*H].
            # This Linear outputs K sets of r-dim vectors, 
            # each set being the coefficients of the r basis vectors. 
            layer_maps.append( nn.Linear(self.attn_infeat_dims[i] * H + TD, r * H * self.K, bias=True) )
            layer_lncat2s.append(LNCat2(self.attn_infeat_dims[i] * H, TD))
            # A specific LayerNorm is applied on each of the K embeddings in each layer.
            layer_out_lns = nn.ModuleList( [ nn.LayerNorm(out_emb_dim, elementwise_affine=True) for k in range(self.K) ] )
            layers_out_lns.append(layer_out_lns)

        self.layer_maps     = nn.ModuleList(layer_maps)
        self.layers_out_lns = nn.ModuleList(layers_out_lns)
        self.layer_lncat2s  = nn.ModuleList(layer_lncat2s)
        self.mask_linear_weights()

        self.has_bias = has_bias
        if has_bias:
            # bias: [16, K, 768].
            self.bias = nn.Parameter(torch.zeros(num_layers, self.K, out_emb_dim), requires_grad=True)
        else:
            self.bias = 0

        print(f"AdaEmbedding {token_string} initialized with {fg_emb_count}/{bg_emb_count}/{self.K} fg/bg/total embs, {self.N} init vectors ({init_words}), {self.r} basis vectors")
        self.call_count = 0
        self.debug = False

    # If masked_layer_idx is specified, then only mask for one layer. Otherwise, mask for all layers.
    # When generating ada embeddings for each layer in turn, only masking one layer will reduce processing time.
    def mask_linear_weights(self, masked_layer_idx=None):
        # Skip masking if is_bg_only.
        if self.H == 1:
            return

        layer_range = range(self.num_layers) if masked_layer_idx is None else [masked_layer_idx]

        for layer_idx in layer_range:
            HALF_D = self.attn_infeat_dims[layer_idx]
            TD     = self.TDs[layer_idx]
            assert self.layer_maps[layer_idx].in_features == HALF_D * self.H + TD
            layer_map_weight = self.layer_maps[layer_idx].weight.data
            # The weight of Linear has shape [out_features, in_features]. 
            # Split the out_features to [K, 2*r].
            layer_map_weight_embs = layer_map_weight.reshape(self.K, self.H * self.r, -1)

            cand_fg_bg_masks = [ torch.cat([ torch.ones(HALF_D), torch.zeros(HALF_D), torch.ones(TD) ]),
                                 torch.cat([ torch.zeros(HALF_D), torch.ones(HALF_D), torch.ones(TD) ]),
                                 torch.ones(HALF_D * self.H + TD) ]
            
            # fg_bg_mask: [K, in_features]
            fg_bg_mask = torch.stack([ cand_fg_bg_masks[emb_infeat_type] for emb_infeat_type in self.emb_infeat_types ], dim=0)
            fg_bg_mask = fg_bg_mask.to(layer_map_weight_embs.device)
            # layer_map_weight_embs: [K, 2*r, in_features]. fg_bg_mask: [K, 1, in_features]
            layer_map_weight_embs *= fg_bg_mask.unsqueeze(1)
        
    # layer_infeat: 4D image feature tensor [B, C, H, W]. C: 320.
    # layer_idx: 0 ~ 24. emb_idx: 0 ~ 15.
    # time_emb: [B, 1280].
    def forward(self, layer_idx, layer_attn_components, time_emb, static_subj_embs, 
                img_mask=None, bp_to_unet=False, cached_infeat_bg=None):
        emb_idx = self.layer_idx2emb_idx[layer_idx]
        pooler  = self.poolers[emb_idx]
        # Some Linears only use either fg or bg features. 
        # So we mask weights at the unused half, since the corresponding weights are 
        # updated during BP and become nonzero. 
        self.mask_linear_weights(emb_idx)
        
        if self.debug:
            breakpoint()

        with torch.autocast(device_type=self.device_type, enabled=True):
            # basis_dyn_weight: [B, r] = [2, 12].
            # We do not BP into the UNet. So cut off the gradient flow here to reduce RAM and compute.
            # infeat_pooled: [B, C_layer]

            # static_subj_layer_emb should be quite similar to the ada embedding at this layer.
            # Use static_subj_layer_emb as the query to do the attention-based pooling.
            static_subj_layer_emb   = static_subj_embs[emb_idx]
            
            # Even if layer_infeat is grad-scaled, the pooler still receives the full gradient.
            # But the grad is scaled when it's further passed to the UNet.
            if self.use_attn_pooler and self.is_bg_only:
                assert cached_infeat_bg is not None, "cached_infeat_bg must be provided when is_bg_only is True."
                infeat_pooled = cached_infeat_bg
                infeat_fg_bg  = None
                infeat_bg     = None
                # Do not mask weights in this case, as infeat_pooled only contains bg features, and 
                # H = 1, i.e., the in_features of the Linear is not doubled.
            else:
                infeat_pooled    = pooler(layer_attn_components, static_subj_layer_emb, img_mask, bp_to_unet)
                if self.use_attn_pooler:
                    infeat_fg, infeat_bg = infeat_pooled
                    infeat_fg_bg = torch.cat([infeat_fg, infeat_bg], dim=-1)
                else:
                    infeat_fg_bg = infeat_pooled
                    infeat_bg    = None

                infeat_pooled = infeat_fg_bg

            # time_emb has a fixed dimension of 1280. But infeat has variable dimensions.
            # Only use the first TD dimensions of the time embedding, 
            # as the time embedding is highly redundant, and the first TD dimensions are sufficient
            # to capture the temporal information.
            # Note to take the first TD dimensions, instead of the last TD dimensions,
            # as the leading dimensions are most sensitive to time change, 
            # and the last dimensions tend to be the same for all time steps.
            # TD is C_layer/2, so that the time embeddings won't dominate the image features infeat_pooled.
            TD = self.TDs[emb_idx]
            
            ablate_time = False
            if ablate_time:
                time_feat = torch.zeros_like(time_emb[:, :TD])
            else:
                time_feat = time_emb[:, :TD]

            # cat(ln(infeat_pooled), ln(time_emb)) as the input features.
            infeat_time      = self.layer_lncat2s[emb_idx](infeat_pooled, time_feat)

            # basis_dyn_weight: [BS, 2*r*K] => [BS, K, 2*r]
            basis_dyn_weight = self.layer_maps[emb_idx](infeat_time).reshape(-1, self.K, self.H * self.r)
            # bias: [1, K, 768]
            bias = self.bias[emb_idx].unsqueeze(0)

            # self.N: number of pre_vecs.
            if self.N > 0:
                # pre_vecs:   [2, N, 768], basis_vecs: [2, r - N, 768]. 
                # basis_vecs: [2, r, 768].
                # 2 at dim 0: H, i.e., fg and bg.
                basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=1)
            else:
                basis_vecs = self.basis_vecs
            
            # basis_vecs: [2, r, 768] => [2*r, 768].
            # Do not initialize basis_vecs as [2*r, 768] in the first place, so that
            # the K-r flattened basis_vecs will be organized as 
            # [pre_vecs, learned_vecs, pre_vecs, learned_vecs].
            # They correspond to the fg/bg parts of basis_dyn_weight.
            basis_vecs = basis_vecs.view(self.H * self.r, self.out_emb_dim)

            # basis_mask: [K, 2*r].
            # [BS, K, 2*r] * [1, K, 2*r] => [BS, K, 2*r].
            basis_dyn_weight_masked = basis_dyn_weight * self.basis_mask
            # [BS, K, 2*r] x [2*r, 768] = [BS, K, 768]
            # out_vecs_unnorm: [BS, K, 768].
            out_vecs_unnorm = torch.matmul(basis_dyn_weight_masked, basis_vecs)

            out_lns = self.layers_out_lns[emb_idx]
            out_vecs0 = torch.stack([ out_lns[k](out_vecs_unnorm[:, k]) for k in range(self.K) ], dim=1)
            # out_emb_dim: 768.
            out_vecs0 = out_vecs0 / np.sqrt(self.out_emb_dim)
            # [BS, K, 768] + [1, K, 768] = [BS, K, 768].
            out_vecs  = out_vecs0 + bias

            if 'call_count' not in self.__dict__:
                self.call_count = 0

            self.verbose = False
            if self.verbose and self.call_count % 10 == 0:
                calc_stats(f'{emb_idx} time_emb', time_emb[:, :TD])
                calc_stats(f'{emb_idx} infeat_fg_bg', infeat_fg_bg)
                calc_stats(f'{emb_idx} basis_dyn_weight', basis_dyn_weight)
                calc_stats(f'{emb_idx} out_vecs0', out_vecs0)
                calc_stats(f'{emb_idx} bias', bias)

            if emb_idx == 24:
                self.call_count += 1

        # Return infeat_bg to be used by another ada_embedder that specializes on the background.
        return out_vecs, infeat_bg

# text_embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
# = LatentDiffusion.cond_stage_model
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            text_embedder,              
            placeholder_strings=None,
            background_string=None,
            initializer_words=None,
            initializer_weights=None,
            # num_vectors_per_token: how many vectors in each layer are allocated to model 
            # the subject (represented as the subject token) and the background. 
            # num_vectors_per_token is an int or a dict.
            num_vectors_per_token=None,
            use_layerwise_embedding=False,
            num_unet_layers=16,
            # If two tokens, LoRA rank= 2 * layerwise_lora_rank_token_ratio = 4. That means,
            # compress 16 embeddings to the linear combination of 4 embeddings,
            # in which two are initialized as the two token embeddings, and two are learned through BP.
            layerwise_lora_rank_token_ratio=2,
            # If no initializer words are specified, then default LoRA rank = 2.
            layerwise_lora_default_rank=2,
            layer_idx2emb_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                  17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            ada_emb_weight=0.5, 
            ada_use_attn_pooler=True,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = OrderedDict()
        
        self.string_to_param_dict = nn.ParameterDict()
        self.string_to_ada_embedder_dict = nn.ModuleDict()
        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.set_ada_emb_weight(ada_emb_weight, init=True)
        self.ada_use_attn_pooler = ada_use_attn_pooler

        self.use_layerwise_embedding = use_layerwise_embedding
        self.layerwise_lora_rank_token_ratio = layerwise_lora_rank_token_ratio
        self.num_unet_layers    = num_unet_layers
        if self.use_layerwise_embedding:
            # self.num_layers_per_embedder specifies the total layers of embeddings for each embedder.
            # There could be multiple embeddings for each layer.
            self.num_layers_per_embedder = num_unet_layers
        else:
            self.num_layers_per_embedder = 1

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
        self.background_string = background_string

        # hasattr(text_embedder, 'tokenizer') -> True
        if hasattr(text_embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_tokens_for_string = partial(get_clip_tokens_for_string, text_embedder.tokenizer)
            get_embeddings_for_tokens = partial(get_embeddings_for_clip_tokens, text_embedder.transformer.text_model.embeddings)
            self.token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_tokens_for_string = partial(get_bert_tokens_for_string, text_embedder.tknz_fn)
            get_embeddings_for_tokens = text_embedder.transformer.token_emb
            self.token_dim = 1280

        # Save this function to be used in load() when doing placeholder substitution.
        self.get_tokens_for_string = get_tokens_for_string

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

            self.string_to_token_dict[placeholder_string] = token
            # initial_embeddings are only used to compute the regularization loss.
            # Wrap with Parameter so that they will be saved to checkpoints.

            # Initialize static/ada embedders.
            # A static/ada embedder can generate K embeddings.
            # layerwise_lora_rank > 0 implies use_layerwise_embedding.
            if layerwise_lora_rank > 0:
                # num_layers_per_embedder = num_unet_layers
                token_params        = StaticLayerwiseEmbedding(self.num_layers_per_embedder, 
                                                               num_vectors_per_token, 
                                                               self.token_dim, 
                                                               layerwise_lora_rank, 
                                                               (0.1, 0.02), 
                                                               init_words,
                                                               init_word_embeddings, init_word_weights, 
                                                               token_string=placeholder_string)

                if placeholder_string == self.background_string:
                    bg_emb_count = 1
                    fg_emb_count = 0
                    num_vectors_per_token = 1
                else:
                    # Around half of the embeddings are fg embeddings, and the other half are bg embeddings.
                    # If num_vectors_per_token[placeholder_string] = 1, then there's only one fg embedding.
                    bg_emb_count = num_vectors_per_token // 2
                    fg_emb_count = num_vectors_per_token - bg_emb_count

                token_ada_embedder  = AdaEmbedding(self.num_layers_per_embedder, 
                                                   num_vectors_per_token, 
                                                   fg_emb_count, 
                                                   bg_emb_count,
                                                   self.token_dim,                                                    
                                                   layerwise_lora_rank, 
                                                   init_words,
                                                   init_word_embeddings,
                                                   use_attn_pooler=ada_use_attn_pooler,
                                                   token_string=placeholder_string)
            else:
                # Degenerate to Textual Inversion. 
                # ANCHOR[id=init_embed] : 16*K vectors are initialized with the same embedding.
                # avg_init_word_embedding: [1, 768] => [1, 2, 768]
                avg_init_word_embedding = avg_init_word_embedding.unsqueeze(0).repeat(self.num_layers_per_embedder, num_vectors_per_token, 1)
                token_params = nn.Parameter(avg_init_word_embedding, requires_grad=True)
                token_ada_embedder = None

            # token_params: an embedding vector or a StaticLayerwiseEmbedding object (when use_layerwise_embedding).
            # Pytorch >= 1.12.0 allows to put an nn.Module object into an nn.ParameterDict.
            self.string_to_param_dict[placeholder_string] = token_params
            self.string_to_ada_embedder_dict[placeholder_string] = token_ada_embedder

            if init_word_embeddings is not None:
                # initial_embeddings won't be optimized. Just used to compute the regularization loss.
                # Use nn.Parameter to put it on cuda. Not to use register_buffer(), since it's a dict.
                self.initial_embeddings[placeholder_string] = nn.Parameter(init_word_embeddings, requires_grad=False)
            else:
                self.initial_embeddings[placeholder_string] = None

        self.static_subj_embs_dict = None
        self.clear_ada_layer_temp_info()
        self.clear_delta_loss_emb_mask()
        self.img_mask = None
        self.cached_infeat_bg = {}
        self.layer_idx2emb_idx = layer_idx2emb_idx
        self.loss_call_count = 0
        # Store the text_embedder to compute the delta loss.
        self.text_embedder = text_embedder
        self.ada_embeddings = None
        self.ada_bp_to_unet = False
        self.force_grad     = False

        print("EmbeddingManager on {} init with {} vec(s), layerwise_lora_rank={}, ada_emb_weight={}, background_string={}".format(
                placeholder_strings, num_vectors_per_token, layerwise_lora_rank, ada_emb_weight, self.background_string))
            
    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # If self.use_layerwise_embedding, then each token expands to num_unet_layers = 16 
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

        # gen_ada_embedding is dynamically switched on/off by  set_ada_layer_temp_info()/clear_ada_layer_temp_info().
        # No need to calculate delta_loss_emb_mask here, as the mask for ada embeddings is 
        # the same as for the static embeddings. 
        # AdaPrompt combines static and ada embeddings. So the static embedding replacement 
        # code below is always called, and delta_loss_emb_mask is always calculated.
        if self.gen_ada_embedding:
            # self.layer_idx, self.layer_infeat, self.time_emb were cached by 
            # a previous call of  set_ada_layer_temp_info() from UNet.
            ada_embedded_text = \
                self.get_ada_embedding(self.layer_idx, self.layer_attn_components, self.time_emb,
                                       tokenized_text, embedded_text, self.static_subj_embs_dict,
                                       self.ada_bp_to_unet)
            # Release ada-specific intermediate variables.
            self.clear_ada_layer_temp_info()

            return ada_embedded_text

        else:
            # Update the token embedding mask used in embedding loss computation.
            self.update_emb_mask(tokenized_text)
            # Clear the cached placeholder indices, and store the new ones computed 
            # in the current function call.
            self.placeholder_indices = None
            # We need to clone embedded_text, as sometimes (when it's not layerwise such as TI) 
            # the modification in get_static_embedding() is in-place. 
            static_embeddings, static_subj_embs_dict = \
                            self.get_static_embedding(tokenized_text, embedded_text.clone(), 
                                                      self.string_to_param_dict,
                                                      B, N, self.num_unet_layers, device)
            # Cache the static embeddings to be used in ada embedding computation later.
            self.static_subj_embs_dict = static_subj_embs_dict

            return static_embeddings
    
    # N: length of sequence (including padding).
    def get_static_embedding(self, tokenized_text, embedded_text, embedder_dict, 
                             B, N, num_unet_layers, device):
        orig_tokenized_text = tokenized_text
        static_subj_embs_dict = {}

        if self.use_layerwise_embedding:
            # embedded_text: [B, N, 768] => [B, 16, N, 768] => [16*B, N, 768].
            # "Tuck" the layer dimension into the batch dimension, 
            # to keep embedded_text in 3D, same as the input.
            embedded_text = embedded_text.unsqueeze(1).repeat(1, num_unet_layers, 1, 1).view(B * num_unet_layers, N, -1)
            # tokenized_text: [B, 16, N] => [16*B, N]
            # tokenized_text has to be repeated along the layer dimension as well, so that 
            # placeholder_indices can index the embedding at each layer in the batch.
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1, num_unet_layers, 1).view(B * num_unet_layers, N)
            # mirror-reflect the embedding along the layer dimension, to make it symmetric 
            # in the encoder & decoder.

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            # If there's only one vector per token, we can do a simple replacement
            placeholder_indices = torch.where(tokenized_text == placeholder_token.to(device))
            # No placeholder token is found in the current batch.
            if placeholder_indices[0].numel() == 0:
                continue
            
            # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
            # Non-layerwise: embedded_text[placeholder_indices]: [2, 768].  placeholder_embeddings: [1, K, 768].
            # layerwise: placeholder_indices =  
            # (tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 
            #  tensor([ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]))
            # embedded_text[placeholder_indices]: [32, 768]. placeholder_embeddings: [16, K, 768].
            # The first 16 elements (0-8) in embedded_text[placeholder_indices] correspond to the 16 layers of the 
            # first instance in the batch.
            # 16 layers of placeholder_embeddings are repeated B times.
            # placeholder_embeddings: [16, 768] repeat=> [32, 768]
            # Note that the 16 layers are initialized with the same embedding. 
            # LINK #init_embed

            # REAL_OCCURS: the real number of occurrences of the placeholder in the current batch,
            # not repetitively counting the occurrences in the embedded_text repeated for M layers.
            REAL_OCCURS = placeholder_indices[0].numel() // self.num_layers_per_embedder

            static_embedder = embedder_dict[placeholder_string].to(device)
            if isinstance(static_embedder, StaticLayerwiseEmbedding):
                # Generate the actual placeholder_embeddings on the fly.
                # The 16 static subject embeddings are formed by linearly combining the basis vectors.
                # The matrix operations are done on the fly.
                # placeholder_embeddings: [16, K, 768].
                placeholder_embeddings = static_embedder()
                static_subj_embs_dict[placeholder_string] = placeholder_embeddings
            else:
                # static_embedder is already the embedding.
                placeholder_embeddings = static_embedder

            for k in range(self.token2num_vectors[placeholder_string]):
                placeholder_indices_k = (placeholder_indices[0], placeholder_indices[1] + k)
                # embedded_text is repeated 16 times along the layer dimension, with size of dim 0 = 16 * BS.
                # The first dim of placeholder_embeddings is the layer dim (size = 16). 
                # So we repeat it BS times to match the batch dim.
                embedded_text[placeholder_indices_k] = placeholder_embeddings[:, k].repeat(REAL_OCCURS, 1)
                
            # Cache the placeholder indices for mix prompt distillation.
            # Note placeholder_indices are recomputed in update_placeholder_indices(), 
            # we don't simply cache placeholder_indices here as they are repeated 16 times 
            # to replace in 16 layers. 
            # But we need them without repetitions for mix prompt distillation.
            # If num_vectors_per_token > 1, then repeat the indices and add to offsets.
            # If background_string is None, then always update the indices. Otherwise, 
            # skip updating placeholder indices of the background string.
            self.update_placeholder_indices(orig_tokenized_text, placeholder_token, self.token2num_vectors[placeholder_string],
                                            is_fg_token=(placeholder_string != self.background_string))

        return embedded_text, static_subj_embs_dict

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # As the output embedding is only generated for a particular layer, 
    # no need to repeat num_unet_layers times as replacing with the static embedding in forward().
    def get_ada_embedding(
            self,
            layer_idx,              # the index of the current layer in the UNet.
            layer_attn_components,  # intermediate features and projections of the UNet attention layer.
            time_emb,               # time embedding of the current iteration.
            tokenized_text,         # [B, N]. Identical B copies along the batch dimension.
            embedded_text,          # [B, N, 768]. Identical B copies along the batch dimension.
            static_subj_embs_dict,  # A dictionary of subject embeddings, used to do attentional pooling.
            ada_bp_to_unet=False
    ):
        device = tokenized_text.device

        assert self.use_layerwise_embedding, "Non-layerwise embedding cannot call get_ada_embedding()."

        # string_to_token_dict is an OrderedDict, with subject tokens added first, and 
        # the background token last (order controlled in main.py). 
        # This order ensures that the background Ada embedder can always use 
        # cached_infeat_bg produced by the previous subject Ada embedder.
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            # There's only one vector per token, we can do a simple replacement
            # embedded_text: [B, N, 768].
            # tokenized_text: [B, N].
            placeholder_indices = torch.where(tokenized_text == placeholder_token.to(device))
            # Skip generating the ada embedding if there's no placeholder token in the batch.
            if placeholder_indices[0].numel() == 0:
                continue

            # Clear the infeat_bg cache before generating subject embedding(s) of a subject token.
            # If the next token is the background token, the keep the cache, so that the background
            # token will reuse the infeat_bg computed by the previous subject token.
            if placeholder_string != self.background_string:
                self.cached_infeat_bg[layer_idx] = None

            ada_embedder = self.string_to_ada_embedder_dict[placeholder_string].to(device)
            assert isinstance(ada_embedder, AdaEmbedding)

            # When it's the turn of the background Ada embedder, the cached_infeat_bg
            # should have been computed by the previous subject Ada embedder. 
            # Otherwise it's a bug.
            if placeholder_string == self.background_string \
                and ada_embedder.is_bg_only and self.cached_infeat_bg[layer_idx] is None:
                breakpoint()

            # Generate the actual placeholder_embeddings on the fly.
            # placeholder_embeddings: [B, K, 768]. B: 2 or 4 (regularization batches).
            # Before this call, we assume static_subj_embs has been generated by 
            # a call to get_static_embedding(). 
            # The pipeline is generate static embeddings first, then generate the ada embeddings. 
            # So this assumption should always hold.
            # For background Ada embedder, cached_infeat_bg is only used when
            # is_bg_only. Otherwise, it's ignored.
            placeholder_embeddings, infeat_bg = \
                        ada_embedder(layer_idx, layer_attn_components, time_emb,
                                     # If K > 1, only take the first embedding out of the K embeddings.
                                     static_subj_embs_dict[placeholder_string][:, 0],
                                     self.img_mask, ada_bp_to_unet, self.cached_infeat_bg[layer_idx])
            
            if placeholder_string != self.background_string:
                # Cache the bg infeat computed by the first (fg) ada embedder, 
                # to be used by the second Ada embedder and the background Ada embedder.
                self.cached_infeat_bg[layer_idx] = infeat_bg

            for k in range(self.token2num_vectors[placeholder_string]):
                # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
                # embedded_text[placeholder_indices]: [2, 768].  placeholder_embeddings: [2, 768].
                # Sometimes (e.g. during inference, some instances contain the placeholder token but
                # others don't. tokenized_text has a batch size of those containing the placeholder token only.
                # But layer_infeat is still of the full batch size. So placeholder_embeddings has a batch size 
                # larger than the number of instances containing the placeholder token. We need to index
                # placeholder_embeddings with placeholder_indices[0] to get the matching new placeholder_embeddings.
                # We can save some computation by generating embeddings only for the instances containing 
                # the placeholder token. But that requires complex processing so not pursued now.
                placeholder_indices_k = (placeholder_indices[0], placeholder_indices[1] + k)
                # placeholder_embeddings: [BS, K, 768]. BS: 2 or 4 (regularization batches).
                embedded_text[placeholder_indices_k] = placeholder_embeddings[placeholder_indices[0], k]
            
        return embedded_text

    # Update token embedding mask.
    def update_emb_mask(self, tokenized_text):
        # Exclude the starting and padding tokens from delta loss.
        delta_loss_emb_mask  = (tokenized_text != 49406 ) & (tokenized_text != 49407)
        # [B, N] => [B, 1, N, 1]
        delta_loss_emb_mask  = delta_loss_emb_mask.float().unsqueeze(1).unsqueeze(3)

        self.set_delta_loss_emb_mask(delta_loss_emb_mask)

    def update_placeholder_indices(self, tokenized_text, placeholder_token, num_vectors_per_token, is_fg_token):
        placeholder_indices_B, placeholder_indices_N = \
            torch.where(tokenized_text == placeholder_token.to(tokenized_text.device))
        # placeholder_indices0: only indices the first embedding of each multi-embedding token.
        placeholder_indices0 = (placeholder_indices_B, placeholder_indices_N)

        if num_vectors_per_token > 1:
            BS = placeholder_indices_B.shape[0]
            placeholder_indices_B = placeholder_indices_B.unsqueeze(1).repeat(1, num_vectors_per_token).view(-1)
            placeholder_indices_N = placeholder_indices_N.unsqueeze(1).repeat(1, num_vectors_per_token).view(-1)
            # Add offsets to the indices of the pseudo-tokens.
            placeholder_indices_N_off = placeholder_indices_N + torch.arange(num_vectors_per_token, device=tokenized_text.device).repeat(BS)
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N_off)
        else:
            # placeholder_indices contains the indices of all placeholder embeddings.
            placeholder_indices = (placeholder_indices_B, placeholder_indices_N)
        
        if is_fg_token:
            self.placeholder_indices_fg, self.placeholder_indices_fg0 = placeholder_indices, placeholder_indices0
        else:
            # background token only has one embedding. So no need to separately store placeholder_indices_bg0.
            self.placeholder_indices_bg = placeholder_indices
            
    def get_ada_emb_weight(self):
        if self.training:
            # 0.5 -> uniform in [0.4, 0.7]. Inject randomness to reduce overfitting.
            rand_ada_emb_weight = self.ada_emb_weight * np.random.uniform(0.8, 1.4)
        else:
            rand_ada_emb_weight = self.ada_emb_weight        
        return rand_ada_emb_weight
    
    def set_ada_emb_weight(self, ada_emb_weight, init=False):
        if init:
            print(f"Setting ada_emb_weight = {ada_emb_weight}")
        else:
            if self.ada_emb_weight != ada_emb_weight:
                print(f"ada_emb_weight: {self.ada_emb_weight} => {ada_emb_weight}")
        self.ada_emb_weight = ada_emb_weight

    def set_ada_layer_temp_info(self, layer_idx, layer_attn_components, time_emb, ada_bp_to_unet):
        self.gen_ada_embedding = True
        self.layer_idx      = layer_idx
        self.time_emb       = time_emb
        self.ada_bp_to_unet = ada_bp_to_unet
        self.layer_attn_components = layer_attn_components
        # Initialize the ada_embeddings cache list.

    # ada_embeddings is used to cache the embeddings of all layers, 
    # for computing the prompt delta loss.
    def reset_ada_embedding_cache(self):
        self.ada_embeddings = [ None for i in range(self.num_unet_layers) ]

    def cache_ada_embedding(self, i, embedding):
        emb_idx = self.layer_idx2emb_idx[i]
        self.ada_embeddings[emb_idx] = embedding

    def set_num_vectors_per_token(self, num_vectors_per_token, placeholder_strings=None):
        if num_vectors_per_token is None or type(num_vectors_per_token) == int:
            # If token2num_vectors is not specified, then set all tokens to have 1 vector.
            # If token2num_vectors is an int, then set all tokens to have 
            # token2num_vectors vectors.
            if num_vectors_per_token is None:
                num_vectors_per_token = 1

            # During inference, placeholder_strings is None. So we use string_to_token_dict 
            # loaded from the ckpt.
            if placeholder_strings is None:
                placeholder_strings = self.string_to_token_dict.keys()
            for k in placeholder_strings:
                self.token2num_vectors[k] = num_vectors_per_token
        else:
            self.token2num_vectors = num_vectors_per_token
        print(f"Set token2num_vectors: {self.token2num_vectors}")

    # Clear layer-specific intermediate variables. Also clear gen_ada_embedding,
    # which will be enabled again through set_ada_layer_temp_info() in ddpm.py.
    def clear_ada_layer_temp_info(self):
        self.gen_ada_embedding = False
        self.layer_idx      = -1
        self.layer_attn_components  = None
        self.time_emb       = None
        
    def clear_ada_embedding_cache(self):
        self.ada_embeddings = None

    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # DDPM.validation_step() -> LatentDiffusion.shared_step() -> .forward()
    # -> .get_learned_conditioning() -> .cond_stage_model.encode()
    # -> EmbeddingManager.forward() -> here.
    # Occasionally, image_logger is called, which calls LatentDiffusion.log_images ->
    # .get_learned_conditioning() -> ... -> here.
    # Such delta_loss_emb_mask won't be used in calc_prompt_delta_loss() and won't be cleared.
    # delta_loss_emb_mask: [B, N, 768], where N is the padded prompt length.
    def set_delta_loss_emb_mask(self, delta_loss_emb_mask):
        self.delta_loss_emb_mask = delta_loss_emb_mask

    def clear_delta_loss_emb_mask(self):
        self.delta_loss_emb_mask = None
 
    # There are image margins after the original image is scaled down.
    # When doing average pooling of image features, the margin area contains no signal, so we use 
    # img_mask to mask it out. 
    # Each image has its own img_mask, so img_mask has a shape of [B, 1, H, W].
    def set_img_mask(self, img_mask):
        self.img_mask = img_mask

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, ckpt_path):
        torch.save({ "string_to_token":         self.string_to_token_dict,
                     "string_to_param":         self.string_to_param_dict,
                     "string_to_ada_embedder":  self.string_to_ada_embedder_dict,
                     "ada_emb_weight":          self.ada_emb_weight,  
                     "token2num_vectors":   self.token2num_vectors,
                   }, 
                    ckpt_path)

    # load custom tokens and their learned embeddings from "embeddings_gs-4200.pt".
    def load(self, ckpt_paths):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        self.string_to_token_dict           = {}
        self.string_to_param_dict           = nn.ParameterDict()
        self.string_to_ada_embedder_dict    = nn.ModuleDict()
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
                self.string_to_token_dict[k2]        = k2_token

                # Mapped from k in ckpt to k2 in the current session.
                for km in ckpt["string_to_param"].keys():
                    # If there are pseudo-tokens within multi-embedding tokens, load them as well.
                    if km.startswith(k):
                        km2 = km.replace(k, k2)
                        self.string_to_param_dict[km2] = ckpt["string_to_param"][km]
                        self.string_to_ada_embedder_dict[km2] = ckpt["string_to_ada_embedder"][km]
                        if km in ckpt["token2num_vectors"]:
                            token2num_vectors[km2] = ckpt["token2num_vectors"][km]
                        print(f"Loaded {km}->{km2} from {ckpt_path}")

            # If multiple checkpoints have different ada_emb_weight, the last one will be used.
            if "ada_emb_weight" in ckpt:
                self.set_ada_emb_weight(ckpt["ada_emb_weight"])
            if "token2num_vectors" in ckpt:
                self.set_num_vectors_per_token(token2num_vectors)

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def optimized_parameters(self):
        params = list(self.string_to_param_dict.parameters()) \
               + list(self.string_to_ada_embedder_dict.parameters())
        
        return params
        
    def embedding_attractor_loss(self):
        loss = 0.
        num_placeholders = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
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
            embeddings = self.string_to_param_dict[key]
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
        loss = 0.
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
            for embobj in (self.string_to_param_dict[key], 
                           self.string_to_ada_embedder_dict[key]):
                # Skip non-layerwise embeddings.
                if not isinstance(embobj, StaticLayerwiseEmbedding) \
                  and not isinstance(embobj, AdaEmbedding):
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
                    loss_bias        = selective_reg_loss(embobj.bias, loss_type=euc_loss_type)
                    # bias_reg_weight is computed dynamically. But it's detached from the graph.
                    # embobj.bias: now [Layers, K, 768].
                    bias_reg_weight  = bias_reg_weight_base  * torch.norm(embobj.bias, dim=-1).mean().item() ** T
                else:
                    loss_bias        = 0.
                    bias_reg_weight  = 0.

                loss_basis       = selective_reg_loss(embobj.basis_vecs, loss_type=euc_loss_type)
                # Like bias_reg_weight, basis_reg_weight is also computed dynamically 
                # and detached from the graph.
                # basis_vecs: now [K, r-N, 768] for Ada embedder, or [r-N, 768] for Static embedder.
                basis_reg_weight = basis_reg_weight_base * torch.norm(embobj.basis_vecs, dim=-1).mean().item() ** T

                # N: number of pre_vecs (init_vecs).
                if embobj.N > 0:
                    # If pre_vecs has a K dim (shape [K, 1, 768]), then init_vecs is automatically broadcasted.
                    loss_pre_vecs = selective_reg_loss(embobj.pre_vecs - init_vecs, loss_type=euc_loss_type)
                else:
                    loss_pre_vecs = 0.

                loss_ada_maps_weight = 0.
                loss_ada_maps_bias   = 0.
                loss_ada_attn_pooler = 0.
                
                if isinstance(embobj, AdaEmbedding):
                    for i, map in enumerate(embobj.layer_maps):
                        loss_ada_maps_weight += selective_reg_loss(map.weight, loss_type=euc_loss_type)
                        loss_ada_maps_bias   += selective_reg_loss(map.bias,   loss_type=euc_loss_type)
                    if self.ada_use_attn_pooler:
                        for i, pooler in enumerate(embobj.poolers):
                            loss_ada_attn_pooler  += selective_reg_loss(pooler.lora_to_k.weight, loss_type=euc_loss_type)
                            loss_ada_attn_pooler  += selective_reg_loss(pooler.lora_to_q.weight, loss_type=euc_loss_type)
                            
                if type(loss_bias) == int:
                    breakpoint()

                # If it's Ada embedder, times K (eff_K = K) to account for each of the K embeddings.
                eff_K = 1 if isinstance(embobj, StaticLayerwiseEmbedding) else embobj.K
                num_out_embeddings += eff_K

                # The losses of most components are times eff_K, except for ada attn pooler, 
                # whose parameters don't inflate with K.
                curr_loss = (  loss_bias             * bias_reg_weight \
                             + loss_basis            * basis_reg_weight \
                             + loss_pre_vecs         * pre_vecs_reg_weight \
                             + loss_ada_maps_weight  * ada_maps_weight_reg_weight \
                             + loss_ada_maps_bias    * ada_maps_bias_reg_weight ) * eff_K \
                             + loss_ada_attn_pooler  * ada_attn_poolers_reg_weight
                
                debug = True
                if debug and self.loss_call_count % 100 == 0:
                    print_str = f'reg_bias={loss_bias.item():.4f}, ' \
                                f'reg_basis={loss_basis.item():.4f}, '
                    if embobj.N > 0:
                        print_str += f'reg_pre_vecs={loss_pre_vecs.item():.4f}, '

                    if isinstance(embobj, AdaEmbedding):
                        print_str += f'loss_ada_maps_weight={loss_ada_maps_weight.item():.4f}, ' \
                                     f'loss_ada_maps_bias={loss_ada_maps_bias.item():.4f}'

                    print(print_str)

                # default: l2
                if euc_loss_type   == 'l2':
                    if isinstance(embobj, AdaEmbedding):
                        loss_boost = ada_l2_loss_boost      # 10
                    else:
                        loss_boost = static_l2_loss_boost   # 5
                else:
                    # l1 loss is numerically much larger than l2 loss, 
                    # hence it has smaller weight than l2 loss.
                    loss_boost = 1.
                loss = loss + curr_loss * loss_boost

        return loss / num_out_embeddings

    def embedding_to_loss(self):
        self.loss_call_count += 1
        if self.use_layerwise_embedding:
            if self.layerwise_lora_rank_token_ratio > 0:
                return self.layerwise_embedding_norm_loss()
            else:
            # Do not use. Performs poorly. 
                return self.layerwise_embedding_attractor_loss()
        else:
            return self.embedding_attractor_loss()

    # Textual inversion is supported, where static_embeddings is only one embedding.
    # static_embeddings: size: [8*16, 77, 768]. 8 = 4 * batch_size. 16: number of UNet layers.
    # embeddings of static_subj_single_emb, static_subj_comp_emb, static_cls_single_emb, static_cls_comp_emb. 
    def calc_prompt_delta_loss(self, static_embeddings, ada_embeddings, do_ada_prompt_delta_reg):
        if self.use_layerwise_embedding:
            num_embed_layers = self.num_unet_layers
        else:
            num_embed_layers = 1

        # num_unet_layers = 16. 
        # If do_ada_prompt_delta_reg, then static_embeddings: [64, 77, 768]. 
        # So BS = 1.
        # static_embeddings / ada_embeddings contain 4 types of embeddings:
        # subj_single, subj_comp, cls_single, cls_comp.
        BS = static_embeddings.shape[0] // (4 * num_embed_layers)
        # static_embeddings: [64, 77, 768] => [4, 16, 77, 768]
        static_embeddings = static_embeddings.view(BS * 4, num_embed_layers, *static_embeddings.shape[1:])

        # cls_*: embeddings generated from prompts containing a class token (as opposed to the subject token).
        # Each is [1, 16, 77, 768]
        static_subj_single_emb, static_subj_comp_emb, static_cls_single_emb, static_cls_comp_emb = \
                static_embeddings.chunk(4)

        if self.delta_loss_emb_mask is not None:
            subj_single_mask, subj_comp_mask, cls_single_mask, cls_comp_mask = \
                    self.delta_loss_emb_mask.chunk(4)
            
            # cls_single_mask == subj_single_mask, cls_comp_mask == subj_comp_mask
            # So only compute using subj_single_mask and subj_comp_mask.
            delta_loss_emb_mask = subj_single_mask + subj_comp_mask
            # If a token appears both in single and comp prompts (all tokens in the single prompts), 
            # the mask value is 2. Convert to 1.
            # If a token appears in only comp prompts (the extra compositional part), 
            # the mask value is 1. Convert to 0.25.
            # If a token is useless (z suffix or padding), the mask value is 0. Convert to 0.
            delta_loss_emb_mask = delta_loss_emb_mask.pow(2) / 4
        else:
            delta_loss_emb_mask = None

        use_ortho_subtract = True
        # cls_delta: [1, 16, 77, 768]. Should be a repeat of a tensor of size [1, 1, 77, 768]. 
        # Delta embedding between class single and comp embeddings.
        # by 16 times along dim=1, as cls_prompt_* doesn't contain placeholder_token.
        # static_delta: [1, 16, 77, 768]. Different values for each layer along dim=1.
        # Delta embedding between subject single and comp embeddings.
        # static_delta / ada_delta should be aligned with cls_delta.
        if use_ortho_subtract:
            cls_delta    = ortho_subtract(static_cls_comp_emb,  static_cls_single_emb)
            static_delta = ortho_subtract(static_subj_comp_emb, static_subj_single_emb)
        else:
            cls_delta    = static_cls_comp_emb - static_cls_single_emb
            static_delta = static_subj_comp_emb - static_subj_single_emb

        static_delta_loss   = calc_delta_loss(static_delta, cls_delta, delta_loss_emb_mask)

        if do_ada_prompt_delta_reg and ada_embeddings is not None:
            # ada_embeddings: [4, 16, 77, 768]
            if isinstance(ada_embeddings, (list, tuple)):
                # twin_single_ada_embeddings: [2, 16, 77, 768]
                # twin_comp_ada_embeddings:   [4, 16, 77, 768]
                # twin_single_ada_embeddings correspond to 2 subj single instances,
                # twin_comp_ada_embeddings correspond to 2 subj comp, 2 mix comp instances.
                # mix comp ada embeddings are generated with the same comp prompts as subj comp.
                # But some layers of the static embeddings are different, so the ada embeddings
                # are different from layer 5.
                twin_single_ada_embeddings, twin_comp_ada_embeddings = ada_embeddings
                # ada_subj_comp_emb: [2, 16, 77, 768]
                ada_subj_comp_emb, ada_cls_comp_emb \
                    = twin_comp_ada_embeddings.chunk(2)
                # ada_subj_single_emb: [2, 16, 77, 768].
                ada_subj_single_emb = twin_single_ada_embeddings
                # static embeddings for the twins in the batch are the same, as the prompts are the same.
                # They only differ on initial noises.
                # So simply repeat cls_delta at the batch dim to match the twin ada embeddings.
                cls_delta = cls_delta.repeat(2, 1, 1, 1)
                delta_loss_emb_mask = delta_loss_emb_mask.repeat(2, 1, 1, 1)
            else:
                # ada_cls_single_emb, ada_cls_comp_emb should be the same as 
                # static_cls_single_emb, static_cls_comp_emb, as class prompts do not contain 
                # the subject token.
                ada_subj_single_emb, ada_subj_comp_emb, ada_cls_single_emb, ada_cls_comp_emb \
                    = ada_embeddings.chunk(4)

            if use_ortho_subtract:
                ada_delta = ortho_subtract(ada_subj_comp_emb, ada_subj_single_emb)
            else:
                ada_delta = ada_subj_comp_emb - ada_subj_single_emb

            ada_delta_loss = calc_delta_loss(ada_delta, cls_delta, delta_loss_emb_mask)
            # The cached ada embeddings are useless now, release them.
            self.clear_ada_embedding_cache()
        else:
            ada_delta_loss = 0
        
        # self.clear_delta_loss_emb_mask()
        # delta_loss = static_delta_loss + ada_delta_loss * ada_comp_loss_boost_ratio
        return static_delta_loss, ada_delta_loss

if __name__ == '__main__':
    # The example code below is obsolete.    
    attnpool = AttentionalPooler()
    x = torch.randn(2, 768, 16, 16)
    mask = torch.randint(2, size=(2, 1, 16, 16)).float()
    y = attnpool(x, x, mask)
    print(y.shape)
