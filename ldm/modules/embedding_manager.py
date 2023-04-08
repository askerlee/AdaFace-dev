import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

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

def demean(x):
    return x - x.mean(dim=-1, keepdim=True)

# Eq.(2) in the StyleGAN-NADA paper.
# delta, ref_delta: [2, 16, 77, 768].
# emb_mask: [2, 77, 1]
def calc_delta_loss(delta, ref_delta, emb_mask=None, exponent=3, do_LN_first=True):
    # Mask out the placeholder suffix token(s).
    # If CLIP skip scheme is "concat", then the text embedding channel number is doubled.
    # In this case, we also duplicate the mask along the channel dimension.
    if (emb_mask is not None) and emb_mask.shape[1] == delta.shape[1] // 2:
        emb_mask = emb_mask.repeat(1, 2, 1)

    try:
        delta = delta * emb_mask if emb_mask is not None else delta
    except:
        breakpoint()

    # Flatten delta and ref_delta, by tucking the layer and token dimensions into the batch dimension.
    # dela: [2464, 768], ref_delta: [2464, 768]
    delta = delta.view(delta.numel() // delta.shape[-1], -1)
    ref_delta = ref_delta.view(ref_delta.numel() // ref_delta.shape[-1], -1)

    # A bias vector to a set of conditioning embeddings doesn't change the attention matrix 
    # (though changes the V tensor). So the bias is better removed.
    # Therefore, do layer normalization before cosine loss, 
    # to remove the effect of bias.
    # IN addition, different ada layers have significantly different scales. 
    # Since cosine is scale invariant, the de-scale is not necessary.
    # LN = demean & de-scale. So LN is equivalent to demean() here.
    if do_LN_first:
        #delta = demean(delta)
        #ref_delta = demean(ref_delta)
        delta     = F.layer_norm(delta, delta.shape[1:])
        ref_delta = F.layer_norm(ref_delta, ref_delta.shape[1:])

    # x * x.abs.pow(exponent - 1) will keep the sign of x after pow(exponent).
    ref_delta_pow = ref_delta * ref_delta.abs().pow(exponent - 1)
    loss = F.cosine_embedding_loss(delta, ref_delta_pow.detach(), 
                                   torch.ones_like(delta[:, 0]))
    return loss

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

    # x: [N, C, H, W], mask: [N, 1, H0, W0]. 
    # H, W: feature map size, H0, W0: original image size.
    # Return: [N, C]
    def forward(self, x, mask=None):
        if mask is None:
            return self.avgpool(x).view(x.shape[0], -1)
        
        mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
        x = x * mask
        x = x.sum(dim=(2,3)) / mask.sum(dim=(2,3))
        return x
        
class StaticLayerwiseEmbedding(nn.Module):
    # dim1: 16 (9 layers out of 25 of UNet are skipped), dim2: 768, r: 12.
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
    def __init__(self, dim1=16, dim2=768, r=12, init_noise_stds=(0.1, 0.04), init_vecs=None, 
                 init_vec_weights=None, init_neg_vecs=None, has_bias=True, device_type="cuda"):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.r = r

        if r > min(dim1, dim2):
            raise ValueError(
                f"StaticLayerwiseEmbedding LoRA rank {r} must be less or equal than {min(dim1, dim2)}"
            )

        if init_vecs is not None:
            # Only one vector is passed in.
            if init_vecs.ndim == 1:
                init_vecs = init_vecs.unsqueeze(0)

            if init_vecs.shape[1] != dim2 or init_vecs.shape[0] > dim1:
                raise ValueError(
                    f"StaticLayerwiseEmbedding init vectors shape {init_vecs.shape} must be (<={dim1}, {dim2})"
                )

            N = self.N = init_vecs.shape[0]
            # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
            # pre_vecs are updated through BP.
            self.pre_vecs = nn.Parameter(init_vecs.clone(), requires_grad=True)
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs.data, dim=1)
        else:
            self.N = 0
            self.pre_vecs = None

        # basis_rand_weights: 16 * r, basis_vecs: r * 768. basis_rand_weights * basis_vecs: 16 * 768.
        self.basis_rand_weights    = nn.Parameter(torch.randn(dim1, r))
        # basis_vecs consists of r basis vectors. Will be updated through BP.
        self.basis_vecs = nn.Parameter(torch.randn(r - N, dim2), requires_grad=True)
        # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
        self.basis_vecs.data = F.normalize(self.basis_vecs, dim=1) / 4.
        # Always set the last basis vector to 0.
        self.basis_vecs.data[-1] = 0

        self.has_bias    = has_bias
        self.device_type = device_type
        # basis_comm_weights are initialized as equal weights, then tuned through BP.
        # basis_comm_weights is added with basis_rand_weights. basis_rand_weights is a random component of the actual weights.
        # So equal weights here won't cause equal graidents (and non-identifiability of parameters).
        self.basis_comm_weights = nn.Parameter(torch.ones(1, r) / r)

        # init_up_noise_stds is only applied when init_vecs is passed in.
        if init_vecs is not None:
            self.basis_comm_weights.data.fill_(1. / N)
            # Lower the weights of the remaining r-N random vectors, to prevent the model  
            # from going too far away from the subspace spanned with init_vecs.
            # By default, these weights are 1/6*0.4 = 0.067.
            self.basis_comm_weights.data[:, N:] *= 0.4

            # basis_comm_weights: [1, r]
            if init_vec_weights is not None:
                assert len(init_vec_weights) == len(init_vecs), f"init_vec_weights must have length {len(init_vecs)}"
                # Assume init_vec_weights is already normalized, i.e., init_vec_weights.sum() = 1.
                self.basis_comm_weights.data[:, :N] = init_vec_weights.unsqueeze(0)

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
            self.basis_rand_weights.data[:, :N]    *= init_noise_stds[1]
            # The last dim1-N block are coefficients of the extra learned vectors.
            # We don't want the result embeddings to be confined 
            # in the subspace of basis_vecs. So we let the extra learned vectors play a bigger role,
            # by making the noises larger (init_noise_stds[0]=0.1).
            self.basis_rand_weights.data[:, N:]    *= init_noise_stds[0]
        else:
            self.N = 0

        if init_neg_vecs is not None:
            # NEG: number of negative initial vectors.
            NEG = init_neg_vecs.shape[0]
            self.NEG = NEG
            # The no. N~N+NEG vectors in basis_vecs are init_neg_vecs (no noise added).
            # The remaining dim1-(N+NEG) vectors are gaussian noises on the unit ball.
            self.basis_vecs.data[:NEG] = -init_neg_vecs.clone()
            # self.basis_comm_weights.data[N:N+NEG] = 1. / NEG
            # Do not tinker with the columns corresponding to negative vectors in basis_rand_weights.
        else:
            self.NEG = 0

        if self.has_bias:
            # bias: 16 * 768.
            self.bias        = nn.Parameter(torch.zeros(dim1, dim2))
        else:
            self.bias = 0

        layer_lns  = []
        for i in range(dim1):
            layer_lns.append( nn.LayerNorm(dim2, elementwise_affine=True) )

        self.layer_lns  = nn.ModuleList(layer_lns)

        print(f"StaticLayerwiseEmbedding initialized with {self.N} init vectors, {self.NEG} negative vectors, {self.r} basis vectors")

    def forward(self, only_bias=False):
        with torch.autocast(device_type=self.device_type, enabled=False):
            if only_bias:
                return self.bias

            # self.basis_comm_weights: [1, r] broadcasted to [16, r].
            basis_weights   = self.basis_rand_weights   + self.basis_comm_weights
            # torch.matmul: matrix multiplication.
            # torch.matmul(lora_up, basis_vecs): 16 * 768.

            if self.N > 0:
                basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=0)
            else:
                basis_vecs = self.basis_vecs

            out_vecs = torch.matmul(basis_weights, basis_vecs)
            # Apply layer-wise layer normalization.
            out_vecs_ln = [ self.layer_lns[i](out_vecs[i]) for i in range(self.dim1) ]
            out_vecs_ln = torch.stack(out_vecs_ln, dim=0) / np.sqrt(self.dim2)

            # Different layers have different biases.
            out_vecs_ln = out_vecs_ln + self.bias
            return out_vecs_ln

class AdaEmbedding(nn.Module):
    # dim1: 16 (9 layers out of 25 of UNet are skipped).
    # dim2: 768, r: 12.
    # infeat_dims: a list of 25 integers, each is the dimension of 
    # the input feature from the respective layer. 9 of them are skipped.
    # infeat_dims are (almost) reflective around the middle layer, except for the first and last layers.
    # Layer indices absent in layer_idx2emb_idx are skipped layers.
    def __init__(self, dim1=16, dim2=768, r=12, init_vecs=None, 
                 infeat_dims = [ 4,    320,  320,  320,  320,  640,  640,  640, 1280, 1280, 1280, 1280, 
                                 1280,
                                 1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640,  640,  320,  320 ],
                 # skipped_layers = [0, 3, 6, 9, 10, 11, 13, 14, 15],
                 layer_idx2emb_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                       17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },
                 has_bias=True, device_type="cuda"):
        super().__init__()

        assert dim1 == len(layer_idx2emb_idx), f"dim1={dim1} != len(layer_idx2emb_idx)={len(layer_idx2emb_idx)}"
        self.dim1 = dim1
        self.dim2 = dim2
        self.r = r
        self.device_type = device_type
        self.layer_idx2emb_idx = layer_idx2emb_idx
        self.emb_idx2layer_idx = { v: k for k, v in layer_idx2emb_idx.items() }

        if init_vecs is not None:
            # Only one vector is passed in.
            if init_vecs.ndim == 1:
                init_vecs = init_vecs.unsqueeze(0)

            if init_vecs.shape[1] != dim2 or init_vecs.shape[0] > dim1:
                raise ValueError(
                    f"AdaEmbedding LoRA init vectors shape {init_vecs.shape} must be (<={dim1}, {dim2})"
                )

            N = self.N = init_vecs.shape[0]
            self.pre_vecs = nn.Parameter(init_vecs.clone(), requires_grad=True)
            # Normalize pre_vecs, to roughly equalize the contributions of different predefined basis vectors.
            # self.pre_vecs.data = F.normalize(self.pre_vecs, dim=1)
        else:
            self.N = 0
            self.pre_vecs = None

        # basis_vecs: [12, 768], consists of r-N basis vectors. Will be updated through BP.
        self.basis_vecs = nn.Parameter(torch.randn(r - N, dim2), requires_grad=True)
        # Normalize basis_vecs, to roughly equalize the contributions of different random vectors.
        self.basis_vecs.data = F.normalize(self.basis_vecs, dim=1) / 4.
        # Always set the last basis vector to 0.
        self.basis_vecs.data[-1] = 0

        self.infeat_dims = list(infeat_dims)
        self.avgpool = MaskedAvgPool2d() # nn.AdaptiveAvgPool2d((1, 1))

        # First TD_frac of dimensions of the time embeddings will be used.
        self.TD_frac = 0.5

        layer_maps = []
        layer_lns  = []
        layer_lncat2s = []
        self.TDs = []

        for i in range(dim1):
            i2 = self.emb_idx2layer_idx[i]
            TD = int(self.TD_frac * infeat_dims[i2])
            self.TDs.append(TD)

            # infeat_dims[i2] + TD because we also include time embeddings (first TD dims) as the input features.
            layer_maps.append( nn.Linear(infeat_dims[i2] + TD, r, bias=True) )
            layer_lns.append( nn.LayerNorm(dim2, elementwise_affine=True) )
            layer_lncat2s.append(LNCat2(infeat_dims[i2], TD))

        self.layer_maps    = nn.ModuleList(layer_maps)
        self.layer_lns     = nn.ModuleList(layer_lns)
        self.layer_lncat2s = nn.ModuleList(layer_lncat2s)

        self.has_bias = has_bias
        if has_bias:
            # bias: 25 * 768.
            self.bias        = nn.Parameter(torch.zeros(dim1, dim2))
        else:
            self.bias        = 0

        print(f"AdaEmbedding initialized with {self.N} init vectors, {self.r} basis vectors")
        self.call_count = 0

    # layer_infeat: 4D image feature tensor [B, C, H, W].
    # layer_idx: 0 ~ 24. emb_idx: 0 ~ 15.
    # time_emb: [B, 1280].
    def forward(self, layer_idx, layer_infeat, time_emb, img_mask=None):
        emb_idx = self.layer_idx2emb_idx[layer_idx]
        self.avgpool = MaskedAvgPool2d()
        
        with torch.autocast(device_type=self.device_type, enabled=True):
            # basis_dyn_weight: [B, r] = [2, 12].
            # We do not BP into the UNet. So cut off the gradient flow here to reduce RAM and compute.
            # infeat_pooled: [B, C_layer]
            infeat_pooled    = self.avgpool(layer_infeat, img_mask).detach()
            # time_emb has a fixed dimension of 1280. But infeat has variable dimensions.
            # Only use the first TD dimensions of the time embedding, 
            # as the time embedding is highly redundant, and the first TD dimensions are sufficient
            # to capture the temporal information.
            # Note to take the first TD dimensions, instead of the last TD dimensions,
            # as the leading dimensions are most sensitive to time change, 
            # and the last dimensions tend to be the same for all time steps.
            # TD is C_layer/2, so that the time embeddings won't dominate the image features infeat_pooled.
            TD = self.TDs[emb_idx]
            # cat(ln(infeat_pooled), ln(time_emb)) as the input features.
            infeat_time      = self.layer_lncat2s[emb_idx](infeat_pooled, time_emb[:, :TD])
            basis_dyn_weight = self.layer_maps[emb_idx](infeat_time)
            # bias: [1, 768]
            bias = self.bias[emb_idx].unsqueeze(0)

            if self.N > 0:
                basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=0)
            else:
                basis_vecs = self.basis_vecs

            ln = self.layer_lns[emb_idx]
            # [2, 12] x [12, 768] = [2, 768], + [1, 768] = [2, 768].
            # dim2: 768.
            out_vec0 = ln(torch.matmul(basis_dyn_weight, basis_vecs)) / np.sqrt(self.dim2)
            out_vec  = out_vec0 + bias

            self.debug = False
            if 'call_count' not in self.__dict__:
                self.call_count = 0

            if self.debug and self.call_count % 10 == 0:
                calc_stats(f'{emb_idx} time_emb', time_emb[:, :TD])
                calc_stats(f'{emb_idx} infeat_pooled', infeat_pooled)
                calc_stats(f'{emb_idx} basis_dyn_weight', basis_dyn_weight)
                calc_stats(f'{emb_idx} out_vec0', out_vec0)
                calc_stats(f'{emb_idx} bias', bias)

            if emb_idx == 24:
                self.call_count += 1

        return out_vec

# Make it compatible with older checkpoints.
LASREmbedding = AdaEmbedding

# embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
# = LatentDiffusion.cond_stage_model
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            initializer_weights=None,
            initializer_neg_words=None,
            placeholder_suffix=None,
            cls_delta_token="person",
            num_vectors_per_token=1,
            progressive_words=False,
            use_layerwise_embedding=False,
            num_unet_layers=16,
            # If two tokens, lora rank=4. That means,
            # compress 16 embeddings to the linear combination of 4 embeddings,
            # in which two are initialized as the two token embeddings, and two are learned through BP.
            layerwise_lora_rank_token_ratio=3,
            # If no initializer words are specified, then lora rank=2.
            layerwise_lora_default_rank=2,
            layer_idx2emb_idx = { 1:  0, 2:  1, 4:  2,  5:  3,  7:  4,  8:  5,  12: 6,  16: 7,
                                  17: 8, 18: 9, 19: 10, 20: 11, 21: 12, 22: 13, 23: 14, 24: 15 },    
            ada_emb_weight=0.5, 
            composition_delta_reg_iter_gap=-1,       
            subj_scale=1.0,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()
        self.string_to_ada_embedder_dict = nn.ModuleDict()
        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0
        self.subj_scale = subj_scale
        self.set_ada_emb_weight(ada_emb_weight, init=True)
        self.composition_delta_reg_iter_gap = composition_delta_reg_iter_gap

        self.use_layerwise_embedding = use_layerwise_embedding
        self.layerwise_lora_rank_token_ratio = layerwise_lora_rank_token_ratio
        self.num_unet_layers    = num_unet_layers
        # change max_vectors_per_token to max_vectors_per_layer_per_token
        # When the passed argument num_vectors_per_token > 1, it means multi-token embedding, 
        # instead of layer-wise embedding.
        self.max_vectors_per_layer_per_token = num_vectors_per_token
        # multi-token and multi-layer embedding are not supported at the same time.
        if self.use_layerwise_embedding:
            assert num_vectors_per_token == 1, \
                "multiple embeddings per token is not supported when using layer-wise embeddings"
            # num_vectors_per_token specifies the total number of embeddings for this token.
            # There's an embedding for each layer.
            num_vectors_per_token = num_unet_layers

        # hasattr(embedder, 'tokenizer') -> True
        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_tokens_for_string = partial(get_clip_tokens_for_string, embedder.tokenizer)
            get_embeddings_for_tokens = partial(get_embeddings_for_clip_tokens, embedder.transformer.text_model.embeddings)
            self.token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_tokens_for_string = partial(get_bert_tokens_for_string, embedder.tknz_fn)
            get_embeddings_for_tokens = embedder.transformer.token_emb
            self.token_dim = 1280

        # Save this function to be used in load() when doing placeholder substitution.
        self.get_tokens_for_string = get_tokens_for_string

        if initializer_neg_words is not None and len(initializer_neg_words) > 0:
            init_neg_embeddings = []
            for neg_words in initializer_neg_words:
                if neg_words is None or len(neg_words) == 0:
                    continue
                neg_token_ids = get_tokens_for_string(neg_words)
                with torch.no_grad():
                    init_neg_embeddings.append(get_embeddings_for_tokens(neg_token_ids.cpu()))
            if len(init_neg_embeddings) > 0:
                init_neg_embeddings = torch.cat(init_neg_embeddings, dim=0)
                NEG = init_neg_embeddings.shape[0]
            else:
                init_neg_embeddings = None
                NEG = 0
        else:
            init_neg_embeddings = None
            NEG = 0

        for idx, placeholder_string in enumerate(placeholder_strings):
            # get_token_for_string <= get_clip_token_for_string.
            # force_single_token = True, as there should be only one token in placeholder_string.
            tokens = get_tokens_for_string(placeholder_string, force_single_token=True)
            token = tokens[0]

            if initializer_words is not None and idx < len(initializer_words):
                init_word_tokens = get_tokens_for_string(initializer_words[idx])
                N = len(init_word_tokens)
                if initializer_weights is not None:
                    init_word_weights = initializer_weights[idx]
                    init_word_weights = torch.tensor(init_word_weights, dtype=torch.float32)
                    init_word_weights = init_word_weights / init_word_weights.sum()
                else:
                    init_word_weights = torch.ones(N, dtype=torch.float32) / N

                layerwise_lora_rank = round(layerwise_lora_rank_token_ratio * N + NEG + 1) if self.use_layerwise_embedding else -1

                with torch.no_grad():
                    init_word_embeddings = get_embeddings_for_tokens(init_word_tokens.cpu())
                    avg_init_word_embedding = (init_word_embeddings * init_word_weights.unsqueeze(1)).sum(dim=0, keepdim=True)

                if layerwise_lora_rank > 0:
                    token_params        = StaticLayerwiseEmbedding(num_vectors_per_token, self.token_dim, layerwise_lora_rank, (0.1, 0.02), 
                                                                   init_word_embeddings, init_word_weights, 
                                                                   init_neg_vecs=init_neg_embeddings)

                    token_ada_embedder  = AdaEmbedding(num_vectors_per_token, self.token_dim, 
                                                       layerwise_lora_rank, init_word_embeddings)                                                        
                else:
                    # ANCHOR[id=init_embed] : num_vectors_per_token vectors are initialized with the same embedding.
                    token_params = torch.nn.Parameter(avg_init_word_embedding.repeat(num_vectors_per_token, 1), requires_grad=True)
                    token_ada_embedder = None
                # initial_embeddings are only used to compute the regularization loss.
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embeddings, requires_grad=False)
            else:
                if self.use_layerwise_embedding and layerwise_lora_default_rank > 0:
                    token_params        = StaticLayerwiseEmbedding(num_vectors_per_token, self.token_dim, layerwise_lora_default_rank, (0.1, 0.02),
                                                              None, None, init_neg_embeddings=init_neg_embeddings)
                                                  
                    token_ada_embedder  = AdaEmbedding(num_vectors_per_token, self.token_dim, 
                                                        layerwise_lora_default_rank, init_word_embeddings)   
                else:
                    token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, self.token_dim), requires_grad=True))
                    token_ada_embedder = None

            self.string_to_token_dict[placeholder_string] = token
            # token_params: an embedding vector or a StaticLayerwiseEmbedding object (when use_layerwise_embedding).
            # Pytorch >= 1.12.0 allows to put an nn.Module object into an nn.ParameterDict.
            self.string_to_param_dict[placeholder_string] = token_params
            self.string_to_ada_embedder_dict[placeholder_string] = token_ada_embedder

            self.cls_delta_token    = cls_delta_token
            if self.cls_delta_token is not None:
                cls_delta_token_ids = get_tokens_for_string(self.cls_delta_token)
                assert len(cls_delta_token_ids) == 1, f"ERROR: cls_delta_token '{cls_delta_token}' must be a single token."
                
            self.placeholder_suffix = placeholder_suffix
            if self.placeholder_suffix is not None:
                # Usually the placeholder word is "z", 
                # so placeholder_suffix variables are named z_suffix_*.
                z_suffix_ids = get_tokens_for_string(self.placeholder_suffix)
                self.z_suffix_ids      = z_suffix_ids
                self.z_suffix_id_count = len(z_suffix_ids)
            else:
                self.z_suffix_ids      = None
                self.z_suffix_id_count = 0

            self.clear_ada_layer_temp_info()
            self.clear_delta_loss_emb_mask()
            self.img_mask = None
            self.layer_idx2emb_idx = layer_idx2emb_idx
            self.loss_call_count = 0
            # Store the embedder to compute the delta loss.
            self.embedder = embedder
            self.token_repl_mask = None
            print("EmbeddingManager initialized with layerwise_lora_rank={}, ada_emb_weight={}, placeholder_suffix={}".format(
                   layerwise_lora_rank, ada_emb_weight, placeholder_suffix))
            
    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # If self.use_layerwise_embedding, then max_vectors_per_token = num_unet_layers = 16.
    def forward(
            self,
            tokenized_text,         # [B, N]. 
            embedded_text,          # [B, N, 768]. 
    ):
        # When delta loss is used, b is not batch_size, but batch_size * 4 * num_compositions_per_image.
        # If bs=2, num_compositions_per_image=2, then b=16.
        # In the iterations when ada delta loss is enabled, in effect num_compositions_per_image is 1, 
        # even if it's specified as 2, so b=8.
        b, n, device = *tokenized_text.shape, tokenized_text.device

        # gen_ada_embedding is dynamically switched on/off by  set_ada_layer_temp_info()/clear_ada_layer_temp_info().
        # No need to calculate delta_loss_emb_mask here, as the mask for ada embeddings is 
        # the same as for the static embeddings. 
        # AdaPrompt combines static and ada embeddings. So the static embedding replacement 
        # code below is always called, and delta_loss_emb_mask is always calculated.
        if self.gen_ada_embedding:
            # self.layer_idx, self.layer_infeat, self.time_emb were cached by 
            # a previous call of  set_ada_layer_temp_info() from UNet.
            embedded_text = \
                self.get_ada_embedding(self.layer_idx, self.layer_infeat, self.time_emb,
                                       tokenized_text, embedded_text)
            emb_idx = self.layer_idx2emb_idx[self.layer_idx]
            # Remove ada-specific intermediate variables.
            self.clear_ada_layer_temp_info()
            return embedded_text

        if self.use_layerwise_embedding:
            # embedded_text: [B, 16, N, 768] => [16*B, N, 768].
            # "Tuck" the layer dimension into the batch dimension, 
            # to keep embedded_text in 3D, same as the input.
            embedded_text = embedded_text.unsqueeze(1).repeat(1, self.num_unet_layers, 1, 1).view(b * self.num_unet_layers, n, -1)
            # tokenized_text: [B, 16, N] => [16*B, N]
            # tokenized_text has to be repeated along the layer dimension as well, so that 
            # placeholder_indices can index the embedding at each layer in the batch.
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1, self.num_unet_layers, 1).view(b * self.num_unet_layers, n)
            # mirror-reflect the embedding along the layer dimension, to make it symmetric 
            # in the encoder & decoder.

        self.token_repl_mask = torch.zeros(embedded_text.shape[0], n, 1, device=device)

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)
            if isinstance(placeholder_embedding, StaticLayerwiseEmbedding):
                # Generate the actual placeholder_embedding on the fly.
                # The 16 Static LoRA embeddings are formed by linearly combining the basis vectors.
                # The matrix operations are done on the fly.
                placeholder_embedding = placeholder_embedding()

            # max_vectors_per_layer_per_token == 1: original num_vectors_per_token == 1, but 
            # self.use_layerwise_embedding could still be True.
            if self.max_vectors_per_layer_per_token == 1: # If there's only one vector per token, we can do a simple replacement
                placeholder_indices = torch.where(tokenized_text == placeholder_token.to(device))
                # No placeholder token is found in the current batch.
                if placeholder_indices[0].numel() == 0:
                    continue

                # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
                # Non-layerwise: embedded_text[placeholder_indices]: [2, 768].  placeholder_embedding: [1, 768].
                # layerwise: placeholder_indices =  
                # (tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 
                #  tensor([ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]))
                # embedded_text[placeholder_indices]: [32, 768]. placeholder_embedding: [16, 768].
                # The first 16 elements (0-8) in embedded_text[placeholder_indices] correspond to the 16 layers of the 
                # first instance in the batch.
                # 16 layers of placeholder_embedding are repeated b times.
                # placeholder_embedding: placeholder_embedding: [16, 768] repeat=> [32, 768]
                # Note that the 16 layers are initialized with the same embedding. 
                # LINK #init_embed
                # Possible BUG: if the placeholder appears in > 1 times in one prompt, then the 
                # filling order may be wrong. Check in the future.
                if self.use_layerwise_embedding:
                    # OCCUR: the actual number of occurrences of the placeholder in the current batch,
                    # not repetitively counting the occurrences in the embedded_text repeated for M layers.
                    OCCUR = placeholder_indices[0].numel() // self.num_unet_layers
                else:
                    OCCUR = placeholder_indices[0].numel()
                
                embedded_text[placeholder_indices] = placeholder_embedding.repeat(OCCUR, 1) * self.subj_scale
                # Mark where the placeholder token is replaced by the embedding.
                self.token_repl_mask[placeholder_indices] = 1

                delta_loss_emb_mask  = torch.ones(b, 1, n, 1, device=device)
                # OCCUR is the real number of occurrences of placeholder. OCCUR <= b.
                # The batch size b is usually small, so this loop is not a bottleneck.
                for i in range(OCCUR):
                    elem_idx  = placeholder_indices[0][i]
                    start_idx = placeholder_indices[1][i] + 1
                    assert tokenized_text[elem_idx][start_idx-1] == placeholder_token
                    has_suffix = True
                    for j in range(self.z_suffix_id_count):
                        if tokenized_text[elem_idx][start_idx+j] != self.z_suffix_ids[j]:
                            has_suffix = False
                            break

                    if has_suffix:
                        end_idx   = placeholder_indices[1][i] + 1 + self.z_suffix_id_count
                        # Simply mask z_suffix_id_count tokens after the placeholder token.
                        # In effect, this masks the placeholder suffix following the placeholder token.
                        delta_loss_emb_mask[elem_idx][0][start_idx:end_idx] = 0

                self.set_delta_loss_emb_mask(delta_loss_emb_mask)
            # *multi-vector latent space*: In this space, S* is embedded into multiple 
            # learned embeddings, an approach that is equivalent to describing
            # the concept through multiple learned pseudo-words. 
            # This setting is experimented in the TI paper, 
            # but AdaPrompt or Static Layerwise Embedding doesn't use it. 
            # So we don't consider this option, and just leave the original code as is.
            else: 
                # otherwise, need to insert and keep track of changing indices
                # *progressive_words*: Begin training with a single embedding vector, introduce a second vector 
                # following 2,000 training steps, and a third vector after 4, 000 steps. 
                # In this scenario, we expect the network to focus on the core details first, 
                # and then leverage the additional pseudo-words to capture finer details.
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_layer_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                # nelement() == numel()
                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col],  
                                               placeholder_embedding[:num_vectors_for_token] * self.subj_scale, 
                                               embedded_text[row][col + 1:]], axis=0)[:n]

                    # Locate the next placeholder token in the row, and replace the embedding in embedded_text.
                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # As the output embedding is only generated for a particular layer, 
    # no need to repeat num_unet_layers times as replacing with the static embedding in forward().
    def get_ada_embedding(
            self,
            layer_idx,              # the index of the current layer in the UNet.
            layer_infeat,           # layer_infeat: intermediate features of the UNet on the noise image.
            time_emb,               # time embedding of the current iteration.
            tokenized_text,         # [B, N]. Identical B copies along the batch dimension.
            embedded_text,          # [B, N, 768]. Identical B copies along the batch dimension.
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        # max_vectors_per_layer_per_token == 1: original num_vectors_per_token == 1, but 
        # self.use_layerwise_embedding could still be True.
        if self.max_vectors_per_layer_per_token > 1: 
            # *multi-vector latent space*: In this space, S* is embedded into multiple 
            # learned embeddings, an approach that is equivalent to describing
            # the concept through multiple learned pseudo-words.
            raise NotImplementedError("multi-vector latent space not implemented yet.")

        if not self.use_layerwise_embedding:
            raise NotImplementedError("non-layerwise embedding not supported in get_ada_embedding().")

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedder = self.string_to_ada_embedder_dict[placeholder_string].to(device)
            assert isinstance(placeholder_embedder, AdaEmbedding)

            # There's only one vector per token, we can do a simple replacement
            # embedded_text: [B, N, 768].
            # tokenized_text: [B, N].
            placeholder_indices = torch.where(tokenized_text == placeholder_token.to(device))
            # Skip generating the ada embedding if there's no placeholder token in the batch.
            if placeholder_indices[0].numel() == 0:
                continue

            # Generate the actual placeholder_embedding on the fly.
            # [B=2, 768]
            placeholder_embedding = placeholder_embedder(layer_idx, layer_infeat, time_emb, self.img_mask)
            # embedded_text[placeholder_indices] indexes the embedding at each instance in the batch.
            # embedded_text[placeholder_indices]: [2, 768].  placeholder_embedding: [2, 768].
            # Sometimes (e.g. during inference, some instances contain the placeholder token but
            # others don't. tokenized_text has a batch size of those containing the placeholder token only.
            # But layer_infeat is still of the full batch size. So placeholder_embedding has a batch size 
            # larger than the number of instances containing the placeholder token. We need to index
            # placeholder_embedding with placeholder_indices[0] to get the matching new placeholder_embedding.
            embedded_text[placeholder_indices] = placeholder_embedding[placeholder_indices[0]] * self.subj_scale

        return embedded_text

    def get_ada_emb_weight(self):
        if self.training:
            # 0.5 -> uniform in [0.4, 0.7]
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

    def set_ada_layer_temp_info(self, layer_idx, layer_infeat, time_emb):
        self.gen_ada_embedding = True
        self.layer_idx      = layer_idx
        self.layer_infeat   = layer_infeat
        self.time_emb       = time_emb
        # Initialize the ada_embeddings cache list.

    # ada_embeddings is used to cache the embeddings of all layers, 
    # for computing the composition delta loss.
    def init_ada_embedding_cache(self):
        self.ada_embeddings = [ None for i in range(self.num_unet_layers) ]

    def cache_ada_embedding(self, i, embedding):
        emb_idx = self.layer_idx2emb_idx[i]
        self.ada_embeddings[emb_idx] = embedding

    # Clear layer-specific intermediate variables. Also clear gen_ada_embedding,
    # which will be enabled again through set_ada_layer_temp_info() in ddpm.py.
    def clear_ada_layer_temp_info(self):
        self.gen_ada_embedding = False
        self.layer_idx      = -1
        self.layer_infeat   = None
        self.time_emb       = None
        
    def clear_ada_embedding_cache(self):
        self.ada_embeddings = None

    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # DDPM.validation_step() -> LatentDiffusion.shared_step() -> .forward()
    # -> .get_learned_conditioning() -> .cond_stage_model.encode()
    # -> EmbeddingManager.forward() -> here.
    # Occasionally, image_logger is called, which calls LatentDiffusion.log_images ->
    # .get_learned_conditioning() -> ... -> here.
    # Such delta_loss_emb_mask won't be used in composition_delta_loss() and won't be cleared.
    # delta_loss_emb_mask: [B, N, 768], where N is the padded prompt length.
    def set_delta_loss_emb_mask(self, delta_loss_emb_mask):
        if self.z_suffix_id_count > 0 and delta_loss_emb_mask is not None:
            self.delta_loss_emb_mask = delta_loss_emb_mask
        # Otherwise, z_suffix_id_count == 0, so we don't need to use it to mask a region 
        # when computing the compositional delta loss.

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
                     "ada_emb_weight":          self.ada_emb_weight, }, 
                    ckpt_path)

    # load custom tokens and their learned embeddings from "embeddings_gs-4200.pt".
    def load(self, ckpt_paths):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        self.string_to_token_dict           = {}
        self.string_to_param_dict           = nn.ParameterDict()
        self.string_to_ada_embedder_dict   = nn.ModuleDict()

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
                self.string_to_param_dict[k2]        = ckpt["string_to_param"][k]
                self.string_to_ada_embedder_dict[k2] = ckpt["string_to_ada_embedder"][k]
                print(f"Loaded {k}->{k2} from {ckpt_path}")

            # If multiple checkpoints have different ada_emb_weight, the last one will be used.
            if "ada_emb_weight" in ckpt:
                self.set_ada_emb_weight(ckpt["ada_emb_weight"])

    # get_embedding_norms_squared() is never used.
    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def embedding_parameters(self):
        return list(self.string_to_param_dict.parameters()) \
               + list(self.string_to_ada_embedder_dict.parameters())

    def embedding_attractor_loss(self):
        loss = 0.
        num_placeholders = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone()
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
                reg_center = self.initial_embeddings[key].clone()
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
        num_placeholders  = len(self.initial_embeddings)
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

        # Dynamically adjust the regularization weights. The larger the norm, the larger the weight.
        # T: temperature. Larger T => when norm grows, the penalty is more severe.
        T = 1.5

        for key in self.initial_embeddings:
            for embobj in (self.string_to_param_dict[key], self.string_to_ada_embedder_dict[key]):
                # Skip non-layerwise embeddings.
                if not isinstance(embobj, StaticLayerwiseEmbedding) \
                  and not isinstance(embobj, AdaEmbedding):
                    continue

                init_vecs = self.initial_embeddings[key].clone()
                # bias, pre_vecs, basis_vecs are structures existing in 
                # both StaticLayerwiseEmbedding and AdaEmbedding.
                # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
                # pre_vecs are updated through BP. Therefore, loss_pre_vecs prevents pre_vecs 
                # from drifting too far from init_vecs.
                if embobj.has_bias:
                    loss_bias        = selective_reg_loss(embobj.bias, loss_type=euc_loss_type)
                    bias_reg_weight  = bias_reg_weight_base  * torch.norm(embobj.bias, dim=1).mean().item() ** T
                else:
                    loss_bias        = 0.
                    bias_reg_weight  = 0.

                loss_basis       = selective_reg_loss(embobj.basis_vecs, loss_type=euc_loss_type)
                basis_reg_weight = basis_reg_weight_base * torch.norm(embobj.basis_vecs, dim=1).mean().item() ** T
                if embobj.N > 0:
                    loss_pre_vecs = selective_reg_loss(embobj.pre_vecs - init_vecs, loss_type=euc_loss_type)
                else:
                    loss_pre_vecs = 0.

                loss_ada_maps_weight = 0.
                loss_ada_maps_bias   = 0.
                if isinstance(embobj, AdaEmbedding):
                    for i, map in enumerate(embobj.layer_maps):
                        loss_ada_maps_weight += selective_reg_loss(map.weight, loss_type=euc_loss_type)
                        loss_ada_maps_bias   += selective_reg_loss(map.bias,   loss_type=euc_loss_type)

                if type(loss_bias) == int:
                    breakpoint()

                curr_loss = loss_bias               * bias_reg_weight \
                            + loss_basis            * basis_reg_weight \
                            + loss_pre_vecs         * pre_vecs_reg_weight \
                            + loss_ada_maps_weight  * ada_maps_weight_reg_weight \
                            + loss_ada_maps_bias    * ada_maps_bias_reg_weight 

                debug = True
                if debug and self.loss_call_count % 100 == 0:
                    print_str = f'loss_bias={loss_bias.item():.4f}, ' \
                                f'loss_basis={loss_basis.item():.4f}, ' \
                                f'loss_pre_vecs={loss_pre_vecs.item():.4f}, '
                    if isinstance(embobj, AdaEmbedding):
                        print_str += f'loss_ada_maps_weight={loss_ada_maps_weight.item():.4f}, ' \
                                     f'loss_ada_maps_bias={loss_ada_maps_bias.item():.4f}'

                    print(print_str)

                if euc_loss_type   == 'l2':
                    if isinstance(embobj, AdaEmbedding):
                        loss_boost = ada_l2_loss_boost
                    else:
                        loss_boost = static_l2_loss_boost
                else:
                    loss_boost = 1.
                loss = loss + curr_loss * loss_boost

        return loss / num_placeholders

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
    # embeddings of subj_prompt_single, subj_prompt_comp, cls_prompt_single, cls_prompt_comp. 
    # cls_prompt_*: embeddings generated from prompts containing a class token (as opposed to the subject token).
    def composition_delta_loss(self, do_ada_comp_delta_reg, static_embeddings):
        # The composition delta loss for ada embeddings is only applied 
        # every composition_delta_reg_iter_gap iterations. So the ada loss 
        # should be boosted proportionally to composition_delta_reg_iter_gap. 
        # Divide it by 2 to reduce the proportion of ada emb loss relative to 
        # static emb loss in the total loss.
        ada_comp_loss_boost_ratio = self.composition_delta_reg_iter_gap / 2
        # If do_ada_comp_delta_reg,     BS = 2.
        # If not do_ada_comp_delta_reg, BS = 2 * num_compositions_per_image = 4.
        BS = static_embeddings.shape[0] // (4 * self.num_unet_layers)
        max_token_num = static_embeddings.shape[1]
        # static_embeddings: [8, 16, 77, 768]
        static_embeddings = static_embeddings.view(BS * 4, -1, max_token_num, static_embeddings.shape[-1])
        # Each is [2, 16, 77, 768]
        subj_prompt_single, subj_prompt_comp, cls_prompt_single, cls_prompt_comp = \
                    static_embeddings.split(BS, dim=0)

        # cls_delta: [2, 16, 77, 768]. Should be a repeat of a tensor [2, 1, 77, 768] 
        # by 16 times along dim=1, as cls_prompt_* doesn't contain placeholder_token.
        cls_delta = cls_prompt_comp - cls_prompt_single
        # static_delta: [2, 16, 77, 768]. Different values for each layer along dim=1.
        static_delta = subj_prompt_comp - subj_prompt_single
        # delta_loss_emb_mask is often obtained from an extended batch. So only uses the first BS elements.
        delta_loss_emb_mask = self.delta_loss_emb_mask[:BS] if self.delta_loss_emb_mask is not None else None
        static_delta_loss   = calc_delta_loss(static_delta, cls_delta, delta_loss_emb_mask)

        if do_ada_comp_delta_reg:
            # Each emb is of [4, 77, 768]. 4 = 2 * batch_size.
            for i, emb in enumerate(self.ada_embeddings):
                # ada embeddings of all layers should have been stored in self.ada_embeddings
                # before calling composition_delta_loss().
                if emb is None:
                    breakpoint()

            # ada_embeddings: [4, 16, 77, 768]
            ada_embeddings = torch.stack(self.ada_embeddings, dim=1)
            ada_subj_emb_single, ada_subj_emb_comp = ada_embeddings.split(BS, dim=0)
            ada_delta = ada_subj_emb_comp - ada_subj_emb_single
            ada_delta_loss = calc_delta_loss(ada_delta, cls_delta, delta_loss_emb_mask)
            # The cached ada embeddings are useless now, release them.
            self.clear_ada_embedding_cache()
        else:
            ada_delta_loss = 0
        
        self.clear_delta_loss_emb_mask()
        delta_loss = static_delta_loss + ada_delta_loss * ada_comp_loss_boost_ratio
        return delta_loss
    