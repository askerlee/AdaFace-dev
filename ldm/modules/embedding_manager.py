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
    # 2: one begin of text, one token.
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

# Eq.(2) in the StyleGAN-NADA paper.
# delta, ref_delta: [2, 16, 77, 768].
def calc_delta_loss(delta, ref_delta, exponent=3):
    # Flatten delta and ref_delta.
    # dela: [2464, 768], ref_delta: [2464, 768]
    delta = delta.view(delta.numel() // delta.shape[-1], -1)
    ref_delta = ref_delta.view(ref_delta.numel() // ref_delta.shape[-1], -1)
    # delta_pow = delta * delta.abs().pow(exponent - 1)
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
            self.bias_scales = nn.Parameter(torch.ones(dim1, 1))
        else:
            self.bias = 0
            self.bias_scales = 0

        lns  = []
        for i in range(dim1):
            lns.append( nn.LayerNorm(dim2, elementwise_affine=True) )

        self.lns  = nn.ModuleList(lns)

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
            out_vecs_ln = [ self.lns[i](out_vecs[i]) for i in range(self.dim1) ]
            out_vecs_ln = torch.stack(out_vecs_ln, dim=0) / np.sqrt(self.dim2)

            # Different layers have different bias scales.
            # Separate bias and bias_scales, for easier regularization on their scales.
            layerwise_bias  = self.bias * self.bias_scales
            out_vecs_ln = out_vecs_ln + layerwise_bias
            return out_vecs_ln

class LASREmbedding(nn.Module):
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
                    f"LoRA init vectors shape {init_vecs.shape} must be (<={dim1}, {dim2})"
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # First TD_frac of dimensions of the time embeddings will be used.
        self.TD_frac = 0.5

        maps = []
        lns  = []
        self.TDs = []

        for i in range(dim1):
            i2 = self.emb_idx2layer_idx[i]
            TD = int(self.TD_frac * infeat_dims[i2])
            self.TDs.append(TD)

            # infeat_dims[i2] + TD because we also include time embeddings (first TD dims) as the input features.
            maps.append( nn.Linear(infeat_dims[i2] + TD, r, bias=True) )
            lns.append( nn.LayerNorm(dim2, elementwise_affine=True) )

        self.maps = nn.ModuleList(maps)
        self.lns  = nn.ModuleList(lns)

        self.has_bias = has_bias
        if has_bias:
            # bias: 25 * 768.
            self.bias        = nn.Parameter(torch.zeros(dim1, dim2))
            self.bias_scales = nn.Parameter(torch.ones(dim1, 1))
        else:
            self.bias = 0
            self.bias_scales = 0

        print(f"LASREmbedding initialized with {self.N} init vectors, {self.r} basis vectors")
        self.call_count = 0

    # layer_infeat: 4D image feature tensor [B, C, H, W].
    # layer_idx: 0 ~ 24. emb_idx: 0 ~ 15.
    # time_emb: [B, 1280].
    def forward(self, layer_idx, layer_infeat, time_emb):
        emb_idx = self.layer_idx2emb_idx[layer_idx]

        with torch.autocast(device_type=self.device_type, enabled=True):
            # basis_dyn_weight: [B, r] = [2, 12].
            # We do not BP into the UNet. So cut off the gradient flow here.
            # infeat_pooled: [B, D_layer]
            infeat_pooled    = self.avgpool(layer_infeat).squeeze(-1).squeeze(-1).detach()
            # time_emb has a fixed dimension of 1280. But infeat has variable dimensions.
            # Only use the first D dimensions of the time embedding, 
            # as the time embedding is highly redundant, and the first D dimensions are sufficient
            # to capture the temporal information.
            # Note to take the first D dimensions, instead of the last D dimensions,
            # as the leading dimensions are sensitive to time change, 
            # and the last dimensions tend to be the same for all time steps.
            # Never allow time embeddings dominate image features infeat.
            TD = self.TDs[emb_idx]
            infeat_time      = torch.cat([infeat_pooled, time_emb[:, :TD]], dim=1)
            basis_dyn_weight = self.maps[emb_idx](infeat_time)
            # Separate bias and bias_scales, for easier regularization on their scales.
            # bias: [1, 768] * [1, 1] = [1, 768].
            bias = self.bias[emb_idx].unsqueeze(0) * self.bias_scales[emb_idx]

            if self.N > 0:
                basis_vecs = torch.cat([self.pre_vecs, self.basis_vecs], dim=0)
            else:
                basis_vecs = self.basis_vecs

            ln = self.lns[emb_idx]
            # [2, 12] x [12, 768] = [2, 768], + [1, 768] = [2, 768].
            out_vec0 = ln(torch.matmul(basis_dyn_weight, basis_vecs)) / np.sqrt(self.dim2)
            out_vec  = out_vec0 + bias

            self.debug = False
            if 'call_count' not in self.__dict__:
                self.call_count = 0

            if self.debug and self.call_count % 10 == 0:
                calc_stats(f'{emb_idx} basis_dyn_weight', basis_dyn_weight)
                calc_stats(f'{emb_idx} out_vec0', out_vec0)
                calc_stats(f'{emb_idx} bias', bias)
            
            if emb_idx == 24:
                self.call_count += 1

        return out_vec

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
            per_image_tokens=False,
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
            lasr_emb_weight=0.5, 
            composition_delta_reg_iter_gap=-1,       
            subj_scale=1.0,
            **kwargs
    ):
        super().__init__()
        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()
        self.string_to_lasr_embedder_dict = nn.ModuleDict()
        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0
        self.subj_scale = subj_scale
        self.lasr_emb_weight = lasr_emb_weight
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

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        if initializer_neg_words is not None and len(initializer_neg_words) > 0:
            init_neg_embeddings = []
            for neg_words in initializer_neg_words:
                if neg_words is None or len(neg_words) == 0:
                    continue
                neg_tokens = get_tokens_for_string(neg_words)
                with torch.no_grad():
                    init_neg_embeddings.append(get_embeddings_for_tokens(neg_tokens.cpu()))
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

                    token_lasr_embedder  = LASREmbedding(num_vectors_per_token, self.token_dim, 
                                                         layerwise_lora_rank, init_word_embeddings)                                                        
                else:
                    # ANCHOR[id=init_embed] : num_vectors_per_token vectors are initialized with the same embedding.
                    token_params = torch.nn.Parameter(avg_init_word_embedding.repeat(num_vectors_per_token, 1), requires_grad=True)
                    token_lasr_embedder = None
                # initial_embeddings are only used to compute the regularization loss.
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embeddings, requires_grad=False)
            else:
                if self.use_layerwise_embedding and layerwise_lora_default_rank > 0:
                    token_params        = StaticLayerwiseEmbedding(num_vectors_per_token, self.token_dim, layerwise_lora_default_rank, (0.1, 0.02),
                                                              None, None, init_neg_embeddings=init_neg_embeddings)
                                                  
                    token_lasr_embedder  = LASREmbedding(num_vectors_per_token, self.token_dim, 
                                                        layerwise_lora_default_rank, init_word_embeddings)   
                else:
                    token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, self.token_dim), requires_grad=True))
                    token_lasr_embedder = None

            self.string_to_token_dict[placeholder_string] = token
            # token_params: an embedding vector or a StaticLayerwiseEmbedding object (when use_layerwise_embedding).
            # Pytorch >= 1.12.0 allows to put an nn.Module object into an nn.ParameterDict.
            self.string_to_param_dict[placeholder_string] = token_params
            self.string_to_lasr_embedder_dict[placeholder_string] = token_lasr_embedder

            self.clear_lasr_layer_info()
            self.layer_idx2emb_idx = layer_idx2emb_idx
            self.loss_call_count = 0
            # Store the embedder to compute the delta loss.
            self.embedder = embedder

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # If self.use_layerwise_embedding, then max_vectors_per_token = num_unet_layers = 16.
    def forward(
            self,
            tokenized_text,         # [B, N]. Identical B copies along the batch dimension.
            embedded_text,          # [B, N, 768]. Identical B copies along the batch dimension.
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device

        # gen_lasr_embedding is dynamically switched on/off by set_lasr_layer_info()/clear_lasr_layer_info().
        if self.gen_lasr_embedding:
            # self.layer_idx, self.layer_infeat, self.time_emb were cached by 
            # a previous call of set_lasr_layer_info() from UNet.
            embedded_text = self.get_lasr_embedding(self.layer_idx, self.layer_infeat, self.time_emb,
                                                    tokenized_text, embedded_text)
            emb_idx = self.layer_idx2emb_idx[self.layer_idx]
            # Cache the computed lasr embedding of the current layer.
            # Before this call, init_lasr_embedding_cache() should have been called somewhere.
            self.lasr_embeddings[emb_idx] = embedded_text
            # Remove lasr-specific intermediate variables.
            self.clear_lasr_layer_info()
            return embedded_text

        if self.use_layerwise_embedding:
            # embedded_text: [B, 16, N, 768] => [16*B, N, 768].
            # "Tuck" the layer dimension into the batch dimension, 
            # to keep embedded_text in 3D, same as the input.
            embedded_text = embedded_text.unsqueeze(1).repeat(1, self.num_unet_layers, 1, 1).view(b * self.num_unet_layers, n, -1)
            # tokenized_text: [B, 16, N] => [16*B, N]
            # tokenized_text has to be repeated along the layer dimension as well, so that 
            # placeholder_idx can index the embedding at each layer in the batch.
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1, self.num_unet_layers, 1).view(b * self.num_unet_layers, n)
            # mirror-reflect the embedding along the layer dimension, to make it symmetric 
            # in the encoder & decoder.

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
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                if placeholder_idx[0].numel() == 0:
                    continue

                # embedded_text[placeholder_idx] indexes the embedding at each instance in the batch.
                # Non-layerwise: embedded_text[placeholder_idx]: [2, 768].  placeholder_embedding: [1, 768].
                # layerwise: placeholder_idx =  
                # (tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 
                #  tensor([ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]))
                # embedded_text[placeholder_idx]: [32, 768]. placeholder_embedding: [16, 768].
                # The first 16 elements (0-8) in embedded_text[placeholder_idx] correspond to the 16 layers of the 
                # first instance in the batch.
                # 16 layers of placeholder_embedding are repeated b times.
                # placeholder_embedding: placeholder_embedding: [16, 768] repeat=> [32, 768]
                # Note that the 16 layers are initialized with the same embedding. 
                # LINK #init_embed
                # Possible BUG: if the placeholder appears in > 1 times in one prompt, then the 
                # filling order may be wrong. Check in the future.
                if self.use_layerwise_embedding:
                    OCCUR = placeholder_idx[0].numel() // self.num_unet_layers
                else:
                    OCCUR = placeholder_idx[0].numel()
                embedded_text[placeholder_idx] = placeholder_embedding.repeat(OCCUR, 1) * self.subj_scale

            # *multi-vector latent space*: In this space, S* is embedded into multiple 
            # learned embeddings, an approach that is equivalent to describing
            # the concept through multiple learned pseudo-words.
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
    def get_lasr_embedding(
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
            raise NotImplementedError("non-layerwise embedding not supported in get_lasr_embedding().")

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedder = self.string_to_lasr_embedder_dict[placeholder_string].to(device)
            assert isinstance(placeholder_embedder, LASREmbedding)

            # There's only one vector per token, we can do a simple replacement
            # embedded_text: [B, N, 768].
            # tokenized_text: [B, N].
            placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
            # Skip generating the lasr embedding if there's no placeholder token in the batch.
            if placeholder_idx[0].numel() == 0:
                continue

            # Generate the actual placeholder_embedding on the fly.
            # [B=2, 768]
            placeholder_embedding = placeholder_embedder(layer_idx, layer_infeat, time_emb)
            # embedded_text[placeholder_idx] indexes the embedding at each instance in the batch.
            # embedded_text[placeholder_idx]: [2, 768].  placeholder_embedding: [2, 768].
            # Sometimes (e.g. during inference, some instances contain the placeholder token but
            # others don't. tokenized_text has a batch size of those containing the placeholder token only.
            # But layer_infeat is still of the full batch size. So placeholder_embedding has a batch size 
            # larger than the number of instances containing the placeholder token. We need to index
            # placeholder_embedding with placeholder_idx[0] to get the matching new placeholder_embedding.
            embedded_text[placeholder_idx] = placeholder_embedding[placeholder_idx[0]] * self.subj_scale

        return embedded_text

    def get_lasr_emb_weight(self):
        return self.lasr_emb_weight
    
    def set_lasr_layer_info(self, layer_idx, layer_infeat, time_emb):
        self.gen_lasr_embedding = True
        self.layer_idx      = layer_idx
        self.layer_infeat   = layer_infeat
        self.time_emb       = time_emb
        # Initialize the lasr_embeddings cache list.

    # lasr_embeddings is used to cache the embeddings of all layers, 
    # for computing the composition delta loss.
    def init_lasr_embedding_cache(self):
        self.lasr_embeddings = [ None for i in range(self.num_unet_layers) ]

    def clear_lasr_layer_info(self):
        self.gen_lasr_embedding = False
        self.layer_idx      = -1
        self.layer_infeat   = None
        self.time_emb       = None
        
    def clear_lasr_embedding_cache(self):
        self.lasr_embeddings = None

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, ckpt_path):
        torch.save({ "string_to_token":         self.string_to_token_dict,
                     "string_to_param":         self.string_to_param_dict,
                     "string_to_lasr_embedder": self.string_to_lasr_embedder_dict,
                     "lasr_emb_weight":         self.lasr_emb_weight, }, 
                    ckpt_path)

    # load custom tokens and their learned embeddings from "embeddings_gs-4200.pt".
    def load(self, ckpt_paths):
        # The default placeholder specified in the config file will be loaded to these dicts.
        # So before loading, remove it from these dicts first.
        self.string_to_token_dict           = {}
        self.string_to_param_dict           = nn.ParameterDict()
        self.string_to_lasr_embedder_dict   = nn.ModuleDict()

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
                self.string_to_lasr_embedder_dict[k2] = ckpt["string_to_lasr_embedder"][k]
                print(f"Loaded {k}->{k2} from {ckpt_path}")

            # If multiple checkpoints have different lasr_emb_weight, the last one will be used.
            if "lasr_emb_weight" in ckpt:
                self.lasr_emb_weight = ckpt["lasr_emb_weight"]

    # get_embedding_norms_squared() is never used.
    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    # Originally returned value is not enclosed in list(), i.e., return a generator.
    # Returned list is list() again. list() the second time won't copy or clone the tensors.
    def embedding_parameters(self):
        return list(self.string_to_param_dict.parameters()) \
               + list(self.string_to_lasr_embedder_dict.parameters())

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

        # each elem in bias_scales is broadcasted to 768 dims. i.e., its effect and gradient is multiplied by 768.
        # So the loss should be divided by 768.
        bias_scales_reg_weight      = 1. / self.token_dim
        bias_reg_weight_base        = 0.1
        basis_reg_weight_base       = 0.1
        lasr_maps_weight_reg_weight = 1.
        lasr_maps_bias_reg_weight   = 0.01
        pre_vecs_reg_weight         = 0.1
        static_l2_loss_boost        = 5
        lasr_static_loss_boost_ratio = 2
        lasr_l2_loss_boost          = static_l2_loss_boost * lasr_static_loss_boost_ratio

        # Dynamically adjust the regularization weights. The larger the norm, the larger the weight.
        # T: temperature. Larger T => when norm grows, the penalty is more severe.
        T = 1.5

        for key in self.initial_embeddings:
            for embobj in (self.string_to_param_dict[key], self.string_to_lasr_embedder_dict[key]):
                # Skip non-LORA embeddings.
                if not isinstance(embobj, StaticLayerwiseEmbedding) \
                  and not isinstance(embobj, LASREmbedding):
                    continue

                init_vecs = self.initial_embeddings[key].clone()
                # bias, pre_vecs, basis_vecs are structures existing in 
                # both StaticLayerwiseEmbedding and LASREmbedding.
                # pre_vecs: basis vectors that are initialized by prespecified init_vecs. 
                # pre_vecs are updated through BP. Therefore, loss_pre_vecs prevents pre_vecs 
                # from drifting too far from init_vecs.
                if embobj.has_bias:
                    # Penalize the bias scales that are larger than 1 with weight 1, 
                    # and those smaller than 1 with weight 0.0001.
                    # If penalizing those < 1 weights too much, they will be slowly driven towards 0.
                    loss_bias_scales_above1 = selective_reg_loss(embobj.bias_scales, loss_type=euc_loss_type, selector=lambda x: x > 1)
                    loss_bias_scales_below1 = selective_reg_loss(embobj.bias_scales, loss_type=euc_loss_type, selector=lambda x: x <= 1)
                    # Penalize those scales bigger than 1 more than those smaller than 1.
                    loss_bias_scales = loss_bias_scales_above1 + loss_bias_scales_below1 * 0.0001
                    loss_bias        = selective_reg_loss(embobj.bias, loss_type=euc_loss_type)
                    bias_reg_weight  = bias_reg_weight_base  * torch.norm(embobj.bias, dim=1).mean().item() ** T
                else:
                    loss_bias_scales = 0.
                    loss_bias        = 0.
                    bias_reg_weight  = 0.

                loss_basis       = selective_reg_loss(embobj.basis_vecs, loss_type=euc_loss_type)
                basis_reg_weight = basis_reg_weight_base * torch.norm(embobj.basis_vecs, dim=1).mean().item() ** T
                if embobj.N > 0:
                    loss_pre_vecs = selective_reg_loss(embobj.pre_vecs - init_vecs, loss_type=euc_loss_type)
                else:
                    loss_pre_vecs = 0.

                loss_lasr_maps_weight = 0.
                loss_lasr_maps_bias   = 0.
                if isinstance(embobj, LASREmbedding):
                    for i, map in enumerate(embobj.maps):
                        loss_lasr_maps_weight += selective_reg_loss(map.weight, loss_type=euc_loss_type)
                        loss_lasr_maps_bias   += selective_reg_loss(map.bias,   loss_type=euc_loss_type)

                if type(loss_bias) == int:
                    breakpoint()

                curr_loss = loss_bias               * bias_reg_weight \
                            + loss_bias_scales      * bias_scales_reg_weight \
                            + loss_basis            * basis_reg_weight \
                            + loss_pre_vecs         * pre_vecs_reg_weight \
                            + loss_lasr_maps_weight * lasr_maps_weight_reg_weight \
                            + loss_lasr_maps_bias   * lasr_maps_bias_reg_weight

                debug = True
                if debug and self.loss_call_count % 100 == 0:
                    print_str = f'loss_bias={loss_bias.item():.4f}, ' \
                                f'loss_bias_scales={loss_bias_scales.item():.4f}, ' \
                                f'loss_basis={loss_basis.item():.4f}, ' \
                                f'loss_pre_vecs={loss_pre_vecs.item():.4f}, '
                    if isinstance(embobj, LASREmbedding):
                        print_str += f'loss_lasr_maps_weight={loss_lasr_maps_weight.item():.4f}, ' \
                                     f'loss_lasr_maps_bias={loss_lasr_maps_bias.item():.4f}'

                    print(print_str)

                if euc_loss_type   == 'l2':
                    if isinstance(embobj, LASREmbedding):
                        loss_boost = lasr_l2_loss_boost
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

    # TODO: support for textual inversion, where static_embeddings is only one embedding.
    # static_embeddings: size: [8*16, 77, 768]. 8 = 4 * batch_size. 16: number of UNet layers.
    # embeddings of subj_prompt_single, subj_prompt_comp, common_prompt_single, common_prompt_comp. 
    # common_prompt_*: embeddings generated from prompts containing a common English name.
    def composition_delta_loss(self, do_lasr_comp_delta_reg, static_embeddings):
        # The composition delta loss for LASR embeddings is only applied 
        # every composition_delta_reg_iter_gap iterations. So boost the loss 
        # by composition_delta_reg_iter_gap times.
        lasr_comp_loss_boost_ratio = self.composition_delta_reg_iter_gap
        BS = static_embeddings.shape[0] // (4 * self.num_unet_layers)
        # static_embeddings: [8, 16, 77, 768]
        static_embeddings = static_embeddings.view(BS * 4, self.num_unet_layers, -1, static_embeddings.shape[-1])
        # Each is [2, 16, 77, 768]
        subj_prompt_single, subj_prompt_comp, common_prompt_single, common_prompt_comp = \
                    static_embeddings.split(BS, dim=0)

        # common_delta: [2, 16, 77, 768]. Should be a repeat of a tensor [2, 1, 77, 768] 
        # by 16 times along dim=1, as common_prompt_* doesn't contain placeholder_token.
        common_delta = common_prompt_comp - common_prompt_single
        # static_delta: [2, 16, 77, 768]. Different values for each layer along dim=1.
        static_delta = subj_prompt_comp - subj_prompt_single
        static_delta_loss   = calc_delta_loss(static_delta, common_delta)

        if do_lasr_comp_delta_reg:
            # Each emb is of [4, 77, 768]. 4 = 2 * batch_size.
            for i, emb in enumerate(self.lasr_embeddings):
                # LASR embeddings of all layers should have been stored in self.lasr_embeddings
                # before calling composition_delta_loss().
                if emb is None:
                    breakpoint()
            # lasr_embeddings: [4, 16, 77, 768]
            lasr_embeddings = torch.stack(self.lasr_embeddings, dim=1)
            lasr_subj_emb_single, lasr_subj_emb_comp = lasr_embeddings.split(BS, dim=0)
            lasr_delta = lasr_subj_emb_comp - lasr_subj_emb_single
            lasr_delta_loss = calc_delta_loss(lasr_delta, common_delta)
            # The cached LASR embeddings are useless now, release them.
            self.clear_lasr_embedding_cache()
        else:
            lasr_delta_loss = 0
        
        # delta_loss is the sum of the delta loss of all layers. Change it to the average.
        delta_loss = (static_delta_loss + lasr_delta_loss * lasr_comp_loss_boost_ratio) / self.num_unet_layers
        return delta_loss
    