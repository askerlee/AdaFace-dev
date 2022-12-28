import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial

DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    assert " " not in string, "Please use a single word for the placeholder string"
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
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]

class LoraEmbedding(nn.Module):
    def __init__(self, dim1, dim2, r=4, init_vec=None):
        super().__init__()

        if r > min(dim1, dim2):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(dim1, dim2)}"
            )
        if init_vec is not None and init_vec.shape != (dim2,):
            raise ValueError(
                f"LoRA init vector shape {init_vec.shape} must be ({dim2},)"
            )

        # lora_up: 25 * r, lora_down: r * 768. lora_up * lora_down: 25 * 768.
        self.lora_up    = nn.Parameter(torch.randn(dim1, r))
        # / sqrt(r) to make the std of (lora_down * lora_up) \approx 1.
        self.lora_down  = nn.Parameter(torch.randn(r, dim2) / np.sqrt(r))
        self.scale = 1.0
        self.r = r

        if init_vec is not None:
            self.bias = nn.Parameter(init_vec.clone(), requires_grad=True)
            '''
            # Each vec in lora_down is init_vec + standard normal noise / 4.
            self.lora_down.data  = self.lora_down.data / 4 + init_vec.unsqueeze(0)
            self.lora_up.data    = self.lora_up.data   / 4 + torch.ones_like(self.lora_up)
            self.lora_up.data   /= r
            '''
        else:
            self.bias = 0
            
    def forward(self):
        with torch.autocast(device_type='cuda', enabled=False):
            # torch.matmul: matrix multiplication.
            # torch.matmul(self.lora_up, self.lora_down): 25 * 768.
            # * self.scale: 25 * 768.
            return torch.matmul(self.lora_up, self.lora_down) * self.scale + self.bias

# embedder: ldm.modules.encoders.modules.FrozenCLIPEmbedder
# = LatentDiffusion.cond_stage_model
class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            use_layerwise_embedding=False,
            layerwise_reflective=False,
            num_unet_enc_layers=12,
            # Compress 12*2+1=25 embeddings to the linear combination of 5 embeddings.
            layerwise_lora_rank=5,          
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.use_layerwise_embedding = use_layerwise_embedding
        self.layerwise_lora_rank    = layerwise_lora_rank if self.use_layerwise_embedding else -1
        self.layerwise_reflective   = layerwise_reflective
        self.num_unet_enc_layers    = num_unet_enc_layers
        self.num_unet_layers        = num_unet_enc_layers * 2 + 1
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
            # if embeddings are symmetric in the encoder & decoder, then
            # it only needs (number of encoder layer) embeddings + 1 embedding (the middle layer).
            if self.layerwise_reflective:
                num_vectors_per_token = num_unet_enc_layers + 1
            else:
                num_vectors_per_token = num_unet_enc_layers * 2 + 1

        # hasattr(embedder, 'tokenizer') -> True
        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            # get_token_for_string = get_clip_token_for_string
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                if self.layerwise_lora_rank > 0:
                    token_params = LoraEmbedding(num_vectors_per_token, token_dim, self.layerwise_lora_rank, init_word_embedding)
                else:
                    # ANCHOR[id=init_embed] : num_vectors_per_token vectors are initialized with the same embedding.
                    token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                # initial_embeddings are for computing the regularization loss.
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                if self.layerwise_lora_rank > 0:
                    token_params = LoraEmbedding(num_vectors_per_token, token_dim, self.layerwise_lora_rank)
                else:
                    token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
            
            self.string_to_token_dict[placeholder_string] = token
            # token_params: embedding vector.
            self.string_to_param_dict[placeholder_string] = token_params

    # "Patch" the returned embeddings of CLIPTextEmbeddings.
    # If self.use_layerwise_embedding, then max_vectors_per_token = num_unet_layers = 25.
    def forward(
            self,
            tokenized_text,         # [B, N]. Identical B copies along the batch dimension.
            embedded_text,          # [B, N, 768]. Identical B copies along the batch dimension.
    ):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        D = embedded_text.shape[-1]

        for placeholder_string, placeholder_token in self.string_to_token_dict.items():

            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)
            # Generate the actual placeholder_embedding on the fly.
            if isinstance(placeholder_embedding, LoraEmbedding):
                placeholder_embedding = placeholder_embedding()

            # max_vectors_per_layer_per_token == 1: original num_vectors_per_token == 1, but 
            # self.use_layerwise_embedding could still be True.
            if self.max_vectors_per_layer_per_token == 1: # If there's only one vector per token, we can do a simple replacement
                if self.use_layerwise_embedding:
                    # embedded_text: [B, 25, N, 768] => [25*B, N, 768].
                    # "Tuck" the layer dimension into the batch dimension, 
                    # to keep embedded_text in 3D, same as the input.
                    embedded_text = embedded_text.unsqueeze(1).repeat(1, self.num_unet_layers, 1, 1).view(b * self.num_unet_layers, n, -1)
                    # tokenized_text: [B, 25, N] => [25*B, N]
                    # tokenized_text has to be repeated along the layer dimension as well, so that 
                    # placeholder_idx can index the embedding at each layer in the batch.
                    tokenized_text = tokenized_text.unsqueeze(1).repeat(1, self.num_unet_layers, 1).view(b * self.num_unet_layers, n)
                    # mirror-reflect the embedding along the layer dimension, to make it symmetric 
                    # in the encoder & decoder.

                    if self.layerwise_reflective:
                        placeholder_embedding_enc = placeholder_embedding[:-1]
                        placeholder_embedding = torch.cat([ placeholder_embedding_enc, 
                                                            placeholder_embedding[-1].unsqueeze(0), 
                                                            placeholder_embedding_enc.flip(0) ], dim=0)

                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                if placeholder_idx[0].numel() == 0:
                    continue

                # embedded_text[placeholder_idx] indexes the embedding at each instance in the batch.
                # Non-layerwise: embedded_text[placeholder_idx]: [2, 768].  placeholder_embedding: [1, 768].
                # layerwise: placeholder_idx =  
                # (tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 
                #  tensor([ 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]))
                # embedded_text[placeholder_idx]: [50, 768]. placeholder_embedding: [25, 768].
                # The first 25 elements (0-8) in embedded_text[placeholder_idx] correspond to the 25 layers of the 
                # first instance in the batch.
                # 25 layers of placeholder_embedding are repeated b times.
                # placeholder_embedding: placeholder_embedding: [25, 768] repeat=> [50, 768]
                # Note that the 25 layers are initialized with the same embedding. 
                # LINK #init_embed
                embedded_text[placeholder_idx] = placeholder_embedding.repeat(b, 1)

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
                    new_embed_row = torch.cat([embedded_text[row][:col],  placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                    # Locate the next placeholder token in the row, and replace the embedding in embedded_text.
                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    # save custom tokens and their learned embeddings to "embeddings_gs-4200.pt".
    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    # load custom tokens and their learned embeddings from "embeddings_gs-4200.pt".
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    # get_embedding_norms_squared() is never used.
    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    def embedding_to_coarse_loss(self):
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss

    def embedding_to_layerwise_loss(self):
        loss = 0.
        num_embeddings = len(self.initial_embeddings)
        euc_loss_type       = 'l2'       # l1, l2
        euc_loss_weight     = 1.0
        cosine_loss_weight  = 1 - euc_loss_weight
        l2_norm_weight      = 0.01
        reg_center_type     = 'init'     # avg, init

        for key in self.initial_embeddings:
            embeddings = self.string_to_param_dict[key]
            # Generate the actual embeddings on the fly.
            if isinstance(embeddings, LoraEmbedding):
                embeddings = embeddings()

            if reg_center_type == 'init':
                # initial_embeddings[key] is already [L, 768]. No need to repeat().
                reg_center = self.initial_embeddings[key].clone().to(embeddings.device)
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

        return loss / num_embeddings
    
    def embedding_to_loss(self):
        if self.use_layerwise_embedding:
            return self.embedding_to_layerwise_loss()
        else:
            return self.embedding_to_coarse_loss()
