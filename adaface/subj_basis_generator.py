# Borrowed from ip-adapter resampler.py.
# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

from torch import einsum
from adaface.util import gen_gradient_scaler
from adaface.arc2face_models import CLIPTextModelWrapper

def reshape_tensor(x, num_heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, num_heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2).contiguous()
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, num_heads, length, -1)
    return x

# FFN. Added a Dropout layer at the end, so that it can still load the old ckpt.
def FeedForward(dim, mult=4, p_dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
        nn.Dropout(p_dropout),
    )

# IP-Adapter FaceID class. Only used in knn-faces.py.
# From: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter_faceid_separate.py
class IP_MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class LayerwiseMLPProjWithSkip(nn.Module):
    def __init__(self, id_embeddings_dim=768, num_layers=16, dim_mult=2):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*dim_mult*num_layers),
            Rearrange('b n (l d) -> b n l d', l=num_layers, d=id_embeddings_dim*dim_mult),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*dim_mult, id_embeddings_dim),
        )
        self.norm = nn.LayerNorm(id_embeddings_dim)

    def forward(self, id_embeds):
        # B N D -> B N L D + B N L D -> B N L D
        x = self.proj(id_embeds) + id_embeds.unsqueeze(1)
        x = self.norm(x)
        return x
        
# group_dim: the tensor dimension that corresponds to the multiple groups.
class LearnedSoftAggregate(nn.Module):
    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim  = group_dim
        # num_feat = 1: element-wise score function & softmax.
        # num_feat > 1: the linear score function is applied to the last dim (features) of the input tensor. 
        self.num_feat   = num_feat
        self.feat2score = nn.Linear(num_feat, 1, bias=False)
        self.keepdim    = keepdim

    def forward(self, x, score_basis=None):
        # If there's only one mode, do nothing.
        if x.shape[self.group_dim] == 1:
            if self.keepdim:
                return x
            else:
                return x.squeeze(self.group_dim)
            
        # Assume the last dim of x is the feature dim.
        if score_basis is None:
            score_basis = x
        
        if self.num_feat == 1:
            mode_scores = self.feat2score(score_basis.unsqueeze(-1)).squeeze(-1)
        else:
            mode_scores = self.feat2score(score_basis)
        attn_probs  = mode_scores.softmax(dim=self.group_dim)
        x_aggr      = (x * attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
        return x_aggr
    
def LoRA_ExpandEmbs(input_dim, lora_rank, output_dim, num_modes, 
                    num_output_vecs, elementwise_affine=True, p_dropout=0.1):
    return nn.Sequential(
        # Project to [BS, lora_rank * output_dim * num_modes].
        # It takes a huge param size. 512 * 32 * 768 * 4 = 6,291,456.
        nn.Linear(input_dim, lora_rank * output_dim * num_modes, bias=False),
        # Reshape to [BS, lora_rank, output_dim].
        Rearrange('b (m q d) -> b m q d', q=lora_rank, m=num_modes, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        # Aggregate [BS, num_modes, loar_rank, output_dim] -> [BS, lora_rank, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False) if num_modes > 1 \
            else Rearrange('b () q d -> b q d'),
        nn.Dropout(p_dropout),
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, num_output_vecs].
        nn.Linear(lora_rank, num_output_vecs, bias=False),
        # Permute to [BS, num_output_vecs, output_dim].
        Rearrange('b d q -> b q d'),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(p_dropout),
    )

def ExpandEmbs(input_dim, output_dim, expansion_ratio, elementwise_affine=True, p_dropout=0.1):
    return nn.Sequential(
        # Project to [BS, num_output_vecs * output_dim].
        nn.Linear(input_dim, expansion_ratio * output_dim, bias=False),
        # Reshape to [BS, num_output_vecs, output_dim].
        Rearrange('b (e d) -> b e d', e=expansion_ratio, d=output_dim),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        nn.Dropout(p_dropout),
    )

# Input: [BS, N, D].
def MultimodeProjection(input_dim, output_dim=-1, num_modes=4, elementwise_affine=True, p_dropout=0.1):
    if output_dim == -1:
        output_dim = input_dim

    return nn.Sequential(
            nn.Linear(input_dim, output_dim * num_modes, bias=False),
            # Reshape to [BS, num_output_vecs, output_dim].
            Rearrange('b n (m d) -> b n m d', m=num_modes, d=output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
            # If num_modes == 1, then simply remove the mode dim. Otherwise, aggregate the modes.
            LearnedSoftAggregate(num_feat=output_dim, group_dim=2, keepdim=False) if num_modes > 1 \
                else Rearrange('b n () d -> b n d'),
            nn.Dropout(p_dropout),
    )

# Low-rank to high-rank transformation.
def Lora2Hira(lora_rank, hira_rank, output_dim, num_modes, elementwise_affine=True, p_dropout=0.1):
    return nn.Sequential(        
        # Permute to [BS, output_dim, lora_rank].
        Rearrange('b q d -> b d q'),
        # Project to [BS, output_dim, hira_rank].
        nn.Linear(lora_rank, hira_rank * num_modes, bias=False),
        # Reshape and permute to [BS, num_modes, num_output_vecs, output_dim].
        Rearrange('b d (m q) -> b m q d', m=num_modes, q=hira_rank),
        nn.LayerNorm(output_dim, elementwise_affine=elementwise_affine),
        # Aggregate [BS, num_modes, hira_rank, output_dim] -> [BS, hira_rank, output_dim].
        LearnedSoftAggregate(num_feat=output_dim, group_dim=1, keepdim=False) if num_modes > 1 \
            else Rearrange('b () q d -> b q d'),       
        nn.Dropout(p_dropout),    
    )

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, num_heads=8, elementwise_affine=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)

        self.to_q   = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv  = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latent_queries):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latent_queries = self.norm2(latent_queries)

        b, l, _ = latent_queries.shape

        q = self.to_q(latent_queries)
        kv_input = torch.cat((x, latent_queries), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.num_heads)
        k = reshape_tensor(k, self.num_heads)
        v = reshape_tensor(v, self.num_heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        attn = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = attn @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class CrossAttention(nn.Module):
    # output_dim is always the same as input_dim.
    # num_q only matters when q_aware_to_v is True. 
    # If q_aware_to_v is False, query x in forward() is still usable.
    def __init__(self, input_dim, num_heads=6, p_dropout=0.05, 
                 identity_to_q=False, identity_to_k=False, identity_to_v=False, v_has_skip=True,
                 q_aware_to_v=True, num_q=416, v_repeat=4, q_aware_to_v_lora_rank=64,
                 identity_to_out=False, out_has_skip=False):
        super().__init__()
        dim_head  = input_dim // num_heads
        inner_dim = dim_head   * num_heads

        self.num_heads      = num_heads
        self.q_aware_to_v   = q_aware_to_v
        self.v_has_skip     = v_has_skip
        self.to_q = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=True) 
                    ) if not identity_to_q else nn.Identity()
        self.to_k = nn.Sequential(
                        nn.Linear(input_dim, inner_dim, bias=False),
                        nn.LayerNorm(inner_dim, elementwise_affine=True) 
                    ) if not identity_to_k else nn.Identity()
        
        self.v_repeat = v_repeat
        self.num_q_group = num_q_group = num_q // v_repeat      # 416 / 4 = 104.

        # If q_aware_to_v is True, then self.to_v consists of num_q projections of input_dim to inner_dim.
        # Otherwise, self.to_v consists of a single projection of input_dim to inner_dim.
        if q_aware_to_v:
            # all_q_mid: 104 * 64 = 6656.
            all_q_mid = num_q_group * q_aware_to_v_lora_rank
            self.to_v = nn.Sequential(
                # number of params: 768 * 6656 = 5,111,808.
                # Input:  [BS, 16, 768]. Output: [BS, 16, 104*64] = [BS, 16, 6656].
                # Each 768-dim vec is dispersed into 104 64-dim vecs.
                nn.Linear(input_dim, all_q_mid, bias=False),
                nn.LayerNorm(all_q_mid, elementwise_affine=True),
                # Change the dim of the tensor to [BS, 6656, 16], as Conv1d transforms dim 1.
                Rearrange('b n q -> b q n', q=all_q_mid),
                # Each q_aware_to_v projection has its own linear layer.
                # The total number of parameters will be 6656*768 = 5,111,808.
                # Output: [BS, 104*768, 16]. Each 64 dim feature is expanded to 768 dim.
                nn.Conv1d(
                    in_channels=all_q_mid,
                    out_channels=num_q_group * input_dim,
                    kernel_size=1,
                    groups=num_q_group,
                    bias=False,
                ),
                # Output: [BS, 104, 16, 768].
                Rearrange('b (q d) n -> b q n d', q=num_q_group, d=input_dim),
                nn.LayerNorm(input_dim, elementwise_affine=True),
            )
        else:
            self.to_v = nn.Sequential(
                            nn.Linear(input_dim, inner_dim, bias=False),
                            nn.LayerNorm(inner_dim, elementwise_affine=True) 
                        ) if not identity_to_v else nn.Identity()

        if identity_to_out:
            assert not out_has_skip, "identity_to_out=True, then out_has_skip has to be False."

        if identity_to_out:
            self.to_out = nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=False),
                nn.Dropout(p_dropout),
                nn.LayerNorm(inner_dim, elementwise_affine=True)
            )

        self.out_has_skip = out_has_skip
        self.attn_drop = nn.Dropout(p_dropout)

    def forward(self, x, context=None, attn_mat=None, return_attn=False):
        h = self.num_heads

        if context is None:
            context = x

        if attn_mat is None:
            # q: [BS, Q, D] -> [BS, Q, D].
            q = self.to_q(x)
            # k: [BS, L, D] -> [BS, L, D].
            k = self.to_k(context)
            # q: [6, 512, 128], k: [6, 17, 128].
            q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))

        if self.q_aware_to_v:
            # context: [BS, L, D]. v: [BS, Q, L, D].
            # There are effectively Q to_v projections.
            v = self.to_v(context)
            if self.v_has_skip:
                v = v + context.unsqueeze(1)
        else:
            # v: [BS, L, D].
            v = self.to_v(context)
            if self.v_has_skip:
                v = v + context

        #print(v.shape)

        if self.q_aware_to_v:
            # v: [6, 64, 17, 128].
            # v is query-specific, so there's an extra dim for the query.
            v = rearrange(v, 'b q n (h d) -> (b h) q n d', h=h).contiguous()
            # Each v is for a query group with 512/64 = 8 queries.
            # So each v is repeated 8 times to match the number of queries.
            # v: [6, 64, 17, 128] -> [6, 512, 17, 128].
            v = v.repeat(1, self.v_repeat, 1, 1)
        else:
            v = rearrange(v, 'b n (h d) -> (b h) n d', h=h).contiguous()

        if attn_mat is None:
            scale = q.size(-1) ** -0.25
            sim = einsum('b i d, b j d -> b i j', q * scale, k * scale)
            # sim: [6, 64, 17]. 6: bs 1 * h 6.
            # attention, what we cannot get enough of
            # NOTE: the normalization is done across tokens, not across pixels.
            # So for each pixel, the sum of attention scores across tokens is 1.
            attn = sim.softmax(dim=-1)
            attn = self.attn_drop(attn)
            #print(attn.std())
        else:
            attn = attn_mat

        if self.q_aware_to_v:
            # attn: [6, 32, 17]. v: [6, 32, 17, 128]. 128: dim of each head. out: [6, 32, 128].
            # out is combined with different attn weights and v for different queries.
            out = einsum('b i j, b i j d -> b i d', attn, v)
        else:
            # v: [6, 17, 128]. out: [6, 32, 128].
            out = einsum('b i j, b j d -> b i d',   attn, v)

        # [6, 32, 128] -> [1, 32, 768].
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).contiguous()

        if self.out_has_skip:
            out = self.to_out(out) + out
        else:
            out = self.to_out(out)

        if return_attn:
            return out, attn
        else:
            return out


class ImgPrompt2TextPrompt(nn.Module):
    def __init__(self, placeholder_is_bg, num_id_vecs, dtype=torch.float16, *args, **kwargs):
        super().__init__()
        self.N_ID  = num_id_vecs
        # If not placeholder_is_bg, then N_SFX will be updated in initialize_text_components().
        self.N_SFX = 0

        if not placeholder_is_bg:
            self.initialize_text_components(*args, **kwargs)

        # prompt2token_proj: arc2face_models.py CLIPTextModelWrapper instance with **custom weights**.
        # prompt2token_proj is with the same architecture as the original arc2face text encoder, 
        # but retrained to do inverse mapping.
        # To be initialized in the subclass.
        self.prompt2token_proj = None
        self.dtype = dtype

    def initialize_static_img_suffix_embs(self, num_static_img_suffix_embs, img_prompt_dim=768):
        self.N_SFX = num_static_img_suffix_embs
        # We always take the first num_static_img_suffix_embs embeddings out of static_img_suffix_embs.
        # So it's OK that static_img_suffix_embs is larger than required number num_static_img_suffix_embs.
        # This holds even if num_static_img_suffix_embs is 0.
        if hasattr(self, 'static_img_suffix_embs') and self.static_img_suffix_embs is not None:
            if self.static_img_suffix_embs.shape[1] == self.N_SFX:
                print(f"static_img_suffix_embs had been initialized to be {self.static_img_suffix_embs.shape[1]} vecs ({self.N_SFX} required). Skip initialization.")
            elif self.static_img_suffix_embs.shape[1] < self.N_SFX:
                print(f"static_img_suffix_embs had been initialized to be {self.static_img_suffix_embs.shape[1]} vecs (< {self.N_SFX} required). Reinitialize.")
                self.static_img_suffix_embs = nn.Parameter(torch.randn(1, self.N_SFX, img_prompt_dim))
            elif self.N_SFX > 0:
                # self.static_img_suffix_embs.shape[1] > self.N_SFX > 0.
                print(f"static_img_suffix_embs had been initialized to be {self.static_img_suffix_embs.shape[1]} vecs (> {self.N_SFX} required). Truncate.")
                self.static_img_suffix_embs = nn.Parameter(self.static_img_suffix_embs[:, :self.N_SFX])
            else:
                # self.static_img_suffix_embs.shape[1] > self.N_SFX == 0.
                print(f"static_img_suffix_embs had been initialized to be {self.static_img_suffix_embs.shape[1]} vecs (0 required). Erase.")
                self.static_img_suffix_embs = None
        else:
            if self.N_SFX > 0:
                # Either static_img_suffix_embs does not exist or is None, 
                # or it's initialized but has fewer than num_static_img_suffix_embs embeddings (this situation should be very rare, 
                # so we don't consider to reuse and extend a shorter static_img_suffix_embs).
                # So we reinitialize it.
                self.static_img_suffix_embs = nn.Parameter(torch.randn(1, self.N_SFX, img_prompt_dim))
            else:
                # If static_img_suffix_embs had been initialized, then it will be set to None, i.e., erased from the SubjBasisGenerator instance.
                self.static_img_suffix_embs = None

    # Implement a separate initialization function, so that it can be called from SubjBasisGenerator
    # after the SubjBasisGenerator is initialized. This can be used to fix old SubjBasisGenerator 
    # ckpts which were not subclassed from ImgPrompt2TextPrompt.
    def initialize_text_components(self, max_prompt_length=77, num_id_vecs=16, 
                                   num_static_img_suffix_embs=0, img_prompt_dim=768):
        self.initialize_static_img_suffix_embs(num_static_img_suffix_embs, img_prompt_dim)
        self.max_prompt_length = max_prompt_length
        self.tokenizer       = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # clip_text_embeddings: CLIPTextEmbeddings instance.
        clip_text_embeddings = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").text_model.embeddings
        # clip_text_embeddings() and clip_text_embeddings.token_embedding() differ in that 
        # clip_text_embeddings() adds positional embeddings, while clip_text_embeddings.token_embedding() doesn't.
        # Adding positional embeddings seems to help somewhat.
        # pad_tokens: pad_token_id 49407 repeated 77 times.
        # pad_token_id is the EOS token. But BOS is 49406.
        pad_tokens = torch.tensor([self.tokenizer.pad_token_id]).repeat(self.max_prompt_length)
        # pad_embeddings: [77, 768]. 
        # pad_embeddings is still on CPU. But should be moved to GPU automatically.
        # Note: detach pad_embeddings from the computation graph, otherwise 
        # deepcopy() in embedding_manager.py:make_frozen_copy_of_subj_basis_generators() will fail.
        self.pad_embeddings = clip_text_embeddings(pad_tokens)[0].detach()

    # image prompt space -> text prompt space.
    # return_emb_types: a list of strings, each string is among 
    # ['full', 'core', 'full_pad', 'full_half_pad'].
    def inverse_img_prompt_embs(self, face_prompt_embs, list_extra_words,
                                return_emb_types, hidden_state_layer_weights=None, 
                                enable_static_img_suffix_embs=False):

        '''
        face_prompt_embs: (BS, self.N_ID, 768), in the image prompt space. 
        Only the core embeddings, no paddings.
        list_extra_words: None or [s_1, ..., s_BS], each s_i is a list of extra words to be added to the prompt.
        '''
        if list_extra_words is not None:
            if len(list_extra_words) != len(face_prompt_embs):
                if len(face_prompt_embs) > 1:
                    print("Warn: list_extra_words has different length as face_prompt_embs.")
                    if len(list_extra_words) == 1:
                        list_extra_words = list_extra_words * len(face_prompt_embs)
                    else:
                        breakpoint()
                else:
                    # len(face_prompt_embs) == 1, this occurs when same_subject_in_batch == True, e.g. in do_feat_distill_on_comp_prompt.
                    # But list_extra_words always corresponds to the actual batch size. So we only take the first element.
                    list_extra_words = list_extra_words[:1]
                    
            for extra_words in list_extra_words:
                assert len(extra_words.split()) <= 2, "Each extra_words string should consist of at most 2 words."
            # 16 or 4 ", " are placeholders for face_prompt_embs.
            prompt_templates = [ "photo of a " + ", " * self.N_ID + list_extra_words[i] for i in range(len(list_extra_words)) ]
        else:
            # 16 or 4 ", " are placeholders for face_prompt_embs.
            # No extra words are added to the prompt. So we add 2 more ", " to the template to keep 
            # the number of tokens roughly the same as when extra words are added.
            prompt_templates = [ "photo of a " + ", " * (self.N_ID + 2) for _ in range(len(face_prompt_embs)) ]

        # This step should be quite fast, and there's no need to cache the input_ids.
        # input_ids: [BS, 77].
        input_ids = self.tokenizer(
                prompt_templates,
                truncation=True,
                padding="max_length",
                max_length=self.max_prompt_length,
                return_tensors="pt",
            ).input_ids.to(face_prompt_embs.device)

        face_prompt_embs_orig_dtype = face_prompt_embs.dtype
        face_prompt_embs            = face_prompt_embs.to(self.dtype)

        ID_END      = 4 + self.N_ID
        PAD_BEGIN   = ID_END + self.N_SFX + 2

        # token_embs: [1, 77, 768]. This call is only to get the template token embeddings (the shallowest mapping).
        token_embs = self.prompt2token_proj(input_ids=input_ids, return_token_embs=True)
        # token 4: first ", " in the template prompt.
        # Replace embeddings of 16 or 4 placeholder ", " with face_prompt_embs.
        token_embs[:, 4:ID_END] = face_prompt_embs
        # Only when do_unet_distill == True, we append the static image suffix embeddings.
        # Otherwise, static image suffix embeddings are ignored,
        # and token_embs[:, ID_END:ID_END+self.N_SFX] are the filler embeddings of the 
        # extra ", " in the template prompt.
        if enable_static_img_suffix_embs and self.N_SFX > 0:
            # Put the static image suffix embeddings right after face_prompt_embs.
            token_embs[:, ID_END:ID_END+self.N_SFX] = self.static_img_suffix_embs[:, :self.N_SFX]

        # This call does the ordinary CLIP text encoding pass.
        prompt_embeds = self.prompt2token_proj(
            input_ids=input_ids,
            input_token_embs=token_embs,
            hidden_state_layer_weights=hidden_state_layer_weights,
            return_token_embs=False
        )[0]

        # Restore the original dtype of prompt_embeds: float16 -> float32.
        prompt_embeds = prompt_embeds.to(face_prompt_embs_orig_dtype)
        # token 4: first ", " in the template prompt.
        # When N_ID == 16,
        # prompt_embeds 4:20 are the most important 16 embeddings that contain the subject's identity.
        # 20:22 are embeddings of the (at most) two extra words.
        # [N, 77, 768] -> [N, 16, 768]
        if enable_static_img_suffix_embs:
            core_prompt_embs = prompt_embeds[:, 4:ID_END+self.N_SFX]
        else:
            core_prompt_embs = prompt_embeds[:, 4:ID_END]

        if list_extra_words is not None:
            # [N, 16, 768] -> [N, 18, 768]
            extra_words_embs = prompt_embeds[:, ID_END+self.N_SFX:PAD_BEGIN]
            core_prompt_embs = torch.cat([core_prompt_embs, extra_words_embs], dim=1)

        returned_prompt_embs = []
        for emb_type in return_emb_types:
            if emb_type == 'full':
                returned_prompt_embs.append(prompt_embeds)
            elif emb_type == 'full_half_pad':
                prompt_embeds2 = prompt_embeds.clone()
                # PAD_BEGIN is 22 or 10. Also exclude the last EOS token. 
                # So we subtract max_prompt_length by (PAD_BEGIN + 1).
                PADS  = self.max_prompt_length - PAD_BEGIN - 1
                if PADS >= 2:
                    # Fill half of the remaining embeddings with pad embeddings.
                    prompt_embeds2[:, PAD_BEGIN:PAD_BEGIN+PADS//2] = self.pad_embeddings[PAD_BEGIN:PAD_BEGIN+PADS//2]
                returned_prompt_embs.append(prompt_embeds2)
            elif emb_type == 'full_pad':
                prompt_embeds2 = prompt_embeds.clone()
                # Replace the PAD_BEGIN-th to the second last embeddings with pad embeddings.
                # Skip replacing the last embedding, which might has special roles.
                # (Although all padding tokens are the same EOS, the last token might acquire special semantics
                # due to its special position.)
                prompt_embeds2[:, PAD_BEGIN:-1] = self.pad_embeddings[PAD_BEGIN:-1]
                returned_prompt_embs.append(prompt_embeds2)
            elif emb_type == 'full_zeroed_extra':
                prompt_embeds2 = prompt_embeds.clone()
                # Only add two pad embeddings. The remaining embeddings are set to 0.
                # Make the positional embeddings align with the actual positions.
                prompt_embeds2[:, 22:24] = self.pad_embeddings[22:24]
                prompt_embeds2[:, 24:-1] = 0
                returned_prompt_embs.append(prompt_embeds2)
            elif emb_type == 'core':
                returned_prompt_embs.append(core_prompt_embs)
            else:
                breakpoint()

        return returned_prompt_embs

class SubjBasisGenerator(ImgPrompt2TextPrompt):
    def __init__(
        self,
        dtype=torch.float16,
        # number of cross-attention heads of the bg prompt translator. 
        # Taken as a half of the number of heads 12 of OpenAI clip-vit-large-patch14:
        # https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
        num_bg_encoder_heads=6,                      
        # number of subject input identity vectors (only when the subject is not face), 
        # or number of background input identity vectors (no matter the subject is face or not).
        # 257: 257 CLIP tokens. 
        num_nonface_in_id_vecs={ 'subj': 77, 'bg': 257 },  
        num_ca_layers=16,
        num_id_vecs=16,                             # num_id_vecs: subj: 16. bg: 4.
        num_static_img_suffix_embs: int = 0,        # Number of extra static learnable image embeddings appended to translated ID embeddings.
        bg_image_embedding_dim=1024,                # CLIP image hidden layer feature dimension, as per config.json above.
        obj_embedding_dim=384,                      # DINO object feature dimension for objects.
        output_dim=768,                             # CLIP text embedding input dimension.
        use_layerwise_proj: bool = False,           # Whether to use layerwise projection.
        placeholder_is_bg: bool = False,            # Whether the placeholder is for the image background tokens.
        learnable_hidden_state_weights_scheme: str = 'per-layer',   # none, per-layer.
        bg_prompt_translator_has_to_out_proj:  bool = False,         # Whether the prompt_trans_layers have a to_out projection.
    ):

        # If not placeholder_is_bg, then it calls initialize_text_components() in the superclass.
        super().__init__(placeholder_is_bg=placeholder_is_bg, num_id_vecs=num_id_vecs, dtype=dtype,
                         max_prompt_length=77, num_static_img_suffix_embs=num_static_img_suffix_embs, 
                         img_prompt_dim=output_dim)

        self.placeholder_is_bg  = placeholder_is_bg
        self.num_ca_layers      = num_ca_layers
        self.num_out_embs       = self.N_ID + self.N_SFX
        self.output_dim         = output_dim
        # num_nonface_in_id_vecs should be the number of core ID embs, 16.
        # However, in such case, pos_embs is not used. So it doesn't matter if it's wrongly set.
        self.num_nonface_in_id_vecs = num_nonface_in_id_vecs['bg'] if placeholder_is_bg else num_nonface_in_id_vecs['subj']
        self.bg_prompt_translator_has_to_out_proj = bg_prompt_translator_has_to_out_proj

        if not self.placeholder_is_bg:
            # [1, 384] -> [1, 16, 768].
            # TODO: use CLIPTextModelWrapper as obj_proj_in.
            self.obj_proj_in = ExpandEmbs(obj_embedding_dim, output_dim, expansion_ratio=self.num_nonface_in_id_vecs)

            # ** prompt2token_proj does the actual job: **
            # it is the inverse projection that maps from faceid2img_prompt_embs to adaface_prompt_embs.
            # self.prompt2token_proj: [1, 16, 768] -> [1, 77, 768] (with paddings) or [1, 16, 768] (without paddings).
            # If self.placeholder_is_bg: prompt2token_proj is set to None.
            # Use an attention dropout of 0.2 to increase robustness.
            clip_dropout_config     = None #CLIPTextConfig.from_pretrained('openai/clip-vit-large-patch14', attention_dropout=0.05, dropout=0.05)
            self.prompt2token_proj  = CLIPTextModelWrapper.from_pretrained('openai/clip-vit-large-patch14',
                                                                           config=clip_dropout_config)
            if use_layerwise_proj:
                # MLPProjWithSkip: MLP with skip connection.
                # [BS, 4, 768] -> [BS, 16, 4, 768]. Extra 16: 16 layers.
                self.layerwise_proj     = LayerwiseMLPProjWithSkip(output_dim, dim_mult=2)
            else:
                self.layerwise_proj     = nn.Identity() #Rearrange('b n d -> b l n d', l=16)

            print(f"Subj prompt2token_proj initialized.")            
            # Only freeze token and positional embeddings of the original CLIPTextModel.
            self.freeze_prompt2token_proj()

            # These multipliers are relative to the original CLIPTextModel.
            self.prompt2token_proj_attention_multipliers = [1] * 12
            self.initialize_hidden_state_layer_weights(learnable_hidden_state_weights_scheme, 'cpu')
            self.bg_proj_in = None
            self.pos_embs = self.pos_embs_ln = self.latent_queries = self.latent_queries_ln = None
        else:
            # For background placeholders, face and object embeddings are not used as they are foreground.
            self.obj_proj_in  = None

            self.bg_proj_in = nn.Sequential(
                nn.Linear(bg_image_embedding_dim, output_dim, bias=False),
                nn.LayerNorm(output_dim),
            )

            self.pos_embs           = nn.Parameter(torch.zeros(1, self.num_nonface_in_id_vecs, output_dim))
            self.pos_embs_ln        = nn.LayerNorm(output_dim)
            self.latent_queries     = nn.Parameter(torch.randn(1, self.num_out_embs, output_dim))
            self.latent_queries_ln  = nn.LayerNorm(output_dim)

            identity_to_v   = False
            v_has_skip      = not identity_to_v                         # True
            identity_to_out = not bg_prompt_translator_has_to_out_proj  # True
            out_has_skip    = not identity_to_out                       # False
            # prompt_translator maps the clip image features (of the background) to the prompt embedding space.
            # It is only used during training when placeholder_is_bg is True.
            # prompt_translator has a to_v projection with skip connection, and doesn't have a to_out projection.
            # dim=768, num_bg_encoder_heads=6.
            self.prompt_translator = \
                CrossAttention(input_dim=output_dim, num_heads=num_bg_encoder_heads, p_dropout=0.05,
                               identity_to_q=False, identity_to_k=False, identity_to_v=identity_to_v,
                               q_aware_to_v=False,  v_has_skip=v_has_skip,
                               num_q=0, # When not q_aware_to_v, num_q is not referenced.
                               identity_to_out=identity_to_out,
                               out_has_skip=out_has_skip)
            
            self.output_scale = output_dim ** -0.5
            
            ''' 
            prompt_translator: CLIPEncoder
            # https://github.com/huggingface/transformers/blob/1872bde7fc6a5d6796bd742bc2dc38eaf8069c5d/src/transformers/models/clip/modeling_clip.py#L566
            # CLIPEncoder.layers: 12 layers of CLIPEncoderLayer, each being
                (0): CLIPEncoderLayer(
                    (self_attn): CLIPAttention(
                        (k_proj): Linear(in_features=768, out_features=768, bias=True)
                        (v_proj): Linear(in_features=768, out_features=768, bias=True)
                        (q_proj): Linear(in_features=768, out_features=768, bias=True)
                        (out_proj): Linear(in_features=768, out_features=768, bias=True)
                    )
                    (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (mlp): CLIPMLP(
                        (activation_fn): QuickGELUActivation()
                        (fc1): Linear(in_features=768, out_features=3072, bias=True)
                        (fc2): Linear(in_features=3072, out_features=768, bias=True)
                    )
                    (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                )
            '''

        print(repr(self))

    # raw_id_embs: only used when the subject is non-faces. In that case it's DINO embeddings.
    # Otherwise, raw_id_embs is not used.
    # faceid2img_prompt_embs: [BS, 16, 768], the core ID prompt embeddings generated by ID2ImgPrompt.
    def forward(self, faceid2img_prompt_embs, clip_features=None, raw_id_embs=None, out_id_embs_cfg_scale=1.0,
                is_face=True, enable_static_img_suffix_embs=False):    
        
        if not self.placeholder_is_bg:
            BS = faceid2img_prompt_embs.shape[0]
        else:
            # If bg, then faceid2img_prompt_embs is set to None, but clip_features is not None.
            BS = clip_features.shape[0]
            clip_features = clip_features.to(self.dtype)
            
        # No need to use raw_id_embs if placeholder_is_bg.
        if not self.placeholder_is_bg:
            if is_face:
                assert faceid2img_prompt_embs is not None
                # id2img_embs has been projected to the (modified) prompt embedding space 
                # by ID2AdaPrompt::map_init_id_to_img_prompt_embs(). This prompt embedding space is modified because 
                # the ID2ImgPrompt module (at least when it's arc2face) may have finetuned the 
                # text encoder and the U-Net.
                # in embedding_manager: [BS, 16, 768] -> [BS, 77, 768].
                # faceid2img_prompt_embs is part of id2img_embs: [BS, 77, 768] -> [BS, 16, 768].
                # adaface_prompt_embs is projected to the prompt embedding spaces. This is the 
                # original U-Net prompt embedding space.

                # hidden_state_layer_weights: [[0.9163], [0.9483], [2.0762]]
                hidden_state_layer_weights = self.hidden_state_layer_weights_grad_scaler(self.hidden_state_layer_weights)

                # faceid2img_prompt_embs -> ada_id_embs: image prompt space -> text prompt space.
                # If list_extra_words is not None, then ada_id_embs: [BS, 18, 768], three leading words, the 16 identity tokens 
                # and (at most) two extra words in adaface_prompt_embs, without BOS and EOS.
                # If list_extra_words is None, then ada_id_embs: [BS, 16, 768], the 16 identity tokens in adaface_prompt_embs.
                # hidden_state_layer_weights: [[0.9163], [0.9483], [2.0762]]
                # ada_id_embs: [BS, 16, 768].
                # return_emb_types: a list of strings, each string is among 
                # ['full', 'core', 'full_pad', 'full_half_pad'].
                ada_id_embs, = \
                    self.inverse_img_prompt_embs(faceid2img_prompt_embs, 
                                                 list_extra_words=None,
                                                 return_emb_types=['core'], 
                                                 hidden_state_layer_weights=hidden_state_layer_weights,
                                                 enable_static_img_suffix_embs=enable_static_img_suffix_embs)
            elif raw_id_embs is not None:
                # id_embs: [BS, 384] -> [BS, 18, 768].
                # obj_proj_in is expected to project the DINO object features to 
                # the token embedding space. So no need to use prompt2token_proj.
                id_embs = self.obj_proj_in(raw_id_embs)
            else:
                breakpoint()
        else:
            # Otherwise, context is the ad-hoc CLIP image features.
            # id_embs: [BS, 257, 768].
            id_embs = self.bg_proj_in(clip_features)

        if self.placeholder_is_bg:
            id_embs = id_embs + self.pos_embs_ln(self.pos_embs)
            latent_queries = self.latent_queries_ln(self.latent_queries).repeat(BS, 1, 1)
            # If bg, we don't have to use a specific attn layer for each 4-vec set. Instead, one attn layer can generate 257 embs, 
            # and we take the first 16*4=64.             
            # Output of prompt_translator is exactly num_out_embs == 64 tokens. id_embs_out: [BS, 64, 768].
            # prompt_translator: better named as bg_prompt_translator. It maps the bg features 
            # to bg prompt embeddings.
            with torch.set_grad_enabled(self.training):
                id_embs_out = self.prompt_translator(latent_queries, id_embs)

            adaface_out_embs = id_embs_out * self.output_scale    # * 0.036
        else:
            # [BS, 16, 768] -> [BS, layers=16, tokens=16, 768]
            adaface_out_embs = self.layerwise_proj(ada_id_embs)
            # If out_id_embs_cfg_scale < 1, adaface_out_embs is a mix of adaface_out_embs and pad_embeddings.
            if out_id_embs_cfg_scale != 1:
                # pad_embeddings: [77, 768] -> [16, 768] -> [1, 1, 16, 768].
                # NOTE: Never do cfg on static image suffix embeddings. 
                # So we take self.N_ID embeddings, instead of self.N_ID + self.N_SFX, 
                # even if enable_static_img_suffix_embs=True.
                pad_embeddings = self.pad_embeddings[4:4+self.N_ID].unsqueeze(0).unsqueeze(1).to(ada_id_embs.device)
                adaface_out_embs[:, :self.N_ID] = ada_id_embs[:, :self.N_ID] * out_id_embs_cfg_scale \
                                                  + pad_embeddings           * (1 - out_id_embs_cfg_scale)

        return adaface_out_embs

    def initialize_hidden_state_layer_weights(self, learnable_hidden_state_weights_scheme, device):
        if learnable_hidden_state_weights_scheme == 'none':
            self.hidden_state_layer_weights = None
            # A grad scaler with alpha =1 is nn.Identity(), which outputs None given None as input.
            self.hidden_state_layer_weights_grad_scaler = gen_gradient_scaler(1)
            print("hidden_state_layer_weights is set to None.")

        elif learnable_hidden_state_weights_scheme == 'per-layer':
            # Learnable weights of the last 3 layers, initialized to putting more focus on the last layer.
            # 'per-layer': Different weights for different layers, but the same for different channels.
            # hidden_state_layer_weights: [3, 1].
            self.hidden_state_layer_weights = nn.Parameter(torch.tensor([[1.0], [2.0], [4.0]], device=device),
                                                            requires_grad=True)
            # A gradient scaler of 5 makes the gradients on hidden_state_layer_weights 5 times larger.
            self.hidden_state_layer_weights_grad_scaler = gen_gradient_scaler(5)
            print("hidden_state_layer_weights initialized as per-layer [1, 2, 4], with grad scaler 5.")
        else:
            breakpoint()

    def extend_prompt2token_proj_attention(self, prompt2token_proj_attention_multipliers=None, 
                                           begin_layer_idx=-1, end_layer_idx=-1, multiplier=1, perturb_std=0.1):
        if begin_layer_idx == -1:
            begin_layer_idx = 0
        if end_layer_idx == -1:
            end_layer_idx = 11
            
        if prompt2token_proj_attention_multipliers is None and multiplier == 1:
            print("prompt2token_proj_attention_multipliers are all 1. No extension is done.")
            return
        
        elif prompt2token_proj_attention_multipliers is None:
            # prompt2token_proj_attention_multipliers are relative to the current prompt2token_proj.
            prompt2token_proj_attention_multipliers = [1] * 12
            for i in range(begin_layer_idx, end_layer_idx+1):
                prompt2token_proj_attention_multipliers[i] = multiplier            
        # Otherwise, use the given prompt2token_proj_attention_multipliers.

        num_extended_layers = self.prompt2token_proj.extend_clip_attention_MKV_multiplier(prompt2token_proj_attention_multipliers, perturb_std)
        # Update prompt2token_proj_attention_multipliers (relative to the original CLIPTextModel).
        for i in range(begin_layer_idx, end_layer_idx+1):
            self.prompt2token_proj_attention_multipliers[i] *= prompt2token_proj_attention_multipliers[i]

        print(f"{num_extended_layers} layers in prompt2token_proj_attention are extended by {prompt2token_proj_attention_multipliers}")
        return num_extended_layers
    
    def squeeze_prompt2token_proj_attention(self, prompt2token_proj_attention_divisors=None, 
                                            begin_layer_idx=-1, end_layer_idx=-1, divisor=1):
        if begin_layer_idx == -1:
            begin_layer_idx = 0
        if end_layer_idx == -1:
            end_layer_idx = 11
        
        if prompt2token_proj_attention_divisors is None and divisor == 1:
            print("prompt2token_proj_attention_divisors are all 1. No squeezing is done.")
            return
        elif prompt2token_proj_attention_divisors is None:
            prompt2token_proj_attention_divisors = [1] * 12
            for i in range(begin_layer_idx, end_layer_idx+1):
                prompt2token_proj_attention_divisors[i] = divisor
        # Otherwise, use the given prompt2token_proj_attention_divisors.

        num_squeezed_layers = self.prompt2token_proj.squeeze_clip_attention_MKV_divisor(prompt2token_proj_attention_divisors)
        # Update prompt2token_proj_attention_multipliers (relative to the original CLIPTextModel).
        for i in range(begin_layer_idx, end_layer_idx+1):
            self.prompt2token_proj_attention_multipliers[i] //= prompt2token_proj_attention_divisors[i]
            
        print(f"{num_squeezed_layers} layers in prompt2token_proj_attention are squeezed by {prompt2token_proj_attention_divisors}")
        return num_squeezed_layers
    
    def freeze_prompt2token_proj(self):
        # Only applicable to fg basis generator.
        if self.placeholder_is_bg:
            return
        
        if self.prompt2token_proj is not None:
            frozen_param_names = []
            for param_name, param in self.prompt2token_proj.text_model.embeddings.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_param_names.append(param_name)
                # If param is already frozen, then no need to freeze it again.
            print(f"{len(frozen_param_names)} params of token_pos_embeddings in Subj prompt2token_proj is frozen.")
            #print(f"Frozen parameters:\n{frozen_param_names}")

    def patch_old_subj_basis_generator_ckpt(self):
        # Fix compatability with the previous version.
        if not hasattr(self, 'bg_prompt_translator_has_to_out_proj'):
            self.bg_prompt_translator_has_to_out_proj = False
        if hasattr(self, 'num_id_vecs') and not hasattr(self, 'N_ID'):
            self.N_ID = self.num_id_vecs
        # Update the number of output embeddings.
        self.num_out_embs = self.N_ID + self.N_SFX

        if not hasattr(self, 'num_nonface_in_id_vecs') and hasattr(self, 'N_ID'):
            self.num_nonface_in_id_vecs = self.N_ID
        if not hasattr(self, 'dtype'):
            self.dtype = torch.float16
        if not hasattr(self, 'num_ca_layers'):
            self.num_ca_layers = 16
            
        if self.placeholder_is_bg:
            if not hasattr(self, 'pos_embs') or self.pos_embs is None:
                self.pos_embs = nn.Parameter(torch.zeros(1, self.num_nonface_in_id_vecs, self.output_dim))
            if not hasattr(self, 'latent_queries') or self.latent_queries is None:        
                self.latent_queries = nn.Parameter(torch.randn(1, self.num_out_embs, self.output_dim))
            # Background encoder doesn't require initializing text components.
        else:
            self.initialize_hidden_state_layer_weights('per-layer', 'cpu')
            if not hasattr(self, 'prompt2token_proj_attention_multipliers'):
                # Please manually set prompt2token_proj_attention_multipliers in the ckpt.
                breakpoint()

            self.initialize_text_components(max_prompt_length=77, num_id_vecs=self.N_ID, 
                                            num_static_img_suffix_embs=self.N_SFX, 
                                            img_prompt_dim=self.output_dim)

            if not hasattr(self, 'use_layerwise_proj'):
                self.use_layerwise_proj = False
            if not hasattr(self, 'layerwise_proj'):
                if self.use_layerwise_proj:
                    self.layerwise_proj = LayerwiseMLPProjWithSkip(self.output_dim, dim_mult=2)
                else:
                    self.layerwise_proj = nn.Identity()

    def __repr__(self):
        type_sig = 'subj' if not self.placeholder_is_bg else 'bg'

        return f"{type_sig} SubjBasisGenerator: num_out_embs={self.num_out_embs}, " \
               f"bg_prompt_translator_has_to_out_proj={self.bg_prompt_translator_has_to_out_proj}"
    
