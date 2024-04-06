"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import math
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR, ConstantLR, OneCycleLR, SequentialLR, \
                                     PolynomialLR, CosineAnnealingWarmRestarts, CyclicLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.modules.subj_basis_generator import CLIPVisionModelWithMask
from transformers import CLIPImageProcessor, CLIPTokenizer, ViTFeatureExtractor, ViTModel
from insightface.app import FaceAnalysis

from ldm.util import    log_txt_as_img, exists, default, ismap, isimage, mean_flat, \
                        count_params, calc_stats, instantiate_from_config, SequentialLR2, \
                        ortho_subtract, ortho_l2loss, gen_gradient_scaler, calc_dyn_loss_scale, \
                        save_grid, chunk_list, normalize_dict_values, normalized_sum, masked_mean, \
                        join_dict_of_indices_with_key_filter, init_x_with_fg_from_training_image, \
                        sel_emb_attns_by_indices, convert_attn_to_spatial_weight, resize_mask_for_feat_or_attn, \
                        calc_ref_cosine_loss, calc_delta_alignment_loss, calc_prompt_emb_delta_loss, \
                        calc_elastic_matching_loss, calc_layer_subj_comp_k_or_v_ortho_loss, \
                        distribute_embedding_to_M_tokens_by_dict, merge_cls_token_embeddings, mix_static_vk_embeddings, \
                        clamp_prompt_embedding, fix_emb_scales, replace_prompt_comp_extra, \
                        halve_token_indices, double_token_indices, extend_indices_N_by_n_times, \
                        gen_comp_extra_indices_by_block, extend_indices_B_by_n_times, \
                        split_indices_by_instance, repeat_selected_instances, \
                        anneal_t_keep_prob, anneal_value, gen_cfg_scales_for_stu_tea, \
                        get_arc2face_id_prompt_embs
                                              

from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from evaluation.clip_eval import CLIPEvaluator
from ldm.prodigy import Prodigy
import copy
from functools import partial
import random
from safetensors.torch import load_file as safetensors_load_file
import sys
import collections
import re, cv2
from PIL import Image

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,    # clip the range of denoised variables, not the CLIP model.
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 unfreeze_model=False,
                 model_lr=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 optimizer_type='Prodigy',
                 grad_clip=0.5,
                 manual_accumulate_grad_batches=1,
                 adam_config=None,
                 prodigy_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 use_layerwise_embedding=False,
                 use_ada_embedding=False,
                 composition_regs_iter_gap=3,
                 static_embedding_reg_weight=0.,
                 ada_embedding_reg_weight=0.,
                 prompt_emb_delta_reg_weight=0.,
                 padding_embs_align_loss_weight=0.,
                 subj_comp_key_ortho_loss_weight=0.,
                 subj_comp_value_ortho_loss_weight=0.,
                 mix_prompt_distill_weight=0.,
                 comp_fg_bg_preserve_loss_weight=0.,
                 fg_bg_complementary_loss_weight=0.,
                 fg_wds_complementary_loss_weight=0.,
                 fg_bg_xlayer_consist_loss_weight=0.,
                 fg_bg_token_emb_ortho_loss_weight=0.,
                 compel_cfg_weight_level_range=[2,2],
                 apply_compel_cfg_prob=0.,
                 wds_bg_recon_discount=1.,
                 do_clip_teacher_filtering=True,
                 num_candidate_teachers=4,
                 use_background_token=True,
                 # 'face portrait' is only valid for humans/animals. 
                 # On objects, use_fp_trick will be ignored, even if it's set to True.
                 use_fp_trick=True,
                 prompt_embedding_clamp_value=-1,
                 normalize_ca_q_and_outfeat=True,
                 do_zero_shot=False,
                 same_subject_in_each_batch=False,
                 arc2face_distill_iter_prob=0.5,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings

        self.use_layerwise_embedding = use_layerwise_embedding
        self.N_CA_LAYERS = 16 if self.use_layerwise_embedding else 1

        self.use_ada_embedding = (use_layerwise_embedding and use_ada_embedding)

        self.static_embedding_reg_weight = static_embedding_reg_weight
        self.ada_embedding_reg_weight    = ada_embedding_reg_weight
        self.composition_regs_iter_gap   = composition_regs_iter_gap

        self.prompt_emb_delta_reg_weight        = prompt_emb_delta_reg_weight
        self.padding_embs_align_loss_weight     = padding_embs_align_loss_weight
        self.subj_comp_key_ortho_loss_weight        = subj_comp_key_ortho_loss_weight
        self.subj_comp_value_ortho_loss_weight      = subj_comp_value_ortho_loss_weight
        self.mix_prompt_distill_weight              = mix_prompt_distill_weight
        self.comp_fg_bg_preserve_loss_weight        = comp_fg_bg_preserve_loss_weight
        self.fg_bg_complementary_loss_weight        = fg_bg_complementary_loss_weight
        self.fg_wds_complementary_loss_weight       = fg_wds_complementary_loss_weight
        self.fg_bg_xlayer_consist_loss_weight       = fg_bg_xlayer_consist_loss_weight
        self.fg_bg_token_emb_ortho_loss_weight      = fg_bg_token_emb_ortho_loss_weight
        # type(compel_cfg_weight_level_range) = ListConfig. So convert it to list.
        self.compel_cfg_weight_level_range          = list(compel_cfg_weight_level_range) if compel_cfg_weight_level_range is not None else None
        self.empty_context                          = None
        self.apply_compel_cfg_prob                  = apply_compel_cfg_prob
        self.do_clip_teacher_filtering              = do_clip_teacher_filtering
        self.num_candidate_teachers                 = num_candidate_teachers
        self.prompt_mix_scheme                      = 'mix_hijk'
        self.wds_bg_recon_discount                  = wds_bg_recon_discount
        
        self.use_background_token                   = use_background_token
        self.use_fp_trick                           = use_fp_trick
        self.normalize_ca_q_and_outfeat             = normalize_ca_q_and_outfeat
        self.do_zero_shot                           = do_zero_shot
        self.arc2face_distill_iter_prob             = arc2face_distill_iter_prob if do_zero_shot else 0
        self.same_subject_in_each_batch             = same_subject_in_each_batch
        self.prompt_embedding_clamp_value           = prompt_embedding_clamp_value
        self.comp_init_fg_from_training_image_fresh_count  = 0
        self.comp_init_fg_from_training_image_reuse_count  = 0
        self.face_encoder                           = None
        self.dino_encoder                           = None
        self.zs_image_encoder_instantiated          = False

        self.cached_inits = {}

        # No matter wheter the scheme is layerwise or not,
        # as long as prompt_emb_delta_reg_weight >= 0, do static comp delta reg.
        # If self.prompt_emb_delta_reg_weight == 0, we still do_static_prompt_delta_reg to monitor this loss, 
        # but this loss won't be involved in the optimization.
        self.do_static_prompt_delta_reg = (self.prompt_emb_delta_reg_weight >= 0)
        
        self.init_iteration_flags()
        
        # Is this for DreamBooth training? Will be overwritten in LatentDiffusion ctor.
        self.is_dreambooth                  = False

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        if self.arc2face_distill_iter_prob > 0:
            self.arc2face = Arc2FaceWrapper()

        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.optimizer_type = optimizer_type
        self.adam_config = adam_config
        self.grad_clip = grad_clip
        self.manual_accumulate_grad_batches = manual_accumulate_grad_batches
        self.automatic_optimization = False
    
        if 'Prodigy' in self.optimizer_type:
            self.prodigy_config = prodigy_config

        self.training_percent = 0.
        
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.unfreeze_model = unfreeze_model
        self.model_lr = model_lr

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # create_clip_evaluator() is called in main.py, so that we can specify device as cuda device.
    def create_clip_evaluator(self, device):        
        self.clip_evaluator = CLIPEvaluator(device=device)
        for param in self.clip_evaluator.model.parameters():
            param.requires_grad = False

        self.num_total_teacher_filter_iters = 0
        self.num_teachable_iters = 0
        # A tiny number to avoid division by zero.
        self.num_total_reuse_filter_iters = 0.001
        self.num_reuse_teachable_iters = 0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        if path.endswith(".ckpt"):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
        elif path.endswith(".safetensors"):
            sd = safetensors_load_file(path, device="cpu")
        else:
            print(f"Unknown checkpoint format: {path}")
            sys.exit(1)

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # self.loss_type: default 'l2'.
    def get_loss(self, pred, target, mean=True, loss_type=None):
        if loss_type is None:
            loss_type = self.loss_type

        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_recon': loss.mean().detach()})
        loss_recon = loss.mean()

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb.mean().detach()})

        loss = loss_recon + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss.mean().detach()})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def init_iteration_flags(self):
        self.iter_flags = { 'calc_clip_loss':               False,
                            'do_normal_recon':              True,
                            'do_arc2face_distill':          False,
                            'is_compos_iter':               False,
                            'do_mix_prompt_distillation':   False,
                            'do_static_prompt_delta_reg':   self.do_static_prompt_delta_reg,
                            'do_ada_prompt_delta_reg':      False,
                            # 'do_teacher_filter':        False,
                            # 'is_teachable':             False,
                            'use_background_token':         False,
                            'use_wds_comp':                 False,  
                            'use_wds_cls_captions':         False,
                            'use_fp_trick':                 False,
                            'reuse_init_conds':             False,
                            'comp_init_fg_from_training_image': False,
                          }
        
    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    #LINK #shared_step
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.init_iteration_flags()
        # global_step is updated every self.manual_accumulate_grad_batches iterations.
        # So training_percent is multiplied by self.manual_accumulate_grad_batches.
        self.training_percent = (self.global_step * self.manual_accumulate_grad_batches) / self.trainer.max_steps

        # How many regularizations are done intermittently during the training iterations?
        cand_reg_types = []
        cand_reg_probs = []

        # do_mix_prompt_distillation implies do_ada_prompt_delta_reg.
        # So if do_mix_prompt_distillation is enabled (mix_prompt_distill_weight > 0),
        # then no need to put do_ada_prompt_delta_reg in cand_reg_types.
        # Otherwise, if prompt_emb_delta_reg_weight > 0, we need to put 
        # do_ada_prompt_delta_reg in cand_reg_types.
        if self.mix_prompt_distill_weight > 0:
            cand_reg_types.append('do_mix_prompt_distillation')
            cand_reg_probs.append(1.)
        elif self.prompt_emb_delta_reg_weight > 0:
            cand_reg_types.append('do_ada_prompt_delta_reg')
            cand_reg_probs.append(1.)

        # NOTE: No need to have standalone ada prompt delta reg, 
        # since each prompt mix reg iter will also do ada prompt delta reg.

        N_CAND_REGS = len(cand_reg_types)
        cand_reg_probs = np.array(cand_reg_probs) / np.sum(cand_reg_probs)

        # If N_CAND_REGS == 0, then no intermittent regularizations, set the two flags to False.
        if N_CAND_REGS > 0 and self.composition_regs_iter_gap > 0:
            if self.global_step % self.composition_regs_iter_gap == 0:
                # Alternate among the regularizations in cand_reg_types. 
                # If both do_ada_prompt_delta_reg and do_mix_prompt_distillation,
                # then alternate between do_ada_prompt_delta_reg and do_mix_prompt_distillation.
                # The two regularizations cannot be done in the same batch, as they require
                # different handling of prompts and are calculated by different loss functions.
                # reg_type_idx = (self.global_step // self.composition_regs_iter_gap) % N_CAND_REGS
                reg_type_idx = np.random.choice(N_CAND_REGS, p=cand_reg_probs)
                iter_reg_type     = cand_reg_types[reg_type_idx]
                if iter_reg_type   == 'do_ada_prompt_delta_reg':
                    self.iter_flags['do_mix_prompt_distillation']  = False
                    self.iter_flags['do_ada_prompt_delta_reg'] = True
                # do_mix_prompt_distillation => do_ada_prompt_delta_reg = True.
                elif iter_reg_type == 'do_mix_prompt_distillation':
                    self.iter_flags['do_mix_prompt_distillation']  = True
                    self.iter_flags['do_ada_prompt_delta_reg'] = True

                self.iter_flags['is_compos_iter']   = True
                # Always calculate clip loss during comp reg iterations, even if self.iter_flags['do_teacher_filter'] is False.
                # This is to monitor how well the model performs on compositionality.
                self.iter_flags['calc_clip_loss']   = True
                self.iter_flags['do_normal_recon']  = False

        if self.iter_flags['do_normal_recon'] and self.arc2face_distill_iter_prob > 0:
            if np.random.rand() < self.arc2face_distill_iter_prob:
                self.iter_flags['do_arc2face_distill'] = True
                # Disable do_static_prompt_delta_reg during arc2face distillation.
                self.iter_flags['do_static_prompt_delta_reg'] = False

        if self.is_dreambooth:
            # DreamBooth uses ConcatDataset to make batch a tuple of train_batch and reg_batch.
            # train_batch: normal subject image recon. reg_batch: general class regularization.
            train_batch = batch[0]
            reg_batch   = batch[1]
            loss_train, loss_dict = self.shared_step(train_batch)
            loss_reg, _ = self.shared_step(reg_batch)
            loss = loss_train + self.db_reg_weight * loss_reg
        else:            
            loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        loss = loss / self.manual_accumulate_grad_batches
        self.manual_backward(loss)

        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            main_optimizer   = optimizers[0]
            extra_optimizers = optimizers[1:]
        else:
            # Single optimizer.
            main_optimizer   = optimizers
            extra_optimizers = None

        if (batch_idx + 1) % self.manual_accumulate_grad_batches == 0:
            self.clip_gradients(main_optimizer, gradient_clip_val=self.grad_clip, 
                                gradient_clip_algorithm="norm")
            
            main_optimizer.step()
            # Execute all optimizers.
            if extra_optimizers is not None:
                for optimizer in extra_optimizers:
                    # Fix double counted global step counter which causes premature end of training.
                    # https://github.com/Lightning-AI/pytorch-lightning/issues/17958
                    # The default two functions do self.optim_step_progress.increment_ready()
                    # and self.optim_step_progress.increment_completed(), respectively.
                    # Removing them will prevent optimizer from updating the global step counter.
                    optimizer._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
                    optimizer._on_after_step  = lambda : self.trainer.profiler.stop("optimizer_step")                
                    optimizer.step()

            main_optimizer.zero_grad()

            # Update LRs of optimizers.
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, list):
                for scheduler in schedulers:
                    scheduler.step()
            else:
                scheduler = schedulers
                # Single scheduler.
                scheduler.step()

        lr = main_optimizer.param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

# LatentDiffusion inherits from DDPM. So:
# LatentDiffusion.model = DiffusionWrapper(unet_config, conditioning_key, use_layerwise_embedding)
class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 personalization_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 # To do DreamBooth training, set is_dreambooth=True.
                 is_dreambooth=False,
                 *args, **kwargs):

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # conditioning_key: crossattn
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        # cond_stage_config is a dict:
        # {'target': 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'}
        # Not sure why it's compared with a string
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


        if not self.unfreeze_model:
            # cond_stage_model = FrozenCLIPEmbedder training = False.
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

            # self.model = DiffusionWrapper() training = False.
            self.model.eval()
            self.model.train = disabled_train
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.is_dreambooth = is_dreambooth
        self.db_reg_weight  = 1.
        if not is_dreambooth:
            # self.embedding_manager.training = True after creation. 
            # *** Where this status is set? ***
            self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)
            # embedding_manager.optimized_parameters(): string_to_static_embedder_dict, 
            # which maps custom tokens to embeddings
            #for param in self.embedding_manager.optimized_parameters():
            #    param.requires_grad = True
        else:
            # For DreamBooth.
            self.embedding_manager = None

        self.generation_cache = []
        self.generation_cache_img_colors = []
        self.cache_start_iter = 0
        self.num_cached_generations = 0

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    #@rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.global_step == 0:
            # Make the behavior deterministic for debugging purposes.
            # In normal runs, disable this statement.
            #random.seed(10000)
            self.create_clip_evaluator(next(self.parameters()).device)

            with torch.no_grad():
                # empty_context_tea_filter is only used for clip teacher filtering.
                self.empty_context_tea_filter = self.get_learned_conditioning([""] * self.num_candidate_teachers)
                self.empty_context_2b = self.get_learned_conditioning([""] * 2)
                empty_context_info = self.get_learned_conditioning([""])
                # empty_context: [16, 77, 768] -> [1, 77, 768]
                self.empty_context = empty_context_info[0][[0]]

        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")


    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        # num_timesteps_cond: 1
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        # cond_stage_trainable = True
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
            self.cond_stage_model = model
            
    
    def instantiate_embedding_manager(self, config, text_embedder):
        model = instantiate_from_config(config, text_embedder=text_embedder)

        if config.params.get("embedding_manager_ckpt", None): # do not load if missing OR empty string
            ckpt_params_perturb_ratio = config.params.get("ckpt_params_perturb_ratio", 0)
            src_placeholders = config.params.get("src_placeholders", None)
            loaded_embedder_components   = config.params.get("loaded_embedder_components", None)
            frozen_placeholder_set       = config.params.get("frozen_placeholder_set", None)
            frozen_embedder_components   = config.params.get("frozen_embedder_components", None)
            model.load(config.params.embedding_manager_ckpt, ckpt_params_perturb_ratio,
                       src_placeholders, loaded_embedder_components,
                       frozen_placeholder_set, frozen_embedder_components)
        
        return model

    def instantiate_zero_shot_image_encoders(self, clip_type='openai'):
        if clip_type == 'laion':
            clip_model_tag = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        else:
            clip_model_tag = 'openai/clip-vit-large-patch14'

        self.clip_image_encoder = CLIPVisionModelWithMask.from_pretrained(clip_model_tag)
        self.clip_preprocessor  = CLIPImageProcessor.from_pretrained(clip_model_tag)
        self.clip_image_encoder = self.clip_image_encoder.to(self.device)
        self.clip_image_encoder.eval()
        print(f'{clip_model_tag} image encoder loaded on {self.device}.')

        # BUG: If device='cpu', then it will throw an exception.
        gpu_id = re.findall(r'\d+', str(self.device)).pop()
        gpu_id = int(gpu_id)
        '''
        {'landmark_3d_68': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc190>, 
         'landmark_2d_106': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc2b0>, 
         'detection': <insightface.model_zoo.retinaface.RetinaFace object at 0x7f8e3f0cc100>, 
         'genderage': <insightface.model_zoo.attribute.Attribute object at 0x7f8e3f0cc1f0>, 
         'recognition': <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x7f8e3f0cc0d0>}
        '''
        # Use the same model as Arc2Face.
        self.face_encoder = FaceAnalysis(name='antelopev2', root='arc2face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_encoder.prepare(ctx_id=gpu_id, det_size=(512, 512))

        self.dino_encoder = ViTModel.from_pretrained('facebook/dino-vits16')
        self.dino_encoder = self.dino_encoder.to(self.device)
        self.dino_encoder.eval()
        self.dino_preprocess = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
        print(f'Face and DINO encoders loaded on {self.device}.')

        self.neg_image_features = None
        self.zs_image_encoder_instantiated = True

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    # cond_in: a batch of prompts like ['an illustration of a dirty z', ...]
    def get_learned_conditioning(self, cond_in, zs_clip_features=None, zs_id_embs=None, 
                                 randomize_clip_weights=False, do_arc2face_distill=False, debug_arc2face_embs=False):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                # cond_in: a list of prompts: ['an illustration of a dirty z', 'an illustration of the cool z']
                # each prompt in c is encoded as [1, 77, 768].
                # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
                self.cond_stage_model.device = self.device
                if randomize_clip_weights:
                    self.cond_stage_model.sample_last_layers_skip_weights()

                # If do_zero_shot, but the prompt is empty (uncond prompt), then zs_clip_features is None.
                # We don't update the zs_clip_features in this case.
                if self.do_zero_shot and (zs_clip_features is not None or zs_id_embs is not None):
                    self.embedding_manager.set_zs_image_features(zs_clip_features, zs_id_embs)
                    # When do_zero_shot, never apply compel_cfg (applied in UNet), as compel cfg has been applied in embedding manager.
                    apply_compel_cfg_prob = 0
                else:
                    apply_compel_cfg_prob = self.apply_compel_cfg_prob

                if self.iter_flags['is_compos_iter']:
                    self.embedding_manager.iter_type = 'distill_iter'
                elif do_arc2face_distill:
                    self.embedding_manager.iter_type = 'arc2face_distill_iter'
                else:
                    self.embedding_manager.iter_type = 'recon_iter'

                # static_prompt_embedding: [128, 77, 768]
                static_prompt_embedding = self.cond_stage_model.encode(cond_in, embedding_manager=self.embedding_manager)
                if debug_arc2face_embs:
                    # static_prompt_embedding: [1, 77, 768]. Need to repeat 16 times for 16 layers, and then BS times for BS instances.
                    static_prompt_embedding = self.embedding_manager.arc2face_embs

                # static_prompt_embedding is tensor. So the following statement is False.
                if isinstance(static_prompt_embedding, DiagonalGaussianDistribution):
                    static_prompt_embedding = static_prompt_embedding.mode()

                if not do_arc2face_distill:
                    emb_global_scales_dict  = self.embedding_manager.get_emb_global_scales_dict(regen=True)
                    # Fix the scales of the static subject embeddings.
                    for placeholder, placeholder_indices in self.embedding_manager.placeholder2indices.items():
                        emb_extra_global_scale = emb_global_scales_dict[placeholder]
                        if placeholder in self.embedding_manager.background_string_dict:
                            emb_extra_global_scale *= self.embedding_manager.background_extra_global_scale

                        static_prompt_embedding = fix_emb_scales(static_prompt_embedding, placeholder_indices,
                                                                 num_layers=self.N_CA_LAYERS,
                                                                 extra_scale=emb_extra_global_scale)

                    # It doesn't matter either merge_cls_token_embeddings() first or fix_emb_scales first().
                    # If cls_delta_string_indices is not empty, then it must be a compositional 
                    # distillation iteration, and placeholder_indices only contains the indices of the subject 
                    # instances. Whereas cls_delta_string_indices only contains the indices of the
                    # class (mix) instances. Switching their order doesn't affect the results.
                    static_prompt_embedding = merge_cls_token_embeddings(static_prompt_embedding, 
                                                                         self.embedding_manager.cls_delta_string_indices,
                                                                         self.embedding_manager.subj_name_to_cls_delta_token_weights)
                else:
                    # Repeat the static prompt embeddings 16 times to match the layerwise prompts.
                    static_prompt_embedding = static_prompt_embedding.unsqueeze(1).repeat(1, 16, 1, 1).reshape(-1, 77, 768)

                '''
                'subj_indices':                  filter_dict_by_key(self.embedding_manager.placeholder2indices,
                                                                    self.embedding_manager.subject_strings),
                # copy.copy(self.embedding_manager.placeholder_indices_fg),
                'bg_indices':                   filter_dict_by_key(self.embedding_manager.placeholder2indices,
                                                                    self.embedding_manager.background_strings),
                #copy.copy(self.embedding_manager.placeholder_indices_bg),
                '''

                extra_info = { 
                                'use_layerwise_context':         self.use_layerwise_embedding, 
                                'use_ada_context':               self.use_ada_embedding,
                                'disable_ada_emb':               self.embedding_manager.get_ada_emb_weight(do_perturb=False) == 0,
                                'use_conv_attn_kernel_size':     self.embedding_manager.use_conv_attn_kernel_size,
                                'placeholder2indices':           copy.copy(self.embedding_manager.placeholder2indices),
                                'prompt_emb_mask':               copy.copy(self.embedding_manager.prompt_emb_mask),
                                # subj2conv_attn_layer_scale: a dict of subject strings, whose values are lists of 16 tensor scalars, 
                                # used to scale conv attention at each CA layer.
                                'subj2conv_attn_layer_scale':    self.embedding_manager.get_subj2conv_attn_layer_scale(),
                                'is_training':                   self.embedding_manager.training,
                                'compel_cfg_weight_level_range': self.compel_cfg_weight_level_range,
                                'apply_compel_cfg_prob':         apply_compel_cfg_prob,
                                'empty_context':                 self.empty_context,
                                # Will set to True in p_losses() if in compositional iterations.
                                'capture_distill_attn':          False,
                             }

                if self.use_ada_embedding:
                    ada_embedder = self.get_layer_ada_conditioning
                    # Initialize the ada embedding cache, so that the subsequent calls to 
                    # self.get_layer_ada_embedding() will store the ada embedding 
                    # for each layer into the cache. 
                    # It also updates the EMA of ada embeddings.
                    # Since get_learned_conditioning() is always called before the loss computations,
                    # it won't cause computation graph errors.
                    # This call is redundant, as the cache has already been cleared in p_losses() 
                    # after finishing each iteration.
                    self.embedding_manager.clear_ada_prompt_embeddings_cache()
                    extra_info['ada_embedder'] = ada_embedder

                c = (static_prompt_embedding, cond_in, extra_info)
            else:
                c = self.cond_stage_model(cond_in)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(cond_in)

        return c

    # get_layer_ada_conditioning() is a callback function called iteratively by each layer in UNet.
    # It returns the conditioning embedding (ada embedding & other token embeddings -> clip encoder) 
    # for the current layer to UNet.
    def get_layer_ada_conditioning(self, c_in, layer_idx, layer_attn_components, time_emb):
        # We don't want to mess with the pipeline of cond_stage_model.encode(), so we pass
        # c_in, layer_idx and layer_infeat directly to embedding_manager. They will be used implicitly
        # when embedding_manager is called within cond_stage_model.encode().
        self.embedding_manager.cache_layer_features_for_ada(layer_idx, layer_attn_components, time_emb)
        # DO NOT call sample_last_layers_skip_weights() here, to make the ada embeddings are generated with 
        # CLIP skip weights consistent with the static embeddings.
        ada_prompt_embedding    = self.cond_stage_model.encode(c_in, embedding_manager=self.embedding_manager)
        #print('ada', ada_prompt_embedding.abs().max())
        emb_global_scales_dict  = self.embedding_manager.get_emb_global_scales_dict(regen=False)
        # The scales of ada embeddings are fixed here.
        # The scales of static subject embeddings are fixed in get_learned_conditioning().
        for placeholder, placeholder_indices in self.embedding_manager.placeholder2indices.items():
            emb_extra_global_scale = emb_global_scales_dict[placeholder]
            if placeholder in self.embedding_manager.background_string_dict:
                emb_extra_global_scale *= self.embedding_manager.background_extra_global_scale

            ada_prompt_embedding = fix_emb_scales(ada_prompt_embedding, placeholder_indices,
                                                  extra_scale=emb_extra_global_scale)
            
        # It doesn't matter either merge_cls_token_embeddings() first or fix_emb_scales first().
        # If cls_delta_string_indices is not empty, then it must be a compositional 
        # distillation iteration, and placeholder_indices only contains the indices of the subject 
        # instances. Whereas cls_delta_string_indices only contains the indices of the
        # class (mix) instances. Switching their order doesn't affect the results.
        ada_prompt_embedding = merge_cls_token_embeddings(ada_prompt_embedding, 
                                                          self.embedding_manager.cls_delta_string_indices,
                                                          self.embedding_manager.subj_name_to_cls_delta_token_weights)
                
        ada_subj_attn_dict = self.embedding_manager.get_ada_subj_attn_dict()

        # Cache the computed ada embedding of the current layer for delta loss computation.
        # Before this call, clear_ada_prompt_embeddings_cache() should have been called somewhere.
        self.embedding_manager.cache_ada_prompt_embedding(layer_idx, ada_prompt_embedding)
        return ada_prompt_embedding, self.embedding_manager.get_ada_emb_weight(), ada_subj_attn_dict

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    # k: key for the images, i.e., 'image'. k is not a number.
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        if 'fg_mask' in batch:
            fg_mask = batch['fg_mask']
            fg_mask = fg_mask.unsqueeze(1).to(x.device)
            #fg_mask = F.interpolate(fg_mask, size=x.shape[-2:], mode='nearest')
        else:
            fg_mask = None

        if 'aug_mask' in batch:
            aug_mask = batch['aug_mask']
            aug_mask = aug_mask.unsqueeze(1).to(x.device)
            #img_mask = F.interpolate(img_mask, size=x.shape[-2:], mode='nearest')
        else:
            aug_mask = None

        if fg_mask is not None or aug_mask is not None:
            mask_dict = {'fg_mask': fg_mask, 'aug_mask': aug_mask}
        else:
            mask_dict = None

        encoder_posterior = self.encode_first_stage(x, mask_dict)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # conditioning_key: 'crossattn'.
        if self.model.conditioning_key is not None:
            if cond_key is None:
                # cond_stage_key: 'caption'.
                cond_key = self.cond_stage_key
            # first_stage_key: 'image'.
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    # batch.keys(): 'image', 'caption'.
                    # batch['caption']: 
                    # ['an illustration of a dirty z', 'an illustration of the cool z']
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            # cond_stage_trainable: True. force_c_encode: False.
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            #if bs is not None:
            #    c = c[:bs]
            #if bs is not None and c.shape[0] != bs:
            #    breakpoint()

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    # output: -1 ~ 1.
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as decode_first_stage() but without torch.no_grad() decorator
    # output: -1 ~ 1.
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x, mask=None):
        if hasattr(self, "split_input_params"):     # False
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i], mask)
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x, mask)
        else:
            # Execute this statement only.
            return self.first_stage_model.encode(x, mask)

    # LatentDiffusion.shared_step() overloads DDPM.shared_step().
    # shared_step() is called in training_step() and (no_grad) validation_step().
    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # batch: { 'caption':               ['an illustration of a dirty z',                    
    #                                    'a depiction of a z'], 
    #          'subj_prompt_comp':     ['an illustration of a dirty z dancing with a boy', 
    #                                    'a depiction of a z kicking a punching bag'],
    #          'cls_prompt_single':     ['an illustration of a dirty person',          
    #                                    'a depiction of a person'],
    #                                    'a depiction of a person kicking a punching bag']
    #          'cls_prompt_comp'  :    ['an illustration of a dirty person dancing with a boy', 
    #                                    'a depiction of a person kicking a punching bag'],
    #          'image':   [2, 512, 512, 3] }
    # 'caption' is not named 'subj_prompt_single' to keep it compatible with older code.
    # ANCHOR[id=shared_step]
    def shared_step(self, batch, **kwargs):
        # captions = batch['caption'].
        # Do not use the returned captions from get_input(). Assign the correct caption later.
        # Encode noise as 4-channel latent features. Get prompts from batch. No gradient into here.
        # NOTE: captions (batch['caption'] or batch['caption_bg'])
        # are only for image reconstruction iterations.
        x_start, _ = self.get_input(batch, self.first_stage_key)
        # Update the training_percent of embedding_manager.
        self.embedding_manager.training_percent = self.training_percent

        batch_have_fg_mask  = batch['has_fg_mask']
        # Temporarily disable fg_mask for debugging.
        disable_fg_mask = False #True
        if disable_fg_mask:
            batch_have_fg_mask[:] = False

        self.iter_flags['fg_mask_avail_ratio']  = batch_have_fg_mask.sum() / batch_have_fg_mask.shape[0]

        self.iter_flags['wds_comp_avail_ratio'] = batch['has_wds_comp'].sum() / batch['has_wds_comp'].shape[0]

        # If it's a compositional distillation iteration, only the first instance in the batch is used.
        # Therefore, self.batch_1st_subject_name is the only subject name in the batch.
        self.batch_1st_subject_name  = batch['subject_name'][0]

        # If cached_inits is available (self.batch_1st_subject_name in self.cached_inits), 
        # cached_inits are only used if do_mix_prompt_distillation = True.
        self.iter_flags['reuse_init_conds']  = (self.iter_flags['do_mix_prompt_distillation'] \
                                                and self.batch_1st_subject_name in self.cached_inits)

        # do_teacher_filter: If not reuse_init_conds and do_teacher_filtering, then we choose the better instance 
        # between the two in the batch, if it's above the usable threshold.
        # do_teacher_filter and reuse_init_conds are mutually exclusive.
        self.iter_flags['do_teacher_filter'] = (self.do_clip_teacher_filtering and self.iter_flags['do_mix_prompt_distillation'] \
                                                and not self.iter_flags['reuse_init_conds'])

        # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
        # or traditional TI.        
        # do_ada_prompt_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
        if self.do_static_prompt_delta_reg or self.iter_flags['do_mix_prompt_distillation']:
            # *_fp prompts are like "a face portrait of ...". They are advantageous over "a photo of ..."
            # when doing compositional mix regularization on humans/animals.
            # For objects (broad_class != 1), even if use_fp_trick = True, 
            # *_fp prompts are not available in batch, so fp_trick won't be used.
            # 'subj_prompt_single_fp' in batch is True <=> (broad_class == 1 and use_fp_trick).
            # If do_mix_prompt_distillation but broad_class == 0 or 2, this statement is False, and 
            # used prompts will be 'subj_prompt_comp', 'cls_prompt_single', 'cls_prompt_comp'...
            p_use_fp_trick = 0.9
            self.iter_flags['use_fp_trick'] = self.iter_flags['do_mix_prompt_distillation'] and self.use_fp_trick \
                                                and 'subj_prompt_single_fp' in batch \
                                                and random.random() < p_use_fp_trick

            # Only use_wds_comp iters if all instances have wds_comp image/prompt pairs.
            # all instances have wds_comp <= all instances have fg_mask, i.e., fg_mask_avail_ratio = 1
            if self.iter_flags['wds_comp_avail_ratio'] == 1:
                if self.iter_flags['is_compos_iter']:
                    # 20%  of compositional distillation iters will be initialized with wds_comp 
                    # *background* images (subject not overlaid).
                    # The comp prompts will be updated with wds_comp_extras that correspond to the wds_comp background images.
                    p_use_wds_comp = 0.2
                else:
                    # 5% of recon iters will be initialized with wds_comp overlay images.
                    # If we set p_use_wds_comp >= 0.1, then the subject embeddings will tend to
                    # attend to and reconstruct the overlay background, leading to inferior results.
                    p_use_wds_comp = 0.05
            else:
                p_use_wds_comp = 0
            
            self.iter_flags['use_wds_comp'] = random.random() < p_use_wds_comp

            if self.iter_flags['use_wds_comp']:
                # Replace the image/caption/mask with the wds_comp image/caption/mask.
                # This block of code has to be before "captions = ..." below, 
                # to avoid using wrong captions (some branches use batch['caption_bg']).
                if self.iter_flags['is_compos_iter']:
                    batch['image']      = batch['wds_image_bgonly'] # batch['wds_image']
                    # In compositional distillation iterations, captions are not used, 
                    # so it doesn't really matter which captions are used.
                    batch['caption']    = batch['wds_caption']
                    batch['caption_bg'] = batch['wds_caption_bg']
                    self.iter_flags['use_wds_cls_captions'] = False
                else:
                    batch['image']  = batch['wds_image']
                    # Use wds_cls_caption at decreasing probability over training.
                    # From 0.9 to 0.1 over first 50% of the training, then keep at 0.1.
                    # Using it more frequently may cause double subjects in the image.
                    p_use_wds_cls_captions = anneal_value(self.training_percent, 0.5, (0.6, 0.1))
                    self.iter_flags['use_wds_cls_captions'] = random.random() < p_use_wds_cls_captions
                    if self.iter_flags['use_wds_cls_captions']:
                        batch['caption']    = batch['wds_cls_caption']
                        batch['caption_bg'] = batch['wds_cls_caption_bg']
                    else:
                        batch['caption']    = batch['wds_caption']
                        batch['caption_bg'] = batch['wds_caption_bg']

                batch['aug_mask']   = batch['wds_aug_mask']
                captions            = batch['wds_cls_caption']
                # first_stage_key: 'image'.
                # get_input() uses image, aug_mask and fg_mask.
                x_start, _ = self.get_input(batch, self.first_stage_key)

            # Slightly larger than 0.5, since comp_init_fg_from_training_image is disabled under reuse_init_conds.
            # So in all distillation iterations, comp_init_fg_from_training_image percentage will be around 0.5.
            # NOTE: If use_wds_comp, then to preserve the foreground, we always enable comp_init_fg_from_training_image.
            # But in this case, the background areas will not be replaced with random noises.
            # p_comp_init_fg_from_training_image = 0 if self.iter_flags['use_wds_comp'] else 1.0
            # p_comp_init_fg_from_training_image: 0.8 -> 1.0 over first 25% of the training, 
            # then keep at 1.0.
            # That is, mix_prompt_distill loss is only enabled at the first 25% of the training 
            # as bootstrapping, then disabled (only keep comp_fg_bg_preserve_loss).
            if not self.do_zero_shot:
                p_comp_init_fg_from_training_image = anneal_value(self.training_percent, 0.5, (0.7, 0.9))
            else:
                # If do_zero_shot, then comp_init_fg_from_training_image is always enabled.
                p_comp_init_fg_from_training_image = 1

            # If reuse_init_conds, comp_init_fg_from_training_image may be set to True later
            # if the previous iteration has comp_init_fg_from_training_image = True.
            self.iter_flags['comp_init_fg_from_training_image'] \
                = self.iter_flags['do_mix_prompt_distillation'] \
                    and not self.iter_flags['reuse_init_conds'] \
                    and self.iter_flags['fg_mask_avail_ratio'] > 0 \
                    and random.random() < p_comp_init_fg_from_training_image
            
            # Mainly use background token on recon iters.
            if self.iter_flags['is_compos_iter']:
                p_use_background_token  = 0
            else:
                # Recon iters.
                if self.iter_flags['use_wds_comp']:
                    # At 95% of the time, use background tokens in recon iters if use_wds_comp.
                    p_use_background_token  = 0.95
                elif self.iter_flags['do_arc2face_distill']:
                    # If do_arc2face_distill, then disable the background token.
                    p_use_background_token  = 0
                else:
                    # To avoid the backgound token taking too much of the foreground, 
                    # we only use the background token on 90% of the training images, to 
                    # force the foreground token to focus on the whole image.
                    p_use_background_token  = 0.9

            # Only use_background_token on recon iters.
            # No need to check do_mix_prompt_distillation, because if do_mix_prompt_distillation,
            # in most iterations p_use_background_token = 0, except for 50% of the iterations when
            # comp_init_fg_from_training_image = True.
            self.iter_flags['use_background_token'] = self.use_background_token \
                                                      and random.random() < p_use_background_token
                        
            if self.iter_flags['use_fp_trick'] and self.iter_flags['use_background_token']:
                captions = batch['caption_bg']
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single_fp_bg'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp_fp_bg'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single_fp_bg'
                CLS_PROMPT_COMP    = 'cls_prompt_comp_fp_bg'
            # use_fp_trick but not use_background_token.
            elif self.iter_flags['use_fp_trick']:
                captions = batch['caption']
                # Never use_fp_trick for recon iters. So no need to have "caption_fp" or "caption_fp_bg".
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single_fp'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp_fp'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single_fp'
                CLS_PROMPT_COMP    = 'cls_prompt_comp_fp'
            # not use_fp_trick and use_background_token.
            elif self.iter_flags['use_background_token']:
                captions = batch['caption_bg']
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single_bg'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp_bg'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single_bg'
                CLS_PROMPT_COMP    = 'cls_prompt_comp_bg'
            # Either do_mix_prompt_distillation but not (use_fp_trick_iter and broad_class == 1), 
            # or recon iters (not do_mix_prompt_distillation) and not use_background_token 
            # We don't use_fp_trick on training images. use_fp_trick is only for compositional regularization.
            else:
                # The above captions returned by self.get_input() is already batch['caption'].
                # Reassign here for clarity.
                captions = batch['caption']
                SUBJ_PROMPT_SINGLE = 'subj_prompt_single'
                SUBJ_PROMPT_COMP   = 'subj_prompt_comp'
                CLS_PROMPT_COMP    = 'cls_prompt_comp'
                CLS_PROMPT_SINGLE  = 'cls_prompt_single'

            # Each prompt_comp consists of multiple prompts separated by "|".
            # Split them into a list of subj_comp_prompts/cls_comp_prompts.
            subj_single_prompts = batch[SUBJ_PROMPT_SINGLE]
            cls_single_prompts  = batch[CLS_PROMPT_SINGLE]
            subj_comp_prompts = []
            for prompt_comp in batch[SUBJ_PROMPT_COMP]:
                subj_comp_prompts.append(prompt_comp.split("|"))
            cls_comp_prompts = []
            for prompt_comp in batch[CLS_PROMPT_COMP]:
                cls_comp_prompts.append(prompt_comp.split("|"))
            # REPEATS: how many prompts correspond to each image.
            REPEATS = len(subj_comp_prompts[0])
            if REPEATS == 1 or self.iter_flags['do_mix_prompt_distillation'] or self.iter_flags['do_ada_prompt_delta_reg']:
                # When this iter computes ada prompt delta loss / prompt mixing loss, 
                # only use the first of the composition prompts (in effect num_compositions_per_image=1),
                # otherwise it will use more than 40G RAM.
                subj_comp_prompts = [ prompts[0] for prompts in subj_comp_prompts ]
                cls_comp_prompts  = [ prompts[0] for prompts in cls_comp_prompts ]
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
            else:
                subj_comp_prompts2 = []
                cls_prompt_comp2   = []
                # Suppose R = num_compositions_per_image, and B the batch size.
                # Each of subj_comp_prompts, cls_comp_prompts is like [ (p1_1,..., p1_R), ..., (pB_1,..., pB_R) ].
                # Interlace the list of composition prompt lists into one list:
                # [ p1_1, p2_1, ..., pB_1, p1_2, p2_2, ..., pB_2, ..., p1_R, p2_R, ..., pB_R ].
                # Interlacing makes it easy to choose the first B prompts (just as for a normal batch). 
                # Do not simply concatenate along B.
                for prompts in zip(*subj_comp_prompts):
                    subj_comp_prompts2 += prompts
                for prompts in zip(*cls_comp_prompts):
                    cls_prompt_comp2 += prompts
                subj_comp_prompts = subj_comp_prompts2
                cls_comp_prompts  = cls_prompt_comp2
                captions = captions * REPEATS
                subj_single_prompts = subj_single_prompts * REPEATS
                cls_single_prompts  = cls_single_prompts * REPEATS
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

            if self.iter_flags['use_wds_comp']:
                # TODO: Currently only support REPEATS == 1.
                assert REPEATS == 1
                # wds_comp_extras is a list of wds compositional extra substrings.
                # Never include the class token in wds_comp_extras, i.e. don't use wds_cls_comp_extra.
                # Because the 4-type prompts are for distillation, which don't use the class token.
                wds_comp_extras     = batch["wds_comp_extra"]
                # Replace the compositional extra substrings in the compositional prompts.
                #print(delta_prompts)
                subj_comp_prompts = replace_prompt_comp_extra(subj_comp_prompts, subj_single_prompts, wds_comp_extras)
                cls_comp_prompts  = replace_prompt_comp_extra(cls_comp_prompts,  cls_single_prompts,  wds_comp_extras)
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
                #print(delta_prompts)

        else:
            delta_prompts = None
            # use_background_token is never enabled in this branch.
            if self.iter_flags['use_background_token']:
                captions = batch['caption_bg']

        if 'aug_mask' in batch:
            img_mask = batch['aug_mask']
            # img_mask: [B, H, W] => [B, 1, H, W]
            img_mask = img_mask.unsqueeze(1).to(x_start.device)
            img_mask = F.interpolate(img_mask, size=x_start.shape[-2:], mode='nearest')
        else:
            img_mask = None

        if 'fg_mask' in batch:
            fg_mask = batch['fg_mask']
            # fg_mask: [B, H, W] => [B, 1, H, W]
            fg_mask = fg_mask.unsqueeze(1).to(x_start.device)
            fg_mask = F.interpolate(fg_mask, size=x_start.shape[-2:], mode='nearest')
        else:
            assert self.iter_flags['fg_mask_avail_ratio'] == 0
            fg_mask = None

        if self.same_subject_in_each_batch and len(set(batch['subject_name'])) > 1:
            print("Different subjects in the batch when same_subject_in_each_batch=True.")
            breakpoint()

        if not self.same_subject_in_each_batch and self.iter_flags['do_mix_prompt_distillation']:
            BS = len(batch['subject_name'])
            batch['subject_name'] = batch['subject_name'][0:1] * BS
            x_start  = x_start[0:1].repeat(BS, 1, 1, 1)
            img_mask = img_mask[0:1].repeat(BS, 1, 1, 1) if img_mask is not None else None
            fg_mask  = fg_mask[0:1].repeat(BS, 1, 1, 1)  if fg_mask  is not None else None
            batch_have_fg_mask    = batch_have_fg_mask[0:1].repeat(BS)
            batch["image_unnorm"] = batch["image_unnorm"][0:1].repeat(BS, 1, 1, 1)
            batch["image_path"]   = batch["image_path"][0:1] * BS
            # captions and delta_prompts don't change, as different subjects share the same placeholder "z".
            self.iter_flags['same_subject_in_batch'] = True
        else:
            self.iter_flags['same_subject_in_batch'] = self.same_subject_in_each_batch

        if not self.iter_flags['do_arc2face_distill']:
            self.batch_subject_names = batch['subject_name']
        else:
            self.batch_subject_names = [ "arc2face" ] * len(batch['subject_name'])
        # Currently, only one subject name is allowed in the batch.
        self.embedding_manager.set_curr_batch_subject_names(self.batch_subject_names)
        print(self.batch_subject_names)

        self.iter_flags['is_face'] = [ self.embedding_manager.subj_name_to_being_faces[subject_name] \
                                        for subject_name in self.batch_subject_names ]

        # aug_mask is renamed as img_mask.
        self.iter_flags['img_mask']             = img_mask
        self.iter_flags['fg_mask']              = fg_mask
        self.iter_flags['batch_have_fg_mask']   = batch_have_fg_mask
        self.iter_flags['delta_prompts']        = delta_prompts

        if self.do_zero_shot:
            # images: 0~255 uint8 tensor [3, 512, 512, 3] -> [3, 3, 512, 512].
            images = batch["image_unnorm"].permute(0, 3, 1, 2).to(x_start.device)
            image_paths = batch["image_path"]

            if not self.iter_flags['do_arc2face_distill']:
                # If self.iter_flags['same_subject_in_batch']:  zs_clip_features: [1, 514, 1280]. zs_id_embs: [1, 512].
                # Otherwise:                      zs_clip_features: [3, 514, 1280]. zs_id_embs: [3, 512].
                # If self.iter_flags['same_subject_in_batch'], then we average the zs_clip_features and zs_id_embs to get 
                # less noisy zero-shot embeddings. Otherwise, we use instance-wise zero-shot embeddings.
                zs_clip_features, zs_id_embs = self.encode_zero_shot_image_features(images, fg_mask.squeeze(1),
                                                                                    # iter_flags['is_face'] is a list of 0/1 elements.
                                                                                    is_face=self.iter_flags['is_face'][0],
                                                                                    calc_avg=self.iter_flags['same_subject_in_batch'],
                                                                                    image_paths=image_paths)

            else:
                zs_clip_features = torch.zeros(x_start.shape[0], 514, 1280).to(x_start.device)
                # zs_id_embs: [4, 512]. arc2face_pos_prompt_emb: [4, 21, 768]
                zs_id_embs, arc2face_pos_prompt_emb \
                    = self.arc2face.gen_rand_arc2face_id_prompt_embs(images.shape[0],
                                                                     gen_neg_prompt=False)
                # During training, zs_id_embs, arc2face_pos_prompt_emb are float16, but x_start is float32.
                zs_id_embs = zs_id_embs.to(x_start.dtype)
                arc2face_pos_prompt_emb = arc2face_pos_prompt_emb.to(x_start.dtype)
                # arc2face_neg_prompt_emb = arc2face_neg_prompt_emb.to(x_start.dtype)
                # arc2face_pos_prompt_emb and arc2face_neg_prompt_emb are used to do CFG-style conditioning on
                # self.arc2face to generate the predicted teacher noise.
                self.iter_flags['arc2face_prompt_emb'] = arc2face_pos_prompt_emb
                # In arc2face distillation, we don't need to consider img_mask and fg_mask.
                self.iter_flags['img_mask'] = None
                self.iter_flags['fg_mask']  = None
                self.iter_flags['batch_have_fg_mask'][:] = False
                # Simply denoise a totally random x_start with arc2face_pos_prompt_emb.
                x_start = torch.randn_like(x_start)
                self.iter_flags['is_face'] = [True] * x_start.shape[0]

            self.iter_flags['zs_clip_features'] = zs_clip_features
            self.iter_flags['zs_id_embs']       = zs_id_embs
        else:
            self.iter_flags['zs_clip_features'] = None
            self.iter_flags['zs_id_embs']       = None

        # reuse_init_conds, discard the prompts offered in shared_step().
        if self.iter_flags['reuse_init_conds']:
            cached_inits = self.cached_inits[self.batch_1st_subject_name]
            # cached_inits['delta_prompts'] is a tuple of 4 lists. No need to split them.
            self.iter_flags['delta_prompts']            = cached_inits['delta_prompts']
            self.iter_flags['img_mask']                 = cached_inits['img_mask']
            self.iter_flags['fg_mask']                  = cached_inits['fg_mask']
            self.iter_flags['batch_have_fg_mask']       = cached_inits['batch_have_fg_mask']
            self.iter_flags['filtered_fg_mask']         = cached_inits['filtered_fg_mask']
            self.iter_flags['use_background_token']     = cached_inits['use_background_token']
            self.iter_flags['use_wds_comp']             = cached_inits['use_wds_comp']
            self.iter_flags['comp_init_fg_from_training_image']   = cached_inits['comp_init_fg_from_training_image']

        loss = self(x_start, captions, **kwargs)

        return loss

    # LatentDiffusion.forward() is only called during training, by shared_step().
    #LINK #shared_step
    def forward(self, x_start, captions, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        ORIG_BS  = len(x_start)

        # Use >=, i.e., assign decay in all iterations after the first 100.
        # This is in case there are skips of iterations of global_step 
        # (shouldn't happen but just in case).

        if self.model.conditioning_key is not None:
            assert captions is not None
            # get_learned_conditioning(): convert captions to a [16*B, 77, 768] tensor.
            if self.cond_stage_trainable:
                # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
                # or traditional TI.
                # do_ada_prompt_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
                # captions: plain prompts like ['an illustration of a dirty z', 'an illustration of the cool z']
                
                if self.iter_flags['do_static_prompt_delta_reg'] or self.iter_flags['do_mix_prompt_distillation']:
                    # reuse_init_conds, discard the prompts offered in shared_step().
                    if self.iter_flags['reuse_init_conds']:
                        # cached_inits['delta_prompts'] is a tuple of 4 lists. No need to split them.
                        delta_prompts = self.cached_inits[self.batch_1st_subject_name]['delta_prompts']
                        # cached_inits will be used in p_losses(), 
                        # so don't delete cached_init[self.batch_1st_subject_name] to False yet.
                    else:
                        # iter_flags['delta_prompts'] is a tuple of 4 lists. No need to split them.
                        delta_prompts = self.iter_flags['delta_prompts']

                    subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = delta_prompts
                    #if self.iter_flags['use_background_token']:
                    #print(subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
                    
                    # Recon iters don't do_ada_prompt_delta_reg.
                    if self.iter_flags['do_mix_prompt_distillation'] or self.iter_flags['do_ada_prompt_delta_reg']:                        
                        # For simplicity, BLOCK_SIZE is fixed at 1. So if ORIG_BS == 2, then BLOCK_SIZE = 1.
                        BLOCK_SIZE  = 1
                        # Only keep the first half of batched prompts to save RAM.
                        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                            subj_single_prompts[:BLOCK_SIZE], subj_comp_prompts[:BLOCK_SIZE], \
                            cls_single_prompts[:BLOCK_SIZE],  cls_comp_prompts[:BLOCK_SIZE]
                    else:
                        # Otherwise, do_static_prompt_delta_reg but not do_ada_prompt_delta_reg.
                        # Do not halve the batch. BLOCK_SIZE = ORIG_BS = 12.
                        # 12 prompts will be fed into get_learned_conditioning().
                        BLOCK_SIZE = ORIG_BS
                                                
                    # If not do_ada_prompt_delta_reg, we still compute the static embeddings 
                    # of the 4 types of prompts, to compute static delta loss. 
                    # But now there are 12 prompts (4 * ORIG_BS = 12), as the batch is not halved.
                    delta_prompts = subj_single_prompts + subj_comp_prompts \
                                    + cls_single_prompts + cls_comp_prompts
                    #print(delta_prompts)
                    # breakpoint()
                    # c_static_emb: the static embeddings for static delta loss.
                    # [4 * N_EMBEDS, 77, 768], 4 * N_EMBEDS = 4 * ORIG_BS * N_CA_LAYERS,
                    # whose layer dimension (N_CA_LAYERS) is tucked into the batch dimension. 
                    # delta_prompts: the concatenation of
                    # (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts).
                    # extra_info: a dict that contains extra info.
                    c_static_emb, _, extra_info = \
                        self.get_learned_conditioning(delta_prompts, 
                                                      self.iter_flags['zs_clip_features'],
                                                      self.iter_flags['zs_id_embs'],
                                                      randomize_clip_weights=True,
                                                      do_arc2face_distill=self.iter_flags['do_arc2face_distill'])
                    
                    # Release zs_clip_features and zs_id_embs.
                    del self.iter_flags['zs_clip_features'], self.iter_flags['zs_id_embs']

                    subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb = \
                        c_static_emb.chunk(4)

                    # *_2b: two sub-blocks of the batch (e.g., subj single prompts and subj comp prompts).
                    # *_1b: one sub-block  of the batch (e.g., only subj single prompts).
                    # Only keep the first half (for single prompts), as the second half is the same 
                    # (for comp prompts, differs at the batch index, but the token index is identical).
                    # placeholder_indices_fg is only for (subj_single_prompts, subj_comp_prompts), since
                    # the placeholder token doesn't appear in the class prompts. 
                    # Now we take the first half of placeholder_indices_fg, so that 
                    # they only account for the subject single prompt, but they are also 
                    # applicable to the other 3 types of prompts as they are all aligned 
                    # at the beginning part of the prompts.
                    # halve_token_indices() can take either a tuple or a dict of tuples.

                    placeholder2indices_2b = extra_info['placeholder2indices']
                    placeholder2indices_1b = {}
                    for k in placeholder2indices_2b:
                        placeholder2indices_1b[k] = halve_token_indices(placeholder2indices_2b[k])
                        if placeholder2indices_1b[k] is None:
                            continue

                    # The subject is represented with a multi-embedding token. The corresponding tokens
                    # in the class prompts are "class , , ,", 
                    # therefore the embeddings of "," need to be patched.
                    # BUG: if the batch size of a mix batch > 4, then the ph_indices_1b_N
                    # corresponds to the indices in more than one instance. But distribute_embedding_to_M_tokens()
                    # treat the indices as if they are always in the same instance.
                    # len(ph_indices_1b_N): embedding number of the subject token.
                    cls_single_emb = distribute_embedding_to_M_tokens_by_dict(cls_single_emb,
                                                                              placeholder2indices_1b)
                    cls_comp_emb   = distribute_embedding_to_M_tokens_by_dict(cls_comp_emb,
                                                                              placeholder2indices_1b)
                    
                    extra_info['placeholder2indices_1b'] = placeholder2indices_1b
                    extra_info['placeholder2indices_2b'] = placeholder2indices_2b

                    # These embeddings are patched. So combine them back into c_static_emb.
                    c_static_emb = torch.cat([subj_single_emb, subj_comp_emb, 
                                              cls_single_emb, cls_comp_emb], dim=0)
                    
                    # [64, 77, 768] => [16, 4, 77, 768].
                    extra_info['c_static_emb_4b'] = c_static_emb.reshape(4 * BLOCK_SIZE, self.N_CA_LAYERS, 
                                                                         *c_static_emb.shape[1:])
                    # if do_ada_prompt_delta_reg, then do_mix_prompt_distillation 
                    # may be True or False, depending whether mix reg is enabled.
                    if self.iter_flags['do_mix_prompt_distillation']:
                        # c_in2 = delta_prompts is used to generate ada embeddings.
                        # c_in2: subj_single_prompts + subj_comp_prompts + cls_single_prompts + cls_comp_prompts
                        # The cls_single_prompts/cls_comp_prompts within c_in2 will only be used to 
                        # generate ordinary prompt embeddings, i.e., 
                        # it doesn't contain subject token, and no ada embedding will be injected by embedding manager.
                        # Instead, subj_single_emb, subj_comp_emb and subject ada embeddings 
                        # are manually mixed into their embeddings.
                        c_in2 = delta_prompts
                        extra_info['iter_type']      = self.prompt_mix_scheme   # 'mix_hijk'
                        # The prompts are either (subj single, subj comp, cls single, cls comp) or
                        # (subj comp, subj comp, cls comp, cls comp) if do_teacher_filter. 
                        # So the first 2 sub-blocks always contain the subject/background tokens, and we use *_2b.    
                        extra_info['placeholder2indices'] = extra_info['placeholder2indices_2b'] 

                    # This iter is a simple ada prompt delta loss iter, without prompt mixing loss. 
                    # This branch is reached only if prompt mixing is not enabled.
                    # "and not self.iter_flags['do_mix_prompt_distillation']" is redundant, because it's at an "elif" branch.
                    # Kept for clarity. 
                    elif self.iter_flags['do_ada_prompt_delta_reg'] and not self.iter_flags['do_mix_prompt_distillation']:
                        # c_in2 consists of four types of prompts: 
                        # subj_single, subj_comp, cls_single, cls_comp.
                        c_in2         = delta_prompts
                        # Do ada prompt delta loss in this iteration. 
                        extra_info['iter_type']     = 'do_ada_prompt_delta_reg'
                        # The prompts are either (subj single, subj comp, cls single, cls comp) or
                        # (subj comp, subj comp, cls comp, cls comp) if do_teacher_filter. 
                        # So the first 2 sub-blocks always contain the subject/background tokens, and we use *_2b.
                        extra_info['placeholder2indices'] = extra_info['placeholder2indices_2b']
                    else:
                        # do_normal_recon. The original scheme. 
                        extra_info['iter_type']      = 'normal_recon'
                        # Use the original "captions" prompts and embeddings.
                        # When num_compositions_per_image > 1, subj_single_prompts contains repeated prompts,
                        # so we only keep the first N_EMBEDS embeddings and the first ORIG_BS prompts.
                        c_in2         = captions
                        # In most cases, captions == subj_single_prompts.
                        # They are not the same, if compos_placeholder_prefix is specified,
                        # or if bg/fg overlay is used. In that case, we need to generate 
                        # c_static_emb from captions.
                        if captions == subj_single_prompts:
                            c_static_emb = subj_single_emb
                            # The blocks as input to get_learned_conditioning() are not halved. 
                            # So BLOCK_SIZE = ORIG_BS = 2. Therefore, for the two instances, we use *_1b.
                            extra_info['placeholder2indices'] = extra_info['placeholder2indices_1b']
                        else:
                            # We are unable to reuse the static embeddings of subject single prompt within 
                            # the 4-type prompts, as captions are different. 
                            # So generate embeddings from the captions from scratch.
                            # placeholder2indices in captions should be the same as in subj_single_prompts.
                            # "captions" consist of subject single prompts only (no comp prompts).
                            # Embeddings don't need patching as there are no class prompts.
                            c_static_emb, _, extra_info0 = \
                                self.get_learned_conditioning(captions, 
                                                              self.iter_flags['zs_clip_features'],
                                                              self.iter_flags['zs_id_embs'],
                                                              randomize_clip_weights=True,
                                                              do_arc2face_distill=self.iter_flags['do_arc2face_distill'])
                            # print(captions)
                            extra_info['placeholder2indices'] = extra_info0['placeholder2indices']

                        extra_info['c_static_emb_1b'] = c_static_emb.reshape(ORIG_BS, self.N_CA_LAYERS, 
                                                                             *c_static_emb.shape[1:])
                        ##### End of normal_recon with static delta loss iters. #####

                    extra_info['cls_single_prompts'] = cls_single_prompts
                    extra_info['cls_comp_prompts']   = cls_comp_prompts
                    # 'delta_prompts' is only used in comp_prompt_mix_reg iters. 
                    # Keep extra_info['delta_prompts'] and iter_flags['delta_prompts'] the same structure.
                    # (Both are tuples of 4 lists. But iter_flags['delta_prompts'] may contain more prompts
                    # than those actually used in this iter.)
                    # iter_flags['delta_prompts'] is not used in p_losses(). Keep it for debugging purpose.
                    extra_info['delta_prompts']      = (subj_single_prompts, subj_comp_prompts, \
                                                        cls_single_prompts,  cls_comp_prompts)

                    # c_static_emb is the full set of embeddings of subj_single_prompts, subj_comp_prompts, 
                    # cls_single_prompts, cls_comp_prompts. 
                    # c_static_emb: [64, 77, 768]                    
                    cond = (c_static_emb, c_in2, extra_info)
                else:
                    # Not (self.iter_flags['do_static_prompt_delta_reg'] or 'do_mix_prompt_distillation').
                    # That is, non-compositional iter, or recon iter without static delta loss. 
                    # Keep the tuple cond unchanged. prompts: subject single.
                    # This branch is only reached when do_static_prompt_delta_reg = False.
                    # Either prompt_emb_delta_reg_weight == 0 (ablation only) or 
                    # it's called by self.validation_step().
                    assert self.iter_flags['do_normal_recon']
                    
                    c_static_emb, c_in, extra_info0 = \
                        self.get_learned_conditioning(captions, 
                                                      self.iter_flags['zs_clip_features'],
                                                      self.iter_flags['zs_id_embs'],                                                      
                                                      randomize_clip_weights=True,
                                                      do_arc2face_distill=self.iter_flags['do_arc2face_distill'])
                    
                    extra_info = extra_info0
                    extra_info['placeholder2indices_1b'] = extra_info['placeholder2indices']
                    extra_info['c_static_emb_1b']   = c_static_emb.reshape(ORIG_BS, self.N_CA_LAYERS, 
                                                                           *c_static_emb.shape[1:])
                                        
                    extra_info['iter_type']         = 'normal_recon'

                    cond = (c_static_emb, c_in, extra_info)
                    ##### End of normal_recon without static delta loss iters. #####

            # shorten_cond_schedule: False. Skipped.
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                # q_sample() is only called during training. 
                # q_sample() calls apply_model(), which estimates the latent code of the image.
                cond = self.q_sample(x_start=captions, t=tc, noise=torch.randn_like(captions.float()))

        # self.model (UNetModel) is called in p_losses().
        #LINK #p_losses
        return self.p_losses(x_start, cond, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    # apply_model() is called both during training and inference.
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # has split_input_params: False.
        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            # Only execute this sentence.
            # self.model: DiffusionWrapper -> 
            # self.model.diffusion_model: ldm.modules.diffusionmodules.openaimodel.UNetModel
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    # images: numpy.ndarray or torch.Tensor.
    # images: a list of np array or tensor [3, Hi, Wi] of different sizes. 
    # is_face: whether the images are faces. If True, then face embedding will be extracted. 
    # Otherwise, DINO embedding will be extracted.
    # fg_masks: a list of [Hi, Wi].
    def encode_zero_shot_image_features(self, images, fg_masks, is_face, size=(512, 512), 
                                        calc_avg=False, image_paths=None):
        if not self.zs_image_encoder_instantiated:
            self.instantiate_zero_shot_image_encoders()

        # Must call self.instantiate_zero_shot_image_encoders() before calling this function.
        image_pixel_values = []
        all_id_embs = []
        # images could be a batch of images that have been collated into a tensor or np array.
        # images can also be a list of images.
        # The code below that processes them one by one can be applied in both cases.
        # If images are a collated batch, processing them one by one will not add much overhead.
        for image in images:
            # input to clip_preprocessor: an image or a batch of images, each being PIL.Image.Image, numpy.ndarray, 
            # torch.Tensor, tf.Tensor or jax.ndarray.
            # Different sizes of images are standardized to the same size 224*224.
            clip_image_pixel_values = self.clip_preprocessor(images=image, return_tensors="pt").pixel_values
            image_pixel_values.append(clip_image_pixel_values)

            if is_face and self.face_encoder is not None:
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy().transpose(1, 2, 0)
                # Resize image to (512, 512). The scheme is Image.NEAREST, to be consistent with 
                # PersonalizedBase dataset class.
                image = np.array(Image.fromarray(image).resize(size, Image.NEAREST))

                '''
                face_encoder:
                {'landmark_3d_68': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc190>, 
                 'landmark_2d_106': <insightface.model_zoo.landmark.Landmark object at 0x7f8e3f0cc2b0>, 
                 'detection': <insightface.model_zoo.retinaface.RetinaFace object at 0x7f8e3f0cc100>, 
                 'genderage': <insightface.model_zoo.attribute.Attribute object at 0x7f8e3f0cc1f0>, 
                 'recognition': <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x7f8e3f0cc0d0>}
                '''                    
                face_info = self.face_encoder.get(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if len(face_info) == 0:
                    # If no face is detected (e.g. animals or objects), then use a zero tensor as the face embedding.
                    id_emb = torch.zeros(512, device=self.device)
                else:
                    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
                    # id_emb: [512,]
                    id_emb = torch.from_numpy(face_info.normed_embedding).to(self.device)
                    
                all_id_embs.append(id_emb)

            elif not is_face:
                # DINO embedding.
                dino_input = self.dino_preprocess(images=image, return_tensors="pt")
                dino_input = dino_input.to(self.device)
                # last_hidden_states: [1, 197, 384]
                last_hidden_states = self.dino_encoder(**dino_input).last_hidden_state
                # We only use CLS token's features, so that the spatial location of the subject will not impact matching. 
                # [1, 197, 384] -> [384,]
                id_emb = last_hidden_states[0, 0]
                all_id_embs.append(id_emb)

        # image_pixel_values: [BS, 3, 224, 224]
        image_pixel_values = torch.cat(image_pixel_values, dim=0)
        image_pixel_values = image_pixel_values.to(self.device)
        if self.face_encoder is not None:
            # all_id_embs: [BS, 512] if is_face, or [BS, 384] if not is_face.
            all_id_embs = torch.stack(all_id_embs, dim=0)
        else:
            all_id_embs = None

        if fg_masks is not None:
            assert len(fg_masks) == len(images)
            # fg_masks is a list of masks.
            if isinstance(fg_masks, (list, tuple)):
                fg_masks2 = []
                for fg_mask in fg_masks:
                    # fg_mask: [Hi, Wi]
                    # BUG: clip_preprocessor will do central crop on images. But fg_mask is not central cropped.
                    # If the ref image is not square, then the fg_mask will not match the image.
                    # TODO: crop fg_mask and images to square before calling encode_zero_shot_image_features().
                    # fg_mask2: [Hi, Wi] -> [1, 1, 224, 224]            
                    fg_mask2 = torch.tensor(fg_mask, device=self.device).float().unsqueeze(0).unsqueeze(0)
                    fg_mask2 = F.interpolate(fg_mask2, size=image_pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                    fg_masks2.append(fg_mask2)
                # fg_masks2: [BS, 224, 224]
                fg_masks2 = torch.cat(fg_masks2, dim=0).squeeze(1)
            else:
                # fg_masks is a collated batch of masks.
                # The actual size doesn't matter, 
                # as fg_mask2 will be resized to the same size as image features 
                # (much smaller than image_pixel_values).            
                fg_masks2 = torch.tensor(fg_masks, device=self.device).float().unsqueeze(1)
                fg_masks2 = F.interpolate(fg_masks2, size=image_pixel_values.shape[-2:], mode='bilinear', align_corners=False)
                fg_masks2 = fg_masks2.squeeze(1)
        else:
            # fg_mask2: [BS, 224, 224]. 
            fg_masks2 = torch.ones_like(image_pixel_values[:, 0, :, :], device=self.device)

        with torch.no_grad():
            if self.neg_image_features is None:
                # neg_pixel_values: [1, 3, 224, 224]
                neg_pixel_values = torch.zeros_like(image_pixel_values[:1], device=self.device)
                self.neg_image_features = self.clip_image_encoder(neg_pixel_values, attn_mask=None, output_hidden_states=True).hidden_states[-2]

            # image_fg_features: [BS, 257, 1280]. 257: 16*16 (patch_embeds) + 1 (class_embeds).
            image_fg_dict  = self.clip_image_encoder(image_pixel_values, attn_mask=fg_masks2, output_hidden_states=True)
            # attn_mask: [BS, 1, 257]
            image_fg_features = image_fg_dict.hidden_states[-2] - self.neg_image_features
            if image_fg_dict.attn_mask is not None:
                image_fg_features = image_fg_features * image_fg_dict.attn_mask

            # A negative mask is used to extract the background features.
            image_bg_dict  = self.clip_image_encoder(image_pixel_values, attn_mask=1-fg_masks2, output_hidden_states=True)
            image_bg_features = image_bg_dict.hidden_states[-2] - self.neg_image_features
            if image_bg_dict.attn_mask is not None:
                image_bg_features = image_bg_features * image_bg_dict.attn_mask        

        # clip_features: [BS, 514, 1280].
        # all_id_embs:   [BS, 512].
        clip_features = torch.cat([image_fg_features, image_bg_features], dim=1)

        if calc_avg:
            # clip_features: [BS, 514, 1280] -> [1, 514, 1280].
            # all_id_embs:       [BS, 512]       -> [1, 512].
            clip_features = clip_features.mean(dim=0, keepdim=True)

            debug = False
            if debug and all_id_embs is not None:
                print(image_paths)                    
                calc_stats('all_id_embs', all_id_embs)
                # Compute pairwise similarities of the embeddings.
                all_id_embs = F.normalize(all_id_embs, p=2, dim=1)
                pairwise_sim = torch.matmul(all_id_embs, all_id_embs.t())
                print('pairwise_sim:', pairwise_sim)
                top_dir = os.path.dirname(image_paths[0]) 
                mean_emb_path = os.path.join(top_dir, "mean_emb.pt")
                if os.path.exists(mean_emb_path):
                    mean_emb = torch.load(mean_emb_path)
                    sim_to_mean = torch.matmul(all_id_embs, mean_emb.t())
                    print('sim_to_mean:', sim_to_mean)

            if all_id_embs is not None:
                id_embs = all_id_embs.mean(dim=0, keepdim=True)
                # Without normalization, id_embs.norm(dim=1) is ~0.9. So normalization doesn't have much effect.
                id_embs = F.normalize(id_embs, p=2, dim=-1)
            # id_embs is None only if face_encoder is None, i.e., disabled by the user.
        else:
            # Don't do average of all_id_embs.
            id_embs = all_id_embs
                    
        return clip_features, id_embs

    # emb_man_prompt_adhoc_info: volatile data structures changing along with the prompts or the input images.
    # Sometimes the prompts changed after generating the static embeddings, 
    # so in such cases we need to manually specify these data structures. 
    # If they are not provided (None),
    # then the data structures stored in embedding_manager are not updated.
    # do_pixel_recon: return denoised images for CLIP evaluation. 
    # if do_pixel_recon and cfg_scale > 1, apply classifier-free guidance. 
    # This is not used for the iter_type 'do_normal_recon'.
    # unet_has_grad: when returning do_pixel_recon (e.g. to select the better instance by smaller clip loss), 
    # to speed up, no BP is done on these instances, so unet_has_grad=False.
    def guided_denoise(self, x_start, noise, t, cond, 
                       emb_man_prompt_adhoc_info,
                       unet_has_grad=True, do_pixel_recon=False, cfg_info=None):
        
        self.embedding_manager.set_prompt_adhoc_info(emb_man_prompt_adhoc_info)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # model_output is the predicted noise.
        # if not unet_has_grad, we save RAM by not storing the computation graph.
        if not unet_has_grad:
            with torch.no_grad():
                model_output = self.apply_model(x_noisy, t, cond)
        else:
            # if unet_has_grad, we don't have to take care of embedding_manager.force_grad.
            # Subject embeddings will naturally have gradients.
            model_output = self.apply_model(x_noisy, t, cond)

        # Save ada embeddings generated during apply_model(), to be used in delta loss. 
        # Otherwise it will be overwritten by uncond denoising.
        # ada_prompt_embeddings: [4, 16, 77, 768] or None, if subj token doesn't appear in the prompts.
        ada_prompt_embeddings = self.embedding_manager.get_cached_ada_prompt_embeddings_as_tensor()

        # Get model output of both conditioned and uncond prompts.
        # Unconditional prompts and reconstructed images are never involved in optimization.
        if do_pixel_recon:
            with torch.no_grad():
                x_start_ = x_start.chunk(2)[0]
                noise_   = noise.chunk(2)[0]
                t_       = t.chunk(2)[0]
                # For efficiency, x_start_, noise_ and t_ are of half-batch size,
                # and only compute model_output_uncond on half of the batch, 
                # as the second half (mix single, mix comp) is generated under the same initial conditions 
                # (only differ on prompts, but uncond means no prompts).
                x_noisy_ = self.q_sample(x_start=x_start_, t=t_, noise=noise_)
                # Clear the cached placeholder indices, as they are for conditional embeddings.
                # Now we generate model_output_uncond under unconditional (negative) prompts,
                # which don't contain placeholder tokens.
                self.embedding_manager.clear_prompt_adhoc_info()
                # self.uncond_context: precomputed unconditional embedding and other info.
                # This statement (executed above) disables deep neg prompts on unconditional embeddings. 
                # Otherwise, it will cancel out the effect of unconditional embeddings.
                model_output_uncond = self.apply_model(x_noisy_, t_, cfg_info['uncond_context'])
                # model_output_uncond: [2, 4, 64, 64] -> [4, 4, 64, 64]
                model_output_uncond = model_output_uncond.repeat(2, 1, 1, 1)
            # Classifier-free guidance to make the contents in the 
            # generated images more pronounced => smaller CLIP loss.
            if cfg_info['cfg_scales'] is not None:
                cfg_scales = cfg_info['cfg_scales'].view(-1, 1, 1, 1)
                pred_noise = model_output * cfg_scales - model_output_uncond * (cfg_scales - 1)
            else:
                pred_noise = model_output

            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=pred_noise)
        else:
            x_recon = None
        
        return model_output, x_recon, ada_prompt_embeddings

    # Release part of the computation graph on unused instances to save RAM.
    def release_plosses_intermediates(self, local_vars):
        for k in ('model_output', 'x_recon', 'clip_images_code'):
            if k in local_vars:
                del local_vars[k]

        extra_info = local_vars['extra_info']
        for k in ('ca_layers_activations'):
            if k in extra_info:
                del extra_info[k]

    # t: steps.
    # cond: (c_static_emb, c_in, extra_info). c_in is the textual prompts. 
    # extra_info: a dict that contains 'ada_embedder' and other fields. 
    # ada_embedder: a function to convert c_in to ada embeddings.
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, cond, t, noise=None):
        # If noise is not None, then use the provided noise.
        # Otherwise, generate noise randomly.
        noise = default(noise, lambda: torch.randn_like(x_start))
        #print(cond[1])

        img_mask            = self.iter_flags['img_mask']
        fg_mask             = self.iter_flags['fg_mask']
        batch_have_fg_mask  = self.iter_flags['batch_have_fg_mask']
        filtered_fg_mask    = self.iter_flags.get('filtered_fg_mask', None)

        c_static_emb, c_in, extra_info = cond

        placeholder2indices2  = placeholder2indices   = extra_info['placeholder2indices']
        prompt_emb_mask2      = prompt_emb_mask       = extra_info['prompt_emb_mask']
        cfg_scales_for_clip_loss = None
        uncond_context      = self.empty_context_2b
        all_subj_indices    = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices'],
                                                                   self.embedding_manager.subject_string_dict)
        all_bg_indices      = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices'],
                                                                   self.embedding_manager.background_string_dict)
        all_subj_indices_1b = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_1b'],
                                                                   self.embedding_manager.subject_string_dict)
        all_bg_indices_1b   = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_1b'],
                                                                   self.embedding_manager.background_string_dict)
        if self.iter_flags['is_compos_iter']:
            all_subj_indices_2b = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_2b'],
                                                                       self.embedding_manager.subject_string_dict)
            all_bg_indices_2b   = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_2b'],
                                                                       self.embedding_manager.background_string_dict)
        
        if self.do_zero_shot:
            # If do_zero_shot, then mix more of the subject embeddings into the class embeddings.
            # Otherwise, the class embeddings are too far from subject embeddings (as the init words are only "person"), 
            # posing too strong regularizations to the subject embeddings.
            if self.iter_flags['comp_init_fg_from_training_image']:
                k_cls_scale_layerwise_range = [1.0, 0.8]
                # Less (compared with the settings below) subject embeddings mixed into v.
                v_cls_scale_layerwise_range = [1.0, 0.7]
            else:
                k_cls_scale_layerwise_range = [1.0, 0.8]
                # More subject embeddings mixed into v.
                v_cls_scale_layerwise_range = [1.0, 0.6]
        else:
            if self.iter_flags['comp_init_fg_from_training_image']:
                k_cls_scale_layerwise_range = [1.0, 1.0]
                # Less (compared with the settings below) subject embeddings mixed into v.
                v_cls_scale_layerwise_range = [1.0, 0.85]
            else:
                k_cls_scale_layerwise_range = [1.0, 1.0]
                # More subject embeddings mixed into v.
                v_cls_scale_layerwise_range = [1.0, 0.7]

        if self.iter_flags['is_compos_iter']:
            # For simplicity, always BLOCK_SIZE = 1, no matter the batch size.
            # We can't afford BLOCK_SIZE=2.
            BLOCK_SIZE = 1

            # Only reuse_init_conds if do_mix_prompt_distillation.
            if self.iter_flags['reuse_init_conds']:
                # If self.iter_flags['reuse_init_conds'], we use the cached x_start and cond.
                # cond is already organized as (subj single, subj comp, mix single, mix comp). 
                # No need to manipulate.
                # noise will be kept as the sampled random noise at the beginning of p_losses(). 
                # NOTE: If reuse_init_conds, and the previous iter has comp_init_fg_from_training_image=True, then 
                # the current iter will also have comp_init_fg_from_training_image=True. 
                # But x_start is the denoised result from the previous iteration (noises have been added above), 
                # so we don't add noise to it again.
                # x_start already has a BS of 4. No need to slice or repeat it.
                x_start = self.cached_inits[self.batch_1st_subject_name]['x_start']
                prev_t  = self.cached_inits[self.batch_1st_subject_name]['t']
                # Clear cache of batch_1st_subject_name, to avoid the cached inits being used in the next mix iter.
                del self.cached_inits[self.batch_1st_subject_name]
                # reuse init iter takes a smaller cfg scale, as in the second denoising step, 
                # a particular scale tend to make the cfg-denoised mixed images more dissimilar 
                # to the subject images than in the first denoising step. 

                # x_start is like (s1, s2, c1, c2) (s1, s2 repeated those in the previous distillation iter).
                # s1 is different from s2, but they are generated from the same initial noise in the 
                # previous reconstruction. This is desired as a simulation of a multi-step inference process.
                # Randomly choose t from the middle 400-700 timesteps, 
                # so as to match the once-denoised x_start.
                # generate the full batch size of t, but actually only use the first block of BLOCK_SIZE.
                # This is to make the code consistent with the non-comp case and avoid unnecessary confusion.
                t_mid = torch.randint(int(self.num_timesteps * 0.4), int(self.num_timesteps * 0.7), 
                                      (x_start.shape[0],), device=x_start.device)
                # t_upperbound: old t - 150. That is, at least 150 steps away from the previous t.
                t_upperbound = prev_t - int(self.num_timesteps * 0.15)
                # t should be at least 150 steps away from the previous t, 
                # so that the noise level is sufficiently different.
                t = torch.minimum(t_mid, t_upperbound)
            else:
                # Fresh compositional iter. May do teacher filtering.
                # In a fresh compositional iter, t is set to be at the last 20% of the timesteps.
                # Randomly choose t from the largest 200 timesteps, to match the completely noisy x_start.
                t_tail = torch.randint(int(self.num_timesteps * 0.8), int(self.num_timesteps * 1), 
                                       (x_start.shape[0],), device=x_start.device)
                t = t_tail

                if self.iter_flags['comp_init_fg_from_training_image'] and self.iter_flags['fg_mask_avail_ratio'] > 0:
                    # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
                    # Therefore, using fg_mask for comp_init_fg_from_training_image will force the model remember 
                    # the background in the training images, which is not desirable.
                    # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0.
                    # fg_mask is 4D (added 1D in shared_step()). So expand batch_have_fg_mask to 4D.
                    filtered_fg_mask    = fg_mask.to(x_start.dtype) * batch_have_fg_mask.view(-1, 1, 1, 1)
                    # If use_wds_comp, then don't fill up the background with gaussian noise 
                    # by doing nothing to x_start.
                    if not self.iter_flags['use_wds_comp']:
                        # If do zero-shot, then don't add extra noise to the foreground.
                        fg_noise_anneal_mean_range = (0.1, 0.4) if not self.do_zero_shot else (0.1, 0.4)
                        x_start, fg_mask, filtered_fg_mask = \
                            init_x_with_fg_from_training_image(x_start, fg_mask, filtered_fg_mask, 
                                                               self.training_percent,
                                                               base_scale_range=(0.7, 1.0),
                                                               fg_noise_anneal_mean_range=fg_noise_anneal_mean_range)

                    # Otherwise it's use_wds_comp, then x_start is kept intact (the noisy wds overlay images).

                elif not self.iter_flags['use_wds_comp']:
                    x_start.normal_()

            if not self.iter_flags['do_mix_prompt_distillation']:
                # Only do ada delta loss. This only happens on ablation (mix_prompt_distill_weight = 0).
                # Generate a batch of 4 instances with the same initial x_start, noise and t.
                # This doubles the batch size to 4, if batch size = 2.
                x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)
                noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)
                t       = t[:BLOCK_SIZE].repeat(4)
                # Calculate CLIP score only for image quality evaluation
                cfg_scales_for_clip_loss = torch.ones_like(t) * 5

                # Update masks to be a 4-fold structure.
                img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = \
                    repeat_selected_instances(slice(0, BLOCK_SIZE), 4, img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask)
                self.iter_flags['fg_mask_avail_ratio'] = batch_have_fg_mask.float().mean()

            else:
                # First iteration of a two-iteration do_mix_prompt_distillation.
                # If num_candidate_teachers=4, generate a batch of 8 instances in *two* sets, 
                # each set with *num_candidate_teachers* instances. 
                # We want to select 1 block of BLOCK_SIZE(=1) instances.
                # Within each set, we set static prompt embeddings to be the same,
                # but initial x_start, noise and t are different (x_start and t may have repetitions
                # if ORIG_HALF_BS < num_candidate_teachers).
                # The corresponding instances across the two sets have the same initial 
                # x_start, noise and t, but different prompt embeddings (subj vs. mix).
                # Then do a no_grad generation, find the best teachable set (if any) and pass to 
                # a with_grad generation for distillation loss computation.
                # If no teachable instances are found, skip the with_grad generation and the distillation loss.
                # Note x_start[0] = x_start[2] != x_start[1] = x_start[3].
                # That means, instances are arranged as: 
                # (subj comp 1, ..., subj comp N, mix comp 1, ..., mix comp N).
                # If BS=3, then N = self.num_candidate_teachers = 3 or 6.
                # If batch size = 3 and N = 6, then we quadruple the batch size to 12
                # (6 subj comp, 6 mix comp).
                if self.iter_flags['do_teacher_filter']:
                    NEW_HALF_BS  = self.num_candidate_teachers * BLOCK_SIZE
                    assert NEW_HALF_BS <= x_start.shape[0], \
                        f"NEW_HALF_BS {NEW_HALF_BS} should be no larger than the batch size {x_start.shape[0]}."
                    # Repeat twice to get a FULL BATCH consisting of two sets of instances.
                    x_start = x_start[:NEW_HALF_BS].repeat(2, 1, 1, 1)
                    # noise and t are repeated in the same way as x_start for two sets. 
                    # Set 1 is for two subj comp instances, set 2 is for two mix comp instances.
                    # Noise and t are different between the two instances within one set.
                    noise   = noise[:NEW_HALF_BS].repeat(2, 1, 1, 1)
                    t       = t[:NEW_HALF_BS].repeat(2)

                    # Make two identical sets of c_static_emb2 and c_in2 (first half batch and second half batch).
                    # The two sets are applied on different initial x_start, noise and t (within each half batch).
                    subj_single_emb, subj_comp_emb, mix_single_emb, mix_comp_emb = \
                        c_static_emb.chunk(4)
                    subj_comp_emb = subj_comp_emb[:BLOCK_SIZE * self.N_CA_LAYERS]
                    mix_comp_emb  = mix_comp_emb[:BLOCK_SIZE * self.N_CA_LAYERS]
                    # Only keep *_comp_emb, but repeat them to form 2x or 3x comp sets.
                    # subj_comp_emb, mix_comp_emb: each contains BLOCK_SIZE instances (truncated in forward()). 
                    # So repeat them by num_candidate_teachers times to match the size of x_start.
                    # subj_comp_emb, mix_comp_emb: [16*BLOCK_SIZE, 77, 768] => [16*BLOCK_SIZE*T, 77, 768].
                    # c_static_emb2: [32*BLOCK_SIZE*T, 77, 768].
                    # We don't need to consider NEW_HALF_BS % ORIG_HALF_BS > 0 and truncation, 
                    # since prompts are decoupled with x_start/noise/t and can be simply repeated
                    # by as many times as needed.
                    c_static_emb2 = torch.cat([ subj_comp_emb.repeat(self.num_candidate_teachers, 1, 1), 
                                                 mix_comp_emb.repeat(self.num_candidate_teachers, 1, 1) ], dim=0)
                    
                    subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                        chunk_list(c_in, 4)
                    # We change the prompts to be twin structure: (subj comp 1, subj comp 2, mix comp 1, mix comp 2).
                    # Since subj comp and subj single have the same placeholder_indices,
                    # We don't need to update placeholder2indices of emb_man_prompt_adhoc_info.
                    c_in2 = subj_comp_prompts * self.num_candidate_teachers + cls_comp_prompts * self.num_candidate_teachers
                    # Back up cond as cond_orig. Replace cond with the cond for the twin comp sets.
                    cond_orig = cond
                    cond = (c_static_emb2, c_in2, extra_info)

                    # Instances are arranged as: 
                    # (subj comp 1, ..., subj comp N, mix comp 1, ..., mix comp N).
                    # So we need to repeat all placeholder_indices_1b by N times.
                    # We have to reinitialize placeholder2indices2. It originally points to placeholder2indices. 
                    # Without reinitialization, the following code will rewrite the contents of placeholder2indices.
                    placeholder2indices2 = {}
                    for k in placeholder2indices:
                        placeholder2indices2[k], _ = extend_indices_B_by_n_times(extra_info['placeholder2indices_1b'][k],
                                                                                 self.num_candidate_teachers,
                                                                                 block_offset=BLOCK_SIZE)
                    
                    subj_single_emb_mask, subj_comp_emb_mask, cls_single_emb_mask, cls_comp_emb_mask = \
                        chunk_list(prompt_emb_mask, 4)
                    subj_comp_emb_mask = subj_comp_emb_mask[:BLOCK_SIZE]
                    cls_comp_emb_mask  = cls_comp_emb_mask[:BLOCK_SIZE]
                    # prompt_emb_mask2: [4 or 6, 77, 1]
                    prompt_emb_mask2 = \
                        torch.cat( [subj_comp_emb_mask.repeat(self.num_candidate_teachers, 1, 1),
                                     cls_comp_emb_mask.repeat(self.num_candidate_teachers, 1, 1)], dim=0)

                    uncond_context = self.empty_context_tea_filter
                    # Update masks to be a b-fold * 2 structure.
                    # Before repeating, img_mask, fg_mask, batch_have_fg_mask should all 
                    # have a batch size of 2*BLOCK_SIZE. So repeat_selected_instances() 
                    # won't discard part of them, but simply repeat them twice.
                    img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = \
                        repeat_selected_instances(slice(0, NEW_HALF_BS), 2, img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask)
                    self.iter_flags['fg_mask_avail_ratio'] = batch_have_fg_mask.float().mean()

                    # do_mix_prompt_distillation. We need to compute CLIP scores for teacher filtering.
                    # Set up cfg configs for guidance to denoise images, which are input to CLIP.
                    # Teachers are slightly more aggressive, to increase the teachable fraction.   
                    cfg_scales_for_clip_loss = gen_cfg_scales_for_stu_tea(6, 5, NEW_HALF_BS, x_start.device)

                # Not self.iter_flags['do_teacher_filter']. This branch is do_mix_prompt_distillation.
                # So it's either reuse_init_conds, or not do_clip_teacher_filtering (globally).
                # In any case, we do not need to change the prompts and static embeddings 
                # and simply do mix reg.
                else:
                    if (not self.do_clip_teacher_filtering) and (not self.iter_flags['reuse_init_conds']):
                        # Usually we shouldn't go here, as do_clip_teacher_filtering is always True.
                        x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)

                    # If reuse_init_conds, prev_t is already 1-repeat-4, and 
                    # x_start is denoised from a 1-repeat-4 x_start in the previous iteration 
                    # (precisely speaking, a 1-repeat-2 x_start that's repeated again to 
                    # approximate a 1-repeat-4 x_start).
                    # But noise, t are not 1-repeat-4. 
                    # So we still need to make them 1-repeat-4.
                    noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)
                    t       = t[:BLOCK_SIZE].repeat(4)

                    # Update masks to be a 1-repeat-4 structure.
                    img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = \
                        repeat_selected_instances(slice(0, BLOCK_SIZE), 4, img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask)
                    self.iter_flags['fg_mask_avail_ratio'] = batch_have_fg_mask.float().mean()

                    cfg_scales_for_clip_loss = \
                        gen_cfg_scales_for_stu_tea(6, 5, BLOCK_SIZE * 2, x_start.device)

                    # use cached x_start and cond. cond already has the 4-type structure. 
                    # No change to cond here.
                    # NOTE cond is mainly determined by the prompts c_in. Since c_in is inherited from
                    # the previous iteration, cond is also almost the same.
            
            # This code block is within if self.iter_flags['is_compos_iter'].
            # The prompts are either 2-repeat-2 (do_teacher_filter) or 1-repeat-4 (distillation) structure.
            # Use cond[1] instead of c_static_emb as input, since cond[1] is updated as 2-repeat-2 
            # in the 'do_teacher_filter' branch. We need to do mixing on the c_static_emb 
            # to be used for denoising.
            # In either case, c_static_emb is of (subject embeddings, class embeddings) structure.
            # Therefore, we don't need to deal with the two cases separately.
            # No matter whether t is 2-repeat-2 or 1-repeat-4 structure, 
            # t.chunk(2)[0] always corresponds to the first two blocks of instances.
            t_frac = t.chunk(2)[0] / self.num_timesteps
            # Embedding mixing is always applied to the subject indices in every instance in the 
            # second half-batch (class instances) only, 
            # and the first half-batch (subject instances) is not affected.
            # So providing all_subj_indices_1b[1] is enough. No need to provide batch indices.
            # The prompts are always repetitions like (subj_comp_prompts * T, cls_comp_prompts * T),
            # so subj indices of the second-half batch are the repetitions of the 
            # original extra_info['placeholder2indices_1b'].
            c_static_emb_vk, emb_v_mixer, emb_v_layers_cls_mix_scales, \
                             emb_k_mixer, emb_k_layers_cls_mix_scales = \
                mix_static_vk_embeddings(cond[0], all_subj_indices_1b[1], 
                                         self.training_percent,
                                         t_frac = t_frac, 
                                         use_layerwise_embedding = self.use_layerwise_embedding,
                                         N_CA_LAYERS = self.N_CA_LAYERS,
                                         K_CLS_SCALE_LAYERWISE_RANGE=k_cls_scale_layerwise_range,
                                         V_CLS_SCALE_LAYERWISE_RANGE=v_cls_scale_layerwise_range)
          
            # Update cond[0] to c_static_emb_vk.
            # Use cond[1] instead of c_in as part of the tuple, since c_in is changed in the
            # 'do_teacher_filter' branch.
            cond = (c_static_emb_vk, cond[1], extra_info)
            # emb_k_layers_cls_mix_scales, emb_v_layers_cls_mix_scales: [4, 16]. 
            # Each set of scales (for 16 layers) is for an instance.
            extra_info['emb_k_mixer'], extra_info['emb_k_layers_cls_mix_scales'] = \
                emb_k_mixer, emb_k_layers_cls_mix_scales
            extra_info['emb_v_mixer'], extra_info['emb_v_layers_cls_mix_scales'] = \
                emb_v_mixer, emb_v_layers_cls_mix_scales            
            
        # Otherwise, it's a recon iter (attentional or unweighted).
        else:
            assert self.iter_flags['do_normal_recon']
            BLOCK_SIZE = x_start.shape[0]
            if self.iter_flags['use_wds_comp']:
                # Decrease t slightly to decrease noise amount and preserve more semantics.
                # Do not add extra noise for use_wds_comp instances, since such instances are 
                # kind of "Out-of-Domain" at the background, and are intrinsically difficult to denoise.
                t = anneal_t_keep_prob(t, self.training_percent, self.num_timesteps, ratio_range=(0.8, 1.0),
                                       keep_prob_range=(0.5, 0.3))
            elif not self.iter_flags['do_arc2face_distill']:
                # Increase t slightly by (1, 1.3) to increase noise amount and make the denoising more challenging.
                # This branch doesn't include the 'do_arc2face_distill' iterations.
                t = anneal_t_keep_prob(t, self.training_percent, self.num_timesteps, ratio_range=(1, 1.3), 
                                       keep_prob_range=(0.5, 0.3))

            # No need to update masks.

        extra_info['capture_distill_attn'] = not self.iter_flags['do_teacher_filter']

        # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
        # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
        # (img_mask is not used in the prompt-guided cross-attention layers).
        # Don't consider img_mask in compositional iterations. Because in compositional iterations,
        # the original images don't play a role (even if comp_init_fg_from_training_image,
        # we still don't consider the actual pixels out of the subject areas, so img_mask doesn't matter).
        extra_info['img_mask']  = None if self.iter_flags['is_compos_iter'] else img_mask

        # img_mask is also used when computing Ada embeddings in embedding_manager.
        # So we pass img_mask to embedding_manager here.
        # Do not consider mask on compositional distillation iterations, 
        # as in such iters, the original pixels (out of the fg_mask) do not matter and 
        # can freely compose any contents.
        emb_man_prompt_adhoc_info = { 'placeholder2indices':    placeholder2indices2,
                                      # In compositional iterations, img_mask is always None.
                                      # No need to consider whether do_teacher_filter or not.
                                      'img_mask':         extra_info['img_mask'],
                                      'prompt_emb_mask':  prompt_emb_mask2 }
        
        # cfg_scales: classifier-free guidance scales.
        # By default, 'capture_distill_attn' = False in a generated context, including uncond_context.
        # So we don't need to set it explicitly.
        cfg_info = { 'cfg_scales':     cfg_scales_for_clip_loss,
                     'uncond_context': uncond_context }
        
        model_output, x_recon, ada_embeddings = \
            self.guided_denoise(x_start, noise, t, cond, 
                                emb_man_prompt_adhoc_info=emb_man_prompt_adhoc_info,
                                unet_has_grad=not self.iter_flags['do_teacher_filter'], 
                                # Reconstruct the images at the pixel level for CLIP loss.
                                # do_pixel_recon is not used for the iter_type 'do_normal_recon'.
                                do_pixel_recon=self.iter_flags['calc_clip_loss'],
                                cfg_info=cfg_info)

        extra_info['capture_distill_attn'] = False

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        # default is "eps", i.e., the UNet predicts noise.
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss = 0
                                
        if self.iter_flags['do_normal_recon']:
            if not self.iter_flags['do_arc2face_distill']:
                loss += self.calc_recon_and_complem_losses(model_output, target, extra_info,
                                                           all_subj_indices, all_bg_indices,
                                                           img_mask, fg_mask, batch_have_fg_mask,
                                                           x_start.shape[0], loss_dict, prefix)
            else:
                # Use the predicted noise by the arc2face UNet as the target.
                # target: [4, 4, 64, 64].
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                target = self.arc2face(x_noisy, t, self.iter_flags['arc2face_prompt_emb'],
                                       batch_contains_neg_instances=False, do_cfg=False)
                loss_recon = self.get_loss(model_output, target.to(model_output.dtype), mean=True)
                loss_dict.update({f'{prefix}/loss_recon': loss_recon.detach()})
                loss += loss_recon

        ###### begin of preparation for is_compos_iter ######
        # is_compos_iter <=> calc_clip_loss. But we keep this redundancy for possible flexibility.
        if self.iter_flags['is_compos_iter'] and self.iter_flags['calc_clip_loss']:
            # losses are stored in loss_dict and are not returned.
            clip_images, are_insts_teachable, best_cand_idx, log_image_colors = \
                    self.calc_clip_losses(x_recon, extra_info, loss_dict, prefix)
            
            is_teachable = are_insts_teachable.any()
            self.iter_flags['is_teachable'] = is_teachable

            if self.do_clip_teacher_filtering:
                # Only do distillation if at least one of teacher instances is teachable.
                # do_teacher_filter implies not reuse_init_conds.
                if self.iter_flags['do_teacher_filter'] and self.iter_flags['is_teachable']:
                    # No need the intermediates of the twin-comp instances. Release them to save RAM.
                    # Cannot release the intermediates outside the "if" branch, as the intermediates
                    # will be used in a reuse_init_conds iter.
                    self.release_plosses_intermediates(locals())
                    self.embedding_manager.clear_ada_prompt_embeddings_cache()

                    # Choose the x_start, noise, and t of the better candidate. 
                    # Repeat 4 times and use them as the condition to do denoising again.
                    # x_start[0] == x_start[2], x_start[1] == x_start[3].
                    # BUG: code below assumes BLOCK_SIZE = 1. To be strict, we should choose
                    # the best candidate *block* instead of *instance*. 
                    # But it will make it quite complicated to update the corresponding subject indices.
                    # If BLOCK_SIZE = 1, the subject indices are the same as the original ones,
                    # so we can still use the old all_subj_indices_1b.
                    x_start_sel = x_start[[best_cand_idx]].repeat(4, 1, 1, 1)
                    noise_sel   = noise[[best_cand_idx]].repeat(4, 1, 1, 1)
                    # t[0] == t[1], t[2] == t[3]. t: [952, 851, 952, 851]
                    t_sel       = t[best_cand_idx].repeat(4)
                    # Mark the best candidate with a purple box, although it's in the first iteration.
                    log_image_colors[best_cand_idx] = 3
                    log_image_colors[best_cand_idx + self.num_candidate_teachers] = 3

                    t_frac      = t_sel.chunk(2)[0] / self.num_timesteps
                    # Mix embeddings to get c_static_emb_orig_vk for cond_orig.
                    # Do mixing on saved cond_orig instead of the updated "cond".
                    # cond_orig is the 4-type prompt embeddings (subj single, subj comp, mix single, mix comp).
                    # but cond  has been re-organized as (subj comp, subj comp, mix comp, mix comp). 
                    # So we use cond_orig.
                    # Embedding mixing is always applied to the subject indices in every instance in the 
                    # second half-batch (class instances) only, 
                    # and the first half-batch (subject instances) is not affected.
                    # So providing all_subj_indices_1b[1] is enough. 
                    # No need to repeat it to be the same size as the first half-batch.                     
                    c_static_emb_orig_vk, emb_v_mixer, emb_v_layers_cls_mix_scales, \
                                          emb_k_mixer, emb_k_layers_cls_mix_scales = \
                        mix_static_vk_embeddings(cond_orig[0], all_subj_indices_1b[1], 
                                                 self.training_percent,
                                                 t_frac = t_frac, 
                                                 use_layerwise_embedding = self.use_layerwise_embedding,
                                                 N_CA_LAYERS = self.N_CA_LAYERS,
                                                 K_CLS_SCALE_LAYERWISE_RANGE=k_cls_scale_layerwise_range,
                                                 V_CLS_SCALE_LAYERWISE_RANGE=v_cls_scale_layerwise_range)

                    # emb_k_layers_cls_mix_scales, emb_v_layers_cls_mix_scales: [2, 16]. 
                    # Each set of scales (for 16 layers) is for an instance.
                    # Different from the emb_v_layers_cls_mix_scales above, which is for the twin comp instances.
                    extra_info['emb_k_mixer'], extra_info['emb_k_layers_cls_mix_scales'] = \
                        emb_k_mixer, emb_k_layers_cls_mix_scales
                    extra_info['emb_v_mixer'], extra_info['emb_v_layers_cls_mix_scales'] = \
                        emb_v_mixer, emb_v_layers_cls_mix_scales
                    # Update c_static_emb.
                    cond_orig_qv = (c_static_emb_orig_vk, cond_orig[1], extra_info)

                    # This branch implies a compositional distillation iter.
                    extra_info['img_mask']  = None
                    extra_info['capture_distill_attn'] = True

                    # BUG: code here assumes BLOCK_SIZE = 1. To be strict, we should choose
                    # the best candidate *block* instead of *instance*. 
                    # But it will make it quite complicated to update the corresponding subject indices.
                    # If BLOCK_SIZE = 1, the subject indices are the same as the original ones,
                    # so we can still use the old placeholder2indices_2b.
                    # Otherwise, we need to update placeholder2indices_2b to correspond to the selected
                    # best candidate block.
                    emb_man_prompt_adhoc_info = { 'placeholder2indices':  extra_info['placeholder2indices_2b'],
                                                  'img_mask':             None,
                                                  'prompt_emb_mask':      extra_info['prompt_emb_mask'] }

                    cfg_scales_for_clip_loss = \
                        gen_cfg_scales_for_stu_tea(6, 5, BLOCK_SIZE * 2, x_start.device)

                    cfg_info = { 'cfg_scales':     cfg_scales_for_clip_loss,
                                 'uncond_context': self.empty_context_2b }

                    # unet_has_grad has to be enabled here. Here is the actual place where the computation graph 
                    # on mix reg and ada embeddings is generated for the delta loss. 
                    # (The previous call to guided_denoise() didn't enable gradients, 
                    # as it's only used to filter a teacher.)
                    # If unet_has_grad=False, the gradients of the ada delta loss
                    # couldn't pass through UNet, reducing the performance.
                    # do_pixel_recon=True: return denoised images x_recon. If cfg_scale > 1, 
                    # do classifier-free guidance, so that x_recon are better instances
                    # to be used to initialize the next reuse_init comp iteration.
                    # student prompts are subject prompts.  
                    model_output, x_recon, ada_embeddings = \
                        self.guided_denoise(x_start_sel, noise_sel, t_sel, cond_orig_qv, 
                                            emb_man_prompt_adhoc_info=emb_man_prompt_adhoc_info,
                                            unet_has_grad=True, 
                                            do_pixel_recon=True, cfg_info=cfg_info)

                    # Update masks according to x_start_sel. Select the masks corresponding to 
                    # the better candidate, indexed by [best_cand_idx] (Keep it as a list).
                    img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = \
                        repeat_selected_instances([best_cand_idx], 4, img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask)
                    self.iter_flags['fg_mask_avail_ratio'] = batch_have_fg_mask.float().mean()
                    # Cache x_recon for the next iteration with a smaller t.
                    # Note the 4 types of prompts have to be the same as this iter, 
                    # since this x_recon was denoised under this cond.
                    # Use the subject half of the batch, chunk(2)[0], instead of the mix half, chunk(2)[1], as 
                    # the subject half is better at subject authenticity (but may be worse on composition).
                    x_recon_sel_rep = x_recon.detach().chunk(2)[0].repeat(2, 1, 1, 1)
                    # Release RAM.
                    del model_output, x_recon

                    # We cannot simply use cond_orig[1], as they are (subj single, subj comp, mix single, mix comp).
                    # mix single = class single, but under some settings, maybe mix comp = subj comp.
                    # cached_inits[self.batch_1st_subject_name]['x_start'] has a batch size of 4.
                    # x_recon_sel_rep doesn't have the 1-repeat-4 structure, instead a 
                    # 1-repeat-2 structure that's repeated twice.
                    # But it approximates a 1-repeat-4 structure, so the distillation should still work.
                    # NOTE: no need to update masks to correspond to x_recon_sel_rep, as x_recon_sel_rep
                    # is half-repeat-2 of (the reconstructed images of) x_start_sel. 
                    # Doing half-repeat-2 on masks won't change them, as they are 1-repeat-4.
                    # NOTE: do_teacher_filter implies not reuse_init_conds. So we always need to cache the inits.
                    self.cached_inits[self.batch_1st_subject_name] = \
                        { 'x_start':                x_recon_sel_rep, 
                          'delta_prompts':          cond_orig[2]['delta_prompts'],
                          't':                      t_sel,
                          # reuse_init_conds implies a compositional iter. So img_mask is always None.
                          'img_mask':               None,   
                          'fg_mask':                fg_mask,
                          'batch_have_fg_mask':     batch_have_fg_mask,
                          'filtered_fg_mask':       filtered_fg_mask,
                          'use_background_token':   self.iter_flags['use_background_token'],
                          'use_wds_comp':           self.iter_flags['use_wds_comp'],
                          'comp_init_fg_from_training_image': self.iter_flags['comp_init_fg_from_training_image'],
                        }
                    
                elif not self.iter_flags['is_teachable']:
                    # If not is_teachable, do not do distillation this time 
                    # (since both instances are not good teachers), 
                    # NOTE: thus we can only compute emb reg and delta loss.

                    # In an self.iter_flags['do_teacher_filter'], and not teachable instances are found.
                    # guided_denoise() above is done on twin comp instances.
                    # ada_embeddings = None => loss_ada_delta = 0.
                    if self.iter_flags['do_teacher_filter']:
                        ada_embeddings = None
                    # Otherwise, it's an reuse_init_conds iter, and no teachable instances are found.
                    # We've computed the ada embeddings for the 4-type instances, 
                    # so just use the existing ada_embeddings (but avoid distillation).

                # Otherwise, it's an reuse_init_conds iter, and a teachable instance is found.
                # Do nothing.
                else:
                    pass

            else:
                # Not do_teacher_filter, nor reuse_init_conds. 
                # So only one possibility: not do_clip_teacher_filtering.
                # The teacher instance is always teachable as teacher filtering is disabled.
                self.iter_flags['is_teachable'] = True
                # Since not self.iter_flags['do_teacher_filter']/reuse_init_conds, log_image_colors are all 0,
                # i.e., no box to be drawn on the images in cache_and_log_generations().
                log_image_colors = torch.zeros(clip_images.shape[0], dtype=int, device=x_start.device)

            self.cache_and_log_generations(clip_images, log_image_colors)

        else:
            # Not a compositional iter. Distillation won't be done in this iter, so is_teachable = False.
            self.iter_flags['is_teachable'] = False
        ###### end of preparation for is_compos_iter ######

        # The Prodigy optimizer seems to suppress the embeddings too much, 
        # so we reduce the scale to 0.5 to dampen the embedding reg loss.
        emb_reg_loss_scale          = 0.5 if self.optimizer_type == 'Prodigy' else 1
        prompt_emb_delta_loss_scale = 0.5 if self.optimizer_type == 'Prodigy' else 1

        if self.do_zero_shot:
            # Completely disable the embedding reg loss if do_zero_shot, as it hurts performance.
            emb_reg_loss_scale = 0
            # Reduce the prompt_emb_delta_loss_scale by 5x (from 0.5 to 0.1) if do_zero_shot, 
            # as it might make learning slow.
            prompt_emb_delta_loss_scale /= 5
        
        if self.static_embedding_reg_weight + self.ada_embedding_reg_weight > 0:
            loss_emb_reg = self.embedding_manager.embedding_reg_loss()
            if self.use_layerwise_embedding:
                loss_static_emb_reg, loss_ada_emb_reg = loss_emb_reg
            else:
                loss_static_emb_reg = loss_emb_reg
                loss_ada_emb_reg    = 0

            if loss_static_emb_reg > 0:
                loss_dict.update({f'{prefix}/loss_static_emb_reg': loss_static_emb_reg.mean().detach().item() })
            if loss_ada_emb_reg > 0:
                loss_dict.update({f'{prefix}/loss_ada_emb_reg':    loss_ada_emb_reg.mean().detach().item() })

            ada_embedding_reg_scale = 0.1 if self.do_zero_shot else 1
            loss_emb_reg = loss_static_emb_reg * self.static_embedding_reg_weight \
                            + loss_ada_emb_reg  * self.ada_embedding_reg_weight * ada_embedding_reg_scale

            loss += loss_emb_reg * emb_reg_loss_scale

        # If do_zero_shot but not self.iter_flags['same_subject_in_batch'], then in each batch,
        # the subject instances are from different subjects, and their embeddings are different as well.
        # The fg_bg_token_embs_ortho_loss will be too complicated to compute (we need to separately consider
        # compositional and non-compositional iterations), so we disable it.
        if not (self.do_zero_shot and not self.iter_flags['same_subject_in_batch']) \
          and self.fg_bg_token_emb_ortho_loss_weight >= 0:
            # If use_background_token, then loss_fg_bg_token_emb_ortho is nonzero.
            # Otherwise, loss_fg_bg_token_emb_ortho is zero.
            loss_fg_bg_token_emb_ortho = \
                self.embedding_manager.calc_fg_bg_token_embs_ortho_loss(ada_grad_scale=0.1, fg_grad_scale=0.5)
            if loss_fg_bg_token_emb_ortho > 0:
                loss_dict.update({f'{prefix}/fg_bg_token_emb_ortho': loss_fg_bg_token_emb_ortho.mean().detach().item() })

            # Completely disable the fg_bg_token_emb_ortho_loss if do_zero_shot, as it hurts performance.
            fg_bg_token_emb_ortho_loss_scale = 0 if self.do_zero_shot else 1

            loss += loss_fg_bg_token_emb_ortho * fg_bg_token_emb_ortho_loss_scale * self.fg_bg_token_emb_ortho_loss_weight

        if self.iter_flags['do_static_prompt_delta_reg']:
            # do_ada_prompt_delta_reg controls whether to do ada comp delta reg here.
            loss_static_prompt_delta,  loss_ada_prompt_delta \
                = calc_prompt_emb_delta_loss( 
                        # 'c_static_emb_4b' is the staticc embedding before mixing.
                        # ada_embeddings is the ada embedding after mixing.
                        extra_info['c_static_emb_4b'], ada_embeddings,
                        extra_info['prompt_emb_mask'],
                        self.iter_flags['do_ada_prompt_delta_reg'],
                        self.prompt_embedding_clamp_value
                    )

            # The cached ada prompt embeddings are useless now, release them.
            self.embedding_manager.clear_ada_prompt_embeddings_cache()

            loss_dict.update({f'{prefix}/static_prompt_delta':      loss_static_prompt_delta.mean().detach().item() })
            if loss_ada_prompt_delta != 0:
                loss_dict.update({f'{prefix}/ada_prompt_delta':     loss_ada_prompt_delta.mean().detach().item() })

            # loss_ada_prompt_delta is only applied every composition_regs_iter_gap iterations. 
            # So it should be scaled up proportionally to composition_regs_iter_gap. 
            # Divide it by 2 to reduce the proportion of ada emb loss relative to 
            # loss_static_prompt_delta in the total loss.
            ada_comp_loss_boost_ratio = self.composition_regs_iter_gap / 2
            
            loss += (loss_static_prompt_delta + loss_ada_prompt_delta * ada_comp_loss_boost_ratio) \
                     * self.prompt_emb_delta_reg_weight * prompt_emb_delta_loss_scale
        
            # Even if padding_embs_align_loss_weight is disabled (=0), we still monitor loss_padding_subj_embs_align.
            if self.padding_embs_align_loss_weight >= 0:
                if self.iter_flags['do_normal_recon']:
                    subj_indices        = all_subj_indices_1b
                    bg_indices          = all_bg_indices_1b
                    # c_static_emb_1b is generated from batch['captions']. 
                    # Do not use c_static_emb_4b, which corresponds to the 4-type prompts.
                    static_embeddings   = extra_info['c_static_emb_1b']
                    # SSB_SIZE: subject sub-batch size.
                    # If do_normal_recon, then both instances are subject instances. 
                    # The subject sub-batch size SSB_SIZE = 2 (1 * BLOCK_SIZE).
                    # If is_compos_iter, then subject sub-batch size SSB_SIZE = 2 * BLOCK_SIZE. 
                    # (subj-single and subj-comp instances).
                    # BLOCK_SIZE = 1, SSB_SIZE = 1.
                    SSB_SIZE = BLOCK_SIZE
                else:
                    subj_indices        = all_subj_indices_2b
                    bg_indices          = all_bg_indices_2b
                    # Both static_embeddings and ada_embeddings are 4-type embeddings.
                    static_embeddings   = extra_info['c_static_emb_4b']
                    # BLOCK_SIZE = 1, SSB_SIZE = 2.
                    SSB_SIZE = 2 * BLOCK_SIZE

                loss_padding_subj_embs_align, loss_padding_cls_embs_align, loss_bg_subj_embs_align \
                    = self.calc_padding_embs_align_loss(static_embeddings, ada_embeddings,
                                                        extra_info['prompt_emb_mask'],
                                                        subj_indices, bg_indices, SSB_SIZE, 
                                                        extra_info['iter_type'],
                                                        self.iter_flags['is_compos_iter'],
                                                        emb_clamp_value=self.prompt_embedding_clamp_value)

                if loss_padding_subj_embs_align != 0:
                    loss_dict.update({f'{prefix}/padding_subj_embs_align': loss_padding_subj_embs_align.mean().detach().item() })
                # Monitor loss_padding_cls_embs_align, viewed as a reference to loss_padding_subj_embs_align.
                # We don't optimize loss_padding_cls_embs_align.
                if loss_padding_cls_embs_align != 0:
                    loss_dict.update({f'{prefix}/padding_cls_embs_align': loss_padding_cls_embs_align.mean().detach().item() })
                if loss_bg_subj_embs_align != 0:
                    loss_dict.update({f'{prefix}/bg_subj_embs_align': loss_bg_subj_embs_align.mean().detach().item() })
                
                padding_subj_embs_align_loss_scale = 0.2
                bg_subj_embs_align_loss_scale  = 0.01 if self.do_zero_shot else 0.1
                # NOTE: loss_padding_cls_embs_align is not optimized, but we still add it with scale 0 
                # to release the computation graph.
                loss_padding_embs_align = (loss_padding_subj_embs_align * padding_subj_embs_align_loss_scale \
                                            + loss_padding_cls_embs_align * 0 \
                                            + loss_bg_subj_embs_align   * bg_subj_embs_align_loss_scale)

                loss += loss_padding_embs_align * self.padding_embs_align_loss_weight

        if self.fg_bg_xlayer_consist_loss_weight > 0 \
          and (self.iter_flags['do_normal_recon'] \
                or (self.iter_flags['do_mix_prompt_distillation'] and self.iter_flags['is_teachable'])):
            # SSB_SIZE: subject sub-batch size.
            # If do_normal_recon, then both instances are subject instances. 
            # The subject sub-batch size SSB_SIZE = 2 (1 * BLOCK_SIZE).
            # If is_compos_iter, then subject sub-batch size SSB_SIZE = 2 * BLOCK_SIZE. 
            # (subj-single and subj-comp instances).
            SSB_SIZE = BLOCK_SIZE if self.iter_flags['do_normal_recon'] else 2 * BLOCK_SIZE
            loss_fg_xlayer_consist, loss_bg_xlayer_consist = \
                self.calc_fg_bg_xlayer_consist_loss(extra_info['ca_layers_activations']['attnscore'],
                                                    all_subj_indices,
                                                    all_bg_indices,
                                                    SSB_SIZE)
            if loss_fg_xlayer_consist > 0:
                loss_dict.update({f'{prefix}/fg_xlayer_consist': loss_fg_xlayer_consist.mean().detach().item() })
            if loss_bg_xlayer_consist > 0:
                loss_dict.update({f'{prefix}/bg_xlayer_consist': loss_bg_xlayer_consist.mean().detach().item() })

            # Reduce the loss_fg_xlayer_consist_loss_scale by 5x if do_zero_shot.
            fg_xlayer_consist_loss_scale = 0.2  if self.do_zero_shot else 1
            bg_xlayer_consist_loss_scale = 0.06 if self.do_zero_shot else 0.3

            loss += (loss_fg_xlayer_consist * fg_xlayer_consist_loss_scale + loss_bg_xlayer_consist * bg_xlayer_consist_loss_scale) \
                    * self.fg_bg_xlayer_consist_loss_weight

        if self.iter_flags['do_mix_prompt_distillation'] and self.iter_flags['is_teachable']:
            # ca_outfeats is a dict as: layer_idx -> ca_outfeat. 
            # It contains the 12 specified cross-attention layers of UNet.
            # i.e., layers 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24.
            # Similar are ca_attns and ca_attnscores.
            ca_outfeats  = extra_info['ca_layers_activations']['outfeat']

            # NOTE: loss_comp_fg_bg_preserve is applied only when this 
            # iteration is teachable, because at such iterations the unet gradient is enabled.
            # The current iteration may be a fresh iteration or a reuse_init_conds iteration.
            # In both cases, if comp_init_fg_from_training_image, then we need to preserve the fg/bg areas.
            # Although fg_mask_avail_ratio > 0 when comp_init_fg_from_training_image,
            # fg_mask_avail_ratio may have been updated after doing teacher filtering 
            # (since x_start has been filtered, masks are also filtered accordingly, 
            # and the same as to fg_mask_avail_ratio). So we need to check it here.
            if self.iter_flags['comp_init_fg_from_training_image'] and self.iter_flags['fg_mask_avail_ratio'] > 0 \
              and self.comp_fg_bg_preserve_loss_weight > 0:
                # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
                # Therefore, using fg_mask for comp_init_fg_from_training_image will force the model remember 
                # the background in the training images, which is not desirable.
                # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0, 
                # excluding the instance from the fg_bg_preserve_loss.
                if self.normalize_ca_q_and_outfeat:
                    ca_q_bns = self.embedding_manager.ca_q_bns
                    ca_outfeat_lns = self.embedding_manager.ca_outfeat_lns
                else:
                    ca_q_bns = None
                    ca_outfeat_lns = None

                loss_comp_single_map_align, loss_sc_ss_fg_match, loss_mc_ms_fg_match, \
                loss_sc_mc_bg_match, loss_comp_subj_bg_attn_suppress, loss_comp_mix_bg_attn_suppress \
                 = self.calc_comp_fg_bg_preserve_loss(ca_outfeats, ca_outfeat_lns, 
                                                      extra_info['ca_layers_activations']['q'],
                                                      ca_q_bns,
                                                      extra_info['ca_layers_activations']['attnscore'], 
                                                      filtered_fg_mask, batch_have_fg_mask,
                                                      all_subj_indices_1b, BLOCK_SIZE)
                
                if loss_comp_subj_bg_attn_suppress > 0:
                    loss_dict.update({f'{prefix}/comp_subj_bg_attn_suppress': loss_comp_subj_bg_attn_suppress.mean().detach().item() })
                # comp_mix_bg_attn_suppress is not optimized, and only recorded for monitoring.
                if loss_comp_mix_bg_attn_suppress > 0:
                    loss_dict.update({f'{prefix}/comp_mix_bg_attn_suppress': loss_comp_mix_bg_attn_suppress.mean().detach().item() })
                if loss_comp_single_map_align > 0:
                    loss_dict.update({f'{prefix}/comp_single_map_align': loss_comp_single_map_align.mean().detach().item() })
                if loss_sc_ss_fg_match > 0:
                    loss_dict.update({f'{prefix}/sc_ss_fg_match': loss_sc_ss_fg_match.mean().detach().item() })
                if loss_mc_ms_fg_match > 0:
                    loss_dict.update({f'{prefix}/mc_ms_fg_match': loss_mc_ms_fg_match.mean().detach().item() })
                if loss_sc_mc_bg_match > 0:
                    loss_dict.update({f'{prefix}/sc_mc_bg_match': loss_sc_mc_bg_match.mean().detach().item() })

                elastic_matching_loss_scale = 1
                # loss_comp_single_map_align is L1 loss on attn maps, so its magnitude is small.
                # But this loss is always very small, so no need to scale it up.
                comp_single_map_align_loss_scale = 1
                # mix single - mix comp matching loss is less important, so scale it down.
                ms_mc_fg_match_loss_scale = 0.1
                comp_subj_bg_attn_suppress_loss_scale = 0.02
                sc_mc_bg_match_loss_scale_base = 2
                sc_mc_bg_match_loss_base = 0.2 
                sc_mc_bg_match_loss_scale = calc_dyn_loss_scale(loss_sc_mc_bg_match,
                                                                sc_mc_bg_match_loss_base,
                                                                sc_mc_bg_match_loss_scale_base,
                                                                min_scale_base_ratio=1,
                                                                max_scale_base_ratio=3)
                
                # No need to scale down loss_comp_mix_bg_attn_suppress, as it's on a 0.05-gs'ed attn map.
                loss_comp_fg_bg_preserve = loss_comp_single_map_align * comp_single_map_align_loss_scale \
                                           + (loss_sc_ss_fg_match + loss_mc_ms_fg_match * ms_mc_fg_match_loss_scale
                                                + loss_sc_mc_bg_match * sc_mc_bg_match_loss_scale) \
                                              * elastic_matching_loss_scale \
                                           + (loss_comp_subj_bg_attn_suppress + loss_comp_mix_bg_attn_suppress) \
                                              * comp_subj_bg_attn_suppress_loss_scale
                
                loss_dict.update({f'{prefix}/comp_fg_bg_preserve': loss_comp_fg_bg_preserve.mean().detach().item() })
                # Keep track of the number of iterations that use comp_init_fg_from_training_image.
                if self.iter_flags['reuse_init_conds']:
                    self.comp_init_fg_from_training_image_reuse_count += 1
                else:
                    self.comp_init_fg_from_training_image_fresh_count += 1
                
                comp_init_fg_from_training_image_reuse_frac = self.comp_init_fg_from_training_image_reuse_count / (self.global_step + 1)
                loss_dict.update({f'{prefix}/comp_init_fg_from_training_image_reuse_frac': comp_init_fg_from_training_image_reuse_frac})
                comp_init_fg_from_training_image_fresh_frac = self.comp_init_fg_from_training_image_fresh_count / (self.global_step + 1)
                loss_dict.update({f'{prefix}/comp_init_fg_from_training_image_fresh_frac': comp_init_fg_from_training_image_fresh_frac})
            else:
                loss_comp_fg_bg_preserve = 0

            comp_fg_bg_preserve_loss_scale = 0.5
            # Scale down loss_comp_fg_bg_preserve if reuse_init_conds, as it's more noisy.
            if self.iter_flags['reuse_init_conds']:
                comp_fg_bg_preserve_loss_scale *= 0.5

            loss += loss_comp_fg_bg_preserve * self.comp_fg_bg_preserve_loss_weight \
                    * comp_fg_bg_preserve_loss_scale

            feat_delta_align_scale = 2
            if self.normalize_ca_q_and_outfeat:
                # Normalize ca_outfeat at 50% chance.
                normalize_ca_outfeat = random.random() < 0.5 #draw_annealed_bool(self.training_percent, 0.5, (0.5, 0.5))
            else:
                normalize_ca_outfeat = False

            if normalize_ca_outfeat:
                ca_outfeat_lns = self.embedding_manager.ca_outfeat_lns
                # If using LN, feat delta is around 5x much smaller. So we scale it up to 
                # match the scale of not using LN.
                feat_delta_align_scale *= 5
            else:
                ca_outfeat_lns = None
                
            # loss_comp_fg_bg_preserve should supercede loss_mix_prompt_distill, 
            # as it should be more accurate (?).
            # So if loss_comp_fg_bg_preserve is active, then loss_mix_prompt_distill is halved.
            # all_subj_indices_2b is used in calc_prompt_mix_loss(), as it's used 
            # to index subj single and subj comp embeddings.
            # The indices will be shifted along the batch dimension (size doubled) 
            # within calc_prompt_mix_loss() to index all the 4 blocks.
            loss_feat_delta_align, loss_subj_attn_delta_align, loss_subj_attn_norm_distill \
                = self.calc_prompt_mix_loss(ca_outfeats, ca_outfeat_lns,
                                            extra_info['ca_layers_activations']['attnscore'], 
                                            all_subj_indices_2b, BLOCK_SIZE)

            if loss_feat_delta_align > 0:
                loss_dict.update({f'{prefix}/feat_delta_align':        loss_feat_delta_align.mean().detach().item() })
            if loss_subj_attn_delta_align > 0:
                loss_dict.update({f'{prefix}/subj_attn_delta_align':   loss_subj_attn_delta_align.mean().detach().item() })
            if loss_subj_attn_norm_distill > 0:
                loss_dict.update({f'{prefix}/subj_attn_norm_distill':  loss_subj_attn_norm_distill.mean().detach().item() })

            # loss_subj_attn_delta_align_* use L2 losses, 
            # so no need to use dynamic loss scale.
            subj_attn_delta_align_loss_scale = 0
            # loss_feat_delta_align is around 0.5~1.5. loss_subj_attn_delta_align is around 0.3~0.6.
            # loss_subj_attn_norm_distill is usually 5~10. 
            # So scale it down by 0.2 to match the other two. New range: 1~2.
            subj_attn_norm_distill_loss_scale_base  = 0.2

            # loss_subj_attn_norm_distill is L1 loss, so need to use dynamic loss scale.
            # The scale of subj_attn_norm_distill_loss based on mix_prompt_distill_weight.
            # subj_attn_norm_distill_loss is DISABLED for faces, but enabled for objects.            
            subj_attn_norm_distill_loss_base = 5.
            subj_attn_norm_distill_loss_scale  = calc_dyn_loss_scale(loss_subj_attn_norm_distill,
                                                                     subj_attn_norm_distill_loss_base,
                                                                     subj_attn_norm_distill_loss_scale_base)

            loss_mix_prompt_distill =   loss_subj_attn_delta_align    * subj_attn_delta_align_loss_scale \
                                        + loss_subj_attn_norm_distill * subj_attn_norm_distill_loss_scale \
                                        + loss_feat_delta_align       * feat_delta_align_scale
                                        
            if loss_mix_prompt_distill > 0:
                loss_dict.update({f'{prefix}/mix_prompt_distill':  loss_mix_prompt_distill.mean().detach().item() })

            if loss_comp_fg_bg_preserve == 0:
                mix_prompt_distill_loss_scale = 1
            else:
                # loss_comp_fg_bg_preserve should supercede loss_mix_prompt_distill, 
                # as it should be more accurate (?).
                # So if loss_comp_fg_bg_preserve is active (>0), then loss_mix_prompt_distill 
                # is discounted to half.
                mix_prompt_distill_loss_scale = 0.5

            # mix_prompt_distill_weight: 1e-4.
            loss += loss_mix_prompt_distill * mix_prompt_distill_loss_scale \
                    * self.mix_prompt_distill_weight

        # Only compute loss_subj_comp_key_ortho/loss_subj_comp_value_ortho on compositional iterations.
        # If subj_comp_key_ortho_loss_weight = 0, we still monitor loss_subj_comp_key_ortho 
        # and loss_subj_comp_value_ortho.
        if self.iter_flags['do_mix_prompt_distillation'] and self.iter_flags['is_teachable'] \
          and self.subj_comp_key_ortho_loss_weight >= 0:
            subj_indices_1b = all_subj_indices_1b
            bg_indices_1b   = all_bg_indices_1b if self.iter_flags['use_background_token'] \
                                else None
            # subj_indices_4b: Extend subj_indices_1b to subj_indices_4b, by adding offset to batch indices.
            # bg_indices_4b:   Extend bg_indices_1b   to bg_indices_4b,   by adding offset to batch indices.
            subj_indices_4b, subj_indices_4b_by_block = \
                extend_indices_B_by_n_times(subj_indices_1b, n=4, block_offset=BLOCK_SIZE)
            bg_indices_4b,   bg_indices_4b_by_block   = \
                extend_indices_B_by_n_times(bg_indices_1b,   n=4, block_offset=BLOCK_SIZE)

            comp_extra_indices_4b_by_block = \
                gen_comp_extra_indices_by_block(extra_info['prompt_emb_mask'],
                                                (subj_indices_4b, bg_indices_4b), block_size=BLOCK_SIZE)

            loss_subj_comp_key_ortho,  loss_subj_comp_value_ortho \
                = self.calc_subj_comp_ortho_loss(extra_info['ca_layers_activations']['k'], 
                                                    extra_info['ca_layers_activations']['v'], 
                                                    extra_info['ca_layers_activations']['attnscore'], 
                                                    subj_indices_4b_by_block, comp_extra_indices_4b_by_block,
                                                    cls_grad_scale=0.05)

            if loss_subj_comp_key_ortho != 0:
                loss_dict.update({f'{prefix}/subj_comp_key_ortho':   loss_subj_comp_key_ortho.mean().detach().item() })
            if loss_subj_comp_value_ortho != 0:
                loss_dict.update({f'{prefix}/subj_comp_value_ortho': loss_subj_comp_value_ortho.mean().detach().item() })
            # ortho losses are less effecive, so scale them down.
            key_ortho_loss_scale_base = 1
            key_ortho_loss_base       = 0.1
            key_ortho_loss_scale = calc_dyn_loss_scale(loss_subj_comp_key_ortho, key_ortho_loss_base,
                                                       key_ortho_loss_scale_base)
            
            # subj_comp_key_ortho_loss_weight:          2e-4, 
            # subj_comp_value_ortho_loss_weight:        0, disabled.
            loss +=   (loss_subj_comp_key_ortho   * key_ortho_loss_scale) * self.subj_comp_key_ortho_loss_weight \
                    + (loss_subj_comp_value_ortho * key_ortho_loss_scale) * self.subj_comp_value_ortho_loss_weight

        if torch.isnan(loss):
            print('NaN loss detected.')
            breakpoint()

        self.release_plosses_intermediates(locals())
        loss_dict.update({f'{prefix}/loss': loss.mean().detach().item() })

        return loss, loss_dict

    # Major losses for normal_recon iterations (loss_recon, loss_fg_bg_complementary, etc.).
    # (But there are still other losses used after calling this function.)
    def calc_recon_and_complem_losses(self, model_output, target, extra_info,
                                      all_subj_indices, all_bg_indices,
                                      img_mask, fg_mask, batch_have_fg_mask, 
                                      BLOCK_SIZE, loss_dict, prefix):
        loss = 0

        if self.fg_bg_complementary_loss_weight > 0:
            # NOTE: Do not check iter_flags['use_background_token'] here. If use_background_token, 
            # then loss_fg_bg_complementary, loss_bg_mf_suppress, loss_fg_bg_mask_contrast 
            # will be nonzero. Otherwise, they are zero, and only loss_subj_mb_suppress is computed.
            # all_subj_indices and all_bg_indices are used, instead of *_1b.
            # But only the indices to the first block are extracted in calc_fg_bg_complementary_loss().
            # do_sqrt_norm=False: we only care about the sum of fg attn scores vs. bg attn scores. 
            # So we don't do sqrt norm.
            loss_fg_bg_complementary, loss_subj_mb_suppress, loss_bg_mf_suppress, loss_fg_bg_mask_contrast = \
                        self.calc_fg_bg_complementary_loss(extra_info['ca_layers_activations']['attnscore'],
                                                            all_subj_indices,
                                                            all_bg_indices,
                                                            BLOCK_SIZE=BLOCK_SIZE,
                                                            fg_grad_scale=0.1,
                                                            fg_mask=fg_mask,
                                                            instance_mask=batch_have_fg_mask,
                                                            do_sqrt_norm=False
                                                            )

            if loss_fg_bg_complementary > 0:
                loss_dict.update({f'{prefix}/fg_bg_complem': loss_fg_bg_complementary.mean().detach()})
            # If fg_mask is None, then loss_subj_mb_suppress = loss_bg_mf_suppress = 0.
            if loss_subj_mb_suppress > 0:
                loss_dict.update({f'{prefix}/subj_mb_suppress': loss_subj_mb_suppress.mean().detach()})
            if loss_bg_mf_suppress > 0:
                loss_dict.update({f'{prefix}/bg_mf_suppress': loss_bg_mf_suppress.mean().detach()})
            if loss_fg_bg_mask_contrast > 0:
                loss_dict.update({f'{prefix}/fg_bg_mask_contrast': loss_fg_bg_mask_contrast.mean().detach()})

            # Reduce the scale of loss_fg_bg_complementary if do_zero_shot, as it hurts performance. 
            loss_fg_bg_complementary_scale = 0.2 if self.do_zero_shot else 1
            loss += (loss_fg_bg_complementary * loss_fg_bg_complementary_scale + loss_subj_mb_suppress \
                        + loss_bg_mf_suppress + loss_fg_bg_mask_contrast) \
                    * self.fg_bg_complementary_loss_weight

        if self.iter_flags['use_wds_comp'] and self.fg_wds_complementary_loss_weight > 0:
            #print(c_in)
            # all_subj_indices and all_bg_indices are used, instead of *_1b.

            # prompt_emb_mask: [2, 77, 1] -> [2, 77].
            comp_extra_mask = extra_info['prompt_emb_mask'].squeeze(-1).clone()
            subj_indices_ext = all_subj_indices

            if self.iter_flags['use_wds_cls_captions']:
                # use_wds_cls_captions: cls token follows the subject tokens, and 
                # precedes wds extra tokens.
                # So we extend the subject indices by 1, to include the cls embedding as part of 
                # the subject embeddings.
                subj_indices_ext = extend_indices_N_by_n_times(subj_indices_ext, n=1)

            # Mask out subject embeddings.
            comp_extra_mask[subj_indices_ext] = 0
            # Mask out background embeddings as well, as we want to encourage the subject embeddings
            # to be complementary to the wds embeddings, without considering the background embeddings.
            if self.iter_flags['use_background_token']:
                comp_extra_mask[all_bg_indices] = 0

            wds_comp_extra_indices = comp_extra_mask.nonzero(as_tuple=True)

            # loss_subj_mb_suppress_wds is the same as above if an instance both use_wds_comp and use_background_token.
            # Otherwise it's different. It's ok if the loss is double-counted sometimes.
            # do_sqrt_norm=True: wds_comp_extra prompts are usually much longer, so we do sqrt norm to scale down 
            # wds_comp_extra attn scores.
            loss_fg_wds_complementary, loss_subj_mb_suppress_wds, loss_wds_mask_align, loss_fg_wds_mask_contrast = \
                        self.calc_fg_bg_complementary_loss(extra_info['ca_layers_activations']['attnscore'],
                                                            subj_indices_ext,
                                                            wds_comp_extra_indices,
                                                            BLOCK_SIZE=BLOCK_SIZE,
                                                            fg_grad_scale=0.1, 
                                                            fg_mask=fg_mask,
                                                            instance_mask=batch_have_fg_mask,
                                                            do_sqrt_norm=True
                                                            )

            if loss_fg_wds_complementary > 0:
                loss_dict.update({f'{prefix}/fg_wds_complem': loss_fg_wds_complementary.mean().detach()})
            # If fg_mask is None, then loss_subj_mb_suppress_wds = loss_wds_mask_align = 0.
            if loss_subj_mb_suppress_wds > 0:
                loss_dict.update({f'{prefix}/subj_mb_suppress_wds': loss_subj_mb_suppress_wds.mean().detach()})
            if loss_wds_mask_align > 0:
                loss_dict.update({f'{prefix}/wds_mask_align': loss_wds_mask_align.mean().detach()})
            if loss_fg_wds_mask_contrast > 0:
                loss_dict.update({f'{prefix}/fg_wds_mask_contrast': loss_fg_wds_mask_contrast.mean().detach()})

            fg_wds_comple_loss_scale    = 1
            wds_mask_align_loss_scale   = 1
            loss += (loss_fg_wds_complementary * fg_wds_comple_loss_scale \
                        + loss_wds_mask_align     * wds_mask_align_loss_scale \
                        + loss_subj_mb_suppress_wds + loss_fg_wds_mask_contrast) \
                    * self.fg_wds_complementary_loss_weight

        if not self.iter_flags['use_background_token'] and not self.iter_flags['use_wds_comp']:
            # bg loss is almost completely ignored. But giving it a little weight may help suppress 
            # subj embeddings' contribution to the background (serving as a contrast to the fg).
            bg_pixel_weight = 0 #0.01
        else:
            if self.iter_flags['use_wds_comp']:
                # fg_pixel_weight/bg_pixel_weight are scalars.
                # an instance of  use_wds_comp: instance_bg_weight is 0.05, instance_fg_weight is 1.
                # an instance not use_wds_comp: instance_bg_weight is 1,    instance_fg_weight is 1.
                # NOTE: We discount the bg weight of the use_wds_comp instances, as bg areas are supposed 
                # to be attended with wds comp extra embeddings. However, the attention may not be perfect,
                # and subject embeddings may be tempted to attend to the background, which will mix the 
                # background features into the subject embeddings, which hurt both authenticity and compositionality.
                ## NOTE: We discount the fg weight of the use_wds_comp instances, as they are less natural and
                ## may incur too high loss to the model (even in the fg areas).
                bg_pixel_weight = self.wds_bg_recon_discount
            else:
                # use_background_token == True and not self.iter_flags['use_wds_comp'].
                # bg loss is somewhat discounted.
                bg_pixel_weight = 0.5
            
        # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
        loss_recon, _ = self.calc_recon_loss(model_output, target, img_mask, fg_mask, 
                                             fg_pixel_weight=1,
                                             bg_pixel_weight=bg_pixel_weight)

        loss_dict.update({f'{prefix}/loss_recon': loss_recon.detach()})
        loss += loss_recon

        return loss

    # pixel-wise recon loss, weighted by fg_pixel_weight and bg_pixel_weight separately.
    # fg_pixel_weight, bg_pixel_weight: could be 1D tensors of batch size, or scalars.
    # img_mask, fg_mask:    [BS, 1, 64, 64] or None.
    # model_output, target: [BS, 4, 64, 64].
    def calc_recon_loss(self, model_output, target, img_mask, fg_mask, 
                        fg_pixel_weight=1, bg_pixel_weight=1):

        if img_mask is None:
            img_mask = torch.ones_like(model_output)
        if fg_mask is None:
            fg_mask = torch.ones_like(model_output)
        
        # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
        model_output = model_output * img_mask
        target       = target       * img_mask
        loss_recon_pixels = self.get_loss(model_output, target, mean=False)

        # fg_mask,              weighted_fg_mask.sum(): 1747, 1747
        # bg_mask=(1-fg_mask),  weighted_fg_mask.sum(): 6445, 887
        weighted_fg_mask = fg_mask       * img_mask * fg_pixel_weight
        weighted_bg_mask = (1 - fg_mask) * img_mask * bg_pixel_weight
        weighted_fg_mask = weighted_fg_mask.expand_as(loss_recon_pixels)
        weighted_bg_mask = weighted_bg_mask.expand_as(loss_recon_pixels)

        loss_recon = (  (loss_recon_pixels * weighted_fg_mask).sum()     \
                      + (loss_recon_pixels * weighted_bg_mask).sum() )   \
                     / (weighted_fg_mask.sum() + weighted_bg_mask.sum())

        return loss_recon, loss_recon_pixels
    
    def calc_clip_losses(self, x_recon, extra_info, loss_dict, prefix):
        # Images generated both under subj_comp_prompts and cls_comp_prompts 
        # are subject to the CLIP text-image matching evaluation.
        # If self.iter_flags['do_teacher_filter'] (implying do_mix_prompt_distillation), 
        # the batch is (subj_comp_emb, subj_comp_emb, mix_comp_emb,  mix_comp_emb).
        # So cls_comp_prompts is used to compute the CLIP text-image matching loss on
        # images guided by the subject or mixed embeddings.
        if self.iter_flags['do_teacher_filter']:
            #del extra_info['ca_layers_activations']
            clip_images_code  = x_recon
            # 4 sets of cls_comp_prompts for (subj comp 1, subj comp 2, mix comp 1, mix comp 2).                
            clip_prompts_comp = extra_info['cls_comp_prompts'] * self.num_candidate_teachers * 2
        else:
            # Either self.iter_flags['reuse_init_conds'], or a pure ada delta loss iter.
            # A batch of either type has the (subj_single, subj_comp, mix_single, mix_comp) structure.
            # Only evaluate the CLIP loss on the comp images. 
            # So the batch size of clip_images_code is halved.
            # If teachable, the whole batch of x_recon is still used for distillation.
            x_recon_subj_single, x_recon_subj_comp, x_recon_mix_single, x_recon_mix_comp = \
                x_recon.chunk(4)
            clip_images_code = torch.cat([x_recon_subj_comp, x_recon_mix_comp], dim=0)
            clip_prompts_comp = extra_info['cls_comp_prompts'] * 2

        # Use CLIP loss as a metric to evaluate the compositionality of the generated images 
        # and do distillation selectively.
        # DO NOT use CLIP loss to optimize the model. It will hurt the performance.
        with torch.no_grad():
            clip_images = self.decode_first_stage(clip_images_code)
            losses_clip_comp   = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_comp,   
                                                                                 clip_images,  
                                                                                 reduction='diag')

        # Instances are arranged as: 
        # (subj comp 1, subj comp 2, mix comp 1, mix comp 2).
        losses_clip_subj_comp, losses_clip_mix_comp \
            = losses_clip_comp.chunk(2)

        loss_diffs_subj_mix = losses_clip_subj_comp - losses_clip_mix_comp

        if not self.iter_flags['reuse_init_conds']:
            loss_dict.update({f'{prefix}/loss_clip_subj_comp': losses_clip_subj_comp.mean().detach().item() })
            loss_dict.update({f'{prefix}/loss_clip_cls_comp':  losses_clip_mix_comp.mean().detach().item() })
        else:
            loss_dict.update({f'{prefix}/reuse_loss_clip_subj_comp': losses_clip_subj_comp.mean().detach().item() })
            loss_dict.update({f'{prefix}/reuse_loss_clip_cls_comp':  losses_clip_mix_comp.mean().detach().item() })

        # If reuse_init_conds, we still check whether the instances are teachable.
        # But it's not called teacher filtering, as there's only one teacher. If it's not teachable,
        # then we skip the distillation loss.
        if self.iter_flags['do_teacher_filter'] or self.iter_flags['reuse_init_conds']:
            # Discard instances that seem to be too far from the text 
            # (it may be a bit premature to make this decision now, as the images are only denoised once).
            # 0.35/0.006: 30%-40% instances will meet these thresholds.
            # 0.33/0.008: 15% instances will meet these thresholds.
            clip_loss_thres         = 0.28
            cls_subj_clip_margin    = 0.002

            # are_insts_teachable: The teacher instances are only teachable if both 
            # the teacher and student are qualified (<= clip_loss_thres), 
            # and the compositional clip loss is smaller than the student.
            # If the student is qualified (<= clip_loss_thres), and the 
            # teacher is teachable, then the teacher is also qualified, 
            # as it has to have a smaller loss to be teachable. 
            # So only need to check losses_clip_subj_comp against clip_loss_thres.
            # Old version: are_insts_teachable = (losses_clip_subj_comp <= clip_loss_thres) & (loss_diffs_subj_mix > cls_subj_clip_margin)
            # This is unreasonable as we shouldn't filter on losses_clip_subj_comp. On the contrary,
            # the worse the student is, the more benefit it will get from the teacher.
            are_insts_teachable = (losses_clip_mix_comp <= clip_loss_thres) & (loss_diffs_subj_mix > cls_subj_clip_margin)
            # print(losses_clip_subj_comp, losses_clip_mix_comp)
            # If any of the two instances is teachable, we consider it as a teachable iteration,
            # and select the better one (with larger loss diff) to teach.
            self.iter_flags['is_teachable']  = are_insts_teachable.sum() > 0

            # best_cand_idx: max of diffs among teachable instances.
            # text-image better aligned -> smaller clip loss.
            # For a 4-instance batch to be teachable, subj comp instances should have larger clip loss
            # than mix comp instances. So loss_diffs_subj_mix should be positive, and the larger, 
            # the more teachable. Therefore we find the argmax of loss_diffs_subj_mix.
            # Sometimes the max among loss_diffs_subj_mix is not teachable (> clip_loss_thres),
            # so we set the loss_diffs_subj_mix of non-teachable instances to be a very large 
            # negative number, so that non-teachable instances will never be selected as the best candidate.
            loss_diffs_subj_mix[~are_insts_teachable] = -1e4
            best_cand_idx = torch.argmax(loss_diffs_subj_mix).item()

            self.num_total_teacher_filter_iters += 1
            self.num_teachable_iters += int(self.iter_flags['is_teachable'])
            teachable_frac = self.num_teachable_iters / self.num_total_teacher_filter_iters
            loss_dict.update({f'{prefix}/teachable_frac': teachable_frac})
            self.num_total_reuse_filter_iters += int(self.iter_flags['reuse_init_conds'])
            self.num_reuse_teachable_iters += int(self.iter_flags['reuse_init_conds'] and self.iter_flags['is_teachable'])
            reuse_teachable_frac = self.num_reuse_teachable_iters / self.num_total_reuse_filter_iters 
            loss_dict.update({f'{prefix}/reuse_teachable_frac': reuse_teachable_frac})

        else:
            are_insts_teachable = torch.ones_like(losses_clip_subj_comp, dtype=torch.bool)
            best_cand_idx = 0

        # Number of filtered pairs is half of the batch size. 
        # Repeat to match the number of instances in the batch.
        # log_image_colors: take values in {0, 1, 2, 3}, 
        # i.e., no box, green, red, purple boxes respectively.
        # 0 no box: not teachable; 
        # 1 green:  teachable in the first iter and not reused yet 
        # (because it's not in the second iter yet).
        # 2 red:    teachable in the first iter but not in the second iter; 
        # 3 purple: teachable in both iters.
        # If self.iter_flags['do_teacher_filter'] and an instance is teachable, then in the log image file, 
        # log_image_flag = 1, so the instance has a green bounary box in the logged image.
        # log_image_colors: 3 * 2. [subj comp * 3, mix comp * 3]
        log_image_colors = are_insts_teachable.repeat(2).int()
        if self.iter_flags['reuse_init_conds']:
            # If reuse_init_conds and an instance is teachable, then in the log image file,
            # log_image_flag = 3, so the instance has a purple bounary box in the logged image.
            log_image_colors += 2

        return clip_images, are_insts_teachable, best_cand_idx, log_image_colors

    def calc_prompt_mix_loss(self, ca_outfeats, ca_outfeat_lns, ca_attnscores, fg_indices_2b, BLOCK_SIZE):
        # do_mix_prompt_distillation iterations. No ordinary image reconstruction loss.
        # Only regularize on intermediate features, i.e., intermediate features generated 
        # under subj_comp_prompts should satisfy the delta loss constraint:
        # F(subj_comp_prompts)  - F(mix(subj_comp_prompts, cls_comp_prompts)) \approx 
        # F(subj_single_prompts) - F(cls_single_prompts)

        # Avoid doing distillation on the first few bottom layers (little difference).
        # distill_layer_weights: relative weight of each distillation layer. 
        # distill_layer_weights are normalized using distill_overall_weight.
        # Most important conditioning layers are 7, 8, 12, 16, 17. All the 5 layers have 1280 channels.
        # But intermediate layers also contribute to distillation. They have small weights.

        # feature map distillation only uses delta loss on the features to reduce the 
        # class polluting the subject features.
        feat_distill_layer_weights = {  7:  0.5, 8: 0.5,   
                                        12: 1.,
                                        16: 1., 17: 1.,
                                        18: 1.,
                                        19: 1., 20: 1., 
                                        21: 1., 22: 1., 
                                        23: 1., 24: 1., 
                                     }

        # attn delta loss is more strict and could cause pollution of subject features with class features.
        # so top layers layers 21, 22, 23, 24 are excluded by setting their weights to 0.
        attn_delta_distill_layer_weights = { 7: 0.5, 8: 0.5,
                                             12: 1.,
                                             16: 1., 17: 1.,
                                             18: 1.,
                                             19: 1., 20: 1., 
                                             21: 1., 22: 1., 
                                             23: 1., 24: 1.,                            
                                            }
        # DISABLE attn delta loss.
        # attn_delta_distill_layer_weights = {}

        # attn norm distillation is applied to almost all conditioning layers.
        attn_norm_distill_layer_weights = { 7: 0.5, 8: 0.5,
                                            12: 1.,
                                            16: 1., 17: 1.,
                                            18: 1.,
                                            19: 1., 20: 1., 
                                            21: 1., 22: 1., 
                                            23: 1., 24: 1.,                                   
                                           }

        feat_size2pooler_spec = { 8: [4, 2], 16: [4, 2], 32: [8, 4], 64: [8, 4] }

        # Normalize the weights above so that each set sum to 1.
        feat_distill_layer_weights          = normalize_dict_values(feat_distill_layer_weights)
        attn_norm_distill_layer_weights     = normalize_dict_values(attn_norm_distill_layer_weights)
        attn_delta_distill_layer_weights    = normalize_dict_values(attn_delta_distill_layer_weights)

        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(fg_indices_2b[0]) // len(torch.unique(fg_indices_2b[0]))
        fg_indices_4b = double_token_indices(fg_indices_2b, BLOCK_SIZE * 2)

        mix_feat_grad_scale = 0.1
        mix_feat_grad_scaler = gen_gradient_scaler(mix_feat_grad_scale)        
        # mix_attn_grad_scale = 0.05, almost zero, effectively no grad to teacher attn. 
        # Setting to 0 may prevent the graph from being released and OOM.
        mix_attn_grad_scale  = 0.05  
        mix_attn_grad_scaler = gen_gradient_scaler(mix_attn_grad_scale)

        loss_layers_subj_attn_delta_align   = []
        loss_layers_feat_delta_align        = []
        loss_layers_subj_attn_norm_distill  = []

        for unet_layer_idx, ca_outfeat in ca_outfeats.items():
            if (unet_layer_idx not in feat_distill_layer_weights) and (unet_layer_idx not in attn_norm_distill_layer_weights):
                continue

            if ca_outfeat_lns is not None:
                ca_outfeat = ca_outfeat_lns[str(unet_layer_idx)](ca_outfeat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # each is [1, 1280, 16, 16]
            subj_single_feat, subj_comp_feat, mix_single_feat, mix_comp_feat \
                = ca_outfeat.chunk(4)
            
            # attn_score_mat: [4, 8, 256, 77] => [4, 77, 8, 256].
            # We don't need BP through attention into UNet.
            attn_score_mat = ca_attnscores[unet_layer_idx].permute(0, 3, 1, 2)
            # subj_attn_4b: [4, 8, 256]  (1 embedding  for 1 token)  => [4, 1, 8, 256] => [4, 8, 256]
            # or            [16, 8, 256] (4 embeddings for 1 token)  => [4, 4, 8, 256] => [4, 8, 256]
            # BLOCK_SIZE*4: this batch contains 4 blocks. Each block should have one instance.
            subj_attn_4b = attn_score_mat[fg_indices_4b].reshape(BLOCK_SIZE*4, K_fg, *attn_score_mat.shape[2:]).sum(dim=1)
            # subj_single_subj_attn, ...: [1, 8, 256] (1 embedding  for 1 token) 
            # or                          [1, 8, 256] (4 embeddings for 1 token)
            subj_single_subj_attn, subj_comp_subj_attn, mix_single_subj_attn, mix_comp_subj_attn \
                = subj_attn_4b.chunk(4)

            if unet_layer_idx in attn_norm_distill_layer_weights:
                attn_norm_distill_layer_weight     = attn_norm_distill_layer_weights[unet_layer_idx]
                attn_delta_distill_layer_weight    = attn_delta_distill_layer_weights[unet_layer_idx]

                # mix_attn_grad_scale = 0.05, almost zero, effectively no grad to mix_comp_subj_attn/mix_single_subj_attn. 
                # Use this scaler to release the graph and avoid OOM.
                mix_comp_subj_attn_gs   = mix_attn_grad_scaler(mix_comp_subj_attn)
                mix_single_subj_attn_gs = mix_attn_grad_scaler(mix_single_subj_attn)

                if attn_delta_distill_layer_weight > 0:
                    # subj_single_subj_attn, subj_comp_subj_attn, mix_single_subj_attn,
                    # mix_comp_subj_attn: [1, 8, 64].
                    # Do gs on mix_* by setting ref_grad_scale = 0.05.
                    # No gs on subj_single_subj_attn, by setting feat_base_grad_scale = 1.
                    loss_dict_layer_subj_attn_delta_align \
                        = calc_delta_alignment_loss(subj_single_subj_attn, subj_comp_subj_attn,
                                                    mix_single_subj_attn,  mix_comp_subj_attn,
                                                    ref_grad_scale=mix_attn_grad_scale,
                                                    feat_base_grad_scale=1,
                                                    use_cosine_loss=True, cosine_exponent=3,
                                                    delta_types=['feat_to_ref'])

                    loss_layers_subj_attn_delta_align.append(loss_dict_layer_subj_attn_delta_align['feat_to_ref'] \
                                                             * attn_delta_distill_layer_weight)

                    '''
                    I don't know why, but loss_comp_attn_delta_distill greatly hurts the performance.
                    # It encourages subj_comp_comp_attn to express at least 1s of mix_comp_comp_attn, i.e.,
                    # comp_attn_align_coeffs should be >= 1. So a loss is incurred if it's < 1.
                    # do_sqr: square the loss, so that the loss is more sensitive to smaller (<< 1) delta_align_coeffs.
                    # subj_comp_comp_attn: [1, 13, 8, 64]. 13: number of extra compositional tokens.
                    # Don't use calc_ref_cosine_loss() as it is scale invariant to mix_comp_comp_attn_gs.
                    # However,we wish subj_comp_comp_attn >= mix_comp_comp_attn_gs.
                    # calc_align_coeff_loss() is ok, since we don't care about high-frequency details
                    # of the compositional part.

                    # comp_attn_score_mat_2b: [2, 13, 8, 64]. 13: number of extra compositional tokens.
                    comp_attn_score_mat_2b = sel_emb_attns_by_indices(attn_score_mat, comp_extra_indices_13b, 
                                                                    do_sum=False, do_sqrt_norm=False)
                    subj_comp_comp_attn, mix_comp_comp_attn = comp_attn_score_mat_2b.chunk(2)
                    mix_comp_comp_attn_gs   = mix_attn_grad_scaler(mix_comp_comp_attn)

                    loss_layer_comp_attn_delta = calc_align_coeff_loss(subj_comp_comp_attn, mix_comp_comp_attn_gs,
                                                                       margin=1., ref_grad_scale=1, do_sqr=True)
                    loss_comp_attn_delta_distill += loss_layer_comp_attn_delta * attn_delta_distill_layer_weight
                    '''
                # mean(dim=-1): average over the 64 feature channels.
                # Align the attention corresponding to each embedding individually.
                # Note mix_*subj_attn use *_gs versions.
                loss_layer_subj_comp_attn_norm   = (subj_comp_subj_attn.mean(dim=-1)   - mix_comp_subj_attn_gs.mean(dim=-1)).abs().mean()
                loss_layer_subj_single_attn_norm = (subj_single_subj_attn.mean(dim=-1) - mix_single_subj_attn_gs.mean(dim=-1)).abs().mean()
                # loss_subj_attn_norm_distill uses L1 loss, which tends to be in 
                # smaller magnitudes than the delta loss. So it will be scaled up later in p_losses().
                loss_layers_subj_attn_norm_distill.append(( loss_layer_subj_comp_attn_norm + loss_layer_subj_single_attn_norm ) \
                                                          * attn_norm_distill_layer_weight)

            if unet_layer_idx not in feat_distill_layer_weights:
                continue

            feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]
            # subj_single_feat, ...: [1, 1280, 16, 16]
            subj_single_feat, subj_comp_feat, mix_single_feat, mix_comp_feat \
                = ca_outfeat.chunk(4)

            # convert_attn_to_spatial_weight() will detach attention weights to 
            # avoid BP through attention.
            # reversed=True: larger subject attention => smaller spatial weight, i.e., 
            # pay more attention to the context.
            spatial_weight_mix_comp, spatial_attn_mix_comp   = convert_attn_to_spatial_weight(mix_comp_subj_attn, BLOCK_SIZE, 
                                                                                                mix_comp_feat.shape[2:],
                                                                                                reversed=True)

            spatial_weight_subj_comp, spatial_attn_subj_comp = convert_attn_to_spatial_weight(subj_comp_subj_attn, BLOCK_SIZE,
                                                                                                subj_comp_feat.shape[2:],
                                                                                                reversed=True)
            # Use mix single/comp weights on both subject-only and mix features, 
            # to reduce misalignment and facilitate distillation.
            # The multiple heads are aggregated by mean(), since the weighted features don't have multiple heads.
            spatial_weight = (spatial_weight_mix_comp + spatial_weight_subj_comp) / 2
            # spatial_attn_mix_comp, spatial_attn_subj_comp are returned for debugging purposes. 
            # Delete them to release RAM.
            del spatial_attn_mix_comp, spatial_attn_subj_comp

            ca_outfeat  = ca_outfeat * spatial_weight

            # 8  -> 4, 2 (output 3),  16 -> 4, 2 (output 7), 
            # 32 -> 8, 4 (output 7),  64 -> 8, 4 (output 15).
            pooler_kernel_size, pooler_stride = feat_size2pooler_spec[ca_outfeat.shape[-1]]
            # feature pooling: allow small perturbations of the locations of pixels.
            # If subj_single_feat is 8x8, then after pooling, it becomes 3x3, too rough.
            # The smallest feat shape > 8x8 is 16x16 => 7x7 after pooling.
            pooler = nn.AvgPool2d(pooler_kernel_size, stride=pooler_stride)

            # ca_outfeat_2d: [4, 1280, 8, 8] -> [4, 1280, 8, 8] -> [4, 1280*7*7] = [4, 62720].
            ca_outfeat_2d = pooler(ca_outfeat).reshape(ca_outfeat.shape[0], -1)
            # subj_single_feat_2d, ...: [1, 1280, 62720]
            subj_single_feat_2d, subj_comp_feat_2d, mix_single_feat_2d, mix_comp_feat_2d \
                = ca_outfeat_2d.chunk(4)

            # mix_feat_grad_scale = 0.1.
            mix_single_feat_2d_gs  = mix_feat_grad_scaler(mix_single_feat_2d)
            mix_comp_feat_2d_gs    = mix_feat_grad_scaler(mix_comp_feat_2d)

            comp_feat_delta   = ortho_subtract(subj_comp_feat_2d,   mix_comp_feat_2d_gs)
            # subj_single_feat is not gs'ed, and I don't know why without gs it still works.
            single_feat_delta = ortho_subtract(subj_single_feat_2d, mix_single_feat_2d_gs)
                
            # single_feat_delta, comp_feat_delta: [1, 1280], ...
            # Pool the spatial dimensions H, W to remove spatial information.
            # The gradient goes back to single_feat_delta -> subj_comp_feat,
            # as well as comp_feat_delta -> mix_comp_feat.
            # If stop_single_grad, the gradients to subj_single_feat and mix_single_feat are stopped, 
            # as these two images should look good by themselves (since they only contain the subject).
            # Note the learning strategy to the single image features should be different from 
            # the single embeddings, as the former should be optimized to look good by itself,
            # while the latter should be optimized to cater for two objectives: 1) the conditioned images look good,
            # and 2) the embeddings are amendable to composition.
            loss_layer_feat_delta_align = ortho_l2loss(comp_feat_delta, single_feat_delta, mean=True)
            loss_layers_feat_delta_align.append(loss_layer_feat_delta_align * feat_distill_layer_weight)

        loss_feat_delta_align       = normalized_sum(loss_layers_feat_delta_align)
        loss_subj_attn_delta_align  = normalized_sum(loss_layers_subj_attn_delta_align)
        loss_subj_attn_norm_distill = normalized_sum(loss_layers_subj_attn_norm_distill)

        return loss_feat_delta_align, loss_subj_attn_delta_align, loss_subj_attn_norm_distill

    def calc_fg_mb_suppress_loss(self, ca_attnscores, subj_indices, 
                                 BLOCK_SIZE, fg_mask, instance_mask=None):
        if (subj_indices is None) or (len(subj_indices) == 0) or (fg_mask is None) \
          or (instance_mask is not None and instance_mask.sum() == 0):
            return 0

        # Discard the first few bottom layers from alignment.
        # attn_align_layer_weights: relative weight of each layer. 
        attn_align_layer_weights = { 7: 0.5, 8: 0.5,
                                     12: 1.,
                                     16: 1., 17: 1.,
                                     18: 1.,
                                     19: 1., 20: 1., 
                                     21: 1., 22: 1., 
                                     23: 1., 24: 1., 
                                   }
                
        # Normalize the weights above so that each set sum to 1.
        attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)
        # K_fg: 9, number of embeddings per subject token.
        K_fg = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        subj_mb_suppress_scale      = 0.05
        mfmb_contrast_score_margin  = 0.4

        # Protect subject emb activations on fg areas.
        subj_score_at_mf_grad_scale = 0.5
        subj_score_at_mf_grad_scaler = gen_gradient_scaler(subj_score_at_mf_grad_scale)

        # In each instance, subj_indices has K_fg times as many elements as bg_indices.
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # bg_indices: ([0, 1, 2, 3], [11, 12, 34, 29]).
        # BLOCK_SIZE = 2, so we only keep instances indexed by [0, 1].
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1], [5, 6, 7, 8, 6, 7, 8, 9]).
        subj_indices = (subj_indices[0][:BLOCK_SIZE*K_fg], subj_indices[1][:BLOCK_SIZE*K_fg])

        loss_layers_subj_mb_suppress    = []

        for unet_layer_idx, unet_attn_score in ca_attnscores.items():
            if (unet_layer_idx not in attn_align_layer_weights):
                continue

            attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]
            # [2, 8, 256, 77] / [2, 8, 64, 77] =>
            # [2, 77, 8, 256] / [2, 77, 8, 64]
            attnscore_mat = unet_attn_score.permute(0, 3, 1, 2)

            # subj_score: [8, 8, 64] -> [2, 4, 8, 64] sum among K_fg embeddings -> [2, 8, 64]
            subj_score = sel_emb_attns_by_indices(attnscore_mat, subj_indices,
                                                  do_sum=True, do_mean=False, do_sqrt_norm=False)

            fg_mask2 = resize_mask_for_feat_or_attn(subj_score, fg_mask, "fg_mask", 
                                                    num_spatial_dims=1,
                                                    mode="nearest|bilinear")
            # Repeat 8 times to match the number of attention heads (for normalization).
            fg_mask2 = fg_mask2.reshape(BLOCK_SIZE, 1, -1).repeat(1, subj_score.shape[1], 1)
            fg_mask3 = torch.zeros_like(fg_mask2)
            fg_mask3[fg_mask2 >  1e-6] = 1.

            bg_mask3 = (1 - fg_mask3)

            if (fg_mask3.sum(dim=(1, 2)) == 0).any():
                # Very rare cases. Safe to skip.
                print("WARNING: fg_mask3 has all-zero masks.")
                continue
            if (bg_mask3.sum(dim=(1, 2)) == 0).any():
                # Very rare cases. Safe to skip.
                print("WARNING: bg_mask3 has all-zero masks.")
                continue

            subj_score_at_mf = subj_score * fg_mask3
            subj_score_at_mf = subj_score_at_mf_grad_scaler(subj_score_at_mf)
            # subj_score_at_mb: [BLOCK_SIZE, 8, 64].
            # mb: mask foreground locations, mask background locations.
            subj_score_at_mb = subj_score * bg_mask3

            # fg_mask3: [BLOCK_SIZE, 8, 64]
            # avg_subj_score_at_mf: [BLOCK_SIZE, 1, 1]
            # keepdim=True, since scores at all locations will use them as references (subtract them).
            avg_subj_score_at_mf = masked_mean(subj_score_at_mf, fg_mask3, dim=(1,2), keepdim=True)
            avg_subj_score_at_mb = masked_mean(subj_score_at_mb, bg_mask3, dim=(1,2), keepdim=True)

            if False and 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
                print(f'layer {unet_layer_idx}')
                print(f'avg_subj_score_at_mf: {avg_subj_score_at_mf.mean():.4f}, avg_subj_score_at_mb: {avg_subj_score_at_mb.mean():.4f}')
            
            # Encourage avg_subj_score_at_mf (subj_score averaged at foreground locations) 
            # to be at least larger by mfmb_contrast_score_margin = 0.4 than 
            # subj_score_at_mb at any background locations.
            # If not, clamp() > 0, incurring a loss.
            # layer_subj_mb_excess: [BLOCK_SIZE, 8, 64].
            layer_subj_mb_excess    = subj_score_at_mb + mfmb_contrast_score_margin - avg_subj_score_at_mf
            # Compared to masked_mean(), mean() is like dynamically reducing the loss weight when more and more 
            # activations conform to the margin restrictions.
            loss_layer_subj_mb_suppress   = masked_mean(layer_subj_mb_excess, 
                                                        layer_subj_mb_excess > 0, 
                                                        instance_weights=instance_mask)

            # loss_layer_subj_bg_contrast_at_mf is usually 0, 
            # so loss_subj_mb_suppress is much smaller than loss_bg_mf_suppress.
            # subj_mb_suppress_scale: 0.05.
            loss_layers_subj_mb_suppress.append(loss_layer_subj_mb_suppress \
                                                * attn_align_layer_weight * subj_mb_suppress_scale)
            
        loss_subj_mb_suppress = normalized_sum(loss_layers_subj_mb_suppress)
    
        return loss_subj_mb_suppress
    
    # Only compute the loss on the first block. If it's a normal_recon iter, 
    # the first block is the whole batch, i.e., BLOCK_SIZE = batch size.
    # bg_indices: we assume the bg tokens appear in all instances in the batch.
    def calc_fg_bg_complementary_loss(self, ca_attnscores,
                                      subj_indices, 
                                      bg_indices,
                                      BLOCK_SIZE, 
                                      fg_grad_scale=0.1,
                                      fg_mask=None, instance_mask=None,
                                      do_sqrt_norm=False):
        
        if subj_indices is None:
            return 0, 0, 0, 0
        
        if subj_indices is not None and bg_indices is None:
            loss_subj_mb_suppress = self.calc_fg_mb_suppress_loss(ca_attnscores, subj_indices, 
                                                                  BLOCK_SIZE, fg_mask, instance_mask)
            
            return 0, loss_subj_mb_suppress, 0, 0

        # Discard the first few bottom layers from alignment.
        # attn_align_layer_weights: relative weight of each layer. 
        attn_align_layer_weights = { 7: 0.5, 8: 0.5,
                                     12: 1.,
                                     16: 1., 17: 1.,
                                     18: 1.,
                                     19: 1., 20: 1., 
                                     21: 1., 22: 1., 
                                     23: 1., 24: 1., 
                                   }
        # 16-18: feature maps 16x16.
        # 19-21: feature maps 32x32.
        # 22-24: feature maps 64x64.
        # The weight is inversely proportional to the feature map size.
        # The larger the feature map, the more details the layer captures, and 
        # fg/bg loss hurts more high-frequency details, therefore it has a smalll weight.

        # Normalize the weights above so that each set sum to 1.
        attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)

        # K_fg: 9, number of embeddings per subject token.
        K_fg = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # K_bg: 4, number of embeddings per background token.
        K_bg = len(bg_indices[0]) // len(torch.unique(bg_indices[0]))

        loss_layers_fg_bg_complementary = []
        loss_layers_subj_mb_suppress    = []
        loss_layers_bg_mf_suppress      = []
        loss_layers_fg_bg_mask_contrast = []

        subj_mb_suppress_scale                = 0.05
        bg_mf_suppress_scale                  = 0.1
        fgbg_emb_contrast_scale               = 0.05
        mfmb_contrast_score_margin            = 0.4
        subj_bg_contrast_at_mf_score_margin   = 0.4 * K_fg / K_bg     # 0.9
        bg_subj_contrast_at_mb_score_margin   = 0.4

        subj_score_at_mf_grad_scale = 0.5
        subj_score_at_mf_grad_scaler = gen_gradient_scaler(subj_score_at_mf_grad_scale)

        # In each instance, subj_indices has K_fg times as many elements as bg_indices.
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # bg_indices: ([0, 1, 2, 3], [11, 12, 34, 29]).
        # BLOCK_SIZE = 2, so we only keep instances indexed by [0, 1].
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1], [5, 6, 7, 8, 6, 7, 8, 9]).
        subj_indices = (subj_indices[0][:BLOCK_SIZE*K_fg], subj_indices[1][:BLOCK_SIZE*K_fg])

        #fg_attn_grad_scale  = 0.5
        #fg_attn_grad_scaler = gen_gradient_scaler(fg_attn_grad_scale)

        for unet_layer_idx, unet_attn_score in ca_attnscores.items():
            if (unet_layer_idx not in attn_align_layer_weights):
                continue

            # [2, 8, 256, 77] / [2, 8, 64, 77] =>
            # [2, 77, 8, 256] / [2, 77, 8, 64]
            attnscore_mat = unet_attn_score.permute(0, 3, 1, 2)
            # subj_score: [8, 8, 64] -> [2, 4, 8, 64] sum among K_fg embeddings -> [2, 8, 64]
            subj_score = sel_emb_attns_by_indices(attnscore_mat, subj_indices, 
                                                  do_sum=True, do_mean=False, do_sqrt_norm=do_sqrt_norm)
            
            # sel_emb_attns_by_indices will split bg_indices to multiple instances,
            # and select the corresponding attention rows for each instance.
            bg_score   = sel_emb_attns_by_indices(attnscore_mat, bg_indices, 
                                                  do_sum=True, do_mean=False, do_sqrt_norm=do_sqrt_norm)

            attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]

            # aim_to_align=False: push bg_score to be orthogonal with subj_score, 
            # so that the two attention maps are complementary.
            # exponent = 2: exponent is 3 by default, which lets the loss focus on large activations.
            # But we don't want to only focus on large activations. So set it to 2.
            # ref_grad_scale = 0.05: small gradients will be BP-ed to the subject embedding,
            # to make the two attention maps more complementary (expect the loss pushes the 
            # subject embedding to a more accurate point).

            # Use subj_score as a reference, and scale down grad to fg attn, 
            # to make fg embeddings more stable.
            # fg_grad_scale: 0.1.
            loss_layer_fg_bg_comple = \
                calc_ref_cosine_loss(bg_score, subj_score, 
                                     exponent=2,    
                                     do_demean_first=False,
                                     first_n_dims_to_flatten=2, 
                                     ref_grad_scale=fg_grad_scale,
                                     aim_to_align=False,
                                     debug=False)

            # loss_fg_bg_complementary doesn't need fg_mask.
            loss_layers_fg_bg_complementary.append(loss_layer_fg_bg_comple * attn_align_layer_weight)

            if (fg_mask is not None) and (instance_mask is None or instance_mask.sum() > 0):
                fg_mask2 = resize_mask_for_feat_or_attn(subj_score, fg_mask, "fg_mask", 
                                                        num_spatial_dims=1,
                                                        mode="nearest|bilinear")
                # Repeat 8 times to match the number of attention heads (for normalization).
                fg_mask2 = fg_mask2.reshape(BLOCK_SIZE, 1, -1).repeat(1, subj_score.shape[1], 1)
                fg_mask3 = torch.zeros_like(fg_mask2)
                fg_mask3[fg_mask2 >  1e-6] = 1.

                bg_mask3 = (1 - fg_mask3)

                if (fg_mask3.sum(dim=(1, 2)) == 0).any():
                    # Very rare cases. Safe to skip.
                    print("WARNING: fg_mask3 has all-zero masks.")
                    continue
                if (bg_mask3.sum(dim=(1, 2)) == 0).any():
                    # Very rare cases. Safe to skip.
                    print("WARNING: bg_mask3 has all-zero masks.")
                    continue

                # subj_score_at_mf: [BLOCK_SIZE, 8, 64].
                # subj, bg: subject embedding,         background embedding.
                # mf,   mb: mask foreground locations, mask background locations.
                subj_score_at_mf = subj_score * fg_mask3
                # Protect subject emb activations on fg areas.
                subj_score_at_mf = subj_score_at_mf_grad_scaler(subj_score_at_mf)

                bg_score_at_mf   = bg_score   * fg_mask3
                subj_score_at_mb = subj_score * bg_mask3
                bg_score_at_mb   = bg_score   * bg_mask3

                # fg_mask3: [BLOCK_SIZE, 8, 64]
                # avg_subj_score_at_mf: [BLOCK_SIZE, 1, 1]
                # keepdim=True, since scores at all locations will use them as references (subtract them).
                # sum(dim=(1,2)): avoid summing across the batch dimension. 
                # It's meaningless to average among the instances.
                avg_subj_score_at_mf = masked_mean(subj_score_at_mf, fg_mask3, dim=(1,2), keepdim=True)
                avg_subj_score_at_mb = masked_mean(subj_score_at_mb, bg_mask3, dim=(1,2), keepdim=True)
                avg_bg_score_at_mf   = masked_mean(bg_score_at_mf,   fg_mask3, dim=(1,2), keepdim=True)
                avg_bg_score_at_mb   = masked_mean(bg_score_at_mb,   bg_mask3, dim=(1,2), keepdim=True)

                if False and 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
                    print(f'layer {unet_layer_idx}')
                    print(f'avg_subj_score_at_mf: {avg_subj_score_at_mf.mean():.4f}, avg_subj_score_at_mb: {avg_subj_score_at_mb.mean():.4f}')
                    print(f'avg_bg_score_at_mf:   {avg_bg_score_at_mf.mean():.4f},   avg_bg_score_at_mb:   {avg_bg_score_at_mb.mean():.4f}')
                
                # Encourage avg_subj_score_at_mf (subj_score averaged at foreground locations) 
                # to be at least larger by mfmb_contrast_score_margin = 0.4 than 
                # subj_score_at_mb at any background locations.
                # If not, clamp() > 0, incurring a loss.
                # layer_subj_mb_excess: [BLOCK_SIZE, 8, 64].
                layer_subj_mb_excess    = subj_score_at_mb + mfmb_contrast_score_margin - avg_subj_score_at_mf
                # Compared to masked_mean(), mean() is like dynamically reducing the loss weight when more and more 
                # activations conform to the margin restrictions.
                loss_layer_subj_mb_suppress   = masked_mean(layer_subj_mb_excess, 
                                                            layer_subj_mb_excess > 0, 
                                                            instance_weights=instance_mask)
                # Encourage avg_bg_score_at_mb (bg_score averaged at background locations)
                # to be at least larger by mfmb_contrast_score_margin = 1 than
                # bg_score_at_mf at any foreground locations.
                # If not, clamp() > 0, incurring a loss.
                layer_bg_mf_suppress          = bg_score_at_mf + mfmb_contrast_score_margin - avg_bg_score_at_mb
                loss_layer_bg_mf_suppress     = masked_mean(layer_bg_mf_suppress, 
                                                            layer_bg_mf_suppress > 0, 
                                                            instance_weights=instance_mask)
                # Encourage avg_subj_score_at_mf (subj_score averaged at foreground locations)
                # to be at least larger by subj_bg_contrast_at_mf_score_margin = 0.8 than
                # bg_score_at_mf at any foreground locations.
                # loss_layer_subj_bg_contrast_at_mf is usually 0, as avg_bg_score_at_mf 
                # usually takes a much smaller value than avg_subj_score_at_mf.
                # avg_subj_score_at_mf.item(): protect subj fg activations.
                layer_subj_bg_contrast_at_mf        = bg_score_at_mf + subj_bg_contrast_at_mf_score_margin - avg_subj_score_at_mf
                loss_layer_subj_bg_contrast_at_mf   = masked_mean(layer_subj_bg_contrast_at_mf, 
                                                                  layer_subj_bg_contrast_at_mf > 0, 
                                                                  instance_weights=instance_mask)
                # Encourage avg_bg_score_at_mb (bg_score averaged at background locations)
                # to be at least larger by subj_bg_contrast_at_mf_score_margin = 0.2 than
                # subj_score_at_mb at any background locations.
                layer_bg_subj_contrast_at_mb        = subj_score_at_mb + bg_subj_contrast_at_mb_score_margin - avg_bg_score_at_mb
                loss_layer_bg_subj_contrast_at_mb   = masked_mean(layer_bg_subj_contrast_at_mb, 
                                                                  layer_bg_subj_contrast_at_mb > 0, 
                                                                  instance_weights=instance_mask)
                # loss_layer_subj_bg_contrast_at_mf is usually 0, 
                # so loss_subj_mb_suppress is much smaller than loss_bg_mf_suppress.
                # subj_mb_suppress_scale: 0.05.
                loss_layers_subj_mb_suppress.append(loss_layer_subj_mb_suppress \
                                                    * attn_align_layer_weight * subj_mb_suppress_scale)
                # bg_mf_suppress_scale: 0.1. More penalty of bg emb activations on fg areas.
                loss_layers_bg_mf_suppress.append(loss_layer_bg_mf_suppress \
                                                    * attn_align_layer_weight * bg_mf_suppress_scale)
                # fgbg_emb_contrast_scale: 0.05. Balanced penalty of fg emb activation 
                # contrast on fg and bg areas.
                loss_layers_fg_bg_mask_contrast.append((loss_layer_subj_bg_contrast_at_mf + loss_layer_bg_subj_contrast_at_mb) \
                                                        * attn_align_layer_weight * fgbg_emb_contrast_scale)
                #print(f'layer {unet_layer_idx}')
                #print(f'subj_contrast: {loss_layer_subj_contrast:.4f}, subj_bg_contrast_at_mf: {loss_layer_subj_bg_contrast_at_mf:.4f},')
                #print(f"bg_contrast:   {loss_layer_bg_contrast:.4f},   subj_bg_contrast_at_mb: {loss_layer_subj_bg_contrast_at_mb:.4f}")

        loss_fg_bg_complementary = normalized_sum(loss_layers_fg_bg_complementary)
        loss_subj_mb_suppress    = normalized_sum(loss_layers_subj_mb_suppress)
        loss_bg_mf_suppress      = normalized_sum(loss_layers_bg_mf_suppress)
        loss_fg_bg_mask_contrast = normalized_sum(loss_layers_fg_bg_mask_contrast)

        return loss_fg_bg_complementary, loss_subj_mb_suppress, loss_bg_mf_suppress, loss_fg_bg_mask_contrast

    # SSB_SIZE: subject sub-batch size.
    # bg_indices could be None if iter_flags['use_background_token'] = False.
    def calc_fg_bg_xlayer_consist_loss(self, ca_attnscores, subj_indices, bg_indices, SSB_SIZE):
        # Discard the first few bottom layers from alignment.
        # attn_align_layer_weights: relative weight of each layer. 
        # layer 7 is absent, since layer 8 aligns with layer 7.
        attn_align_layer_weights = { 8: 0.5,
                                     12: 1.,
                                     16: 1., 17: 1.,
                                     18: 1.,
                                     19: 0.5, 20: 0.5, 
                                     21: 0.5,  22: 0.25, 
                                     23: 0.25, 24: 0.25, 
                                   }
        # 16-18: feature maps 16x16.
        # 19-21: feature maps 32x32.
        # 22-24: feature maps 64x64.
        # The weight is inversely proportional to the feature map size.
        # The larger the feature map, the more details the layer captures, and 
        # fg/bg loss hurts more high-frequency details, therefore it has a smalll weight.
                
        # Align a layer with the layer below it.
        attn_align_xlayer_maps = { 8: 7, 12: 8, 16: 12, 17: 16, 18: 17, 19: 18, 
                                   20: 19, 21: 20, 22: 21, 23: 22, 24: 23 }

        # Normalize the weights above so that each set sum to 1.
        attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)

        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # In each instance, subj_indices has K_fg elements, 
        # and bg_indices has K_bg elements or 0 elements 
        # (if iter_flags['use_background_token'] = False)
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                          [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # bg_indices: ([0, 1, 2, 3, 4, 5, 6, 7], [11, 12, 34, 29, 11, 12, 34, 29]).
        # SSB_SIZE = 2, so we only keep instances indexed by [0, 1].
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1], [5, 6, 7, 8, 6, 7, 8, 9]).
        subj_indices = (subj_indices[0][:SSB_SIZE*K_fg], subj_indices[1][:SSB_SIZE*K_fg])

        if bg_indices is not None:
            # K_bg: 1 or 2, number of embeddings per background token.
            K_bg = len(bg_indices[0]) // len(torch.unique(bg_indices[0]))
            # bg_indices: ([0, 1], [11, 12]).
            bg_indices = (bg_indices[0][:SSB_SIZE*K_bg], bg_indices[1][:SSB_SIZE*K_bg])

        loss_layers_fg_xlayer_consist = []
        loss_layers_bg_xlayer_consist = []

        for unet_layer_idx, unet_attn_score in ca_attnscores.items():
            if (unet_layer_idx not in attn_align_layer_weights):
                continue

            # [2, 8, 256, 77] => [2, 77, 8, 256]
            attn_score_mat        = unet_attn_score.permute(0, 3, 1, 2)
            # [2, 8, 64, 77]  => [2, 77, 8, 64]
            attn_score_mat_xlayer = ca_attnscores[attn_align_xlayer_maps[unet_layer_idx]].permute(0, 3, 1, 2)
            
            # Make sure attn_score_mat_xlayer is always smaller than attn_score_mat.
            # So we always scale down attn_score_mat to match attn_score_mat_xlayer.
            if attn_score_mat_xlayer.shape[-1] > attn_score_mat.shape[-1]:
                attn_score_mat, attn_score_mat_xlayer = attn_score_mat_xlayer, attn_score_mat

            # H: 16, Hx: 8
            H  = int(np.sqrt(attn_score_mat.shape[-1]))
            Hx = int(np.sqrt(attn_score_mat_xlayer.shape[-1]))

            # Why taking mean over 8 heads: In a CA layer, FFN features added to the CA features. 
            # If there were no FFN pathways, the CA features of the previous CA layer would be better aligned
            # with the current CA layer (ignoring the transformations by the intermediate conv layers).
            # Therefore, the CA features in the current CA layer may be barely aligned 
            # with the CA features in the L_xlayer-th CA layer. So it's of little benefit
            # to align the corresponding heads across CA layers.
            # subj_attn:        [8, 8, 256] -> [2, 9, 8, 256] -> mean over 8 heads, sum over 9 embs -> [2, 256] 
            # Average out head, because the head i in layer a may not correspond to head i in layer b.
            subj_attn        = attn_score_mat[subj_indices].reshape(       SSB_SIZE, K_fg, *attn_score_mat.shape[2:]).mean(dim=2).sum(dim=1)
            # subj_attn_xlayer: [8, 8, 64]  -> [2, 9, 8, 64]  -> mean over 8 heads, sum over 9 embs -> [2, 64]
            subj_attn_xlayer = attn_score_mat_xlayer[subj_indices].reshape(SSB_SIZE, K_fg, *attn_score_mat_xlayer.shape[2:]).mean(dim=2).sum(dim=1)

            # subj_attn: [2, 256] -> [2, 16, 16] -> [2, 8, 8] -> [2, 64]
            subj_attn = subj_attn.reshape(SSB_SIZE, 1, H, H)
            subj_attn = F.interpolate(subj_attn, size=(Hx, Hx), mode="bilinear", align_corners=False)
            subj_attn = subj_attn.reshape(SSB_SIZE, Hx*Hx)

            if bg_indices is not None:
                # bg_attn:   [8, 8, 256] -> [2, 4, 8, 256] -> mean over 8 heads, sum over 4 embs -> [2, 256]
                # 8: 8 attention heads. Last dim 256: number of image tokens.
                # Average out head, because the head i in layer a may not correspond to head i in layer b.
                bg_attn         = attn_score_mat[bg_indices].reshape(SSB_SIZE, K_bg, *attn_score_mat.shape[2:]).mean(dim=2).sum(dim=1)
                bg_attn_xlayer  = attn_score_mat_xlayer[bg_indices].reshape(SSB_SIZE, K_bg, *attn_score_mat_xlayer.shape[2:]).mean(dim=2).sum(dim=1)
                # bg_attn: [2, 4, 256] -> [2, 4, 16, 16] -> [2, 4, 8, 8] -> [2, 4, 64]
                bg_attn   = bg_attn.reshape(SSB_SIZE, 1, H, H)
                bg_attn   = F.interpolate(bg_attn, size=(Hx, Hx), mode="bilinear", align_corners=False)
                bg_attn   = bg_attn.reshape(SSB_SIZE, Hx*Hx)
            
            attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]

            #loss_layer_fg_xlayer_consist = ortho_l2loss(subj_attn, subj_attn_xlayer, mean=True)
            loss_layer_fg_xlayer_consist = calc_ref_cosine_loss(subj_attn, subj_attn_xlayer,
                                                                exponent=2,    
                                                                do_demean_first=True,
                                                                first_n_dims_to_flatten=1,
                                                                ref_grad_scale=1,
                                                                aim_to_align=True)
            
            loss_layers_fg_xlayer_consist.append(loss_layer_fg_xlayer_consist * attn_align_layer_weight)
            
            if bg_indices is not None:
                #loss_layer_bg_xlayer_consist = ortho_l2loss(bg_attn, bg_attn_xlayer, mean=True)
                loss_layer_bg_xlayer_consist = calc_ref_cosine_loss(bg_attn, bg_attn_xlayer,
                                                                    exponent=2,    
                                                                    do_demean_first=True,
                                                                    first_n_dims_to_flatten=1,
                                                                    ref_grad_scale=1,
                                                                    aim_to_align=True)
                
                loss_layers_bg_xlayer_consist.append(loss_layer_bg_xlayer_consist * attn_align_layer_weight)

        loss_fg_xlayer_consist = normalized_sum(loss_layers_fg_xlayer_consist)
        loss_bg_xlayer_consist = normalized_sum(loss_layers_bg_xlayer_consist)

        return loss_fg_xlayer_consist, loss_bg_xlayer_consist

    # DISABLED. No obvious improvement.
    def calc_subj_comp_ortho_loss(self, ca_ks, ca_vs, ca_attnscores,
                                  subj_indices_by_block, comp_extra_indices_by_block, 
                                  cls_grad_scale=0.05):

        # Discard the first few bottom layers from the orthogonal loss.
        k_ortho_layer_weights = { 7: 0.5, 8: 0.5,
                                  12: 1.,
                                  16: 1., 17: 1.,
                                  18: 1.,
                                  19: 1., 20: 1., 
                                  21: 1., 22: 1., 
                                  23: 1., 24: 1.,     
                                }
        
        v_ortho_layer_weights = { 7: 0.5, 8: 0.5,
                                  12: 1.,
                                  16: 1., 17: 1.,
                                  18: 0.5,
                                  19: 0.5, 20: 0.5, 
                                  21: 0.25, 22: 0.25, 
                                  23: 0.25, 24: 0.25,                                
                                }
        
        # Normalize the weights above so that each set sum to 1.
        k_ortho_layer_weights = normalize_dict_values(k_ortho_layer_weights)
        v_ortho_layer_weights = normalize_dict_values(v_ortho_layer_weights)

        loss_subj_comp_key_ortho   = 0
        loss_subj_comp_value_ortho = 0

        subj_subj_indices = subj_indices_by_block[1]
        subj_comp_indices = comp_extra_indices_by_block[1]
        # cls blocks are available, so the ortho loss has to align the delta with the cls delta.
        cls_subj_indices  = subj_indices_by_block[3]
        cls_comp_indices  = comp_extra_indices_by_block[3]

        for unet_layer_idx, seq_k in ca_ks.items():
            if (unet_layer_idx not in k_ortho_layer_weights):
                continue

            # attn_score_mat: [4, 8, 256, 77] => [4, 77, 8, 256].
            # We don't need BP through attention into UNet.
            attn_score_mat = ca_attnscores[unet_layer_idx].permute(0, 3, 1, 2)
            # avg_attn_scores: the average attention score of each prompt token to all image patches.
            # The higher the average attention score, the more important role the token plays 
            # in conditioning the image.
            # Therefore, avg_attn_scores is used to weight the ks/vs of tokens when computing their mean k/v.
            avg_attn_scores = attn_score_mat.mean(dim=(2,3), keepdim=False).clamp(min=0)

            loss_layer_subj_comp_key_ortho = \
                calc_layer_subj_comp_k_or_v_ortho_loss(seq_k, 
                                                       subj_subj_indices, 
                                                       subj_comp_indices, 
                                                       cls_subj_indices, 
                                                       cls_comp_indices,
                                                       all_token_weights=avg_attn_scores,
                                                       do_demean_first=False, 
                                                       cls_grad_scale=cls_grad_scale,
                                                       margin=0.6)
            loss_layer_subj_comp_value_ortho = \
                calc_layer_subj_comp_k_or_v_ortho_loss(ca_vs[unet_layer_idx], 
                                                       subj_subj_indices, 
                                                       subj_comp_indices, 
                                                       cls_subj_indices, 
                                                       cls_comp_indices,
                                                       all_token_weights=avg_attn_scores,
                                                       do_demean_first=False, 
                                                       cls_grad_scale=cls_grad_scale,
                                                       margin=0.7)
                    
            k_ortho_layer_weight = k_ortho_layer_weights[unet_layer_idx]
            v_ortho_layer_weight = v_ortho_layer_weights[unet_layer_idx]
            loss_subj_comp_key_ortho   += loss_layer_subj_comp_key_ortho   * k_ortho_layer_weight
            loss_subj_comp_value_ortho += loss_layer_subj_comp_value_ortho * v_ortho_layer_weight

        return loss_subj_comp_key_ortho, loss_subj_comp_value_ortho

    # Intuition of comp_fg_bg_preserve_loss: 
    # In distillation iterations, if comp_init_fg_from_training_image, then at fg_mask areas, x_start is initialized with 
    # the noisy input images. (Otherwise in distillation iterations, x_start is initialized as pure noise.)
    # Essentially, it's to mask the background out of the input images with noise.
    # Therefore, intermediate features at the foreground with single prompts should be close to those of the original images.
    # Features with comp prompts should be similar with the original images at the foreground.
    # So features under comp prompts should be close to features under single prompts, at fg_mask areas.
    # (The features at background areas under comp prompts are the compositional contents, which shouldn't be regularized.) 
    # NOTE: subj_indices are used to compute loss_comp_subj_bg_attn_suppress and loss_comp_mix_bg_attn_suppress.
    def calc_comp_fg_bg_preserve_loss(self, ca_outfeats, ca_outfeat_lns, ca_qs, ca_q_bns, ca_attnscores, 
                                      fg_mask, batch_have_fg_mask, subj_indices, BLOCK_SIZE):
        # No masks available. loss_comp_subj_fg_feat_preserve, loss_comp_subj_bg_attn_suppress are both 0.
        if fg_mask is None or batch_have_fg_mask.sum() == 0:
            return 0, 0, 0, 0, 0, 0

        feat_distill_layer_weights = {  7: 0.5, 8: 0.5,   
                                        12: 1.,
                                        16: 1., 17: 1.,
                                        18: 1.,
                                        19: 1, 20: 1, 
                                        21: 1, 22: 1, 
                                        23: 1, 24: 1, 
                                     }

        # fg_mask is 4D. So expand batch_have_fg_mask to 4D.
        # *_4b means it corresponds to a 4-block batch (batch size = 4 * BLOCK_SIZE).
        fg_mask_4b = fg_mask            * batch_have_fg_mask.view(-1, 1, 1, 1)

        # K_fg: 4, number of embeddings per subject token.
        K_fg   = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # ind_subj_subj_B_1b, ind_subj_subj_N_1b: [0, 0, 0, 0], [5, 6, 7, 8].
        ind_subj_subj_B_1b, ind_subj_subj_N_1b = subj_indices[0][:BLOCK_SIZE*K_fg], subj_indices[1][:BLOCK_SIZE*K_fg]
        ind_subj_B = torch.cat([ind_subj_subj_B_1b,                     ind_subj_subj_B_1b + BLOCK_SIZE,
                                ind_subj_subj_B_1b + 2 * BLOCK_SIZE,    ind_subj_subj_B_1b + 3 * BLOCK_SIZE], dim=0)
        ind_subj_N = ind_subj_subj_N_1b.repeat(4)
        
        # Normalize the weights above so that each set sum to 1.
        feat_distill_layer_weights  = normalize_dict_values(feat_distill_layer_weights)
        mix_grad_scale      = 0.02
        mix_grad_scaler     = gen_gradient_scaler(mix_grad_scale)

        loss_layers_comp_single_map_align      = []
        loss_layers_sc_ss_fg_match             = []
        loss_layers_mc_ms_fg_match             = []
        loss_layers_sc_mc_bg_match             = []

        loss_layers_comp_subj_bg_attn_suppress = []
        loss_layers_comp_mix_bg_attn_suppress  = []
        
        for unet_layer_idx, ca_outfeat in ca_outfeats.items():
            if unet_layer_idx not in feat_distill_layer_weights:
                continue
            feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]

            # ca_outfeat: [4, 1280, 8, 8]
            # ca_layer_q: [4, 8, 64, 160] -> [4, 8, 160, 64] -> [4, 8*160, 8, 8]
            ca_layer_q = ca_qs[unet_layer_idx]
            ca_q_h = int(np.sqrt(ca_layer_q.shape[2] * ca_outfeat.shape[2] // ca_outfeat.shape[3]))
            ca_q_w = ca_layer_q.shape[2] // ca_q_h
            ca_layer_q = ca_layer_q.permute(0, 1, 3, 2).reshape(ca_layer_q.shape[0], -1, ca_q_h, ca_q_w)
            if ca_q_bns is not None:
                ca_layer_q = ca_q_bns[str(unet_layer_idx)](ca_layer_q)

            # Some layers resize the input feature maps. So we need to resize ca_outfeat to match ca_layer_q.
            if ca_outfeat.shape[2:] != ca_layer_q.shape[2:]:
                ca_outfeat = F.interpolate(ca_outfeat, size=ca_layer_q.shape[2:], mode="bilinear", align_corners=False)

            if ca_outfeat_lns is not None:
                ca_outfeat = ca_outfeat_lns[str(unet_layer_idx)](ca_outfeat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            do_feat_pooling = True
            feat_pool_kernel_size = 4
            feat_pool_stride      = 2
            # feature pooling: allow small perturbations of the locations of pixels.
            # calc_comp_fg_bg_preserve_loss() can have higher spatial precision than calc_prompt_mix_loss().
            if do_feat_pooling and ca_outfeat.shape[-1] > 8:
                pooler = nn.AvgPool2d(feat_pool_kernel_size, stride=feat_pool_stride)
            else:
                pooler = nn.Identity()

            ###### elastic matching loss ######
            # layer_q: [4, 1280, 8, 8] -> [4, 1280, 8, 8] -> [4, 1280, 64].
            ca_layer_q_pooled   = pooler(ca_layer_q).reshape(*ca_layer_q.shape[:2], -1)
            # ca_outfeat_pooled: [4, 1280, 8, 8] -> [4, 1280, 8, 8] -> [4, 1280, 64].
            ca_outfeat_pooled   = pooler(ca_outfeat).reshape(*ca_outfeat.shape[:2], -1)
            # fg_attn_mask_4b: [4, 1, 64, 64] => [4, 1, 8, 8]
            fg_attn_mask_4b \
                = resize_mask_for_feat_or_attn(ca_outfeat, fg_mask_4b, "fg_mask_4b", 
                                               num_spatial_dims=2,
                                               mode="nearest|bilinear", warn_on_all_zero=False)
            # fg_attn_mask_4b: [4, 1, 8, 8] -> [4, 1, 8, 8]
            # Since fg_attn_mask_4b is binary, maybe 
            # resize_mask_for_feat_or_attn(ca_outfeat_pooled, ...) is equivalent to
            # resize_mask_for_feat_or_attn(ca_outfeat, ...) then pooler().
            fg_attn_mask_pooled_4b = pooler(fg_attn_mask_4b)
            # fg_attn_mask_pooled: [4, 1, 8, 8] -> [1, 1, 8, 8] 
            fg_attn_mask_pooled = fg_attn_mask_pooled_4b.chunk(4)[0]
            # fg_attn_mask_pooled: [1, 1, 8, 8] -> [1, 1, 64]
            fg_attn_mask_pooled = fg_attn_mask_pooled.reshape(*fg_attn_mask_pooled.shape[:2], -1)

            # sc_map_ss_fg_prob, mc_map_ms_fg_prob: [1, 1, 64]
            # removed loss_layer_ms_mc_fg_match to save computation.
            loss_layer_comp_single_align_map, loss_layer_ss_sc_fg_match, \
            loss_layer_sc_mc_bg_match, sc_map_ss_fg_prob_below_mean, mc_map_ss_fg_prob_below_mean \
                = calc_elastic_matching_loss(ca_layer_q_pooled, ca_outfeat_pooled, fg_attn_mask_pooled, 
                                             fg_bg_cutoff_prob=0.25,
                                             single_q_grad_scale=0.1, single_feat_grad_scale=0.01,
                                             mix_feat_grad_scale=0.05)

            loss_layers_comp_single_map_align.append(loss_layer_comp_single_align_map * feat_distill_layer_weight)
            loss_layers_sc_ss_fg_match.append(loss_layer_ss_sc_fg_match * feat_distill_layer_weight)
            # loss_mc_ms_fg_match += loss_layer_ms_mc_fg_match * feat_distill_layer_weight
            loss_layers_sc_mc_bg_match.append(loss_layer_sc_mc_bg_match * feat_distill_layer_weight)

            if sc_map_ss_fg_prob_below_mean is None or mc_map_ss_fg_prob_below_mean is None:
                continue
            
            ##### unet_attn_score fg preservation loss & bg suppression loss #####
            unet_attn_score = ca_attnscores[unet_layer_idx]
            # attn_score_mat: [4, 8, 256, 77] => [4, 77, 8, 256] 
            attn_score_mat = unet_attn_score.permute(0, 3, 1, 2)
            # subj_subj_attn: [4, 77, 8, 256] -> [4 * K_fg, 8, 256] -> [4, K_fg, 8, 256]
            # attn_score_mat and subj_subj_attn are not pooled.
            subj_attn = attn_score_mat[ind_subj_B, ind_subj_N].reshape(BLOCK_SIZE * 4, K_fg, *attn_score_mat.shape[2:])
            # Sum over 9 subject embeddings. [4, K_fg, 8, 256] -> [4, 8, 256].
            # The scale of the summed attention won't be overly large, since we've done 
            # distribute_embedding_to_M_tokens() to them.
            subj_attn = subj_attn.sum(dim=1)
            H = int(np.sqrt(subj_attn.shape[-1]))
            # subj_attn_hw: [4, 8, 256] -> [4, 8, 8, 8].
            subj_attn_hw = subj_attn.reshape(*subj_attn.shape[:2], H, H)
            # At some layers, the output features are upsampled. So we need to 
            # upsample the attn map to match the output features.
            if subj_attn_hw.shape[2:] != ca_outfeat.shape[2:]:
                subj_attn_hw = F.interpolate(subj_attn_hw, size=ca_outfeat.shape[2:], mode="bilinear", align_corners=False)

            # subj_attn_pooled: [4, 8, 8, 8] -> [4, 8, 8, 8] -> [4, 8, 64].
            subj_attn_pooled = pooler(subj_attn_hw).reshape(*subj_attn_hw.shape[:2], -1)

            subj_single_subj_attn, subj_comp_subj_attn, mix_single_subj_attn, mix_comp_subj_attn \
                = subj_attn_pooled.chunk(4)

            mix_comp_subj_attn_gs = mix_grad_scaler(mix_comp_subj_attn)

            subj_comp_subj_attn_pos   = subj_comp_subj_attn.clamp(min=0)
            mix_comp_subj_attn_gs_pos = mix_comp_subj_attn_gs.clamp(min=0)

            # Suppress the subj attention scores on background areas in comp instances.
            # subj_comp_subj_attn: [1, 8, 64]. ss_bg_mask_map_to_sc: [1, 1, 64].
            # Some elements in subj_comp_subj_attn are negative. 
            # We allow pushing them to -inf, doing which seems to perform better.
            loss_layer_subj_bg_attn_suppress = masked_mean(subj_comp_subj_attn_pos, 
                                                           sc_map_ss_fg_prob_below_mean)
            loss_layer_mix_bg_attn_suppress  = masked_mean(mix_comp_subj_attn_gs_pos,  
                                                           mc_map_ss_fg_prob_below_mean)

            loss_layers_comp_subj_bg_attn_suppress.append(loss_layer_subj_bg_attn_suppress * feat_distill_layer_weight)
            loss_layers_comp_mix_bg_attn_suppress.append(loss_layer_mix_bg_attn_suppress  * feat_distill_layer_weight)

        loss_comp_single_map_align      = normalized_sum(loss_layers_comp_single_map_align)
        loss_sc_ss_fg_match             = normalized_sum(loss_layers_sc_ss_fg_match)
        # loss_mc_ms_fg_match is disabled for efficiency.
        loss_mc_ms_fg_match             = 0 # normalized_sum(loss_layers_mc_ms_fg_match)
        loss_sc_mc_bg_match             = normalized_sum(loss_layers_sc_mc_bg_match)
        loss_comp_subj_bg_attn_suppress = normalized_sum(loss_layers_comp_subj_bg_attn_suppress)
        loss_comp_mix_bg_attn_suppress  = normalized_sum(loss_layers_comp_mix_bg_attn_suppress)

        return loss_comp_single_map_align, loss_sc_ss_fg_match, loss_mc_ms_fg_match, \
               loss_sc_mc_bg_match, loss_comp_subj_bg_attn_suppress, loss_comp_mix_bg_attn_suppress
               

    # SSB_SIZE: subject sub-batch size.
    # If do_normal_recon, then both instances are subject instances. 
    # The subject sub-batch size SSB_SIZE = 3 (1 * BLOCK_SIZE, BLOCK_SIZE=3).
    # If is_compos_iter, then subject sub-batch size SSB_SIZE = 2 (2 * BLOCK_SIZE, BLOCK_SIZE=1).
    # (subj-single and subj-comp blocks).
    def calc_padding_embs_align_loss(self, static_prompt_embeddings, ada_prompt_embeddings, 
                                     prompt_emb_mask, subj_indices, bg_indices,
                                     SSB_SIZE, iter_type, is_compos_iter, emb_clamp_value=-1):
        # If do_normal_recon:
        # static_prompt_embeddings: [3, 16, 77, 768].
        # ada_prompt_embeddings:    [3, 16, 77, 768].
        # prompt_emb_mask:          [12, 77, 1] (correspoinding to the 4-type static prompts)
        # Otherwise, is_compos_iter:
        # static_prompt_embeddings: [4, 16, 77, 768].
        # ada_prompt_embeddings:    [4, 16, 77, 768].
        # prompt_emb_mask:          [4, 77,   1].

        static_prompt_embeddings, ada_prompt_embeddings = \
            clamp_prompt_embedding(emb_clamp_value, static_prompt_embeddings, ada_prompt_embeddings)

        if ada_prompt_embeddings is not None:
            # During training, ada_emb_weight is randomly drawn from [0.4, 0.7].
            ada_emb_weight = self.embedding_manager.get_ada_emb_weight(do_perturb=False) 
            cond_prompt_embeddings = static_prompt_embeddings * (1 - ada_emb_weight) \
                                      + ada_prompt_embeddings * ada_emb_weight
        else:
            cond_prompt_embeddings = static_prompt_embeddings
        
        if not is_compos_iter:
            cond_prompt_embeddings_delta = cond_prompt_embeddings
        else:
            subj_prompt_embeddings, mix_prompt_embeddings = cond_prompt_embeddings.chunk(2)
            # Calculate the delta between the subject embeddings and the mix embeddings.
            # This should have similar effect as demeaning, and at the same time highlight
            # the extra semantics of the subject.
            cond_prompt_embeddings_delta = ortho_subtract(subj_prompt_embeddings, mix_prompt_embeddings)
            
        # padding_mask: [12, 77, 1] => [12, 77]
        padding_mask = 1 - prompt_emb_mask.squeeze(-1)
        # The SOT token always receives the highest attention, which is the normal behavior.
        # So we exclude the SOT token from the align padding loss.
        padding_mask[:, 0] = 0
        # In addition, we also exclude the EOT token from the align padding loss.
        # padding_mask[:, -1] = 0
        # Both subj_embs_contrast_paddings_grad_scaler and subj_embs_contrast_bg_grad_scaler
        # are applied on subj embeddings, but first is for contrasting with padding embeddings,
        # and second is for contrasting with bg embeddings.
        # padding embeddings cannot push the subject embeddings away. So gs is very small.
        subj_embs_contrast_paddings_grad_scale  = 0.02
        subj_embs_contrast_paddings_grad_scaler = gen_gradient_scaler(subj_embs_contrast_paddings_grad_scale)
        # bg embeddings can push the subject embeddings away, but less effectively than 
        # the subject embeddings pushing the bg embeddings away.
        subj_embs_contrast_bg_grad_scale  = 0.3
        subj_embs_contrast_bg_grad_scaler = gen_gradient_scaler(subj_embs_contrast_bg_grad_scale)

        subj_subj_indices_B, subj_subj_indices_N = subj_indices

        K_fg = len(subj_subj_indices_B) // len(torch.unique(subj_subj_indices_B))
        # subj_subj_embs: [3*9, 16, 768]. The 3*9 sets of embeddings are ordered first along tokens, 
        # then along instances. reshape(SSB_SIZE, K_fg) corresponds to this order.
        subj_subj_embs = cond_prompt_embeddings[subj_subj_indices_B, :, subj_subj_indices_N]
        # subj_subj_embs: [27, 16, 768] => [3, 9, 16, 768] => [3, 1, 16, 768]. "1" is for broadcasting.
        subj_subj_embs = subj_subj_embs.reshape(SSB_SIZE, K_fg, self.N_CA_LAYERS, -1).sum(dim=1, keepdim=True)
        subj_subj_embs_gs = subj_embs_contrast_paddings_grad_scaler(subj_subj_embs)

        loss_padding_subj_embs_align = 0
        loss_padding_cls_embs_align  = 0
        loss_bg_subj_embs_align      = 0

        # Subject tokens are always in instances 0 .. SSB_SIZE-1.
        for i in range(SSB_SIZE):
            #subj_padding_indices_i = subj_padding_indices_by_instance[i]
            # If do_normal_recon, padding_mask is [12, 77] for all the 4 types of static prompts. 
            # But SSB_SIZE = 3. It's fine, as we only look at the first SSB_SIZE instances.
            subj_padding_indices_i_N = padding_mask[i].nonzero().squeeze(-1)
            # subj_padding_embs_i: [16, 56, 768] -> [56, 16, 768].
            # Note that subj_padding_embs_i is taken from cond_prompt_embeddings_delta,
            # as i always points to a subject instance.
            subj_padding_embs_i = cond_prompt_embeddings_delta[i, :, subj_padding_indices_i_N].permute(1, 0, 2)
            # subj_subj_embs_gs_i: [1,  16, 768] => [56, 16, 768]. 
            # Same subject embedding paired for all the 56 padding tokens.
            subj_subj_embs_gs_i = subj_subj_embs_gs[i].expand_as(subj_padding_embs_i)
            # Penalize any positive coeffs within padding_subj_embs_align_coeffs_i: [56, 16].
            loss_inst_padding_subj_embs_align \
                = calc_ref_cosine_loss(subj_padding_embs_i, subj_subj_embs_gs_i,
                                       exponent=2, do_demean_first=True, 
                                       first_n_dims_to_flatten=2, 
                                       ref_grad_scale=1, aim_to_align=False)
            
            loss_padding_subj_embs_align += loss_inst_padding_subj_embs_align
        
        loss_padding_subj_embs_align /= SSB_SIZE

        # If is_compos_iter, loss_padding_cls_embs_align is computed as a reference (but not optimized).
        if is_compos_iter:
            cls_subj_indices_B, cls_subj_indices_N = subj_subj_indices_B + SSB_SIZE, subj_subj_indices_N
            cls_subj_embs = cond_prompt_embeddings[cls_subj_indices_B, :, cls_subj_indices_N]
            cls_subj_embs = cls_subj_embs.reshape(SSB_SIZE, K_fg, self.N_CA_LAYERS, -1).sum(dim=1, keepdim=True)
            cls_padding_indices = padding_mask[SSB_SIZE:].nonzero(as_tuple=True)
            cls_padding_indices_by_instance = split_indices_by_instance(cls_padding_indices)

            for i in range(len(cls_padding_indices_by_instance)):
                cls_padding_indices_i_B, cls_padding_indices_i_N = cls_padding_indices_by_instance[i]
                # cls_padding_indices_i_B contains the same batch index multiple times. So take the first one.
                # cls_padding_indices corresponds to padding_mask[SSB_SIZE:], so adding SSB_SIZE to correct the offset.
                i2 = cls_padding_indices_i_B[0] + SSB_SIZE
                # cls_padding_embs_i: [16, 61, 768] -> [61, 16, 768].
                # Note that cls_padding_embs_i is taken from cond_prompt_embeddings,
                # instead of cond_prompt_embeddings_delta, as i2 always points to a cls instance.
                cls_padding_embs_i = cond_prompt_embeddings[i2, :, cls_padding_indices_i_N].permute(1, 0, 2)
                # cls_subj_embs_i:    [1, 16, 768]  -> [61, 16, 768].
                cls_subj_embs_i = cls_subj_embs[i].expand_as(cls_padding_embs_i)
                # we don't do gs on cls_subj_embs, since loss_padding_cls_embs_align is not optimized.
                loss_inst_padding_cls_embs_align \
                    = calc_ref_cosine_loss(cls_padding_embs_i, cls_subj_embs_i,
                                           exponent=2, do_demean_first=True, 
                                           first_n_dims_to_flatten=2,                                              
                                           ref_grad_scale=1, aim_to_align=False)
                
                loss_padding_cls_embs_align += loss_inst_padding_cls_embs_align

            if len(cls_padding_indices_by_instance) > 0:
                loss_padding_cls_embs_align /= len(cls_padding_indices_by_instance)

        # Should be a do_normal_recon iter. Otherwise bg tokens are not in the prompts.
        if bg_indices is not None:
            bg_padding_mask = torch.zeros_like(padding_mask)
            bg_padding_mask[bg_indices] = 1
            # subj_embs_contrast_bg_grad_scale  = 0.3.
            # We use a 0.3 gs (relatively large) on subj_subj_embs, 
            # since we also want to avoid the bg semantics 
            # contaminating the subj semantics. So the loss should be *bi-directional*.
            # bg embeddings can push the subject embeddings away, but less effectively than 
            # the subject embeddings pushing the bg embeddings away.
            subj_subj_embs_gs2 = subj_embs_contrast_bg_grad_scaler(subj_subj_embs)

            # bg tokens may appear in class prompts (beyond SSB_SIZE), 
            # but subj tokens are limited to SSB_SIZE. So we only look at the first SSB_SIZE instances.
            for i in range(SSB_SIZE):
                #subj_padding_indices_i = subj_padding_indices_by_instance[i]
                bg_indices_i_N = bg_padding_mask[i].nonzero().squeeze(-1)
                if len(bg_indices_i_N) > 0:
                    # bg_embs_i: [16, 4, 768] => [4, 16, 768].
                    # Note that bg_embs_i is taken from cond_prompt_embeddings,
                    # as we don't contrast the extra semantics, but the complete fg vs. bg semantics.
                    bg_embs_i = cond_prompt_embeddings[i, :, bg_indices_i_N].permute(1, 0, 2)
                    # subj_subj_embs_gs2_i: [1,  16, 768].
                    subj_subj_embs_gs2_i = subj_subj_embs_gs2[i].expand_as(bg_embs_i)
                    # Penalize any positive coeffs within bg_subj_embs_align_coeffs_i: [4, 16].
                    loss_inst_bg_subj_embs_align \
                        = calc_ref_cosine_loss(bg_embs_i, subj_subj_embs_gs2_i,
                                               exponent=2, do_demean_first=True, 
                                               first_n_dims_to_flatten=2, 
                                               ref_grad_scale=1, aim_to_align=False)
                    
                    loss_bg_subj_embs_align += loss_inst_bg_subj_embs_align

            loss_bg_subj_embs_align /= SSB_SIZE

        return loss_padding_subj_embs_align, loss_padding_cls_embs_align, loss_bg_subj_embs_align

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            # The code below doesn't work. Since the ddim_sampler API has been revised.
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @rank_zero_only
    def cache_and_log_generations(self, samples, img_colors, max_cache_size=48):
        self.generation_cache.append(samples)
        self.generation_cache_img_colors.append(img_colors)
        self.num_cached_generations += len(samples)

        if self.num_cached_generations >= max_cache_size:
            grid_folder = self.logger._save_dir + f'/samples'
            os.makedirs(grid_folder, exist_ok=True)
            grid_filename = grid_folder + f'/{self.cache_start_iter:04d}-{self.global_step:04d}.png'
            save_grid(self.generation_cache, self.generation_cache_img_colors, 
                      grid_filename, 12, do_normalize=True)
            print(f"{self.num_cached_generations} generations saved to {grid_filename}")
            
            # Clear the cache. If num_cached_generations > max_cache_size,
            # some samples at the end of the cache will be discarded.
            self.generation_cache = []
            self.generation_cache_img_colors = []
            self.num_cached_generations = 0
            self.cache_start_iter = self.global_step + 1

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch['caption'])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
            
            uc = self.get_learned_conditioning(N * [""])
            sample_scaled, _ = self.sample_log(cond=c, 
                                               batch_size=N, 
                                               ddim=use_ddim, 
                                               ddim_steps=ddim_steps,
                                               eta=ddim_eta,                                                 
                                               unconditional_guidance_scale=5.0,
                                               unconditional_conditioning=uc)
            log["samples_scaled"] = self.decode_first_stage(sample_scaled)

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    # configure_optimizers() is called later as a hook function by pytorch_lightning.
    # call stack: main.py: trainer.fit()
    # ...
    # pytorch_lightning/core/optimizer.py:
    # optim_conf = model.trainer._call_lightning_module_hook("configure_optimizers", pl_module=model)
    def configure_optimizers(self):
        if self.optimizer_type == 'AdamW':
            OptimizerClass = torch.optim.AdamW
        elif self.optimizer_type == 'NAdam':
            # In torch 1.13, decoupled_weight_decay is not supported. 
            # But since we disabled weight decay, it doesn't matter.
            OptimizerClass = torch.optim.NAdam
        elif self.optimizer_type == 'Prodigy':
            OptimizerClass = Prodigy
        elif self.optimizer_type == 'ProdigyAdamW':
            OptimizerClass = Prodigy
        else:
            raise NotImplementedError()
            
        # self.learning_rate and self.weight_decay are set in main.py.
        # self.learning_rate = base_learning_rate * 2, 2 is the batch size.
        lr      = self.learning_rate
        scheduler = None
        prodigy_adam_opt = None
        prodigy_adamw_scheduler = None

        # If using textual inversion, then embedding_manager is not None.
        if self.embedding_manager is not None: 
            embedding_params_with_lr_ratios = self.embedding_manager.optimized_parameters()

            embedding_params_with_lrs = []
            for param_group in embedding_params_with_lr_ratios:
                params = param_group['params']
                param_lr = lr * param_group['lr_ratio']
                # Not sure if it's necessary to set requires_grad=True here.
                #for param in params:
                #    param.requires_grad = True
                    
                embedding_params_with_lrs.append( {'params': params, 'lr': param_lr, 
                                                   'excluded_from_prodigy': param_group['excluded_from_prodigy']} )

            # unfreeze_model:
            # Are we allowing the base model to train? If so, set two different parameter groups.
            if self.unfreeze_model: 
                model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
                opt_params_with_lrs = embedding_params_with_lrs + [ {"params": model_params, "lr": self.model_lr,
                                                                    "excluded_from_prodigy": False} ]
            else:
                # Otherwise, train only embeddings
                opt_params_with_lrs = embedding_params_with_lrs
            
            if 'Prodigy' not in self.optimizer_type:
                opt = OptimizerClass(opt_params_with_lrs, weight_decay=self.weight_decay,
                                     betas=self.adam_config.betas)
                
                assert 'target' in self.adam_config.scheduler_config
                self.adam_config.scheduler_config.params.max_decay_steps = self.trainer.max_steps
                lambda_scheduler = instantiate_from_config(self.adam_config.scheduler_config)
                print("Setting up LambdaLR scheduler...")
                scheduler = LambdaLR(opt, lr_lambda=lambda_scheduler.schedule)

            else:
                prodigy_params = [ param_group['params'] for param_group in opt_params_with_lrs \
                                   if not param_group['excluded_from_prodigy'] ]
                prodigy_params = sum(prodigy_params, [])

                if self.do_zero_shot:
                    # [0.9, 0.999]. Converge more slowly.
                    betas = self.prodigy_config.zs_betas
                else:
                    # [0.985, 0.993]. Converge faster.
                    betas = self.prodigy_config.betas

                # Prodigy uses an LR = 1.
                opt = OptimizerClass(prodigy_params, lr=1., weight_decay=self.weight_decay,
                                     betas=betas,   # default: [0.985, 0.993]
                                     d_coef=self.prodigy_config.d_coef, # default: 5
                                     safeguard_warmup= self.prodigy_config.scheduler_cycles > 1, 
                                     use_bias_correction=True)

                total_cycle_steps  = self.trainer.max_steps - self.prodigy_config.warm_up_steps
                transition_milestones = [self.prodigy_config.warm_up_steps]
                # Since factor=1, we don't need to make sure the last step of the scheduler is called,
                # which restores the LR to the original value.
                warmup_scheduler    = ConstantLR(opt, factor=1., total_iters=self.prodigy_config.warm_up_steps)
                num_scheduler_cycles = self.prodigy_config.scheduler_cycles
                if self.prodigy_config.scheduler_type == 'CyclicLR':
                    # CyclicLR will do a downward half-cycle first. So we subtract 0.5
                    # from num_scheduler_cycles. If self.prodigy_config.scheduler_cycles = 2,
                    # then num_scheduler_cycles = 1.5, which means there'll be an extra up-down cycle.
                    num_scheduler_cycles -= 0.5

                # single_cycle_steps = 750, if max_steps = 2000, warm_up_steps = 500 and scheduler_cycles = 2.
                single_cycle_steps  = total_cycle_steps / num_scheduler_cycles
                last_cycle_steps    = total_cycle_steps - single_cycle_steps * (num_scheduler_cycles - 1)
                schedulers = [warmup_scheduler]
                print(f"Setting up {num_scheduler_cycles} * {single_cycle_steps} cycles, {self.prodigy_config.warm_up_steps} warm up steps.")

                if self.prodigy_config.scheduler_type == 'Linear':
                    num_scheduler_cycles = int(num_scheduler_cycles)
                    for c in range(num_scheduler_cycles):
                        if c == num_scheduler_cycles - 1:
                            # The last cycle.
                            cycle_steps = last_cycle_steps
                        else:
                            cycle_steps = single_cycle_steps
                            transition_milestones.append(transition_milestones[-1] + cycle_steps)

                        # total_iters = second_phase_steps * 1.1, so that the LR is reduced to 0.1/1.1 = 0.09
                        # of the full LR at the end.
                        linear_cycle_scheduler = PolynomialLR(opt, power=1,
                                                              total_iters=cycle_steps * 1.1)
                        schedulers.append(linear_cycle_scheduler)
                elif self.prodigy_config.scheduler_type == 'CosineAnnealingWarmRestarts':
                    # eta_min should be 0.1 instead of 0.1 * LR, since the full LR is 1 for Prodigy.
                    schedulers.append(CosineAnnealingWarmRestarts(opt, T_0=int(single_cycle_steps), T_mult=1, 
                                                                  eta_min=0.1,
                                                                  last_epoch=-1))
                elif self.prodigy_config.scheduler_type == 'CyclicLR':
                    # step_size_up = step_size_down = single_cycle_steps / 2 (float).
                    # last_epoch will be updated to single_cycle_steps / 2 in training_step(), 
                    # so that the LR begins with max_lr.
                    # We can't initialize it here, since SequentialLR will manually call 
                    # scheduler.step(0) at the first iteration, which will set the last_epoch to 0.
                    # Therefore, after the first scheduler.step(), we set the last_epoch of CyclicLR 
                    # to single_cycle_steps / 2.
                    schedulers.append(CyclicLR(opt, base_lr=0.1, max_lr=1, step_size_up=single_cycle_steps / 2,
                                               last_epoch=single_cycle_steps / 2 - 1, cycle_momentum=False))
                    # Disable SequentialLR2 from calling scheduler.step(0) at the first iteration, which will 
                    # set the last_epoch of CyclicLR to 0.
                    schedulers[-1].start_from_epoch_0 = False

                #print(scheduler._schedulers[-1].get_lr())
                else:
                    raise NotImplementedError()
                
                scheduler = SequentialLR2(opt, schedulers=schedulers,
                                          milestones=transition_milestones)

                if self.optimizer_type == 'ProdigyAdamW':
                    # If using ProdigyAdamW, AdamW is initialized with param group opt_params_with_lrs.
                    # It's different from Prodigy, which is initialized with param group opt_params.
                    # So the two data structures won't interfere with each other.
                    prodigy_adam_opt = torch.optim.AdamW(opt_params_with_lrs, weight_decay=self.weight_decay,
                                                         betas=self.adam_config.betas)
                    # Use AdamW with LR = self.learning_rate / 10000, in the beginning half of the total steps.
                    # This is to warm-up the AdamW optimizer, and delegate the optimization to Prodigy.
                    # total_iters = adamw_kickin_iter - 2: 
                    # at the last iteration, ConstantLR will revert the 
                    # LR to the original values. If total_iters > milestones of SequentialLR, this step
                    # will not be called, and the LR of AdamW will be kept at very small values, which 
                    # is not intended. Therefore we set total_iters=adamw_kickin_iter - 2.
                    # It will revert the LRs to initial LRs before the last cycle of Prodigy.
                    adamw_kickin_iter   = self.trainer.max_steps - last_cycle_steps
                    zero_scheduler      = ConstantLR(prodigy_adam_opt, factor=1e-4,  
                                                     total_iters=adamw_kickin_iter - 2)
                    # Enable adamw optimizer in the last cycle of Prodigy.
                    # max_LR = lr / 4. Initial LR = max_lr / div_factor = lr / 40. 
                    # No need to use a smaller initial LR, as AdamW has been warmed-up 
                    # (operating with LR=0 for half of the total steps).
                    # pct_start=0.3 (default). If max_steps = 2000, then total_steps = 1000, 
                    # max_lr is achieved at relative step = 300 (absolute step = 1300).
                    # final_div_factor = initial LR / 1 = lr / 40.
                    onecycle_scheduler  = OneCycleLR(prodigy_adam_opt, max_lr=lr / 4, 
                                                     total_steps=last_cycle_steps,
                                                     div_factor=10, final_div_factor=1)
                    prodigy_adamw_scheduler = SequentialLR(prodigy_adam_opt, schedulers=[zero_scheduler, onecycle_scheduler], 
                                                           milestones=[adamw_kickin_iter])

        else:
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.append(self.logvar)

            opt = OptimizerClass(params, lr=lr, weight_decay=self.weight_decay,
                                 betas=self.adam_config.betas)

            assert 'target' in self.adam_config.scheduler_config
            self.adam_config.scheduler_config.params.max_decay_steps = self.trainer.max_steps

            lambda_scheduler = instantiate_from_config(self.adam_config.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaLR(opt, lr_lambda=lambda_scheduler.schedule)

        if scheduler is None:
            return opt
        
        optimizers = [ {'optimizer': opt, 'frequency': 1, 
                        'lr_scheduler': {
                            'scheduler': scheduler,
                            'interval': 'step', # No need to specify in config yaml.
                            'frequency': 1
                        }} ]

        if self.optimizer_type == 'ProdigyAdamW':
            optimizers.append(
                {
                    'optimizer': prodigy_adam_opt, 'frequency': 1, 
                    'lr_scheduler': {
                        'scheduler': prodigy_adamw_scheduler,
                        'interval': 'step',
                        'frequency': 1
                    }
                })

        # self.scheduler = schedulers[0]['scheduler']
        return optimizers


    # configure_opt_embedding() is never called.
    def configure_opt_embedding(self):
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.embedding_manager.optimized_parameters():
            param.requires_grad = True

        lr = self.learning_rate
        params = list(self.embedding_manager.optimized_parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    # configure_opt_model() is never called.
    def configure_opt_model(self):
        for param in self.cond_stage_model.parameters():
            param.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.embedding_manager.optimized_parameters():
            param.requires_grad = True

        model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
        embedding_params = list(self.embedding_manager.optimized_parameters())
        return torch.optim.AdamW([{"params": embedding_params, "lr": self.learning_rate}, {"params": model_params}], lr=self.model_lr)

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):

        if not self.unfreeze_model: # If we are not tuning the model itself, zero-out the checkpoint content to preserve memory.
            checkpoint.clear()
        
        if os.path.isdir(self.trainer.checkpoint_callback.dirpath) and self.embedding_manager is not None:
            self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, "embeddings.pt"))
            self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, f"embeddings_gs-{self.global_step}.pt"))

class Arc2FaceWrapper(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        from diffusers import UNet2DConditionModel
        from arc2face.arc2face import CLIPTextModelWrapper
        self.unet = UNet2DConditionModel.from_pretrained(
                        #"runwayml/stable-diffusion-v1-5", subfolder="unet"
                        'arc2face/models', subfolder="arc2face", torch_dtype=torch.float16
                    )
        self.text_encoder = CLIPTextModelWrapper.from_pretrained(
                            'arc2face/models', subfolder="encoder", torch_dtype=torch.float16
                            )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def gen_rand_arc2face_id_prompt_embs(self, batch_size, gen_neg_prompt=False):
        # Returns faceid_embeds, arc2face_pos_prompt_emb.
        return get_arc2face_id_prompt_embs(None, self.tokenizer, self.text_encoder,
                                           image_folder=None, image_paths=None,
                                           images_np=None, example_image_count=0,
                                           out_image_count=batch_size,
                                           device=self.device,
                                           input_max_length=21, # Remove all paddings.
                                           rand_face=True, noise_level=0, 
                                           gen_neg_prompt=gen_neg_prompt, verbose=False)
    
    # Keep compatible with the original openaimodel.UNetModel:forward().
    # Only used for inference/distillation, so no_grad() is used.
    @torch.no_grad()
    def forward(self, x, timesteps=None, context=None, y=None, 
                context_in=None, extra_info=None, batch_contains_neg_instances=False, 
                do_cfg=False, cfg_scale=3.0):
        if do_cfg and not batch_contains_neg_instances:
            breakpoint()

        with torch.autocast(device_type='cuda', dtype=torch.float16):            
            if batch_contains_neg_instances:
                if context.shape[0] == x.shape[0] * 16 or context.shape[0] == x.shape[0] * 32:
                    # Batch contains half positive, half negative instances.
                    pos_context, neg_context = context.chunk(2, dim=0)
                    # If we intercept layerwise prompts here, then context.shape[0] == x.shape[0] * 16.
                    # Otherwise, it's ordinary prompts, and context.shape[0] == x.shape[0].
                    pos_context = pos_context.reshape(-1, 16, *context.shape[1:]).mean(dim=1)
                    neg_context = neg_context.reshape(-1, 16, *context.shape[1:]).mean(dim=1)
                    context = torch.cat([pos_context, neg_context], dim=0)
                # Repeat x to match negative context.
                # If context.shape[0] == x.shape[0] * 32 above, then now context.shape[0] == x.shape[0] * 2.
                if context.shape[0] == x.shape[0] * 2:
                    x = x.repeat(2, 1, 1, 1)
                    timesteps = timesteps.repeat(2)
                else:
                    breakpoint()
            else:
                # Batch only contains positive instances. No need to split.
                # If do_arc2face_distill, then context is [BS=6, 21, 768]. The condition below is False.
                # Either context.shape[0] == x.shape[0] * 16, or context.shape[0] == x.shape[0].
                if context.shape[0] == x.shape[0] * 16:
                    context = context.reshape(-1, 16, *context.shape[1:]).mean(dim=1)
                # Otherwise, we don't need to do anything.
                    
            noise_pred = self.unet(sample=x, timestep=timesteps, encoder_hidden_states=context,
                                   return_dict=False)[0]
            if do_cfg:
                pos_noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
                # Classifier-free guidance, but the predicted noise is scaled down by cfg_scale,
                # to match the magnitude of the original UNet predicted noise.
                noise_pred = pos_noise_pred + neg_noise_pred * (1 - cfg_scale) / cfg_scale

        return noise_pred
    
class DiffusionWrapper(pl.LightningModule): 
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # diffusion_model: UNetModel
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    # t: a 1-D batch of timesteps (during training: randomly sample one timestep for each instance).
    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            # For textual inversion, there's usually only one element tensor in c_crossattn.
            # So we take c_crossattn[0] directly, instead of torch.cat() on a list of tensors.
            if isinstance(c_crossattn[0], tuple):
                c_static_emb, c_in, extra_info = c_crossattn[0]
            else:
                c_static_emb    = c_crossattn[0]
                c_in            = None
                extra_info      = None
            # c_static_emb = torch.cat(c_crossattn, 1)
            # self.diffusion_model: UNetModel.
            #if 'iter_type' in extra_info:
            #    print(extra_info['iter_type'], c_in)
            out = self.diffusion_model(x, t, context=c_static_emb, context_in=c_in, extra_info=extra_info)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs
