import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR, ConstantLR, PolynomialLR, \
                                     CosineAnnealingWarmRestarts, CyclicLR
from ldm.modules.lr_scheduler import SequentialLR2
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_only
import bitsandbytes as bnb
from diffusers import UNet2DConditionModel

from ldm.util import    exists, default, instantiate_from_config, disabled_train, \
                        ortho_subtract, ortho_l2loss, gen_gradient_scaler, calc_ref_cosine_loss, \
                        calc_delta_alignment_loss, calc_prompt_emb_delta_loss, calc_elastic_matching_loss, \
                        save_grid, normalize_dict_values, normalized_sum, masked_mean, \
                        init_x_with_fg_from_training_image, resize_mask_for_feat_or_attn, convert_attn_to_spatial_weight, \
                        sel_emb_attns_by_indices, distribute_embedding_to_M_tokens_by_dict, \
                        join_dict_of_indices_with_key_filter, repeat_selected_instances, halve_token_indices, \
                        double_token_indices, merge_cls_token_embeddings, anneal_perturb_embedding, \
                        probably_anneal_t, count_optimized_params, count_params

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from ldm.prodigy import Prodigy
from ldm.ortho_nesterov import CombinedOptimizer, OrthogonalNesterov
from ldm.ademamix import AdEMAMix
from ldm.ademamix_shampoo import AdEMAMixDistributedShampoo

from adaface.unet_teachers import create_unet_teacher

import copy
from functools import partial
import random
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
import sys

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 monitor=None,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,    # clip the range of denoised variables, not the CLIP model.
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 unfreeze_unet=False,
                 unet_lr=0.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 optimizer_type='Prodigy',
                 grad_clip=0.5,
                 adam_config=None,
                 prodigy_config=None,
                 comp_distill_iter_gap=-1,
                 cls_subj_mix_scale=0.8,
                 prompt_emb_delta_reg_weight=0.,
                 comp_prompt_distill_weight=1e-4,
                 comp_fg_bg_preserve_loss_weight=0.,
                 fg_bg_complementary_loss_weight=0.,
                 fg_bg_xlayer_consist_loss_weight=0.,
                 enable_background_token=True,
                 # 'face portrait' is only valid for humans/animals. 
                 # On objects, use_fp_trick will be ignored, even if it's set to True.
                 use_fp_trick=True,
                 normalize_ca_q_and_outfeat=True,
                 p_unet_distill_iter=0,
                 unet_teacher_types=None,
                 max_num_unet_distill_denoising_steps=5,
                 p_unet_teacher_uses_cfg=0.6,
                 unet_teacher_cfg_scale_range=[1.3, 2],
                 max_num_comp_distill_init_prep_denoising_steps=6,
                 p_unet_distill_uses_comp_prompt=0.1,
                 id2img_prompt_encoder_lr_ratio=0.001,
                 extra_unet_dirpaths=None,
                 unet_weights=None,
                 p_gen_id2img_rand_id=0.4,
                 p_perturb_face_id_embs=0.6,
                 perturb_face_id_embs_std_range=[0.5, 1.5],
                 extend_prompt2token_proj_attention_multiplier=1,
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

        self.comp_distill_iter_gap                  = comp_distill_iter_gap
        self.prompt_emb_delta_reg_weight            = prompt_emb_delta_reg_weight
        self.comp_prompt_distill_weight             = comp_prompt_distill_weight
        self.comp_fg_bg_preserve_loss_weight        = comp_fg_bg_preserve_loss_weight
        self.fg_bg_complementary_loss_weight        = fg_bg_complementary_loss_weight
        self.fg_bg_xlayer_consist_loss_weight       = fg_bg_xlayer_consist_loss_weight
        # mix some of the subject embedding denoising results into the class embedding denoising results for faster convergence.
        # Otherwise, the class embeddings are too far from subject embeddings (person, man, woman), 
        # posing too large losses to the subject embeddings.
        self.cls_subj_mix_scale                     = cls_subj_mix_scale

        self.enable_background_token                = enable_background_token
        self.use_fp_trick                           = use_fp_trick
        self.normalize_ca_q_and_outfeat             = normalize_ca_q_and_outfeat
        self.p_unet_distill_iter                    = p_unet_distill_iter if self.training else 0
        self.unet_teacher_types                     = list(unet_teacher_types) if unet_teacher_types is not None else None
        self.p_unet_teacher_uses_cfg                = p_unet_teacher_uses_cfg
        self.unet_teacher_cfg_scale_range           = unet_teacher_cfg_scale_range
        self.max_num_unet_distill_denoising_steps   = max_num_unet_distill_denoising_steps
        # Sometimes we use the subject compositional prompts as the distillation target on a UNet ensemble teacher.
        # If unet_teacher_types == ['arc2face'], then p_unet_distill_uses_comp_prompt == 0, i.e., we
        # never use the compositional prompts as the distillation target of arc2face.
        # If unet_teacher_types is ['consistentID', 'arc2face'], then p_unet_distill_uses_comp_prompt == 0.1.
        # If unet_teacher_types == ['consistentID'], then p_unet_distill_uses_comp_prompt == 0.2.
        # NOTE: If compositional iterations are enabled, then we don't do unet distillation on the compositional prompts.
        if self.unet_teacher_types != ['arc2face'] and self.comp_distill_iter_gap <= 0:
            self.p_unet_distill_uses_comp_prompt = p_unet_distill_uses_comp_prompt
        else:
            self.p_unet_distill_uses_comp_prompt = 0
        self.id2img_prompt_encoder_lr_ratio         = id2img_prompt_encoder_lr_ratio
        self.extra_unet_dirpaths                    = extra_unet_dirpaths
        self.unet_weights                           = unet_weights
        
        self.p_gen_id2img_rand_id                   = p_gen_id2img_rand_id
        self.p_perturb_face_id_embs                 = p_perturb_face_id_embs
        self.perturb_face_id_embs_std_range         = perturb_face_id_embs_std_range

        self.max_num_comp_distill_init_prep_denoising_steps = max_num_comp_distill_init_prep_denoising_steps

        self.extend_prompt2token_proj_attention_multiplier = extend_prompt2token_proj_attention_multiplier
        self.comp_init_fg_from_training_image_count  = 0

        self.cached_inits = {}
        self.do_prompt_emb_delta_reg = (self.prompt_emb_delta_reg_weight > 0)
        
        self.init_iteration_flags()

        self.model = DiffusionWrapper(unet_config)

        count_params(self.model, verbose=True)

        self.optimizer_type = optimizer_type
        self.adam_config = adam_config
        self.grad_clip = grad_clip
    
        if 'Prodigy' in self.optimizer_type:
            self.prodigy_config = prodigy_config

        self.training_percent = 0.
        self.non_comp_iter_counter = 0

        self.unfreeze_unet = unfreeze_unet
        self.unet_lr = unet_lr

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

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
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
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

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

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

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def init_iteration_flags(self):
        self.iter_flags = { 'calc_clip_loss':               False,
                            'do_normal_recon':              True,
                            'do_unet_distill':              False,
                            'gen_id2img_rand_id':           False,
                            'id2img_prompt_embs':           None,
                            'id2img_neg_prompt_embs':       None,
                            'perturb_face_id_embs':         False,
                            'faceless_img_count':           0,
                            'num_denoising_steps':          1,
                            'do_comp_prompt_distillation':  False,
                            'do_prompt_emb_delta_reg':      False,
                            'unet_distill_uses_comp_prompt': False,
                            'use_background_token':         False,
                            'use_fp_trick':                 False,
                            'comp_init_fg_from_training_image': False,
                          }
        
    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    def shared_step(self, batch):
        raise NotImplementedError("shared_step() is not implemented in DDPM.")

    def training_step(self, batch, batch_idx):
        self.init_iteration_flags()
        self.training_percent = self.global_step / self.trainer.max_steps
        
        # NOTE: No need to have standalone ada prompt delta reg, 
        # since each prompt mix reg iter will also do ada prompt delta reg.

        # If N_CAND_REGS == 0, then no prompt distillation/regularizations, 
        # and the flags below take the default False value.
        if self.comp_distill_iter_gap > 0 and self.global_step % self.comp_distill_iter_gap == 0:
            self.iter_flags['do_comp_prompt_distillation']  = True
            self.iter_flags['do_normal_recon']              = False
            self.iter_flags['do_unet_distill']              = False
            self.iter_flags['do_prompt_emb_delta_reg']      = self.do_prompt_emb_delta_reg
        else:
            self.iter_flags['do_comp_prompt_distillation']  = False
            self.non_comp_iter_counter += 1
            if self.non_comp_iter_counter % 2 == 0:
                self.iter_flags['do_normal_recon']  = False
                self.iter_flags['do_unet_distill']  = True
                # Disable do_prompt_emb_delta_reg during unet distillation.
                self.iter_flags['do_prompt_emb_delta_reg'] = False
            else:
                self.iter_flags['do_normal_recon']  = True
                self.iter_flags['do_unet_distill']  = False
                self.iter_flags['do_prompt_emb_delta_reg'] = self.do_prompt_emb_delta_reg

        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

# LatentDiffusion inherits from DDPM. So:
# LatentDiffusion.model = DiffusionWrapper(unet_config)
class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 personalization_config,
                 cond_stage_key="image",
                 embedding_manager_trainable=True,
                 concat_mode=True,
                 cond_stage_forward=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):

        self.scale_by_std = scale_by_std
        # for backwards compatibility after implementation of DiffusionWrapper

        # cond_stage_config is a dict:
        # {'target': 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'}
        # Not sure why it's compared with a string

        # base_model_path and ignore_keys are popped from kwargs, so that they won't be passed to the base class DDPM.
        base_model_path = kwargs.pop("base_model_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(*args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.embedding_manager_trainable = embedding_manager_trainable

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

        self.restarted_from_ckpt = (base_model_path is not None)
        # base_model_path is popped from kwargs, so that it won't be passed to the base class DDPM.
        # As a result, the model weight is only loaded here, not in DDPM.
        if base_model_path is not None:
            self.init_from_ckpt(base_model_path, ignore_keys)
        
        if self.p_unet_distill_iter > 0 and self.unet_teacher_types is not None:
            # When unet_teacher_types == 'unet_ensemble' or unet_teacher_types contains multiple values,
            # device, unets, extra_unet_dirpaths and unet_weights are needed. 
            # Otherwise, they are not needed.
            self.unet_teacher = create_unet_teacher(self.unet_teacher_types, 
                                                    device='cpu',
                                                    unets=None,
                                                    extra_unet_dirpaths=self.extra_unet_dirpaths,
                                                    unet_weights=self.unet_weights,
                                                    p_uses_cfg=self.p_unet_teacher_uses_cfg,
                                                    cfg_scale_range=self.unet_teacher_cfg_scale_range)
        else:
            self.unet_teacher = None

        if self.comp_distill_iter_gap > 0:
            # Use RealisticVision unet to prepare x_start for compositional distillation, since it's more compositional.
            unet = UNet2DConditionModel.from_pretrained('models/ensemble/rv4-unet', torch_dtype=torch.float16)
            # comp_distill_unet is a diffusers unet used to do a few steps of denoising 
            # on the compositional prompts, before the actual compositional distillation.
            # So float16 is sufficient.
            self.comp_distill_init_prep_unet = \
                create_unet_teacher('unet_ensemble', 
                                    # A trick to avoid creating multiple UNet instances.
                                    unets = [unet, unet],
                                    unet_types=None,
                                    extra_unet_dirpaths=None,
                                    # When aggregating the results of using subject embeddings vs. class embeddings,
                                    # we give more weights to the class embeddings for better compositionality.
                                    # NOTE: subject embeddings first, then class embeddings.
                                    unet_weights = [1 - self.cls_subj_mix_scale, self.cls_subj_mix_scale],
                                    p_uses_cfg=1, # Always uses CFG for init prep denoising.
                                    cfg_scale_range=[2, 4],
                                    torch_dtype=torch.float16)
            self.comp_distill_init_prep_unet.train = disabled_train

        # cond_stage_model = FrozenCLIPEmbedder training = False.
        # We never train the CLIP text encoder. So disable the training of the CLIP text encoder.
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        if self.unfreeze_unet:
            self.model.train()
            embed_param_count = 0
            trainable_param_count = 0
            excluded_key_pats = [ 'time_embed', 'emb_layers', 'input_blocks' ]
            for key, param in self.model.named_parameters():
                # Freeze embedding layers. Finetune other parameters.
                if any([pat in key for pat in excluded_key_pats]):
                    param.requires_grad = False
                    embed_param_count += 1
                else:
                    param.requires_grad = True
                    trainable_param_count += 1
            print(f"Freeze {embed_param_count} embedding parameters, train {trainable_param_count} parameters.")

        else:
            # self.model = DiffusionWrapper() training = False.
            # If not unfreeze_unet, then disable the training of the UNetk, 
            # and only train the embedding_manager.
            self.model.eval()
            self.model.train = disabled_train
            for param in self.model.parameters():
                param.requires_grad = False

        self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)
        if self.embedding_manager_trainable:
            # embedding_manager contains subj_basis_generator, which is based on extended CLIP image encoder,
            # which has attention dropout. Therefore setting embedding_manager.train() is necessary.
            self.embedding_manager.train()
        self.num_id_vecs                = self.embedding_manager.id2ada_prompt_encoder.num_id_vecs
        self.num_static_img_suffix_embs = self.embedding_manager.id2ada_prompt_encoder.num_static_img_suffix_embs

        self.generation_cache = []
        self.generation_cache_img_colors = []
        self.cache_start_iter = 0
        self.num_cached_generations = 0

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == 0:
            # Make the behavior deterministic for debugging purposes.
            # In normal runs, disable this statement.
            #random.seed(10000)
            self.num_teachable_iters = 0
            self.num_reuse_teachable_iters = 0
            # uncond_context is a tuple of (uncond_emb, uncond_c_in, extra_info).
            # uncond_context[0]: [1, 77, 768].
            self.uncond_context         = self.get_text_conditioning([""], text_conditioning_iter_type='plain_text_iter')
            # "photo of a" is the template of Arc2face. Including an extra BOS token, the length is 4.
            img_prompt_prefix_context   = self.get_text_conditioning(["photo of a"], text_conditioning_iter_type='plain_text_iter')
            # img_prompt_prefix_context: [1, 4, 768]. Abandon the remaining text paddings.
            self.img_prompt_prefix_embs = img_prompt_prefix_context[0][:1, :4]

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

    # We never train the VAE. So disable the training of the VAE.
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        assert config != '__is_first_stage__'
        assert config != '__is_unconditional__'
        # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
        self.cond_stage_model = instantiate_from_config(config)
        
    def instantiate_embedding_manager(self, config, text_embedder):
        model = instantiate_from_config(config, text_embedder=text_embedder)
        return model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    # Number of calls to get_text_conditioning() during training:
    # If do_comp_prompt_distillation, then 1 call on delta prompts (NOTE: delta prompts have a large batch size).
    # If do_normal_recon / do_unet_distll with delta loss, then 2 calls (one on delta prompts, one on subject single prompts). 
    # NOTE: the delta prompts consumes extram RAM.
    # If do_normal_recon / do_unet_distll without delta loss, then 1 call.
    # cond_in: a batch of prompts like ['an illustration of a dirty z', ...]
    # return_prompt_embs_type: ['text', 'id', 'text_id'].
    # 'text': default, the conventional text embeddings produced from the embedding manager.
    # 'id': the input subj_id2img_prompt_embs, generated by an ID2ImgPrompt module.
    # 'text_id': concatenate the text embeddings with the ID2ImgPrompt embeddings.
    # 'id' or 'text_id' are used when we want to evaluate the original ID2ImgPrompt module.
    def get_text_conditioning(self, cond_in, subj_id2img_prompt_embs=None, clip_bg_features=None, 
                              randomize_clip_weights=False, return_prompt_embs_type='text', 
                              text_conditioning_iter_type=None):
        # cond_in: a list of prompts: ['an illustration of a dirty z', 'an illustration of the cool z']
        # each prompt in c is encoded as [1, 77, 768].
        # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
        self.cond_stage_model.device = self.device
        if randomize_clip_weights:
            self.cond_stage_model.sample_last_layers_skip_weights()
            
        if text_conditioning_iter_type is None:
            if self.iter_flags['do_comp_prompt_distillation']:
                text_conditioning_iter_type = 'compos_distill_iter'
            elif self.iter_flags['do_unet_distill']:
                text_conditioning_iter_type = 'unet_distill_iter'
            else:
                # Even if return_prompt_embs_type == 'id' or 'text_id', we still
                # generate the conventional text embeddings.
                # Therefore, text_conditioning_iter_type is set to 'recon_iter'.
                text_conditioning_iter_type = 'recon_iter'

        # Update subj_id2img_prompt_embs, clip_bg_features to be used to generate ada embeddings in the embedding manager.
        # If the prompt is "" (negative prompt), then clip_bg_features is None.
        # Also update the iteration type of the embedding manager, according to the arguments passed in.
        self.embedding_manager.set_image_prompts_and_iter_type(subj_id2img_prompt_embs, clip_bg_features,
                                                               text_conditioning_iter_type)

        # prompt_embeddings: [B, 77, 768]
        prompt_embeddings = self.cond_stage_model.encode(cond_in, embedding_manager=self.embedding_manager)

        if self.training:
            # If cls_delta_string_indices is not empty, then it must be a compositional 
            # distillation iteration, and placeholder_indices only contains the indices of the subject 
            # instances. Whereas cls_delta_string_indices only contains the indices of the
            # class instances. Therefore, cls_delta_string_indices is used here.
            prompt_embeddings = merge_cls_token_embeddings(prompt_embeddings, 
                                                           self.embedding_manager.cls_delta_string_indices)
            
        # return_prompt_embs_type: ['text', 'id', 'text_id']. Default: 'text', i.e., 
        # the conventional text embeddings returned by the clip encoder (embedding manager in the middle).
        if return_prompt_embs_type in ['id', 'text_id']:
            # if text_conditioning_iter_type == 'plain_text_iter', the current prompt is a plain text (e.g., a negative prompt).
            # So subj_id2img_prompt_embs serves as the negative ID prompt embeddings (only applicable to ConsistentID).
            # NOTE: These are not really negative ID embeddings, and is just to make the prompt to have the correct length.
            if text_conditioning_iter_type == 'plain_text_iter' and subj_id2img_prompt_embs is None:
                if return_prompt_embs_type == 'id':
                    # If subj_id2img_prompt_embs is used as standalone negative embeddings,
                    # then we take the beginning N embeddings of prompt_embeddings.
                    subj_id2img_prompt_embs = prompt_embeddings[:, :self.num_id_vecs, :]
                else:
                    # If subj_id2img_prompt_embs is to be postpended to the negative embeddings,
                    # then we take the ending N embeddings of prompt_embeddings,
                    # to avoid two BOS tokens appearing in the same prompt.
                    subj_id2img_prompt_embs = prompt_embeddings[:, -self.num_id_vecs:, :]
                # Since subj_id2img_prompt_embs is taken from a part of prompt_embeddings,
                # we don't need to repeat it.

            elif subj_id2img_prompt_embs is not None:
                # During training, subj_id2img_prompt_embs is CLIP(Ada(id_img_prompt, static_img_suffix_embs)).
                assert subj_id2img_prompt_embs.shape[1] == self.num_id_vecs + self.num_static_img_suffix_embs
                # subj_id2img_prompt_embs is the embedding generated by the ID2ImgPrompt module. 
                # NOTE: It's not the inverse embeddings. return_prompt_embs_type is enabled only 
                # when we wish to evaluate the original ID2ImgPrompt module.
                # subj_id2img_prompt_embs: [1, 4, 768] or [1, 16, 768]. 
                # Need to repeat BS times for BS instances.
                BS_repeat = len(cond_in) // subj_id2img_prompt_embs.shape[0]
                # subj_id2img_prompt_embs: [1, 4, 768] or [1, 16, 768] => repeat to [BS, 4/16, 768].
                subj_id2img_prompt_embs = subj_id2img_prompt_embs.repeat(BS_repeat, 1, 1)
                
            if return_prompt_embs_type == 'id':
                # Only return the ID2ImgPrompt embeddings, and discard the text embeddings.
                prompt_embeddings = subj_id2img_prompt_embs
            elif return_prompt_embs_type == 'text_id':
                # NOTE: always append the id2img prompts to the end of the prompt embeddings
                # Arc2face doesn't care about the order of the prompts. But consistentID only works when
                # the id2img prompt embeddings are postpended to the end of the prompt embeddings.
                # prompt_embeddings: [BS, 81, 768]. 81: 77 + 4.
                prompt_embeddings = torch.cat([prompt_embeddings, subj_id2img_prompt_embs], dim=1)

        # Otherwise, inference and no special return_prompt_embs_type, we do nothing to the prompt_embeddings.

        # 'placeholder2indices' and 'prompt_emb_mask' are cached to be used in forward() and p_losses().
        extra_info = { 
                        'placeholder2indices':           copy.copy(self.embedding_manager.placeholder2indices),
                        'prompt_emb_mask':               copy.copy(self.embedding_manager.prompt_emb_mask),
                        # Will be updated to True in p_losses() when in compositional iterations.
                        'capture_ca_layers_activations':  False,
                     }

        c = (prompt_embeddings, cond_in, extra_info)

        return c

    # k: key for the images, i.e., 'image'. k is not a number.
    @torch.no_grad()
    def get_input(self, batch, k, bs=None):
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

        return z

    # output: -1 ~ 1.
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    # same as decode_first_stage() but without torch.no_grad() decorator
    # output: -1 ~ 1.
    def decode_first_stage_with_grad(self, z, predict_cids=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x, mask=None):
        return self.first_stage_model.encode(x, mask)

    # LatentDiffusion.shared_step() overloads DDPM.shared_step().
    # shared_step() is called in training_step() and (no_grad) validation_step().
    # In the beginning of an epoch, a few validation_step() is called. But I don't know why.
    # batch: { 'caption':               ['an illustration of a dirty z',                    
    #                                    'a depiction of a z'], 
    #          'subj_comp_prompt':     ['an illustration of a dirty z dancing with a boy', 
    #                                    'a depiction of a z kicking a punching bag'],
    #          'cls_single_prompt':     ['an illustration of a dirty person',          
    #                                    'a depiction of a person'],
    #                                    'a depiction of a person kicking a punching bag']
    #          'cls_comp_prompt'  :    ['an illustration of a dirty person dancing with a boy', 
    #                                    'a depiction of a person kicking a punching bag'],
    #          'image':   [2, 512, 512, 3] }
    # 'caption' is not named 'subj_single_prompt' to keep it compatible with older code.
    # ANCHOR[id=shared_step]
    def shared_step(self, batch):
        # Encode noise as 4-channel latent features.
        # first_stage_key="image"
        x_start = self.get_input(batch, self.first_stage_key)
        # Update the training_percent of embedding_manager.
        self.embedding_manager.training_percent = self.training_percent

        batch_have_fg_mask  = batch['has_fg_mask']
        # Temporarily disable fg_mask for debugging.
        disable_fg_mask = False #True
        if disable_fg_mask:
            batch_have_fg_mask[:] = False

        self.iter_flags['fg_mask_avail_ratio']  = batch_have_fg_mask.sum() / batch_have_fg_mask.shape[0]

        # If it's a compositional distillation iteration, only the first instance in the batch is used.
        # Therefore, self.batch_1st_subject_name is the only subject name in the batch.
        self.batch_1st_subject_name  = batch['subject_name'][0]
        self.batch_1st_subject_is_in_mix_subj_folder = batch['is_in_mix_subj_folder'][0]

        # NOTE: *_fp prompts are like "face portrait of ..." or "a portrait of ...". 
        # They highlight the face features compared to the normal prompts.
        # When doing compositional distillation on humans/animals they are a little bit better.
        # For objects, even if use_fp_trick = True, *_fp prompts are not available in batch, 
        # so fp_trick won't be used.
        if self.use_fp_trick and 'subj_single_prompt_fp' in batch:
            if self.iter_flags['do_comp_prompt_distillation'] :
                p_use_fp_trick = 0.7
            # If compositional distillation is enabled, then in normal recon iterations,
            # we use the fp_trick most of the time, to better reconstructing single-face input images.
            # However, we still keep 20% of the do_normal_recon iterations to not use the fp_trick,
            # to encourage a bias towards larger facial areas in the output images.
            elif self.iter_flags['do_normal_recon'] and self.comp_distill_iter_gap > 0:
                p_use_fp_trick = 0.8
            else:
                # If not doing compositional distillation, then use_fp_trick is disabled on do_normal_recon iterations,
                # so that the ID embeddings alone are expected to reconstruct the subject portraits.
                p_use_fp_trick = 0
        else:
            p_use_fp_trick = 0

        self.iter_flags['use_fp_trick'] = (torch.rand(1) < p_use_fp_trick).item()

        if self.iter_flags['do_comp_prompt_distillation'] and self.iter_flags['fg_mask_avail_ratio'] > 0:
            # If do_comp_prompt_distillation, comp_init_fg_from_training_image is always enabled.
            # It's OK, since when do_zero_shot, we have a large diverse set of training images,
            # and always initializing from training images won't lead to overfitting.
            p_comp_init_fg_from_training_image = 1
        else:
            p_comp_init_fg_from_training_image = 0

        self.iter_flags['comp_init_fg_from_training_image'] \
            = (torch.rand(1) < p_comp_init_fg_from_training_image).item()
        
        if self.iter_flags['do_unet_distill']:
            # If do_unet_distill, then only use the background tokens in a small percentage of the iterations.
            # Because for ConsistentID, the background is a bit noisy, but there has been 
            # 4 prompt embeddings serving as the background tokens to absorb the background noise.
            # For Arc2face, the background is simple and we probably don't need to absorb the 
            # background noise with background tokens.
            p_use_background_token  = 0.1
        elif self.iter_flags['do_normal_recon']:
            # We lower p_use_background_token from the previous value 0.9 to 0.3 to avoid the background token
            # taking too much of the foreground (i.e., capturing the subject features).
            p_use_background_token  = 0.3
        elif self.iter_flags['do_comp_prompt_distillation']:
            # When doing compositional distillation, the background is quite different between 
            # single prompts and comp prompts. So using a background token is probably not a good idea.
            p_use_background_token  = 0
        else:
            breakpoint()

        # enable_background_token is True, as long as background token is specified in 
        # the command line of train.py.
        self.iter_flags['use_background_token'] = self.enable_background_token \
                                                    and (torch.rand(1) < p_use_background_token)

        # ** use_fp_trick is only for compositional iterations. **
        if self.iter_flags['use_fp_trick'] and self.iter_flags['use_background_token']:
            SUBJ_SINGLE_PROMPT = 'subj_single_prompt_fp_bg'
            SUBJ_COMP_PROMPT   = 'subj_comp_prompt_fp_bg'
            CLS_SINGLE_PROMPT  = 'cls_single_prompt_fp_bg'
            CLS_COMP_PROMPT    = 'cls_comp_prompt_fp_bg'
        # use_fp_trick but not use_background_token.
        elif self.iter_flags['use_fp_trick']:
            # Never use_fp_trick for recon iters. So no need to have "caption_fp" or "caption_fp_bg".
            SUBJ_SINGLE_PROMPT = 'subj_single_prompt_fp'
            SUBJ_COMP_PROMPT   = 'subj_comp_prompt_fp'
            CLS_SINGLE_PROMPT  = 'cls_single_prompt_fp'
            CLS_COMP_PROMPT    = 'cls_comp_prompt_fp'
        # not use_fp_trick and use_background_token.
        elif self.iter_flags['use_background_token']:
            SUBJ_SINGLE_PROMPT = 'subj_single_prompt_bg'
            SUBJ_COMP_PROMPT   = 'subj_comp_prompt_bg'
            CLS_SINGLE_PROMPT  = 'cls_single_prompt_bg'
            CLS_COMP_PROMPT    = 'cls_comp_prompt_bg'
        # Either do_comp_prompt_distillation but not use_fp_trick_iter, 
        # or recon/unet_distill iters (not do_comp_prompt_distillation) and not use_background_token.
        # We don't use_fp_trick on training images. 
        else:
            SUBJ_SINGLE_PROMPT = 'subj_single_prompt'
            SUBJ_COMP_PROMPT   = 'subj_comp_prompt'
            CLS_COMP_PROMPT    = 'cls_comp_prompt'
            CLS_SINGLE_PROMPT  = 'cls_single_prompt'

        subj_single_prompts = batch[SUBJ_SINGLE_PROMPT]
        cls_single_prompts  = batch[CLS_SINGLE_PROMPT]
        subj_comp_prompts   = batch[SUBJ_COMP_PROMPT]
        cls_comp_prompts    = batch[CLS_COMP_PROMPT]
        captions            = subj_single_prompts

        delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

        if 'aug_mask' in batch:
            # img_mask is another name of aug_mask.
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

        print(f"Rank {self.trainer.global_rank}: {batch['subject_name']}")

        BS = len(batch['subject_name'])
        # If do_comp_prompt_distillation, we repeat the instances in the batch, 
        # so that all instances are the same.
        if self.iter_flags['do_comp_prompt_distillation']:
            # Change the batch to have the (1 subject image) * BS strcture.
            # "captions" and "delta_prompts" don't change, as different subjects share the same placeholder "z".
            # After image_unnorm is repeated, the extracted zs_clip_fgbg_features and face_id_embs, extracted from image_unnorm,
            # will be repeated automatically. Therefore, we don't need to manually repeat them later.
            batch['subject_name'], batch["image_path"], batch["image_unnorm"], x_start, img_mask, fg_mask, batch_have_fg_mask = \
                repeat_selected_instances(slice(0, 1), BS, batch['subject_name'], batch["image_path"], batch["image_unnorm"], 
                                          x_start, img_mask, fg_mask, batch_have_fg_mask)
            self.iter_flags['same_subject_in_batch'] = True
        else:
            self.iter_flags['same_subject_in_batch'] = False

        # do_unet_distill and random() < p_unet_distill_iter.
        # p_gen_id2img_rand_id: 0.4 if distilling on arc2face. 0.2 if distilling on consistentID,
        # 0.1 if distilling on jointIDs.
        if self.iter_flags['do_unet_distill'] and (torch.rand(1) < self.p_gen_id2img_rand_id):
            self.iter_flags['gen_id2img_rand_id'] = True
            self.batch_subject_names = [ "rand_id_to_img_prompt" ] * len(batch['subject_name'])
        else:
            self.iter_flags['gen_id2img_rand_id'] = False
            self.batch_subject_names = batch['subject_name']            

        batch_images_unnorm = batch["image_unnorm"]

        # images: 0~255 uint8 tensor [3, 512, 512, 3] -> [3, 3, 512, 512].
        images = batch["image_unnorm"].permute(0, 3, 1, 2).to(x_start.device)
        image_paths = batch["image_path"]

        # gen_id2img_rand_id. The recon/distillation is on random ID embeddings. So there's no ground truth input images.
        # Therefore, zs_clip_fgbg_features are not available and are randomly generated as well.
        # gen_id2img_rand_id implies (not do_comp_prompt_distillation).
        # NOTE: the faces generated with gen_id2img_rand_id are usually atypical outliers,
        # so adding a small proportion of them to the training data may help increase the authenticity on
        # atypical faces, but adding too much of them may harm the performance on typical faces.
        if self.iter_flags['gen_id2img_rand_id']:
            # FACE_ID_DIM: 512 for each encoder. 1024 for two encoders.
            # FACE_ID_DIM is the sum of all encoders' face ID dimensions.
            FACE_ID_DIM             = self.embedding_manager.id2ada_prompt_encoder.face_id_dim
            face_id_embs            = torch.randn(BS, FACE_ID_DIM, device=x_start.device)
            CLIP_DIM                = self.embedding_manager.id2ada_prompt_encoder.clip_embedding_dim
            # 514 is for fg and bg tokens (257 * 2). 
            # CLIP_DIM is the total dimension of the CLIP embeddings. If there are two encoders,
            # then CLIP_DIM is the sum of both encoders' CLIP dimensions.
            zs_clip_fgbg_features   = torch.randn(BS, 514, CLIP_DIM, device=x_start.device)
            # On random IDs, we don't need to consider img_mask and fg_mask.
            img_mask = None
            fg_mask  = None
            batch_have_fg_mask[:] = False
            # In a gen_id2img_rand_id iteration, simply denoise a totally random x_start.
            x_start = torch.randn_like(x_start)
            self.iter_flags['faceless_img_count'] = 0
            # A batch of random faces share no similarity with each other, so same_subject_in_batch is False.
            self.iter_flags['same_subject_in_batch'] = False

        # Not gen_id2img_rand_id. The recon/distillation is on real ID embeddings.
        # 'gen_id2img_rand_id' is only True in do_unet_distill iters.
        # So if not do_unet_distill, then this branch is always executed.
        #    If     do_unet_distill, then this branch is executed at 50% of the time.
        else:
            # If self.iter_flags['same_subject_in_batch']:  zs_clip_fgbg_features: [1, 514, 1280]. face_id_embs: [1, 512].
            # Otherwise:                                    zs_clip_fgbg_features: [3, 514, 1280]. face_id_embs: [3, 512].
            # If self.iter_flags['same_subject_in_batch'], then we average the zs_clip_fgbg_features and face_id_embs to get 
            # less noisy zero-shot embeddings. Otherwise, we use instance-wise zero-shot embeddings.
            # If do_comp_prompt_distillation, then we have repeated the instances in the batch, 
            # so that all instances are the same, and self.iter_flags['same_subject_in_batch'] == True.
            # ** We don't cache and provide zs_clip_neg_features later, as it is constant and
            # is cached in the FaceID2AdaPrompt object.
            faceless_img_count, face_id_embs, zs_clip_fgbg_features = \
                self.embedding_manager.id2ada_prompt_encoder.extract_init_id_embeds_from_images(\
                    images, image_paths, fg_mask.squeeze(1), skip_non_faces=False, 
                    calc_avg=False)

            # faceless_img_count: number of images in the batch in which no faces are detected.
            self.iter_flags['faceless_img_count'] = faceless_img_count
            # If there are faceless input images in the batch, then the face ID embeddings are randomly generated.
            # If do_normal_recon, then we have to change the iteration type to do_unet_distill. Otherwise there'll 
            # be a large recon error, as the face ID embeddings don't correspond to input images.
            # If this is an compositional distillation iteration, then it's OK to use random ID embeddings.
            if faceless_img_count > 0 and self.iter_flags['do_normal_recon']:
                self.iter_flags['do_normal_recon'] = False
                self.iter_flags['do_unet_distill'] = True
                # Disable do_prompt_emb_delta_reg during unet distillation.
                self.iter_flags['do_prompt_emb_delta_reg']  = False

        # get_batched_img_prompt_embs() encodes face_id_embs to id2img_prompt_embs.
        # results is (face_image_count, faceid_embeds, pos_prompt_embs, neg_prompt_embs).
        # if the encoder is arc2face, neg_prompt_embs is None.
        # If it's consistentID or jointIDs, neg_prompt_embs is not None.
        results = self.embedding_manager.id2ada_prompt_encoder.get_batched_img_prompt_embs(
                    images.shape[0], init_id_embs=face_id_embs, 
                    pre_clip_features=zs_clip_fgbg_features)
                    
        # id2img_prompt_embs, id2img_neg_prompt_embs: [4, 21, 768]
        # If UNet teacher is not consistentID, then id2img_neg_prompt_embs == None.
        id2img_prompt_embs, id2img_neg_prompt_embs = results[2], results[3]
        # During training, id2img_prompt_embs is float16, but x_start is float32.
        id2img_prompt_embs = id2img_prompt_embs.to(x_start.dtype)
        if id2img_neg_prompt_embs is not None:
            id2img_neg_prompt_embs = id2img_neg_prompt_embs.to(x_start.dtype)

        # If do_comp_prompt_distillation, then we don't add noise to the zero-shot ID embeddings, 
        # to avoid distorting the ID information.
        p_perturb_face_id_embs = self.p_perturb_face_id_embs if self.iter_flags['do_unet_distill'] else 0                
        # p_perturb_face_id_embs: default 0.6.
        # The overall prob of perturb_face_id_embs: (1 - 0.5) * 0.6 = 0.3.
        self.iter_flags['perturb_face_id_embs'] = (torch.rand(1) < p_perturb_face_id_embs).item()
        if self.iter_flags['perturb_face_id_embs']:
            if not self.iter_flags['same_subject_in_batch']:
                self.iter_flags['same_subject_in_batch'] = True
                # Change the ID features of multiple subjects in the batch to the ID features of 
                # the first subject, before adding noise to the ID features.
                # Doing so is similar to contrastive learning: the embeddings in a batch are similar
                # (the first subject embedding + randn noise), but the generated images are quite different.
                # Therefore, the model may learn to distinguish the tiny differences in the embeddings.
                # As the embeddings are coupled with x_start and fg_mask, we need to change them to 
                # the first subject's as well.
                # Change the batch to have the (1 subject image) * BS strcture.
                # "captions" and "delta_prompts" don't change, as different subjects share the same placeholder "z".
                # clip_bg_features is used by adaface encoder, so we repeat zs_clip_fgbg_features accordingly.
                # We don't repeat id2img_neg_prompt_embs, as it's constant and identical for different instances.
                x_start, batch_images_unnorm, img_mask, fg_mask, \
                batch_have_fg_mask, self.batch_subject_names, \
                id2img_prompt_embs, zs_clip_fgbg_features = \
                    repeat_selected_instances(slice(0, 1), BS, 
                                              x_start, batch_images_unnorm, img_mask, fg_mask, 
                                              batch_have_fg_mask, self.batch_subject_names, 
                                              id2img_prompt_embs, zs_clip_fgbg_features)
                
            # ** Perturb the zero-shot ID image prompt embeddings with probability 0.6. **
            # The noise is added to the image prompt embeddings instead of the initial face ID embeddings.
            # Because for ConsistentID, both the ID embeddings and the CLIP features are used to generate the image prompt embeddings.
            # Each embedding has different roles in depicting the facial features.
            # If we perturb both, we cannot guarantee their consistency and the perturbed faces may be quite distorted.
            # perturb_std_is_relative=True: The perturb_std is relative to the std of the last dim (512) of face_id_embs.
            # If the subject is not face, then face_id_embs is DINO embeddings. We can still add noise to them.
            # Keep the first ID embedding as it is, and add noise to the rest.
            # ** After perturbation, consistentID embeddings and arc2face embeddings are slightly inconsistent. **
            # Therefore, for jointIDs, we should reduce perturb_face_id_embs_std_range to [0.3, 0.6].
            id2img_prompt_embs[1:] = \
                anneal_perturb_embedding(id2img_prompt_embs[1:], training_percent=0, 
                                         begin_noise_std_range=self.perturb_face_id_embs_std_range, 
                                         end_noise_std_range=None, 
                                         perturb_prob=1, perturb_std_is_relative=True, 
                                         keep_norm=True, verbose=True)

        if self.iter_flags['do_unet_distill']:
            # Iterate among 1 ~ 5. We don't draw random numbers, so that different ranks have the same num_denoising_steps,
            # which would be faster for synchronization.
            # Note since comp_distill_iter_gap == 3 or 4, we should choose a number that is co-prime with 3 and 4.
            # Otherwise, some values, e.g., 0 and 3, will never be chosen.
            num_denoising_steps = self.global_step % 5 + 1
            self.iter_flags['num_denoising_steps'] = num_denoising_steps

            # Sometimes we use the subject compositional prompts as the distillation target on a UNet ensemble teacher.
            # If unet_teacher_types == ['arc2face'], then p_unet_distill_uses_comp_prompt == 0, i.e., we
            # never use the compositional prompts as the distillation target of arc2face.
            # If unet_teacher_types is ['consistentID', 'arc2face'], then p_unet_distill_uses_comp_prompt == 0.1.
            # If unet_teacher_types == ['consistentID'], then p_unet_distill_uses_comp_prompt == 0.2.
            if torch.rand(1) < self.p_unet_distill_uses_comp_prompt:
                self.iter_flags['unet_distill_uses_comp_prompt'] = True
                captions = batch[SUBJ_COMP_PROMPT]

            if num_denoising_steps > 1:
                # Only use the first 1/num_denoising_steps of the batch to avoid OOM.
                # If num_denoising_steps >= 2, BS == 1 or 2, then HALF_BS = 1.
                # If num_denoising_steps == 2 or 3, BS == 4, then HALF_BS = 2. 
                # If num_denoising_steps == 4 or 5, BS == 4, then HALF_BS = 1.
                HALF_BS = torch.arange(BS).chunk(num_denoising_steps)[0].shape[0]
                # Setting the minimal batch size to be 2 requires skipping 3 steps if num_denoising_steps == 6.
                # Seems doing so will introduce too much artifact. Therefore it's DISABLED.
                ## The batch size when doing multi-step denoising is at least 2. 
                ## But naively doing so when num_denoising_steps >= 3 may cause OOM.
                ## In that case, we need to discard the first few steps from loss computation.
                ## HALF_BS = max(2, HALF_BS)

                # REPEAT = 1 in repeat_selected_instances(), so that it **only selects** the 
                # first HALF_BS elements without repeating.
                # clip_bg_features is used by ConsistentID adaface encoder, 
                # so we repeat zs_clip_fgbg_features as well.
                x_start, batch_images_unnorm, img_mask, fg_mask, \
                batch_have_fg_mask, self.batch_subject_names, \
                zs_clip_fgbg_features, \
                id2img_prompt_embs, id2img_neg_prompt_embs, \
                captions, subj_single_prompts, subj_comp_prompts, \
                cls_single_prompts, cls_comp_prompts \
                = repeat_selected_instances(slice(0, HALF_BS), 1, 
                                            x_start, batch_images_unnorm, img_mask, fg_mask, 
                                            batch_have_fg_mask, self.batch_subject_names, 
                                            zs_clip_fgbg_features,
                                            id2img_prompt_embs, id2img_neg_prompt_embs,
                                            captions, subj_single_prompts, subj_comp_prompts,
                                            cls_single_prompts, cls_comp_prompts)
                
                # Update delta_prompts to have the first HALF_BS prompts.
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

        # aug_mask is renamed as img_mask.
        self.iter_flags['img_mask']                 = img_mask
        self.iter_flags['fg_mask']                  = fg_mask
        self.iter_flags['batch_have_fg_mask']       = batch_have_fg_mask
        self.iter_flags['delta_prompts']            = delta_prompts
        self.iter_flags['image_unnorm']             = batch_images_unnorm

        self.iter_flags['id2img_prompt_embs']       = id2img_prompt_embs
        self.iter_flags['id2img_neg_prompt_embs']   = id2img_neg_prompt_embs
        if self.embedding_manager.id2ada_prompt_encoder.name == 'jointIDs':
            self.iter_flags['encoders_num_id_vecs']     = self.embedding_manager.id2ada_prompt_encoder.encoders_num_id_vecs
        else:
            self.iter_flags['encoders_num_id_vecs']     = None

        if zs_clip_fgbg_features is not None:
            self.iter_flags['clip_bg_features']  = zs_clip_fgbg_features.chunk(2, dim=1)[1]
        else:
            self.iter_flags['clip_bg_features']  = None

        # In get_text_conditioning(), text_conditioning_iter_type will be set again.
        # Setting it here is necessary, as set_curr_batch_subject_names() maps curr_batch_subj_names to cls_delta_strings,
        # whose behavior depends on the correct text_conditioning_iter_type.
        if self.iter_flags['do_comp_prompt_distillation']:
            text_conditioning_iter_type = 'compos_distill_iter'
        elif self.iter_flags['do_unet_distill']:
            text_conditioning_iter_type = 'unet_distill_iter'
        else:
            text_conditioning_iter_type = 'recon_iter'
        self.iter_flags['text_conditioning_iter_type'] = text_conditioning_iter_type

        self.embedding_manager.set_curr_batch_subject_names(self.batch_subject_names)

        loss = self(x_start, captions)

        return loss

    # LatentDiffusion.forward() is only called during training, by shared_step().
    #LINK #shared_step
    def forward(self, x_start, captions):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        ORIG_BS  = len(x_start)

        # Use >=, i.e., assign decay in all iterations after the first 100.
        # This is in case there are skips of iterations of global_step 
        # (shouldn't happen but just in case).

        assert captions is not None
        # get_text_conditioning(): convert captions to a [BS, 77, 768] tensor.
        # captions: plain prompts like ['an illustration of a dirty z', 'an illustration of the cool z']
        # When do_unet_distill and distilling on ConsistentID, we still
        # need to provide cls_comp_prompts embeddings to the UNet teacher as condition.

        # iter_flags['delta_prompts'] is a tuple of 4 lists. No need to split them.
        delta_prompts = self.iter_flags['delta_prompts']

        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = delta_prompts
        #if self.iter_flags['use_background_token']:
        #print(subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
        
        if self.iter_flags['do_comp_prompt_distillation']:                        
            # For simplicity, BLOCK_SIZE is fixed at 1. So if ORIG_BS == 2, then BLOCK_SIZE = 1.
            BLOCK_SIZE  = 1
            # Only keep the first half of batched prompts to save RAM.
            subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                subj_single_prompts[:BLOCK_SIZE], subj_comp_prompts[:BLOCK_SIZE], \
                cls_single_prompts[:BLOCK_SIZE],  cls_comp_prompts[:BLOCK_SIZE]
        else:
            # Otherwise, do_prompt_emb_delta_reg.
            # Do not halve the batch. BLOCK_SIZE = ORIG_BS = 12.
            # 12 prompts will be fed into get_text_conditioning().
            BLOCK_SIZE = ORIG_BS
                                    
        # We still compute the prompt embeddings of the 4 types of prompts, 
        # to compute prompt delta loss. 
        # But now there are 16 prompts (4 * ORIG_BS = 16), as the batch is not halved.
        delta_prompts = subj_single_prompts + subj_comp_prompts \
                        + cls_single_prompts + cls_comp_prompts
        #print(delta_prompts)
        # breakpoint()
        # c_prompt_emb: the prompt embeddings for prompt delta loss [4, 77, 768].
        # delta_prompts: the concatenation of
        # (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts).
        # extra_info: a dict that contains extra info.
        c_prompt_emb, _, extra_info = \
            self.get_text_conditioning(delta_prompts, 
                                       self.iter_flags['id2img_prompt_embs'],
                                       self.iter_flags['clip_bg_features'],
                                       randomize_clip_weights=True,
                                       text_conditioning_iter_type=self.iter_flags['text_conditioning_iter_type'])

        subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb = \
            c_prompt_emb.chunk(4)

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

        # placeholder2indices_2b is copied from self.embedding_manager.placeholder2indices 
        # during get_text_conditioning(). Because such indices are volatile 
        # (change with different prompts), we need to cache them immediately for later use.
        placeholder2indices_2b = extra_info['placeholder2indices']
        placeholder2indices_1b = {}
        for k in placeholder2indices_2b:
            placeholder2indices_1b[k] = halve_token_indices(placeholder2indices_2b[k])
            if placeholder2indices_1b[k] is None:
                continue

        # NOTE: if there are multiple subject tokens (e.g., 28 tokens), then only the first subject token
        # is aligned with the "class-token , , , ...". 
        # The rest 27 tokens are aligned with the embeddings of ", ".
        # This misalignment is patched below by distributing the class embeddings to the consecutive 28 tokens.
        cls_single_emb = distribute_embedding_to_M_tokens_by_dict(cls_single_emb, placeholder2indices_1b)
        cls_comp_emb   = distribute_embedding_to_M_tokens_by_dict(cls_comp_emb,   placeholder2indices_1b)
        
        extra_info['placeholder2indices_1b'] = placeholder2indices_1b
        extra_info['placeholder2indices_2b'] = placeholder2indices_2b

        # These embeddings are patched. So combine them back into c_prompt_emb.
        # [64, 77, 768].
        c_prompt_emb = torch.cat([subj_single_emb, subj_comp_emb, 
                                  cls_single_emb,  cls_comp_emb], dim=0)
        extra_info['c_prompt_emb_4b'] = c_prompt_emb

        if self.iter_flags['do_comp_prompt_distillation']:
            # c_in = delta_prompts is used to generate ada embeddings.
            # c_in: subj_single_prompts + subj_comp_prompts + cls_single_prompts + cls_comp_prompts
            # The cls_single_prompts/cls_comp_prompts within c_in will only be used to 
            # generate ordinary prompt embeddings, i.e., 
            # it doesn't contain subject token, and no ada embedding will be injected by embedding manager.
            # Instead, subj_single_emb, subj_comp_emb and subject ada embeddings 
            # are manually mixed into their embeddings.
            c_in = delta_prompts
            # The prompts are either (subj single, subj comp, cls single, cls comp).
            # So the first 2 sub-blocks always contain the subject/background tokens, and we use *_2b.    
            extra_info['placeholder2indices'] = extra_info['placeholder2indices_2b']
        else:
            # do_normal_recon or do_unet_distill.
            c_in = captions
            # Use the original "captions" prompts and embeddings.
            # captions == subj_single_prompts doesn't hold when unet_distill_uses_comp_prompt.
            # it holds in all other cases.
            if not self.iter_flags['unet_distill_uses_comp_prompt']:
                assert captions == subj_single_prompts
            else:
                assert captions == subj_comp_prompts
            # When unet_distill_uses_comp_prompt, captions is subj_comp_prompts. 
            # So in this case, subj_single_emb == subj_comp_emb.
            c_prompt_emb = subj_single_emb
            # The blocks as input to get_text_conditioning() are not halved. 
            # So BLOCK_SIZE = ORIG_BS = 2. Therefore, for the two instances, we use *_1b.
            extra_info['placeholder2indices']   = extra_info['placeholder2indices_1b']
            extra_info['c_prompt_emb_1b']       = c_prompt_emb

            # extra_info['c_prompt_emb_4b'] is already [16, 4, 77, 768]. Replace the first block [4, 4, 77, 768].
            # As adaface_subj_embs0 is only the subject embeddings, we need to rely on placeholder_indices 
            # to do the replacement.
            # extra_info['c_prompt_emb_4b'][:BLOCK_SIZE] = self.embedding_manager.adaface_subj_embs0
                                
            ##### End of normal_recon with prompt delta loss iters. #####

        extra_info['cls_single_prompts'] = cls_single_prompts
        extra_info['cls_single_emb']     = cls_single_emb
        extra_info['cls_comp_prompts']   = cls_comp_prompts
        extra_info['cls_comp_emb']       = cls_comp_emb
                            
        # 'delta_prompts' is only used in comp_prompt_mix_reg iters. 
        # Keep extra_info['delta_prompts'] and iter_flags['delta_prompts'] the same structure.
        # (Both are tuples of 4 lists. But iter_flags['delta_prompts'] may contain more prompts
        # than those actually used in this iter.)
        # iter_flags['delta_prompts'] is not used in p_losses(). Keep it for debugging purpose.
        extra_info['delta_prompts']      = (subj_single_prompts, subj_comp_prompts, \
                                            cls_single_prompts,  cls_comp_prompts)

        # c_prompt_emb is the full set of embeddings of subj_single_prompts, subj_comp_prompts, 
        # cls_single_prompts, cls_comp_prompts. 
        # c_prompt_emb: [64, 77, 768]                    
        cond = (c_prompt_emb, c_in, extra_info)

        # self.model (UNetModel) is called in p_losses().
        #LINK #p_losses
        c_prompt_emb, c_in, extra_info = cond
        return self.p_losses(x_start, t, c_prompt_emb, c_in, extra_info)

    # apply_model() is called both during training and inference.
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        # self.model: DiffusionWrapper -> 
        # self.model.diffusion_model: ldm.modules.diffusionmodules.openaimodel.UNetModel
        x_recon = self.model(x_noisy, t, cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    # text_prompt_adhoc_info: volatile data structures changing along with the prompts or the input images.
    # Sometimes the prompts changed after generating the prompt embeddings, 
    # so in such cases we need to manually specify these data structures. 
    # If they are not provided (None),
    # then the data structures stored in embedding_manager are not updated.
    # do_pixel_recon: return denoised images for CLIP evaluation. 
    # if do_pixel_recon and cfg_scale > 1, apply classifier-free guidance. 
    # This is not used for the iter_type 'do_normal_recon'.
    # unet_has_grad: 'all', 'none', 'subject-half'.
    def guided_denoise(self, x_start, noise, t, cond, text_prompt_adhoc_info, 
                       unet_has_grad='all', do_pixel_recon=False, cfg_scale=-1):
        
        self.embedding_manager.set_prompt_adhoc_info(text_prompt_adhoc_info)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # model_output is the predicted noise.
        # if not unet_has_grad, we save RAM by not storing the computation graph.
        # if unet_has_grad, we don't have to take care of embedding_manager.force_grad.
        # Subject embeddings will naturally have gradients.
        if unet_has_grad == 'none':
            with torch.no_grad():
                model_output = self.apply_model(x_noisy, t, cond)
        elif unet_has_grad == 'all':
            model_output = self.apply_model(x_noisy, t, cond)
        elif unet_has_grad == 'subject-half':
            # Split x_noisy, t, cond into two halves.
            x_noisy_1, x_noisy_2 = x_noisy.chunk(2)
            t_1, t_2 = t.chunk(2)
            c_prompt_emb, c_in, extra_info = cond
            c_prompt_emb_1, c_prompt_emb_2 = c_prompt_emb.chunk(2)
            c_in_1, c_in_2 = c_in[:len(x_noisy_1)], c_in[len(x_noisy_1):]
            extra_info_1, extra_info_2 = extra_info, copy.copy(extra_info)
            model_output_1 = self.apply_model(x_noisy_1, t_1, (c_prompt_emb_1, c_in_1, extra_info_1))
            with torch.no_grad():
                model_output_2 = self.apply_model(x_noisy_2, t_2, (c_prompt_emb_2, c_in_2, extra_info_2))

            model_output = torch.cat([model_output_1, model_output_2], dim=0)
            if extra_info['capture_ca_layers_activations']:
                # Concatenate the two attention weights.
                for layer_idx in extra_info_1['ca_layers_activations'].keys():
                    for k in extra_info_1['ca_layers_activations'][layer_idx].keys():
                        extra_info_1['ca_layers_activations'][layer_idx][k] = \
                            torch.cat([extra_info_1['ca_layers_activations'][layer_idx][k], 
                                       extra_info_2['ca_layers_activations'][layer_idx][k]], dim=0)
        else:
            breakpoint()

        # Get model output of both conditioned and uncond prompts.
        # Unconditional prompts and reconstructed images are never involved in optimization.
        if cfg_scale > 1:
            # We never needs gradients on unconditional generation.
            with torch.no_grad():
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                # Clear the cached placeholder indices, as they are for conditional embeddings.
                # Now we generate model_output_uncond under unconditional (negative) prompts,
                # which don't contain placeholder tokens.
                self.embedding_manager.clear_prompt_adhoc_info()
                # uncond_context is a tuple of (uncond_emb, uncond_c_in, extra_info).
                # By default, 'capture_ca_layers_activations' = False in a generated text context, 
                # including uncond_context. So we don't need to set it in self.uncond_context explicitly.                
                uncond_emb  = self.uncond_context[0].repeat(x_noisy.shape[0], 1, 1)
                uncond_c_in = self.uncond_context[1] * x_noisy.shape[0]
                uncond_context = (uncond_emb, uncond_c_in, self.uncond_context[2])
                # model_output_uncond: [BS, 4, 64, 64]
                model_output_uncond = self.apply_model(x_noisy, t, uncond_context)
            # If do clip filtering, CFG makes the contents in the 
            # generated images more pronounced => smaller CLIP loss.
            noise_pred = model_output * cfg_scale - model_output_uncond * (cfg_scale - 1)
        else:
            noise_pred = model_output

        if do_pixel_recon:
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=noise_pred)
        else:
            x_recon = None
        
        return model_output, x_recon

    # Release part of the computation graph on unused instances to save RAM.
    def release_plosses_intermediates(self, local_vars):
        for k in ('model_output', 'x_recon', 'clip_images_code'):
            if k in local_vars:
                del local_vars[k]

        extra_info = local_vars['extra_info']
        for k in ('ca_layers_activations'):
            if k in extra_info:
                del extra_info[k]

    # t: timesteps.
    # c_in is the textual prompts. 
    # extra_info: a dict that contains various fields. 
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, t, c_prompt_emb, c_in, extra_info):
        #print(c_in)
        cond = (c_prompt_emb, c_in, extra_info)
        img_mask            = self.iter_flags['img_mask']
        fg_mask             = self.iter_flags['fg_mask']
        batch_have_fg_mask  = self.iter_flags['batch_have_fg_mask']
        filtered_fg_mask    = self.iter_flags.get('filtered_fg_mask', None)
        placeholder2indices = extra_info['placeholder2indices']
        prompt_emb_mask     = extra_info['prompt_emb_mask']
        
        # all_subj_indices, all_bg_indices are used to extract the attention weights
        # of the subject and background tokens for the attention loss computation.
        # join_dict_of_indices_with_key_filter(): separate the indices of the subject tokens from those of 
        # the background tokens, using subject_string_dict and background_string_dict as the filters, separately.
        all_subj_indices    = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices'],
                                                                   self.embedding_manager.subject_string_dict)
        all_bg_indices      = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices'],
                                                                   self.embedding_manager.background_string_dict)
        if self.iter_flags['do_comp_prompt_distillation']:
            # all_subj_indices_2b is used in calc_feat_attn_delta_loss() in calc_comp_prompt_distill_loss().
            all_subj_indices_2b = \
                join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_2b'],
                                                     self.embedding_manager.subject_string_dict)
            # all_subj_indices_1b is used in calc_comp_fg_bg_preserve_loss() in calc_comp_prompt_distill_loss().
            all_subj_indices_1b = \
                join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_1b'],
                                                     self.embedding_manager.subject_string_dict)

        # prompt_emb_mask is used by calc_prompt_emb_delta_loss() to highlight the subject tokens.
        text_prompt_adhoc_info = { 'placeholder2indices':   placeholder2indices,
                                   'prompt_emb_mask':       prompt_emb_mask }

        noise = torch.randn_like(x_start) 

        # If do_comp_prompt_distillation, we prepare the attention activations 
        # for computing distillation losses.
        if self.iter_flags['do_comp_prompt_distillation']:
            # For simplicity, we fix BLOCK_SIZE = 1, no matter the batch size.
            # We can't afford BLOCK_SIZE=2 on a 48GB GPU as it will double the memory usage.            
            BLOCK_SIZE = 1
            masks = (img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask)
            # noise and masks are updated to be a 1-repeat-4 structure in do_comp_prompt_denoising().
            # We return noise to make the gt_target up-to-date, which is the recon objective.
            # But gt_target is probably not referred to in the following loss computations,
            # since the current iteration is do_comp_prompt_distillation. We update it just in case.
            # masks will still be used in the loss computation. So we update them as well.
            x_recon, noise, masks, init_prep_context_type = \
                self.do_comp_prompt_denoising(cond, x_start, noise, text_prompt_adhoc_info,
                                              masks, fg_noise_anneal_mean_range=(0.3, 0.3),
                                              BLOCK_SIZE=BLOCK_SIZE)
            # Update masks.
            img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = masks

            # Log the training image (first image in the batch) and the denoised images for diagnosis.
            # The four instances in iter_flags['image_unnorm'] are different,
            # but only the first one is actually used. So we only log the first one.
            # input_image: torch.uint8, 0~255, [1, 512, 512, 3] -> [1, 3, 512, 512].
            input_image = self.iter_flags['image_unnorm'][[0]]
            input_image = input_image.permute(0, 3, 1, 2)
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
            # All of them are 1, indicating green.
            log_image_colors = torch.ones(input_image.shape[0], dtype=int, device=x_start.device)
            self.cache_and_log_generations(input_image, log_image_colors, do_normalize=False)
            recon_images = self.decode_first_stage(x_recon)
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
            # If using subj context to do init prep, then the color is green.
            # If using comp context to do init prep, then the color is purple.
            if init_prep_context_type == 'subj':
                log_image_colors = torch.ones(recon_images.shape[0], dtype=int, device=x_start.device)
            elif init_prep_context_type == 'cls':
                log_image_colors = torch.ones(recon_images.shape[0], dtype=int, device=x_start.device) * 3
            elif init_prep_context_type == 'ensemble':
                log_image_colors = torch.zeros(recon_images.shape[0], dtype=int, device=x_start.device)

            self.cache_and_log_generations(recon_images, log_image_colors, do_normalize=True)

            # x_recon is not used for comp prompt distillation loss computation.
            # Only the captured attention matrics are used. So we can release x_recon.
            del x_recon

        ###### Begin of loss computation. ######
        loss_dict = {}
        session_prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            gt_target = x_start
        # default is "eps", i.e., the UNet predicts noise.
        elif self.parameterization == "eps":
            gt_target = noise
        else:
            raise NotImplementedError()

        loss = 0
                                
        if self.iter_flags['do_normal_recon']:          
            BLOCK_SIZE = x_start.shape[0]

            # Increase t slightly by (1.3, 1.6) to increase noise amount and make the denoising more challenging,
            # with smaller prob to keep the original t.
            t = probably_anneal_t(t, self.training_percent, self.num_timesteps, ratio_range=(1.3, 1.6),
                                  keep_prob_range=(0.2, 0.1))
            if self.iter_flags['num_denoising_steps'] > 1:
                # Take a weighted average of t and 1000, to shift t to larger values, 
                # so that the 2nd-6th denoising steps fall in more reasonable ranges.
                t = (4 * t + (self.iter_flags['num_denoising_steps'] - 1) * self.num_timesteps) // (3 + self.iter_flags['num_denoising_steps'])

            # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
            # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
            # (img_mask is not used in the prompt-guided cross-attention layers).
            extra_info['img_mask']  = img_mask

            extra_info['capture_ca_layers_activations'] = True

            model_output, x_recon = \
                self.guided_denoise(x_start, noise, t, cond, 
                                    text_prompt_adhoc_info=text_prompt_adhoc_info,
                                    unet_has_grad='all', 
                                    # Reconstruct the images at the pixel level for CLIP loss.
                                    # do_pixel_recon is enabled only when do_comp_prompt_distillation.
                                    # This is to cache the denoised images as future initialization.
                                    do_pixel_recon=self.iter_flags['do_comp_prompt_distillation'],
                                    # Do not use cfg_scale for normal recon iterations. Only do recon 
                                    # using the positive prompt.
                                    cfg_scale=-1)

            extra_info['capture_ca_layers_activations'] = False

            # If do_normal_recon, then there's only 1 objective:
            # **Objective 1**: Align the student predicted noise with the ground truth noise.
            if not self.iter_flags['use_background_token']:
                # bg loss is almost completely ignored. But giving it a little weight may help suppress 
                # subj embeddings' contribution to the background (serving as a contrast to the fg).
                bg_pixel_weight = 0 #0.01
            else:
                # use_background_token == True. bg loss is somewhat discounted.
                # p_use_background_token = 0.1 if do_unet_distill. 
                bg_pixel_weight = 0.1
                                
            loss_fg_bg_contrast, loss_recon = \
                self.calc_recon_and_complem_losses(model_output, gt_target, extra_info,
                                                   all_subj_indices, all_bg_indices,
                                                   img_mask, fg_mask, batch_have_fg_mask,
                                                   bg_pixel_weight,
                                                   x_start.shape[0], loss_dict, session_prefix)
            loss += loss_recon + loss_fg_bg_contrast
            v_loss_recon = loss_recon.mean().detach().item()
            loss_dict.update({f'{session_prefix}/loss_recon': v_loss_recon})
            print(f"Rank {self.trainer.global_rank} single-step recon: {t.tolist()}, {v_loss_recon:.4f}")
        elif self.iter_flags['do_unet_distill']:
            loss_unet_distill = \
                self.calc_unet_distill_loss(x_start, noise, t, cond, extra_info, 
                                            text_prompt_adhoc_info, img_mask, fg_mask)

            v_loss_unet_distill = loss_unet_distill.mean().detach().item()
            loss_dict.update({f'{session_prefix}/loss_unet_distill': v_loss_unet_distill})
            loss += loss_unet_distill

        if self.iter_flags['do_prompt_emb_delta_reg']:
            # 'c_prompt_emb_4b' is the prompt embeddings before mixing.
            loss_prompt_emb_delta = calc_prompt_emb_delta_loss( 
                        extra_info['c_prompt_emb_4b'], extra_info['prompt_emb_mask'])

            loss_dict.update({f'{session_prefix}/prompt_emb_delta': loss_prompt_emb_delta.mean().detach().item() })

            # The Prodigy optimizer seems to suppress the embeddings too much, 
            # so it uses a smaller scale to reduce the negative effect of prompt_emb_delta_loss.
            prompt_emb_delta_loss_scale = 1 if self.optimizer_type == 'Prodigy' else 2
            # prompt_emb_delta_reg_weight: 1e-5.
            loss += loss_prompt_emb_delta * self.prompt_emb_delta_reg_weight * prompt_emb_delta_loss_scale

        # fg_bg_xlayer_consist_loss_weight == 5e-5. 
        if self.fg_bg_xlayer_consist_loss_weight > 0 \
          and ( self.iter_flags['do_normal_recon']  or self.iter_flags['do_comp_prompt_distillation'] ):
            # SSB_SIZE: subject sub-batch size.
            # If do_normal_recon, then both instances are subject instances. 
            # The subject sub-batch size SSB_SIZE = 2 (1 * BLOCK_SIZE).
            # If do_comp_prompt_distillation, then subject sub-batch size SSB_SIZE = 2 * BLOCK_SIZE. 
            # (subj-single and subj-comp instances).
            SSB_SIZE = BLOCK_SIZE if self.iter_flags['do_normal_recon'] else 2 * BLOCK_SIZE
            loss_fg_xlayer_consist, loss_bg_xlayer_consist = \
                self.calc_fg_bg_xlayer_consist_loss(extra_info['ca_layers_activations']['attnscore'],
                                                    all_subj_indices,
                                                    all_bg_indices,
                                                    SSB_SIZE)
            if loss_fg_xlayer_consist > 0:
                loss_dict.update({f'{session_prefix}/fg_xlayer_consist': loss_fg_xlayer_consist.mean().detach().item() })
            if loss_bg_xlayer_consist > 0:
                loss_dict.update({f'{session_prefix}/bg_xlayer_consist': loss_bg_xlayer_consist.mean().detach().item() })

            # Reduce the loss_fg_xlayer_consist_loss_scale by 5x if do_zero_shot.
            fg_xlayer_consist_loss_scale = 0.2
            bg_xlayer_consist_loss_scale = 0.06

            loss += (loss_fg_xlayer_consist * fg_xlayer_consist_loss_scale + loss_bg_xlayer_consist * bg_xlayer_consist_loss_scale) \
                    * self.fg_bg_xlayer_consist_loss_weight

        ###### begin of do_comp_prompt_distillation ######
        if self.iter_flags['do_comp_prompt_distillation']:
            loss_comp_prompt_distill, loss_comp_fg_bg_preserve = \
                self.calc_comp_prompt_distill_loss(extra_info['ca_layers_activations'], filtered_fg_mask, batch_have_fg_mask, 
                                                   all_subj_indices_1b, all_subj_indices_2b, 
                                                   BLOCK_SIZE, loss_dict, session_prefix)
            
            if loss_comp_prompt_distill > 0:
                loss_dict.update({f'{session_prefix}/comp_prompt_distill':  loss_comp_prompt_distill.mean().detach().item() })
            # loss_comp_fg_bg_preserve = 0 if comp_init_fg_from_training_image and there's a valid fg_mask.
            if loss_comp_fg_bg_preserve > 0:
                loss_dict.update({f'{session_prefix}/comp_fg_bg_preserve': loss_comp_fg_bg_preserve.mean().detach().item() })
                # Keep track of the number of iterations that use comp_init_fg_from_training_image.
                self.comp_init_fg_from_training_image_count += 1
                comp_init_fg_from_training_image_frac = self.comp_init_fg_from_training_image_count / (self.global_step + 1)
                loss_dict.update({f'{session_prefix}/comp_init_fg_from_training_image_frac': comp_init_fg_from_training_image_frac})

            # comp_fg_bg_preserve_loss_weight: 1e-3. loss_comp_fg_bg_preserve: 0.5-0.6.
            loss += loss_comp_fg_bg_preserve * self.comp_fg_bg_preserve_loss_weight
            # comp_prompt_distill_weight: 1e-3. loss_comp_prompt_distill: 2-3.
            loss += loss_comp_prompt_distill * self.comp_prompt_distill_weight

        if torch.isnan(loss) and self.trainer.global_rank == 0:
            print('NaN loss detected.')
            breakpoint()

        self.release_plosses_intermediates(locals())
        loss_dict.update({f'{session_prefix}/loss': loss.mean().detach().item() })

        return loss, loss_dict

    # Do denoising, collect the attention activations for computing the losses later.
    # masks: (img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask). 
    # Put them in a tuple to avoid too many arguments. The updated masks are returned.
    # For simplicity, we fix BLOCK_SIZE = 1, no matter the batch size.
    # We can't afford BLOCK_SIZE=2 on a 48GB GPU as it will double the memory usage.
    def do_comp_prompt_denoising(self, cond, x_start, noise, text_prompt_adhoc_info,
                                 masks, fg_noise_anneal_mean_range=(0.3, 0.3),
                                 BLOCK_SIZE=1):
        c_prompt_emb, c_in, extra_info = cond

        # Compositional iter.
        # In a compositional iter, we don't use the t passed into p_losses().
        # Instead, t is randomly drawn from the middle (noisy but not too noisy) 30% segment of the timesteps.
        t_middle = torch.randint(int(self.num_timesteps * 0.4), int(self.num_timesteps * 0.7), 
                                 (x_start.shape[0],), device=x_start.device)
        t = t_middle

        # Although img_mask is not explicitly referred to in the following code,
        # it's updated within repeat_selected_instances(slice(0, BLOCK_SIZE), 4, *masks).
        img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask = masks

        if self.iter_flags['comp_init_fg_from_training_image'] and self.iter_flags['fg_mask_avail_ratio'] > 0:
            # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
            # Therefore, using fg_mask for comp_init_fg_from_training_image will force the model remember 
            # the background in the training images, which is not desirable.
            # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0.
            # fg_mask is 4D (added 1D in shared_step()). So expand batch_have_fg_mask to 4D.
            filtered_fg_mask    = fg_mask.to(x_start.dtype) * batch_have_fg_mask.view(-1, 1, 1, 1)
            # x_start, fg_mask, filtered_fg_mask are scaled in init_x_with_fg_from_training_image()
            # by the same scale, to make the fg_mask and filtered_fg_mask consistent with x_start.
            # fg_noise_anneal_mean_range = (0.3, 0.3)
            x_start, fg_mask, filtered_fg_mask = \
                init_x_with_fg_from_training_image(x_start, fg_mask, filtered_fg_mask, 
                                                    self.training_percent,
                                                    base_scale_range=(0.8, 1.0),
                                                    fg_noise_anneal_mean_range=fg_noise_anneal_mean_range)

        else:
            # We have to use random noise for x_start, as the training images 
            # are not used for initialization.
            x_start.normal_()

        # Make the 4 instances in x_start, noise and t the same.
        x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)
        noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)
        t       = t[:BLOCK_SIZE].repeat(4)

        # masks may have been changed in init_x_with_fg_from_training_image(). So we update it.
        masks = (img_mask, fg_mask, filtered_fg_mask, batch_have_fg_mask)
        # Update masks to be a 1-repeat-4 structure.
        masks = \
            repeat_selected_instances(slice(0, BLOCK_SIZE), 4, *masks)
        batch_have_fg_mask = masks[-1]
        self.iter_flags['fg_mask_avail_ratio'] = batch_have_fg_mask.float().mean()

        # This code block is within self.iter_flags['do_comp_prompt_distillation'].
        # The prompts are of 1-repeat-4 (distillation) structure.
        # Embedding mixing is always applied to the subject indices in every instance in the 
        # second half-batch (class instances) only, 
        # and the first half-batch (subject instances) is not affected.
        # So providing all_subj_indices_1b[1] is enough. No need to provide batch indices.
        # The prompts are always repetitions like (subj_comp_prompts * T, cls_comp_prompts * T),
        # so subj indices of the second-half batch are the repetitions of the 
        # original extra_info['placeholder2indices_1b'].
        #c_prompt_emb_mixed = mix_cls_subj_embeddings(c_prompt_emb, all_subj_indices_1b[1], 
        #                                             cls_subj_mix_scale=self.cls_subj_mix_scale)
        
        # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
        # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
        # (img_mask is not used in the prompt-guided cross-attention layers).
        # But we don't use img_mask in compositional iterations. Because in compositional iterations,
        # the original images don't play a role (even if comp_init_fg_from_training_image,
        # the unet doesn't consider the actual pixels outside of the subject areas, so img_mask is
        # set to None in extra_info).
        extra_info['img_mask']  = None

        # Update cond[0] to c_prompt_emb_mixed, to prepare for future reference.
        # cond = (c_prompt_emb, c_in, extra_info)
        # print(f"Rank {self.trainer.global_rank} step {self.global_step}: do_comp_prompt_distillation\n", c_in)

        # We allow 2 to 6 denoising steps for preparing class instances for comp prompt distillation.
        '''
        num_init_prep_denoising_steps = \
            sample_num_denoising_steps( self.max_num_comp_distill_init_prep_denoising_steps, 
                                        p_num_denoising_steps    = [0.1, 0.3, 0.3, 0.3], 
                                        cand_num_denoising_steps = [1,   2,   3,   4] )
        '''
        
        # num_init_prep_denoising_steps iterates from 1 to 5.
        # Note 5 is co-prime with comp_distill_iter_gap == 3 or 4.
        # Therefore, this will be a uniform distribution of 1 to 5.
        # Otherwise say, if num_denoising_steps == self.global_step % 6, then since for comp_distill_iterations,
        # the global_step is always a multiple of 3, then num_denoising_steps will always be 0 or 3, which is not desired.
        num_init_prep_denoising_steps = self.global_step % 5 + 1

        if num_init_prep_denoising_steps > 0:
            # Class prep denoising: Denoise x_start with the class prompts 
            # for num_init_prep_denoising_steps times, using self.comp_distill_init_prep_unet.
            # We only use the second half-batch (class instances) for class prep denoising.
            x_start_2 = x_start.chunk(2)[1]
            noise_2 = noise.chunk(2)[1]
            t_2 = t.chunk(2)[1]
            c_prompt_emb_1, c_prompt_emb_2   = c_prompt_emb.chunk(2)
            uncond_emb       = self.uncond_context[0].repeat(x_start_2.shape[0], 1, 1)
            # *_double_context contains both the positive and negative prompt embeddings.
            subj_double_context = torch.cat([c_prompt_emb_1, uncond_emb], dim=0)
            cls_double_context  = torch.cat([c_prompt_emb_2, uncond_emb], dim=0)
            init_prep_context_type = 'ensemble'
            
            '''
            # Randomly switch between the subject and class double contexts.
            if torch.rand(1) < 0.5:
                init_prep_context = subj_double_context
                init_prep_context_type = 'subj'
            else:
                init_prep_context = cls_double_context
                init_prep_context_type = 'cls'
            '''

            # Since we always use CFG for class prep denoising,
            # we need to pass the negative prompt as well.
            # cfg_scale_range=[2, 4].
            with torch.no_grad():
                init_prep_noise_preds, init_prep_x_starts, init_prep_noises, all_t = \
                    self.comp_distill_init_prep_unet(self, x_start_2, noise_2, t_2, 
                                                     # The unet ensemble will do denoising with both subj_double_context
                                                     # and cls_double_context, and average the results.
                                                     teacher_context=[subj_double_context, cls_double_context], 
                                                     num_denoising_steps=num_init_prep_denoising_steps,
                                                     uses_same_t=True)
                
            all_t_list = [ t[0].item() for t in all_t ]
            print(f"Rank {self.trainer.global_rank} step {self.global_step}: "
                  f"{num_init_prep_denoising_steps} subj-cls ensemble prep denoising steps {all_t_list}")
            
            # The last init_prep_x_start is the final denoised image (with the smallest t).
            # So we use it as the x_start to be denoised by the 4-type prompt set.
            # We need to let the subject and class instances use the same x_start. 
            # Therefore, we repeat init_prep_x_starts[-1] twice.
            x_start = init_prep_x_starts[-1].repeat(2, 1, 1, 1).to(dtype=x_start.dtype)
            del init_prep_noise_preds, init_prep_x_starts, init_prep_noises, all_t

        extra_info['capture_ca_layers_activations'] = True

        # model_output is not used by the caller.
        model_output, x_recon = \
            self.guided_denoise(x_start, noise, t, cond, 
                                text_prompt_adhoc_info=text_prompt_adhoc_info,
                                unet_has_grad='subject-half', 
                                # Reconstruct the images at the pixel level for CLIP loss.
                                # do_pixel_recon is enabled only when do_comp_prompt_distillation.
                                # This is to log the denoised images for debugging and visualization.
                                do_pixel_recon=self.iter_flags['do_comp_prompt_distillation'],
                                cfg_scale=5)

        extra_info['capture_ca_layers_activations'] = False

        # noise and masks are updated to be a 1-repeat-4 structure in do_comp_prompt_denoising().
        # We return noise to make the gt_target up-to-date, which is the recon objective.
        # But gt_target is probably not referred to in the following loss computations,
        # since the current iteration is do_comp_prompt_distillation. We update it just in case.
        # masks will still be used in the loss computation. So we return updated masks as well.
        return x_recon, noise, masks, init_prep_context_type
    
    # Major losses for normal_recon iterations (loss_recon, loss_fg_bg_complementary, etc.).
    # (But there are still other losses used after calling this function.)
    def calc_recon_and_complem_losses(self, model_output, target, extra_info,
                                      all_subj_indices, all_bg_indices,
                                      img_mask, fg_mask, batch_have_fg_mask, 
                                      bg_pixel_weight, BLOCK_SIZE, loss_dict, session_prefix):
        loss_fg_bg_contrast = 0

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
                loss_dict.update({f'{session_prefix}/fg_bg_complem': loss_fg_bg_complementary.mean().detach().item()})
            # If fg_mask is None, then loss_subj_mb_suppress = loss_bg_mf_suppress = 0.
            if loss_subj_mb_suppress > 0:
                loss_dict.update({f'{session_prefix}/subj_mb_suppress': loss_subj_mb_suppress.mean().detach().item()})
            if loss_bg_mf_suppress > 0:
                loss_dict.update({f'{session_prefix}/bg_mf_suppress': loss_bg_mf_suppress.mean().detach().item()})
            if loss_fg_bg_mask_contrast > 0:
                loss_dict.update({f'{session_prefix}/fg_bg_mask_contrast': loss_fg_bg_mask_contrast.mean().detach().item()})

            # Reduce the scale of loss_fg_bg_complementary if do_zero_shot, as it hurts performance. 
            loss_fg_bg_complementary_scale = 0.2
            loss_fg_bg_contrast += (loss_fg_bg_complementary * loss_fg_bg_complementary_scale + loss_subj_mb_suppress \
                                    + loss_bg_mf_suppress + loss_fg_bg_mask_contrast) \
                                   * self.fg_bg_complementary_loss_weight

        # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
        loss_recon, _ = self.calc_recon_loss(model_output, target, img_mask, fg_mask, 
                                             fg_pixel_weight=1,
                                             bg_pixel_weight=bg_pixel_weight)

        return loss_fg_bg_contrast, loss_recon

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
                     / (weighted_fg_mask.sum() + weighted_bg_mask.sum() + 1e-6)

        return loss_recon, loss_recon_pixels
    
    def calc_unet_distill_loss(self, x_start, noise, t, cond, extra_info, 
                               text_prompt_adhoc_info, img_mask, fg_mask):
        # num_denoising_steps > 1 implies do_unet_distill, but not vice versa.
        num_denoising_steps = self.iter_flags['num_denoising_steps']
        # student_prompt_embs is the prompt embedding of the student model.
        student_prompt_embs = cond[0]
        # NOTE: when unet_teacher_types == ['unet_ensemble'], unets are specified in 
        # extra_unet_dirpaths (finetuned unets on the original SD unet); 
        # in this case they are surely not 'arc2face' or 'consistentID'.
        # The same student_prompt_embs is used by all unet_teachers.
        if self.unet_teacher_types == ['unet_ensemble']:
            teacher_contexts = [student_prompt_embs]
        else:
            teacher_contexts = []
            encoders_num_id_vecs = self.iter_flags['encoders_num_id_vecs']
            # If id2ada_prompt_encoder.name == 'jointIDs',         then encoders_num_id_vecs is not None.
            # Otherwise, id2ada_prompt_encoder is a single encoder, and encoders_num_id_vecs is None.
            if encoders_num_id_vecs is not None:
                all_id2img_prompt_embs      = self.iter_flags['id2img_prompt_embs'].split(encoders_num_id_vecs, dim=1)
                all_id2img_neg_prompt_embs  = self.iter_flags['id2img_neg_prompt_embs'].split(encoders_num_id_vecs, dim=1)
            else:
                # Single FaceID2AdaPrompt encoder. No need to split id2img_prompt_embs/id2img_neg_prompt_embs.
                all_id2img_prompt_embs      = [ self.iter_flags['id2img_prompt_embs'] ]
                all_id2img_neg_prompt_embs  = [ self.iter_flags['id2img_neg_prompt_embs'] ]

            for i, unet_teacher_type in enumerate(self.unet_teacher_types):
                if unet_teacher_type not in ['arc2face', 'consistentID']:
                    breakpoint()
                if unet_teacher_type == 'arc2face':
                    # For arc2face, p_unet_teacher_uses_cfg is always 0. So we only pass pos_prompt_embs.
                    img_prompt_prefix_embs = self.img_prompt_prefix_embs.repeat(x_start.shape[0], 1, 1)
                    # teacher_context: [BS, 4+16, 768] = [BS, 20, 768]
                    teacher_context = torch.cat([img_prompt_prefix_embs, all_id2img_prompt_embs[i]], dim=1)

                    if self.p_unet_teacher_uses_cfg > 0:
                        # When p_unet_teacher_uses_cfg > 0, we provide both pos_prompt_embs and neg_prompt_embs 
                        # to the teacher.
                        # self.uncond_context is a tuple of (uncond_embs, uncond_c_in, extra_info).
                        # Truncate the uncond_embs to the same length as teacher_context.
                        LEN_POS_PROMPT = teacher_context.shape[1]
                        teacher_neg_context = self.uncond_context[0][:1, :LEN_POS_PROMPT].repeat(x_start.shape[0], 1, 1)
                        # The concatenation of teacher_context and teacher_neg_context is done on dim 0.
                        teacher_context = torch.cat([teacher_context, teacher_neg_context], dim=0)

                elif unet_teacher_type == 'consistentID':
                    global_id_embeds = all_id2img_prompt_embs[i]
                    # global_id_embeds: [BS, 4,  768]
                    # cls_prompt_embs:  [BS, 77, 768]
                    cls_emb_key = 'cls_comp_emb' if self.iter_flags['unet_distill_uses_comp_prompt'] else 'cls_single_emb'
                    cls_prompt_embs = extra_info[cls_emb_key]
                    # Always append the ID prompt embeddings to the class (general) prompt embeddings.
                    # teacher_context: [BS, 81, 768]
                    teacher_context = torch.cat([cls_prompt_embs, global_id_embeds], dim=1)    
                    if self.p_unet_teacher_uses_cfg > 0:
                        # When p_unet_teacher_uses_cfg > 0, we provide both pos_prompt_embs and neg_prompt_embs 
                        # to the teacher.
                        global_neg_id_embs = all_id2img_neg_prompt_embs[i]
                        # uncond_context is a tuple of (uncond_emb, uncond_c_in, extra_info).
                        # uncond_context[0]: [16, 77, 768] -> [1, 77, 768] -> [BS, 77, 768]
                        cls_neg_prompt_embs = self.uncond_context[0][:1].repeat(teacher_context.shape[0], 1, 1)
                        # teacher_neg_context: [BS, 81, 768]
                        teacher_neg_context = torch.cat([cls_neg_prompt_embs, global_neg_id_embs], dim=1)
                        # The concatenation of teacher_context and teacher_neg_context is done on dim 0.
                        # This is kind of arbitrary (we can also concate them on dim 1), 
                        # since we always chunk(2) on the same dimension to restore the two parts.
                        teacher_context = torch.cat([teacher_context, teacher_neg_context], dim=0)            

                teacher_contexts.append(teacher_context)
            # If there's only one teacher, then self.unet_teacher is not a UNetEnsembleTeacher.
            # So we dereference the list.
            if len(teacher_contexts) == 1:
                teacher_contexts = teacher_contexts[0]

        with torch.no_grad():
            unet_teacher_noise_preds, unet_teacher_x_starts, unet_teacher_noises, all_t = \
                self.unet_teacher(self, x_start, noise, t, teacher_contexts, 
                                  num_denoising_steps=num_denoising_steps)
        
        # **Objective 2**: Align student noise predictions with teacher noise predictions.
        # targets: replaced as the reconstructed x0 by the teacher UNet.
        # If ND = num_denoising_steps > 1, then unet_teacher_noise_preds contain ND 
        # unet_teacher predicted noises (of different ts).
        # targets: [HALF_BS, 4, 64, 64] * num_denoising_steps.
        targets = unet_teacher_noise_preds

        # The outputs of the remaining denoising steps will be appended to model_outputs.
        model_outputs = []
        #all_recon_images = []

        for s in range(num_denoising_steps):
            # Predict the noise with t_s (a set of earlier t).
            # When s > 1, x_start_s is the unet_teacher predicted images in the previous step,
            # used to seed the second denoising step. 
            x_start_s = unet_teacher_x_starts[s].to(x_start.dtype)
            # noise_t, t_s are the s-th noise/t used to by unet_teacher.
            noise_t   = unet_teacher_noises[s].to(x_start.dtype)
            t_s       = all_t[s]

            # Here x_start_s is used as x_start.
            # ** unet_teacher.cfg_scale is randomly sampled from unet_teacher_cfg_scale_range in unet_teacher(). **
            # ** DO make sure unet_teacher() was called before guided_denoise() below. **
            # We need to make the student's CFG scale consistent with the teacher UNet's.
            # If not self.p_unet_teacher_uses_cfg, then self.unet_teacher.cfg_scale = 1, 
            # and the cfg_scale is not used in guided_denoise().
            model_output_s, x_recon_s = \
                self.guided_denoise(x_start_s, noise_t, t_s, cond, 
                                    text_prompt_adhoc_info=text_prompt_adhoc_info,
                                    unet_has_grad='all', do_pixel_recon=True, 
                                    cfg_scale=self.unet_teacher.cfg_scale)
            model_outputs.append(model_output_s)

            recon_images_s = self.decode_first_stage(x_recon_s)
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
            # all of them are 2, indicating red.
            log_image_colors = torch.ones(recon_images_s.shape[0], dtype=int, device=x_start.device) * 2
            self.cache_and_log_generations(recon_images_s, log_image_colors, do_normalize=True)

        print(f"Rank {self.trainer.global_rank} {len(model_outputs)}-step distillation:")
        losses_unet_distill = []

        for s in range(len(model_outputs)):
            try:
                model_output, target = model_outputs[s], targets[s]
            except:
                breakpoint()

            # If we use the original image (noise) as target, and still wish to keep the original background
            # after being reconstructed with id2img_prompt_embs, so as not to suppress the background pixels. 
            # Therefore, bg_pixel_weight = 0.1.
            if not self.iter_flags['do_unet_distill']:
                bg_pixel_weight = 0.1
            # If we use comp_prompt as condition, then the background is compositional, and 
            # we want to do recon on the whole image. But considering background is not perfect, 
            # esp. for consistentID whose compositionality is not so good, so bg_pixel_weight = 0.5.
            elif self.iter_flags['unet_distill_uses_comp_prompt']:
                bg_pixel_weight = 0.5
            else:
                # unet_teacher_type == ['arc2face'] or ['consistentID'] or ['consistentID', 'arc2face'].
                bg_pixel_weight = 0

            # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
            loss_unet_distill, _ = \
                self.calc_recon_loss(model_output, target.to(model_output.dtype), 
                                        img_mask, fg_mask, fg_pixel_weight=1,
                                        bg_pixel_weight=bg_pixel_weight)

            print(f"Rank {self.trainer.global_rank} Step {s}: {all_t[s].tolist()}, {loss_unet_distill.item():.4f}",
                  end='')
            
            losses_unet_distill.append(loss_unet_distill)

            # Try hard to release memory after each step. But since they are part of the computation graph,
            # doing so may not have any effect :(
            model_outputs[s], targets[s] = None, None

        # If num_denoising_steps > 1, most loss_unet_distill are usually 0.001~0.005, but sometimes there are a few large loss_unet_distill.
        # In order not to dilute the large loss_unet_distill, we don't divide by num_denoising_steps.
        # Instead, only increase the normalizer sub-linearly.
        loss_unet_distill = sum(losses_unet_distill) / np.sqrt(num_denoising_steps)

        return loss_unet_distill

    def calc_comp_prompt_distill_loss(self, ca_layers_activations, filtered_fg_mask, batch_have_fg_mask,
                                      all_subj_indices_1b, all_subj_indices_2b, BLOCK_SIZE, loss_dict, session_prefix):
        # ca_outfeats is a dict as: layer_idx -> ca_outfeat. 
        # It contains the 12 specified cross-attention layers of UNet.
        # i.e., layers 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24.
        # Similar are ca_attns and ca_attnscores.
        ca_outfeats  = ca_layers_activations['outfeat']

        # NOTE: loss_comp_fg_bg_preserve is applied only when this 
        # iteration is teachable, because at such iterations the unet gradient is enabled.
        # If comp_init_fg_from_training_image, then we need to preserve the fg/bg areas.
        # Although fg_mask_avail_ratio > 0 when comp_init_fg_from_training_image,
        # fg_mask_avail_ratio may have been updated after doing teacher filtering 
        # (since x_start has been filtered, masks are also filtered accordingly, 
        # and the same as to fg_mask_avail_ratio). So we need to check it here.
        if self.iter_flags['comp_init_fg_from_training_image'] and self.iter_flags['fg_mask_avail_ratio'] > 0:
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
                                                     ca_layers_activations['q'],
                                                     ca_q_bns,
                                                     ca_layers_activations['attnscore'], 
                                                     filtered_fg_mask, batch_have_fg_mask,
                                                     all_subj_indices_1b, BLOCK_SIZE)
            
            if loss_comp_subj_bg_attn_suppress > 0:
                loss_dict.update({f'{session_prefix}/comp_subj_bg_attn_suppress': loss_comp_subj_bg_attn_suppress.mean().detach().item() })
            # comp_mix_bg_attn_suppress is not optimized, and only recorded for monitoring.
            if loss_comp_mix_bg_attn_suppress > 0:
                loss_dict.update({f'{session_prefix}/comp_mix_bg_attn_suppress': loss_comp_mix_bg_attn_suppress.mean().detach().item() })
            if loss_comp_single_map_align > 0:
                loss_dict.update({f'{session_prefix}/comp_single_map_align': loss_comp_single_map_align.mean().detach().item() })
            if loss_sc_ss_fg_match > 0:
                loss_dict.update({f'{session_prefix}/sc_ss_fg_match': loss_sc_ss_fg_match.mean().detach().item() })
            if loss_mc_ms_fg_match > 0:
                loss_dict.update({f'{session_prefix}/mc_ms_fg_match': loss_mc_ms_fg_match.mean().detach().item() })
            if loss_sc_mc_bg_match > 0:
                loss_dict.update({f'{session_prefix}/sc_mc_bg_match': loss_sc_mc_bg_match.mean().detach().item() })

            elastic_matching_loss_scale = 1
            # loss_comp_single_map_align is L1 loss on attn maps, so its magnitude is small.
            # But this loss is always very small, so no need to scale it up.
            comp_single_map_align_loss_scale = 1
            # mix single - mix comp matching loss is less important, so scale it down.
            ms_mc_fg_match_loss_scale = 0.1
            comp_subj_bg_attn_suppress_loss_scale = 0.02
            sc_mc_bg_match_loss_scale = 1
            
            # No need to scale down loss_comp_mix_bg_attn_suppress, as it's on a 0.05-gs'ed attn map.
            loss_comp_fg_bg_preserve = loss_comp_single_map_align * comp_single_map_align_loss_scale \
                                        + (loss_sc_ss_fg_match + loss_mc_ms_fg_match * ms_mc_fg_match_loss_scale
                                            + loss_sc_mc_bg_match * sc_mc_bg_match_loss_scale) \
                                            * elastic_matching_loss_scale \
                                        + (loss_comp_subj_bg_attn_suppress + loss_comp_mix_bg_attn_suppress) \
                                            * comp_subj_bg_attn_suppress_loss_scale
        else:
            loss_comp_fg_bg_preserve = 0

        feat_delta_align_scale = 0.5
        if self.normalize_ca_q_and_outfeat:
            # Normalize ca_outfeat at 50% chance.
            normalize_ca_outfeat = (torch.rand(1) < 0.5).item()
        else:
            normalize_ca_outfeat = False

        # normalize_ca_outfeat is enabled 50% of the time.
        if normalize_ca_outfeat:
            ca_outfeat_lns = self.embedding_manager.ca_outfeat_lns
            # If using LN, feat delta is around 5x smaller. So we scale it up to 
            # match the scale of not using LN.
            feat_delta_align_scale *= 5
        else:
            ca_outfeat_lns = None
            
        # loss_comp_fg_bg_preserve should supercede loss_comp_prompt_distill, 
        # as it should be more accurate (?).
        # So if loss_comp_fg_bg_preserve is active, then loss_comp_prompt_distill is halved.
        # all_subj_indices_2b is used in calc_feat_attn_delta_loss(), as it's used 
        # to index subj single and subj comp embeddings.
        # The indices will be shifted along the batch dimension (size doubled) 
        # within calc_feat_attn_delta_loss() to index all the 4 blocks.
        loss_feat_delta_align, loss_subj_attn_delta_align, loss_subj_attn_norm_distill \
            = self.calc_feat_attn_delta_loss(ca_outfeats, ca_outfeat_lns,
                                             ca_layers_activations['attnscore'], 
                                             all_subj_indices_2b, BLOCK_SIZE)

        if loss_feat_delta_align > 0:
            loss_dict.update({f'{session_prefix}/feat_delta_align':        loss_feat_delta_align.mean().detach().item() })
        if loss_subj_attn_delta_align > 0:
            loss_dict.update({f'{session_prefix}/subj_attn_delta_align':   loss_subj_attn_delta_align.mean().detach().item() })
        if loss_subj_attn_norm_distill > 0:
            loss_dict.update({f'{session_prefix}/subj_attn_norm_distill':  loss_subj_attn_norm_distill.mean().detach().item() })

        # Seems that loss_subj_attn_delta_align increases as the model gets better. Therefore we reduce 
        # subj_attn_delta_align_loss_scale from 0.1 to 0.01, to minimize its negative impact.
        # TODO: check if we need to totally disable loss_subj_attn_delta_align, by setting its scale to 0.
        subj_attn_delta_align_loss_scale = 0.01
        # loss_feat_delta_align is around 0.2. loss_subj_attn_delta_align is around 0.4-0.5.

        # If do_zero_shot, loss_subj_attn_norm_distill is quite stable (10~20, depending on various settings). 
        # So no need to use a dynamic loss scale.
        subj_attn_norm_distill_loss_scale = 1

        loss_comp_prompt_distill =   loss_subj_attn_delta_align  * subj_attn_delta_align_loss_scale \
                                   + loss_subj_attn_norm_distill * subj_attn_norm_distill_loss_scale \
                                   + loss_feat_delta_align       * feat_delta_align_scale
        
        return loss_comp_prompt_distill, loss_comp_fg_bg_preserve
    
    # calc_feat_attn_delta_loss() is used by calc_comp_prompt_distill_loss().
    def calc_feat_attn_delta_loss(self, ca_outfeats, ca_outfeat_lns, ca_attnscores, fg_indices_2b, BLOCK_SIZE):
        # do_comp_prompt_distillation iterations. No ordinary image reconstruction loss.
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

            '''
            avg_subj_score_at_mb = masked_mean(subj_score_at_mb, bg_mask3, dim=(1,2), keepdim=True)
            if 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
                print(f'layer {unet_layer_idx}')
                print(f'avg_subj_score_at_mf: {avg_subj_score_at_mf.mean():.4f}, avg_subj_score_at_mb: {avg_subj_score_at_mb.mean():.4f}')
            '''

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
                                     do_demeans=[False, False],
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
            if (unet_layer_idx not in attn_align_layer_weights) \
                or (attn_align_xlayer_maps[unet_layer_idx] not in ca_attnscores):
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
                                                                do_demeans=[True, True],
                                                                first_n_dims_to_flatten=1,
                                                                ref_grad_scale=1,
                                                                aim_to_align=True)
            
            loss_layers_fg_xlayer_consist.append(loss_layer_fg_xlayer_consist * attn_align_layer_weight)
            
            if bg_indices is not None:
                #loss_layer_bg_xlayer_consist = ortho_l2loss(bg_attn, bg_attn_xlayer, mean=True)
                loss_layer_bg_xlayer_consist = calc_ref_cosine_loss(bg_attn, bg_attn_xlayer,
                                                                    exponent=2,    
                                                                    do_demeans=[True, True],
                                                                    first_n_dims_to_flatten=1,
                                                                    ref_grad_scale=1,
                                                                    aim_to_align=True)
                
                loss_layers_bg_xlayer_consist.append(loss_layer_bg_xlayer_consist * attn_align_layer_weight)

        loss_fg_xlayer_consist = normalized_sum(loss_layers_fg_xlayer_consist)
        loss_bg_xlayer_consist = normalized_sum(loss_layers_bg_xlayer_consist)

        return loss_fg_xlayer_consist, loss_bg_xlayer_consist

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
            # calc_comp_fg_bg_preserve_loss() can have higher spatial precision 
            # than calc_feat_attn_delta_loss(), but it requires fg_mask, 
            # which is not always available.
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
            # loss_layer_comp_single_align_map: loss of alignment between two soft mappings: sc_map_ss_prob and mc_map_ms_prob.
            loss_layer_comp_single_align_map, loss_layer_sc_ss_fg_match, \
            loss_layer_sc_mc_bg_match, sc_map_ss_fg_prob_below_mean, mc_map_ss_fg_prob_below_mean \
                = calc_elastic_matching_loss(ca_layer_q_pooled, ca_outfeat_pooled, fg_attn_mask_pooled, 
                                             fg_bg_cutoff_prob=0.25,
                                             single_q_grad_scale=0.1, single_feat_grad_scale=0.01,
                                             mix_feat_grad_scale=0.05)

            loss_layers_comp_single_map_align.append(loss_layer_comp_single_align_map * feat_distill_layer_weight)
            loss_layers_sc_ss_fg_match.append(loss_layer_sc_ss_fg_match * feat_distill_layer_weight)
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
        loss_mc_ms_fg_match             = 0
        loss_sc_mc_bg_match             = normalized_sum(loss_layers_sc_mc_bg_match)
        loss_comp_subj_bg_attn_suppress = normalized_sum(loss_layers_comp_subj_bg_attn_suppress)
        loss_comp_mix_bg_attn_suppress  = normalized_sum(loss_layers_comp_mix_bg_attn_suppress)

        return loss_comp_single_map_align, loss_sc_ss_fg_match, loss_mc_ms_fg_match, \
               loss_sc_mc_bg_match, loss_comp_subj_bg_attn_suppress, loss_comp_mix_bg_attn_suppress

    # samples: a single 4D [B, C, H, W] np array, or a single 4D [B, C, H, W] torch tensor, 
    # or a list of 3D [C, H, W] torch tensors.
    # Data type of samples could be uint (0-25), or float (-1, 1) or (0, 1).
    # If (-1, 1), then we should set do_normalize=True.
    # img_colors: a single 1D torch tensor, indexing colors = [ None, 'green', 'red', 'purple' ]
    # For raw output from raw output from SD decode_first_stage(),
    # samples are be between [-1, 1], so we set do_normalize=True.
    @rank_zero_only
    def cache_and_log_generations(self, samples, img_colors, do_normalize=True, max_cache_size=48):
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)

        # samples is a list of 3D tensor: (C, H, W)
        if not isinstance(samples, torch.Tensor):
            # Make sample a 4D tensor: (B, C, H, W)
            samples = torch.cat(samples, 0)

        if samples.dtype != torch.uint8:
            if do_normalize:
                samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
            samples = (255. * samples).to(torch.uint8)

        # img_colors is a 1D tensor: (B,)
        if img_colors is None:
            img_colors = torch.zeros(samples.size(0), dtype=torch.int)

        self.generation_cache.append(samples)
        self.generation_cache_img_colors.append(img_colors)
        self.num_cached_generations += len(samples)

        if self.num_cached_generations >= max_cache_size:
            grid_folder = self.logger._save_dir + f'/samples'
            os.makedirs(grid_folder, exist_ok=True)
            grid_filename = grid_folder + f'/{self.cache_start_iter:04d}-{self.global_step:04d}.png'
            cached_images     = torch.cat(self.generation_cache,            0)
            cached_img_colors = torch.cat(self.generation_cache_img_colors, 0)
            # samples:    a (B, C, H, W) tensor.
            # img_colors: a tensor of (B,) ints.
            # samples should be between [0, 255] (uint8).
            save_grid(cached_images, cached_img_colors, grid_filename, nrow=12)
            print(f"{self.num_cached_generations} generations saved to {grid_filename}")
            
            # Clear the cache. If num_cached_generations > max_cache_size,
            # some samples at the end of the cache will be discarded.
            self.generation_cache = []
            self.generation_cache_img_colors = []
            self.num_cached_generations = 0
            self.cache_start_iter = self.global_step + 1

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
        elif self.optimizer_type == 'Adam8bit':
            OptimizerClass = bnb.optim.Adam8bit
        elif self.optimizer_type == 'AdamW8bit':
            OptimizerClass = bnb.optim.AdamW8bit
        elif self.optimizer_type == 'OrthogonalNesterov':
            OptimizerClass = CombinedOptimizer
        elif self.optimizer_type == 'Prodigy':
            OptimizerClass = Prodigy
        elif self.optimizer_type == 'AdEMAMix':
            OptimizerClass = AdEMAMix
        elif self.optimizer_type == 'AdEMAMixDistributedShampoo':
            OptimizerClass = AdEMAMixDistributedShampoo
        else:
            raise NotImplementedError()
            
        # self.learning_rate and self.weight_decay are set in main.py.
        # self.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr.
        # If accumulate_grad_batches = 2, ngpu = 2, bs = 4, base_lr = 8e-04, then
        # learning_rate = 2 * 2 * 4 * 1e-05 = 1.6e-04.
        lr          = self.learning_rate
        scheduler   = None

        opt_params_with_lrs = []
        if self.embedding_manager_trainable:
            embedding_params = self.embedding_manager.optimized_parameters()
            embedding_params_with_lrs = [ {'params': embedding_params, 'lr': lr} ]
            opt_params_with_lrs += embedding_params_with_lrs

        # Are we allowing the base model to train? If so, set two different parameter groups.
        if self.unfreeze_unet: 
            model_params = list(self.model.parameters())
            # unet_lr: default 2e-6 set in finetune-unet.yaml.
            opt_params_with_lrs += [ {"params": model_params, "lr": self.unet_lr} ]

        count_optimized_params(opt_params_with_lrs)

        # Adam series, AdEMAMix series, or OrthogonalNesterov.
        if 'Prodigy' not in self.optimizer_type:
            if 'adam' in self.optimizer_type.lower():
                opt = OptimizerClass(opt_params_with_lrs, weight_decay=self.weight_decay,
                                    betas=self.adam_config.betas)
            # AdEMAMix, AdEMAMixDistributedShampoo.
            if 'AdEMAMix' in self.optimizer_type:
                # AdEMAMix uses three betas. We use the default 0.9999 for the third beta.
                opt = OptimizerClass(opt_params_with_lrs, weight_decay=self.weight_decay,
                                     betas=self.adam_config.betas + (0.9999,))
                
            elif self.optimizer_type == 'OrthogonalNesterov':
                adam_config = {'betas': self.adam_config.betas, 'weight_decay': self.weight_decay}
                # ortho_nevstrov uses a much smaller LR. Theses settings are copied from
                # https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
                ortho_nesterov_config = {'lr': 0.1 * lr, 'momentum': 0.95}
                opt = OptimizerClass(opt_params_with_lrs, optimizer_types=(OrthogonalNesterov, torch.optim.AdamW),
                                     configs=(ortho_nesterov_config, adam_config))
                
            assert 'target' in self.adam_config.scheduler_config
            self.adam_config.scheduler_config.params.max_decay_steps = self.trainer.max_steps
            lambda_scheduler = instantiate_from_config(self.adam_config.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaLR(opt, lr_lambda=lambda_scheduler.schedule)

        else:
            # Use Prodigy. Remove 'lr' from the parameter groups, since Prodigy doesn't use it.
            prodigy_params = [ param_group['params'] for param_group in opt_params_with_lrs ]
            prodigy_params = sum(prodigy_params, [])

            # [0.9, 0.999]. Converge more slowly.
            betas = self.prodigy_config.zs_betas

            # Prodigy uses an LR = 1.
            # weight_decay is always disabled (set to 0).
            opt = OptimizerClass(prodigy_params, lr=1., weight_decay=self.weight_decay,
                                 betas=betas,   # default: [0.985, 0.993]
                                 d_coef=self.prodigy_config.d_coef, # default: 5
                                 safeguard_warmup = self.prodigy_config.scheduler_cycles > 1, 
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
                schedulers.append(CyclicLR(opt, base_lr=0.1, max_lr=1, 
                                           step_size_up = single_cycle_steps / 2,
                                           last_epoch = single_cycle_steps / 2 - 1, 
                                           cycle_momentum=False))
                # Disable SequentialLR2 from calling scheduler.step(0) at the first iteration, which will 
                # set the last_epoch of CyclicLR to 0.
                schedulers[-1].start_from_epoch_0 = False

            else:
                raise NotImplementedError()
            
            scheduler = SequentialLR2(opt, schedulers=schedulers,
                                      milestones=transition_milestones)

        if scheduler is None:
            return opt
        
        optimizers = [ {'optimizer': opt, 'frequency': 1, 
                        'lr_scheduler': {
                            'scheduler': scheduler,
                            'interval': 'step', # No need to specify in config yaml.
                            'frequency': 1
                        }} 
                     ]

        return optimizers

    # Called by modelcheckpoint in config.yaml.
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        print(self.trainer.global_rank, "Saving checkpoint...")
    
        checkpoint.clear()
        
        if os.path.isdir(self.trainer.checkpoint_callback.dirpath): 
            if self.embedding_manager_trainable:
                self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, f"embeddings_gs-{self.global_step}.pt"))

            if self.unfreeze_unet:
                # Save the UNetModel state_dict.
                # self.model is a DiffusionWrapper, whose parameters are the same as the UNetModel member,
                # but with an extra diffusion_model session_prefix. This would be handled during checkpoint conversion.
                # The unet has different parameter names from diffusers.
                # It can be converted with convert_ldm_unet_checkpoint().

                state_dict = self.model.state_dict()
                state_dict2 = {}
                for k in state_dict:
                    # Skip ema weights
                    if k.startswith("model_ema."):
                        continue    
                    if state_dict[k].dtype == torch.float32:
                        state_dict2[k] = state_dict[k].half()
                    else:
                        state_dict2[k] = state_dict[k]

                unet_save_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 
                                              f"unet-{self.global_step}.safetensors")
                safetensors_save_file(state_dict2, unet_save_path)
                print(f"Saved {unet_save_path}")

class DiffusionWrapper(pl.LightningModule): 
    def __init__(self, diff_model_config):
        super().__init__()
        # diffusion_model: UNetModel
        self.diffusion_model = instantiate_from_config(diff_model_config)

    # t: a 1-D batch of timesteps (during training: randomly sample one timestep for each instance).
    def forward(self, x, t, cond):
        c_prompt_emb, c_in, extra_info = cond
        out = self.diffusion_model(x, t, context=c_prompt_emb, context_in=c_in, extra_info=extra_info)

        return out
