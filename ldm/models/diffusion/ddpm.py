import torch
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
                        ortho_subtract, gen_gradient_scaler, calc_ref_cosine_loss, \
                        calc_prompt_emb_delta_loss, calc_elastic_matching_loss, calc_recon_loss, \
                        save_grid, normalize_dict_values, masked_mean, pool_feat_or_attn_mat, \
                        init_x_with_fg_from_training_image, resize_mask_to_target_size, \
                        sel_emb_attns_by_indices, distribute_embedding_to_M_tokens_by_dict, \
                        join_dict_of_indices_with_key_filter, collate_dicts, select_and_repeat_instances, \
                        halve_token_indices, double_token_indices, merge_cls_token_embeddings, anneal_perturb_embedding, \
                        count_optimized_params, count_params, add_dict_to_dict, calc_dyn_loss_scale
                        
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from adaface.diffusers_attn_lora_capture import AttnProcessor_LoRA_Capture, CrossAttnUpBlock2D_forward_capture
from ldm.prodigy import Prodigy
from adaface.unet_teachers import create_unet_teacher
from gma.network import GMA
from gma.utils.utils import load_checkpoint as gma_load_checkpoint

import copy
from functools import partial
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from evaluation.arcface_wrapper import ArcFaceWrapper

import sys
torch.set_printoptions(precision=4, sci_mode=False)

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 automatic_optimization=True,
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
                 optimizer_type='AdamW',
                 grad_clip=0.5,
                 adam_config=None,
                 prodigy_config=None,
                 comp_distill_iter_gap=-1,
                 cls_subj_mix_scale=0.8,
                 prompt_emb_delta_reg_weight=0.,
                 comp_fg_bg_preserve_loss_weight=0.,
                 recon_subj_bg_suppress_loss_weight=0.,
                 pred_l2_loss_weight=0, #1e-4,
                 subj_attn_norm_distill_loss_weight=0,
                 # 'face portrait' is only valid for humans/animals. 
                 # On objects, use_fp_trick will be ignored, even if it's set to True.
                 use_fp_trick=True,
                 unet_distill_iter_gap=2,
                 unet_distill_weight=8,
                 unet_teacher_types=None,
                 max_num_unet_distill_denoising_steps=4,
                 max_num_comp_priming_denoising_steps=4,
                 comp_distill_denoising_steps_range=[2, 3],
                 p_unet_teacher_uses_cfg=0.6,
                 unet_teacher_cfg_scale_range=[1.3, 2],
                 p_unet_distill_uses_comp_prompt=0.1,
                 extra_unet_dirpaths=None,
                 unet_weights=None,
                 p_gen_rand_id_for_id2img=0.4,
                 p_perturb_face_id_embs=0.6,
                 perturb_face_id_embs_std_range=[0.3, 0.6],
                 extend_prompt2token_proj_attention_multiplier=1,
                 use_face_flow_for_sc_matching_loss=False,
                 use_arcface_loss=True,
                 arcface_align_loss_weight=4e-2,
                 use_ldm_unet=True,
                 diffusers_unet_path='models/ensemble/sd15-unet',
                 diffusers_unet_uses_lora=False,
                 ):
        
        super().__init__()
        self.automatic_optimization = automatic_optimization

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
        self.comp_fg_bg_preserve_loss_weight        = comp_fg_bg_preserve_loss_weight
        self.recon_subj_bg_suppress_loss_weight     = recon_subj_bg_suppress_loss_weight
        self.pred_l2_loss_weight                    = pred_l2_loss_weight
        self.subj_attn_norm_distill_loss_weight     = subj_attn_norm_distill_loss_weight
        # mix some of the subject embedding denoising results into the class embedding denoising results for faster convergence.
        # Otherwise, the class embeddings are too far from subject embeddings (person, man, woman), 
        # posing too large losses to the subject embeddings.
        self.cls_subj_mix_scale                     = cls_subj_mix_scale

        self.use_fp_trick                           = use_fp_trick
        self.unet_distill_iter_gap                  = unet_distill_iter_gap if self.training else 0
        self.unet_distill_weight                    = unet_distill_weight
        self.unet_teacher_types                     = list(unet_teacher_types) if unet_teacher_types is not None else None
        self.p_unet_teacher_uses_cfg                = p_unet_teacher_uses_cfg
        self.unet_teacher_cfg_scale_range           = unet_teacher_cfg_scale_range
        self.max_num_unet_distill_denoising_steps   = max_num_unet_distill_denoising_steps
        self.max_num_comp_priming_denoising_steps   = max_num_comp_priming_denoising_steps
        self.comp_distill_denoising_steps_range     = comp_distill_denoising_steps_range
        # Sometimes we use the subject compositional prompts as the distillation target on a UNet ensemble teacher.
        # If unet_teacher_types == ['arc2face'], then p_unet_distill_uses_comp_prompt == 0, i.e., we
        # never use the compositional prompts as the distillation target of arc2face.
        # If unet_teacher_types is ['consistentID', 'arc2face'], then p_unet_distill_uses_comp_prompt == 0.
        # If unet_teacher_types == ['consistentID'], then p_unet_distill_uses_comp_prompt == 0.1.
        # NOTE: If compositional iterations are enabled, then we don't do unet distillation on the compositional prompts.
        if self.unet_teacher_types == ['consistentID'] and self.comp_distill_iter_gap <= 0:
            self.p_unet_distill_uses_comp_prompt = p_unet_distill_uses_comp_prompt
        else:
            self.p_unet_distill_uses_comp_prompt = 0
            
        self.extra_unet_dirpaths                    = extra_unet_dirpaths
        self.unet_weights                           = unet_weights
        
        self.p_gen_rand_id_for_id2img               = p_gen_rand_id_for_id2img
        self.p_perturb_face_id_embs                 = p_perturb_face_id_embs
        self.perturb_face_id_embs_std_range         = perturb_face_id_embs_std_range

        self.extend_prompt2token_proj_attention_multiplier = extend_prompt2token_proj_attention_multiplier
        self.comp_iters_count                        = 0
        self.non_comp_iters_count                    = 0
        self.comp_iters_face_detected_count          = 0
        self.comp_iters_bg_match_loss_count          = 0
        self.comp_init_fg_from_training_image_count  = 0

        self.cached_inits = {}
        self.do_prompt_emb_delta_reg = (self.prompt_emb_delta_reg_weight > 0)

        self.init_iteration_flags()

        self.use_ldm_unet = use_ldm_unet
        self.diffusers_unet_uses_lora = diffusers_unet_uses_lora
        if self.use_ldm_unet:
            self.model = DiffusionWrapper(unet_config)
        else:
            self.model = DiffusersUNetWrapper(unet_dirpath=diffusers_unet_path, 
                                              torch_dtype=torch.float16,
                                              enable_lora=self.diffusers_unet_uses_lora)

        count_params(self.model, verbose=True)

        self.optimizer_type = optimizer_type
        self.adam_config = adam_config
        self.grad_clip = grad_clip
        self.use_face_flow_for_sc_matching_loss = use_face_flow_for_sc_matching_loss
        self.use_arcface_loss = use_arcface_loss
        self.arcface_align_loss_weight = arcface_align_loss_weight

        if 'Prodigy' in self.optimizer_type:
            self.prodigy_config = prodigy_config

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

        num_del_keys = 0
        deleted_keys = []
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
                    deleted_keys.append(k)
                    num_del_keys += 1

        print(f"Deleting {num_del_keys} keys {deleted_keys} from state_dict.")
        num_remaining_keys = len(list(sd.keys()))
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

        print(f"Successfully loaded {num_remaining_keys - len(unexpected)} keys")

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
    def get_loss_func(self, loss_type=None):
        if loss_type is None:
            loss_type = self.loss_type

        if loss_type == 'l1':
            loss_func = F.l1_loss
        elif loss_type == 'l2':
            loss_func = F.mse_loss
        else:
            raise NotImplementedError("Unknown loss type '{loss_type}'")

        return loss_func

    def get_input(self, batch, k):
        x = batch[k]

        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def init_iteration_flags(self):
        self.iter_flags = { 'calc_clip_loss':                   False,
                            'do_normal_recon':                  False,
                            'do_unet_distill':                  False,
                            'gen_rand_id_for_id2img':           False,
                            'id2img_prompt_embs':               None,
                            'id2img_neg_prompt_embs':           None,
                            'perturb_face_id_embs':             False,
                            'faceless_img_count':               0,
                            'do_feat_distill_on_comp_prompt':   False,
                            'do_prompt_emb_delta_reg':          False,
                            'unet_distill_uses_comp_prompt':    False,
                            'use_fp_trick':                     False,
                            'comp_init_fg_from_training_image': False,
                          }
        
    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    def shared_step(self, batch):
        raise NotImplementedError("shared_step() is not implemented in DDPM.")

    def training_step(self, batch, batch_idx):
        self.init_iteration_flags()
        
        # NOTE: No need to have standalone ada prompt delta reg, 
        # since each prompt mix reg iter will also do ada prompt delta reg.

        # If N_CAND_REGS == 0, then no prompt distillation/regularizations, 
        # and the flags below take the default False value.
        if self.comp_distill_iter_gap > 0 and self.global_step % self.comp_distill_iter_gap == 0:
            self.iter_flags['do_feat_distill_on_comp_prompt']  = True
            self.iter_flags['do_normal_recon']              = False
            self.iter_flags['do_unet_distill']              = False
            self.iter_flags['do_prompt_emb_delta_reg']      = self.do_prompt_emb_delta_reg
            self.comp_iters_count += 1
        else:
            self.iter_flags['do_feat_distill_on_comp_prompt']  = False
            self.non_comp_iters_count += 1
            if self.unet_distill_iter_gap > 0 and self.non_comp_iters_count % self.unet_distill_iter_gap == 0:
                self.iter_flags['do_normal_recon']  = False
                self.iter_flags['do_unet_distill']  = True
                # Disable do_prompt_emb_delta_reg during unet distillation.
                self.iter_flags['do_prompt_emb_delta_reg'] = False
            else:
                self.iter_flags['do_normal_recon']  = True
                self.iter_flags['do_unet_distill']  = False
                self.iter_flags['do_prompt_emb_delta_reg'] = self.do_prompt_emb_delta_reg

        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if not self.automatic_optimization:
            self.manual_backward(loss)
            self.clip_gradients(optimizer, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

            if (batch_idx + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

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

        # use_ldm_unet is gotten from kwargs, so it will still be passed to the base class DDPM.
        use_ldm_unet    = kwargs.get("use_ldm_unet", True)

        # base_model_path and ignore_keys are popped from kwargs, so they won't be passed to the base class DDPM.
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
        if base_model_path is not None:
            # Ignore all keys of the UNet model, since we are using a diffusers UNet model.
            # We still need to load the VAE and CLIP weights.
            if not use_ldm_unet:
                ignore_keys.append('model')
            self.init_from_ckpt(base_model_path, ignore_keys)
        
        if self.unet_distill_iter_gap > 0 and self.unet_teacher_types is not None:
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
            # Use SAR UNet to prime x_start for compositional distillation, since 
            # it's more compositional than SD15 UNet and closer to SD15 UNet than RealisticVision4 UNet.
            unet = UNet2DConditionModel.from_pretrained('models/ensemble/sar-unet', torch_dtype=torch.float16)
            # comp_distill_unet is a diffusers unet used to do a few steps of denoising 
            # on the compositional prompts, before the actual compositional distillation.
            # So float16 is sufficient.
            self.comp_distill_priming_unet = \
                create_unet_teacher('unet_ensemble', 
                                    # A trick to avoid creating multiple UNet instances.
                                    unets = [unet, unet],
                                    unet_types=None,
                                    extra_unet_dirpaths=None,
                                    # unet_weights: [0.2, 0.8]. The "first unet" uses subject embeddings, 
                                    # the second uses class embeddings. This means that,
                                    # when aggregating the results of using subject embeddings vs. class embeddings,
                                    # we give more weights to the class embeddings for better compositionality.
                                    unet_weights = [1 - self.cls_subj_mix_scale, self.cls_subj_mix_scale],
                                    p_uses_cfg=1, # Always uses CFG for priming denoising.
                                    cfg_scale_range=[2, 4],
                                    torch_dtype=torch.float16)
            self.comp_distill_priming_unet.train = disabled_train

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

        if self.use_face_flow_for_sc_matching_loss and self.comp_distill_iter_gap > 0:
            flow_model_config = { 'mixed_precision': True }
            self.flow_model = GMA(flow_model_config)
            self.flow_model.eval()
            for param in self.flow_model.parameters():
                param.requires_grad = False

            flow_model_ckpt_path = "models/gma-sintel.pth"
            gma_load_checkpoint(self.flow_model, flow_model_ckpt_path)
        else:
            self.flow_model = None

        if self.use_arcface_loss:
            self.arcface = ArcFaceWrapper('cpu')
            # Disable training mode, as this mode 
            # doesn't accept only 1 image as input.
            self.arcface.train = disabled_train
        else:
            self.arcface = None

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
        if (not self.use_ldm_unet) and self.diffusers_unet_uses_lora:
            unet_hooked_attn_procs = self.model.hooked_attn_procs
        else:
            unet_hooked_attn_procs = None
        model = instantiate_from_config(config, text_embedder=text_embedder,
                                        unet_hooked_attn_procs=unet_hooked_attn_procs)
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
    # If do_feat_distill_on_comp_prompt, then 1 call on delta prompts (NOTE: delta prompts have a large batch size).
    # If do_normal_recon / do_unet_distll with delta loss, then 2 calls (one on delta prompts, one on subject single prompts). 
    # NOTE: the delta prompts consumes extram RAM.
    # If do_normal_recon / do_unet_distll without delta loss, then 1 call.
    # cond_in: a batch of prompts like ['an illustration of a dirty z', ...]
    # return_prompt_embs_type: ['text', 'id', 'text_id'].
    # 'text': default, the conventional text embeddings produced from the embedding manager.
    # 'id': the input subj_id2img_prompt_embs, generated by an ID2ImgPrompt module.
    # 'text_id': concatenate the text embeddings with the subject IMAGE embeddings.
    # 'id' or 'text_id' are ablation settings to evaluate the original ID2ImgPrompt module 
    # without them going through CLIP.
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
            # Guess text_conditioning_iter_type from the iteration flags.
            if self.iter_flags['do_feat_distill_on_comp_prompt']:
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
            
        # return_prompt_embs_type: ['id', 'text_id']. Training default: 'text', i.e., 
        # the conventional text embeddings returned by the clip encoder (embedding manager in the middle).
        # 'id': the subject embeddings only. 
        # 'text_id': concatenate the text embeddings with the subject IMAGE embeddings.
        if return_prompt_embs_type in ['id', 'text_id']:
            # if text_conditioning_iter_type == 'plain_text_iter', the current prompt is a plain text 
            # without the subject string (probably a negative prompt).
            # So subj_id2img_prompt_embs serves as the negative ID prompt embeddings.
            # NOTE: These are not really negative ID embeddings, and is just to make the negative prompt embeddings
            # to have the same length as the positive prompt embeddings.
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

            # Ordinary prompt containing the subject string.
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
            
            # 'id' and 'text_id' are ablation settings to evaluate the original ID2ImgPrompt module.
            # NOTE the subject image embeddings don't go through CLIP.
            if return_prompt_embs_type == 'id':
                # Only return the ID2ImgPrompt embeddings, and discard the text embeddings.
                prompt_embeddings = subj_id2img_prompt_embs
            elif return_prompt_embs_type == 'text_id':
                # NOTE: always append the id2img prompts to the end of the prompt embeddings
                # Arc2face doesn't care about the order of the prompts. But consistentID only works when
                # the id2img prompt embeddings are postpended to the end of the prompt embeddings.
                # prompt_embeddings is already in the image embedding space. So it can be concatenated with
                # the subject image embeddings.
                # prompt_embeddings: [BS, 81, 768]. 81: 77 + 4.
                prompt_embeddings = torch.cat([prompt_embeddings, subj_id2img_prompt_embs], dim=1)

        # Otherwise, inference and no special return_prompt_embs_type, we do nothing to the prompt_embeddings.

        # 'placeholder2indices' and 'prompt_emb_mask' are cached to be used in forward() and p_losses().
        extra_info = { 
                        'placeholder2indices':           copy.copy(self.embedding_manager.placeholder2indices),
                        'prompt_emb_mask':               copy.copy(self.embedding_manager.prompt_emb_mask),
                        # Will be updated to True in p_losses() when in compositional iterations.
                        'capture_ca_activations':  False,
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
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    # same as decode_first_stage() but without torch.no_grad() decorator
    # output: -1 ~ 1.
    def decode_first_stage_with_grad(self, z):
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

        instances_have_fg_mask  = batch['has_fg_mask']
        # Temporarily disable fg_mask for debugging.
        disable_fg_mask = False #True
        if disable_fg_mask:
            instances_have_fg_mask[:] = False

        self.iter_flags['fg_mask_avail_ratio']  = instances_have_fg_mask.sum() / instances_have_fg_mask.shape[0]

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
            if self.iter_flags['do_feat_distill_on_comp_prompt']:
                p_use_fp_trick = 1
            # If compositional distillation is enabled, then in normal recon iterations,
            # we use the fp_trick most of the time, to better reconstructing single-face input images.
            # However, we still keep 20% of the do_normal_recon iterations to not use the fp_trick,
            # to encourage a bias towards larger facial areas in the output images.
            elif self.iter_flags['do_normal_recon'] and self.comp_distill_iter_gap > 0:
                p_use_fp_trick = 0.8
            else:
                # If not doing compositional distillation and only doing do_normal_recon, 
                # then use_fp_trick is disabled, so that the ID embeddings alone are expected 
                # to reconstruct the subject portraits.
                p_use_fp_trick = 0
        else:
            p_use_fp_trick = 0

        self.iter_flags['use_fp_trick'] = (torch.rand(1) < p_use_fp_trick).item()

        if self.iter_flags['do_feat_distill_on_comp_prompt'] and self.iter_flags['fg_mask_avail_ratio'] > 0:
            # If do_feat_distill_on_comp_prompt, comp_init_fg_from_training_image is always enabled.
            # It's OK, since when do_zero_shot, we have a large diverse set of training images,
            # and always initializing from training images won't lead to overfitting.
            p_comp_init_fg_from_training_image = 1
        else:
            p_comp_init_fg_from_training_image = 0

        self.iter_flags['comp_init_fg_from_training_image'] \
            = (torch.rand(1) < p_comp_init_fg_from_training_image).item()

        # ** use_fp_trick is only for compositional iterations. **
        if self.iter_flags['use_fp_trick']:
            # Never use_fp_trick for recon iters. So no need to have "caption_fp".
            SUBJ_SINGLE_PROMPT = 'subj_single_prompt_fp'
            SUBJ_COMP_PROMPT   = 'subj_comp_prompt_fp'
            CLS_SINGLE_PROMPT  = 'cls_single_prompt_fp'
            CLS_COMP_PROMPT    = 'cls_comp_prompt_fp'
        # Either do_feat_distill_on_comp_prompt but not use_fp_trick_iter, 
        # or recon/unet_distill iters (not do_feat_distill_on_comp_prompt).
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
            # aug_mask indicates the valid region of the image, due to the augmentation.
            # img_mask is just another name of aug_mask.
            img_mask = batch['aug_mask']
            # img_mask: [B, H, W] => [B, 1, H, W]
            img_mask = img_mask.unsqueeze(1).to(x_start.device)
            img_mask = F.interpolate(img_mask, size=x_start.shape[-2:], mode='nearest')
        else:
            img_mask = None

        if 'fg_mask' in batch:
            # fg_mask indicates the foreground region of the image. On face images,
            # the human face and body regions are the foreground region. 
            fg_mask = batch['fg_mask']
            # fg_mask: [B, H, W] => [B, 1, H, W]
            fg_mask = fg_mask.unsqueeze(1).to(x_start.device)
            fg_mask = F.interpolate(fg_mask, size=x_start.shape[-2:], mode='nearest')
        else:
            assert self.iter_flags['fg_mask_avail_ratio'] == 0
            fg_mask = None

        print(f"Rank {self.trainer.global_rank}: {batch['subject_name']}")

        BS = len(batch['subject_name'])
        # If do_feat_distill_on_comp_prompt, we repeat the instances in the batch, 
        # so that all instances are the same.
        if self.iter_flags['do_feat_distill_on_comp_prompt']:
            # Change the batch to have the (1 subject image) * BS strcture.
            # "captions" and "delta_prompts" don't change, as different subjects share the same placeholder "z".
            # After image_unnorm is repeated, the extracted zs_clip_fgbg_features and face_id_embs, extracted from image_unnorm,
            # will be repeated automatically. Therefore, we don't need to manually repeat them later.
            batch['subject_name'], batch["image_path"], batch["image_unnorm"], x_start, img_mask, fg_mask, instances_have_fg_mask = \
                select_and_repeat_instances(slice(0, 1), BS, batch['subject_name'], batch["image_path"], batch["image_unnorm"], 
                                            x_start, img_mask, fg_mask, instances_have_fg_mask)
            self.iter_flags['same_subject_in_batch'] = True
        else:
            self.iter_flags['same_subject_in_batch'] = False

        # do_unet_distill and random() < unet_distill_iter_gap.
        # p_gen_rand_id_for_id2img: 0.4 if distilling on arc2face. 0.2 if distilling on consistentID,
        # 0.1 if distilling on jointIDs.
        if self.iter_flags['do_unet_distill'] and (torch.rand(1) < self.p_gen_rand_id_for_id2img):
            self.iter_flags['gen_rand_id_for_id2img'] = True
            self.batch_subject_names = [ "rand_id_to_img_prompt" ] * len(batch['subject_name'])
        else:
            self.iter_flags['gen_rand_id_for_id2img'] = False
            self.batch_subject_names = batch['subject_name']            

        batch_images_unnorm = batch["image_unnorm"]

        # images: 0~255 uint8 tensor [3, 512, 512, 3] -> [3, 3, 512, 512].
        images = batch["image_unnorm"].permute(0, 3, 1, 2).to(x_start.device)
        image_paths = batch["image_path"]

        # gen_rand_id_for_id2img. The recon/distillation is on random ID embeddings. So there's no ground truth input images.
        # Therefore, zs_clip_fgbg_features are not available and are randomly generated as well.
        # gen_rand_id_for_id2img implies (not do_feat_distill_on_comp_prompt).
        # NOTE: the faces generated with gen_rand_id_for_id2img are usually atypical outliers,
        # so adding a small proportion of them to the training data may help increase the authenticity on
        # atypical faces, but adding too much of them may harm the performance on typical faces.
        if self.iter_flags['gen_rand_id_for_id2img']:
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
            instances_have_fg_mask[:] = False
            # In a gen_rand_id_for_id2img iteration, simply denoise a totally random x_start.
            x_start = torch.randn_like(x_start)
            self.iter_flags['faceless_img_count'] = 0
            # A batch of random faces share no similarity with each other, so same_subject_in_batch is False.
            self.iter_flags['same_subject_in_batch'] = False

        # Not gen_rand_id_for_id2img. The recon/distillation is on real ID embeddings.
        # 'gen_rand_id_for_id2img' is only True in do_unet_distill iters.
        # So if not do_unet_distill, then this branch is always executed.
        #    If     do_unet_distill, then this branch is executed at 50% of the time.
        else:
            # If self.iter_flags['same_subject_in_batch']:  zs_clip_fgbg_features: [1, 514, 1280]. face_id_embs: [1, 512].
            # Otherwise:                                    zs_clip_fgbg_features: [3, 514, 1280]. face_id_embs: [3, 512].
            # If self.iter_flags['same_subject_in_batch'], then we average the zs_clip_fgbg_features and face_id_embs to get 
            # less noisy zero-shot embeddings. Otherwise, we use instance-wise zero-shot embeddings.
            # If do_feat_distill_on_comp_prompt, then we have repeated the instances in the batch, 
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
            if faceless_img_count > 0:
                self.iter_flags['do_normal_recon'] = False
                self.iter_flags['do_unet_distill'] = True
                # Disable do_feat_distill_on_comp_prompt during unet distillation.
                self.iter_flags['do_feat_distill_on_comp_prompt']  = False

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

        # If do_feat_distill_on_comp_prompt, then we don't add noise to the zero-shot ID embeddings, 
        # to avoid distorting the ID information.
        p_perturb_face_id_embs = self.p_perturb_face_id_embs if self.iter_flags['do_unet_distill'] else 0                
        # p_perturb_face_id_embs: default 0.6.
        # The overall prob of perturb_face_id_embs: (1 - 0.5) * 0.6 = 0.3.
        self.iter_flags['perturb_face_id_embs'] = (torch.rand(1) < p_perturb_face_id_embs).item()
        if self.iter_flags['perturb_face_id_embs']:
            if not self.iter_flags['same_subject_in_batch']:
                self.iter_flags['same_subject_in_batch'] = True
                # Replace the ID features of multiple subjects in the batch to multiple copies of 
                # the first subject, before adding noise to the ID features.
                # Doing so is similar to contrastive learning: the embeddings in a batch are similar
                # (the first subject embedding + randn noise), but the generated images are quite different.
                # Therefore, the model may learn to distinguish the tiny differences in the embeddings.
                # As the embeddings are coupled with x_start and fg_mask, we need to change them to 
                # those of the first subject as well.
                # Change the batch to have the (1 subject image) * BS strcture.
                # "captions" and "delta_prompts" don't change, as different subjects share the same placeholder "z".
                # clip_bg_features is used by adaface encoder, so we repeat zs_clip_fgbg_features accordingly.
                # We don't repeat id2img_neg_prompt_embs, as it's constant and identical for different instances.
                x_start, batch_images_unnorm, img_mask, fg_mask, \
                instances_have_fg_mask, self.batch_subject_names, \
                id2img_prompt_embs, zs_clip_fgbg_features = \
                    select_and_repeat_instances(slice(0, 1), BS, 
                                                x_start, batch_images_unnorm, img_mask, fg_mask, 
                                                instances_have_fg_mask, self.batch_subject_names, 
                                                id2img_prompt_embs, zs_clip_fgbg_features)
                
            # ** Perturb the zero-shot ID image prompt embeddings with probability 0.6. **
            # ** The perturbation here is not to make the img2ada encoder more robust to random perturbations,
            # ** but to find neighbors of the subject image embeddings for UNet distillation.
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
            # Iterate among 3 ~ 5. We don't draw random numbers, so that different ranks have the same num_unet_denoising_steps,
            # which would be faster for synchronization.
            # Note since comp_distill_iter_gap == 3 or 4, we should choose a number that is co-prime with 3 and 4.
            # Otherwise, some values, e.g., 0 and 3, will never be chosen.
            num_unet_denoising_steps = self.non_comp_iters_count % 3 + 2
            self.iter_flags['num_unet_denoising_steps'] = num_unet_denoising_steps

            # Sometimes we use the subject compositional prompts as the distillation target on a UNet ensemble teacher.
            # If unet_teacher_types == ['arc2face'], then p_unet_distill_uses_comp_prompt == 0, i.e., we
            # never use the compositional prompts as the distillation target of arc2face.
            # If unet_teacher_types is ['consistentID', 'arc2face'], then p_unet_distill_uses_comp_prompt == 0.1.
            # If unet_teacher_types == ['consistentID'], then p_unet_distill_uses_comp_prompt == 0.2.
            if torch.rand(1) < self.p_unet_distill_uses_comp_prompt:
                self.iter_flags['unet_distill_uses_comp_prompt'] = True
                captions = batch[SUBJ_COMP_PROMPT]

            if num_unet_denoising_steps > 1:
                # Only use the first 1/num_unet_denoising_steps of the batch to avoid OOM.
                # If num_unet_denoising_steps >= 2, BS == 1 or 2, then HALF_BS = 1.
                # If num_unet_denoising_steps == 2 or 3, BS == 4, then HALF_BS = 2. 
                # If num_unet_denoising_steps == 4 or 5, BS == 4, then HALF_BS = 1.
                HALF_BS = torch.arange(BS).chunk(num_unet_denoising_steps)[0].shape[0]
                # Setting the minimal batch size to be 2 requires skipping 3 steps if num_unet_denoising_steps == 6.
                # Seems doing so will introduce too much artifact. Therefore it's DISABLED.
                ## The batch size when doing multi-step denoising is at least 2. 
                ## But naively doing so when num_unet_denoising_steps >= 3 may cause OOM.
                ## In that case, we need to discard the first few steps from loss computation.
                ## HALF_BS = max(2, HALF_BS)

                # REPEAT = 1 in select_and_repeat_instances(), so that it **only selects** the 
                # first HALF_BS elements without repeating.
                # clip_bg_features is used by ConsistentID adaface encoder, 
                # so we repeat zs_clip_fgbg_features as well.
                x_start, batch_images_unnorm, img_mask, fg_mask, \
                instances_have_fg_mask, self.batch_subject_names, \
                zs_clip_fgbg_features, \
                id2img_prompt_embs, id2img_neg_prompt_embs, \
                captions, subj_single_prompts, subj_comp_prompts, \
                cls_single_prompts, cls_comp_prompts \
                = select_and_repeat_instances(slice(0, HALF_BS), 1, 
                                              x_start, batch_images_unnorm, img_mask, fg_mask, 
                                              instances_have_fg_mask, self.batch_subject_names, 
                                              zs_clip_fgbg_features,
                                              id2img_prompt_embs, id2img_neg_prompt_embs,
                                              captions, subj_single_prompts, subj_comp_prompts,
                                              cls_single_prompts, cls_comp_prompts)
                
                # Update delta_prompts to have the first HALF_BS prompts.
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

        # aug_mask is renamed as img_mask.
        self.iter_flags['img_mask']                 = img_mask
        self.iter_flags['fg_mask']                  = fg_mask
        self.iter_flags['instances_have_fg_mask']       = instances_have_fg_mask
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
        if self.iter_flags['do_feat_distill_on_comp_prompt']:
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

        if self.iter_flags['do_feat_distill_on_comp_prompt']:                        
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

        if self.iter_flags['do_feat_distill_on_comp_prompt']:
            # c_in = delta_prompts is used to generate ada embeddings.
            # c_in: subj_single_prompts + subj_comp_prompts + cls_single_prompts + cls_comp_prompts
            # The cls_single_prompts/cls_comp_prompts within c_in will only be used to 
            # generate ordinary prompt embeddings, i.e., 
            # it doesn't contain subject token, and no ada embedding will be injected by embedding manager.
            # Instead, subj_single_emb, subj_comp_emb and subject ada embeddings 
            # are manually mixed into their embeddings.
            c_in = delta_prompts
            # The prompts are either (subj single, subj comp, cls single, cls comp).
            # So the first 2 sub-blocks always contain the subject tokens, and we use *_2b.    
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
                            
        # Keep extra_info['delta_prompts'] and iter_flags['delta_prompts'] the same structure.
        # (Both are tuples of 4 lists. But iter_flags['delta_prompts'] may contain more prompts
        # than those actually used in this iter.)
        # iter_flags['delta_prompts'] is not used in p_losses(). Keep it for debugging purpose.
        extra_info['delta_prompts']      = (subj_single_prompts, subj_comp_prompts, \
                                            cls_single_prompts,  cls_comp_prompts)
        extra_info['enable_lora']        = self.diffusers_unet_uses_lora

        # c_prompt_emb is the full set of embeddings of subj_single_prompts, subj_comp_prompts, 
        # cls_single_prompts, cls_comp_prompts. 
        # c_prompt_emb: [64, 77, 768]                    
        cond_context = (c_prompt_emb, c_in, extra_info)

        # self.model (UNetModel) is called in p_losses().
        #LINK #p_losses
        c_prompt_emb, c_in, extra_info = cond_context
        return self.p_losses(x_start, c_prompt_emb, c_in, extra_info)

    # apply_model() is called both during training and inference.
    def apply_model(self, x_noisy, t, cond_context):
        # self.model: DiffusionWrapper -> 
        # self.model.diffusion_model: ldm.modules.diffusionmodules.openaimodel.UNetModel
        x_recon = self.model(x_noisy, t, cond_context)
        return x_recon

    def sliced_apply_model(self, x_noisy, t, cond_context, slice_inst, enable_grad, enable_lora):
        x_noisy_ = x_noisy[slice_inst]
        t_ = t[slice_inst]
        c_prompt_emb, c_in, extra_info = cond_context
        c_prompt_emb_ = c_prompt_emb[slice_inst]
        c_in_ = c_in[slice_inst]
        extra_info_ = copy.copy(extra_info)
        extra_info_['enable_lora'] = enable_lora
        with torch.set_grad_enabled(enable_grad):
            model_output = self.apply_model(x_noisy_, t_, (c_prompt_emb_, c_in_, extra_info_))
        return model_output, extra_info_

    # do_pixel_recon: return denoised images for CLIP evaluation. 
    # if do_pixel_recon and cfg_scale > 1, apply classifier-free guidance. 
    # This is not used for the iter_type 'do_normal_recon'.
    # batch_part_has_grad: 'all', 'none', 'subject-compos'.
    def guided_denoise(self, x_start, noise, t, cond_context,
                       uncond_emb=None, img_mask=None, batch_part_has_grad='all', 
                       do_pixel_recon=False, cfg_scale=-1, capture_ca_activations=False):
        
        x_noisy = self.q_sample(x_start, t, noise)
        ca_layers_activations = None

        extra_info = cond_context[2]
        extra_info['capture_ca_activations'] = capture_ca_activations
        extra_info['img_mask'] = img_mask

        # model_output is the predicted noise.
        # if not batch_part_has_grad, we save RAM by not storing the computation graph.
        # if batch_part_has_grad, we don't have to take care of embedding_manager.force_grad.
        # Subject embeddings will naturally have gradients.
        if batch_part_has_grad == 'none':
            with torch.no_grad():
                model_output = self.apply_model(x_noisy, t, cond_context)

            if capture_ca_activations:
                ca_layers_activations = extra_info['ca_layers_activations']

        elif batch_part_has_grad == 'all':
            model_output = self.apply_model(x_noisy, t, cond_context)

            if capture_ca_activations:
                ca_layers_activations = extra_info['ca_layers_activations']

        elif batch_part_has_grad == 'subject-compos':
            # Although enable_lora is set to True, if self.diffusers_unet_uses_lora is False, it will be overridden
            # in the unet.
            model_output_ss, extra_info_ss = self.sliced_apply_model(x_noisy, t, cond_context, slice_inst=slice(0, 1), 
                                                                     enable_grad=False, enable_lora=True)
            model_output_sc, extra_info_sc = self.sliced_apply_model(x_noisy, t, cond_context, slice_inst=slice(1, 2),
                                                                     enable_grad=True,  enable_lora=True)
            model_output_c2, extra_info_c2 = self.sliced_apply_model(x_noisy, t, cond_context, slice_inst=slice(2, 4),
                                                                     enable_grad=False, enable_lora=False)
            
            model_output = torch.cat([model_output_ss, model_output_sc, model_output_c2], dim=0)
            extra_info = cond_context[2]
            if capture_ca_activations:
                # Collate three captured activation dicts into extra_info.
                ca_layers_activations = collate_dicts([extra_info_ss['ca_layers_activations'],
                                                       extra_info_sc['ca_layers_activations'],
                                                       extra_info_c2['ca_layers_activations']])
        else:
            breakpoint()

        # Get model output of both conditioned and uncond prompts.
        # Unconditional prompts and reconstructed images are never involved in optimization.
        if cfg_scale > 1:
            if uncond_emb is None:
                # Use self.uncond_context as the unconditional context.
                # uncond_context is a tuple of (uncond_emb, uncond_c_in, extra_info).
                # By default, 'capture_ca_activations' = False in a generated text context, 
                # including uncond_context. So we don't need to set it in self.uncond_context explicitly.                
                uncond_emb  = self.uncond_context[0].repeat(x_noisy.shape[0], 1, 1)

            uncond_c_in = self.uncond_context[1] * x_noisy.shape[0]
            uncond_context = (uncond_emb, uncond_c_in, self.uncond_context[2])

            # We never needs gradients on unconditional generation.
            with torch.no_grad():
                x_noisy = self.q_sample(x_start, t, noise)
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
        
        return noise_pred, x_recon, ca_layers_activations

    def multistep_denoise(self, x_start, noise, t, cond_context, 
                          uncond_emb=None, img_mask=None, batch_part_has_grad='subject-compos', 
                          cfg_scale=-1, capture_ca_activations=False,
                          num_denoising_steps=1, 
                          same_t_noise_across_instances=False):
        assert num_denoising_steps <= 10

        # Initially, x_starts only contains the original x_start.
        x_starts    = [ x_start ]
        noises      = [ noise ]
        ts          = [ t ]
        noise_preds = []
        x_recons    = []
        ca_layers_activations_list = []

        for i in range(num_denoising_steps):
            x_start = x_starts[i]
            t       = ts[i]
            noise   = noises[i]

            # batch_part_has_grad == 'subject-compos', i.e., only the subject compositional instance has gradients.
            noise_pred, x_recon, ca_layers_activations = \
                self.guided_denoise(x_start, noise, t, cond_context,
                                    uncond_emb, img_mask, batch_part_has_grad, 
                                    do_pixel_recon=True, cfg_scale=cfg_scale, 
                                    capture_ca_activations=capture_ca_activations)
            
            noise_preds.append(noise_pred)
            pred_x0 = x_recon
            # The predicted x0 is used as the x_start for the next denoising step.
            x_starts.append(pred_x0)
            x_recons.append(x_recon)
            ca_layers_activations_list.append(ca_layers_activations)

            # Sample an earlier timestep for the next denoising step.
            if i < num_denoising_steps - 1:
                # NOTE: rand_like() samples from U(0, 1), not like randn_like().
                unscaled_ts = torch.rand_like(t.float())
                # Make sure at the middle step (i = sqrt(num_denoising_steps - 1), the timestep 
                # is between 50% and 70% of the current timestep. So if num_denoising_steps = 5,
                # we take timesteps within [0.5^0.66, 0.7^0.66] = [0.63, 0.79] of the current timestep.
                # If num_denoising_steps = 4, we take timesteps within [0.5^0.72, 0.7^0.72] = [0.61, 0.77] 
                # of the current timestep.
                t_lb = t * np.power(0.5, np.power(num_denoising_steps - 1, -0.3))
                t_ub = t * np.power(0.7, np.power(num_denoising_steps - 1, -0.3))
                earlier_timesteps = (t_ub - t_lb) * unscaled_ts + t_lb
                earlier_timesteps = earlier_timesteps.long()
                noise = torch.randn_like(pred_x0)

                if same_t_noise_across_instances:
                    # If same_t_noise_across_instances, we use the same earlier_timesteps and noise for all instances.
                    earlier_timesteps = earlier_timesteps[0].repeat(x_start.shape[0])
                    noise = noise[:1].repeat(x_start.shape[0], 1, 1, 1)

                # earlier_timesteps = ts[i+1] < ts[i].
                ts.append(earlier_timesteps)
                noises.append(noise)

        return noise_preds, x_starts, x_recons, noises, ts, ca_layers_activations_list
            
    # t: timesteps.
    # c_in is the textual prompts. 
    # extra_info: a dict that contains various fields. 
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, c_prompt_emb, c_in, extra_info):
        #print(c_in)
        cond_context = (c_prompt_emb, c_in, extra_info)
        img_mask            = self.iter_flags['img_mask']
        fg_mask             = self.iter_flags['fg_mask']
        instances_have_fg_mask  = self.iter_flags['instances_have_fg_mask']
        filtered_fg_mask    = self.iter_flags.get('filtered_fg_mask', None)

        # all_subj_indices are used to extract the attention weights
        # of the subject tokens for the attention loss computation.
        # Then combine all subject indices into all_subj_indices.
        all_subj_indices    = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices'],
                                                                   self.embedding_manager.subject_string_dict)
        if self.iter_flags['do_feat_distill_on_comp_prompt']:
            # all_subj_indices_2b is used in calc_feat_delta_and_attn_norm_loss() in calc_comp_prompt_distill_loss().
            all_subj_indices_2b = \
                join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_2b'],
                                                     self.embedding_manager.subject_string_dict)
            # all_subj_indices_1b is used in calc_comp_subj_bg_preserve_loss() in calc_comp_prompt_distill_loss().
            all_subj_indices_1b = \
                join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_1b'],
                                                     self.embedding_manager.subject_string_dict)

        noise = torch.randn_like(x_start) 

        # If do_feat_distill_on_comp_prompt, we prepare the attention activations 
        # for computing distillation losses.
        if self.iter_flags['do_feat_distill_on_comp_prompt']:
            # For simplicity, we fix BLOCK_SIZE = 1, no matter the batch size.
            # We can't afford BLOCK_SIZE=2 on a 48GB GPU as it will double the memory usage.            
            BLOCK_SIZE = 1
            masks = (img_mask, fg_mask, filtered_fg_mask, instances_have_fg_mask)
            # x_start_maskfilled: transformed x_start, in which the fg area is scaled down from the input image,
            # and the bg mask area filled with noise. Returned only for logging.
            # x_start_primed: the primed (denoised) x_start_maskfilled, ready for denoising.
            # noise and masks are updated to be a 1-repeat-4 structure in prime_x_start_for_comp_prompts().
            # We return noise to make the gt_target up-to-date, which is the recon objective.
            # But gt_target is probably not referred to in the following loss computations,
            # since the current iteration is do_feat_distill_on_comp_prompt. We update it just in case.
            # masks will still be used in the loss computation. So we update them as well.
            x_start_maskfilled, x_start_primed, noise, masks, num_primed_denoising_steps = \
                self.prime_x_start_for_comp_prompts(cond_context, x_start, noise,
                                                    masks, fg_noise_amount=0.2,
                                                    BLOCK_SIZE=BLOCK_SIZE)
            # Update masks.
            img_mask, fg_mask, filtered_fg_mask, instances_have_fg_mask = masks
            self.iter_flags['fg_mask_avail_ratio'] = instances_have_fg_mask.float().mean()

            uncond_emb  = self.uncond_context[0].repeat(BLOCK_SIZE * 4, 1, 1)

            # t is randomly drawn from the middle rear 30% segment of the timesteps (noisy but not too noisy).
            t_midrear = torch.randint(int(self.num_timesteps * 0.5), int(self.num_timesteps * 0.8), 
                                      (BLOCK_SIZE,), device=x_start.device)
            # Same t_mid for all instances.
            t_midrear = t_midrear.repeat(BLOCK_SIZE * 4)

            # comp_distill_denoising_steps_range: [2, 2].
            # num_denoising_steps iterates among 2 ~ 3. We don't draw random numbers, 
            # so that different ranks have the same num_denoising_steps,
            # which might be faster for synchronization.
            W = self.comp_distill_denoising_steps_range[1] - self.comp_distill_denoising_steps_range[0] + 1
            num_comp_denoising_steps = self.comp_iters_count % W + self.comp_distill_denoising_steps_range[0]

            # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
            # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
            # (img_mask is not used in the prompt-guided cross-attention layers).
            # But we don't use img_mask in compositional iterations. Because in compositional iterations,
            # the original images don't play a role (even if comp_init_fg_from_training_image,
            # the unet doesn't consider the actual pixels outside of the subject areas, so img_mask is
            # set to None).

            # ca_layers_activations_list will be used in calc_comp_prompt_distill_loss().
            # noise_preds is not used for loss computation.
            # x_recons[-1] will be used for arcface align loss computation.
            noise_preds, x_starts, x_recons, noises, ts, ca_layers_activations_list = \
                self.multistep_denoise(x_start_primed, noise, t_midrear, cond_context,
                                       uncond_emb=uncond_emb, img_mask=None, 
                                       batch_part_has_grad='subject-compos', 
                                       cfg_scale=5, capture_ca_activations=True,
                                       num_denoising_steps=num_comp_denoising_steps,
                                       same_t_noise_across_instances=True)

            ts_1st = [ t[0].item() for t in ts ]
            print(f"comp distill denoising steps: {num_comp_denoising_steps}, ts: {ts_1st}")

            # Log x_start, x_start_maskfilled (noisy and scaled version of the first image in the batch),
            # x_start_primed (x_start_maskfilled denoised for a few steps), and the denoised images for diagnosis.
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
            # All of them are 1, indicating green.
            x_start0 = x_start[:1]
            input_image = self.decode_first_stage(x_start0)
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
            # All of them are 1, indicating green.
            log_image_colors = torch.ones(input_image.shape[0], dtype=int, device=x_start.device)
            self.cache_and_log_generations(input_image, log_image_colors, do_normalize=True)

            x_start_maskfilled = x_start_maskfilled[[0]]
            log_image_colors = torch.ones(x_start_maskfilled.shape[0], dtype=int, device=x_start.device)
            x_start_maskfilled_decoded = self.decode_first_stage(x_start_maskfilled)
            self.cache_and_log_generations(x_start_maskfilled_decoded, log_image_colors, do_normalize=True)

            x_start_primed = x_start_primed.chunk(2)[0]
            log_image_colors = torch.ones(x_start_primed.shape[0], dtype=int, device=x_start.device)
            x_start_primed_decoded = self.decode_first_stage(x_start_primed)
            self.cache_and_log_generations(x_start_primed_decoded, log_image_colors, do_normalize=True)
            
            for i, x_recon in enumerate(x_recons):
                recon_images = self.decode_first_stage(x_recon)
                # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
                # If there are multiple denoising steps, the output images are assigned different colors.
                log_image_colors = torch.ones(recon_images.shape[0], dtype=int, device=x_start.device) * (i % 4)

                self.cache_and_log_generations(recon_images, log_image_colors, do_normalize=True)

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

        # do_prompt_emb_delta_reg is always done, regardless of the iter_type.
        if self.iter_flags['do_prompt_emb_delta_reg']:
            loss_prompt_emb_delta = calc_prompt_emb_delta_loss( 
                        extra_info['c_prompt_emb_4b'], extra_info['prompt_emb_mask'])

            loss_dict.update({f'{session_prefix}/prompt_emb_delta': loss_prompt_emb_delta.mean().detach().item() })

            # The Prodigy optimizer seems to suppress the embeddings too much, 
            # so it uses a smaller scale to reduce the negative effect of prompt_emb_delta_loss.
            prompt_emb_delta_loss_scale = 1 if self.optimizer_type == 'Prodigy' else 2
            # prompt_emb_delta_reg_weight: 1e-5.
            loss += loss_prompt_emb_delta * self.prompt_emb_delta_reg_weight * prompt_emb_delta_loss_scale

        ##### begin of do_normal_recon #####
        if self.iter_flags['do_normal_recon']:          
            BLOCK_SIZE = x_start.shape[0]
            t = torch.randint(self.num_timesteps // 2, self.num_timesteps, (x_start.shape[0],), device=self.device).long()

            # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
            # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
            # (img_mask is not used in the prompt-guided cross-attention layers).
            # Don't do CFG. So uncond_emb is None.
            model_output, x_recon, ca_layers_activations = \
                self.guided_denoise(x_start, noise, t, cond_context, 
                                    uncond_emb=None, img_mask=img_mask,
                                    batch_part_has_grad='all', 
                                    # Reconstruct the images at the pixel level for CLIP loss.
                                    do_pixel_recon=True,
                                    # Do not use cfg_scale for normal recon iterations. Only do recon 
                                    # using the positive prompt.
                                    cfg_scale=-1, capture_ca_activations=True)

            # If do_normal_recon, then there's only 1 objective:
            # **Objective 1**: Align the student predicted noise with the ground truth noise.
            # bg loss is completely ignored. 
            bg_pixel_weight = 0 

            loss_subj_mb_suppress, loss_recon, loss_pred_l2 = \
                self.calc_recon_and_complem_losses(model_output, gt_target, ca_layers_activations,
                                                   all_subj_indices, img_mask, fg_mask, instances_have_fg_mask,
                                                   bg_pixel_weight, x_start.shape[0])
            v_loss_recon = loss_recon.mean().detach().item()
        
            # If fg_mask is None, then loss_subj_mb_suppress = loss_bg_mf_suppress = 0.
            if loss_subj_mb_suppress > 0:
                loss_dict.update({f'{session_prefix}/subj_mb_suppress': loss_subj_mb_suppress.mean().detach().item()})

            loss_dict.update({f'{session_prefix}/loss_recon': v_loss_recon})
            loss_dict.update({f'{session_prefix}/pred_l2': loss_pred_l2.mean().detach().item()})
            print(f"Rank {self.trainer.global_rank} single-step recon: {t.tolist()}, {v_loss_recon:.4f}")
            # loss_recon: 0.02~0.03.
            # loss_pred_l2: 0.97~0.99. Quite stable. pred_l2_loss_weight: 1e-3 -> 1e-3. 1/20~1/30 of recon loss.
            loss += loss_recon + loss_pred_l2 * self.pred_l2_loss_weight \
                    + loss_subj_mb_suppress * self.recon_subj_bg_suppress_loss_weight

            if self.use_arcface_loss and (self.arcface is not None):
                # We can only afford doing arcface_align_loss on two instances. Otherwise, OOM.
                loss_arcface_align = self.calc_arcface_align_loss(x_start[:2], x_recon[:2])
                if loss_arcface_align > 0:
                    loss_dict.update({f'{session_prefix}/arcface_align': loss_arcface_align.mean().detach().item() })
                    # loss_arcface_align: 0.5-0.8. arcface_align_loss_weight: 4e-2 => 0.02-0.032.
                    # This loss is around 1/5 of recon/distill losses (0.1).
                    loss += loss_arcface_align * self.arcface_align_loss_weight

            recon_images = self.decode_first_stage(x_recon)
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple' ]
            # all of them are 2, indicating red.
            log_image_colors = torch.ones(recon_images.shape[0], dtype=int, device=x_start.device) * 3
            self.cache_and_log_generations(recon_images, log_image_colors, do_normalize=True)
        ##### end of do_normal_recon #####

        ##### begin of do_unet_distill #####
        elif self.iter_flags['do_unet_distill']:
            t = torch.randint(self.num_timesteps // 2, self.num_timesteps, (x_start.shape[0],), device=self.device).long()

            # img_mask, fg_mask are used in recon_loss().
            loss_unet_distill = \
                self.calc_unet_distill_loss(x_start, noise, t, cond_context, extra_info, img_mask, fg_mask)

            v_loss_unet_distill = loss_unet_distill.mean().detach().item()
            loss_dict.update({f'{session_prefix}/loss_unet_distill': v_loss_unet_distill})
            # loss_unet_distill: ~0.01, so we use a very large unet_distill_weight==8 to
            # make it comparable to the recon loss.
            loss += loss_unet_distill * self.unet_distill_weight
        ##### end of do_unet_distill #####

        ###### begin of do_feat_distill_on_comp_prompt ######
        elif self.iter_flags['do_feat_distill_on_comp_prompt']:
            losses_comp_fg_bg_preserve = []
            losses_sc_mc_bg_match = []
            losses_subj_attn_norm_distill = []

            loss_names = [ 'loss_subj_comp_map_single_align_with_cls', 'loss_sc_recon_ss_fg_attn_agg', 
                           'loss_sc_recon_ss_fg_flow', 'loss_sc_recon_ss_fg_min', 'loss_sc_mc_bg_match', 
                           'loss_comp_subj_bg_attn_suppress', 'loss_comp_cls_bg_attn_suppress' ]
            
            for loss_name in loss_names:
                loss_name2 = loss_name.replace('loss_', '')
                loss_name2 = f'{session_prefix}/{loss_name2}'
                loss_dict[loss_name2] = 0

            for step_idx, ca_layers_activations in enumerate(ca_layers_activations_list):
                # If we take 3 denoising steps, then recon_loss_discard_thres will be 0.4, 0.5, 0.6, respectively,
                # i.e., more strict for the first step, and more relaxed for the last step.
                recon_loss_discard_thres = 0.4 + 0.1 * step_idx
                loss_comp_fg_bg_preserve, loss_sc_mc_bg_match = \
                    self.calc_comp_prompt_distill_loss(ca_layers_activations, filtered_fg_mask, 
                                                       instances_have_fg_mask, all_subj_indices_1b, 
                                                       BLOCK_SIZE, loss_dict, session_prefix,
                                                       recon_loss_discard_thres=recon_loss_discard_thres)

                # ca_layers_activations['outfeat'] is a dict as: layer_idx -> ca_outfeat. 
                # It contains the 3 specified cross-attention layers of UNet. i.e., layers 22, 23, 24.
                # Similar are ca_attns and ca_attns, each ca_outfeats in ca_outfeats is already 4D like [4, 8, 64, 64].

                # NOTE: loss_subj_attn_norm_distill is disabled. Since we use L2 loss for loss_sc_mc_bg_match,
                # the subj attn values are learned to not overly express in the background tokens, so no need to suppress them. 
                # Actually, explicitly discouraging the subject attn values from being too large will reduce subject authenticity.
                # NOTE: loss_feat_delta_align is disabled. It doesn't have spatial correspondance when 
                # doing the feature delta calculation.
                # These two losses are only used for monitoring the training process.

                # all_subj_indices_2b is used in calc_feat_delta_and_attn_norm_loss(), as it's used 
                # to index subj single and subj comp embeddings.
                # The indices will be shifted along the batch dimension (size doubled) 
                # within calc_feat_delta_and_attn_norm_loss() to index all the 4 blocks.
                loss_feat_delta_align, loss_subj_attn_norm_distill \
                    = self.calc_feat_delta_and_attn_norm_loss(ca_layers_activations['outfeat'], 
                                                              ca_layers_activations['attn'], 
                                                              all_subj_indices_2b, BLOCK_SIZE)

                # loss_feat_delta_align: 0.02~0.03. 
                if loss_feat_delta_align > 0:
                    loss_dict.update({f'{session_prefix}/feat_delta_align': \
                                      loss_feat_delta_align.mean().detach().item() })

                losses_comp_fg_bg_preserve.append(loss_comp_fg_bg_preserve)
                losses_subj_attn_norm_distill.append(loss_subj_attn_norm_distill)
                losses_sc_mc_bg_match.append(loss_sc_mc_bg_match)
            
            for loss_name in loss_names:
                loss_name2 = loss_name.replace('loss_', '')
                loss_name2 = f'{session_prefix}/{loss_name2}'
                if loss_name2 in loss_dict:
                    loss_dict[loss_name2] = loss_dict[loss_name2] / len(ca_layers_activations_list)

            loss_comp_fg_bg_preserve    = torch.stack(losses_comp_fg_bg_preserve).mean()
            loss_subj_attn_norm_distill = torch.stack(losses_subj_attn_norm_distill).mean()
            loss_sc_mc_bg_match         = torch.stack(losses_sc_mc_bg_match).mean()

            if loss_sc_mc_bg_match > 0:
                self.comp_iters_bg_match_loss_count += 1
                sc_mc_bg_match_loss_frac = self.comp_iters_bg_match_loss_count / (self.comp_iters_count + 1)
                loss_dict.update({f'{session_prefix}/sc_mc_bg_match_loss_frac': sc_mc_bg_match_loss_frac})

            # loss_comp_fg_bg_preserve = 0 if comp_init_fg_from_training_image and there's a valid fg_mask.
            if loss_comp_fg_bg_preserve > 0:
                loss_dict.update({f'{session_prefix}/comp_fg_bg_preserve': loss_comp_fg_bg_preserve.mean().detach().item() })
                # Keep track of the number of iterations that use comp_init_fg_from_training_image.
                self.comp_init_fg_from_training_image_count += 1
                comp_init_fg_from_training_image_frac = self.comp_init_fg_from_training_image_count / (self.comp_iters_count + 1)
                loss_dict.update({f'{session_prefix}/comp_init_fg_from_training_image_frac': comp_init_fg_from_training_image_frac})
            # loss_subj_attn_norm_distill: 0.08~0.12.
            if loss_subj_attn_norm_distill > 0:
                loss_dict.update({f'{session_prefix}/subj_attn_norm_distill':  loss_subj_attn_norm_distill.mean().detach().item() })
            
            # comp_fg_bg_preserve_loss_weight: 1e-2. loss_comp_fg_bg_preserve: 0.5-0.6.
            # loss_subj_attn_norm_distill: 0.08~0.12. DISABLED.
            # loss_sc_mc_bg_match is L2 loss, which are very small. So we scale them up by 5x to 50x.
            # loss_sc_mc_bg_match: 0.002~0.01, sc_mc_bg_match_loss_scale: 10~50 => 0.02~5.
            # rel_scale_range=(0, 1): the absolute range of the scale will be 5~50.
            sc_mc_bg_match_loss_scale = calc_dyn_loss_scale(loss_sc_mc_bg_match, (0.001, 5), (0.01, 50), 
                                                            rel_scale_range=(0, 2))
            loss += (loss_comp_fg_bg_preserve + loss_sc_mc_bg_match * sc_mc_bg_match_loss_scale) \
                    * self.comp_fg_bg_preserve_loss_weight
            
            if self.use_arcface_loss and (self.arcface is not None):
                # Trying to calc arcface_align_loss from difficult to easy steps.
                # sel_step: 0~2. 0 is the hardest for face detection (denoised once), and 2 is the easiest (denoised 3 times).
                loss_calc_count = 0
                max_loss_calc_count = 1
                for sel_step in range(len(x_recons)):
                    x_recon  = x_recons[sel_step]
                    # If there are faceless input images, then do_feat_distill_on_comp_prompt is always False.
                    # Thus, here do_feat_distill_on_comp_prompt is always True, and x_start[0] is a valid face image.
                    x_start0    = x_start.chunk(4)[0]
                    subj_recon  = x_recon.chunk(2)[0]
                    loss_arcface_align = self.calc_arcface_align_loss(x_start0, subj_recon)
                    # Found valid face images. Stop trying, since we cannot afford calculating arcface_align_loss for > 1 steps.
                    if loss_arcface_align > 0:
                        print(f"Rank-{self.trainer.global_rank} arcface_align step {sel_step+1}/{len(x_recons)}")
                        loss_dict.update({f'{session_prefix}/arcface_align': loss_arcface_align.mean().detach().item() })
                        # loss_arcface_align: 0.5-0.8. arcface_align_loss_weight: 1e-3 => 0.0005-0.0008.
                        # This loss is around 1/150 of recon/distill losses (0.1).
                        loss += loss_arcface_align * self.arcface_align_loss_weight
                        loss_calc_count += 1
                        if loss_calc_count >= max_loss_calc_count:
                            break

                if loss_calc_count > 0:
                    self.comp_iters_face_detected_count += 1
                    comp_iters_face_detected_frac = self.comp_iters_face_detected_count / self.comp_iters_count
                    loss_dict.update({f'{session_prefix}/comp_iters_face_detected_frac': comp_iters_face_detected_frac})
        ##### end of do_feat_distill_on_comp_prompt #####

        else:
            breakpoint()

        if torch.isnan(loss) and self.trainer.global_rank == 0:
            print('NaN loss detected.')
            breakpoint()

        loss_dict.update({f'{session_prefix}/loss': loss.mean().detach().item() })

        return loss, loss_dict

    def calc_arcface_align_loss(self, x_start, x_recon):
        # If there are faceless input images, then do_feat_distill_on_comp_prompt is always False.
        # Thus, here do_feat_distill_on_comp_prompt is always True, and x_start[0] is a valid face image.
        x_start_pixels    = self.decode_first_stage(x_start)
        # subj-comp instance. 
        # NOTE: use the with_grad version of decode_first_stage. Otherwise no effect.
        subj_recon_pixels = self.decode_first_stage_with_grad(x_recon)
        loss_arcface_align = self.arcface.calc_arcface_align_loss(x_start_pixels, subj_recon_pixels)
        loss_arcface_align = loss_arcface_align.to(x_start.dtype)
        return loss_arcface_align
    
    # Major losses for normal_recon iterations (loss_recon, loss_recon_subj_mb_suppress, etc.).
    # (But there are still other losses used after calling this function.)
    def calc_recon_and_complem_losses(self, model_output, target, ca_layers_activations,
                                      all_subj_indices, img_mask, fg_mask, instances_have_fg_mask, 
                                      bg_pixel_weight, BLOCK_SIZE):

        loss_subj_mb_suppress = \
            self.calc_subj_masked_bg_suppress_loss(
                ca_layers_activations['attnscore'],
                all_subj_indices, BLOCK_SIZE, fg_mask, instances_have_fg_mask)

        # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
        loss_recon, _ = calc_recon_loss(self.get_loss_func(), model_output, target, img_mask, fg_mask, 
                                        fg_pixel_weight=1, bg_pixel_weight=bg_pixel_weight)

        # Calc the L2 norm of model_output.
        loss_pred_l2 = (model_output ** 2).mean()
        return loss_subj_mb_suppress, loss_recon, loss_pred_l2


    def calc_unet_distill_loss(self, x_start, noise, t, cond_context, extra_info, img_mask, fg_mask):
        c_prompt_emb, c_in, extra_info = cond_context
        BLOCK_SIZE = x_start.shape[0]

        # num_unet_denoising_steps > 1 implies do_unet_distill, but not vice versa.
        num_unet_denoising_steps = self.iter_flags['num_unet_denoising_steps']
        # student_prompt_embs is the prompt embedding of the student model.
        student_prompt_embs = cond_context[0]
        # NOTE: when unet_teacher_types == ['unet_ensemble'], unets are specified in 
        # extra_unet_dirpaths (finetuned unets on the original SD unet); 
        # in this case they are surely not 'arc2face' or 'consistentID'.
        # The same student_prompt_embs is used by all unet_teachers.
        if self.unet_teacher_types == ['unet_ensemble']:
            teacher_contexts = [student_prompt_embs]
        else:
            # Only enable a subset of teachers. The whole set of teachers may have been initialized,
            # if id2ada_prompt_encoder.name == 'jointIDs' by setting 
            # personalization_config.params.adaface_encoder_types = ['consistentID', 'arc2face'])
            # but some are disabled by setting
            # personalization_config.params.enabled_encoders = ['consistentID'] or ['arc2face'].
            teacher_contexts = []
            encoders_num_id_vecs = self.iter_flags['encoders_num_id_vecs']
            # If id2ada_prompt_encoder.name == 'jointIDs',         then encoders_num_id_vecs is not None.
            # Otherwise, id2ada_prompt_encoder is a single encoder, and encoders_num_id_vecs is None.
            if encoders_num_id_vecs is not None:
                all_id2img_prompt_embs      = self.iter_flags['id2img_prompt_embs'].split(encoders_num_id_vecs, dim=1)
                all_id2img_neg_prompt_embs  = self.iter_flags['id2img_neg_prompt_embs'].split(encoders_num_id_vecs, dim=1)
                # If id2ada_prompt_encoder.name == 'jointIDs', the img_prompt_embs are ordered as such.
                encoder_name2idx = { 'consistentID': 0, 'arc2face': 1 }
            else:
                # Single FaceID2AdaPrompt encoder. No need to split id2img_prompt_embs/id2img_neg_prompt_embs.
                all_id2img_prompt_embs      = [ self.iter_flags['id2img_prompt_embs'] ]
                all_id2img_neg_prompt_embs  = [ self.iter_flags['id2img_neg_prompt_embs'] ]
                encoder_name2idx = { self.unet_teacher_types[0]: 0 }
                
            for unet_teacher_type in self.unet_teacher_types:
                if unet_teacher_type not in ['consistentID', 'arc2face']:
                    breakpoint()
                
                teacher_idx = encoder_name2idx[unet_teacher_type]
                if unet_teacher_type == 'arc2face':
                    # img_prompt_prefix_embs: the embeddings of a template prompt "photo of a"
                    # For arc2face, p_unet_teacher_uses_cfg is always 0. So we only pass pos_prompt_embs.
                    img_prompt_prefix_embs = self.img_prompt_prefix_embs.repeat(x_start.shape[0], 1, 1)
                    # teacher_context: [BS, 4+16, 768] = [BS, 20, 768]
                    teacher_context = torch.cat([img_prompt_prefix_embs, all_id2img_prompt_embs[teacher_idx]], dim=1)

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
                    global_id_embeds = all_id2img_prompt_embs[teacher_idx]
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
                        global_neg_id_embs = all_id2img_neg_prompt_embs[teacher_idx]
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
                                  num_denoising_steps=num_unet_denoising_steps)
        
        # **Objective 2**: Align student noise predictions with teacher noise predictions.
        # targets: replaced as the reconstructed x0 by the teacher UNet.
        # If ND = num_unet_denoising_steps > 1, then unet_teacher_noise_preds contain ND 
        # unet_teacher predicted noises (of different ts).
        # targets: [HALF_BS, 4, 64, 64] * num_unet_denoising_steps.
        targets = unet_teacher_noise_preds

        # The outputs of the remaining denoising steps will be appended to model_outputs.
        model_outputs = []
        #all_recon_images = []

        # DON'T apply neg_id_emb for recon iterations. 
        # But still apply CFG to match the teacher. So uncond_emb is the real uncond_emb.
        uncond_emb = self.uncond_context[0].repeat(BLOCK_SIZE, 1, 1)

        for s in range(num_unet_denoising_steps):
            # Predict the noise with t_s (a set of earlier t).
            # When s > 1, x_start_s is the unet_teacher predicted images in the previous step,
            # used to seed the second denoising step. 
            x_start_s = unet_teacher_x_starts[s].to(x_start.dtype)
            # noise_t, t_s are the s-th noise/t used to by unet_teacher.
            noise_t   = unet_teacher_noises[s].to(x_start.dtype)
            t_s       = all_t[s]

            # x_start_s, noise_t, t_s, unet_teacher.cfg_scale
            # are all randomly sampled from unet_teacher_cfg_scale_range in unet_teacher().
            # So, make sure unet_teacher() was called before guided_denoise() below.
            # We need to make the student's CFG scale consistent with the teacher UNet's.
            # If not self.p_unet_teacher_uses_cfg, then self.unet_teacher.cfg_scale = 1, 
            # and the cfg_scale is not used in guided_denoise().
            # ca_layers_activations is not used in unet distillation.
            # ** Intentionally do not use img_mask in unet distillation. 
            # Otherwise the task will be too easy for the student.
            model_output_s, x_recon_s, ca_layers_activations = \
                self.guided_denoise(x_start_s, noise_t, t_s, cond_context, 
                                    uncond_emb=uncond_emb, img_mask=None,
                                    batch_part_has_grad='all', do_pixel_recon=True, 
                                    cfg_scale=self.unet_teacher.cfg_scale,
                                    capture_ca_activations=False)
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
            # In the compositional iterations, unet_distill_uses_comp_prompt is always False.
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
                calc_recon_loss(self.get_loss_func(), model_output, target.to(model_output.dtype), 
                                img_mask, fg_mask, fg_pixel_weight=1,
                                bg_pixel_weight=bg_pixel_weight)

            print(f"Rank {self.trainer.global_rank} Step {s}: {all_t[s].tolist()}, {loss_unet_distill.item():.4f}")
            
            losses_unet_distill.append(loss_unet_distill)

            # Try hard to release memory after each step. But since they are part of the computation graph,
            # doing so may not have any effect :(
            model_outputs[s], targets[s] = None, None

        # If num_unet_denoising_steps > 1, most loss_unet_distill are usually 0.001~0.005, but sometimes there are a few large loss_unet_distill.
        # In order not to dilute the large loss_unet_distill, we don't divide by num_unet_denoising_steps.
        # Instead, only increase the normalizer sub-linearly.
        loss_unet_distill = sum(losses_unet_distill) / np.sqrt(num_unet_denoising_steps)

        return loss_unet_distill

    # Do denoising, collect the attention activations for computing the losses later.
    # masks: (img_mask, fg_mask, filtered_fg_mask, instances_have_fg_mask). 
    # Put them in a tuple to avoid too many arguments. The updated masks are returned.
    # For simplicity, we fix BLOCK_SIZE = 1, no matter the batch size.
    # We can't afford BLOCK_SIZE=2 on a 48GB GPU as it will double the memory usage.
    def prime_x_start_for_comp_prompts(self, cond_context, x_start, noise,
                                       masks, fg_noise_amount=0.2, BLOCK_SIZE=1):
        c_prompt_emb, c_in, extra_info = cond_context

        # Although img_mask is not explicitly referred to in the following code,
        # it's updated within select_and_repeat_instances(slice(0, BLOCK_SIZE), 4, *masks).
        img_mask, fg_mask, filtered_fg_mask, instances_have_fg_mask = masks

        if self.iter_flags['comp_init_fg_from_training_image'] and self.iter_flags['fg_mask_avail_ratio'] > 0:
            # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
            # Therefore, using fg_mask for comp_init_fg_from_training_image will force the model remember 
            # the background in the training images, which is not desirable.
            # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0.
            # fg_mask is 4D (added 1D in shared_step()). So expand instances_have_fg_mask to 4D.
            filtered_fg_mask = fg_mask.to(x_start.dtype) * instances_have_fg_mask.view(-1, 1, 1, 1)
            # x_start, fg_mask, filtered_fg_mask are scaled in init_x_with_fg_from_training_image()
            # by the same scale, to make the fg_mask and filtered_fg_mask consistent with x_start.
            # fg_noise_amount = 0.2
            x_start, fg_mask, filtered_fg_mask = \
                init_x_with_fg_from_training_image(x_start, fg_mask, filtered_fg_mask, 
                                                   base_scale_range=(0.8, 1.0),
                                                   fg_noise_amount=fg_noise_amount)

        else:
            # We have to use random noise for x_start, as the training images 
            # are not used for initialization.
            x_start.normal_()

        # Make the 4 instances in x_start, noise and t the same.
        x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)
        noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)

        x_start_maskfilled = x_start
        
        # masks may have been changed in init_x_with_fg_from_training_image(). So we update it.
        masks = (img_mask, fg_mask, filtered_fg_mask, instances_have_fg_mask)
        # Update masks to be a 1-repeat-4 structure.
        masks = select_and_repeat_instances(slice(0, BLOCK_SIZE), 4, *masks)

        # num_primed_denoising_steps iterates from 2 to 5.
        # num_primed_denoising_steps will always follow a uniform distribution of [1, 2, 3, 4].
        num_primed_denoising_steps = self.comp_iters_count % self.max_num_comp_priming_denoising_steps + 1

        # We hard-coded MIN_N_SHARED = 1, and num_shared_denoising_steps >= 1. So taking
        # num_shared_denoising_steps = max(num_primed_denoising_steps - MAX_N_SEP, MIN_N_SHARED) is always valid.
        MIN_N_SHARED = 1
        MAX_N_SEP = 2
        # If num_primed_denoising_steps > MAX_N_SEP, then we split the denoising steps into
        # shared denoising steps and separate denoising steps.
        # This is to make sure the subj init x_start and cls init x_start do not deviate too much.
        num_shared_denoising_steps = max(num_primed_denoising_steps - MAX_N_SEP, MIN_N_SHARED)
        num_sep_denoising_steps    = num_primed_denoising_steps - num_shared_denoising_steps
        all_t_list = []

        # In priming denoising steps, t is randomly drawn from the terminal 25% segment of the timesteps (very noisy).
        t_rear = torch.randint(int(self.num_timesteps * 0.75), int(self.num_timesteps * 1), 
                                (BLOCK_SIZE,), device=x_start.device)
        t      = t_rear.repeat(4)

        # ** Do num_shared_denoising_steps of shared denoising steps with the subj-mix-cls comp prompts.
        if num_shared_denoising_steps > 0:
            # Class priming denoising: Denoise x_start_1 with the comp prompts 
            # for num_shared_denoising_steps times, using self.comp_distill_priming_unet.
            x_start_1                      = x_start.chunk(4)[0]
            noise_1                        = noise.chunk(4)[0]
            t_1                            = t.chunk(4)[0]
            
            _, subj_comp_prompt_emb, _, cls_comp_prompt_emb = c_prompt_emb.chunk(4)
            uncond_emb          = self.uncond_context[0].repeat(x_start_1.shape[0], 1, 1)
            # Both cond and uncond embeddings are provided for CFG denoising.
            subj_double_context = torch.cat([subj_comp_prompt_emb, uncond_emb], dim=0)
            cls_double_context  = torch.cat([cls_comp_prompt_emb,  uncond_emb], dim=0)

            # Since we always use CFG for class priming denoising,
            # we need to pass the negative prompt as well.
            # cfg_scale_range of comp_distill_priming_unet is [2, 4].
            # primed_noises: the noises that have been used in the denoising.
            with torch.no_grad():
                primed_noise_preds, primed_x_starts, primed_noises, all_t = \
                    self.comp_distill_priming_unet(self, x_start_1, noise_1, t_1, 
                                                    # In each timestep, the unet ensemble will do denoising on the same x_start_1 
                                                    # with both subj_double_context and cls_double_context, then average the results.
                                                    # It's similar to do averaging on the prompt embeddings, but yields sharper results.
                                                    # From the outside, the unet ensemble is transparent, like a single unet.
                                                    teacher_context=[subj_double_context, cls_double_context], 
                                                    num_denoising_steps=num_shared_denoising_steps,
                                                    # Same t and noise across instances.
                                                    same_t_noise_across_instances=True,
                                                    global_t_lb=300)
                
            # Repeat the 1-instance denoised x_start_1 to 2-instance x_start_2, i.e., one single, one comp instances.
            x_start_2   = primed_x_starts[-1].repeat(2, 1, 1, 1).to(dtype=x_start.dtype)
            t_2         = all_t[-1].repeat(2)
            all_t_list  += [ ti[0].item() for ti in all_t ]
        else:
            # Class priming denoising: Denoise x_start_2 with the class single/comp prompts 
            # for num_sep_denoising_steps times, using self.comp_distill_priming_unet.
            # We only use the second half-batch (class instances) for class priming denoising.
            # x_start and t are initialized as 1-repeat-4 at above, so the second half is 1-repeat-2.
            # i.e., the denoising only differs in the prompt embeddings, but not in the x_start, t, and noise.
            x_start_2   = x_start.chunk(2)[1]
            t_2         = t.chunk(2)[1]

        # Ensure the two instances (one single, one comp) use the same t, although on different x_start_2 and noise.
        noise_2 = torch.randn_like(x_start[:2*BLOCK_SIZE]) #.repeat(2, 1, 1, 1)
        subj_prompt_emb, cls_prompt_emb = c_prompt_emb.chunk(2)
        uncond_emb          = self.uncond_context[0].repeat(x_start_2.shape[0], 1, 1)
        # x_double_context contains both the positive and negative prompt embeddings.
        subj_double_context = torch.cat([subj_prompt_emb, uncond_emb], dim=0)
        cls_double_context  = torch.cat([cls_prompt_emb,  uncond_emb], dim=0)

        # ** Do num_sep_denoising_steps of separate denoising steps with the single-comp prompts.
        #     x_start_2[0] is denoised with the single prompt (both subj single and cls single before averaging), 
        # and x_start_2[1] is denoised with the comp   prompt (both subj comp   and cls comp   before averaging).
        # Since we always use CFG for class priming denoising, we need to pass the negative prompt as well.
        # default cfg_scale_range=[2, 4].
        with torch.no_grad():
            primed_noise_preds, primed_x_starts, primed_noises, all_t = \
                self.comp_distill_priming_unet(self, x_start_2, noise_2, t_2, 
                                                # In each timestep, the unet ensemble will do denoising on the same x_start_2 
                                                # with both subj_double_context and cls_double_context, then average the results.
                                                # It's similar to do averaging on the prompt embeddings, but yields sharper results.
                                                # From the outside, the unet ensemble is transparent, like a single unet.
                                                teacher_context=[subj_double_context, cls_double_context], 
                                                num_denoising_steps=num_sep_denoising_steps,
                                                # Same t and noise across instances.
                                                same_t_noise_across_instances=True)
        
        all_t_list += [ ti[0].item() for ti in all_t ]
        print(f"Rank {self.trainer.global_rank} step {self.global_step}: "
                f"subj-cls ensemble prime denoising {num_primed_denoising_steps} steps {all_t_list}")
        
        # The last primed_x_start is the final denoised image (with the smallest t).
        # So we use it as the x_start to be denoised by the 4-type prompt set.
        # We need to let the subject and class instances use the same x_start. 
        # Therefore, we repeat primed_x_starts[-1] twice.
        x_start = primed_x_starts[-1].repeat(2, 1, 1, 1).to(dtype=x_start.dtype)

        # Regenerate the noise, since the noise has been used above.
        # Ensure the two types of instances (single, comp) use different noise.
        # ** But subj and cls instances use the same noise.
        noise           = torch.randn_like(x_start[:BLOCK_SIZE]).repeat(4, 1, 1, 1)
        x_start_primed  = x_start
        # noise and masks are updated to be a 1-repeat-4 structure in prime_x_start_for_comp_prompts().
        # We return noise to make the gt_target up-to-date, which is the recon objective.
        # But gt_target is probably not referred to in the following loss computations,
        # since the current iteration is do_feat_distill_on_comp_prompt. We update it just in case.
        # masks will still be used in the loss computation. So we return updated masks as well.
        return x_start_maskfilled, x_start_primed, noise, masks, num_primed_denoising_steps
    
    def calc_comp_prompt_distill_loss(self, ca_layers_activations, filtered_fg_mask, instances_have_fg_mask,
                                      all_subj_indices_1b, BLOCK_SIZE, loss_dict, session_prefix,
                                      recon_loss_discard_thres=0.4):
        # ca_outfeats is a dict as: layer_idx -> ca_outfeat. 
        # It contains the 3 specified cross-attention layers of UNet. i.e., layers 22, 23, 24.
        # Similar are ca_attns and ca_attns, each ca_outfeats in ca_outfeats is already 4D like [4, 8, 64, 64].
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

            comp_subj_bg_preserve_loss_dict = \
                self.calc_comp_subj_bg_preserve_loss(ca_outfeats,
                                                     ca_layers_activations['attn_out'],
                                                     ca_layers_activations['q'],
                                                     ca_layers_activations['attn'], 
                                                     filtered_fg_mask, instances_have_fg_mask,
                                                     all_subj_indices_1b, BLOCK_SIZE,
                                                     recon_feat_objectives={'attn_out': 'L2', 'outfeat': 'L2'},
                                                     recon_loss_discard_thres=recon_loss_discard_thres,
                                                     bg_align_loss_scheme='L2',
                                                     do_feat_attn_pooling=True)
            
            loss_names = [ 'loss_subj_comp_map_single_align_with_cls', 'loss_sc_recon_ss_fg_attn_agg', 
                           'loss_sc_recon_ss_fg_flow', 'loss_sc_recon_ss_fg_min', 'loss_sc_mc_bg_match', 
                           'loss_comp_subj_bg_attn_suppress', 'loss_comp_cls_bg_attn_suppress' ]
            
            # loss_sc_recon_ss_fg_attn_agg and loss_sc_recon_ss_fg_flow, loss_comp_cls_bg_attn_suppress 
            # are returned to be monitored, not to be optimized.
            # Only their counterparts -- loss_sc_recon_ss_fg_min, loss_comp_subj_bg_attn_suppress 
            # are optimized.
            loss_subj_comp_map_single_align_with_cls, loss_sc_recon_ss_fg_attn_agg, \
            loss_sc_recon_ss_fg_flow, loss_sc_recon_ss_fg_min, loss_sc_mc_bg_match, \
            loss_comp_subj_bg_attn_suppress, loss_comp_cls_bg_attn_suppress \
                = [ comp_subj_bg_preserve_loss_dict.get(loss_name, 0) for loss_name in loss_names ] 

            for loss_name in loss_names:
                if loss_name in comp_subj_bg_preserve_loss_dict and comp_subj_bg_preserve_loss_dict[loss_name] > 0:
                    loss_name2 = loss_name.replace('loss_', '')
                    # Accumulate the loss values to loss_dict when there are multiple denoising steps.
                    add_dict_to_dict(loss_dict, {f'{session_prefix}/{loss_name2}': comp_subj_bg_preserve_loss_dict[loss_name].mean().detach().item() })

            # loss_subj_comp_map_single_align_with_cls is L1 loss on attn maps, so it's is small: 0.5~2e-5.
            subj_comp_map_single_align_with_cls_loss_scale = 1
            comp_subj_bg_attn_suppress_loss_scale = 0.02
            sc_recon_ss_fg_min_loss_scale = 10

            # loss_sc_recon_ss_fg_min: 0.1~0.12. -> 1~1.2.
            loss_sc_ss_fg_recon = loss_sc_recon_ss_fg_min * sc_recon_ss_fg_min_loss_scale
            # loss_sc_mc_bg_match:             0.005~0.008 -> 0.25~0.4.
            # loss_comp_subj_bg_attn_suppress: 0.1~0.2     -> 0.002~0.004.
            # loss_sc_mc_bg_match has similar effects to suppress the subject attn values in the background tokens.
            # Therefore, we use a very small comp_subj_bg_attn_suppress_loss_scale = 0.02.
            loss_comp_fg_bg_preserve = loss_subj_comp_map_single_align_with_cls * subj_comp_map_single_align_with_cls_loss_scale \
                                        + loss_sc_ss_fg_recon + loss_comp_subj_bg_attn_suppress * comp_subj_bg_attn_suppress_loss_scale
        else:
            loss_comp_fg_bg_preserve = loss_sc_mc_bg_match = 0

        return loss_comp_fg_bg_preserve, loss_sc_mc_bg_match
    
    # calc_feat_delta_and_attn_norm_loss() is used by calc_comp_prompt_distill_loss().
    def calc_feat_delta_and_attn_norm_loss(self, ca_outfeats, ca_attns, subj_indices_2b, BLOCK_SIZE):
        # do_feat_distill_on_comp_prompt iterations. No ordinary image reconstruction loss.
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
        # Feature map spatial sizes are all 64*64.
        feat_distill_layer_weights = { 23: 1, 24: 1, 
                                     }

        # attn norm distillation is applied to almost all conditioning layers.
        attn_norm_distill_layer_weights = { 7: 0.5, 8: 0.5,
                                            12: 1.,
                                            16: 1., 17: 1.,
                                            18: 1.,
                                            19: 1., 20: 1., 
                                            21: 1., 22: 1., 
                                            23: 1., 24: 1.,                                   
                                           }

        # Normalize the weights above so that each set sum to 1.
        feat_distill_layer_weights          = normalize_dict_values(feat_distill_layer_weights)
        attn_norm_distill_layer_weights     = normalize_dict_values(attn_norm_distill_layer_weights)

        # K_subj: 4, number of embeddings per subject token.
        K_subj = len(subj_indices_2b[0]) // len(torch.unique(subj_indices_2b[0]))
        subj_indices_4b = double_token_indices(subj_indices_2b, BLOCK_SIZE * 2)

        loss_layers_feat_delta_align        = []
        loss_layers_subj_attn_norm_distill  = []

        subj_single_feat_grad_scaler = gen_gradient_scaler(0.1)

        for unet_layer_idx, ca_outfeat in ca_outfeats.items():
            if (unet_layer_idx not in feat_distill_layer_weights) and (unet_layer_idx not in attn_norm_distill_layer_weights):
                continue

            # attn_mat: [4, 8, 256, 77] => [4, 77, 8, 256].
            # We don't need BP through attention into UNet.
            attn_mat = ca_attns[unet_layer_idx].permute(0, 3, 1, 2)
            # subj_attn_4b: [4, 8, 256]  (1 embedding  for 1 token)  => [4, 1, 8, 256] => [4, 8, 256]
            # or            [16, 8, 256] (4 embeddings for 1 token)  => [4, 4, 8, 256] => [4, 8, 256]
            # BLOCK_SIZE*4: this batch contains 4 blocks. Each block should have one instance.
            subj_attn_4b = attn_mat[subj_indices_4b].reshape(BLOCK_SIZE*4, K_subj, *attn_mat.shape[2:]).sum(dim=1)
            # subj_single_subj_attn, ...: [1, 8, 256] (1 embedding  for 1 token) 
            # or                          [1, 8, 256] (4 embeddings for 1 token)
            subj_single_subj_attn, subj_comp_subj_attn, cls_single_subj_attn, cls_comp_subj_attn \
                = subj_attn_4b.chunk(4)

            if unet_layer_idx in attn_norm_distill_layer_weights:
                attn_norm_distill_layer_weight     = attn_norm_distill_layer_weights[unet_layer_idx]

                cls_comp_subj_attn_gs       = cls_comp_subj_attn.detach()
                cls_single_subj_attn_gs     = cls_single_subj_attn.detach()

                # mean(dim=-1): average across the 64 feature channels.
                # Align the attention corresponding to each embedding individually.
                # Note cls_*subj_attn use *_gs versions.
                # The L1 loss of the average attention values of the subject tokens, at each head and each instance.
                loss_layer_subj_comp_attn_norm   = F.l1_loss(subj_comp_subj_attn.abs().mean(dim=-1), cls_comp_subj_attn_gs.abs().mean(dim=-1))
                loss_layer_subj_single_attn_norm = F.l1_loss(subj_single_subj_attn.abs().mean(dim=-1), cls_single_subj_attn_gs.abs().mean(dim=-1))
                # loss_subj_attn_norm_distill uses L1 loss, which tends to be in 
                # smaller magnitudes than the delta loss. So it will be scaled up later in p_losses().
                loss_layers_subj_attn_norm_distill.append(( loss_layer_subj_comp_attn_norm + loss_layer_subj_single_attn_norm ) \
                                                          * attn_norm_distill_layer_weight)

            if unet_layer_idx not in feat_distill_layer_weights:
                continue

            feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]
            # subj_single_feat, ...: [1, 1280, 16, 16]
            subj_single_feat, subj_comp_feat, cls_single_feat, cls_comp_feat \
                = ca_outfeat.chunk(4)

            # [4, 320, 64, 64] -> [4, 320, 31, 31]
            ca_outfeat = pool_feat_or_attn_mat(ca_outfeat)
            # ca_outfeat_3d: [4, 320, 31, 31] -> [4, 320, 961] -> [4, 961, 320]
            ca_outfeat_3d = ca_outfeat.reshape(*ca_outfeat.shape[:2], -1).permute(0, 2, 1)
            # subj_single_feat_3d, ...: [1, 961, 320]
            subj_single_feat_3d, subj_comp_feat_3d, cls_single_feat_3d, cls_comp_feat_3d \
                = ca_outfeat_3d.chunk(4)

            cls_single_feat_3d_gs  = cls_single_feat_3d.detach()
            cls_comp_feat_3d_gs    = cls_comp_feat_3d.detach()
            # subj_single_feat_grad_scaler reduces grad by 10x.
            subj_single_feat_3d_gs = subj_single_feat_grad_scaler(subj_single_feat_3d)

            # ortho_subtract() is done on each image token individually, potentially with different scales.
            comp_feat_delta   = ortho_subtract(subj_comp_feat_3d,      cls_comp_feat_3d_gs)
            # subj_single_feat is gs'ed by 10x to avoid it from degeneration.
            single_feat_delta = ortho_subtract(subj_single_feat_3d_gs, cls_single_feat_3d_gs)
                
            # single_feat_delta, comp_feat_delta: [1, 320], ...
            # Pool the spatial dimensions H, W to remove spatial information.
            # The gradient goes back to single_feat_delta -> subj_comp_feat,
            # as well as comp_feat_delta -> cls_comp_feat.
            # If stop_single_grad, the gradients to subj_single_feat and cls_single_feat are stopped, 
            # as these two images should look good by themselves (since they only contain the subject).
            # Note the learning strategy to the single image features should be different from 
            # the single embeddings, as the former should be optimized to look good by itself,
            # while the latter should be optimized to cater for two objectives: 1) the conditioned images look good,
            # and 2) the embeddings are amendable to composition.
            loss_layer_feat_delta_align = \
                calc_ref_cosine_loss(comp_feat_delta, single_feat_delta, 
                                     emb_mask=None,
                                     exponent=2, do_demeans=[False, False],
                                     first_n_dims_into_instances=2, 
                                     aim_to_align=True, 
                                     ref_grad_scale=1)
            loss_layers_feat_delta_align.append(loss_layer_feat_delta_align * feat_distill_layer_weight)

        loss_feat_delta_align       = sum(loss_layers_feat_delta_align)
        loss_subj_attn_norm_distill = sum(loss_layers_subj_attn_norm_distill)

        return loss_feat_delta_align, loss_subj_attn_norm_distill

    def calc_subj_masked_bg_suppress_loss(self, ca_attnscore, subj_indices, 
                                          BLOCK_SIZE, fg_mask, instance_has_fg_mask=None):
        if (subj_indices is None) or (len(subj_indices) == 0) or (fg_mask is None) \
          or (instance_has_fg_mask is not None and instance_has_fg_mask.sum() == 0):
            return 0

        # Discard the first few bottom layers from alignment.
        # attn_align_layer_weights: relative weight of each layer. 
        # Feature map spatial sizes are all 64*64.
        attn_align_layer_weights = { 23: 1, 24: 1, 
                                   }
                
        # Normalize the weights above so that each set sum to 1.
        attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)
        # K_subj: 9, number of embeddings per subject token.
        K_subj = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        subj_mb_suppress_scale      = 0.05
        mfmb_contrast_attn_margin   = 0.4

        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        subj_indices = (subj_indices[0][:BLOCK_SIZE*K_subj], subj_indices[1][:BLOCK_SIZE*K_subj])

        loss_layers_subj_mb_suppress    = []

        for unet_layer_idx, unet_attn in ca_attnscore.items():
            if (unet_layer_idx not in attn_align_layer_weights):
                continue

            attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]
            # [2, 8, 256, 77] / [2, 8, 64, 77] =>
            # [2, 77, 8, 256] / [2, 77, 8, 64]
            attn_mat = unet_attn.permute(0, 3, 1, 2)

            # subj_attn: [8, 8, 64] -> [2, 4, 8, 64] sum among K_subj embeddings -> [2, 8, 64]
            subj_attn = sel_emb_attns_by_indices(attn_mat, subj_indices, do_sum=True, do_mean=False)

            fg_mask2 = resize_mask_to_target_size(fg_mask, "fg_mask", subj_attn.shape[-1], 
                                                  mode="nearest|bilinear")
            # Repeat 8 times to match the number of attention heads (for normalization).
            fg_mask2 = fg_mask2.reshape(BLOCK_SIZE, 1, -1).repeat(1, subj_attn.shape[1], 1)
            fg_mask3 = torch.zeros_like(fg_mask2)
            # Set fractional values (due to resizing) to 1.
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

            subj_attn_at_mf = subj_attn * fg_mask3
            # Protect subject emb activations on fg areas.
            subj_attn_at_mf = subj_attn_at_mf.detach()
            # subj_attn_at_mb: [BLOCK_SIZE, 8, 64].
            # mb: mask foreground locations, mask background locations.
            subj_attn_at_mb = subj_attn * bg_mask3

            # fg_mask3: [BLOCK_SIZE, 8, 64]
            # avg_subj_attn_at_mf: [BLOCK_SIZE, 1, 1]
            # keepdim=True, since attn probs at all locations will use them as references (subtract them).
            avg_subj_attn_at_mf = masked_mean(subj_attn_at_mf, fg_mask3, dim=(1,2), keepdim=True)

            '''
            avg_subj_attn_at_mb = masked_mean(subj_attn_at_mb, bg_mask3, dim=(1,2), keepdim=True)
            if 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
                print(f'layer {unet_layer_idx}')
                print(f'avg_subj_attn_at_mf: {avg_subj_attn_at_mf.mean():.4f}, avg_subj_attn_at_mb: {avg_subj_attn_at_mb.mean():.4f}')
            '''

            # Encourage avg_subj_attn_at_mf (subj_attn averaged at foreground locations) 
            # to be at least larger by mfmb_contrast_attn_margin = 0.4 than 
            # subj_attn_at_mb at any background locations.
            # If not, clamp() > 0, incurring a loss.
            # layer_subj_mb_excess: [BLOCK_SIZE, 8, 64].
            layer_subj_mb_excess = subj_attn_at_mb + mfmb_contrast_attn_margin - avg_subj_attn_at_mf
            # Compared to masked_mean(), mean() is like dynamically reducing the loss weight when more and more 
            # activations conform to the margin restrictions.
            loss_layer_subj_mb_suppress   = masked_mean(layer_subj_mb_excess, 
                                                        layer_subj_mb_excess > 0, 
                                                        instance_weights=instance_has_fg_mask)

            # loss_layer_subj_bg_contrast_at_mf is usually 0, 
            # so loss_subj_mb_suppress is much smaller than loss_bg_mf_suppress.
            # subj_mb_suppress_scale: 0.05.
            loss_layers_subj_mb_suppress.append(loss_layer_subj_mb_suppress \
                                                * attn_align_layer_weight * subj_mb_suppress_scale)
            
        loss_subj_mb_suppress = sum(loss_layers_subj_mb_suppress)
    
        return loss_subj_mb_suppress

    # Intuition of comp_fg_bg_preserve_loss: 
    # In distillation iterations, if comp_init_fg_from_training_image, then at fg_mask areas, x_start is initialized with 
    # the noisy input images. (Otherwise in distillation iterations, x_start is initialized as pure noise.)
    # Essentially, it's to mask the background out of the input images with noise.
    # Therefore, intermediate features at the foreground with single prompts should be close to those of the original images.
    # Features with comp prompts should be similar with the original images at the foreground.
    # So features under comp prompts should be close to features under single prompts, at fg_mask areas.
    # (The features at background areas under comp prompts are the compositional contents, which shouldn't be regularized.) 
    # NOTE: subj_indices are used to compute loss_comp_subj_bg_attn_suppress and loss_comp_cls_bg_attn_suppress.
    def calc_comp_subj_bg_preserve_loss(self, ca_outfeats, ca_attn_outs, ca_qs, ca_attns, 
                                        fg_mask, instances_have_fg_mask, subj_indices, BLOCK_SIZE,
                                        recon_feat_objectives={'attn_out': 'L2', 'outfeat': 'cosine'},
                                        recon_loss_discard_thres=0.4,
                                        bg_align_loss_scheme='L2', do_feat_attn_pooling=True):
        # No masks available. loss_comp_subj_fg_feat_preserve, loss_comp_subj_bg_attn_suppress are both 0.
        if fg_mask is None or instances_have_fg_mask.sum() == 0:
            return 0, 0, 0, 0, 0, 0

        # Feature map spatial sizes are all 64*64.
        # Remove layer 22, as the losses at this layer are often too large 
        # and are discarded at a high percentage.
        elastic_matching_layer_weights = { 23: 1, 24: 1, 
                                         }
        
        # Normalize the weights above so that each set sum to 1.
        elastic_matching_layer_weights  = normalize_dict_values(elastic_matching_layer_weights)
        
        # fg_mask is 4D. So expand instances_have_fg_mask to 4D.
        # *_4b means it corresponds to a 4-block batch (batch size = 4 * BLOCK_SIZE).
        fg_mask_4b = fg_mask * instances_have_fg_mask.view(-1, 1, 1, 1)

        # K_subj: 4, number of embeddings per subject token.
        K_subj   = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
        # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
        #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
        # ind_subj_subj_B_1b, ind_subj_subj_N_1b: [0, 0, 0, 0], [5, 6, 7, 8].
        ind_subj_subj_B_1b, ind_subj_subj_N_1b = subj_indices[0][:BLOCK_SIZE*K_subj], subj_indices[1][:BLOCK_SIZE*K_subj]
        ind_subj_B = torch.cat([ind_subj_subj_B_1b,                     ind_subj_subj_B_1b + BLOCK_SIZE,
                                ind_subj_subj_B_1b + 2 * BLOCK_SIZE,    ind_subj_subj_B_1b + 3 * BLOCK_SIZE], dim=0)
        ind_subj_N = ind_subj_subj_N_1b.repeat(4)
        
        loss_dict = {}
        
        for unet_layer_idx, ca_outfeat in ca_outfeats.items():
            if unet_layer_idx not in elastic_matching_layer_weights:
                continue
            elastic_matching_layer_weight = elastic_matching_layer_weights[unet_layer_idx]

            # ca_outfeat: [4, 1280, 8, 8]
            ca_feat_h, ca_feat_w = ca_outfeat.shape[-2:]

            # ca_layer_q: [4, 1280, 64] -> [4, 1280, 8, 8]
            ca_layer_q  = ca_qs[unet_layer_idx]
            ca_attn_out = ca_attn_outs[unet_layer_idx]
            # This way of calculation ca_q_h is to consider the case when the height and width might not be the same.
            ca_q_h = int(np.sqrt(ca_layer_q.shape[2] * ca_outfeat.shape[2] // ca_outfeat.shape[3]))
            ca_q_w = ca_layer_q.shape[2] // ca_q_h
            ca_layer_q = ca_layer_q.reshape(ca_layer_q.shape[0], -1, ca_q_h, ca_q_w)

            # ca_attn_out: [B, D, N] -> [B, D, H, W].
            ca_attn_out = ca_attn_out.reshape(*ca_attn_out.shape[:2], ca_feat_h, ca_feat_w)

            # Some layers resize the input feature maps. So we need to resize ca_outfeat to match ca_layer_q.
            if ca_outfeat.shape[2:] != ca_layer_q.shape[2:]:
                ca_outfeat = F.interpolate(ca_outfeat, size=ca_layer_q.shape[2:], mode="bilinear", align_corners=False)
                
            ###### elastic matching loss ######
            # q of each layer is used to compute the correlation matrix between subject-single and subject-comp instances,
            # as well as class-single and class-comp instances.
            # ca_attn_out is used to compute the reconstruction loss between subject-single and subject-comp instances 
            # (using the correlation matrix), as well as class-single and class-comp instances.
            # Flatten the spatial dimensions of ca_attn_out.
            # ca_layer_q, ca_attn_out, ca_outfeat: [4, 1280, 8, 8] -> [4, 1280, 64].
            ca_layer_q  = ca_layer_q.reshape(*ca_layer_q.shape[:2], -1)
            ca_attn_out = ca_attn_out.reshape(*ca_attn_out.shape[:2], -1)
            ca_outfeat  = ca_outfeat.reshape(*ca_outfeat.shape[:2], -1)
            # fg_mask_4b: [4, 1, 64, 64] => [4, 1, 8, 8]
            fg_mask_4b \
                = resize_mask_to_target_size(fg_mask_4b, "fg_mask_4b", (ca_feat_h, ca_feat_w), 
                                             mode="nearest|bilinear", warn_on_all_zero=False)
            # ss_fg_mask: [4, 1, 8, 8] -> [1, 1, 8, 8] 
            ss_fg_mask = fg_mask_4b.chunk(4)[0]
            # ss_fg_mask: [1, 1, 8, 8] -> [1, 1, 64]. Spatial dims are collapsed.
            ss_fg_mask = ss_fg_mask.reshape(*ss_fg_mask.shape[:2], -1)

            # sc_map_ss_fg_prob, mc_map_ms_fg_prob: [1, 1, 64]
            # removed loss_layer_ms_mc_fg_match to save computation.
            # loss_layer_subj_comp_map_single_align_with_cls: loss of alignment between two soft mappings: sc_map_ss_prob and mc_map_ms_prob.
            # sc_map_ss_fg_prob_below_mean and mc_map_ms_fg_prob_below_mean are used as fg/bg soft masks of comp instances
            # to suppress the activations on background areas.
            
            loss_layer_subj_comp_map_single_align_with_cls, losses_sc_recon_ss_fg, \
            loss_layer_sc_mc_bg_match, sc_map_ss_fg_prob_below_mean, mc_map_ss_fg_prob_below_mean \
                = calc_elastic_matching_loss(unet_layer_idx, self.flow_model, 
                                             ca_layer_q, ca_attn_out, ca_outfeat, 
                                             ss_fg_mask, ca_feat_h, ca_feat_w, 
                                             recon_feat_objectives=recon_feat_objectives,
                                             recon_loss_discard_thres=recon_loss_discard_thres,
                                             fg_bg_cutoff_prob=0.25, num_flow_est_iters=12,
                                             bg_align_loss_scheme=bg_align_loss_scheme,
                                             do_feat_attn_pooling=do_feat_attn_pooling)

            loss_sc_recon_ss_fg_attn_agg, loss_sc_recon_ss_fg_flow, loss_sc_recon_ss_fg_min = losses_sc_recon_ss_fg

            add_dict_to_dict(loss_dict, 
                              { 'loss_subj_comp_map_single_align_with_cls': loss_layer_subj_comp_map_single_align_with_cls * elastic_matching_layer_weight,
                                'loss_sc_recon_ss_fg_attn_agg':   loss_sc_recon_ss_fg_attn_agg * elastic_matching_layer_weight,
                                'loss_sc_recon_ss_fg_flow':       loss_sc_recon_ss_fg_flow * elastic_matching_layer_weight,
                                'loss_sc_recon_ss_fg_min':        loss_sc_recon_ss_fg_min * elastic_matching_layer_weight,
                                'loss_sc_mc_bg_match':            loss_layer_sc_mc_bg_match * elastic_matching_layer_weight })
                
            if sc_map_ss_fg_prob_below_mean is None or mc_map_ss_fg_prob_below_mean is None:
                continue
            
            ##### unet_attn fg preservation loss & bg suppression loss #####
            unet_attn = ca_attns[unet_layer_idx]
            # attn_mat: [4, 8, 256, 77] => [4, 77, 8, 256] 
            attn_mat = unet_attn.permute(0, 3, 1, 2)
            # subj_subj_attn: [4, 77, 8, 256] -> [4 * K_subj, 8, 256] -> [4, K_subj, 8, 256]
            # attn_mat and subj_subj_attn are not pooled.
            subj_attn = attn_mat[ind_subj_B, ind_subj_N].reshape(BLOCK_SIZE * 4, K_subj, *attn_mat.shape[2:])
            # Sum over 9 subject embeddings. [4, K_subj, 8, 256] -> [4, 8, 256].
            # The scale of the summed attention won't be overly large, since we've done 
            # distribute_embedding_to_M_tokens() to them.
            subj_attn = subj_attn.sum(dim=1)
            H = int(np.sqrt(subj_attn.shape[-1]))
            # subj_attn_hw: [4, 8, 256] -> [4, 8, 8, 8].
            subj_attn_hw = subj_attn.reshape(*subj_attn.shape[:2], H, H)
            # At some layers, the output features are upsampled. So we need to 
            # upsample the attn map to match the output features.
            if subj_attn_hw.shape[2:] != (ca_feat_h, ca_feat_w):
                subj_attn_hw = F.interpolate(subj_attn_hw, size=(ca_feat_h, ca_feat_w), mode="bilinear", align_corners=False)

            # subj_attn_hw: [4, 8, 8, 8] -> [4, 8, 8, 8] -> [4, 8, 64].
            subj_attn_flat = subj_attn_hw.reshape(*subj_attn_hw.shape[:2], -1)

            subj_single_subj_attn, subj_comp_subj_attn, cls_single_subj_attn, cls_comp_subj_attn \
                = subj_attn_flat.chunk(4)

            cls_comp_subj_attn_gs = cls_comp_subj_attn.detach()

            subj_comp_subj_attn_pos   = subj_comp_subj_attn.clamp(min=0)
            cls_comp_subj_attn_gs_pos = cls_comp_subj_attn_gs.clamp(min=0)

            if do_feat_attn_pooling:
                subj_comp_subj_attn_pos   = pool_feat_or_attn_mat(subj_comp_subj_attn_pos,   (ca_feat_h, ca_feat_w))
                cls_comp_subj_attn_gs_pos = pool_feat_or_attn_mat(cls_comp_subj_attn_gs_pos, (ca_feat_h, ca_feat_w))

            # Suppress the subj attention probs on background areas in comp instances.
            # subj_comp_subj_attn: [1, 8, 64]. ss_bg_mask_map_to_sc: [1, 1, 64].
            # sc_map_ss_fg_prob_below_mean: bg token should have fg attn probs below mean. Therefore
            # these token are regarded as bg tokens.
            loss_layer_comp_subj_bg_attn_suppress = masked_mean(subj_comp_subj_attn_pos, 
                                                                sc_map_ss_fg_prob_below_mean)
            loss_layer_comp_cls_bg_attn_suppress  = masked_mean(cls_comp_subj_attn_gs_pos,  
                                                                mc_map_ss_fg_prob_below_mean)

            add_dict_to_dict(loss_dict,
                             { 'loss_comp_subj_bg_attn_suppress': loss_layer_comp_subj_bg_attn_suppress * elastic_matching_layer_weight,
                               'loss_comp_cls_bg_attn_suppress':  loss_layer_comp_cls_bg_attn_suppress * elastic_matching_layer_weight })
            
        return loss_dict
    
    # samples: a single 4D [B, C, H, W] np array, or a single 4D [B, C, H, W] torch tensor, 
    # or a list of 3D [C, H, W] torch tensors.
    # Data type of samples could be uint (0-25), or float (-1, 1) or (0, 1).
    # If (-1, 1), then we should set do_normalize=True.
    # img_colors: a single 1D torch tensor, indexing colors = [ None, 'green', 'red', 'purple' ]
    # For raw output from raw output from SD decode_first_stage(),
    # samples are be between [-1, 1], so we set do_normalize=True, which will convert and clamp to [0, 1].
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
        elif self.optimizer_type == 'Prodigy':
            OptimizerClass = Prodigy
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

        # Adam series, AdEMAMix series.
        if 'Prodigy' not in self.optimizer_type:
            if 'adam' in self.optimizer_type.lower():
                opt = OptimizerClass(opt_params_with_lrs, weight_decay=self.weight_decay,
                                    betas=self.adam_config.betas)

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

# The old LDM UNet wrapper.
class DiffusionWrapper(pl.LightningModule): 
    def __init__(self, diff_model_config):
        super().__init__()
        # diffusion_model: UNetModel
        self.diffusion_model = instantiate_from_config(diff_model_config)

    # t: a 1-D batch of timesteps (during training: randomly sample one timestep for each instance).
    def forward(self, x, t, cond_context):
        c_prompt_emb, c_in, extra_info = cond_context
        out = self.diffusion_model(x, t, context=c_prompt_emb, context_in=c_in, extra_info=extra_info)

        return out

# The diffusers UNet wrapper.
class DiffusersUNetWrapper(pl.LightningModule):
    def __init__(self, unet_dirpath, torch_dtype=torch.bfloat16,
                 enable_lora=False, lora_rank=128):
        super().__init__()
        # diffusion_model is actually a UNet. Use this variable name to be 
        # consistent with DiffusionWrapper.
        # By default, .eval() is called in the constructor to deactivate DropOut modules.
        self.diffusion_model = UNet2DConditionModel.from_pretrained(
                                unet_dirpath, torch_dtype=torch_dtype
                               )
        # Conform with main.py() of setting debug_attn.
        self.diffusion_model.debug_attn = False
        self.to(torch_dtype)

        # Only capture the activations of the last 3 CA layers.
        self.captured_layer_indices = [22, 23, 24] # => 13, 14, 15
        self.global_enable_lora = enable_lora
        self.lora_rank = lora_rank
        hooked_attn_procs = self.set_attn_processors()
        self.hooked_attn_procs = torch.nn.ModuleList(hooked_attn_procs.values())

    # Adapted from ConsistentIDPipeline:set_ip_adapter().
    def set_attn_processors(self):
        unet = self.diffusion_model
        attn_procs = {}
        hooked_attn_procs = {}

        for name, attn_proc in unet.attn_processors.items():
            # Only capture the activations of the last 3 CA layers.
            if not name.startswith("up_blocks.3"):
                # Not the last 3 CA layers. Don't enable LoRA or capture activations.
                # The difference with the default attn_proc is that AttnProcessor_LoRA_Capture handles img_mask.
                attn_procs[name] = AttnProcessor_LoRA_Capture(
                    capture_ca_activations=False, enable_lora=False)
                continue
            # cross_attention_dim: 768.
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if cross_attention_dim is None:
                # Self attention. Don't enable LoRA or capture activations.
                # The difference with the default attn_proc is that AttnProcessor_LoRA_Capture handles img_mask.
                attn_procs[name] = AttnProcessor_LoRA_Capture(
                    capture_ca_activations=False, enable_lora=False)
                continue

            block_id = 3
            # hidden_size: 320
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            hooked_attn_proc = AttnProcessor_LoRA_Capture(
                capture_ca_activations=True, enable_lora=self.global_enable_lora,
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, 
                # LoRA up is initialized to 0. So no need to worry that the LoRA output may be too large.
                lora_rank=self.lora_rank, lora_scale=1
            )
            attn_procs[name]        = hooked_attn_proc
            hooked_attn_procs[name] = hooked_attn_proc
        
        unet.set_attn_processor(attn_procs)
        print(f"Set {len(hooked_attn_procs)} CrossAttn LoRA processors on {hooked_attn_procs.keys()}.")
        return hooked_attn_procs
    
    def forward(self, x, t, cond_context, out_dtype=torch.float32):
        c_prompt_emb, c_in, extra_info = cond_context
        img_mask = extra_info.get('img_mask', None) if extra_info is not None else None
        capture_ca_activations = extra_info.get('capture_ca_activations', False) if extra_info is not None else False
        # self.global_enable_lora is the global flag. 
        # We can override this call by setting extra_info['enable_lora'].
        enable_lora = extra_info.get('enable_lora', True) if extra_info is not None else True
        # If enable_lora is set to False globally, then disable it in this call.
        enable_lora = enable_lora and self.global_enable_lora

        # capture_ca_activations is set to capture_ca_activations in reset_attn_cache_and_flags().
        for hooked_attn_proc in self.hooked_attn_procs:
            hooked_attn_proc.reset_attn_cache_and_flags(capture_ca_activations, enable_lora)

        if capture_ca_activations:
            # Only get the 3-layer output features of the last up blocks (which contains the last 3 CA layers).
            self.diffusion_model.up_blocks[3].capture_outfeats = capture_ca_activations
            # Back up the forward() method of the last up block.
            up_blocks3_forward = self.diffusion_model.up_blocks[3].forward
            # Replace the forward() method of the last up block with a capturing method.
            self.diffusion_model.up_blocks[3].forward = \
                CrossAttnUpBlock2D_forward_capture.__get__(self.diffusion_model.up_blocks[3])

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            out = self.diffusion_model(sample=x, timestep=t, encoder_hidden_states=c_prompt_emb, 
                                       cross_attention_kwargs={'img_mask': img_mask},
                                       return_dict=False)[0]

        captured_activations = { k: {} for k in ('outfeat', 'attn', 'attnscore', 'q', 'attn_out') }

        if capture_ca_activations:
            # 3 output feature tensors of the three (resnet, attn) pairs in the last up block.
            # Each (resnet, attn) pair corresponds to a TimestepEmbedSequential layer in the LDM implementation.
            #LINK ldm/modules/diffusionmodules/openaimodel.py#unet_layers
            cached_outfeats = self.diffusion_model.up_blocks[3].cached_outfeats

            # Restore everything.
            # capture_ca_activations is set to False in reset_attn_cache_and_flags().
            self.diffusion_model.up_blocks[3].capture_outfeats = False
            self.diffusion_model.up_blocks[3].forward = up_blocks3_forward
            # Release one of the references to the cached outfeats.
            self.diffusion_model.up_blocks[3].cached_outfeats = {}

            for layer_idx in self.captured_layer_indices:
                # Subtract 22 to ca_layer_idx to match the layer index in up_blocks[3].
                # 22, 23, 24 -> 0, 1, 2.
                layer_idx2 = layer_idx - 22
                for k in captured_activations.keys():
                    if k == 'outfeat':
                        captured_activations['outfeat'][layer_idx] = cached_outfeats[layer_idx2].to(out_dtype)
                    else:
                        cached_activations = self.hooked_attn_procs[layer_idx2].cached_activations
                        captured_activations[k][layer_idx] = cached_activations[k].to(out_dtype)

                # Restore enable_lora to the global flag.
                self.hooked_attn_procs[layer_idx2].reset_attn_cache_and_flags(False, self.global_enable_lora)

        extra_info['ca_layers_activations'] = captured_activations
        out = out.to(out_dtype)

        return out
