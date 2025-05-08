import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR, ConstantLR, PolynomialLR, \
                                     CosineAnnealingWarmRestarts, CyclicLR
from ldm.modules.lr_scheduler import SequentialLR2
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_only
from ldm.c_adamw import AdamW as CAdamW
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
import queue
from threading import Thread

from ldm.util import    exists, default, instantiate_from_config, disabled_train, load_ckpt_to_cpu, inplace_model_copy, \
                        calc_prompt_emb_delta_loss, calc_comp_subj_bg_preserve_loss, calc_recon_loss, \
                        calc_recon_and_suppress_losses, calc_attn_norm_loss, calc_sc_rep_attn_distill_loss, \
                        calc_subj_masked_bg_suppress_loss, calc_subj_attn_cross_t_diff_loss, \
                        distribute_embedding_to_M_tokens_by_dict, join_dict_of_indices_with_key_filter, \
                        collate_dicts, split_dict, select_and_repeat_instances, halve_token_indices, \
                        merge_cls_token_embeddings, anneal_perturb_embedding, calc_dyn_loss_scale, \
                        count_optimized_params, count_params, torch_uniform, pixel_bboxes_to_latent, \
                        RollingStats, save_grid, set_seed_per_rank_and_batch
                        
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from adaface.diffusers_attn_lora_capture import set_up_attn_processors, set_up_ffn_loras, \
                                                set_lora_and_capture_flags, CrossAttnUpBlock2D_forward_capture, \
                                                get_captured_activations
from peft.utils.constants import DUMMY_TARGET_MODULES
from ldm.prodigy import Prodigy
from adaface.unet_teachers import create_unet_teacher
from gma.network import GMA
from gma.utils.utils import load_checkpoint as gma_load_checkpoint

import copy
from functools import partial
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from ldm.modules.arcface_wrapper import ArcFaceWrapper

import sys
torch.set_printoptions(precision=4, sci_mode=False)
import cv2
import platform
import insightface
import itertools

# Check the architecture
arch = platform.machine()

if arch != "arm64" and arch != "aarch64":
    try:
        import bitsandbytes as bnb
        print("bitsandbytes imported successfully!")
    except ImportError:
        print("bitsandbytes is not installed or cannot be imported.")
else:
    print("Skipping bitsandbytes import on arm64 architecture.")

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 base_model_path,
                 comp_unet_weight_path=None,
                 lightning_auto_optimization=True,
                 timesteps=1000,
                 beta_schedule="linear",
                 monitor=None,
                 first_stage_key="image",
                 channels=3,
                 clip_denoised=True,    # clip the range of denoised variables, not the CLIP model.
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 unfreeze_unet=False,
                 unet_lr=0.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 optimizer_type='CAdamW',
                 adam_config=None,
                 prodigy_config=None,
                 comp_distill_iter_gap=5,
                 cls_subj_mix_ratio=0.4,        
                 prompt_emb_delta_reg_weight=1e-4,
                 recon_subj_mb_suppress_loss_weight=0.2, 
                 comp_sc_subj_mb_suppress_loss_weight=0.2,
                 sc_fg_face_suppress_mask_shrink_ratio=0.3,
                 # Percent in each edge: [0.15, 0.6].
                 comp_sc_fg_mask_percent_range=[0.0225, 0.36],
                 # Maybe we should set the face align loss threshold higher during the earlier stages, 
                 # and reduce it gradually as the training progresses,
                 # since the adaface model can better and better capture the face features.
                 recon_face_align_loss_thres=0.7,
                 comp_sc_face_align_loss_thres=0.7,
                 # 'face portrait' is only valid for humans/animals. 
                 # On objects, use_fp_trick will be ignored, even if it's set to True.
                 use_fp_trick=True,
                 unet_distill_iter_gap=3,
                 unet_distill_weight=8, # Boost up the unet distillation loss by 8 times.
                 unet_teacher_types=None,
                 max_num_unet_distill_denoising_steps=4,
                 max_num_comp_priming_denoising_steps=4,
                 num_recon_denoising_steps=2,
                 num_comp_distill_denoising_steps=4,
                 redenoise_subj_single_crop_mix_weight=0.3,
                 p_unet_teacher_uses_cfg=0.6,
                 unet_teacher_cfg_scale_range=[1.5, 2.5],
                 p_unet_distill_uses_comp_prompt=0,
                 p_gen_rand_id_for_id2img=0,
                 p_perturb_face_id_embs=0.2,
                 perturb_face_id_embs_std_range=[0.3, 0.6],
                 p_normal_recon_on_pure_noise=0.4,
                 p_unet_distill_on_pure_noise=0.5,
                 subj_rep_prompts_count=2,
                 p_do_adv_attack_when_recon_on_images=0,
                 recon_adv_mod_mag_range=[0.001, 0.003],
                 recon_bg_pixel_weight=0.1,
                 use_face_flow_for_sc_matching_loss=True,
                 arcface_align_loss_weight=1e-2,
                 unet_uses_attn_lora=True,
                 recon_uses_ffn_lora=True,
                 comp_uses_ffn_lora=True,
                 unet_lora_rank=192,
                 unet_lora_scale_down=8,
                 attn_lora_layer_names=['q', 'k', 'v', 'out'],
                 q_lora_updates_query=False,
                 # ps_comp_attn_aug: [p for no aug, p for shrink cross attn, p for mix mc attn with sc].
                 # We don't apply shrink cross attn and mix mc attn with sc at the same time.
                 # Since ps_comp_attn_aug = [0, 0.5, 0.5], we always do 
                 # either shrink_cross_attn or mix_sc_mc_attn.
                 ps_comp_attn_aug=[0, 0.5, 0.5],
                 cross_attn_shrink_factor=0.5,
                 # res_hidden_states_gradscale: gradient scale for residual hidden states.
                 res_hidden_states_gradscale=0.5,
                 log_attn_level=0,
                 ablate_img_embs=False,
                 lora_weight_decay=0.01,
                 use_diffusers_vae_for_encoding=False,
                ):
        
        super().__init__()
        self.lightning_auto_optimization = lightning_auto_optimization

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        # clip_denoised: clip the range of denoised variables, not the CLIP model.
        self.clip_denoised = clip_denoised
        self.first_stage_key = first_stage_key
        self.channels = channels

        self.comp_unet_weight_path                  = comp_unet_weight_path
        self.comp_distill_iter_gap                  = comp_distill_iter_gap
        self.prompt_emb_delta_reg_weight            = prompt_emb_delta_reg_weight
        self.recon_subj_mb_suppress_loss_weight     = recon_subj_mb_suppress_loss_weight
        self.recon_face_align_loss_thres            = recon_face_align_loss_thres
        self.comp_sc_subj_mb_suppress_loss_weight   = comp_sc_subj_mb_suppress_loss_weight
        self.sc_fg_face_suppress_mask_shrink_ratio  = sc_fg_face_suppress_mask_shrink_ratio
        self.comp_sc_fg_mask_percent_range          = comp_sc_fg_mask_percent_range
        self.comp_sc_face_align_loss_thres          = comp_sc_face_align_loss_thres
        # mix some of the subject embedding denoising results into the class embedding denoising results for faster convergence.
        # Otherwise, the class embeddings are too far from subject embeddings (person, man, woman), 
        # posing too large losses to the subject embeddings.
        self.cls_subj_mix_ratio                     = cls_subj_mix_ratio

        self.use_fp_trick                           = use_fp_trick
        self.unet_distill_iter_gap                  = unet_distill_iter_gap if self.training else 0
        self.unet_distill_weight                    = unet_distill_weight
        self.unet_teacher_types                     = list(unet_teacher_types) if unet_teacher_types is not None else None
        self.p_unet_teacher_uses_cfg                = p_unet_teacher_uses_cfg
        self.unet_teacher_cfg_scale_range           = unet_teacher_cfg_scale_range
        self.max_num_unet_distill_denoising_steps   = max_num_unet_distill_denoising_steps
        self.p_unet_distill_on_pure_noise           = p_unet_distill_on_pure_noise

        self.max_num_comp_priming_denoising_steps   = max_num_comp_priming_denoising_steps
        self.num_recon_denoising_steps              = num_recon_denoising_steps
        self.num_comp_distill_denoising_steps       = num_comp_distill_denoising_steps
        self.redenoise_subj_single_crop_mix_weight  = redenoise_subj_single_crop_mix_weight

        # Sometimes we use the subject compositional instances as the distillation target on a UNet ensemble teacher.
        # If unet_teacher_types == ['arc2face'], then p_unet_distill_uses_comp_prompt == 0, i.e., we
        # never use the compositional instances as the distillation target of arc2face.
        # If unet_teacher_types is ['consistentID', 'arc2face'], then p_unet_distill_uses_comp_prompt == 0.
        # If unet_teacher_types == ['consistentID'], then p_unet_distill_uses_comp_prompt == 0.1.
        # NOTE: If compositional iterations are enabled, then we don't do unet distillation on the compositional prompts.
        if self.unet_teacher_types == ['consistentID'] and self.comp_distill_iter_gap <= 0:
            self.p_unet_distill_uses_comp_prompt = p_unet_distill_uses_comp_prompt
        else:
            self.p_unet_distill_uses_comp_prompt = 0

        self.p_gen_rand_id_for_id2img               = p_gen_rand_id_for_id2img
        self.p_perturb_face_id_embs                 = p_perturb_face_id_embs
        self.perturb_face_id_embs_std_range         = perturb_face_id_embs_std_range
        self.p_normal_recon_on_pure_noise           = p_normal_recon_on_pure_noise
        self.subj_rep_prompts_count                 = subj_rep_prompts_count
        self.p_do_adv_attack_when_recon_on_images   = p_do_adv_attack_when_recon_on_images
        self.recon_adv_mod_mag_range                = recon_adv_mod_mag_range
        self.recon_bg_pixel_weight                  = recon_bg_pixel_weight

        self.comp_iters_count                        = 0
        self.non_comp_iters_count                    = 0
        self.unet_distill_iters_count                = 0
        self.unet_distill_on_noise_iters_count       = 0
        self.normal_recon_iters_count                = 0
        self.normal_recon_face_images_on_image_stats = RollingStats(num_values=2, window_size=600, stat_type='sum')
        self.normal_recon_face_images_on_noise_stats = RollingStats(num_values=2, window_size=200, stat_type='sum')
        self.normal_recon_face_align_loss_kept_frac  = RollingStats(num_values=1, window_size=200, stat_type='mean')
        self.comp_sc_face_detected_frac              = RollingStats(num_values=1, window_size=200, stat_type='mean')
        self.comp_mc_face_detected_frac              = RollingStats(num_values=1, window_size=200, stat_type='mean')
        self.comp_sc_face_suppressed_frac            = RollingStats(num_values=1, window_size=200, stat_type='mean')
        self.comp_sc_face_align_loss_kept_frac       = RollingStats(num_values=1, window_size=200, stat_type='mean')
        self.comp_iters_bg_has_face_count            = 0
        self.comp_iters_bg_match_loss_count          = 0
        self.adaface_adv_iters_count                 = 0
        self.adaface_adv_success_iters_count         = 0

        self.cached_inits = {}
        self.do_prompt_emb_delta_reg = (self.prompt_emb_delta_reg_weight > 0)

        self.init_iteration_flags()

        self.unet_uses_attn_lora    = unet_uses_attn_lora
        self.recon_uses_ffn_lora    = recon_uses_ffn_lora
        self.comp_uses_ffn_lora     = comp_uses_ffn_lora
        self.unet_lora_rank         = unet_lora_rank
        self.unet_lora_scale_down   = unet_lora_scale_down
        self.attn_lora_layer_names  = attn_lora_layer_names
        self.q_lora_updates_query   = q_lora_updates_query
        self.ps_comp_attn_aug       = torch.tensor(ps_comp_attn_aug)
        self.cross_attn_shrink_factor = cross_attn_shrink_factor
        self.res_hidden_states_gradscale = res_hidden_states_gradscale

        self.lora_weight_decay      = lora_weight_decay
        self.ablate_img_embs        = ablate_img_embs
        self.log_attn_level         = log_attn_level

        self.model = DiffusersUNetWrapper(base_model_path=base_model_path, 
                                            torch_dtype=torch.float16,
                                            use_attn_lora=self.unet_uses_attn_lora,
                                            # attn_lora_layer_names: ['q', 'k', 'v', 'out'], 
                                            # add lora layers to all components in the designated cross-attn layers.
                                            attn_lora_layer_names=self.attn_lora_layer_names,
                                            use_ffn_lora=True,
                                            # attn QKV dim: 768, lora_rank: 192, 1/4 of 768.
                                            lora_rank=self.unet_lora_rank, 
                                            attn_lora_scale_down=self.unet_lora_scale_down,   # 8
                                            ffn_lora_scale_down=self.unet_lora_scale_down,    # 8
                                            cross_attn_shrink_factor=self.cross_attn_shrink_factor,
                                            # q_lora_updates_query = True: q is updated by the LoRA layer.
                                            # False: q is not updated, and an additional q2 is updated and returned.
                                            q_lora_updates_query=self.q_lora_updates_query
                                            )
        self.model.setup_hooks_and_loras()
        # Always use diffusers vae for decoding
        self.vae = self.model.pipeline.vae
        # Use diffusers vae for encoding if use_diffusers_vae_for_encoding.
        self.use_diffusers_vae_for_encoding = use_diffusers_vae_for_encoding

        if comp_unet_weight_path is not None:
            # base_unet_state_dict and comp_unet_state_dict are on CPU, and won't consume extra GPU RAM.
            self.base_unet_state_dict = { k: v.cpu().pin_memory() for k, v in self.model.diffusion_model.state_dict().items() }
            self.comp_unet_state_dict = load_ckpt_to_cpu(comp_unet_weight_path, pinned=True)
        else:
            self.base_unet_state_dict = None
            self.comp_unet_state_dict = None

        # face_detector is a light weight and highly efficient face detector.
        self.face_detector = insightface.model_zoo.get_model('models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx')
        self.face_detector.prepare(ctx_id=0, input_size=(640, 640))

        count_params(self.model, verbose=True)

        self.optimizer_type = optimizer_type
        self.adam_config = adam_config
        self.use_face_flow_for_sc_matching_loss = use_face_flow_for_sc_matching_loss
        self.arcface_align_loss_weight = arcface_align_loss_weight

        if 'Prodigy' in self.optimizer_type:
            self.prodigy_config = prodigy_config

        self.unfreeze_unet = unfreeze_unet
        self.unet_lr = unet_lr

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

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

        print(f"Deleting {num_del_keys} keys {deleted_keys[0]} ... {deleted_keys[-1]} from state_dict.")
        num_remaining_keys = len(list(sd.keys()))
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        # Restored from models/stable-diffusion-v-1-5/v1-5-dste8-vae.safetensors with 1018 or 1581 
        # missing and 1 unexpected keys (1018: including VAE keys, 1581: deleting VAE keys).
        # This is OK, because the missing keys are from the UNet model, which is replaced by DiffusersUNetWrapper,
        # and the key names are different with the original LDM UNet model.
        # NOTE: if not use_diffusers_vae_for_encoding, we still load first_stage_model from the checkpoint. 
        # len(self.first_stage_model.state_dict().keys()) = 248. 1018 + 248 = 1266.
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing[0]} ... {missing[-1]}")
        if len(unexpected) > 0:
            if len(unexpected) > 1:
                print(f"Unexpected Keys: {unexpected[0]} ... {unexpected[-1]}")
            else:
                print(f"Unexpected Key: {unexpected[0]}")

        print(f"Successfully loaded {num_remaining_keys - len(unexpected)} keys")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod,   t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod,           t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

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
                            'do_comp_feat_distill':             False,
                            'use_comp_distill_weights':         False,
                            'do_prompt_emb_delta_reg':          False,
                            'unet_distill_uses_comp_prompt':    False,
                            'unet_distill_on_pure_noise':       False,
                            'use_fp_trick':                     False,
                            'normal_recon_on_pure_noise':              False,
                          }
        
    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    def shared_step(self, batch):
        raise NotImplementedError("shared_step() is not implemented in DDPM.")

    def training_step(self, batch, batch_idx):
        prev_iter_use_comp_distill_weights = self.iter_flags.get('use_comp_distill_weights', False)
        self.init_iteration_flags()

        epoch = self.trainer.current_epoch
        # For reproducibility, fix the seed for each batch.
        # Don't use global_step to set the seed, as it repeats when using grad accumulation.
        # Use batch_idx to set the seed, as it is different for each batch.
        set_seed_per_rank_and_batch(self.trainer.global_rank, epoch, batch_idx)

        # If we use global_step to decide the iter type, then
        # ** due to grad accumulation (global_step increases 1 after 2 iterations), 
        # ** each type of iter is actually executed twice in a row,
        # which might not be ideal for optimization (iter types are not fully diversified across iterations).
        # But since we need to switch unet weights according to the iter type during training, 
        # it would be more efficient to take two consecutive accumulation steps 
        # of the same type on the same weight.
        if self.comp_distill_iter_gap > 0 and (self.global_step % self.comp_distill_iter_gap == 0):
            self.iter_flags['do_comp_feat_distill']     = True
            self.iter_flags['do_normal_recon']          = False
            self.iter_flags['do_unet_distill']          = False
            self.iter_flags['do_prompt_emb_delta_reg']  = self.do_prompt_emb_delta_reg
            self.comp_iters_count += 1
        else:
            self.iter_flags['do_comp_feat_distill']     = False
            self.non_comp_iters_count += 1
            if self.unet_distill_iter_gap > 0 and (self.non_comp_iters_count % self.unet_distill_iter_gap == 0):
                self.iter_flags['do_normal_recon']      = False
                self.iter_flags['do_unet_distill']      = True
                # Disable do_prompt_emb_delta_reg during unet distillation.
                self.iter_flags['do_prompt_emb_delta_reg'] = False
                self.unet_distill_iters_count += 1
            else:
                self.iter_flags['do_normal_recon']      = True
                self.iter_flags['do_unet_distill']      = False
                self.iter_flags['do_prompt_emb_delta_reg'] = self.do_prompt_emb_delta_reg
                self.normal_recon_iters_count += 1

        # Switch model weights when switching between normal recon / unet distill and comp feat distill.
        # ** Only switch half of the time ** to avoid feature space degeneration.
        if not prev_iter_use_comp_distill_weights and self.iter_flags['do_comp_feat_distill'] \
          and self.comp_unet_state_dict: # and torch.rand(1) < 0.5:
            print("Switching to comp distill unet weights")
            self.iter_flags['use_comp_distill_weights'] = True
            self.model.load_unet_state_dict(self.comp_unet_state_dict)
        elif prev_iter_use_comp_distill_weights and (not self.iter_flags['do_comp_feat_distill']) \
          and self.base_unet_state_dict:
            print("Switching to base unet weights")
            self.iter_flags['use_comp_distill_weights'] = False
            self.model.load_unet_state_dict(self.base_unet_state_dict)

        loss, mon_loss_dict = self.shared_step(batch)
        self.log_dict(mon_loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if not self.lightning_auto_optimization:
            self.manual_backward(loss)
            self.clip_gradients(optimizer, gradient_clip_val=self.trainer.gradient_clip_val, 
                                gradient_clip_algorithm=self.trainer.gradient_clip_algorithm)

            if (batch_idx + 1) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

        return loss

# LatentDiffusion inherits from DDPM. So:
# LatentDiffusion.model = DiffusersUNetWrapper()
class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 personalization_config,
                 cond_stage_key="image",
                 is_embedding_manager_trainable=True,
                 concat_mode=True,
                 cond_stage_forward=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):

        self.scale_by_std = scale_by_std    # Always False
        # for backwards compatibility after implementation of DiffusionWrapper

        # cond_stage_config is a dict:
        # {'target': 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'}
        # Not sure why it's compared with a string

        # base_model_path and ignore_keys are popped from kwargs, so they won't be passed to the base class DDPM.
        base_model_path = kwargs.get("base_model_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(*args, **kwargs)

        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.is_embedding_manager_trainable = is_embedding_manager_trainable

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        if not self.use_diffusers_vae_for_encoding:
            self.instantiate_first_stage(first_stage_config)
        else:
            self.first_stage_model = None
        self.instantiate_cond_stage(cond_stage_config)

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = (base_model_path is not None)
        if base_model_path is not None:
            # Don't load the position embedding of CLIP, as we may change the max number of tokens in the prompt.
            # In addition, we've loaded the CLIP model weights, including the position embedding, in the ctor of
            # FrozenCLIPEmbedder.
            ignore_keys.append('cond_stage_model.transformer.text_model.embeddings.position_embedding.weight')
            # Ignore all keys of the UNet model (key starts with 'model') and the VAE (first_stage_model), 
            # since we are using diffusers UNet model and VAE.
            # We still need to load the CLIP (cond_stage_model).
            # We've changed the openai CLIP to transformers CLIP, so in principle we don't need to load the CLIP weights again.
            # However, in the ckpt, the CLIP may be finetuned and better than the pretrained CLIP weights.
            # NOTE: we use diffusers vae to decode, but still use ldm VAE to encode.
            ignore_keys.extend(['model'])
            if self.use_diffusers_vae_for_encoding:
                # Ignore the VAE keys, since we are using diffusers VAE.
                ignore_keys.extend(['first_stage_model'])
            self.init_from_ckpt(base_model_path, ignore_keys)
        
        if self.unet_distill_iter_gap > 0 and self.unet_teacher_types is not None:
            # ** OBSOLETE ** When unet_teacher_types == 'unet_ensemble' or unet_teacher_types contains multiple values,
            # device, unets, extra_unet_dirpaths and unet_weights_in_ensemble are needed. 
            # Otherwise, they are not needed.
            self.unet_teacher = create_unet_teacher(self.unet_teacher_types, 
                                                    device='cpu',
                                                    unets=None,
                                                    extra_unet_dirpaths=None,
                                                    unet_weights_in_ensemble=None,
                                                    p_uses_cfg=self.p_unet_teacher_uses_cfg,
                                                    cfg_scale_range=self.unet_teacher_cfg_scale_range)
        else:
            self.unet_teacher = None

        if self.comp_distill_iter_gap > 0:
            # self.comp_unet_weight_path: "models/ensemble/sar-unet".
            # Although using RealisticVision UNet has better compositionality than sar UNet,
            # seems the domain gap with original SD is wider, and the semantics doesn't effectively 
            # pass to the subsequent denoising by the sar UNet.
            # Therefore, we still use sar UNET to prime x_start for compositional distillation.
            unet = UNet2DConditionModel.from_pretrained(self.comp_unet_weight_path, torch_dtype=torch.float16)
            # comp_distill_unet is a diffusers unet used to do a few steps of denoising 
            # on the compositional prompts, before the actual compositional distillation.
            # So float16 is sufficient.
            unets = [unet]
            unet_weights_in_ensemble = [1]

            self.comp_distill_priming_unet = \
                create_unet_teacher('unet_ensemble', 
                                    # A trick to avoid creating multiple UNet instances.
                                    # Same underlying unet, applied with different prompts, then mixed.
                                    unets = unets,
                                    unet_types=None,
                                    extra_unet_dirpaths=None,
                                    # unet_weights_in_ensemble: [0.2, 0.8]. The "first unet" uses subject embeddings, 
                                    # the second uses class embeddings. This means that,
                                    # when aggregating the results of using subject embeddings vs. class embeddings,
                                    # we give more weights to the class embeddings for better compositionality.
                                    unet_weights_in_ensemble = unet_weights_in_ensemble,
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
            # If not unfreeze_unet, then disable the training of the UNetk, 
            # and only train the embedding_manager.
            self.model.eval()
            self.model.train = disabled_train
            # unet lora params are set to requires_grad = False here.
            # But in embedding_manager.optimized_parameters(), they are set to requires_grad = True.
            for param in self.model.parameters():
                param.requires_grad = False

        self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)
        if self.is_embedding_manager_trainable:
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

        if self.arcface_align_loss_weight > 0:
            # arcface will be moved to GPU automatically.
            self.arcface = ArcFaceWrapper('cpu')
            # Disable training mode, as this mode doesn't accept only 1 image as input.
            self.arcface.train = disabled_train
            for param in self.arcface.parameters():
                param.requires_grad = False
        else:
            self.arcface = None

        self.sample_image_queue = queue.Queue(maxsize=120)
        self.cache_start_iter = 0
        self.sample_save_thread = None

    def on_train_start(self):
        if self.trainer.global_rank == 0 and self.sample_save_thread is None:
            self.sample_save_thread = Thread(target=self.save_samples_worker, args=(60,), daemon=True)
            self.sample_save_thread.start()
            print("Started individual sample saving thread...")

        # uncond_context is a tuple of (uncond_emb, uncond_prompt_in, extra_info).
        # uncond_context[0]: [1, 77, 768].
        with torch.no_grad():
            self.uncond_context         = self.get_text_conditioning([""], text_conditioning_iter_type='plain_text_iter')
            # "photo of a" is the template of Arc2face. Including an extra BOS token, the length is 4.
            img_prompt_prefix_context   = self.get_text_conditioning(["photo of a"], text_conditioning_iter_type='plain_text_iter')
        # img_prompt_prefix_context: [1, 4, 768]. Abandon the remaining text paddings.
        self.img_prompt_prefix_embs = img_prompt_prefix_context[0][:1, :4]

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
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
        self.cond_stage_model.initialize_hooks()
        
    def instantiate_embedding_manager(self, config, text_embedder):
        unet_lora_modules = self.model.unet_lora_modules
        model = instantiate_from_config(config, text_embedder=text_embedder,
                                        unet_lora_modules=unet_lora_modules)
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
    # If do_comp_feat_distill, then 1 call on delta prompts (NOTE: delta prompts have a large batch size).
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
                              text_conditioning_iter_type=None, real_batch_size=-1):
        # cond_in: a list of prompts: ['an illustration of a dirty z', 'an illustration of the cool z']
        # each prompt in c is encoded as [1, 77, 768].
        # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
        self.cond_stage_model.device = self.device
        if randomize_clip_weights:
            self.cond_stage_model.sample_last_layers_skip_weights()
            
        if text_conditioning_iter_type is None:
            # Guess text_conditioning_iter_type from the iteration flags.
            if self.iter_flags['do_comp_feat_distill']:
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
                                                               text_conditioning_iter_type,
                                                               real_batch_size)

        # prompt_embeddings: [B, 77, 768]
        prompt_embeddings = self.cond_stage_model.encode(cond_in, embedding_manager=self.embedding_manager)

        if self.training:
            # If cls_delta_string_indices is not empty, then it must be a compositional 
            # distillation iteration, and placeholder_indices only contains the indices of the subject 
            # instances. Whereas cls_delta_string_indices only contains the indices of the
            # class instances. Therefore, cls_delta_string_indices is used here.
            # NOTE: after merging, if there are multiple cls tokens, their embeddings become one (averaged),
            # and the consequent token embeddings are moved towards the beginning of prompt_embeddings.
            # Example: cls_delta_string_indices = [(2, 4, 2, 'lisa'), (3, 4, 2, 'lisa')]
            # Then in the 2nd and 3rd instances, the 4th and 5th tokens are cls tokens and are averaged.
            # The 6th~76th token embeddings are moved towards the beginning of prompt_embeddings.
            # BUG: after merge_cls_token_embeddings(), embedding_manager.prompt_emb_mask is not updated.
            # But this should be a minor issue.
            prompt_embeddings = merge_cls_token_embeddings(prompt_embeddings, 
                                                           self.embedding_manager.cls_delta_string_indices)

        # return_prompt_embs_type: ['id', 'text', 'text_id']. Training default: 'text', i.e., 
        # the conventional text embeddings returned by the clip encoder (embedding manager in the middle).
        # 'id': the subject embeddings only. 
        # 'text_id': concatenate the text embeddings with the subject IMAGE embeddings.
        # 'id' and 'text_id' are ablation settings to evaluate the original ID2ImgPrompt module.
        # NOTE the subject image embeddings don't go through CLIP.
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
                        'placeholder2indices':  copy.copy(self.embedding_manager.placeholder2indices),
                        'prompt_emb_mask':      copy.copy(self.embedding_manager.prompt_emb_mask),
                        'prompt_pad_mask':      copy.copy(self.embedding_manager.prompt_pad_mask),
                        # Will be updated to True in p_losses() when in compositional iterations.
                        'capture_ca_activations':               False,
                        'use_attn_lora':                        False,
                        'use_ffn_lora':                         False,
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

    # output: roughly -1 ~ 1, but sometimes it could go beyond [-1, 1].
    @torch.no_grad()
    def decode_first_stage(self, z):
        # Revised from StableDiffusionPipeline::decode_latents().
        z = z.to(self.model.pipeline.dtype)
        z = 1 / self.vae.config.scaling_factor * z
        # NOTE: vae.decode() doesn't clip the output to [-1, 1].
        # image: roughly [-1, 1], but sometimes it could go beyond [-1, 1].
        image = self.vae.decode(z, return_dict=False)[0]
        return image
        
    # same as decode_first_stage() but without torch.no_grad() decorator
    # output: roughly -1 ~ 1, but sometimes it could go beyond [-1, 1].
    def decode_first_stage_with_grad(self, z):
        # Revised from StableDiffusionPipeline::decode_latents().
        # from diffusers import AutoencoderKL
        z = z.to(self.model.pipeline.dtype)
        z = 1 / self.vae.config.scaling_factor * z
        # image: [-1, 1]
        image = self.vae.decode(z, return_dict=False)[0]
        return image
        
    @torch.no_grad()
    def encode_first_stage(self, x, mask=None):
        # x: cuda float tensor of [BS, 3, 512, 512], normalized to [-1, 1]
        # mask: dict of {'fg_mask':  cuda torch.uint8 [BS, 1, 512, 512], 
        #                'aug_mask': cuda torch.uint8 [BS, 1, 512, 512]}.
        if not self.use_diffusers_vae_for_encoding:
            return self.first_stage_model.encode(x, mask)
        else:
            # Since diffusers VAE doesn't support mask, mask is ignored.
            latents = self.vae.encode(x.to(torch.float16)).latent_dist.mode()
            return latents        

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
        # Encode the input image/noise as 4-channel latent features.
        # first_stage_key="image"
        x_start = self.get_input(batch, self.first_stage_key)

        if self.iter_flags['do_normal_recon']:
            p_normal_recon_on_pure_noise = self.p_normal_recon_on_pure_noise
        else:
            p_normal_recon_on_pure_noise = 0

        if self.iter_flags['do_unet_distill']:
            p_unet_distill_on_pure_noise = self.p_unet_distill_on_pure_noise
        else:
            p_unet_distill_on_pure_noise = 0

        if self.iter_flags['do_comp_feat_distill']:
            attn_aug_names = ['no_attn_aug', 'shrink_cross_attn', 'mix_sc_mc_attn']
            attn_aug_idx = torch.multinomial(self.ps_comp_attn_aug, 1).item()
            # NOTE: the multinomial distribution guarantees that 
            # we don't do shrink_cross_attn and mix_sc_mc_attn at the same time.
            # Since ps_comp_attn_aug = [0, 0.5, 0.5], we always do 
            # either shrink_cross_attn or mix_sc_mc_attn.
            for i in range(len(attn_aug_names)):
                if i == attn_aug_idx:
                    self.iter_flags[attn_aug_names[i]] = True
                    print("Attn augmentation: ", attn_aug_names[i])
                else:
                    self.iter_flags[attn_aug_names[i]] = False
        else:
            self.iter_flags['no_attn_aug']          = True
            self.iter_flags['shrink_cross_attn']    = False
            self.iter_flags['mix_sc_mc_attn']       = False

        self.iter_flags['normal_recon_on_pure_noise']        = (torch.rand(1) < p_normal_recon_on_pure_noise).item()
        self.iter_flags['unet_distill_on_pure_noise'] = (torch.rand(1) < p_unet_distill_on_pure_noise).item()
        
        # NOTE: *_fp prompts are like "face portrait of ..." or "a portrait of ...". 
        # They highlight the face features compared to the normal prompts.
        if self.use_fp_trick and 'subj_single_prompt_fp' in batch:
            if self.iter_flags['do_comp_feat_distill']:
                # Use the fp trick all the time on compositional distillation iterations,
                # so that class comp prompts will generate clear face areas.
                p_use_fp_trick = 0.5
            # If compositional distillation is enabled, then in normal recon iterations,
            # we use the fp_trick most of the time, to better reconstructing single-face input images.
            # However, we still keep 20% of the do_normal_recon iterations to not use the fp_trick,
            # to encourage a bias towards larger facial areas in the output images.
            elif self.iter_flags['do_normal_recon'] and self.comp_distill_iter_gap > 0:
                p_use_fp_trick = 1
            else:
                # If not doing compositional distillation and only doing do_normal_recon, 
                # then use_fp_trick is disabled, so that the ID embeddings alone are expected 
                # to reconstruct the subject portraits.
                p_use_fp_trick = 0
        else:
            p_use_fp_trick = 0

        self.iter_flags['use_fp_trick'] = (torch.rand(1) < p_use_fp_trick).item()

        # NOTE: If normal_recon_on_pure_noise, there are no ground truth images, and we use 
        # subj_single_prompt to reconstruct the foreground face, and use cls_single_prompt
        # to reconstruct the background. Therefore, modifiers could be added to the prompts.
        # So in such cases, we use the mod prompts. 
        # Otherwise, there are ground truth images, and we can only use prompts without modifiers.
        if self.iter_flags['use_fp_trick']:
            if self.iter_flags['do_comp_feat_distill'] or \
              (self.iter_flags['do_normal_recon'] and self.iter_flags['normal_recon_on_pure_noise']):
                # If doing compositional distillation, then use the subj single prompts with styles, lighting, etc.
                SUBJ_SINGLE_PROMPT = 'subj_single_mod_prompt_fp'
                # SUBJ_COMP_PROMPT, CLS_SINGLE_PROMPT, CLS_COMP_PROMPT have to match 
                # SUBJ_SINGLE_PROMPT for prompt delta loss.
                CLS_SINGLE_PROMPT  = 'cls_single_mod_prompt_fp'
                SUBJ_COMP_PROMPT   = 'subj_comp_mod_prompt_fp'
                CLS_COMP_PROMPT    = 'cls_comp_mod_prompt_fp'
            else:
                # If normal recon or unet distillation, then use the subj single prompts without styles, lighting, etc.
                SUBJ_SINGLE_PROMPT = 'subj_single_prompt_fp'
                CLS_SINGLE_PROMPT  = 'cls_single_prompt_fp'
                # UNet distillation on subj_comp prompts (on joint face encoders) hasn't been implemented yet, 
                # maybe in the future. So how to set SUBJ_COMP_PROMPT doesn't matter yet.
                SUBJ_COMP_PROMPT   = 'subj_comp_prompt_fp'
                # CLS_COMP_PROMPT has to match SUBJ_COMP_PROMPT for prompt delta loss.
                CLS_COMP_PROMPT    = 'cls_comp_prompt_fp'

        else:
            if self.iter_flags['do_comp_feat_distill'] or \
              (self.iter_flags['do_normal_recon'] and self.iter_flags['normal_recon_on_pure_noise']):
                # If doing compositional distillation, then use the subj single prompts with styles, lighting, etc.
                SUBJ_SINGLE_PROMPT  = 'subj_single_mod_prompt_fp'
                SUBJ_COMP_PROMPT    = 'subj_comp_mod_prompt'
                # SUBJ_COMP_PROMPT, CLS_SINGLE_PROMPT, CLS_COMP_PROMPT have to match 
                # SUBJ_SINGLE_PROMPT for prompt delta loss.
                CLS_SINGLE_PROMPT   = 'cls_single_mod_prompt_fp'
                # Sometimes the cls comp instances don't have clear faces.
                # So use the fp trick on cls comp prompts at 75% of the time.
                if self.comp_iters_count % 4 != 0:
                    CLS_COMP_PROMPT = 'cls_comp_mod_prompt_fp'
                else:
                    CLS_COMP_PROMPT = 'cls_comp_mod_prompt'
            else:
                # If normal recon or unet distillation, then use the subj single prompts without styles, lighting, etc.
                # cls prompts are only used for delta loss, so they don't need to be fp prompts.
                SUBJ_SINGLE_PROMPT  = 'subj_single_prompt'
                CLS_SINGLE_PROMPT   = 'cls_single_prompt'
                # UNet distillation on subj_comp prompts (on joint face encoders) hasn't been implemented yet,
                # maybe in the future. So how to set SUBJ_COMP_PROMPT doesn't matter yet.
                SUBJ_COMP_PROMPT    = 'subj_comp_prompt'
                # SUBJ_COMP_PROMPT, CLS_SINGLE_PROMPT, CLS_COMP_PROMPT have to match SUBJ_SINGLE_PROMPT for prompt delta loss.
                CLS_COMP_PROMPT     = 'cls_comp_prompt'

        subj_single_prompts = batch[SUBJ_SINGLE_PROMPT]
        cls_single_prompts  = batch[CLS_SINGLE_PROMPT]
        subj_comp_prompts   = batch[SUBJ_COMP_PROMPT]
        cls_comp_prompts    = batch[CLS_COMP_PROMPT]

        # Don't use fp trick and the 'clear face' suffix at the same time.
        if self.iter_flags['do_comp_feat_distill']:
            p_clear_face = 0.8
            p_front_view = 0.8
        else:
            p_clear_face = 0
            p_front_view = 0
            
        if torch.rand(1) < p_clear_face:
            # Add 'clear face' to the 4 types of prompts. Its effect is weaker than the fp trick.
            cls_single_prompts, cls_comp_prompts, subj_single_prompts, subj_comp_prompts = \
                [ [ p + ', clear face' for p in prompts ] for prompts in \
                    (cls_single_prompts, cls_comp_prompts, subj_single_prompts, subj_comp_prompts) ]

        if torch.rand(1) < p_front_view:
            # Add 'front view' to the 4 types of prompts.
            # Chance is 'front view' may have been added to the prompts already. 
            # But it doesn't matter to repeat it.
            cls_single_prompts, cls_comp_prompts, subj_single_prompts, subj_comp_prompts = \
                [ [ p + ', front view' for p in prompts ] for prompts in \
                    (cls_single_prompts, cls_comp_prompts, subj_single_prompts, subj_comp_prompts) ]
                    
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
            breakpoint()

        print(f"Rank {self.trainer.global_rank}: {batch['subject_name']}")

        BS = len(batch['subject_name'])
        # If do_comp_feat_distill, we repeat the instances in the batch, 
        # so that all instances are the same.
        if self.iter_flags['do_comp_feat_distill']:
            self.iter_flags['same_subject_in_batch'] = True
            # Change the batch to have the (1 subject image) * BS strcture.
            # "delta_prompts" don't change, as different subjects share the same placeholder "z".
            # After image_unnorm is repeated, the extracted zs_clip_fgbg_features and face_id_embs, extracted from image_unnorm,
            # will be repeated automatically. Therefore, we don't need to manually repeat them later.
            batch['subject_name'], batch["image_path"], batch["image_unnorm"], x_start, img_mask, fg_mask = \
                select_and_repeat_instances(slice(0, 1), BS, batch['subject_name'], batch["image_path"], batch["image_unnorm"], 
                                            x_start, img_mask, fg_mask)
        else:
            self.iter_flags['same_subject_in_batch'] = False
            if self.iter_flags['do_normal_recon']:
                # Reduce the batch size to have only 2 instances.
                batch['subject_name'], batch["image_path"], batch["image_unnorm"], x_start, img_mask, fg_mask = \
                    select_and_repeat_instances(slice(0, 2), 1, batch['subject_name'], batch["image_path"], batch["image_unnorm"], 
                                                x_start, img_mask, fg_mask)

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
        # gen_rand_id_for_id2img implies (not do_comp_feat_distill).
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
            # If do_comp_feat_distill, then we have repeated the instances in the batch, 
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
                # If the iteration is do_comp_feat_distill/do_normal_recon, convert it to a do_unet_distill iteration.
                # We don't have to update self.unet_distill_iters_count, self.normal_recon_iters_count, etc., 
                # as they don't have to be so precise. Moreover, updating them will break the synchronization 
                # between different training instances.
                self.iter_flags['do_normal_recon']       = False
                self.iter_flags['do_comp_feat_distill']  = False
                self.iter_flags['do_unet_distill']       = True

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

        # If do_comp_feat_distill, then we don't add noise to the zero-shot ID embeddings, 
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
                # "delta_prompts" don't change, as different subjects share the same placeholder "z".
                # clip_bg_features is used by adaface encoder, so we repeat zs_clip_fgbg_features accordingly.
                # We don't repeat id2img_neg_prompt_embs, as it's constant and identical for different instances.
                x_start, batch_images_unnorm, img_mask, fg_mask, \
                self.batch_subject_names, id2img_prompt_embs, zs_clip_fgbg_features = \
                    select_and_repeat_instances(slice(0, 1), BS, 
                                                x_start, batch_images_unnorm, img_mask, fg_mask, 
                                                self.batch_subject_names, id2img_prompt_embs, zs_clip_fgbg_features)
                
            # ** Perturb the zero-shot ID image prompt embeddings with probability 0.2. **
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
            # Therefore, for jointIDs, we reduce perturb_face_id_embs_std_range to [0.3, 0.6].
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
            num_unet_denoising_steps = self.unet_distill_iters_count % 3 + 2
            self.iter_flags['num_unet_denoising_steps'] = num_unet_denoising_steps

            if (torch.rand(1) < self.p_unet_distill_uses_comp_prompt):
                # Sometimes we use the subject compositional instances as the distillation target on a UNet ensemble teacher.
                # If unet_teacher_types == ['arc2face'], then p_unet_distill_uses_comp_prompt == 0, i.e., we
                # never use the compositional instances as the distillation target of arc2face.
                # If unet_teacher_types is ['consistentID', 'arc2face'], then p_unet_distill_uses_comp_prompt == 0.1.
                # If unet_teacher_types == ['consistentID'], then p_unet_distill_uses_comp_prompt == 0.2.
                # 'unet_distill_uses_comp_prompt' is only applicable to composable teachers, such as consistentID.
                # They are exclusive to each other, and are never enabled at the same time.
                self.iter_flags['unet_distill_uses_comp_prompt'] = True

            if num_unet_denoising_steps > 1:
                # If denoising steps are a few, then reduce batch size to avoid OOM.
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
                self.batch_subject_names, zs_clip_fgbg_features, \
                id2img_prompt_embs, id2img_neg_prompt_embs, \
                subj_single_prompts, subj_comp_prompts, \
                cls_single_prompts, cls_comp_prompts \
                    = select_and_repeat_instances(slice(0, HALF_BS), 1, 
                                                  x_start, batch_images_unnorm, img_mask, fg_mask, 
                                                  self.batch_subject_names, zs_clip_fgbg_features,
                                                  id2img_prompt_embs, id2img_neg_prompt_embs,
                                                  subj_single_prompts, subj_comp_prompts,
                                                  cls_single_prompts, cls_comp_prompts)
                    
                # Update delta_prompts to have the first HALF_BS prompts.
                delta_prompts = (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

        # aug_mask is renamed as img_mask.
        self.iter_flags['img_mask']                 = img_mask
        self.iter_flags['fg_mask']                  = fg_mask
        self.iter_flags['delta_prompts']            = delta_prompts
        self.iter_flags['compos_partial_prompt']    = batch['compos_partial_prompt']
        self.iter_flags['prompt_modifier']          = batch['prompt_modifier']
        self.iter_flags['image_unnorm']             = batch_images_unnorm

        self.iter_flags['id2img_prompt_embs']       = id2img_prompt_embs
        self.iter_flags['id2img_neg_prompt_embs']   = id2img_neg_prompt_embs
        if self.embedding_manager.id2ada_prompt_encoder.name == 'jointIDs':
            self.iter_flags['encoders_num_id_vecs'] = self.embedding_manager.id2ada_prompt_encoder.encoders_num_id_vecs
        else:
            self.iter_flags['encoders_num_id_vecs'] = None

        if zs_clip_fgbg_features is not None:
            self.iter_flags['clip_bg_features']  = zs_clip_fgbg_features.chunk(2, dim=1)[1]
        else:
            self.iter_flags['clip_bg_features']  = None

        # In get_text_conditioning(), text_conditioning_iter_type will be set again.
        # Setting it here is necessary, as set_curr_batch_subject_names() maps curr_batch_subj_names to cls_delta_strings,
        # whose behavior depends on the correct text_conditioning_iter_type.
        if self.iter_flags['do_comp_feat_distill']:
            text_conditioning_iter_type = 'compos_distill_iter'
        elif self.iter_flags['do_unet_distill']:
            text_conditioning_iter_type = 'unet_distill_iter'
        else:
            text_conditioning_iter_type = 'recon_iter'
        self.iter_flags['text_conditioning_iter_type'] = text_conditioning_iter_type

        self.embedding_manager.set_curr_batch_subject_names(self.batch_subject_names)

        loss = self(x_start)
        # Release temporary variables stored in iter_flags.
        self.init_iteration_flags()
        return loss

    # LatentDiffusion.forward() is only called during training, by shared_step().
    #LINK #shared_step
    def forward(self, x_start):
        ORIG_BS  = len(x_start)

        # Use >=, i.e., assign decay in all iterations after the first 100.
        # This is in case there are skips of iterations of global_step 
        # (shouldn't happen but just in case).

        # iter_flags['delta_prompts'] is a tuple of 4 lists. No need to split them.
        delta_prompts           = self.iter_flags['delta_prompts']
        compos_partial_prompt   = self.iter_flags['compos_partial_prompt']
        prompt_modifier         = self.iter_flags['prompt_modifier']

        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = delta_prompts

        if self.iter_flags['do_comp_feat_distill']:                        
            # For simplicity, BLOCK_SIZE is fixed at 1. So if ORIG_BS == 2, then BLOCK_SIZE = 1.
            BLOCK_SIZE = 1
        else:
            # Otherwise, do_prompt_emb_delta_reg.
            # Do not halve the batch. BLOCK_SIZE = ORIG_BS = 12.
            # 12 prompts will be fed into get_text_conditioning().
            BLOCK_SIZE = ORIG_BS

        # Only keep the first BLOCK_SIZE of batched prompts.
        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts, \
        compos_partial_prompt, prompt_modifier = \
            select_and_repeat_instances(slice(0, BLOCK_SIZE), 1, 
            subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts,
            compos_partial_prompt, prompt_modifier)

        # Repeat the compositional prompts, to further highlight the compositional features.
        # NOTE: the prompt_modifier is repeated at most once, no matter subj_rep_prompts_count.
        # Since subj_comp_prompts already contains 1 copy of the modifier,
        # in total subj_comp_rep_prompts contains 2 copies of the modifier, and maybe 3 copies of compos_partial_prompt.
        # This is to avoid the subj comp instance receives too much style guidance from the subj_comp_rep instances,
        # and becomes overly stylized.

        # Add prompt_modifier only once.
        # Repeat compos_partial_prompt subj_rep_prompts_count = 2 times.
        subj_comp_rep_prompts = [ subj_comp_prompts[i] + \
                                  ", ".join([ prompt_modifier[i] + ", " + compos_partial_prompt[i] ] * self.subj_rep_prompts_count) \
                                    for i in range(BLOCK_SIZE) ]
                    
        # We still compute the prompt embeddings of the first 4 types of prompts, 
        # to compute prompt delta loss. 
        # But now there are 16 prompts (4 * ORIG_BS = 16), as the batch is not halved.
        delta_prompts = subj_single_prompts + subj_comp_prompts \
                        + cls_single_prompts + cls_comp_prompts
        # prompt_emb: the prompt embeddings for prompt delta loss [4, 77, 768].
        # delta_prompts: the concatenation of
        # (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts).
        # extra_info: a dict that contains extra info.
        prompt_emb, _, extra_info = \
            self.get_text_conditioning(delta_prompts, 
                                       self.iter_flags['id2img_prompt_embs'],
                                       self.iter_flags['clip_bg_features'],
                                       randomize_clip_weights=True,
                                       text_conditioning_iter_type=self.iter_flags['text_conditioning_iter_type'],
                                       real_batch_size=ORIG_BS)

        subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb = \
            prompt_emb.chunk(4)

        subj_comp_rep_emb, _, extra_info_sc_rep = \
            self.get_text_conditioning(subj_comp_rep_prompts,
                                       self.iter_flags['id2img_prompt_embs'],
                                       self.iter_flags['clip_bg_features'],
                                       randomize_clip_weights=False,
                                       text_conditioning_iter_type=self.iter_flags['text_conditioning_iter_type'],
                                       real_batch_size=ORIG_BS)

        # Rename extra_info['prompt_emb_mask'] to extra_info['prompt_emb_mask_4b_orig'],
        #        extra_info['prompt_pad_mask'] to extra_info['prompt_pad_mask_4b_orig'].
        extra_info['prompt_emb_mask_4b_orig']   = extra_info.pop('prompt_emb_mask')
        extra_info['prompt_pad_mask_4b_orig']   = extra_info.pop('prompt_pad_mask')

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

        # uncond_prompt_embeddings: [1, 77, 768].
        uncond_prompt_embeddings = self.uncond_context[0]
        # NOTE: if there are multiple subject tokens (e.g., 28 tokens), then only the first subject token
        # is aligned with the "class-token , , , ...". 
        # The rest 27 tokens are aligned with the embeddings of ", ".
        # So we patch this misalignment by distributing the class embeddings of one token 
        # evenly to the 28 tokens.
        # After patching, often the class "person" is not fully expressed in the intermediate latent images.
        # Therefore we use two tricks to enhance its expression:
        # emb_cfg = 2:         compel style CFG for embeddings, i.e., cond_emb * 2 - uncond_emb.
        # emb_extra_boost = 2: increasing the embedding magnitude after distribution by 2.
        cls_single_emb_dist = \
            distribute_embedding_to_M_tokens_by_dict(cls_single_emb, uncond_prompt_embeddings, 
                                                     placeholder2indices_1b, divide_scheme='sqrt_M', 
                                                     emb_cfg=2, emb_extra_boost=2)
        cls_comp_emb_dist   = \
            distribute_embedding_to_M_tokens_by_dict(cls_comp_emb,   uncond_prompt_embeddings, 
                                                     placeholder2indices_1b, divide_scheme='sqrt_M', 
                                                     emb_cfg=2, emb_extra_boost=2)
        
        extra_info['placeholder2indices_1b'] = placeholder2indices_1b
        extra_info['placeholder2indices_2b'] = placeholder2indices_2b

        # Replace encoded subject embeddings with image embeddings.
        if self.iter_flags['do_comp_feat_distill'] and self.ablate_img_embs:
            subj_indices_1b = placeholder2indices_1b['z'][1]
            # Only use the first instance in the batch to generate the adaface_subj_embs,
            # as the whole batch is of the same subject.           
            # id2img_prompt_embs: [3, 20, 768] (a batch of three embeddings of the same values).
            # subj_single_emb: [1, 97, 768] -> [1, 20, 768]
            subj_single_emb = subj_single_emb.clone()
            subj_comp_emb   = subj_comp_emb.clone()
            subj_single_emb[:, subj_indices_1b] = self.iter_flags['id2img_prompt_embs'][:1]
            subj_comp_emb[:, subj_indices_1b]   = self.iter_flags['id2img_prompt_embs'][:1]

        # prompt_emb_4b_rep_nonmix: in which cls_comp_emb is not mixed with subj_comp_emb.
        prompt_emb_4b_rep_nonmix = torch.cat([subj_single_emb,   subj_comp_emb, 
                                              subj_comp_rep_emb, cls_comp_emb], dim=0)
        # cls_single_emb and cls_comp_emb have been patched above. 
        # Then combine them back into prompt_emb_4b_orig_dist.
        # orig means it doesn't contain subj_comp_rep_emb.
        # prompt_emb_4b_orig_dist: [4, 77, 768].
        prompt_emb_4b_orig_dist  = torch.cat([subj_single_emb,     subj_comp_emb,
                                              cls_single_emb_dist, cls_comp_emb_dist], dim=0)
        extra_info['prompt_emb_4b_rep_nonmix']  = prompt_emb_4b_rep_nonmix
        extra_info['prompt_emb_4b_orig_dist']   = prompt_emb_4b_orig_dist

        if self.iter_flags['do_comp_feat_distill']:
            # prompt_in: subj_single_prompts + subj_comp_prompts + subj_comp_rep_prompts + cls_comp_prompts
            # The cls_single_prompts/cls_comp_prompts within prompt_in will only be used to 
            # generate ordinary prompt embeddings, i.e., 
            # it doesn't contain subject tokens.
            # The 4 blocks of instances are (subj_single, subj_comp, subj_comp_rep, cls_comp).
            # *** subj_comp_prompts repeats the compositional part once,
            # *** and subj_comp_rep_prompts repeats the compositional part twice.
            prompt_in = subj_single_prompts + subj_comp_prompts + subj_comp_rep_prompts + cls_comp_prompts

            # Mix 0.2 of the subject comp embeddings with 0.8 of the cls comp embeddings as cls_comp_emb2.
            cls_comp_emb2 = subj_comp_emb * (1 - self.cls_subj_mix_ratio) + cls_comp_emb * self.cls_subj_mix_ratio

            prompt_emb = torch.cat([subj_single_emb, subj_comp_emb, subj_comp_rep_emb, cls_comp_emb2], dim=0)

            # Update the cls_single (mc) embedding mask and padding mask to be those of sc_rep.
            # The mask before update is prompt_emb_mask_4b_orig, which matches prompt_emb_4b_orig_dist.
            # The mask after  update is prompt_emb_mask_4b,      which matches prompt_emb_4b_rep_nonmix and prompt_emb.
            prompt_emb_mask_4b  = extra_info['prompt_emb_mask_4b_orig'].clone()
            prompt_pad_mask_4b  = extra_info['prompt_pad_mask_4b_orig'].clone()
            prompt_emb_mask_4b[BLOCK_SIZE*2:BLOCK_SIZE*3] = extra_info_sc_rep['prompt_emb_mask']
            prompt_pad_mask_4b[BLOCK_SIZE*2:BLOCK_SIZE*3] = extra_info_sc_rep['prompt_pad_mask']
            extra_info['prompt_emb_mask_4b'] = prompt_emb_mask_4b
            extra_info['prompt_pad_mask_4b'] = prompt_pad_mask_4b

            # The prompts are either (subj single, subj comp, cls single, cls comp).
            # So the first 2 sub-blocks always contain the subject tokens, and we use *_2b.    
            extra_info['placeholder2indices'] = extra_info['placeholder2indices_2b']
        else:
            # do_normal_recon or do_unet_distill.
            if self.iter_flags['unet_distill_uses_comp_prompt']:
                prompt_in   = subj_comp_prompts
                prompt_emb  = subj_comp_emb
            else:
                prompt_in   = subj_single_prompts
                prompt_emb  = subj_single_emb

            # The blocks as input to get_text_conditioning() are not halved. 
            # So BLOCK_SIZE = ORIG_BS = 2. Therefore, for the two instances, we use *_1b.
            extra_info['placeholder2indices'] = extra_info['placeholder2indices_1b']
                            
        # extra_info['cls_single_emb'] and extra_info['cls_comp_emb'] are used in unet distillation.
        # cls_comp_emb is only mixed in do_comp_feat_distill iterations, so it's fine.
        extra_info['cls_single_prompts']    = cls_single_prompts
        extra_info['cls_single_emb']        = cls_single_emb
        extra_info['cls_comp_prompts']      = cls_comp_prompts
        extra_info['cls_comp_emb']          = cls_comp_emb                          
        extra_info['compos_partial_prompt'] = compos_partial_prompt

        # prompt_emb: [4, 77, 768]                    
        cond_context = (prompt_emb, prompt_in, extra_info)

        # self.model (UNetModel) is called in p_losses().
        #LINK #p_losses
        prompt_emb, prompt_in, extra_info = cond_context
        return self.p_losses(x_start, prompt_emb, prompt_in, extra_info)

    # apply_model() is called both during training and inference.
    # apply_model() is called in sliced_apply_model() and guided_denoise().
    def apply_model(self, x_noisy, t, cond_context, use_attn_lora=False, 
                    use_ffn_lora=False, ffn_lora_adapter_name=None):
        # self.model: DiffusersUNetWrapper -> 
        # self.model.diffusion_model: diffusers UNet2DConditionModel.
        # cond_context[2]: extra_info.
        cond_context[2]['use_attn_lora'] = use_attn_lora
        cond_context[2]['use_ffn_lora']  = use_ffn_lora
        cond_context[2]['ffn_lora_adapter_name'] = ffn_lora_adapter_name
        x_recon = self.model(x_noisy, t, cond_context)
        return x_recon

    # sliced_apply_model() is only called within guided_denoise().
    def sliced_apply_model(self, x_noisy, t, cond_context, slice_indices, 
                           enable_grad, use_attn_lora=False, 
                           use_ffn_lora=False, ffn_lora_adapter_name=None):
        x_noisy_ = x_noisy[slice_indices]
        t_       = t[slice_indices]
        prompt_emb, prompt_in, extra_info = cond_context
        prompt_emb_ = prompt_emb[slice_indices]
        prompt_in_  = [ prompt_in[i] for i in slice_indices ]
        cond_context_ = (prompt_emb_, prompt_in_, extra_info)
        with torch.set_grad_enabled(enable_grad):
            # use_attn_lora, use_ffn_lora, and ffn_lora_adapter_name are set in apply_model().
            noise_pred = self.apply_model(x_noisy_, t_, cond_context_, 
                                          use_attn_lora=use_attn_lora, 
                                          use_ffn_lora=use_ffn_lora, 
                                          ffn_lora_adapter_name=ffn_lora_adapter_name)
        return noise_pred

    # do_pixel_recon: return denoised images for CLIP evaluation. 
    # if do_pixel_recon and cfg_scale > 1, apply classifier-free guidance. 
    # This is not used for the iter_type 'do_normal_recon'.
    # batch_part_has_grad: 'all', 'none', 'subject-compos'.
    # shrink_cross_attn: only enabled on compositional distillation iterations, 
    # not on normal recon / unet distillation iterations.
    # If use_attn_lora is set to True and self.unet_uses_attn_lora is False, 
    # it will be overridden in the unet.
    def guided_denoise(self, x_start, noise, t, cond_context,
                       uncond_emb=None, img_mask=None, 
                       shrink_cross_attn=False, mix_sc_mc_attn=False, 
                       batch_part_has_grad='all', do_pixel_recon=False, cfg_scale=-1, 
                       capture_ca_activations=False, 
                       res_hidden_states_gradscale=1,
                       use_attn_lora=False, use_ffn_lora=False, ffn_lora_adapter_name=None):
        
        x_noisy = self.q_sample(x_start, t, noise)
        ca_layers_activations = None

        extra_info = cond_context[2]
        extra_info['capture_ca_activations']              = capture_ca_activations
        extra_info['res_hidden_states_gradscale']         = res_hidden_states_gradscale
        extra_info['img_mask']                            = img_mask
        extra_info['shrink_cross_attn']                   = shrink_cross_attn

        # noise_pred is the predicted noise.
        # if not batch_part_has_grad, we save RAM by not storing the computation graph.
        # if batch_part_has_grad, we don't have to take care of embedding_manager.force_grad.
        # Subject embeddings will naturally have gradients.
        if batch_part_has_grad == 'none':
            with torch.no_grad():
                noise_pred = self.apply_model(x_noisy, t, cond_context, use_attn_lora=use_attn_lora,
                                              use_ffn_lora=use_ffn_lora, ffn_lora_adapter_name=ffn_lora_adapter_name)

            if capture_ca_activations:
                ca_layers_activations = extra_info['ca_layers_activations']

        elif batch_part_has_grad == 'all':
            noise_pred = self.apply_model(x_noisy, t, cond_context, use_attn_lora=use_attn_lora,
                                          use_ffn_lora=use_ffn_lora, ffn_lora_adapter_name=ffn_lora_adapter_name)

            if capture_ca_activations:
                ca_layers_activations = extra_info['ca_layers_activations']

        elif batch_part_has_grad == 'subject-compos':    
            ##### SS instance generation #####
            extra_info_ss = copy.copy(extra_info)
            extra_info_ss['shrink_cross_attn']  = shrink_cross_attn
            #extra_info_ss['debug']  = False
            cond_context2 = (cond_context[0], cond_context[1], extra_info_ss)
            noise_pred_ss = self.sliced_apply_model(x_noisy, t, cond_context2, slice_indices=[0], 
                                                    enable_grad=False, use_attn_lora=use_attn_lora,
                                                    use_ffn_lora=use_ffn_lora, 
                                                    ffn_lora_adapter_name=ffn_lora_adapter_name)
            ss_ca_layers_activations = extra_info_ss['ca_layers_activations']

            ##### SR (sc_rep) instance generation #####
            extra_info_sr = copy.copy(extra_info)
            # The ms instance is actually sc_comp_rep.
            # So we use the same shrink_cross_attn as the sc instance.
            extra_info_sr['shrink_cross_attn']      = shrink_cross_attn
            extra_info_sr['mix_attn_mats_in_batch'] = False

            cond_context2 = (cond_context[0], cond_context[1], extra_info_sr)
            noise_pred_sr = self.sliced_apply_model(x_noisy, t, cond_context2, slice_indices=[2],
                                                    enable_grad=False, use_attn_lora=use_attn_lora, 
                                                    use_ffn_lora=use_ffn_lora, 
                                                    ffn_lora_adapter_name=ffn_lora_adapter_name)
            sr_ca_layers_activations = extra_info_sr['ca_layers_activations']

            if mix_sc_mc_attn:
                ##### SC and MC instances generation #####
                extra_info_sm = copy.copy(extra_info)
                extra_info_sm['shrink_cross_attn']      = False
                extra_info_sm['mix_attn_mats_in_batch'] = True
                cond_context2 = (cond_context[0], cond_context[1], extra_info_sm)
                # Do not use attn and ffn LoRAs on joint sc-mc instances.
                noise_pred_sm = self.sliced_apply_model(x_noisy, t, cond_context2, slice_indices=[1, 3],
                                                        enable_grad=False, use_attn_lora=False,
                                                        use_ffn_lora=False, ffn_lora_adapter_name=ffn_lora_adapter_name)
                noise_pred_sc, noise_pred_mc = noise_pred_sm.chunk(2, dim=0)
                sc_ca_layers_activations, mc_ca_layers_activations = \
                    split_dict(extra_info_sm['ca_layers_activations'], 2)

            else:
                ##### SC instance generation #####
                extra_info_sc = copy.copy(extra_info)
                extra_info_sc['shrink_cross_attn']      = shrink_cross_attn
                extra_info_sc['mix_attn_mats_in_batch'] = False
                cond_context2 = (cond_context[0], cond_context[1], extra_info_sc)
                noise_pred_sc = self.sliced_apply_model(x_noisy, t, cond_context2, slice_indices=[1],
                                                        enable_grad=True,  use_attn_lora=use_attn_lora,
                                                        use_ffn_lora=use_ffn_lora, 
                                                        ffn_lora_adapter_name=ffn_lora_adapter_name)
                sc_ca_layers_activations = extra_info_sc['ca_layers_activations']

                ##### MC instance generation #####
                extra_info_mc = copy.copy(extra_info)
                extra_info_mc['shrink_cross_attn']   = False
                cond_context2 = (cond_context[0], cond_context[1], extra_info_mc)
                # Never use attn and ffn LoRAs on mc instances.
                # FFN LoRAs on mc instances may lead to trivial solutions of suppressing 
                # background/compositional components and focusing on the subject only.
                noise_pred_mc = self.sliced_apply_model(x_noisy, t, cond_context2, slice_indices=[3],
                                                        enable_grad=False, use_attn_lora=False,
                                                        use_ffn_lora=False, 
                                                        ffn_lora_adapter_name=ffn_lora_adapter_name)
                mc_ca_layers_activations = extra_info_mc['ca_layers_activations']

            noise_pred = torch.cat([noise_pred_ss, noise_pred_sc, noise_pred_sr, noise_pred_mc], dim=0)
            extra_info = cond_context[2]
            if capture_ca_activations:
                # Collate three captured activation dicts into extra_info.
                ca_layers_activations = collate_dicts([ss_ca_layers_activations,
                                                       sc_ca_layers_activations,
                                                       sr_ca_layers_activations,
                                                       mc_ca_layers_activations])
        else:
            breakpoint()

        # Get model output of both conditioned and uncond prompts.
        # Unconditional prompts and reconstructed images are never involved in optimization.
        if cfg_scale > 1:
            if uncond_emb is None:
                # Use self.uncond_context as the unconditional context.
                # uncond_context is a tuple of (uncond_emb, uncond_prompt_in, extra_info).
                # By default, 'capture_ca_activations' = False in a generated text context, 
                # including uncond_context. So we don't need to set it in self.uncond_context explicitly.                
                uncond_emb  = self.uncond_context[0].repeat(x_noisy.shape[0], 1, 1)

            uncond_prompt_in = self.uncond_context[1] * x_noisy.shape[0]
            uncond_extra_info = copy.copy(self.uncond_context[2])
            uncond_context = (uncond_emb, uncond_prompt_in, uncond_extra_info)

            # We never needs gradients on unconditional generation.
            with torch.no_grad():
                # noise_pred_uncond: [BS, 4, 64, 64]
                noise_pred_uncond = self.apply_model(x_noisy, t, uncond_context, use_attn_lora=False, 
                                                     use_ffn_lora=use_ffn_lora, ffn_lora_adapter_name=ffn_lora_adapter_name)
            # If do clip filtering, CFG makes the contents in the 
            # generated images more pronounced => smaller CLIP loss.
            noise_pred = noise_pred * cfg_scale - noise_pred_uncond * (cfg_scale - 1)
        else:
            noise_pred = noise_pred

        if do_pixel_recon:
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=noise_pred)
        else:
            x_recon = None
        
        return noise_pred, x_recon, ca_layers_activations

    # If not do_adv_attack, then mon_loss_dict and session_prefix are not used and could be set to None.
    def recon_multistep_denoise(self, mon_loss_dict, session_prefix,
                                x_start0, noise, t, subj_context, cls_context,
                                uncond_emb, img_mask, fg_mask, cfg_scale, 
                                num_denoising_steps, num_priming_steps,
                                normal_recon_on_pure_noise, enable_unet_attn_lora, enable_unet_ffn_lora, 
                                ffn_lora_adapter_name, do_adv_attack, DO_ADV_BS):

        assert num_denoising_steps <= 10
        
        # Initially, x_starts only contains the original x_start0.
        x_starts    = [ x_start0 ]
        noises      = [ noise ]
        ts          = [ t ]
        noise_preds = []
        x_recons    = []
        ca_layers_activations_list = []

        if cls_context is not None:
            noise_preds_cls = []
            x_recons_cls    = []
        else:
            noise_preds_cls = None
            x_recons_cls    = None

        for i in range(num_denoising_steps):
            x_start = x_starts[i]
            t       = ts[i]
            noise   = noises[i]
            on_priming_steps = (i < num_priming_steps)
            # Half of the priming steps are done using cls_context, 
            # and half using subj_context.
            if on_priming_steps and (cls_context is not None) and (i % 2 == 0):
                context = cls_context
            else:
                context = subj_context

            # If normal_recon_on_pure_noise, enable_unet_attn_lora, enable_unet_ffn_lora are False
            # Otherwise, enable_unet_ffn_lora = self.recon_uses_ffn_lora = True.
            use_ffn_lora = enable_unet_ffn_lora and (not on_priming_steps)

            # Only enable gradients after num_priming_steps.
            noise_pred, x_recon, ca_layers_activations = \
                self.guided_denoise(x_start, noise, t, context,
                                    uncond_emb, img_mask, 
                                    shrink_cross_attn=False,
                                    batch_part_has_grad='all' if (not on_priming_steps) else 'none',
                                    do_pixel_recon=True, cfg_scale=cfg_scale, 
                                    capture_ca_activations=(not on_priming_steps),
                                    # When doing normal recon, res_hidden_states_gradscale is always 0, 
                                    # i.e., gradients don't flow back through UNet skip connections.
                                    res_hidden_states_gradscale=0,
                                    # enable_unet_attn_lora: randomly set to True 50% of the time.
                                    use_attn_lora=enable_unet_attn_lora,
                                    use_ffn_lora=use_ffn_lora, 
                                    ffn_lora_adapter_name=ffn_lora_adapter_name)

            noise_preds.append(noise_pred)
            ca_layers_activations_list.append(ca_layers_activations)
            x_recons.append(x_recon)

            # In our current implementation, num_priming_steps > 0 only if normal_recon_on_pure_noise.
            # So no need to check (i < num_priming_steps).
            if normal_recon_on_pure_noise or on_priming_steps:
                # The predicted x0 is used as the x_start in the next denoising step.
                ## NOTE: we detach the predicted x0, so that the gradients don't flow back to the previous denoising steps.
                x_starts.append(x_recon.detach())
            else:
                # The original x_start0 is used as the x_start in the next denoising step.
                x_starts.append(x_start0)

            # NOTE: cls_context reuses extra_info from subj_context.
            # But since we don't capture ca activations in cls_context,
            # it won't affect the ca activations gotten captured in subj_context.
            if cls_context is not None:
                noise_pred_cls, x_recon_cls, _ = \
                    self.guided_denoise(x_start, noise, t, cls_context,
                                        uncond_emb, img_mask, 
                                        shrink_cross_attn=False,
                                        batch_part_has_grad='none',
                                        do_pixel_recon=True, cfg_scale=cfg_scale, 
                                        capture_ca_activations=False,
                                        # When doing normal recon, res_hidden_states_gradscale is always 0, 
                                        # i.e., gradients don't flow back through UNet skip connections.
                                        res_hidden_states_gradscale=0,
                                        # enable_unet_attn_lora: randomly set to True 50% of the time.
                                        use_attn_lora=enable_unet_attn_lora,
                                        # enable_unet_ffn_lora = self.recon_uses_ffn_lora = True.
                                        use_ffn_lora=enable_unet_ffn_lora, 
                                        ffn_lora_adapter_name=ffn_lora_adapter_name)
                
                noise_preds_cls.append(noise_pred_cls)
                x_recons_cls.append(x_recon_cls)

            # Sample an earlier timestep for the next denoising step.
            if i < num_denoising_steps - 1:
                t0 = t
                # NOTE: rand_like() samples from U(0, 1), not like randn_like().
                rand_ts = torch.rand_like(t0.float())
                # Make sure at the middle step (i < num_denoising_steps - 1), the timestep 
                # is between 50% and 70% of the current timestep. So if num_denoising_steps = 5,
                # we take timesteps within [0.5^0.66, 0.7^0.66] = [0.63, 0.79] of the current timestep.
                # If num_denoising_steps = 4, we take timesteps within [0.5^0.72, 0.7^0.72] = [0.61, 0.77] 
                # of the current timestep.
                # In general, the larger num_denoising_steps, the ratio between et and t0 is closer to 1.
                t_lb = t0 * np.power(0.5, np.power(num_denoising_steps - 1, -0.3))
                t_ub = t0 * np.power(0.7, np.power(num_denoising_steps - 1, -0.3))
                # et: earlier timestep, ts[i+1] < ts[i].
                # et is randomly sampled between [t_lb, t_ub].
                et = (t_ub - t_lb) * rand_ts + t_lb
                et = et.long()
                earlier_timesteps = et
                ts.append(earlier_timesteps)

                noise = torch.randn_like(x_start)
                # Do adversarial "attack" (edit) on x_start, so that it's harder to reconstruct.
                # This way, we force the adaface encoders to better reconstruct the subject.
                # recon_with_adv_attack_iter_gap = 3, i.e., adversarial attack on the input images 
                # every 3 non-comp recon iterations.
                # Doing adversarial attack on the input images seems to introduce high-frequency noise 
                # to the whole image (not just the face area), so we only do it after the first denoise step.
                if do_adv_attack:
                    adv_grad = self.calc_arcface_adv_grad(x_start[:DO_ADV_BS])
                    self.adaface_adv_iters_count += 1
                    if adv_grad is not None:
                        # adv_grad_max: 1e-3
                        adv_grad_max = adv_grad.abs().max().detach().item()
                        mon_loss_dict.update({f'{session_prefix}/adv_grad_max': adv_grad_max})
                        # adv_grad_mean is always 4~5e-6.
                        # mon_loss_dict.update({f'{session_prefix}/adv_grad_mean': adv_grad.abs().mean().detach().item()})
                        faceloss_fg_mask = fg_mask[:DO_ADV_BS].repeat(1, 4, 1, 1)
                        # adv_grad_fg_mean: 8~9e-6.
                        adv_grad_fg_mean = adv_grad[faceloss_fg_mask.bool()].abs().mean().detach().item()
                        mon_loss_dict.update({f'{session_prefix}/adv_grad_fg_mean': adv_grad_fg_mean})
                        # adv_grad_mag: ~1e-4.
                        adv_grad_mag = np.sqrt(adv_grad_max * adv_grad_fg_mean)
                        # recon_adv_mod_mag_range: [0.001, 0.003].
                        recon_adv_mod_mag = torch_uniform(*self.recon_adv_mod_mag_range).item()
                        # recon_adv_mod_mag: 0.001~0.003. adv_grad_scale: 4~10.
                        adv_grad_scale = recon_adv_mod_mag / (adv_grad_mag + 1e-6)
                        mon_loss_dict.update({f'{session_prefix}/adv_grad_scale': adv_grad_scale})
                        # Cap the adv_grad_scale to 10, as we observe most adv_grad_scale are below 10.
                        # adv_grad mean at fg area after scaling: 1e-3.
                        adv_grad = adv_grad * min(adv_grad_scale, 10)
                        # x_start - lambda * adv_grad minimizes the face embedding magnitudes.
                        # We subtract adv_grad from noise, then after noise is mixed with x_start, 
                        # adv_grad is effectively subtracted from x_start, minimizing the face embedding magnitudes.
                        # We predict the updated noise to remain consistent with the training paradigm.
                        # Q: should we subtract adv_grad from x_start instead of noise? I'm not sure.
                        noise[:DO_ADV_BS] -= adv_grad

                        self.adaface_adv_success_iters_count += 1
                        # adaface_adv_success_rate is always close to 1, so we don't monitor it.
                        # adaface_adv_success_rate = self.adaface_adv_success_iters_count / self.adaface_adv_iters_count
                        #mon_loss_dict.update({f'{session_prefix}/adaface_adv_success_rate': adaface_adv_success_rate})

                noises.append(noise)

        return noise_preds, noise_preds_cls, x_starts, x_recons, x_recons_cls, \
               noises, ts, ca_layers_activations_list


    # Do denoising, collect the attention activations for computing the losses later.
    # masks: (img_mask, fg_mask). 
    # Put them in a tuple to avoid too many arguments. The updated masks are returned.
    # On a 48GB GPU, we can only afford BLOCK_SIZE=1, otherwise OOM.
    def prime_x_start_for_comp_prompts(self, subj_context, x_start, noise,
                                       masks, num_comp_priming_denoising_steps, BLOCK_SIZE=1):
        prompt_emb, prompt_in, extra_info = subj_context
        # Although img_mask is not explicitly referred to in the following code,
        # it's updated within select_and_repeat_instances(slice(0, BLOCK_SIZE), 4, *masks).
        img_mask, fg_mask = masks

        # We use random noise for x_start, and 80% of the time, we use the training images.
        # NOTE: DO NOT x_start.normal_() here, as it will overwrite the x_start in the caller,
        # which is useful for loss computation.
        x_start = torch.randn_like(x_start) 
        # Set fg_mask to be the whole image.
        fg_mask = torch.ones_like(fg_mask)

        # Make the 4 instances in x_start, noise and t the same.
        x_start = x_start[:BLOCK_SIZE].repeat(4, 1, 1, 1)
        noise   = noise[:BLOCK_SIZE].repeat(4, 1, 1, 1)
        # In priming denoising steps, t is randomly drawn from the tail 20% segment of the timesteps 
        # (highly noisy).
        t_rear = torch.randint(int(self.num_timesteps * 0.7), int(self.num_timesteps * 0.9), 
                                (BLOCK_SIZE,), device=x_start.device)
        t      = t_rear.repeat(4)

        subj_single_prompt_emb, subj_comp_prompt_emb, subj_comp_rep_prompt_emb, cls_comp_prompt_emb = prompt_emb.chunk(4)
        cls_comp_prompt_emb2 = (1 - self.cls_subj_mix_ratio) * subj_comp_rep_prompt_emb + self.cls_subj_mix_ratio * cls_comp_prompt_emb

        # masks may have been changed in init_x_with_fg_from_training_image(). So we update it.
        # Update masks to be a 1-repeat-4 structure.
        masks = select_and_repeat_instances(slice(0, BLOCK_SIZE), 4, img_mask, fg_mask)

        all_t_list = []

        uncond_emb = self.uncond_context[0]

        # Class priming denoising: Denoise x_start_2 with the class single/comp prompts 
        # for num_sep_denoising_steps times, using self.comp_distill_priming_unet.
        # We only use half of the batch for faster class priming denoising.
        # x_start, t and noise are initialized as 1-repeat-4 at above, so the half is 1-repeat-2.
        # i.e., the denoising only differs in the prompt embeddings, but not in the x_start, t, and noise.
        x_start_2   = x_start.chunk(2)[0]
        t_2         = t.chunk(2)[1]
        noise_2     = noise.chunk(2)[0]

        # ** Do num_sep_denoising_steps of separate denoising steps with the single-comp prompts.
        # x_start_2[0] is denoised with the single prompt (both subj single and cls single before averaging), 
        # x_start_2[1] is denoised with the comp   prompt (both subj comp   and cls comp   before averaging).
        # Since we always use CFG for class priming denoising, we need to pass the negative prompt as well.
        # default cfg_scale_range=[2, 4].

        # cls_comp_prompt_emb2 + the SAR checkpoint will generate good compositional images. 
        # Later the two instances of (subj single, cls mix comp) will be repeated twice to get:
        # (subj single, cls mix comp, subj single, cls mix comp) as the primed x_start of
        # (subj single, subj comp,    cls single,  cls comp).
        # NOTE "cls mix comp" is used to initialize subj comp   and cls comp   instance denoising.
        #      "subj single"  is used to initialize subj single and cls single instance denoising.
        # Although there is misalignment on 3 out of 4 pairs, the 3 pairs are still semantically very close. 
        # So this should be fine.
        teacher_context=[ torch.cat([subj_single_prompt_emb, cls_comp_prompt_emb2], dim=0) ]

        with torch.no_grad():
            primed_noise_preds, primed_x_starts, primed_noises, all_t = \
                self.comp_distill_priming_unet(self, x_start_2, noise_2, t_2, 
                                               # In each timestep, the unet ensemble will do denoising on the same x_start_2 
                                               # with subj_double_prompt_emb and cls_double_prompt_emb, then average the results.
                                               # It's similar to do averaging on the prompt embeddings, but yields sharper results.
                                               # From the outside, the unet ensemble is transparent, like a single unet.
                                               teacher_context=teacher_context, 
                                               negative_context=uncond_emb,
                                               num_denoising_steps=num_comp_priming_denoising_steps,
                                               # Same t and noise across instances.
                                               same_t_noise_across_instances=True)
        
        all_t_list += [ ti[0].item() for ti in all_t ]
        print(f"Rank {self.trainer.global_rank} step {self.global_step}: "
                f"subj-cls ensemble prime denoising {num_comp_priming_denoising_steps} steps {all_t_list}")
        
        # The last primed_x_start is the final denoised image (with the smallest t).
        # So we use it as the x_start_primed to be denoised by the 4-type prompt set.
        # The subject and class instances use the same primed x_start. So we repeat primed_x_starts[-1] twice.
        x_start_primed  = primed_x_starts[-1].repeat(2, 1, 1, 1).to(dtype=x_start.dtype)
        # noise and masks are updated to be a 1-repeat-4 structure in prime_x_start_for_comp_prompts().
        # We return noise to make the noise_gt up-to-date, which is the recon objective.
        # But noise_gt is probably not referred to in the following loss computations,
        # since the current iteration is do_comp_feat_distill. We update it just in case.
        # masks will still be used in the loss computation. So we return updated masks as well.
        return x_start_primed, masks

    # In ordinary comp denoising,  noises = [noise],                 ts = [t], where noise and t are of 1-repeat-4 structures.
    # In subject-single denoising, noises = [noise_0, noise_1, ...], ts = [t_0, t_1, ...], i.e., 
    # for all num_denoising_steps, they are provided and shouldn't be randomly sampled.
    def comp_distill_multistep_denoise(self, x_start, noises, ts, subj_context, 
                                       uncond_emb, shrink_cross_attn=False, mix_sc_mc_attn=False,
                                       cfg_scale=2.5, num_denoising_steps=4, 
                                       use_attn_lora=False, use_ffn_lora=False, ffn_lora_adapter_name=None,
                                       BLKS=4):
        assert num_denoising_steps <= 10

        # When mixing sc and mc attention, disable attn and ffn LoRAs on all instances.
        use_attn_lora = use_attn_lora and (not mix_sc_mc_attn)
        use_ffn_lora  = use_ffn_lora  and (not mix_sc_mc_attn)

        # Initially, x_starts only contains the original x_start.
        x_starts    = [ x_start ]
        noise_preds = []
        x_recons    = []
        ca_layers_activations_list = []

        for i in range(num_denoising_steps):
            x_start = x_starts[i]
            t       = ts[i]
            noise   = noises[i]

            # batch_part_has_grad == 'subject-compos', i.e., only the subject compositional instance has gradients.
            '''
            (tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 
            tensor([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23]))
            '''

            noise_pred, x_recon, ca_layers_activations = \
                self.guided_denoise(x_start, noise, t, subj_context,
                                    uncond_emb, img_mask=None, 
                                    shrink_cross_attn=shrink_cross_attn,
                                    mix_sc_mc_attn=mix_sc_mc_attn,
                                    batch_part_has_grad='subject-compos' if BLKS > 1 else 'none',
                                    do_pixel_recon=True, cfg_scale=cfg_scale, 
                                    capture_ca_activations=True,
                                    # res_hidden_states_gradscale: 0.5, i.e.,
                                    # gradients are halved when flowing back through UNet skip connections.
                                    res_hidden_states_gradscale=self.res_hidden_states_gradscale,
                                    # use_attn_lora == self.unet_uses_attn_lora == True.
                                    # Enable the attn lora in subject-compos batches, as long as 
                                    # attn lora is globally enabled.
                                    use_attn_lora=use_attn_lora,
                                    # Don't use ffn lora in subject-compos batches.
                                    use_ffn_lora=use_ffn_lora,
                                    ffn_lora_adapter_name=ffn_lora_adapter_name)
        
            noise_preds.append(noise_pred)
            x_starts.append(x_recon.detach())
            x_recons.append(x_recon)
            ca_layers_activations_list.append(ca_layers_activations)

            # If noises[i+1] and t[i+1] are not provided as arguments, we need to sample an earlier timestep t
            # and noise for the next denoising step i+1.
            if i < num_denoising_steps - 1 and len(noises) <= i + 1:
                noise = torch.randn_like(x_start.chunk(BLKS)[0]).repeat(BLKS, 1, 1, 1)

                t0 = t.chunk(BLKS)[0]
                # NOTE: rand_like() samples from U(0, 1), not like randn_like().
                rand_ts = torch.rand_like(t0.float())
                # Make sure at the middle step (i < num_denoising_steps - 1), the timestep 
                # is between 50% and 70% of the current timestep. So if num_denoising_steps = 5,
                # we take timesteps within [0.5^0.66, 0.7^0.66] = [0.63, 0.79] of the current timestep.
                # If num_denoising_steps = 4, we take timesteps within [0.5^0.72, 0.7^0.72] = [0.61, 0.77] 
                # of the current timestep.
                # In general, the larger num_denoising_steps, the ratio between et and t0 is closer to 1.
                t_lb = t0 * np.power(0.5, np.power(num_denoising_steps - 1, -0.3))
                t_ub = t0 * np.power(0.7, np.power(num_denoising_steps - 1, -0.3))
                # et: earlier timestep, ts[i+1] < ts[i].
                # et is randomly sampled between [t_lb, t_ub].
                et = (t_ub - t_lb) * rand_ts + t_lb
                et = et.long()
                # Use the same t and noise for all instances.
                earlier_timesteps = et.repeat(BLKS)
                ts.append(earlier_timesteps)
                noises.append(noise)

        return noise_preds, x_starts, x_recons, noises, ts, ca_layers_activations_list
            
    # t: timesteps.
    # prompt_in is the textual prompts. 
    # extra_info: a dict that contains various fields. 
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, prompt_emb, prompt_in, extra_info):
        #print(prompt_in)
        subj_context = (prompt_emb, prompt_in, extra_info)
        img_mask     = self.iter_flags['img_mask']
        fg_mask      = self.iter_flags['fg_mask']

        noise = torch.randn_like(x_start) 

        ###### Begin of loss computation of the 3 types of iterations ######
        mon_loss_dict = {}
        session_prefix = 'train' if self.training else 'val'
        loss = 0

        # do_prompt_emb_delta_reg is always done, regardless of the iter_type.
        if self.iter_flags['do_prompt_emb_delta_reg']:
            loss_prompt_emb_delta = \
                calc_prompt_emb_delta_loss(extra_info['prompt_emb_4b_orig_dist'], extra_info['prompt_emb_mask_4b_orig'])

            mon_loss_dict.update({f'{session_prefix}/prompt_emb_delta': loss_prompt_emb_delta.mean().detach().item() })

            # prompt_emb_delta_reg_weight: 1e-4.
            loss += loss_prompt_emb_delta * self.prompt_emb_delta_reg_weight

        ##### begin of do_normal_recon #####
        if self.iter_flags['do_normal_recon']:  
            # all_subj_indices are used to extract the attention weights
            # of the subject tokens for the attention loss computation.
            # Then combine all subject indices into all_subj_indices.
            # self.embedding_manager.subject_string_dict: the key filter list. Only contains 'z' 
            # when each image contains a single subject.
            all_subj_indices = join_dict_of_indices_with_key_filter(extra_info['placeholder2indices'],
                                                                    self.embedding_manager.subject_string_dict)

            # NOTE: If normal_recon_on_pure_noise, then disable all LoRAs to avoid biases within LoRAs 
            # being introduced to images generated when normal_recon_on_pure_noise.
            # Enable attn LoRAs on UNet 50% of the time during recon iterations, to prevent
            # attn LoRAs don't degerate in comp distillation iterations.
            if not self.iter_flags['normal_recon_on_pure_noise']:
                enable_unet_attn_lora = self.unet_uses_attn_lora and (torch.rand(1).item() < 0.5)
                enable_unet_ffn_lora  = self.recon_uses_ffn_lora
                if self.comp_uses_ffn_lora and (torch.randn(1).item() < 0.25):
                    # 1/4 of the time we use comp_distill ffn lora adapter, 
                    # to prevent the lora from degeneration.
                    # But if not comp_uses_ffn_lora, then we always use the recon_loss lora.
                    ffn_lora_adapter_name = 'comp_distill'
                else:
                    ffn_lora_adapter_name = 'recon_loss'
            else:
                enable_unet_attn_lora = False
                enable_unet_ffn_lora  = False
                ffn_lora_adapter_name = 'recon_loss'    # placeholder. Not really used.

            # recon_with_adv_attack_iter_gap = 3, i.e., adversarial attack on the input images every 
            # 3 non-comp recon iterations.
            # Previously, recon_with_adv_attack_iter_gap = 4 and do_adv_attack on both non-comp and 
            # comp recon iterations. That is 25% of recon iterations.
            # Now,        recon_with_adv_attack_iter_gap = 3 and do_adv_attack only on non-comp recon iterations.
            # That is 80% / 3 = 26.67% of recon iterations. Almost the same as before.
            # Doing adversarial attack on the input images seems to introduce high-frequency noise 
            # to the whole image (not just the face area), so we only do it after the first denoise step.
            do_adv_attack = (torch.rand(1) < self.p_do_adv_attack_when_recon_on_images) \
                            and (not self.iter_flags['normal_recon_on_pure_noise'])
            # diffusers VAE is fp16, more memory efficient. So we can afford DO_ADV_BS=2.
            DO_ADV_BS = min(x_start.shape[0], 2)

            cls_context = (extra_info['cls_single_emb'], prompt_in, extra_info)

            loss_normal_recon = \
                self.calc_normal_recon_loss(mon_loss_dict, session_prefix, 
                                            # num_recon_denoising_steps: 2.
                                            self.num_recon_denoising_steps, x_start, noise, 
                                            subj_context, cls_context,
                                            img_mask, fg_mask, all_subj_indices, self.recon_bg_pixel_weight, 
                                            self.iter_flags['normal_recon_on_pure_noise'], 
                                            enable_unet_attn_lora, enable_unet_ffn_lora, ffn_lora_adapter_name,
                                            do_adv_attack, DO_ADV_BS)
            loss += loss_normal_recon
        ##### end of do_normal_recon #####

        ##### begin of do_unet_distill #####
        elif self.iter_flags['do_unet_distill']:
            # img_mask, fg_mask are used in recon_loss().
            loss_unet_distill = \
                self.calc_unet_distill_loss(mon_loss_dict, session_prefix,
                                            x_start, noise, subj_context, 
                                            img_mask, fg_mask, 
                                            self.iter_flags['num_unet_denoising_steps'], 
                                            self.iter_flags['unet_distill_on_pure_noise'])
            # loss_unet_distill: < 0.01, so we use a very large unet_distill_weight==8 to
            # make it comparable to the recon loss. Otherwise, loss_unet_distill will 
            # be dominated by the recon loss.
            loss += loss_unet_distill * self.unet_distill_weight
        ##### end of do_unet_distill #####

        ###### begin of do_comp_feat_distill ######
        elif self.iter_flags['do_comp_feat_distill']:
            # For simplicity, we fix BLOCK_SIZE = 1, no matter the batch size.
            # We can't afford BLOCK_SIZE=2 on a 48GB GPU as it will double the memory usage.            
            BLOCK_SIZE = 1
            all_subj_indices_1b = \
                join_dict_of_indices_with_key_filter(extra_info['placeholder2indices_1b'],
                                                     self.embedding_manager.subject_string_dict)

            # subj_context contains 
            #    (subj_single_emb, subj_comp_emb,      subj_comp_rep_emb, cls_comp_emb) 
            # But in order to do priming, we need cond_context_orig which contains
            # (subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb).
            # Therefore, we use extra_info['prompt_emb_4b_rep_nonmix'] to get the old context.
            cond_context_orig = (extra_info['prompt_emb_4b_rep_nonmix'], subj_context[1], subj_context[2])
            # ss_context: the context used for denoising the subject single instance alone.
            ss_context = (prompt_emb.chunk(4)[0], prompt_in[:BLOCK_SIZE], copy.copy(extra_info))

            # x_start_primed: the primed (denoised) x_start, ready for denoising.
            # noise and masks are updated to be a 1-repeat-4 structure in prime_x_start_for_comp_prompts().
            # We return noise to make the noise_gt up-to-date, which is the recon objective.
            # But noise_gt is probably not referred to in the following loss computations,
            # since the current iteration is do_comp_feat_distill. We update it just in case.
            # masks will still be used in the loss computation. So we update them as well.
            # NOTE: x_start still contains valid face images, as it's unchanged after priming.
            # Later it will be used for loss computation.            

            # num_comp_priming_denoising_steps alternates between 3 and 4.
            num_comp_priming_denoising_steps = self.comp_iters_count % 2 - 1 + self.max_num_comp_priming_denoising_steps

            x_start0 = x_start
            # Only the first 1/4 of the batch (actually 1 image), i.e., x_start0_ss, is used for priming.
            # They are repeated 4 times to form a primed batch.
            x_start_primed, masks = \
                self.prime_x_start_for_comp_prompts(cond_context_orig, x_start, noise,
                                                    (img_mask, fg_mask), num_comp_priming_denoising_steps, 
                                                    BLOCK_SIZE=BLOCK_SIZE)
            
            # Update masks.
            img_mask, fg_mask = masks
            # Regenerate the noise, since the noise above has been used in prime_x_start_for_comp_prompts().
            noise = torch.randn_like(x_start[:BLOCK_SIZE]).repeat(4, 1, 1, 1)

            x_start_ss, x_start_sc, x_start_sr, x_start_mc = x_start_primed.chunk(4)
            # Block 2 is the subject comp repeat (sc-repeat) instance.
            # Make the sc-repeat and sc blocks use the same x_start, so that their output features
            # are more aligned, and more effective for distillation.
            x_start_primed = torch.cat([x_start_ss, x_start_sc, x_start_sc, x_start_mc], dim=0)

            uncond_emb  = self.uncond_context[0].repeat(BLOCK_SIZE * 4, 1, 1)

            # t is randomly drawn from the middle rear 25% segment of the timesteps (noisy but not too noisy).
            t_midrear = torch.randint(int(self.num_timesteps * 0.45), int(self.num_timesteps * 0.7), 
                                      (BLOCK_SIZE,), device=x_start.device)
            # Same t_mid for all instances.
            t_midrear = t_midrear.repeat(BLOCK_SIZE * 4)

            # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
            # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
            # (img_mask is not used in the prompt-guided cross-attention layers).
            # NOTE: We don't use img_mask in compositional iterations. Because in compositional iterations,
            # the original images don't play a role, and the unet is supposed to generate the face from scratch.
            # so img_mask doesn't match the generated face and is set to None).

            # ca_layers_activations_list will be used in calc_comp_prompt_distill_loss().
            # noise_preds is not used for loss computation.
            # x_recons[-1] will be used to detect faces.
            # All x_recons with faces detected will be used for arcface align loss computation.
            noise_preds, x_starts, x_recons, noises, ts, ca_layers_activations_list = \
                self.comp_distill_multistep_denoise(x_start_primed, [noise], [t_midrear], subj_context,
                                                    uncond_emb=uncond_emb, 
                                                    shrink_cross_attn=self.iter_flags['shrink_cross_attn'],
                                                    mix_sc_mc_attn=self.iter_flags['mix_sc_mc_attn'],
                                                    cfg_scale=2.5, num_denoising_steps=self.num_comp_distill_denoising_steps,
                                                    use_attn_lora=self.unet_uses_attn_lora, use_ffn_lora=self.comp_uses_ffn_lora,
                                                    ffn_lora_adapter_name='comp_distill')

            ts_1st = [ t[0].item() for t in ts ]
            print(f"comp distill denoising steps: {self.num_comp_distill_denoising_steps}, ts: {ts_1st}")

            # Log x_start0 (augmented version of the input images),
            # x_start_primed (pure noise denoised for a few steps), and the denoised images for diagnosis.
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
            # All of them are 1, indicating green.
            x_start0_ss       = x_start0[:BLOCK_SIZE]
            x_start_primed_ss = x_start_primed[:BLOCK_SIZE]
            input_image = self.decode_first_stage(x_start0_ss)
            # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
            # All of them are 1, indicating green.
            log_image_colors = torch.ones(input_image.shape[0], dtype=int, device=x_start.device)
            self.cache_and_log_generations(input_image, log_image_colors, do_normalize=True)

            # NOTE: x_start_primed is primed with 0.3 subj embeddings and 0.7 cls embeddings. Therefore,
            # the faces still don't look like the subject. What matters is that the background is compositional.
            x_start_primed = x_start_primed.chunk(2)[0]
            log_image_colors = torch.ones(x_start_primed.shape[0], dtype=int, device=x_start.device)
            x_start_primed_decoded = self.decode_first_stage(x_start_primed)
            self.cache_and_log_generations(x_start_primed_decoded, log_image_colors, do_normalize=True)
            
            for i, x_recon in enumerate(x_recons):
                recon_images = self.decode_first_stage(x_recon)
                # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
                # If there are multiple denoising steps, the output images are assigned different colors.
                log_image_colors = torch.ones(recon_images.shape[0], dtype=int, device=x_start.device) * (i % 4)

                self.cache_and_log_generations(recon_images, log_image_colors, do_normalize=True)

                if self.log_attn_level > 0:
                    self.log_attention(ca_layers_activations_list[i], all_subj_indices_1b)

            # x_start0: x_start before priming, i.e., the input latent images. 
            loss_comp_feat_distill = \
                self.calc_comp_feat_distill_loss(mon_loss_dict, session_prefix,
                                                 x_start0_ss, x_start_primed_ss, x_recons, noise_preds, noises, ts,
                                                 ca_layers_activations_list, all_subj_indices_1b, 
                                                 ss_context, self.uncond_context[0], # Only pass one block of uncond embedding.
                                                 extra_info['prompt_emb_mask_4b'],
                                                 extra_info['prompt_pad_mask_4b'],
                                                 BLOCK_SIZE, self.sc_fg_face_suppress_mask_shrink_ratio,
                                                 use_attn_lora=self.unet_uses_attn_lora,
                                                 use_ffn_lora=self.comp_uses_ffn_lora)
            loss += loss_comp_feat_distill
        ##### end of do_comp_feat_distill #####

        else:
            breakpoint()

        if (torch.isnan(loss) or torch.isinf(loss)) and self.trainer.global_rank == 0:
            print('NaN/inf loss detected.')
            breakpoint()

        mon_loss_dict.update({f'{session_prefix}/loss': loss.mean().detach().item() })
        return loss, mon_loss_dict

    def are_faces_present_in_latents(self, latents):
        # images: [4, 3, 512, 512].
        images = self.decode_first_stage(latents)
        images = torch.clamp(images, -1, 1)
        are_face_detected = []
        for image in images:
            image = (image + 1) * 127.5
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')
            faces = self.face_detector.detect(image)
            are_face_detected.append(len(faces) > 0)

        are_face_detected = torch.tensor(are_face_detected, device=latents.device)
        return are_face_detected, images
    
    # If no faces are detected in x_recon, loss_arcface_align is 0, and face_bboxes is None.
    def calc_arcface_align_loss(self, x_start, x_recon, fg_faces_grad_mask_ratios=(1, 0.3)):
        # If there are faceless input images, then do_comp_feat_distill is always False.
        # Thus, here do_comp_feat_distill is always True, and x_start[0] is a valid face image.
        x_start_pixels = self.decode_first_stage(x_start)
        # NOTE: use the with_grad version of decode_first_stage. Otherwise no effect.
        subj_recon_pixels = self.decode_first_stage_with_grad(x_recon)

        # recon_fg_face_bboxes: long tensor of [BS, 4], where BS is the batch size.
        # If no face is detected in instance i, recon_fg_face_bboxes[i] is the full image size.
        # recon_fg_face_detected_inst_mask: binary tensor of [BS]
        loss_arcface_align, loss_fg_faces_suppress, loss_bg_faces_suppress, \
        recon_fg_face_bboxes, recon_fg_face_detected_inst_mask = \
            self.arcface.calc_arcface_align_loss(x_start_pixels, subj_recon_pixels, 
                                                 fg_faces_grad_mask_ratios=fg_faces_grad_mask_ratios)
        loss_arcface_align      = loss_arcface_align.to(x_start.dtype)
        loss_bg_faces_suppress  = loss_bg_faces_suppress.to(x_start.dtype)
        # Map the recon_fg_face_bboxes from the pixel space to latent space (scale down by 8x).
        # recon_fg_face_bboxes is None if there are no faces detected in x_recon.
        recon_fg_face_bboxes = pixel_bboxes_to_latent(recon_fg_face_bboxes, x_start_pixels.shape[-1], x_start.shape[-1])
        return loss_arcface_align, loss_fg_faces_suppress, loss_bg_faces_suppress, \
               recon_fg_face_bboxes, recon_fg_face_detected_inst_mask

    def calc_arcface_adv_grad(self, x_start):
        x_start.requires_grad = True
        # NOTE: To avoid horrible adv_grad artifacts in the background, we should use mask here to separate 
        # fg and bg when decode(). However, diffusers vae doesn't support mask, so we should avoid do_adv_attack. 
        orig_image = self.decode_first_stage_with_grad(x_start)
        # T=20: the smallest face size to be detected is 20x20. Note this is in the pixel space, 
        # so such faces are really small.
        face_embs_center, _, _, fg_face_bboxes, face_detected_inst_mask = \
            self.arcface.embed_image_tensor(orig_image, T=20, enable_grad=True, fg_faces_grad_mask_ratios=(0.9, 0.9))
        no_face_img_num = (1 - face_detected_inst_mask).sum()
        if no_face_img_num.sum() > 0:
            print(f"Failed to detect faces in {no_face_img_num} image, unable to compute adv_grad.")
            return None
        
        # NOTE: We want to push face_embs_center towards the negative direction, 
        # which is equivalent to push face_embs_center towards 0 == minimize (face_embs_center*face_embs_center).mean().
        # It's not simply reduce the magnitude of the face embedding. Since we add noise to the face image,
        # which introduces other random directions of the face embedding. When we reduce the  
        # face embedding magnitude along the original direction, we boost the noisy face embedding 
        # along the other directions more effectively.
        # Randomly drop 30% of the face embeddings, i.e., backprop only based on a subset of 
        # the face embeddings, to make the generated adv grad more stochastic 
        # and less artificial.
        face_embs_center = F.dropout(face_embs_center, p=0.3, training=True)
        self_align_loss = (face_embs_center ** 2).mean()
        # self_align_loss.backward() won't cause gradient syncing between GPUs, 
        # so we don't need to add self.trainer.model.no_sync() context here.
        self_align_loss.backward()
            
        adv_grad = x_start.grad
        x_start.grad = None
        x_start.requires_grad = False

        # Map the fg_face_bboxes from the pixel space to latent space (scale down by 8x).
        # fg_face_bboxes is None if there are no faces detected in x_recon.
        fg_face_bboxes = pixel_bboxes_to_latent(fg_face_bboxes, orig_image.shape[-1], x_start.shape[-1])
        # Set areas outside the fg_face_bboxes of adv_grad to negative.
        # NOTE: seems this trick still couldn't prevent the bg from being corrupted. Therefore,
        # we should avoid do_adv_attack.
        face_mask = torch.zeros_like(adv_grad)
        for i in range(x_start.shape[0]):
            x1, y1, x2, y2 = fg_face_bboxes[i]
            face_mask[i, :, y1:y2, x1:x2] = 1
        adv_grad = adv_grad * face_mask

        return adv_grad

    # cls_context: class single embeddings.
    # NOTE: cls_context is used to align the background denoised using ada embeddings 
    # with images denoised using the class embeddings. This is an important trick
    # *** to mitigate that the subject gradually dominates the whole image and a lot of artifacts
    # *** are generated in the background.
    # With this trick, we can be reassured to use arcface align loss without worrying it bringing a lot
    # of high-frequency noise to the background.
    # enable_unet_attn_lora: randomly set to True 50% of the time.
    # enable_unet_ffn_lora: if not normal_recon_on_pure_noise, then True. Otherwise False.
    # recon_bg_pixel_weight == 0.1, a small penalty on bg errors.
    def calc_normal_recon_loss(self, mon_loss_dict, session_prefix, 
                               num_denoising_steps, x_start, noise, subj_context, cls_context,
                               img_mask, fg_mask, all_subj_indices, recon_bg_pixel_weight,
                               normal_recon_on_pure_noise, 
                               enable_unet_attn_lora, enable_unet_ffn_lora, ffn_lora_adapter_name,
                               do_adv_attack, DO_ADV_BS):
        loss_normal_recon = torch.tensor(0.0, device=x_start.device)

        BLOCK_SIZE = x_start.shape[0]
        # The t at the first denoising step are sampled from the middle rear 0.5 ~ 0.8 of the timesteps.
        # In the subsequent denoising steps, the timesteps become gradually earlier.
        if normal_recon_on_pure_noise:
            t = torch.randint(int(self.num_timesteps * 0.7), int(self.num_timesteps * 0.9), 
                              (BLOCK_SIZE,), device=x_start.device).long()
            x_start0 = torch.randn_like(x_start)
            num_recon_priming_steps = 4
            num_denoising_steps += num_recon_priming_steps
        else:
            t = torch.randint(int(self.num_timesteps * 0.5), int(self.num_timesteps * 0.8), 
                              (x_start.shape[0],), device=self.device).long()
            x_start0 = x_start
            num_recon_priming_steps = 0

        if num_denoising_steps > 1 or normal_recon_on_pure_noise:
            # When doing multi-step denoising, we apply CFG on the recon images.
            # Use the null prompt as the negative prompt.
            uncond_emb = self.uncond_context[0].repeat(BLOCK_SIZE, 1, 1)
            # If cfg_scale == 2, result = 2 * noise_pred - noise_pred_neg.
            cfg_scale = 2
            if normal_recon_on_pure_noise:
                img_mask = None
                fg_mask  = torch.ones_like(fg_mask)
        else:
            # Use the default negative prompts.
            # Don't do CFG. So uncond_emb is None.
            uncond_emb = None
            cfg_scale  = -1

        input_images = self.decode_first_stage(x_start)
        # log_image_colors: a list of 3, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
        # All of them are 3, purple.
        log_image_colors = torch.ones(input_images.shape[0], dtype=int, device=x_start.device) * 3
        self.cache_and_log_generations(input_images, log_image_colors, do_normalize=True)

        # mon_loss_dict is used to log adv_attack stats.
        # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
        # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
        # (img_mask is not used in the prompt-guided cross-attention layers).
        noise_preds, noise_preds_cls, x_starts, x_recons, x_recons_cls, noises, ts, ca_layers_activations_list = \
            self.recon_multistep_denoise(mon_loss_dict, session_prefix, 
                                         # img_mask is used to mask the blank areas around the augmented images.
                                         # fg_mask is used to confine adv_grad to the foreground area.
                                         x_start0, noise, t, subj_context, cls_context, 
                                         uncond_emb, img_mask, fg_mask,
                                         cfg_scale, num_denoising_steps, num_recon_priming_steps,
                                         # If normal_recon_on_pure_noise, enable_unet_attn_lora, enable_unet_ffn_lora are False
                                         # Otherwise, enable_unet_attn_lora: randomly set to True 50% of the time.
                                         # enable_unet_ffn_lora: True.
                                         normal_recon_on_pure_noise, enable_unet_attn_lora, enable_unet_ffn_lora,
                                         ffn_lora_adapter_name, # Switching between 'recon_loss' or 'comp_distill'.
                                         do_adv_attack, DO_ADV_BS)

        losses_recon = []
        losses_recon_cls = []
        recon_loss_scales = []
        losses_recon_subj_mb_suppress = []
        losses_arcface_align_recon    = []
        losses_arcface_align_recon_stat = []
        losses_bg_faces_suppress      = []
        pred_l2s        = []
        latent_shape, device = x_start.shape, x_start.device

        # Skip the first num_recon_priming_steps denoising steps from the recon loss computation.
        for i in range(num_recon_priming_steps, num_denoising_steps):
            noise, noise_pred, noise_pred_cls, x_recon, \
            x_recon_cls, ca_layers_activations = \
                noises[i], noise_preds[i], noise_preds_cls[i], x_recons[i], \
                x_recons_cls[i], ca_layers_activations_list[i]

            recon_images_cls = self.decode_first_stage(x_recon_cls)
            recon_images     = self.decode_first_stage(x_recon)
            log_image_colors = torch.ones(recon_images_cls.shape[0], dtype=int, device=x_start.device) * 3 \
                                + i + 1 - num_recon_priming_steps
            self.cache_and_log_generations(recon_images_cls, log_image_colors, do_normalize=True)
            # log_image_colors: a list of 4 or 5, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
            # 4 or 5: orange for the first denoising step, blue for the second denoising step.
            log_image_colors = torch.ones(recon_images.shape[0], dtype=int, device=x_start.device) * 3 \
                                + i + 1 - num_recon_priming_steps
            self.cache_and_log_generations(recon_images, log_image_colors, do_normalize=True)
            
            # Calc the L2 norm of noise_pred.
            pred_l2_step = (noise_pred ** 2).mean()
            pred_l2s.append(pred_l2_step)

            # Only compute loss_recon and loss_recon_subj_mb_suppress when faces are detected in x_recon.
            # loss_arcface_align_recon_step > 0 implies there's a face detected in each instances in x_recon.
            if self.arcface_align_loss_weight > 0:
                # If no faces are detected in x_recon, loss_arcface_align is 0, and fg_face_bboxes is None.
                # If there is no face detected in any of the instances, then loss_arcface_align_recon_step is 0,
                # and recon loss is skipped. 
                # We still compute the recon loss on the good instances with face detected, 
                # which are indicated by face_detected_inst_mask.
                # face_detected_inst_mask: binary tensor of [BS].
                # loss_fg_faces_suppress_step is not optimized. 
                # So the second item in fg_faces_grad_mask_ratios has no effect.
                loss_arcface_align_recon_step, loss_fg_faces_suppress_step, loss_bg_faces_suppress_step, fg_face_bboxes, face_detected_inst_mask = \
                    self.calc_arcface_align_loss(x_start, x_recon, fg_faces_grad_mask_ratios=(1, 0.7))

                # Count normal_recon_on_pure_noise stats and non-pure-noise stats separately.
                # normal_recon_face_images_on_*_stats contain face_images_count and all_images_count.
                if normal_recon_on_pure_noise:
                    self.normal_recon_face_images_on_noise_stats.update([face_detected_inst_mask.sum().item(),
                                                                         face_detected_inst_mask.shape[0]])
                else:
                    self.normal_recon_face_images_on_image_stats.update([face_detected_inst_mask.sum().item(),
                                                                         face_detected_inst_mask.shape[0]])

                if loss_arcface_align_recon_step > 0:
                    # recon_face_align_loss_thres: 0.7. Only optimize the face align loss if <= 0.7.
                    if (self.recon_face_align_loss_thres <= 0) or (loss_arcface_align_recon_step < self.recon_face_align_loss_thres):
                        losses_arcface_align_recon.append(loss_arcface_align_recon_step)
                        self.normal_recon_face_align_loss_kept_frac.update(1)
                    else:
                        self.normal_recon_face_align_loss_kept_frac.update(0)
                        
                    # loss_arcface_align_recon_stat is the mean of all loss_arcface_align_recon_step 
                    # without filtering.
                    losses_arcface_align_recon_stat.append(loss_arcface_align_recon_step)
                    # In this branch, at least one face is detected in x_recon.
                    # Set the weights of the instances without faces detected to 0.1, 
                    # to downscale corresponding gradients which are more noisy than those with faces detected.
                    face_detected_inst_weights = face_detected_inst_mask.clone()
                    face_detected_inst_weights[face_detected_inst_mask==0] = 0.1                        
                    recon_loss_scale = 1.
                    # If there are at least one face detected, then we get face_bb_mask.
                    # For failed instances, the fg_face_bboxes are the full image size. Therefore,
                    # multiplying with fg_mask has no effect, and the recon loss still applies to the whole image.
                    # NOTE: fg_face_bboxes has been converted to the coords of the latents space, 64*64,
                    # in calc_arcface_align_loss(). So we DON'T need to convert it again.
                    face_bb_mask = torch.zeros(BLOCK_SIZE, 1, latent_shape[-2], latent_shape[-1], device=device)
                    for j in range(len(fg_face_bboxes)):
                        x1, y1, x2, y2 = fg_face_bboxes[j]
                        face_bb_mask[j, :, y1:y2, x1:x2] = 1
                        print(f"Rank {self.trainer.global_rank} recon face coords {j}: {fg_face_bboxes[j].detach().cpu().numpy()}.", end=' ')

                    fg_mask2 = fg_mask * face_bb_mask
                    mask_overlap_ratio = fg_mask2.sum() / fg_mask.sum()
                    print(f"Recon face detected/segmented mask overlap: {mask_overlap_ratio:.2f}")
                else:
                    # No faces detected in x_recon.
                    # face_detected_inst_weights set to all ones, i.e., we recon all instances equally.
                    # After that, scale down the recon loss by 0.1, for the same purpose 
                    # as part of face_detected_inst_weights being set to 0.1 above.
                    # NOTE: if no faces are detected in x_recon and we set face_detected_inst_weights to 
                    # all 0.1, then it's equivalent to setting face_detected_inst_weights to all ones,
                    # since we normalize the pixel-wise recon losses by the number of weighted face pixels.
                    recon_loss_scale = 0.1
                    face_detected_inst_weights = torch.ones_like(face_detected_inst_mask)
                    fg_mask2 = fg_mask

                # NOTE: recon_bg_pixel_weight = 0.1, i.e.,
                # bg loss is given a small weight to suppress multi-face artifacts.
                # NOTE: img_mask is set to None, because calc_recon_loss() only considers pixels
                # not masked by img_mask. Therefore, blank pixels due to augmentation are not regularized.
                # If img_mask = None, then blank pixels are regularized as bg pixels.
                loss_recon_step, loss_recon_cls_step, loss_recon_subj_mb_suppress_step = \
                    calc_recon_and_suppress_losses(noise, noise_pred, noise_pred_cls, 
                                                   face_detected_inst_weights, 
                                                   ca_layers_activations,
                                                   all_subj_indices, None, fg_mask2,
                                                   recon_bg_pixel_weight, x_start.shape[0],
                                                   normal_recon_on_pure_noise)
                
                losses_recon.append(loss_recon_step)
                losses_recon_cls.append(loss_recon_cls_step)
                recon_loss_scales.append(recon_loss_scale)
                losses_recon_subj_mb_suppress.append(loss_recon_subj_mb_suppress_step)
                if loss_bg_faces_suppress_step > 0:
                    losses_bg_faces_suppress.append(loss_bg_faces_suppress_step)

        # recon_face_images_on_noise_frac: the window-accumulated ratio of (num of noise recon face images / num of all recon images).
        recon_face_images_on_noise_frac = self.normal_recon_face_images_on_noise_stats.sums[0] / (self.normal_recon_face_images_on_noise_stats.sums[1] + 1e-2)
        mon_loss_dict.update({f'{session_prefix}/recon_face_images_on_noise_frac': recon_face_images_on_noise_frac})
        # recon_face_images_on_image_frac: the window-accumulated ratio of (num of normal recon face images / num of all recon images).
        recon_face_images_on_image_frac = self.normal_recon_face_images_on_image_stats.sums[0] / (self.normal_recon_face_images_on_image_stats.sums[1] + 1e-2)
        mon_loss_dict.update({f'{session_prefix}/recon_face_images_on_image_frac': recon_face_images_on_image_frac})
        mon_loss_dict.update({f'{session_prefix}/recon_face_align_loss_kept_frac': self.normal_recon_face_align_loss_kept_frac.mean})

        arcface_align_recon_loss_scale = 1
        #### loss_arcface_align_recon ####
        if len(losses_arcface_align_recon) > 0:
            loss_arcface_align_recon = torch.stack(losses_arcface_align_recon).mean()
            if loss_arcface_align_recon > 0:
                if normal_recon_on_pure_noise:
                    mon_loss_dict.update({f'{session_prefix}/arcface_align_recon_on_noise_opt': loss_arcface_align_recon.mean().detach().item() })
                    # When normal_recon_on_pure_noise, loss_arcface_align_recon is the only loss, 
                    # so we scale it up to 4x.
                    arcface_align_recon_loss_scale = 4
                else:
                    mon_loss_dict.update({f'{session_prefix}/arcface_align_recon_on_image_opt': loss_arcface_align_recon.mean().detach().item() })
                # loss_arcface_align_recon: 0.4-0.5. arcface_align_loss_weight: 0.01 => 0.004-0.005.
                # This loss is around 1/20 of recon/distill losses (0.1).
                # If normal_recon_on_pure_noise, then loss_arcface_align_recon => 0.016-0.02.
                loss_normal_recon += loss_arcface_align_recon * self.arcface_align_loss_weight * arcface_align_recon_loss_scale

        if len(losses_arcface_align_recon_stat) > 0:
            loss_arcface_align_recon_stat = torch.stack(losses_arcface_align_recon_stat).mean()
            if normal_recon_on_pure_noise:
                mon_loss_dict.update({f'{session_prefix}/arcface_align_recon_on_noise': loss_arcface_align_recon_stat.mean().detach().item() })
                print(f"Rank {self.trainer.global_rank} arcface_align_recon_on_noise: {loss_arcface_align_recon_stat.detach().item():.4f}")
            else:
                mon_loss_dict.update({f'{session_prefix}/arcface_align_recon_on_image': loss_arcface_align_recon_stat.mean().detach().item() })
                print(f"Rank {self.trainer.global_rank} arcface_align_recon_on_image: {loss_arcface_align_recon_stat.detach().item():.4f}")

        #### loss_bg_faces_suppress ####
        if len(losses_bg_faces_suppress) > 0:
            loss_bg_faces_suppress = torch.stack(losses_bg_faces_suppress).mean()
            if loss_bg_faces_suppress > 0:
                mon_loss_dict.update({f'{session_prefix}/recon_bg_faces_suppress': loss_bg_faces_suppress.mean().detach().item() })
                # loss_bg_faces_suppress_comp is a mean L2 loss, only ~0.02. * 2 => 0.04,
                # same scale as loss_bg_faces_suppress_comp.
                # Although this is ~10x of loss_arcface_align_recon, it's very infraquently triggered.
                recon_bg_faces_suppress_loss_scale = 2 * arcface_align_recon_loss_scale
                loss_normal_recon += loss_bg_faces_suppress * recon_bg_faces_suppress_loss_scale

        #### pred_l2 ####
        pred_l2 = torch.stack(pred_l2s).mean()
        # pred_l2: 0.92~0.99.
        mon_loss_dict.update({f'{session_prefix}/pred_l2': pred_l2.mean().detach().item()})

        #### loss_recon_subj_mb_suppress ####
        loss_recon_subj_mb_suppress = torch.stack(losses_recon_subj_mb_suppress).mean()
        # If fg_mask is None, then loss_recon_subj_mb_suppress = loss_bg_mf_suppress = 0.
        # If normal_recon_on_pure_noise, then loss_recon_subj_mb_suppress is not optimized, and is only for monitoring.
        if loss_recon_subj_mb_suppress > 0:
            mon_loss_dict.update({f'{session_prefix}/recon_subj_mb_suppress': loss_recon_subj_mb_suppress.mean().detach().item()})
        recon_loss_scales = torch.tensor(recon_loss_scales, device=device)

        #### loss_recon ####
        if not normal_recon_on_pure_noise:
            losses_recon = torch.stack(losses_recon)
            loss_recon   = (losses_recon * recon_loss_scales).mean()
            # The scaled loss_recon may be smaller when none of the images have faces detected
            # due to discount.
            # Therefore, the unscaled loss_recon shows the real difficulty of the recon task.
            loss_recon_unscaled = losses_recon.mean()
            v_loss_recon = loss_recon_unscaled.detach().item()

            mon_loss_dict.update({f'{session_prefix}/loss_recon': v_loss_recon})

            ts_str = ", ".join([ f"{t.tolist()}" for t in ts ])
            print(f"Rank {self.trainer.global_rank} {num_denoising_steps}-step recon: {ts_str}, {v_loss_recon:.4f}")
            # loss_recon: ~0.1
            loss_normal_recon += loss_recon
            # loss_recon_subj_mb_suppress: 0.1, recon_subj_mb_suppress_loss_weight: 0.05 -> 0.005.
            # recon loss: 0.1, loss_recon_subj_mb_suppress is 1/20 of recon loss.
            loss_normal_recon += loss_recon_subj_mb_suppress * self.recon_subj_mb_suppress_loss_weight
        else:
            print(f"Rank {self.trainer.global_rank} normal_recon_on_pure_noise.")

        #### loss_recon_cls ####
        losses_recon_cls = torch.stack(losses_recon_cls)
        loss_recon_cls   = (losses_recon_cls * recon_loss_scales).mean()
        loss_normal_recon += loss_recon_cls
        loss_recon_cls_unscaled = losses_recon_cls.mean()
        v_loss_recon_cls = loss_recon_cls_unscaled.detach().item()
        mon_loss_dict.update({f'{session_prefix}/loss_recon_cls': v_loss_recon_cls})
        print(f"Rank {self.trainer.global_rank} loss_recon_cls: {v_loss_recon_cls:.4f}")

        mon_loss_dict.update({f'{session_prefix}/normal_recon_total': loss_normal_recon.mean().detach().item()})

        return loss_normal_recon

    def prepare_teacher_context(self, subj_context, uncond_context, BLOCK_SIZE, 
                                id2img_prompt_embs, id2img_neg_prompt_embs, img_prompt_prefix_embs,
                                unet_teacher_types, encoders_num_id_vecs, 
                                p_unet_teacher_uses_cfg, unet_distill_uses_comp_prompt):

        prompt_emb, prompt_in, extra_info = subj_context
        # student_prompt_embs is the prompt embedding of the student model.
        student_prompt_embs = subj_context[0]

        ####### Begin of Preparing the teacher context #######
        # ** OBSOLETE ** NOTE: when unet_teacher_types == ['unet_ensemble'], unets are specified in 
        # extra_unet_dirpaths (finetuned unets on the original SD unet); 
        # In this case, they are surely not 'arc2face' or 'consistentID'.
        # The same student_prompt_embs is used by all unet_teachers.
        if unet_teacher_types == ['unet_ensemble']:
            teacher_contexts = [student_prompt_embs]
        else:
            # The whole set of teachers have been initialized,
            # if id2ada_prompt_encoder.name == 'jointIDs' by setting 
            # personalization_config.params.adaface_encoder_types = ['consistentID', 'arc2face']).
            # But some may be disabled by setting
            # personalization_config.params.enabled_encoders = ['consistentID'] or ['arc2face'].
            teacher_contexts = []
            # If id2ada_prompt_encoder.name == 'jointIDs',         then encoders_num_id_vecs is not None.
            # Otherwise, id2ada_prompt_encoder is a single encoder, and encoders_num_id_vecs is None.
            if encoders_num_id_vecs is not None:
                all_id2img_prompt_embs      = id2img_prompt_embs.split(encoders_num_id_vecs, dim=1)
                all_id2img_neg_prompt_embs  = id2img_neg_prompt_embs.split(encoders_num_id_vecs, dim=1)
                # If id2ada_prompt_encoder.name == 'jointIDs', the img_prompt_embs are ordered as such.
                encoder_name2idx = { 'consistentID': 0, 'arc2face': 1 }
            else:
                # Single FaceID2AdaPrompt encoder. No need to split id2img_prompt_embs/id2img_neg_prompt_embs.
                all_id2img_prompt_embs      = [ id2img_prompt_embs ]
                all_id2img_neg_prompt_embs  = [ id2img_neg_prompt_embs ]
                encoder_name2idx = { unet_teacher_types[0]: 0 }
                
            for unet_teacher_type in unet_teacher_types:
                if unet_teacher_type not in ['consistentID', 'arc2face']:
                    breakpoint()
                
                teacher_idx = encoder_name2idx[unet_teacher_type]
                if unet_teacher_type == 'arc2face':
                    # img_prompt_prefix_embs: the embeddings of a template prompt "photo of a"
                    # For arc2face, p_unet_teacher_uses_cfg is always 0. So we only pass pos_prompt_embs.
                    img_prompt_prefix_embs = img_prompt_prefix_embs.repeat(BLOCK_SIZE, 1, 1)
                    # teacher_context: [BS, 4+16, 768] = [BS, 20, 768]
                    teacher_context = torch.cat([img_prompt_prefix_embs, all_id2img_prompt_embs[teacher_idx]], dim=1)

                    if p_unet_teacher_uses_cfg > 0:
                        # When p_unet_teacher_uses_cfg > 0, we provide both pos_prompt_embs and neg_prompt_embs 
                        # to the teacher.
                        # uncond_context is a tuple of (uncond_embs, uncond_prompt_in, extra_info).
                        # Truncate the uncond_embs to the same length as teacher_context.
                        LEN_POS_PROMPT = teacher_context.shape[1]
                        # NOTE: Since arc2face doesn't respond to compositional prompts, 
                        # even if unet_distill_uses_comp_prompt,
                        # we don't need to set teacher_neg_context as the negative compositional prompts.
                        teacher_neg_context = uncond_context[0][:, :LEN_POS_PROMPT].repeat(BLOCK_SIZE, 1, 1)
                        # The concatenation of teacher_context and teacher_neg_context is done on dim 0.
                        teacher_context = torch.cat([teacher_context, teacher_neg_context], dim=0)

                elif unet_teacher_type == 'consistentID':
                    global_id_embeds = all_id2img_prompt_embs[teacher_idx]
                    # global_id_embeds: [BS, 4,  768]
                    # cls_prompt_embs:  [BS, 77, 768]
                    if unet_distill_uses_comp_prompt:
                        cls_emb_key = 'cls_comp_emb'  
                    else:
                        cls_emb_key = 'cls_single_emb'

                    cls_prompt_embs = extra_info[cls_emb_key]
                    # Always append the ID prompt embeddings to the class (general) prompt embeddings.
                    # teacher_context: [BS, 81, 768]
                    teacher_context = torch.cat([cls_prompt_embs, global_id_embeds], dim=1)    
                    if p_unet_teacher_uses_cfg > 0:
                        # When p_unet_teacher_uses_cfg > 0, we provide both pos_prompt_embs and neg_prompt_embs 
                        # to the teacher.
                        global_neg_id_embs = all_id2img_neg_prompt_embs[teacher_idx]
                        # uncond_context is a tuple of (uncond_emb, uncond_prompt_in, extra_info).
                        # uncond_context[0]: [1, 77, 768] -> [BS, 77, 768]
                        cls_neg_prompt_embs = uncond_context[0].repeat(teacher_context.shape[0], 1, 1)

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
        ####### End of Preparing the teacher context #######

        return teacher_contexts
    
    def calc_unet_distill_loss(self, mon_loss_dict, session_prefix, 
                               x_start, noise, subj_context, 
                               img_mask, fg_mask, num_unet_denoising_steps, unet_distill_on_pure_noise):
        if unet_distill_on_pure_noise:
            # Alternate between priming using Adaface and priming using the teacher.
            priming_using_adaface = (self.unet_distill_on_noise_iters_count % 2 == 0)
            # 6: pink, 7: magenta
            log_color_idx = 6 if priming_using_adaface else 7
        else:
            log_color_idx = 2

        BLOCK_SIZE = x_start.shape[0]
        uncond_emb = self.uncond_context[0].repeat(BLOCK_SIZE, 1, 1)
        x_start_pixels = self.decode_first_stage(x_start)
        
        # Regenerate a slightly smaller t for unet distillation.
        t = torch.randint(int(self.num_timesteps * 0.7), int(self.num_timesteps * 0.9), 
                          (x_start.shape[0],), device=x_start.device).long()

        # log_image_colors: a list of 0-6, indexing colors 
        # = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
        # If unet_distill_on_pure_noise: all of them are 6 or 7, indicating pink or magenta.
        # If unet_distill_on_image:      all of them are 2,      indicating red.
        log_image_colors = torch.ones(x_start_pixels.shape[0], dtype=int, device=x_start.device) * log_color_idx
        self.cache_and_log_generations(x_start_pixels, log_image_colors, do_normalize=True)
        teacher_contexts = self.prepare_teacher_context(subj_context, self.uncond_context, BLOCK_SIZE,
                                                        self.iter_flags['id2img_prompt_embs'],
                                                        self.iter_flags['id2img_neg_prompt_embs'],
                                                        self.img_prompt_prefix_embs,
                                                        self.unet_teacher_types,
                                                        self.iter_flags['encoders_num_id_vecs'],
                                                        self.p_unet_teacher_uses_cfg,
                                                        self.iter_flags['unet_distill_uses_comp_prompt'])
        
        # unet_distill_on_pure_noise: Use totally random x_start as the input latent images.
        if unet_distill_on_pure_noise:
            num_distill_priming_steps = 4
            # If all num_priming_trials of priming trials fail (no face detected),
            # then we stop trying and go with the last x_start.
            num_priming_trials = 3

            self.unet_distill_on_noise_iters_count += 1
            # img_mask is used to mask the blank areas around the augmented images.
            # As unet_distill_on_pure_noise, we set img_mask = None.
            img_mask0 = None
            fg_mask0  = torch.ones_like(fg_mask)

            for trial_idx in range(num_priming_trials):
                # Regenerate x_start, noise0 and t0 for each trial.
                x_start = torch.randn_like(x_start)
                noise0  = torch.randn_like(noise)
                t0      = torch.randint(int(self.num_timesteps * 0.75), int(self.num_timesteps * 0.9), 
                                        (x_start.shape[0],), device=x_start.device).long()

                if priming_using_adaface:
                    # Since not do_adv_attack, mon_loss_dict and session_prefix are not used and could be set to None.
                    # We set num_denoising_steps = num_distill_priming_steps, 
                    # i.e., we only do priming steps in recon_multistep_denoise().
                    # recon_multistep_denoise() always uses CFG.
                    noise_preds, _, x_starts, x_recons, _, noises, ts, ca_layers_activations_list = \
                        self.recon_multistep_denoise(None, None, 
                                                     x_start, noise0, t0, subj_context, None, 
                                                     uncond_emb, img_mask0, fg_mask0,
                                                     cfg_scale=2, num_denoising_steps=num_distill_priming_steps, 
                                                     num_priming_steps=num_distill_priming_steps,
                                                     normal_recon_on_pure_noise=True, 
                                                     enable_unet_attn_lora=False, enable_unet_ffn_lora=False, 
                                                     # ffn_lora_adapter_name is 'unet_distill', 
                                                     # to get an x_start compatible with the teacher.
                                                     ffn_lora_adapter_name='recon_loss',  
                                                     do_adv_attack=False, DO_ADV_BS=-1)
                else:
                    with torch.no_grad():
                        # force_uses_cfg=True: unet_teacher() always uses CFG.
                        noise_preds, x_starts, noises, ts = \
                            self.unet_teacher(self, x_start, noise0, t0, teacher_contexts, 
                                              num_denoising_steps=num_distill_priming_steps,
                                              force_uses_cfg=True)

                # x_starts are denoised image latents, and x_start is the last denoised image latent.
                x_start = x_starts[-1]
                are_face_detected, x_start_pixels = self.are_faces_present_in_latents(x_start)
                priming_model_name = 'Adaface' if priming_using_adaface else 'Teacher'
                if torch.all(are_face_detected):
                    # If all instances contain faces, then we can stop the priming trials.
                    print(f"Rank {self.trainer.global_rank} {priming_model_name} distill priming trial {trial_idx+1}/{num_priming_trials} succeeded. Stop.")
                    break
                else:
                    continue_or_give_up = 'continue' if trial_idx < num_priming_trials - 1 else 'give up'
                    print(f"Rank {self.trainer.global_rank} {priming_model_name} distill priming trial {trial_idx+1}/{num_priming_trials} failed. {continue_or_give_up}."
                          f"Face detected: {are_face_detected.sum().item()}/{are_face_detected.shape[0]}.")

            # Log the primed x_start images.
            log_image_colors = torch.ones(x_start_pixels.shape[0], dtype=int, device=x_start.device) * log_color_idx
            self.cache_and_log_generations(x_start_pixels, log_image_colors, do_normalize=True)

        with torch.no_grad():
            unet_teacher_noise_preds, unet_teacher_x_starts, unet_teacher_noises, all_t = \
                self.unet_teacher(self, x_start, noise, t, teacher_contexts, 
                                  num_denoising_steps=num_unet_denoising_steps,
                                  force_uses_cfg=False)
        
        # **Objective 2**: Align student noise predictions with teacher noise predictions.
        # noise_gts: replaced as the reconstructed x0 by the teacher UNet.
        # If ND = num_unet_denoising_steps > 1, then unet_teacher_noise_preds contain ND 
        # unet_teacher predicted noises (of different ts).
        # noise_gts: [HALF_BS, 4, 64, 64] * num_unet_denoising_steps.
        noise_gts = unet_teacher_noise_preds

        # The outputs of the remaining denoising steps will be appended to noise_preds.
        noise_preds = []

        for s in range(num_unet_denoising_steps):
            # Predict the noise with t_s (a set of earlier t).
            # When s > 1, x_start_s is the unet_teacher predicted images in the previous step,
            # used to seed the second denoising step. 
            # Some unet_teacher_x_starts used CFG and some did not.
            # NOTE: CFG only changes unet_teacher_x_starts and x_recon_s, 
            # not the predicted noises noise_pred_s, on which we compute the distillation loss.
            # Therefore, using CFG or not doesn't impact the distillation loss.
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
            # We intentionally do not use img_mask in unet distillation. 
            # Otherwise the task will be too easy for the student.
            noise_pred_s, x_recon_s, ca_layers_activations = \
                self.guided_denoise(x_start_s, noise_t, t_s, subj_context, 
                                    uncond_emb=uncond_emb, img_mask=None,
                                    shrink_cross_attn=False,
                                    # do_pixel_recon implies using CFG to get x_recon_s.
                                    batch_part_has_grad='all', do_pixel_recon=True, 
                                    cfg_scale=self.unet_teacher.cfg_scale,
                                    capture_ca_activations=False,
                                    # When doing unet distillation, res_hidden_states_gradscale is always 0, 
                                    # i.e., gradients don't flow back through UNet skip connections.
                                    res_hidden_states_gradscale=0,
                                    # ** Always disable attn LoRAs on unet distillation.
                                    use_attn_lora=False,                    
                                    # ** Always enable ffn LoRAs on unet distillation to reduce domain gap.
                                    use_ffn_lora=True, 
                                    ffn_lora_adapter_name='unet_distill')

            noise_preds.append(noise_pred_s)

            recon_images_s = self.decode_first_stage(x_recon_s)
            # log_image_colors: a list of 0-6, indexing colors 
            # = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
            # If unet_distill_on_pure_noise: all of them are 6, indicating pink.
            # If unet_distill_on_image:      all of them are 2, indicating red.
            log_image_colors = torch.ones(recon_images_s.shape[0], dtype=int, device=x_start.device) * log_color_idx
            self.cache_and_log_generations(recon_images_s, log_image_colors, do_normalize=True)

        iter_type_str = f'on {num_distill_priming_steps}-step primed noise' if unet_distill_on_pure_noise else 'on image'
        print(f"Rank {self.trainer.global_rank} {len(noise_preds)}-step distillation ({iter_type_str}):")
        losses_unet_distill = []

        for s in range(len(noise_preds)):
            noise_pred, noise_gt = noise_preds[s], noise_gts[s]

            # In the compositional iterations, unet_distill_uses_comp_prompt is always False.
            # If we use comp_prompt as condition, then the background is compositional, and 
            # we want to do recon on the whole image. But considering background is not perfect, 
            # esp. for consistentID whose compositionality is not so good, so recon_bg_pixel_weight = 0.5.
            if self.iter_flags['unet_distill_uses_comp_prompt']:
                recon_bg_pixel_weight = 0.5
            else:
                # unet_teacher_type == ['arc2face'] or ['consistentID'] or ['consistentID', 'arc2face'].
                recon_bg_pixel_weight = 0

            # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
            loss_unet_distill, _ = \
                calc_recon_loss(F.mse_loss, noise_pred, noise_gt.to(noise_pred.dtype), 
                                img_mask, fg_mask, instance_weights=None, 
                                fg_pixel_weight=1, bg_pixel_weight=recon_bg_pixel_weight)

            print(f"Rank {self.trainer.global_rank} Step {s}: {all_t[s].tolist()}, {loss_unet_distill.detach().item():.4f}")
            losses_unet_distill.append(loss_unet_distill)

        # If num_unet_denoising_steps > 1, most loss_unet_distill are usually 0.001~0.005, but sometimes there are a few large loss_unet_distill.
        # In order not to dilute the large loss_unet_distill, we don't divide by num_unet_denoising_steps.
        # Instead, only increase the normalizer sub-linearly.
        loss_unet_distill = sum(losses_unet_distill) / np.sqrt(num_unet_denoising_steps)

        v_loss_unet_distill = loss_unet_distill.mean().detach().item()
        if unet_distill_on_pure_noise:
            mon_loss_dict.update({f'{session_prefix}/unet_distill_on_noise': v_loss_unet_distill})
        else:
            mon_loss_dict.update({f'{session_prefix}/unet_distill_on_image': v_loss_unet_distill})

        return loss_unet_distill

    # Replace the first block of each activation data in ca_layers_activations_list to 
    # the re-denoised subject-single instance activations.
    def redenoise_subj_single(self, x_start0_ss, noises, ts, ca_layers_activations_list, 
                              ss_context, uncond_emb, sc_fg_face_bboxes, latent_shape, 
                              use_attn_lora, use_ffn_lora, crop_mix_weight=0.3):
        device = x_start0_ss.device
        # Crop the face area in the x_start0_ss and noises of the ss instance, 
        # and re-denoise it and replace the ss instance results.
        noises_ss = [ noise.chunk(4)[0] for noise in noises ]
        x_start0_ss_crop = []
        noises_ss_crop = [ [] for _ in range(len(noises)) ]
        # If BLOCK_SIZE == 1, then i only takes 0.
        for i in range(len(sc_fg_face_bboxes)):
            x1, y1, x2, y2 = sc_fg_face_bboxes[i]
            # x_start0_ss: [1, 4, 64, 64].
            ss_crop_i = x_start0_ss[i:i+1, :, y1:y2, x1:x2]
            # Crop the sc face area in the x_start0_ss and noises of the ss instance.
            # Resize them to the original size of (64, 64), then add to the original x_start0_ss / noises_ss.
            # IF BLOCK_SIZE > 1, then different sc_fg_face_bboxes may have different sizes. 
            # Therefore we have to crop and resize them separately.
            ss_crop_i = F.interpolate(ss_crop_i, latent_shape[-2:], mode='bilinear', align_corners=False)
            x_start0_ss_crop.append(ss_crop_i)
            for step, noise in enumerate(noises):
                noise_ss = noise[i:i+1, :, y1:y2, x1:x2]
                noise_ss = F.interpolate(noise_ss, latent_shape[-2:], mode='bilinear', align_corners=False)
                noises_ss_crop[step].append(noise_ss)

        x_start0_ss_crop = torch.cat(x_start0_ss_crop, dim=0)
        # Simply scaling up x_start0_ss_crop, noises_ss_crop to the original size will introduce
        # too many low frequency artifacts. Therefore we take a weighted average of the original and cropped-and-scaled ones.
        x_start0_ss_crop = x_start0_ss_crop * crop_mix_weight + x_start0_ss * (1 - crop_mix_weight)
        for step in range(len(noises_ss_crop)):
            noises_ss_crop[step] = torch.cat(noises_ss_crop[step], dim=0)
            noises_ss_crop[step] = noises_ss_crop[step] * crop_mix_weight + noises_ss[step] * (1 - crop_mix_weight)

        # ts_ss: only keep the first 1/4 (the SS block) of each t in the t sequence.
        ts_ss = [ t.chunk(4)[0] for t in ts ]

        # The input noises and ts are the randomly sampled noise and t sequences 
        # used to denoise the whole batch previously. So that the denoised results are aligned with
        # the SC instance.
        # ss_context: (prompt_emb_ss, prompt_in_ss, extra_info).
        # BLKS=1: only one block of the subject-single instance.
        noise_preds_ss, x_starts_ss, x_recons_ss, noises_ss_crop, ts_ss, ca_layers_activations_list_ss = \
            self.comp_distill_multistep_denoise(x_start0_ss_crop, noises_ss_crop, ts_ss, ss_context,
                                                uncond_emb=uncond_emb, 
                                                shrink_cross_attn=self.iter_flags['shrink_cross_attn'],
                                                mix_sc_mc_attn=self.iter_flags['mix_sc_mc_attn'],
                                                cfg_scale=2.5, num_denoising_steps=self.num_comp_distill_denoising_steps,
                                                use_attn_lora=use_attn_lora, use_ffn_lora=use_ffn_lora,
                                                ffn_lora_adapter_name='comp_distill', 
                                                BLKS=1)
        
        x_recon_ss2 = x_recons_ss[-1]
        x_recon_ss_pixels2 = self.decode_first_stage(x_recon_ss2)
        # The cropping operation is wrapped with torch.no_grad() in retinaface implementation.
        # So we don't need to wrap it here.
        ss_fg_face_crops2, ss_bg_face_crops_flat2, ss_fg_face_bboxes2, ss_face_detected_inst_mask2 = \
            self.arcface.retinaface.crop_faces(x_recon_ss_pixels2, out_size=(128, 128), T=20)

        # Only replace the ss instance results when all ss instances have faces detected.
        if (1 - ss_face_detected_inst_mask2).sum() == 0:
            # If there are no failed indices, then we get ss_fg_face_bboxes2.
            # NOTE: ss_fg_face_bboxes2 are coords on x_recon_ss_pixels2, 512*512.
            # In calc_elastic_matching_loss(), it is used to crop the latents, 64*64. 
            # So we scale ss_fg_face_bboxes2 down by 8.
            ss_fg_face_bboxes2 = pixel_bboxes_to_latent(ss_fg_face_bboxes2, x_recon_ss_pixels2.shape[-1], latent_shape[-1])
            # len(ss_fg_face_bboxes2) == BLOCK_SIZE, usually 1.
            for i in range(len(ss_fg_face_bboxes2)):
                print(f"Rank {self.trainer.global_rank} 2nd SS face coords {i}: {ss_fg_face_bboxes2[i].detach().cpu().numpy()}.", end=' ')

            for i, x_recon_ss2 in enumerate(x_recons_ss):
                recon_images_ss = self.decode_first_stage(x_recon_ss2)
                # log_image_colors: a list of 0-3, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
                # If there are multiple denoising steps, the output images are assigned different colors.
                log_image_colors = torch.ones(recon_images_ss.shape[0], dtype=int, device=device) * (i % 4)
                # Since the 1st round of recon images have been logged already,
                # we can only put the second round of SS recon images at the end of 
                # this batch of logged images.
                self.cache_and_log_generations(recon_images_ss, log_image_colors, do_normalize=True)

            for i, ca_layers_activations_ss in enumerate(ca_layers_activations_list_ss):
                ca_layers_activations = ca_layers_activations_list[i]
                ca_ss, ca_sc, ca_ss_rep, ca_mc = \
                    split_dict(ca_layers_activations, 4)
                # Replace the first round SS instance CA activations with 
                # the second round of SS instance CA activations.
                ca_layers_activations = collate_dicts([ca_layers_activations_ss, ca_sc, ca_ss_rep, ca_mc])
                ca_layers_activations_list[i] = ca_layers_activations

            return ss_fg_face_bboxes2
        # Otherwise, we keep the original activations and ss_fg_face_bboxes.
        else:
            return None
        
    # x_start0_ss: the latent of the first 1/4 batch of input images (actually only 1 image), 
    # ** without priming.
    # sc_fg_face_suppress_mask_shrink_ratio: 0.3.
    def calc_comp_feat_distill_loss(self, mon_loss_dict, session_prefix,
                                    x_start0_ss, x_start_primed_ss, x_recons, noise_preds, noises, 
                                    ts, ca_layers_activations_list, 
                                    all_subj_indices_1b, ss_context, uncond_emb, 
                                    prompt_emb_mask_4b, prompt_pad_mask_4b,
                                    BLOCK_SIZE, sc_fg_face_suppress_mask_shrink_ratio, 
                                    use_attn_lora, use_ffn_lora):
        losses_comp_fg_bg_preserve          = []
        losses_comp_rep_distill_subj_attn   = []
        losses_comp_rep_distill_subj_k      = []
        losses_comp_rep_distill_nonsubj_k   = []
        losses_comp_rep_distill_subj_v      = []
        losses_comp_rep_distill_nonsubj_v   = []
        subj_attn_cross_t_diffs    = []
        pred_l2s                            = []

        device = x_start0_ss.device
        dtype  = x_start0_ss.dtype
        latent_shape = x_start0_ss.shape

        sc_fg_mask, mc_fg_mask = None, None
        ss_fg_face_bboxes, sc_fg_face_bboxes = None, None
        all_ss_contain_faces = False
        first_step_when_sc_face_is_detected = -1
        # When sc_fg_mask_percent >= 0.1, we think the face has a chance to be too large and 
        # do subj_comp_rep_distill to discourage it.
        # 0.2 is borderline large, and 0.25 is too large.
        # 0.25 means when sc_fg_mask_percent >= 0.25, the loss scale is at the max value 1.
        rep_dist_fg_bounds      = (0.1, 0.20, 0.25)
        loss_comp_feat_distill  = torch.tensor(0., device=device, dtype=dtype)
        loss_fg_faces_suppress_comp = torch.tensor(0., device=device, dtype=dtype)
        loss_arcface_align_comp = torch.tensor(0., device=device, dtype=dtype)

        if self.arcface_align_loss_weight > 0:
            # ** The recon image in the last step is the clearest. Therefore,
            # we use the reconstructed images of the subject-single block in the last step
            # to detect the face area in the subject-single images. 
            x_recon_ss = x_recons[-1].chunk(4)[0]
            x_recon_ss_pixels = self.decode_first_stage(x_recon_ss)
            # The cropping operation is wrapped with torch.no_grad() in retinaface implementation.
            # So we don't need to wrap it here.
            ss_fg_face_crops, ss_bg_face_crops_flat, ss_fg_face_bboxes, ss_face_detected_inst_mask = \
                self.arcface.retinaface.crop_faces(x_recon_ss_pixels, out_size=(128, 128), T=20)

            # Only compute losses when all ss instances have faces detected.
            if (1 - ss_face_detected_inst_mask).sum() == 0:
                all_ss_contain_faces = True
                # If there are no failed indices, then we get ss_fg_face_bboxes.
                # NOTE: ss_fg_face_bboxes are coords on x_recon_ss_pixels, 512*512.
                # ss cropping is done on the latents, 64*64. So we scale ss_fg_face_bboxes down by 8.
                ss_fg_face_bboxes = pixel_bboxes_to_latent(ss_fg_face_bboxes, x_recon_ss_pixels.shape[-1], latent_shape[-1])
                for i in range(len(ss_fg_face_bboxes)):
                    print(f"Rank {self.trainer.global_rank} 1st SS face coords {i}: {ss_fg_face_bboxes[i].detach().cpu().numpy()}.", end=' ')

                # If a face cannot be detected in the subject-single instance, then it probably
                # won't be detected in the subject-compositional instance either.
                # comp_sc_face_align_loss_thres: 0.7
                loss_arcface_align_comp, loss_fg_faces_suppress_comp, loss_bg_faces_suppress_comp, \
                loss_comp_sc_subj_mb_suppress, sc_fg_mask, sc_fg_face_bboxes, first_step_when_sc_face_is_detected = \
                    self.calc_comp_face_align_and_mb_suppress_losses(mon_loss_dict, session_prefix, x_start0_ss, x_recons, 
                                                                     ca_layers_activations_list,
                                                                     all_subj_indices_1b, 
                                                                     # fg_faces_grad_mask_ratios = (0.9, 0.3)
                                                                     fg_faces_grad_mask_ratios=(0.9, sc_fg_face_suppress_mask_shrink_ratio), 
                                                                     BLOCK_SIZE=BLOCK_SIZE,
                                                                     comp_sc_face_align_loss_kept_frac=self.comp_sc_face_align_loss_kept_frac,
                                                                     comp_sc_face_align_loss_thres=self.comp_sc_face_align_loss_thres)

                # loss_bg_faces_suppress_comp is a mean L2 loss, only ~0.02. * 400 * 0.01 => 0.08.
                # Although this is ~10x of loss_arcface_align_comp, it's very infraquently triggered.
                comp_bg_faces_suppress_loss_scale = 400
                loss_comp_feat_distill += loss_bg_faces_suppress_comp * comp_bg_faces_suppress_loss_scale * self.arcface_align_loss_weight
                # loss_comp_sc_subj_mb_suppress: ~0.2, comp_sc_subj_mb_suppress_loss_weight: 0.2 => 0.04.
                # loss_comp_feat_distill: 0.07, 60% of comp distillation loss.
                loss_comp_feat_distill += loss_comp_sc_subj_mb_suppress * self.comp_sc_subj_mb_suppress_loss_weight

                if loss_bg_faces_suppress_comp > 0:
                    self.comp_iters_bg_has_face_count += 1
                    comp_iters_bg_has_face_frac = self.comp_iters_bg_has_face_count / self.comp_iters_count
                    mon_loss_dict.update({f'{session_prefix}/comp_iters_bg_has_face_frac': comp_iters_bg_has_face_frac})

            mc_x_recon = x_recons[-1].chunk(4)[3]
            mc_x_recon_pixels = self.decode_first_stage(mc_x_recon)
            # The cropping operation is wrapped with torch.no_grad() in retinaface implementation.
            # So we don't need to wrap it here.
            mc_fg_face_crops, mc_bg_face_crops_flat, mc_fg_face_bboxes, mc_face_detected_inst_mask = \
                self.arcface.retinaface.crop_faces(mc_x_recon_pixels, out_size=(128, 128), T=20)
            
            if (1 - mc_face_detected_inst_mask).sum() == 0:
                mc_fg_face_bboxes = pixel_bboxes_to_latent(mc_fg_face_bboxes, mc_x_recon_pixels.shape[-1], latent_shape[-1])
                mc_fg_mask = torch.zeros(BLOCK_SIZE, 1, latent_shape[-2], latent_shape[-1], device=device)
                # mc_fg_mask is zero-initialized.
                # len(mc_fg_face_bboxes) == BLOCK_SIZE == len(mc_fg_mask), usually 1.
                for i in range(len(mc_fg_face_bboxes)):
                    x1, y1, x2, y2 = mc_fg_face_bboxes[i]
                    mc_fg_mask[i, :, y1:y2, x1:x2] = 1
                    print(f"Rank {self.trainer.global_rank} MC face coords {i}: {mc_fg_face_bboxes[i].detach().cpu().numpy()}.", end=' ')

        monitor_loss_names = \
            [ 'loss_sc_recon_ssfg_attn_agg', 'loss_sc_recon_ssfg_flow', 'loss_sc_recon_ssfg_sameloc', 'loss_sc_recon_ssfg_min', 
              'loss_sc_recon_mc_attn_agg',   'loss_sc_recon_mc_flow',   'loss_sc_recon_mc_sameloc',   'loss_sc_recon_mc_min',
              'loss_sc_to_ssfg_sparse_attns_distill', 'loss_sc_to_mc_sparse_attns_distill',
              'ssfg_sameloc_win_rate', 'ssfg_flow_win_rate', 'mc_flow_win_rate', 'mc_sameloc_win_rate',
              'ssfg_avg_sparse_distill_weight', 'mc_avg_sparse_distill_weight', 
              'sc_bg_percent', 'discarded_loss_ratio' ]
        
        for loss_name in monitor_loss_names:
            loss_name2 = f'{session_prefix}/{loss_name}'
            mon_loss_dict[loss_name2] = 0

        # If no face is detected in either the SC or the MC instance, then both sc_fg_mask and mc_fg_mask are None.
        if sc_fg_mask is not None:
            sc_fg_mask_percent = sc_fg_mask.float().mean().item()
            mon_loss_dict.update({f'{session_prefix}/sc_fg_mask_percent': sc_fg_mask_percent })
        else:
            sc_fg_mask_percent = 0
        
        # If no face is detected in either the SC or the MC instance, then both sc_fg_mask and mc_fg_mask are None.
        if mc_fg_mask is not None:
            mc_fg_mask_percent = mc_fg_mask.float().mean().item()
            mon_loss_dict.update({f'{session_prefix}/mc_fg_mask_percent': mc_fg_mask_percent })
            # This iteration is counted as face detected, as long as the face is detected in one step.
            comp_mc_face_detected_frac = self.comp_mc_face_detected_frac.update(1)
        else:
            mc_fg_mask_percent = 0
            comp_mc_face_detected_frac = self.comp_mc_face_detected_frac.update(0)

        # Even if the face is not detected in this step, we still update comp_mc_face_detected_frac,
        # because it's an accumulated value that's inherently continuous.
        mon_loss_dict.update({f'{session_prefix}/comp_mc_face_detected_frac': comp_mc_face_detected_frac})
        mon_loss_dict.update({f'{session_prefix}/comp_sc_face_align_loss_kept_frac': self.comp_sc_face_align_loss_kept_frac.mean})

        if sc_fg_mask_percent == 0:
            sc_face_proportion_type = 'sc-noface'
        # If mc doesn't contain a face, and sc contains a small face, it's allowed.
        # If mc doesn't contain a face, but sc contains a sufficiently large face,
        # it is not desired.
        # comp_sc_fg_mask_percent_range[1] = 0.36, so sc_fg_mask_percent >= 0.16 * 0.36 = 0.0576, meaning
        # each edge of the face is at most 0.24.
        elif mc_fg_mask_percent == 0 and sc_fg_mask_percent >= 0.16 * self.comp_sc_fg_mask_percent_range[1]:
            sc_face_proportion_type = 'mc-no-sc-large'
        # When sc fg and mc fg masks have no overlap or very little overlap, 
        # we suppress the face in the sc instance. 1/6.25 = 0.16.
        # In this branch, sc_fg_mask_percent > 0 always holds. So no need to check it.
        elif mc_fg_mask_percent > 0 and ((sc_fg_mask * mc_fg_mask).sum() / sc_fg_mask.sum()) < 0.16:
            sc_face_proportion_type = 'little-no-overlap'
        # Usually sc_fg_mask_percent is around 0.05~0.25.
        # comp_sc_fg_mask_percent_range: 0.0225~0.36, each dim [0.15, 0.6].
        # If sc_fg_mask_percent >= 0.36, it means the image is dominated by the face (each edge >= 0.6), 
        # then we don't need to preserve the background as it will be highly 
        # challenging to match the background with the mc instance.
        # If sc_fg_mask_percent <= 0.0225, it means the face is too small (each edge <= 0.15 = 77 pixels).
        elif sc_fg_mask_percent <= self.comp_sc_fg_mask_percent_range[0]:
            sc_face_proportion_type = 'too-small'
        # If a face is detected in the mc instance, then the face in the sc instance is at most 6.25x of its size
        # (2.5x at each dimension),
        # Otherwise, the face in the sc instance is at most 1/4 of the max allowed size = 0.25 * 0.36 = 0.09,
        # i.e., each edge of the face is at most 0.3.
        elif (sc_fg_mask_percent >= self.comp_sc_fg_mask_percent_range[1]) \
          or (mc_fg_mask_percent > 0 and sc_fg_mask_percent >= 6.25 * mc_fg_mask_percent):
            # Skip calc_comp_subj_bg_preserve_loss() before sc_face is detected.
            sc_face_proportion_type = 'too-large'
        else:
            sc_face_proportion_type = 'good'

        print(f"Rank {self.trainer.global_rank}-{self.global_step} sc_face_proportion_type: {sc_face_proportion_type}")

        # loss_arcface_align_comp: 0.5-0.8. arcface_align_loss_weight * scale: 0.01 * 10 => 0.05-0.08.
        # This loss is around 1/15 of comp distill losses (0.1).
        # NOTE: if arcface_align_loss_weight is too large (e.g., 0.05), then it will introduce a lot of artifacts to the 
        # whole image, not just the face area. So we need to keep it small.
        # If comp_sc_face_detected_frac becomes smaller over time, 
        # then gradually increase the weight of loss_arcface_align_comp through arcface_align_comp_loss_scale.
        # If comp_sc_face_detected_frac=0.9, then arcface_align_comp_loss_scale= 1.25*3 = 3.75.
        # If comp_sc_face_detected_frac=0.5, then arcface_align_comp_loss_scale=12 (maximum).
        if loss_arcface_align_comp > 0 and sc_face_proportion_type in ['too-small', 'good']:
            # This iteration is counted as face detected, as long as the face is detected in one step.
            comp_sc_face_detected_frac = self.comp_sc_face_detected_frac.update(1)
            arcface_align_comp_loss_scale = 3 * min(4, 1 / (comp_sc_face_detected_frac**2 + 0.01))
            loss_comp_feat_distill += loss_arcface_align_comp * arcface_align_comp_loss_scale * self.arcface_align_loss_weight
        else:
            comp_sc_face_detected_frac = self.comp_sc_face_detected_frac.update(0)

        # Even if the face is not detected in this step, we still update comp_sc_face_detected_frac,
        # because it's an accumulated value that's inherently continuous.
        mon_loss_dict.update({f'{session_prefix}/comp_sc_face_detected_frac': comp_sc_face_detected_frac})
        if sc_face_proportion_type not in ['sc-noface']:
            # Update ca_layers_activations_list in-place.
            # If a face is detected in the ss instance of the last denoising step, then 
            # the new ss_fg_face_bboxes is returned, and we replace ss_fg_face_bboxes with it.
            ss_fg_face_bboxes2 = \
                self.redenoise_subj_single(x_start_primed_ss, noises, ts, ca_layers_activations_list,
                                           ss_context, uncond_emb, sc_fg_face_bboxes, latent_shape,
                                           use_attn_lora=use_attn_lora, use_ffn_lora=use_ffn_lora, 
                                           crop_mix_weight=self.redenoise_subj_single_crop_mix_weight)
            if ss_fg_face_bboxes2 is not None:
                ss_fg_face_bboxes = ss_fg_face_bboxes2
                ss_redenoised = True
            else:
                # Otherwise, the redenoising failed, i.e., no face is detected in the new ss instance.
                ss_redenoised = False
        else:
            # Otherwise, no face is detected in the sc instance. 
            # So we keep the original SS activations and ss_fg_face_bboxes.
            ss_redenoised = False

        if sc_face_proportion_type in ['mc-no-sc-large', 'little-no-overlap', 'too-large']:
            do_sc_fg_faces_suppress = True
            # If sc_face_proportion_type is 'mc-no-sc-large', 'little-no-overlap' or 'too-large', then 
            # ** we optimize both the arcface align loss and the face suppression loss, to drive the face
            # into the center of the face area, and keep the face identity at the same time.
            comp_fg_faces_suppress_loss_scale_dict = \
                    { 'mc-no-sc-large': 10, 'little-no-overlap': 5, 'too-large': 5 }
            # Suppress the face in the sc instance, which is at the "background" of the mc instance.
            comp_fg_faces_suppress_loss_scale = comp_fg_faces_suppress_loss_scale_dict[sc_face_proportion_type]
            comp_sc_face_suppressed_frac = self.comp_sc_face_suppressed_frac.update(1)
            # comp_sc_face_suppressed_frac: 0.2~0.5.
            # If comp_sc_face_suppressed_frac=0.5, then extra_suppress_loss_scale = 15.625.
            # If comp_sc_face_suppressed_frac=0.2, then extra_suppress_loss_scale = 1.
            # loss_fg_faces_suppress_comp: 0.03 -> 0.03 * 10 * 15 * 0.01 = 0.045.
            extra_suppress_loss_scale = 15
            loss_comp_feat_distill += loss_fg_faces_suppress_comp * comp_fg_faces_suppress_loss_scale \
                                      * extra_suppress_loss_scale * self.arcface_align_loss_weight
            sc_face_shrink_ratio_for_bg_matching_mask = sc_fg_face_suppress_mask_shrink_ratio  # 0.3
        else:
            do_sc_fg_faces_suppress = False
            # If sc_face_proportion_type is 'sc-noface', 'too-small' or 'good', then
            # ** we don't optimize the face suppression loss.
            comp_sc_face_suppressed_frac = self.comp_sc_face_suppressed_frac.update(0)
            sc_face_shrink_ratio_for_bg_matching_mask = 1

        mon_loss_dict.update({f'{session_prefix}/comp_sc_face_suppressed_frac': comp_sc_face_suppressed_frac})

        for step_idx, ca_layers_activations in enumerate(ca_layers_activations_list):
            # Calc the L2 norm of noise_pred.
            pred_l2_step = (noise_preds[step_idx] ** 2).mean()   
            pred_l2s.append(pred_l2_step)
        
            loss_comp_rep_distill_subj_attn, loss_comp_rep_distill_subj_k, loss_comp_rep_distill_nonsubj_k, \
            loss_comp_rep_distill_subj_v, loss_comp_rep_distill_nonsubj_v = \
                calc_sc_rep_attn_distill_loss(ca_layers_activations, all_subj_indices_1b, 
                                              prompt_emb_mask_4b,    prompt_pad_mask_4b,
                                              sc_fg_mask_percent,    FG_THRES=rep_dist_fg_bounds[0])
            
            if loss_comp_rep_distill_subj_attn == 0:
                loss_comp_rep_distill_subj_attn, loss_comp_rep_distill_subj_k, loss_comp_rep_distill_nonsubj_k, \
                loss_comp_rep_distill_subj_v, loss_comp_rep_distill_nonsubj_v = \
                    [ torch.tensor(0., device=device, dtype=dtype) ] * 5

            losses_comp_rep_distill_subj_attn.append(loss_comp_rep_distill_subj_attn)
            losses_comp_rep_distill_subj_k.append(loss_comp_rep_distill_subj_k)
            losses_comp_rep_distill_nonsubj_k.append(loss_comp_rep_distill_nonsubj_k)
            losses_comp_rep_distill_subj_v.append(loss_comp_rep_distill_subj_v)
            losses_comp_rep_distill_nonsubj_v.append(loss_comp_rep_distill_nonsubj_v)

            # NOTE: Skip computing loss_comp_fg_bg_preserve before sc_face is detected.
            # If first_step_when_sc_face_is_detected == -1, then in all steps, sc_face is not detected.
            # So we skip all the steps.
            # We also skip loss_comp_fg_bg_preserve on all steps if not all_ss_contain_faces, 
            # i.e., faces are not detected in some subject-single instances. But this should be rare.
            if (not all_ss_contain_faces) or (first_step_when_sc_face_is_detected == -1):
                continue

            if (step_idx < len(ca_layers_activations_list) - 1) and (step_idx >= first_step_when_sc_face_is_detected - 1):
                subj_attn_cross_t_diff_step = \
                    calc_subj_attn_cross_t_diff_loss(ca_layers_activations, ca_layers_activations_list[step_idx+1],
                                                     all_subj_indices_1b)
                subj_attn_cross_t_diffs.append(subj_attn_cross_t_diff_step)
            
            if (step_idx < first_step_when_sc_face_is_detected) or (sc_face_proportion_type in ['sc-noface']):
                continue

            loss_comp_fg_bg_preserve_step = \
                calc_comp_subj_bg_preserve_loss(mon_loss_dict, session_prefix, device,
                                                self.flow_model, ca_layers_activations, 
                                                ss_fg_face_bboxes, sc_fg_face_bboxes,
                                                sc_face_shrink_ratio_for_bg_matching_mask=sc_face_shrink_ratio_for_bg_matching_mask,
                                                recon_scaled_loss_threses={'mc': 0.4, 'ssfg': 0.4},
                                                recon_max_scale_of_threses=5,
                                                do_sc_fg_faces_suppress=do_sc_fg_faces_suppress
                                               )
            losses_comp_fg_bg_preserve.append(loss_comp_fg_bg_preserve_step)

            # ca_layers_activations['outfeat'] is a dict as: layer_idx -> ca_outfeat. 
            # It contains the 3 specified cross-attention layers of UNet. i.e., layers 22, 23, 24.
            # Similar are ca_attns and ca_attns, each ca_outfeats in ca_outfeats is already 4D like [4, 8, 64, 64].

        comp_fg_bg_preserve_loss_count = len(losses_comp_fg_bg_preserve) + 1e-6
        for loss_name in monitor_loss_names:
            loss_name2 = f'{session_prefix}/{loss_name}'
            if loss_name2 in mon_loss_dict:
                if mon_loss_dict[loss_name2] > 0:
                    mon_loss_dict[loss_name2] = mon_loss_dict[loss_name2] / comp_fg_bg_preserve_loss_count
                    loss_name3 = loss_name2.replace('loss_', '')
                    # Rename the losses to more concise names by removing the 'loss_' prefix.
                    mon_loss_dict[loss_name3] = mon_loss_dict.pop(loss_name2)
                else:
                    # Remove 0 losses from the mon_loss_dict.
                    del mon_loss_dict[loss_name2]

        # These 4 losses_* are always non-empty.
        loss_comp_rep_distill_subj_attn  = torch.stack(losses_comp_rep_distill_subj_attn).mean()
        loss_comp_rep_distill_subj_k     = torch.stack(losses_comp_rep_distill_subj_k).mean()
        loss_comp_rep_distill_nonsubj_k  = torch.stack(losses_comp_rep_distill_nonsubj_k).mean()
        loss_comp_rep_distill_subj_v     = torch.stack(losses_comp_rep_distill_subj_v).mean()
        loss_comp_rep_distill_nonsubj_v  = torch.stack(losses_comp_rep_distill_nonsubj_v).mean()

        # Chance is we may have skipped all the steps for computing 
        # loss_comp_fg_bg_preserve. Therefore we need to check if 
        # losses_comp_fg_bg_preserve is empty.
        if len(losses_comp_fg_bg_preserve) > 0:
            loss_comp_fg_bg_preserve = torch.stack(losses_comp_fg_bg_preserve).mean()
            mon_loss_dict.update({f'{session_prefix}/comp_fg_bg_preserve': loss_comp_fg_bg_preserve.mean().detach().item() })
            # loss_comp_fg_bg_preserve: 0.06~0.08.
            # loss_sc_recon_ssfg_min and loss_sc_recon_mc_min is absorbed into loss_comp_fg_bg_preserve.
            loss_comp_feat_distill += loss_comp_fg_bg_preserve

        if len(subj_attn_cross_t_diffs) > 0:
            subj_attn_cross_t_diff = torch.stack(subj_attn_cross_t_diffs).mean()
            mon_loss_dict.update({f'{session_prefix}/subj_attn_cross_t_diff': subj_attn_cross_t_diff.mean().detach().item() })
            # subj_attn_cross_t_diff: 1e-5~2e-5 * 0 => DISABLED.
            # loss_comp_feat_distill += subj_attn_cross_t_diff * self.comp_sc_subj_attn_cross_t_diff_loss_weight

        if loss_comp_rep_distill_subj_attn > 0:
            mon_loss_dict.update({f'{session_prefix}/comp_rep_distill_subj_attn':  loss_comp_rep_distill_subj_attn.detach().item() })
            mon_loss_dict.update({f'{session_prefix}/comp_rep_distill_subj_k':     loss_comp_rep_distill_subj_k.detach().item() })
            mon_loss_dict.update({f'{session_prefix}/comp_rep_distill_nonsubj_k':  loss_comp_rep_distill_nonsubj_k.detach().item() })
            mon_loss_dict.update({f'{session_prefix}/comp_rep_distill_subj_v':     loss_comp_rep_distill_subj_v.detach().item() })
            mon_loss_dict.update({f'{session_prefix}/comp_rep_distill_nonsubj_v':  loss_comp_rep_distill_nonsubj_v.detach().item() })
            # If sc_fg_mask_percent == 0.2, then fg_percent_rep_distill_scale = 0.5.
            # If sc_fg_mask_percent >= 0.25, then fg_percent_rep_distill_scale = 2.
            # valid_scale_range=(0.05, 1): If sc_fg_mask_percent = 0.1, then fg_percent_rep_distill_scale = 0.05.
            if sc_fg_mask_percent > 0:
                fg_percent_rep_distill_scale = \
                    calc_dyn_loss_scale(sc_fg_mask_percent, (rep_dist_fg_bounds[1], 0.5), (rep_dist_fg_bounds[2], 2), 
                                        valid_scale_range=(0.05, 2))
            else:
                # sc_fg_mask_percent == 0 means no face is detected in the subject-comp instance.
                # In this case, we don't do distillation on the subject-comp-rep instance.
                fg_percent_rep_distill_scale = 0

            # loss_comp_rep_distill_nonsubj_k: 0.003.
            comp_rep_distill_nonsubj_k_loss_scale = 5
            # Weaker regularization on the non-subj v alignment.
            # loss_comp_rep_distill_nonsubj_v: 0.001.
            comp_rep_distill_nonsubj_v_loss_scale = 2
            # If do_comp_feat_distill is less frequent, then increase the weight of loss_subj_comp_rep_distill_*.
            subj_comp_rep_distill_loss_scale = fg_percent_rep_distill_scale

            # The right side term should be < 0.01.
            loss_comp_feat_distill += (loss_comp_rep_distill_subj_attn + loss_comp_rep_distill_subj_k + loss_comp_rep_distill_subj_v + \
                                       loss_comp_rep_distill_nonsubj_k * comp_rep_distill_nonsubj_k_loss_scale + \
                                       loss_comp_rep_distill_nonsubj_v * comp_rep_distill_nonsubj_v_loss_scale) \
                                      * subj_comp_rep_distill_loss_scale
            
        v_loss_comp_feat_distill = loss_comp_feat_distill.mean().detach().item()
        if v_loss_comp_feat_distill > 0:
            mon_loss_dict.update({f'{session_prefix}/comp_feat_distill_total': v_loss_comp_feat_distill})

        pred_l2 = torch.stack(pred_l2s).mean()
        # pred_l2: 0.92~0.99. But we don't optimize it; instead, it's just for monitoring.
        mon_loss_dict.update({f'{session_prefix}/pred_l2': pred_l2.mean().detach().item()})

        return loss_comp_feat_distill            

    def calc_comp_face_align_and_mb_suppress_losses(self, mon_loss_dict, session_prefix, 
                                                    x_start0_ss, x_recons, ca_layers_activations_list,
                                                    all_subj_indices_1b, fg_faces_grad_mask_ratios,
                                                    BLOCK_SIZE, comp_sc_face_align_loss_kept_frac,
                                                    comp_sc_face_align_loss_thres):
        # We cannot afford calculating loss_arcface_align_comp for > 1 steps. Otherwise, OOM.
        max_arcface_align_loss_count        = 3
        arcface_align_loss_count            = 0
        arcface_align_loss_stat_count       = 0
        comp_sc_subj_mb_suppress_loss_count = 0
        fg_faces_suppress_loss_count, bg_faces_suppress_loss_count = 0, 0
        sc_fg_mask, sc_fg_face_bboxes       = None, None
        first_step_when_sc_face_is_detected = -1

        # Don't initialize zero_losses as torch.zeros(3), otherwise the losses cannot do inplace addition.
        zero_losses = [ torch.tensor(0., device=x_start0_ss.device, dtype=x_start0_ss.dtype) for _ in range(5) ]
        loss_comp_sc_subj_mb_suppress = zero_losses[0]
        loss_arcface_align_comp       = zero_losses[1]
        loss_arcface_align_comp_stat  = zero_losses[2]
        loss_fg_faces_suppress_comp   = zero_losses[3]
        loss_bg_faces_suppress_comp   = zero_losses[4]

        if self.arcface_align_loss_weight > 0:
            # Trying to calc arcface_align_loss from easy to difficult steps.
            # sel_step: 2~0. 0 is the hardest for face detection (denoised once), and 2 is the easiest (denoised 3 times).
            # Reverse the steps, so we start from the clearest face image, and get the most accurate sc_fg_mask.
            for sel_step in range(len(x_recons) - 1, -1, -1):
                x_recon = x_recons[sel_step]
                # We have checked that x_start0_ss is always a valid face image.
                # Align subj_comp_recon to x_start0_ss.
                # Only optimize subj comp instances w.r.t. arcface_align_loss. 
                # Since the subj single instances were generated without gradient, 
                # we cannot optimize them.
                subj_comp_recon  = x_recon.chunk(4)[1]
                # x_start0_ss and subj_comp_recon are latent images, [1, 4, 64, 64]. 
                # They need to be decoded first.
                # If no faces are detected in x_recon, loss_arcface_align_comp_step is 0, 
                # and sc_fg_face_bboxes is None.
                # NOTE: In the first iteration, sc_fg_mask is None, 
                # and we'll generate it based on sc_fg_face_bboxes.
                
                # Only do arcface_align_loss for the first iteration (i.e., the last denoising iteration).
                if arcface_align_loss_count < max_arcface_align_loss_count:
                    # Since the sc block only contains 1 instance, if loss_arcface_align_comp_step > 0, that means a face is detected.
                    # So we don't need to check sc_fg_face_detected_inst_mask since it's always [1].
                    # sc_fg_face_detected_inst_mask: binary tensor of [BS].
                    # fg_faces_grad_mask_ratios: (0.9, 0.3) means:
                    # For loss_arcface_align_comp,     we encourage the central 90% of the face area, 
                    # For loss_fg_faces_suppress_comp, we suppress  the border  70% of the face area.
                    loss_arcface_align_comp_step, loss_fg_faces_suppress_comp_step, loss_bg_faces_suppress_comp_step, \
                    sc_fg_face_bboxes_, sc_fg_face_detected_inst_mask = \
                        self.calc_arcface_align_loss(x_start0_ss, subj_comp_recon, fg_faces_grad_mask_ratios)
                    # Found valid face images. Stop trying, since we cannot afford calculating loss_arcface_align_comp for > 1 steps.
                    if loss_arcface_align_comp_step > 0:
                        print(f"Rank-{self.trainer.global_rank} arcface_align_comp step {sel_step+1}/{len(x_recons)}")
                        # If comp_sc_face_align_loss_thres is not specified (-1), or the loss is smaller than the threshold,
                        # then we optimize the loss_arcface_align_comp_step.
                        # comp_sc_face_align_loss_thres: 0.7.
                        if (comp_sc_face_align_loss_thres <= 0) or (loss_arcface_align_comp_step <= comp_sc_face_align_loss_thres):
                            loss_arcface_align_comp += loss_arcface_align_comp_step
                            arcface_align_loss_count += 1
                            loss_arcface_align_comp_stat += loss_arcface_align_comp_step
                            arcface_align_loss_stat_count += 1
                            comp_sc_face_align_loss_kept_frac.update(1)
                        else:
                            # Skip the loss_arcface_align_comp_step if it's too big and the gradient is too noisy.
                            # Only calculate the unoptimized statistics.
                            loss_arcface_align_comp_stat += loss_arcface_align_comp_step
                            arcface_align_loss_stat_count += 1
                            comp_sc_face_align_loss_kept_frac.update(0)

                        ca_layers_activations = ca_layers_activations_list[sel_step]
                        if first_step_when_sc_face_is_detected == -1:
                            first_step_when_sc_face_is_detected = sel_step

                        # Generate sc_fg_mask for the first time, based on the detected face area.
                        if sc_fg_mask is None:
                            sc_fg_face_bboxes = sc_fg_face_bboxes_
                            # sc_fg_mask: [1, 1, 64, 64].
                            sc_fg_mask = torch.zeros_like(subj_comp_recon[:, :1])
                            # When loss_arcface_align_comp > 0, sc_fg_face_bboxes is always not None.
                            # sc_fg_face_bboxes: [[22, 15, 36, 33]], already scaled down to 64*64.
                            for i in range(len(sc_fg_face_bboxes)):
                                x1, y1, x2, y2 = sc_fg_face_bboxes[i]
                                sc_fg_mask[i, :, y1:y2, x1:x2] = 1

                        if loss_fg_faces_suppress_comp_step > 0:
                            loss_fg_faces_suppress_comp += loss_fg_faces_suppress_comp_step
                            fg_faces_suppress_loss_count += 1
                        if loss_bg_faces_suppress_comp_step > 0:
                            loss_bg_faces_suppress_comp += loss_bg_faces_suppress_comp_step
                            bg_faces_suppress_loss_count += 1
                
                # Always calculate loss_comp_sc_subj_mb_suppress, as long as sc_fg_mask is available,
                # since it's fast to compute and doesn't require much memory.
                # NOTE: loss_comp_sc_subj_mb_suppress will help mitigate "attention deficits" 
                # in early denoising steps.
                if sc_fg_mask is not None:
                    # ca_layers_activations['attn']: { 22 -> [4, 8, 4096, 77], 23 -> [4, 8, 4096, 77], 24 -> [4, 8, 4096, 77] }.
                    # sc_attn_dict: { 22 -> [1, 8, 64, 64], 23 -> [1, 8, 64, 64], 24 -> [1, 8, 64, 64] }.
                    sc_attn_dict = { layer_idx: attn.chunk(4)[1] for layer_idx, attn in ca_layers_activations['attn'].items() }
                    # Suppress the activation of the subject embeddings at the background area, to reduce double-face artifacts.
                    loss_comp_sc_subj_mb_suppress_step = \
                        calc_subj_masked_bg_suppress_loss(sc_attn_dict, all_subj_indices_1b, BLOCK_SIZE, sc_fg_mask)
                    loss_comp_sc_subj_mb_suppress += loss_comp_sc_subj_mb_suppress_step
                    comp_sc_subj_mb_suppress_loss_count += 1

            if arcface_align_loss_count > 0:
                loss_arcface_align_comp = loss_arcface_align_comp / arcface_align_loss_count
                mon_loss_dict.update({f'{session_prefix}/arcface_align_comp_opt': loss_arcface_align_comp.mean().detach().item() })

            if arcface_align_loss_stat_count > 0:
                loss_arcface_align_comp_stat = loss_arcface_align_comp_stat / arcface_align_loss_stat_count
                mon_loss_dict.update({f'{session_prefix}/arcface_align_comp': loss_arcface_align_comp_stat.mean().detach().item() })
                
            if comp_sc_subj_mb_suppress_loss_count > 0:
                loss_comp_sc_subj_mb_suppress = loss_comp_sc_subj_mb_suppress / comp_sc_subj_mb_suppress_loss_count
                mon_loss_dict.update({f'{session_prefix}/comp_sc_subj_mb_suppress': loss_comp_sc_subj_mb_suppress.mean().detach().item() })

            if fg_faces_suppress_loss_count > 0:
                loss_fg_faces_suppress_comp = loss_fg_faces_suppress_comp / fg_faces_suppress_loss_count
                mon_loss_dict.update({f'{session_prefix}/comp_fg_faces_suppress': loss_fg_faces_suppress_comp.mean().detach().item() })

            if bg_faces_suppress_loss_count > 0:
                loss_bg_faces_suppress_comp = loss_bg_faces_suppress_comp / bg_faces_suppress_loss_count
                mon_loss_dict.update({f'{session_prefix}/comp_bg_faces_suppress': loss_bg_faces_suppress_comp.mean().detach().item() })

        return loss_arcface_align_comp, loss_fg_faces_suppress_comp, loss_bg_faces_suppress_comp, \
               loss_comp_sc_subj_mb_suppress, sc_fg_mask, sc_fg_face_bboxes, first_step_when_sc_face_is_detected
    
    def log_attention(self, ca_layers_activations, all_subj_indices_1b):
        # attn: ['23': [4, 8, 4096, 97], '24': [4, 8, 4096, 97]]. 
        # 97: CLIP text embeddings extended from 77 to 97.
        attn = ca_layers_activations['attn']
        heatmaps = []
        for layer_idx in attn.keys():
            # subj_attn: [8, 4096, 20] -> [8, 4096] -> [4096]
            subj_attn_1d = attn[layer_idx][1, :, :, all_subj_indices_1b[1]].sum(dim=-1).mean(dim=0)
            subj_attn_2d = subj_attn_1d.view(64, 64)
            # Normalize the attention weights to [0, 1].
            subj_attn_2d = subj_attn_2d - subj_attn_2d.min()
            subj_attn_2d = subj_attn_2d / subj_attn_2d.max()

            # [64, 64] -> [512, 512].
            subj_attn_2d = F.interpolate(subj_attn_2d.reshape(1, 1, *subj_attn_2d.shape), size=(512, 512), mode='bilinear').squeeze()
            subj_attn_2d_np = np.uint8(subj_attn_2d.detach().cpu().numpy() * 255)
            # heatmap: [512, 512, 3]. Note to convert BGR to RGB.
            heatmap = cv2.applyColorMap(subj_attn_2d_np, cv2.COLORMAP_JET)[..., ::-1].copy()
            # heatmap: [512, 512, 3] -> [3, 512, 512].
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).to(subj_attn_1d.device)
            heatmaps.append(heatmap)

        # heatmaps: [2, 3, 512, 512].
        heatmaps = torch.stack(heatmaps, dim=0)
        avg_heatmap = heatmaps.float().mean(dim=0, keepdim=True).to(heatmaps.dtype)
        if self.log_attn_level == 2:
            heatmaps = torch.cat([heatmaps, avg_heatmap], dim=0)
        else:
            # If log_attn_level == 1, only log the average heatmap.
            heatmaps = avg_heatmap
        self.cache_and_log_generations(heatmaps, None, do_normalize=False)

    # samples: a single 4D [B, C, H, W] np array, or a single 4D [B, C, H, W] torch tensor, 
    # or a list of 3D [C, H, W] torch tensors.
    # Data type of samples could be uint (0-25), or float (-1, 1) or (0, 1).
    # If (-1, 1), then we should set do_normalize=True.
    # img_colors: a single 1D torch tensor, indexing colors = [ None, 'green', 'red', 'purple', 'orange', 'blue', 'pink', 'magenta' ]
    # For raw output from raw output from SD decode_first_stage(),
    # samples are be between [-1, 1], so we set do_normalize=True, which will convert and clamp to [0, 1].
    @rank_zero_only
    def cache_and_log_generations(self, samples, img_colors, do_normalize=True):
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
            img_colors = torch.zeros(samples.size(0), dtype=torch.int, device=samples.device)

        self.sample_image_queue.put((samples.cpu(), img_colors.cpu()))

    def save_samples_worker(self, max_cache_size=60):
        # max_cache_size = 60, nrow=12, so we save a 4*12 grid of samples.
        # Each sample is 512*512*3, so the grid is 512*512*3*4*12*4 bytes = 37.7M * 4 = 150M.
        """Worker that saves images only when pending_samples has at least max_cache_size items"""
        pending_samples = []
        pending_sample_colors = []
        pending_sample_count = 0

        while True:
            try:
                # samples:    a (B, C, H, W) tensor.
                # img_colors: a tensor of (B,) ints.
                # samples should be between [0, 255] (uint8).
                samples, img_colors = self.sample_image_queue.get(timeout=1)
                # Not enough items yet, mark this one as done and continue waiting
                self.sample_image_queue.task_done()                
                pending_samples.append(samples)
                pending_sample_colors.append(img_colors)
                pending_sample_count += len(samples)
            except queue.Empty:
                continue

            if pending_sample_count >= max_cache_size:
                grid_folder = os.path.join(self.logger._save_dir, 'samples')
                os.makedirs(grid_folder, exist_ok=True)
                grid_filename = os.path.join(grid_folder, f'{self.cache_start_iter:04d}-{self.global_step:04d}.png')
                pending_images     = torch.cat(pending_samples,       0)
                pending_img_colors = torch.cat(pending_sample_colors, 0)
                save_grid(pending_images, pending_img_colors, grid_filename, 12)   
                
                # Clear the cache. If num_cached_generations > max_cache_size,
                # some samples at the end of the cache will be discarded.
                pending_samples.clear()
                pending_sample_colors.clear()
                pending_sample_count  = 0
                self.cache_start_iter = self.global_step


    # configure_optimizers() is called later as a hook function by pytorch_lightning.
    # call stack: main.py: trainer.fit()
    # ...
    # pytorch_lightning/core/optimizer.py:
    # optim_conf = model.trainer._call_lightning_module_hook("configure_optimizers", pl_module=model)
    def configure_optimizers(self):
        if self.optimizer_type == 'AdamW':
            OptimizerClass = torch.optim.AdamW
        elif self.optimizer_type == 'CAdamW':
            OptimizerClass = CAdamW
        elif self.optimizer_type == 'NAdam':
            # In torch 1.13, decoupled_weight_decay is not supported. 
            # But since we disabled weight decay, it doesn't matter.
            OptimizerClass = torch.optim.NAdam
        # 8bit optimizers are not supported under arm64.
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

        opt_param_groups = []
        # In most cases (unless we finetune unet only), is_embedding_manager_trainable = True.
        if self.is_embedding_manager_trainable:
            # embedding_param_groups contains two param groups: adaface encoders, and unet lora params.
            embedding_param_groups = self.embedding_manager.optimized_parameters(lr, self.weight_decay, self.lora_weight_decay)
            opt_param_groups += embedding_param_groups
            # For CAdamW, we are unable to set the learning rate of the embedding_params individually.

        # Are we allowing the base model to train? If so, set two different parameter groups.
        if self.unfreeze_unet: 
            model_params = list(self.model.parameters())
            # unet_lr: default 2e-6 set in finetune-unet.yaml.
            opt_param_groups += [ {"params": model_params, "lr": self.unet_lr, "weight_decay": self.weight_decay} ]

        opt_params = list(itertools.chain.from_iterable(
            group["params"] for group in opt_param_groups
        ))
        self._params_being_optimized = opt_params
        count_optimized_params(opt_param_groups)
        
        if 'adam' in self.optimizer_type.lower():
            opt = OptimizerClass(opt_param_groups, betas=self.adam_config.betas)
            assert 'target' in self.adam_config.scheduler_config
            self.adam_config.scheduler_config.params.max_decay_steps = self.trainer.max_steps
            lambda_scheduler = instantiate_from_config(self.adam_config.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaLR(opt, lr_lambda=lambda_scheduler.schedule)

        elif self.optimizer_type == 'Prodigy':
            # [0.9, 0.999]. Converge more slowly.
            betas = self.prodigy_config.zs_betas

            # Prodigy uses an LR = 1.
            # weight_decay is always disabled (set to 0).
            opt = OptimizerClass(opt_params, lr=1., weight_decay=self.weight_decay,
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

        else:
            # Unsupported optimizer.
            breakpoint()

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

    @torch.no_grad()
    def on_after_backward(self):
        total_grad_norm = clip_grad_norm_(self._params_being_optimized, max_norm=float('inf'), norm_type=2)
        max_grad        = clip_grad_norm_(self._params_being_optimized, max_norm=float('inf'), norm_type=float('inf'))
        self.log("grad_norm", total_grad_norm)
        self.log("max_grad", max_grad)
        
    # Called by modelcheckpoint in config.yaml.
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()

        print(self.trainer.global_rank, "Saving checkpoint...")

        if os.path.isdir(self.custom_checkpoint_saver.save_dir): 
            if self.is_embedding_manager_trainable:
                emb_man_ckpt_path = os.path.join(self.custom_checkpoint_saver.save_dir, f"embeddings_gs-{self.global_step}.pt")
                self.embedding_manager.save(emb_man_ckpt_path)

            if self.unfreeze_unet:
                # Save the UNetModel state_dict.
                # self.model is a DiffusersUNetWrapper, whose parameters are the same as the UNetModel member,
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

                unet_save_path = os.path.join(self.custom_checkpoint_saver.save_dir, 
                                            f"unet-{self.global_step}.safetensors")
                safetensors_save_file(state_dict2, unet_save_path)
                print(f"Saved {unet_save_path}")

'''
# The old LDM UNet wrapper. OBSOLETE.
class DiffusionWrapper(pl.LightningModule): 
    def __init__(self, diff_model_config):
        super().__init__()
        # diffusion_model: UNetModel
        self.diffusion_model = instantiate_from_config(diff_model_config)

    # t: a 1-D batch of timesteps (during training: randomly sample one timestep for each instance).
    def forward(self, x, t, cond_context):
        prompt_emb, prompt_in, extra_info = cond_context
        out = self.diffusion_model(x, t, context=prompt_emb, context_in=prompt_in, extra_info=extra_info)

        return out
'''

# The diffusers UNet wrapper.
# attn_lora_layer_names=['q', 'k', 'v', 'out']: add lora layers to the q, k, v, out projections.
# q_lora_updates_query: If True, the q projection is updated by the LoRA layer.
# if False, the q projection is not updated by the LoRA layer. An additional q2 projection is updated.
class DiffusersUNetWrapper(pl.LightningModule):
    def __init__(self, base_model_path, torch_dtype=torch.float16,
                 use_attn_lora=False, attn_lora_layer_names=['q', 'k', 'v', 'out'], 
                 use_ffn_lora=True, lora_rank=192, 
                 attn_lora_scale_down=8, ffn_lora_scale_down=8,
                 cross_attn_shrink_factor=0.5, q_lora_updates_query=False):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_single_file(base_model_path, torch_dtype=torch_dtype)

        # diffusion_model is actually a UNet. Use this variable name to be 
        # consistent with DiffusionWrapper.
        # By default, .eval() is called in the constructor to deactivate DropOut modules.
        self.diffusion_model = self.pipeline.unet
        # Conform with main.py() which sets debug_attn.
        self.diffusion_model.debug_attn = False
        # _DeviceDtypeModuleMixin class sets self.dtype = torch_dtype.
        self.to(torch_dtype)

        self.use_attn_lora              = use_attn_lora
        self.use_ffn_lora               = use_ffn_lora
        self.lora_rank                  = lora_rank
        self.attn_lora_scale_down       = attn_lora_scale_down
        self.ffn_lora_scale_down        = ffn_lora_scale_down
        self.cross_attn_shrink_factor   = cross_attn_shrink_factor
        self.q_lora_updates_query       = q_lora_updates_query
        self.attn_lora_layer_names      = attn_lora_layer_names

    def setup_hooks_and_loras(self):
        # Keep a reference to self.attn_capture_procs to change their flags later.
        attn_capture_procs, attn_opt_modules = \
            set_up_attn_processors(self.diffusion_model, self.use_attn_lora, 
                                   attn_lora_layer_names=self.attn_lora_layer_names,
                                   lora_rank=self.lora_rank, lora_scale_down=self.attn_lora_scale_down,
                                   cross_attn_shrink_factor=self.cross_attn_shrink_factor,
                                   q_lora_updates_query=self.q_lora_updates_query)
        self.attn_capture_procs = list(attn_capture_procs.values())

        # up_blocks[0] is a different type of block, so we skip it.
        self.res_hidden_states_gradscale_blocks = self.diffusion_model.up_blocks[1:]
        for block in self.res_hidden_states_gradscale_blocks:
            block.forward = CrossAttnUpBlock2D_forward_capture.__get__(block)

        # Replace the forward() method of the last up block with a capturing method.
        self.outfeat_capture_blocks = [ self.diffusion_model.up_blocks[3] ]
        # Intercept the forward() method of the last 3 CA layers.
        for block in self.outfeat_capture_blocks:
            block.forward = CrossAttnUpBlock2D_forward_capture.__get__(block)
        
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        if self.use_attn_lora or self.use_ffn_lora:
            # LoRA scaling is always 0.125, the same as the LoRAs in AttnProcessor_LoRA_Capture
            # for cross attention layers.
            # attn_capture_procs and ffn_lora_layers are used to set the flags.
            # Replace self.diffusion_model with the PEFT wrapper model.
            # NOTE: cross-attn layers are INCLUDED in the returned lora_modules.
            # cross-attn layers are not included in ffn_lora_layers.
            # The first returned value is the PEFT wrapper model, 
            # which replaces the original unet, self.diffusion_model.
            # Even if use_ffn_lora is False, we still generate ffn_lora_layers.
            # We'll always disable them in set_lora_and_capture_flags().
            # This is to convert the unet to a PEFT model, which can handle fp16 well.
            if self.use_ffn_lora:
                target_modules_pat = 'up_blocks.3.resnets.[12].conv[a-z0-9_]+'
            else:
                # A special pattern, "dummy-target-modules" tells PEFT to add loras on NONE of the layers.
                # We couldn't simply skip PEFT initialization (converting unet to a PEFT model),
                # otherwise the attn lora layers will cause nan quickly during a fp16 training.
                target_modules_pat = DUMMY_TARGET_MODULES

            # By default, ffn_lora_scale_down = 16, i.e., the impact of LoRA is 1/16.
            ffn_lora_layers, ffn_opt_modules = \
                set_up_ffn_loras(self.diffusion_model, target_modules_pat=target_modules_pat,
                                 lora_uses_dora=True, lora_rank=self.lora_rank, 
                                 lora_alpha=self.lora_rank // self.ffn_lora_scale_down,
                                )
            self.ffn_lora_layers = list(ffn_lora_layers.values())

            # Combine attn_opt_modules and ffn_opt_modules into unet_lora_modules.
            # unet_lora_modules is for optimization and loading/saving.
            unet_lora_modules = {}
            # attn_opt_modules and ffn_opt_modules have different depths of keys.
            # attn_opt_modules:
            # up_blocks_3_attentions_1_transformer_blocks_0_attn2_processor_to_q_lora_lora_A, ...
            # ffn_opt_modules:
            # up_blocks_3_resnets_1_conv1_lora_A, ...
            unet_lora_modules.update(attn_opt_modules)
            unet_lora_modules.update(ffn_opt_modules)
            # ParameterDict can contain both Parameter and nn.Module.
            # TODO: maybe in the future, we couldn't put nn.Module in nn.ParameterDict.
            self.unet_lora_modules  = torch.nn.ParameterDict(unet_lora_modules)
            for param in self.unet_lora_modules.parameters():
                param.requires_grad = True
                param.data = param.data.to(torch.float32)

            print(f"Set up LoRAs with {len(self.unet_lora_modules)} modules: {self.unet_lora_modules.keys()}")
        else:
            self.ffn_lora_layers    = []
            self.unet_lora_modules  = None

    def load_unet_state_dict(self, unet_state_dict):
        inplace_model_copy(self.diffusion_model, unet_state_dict)

    def forward(self, x, t, cond_context, out_dtype=torch.float32):
        prompt_emb, prompt_in, extra_info = cond_context
        # img_mask is only used in normal_recon iterations. Not in unet distillation or comp distillation.
        # img_mask is used in BasicTransformerBlock.attn1 (self-attention of image tokens),
        # to avoid mixing the invalid blank areas around the augmented images with the valid areas.
        # img_mask is not used in the prompt-image cross-attention layers.
        img_mask     = extra_info.get('img_mask',     None) if extra_info is not None else None
        # shrink_cross_attn is only set to the LoRA'ed attn layers, i.e., layers 22, 23, 24.
        # Other layers will always have shrink_cross_attn = False.
        shrink_cross_attn = extra_info.get('shrink_cross_attn', False) if extra_info is not None else False
        # mix_attn_mats_in_batch: extra attn matrices to be mixed with the original ones.
        mix_attn_mats_in_batch = extra_info.get('mix_attn_mats_in_batch', False) if extra_info is not None else False
        debug = extra_info.get('debug', False) if extra_info is not None else False

        capture_ca_activations = extra_info.get('capture_ca_activations', False) if extra_info is not None else False
        # self.use_attn_lora and self.use_ffn_lora are the global flag. 
        # We can override them by setting extra_info['use_attn_lora'] and extra_info['use_ffn_lora'].
        # If use_attn_lora is set to False globally, then disable it in this call.
        use_attn_lora          = extra_info.get('use_attn_lora', self.use_attn_lora) if extra_info is not None else self.use_attn_lora
        use_ffn_lora           = extra_info.get('use_ffn_lora',  self.use_ffn_lora)  if extra_info is not None else self.use_ffn_lora
        ffn_lora_adapter_name  = extra_info.get('ffn_lora_adapter_name', None) if extra_info is not None else None
        res_hidden_states_gradscale = extra_info.get('res_hidden_states_gradscale', 1) if extra_info is not None else 1

        # set_lora_and_capture_flags() accesses self.attn_capture_procs and self.outfeat_capture_blocks.
        # The activation capture flags and caches in attn_capture_procs and outfeat_capture_blocks are set differently.
        # So we keep them in different lists.
        # use_attn_lora, capture_ca_activations, shrink_cross_attn are only applied to layers 
        # in self.attn_capture_procs.
        set_lora_and_capture_flags(self.diffusion_model, self.unet_lora_modules, 
                                   self.attn_capture_procs, self.outfeat_capture_blocks, self.res_hidden_states_gradscale_blocks,
                                   use_attn_lora, use_ffn_lora, ffn_lora_adapter_name, capture_ca_activations, 
                                   shrink_cross_attn, mix_attn_mats_in_batch, res_hidden_states_gradscale)

        # x: x_noisy from LatentDiffusion.apply_model().
        x, prompt_emb, img_mask = [ ts.to(self.dtype) if ts is not None else None \
                                       for ts in (x, prompt_emb, img_mask) ]
        
        with torch.amp.autocast("cuda", enabled=(self.dtype == torch.float16)):
            out = self.diffusion_model(sample=x, timestep=t, encoder_hidden_states=prompt_emb, 
                                       cross_attention_kwargs={'img_mask': img_mask, 
                                                               'debug':    debug},
                                       return_dict=False)[0]

        # 3 output feature tensors of the three (resnet, attn) pairs in the last up block.
        # Each (resnet, attn) pair corresponds to a TimestepEmbedSequential layer in the LDM implementation.
        #LINK ldm/modules/diffusionmodules/openaimodel.py#unet_layers
        # If not capture_ca_activations, then get_captured_activations() returns a dict with only keys and empty values.
        # NOTE: Layer 22 capturing is not supported, as layer 22 has internal_idx 0, and -1 maps
        # to the last layer in attn_capture_procs, which is layer 24.        
        extra_info['ca_layers_activations'] = \
            get_captured_activations(capture_ca_activations, self.attn_capture_procs, 
                                     self.outfeat_capture_blocks,
                                     # Only capture the activations of the last 2 CA layers.
                                     captured_layer_indices = [22, 23, 24],
                                     out_dtype=out_dtype)

        # Restore capture_ca_activations to False, and disable all attn loras. 
        # NOTE: FFN loras has been disabled above.
        set_lora_and_capture_flags(self.diffusion_model, self.unet_lora_modules, 
                                   self.attn_capture_procs, self.outfeat_capture_blocks, self.res_hidden_states_gradscale_blocks,
                                   False, False, None, False, False, None, res_hidden_states_gradscale)

        out = out.to(out_dtype)
        return out
