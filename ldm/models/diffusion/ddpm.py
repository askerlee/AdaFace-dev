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
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, \
                       count_params, instantiate_from_config, mix_embeddings, \
                       scale_emb_in_embs, ortho_subtract, calc_stats, \
                       GradientScaler, gen_gradient_scaler, \
                       convert_attn_to_spatial_weight, calc_delta_loss, \
                       save_grid, chunk_list, patch_multi_embeddings

from ldm.modules.ema import LitEma
from ldm.modules.sophia import SophiaG
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from evaluation.clip_eval import CLIPEvaluator, NoisedCLIPEvaluator
import copy
from functools import partial
import random
from safetensors.torch import load_file as safetensors_load_file
import sys

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
                 embedding_reg_weight=0.,
                 unfreeze_model=False,
                 model_lr=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 optimizer_type='AdamW',
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 use_layerwise_embedding=False,
                 use_ada_embedding=False,
                 composition_regs_iter_gap=-1,
                 prompt_delta_reg_weight=0.,
                 composition_prompt_mix_reg_weight=0.,
                 fg_bg_complementary_loss_weight=0.,
                 do_clip_teacher_filtering=False,
                 do_attn_recon_loss=True,
                 use_background_token=False,
                 use_conv_attn=False,
                 # 'face portrait' is only valid for humans/animals. On objects, use_fp_trick will be ignored.
                 use_fp_trick=True,       
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
        self.use_ada_embedding = (use_layerwise_embedding and use_ada_embedding)

        self.composition_regs_iter_gap       = composition_regs_iter_gap
        self.prompt_delta_reg_weight         = prompt_delta_reg_weight
        self.composition_prompt_mix_reg_weight = composition_prompt_mix_reg_weight
        self.fg_bg_complementary_loss_weight   = fg_bg_complementary_loss_weight
        self.do_clip_teacher_filtering      = do_clip_teacher_filtering
        self.prompt_mix_scheme              = 'mix_hijk'

        self.do_attn_recon_loss             = do_attn_recon_loss
        self.use_conv_attn                  = use_conv_attn
        self.use_background_token           = use_background_token
        # If use_conv_attn, the subject is well expressed, and use_fp_trick is unnecessary 
        # (actually harmful).
        self.use_fp_trick = False if self.use_conv_attn else use_fp_trick

        self.cached_inits_available         = False
        self.cached_inits                   = None
        self.init_iter_flags()

        # Training flags. 
        # No matter wheter the scheme is layerwise or not,
        # as long as composition_regs_iter_gap > 0 and prompt_delta_reg_weight > 0, 
        # do static comp delta reg.
        self.do_static_prompt_delta_reg = self.composition_regs_iter_gap > 0 \
                                            and self.prompt_delta_reg_weight > 0
        # Is this for DreamBooth training? Will be overwritten in LatentDiffusion ctor.
        self.is_dreambooth                  = False

        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.optimizer_type = optimizer_type
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
            self.warm_up_steps = scheduler_config.params.warm_up_steps
        else:
            self.scheduler = None
            self.warm_up_steps = 500

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.embedding_reg_weight = embedding_reg_weight
        self.distill_loss_scale = 0

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
    # We couldn't create clip_evaluator on cpu and then move it to cuda device, because 
    # NoisedCLIPEvaluator is not properly implemented to support this.
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

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean().detach()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb.detach()})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss.detach()})

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

    def init_iter_flags(self):
        self.iter_flags = { 'calc_clip_loss':           False,
                            'do_normal_recon':          True,
                            'do_attn_recon_loss':       False,
                            'is_comp_iter':             False,
                            'do_comp_prompt_mix_reg':   False,
                            'do_ada_prompt_delta_reg':  False,
                            # 'do_teacher_filter':        False,
                            # 'is_teachable':             False,
                            'use_background_token':     False,
                            'reuse_init_conds':         False,
                          }
        
    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    #LINK #shared_step
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.init_iter_flags()

        # How many regularizations are done intermittently during the training iterations?
        interm_reg_types = []
        interm_reg_probs = []
        # There's only one reg type: 'do_comp_prompt_mix_reg' in interm_reg_types.
        # This structure is kept for possible future extensions.
        if self.composition_prompt_mix_reg_weight > 0:
            interm_reg_types.append('do_comp_prompt_mix_reg')
            interm_reg_probs.append(2.)
        # NOTE: No need to have standalone ada prompt delta reg, 
        # since each prompt mix reg iter will also do ada prompt delta reg.

        N_INTERM_REGS = len(interm_reg_types)
        interm_reg_probs = np.array(interm_reg_probs) / np.sum(interm_reg_probs)

        # If N_INTERM_REGS == 0, then no intermittent regularizations, set the two flags to False.
        if N_INTERM_REGS > 0 and self.composition_regs_iter_gap > 0 \
            and self.global_step % self.composition_regs_iter_gap == 0:
            # Alternate among the regularizations in interm_reg_types. 
            # If both do_ada_prompt_delta_reg and do_comp_prompt_mix_reg,
            # then alternate between do_ada_prompt_delta_reg and do_comp_prompt_mix_reg.
            # The two regularizations cannot be done in the same batch, as they require
            # different handling of prompts and are calculated by different loss functions.
            # reg_type_idx = (self.global_step // self.composition_regs_iter_gap) % N_INTERM_REGS
            reg_type_idx = np.random.choice(N_INTERM_REGS, p=interm_reg_probs)
            iter_reg_type     = interm_reg_types[reg_type_idx]
            if iter_reg_type   == 'do_ada_prompt_delta_reg':
                self.iter_flags['do_comp_prompt_mix_reg']  = False
                self.iter_flags['do_ada_prompt_delta_reg'] = True
            # do_comp_prompt_mix_reg => do_ada_prompt_delta_reg = True.
            elif iter_reg_type == 'do_comp_prompt_mix_reg':
                self.iter_flags['do_comp_prompt_mix_reg']  = True
                self.iter_flags['do_ada_prompt_delta_reg'] = True

            self.iter_flags['is_comp_iter'] = True
            # Always calculate clip loss during comp reg iterations, even if self.iter_flags['do_teacher_filter'] is False.
            # This is to monitor how well the model performs on compositionality.
            self.iter_flags['calc_clip_loss'] = True
            self.iter_flags['do_normal_recon']    = False
            self.iter_flags['do_attn_recon_loss'] = False

        elif self.global_step >= 0:
            # do_attn_recon_loss only if not is_comp_iter, i.e., not a compositional reg iteration.
            self.iter_flags['do_attn_recon_loss'] = self.do_attn_recon_loss

        # Borrow the LR LambdaWarmUpCosineScheduler to control the mix weight.
        if self.scheduler is not None:
            lr_scale = self.scheduler.get_last_lr()[0] / self.scheduler.base_lrs[0]
            # distill_loss_scale only increases. Once it reaches 1 after LR warm-up, it stays at 1.
            self.distill_loss_scale = max(self.distill_loss_scale, lr_scale)
            # print(f'lr_lambda: {lr_lambda}')
        else:
            self.distill_loss_scale = 1.0

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

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
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
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

            self.model.eval()
            self.model.train = disabled_train
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.is_dreambooth = is_dreambooth
        self.db_reg_weight  = 1.
        if not is_dreambooth:
            self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)
            # embedding_manager.optimized_parameters(): string_to_param_dict, 
            # which maps custom tokens to embeddings
            for param in self.embedding_manager.optimized_parameters():
                param.requires_grad = True
            self.num_vectors_per_token = max(self.embedding_manager.token2num_vectors.values())
        else:
            # For DreamBooth.
            self.embedding_manager = None
            self.num_vectors_per_token = 1

        self.generation_cache = []
        self.generation_cache_img_flags = []
        self.cache_start_iter = 0
        self.num_cached_generations = 0
        
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.global_step == 0:
            self.create_clip_evaluator(next(self.parameters()).device)
            with torch.no_grad():
                self.uncond = self.get_learned_conditioning([""] * 2)

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
            model.load(config.params.embedding_manager_ckpt)
        
        return model

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
    def get_learned_conditioning(self, cond_in, img_mask=None):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                # cond_in: a list of prompts: ['an illustration of a dirty z', 'an illustration of the cool z']
                # each prompt in c is encoded as [1, 77, 768].
                # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
                c_in = copy.copy(cond_in)
                # c: [128, 77, 768]
                self.cond_stage_model.sample_last_layers_skip_weights()
                c = self.cond_stage_model.encode(cond_in, embedding_manager=self.embedding_manager)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
                
                extra_info = { 
                                'use_layerwise_context': self.use_layerwise_embedding, 
                                'use_ada_context':       self.use_ada_embedding,
                                'use_conv_attn':         self.use_conv_attn,
                                'subj_indices':          copy.copy(self.embedding_manager.placeholder_indices_fg),
                             }
                
                if self.use_ada_embedding:
                    ada_embedder = self.get_ada_conditioning
                    # Initialize the ada embedding cache, so that the subsequent calls to 
                    # EmbeddingManager.get_ada_embedding() will store the ada embedding 
                    # for each layer into the cache. 
                    # The cache will be used in calc_prompt_delta_loss().
                    self.embedding_manager.reset_ada_embedding_cache()
                    # The image mask here is used when computing Ada embeddings in embedding_manager.
                    # Do not consider mask on compositional reg iterations.
                    if img_mask is not None and \
                        (   self.iter_flags['do_comp_prompt_mix_reg'] \
                         or self.iter_flags['do_ada_prompt_delta_reg']):
                        img_mask = None

                    # img_mask is used by the ada embedding generator. 
                    # So we pass img_mask to embedding_manager here.
                    self.embedding_manager.set_img_mask(img_mask)
                    extra_info['ada_embedder'] = ada_embedder

                c = (c, c_in, extra_info)
            else:
                c = self.cond_stage_model(cond_in)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(cond_in)

        return c

    # get_ada_conditioning() is a callback function called iteratively by each layer in UNet.
    # It returns the conditioning embedding (ada embedding & other token embeddings -> clip encoder) 
    # for the current layer to UNet.
    def get_ada_conditioning(self, c_in, layer_idx, layer_attn_components, time_emb, ada_bp_to_unet):
        # We don't want to mess with the pipeline of cond_stage_model.encode(), so we pass
        # c_in, layer_idx and layer_infeat directly to embedding_manager. They will be used implicitly
        # when embedding_manager is called within cond_stage_model.encode().
        self.embedding_manager.set_ada_layer_temp_info(layer_idx, layer_attn_components, time_emb, ada_bp_to_unet)
        # DO NOT call sample_last_layers_skip_weights() here, to make the ada embeddings are generated with 
        # CLIP skip weights consistent with the static embeddings.
        c = self.cond_stage_model.encode(c_in, embedding_manager=self.embedding_manager)
        # Cache the computed ada embedding of the current layer for delta loss computation.
        # Before this call, reset_ada_embedding_cache() should have been called somewhere.
        self.embedding_manager.cache_ada_embedding(layer_idx, c)
        return c, self.embedding_manager.get_ada_emb_weight() #, self.embedding_manager.token_attn_weights

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
        encoder_posterior = self.encode_first_stage(x)
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
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
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

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
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
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

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
        # c = batch["caption"]
        # Encode noise as 4-channel latent features. Get prompts from batch. No gradient into here.
        x, c = self.get_input(batch, self.first_stage_key)
        
        if self.use_conv_attn:
            p_bg_token = 0.9
        else:
            p_bg_token = 0.8

        # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
        # or traditional TI.        
        # do_ada_prompt_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
        if self.do_static_prompt_delta_reg or self.iter_flags['do_comp_prompt_mix_reg']:
            subj_comp_prompts = []
            # *_fp prompts are like "a face portrait of ...". They are advantageous over "a photo of ..."
            # when doing compositional mix regularization. 
            # However this trick is only applicable to humans/animals.
            # For objects, even if use_fp_trick = True, 
            # *_fp prompts are not available in batch, so they won't be used.
            # Only use_background_token on training images. So if do_comp_prompt_mix_reg, 
            # then no need to check use_background_token.
            # 'subj_prompt_single_fp' in batch is True <=> (broad_class == 1 and use_fp_trick).
            # If do_comp_prompt_mix_reg but broad_class == 0 or 2, this statement is False, and 
            # used prompts will be 'subj_prompt_comp', 'cls_prompt_single', 'cls_prompt_comp'...
            if self.iter_flags['do_comp_prompt_mix_reg'] and (self.use_fp_trick and 'subj_prompt_single_fp' in batch):
                # Replace c from the default batch['caption'] to batch['subj_prompt_single_fp'].
                c = batch['subj_prompt_single_fp']
                SUBJ_PROMPT_COMP  = 'subj_prompt_comp_fp'
                CLS_PROMPT_COMP   = 'cls_prompt_comp_fp'
                CLS_PROMPT_SINGLE = 'cls_prompt_single_fp'
            # "not self.iter_flags['do_comp_prompt_mix_reg']": implies no ada delta loss.
            # So this iter is only doing recon on training images.
            # Only use_background_token on such cases. 
            # *_comp_bg: used for static delta loss. 
            # To avoid the backgound token taking too much of the foreground, 
            # we only use the background token on 80% of the training images.
            elif not self.iter_flags['do_comp_prompt_mix_reg'] and self.use_background_token \
              and self.global_step >= 0 \
              and random.random() < p_bg_token:
                c = batch['subj_prompt_single_bg']
                SUBJ_PROMPT_COMP  = 'subj_prompt_comp_bg'
                CLS_PROMPT_COMP   = 'cls_prompt_comp_bg'
                CLS_PROMPT_SINGLE = 'cls_prompt_single_bg'
                self.iter_flags['use_background_token'] = True
            # Either do_comp_prompt_mix_reg but not (use_fp_trick and broad_class == 1), 
            # or recon iters (not do_comp_prompt_mix_reg) and not use_background_token 
            # We don't use_fp_tricks on training images. use_fp_tricks is only for compositional regularization.
            else:
                # c has been assigned at the first line of this function.
                SUBJ_PROMPT_COMP  = 'subj_prompt_comp'
                CLS_PROMPT_COMP   = 'cls_prompt_comp'
                CLS_PROMPT_SINGLE = 'cls_prompt_single'

            # Each prompt_comp consists of multiple prompts separated by "|".
            # Split them into a list of subj_comp_prompts/cls_comp_prompts.
            for prompt_comp in batch[SUBJ_PROMPT_COMP]:
                subj_comp_prompts.append(prompt_comp.split("|"))
            cls_comp_prompts = []
            for prompt_comp in batch[CLS_PROMPT_COMP]:
                cls_comp_prompts.append(prompt_comp.split("|"))
            cls_single_prompts = batch[CLS_PROMPT_SINGLE]
            # REPEATS: how many prompts correspond to each image.
            REPEATS = len(subj_comp_prompts[0])
            if REPEATS == 1 or self.iter_flags['do_comp_prompt_mix_reg'] or self.iter_flags['do_ada_prompt_delta_reg']:
                # When this iter computes ada prompt delta loss / prompt mixing loss, 
                # only use the first of the composition prompts (in effect num_compositions_per_image=1),
                # otherwise it will use more than 40G RAM.
                subj_comp_prompts = [ prompts[0] for prompts in subj_comp_prompts ]
                cls_comp_prompts  = [ prompts[0] for prompts in cls_comp_prompts ]
                delta_prompts = (subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
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
                c = c * REPEATS
                cls_single_prompts = cls_single_prompts * REPEATS
                delta_prompts = (subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
        else:
            delta_prompts = None

        if 'mask' in batch:
            img_mask = batch['mask']
            img_mask = img_mask.unsqueeze(1).to(x.device)
            img_mask = F.interpolate(img_mask, size=x.shape[-2:], mode='nearest')
        else:
            img_mask = None

        loss = self(x, c, delta_prompts, img_mask=img_mask, **kwargs)
        return loss

    # LatentDiffusion.forward() is only called during training, by shared_step().
    #LINK #shared_step
    def forward(self, x, c, delta_prompts=None, img_mask=None, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        self.iter_flags['reuse_init_conds']     = (self.do_clip_teacher_filtering and self.iter_flags['do_comp_prompt_mix_reg'] \
                                                   and self.cached_inits_available)

        if self.model.conditioning_key is not None:
            assert c is not None
            # c: condition, a prompt template. 
            # get_learned_conditioning(): convert c to a [B, 77, 768] tensor.
            if self.cond_stage_trainable:
                # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
                # or traditional TI.
                # do_ada_prompt_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
                # c: subj_single_prompts, which are plain prompts like 
                # ['an illustration of a dirty z', 'an illustration of the cool z']
                if self.do_static_prompt_delta_reg or self.iter_flags['do_comp_prompt_mix_reg'] or self.iter_flags['do_attn_recon_loss']:
                    if not self.iter_flags['reuse_init_conds']:
                        subj_single_prompts = c
                        subj_comp_prompts, cls_single_prompts, cls_comp_prompts = delta_prompts
                    else:
                        # self.iter_flags['reuse_init_conds'], discard the prompts offered in shared_step().
                        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                            chunk_list(self.cached_inits['delta_prompts'], 4)
                        # cached_inits will be used in p_losses(), 
                        # so don't turn off cached_inits_available yet.

                    #if self.iter_flags['use_background_token']:
                    #    print(subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts)

                    N_LAYERS = 16 if self.use_layerwise_embedding else 1
                    ORIG_BS  = len(x)
                    N_EMBEDS = ORIG_BS * N_LAYERS
                    # HALF_BS is at least 1. So if ORIG_BS == 1, then HALF_BS = 1.
                    HALF_BS  = max(ORIG_BS // 2, 1)
                    
                    if self.iter_flags['do_comp_prompt_mix_reg'] or self.iter_flags['do_ada_prompt_delta_reg']:
                        # Only keep the first half of batched prompts to save RAM.
                        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                            subj_single_prompts[:HALF_BS], subj_comp_prompts[:HALF_BS], \
                            cls_single_prompts[:HALF_BS],  cls_comp_prompts[:HALF_BS]
                                            
                    # If not do_ada_prompt_delta_reg, we still compute the static embeddings 
                    # of the 4 types of prompts, to compute static delta loss. But now there are 8 prompts (4*BS=8).
                    delta_prompts = subj_single_prompts + subj_comp_prompts \
                                    + cls_single_prompts + cls_comp_prompts
                    # print(delta_prompts)
                    # breakpoint()
                    # c_static_emb: the static embeddings [4 * N_EMBEDS, 77, 768], 
                    # 4 * N_EMBEDS = 4 * ORIG_BS * N_LAYERS,
                    # whose layer dimension (N_LAYERS) is tucked into the batch dimension. 
                    # c_in: a copy of delta_prompts, the concatenation of
                    # (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts).
                    # extra_info: a dict that contains extra info.
                    c_static_emb, c_in, extra_info = self.get_learned_conditioning(delta_prompts, img_mask=img_mask)
                    subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb = \
                        c_static_emb.chunk(4)

                    # Only keep the first half (for single prompts), as the second half is the same 
                    # (for comp prompts, differs at the batch index, but the token index is identical).
                    # placeholder_indices_fg is only for (subj_single_prompts, subj_comp_prompts), since
                    # the placeholder token doesn't appear in the class prompts. 
                    # Now we take the first half of placeholder_indices_fg, so that 
                    # they only account for the subject single prompt, but they are also 
                    # applicable to the other 3 types of prompts as they are all aligned 
                    # at the beginning part of the prompts.
                    # BUG: if the batch size of a mix batch > 4, then the subj_indices_half_N
                    # corresponds to the indices in more than one instance. But patch_multi_embeddings()
                    # treat the indices as if they are always in the same instance.
                    if self.embedding_manager.placeholder_indices_fg is None:
                        breakpoint()
                        
                    # subj_indices_half_B, subj_indices_half_N: subject token indices 
                    # within the subject single prompt (BS=1).
                    # len(subj_indices_half_N): embedding number of the subject token.
                    subj_indices_half_B  = self.embedding_manager.placeholder_indices_fg[0].chunk(2)[0]
                    subj_indices_half_N  = self.embedding_manager.placeholder_indices_fg[1].chunk(2)[0]

                    if self.iter_flags['do_comp_prompt_mix_reg'] or self.iter_flags['do_ada_prompt_delta_reg']:
                        assert not self.iter_flags['do_attn_recon_loss']
                        extra_info['subj_indices'] = copy.copy(self.embedding_manager.placeholder_indices_fg)
                    else:
                        assert self.iter_flags['do_attn_recon_loss']
                        extra_info['subj_indices'] = (subj_indices_half_B, subj_indices_half_N)

                    subj_indices0_half_B = self.embedding_manager.placeholder_indices_fg0[0].chunk(2)[0]
                    subj_indices0_half_N = self.embedding_manager.placeholder_indices_fg0[1].chunk(2)[0]
                    extra_info['subj_indices0'] = (subj_indices0_half_B, subj_indices0_half_N)

                    # if do_ada_prompt_delta_reg, then do_comp_prompt_mix_reg 
                    # may be True or False, depending whether mix reg is enabled.
                    if self.iter_flags['do_comp_prompt_mix_reg'] or self.iter_flags['do_attn_recon_loss']:
                        # c_in2 is used to generate ada embeddings.
                        # Arrange c_in2 in the same layout as the static embeddings.
                        # The mix_single_prompts within c_in2 will only be used to generate ordinary 
                        # prompt embeddings, i.e., 
                        # it doesn't contain subject token, and no ada embedding will be injected.
                        # despite the fact that subj_single_emb, cls_single_emb are mixed into 
                        # the corresponding static embeddings.
                        # Tried to use subj_single_prompts as mix_single_prompts, leading to worse performance.
                        mix_single_prompts = cls_single_prompts
                        # The last set corresponds the mixed embeddings of (subj_comp_prompts, cls_comp_prompts). 
                        # Using subj_comp_prompts as mix_comp_prompts will enhance authenticity but hurt compositionality.
                        # Using cls_comp_prompts  as mix_comp_prompts will enhance compositionality but hurt authenticity.
                        #mix_comp_prompts = subj_comp_prompts
                        mix_comp_prompts = cls_comp_prompts

                        c_in2 = subj_single_prompts + subj_comp_prompts + mix_single_prompts + mix_comp_prompts

                        # The subject is represented with a multi-embedding token.
                        if len(subj_indices_half_N) > 1:
                            cls_single_emb = patch_multi_embeddings(cls_single_emb, subj_indices_half_N)
                            cls_comp_emb   = patch_multi_embeddings(cls_comp_emb,   subj_indices_half_N)
                            
                        # The static embeddings of subj_comp_prompts and cls_comp_prompts,
                        # i.e., subj_comp_emb and cls_comp_emb will be mixed.
                        # In subj_comp_emb_v_mix, subj_single_emb_v_mix, the K subject embeddings are scaled down by 0.5,
                        # /* and added with 0.5 * class embeddings */, 
                        # so that other components will express more during guidance. 
                        # and the token number will be the double of subj_comp_emb.
                        # The first half of the embeddings will be used as the q/v in cross attention layers.
                        # The second half of the embeddings will be used as the k in cross attention layers.
                        # This is only for static embeddings. The dynamically generated ada embeddings 
                        # won't be mixed, but simply repeated.

                        """                         
                        subj_comp_emb_v   = scale_emb_in_embs(subj_comp_emb,   subj_indices_half_N, 
                                                               scale=subj_emb_scale, scale_first_only=False)
                        subj_single_emb_v = scale_emb_in_embs(subj_single_emb, subj_indices_half_N, 
                                                               scale=subj_emb_scale, scale_first_only=False)
                        """
                        
                        total_training_steps = self.trainer.max_steps
                        INIT_CLS_EMB_SCALE  = 0.1
                        FINAL_CLS_EMB_SCALE = 0.3
                        # Linearly increase the scale of the class embeddings from 0.1 to 0.3, i.e., 
                        # Linearly decrease the scale of the subject embeddings from 0.9 to 0.7, 
                        # so that the distillation keeps being effective. Otherwise the teacher 
                        # will gradually become very similar to the student in the end.
                        subj_emb_scale = 1 - INIT_CLS_EMB_SCALE \
                                         - (FINAL_CLS_EMB_SCALE - INIT_CLS_EMB_SCALE) * np.random.uniform(0.8, 1.3) \
                                            * self.global_step / total_training_steps
                        # mix_embeddings('add'):  being subj_comp_emb almost everywhere, except those at subj_indices_half_N,
                        # where they are subj_comp_emb * subj_emb_scale + cls_comp_emb * (1 - subj_emb_scale).
                        # subj_comp_emb, cls_comp_emb, subj_single_emb, cls_single_emb: [16, 77, 768].
                        # Each is of a single instance. So only provides subj_indices_half_N 
                        # (multiple token indices of the same instance).
                        subj_comp_emb_v   = mix_embeddings('add', subj_comp_emb, cls_comp_emb,
                                                            None, c1_subj_scale=subj_emb_scale)
                        subj_single_emb_v = mix_embeddings('add', subj_single_emb, cls_single_emb,
                                                            None, c1_subj_scale=subj_emb_scale)

                        if random.random() < 1: #0.5:
                            mix_comp_emb_all_layers   = torch.cat([subj_comp_emb_v,   cls_comp_emb],   dim=1)
                            mix_single_emb_all_layers = torch.cat([subj_single_emb_v, cls_single_emb], dim=1)
                        else:
                            # Swap embeddings that produce q and v.
                            mix_comp_emb_all_layers   = torch.cat([cls_comp_emb,   subj_comp_emb_v],   dim=1)
                            mix_single_emb_all_layers = torch.cat([cls_single_emb, subj_single_emb_v], dim=1)

                        #mix_comp_emb_all_layers  = cls_comp_emb
                        #mix_single_emb_all_layers = cls_single_emb
                        # If prompt_mix_grad_scale == 0, stop gradient on mix_comp_emb, 
                        # since it serves as the reference.
                        # If we don't stop gradient on mix_comp_emb, 
                        # then chance is that mix_comp_emb might be dominated by subj_comp_emb2,
                        # so that mix_comp_emb will produce images similar as subj_comp_emb2 does.
                        # Stopping the gradient will improve compositionality but reduce face similarity.
                        prompt_mix_grad_scale = 0.01
                        if prompt_mix_grad_scale == 0:
                            mix_comp_emb_all_layers   = mix_comp_emb_all_layers.detach()
                            mix_single_emb_all_layers = mix_single_emb_all_layers.detach()
                        elif prompt_mix_grad_scale != 1:
                            grad_scaler = GradientScaler(prompt_mix_grad_scale)
                            mix_comp_emb_all_layers   = grad_scaler(mix_comp_emb_all_layers)
                            mix_single_emb_all_layers = grad_scaler(mix_single_emb_all_layers)

                        if self.use_layerwise_embedding:
                            # 4, 5, 6, 7, 8, 9 correspond to original layer indices 7, 8, 12, 16, 17, 18.
                            # (same as used in computing mixing loss)
                            sync_layer_indices = [4, 5, 6, 7, 8, 9]
                            layer_mask = torch.zeros_like(mix_comp_emb_all_layers).reshape(-1, N_LAYERS, *mix_comp_emb_all_layers.shape[1:])
                            layer_mask[:, sync_layer_indices] = 1
                            layer_mask = layer_mask.reshape(-1, *mix_comp_emb_all_layers.shape[1:])

                            # This copy of subj_single_emb, subj_comp_emb2 will be simply 
                            # repeated at the token dimension to match the token number of the mixed (concatenated) 
                            # mix_single_emb and mix_comp_emb embeddings.
                            subj_comp_emb2   = subj_comp_emb.repeat(1, 2, 1)
                            subj_single_emb2 = subj_single_emb.repeat(1, 2, 1)

                            # Otherwise, the second halves of subj_comp_emb2/cls_comp_emb
                            # are already key embeddings. No need to repeat.

                            # Use most of the layers of embeddings in subj_comp_emb2, but 
                            # replace sync_layer_indices layers with those from mix_comp_emb_all_layers.
                            # Do not assign with sync_layers as indices, which destroys the computation graph.
                            mix_comp_emb   = subj_comp_emb2 * (1 - layer_mask) \
                                             + mix_comp_emb_all_layers * layer_mask

                            mix_single_emb = subj_single_emb2 * (1 - layer_mask) \
                                             + mix_single_emb_all_layers * layer_mask
                        else:
                            # There is only one layer of embeddings.
                            mix_comp_emb   = mix_comp_emb_all_layers
                            mix_single_emb = mix_single_emb_all_layers

                        if self.iter_flags['do_comp_prompt_mix_reg']:
                            # c_static_emb2 will be added with the ada embeddings to form the 
                            # conditioning embeddings in the U-Net.
                            # Unmixed embeddings and mixed embeddings will be merged in one batch for guiding
                            # image generation and computing compositional mix loss.
                            c_static_emb2 = torch.cat([ subj_single_emb2, subj_comp_emb2, 
                                                        mix_single_emb,   mix_comp_emb ], dim=0)
                            
                            extra_info['iter_type']      = self.prompt_mix_scheme
                            # Set ada_bp_to_unet to False will reduce performance.
                            extra_info['ada_bp_to_unet'] = True
                        # do_attn_recon_loss
                        else:
                            # c_in_cls         = cls_single_prompts
                            # Use the mixed embeddings as the static embeddings of the class prompts.
                            # It strikes a balance between pure class embeddings and subject embeddings.
                            c_static_emb_cls = mix_single_emb
                            extra_info2 = copy.copy(extra_info)
                            # cond_mix uses class single prompts. Set iter_type to 'mix_recon' to 
                            # distinguish it from 'normal_recon' which uses subject single prompts.
                            extra_info2['iter_type']      = 'mix_recon'
                            # Inherit 'use_conv_attn' from extra_info.
                            # extra_info2['use_conv_attn']  = False
                            cond_mix = (c_static_emb_cls, cls_single_prompts, extra_info2)
                            # There's a loopy reference extra_info -> c_cls -> extra_info, but it's fine.
                            extra_info['cond_mix'] = cond_mix

                            # Same inits as the normal recon loss below.
                            c_in2         = subj_single_prompts[:ORIG_BS]
                            c_static_emb2 = subj_single_emb[:N_EMBEDS]

                            assert self.iter_flags['do_normal_recon']
                            extra_info['iter_type']      = 'normal_recon'
                            extra_info['ada_bp_to_unet'] = False
                            
                    # This iter is a simple ada prompt delta loss iter, without prompt mixing loss. 
                    # This branch is reached only if prompt mixing is not enabled.
                    # "and not self.iter_flags['do_comp_prompt_mix_reg']" is redundant, because it's at an "elif" branch.
                    # Kept for clarity. 
                    elif self.iter_flags['do_ada_prompt_delta_reg'] and not self.iter_flags['do_comp_prompt_mix_reg']:
                        # Do ada prompt delta loss in this iteration. 
                        # c_static_emb: static embeddings, [64, 77, 768]. 128 = 4 * 16. 
                        # 4 is the total number of prompts in delta_prompts. 
                        # Each prompt is converted to 16 embeddings.
                        c_static_emb2 = c_static_emb
                        c_in2         = c_in
                        # c_in2 consists of four types of prompts: 
                        # subj_single, subj_comp, cls_single, cls_comp.
                        extra_info['iter_type']      = 'do_ada_prompt_delta_reg'
                        extra_info['ada_bp_to_unet'] = False
                        
                    else:
                        # The original scheme. Use the original subj_single_prompts embeddings and prompts.
                        # When num_compositions_per_image > 1, subj_single_prompts contains repeated prompts,
                        # so we only keep the first N_EMBEDS embeddings and the first ORIG_BS prompts.
                        c_in2         = subj_single_prompts[:ORIG_BS]
                        c_static_emb2 = subj_single_emb[:N_EMBEDS]

                        assert self.iter_flags['do_normal_recon']
                        extra_info['iter_type']      = 'normal_recon'
                        extra_info['ada_bp_to_unet'] = False

                    extra_info['cls_comp_prompts']   = cls_comp_prompts
                    extra_info['cls_single_prompts'] = cls_single_prompts
                    # 'delta_prompts' is only used in comp_prompt_mix_reg iters.
                    extra_info['delta_prompts']      = c_in

                    # c_static_emb2 is the static embeddings of the prompts used in losses other than 
                    # the static delta loss, e.g., used to estimate the ada embeddings.
                    # If use_ada_embedding, then c_in2 will be fed again to CLIP text encoder to 
                    # get the ada embeddings. Otherwise, c_in2 will be useless and ignored.
                    c = (c_static_emb2, c_in2, extra_info)
                    # c_static_emb is the full set of embeddings of subj_single_prompts, subj_comp_prompts, 
                    # cls_single_prompts, cls_comp_prompts. 
                    # c_static_emb is backed up to be used to compute the static delta loss.
                    self.c_static_emb = c_static_emb

                else:
                    # Not (self.do_static_prompt_delta_reg or 'do_comp_prompt_mix_reg' or 'do_attn_recon_loss'.
                    # That is, no delta loss, compositional mix loss, or attention-weighted recon loss. 
                    # Keep the tuple c unchanged.
                    c = self.get_learned_conditioning(c)
                    # c[2]: extra_info. Here is only reached when do_static_prompt_delta_reg = False.
                    # Either prompt_delta_reg_weight == 0 or it's called by self.validation_step().
                    assert self.iter_flags['do_normal_recon']
                    c[2]['iter_type'] = 'normal_recon'
                    extra_info['ada_bp_to_unet'] = False
                    extra_info['subj_indices'] = copy.copy(self.embedding_manager.placeholder_indices_fg)
                    self.c_static_emb = None

            # shorten_cond_schedule: False. Skipped.
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                # q_sample() is only called during training. 
                # q_sample() calls apply_model(), which estimates the latent code of the image.
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        # self.model (UNetModel) is called in p_losses().
        #LINK #p_losses
        return self.p_losses(x, c, t, img_mask=img_mask, *args, **kwargs)

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

    # do_recon: return denoised images. 
    # if do_recon and cfg_scale > 1, apply classifier-free guidance. 
    # unet_has_grad: when returning do_recon (e.g. to select the better instance by smaller clip loss), 
    # to speed up, no BP is done on these instances, so unet_has_grad=False.
    def guided_denoise(self, x_start, noise, t, cond, unet_has_grad=True, 
                       crossattn_force_grad=False, 
                       do_recon=False, cfg_scales=None):
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if unet_has_grad:
            # No need to force grad on cross attention layers, as the whole U-Net has grad.
            crossattn_force_grad = False

        # model_output is the predicted noise.
        # if not unet_has_grad, we save RAM by not storing the computation graph.
        # crossattn_force_grad: even if unet_has_grad = False, 
        # we can still enable gradients on cross attention layers,
        # so that limited optimizations w.r.t. the embedders can be done (but unable to BP through UNet).
        if not unet_has_grad:
            cond[2]['crossattn_force_grad'] = crossattn_force_grad
            with torch.no_grad():
                model_output = self.apply_model(x_noisy, t, cond)
            del cond[2]['crossattn_force_grad']
        else:
            # if unet_has_grad, we don't have to take care of embedding_manager.force_grad.
            # Subject embeddings will naturally have gradients.
            model_output = self.apply_model(x_noisy, t, cond)

        # Save ada embeddings generated during apply_model(), to be used in delta loss. 
        # Otherwise it will be overwritten by uncond denoising.
        if self.embedding_manager.ada_embeddings is not None:
            # ada_embeddings: [4, 16, 77, 768]
            ada_embeddings = torch.stack(self.embedding_manager.ada_embeddings, dim=1)
        else:
            ada_embeddings = None

        # Unconditional noises are never optimized.
        if do_recon:
            with torch.no_grad():
                x_start_ = x_start.chunk(2)[0]
                noise_   = noise.chunk(2)[0]
                t_       = t.chunk(2)[0]
                # For efficiency, x_start_, noise_ and t_ are of half-batch size,
                # and only compute model_output_uncond on half of the batch, 
                # as the second half (mix single, mix comp) is generated under the same initial conditions 
                # (only differ on prompts, but uncond means no prompts).
                x_noisy_ = self.q_sample(x_start=x_start_, t=t_, noise=noise_)
                # self.uncond: precomputed unconditional embeddings.
                model_output_uncond = self.apply_model(x_noisy_, t_, self.uncond)
                # model_output_uncond: [2, 4, 64, 64] -> [4, 4, 64, 64]
                model_output_uncond = model_output_uncond.repeat(2, 1, 1, 1)
                
            # Classifier-free guidance to make the contents in the 
            # generated images more pronounced => smaller CLIP loss.
            if cfg_scales is not None:
                cfg_scales = cfg_scales.view(-1, 1, 1, 1)
                pred_noise = model_output * cfg_scales - model_output_uncond * (cfg_scales - 1)
            else:
                pred_noise = model_output
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=pred_noise)
        else:
            x_recon = None
        
        return model_output, x_recon, ada_embeddings

    # Release part of the computation graph on unused instances to save RAM.
    def release_plosses_intermediates(self, local_vars):
        del local_vars['model_output'], local_vars['x_recon'], local_vars['clip_images_code']
        cond = local_vars['cond']
        if 'unet_feats' in cond[2]:
            del cond[2]['unet_feats']
        if 'unet_attns' in cond[2]:
            del cond[2]['unet_attns']        

    # t: steps.
    # cond: (c_static_emb, c_in, extra_info). c_in is the textual prompts. 
    # extra_info: a dict that contains 'ada_embedder' and other fields. 
    # ada_embedder: a function to convert c_in to ada embeddings.
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, cond, t, noise=None, img_mask=None):
        # If noise is not None, then use the provided noise.
        # Otherwise, generate noise randomly.
        noise = default(noise, lambda: torch.randn_like(x_start))
        #print(cond[1])

        # If cached_inits_available, cached_inits are only used if do_comp_prompt_mix_reg = True.
        # In this case, do_teacher_filtering = True, and we choose the better instance 
        # between the two in the batch, if it's above the usable threshold.
        # do_teacher_filtering and cached_inits_available are mutually exclusive.
        self.iter_flags['do_teacher_filter'] = (self.do_clip_teacher_filtering and self.iter_flags['do_comp_prompt_mix_reg'] \
                                                and not self.cached_inits_available)
        # reuse_init_conds was evaluated in LatentDiffusion::forward().
        # self.iter_flags['reuse_init_conds']     = (self.do_clip_teacher_filtering and self.iter_flags['do_comp_prompt_mix_reg'] \
        #                                            and self.cached_inits_available)

        if self.iter_flags['reuse_init_conds']:
            # If self.iter_flags['reuse_init_conds'], we use the cached x_start and cond.
            # cond is already organized as (subj single, subj comp, mix single, mix comp). 
            # No need to manipulate.
            # noise will be kept as the sampled random noise at the beginning of p_losses(). 
            x_start = self.cached_inits['x_start']
            prev_t  = self.cached_inits['t']
            # Avoid the next mix iter to still use the cached inits.
            self.cached_inits_available = False
            self.cached_inits = None
            # reuse init iter takes a smaller cfg scale, as in the second denoising step, 
            # a particular scale tend to make the cfg-denoised mixed images more dissimilar 
            # to the subject images than in the first denoising step. 

        cfg_scales_for_clip_loss = None
        c_static_emb, c_in, extra_info = cond

        if self.iter_flags['is_comp_iter']:
            # If bs=2, then HALF_BS=1.
            if not self.iter_flags['reuse_init_conds']:
                HALF_BS  = max(x_start.shape[0] // 2, 1)
                # At 80% of the chance, randomly initialize x_start and t. Note the batch size is still 2 here.
                # At 20% of the chance, use a noisy x_start based on the training images. 
                # This may help the model ignore the background in the training images given prompts, 
                # i.e., give prompts higher priority over the background.
                if random.random() < 0.8:
                    x_start.normal_()
                else:
                    x_start = torch.randn_like(x_start) * 0.9 + x_start * 0.1
                # Randomly choose t from the largest 150 timesteps, so as to match the completely noisy x_start.
                t_tail = torch.randint(int(self.num_timesteps * 0.85), self.num_timesteps, (x_start.shape[0],), device=x_start.device)
                t = t_tail
            else:
                # If self.iter_flags['reuse_init_conds'], BS is already doubled in the previous 
                # compositional iteration. So divide by 4.
                HALF_BS  = max(x_start.shape[0] // 4, 1)
                # Randomly choose t from the middle 300-600 timesteps, 
                # so as to match the once-denoised x_start.
                # generate the full batch size of t, but actually only use the first HALF_BS.
                # This is to make the code consistent with the non-comp case and avoid unnecessary confusion.
                t_mid = torch.randint(int(self.num_timesteps * 0.6), int(self.num_timesteps * 0.75), 
                                      (x_start.shape[0],), device=x_start.device)
                # t_upperbound: old t - 150. That is, at least 150 steps away from the previous t.
                t_upperbound = prev_t - int(self.num_timesteps * 0.15)
                # t should be at least 150 steps away from the previous t, 
                # so that the noise level is sufficiently different.
                t = torch.minimum(t_mid, t_upperbound)

            # Ignore img_mask.
            img_mask = None

            # Only do delta loss reg. This happens if composition_prompt_mix_reg_weight = 0.
            if not self.iter_flags['do_comp_prompt_mix_reg']:
                # Generate a batch of 4 instances with the same initial x_start, noise and t.
                # This doubles the batch size to 4, if bs=2.
                x_start = x_start[:HALF_BS].repeat(4, 1, 1, 1)
                noise   = noise[:HALF_BS].repeat(4, 1, 1, 1)
                t       = t[:HALF_BS].repeat(4)
                # Calculate CLIP score only for image quality evaluation
                cfg_scales_for_clip_loss = torch.ones_like(t) * 5
            else:
                # Teachers are slightly more aggressive, to increase the teachable fraction.                
                cfg_scale_for_teacher  = 6
                cfg_scale_for_student  = 5
                cfg_scales_for_teacher   = torch.ones(HALF_BS*2, device=x_start.device) * cfg_scale_for_teacher
                cfg_scales_for_student   = torch.ones(HALF_BS*2, device=x_start.device) * cfg_scale_for_student
                cfg_scales_for_clip_loss = torch.cat([cfg_scales_for_student, cfg_scales_for_teacher], dim=0)

                # First iteration of a two-iteration do_comp_prompt_mix_reg.
                # Generate a batch of 4 instances in two sets, each set with the same initial x_start, noise and t.
                # Then filter, find the best teachable set (if any) and pass to the recursive iteration.
                # If no teachable set is found, skip recursion and the prompt mix reg.
                # Note x_start[0] = x_start[2] != x_start[1] = x_start[3].
                # That means, instances are arranged as: 
                # (subj comp 1, subj comp 2, mix comp 1, mix comp 2).
                # This doubles the batch size to 4, if bs=2.
                if self.iter_flags['do_teacher_filter']:
                    x_start = x_start.repeat(2, 1, 1, 1)
                    noise   = noise.repeat(2, 1, 1, 1)
                    t       = t.repeat(2)

                    # print(c_in)

                    # Make two identical sets of c_static_emb2 and c_in2.
                    subj_single_emb, subj_comp_emb, mix_single_emb, mix_comp_emb = \
                        c_static_emb.chunk(4)
                    # Only keep *comp_emb, but repeat them to form twin comp sets.
                    c_static_emb2 = torch.cat([ subj_comp_emb, subj_comp_emb, 
                                                mix_comp_emb,  mix_comp_emb ], dim=0)
                    
                    subj_single_prompts, subj_comp_prompts, mix_single_prompts, mix_comp_prompts = \
                        chunk_list(c_in, 4)
                    c_in2 = subj_comp_prompts + subj_comp_prompts + mix_comp_prompts + mix_comp_prompts
                    cond_orig = cond
                    # Replace cond with the cond for twin comp sets.
                    cond = (c_static_emb2, c_in2, extra_info)
                # Not self.iter_flags['do_teacher_filter']. Either self.iter_flags['reuse_init_conds'], or not self.do_clip_teacher_filtering.
                # In the latter case, only compute the emb reg and delta loss on the mixed images.
                else:
                    noise   = noise[:HALF_BS].repeat(4, 1, 1, 1)
                    t       = t[:HALF_BS].repeat(4)
                    # use cached x_start and cond. Already has the 4-type structure. 
                    # No change to the condition here.

        extra_info['use_background_token'] = self.iter_flags['use_background_token']
        iter_type = cond[2]['iter_type']

        # There are always some subj prompts in this batch. So if self.use_conv_attn,
        # then cond[2]['use_conv_attn'] = True, it will inform U-Net to do conv attn.
        model_output, x_recon, ada_embeddings = \
            self.guided_denoise(x_start, noise, t, cond, 
                                unet_has_grad=not self.iter_flags['do_teacher_filter'], 
                                crossattn_force_grad=False,
                                do_recon=self.iter_flags['calc_clip_loss'],
                                # cfg_scales: classifier-free guidance scales.
                                cfg_scales=cfg_scales_for_clip_loss)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        # default is "eps", i.e., the UNet predicts noise.
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # Only compute the loss on the masked region.
        if img_mask is not None:
            target       = target       * img_mask
            model_output = model_output * img_mask
        
        loss = 0
        twin_comp_ada_embeddings = None

        if self.iter_flags['do_normal_recon']:
            # Compute subj_fg_bg_complementary_loss, only when 
            # we don't compute mix_fg_bg_complementary_loss. Otherwise it'll take too much RAM.
            # mix_fg_bg_complementary_loss is preferred over subj_fg_bg_complementary_loss,
            # since the mix prompts produce slightly more accurate attention maps on the 
            # foreground subject.
            if self.iter_flags['use_background_token'] and not self.iter_flags['do_attn_recon_loss']:
                subj_fg_bg_complementary_loss = \
                            self.calc_fg_bg_complementary_loss(cond[2]['unet_attns'], 
                                                               self.embedding_manager.placeholder_indices_fg,
                                                               self.embedding_manager.placeholder_indices_bg,
                                                               x_start.shape[0],
                                                               fg_grad_scale=0.01,
                                                               do_demean_first=False
                                                              )
                
                # Release RAM.
                del cond[2]['unet_attns'], cond[2]['unet_feats']
                loss += self.fg_bg_complementary_loss_weight * subj_fg_bg_complementary_loss
                loss_dict.update({f'{prefix}/fg_bg_complem_subj': subj_fg_bg_complementary_loss.item()})
                # print(fg_bg_complementary_loss)

            if self.iter_flags['do_attn_recon_loss']:
                subj_indices0 = self.embedding_manager.placeholder_indices_fg0
                # For 'do_normal_recon' but do_attn_recon_loss_iter=True, same to mix reg iters or ada reg iters, 
                # 4 types of prompts are fed to embedding_manager to generate static embeddings in forward(). 
                # subj_indices00 has a BS of 4, for the 2 types of subject prompts, each type 2 prompts.
                # So we only keep the first half, which correspond to the 2 subject-single prompts.
                subj_indices0 = (subj_indices0[0].chunk(2)[0], subj_indices0[1].chunk(2)[0])

                cond_mix = extra_info['cond_mix']
                #print(cond_mix[1])
                
                # We only add a small amount of noise to x_start, so that the obtained attention maps are more accurate.
                #t_small = torch.randint(0, int(self.num_timesteps * 0.1), 
                #                        size=t.shape, device=x_start.device)
                t0 = torch.zeros_like(t)
                # crossattn_force_grad: override the cross attention layers' autograd status, so that we can BP
                # from the complementary loss through the cross attention.
                # cond_mix: cls prompts. So do not use_conv_attn. cond_mix[2]['use_conv_attn'] is False
                self.guided_denoise(x_start, noise, t0, cond_mix, 
                                    unet_has_grad=False, 
                                    crossattn_force_grad=True,
                                    do_recon=False,
                                    cfg_scales=None)
                unet_attns_mix_single = cond_mix[2]['unet_attns']

                # Compute mix_fg_bg_complementary_loss, which is similar as
                # subj_fg_bg_complementary_loss but with the mix prompts.
                # placeholder_indices_fg0 here is the cached placeholder_indices_fg0 on subject prompts.
                # fg_grad_scale > 0, because we override the embedding manager's autograd status,
                # so that gradients can still be BP-ed from the cross attention layers, even if 
                # unet_has_grad=False.
                if self.iter_flags['use_background_token']:
                    # placeholder_indices_bg contains the background token indices of the 4 types of prompts.
                    # So we only take the beginning BS indices out of it.
                    mix_fg_bg_complementary_loss = \
                                self.calc_fg_bg_complementary_loss(unet_attns_mix_single, 
                                                                   self.embedding_manager.placeholder_indices_fg,
                                                                   self.embedding_manager.placeholder_indices_bg,
                                                                   x_start.shape[0],
                                                                   fg_grad_scale=0.01,
                                                                   do_demean_first=False
                                                                  )
                    
                    # Do not delete cond_mix[2]['unet_attns'], as it will be used to compute the spatial weights.
                    loss += self.fg_bg_complementary_loss_weight * mix_fg_bg_complementary_loss
                    loss_dict.update({f'{prefix}/fg_bg_complem_mix': mix_fg_bg_complementary_loss.item()})

                # unet_attns is a dict as: layer_idx -> attn_mat. 
                # It contains the 6 specified conditioned layers of UNet attentions, 
                # i.e., layers 7, 8, 12, 16, 17, 18.
                # unet_attns_mix_single = cond[2]['unet_attns']
                # Use the top-most captured attn layer to get the subject attention.
                attn_layer_idx = 17
                # [4, 8, 256, 77] / [4, 8, 64, 77] =>
                # [4, 77, 8, 256] / [4, 77, 8, 64]
                attn_mat_mix_single = unet_attns_mix_single[attn_layer_idx].permute(0, 3, 1, 2)
                # subj_indices0: ([0, 1], [6, 7]).
                # subj_attn: [2, 8, 256 / 64] (only consider the first embedding 
                # even if there are mult-embeddings for the subject token).
                subj_attn_mix_single = attn_mat_mix_single[subj_indices0]
                # spatial_weight_mix_single: [2, 1, 64, 64].
                # model_output, target:      [2, 4, 64, 64].
                spatial_weight_mix_single, spatial_attn_mix_single = \
                    convert_attn_to_spatial_weight(subj_attn_mix_single, 
                                                   x_start.shape[0], 
                                                   x_start.shape[2:],
                                                   reversed=False)  
                
                # Release RAM.
                del cond_mix[2]['unet_attns'], cond_mix[2]['unet_feats']
                # Inverse of the average weight of the foreground pixels, i.e. those
                # with spatial_weight_mix_single > 1. 
                # After applying fg_inv_scale, foreground pixels will receive similar weights
                # as without using attentional weighting. But background pixels will be
                # far less weighted, as their weights in spatial_weight_mix_single < 1, and 
                # are further scaled down by fg_inv_scale.
                fg_inv_scale = (spatial_weight_mix_single > 1.).sum() \
                                / ((spatial_weight_mix_single > 1.) * spatial_weight_mix_single).sum() 
                self.l_simple_weight_iter = self.l_simple_weight * fg_inv_scale
                loss_dict.update({f'{prefix}/fg_inv_scale': fg_inv_scale.item()})

                # Weight the loss at each pixel with the attention-derived spatial weights.          
                model_output = model_output * spatial_weight_mix_single
                target       = target       * spatial_weight_mix_single
            else:
                self.l_simple_weight_iter = self.l_simple_weight

            # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean().detach()})

            logvar_t = self.logvar.to(self.device)[t]
            # In theory, the loss can be weighted according to t. However in practice,
            # all the weights are the same, so this step is useless.
            loss_simple = loss_simple / torch.exp(logvar_t) + logvar_t
            # loss = loss_simple / torch.exp(self.logvar) + self.logvar
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss_simple.mean()})
                loss_dict.update({'logvar': self.logvar.data.mean().detach()})

            # If do_attn_recon_loss_iter, then 
            # l_simple_weight_iter = l_simple_weight (1) * fg_inv_scale, i.e.,
            # the recon loss is scaled down by the average weight of focused pixels.
            # The background pixels will be applied with a weight of << 1.
            loss = self.l_simple_weight_iter * loss_simple.mean()

            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.detach()})
            # original_elbo_weight = 0, so that loss_vlb is disabled.
            loss += (self.original_elbo_weight * loss_vlb)

        # is_comp_iter <=> calc_clip_loss. But we keep this redundancy for possible flexibility.
        if self.iter_flags['is_comp_iter'] and self.iter_flags['calc_clip_loss']:
            # Images generated both under subj_comp_prompts and mix_comp_prompts 
            # are subject to the CLIP text-image matching evaluation.
            # If self.iter_flags['do_teacher_filter'] (implying do_comp_prompt_mix_reg), 
            # the batch is (subj_comp_emb, subj_comp_emb, mix_comp_emb,  mix_comp_emb).
            # So cls_comp_prompts is used to compute the CLIP text-image matching loss on
            # images guided by the subject or mixed embeddings.
            if self.iter_flags['do_teacher_filter']:
                clip_images_code  = x_recon
                # 4 sets of cls_comp_prompts for (subj comp 1, subj comp 2, mix comp 1, mix comp 2).                
                clip_prompts_comp = cond[2]['cls_comp_prompts'] * 4            
            else:
                # Either self.iter_flags['reuse_init_conds'], or an ada delta loss iter.
                # A batch of either type has the (subj_single, subj_comp, mix_single, mix_comp) structure.
                # Only evaluate the CLIP loss on the comp images. 
                # So the batch size of clip_images_code is halved.
                x_recon_subj_single, x_recon_subj_comp, x_recon_mix_single, x_recon_mix_comp = \
                    x_recon.chunk(4)
                clip_images_code = torch.cat([x_recon_subj_comp, x_recon_mix_comp], dim=0)
                clip_prompts_comp = cond[2]['cls_comp_prompts'] * 2

            if len(clip_images_code) != len(clip_prompts_comp):
                breakpoint()

            # Use CLIP loss as a metric to evaluate the compositionality of the generated images 
            # and do distillation selectively.
            # DO NOT use CLIP loss to optimize the model. It will hurt the performance.
            with torch.no_grad():
                clip_images = self.decode_first_stage(clip_images_code)
                losses_clip_comp   = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_comp,   
                                                                                     clip_images,  
                                                                                     reduction='diag')
                #print(clip_prompts_comp)

            # Instances are arranged as: 
            # (subj comp 1, subj comp 2, mix comp 1, mix comp 2).
            losses_clip_subj_comp, losses_clip_mix_comp \
                = losses_clip_comp.chunk(2)
            
            if not self.iter_flags['reuse_init_conds']:
                loss_dict.update({f'{prefix}/loss_clip_subj_comp': losses_clip_subj_comp.mean()})
                loss_dict.update({f'{prefix}/loss_clip_cls_comp':  losses_clip_mix_comp.mean()})
            else:
                loss_dict.update({f'{prefix}/reuse_loss_clip_subj_comp': losses_clip_subj_comp.mean()})
                loss_dict.update({f'{prefix}/reuse_loss_clip_cls_comp':  losses_clip_mix_comp.mean()})

            if self.iter_flags['do_teacher_filter'] or self.iter_flags['reuse_init_conds']:
                # Discard instances that seem to be too far from the text 
                # (it may be a bit premature to make this decision now, as the images are only denoised once).
                # 0.35/0.006: 30%-40% instances will meet these thresholds.
                # 0.33/0.008: 15% instances will meet these thresholds.
                clip_loss_thres             = 0.28
                cls_subj_clip_margin        = 0.002
                clip_loss_thres_base        = 0.26
                cls_subj_clip_margin_base   = 0.003

                # are_insts_teachable: The teacher instances are only teachable if both 
                # the teacher and student are qualified (<= clip_loss_thres), 
                # and the compositional clip loss is smaller than the student.
                # If the student is qualified (<= clip_loss_thres), and the 
                # teacher is teachable, then the teacher is also qualified, 
                # as it has to have a smaller loss to be teachable. 
                # So only need to check losses_clip_subj_comp against clip_loss_thres.
                loss_diffs_subj_mix = losses_clip_subj_comp - losses_clip_mix_comp
                are_insts_teachable = (losses_clip_subj_comp <= clip_loss_thres) & (loss_diffs_subj_mix > cls_subj_clip_margin)
                # print(losses_clip_subj_comp, losses_clip_mix_comp)
                # If any of the two instances is teachable, we consider it as a teachable iteration,
                # and select the better one (with larger loss diff) to teach.
                self.iter_flags['is_teachable']  = are_insts_teachable.sum() > 0
                # Number of filtered pairs is half of the batch size. 
                # Repeat to match the number of instances in the batch.
                # log_image_flags: take values in {0, 1, 2, 3}, 
                # i.e., no box, green, red, purple boxes respectively.
                # If self.iter_flags['do_teacher_filter'] and an instance is teachable, then in the log image file, 
                # log_image_flag = 1, so the instance has a green bounary box in the logged image.
                log_image_flags = are_insts_teachable.repeat(2).int()
                if self.iter_flags['reuse_init_conds']:
                    # If reuse_init_conds and an instance is teachable, then in the log image file,
                    # log_image_flag = 3, so the instance has a purple bounary box in the logged image.
                    log_image_flags += 2

                self.num_total_teacher_filter_iters += 1
                self.num_teachable_iters += int(self.iter_flags['is_teachable'])
                teachable_frac = self.num_teachable_iters / self.num_total_teacher_filter_iters
                loss_dict.update({f'{prefix}/teachable_frac': teachable_frac})
                self.num_total_reuse_filter_iters += int(self.iter_flags['reuse_init_conds'])
                self.num_reuse_teachable_iters += int(self.iter_flags['reuse_init_conds'] and self.iter_flags['is_teachable'])
                reuse_teachable_frac = self.num_reuse_teachable_iters / self.num_total_reuse_filter_iters 
                loss_dict.update({f'{prefix}/reuse_teachable_frac': reuse_teachable_frac})

                # Only do distillation if at least one of teacher instances is teachable.
                if self.iter_flags['do_teacher_filter'] and self.iter_flags['is_teachable']:
                    # No need the intermediates of the twin-comp instances. Release them to save RAM.
                    self.release_plosses_intermediates(locals())
                    # reset_ada_embedding_cache() will implicitly clear the cache.
                    self.embedding_manager.reset_ada_embedding_cache()

                    better_cand_idx = torch.argmax(loss_diffs_subj_mix)
                    loss_clip_subj_comp = losses_clip_subj_comp[better_cand_idx]
                    loss_diff_subj_mix  = loss_diffs_subj_mix[better_cand_idx]
                    # If loss_clip_subj_comp <= clip_loss_thres, then distill_loss_clip_discount = exp(0) = 1.
                    # If loss_clip_subj_comp = 0.35, distill_loss_clip_discount = exp(-2)   = 0.14.
                    #                          0.34                               exp(-0.1) = 0.37.
                    self.distill_loss_clip_discount = 1 / np.exp( max(loss_clip_subj_comp.cpu().item() - clip_loss_thres_base, 0) / 0.01 
                                                                  +
                                                                  max(cls_subj_clip_margin_base - loss_diff_subj_mix.cpu().item(), 0) / 0.001 )

                    # Choose the x_start, noise, and t of the better candidate. 
                    # Repeat 4 times and use them as the condition to do denoising again.
                    x_start = x_start[better_cand_idx].repeat(4, 1, 1, 1)
                    noise   = noise[better_cand_idx].repeat(4, 1, 1, 1)
                    t       = t[better_cand_idx].repeat(4)
                    # Use the original cond, which is organized as 
                    # (subj single, subj comp, mix single, mix comp).

                    # do_recon=True: return denoised images x_recon. If cfg_scale > 1, 
                    # do classifier-free guidance, so that x_recon are better instances
                    # to be used to initialize the next reuse_init comp iteration.
                    # student prompts are subject prompts. So in cond_orig, 
                    # use_conv_attn = self.use_conv_attn.
                    model_output, x_recon, ada_embeddings = \
                        self.guided_denoise(x_start, noise, t, cond_orig, unet_has_grad=True, 
                                            do_recon=True, cfg_scales=cfg_scales_for_clip_loss)
                    # Cache x_recon for the next iteration with a smaller t.
                    # Note the 4 types of prompts have to be the same as this iter, 
                    # since this x_recon was denoised under this cond.
                    # Use the mix half of the batch, chunk(2)[1], instead of the subject half, chunk(2)[0], as 
                    # the mix half is better at composition (it's worse on subject authenticity, but that's fine).
                    x_start_reuse = x_recon.detach().chunk(2)[1].repeat(2, 1, 1, 1)
                    # We cannot simply use cond_orig[1], as they are (subj single, subj comp, mix single, mix comp).
                    # mix single = class single, but mix comp = subj comp.
                    self.cached_inits = { 'x_start':        x_start_reuse, 
                                          'delta_prompts':  cond_orig[2]['delta_prompts'],
                                          't':              t }
                    # Do not put cached_inits_available on self.iter_flags, as iter_flags will be reset
                    # in the beginning of the next iteration.
                    self.cached_inits_available = True

                elif not self.iter_flags['is_teachable']:
                    # If not is_teachable, do not do distillation this time 
                    # (since both instances are not good teachers), 
                    # so only compute emb reg and static/ada delta loss.
                    self.release_plosses_intermediates(locals())

                    # In an self.iter_flags['do_teacher_filter'], and not teachable instances are found.
                    # guided_denoise() above is done on twin comp instances, 
                    # and we've computed twin_comp_ada_embeddings.
                    if self.iter_flags['do_teacher_filter']:
                        # twin_comp_ada_embeddings is within no_grad(), so it won't receive gradients.
                        # Nonetheless, the delta loss takes both twin_single_ada_embeddings 
                        # and twin_comp_ada_embeddings as input, the ada embeddings 
                        # can still be optimized through twin_single_ada_embeddings.
                        twin_comp_ada_embeddings = ada_embeddings
                        # Ada delta loss requires twin subj-single embeddings that match the 
                        # previously cached twin comp embeddings.
                        # Prepare for twin subj-single condition. Only use the ada embeddings for delta loss.
                        # BS = 2. Corresponds to the twin comp instances (subj comp 1, subj comp 2).
                        c_static_emb3 = subj_single_emb.repeat(2, 1, 1, 1)
                        c_in3 = subj_single_prompts * 2
                        cond_single = (c_static_emb3, c_in3, extra_info)
                        # Previously retured ada_embeddings from guided_denoise() is twin_comp_ada_embeddings. 
                        # twin_comp_ada_embeddings: [4, 16, 77, 768], 
                        # the two sets of ada embeddings with compositional prompts.
                        self.embedding_manager.reset_ada_embedding_cache()

                        # Only take the first half of the batch, as the init conditions 
                        # of the second half is a repeat of the first half.
                        x_start_ = x_start.chunk(2)[0]
                        noise_   = noise.chunk(2)[0]
                        t_       = t.chunk(2)[0]

                        # twin_single_ada_embeddings: [2, 16, 77, 768]. BS = 2, 
                        # as it's only the twin subj-single instances. 
                        # (cls-single ada embeddings are the same as cls-single static embeddings, 
                        # so no need to generate them)
                        # twin_single_ada_embeddings has gradients, as has_grad=True by default.
                        # The iter_type is still 'mix_hijk'. But it should output the same results as 
                        # normal prompt embeddings, since subj_single_emb 
                        # consists of identical q, k embeddings.
                        # Both prompts are subj-single prompts, so in cond_single 
                        # (inherits extra_info from cond),
                        # use_conv_attn = self.use_conv_attn.
                        model_output, x_recon, twin_single_ada_embeddings = \
                            self.guided_denoise(x_start_, noise_, t_, cond_single,
                                                unet_has_grad=False, crossattn_force_grad=True)
                        
                        # No need to concatenate these two, instead let calc_prompt_delta_loss() handle the intricacy.
                        ada_embeddings = (twin_single_ada_embeddings, twin_comp_ada_embeddings)

                        self.release_plosses_intermediates(locals())

                    # Otherwise, it's an reuse_init_conds iter, and no teachable instances are found.
                    # We've computed the ada embeddings for the 4-type instances, 
                    # so just use the existing ada_embeddings (but avoid distillation).

                # Otherwise, it's an reuse_init_conds iter, and a teachable instance is found.
                # Do nothing.

            else:
                # Not self.iter_flags['do_teacher_filter'], nor reuse_init_conds. 
                # Only one possibility: not self.do_clip_teacher_filtering.
                # The teacher instance is always teachable as filtering is totally disabled.
                self.iter_flags['is_teachable'] = True
                self.distill_loss_clip_discount = 1
                # Since not self.iter_flags['do_teacher_filter']/reuse_init_conds, log_image_flags are all 0,
                # i.e., no box to be drawn on the images in cache_and_log_generations().
                log_image_flags = torch.zeros(clip_images.shape[0], device=x_start.device)

            self.cache_and_log_generations(clip_images, log_image_flags)

        else:
            # Not is_comp_iter. Distillation won't be done in this iter, so is_teachable = False.
            self.iter_flags['is_teachable'] = False
            
        if self.embedding_reg_weight > 0:
            loss_embedding_reg = self.embedding_manager.embedding_to_loss().mean()
            loss_dict.update({f'{prefix}/loss_emb_reg': loss_embedding_reg.detach()})
            loss += (self.embedding_reg_weight * loss_embedding_reg)

        if self.do_static_prompt_delta_reg:
            # do_ada_prompt_delta_reg controls whether to do ada comp delta reg here.
            static_delta_loss, ada_delta_loss, subj_emb_diff_loss = \
                self.embedding_manager.calc_prompt_delta_loss( 
                                        self.c_static_emb, ada_embeddings,
                                        self.iter_flags['do_ada_prompt_delta_reg'],
                                        extra_info['subj_indices0']
                                       )
            
            loss_dict.update({f'{prefix}/static_delta_loss': static_delta_loss.mean().detach()})
            if ada_delta_loss != 0:
                loss_dict.update({f'{prefix}/ada_delta_loss': ada_delta_loss.mean().detach()})
            if subj_emb_diff_loss != 0:
                loss_dict.update({f'{prefix}/subj_emb_diff_loss': subj_emb_diff_loss.mean().detach()})

            # The prompt delta loss for ada embeddings is only applied 
            # every self.composition_regs_iter_gap iterations. So the ada loss 
            # should be boosted proportionally to composition_regs_iter_gap. 
            # Divide it by 2 to reduce the proportion of ada emb loss relative to 
            # static emb loss in the total loss.                
            ada_comp_loss_boost_ratio = self.composition_regs_iter_gap / 2
            subj_emb_diff_loss_weight = 0.002
            loss_prompt_delta_reg = static_delta_loss + ada_comp_loss_boost_ratio * ada_delta_loss \
                                    + subj_emb_diff_loss * subj_emb_diff_loss_weight
            loss += (self.prompt_delta_reg_weight * loss_prompt_delta_reg)
        
        if self.iter_flags['do_comp_prompt_mix_reg'] and self.iter_flags['is_teachable']:
            # unet_feats is a dict as: layer_idx -> unet_feat. 
            # It contains all the intermediate 25 layers of UNet features.
            unet_feats = cond[2]['unet_feats']
            # unet_attns is a dict as: layer_idx -> attn_mat. 
            # It contains the 6 specified conditioned layers of UNet attentions, 
            # i.e., layers 7, 8, 12, 16, 17, 18.
            unet_attns = cond[2]['unet_attns']
            loss_subj_attn_delta_distill, loss_subj_attn_norm_distill, loss_feat_distill = \
                                self.calc_prompt_mix_loss(unet_feats, unet_attns, 
                                                          self.embedding_manager.placeholder_indices_fg,
                                                          HALF_BS)
            
            loss_dict.update({f'{prefix}/loss_feat_distill': loss_feat_distill.detach()})
            loss_dict.update({f'{prefix}/loss_subj_attn_delta_distill': loss_subj_attn_delta_distill.detach()})
            loss_dict.update({f'{prefix}/loss_subj_attn_norm_distill': loss_subj_attn_norm_distill.detach()})

            # In a reused init iter, the denoise image may not look so authentic, so
            # it receives a smaller weight.
            distill_feat_weight      = 0.5 if (not self.iter_flags['reuse_init_conds']) else 0.3
            # Set to 0 to disable distillation on attention weights of the subject.
            distill_subj_attn_weight = 0.4

            loss_prompt_mix_reg =   (loss_subj_attn_delta_distill + loss_subj_attn_norm_distill) * distill_subj_attn_weight + \
                                    loss_feat_distill      * distill_feat_weight 
            
            loss += self.composition_prompt_mix_reg_weight * loss_prompt_mix_reg * self.distill_loss_scale * self.distill_loss_clip_discount
            self.release_plosses_intermediates(locals())

        loss_dict.update({f'{prefix}/loss': loss.detach()})

        return loss, loss_dict

    def calc_prompt_mix_loss(self, unet_feats, unet_attns, placeholder_indices, HALF_BS):
        # do_comp_prompt_mix_reg iterations. No ordinary image reconstruction loss.
        # Only regularize on intermediate features, i.e., intermediate features generated 
        # under subj_comp_prompts should satisfy the delta loss constraint:
        # F(subj_comp_prompts)  - F(mix(subj_comp_prompts, cls_comp_prompts)) \approx 
        # F(subj_single_prompts) - F(cls_single_prompts)
        loss_subj_attn_delta_distill = 0
        loss_subj_attn_norm_distill  = 0
        loss_feat_distill      = 0

        delta_attn_loss_scale    = 1
        direct_attn_loss_scale   = 2
        # The norm is actually the abs().mean(), so it has small magnitudes and should be scaled up.
        direct_attn_norm_loss_scale = 6

        # Discard top layers and the first few bottom layers from distillation.
        # distill_layer_weights: relative weight of each distillation layer. 
        # distill_layer_weights are normalized using distill_overall_weight.
        # Conditioning layers are 7, 8, 12, 16, 17. All the 5 layers have 1280 channels.
        # But intermediate layers also contribute to distillation. They have small weights.
        # Layer 16 has strong face semantics, so it is given a small weight.
        feat_distill_layer_weights = { 7:  1., 8: 1.,   
                                       #9:  0.5, 10: 0.5, 11: 0.5, 
                                       12: 0.5, 
                                       16: 0.5, 17: 0.5,
                                     }

        attn_distill_layer_weights = { 7:  1., 8: 1.,
                                       #9:  0.5, 10: 0.5, 11: 0.5,
                                       12: 1.,
                                       16: 1., 17: 1.,
                                     }
        
        # Normalize the weights above so that each set sum to 1.
        feat_distill_layer_weight_sum = np.sum(list(feat_distill_layer_weights.values()))
        feat_distill_layer_weights = { k: v / feat_distill_layer_weight_sum for k, v in feat_distill_layer_weights.items() }
        attn_distill_layer_weight_sum = np.sum(list(attn_distill_layer_weights.values()))
        attn_distill_layer_weights = { k: v / attn_distill_layer_weight_sum for k, v in attn_distill_layer_weights.items() }
        
        orig_placeholder_ind_B, orig_placeholder_ind_T = placeholder_indices
        # The class prompts are at the latter half of the batch.
        # So we need to add the batch indices of the subject prompts, to locate
        # the corresponding class prompts.
        cls_placeholder_ind_B = orig_placeholder_ind_B + HALF_BS * 2
        # Concatenate the placeholder indices of the subject prompts and class prompts.
        placeholder_indices_B = torch.cat([orig_placeholder_ind_B, cls_placeholder_ind_B ], dim=0)
        placeholder_indices_T = torch.cat([orig_placeholder_ind_T, orig_placeholder_ind_T], dim=0)
        # placeholder_indices: 
        # ( tensor([0,  0,   1, 1,   2, 2,   3, 3]), 
        #   tensor([6,  7,   6, 7,   6, 7,   6, 7]) )
        placeholder_indices = (placeholder_indices_B, placeholder_indices_T)
        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(orig_placeholder_ind_B) // len(torch.unique(orig_placeholder_ind_B))

        mix_feat_grad_scale = 0.1
        mix_feat_grad_scaler = gen_gradient_scaler(mix_feat_grad_scale)
        # mix_attn_grad_scale = 0.01, almost zero, effectively no grad to teacher attn. 
        # Setting to 0 may prevent the graph from being released and OOM.
        mix_attn_grad_scale  = 0.01  
        mix_attn_grad_scaler = gen_gradient_scaler(mix_attn_grad_scale)

        for unet_layer_idx, unet_feat in unet_feats.items():
            if (unet_layer_idx not in feat_distill_layer_weights) and (unet_layer_idx not in attn_distill_layer_weights):
                continue

            # each is [1, 1280, 16, 16]
            feat_subj_single, feat_subj_comp, feat_mix_single, feat_mix_comp \
                = unet_feat.chunk(4)
            
            # attn_mat: [4, 8, 256, 77] => [4, 77, 8, 256].
            # We don't need BP through attention into UNet.
            attn_mat = unet_attns[unet_layer_idx].permute(0, 3, 1, 2)
            # subj_attn: [4, 8, 256] (1 embedding  for 1 token)  => [4, 1, 8, 256] mean => [4, 8, 256]
            # or         [16, 8, 256] (4 embeddings for 1 token) => [4, 4, 8, 256] mean => [4, 8, 256]
            # mean is taken across the multiple subject tokens.
            subj_attn = attn_mat[placeholder_indices].reshape(HALF_BS*4, K_fg, *attn_mat.shape[2:]).mean(dim=1)
            # subj_attn_subj_single, ...: [1, 8, 256] (1 embedding  for 1 token) 
            # or                          [1, 8, 256] (4 embeddings for 1 token)
            subj_attn_subj_single, subj_attn_subj_comp, subj_attn_mix_single,  subj_attn_mix_comp \
                = subj_attn.chunk(4)

            if (unet_layer_idx in attn_distill_layer_weights):
                attn_distill_layer_weight = attn_distill_layer_weights[unet_layer_idx]
                attn_subj_delta = subj_attn_subj_comp - subj_attn_subj_single
                attn_mix_delta  = subj_attn_mix_comp  - subj_attn_mix_single
                # Setting exponent as 2 seems to push too hard restriction on subject embeddings 
                # towards class embeddings, hurting authenticity.
                loss_layer_subj_delta_attn = calc_delta_loss(attn_subj_delta, attn_mix_delta, 
                                                             exponent=3,
                                                             first_n_dims_to_flatten=2, 
                                                             ref_grad_scale=0.01)
                
                # mix_attn_grad_scale = 0.01, almost zero, effectively no grad to subj_attn_mix_comp/subj_attn_mix_single. 
                # Use this scaler to release the graph and avoid OOM.
                subj_attn_mix_comp_gs   = mix_attn_grad_scaler(subj_attn_mix_comp)
                subj_attn_mix_single_gs = mix_attn_grad_scaler(subj_attn_mix_single)
                loss_layer_subj_comp_attn   = self.get_loss(subj_attn_subj_comp,   subj_attn_mix_comp_gs,   mean=True)
                loss_layer_subj_single_attn = self.get_loss(subj_attn_subj_single, subj_attn_mix_single_gs, mean=True)

                # Align the attention corresponding to each embedding individually.
                loss_layer_subj_comp_attn_norm   = (subj_attn_subj_comp.mean(dim=(1,2))   - subj_attn_mix_comp_gs.mean(dim=(1,2))).abs().mean()
                loss_layer_subj_single_attn_norm = (subj_attn_subj_single.mean(dim=(1,2)) - subj_attn_mix_single_gs.mean(dim=(1,2))).abs().mean()
                # print(loss_layer_subj_comp_attn_norm, loss_layer_subj_single_attn_norm)

                # loss_layer_subj_attn_distill = self.get_loss(attn_subj_delta, attn_mix_delta, mean=True)
                # L2 loss tends to be smaller than delta loss. So we scale it up by 10.
                loss_subj_attn_delta_distill += ( loss_layer_subj_delta_attn * delta_attn_loss_scale \
                                                + (loss_layer_subj_comp_attn + loss_layer_subj_single_attn) * direct_attn_loss_scale \
                                                ) * attn_distill_layer_weight
                loss_subj_attn_norm_distill  += ( loss_layer_subj_comp_attn_norm + loss_layer_subj_single_attn_norm ) * direct_attn_norm_loss_scale * attn_distill_layer_weight

            use_subj_attn_as_spatial_weights = True
            if use_subj_attn_as_spatial_weights:
                if unet_layer_idx not in feat_distill_layer_weights:
                    continue
                feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]

                # feat_subj_single, ...: [1, 1280, 16, 16]
                feat_subj_single, feat_subj_comp, feat_mix_single, feat_mix_comp \
                    = unet_feat.chunk(4)

                # convert_attn_to_spatial_weight() will detach attention weights to 
                # avoid BP through attention.
                spatial_weight_mix_comp, spatial_attn_mix_comp   = convert_attn_to_spatial_weight(subj_attn_mix_comp, HALF_BS, 
                                                                                                  feat_mix_comp.shape[2:],
                                                                                                  reversed=True)

                spatial_weight_subj_comp, spatial_attn_subj_comp = convert_attn_to_spatial_weight(subj_attn_subj_comp, HALF_BS,
                                                                                                  feat_subj_comp.shape[2:],
                                                                                                  reversed=True)
                spatial_weight = (spatial_weight_mix_comp + spatial_weight_subj_comp) / 2

                # spatial_attn_mix_comp, spatial_attn_subj_comp are returned for debugging purposes. 
                # Delete them to release RAM.
                del spatial_attn_mix_comp, spatial_attn_subj_comp

                # Use mix single/comp weights on both subject-only and mix features, 
                # to reduce misalignment and facilitate distillation.
                feat_subj_single = feat_subj_single * spatial_weight
                feat_subj_comp   = feat_subj_comp   * spatial_weight
                feat_mix_single  = feat_mix_single  * spatial_weight
                feat_mix_comp    = feat_mix_comp    * spatial_weight

            do_feat_pooling = True
            feat_pool_kernel_size = 4
            feat_pool_stride      = 2
            # feature pooling: allow small perturbations of the locations of pixels.
            if do_feat_pooling:
                pooler = nn.AvgPool2d(feat_pool_kernel_size, stride=feat_pool_stride)
            else:
                pooler = nn.Identity()

            # Pool the H, W dimensions to remove spatial information.
            # After pooling, feat_subj_single, feat_subj_comp, 
            # feat_mix_single, feat_mix_comp: [1, 1280] or [1, 640], ...
            feat_subj_single = pooler(feat_subj_single).reshape(feat_subj_single.shape[0], -1)
            feat_subj_comp   = pooler(feat_subj_comp).reshape(feat_subj_comp.shape[0], -1)
            feat_mix_single  = pooler(feat_mix_single).reshape(feat_mix_single.shape[0], -1)
            feat_mix_comp    = pooler(feat_mix_comp).reshape(feat_mix_comp.shape[0], -1)

            feat_mix_single  = mix_feat_grad_scaler(feat_mix_single)
            feat_mix_comp    = mix_feat_grad_scaler(feat_mix_comp)

            distill_on_delta = True
            if distill_on_delta:
                # ortho_subtract is in terms of the last dimension. So we pool the spatial dimensions first above.
                feat_mix_delta  = ortho_subtract(feat_mix_comp,  feat_mix_single)
                feat_subj_delta = ortho_subtract(feat_subj_comp, feat_subj_single)
            else:
                feat_mix_delta  = feat_mix_comp
                feat_subj_delta = feat_subj_comp
                
            # feat_subj_delta, feat_mix_delta: [1, 1280], ...
            # Pool the spatial dimensions H, W to remove spatial information.
            # The gradient goes back to feat_subj_delta -> feat_subj_comp,
            # as well as feat_mix_delta -> feat_mix_comp.
            # If stop_single_grad, the gradients to feat_subj_single and feat_mix_single are stopped, 
            # as these two images should look good by themselves (since they only contain the subject).
            # Note the learning strategy to the single image features should be different from 
            # the single embeddings, as the former should be optimized to look good by itself,
            # while the latter should be optimized to cater for two objectives: 1) the conditioned images look good,
            # and 2) the embeddings are amendable to composition.
            loss_layer_feat_distill = self.get_loss(feat_subj_delta, feat_mix_delta, mean=True)
            
            # print(f'layer {unet_layer_idx} loss: {loss_layer_prompt_mix_reg:.4f}')
            loss_feat_distill += loss_layer_feat_distill * feat_distill_layer_weight

        return loss_subj_attn_delta_distill, loss_subj_attn_norm_distill, loss_feat_distill

    def calc_fg_bg_complementary_loss(self, unet_attns, 
                                      placeholder_indices_fg, 
                                      placeholder_indices_bg, 
                                      BS, fg_grad_scale=0.01,
                                      do_demean_first=False):

        # Discard top layers and the first few bottom layers from distillation.
        # distill_layer_weights: relative weight of each distillation layer. 
        # distill_layer_weights are normalized using distill_overall_weight.
        # Conditioning layers are 7, 8, 12, 16, 17. All the 5 layers have 1280 channels.
        # But intermediate layers also contribute to distillation. They have small weights.
        # Layer 16 has strong face semantics, so it is given a small weight.
        attn_distill_layer_weights = { #7:  1., 8: 1.,
                                       #9:  0.5, 10: 0.5, 11: 0.5,
                                       12: 1.,
                                       16: 1., 17: 1., 
                                       18: 1.,
                                     }
        
        # Normalize the weights above so that each set sum to 1.
        attn_distill_layer_weight_sum = np.sum(list(attn_distill_layer_weights.values()))
        attn_distill_layer_weights = { k: v / attn_distill_layer_weight_sum for k, v in attn_distill_layer_weights.items() }
        # K_fg: 4, number of embeddings per subject token.
        K_fg = len(placeholder_indices_fg[0]) // len(torch.unique(placeholder_indices_fg[0]))
        # K_bg: 1 or 2, number of embeddings per background token.
        K_bg = len(placeholder_indices_bg[0]) // len(torch.unique(placeholder_indices_bg[0]))
        assert self.num_vectors_per_token == K_fg

        loss_fg_bg_complementary = 0

        for unet_layer_idx, unet_attn in unet_attns.items():
            if (unet_layer_idx not in attn_distill_layer_weights):
                continue

            # [2, 8, 256, 77] / [2, 8, 64, 77] =>
            # [2, 77, 8, 256] / [2, 77, 8, 64]
            attn_mat = unet_attn.permute(0, 3, 1, 2)
            # In each instance, placeholder_indices_fg has K_fg times as many elements as placeholder_indices_bg.
            # placeholder_indices_fg: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
            #                          [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
            # placeholder_indices_bg: ([0, 1, 2, 3, 4, 5, 6, 7], [11, 12, 34, 29, 11, 12, 34, 29]).
            # BS = 2, so we only keep instances indexed by [0, 1].
            # placeholder_indices_fg: ([0, 0, 0, 0, 1, 1, 1, 1], [5, 6, 7, 8, 6, 7, 8, 9]).
            placeholder_indices_fg = (placeholder_indices_fg[0][:BS*K_fg], placeholder_indices_fg[1][:BS*K_fg])
            # placeholder_indices_bg: ([0, 1], [11, 12]).
            placeholder_indices_bg = (placeholder_indices_bg[0][:BS*K_bg], placeholder_indices_bg[1][:BS*K_bg])
            # subj_attn: [8, 8, 64] -> [2, 4, 8, 64] mean among K_fg embeddings -> [2, 8, 64]
            subj_attn = attn_mat[placeholder_indices_fg].reshape(BS, K_fg, *attn_mat.shape[2:]).mean(dim=1)
            # bg_attn:   [4, 8, 64] -> [2, 2, 8, 64] mean among K_bg embeddings -> [2, 8, 64]
            # 8: 8 attention heads. Last dim 64: number of image tokens.
            bg_attn   = attn_mat[placeholder_indices_bg].reshape(BS, K_bg, *attn_mat.shape[2:]).mean(dim=1)

            attn_distill_layer_weight = attn_distill_layer_weights[unet_layer_idx]
            # Align bg_attn with (1 - subj_attn), so that the two attention maps are complementary.
            # exponent = 2: exponent is 3 by default, which lets the loss focus on large activations.
            # But we don't want to only focus on large activations. So set it to 2.
            # do_demean_first: remove the means from both embeddings before calculating the delta loss.
            # This normalizes (1 - subj_attn), since most elements in subj_attn are almost 0.
            # ref_grad_scale = 0.05: small gradients will be BP-ed to the subject embedding,
            # to make the two attention maps more complementary (expect the loss pushes the 
            # subject embedding to a more accurate point).
            loss_layer_fg_bg_comple = calc_delta_loss(bg_attn, subj_attn.max().detach() - subj_attn, 
                                                      exponent=2,    
                                                      do_demean_first=do_demean_first,  # False
                                                      first_n_dims_to_flatten=2, 
                                                      ref_grad_scale=fg_grad_scale,
                                                      debug=False)
            
            loss_fg_bg_complementary += loss_layer_fg_bg_comple * attn_distill_layer_weight

        # If not doing demean, the loss will be smaller in magnitude, so we scale it up.
        # Do demean: loss_fg_bg_complementary is 1 ~ 1.2. 
        # No demean: loss_fg_bg_complementary is 0.2 ~ 0.3.
        if not do_demean_first:
            loss_fg_bg_complementary *= 3

        return loss_fg_bg_complementary

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
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    def cache_and_log_generations(self, samples, img_flags, max_cache_size=48):
        self.generation_cache.append(samples)
        self.generation_cache_img_flags.append(img_flags)
        self.num_cached_generations += len(samples)

        if self.num_cached_generations >= max_cache_size:
            grid_folder = self.logger._save_dir + f'/samples'
            os.makedirs(grid_folder, exist_ok=True)
            grid_filename = grid_folder + f'/{self.cache_start_iter:04d}-{self.global_step:04d}.png'
            save_grid(self.generation_cache, self.generation_cache_img_flags, 
                      grid_filename, 12, do_normalize=True)
            print(f"{self.num_cached_generations} generations saved to {grid_filename}")
            
            # Clear the cache. If num_cached_generations > max_cache_size,
            # some samples at the end of the cache will be discarded.
            self.generation_cache = []
            self.generation_cache_img_flags = []
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
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
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
        # self.learning_rate and self.weight_decay are set in main.py.
        # self.learning_rate = base_learning_rate * 2, 2 is the batch size.
        lr = self.learning_rate
        model_lr = self.model_lr
        weight_decay = self.weight_decay

        if self.optimizer_type == 'sophia':
            lr = lr / 2
            model_lr = model_lr / 2
            OptimizerClass = SophiaG
        else:
            OptimizerClass = torch.optim.AdamW

        # If using textual inversion, then embedding_manager is not None.
        if self.embedding_manager is not None: 
            embedding_params = list(self.embedding_manager.optimized_parameters())
            # unfreeze_model:
            # Are we allowing the base model to train? If so, set two different parameter groups.
            if self.unfreeze_model: 
                model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
                opt = OptimizerClass([{"params": embedding_params, "lr": lr}, {"params": model_params}], lr=model_lr)
            # Otherwise, train only embedding
            else:
                opt = OptimizerClass(embedding_params, lr=lr, weight_decay=weight_decay)
        else:
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.ap

            opt = OptimizerClass(params, lr=lr, weight_decay=weight_decay)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            
            self.scheduler = scheduler[0]['scheduler']
            return [opt], scheduler
        
        return opt

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
