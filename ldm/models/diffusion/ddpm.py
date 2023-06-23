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
                       ortho_subtract, calc_stats, rand_like, GradientScaler, \
                       calc_chan_locality, convert_attn_to_spatial_weight, calc_delta_loss, \
                       save_grid, divide_chunks

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
import math
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
                 filter_with_clip_loss=False,
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
        self.filter_with_clip_loss          = filter_with_clip_loss
        self.prompt_mix_scheme              = 'mix_hijk'
        self.use_fp_trick                   = use_fp_trick

        self.do_static_prompt_delta_reg     = False
        self.do_ada_prompt_delta_reg        = False
        self.do_comp_prompt_mix_reg         = False
        self.calc_clip_loss                 = False        
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
            self.warmup_steps     = scheduler_config.params.warm_up_steps
        else:
            self.scheduler = None
            self.warmup_steps = 500

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.embedding_reg_weight = embedding_reg_weight

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
    def create_clip_evaluator(self, device, use_noised_clip=False):
        self.use_noised_clip = use_noised_clip
        
        if self.use_noised_clip:
            self.clip_evaluator = NoisedCLIPEvaluator(device=device)
            for param in self.clip_evaluator.model.image_encoder.parameters():
                param.requires_grad = False
            for param in self.clip_evaluator.model.text_encoder.parameters():
                param.requires_grad = False
        else:
            self.clip_evaluator = CLIPEvaluator(device=device)
            for param in self.clip_evaluator.model.parameters():
                param.requires_grad = False

        self.num_total_clip_iters = 0
        self.num_teachable_iters = 0

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

    # This shared_step() is overridden by LatentDiffusion::shared_step() and never called. 
    #LINK #shared_step
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # No matter wheter the scheme is layerwise or not,
        # as long as composition_regs_iter_gap > 0 and prompt_delta_reg_weight > 0, 
        # do static comp delta reg.
        self.do_static_prompt_delta_reg = self.composition_regs_iter_gap > 0 \
                                            and self.prompt_delta_reg_weight > 0

        # How many regularizations are done intermittently during the training iterations?
        interm_reg_types = []
        interm_reg_probs = []
        # Only do comp_prompt_mix_reg after warm_up_steps (500) iterations.
        # So that subject-level prompts are already able to generate rough subject images 
        # to compute a reasonable mixing loss.
        if self.composition_prompt_mix_reg_weight > 0:
            interm_reg_types.append('do_comp_prompt_mix_reg')
            interm_reg_probs.append(2.)
        # do_ada_prompt_delta_reg only if do_static_prompt_delta_reg and use_ada_embedding,
        # and not do_comp_prompt_mix_reg.
        elif self.do_static_prompt_delta_reg and self.use_ada_embedding:
            interm_reg_types.append('do_ada_prompt_delta_reg')
            interm_reg_probs.append(1.)

        N_INTERM_REGS = len(interm_reg_types)
        interm_reg_probs = np.array(interm_reg_probs) / np.sum(interm_reg_probs)
        self.do_ada_prompt_delta_reg  = False
        self.do_comp_prompt_mix_reg   = False
        self.calc_clip_loss           = False

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
            reg_type     = interm_reg_types[reg_type_idx]
            if reg_type   == 'do_ada_prompt_delta_reg':
                self.do_comp_prompt_mix_reg   = False
                self.do_ada_prompt_delta_reg  = True
            # do_comp_prompt_mix_reg implies do_ada_prompt_delta_reg.
            elif reg_type == 'do_comp_prompt_mix_reg':
                self.do_comp_prompt_mix_reg   = True
                self.do_ada_prompt_delta_reg  = True

            self.calc_clip_loss = True

        # Borrow the LR LambdaWarmUpCosineScheduler to control the mix weight.
        if self.scheduler is not None:
            lr_scale = self.scheduler.get_last_lr()[0] / self.scheduler.base_lrs[0]
            self.distill_loss_scale = lr_scale
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
        else:
            # For DreamBooth.
            self.embedding_manager = None

        self.generation_cache = []
        self.cache_start_iter = 0
        self.num_cached_generations = 0
        
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
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
            
    
    def instantiate_embedding_manager(self, config, embedder):
        model = instantiate_from_config(config, embedder=embedder)

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
                c = self.cond_stage_model.encode(cond_in, embedding_manager=self.embedding_manager)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
                
                extra_info = { 'use_layerwise_context': self.use_layerwise_embedding, 
                               'use_ada_context':       self.use_ada_embedding,
                             }
                
                if self.use_ada_embedding:
                    ada_embedder = self.get_ada_conditioning
                    # Initialize the ada embedding cache, so that the subsequent calls to 
                    # EmbeddingManager.get_ada_embedding() will store the ada embedding 
                    # for each layer into the cache. 
                    # The cache will be used in calc_prompt_delta_loss().
                    self.embedding_manager.init_ada_embedding_cache()
                    # The image mask here is used when computing Ada embeddings in embedding_manager.
                    # Do not consider mask on compositional reg iterations.
                    if img_mask is not None and \
                        (self.do_comp_prompt_mix_reg or self.do_ada_prompt_delta_reg):
                            img_mask = None
                            """                             
                            # If do_comp_prompt_mix_reg, the image mask is also needed to repeat. 
                            HALF_BS  = max(img_mask.shape[0] // 2, 1)
                            # The image batch will be repeated 4 times in p_losses(),
                            # so img_mask is also repeated 4 times.
                            img_mask = img_mask[:HALF_BS].repeat(4, 1, 1, 1) 
                            """

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

    # get_ada_conditioning() is a callback function called iteratively by each layer in UNet
    # It returns the conditioning embedding (ada embedding & other token embeddings -> clip encoder) 
    # for the current layer to UNet.
    def get_ada_conditioning(self, c_in, layer_idx, layer_infeat, time_emb, ada_bp_to_unet):
        # We don't want to mess with the pipeline of cond_stage_model.encode(), so we pass
        # c_in, layer_idx and layer_infeat directly to embedding_manager. They will be used implicitly
        # when embedding_manager is called within cond_stage_model.encode().
        self.embedding_manager.set_ada_layer_temp_info(layer_idx, layer_infeat, time_emb, ada_bp_to_unet)
        c = self.cond_stage_model.encode(c_in, embedding_manager=self.embedding_manager)
        # Cache the computed ada embedding of the current layer for delta loss computation.
        # Before this call, init_ada_embedding_cache() should have been called somewhere.
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
        # do_static_prompt_delta_reg is applicable to Ada, Static layerwise embedding 
        # or traditional TI.        
        # do_ada_prompt_delta_reg implies do_static_prompt_delta_reg. So only check do_static_prompt_delta_reg.
        if self.do_static_prompt_delta_reg or self.do_comp_prompt_mix_reg:
            subj_comp_prompts = []
            # *_fp prompts are like "a face portrait of ...". They are advantageous over "a photo of ..."
            # when doing compositional mix regularization. 
            # However this trick is only applicable to humans/animals.
            # For objects, *_fp prompts are not available in batch, so they won't be used.
            if self.do_comp_prompt_mix_reg and self.use_fp_trick and 'subj_prompt_single_fp' in batch:
                # Replace c = batch['caption'] by c = batch['subj_prompt_single_fp'].
                c = batch['subj_prompt_single_fp']
                SUBJ_PROMPT_COMP  = 'subj_prompt_comp_fp'
                CLS_PROMPT_COMP   = 'cls_prompt_comp_fp'
                CLS_PROMPT_SINGLE = 'cls_prompt_single_fp'
            else:
                SUBJ_PROMPT_COMP  = 'subj_prompt_comp'
                CLS_PROMPT_COMP   = 'cls_prompt_comp'
                CLS_PROMPT_SINGLE = 'cls_prompt_single'

            # Each prompt_comps consists of multiple prompts separated by "|".
            # Split them into a list of subj_comp_prompts/cls_comp_prompts.
            for prompt_comps in batch[SUBJ_PROMPT_COMP]:
                subj_comp_prompts.append(prompt_comps.split("|"))
            cls_comp_prompts = []
            for prompt_comps in batch[CLS_PROMPT_COMP]:
                cls_comp_prompts.append(prompt_comps.split("|"))
            cls_single_prompts = batch[CLS_PROMPT_SINGLE]
            # REPEATS: how many prompts correspond to each image.
            REPEATS = len(subj_comp_prompts[0])
            if REPEATS == 1 or self.do_comp_prompt_mix_reg or self.do_ada_prompt_delta_reg:
                # When this iter computes ada prompt delta loss / prompt mixing loss, 
                # only use the first of the composition prompts (in effect num_compositions_per_image=1),
                # otherwise it will use more than 40G RAM.
                subj_comp_prompts = [ prompts[0] for prompts in subj_comp_prompts ]
                cls_comp_prompts  = [ prompts[0] for prompts in cls_comp_prompts ]
                prompt_delta_prompts = (subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
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
                prompt_delta_prompts = (subj_comp_prompts, cls_single_prompts, cls_comp_prompts)
        else:
            prompt_delta_prompts = None

        if 'mask' in batch:
            img_mask = batch['mask']
            img_mask = img_mask.unsqueeze(1).to(x.device)
            img_mask = F.interpolate(img_mask, size=x.shape[-2:], mode='nearest')
        else:
            img_mask = None

        loss = self(x, c, prompt_delta_prompts, img_mask=img_mask, **kwargs)
        return loss

    # LatentDiffusion.forward() is only called during training, by shared_step().
    #LINK #shared_step
    def forward(self, x, c, prompt_delta_prompts=None, img_mask=None, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
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
                if self.do_static_prompt_delta_reg or self.do_comp_prompt_mix_reg:
                    subj_comp_prompts, cls_single_prompts, cls_comp_prompts = prompt_delta_prompts

                    N_LAYERS = 16 if self.use_layerwise_embedding else 1
                    ORIG_BS  = len(x)
                    N_EMBEDS = ORIG_BS * N_LAYERS
                    # HALF_BS is at least 1. So if ORIG_BS == 1, then HALF_BS = 1.
                    HALF_BS  = max(ORIG_BS // 2, 1)

                    subj_single_prompts = c

                    if self.do_comp_prompt_mix_reg or self.do_ada_prompt_delta_reg:
                        # Only keep the first half of batched prompts to save RAM.
                        subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts = \
                            subj_single_prompts[:HALF_BS], subj_comp_prompts[:HALF_BS], \
                            cls_single_prompts[:HALF_BS],  cls_comp_prompts[:HALF_BS]
                                            
                    # PROMPT_BS = ORIG_BS * num_compositions_per_image.
                    # So when num_compositions_per_image > 1, subj_single_prompts/cls_single_prompts contains repeated prompts,
                    # subj_comp_prompts/cls_comp_prompts contains varying compositional prompts 
                    # that make PROMPT_BS > ORIG_BS.
                    PROMPT_BS = len(subj_single_prompts)
                    prompts_delta = subj_single_prompts + subj_comp_prompts \
                                    + cls_single_prompts + cls_comp_prompts
                    PROMPT_N_EMBEDS = PROMPT_BS * N_LAYERS
                    # c_static_emb: the static embeddings [4 * N_EMBEDS, 77, 768], 
                    # 4 * N_EMBEDS = 4 * ORIG_BS * N_LAYERS,
                    # whose layer dimension (N_LAYERS) is tucked into the batch dimension. 
                    # c_in: a copy of prompts_delta, the concatenation of
                    # (subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts).
                    # extra_info: a dict that contains extra info.
                    c_static_emb, c_in, extra_info = self.get_learned_conditioning(prompts_delta, img_mask=img_mask)
                    subj_single_emb, subj_comps_emb, cls_single_emb, cls_comps_emb = \
                        torch.split(c_static_emb, PROMPT_N_EMBEDS, dim=0)
                    
                    # if do_ada_prompt_delta_reg, then do_comp_prompt_mix_reg 
                    # may be True or False, depending whether mix reg is enabled.
                    if self.do_comp_prompt_mix_reg:
                        # c_in2 is used to generate ada embeddings.
                        # Arrange c_in2 in the same layout as the static embeddings.
                        # The cls_single_prompts within c_in2 will only be used to generate ordinary 
                        # prompt embeddings, i.e., 
                        # it doesn't contain subject token, and no ada embedding will be injected.
                        # despite the fact that subj_single_emb, cls_single_emb are mixed into 
                        # the corresponding static embeddings.
                        # Tried to use subj_single_prompts here, and it's worse.
                        # The last set is another subj_comp_prompts, which is NOT A BUG.
                        # This subj_comp_prompts is used to generate the ada embedding for
                        # the subj_comp_prompts, used for the mixed embeddings of 
                        # (subj_comp_prompts, cls_comp_prompts).
                        c_in2 = subj_single_prompts + subj_comp_prompts + cls_single_prompts + cls_comp_prompts
                        #print(c_in2)

                        # The static embeddings of subj_comp_prompts and cls_comp_prompts,
                        # i.e., subj_comps_emb and cls_comps_emb will be mixed (concatenated),
                        # and the token number will be the double of subj_comps_emb.
                        # Ada embeddings won't be mixed.
                        # Mixed embedding subj_comps_emb_mix = 
                        # concat(subj_comps_emb, cls_comps_emb -| subj_comps_emb)_dim1. 
                        # -| means orthogonal subtraction.
                        subj_comps_emb_mix_all_layers  = mix_embeddings(subj_comps_emb, cls_comps_emb, 
                                                                        c2_mix_weight=1,
                                                                        mix_scheme='adeltaconcat',
                                                                        use_ortho_subtract=True)
                        subj_single_emb_mix_all_layers = mix_embeddings(subj_single_emb, cls_single_emb,
                                                                        c2_mix_weight=1,
                                                                        mix_scheme='adeltaconcat',
                                                                        use_ortho_subtract=True)
                        
                        #subj_comps_emb_mix_all_layers  = cls_comps_emb
                        #subj_single_emb_mix_all_layers = cls_single_emb
                        # If stop_prompt_mix_grad, stop gradient on subj_comps_emb_mix, 
                        # since it serves as the reference.
                        # If we don't stop gradient on subj_comps_emb_mix, 
                        # then chance is that subj_comps_emb_mix might be dominated by subj_comps_emb,
                        # so that subj_comps_emb_mix will produce images similar as subj_comps_emb does.
                        # stop_prompt_mix_grad will improve compositionality but reduce face similarity.
                        stop_prompt_mix_grad = False
                        prompt_mix_grad_scale = 0.1
                        if stop_prompt_mix_grad:
                            subj_comps_emb_mix_all_layers  = subj_comps_emb_mix_all_layers.detach()
                            subj_single_emb_mix_all_layers = subj_single_emb_mix_all_layers.detach()
                        elif prompt_mix_grad_scale != 1:
                            grad_scaler = GradientScaler(prompt_mix_grad_scale)
                            subj_comps_emb_mix_all_layers  = grad_scaler(subj_comps_emb_mix_all_layers)
                            subj_single_emb_mix_all_layers = grad_scaler(subj_single_emb_mix_all_layers)

                        if self.use_layerwise_embedding:
                            # 4, 5, 6, 7, 8 correspond to original layer indices 7, 8, 12, 16, 17
                            # (same as used in computing mixing loss)
                            sync_layer_indices = [4, 5, 6, 7, 8]
                            layer_mask = torch.zeros_like(subj_comps_emb_mix_all_layers).reshape(-1, N_LAYERS, *subj_comps_emb_mix_all_layers.shape[1:])
                            layer_mask[:, sync_layer_indices] = 1
                            layer_mask = layer_mask.reshape(-1, *subj_comps_emb_mix_all_layers.shape[1:])

                            # This copy of subj_single_emb, subj_comps_emb will be simply 
                            # repeated at the token dimension to match 
                            # the token number of the mixed (concatenated) 
                            # subj_single_emb_mix and subj_comps_emb_mix embeddings.
                            subj_comps_emb  = subj_comps_emb.repeat(1, 2, 1)
                            subj_single_emb = subj_single_emb.repeat(1, 2, 1)

                            # Otherwise, the second halves of subj_comps_emb/cls_comps_emb
                            # are already key embeddings. No need to repeat.

                            # Use most of the layers of embeddings in subj_comps_emb, but 
                            # replace sync_layer_indices layers with those from subj_comps_emb_mix_all_layers.
                            # Do not assign with sync_layers as indices, which destroys the computation graph.
                            subj_comps_emb_mix  = subj_comps_emb * (1 - layer_mask) \
                                                  + subj_comps_emb_mix_all_layers * layer_mask

                            subj_single_emb_mix = subj_single_emb * (1 - layer_mask) \
                                                  + subj_single_emb_mix_all_layers * layer_mask
                        else:
                            # There is only one layer of embeddings.
                            subj_comps_emb_mix  = subj_comps_emb_mix_all_layers
                            subj_single_emb_mix = subj_single_emb_mix_all_layers

                        # c_static_emb2 will be added with the ada embeddings to form the 
                        # conditioning embeddings in the U-Net.
                        # Unmixed embeddings and mixed embeddings will be merged in one batch for guiding
                        # image generation and computing compositional mix loss.
                        c_static_emb2 = torch.cat([ subj_single_emb, 
                                                    subj_comps_emb, 
                                                    subj_single_emb_mix, 
                                                    subj_comps_emb_mix ], dim=0)
                        
                        extra_info['iter_type']      = self.prompt_mix_scheme
                        # Set ada_bp_to_unet to False will reduce performance.
                        extra_info['ada_bp_to_unet'] = True

                    # This iter is a simple ada prompt delta loss iter, without prompt mixing loss. 
                    # This branch is reached only if prompt mixing is not enabled.
                    # "and not self.do_comp_prompt_mix_reg" is redundant, because it's "elif".
                    # Added for clarity. 
                    elif self.do_ada_prompt_delta_reg and not self.do_comp_prompt_mix_reg:
                        # Do ada prompt delta loss in this iteration. 
                        # c_static_emb: static embeddings, [128, 77, 768]. 128 = 8 * 16. 
                        # 8 is the total number of prompts in prompts_delta. 
                        # Each prompt is converted to 16 embeddings.
                        # 8 = 2 * 4. 2: bs. 4: subj_single_prompts, subj_comp_prompts, cls_single_prompts, cls_comp_prompts.                       
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
                        c_static_emb2 = subj_single_emb[:N_EMBEDS]
                        c_in2         = subj_single_prompts[:ORIG_BS]
                        extra_info['iter_type']      = 'normal_recon'
                        extra_info['ada_bp_to_unet'] = False

                    extra_info['cls_comp_prompts']   = cls_comp_prompts
                    extra_info['cls_single_prompts'] = cls_single_prompts

                    # If use_ada_embedding, then c_in2 will be fed again to CLIP text encoder to 
                    # get the ada embeddings. Otherwise, c_in2 will be useless and ignored.
                    c = (c_static_emb2, c_in2, extra_info)
                    # The full c_static_emb (embeddings of subj_single_prompts, subj_comp_prompts, 
                    # cls_single_prompts, cls_comp_prompts) is backed up to be used 
                    # to compute the static delta loss later.
                    self.c_static_emb = c_static_emb

                else:
                    # No delta loss or compositional mix loss. Keep the tuple c unchanged.
                    c = self.get_learned_conditioning(c)
                    # c[2]: extra_info. Here is only reached when do_static_prompt_delta_reg = False.
                    # Either prompt_delta_reg_weight == 0 or it's called by self.validation_step().
                    c[2]['iter_type'] = 'normal_recon'
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

    # t: steps.
    # cond: (c_static_emb, c_in, extra_info). c_in is the textual prompts. 
    # extra_info: a dict that contains 'ada_embedder' and other fields. 
    # ada_embedder: a function to convert c_in to ada embeddings.
    # ANCHOR[id=p_losses]
    def p_losses(self, x_start, cond, t, noise=None, img_mask=None, recur_depth=0):
        is_comp_iter = (self.do_comp_prompt_mix_reg or self.do_ada_prompt_delta_reg)
        noise = default(noise, lambda: torch.randn_like(x_start))

        if is_comp_iter:
            x_start_ = x_start
            t_       = t
            noise_   = noise
            img_mask_ = img_mask

            HALF_BS  = max(x_start.shape[0] // 2, 1)
            # Randomly choose t from the largest 250 timesteps, so as to match the total noise input.
            rand_timestep = np.random.randint(int(self.num_timesteps * 0.75), self.num_timesteps)
            t.fill_(rand_timestep)
            t = t[:HALF_BS].repeat(4)
            # Make x_start random.
            x_start.normal_()
            # Use the same x_start across the 4 instances.
            x_start  = x_start[:HALF_BS].repeat(4, 1, 1, 1)
            # Use the same noise across the 4 instances.
            noise    = noise[:HALF_BS].repeat(4, 1, 1, 1)
            # Ignore img_mask.
            img_mask = None

            if self.do_comp_prompt_mix_reg:
                c_static_emb, c_in = cond[0], cond[1]
                cond_ = cond
                subj_single_emb, subj_comps_emb, subj_single_emb_mix, subj_comps_emb_mix = \
                    torch.split(c_static_emb, c_static_emb.shape[0] // 4, dim=0)
                c_static_emb2 = torch.cat([ subj_comps_emb, subj_comps_emb, 
                                            subj_comps_emb_mix, subj_comps_emb_mix ], dim=0)
                
                subj_single_prompts, subj_comp_prompts, subj_single_prompts, subj_comp_prompts = \
                    divide_chunks(c_in, len(c_in) // 4)
                c_in2 = subj_comp_prompts + subj_comp_prompts + subj_comp_prompts + subj_comp_prompts
                # cond = (c_static_emb2, c_in2, cond_[2])

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # model_output is predicted noise.
        model_output = self.apply_model(x_noisy, t, cond)

        # compositional reg iterations.
        if is_comp_iter and self.calc_clip_loss:
            # If we do compositional prompt mixing, we need the final images of the 
            # second half as the reconstruction objective for compositional regularization.
            # The subj_single and cls_single images, although seemingly subject to
            # the image recon loss, are actually not reconstructable, 
            # since 75% of the chance, x_start is totally randomized.
            # LINK #shared_step
            # Note model_output is predicted noise.
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
            model_outputs = torch.split(x_recon, model_output.shape[0] // 4, dim=0)
            # Images generated both under subj_comp_prompts and subj_prompt_mix_comps 
            # are subject to the CLIP text-image matching evaluation.
            # cls_comp_prompts is used to compute the CLIP text-image matching loss on
            # images guided by the mixed embeddings.
            clip_images_code  = torch.cat([ model_outputs[1], model_outputs[3] ], dim=0)
            # The first  cls_comp_prompts is for subj_comps_emb, and 
            # the second cls_comp_prompts is for subj_comps_emb_mix.                
            clip_prompts_comp = cond[2]['cls_comp_prompts'] + cond[2]['cls_comp_prompts']

            # Compositional images are also subject to the single-prompt CLIP loss,
            # as they must contain the subject. This is to reduce the chance that
            # the output image only contains elements other than the subject.
            clip_prompts_single = cond[2]['cls_single_prompts'] + cond[2]['cls_single_prompts']
            if len(clip_images_code) != len(clip_prompts_comp) or len(clip_images_code) != len(clip_prompts_single):
                breakpoint()

        # Otherwise, ordinary image reconstruction loss. No need to split the model output.

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
        
        iter_type = cond[2]['iter_type']
        loss = 0

        if iter_type == 'normal_recon':
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

            # l_simple_weight = 1.
            loss = self.l_simple_weight * loss_simple.mean()

            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
            loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.detach()})
            # original_elbo_weight = 0, so that loss_vlb is disabled.
            loss += (self.original_elbo_weight * loss_vlb)

        if self.calc_clip_loss:
            with torch.no_grad():
                clip_images = self.decode_first_stage(clip_images_code)
                clip_images_np = clip_images.cpu().numpy()

                losses_clip_comp   = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_comp,   clip_images,  
                                                                                        reduction='diag')
                #losses_clip_single = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_single, clip_images, 
                #                                                                     reduction='diag')

            self.cache_and_log_generations(clip_images_np)

            losses_clip = losses_clip_comp #* 1.3 - losses_clip_single * 0.3
            # loss_dict is only used for logging. So we can pass the unfiltered detached loss.
            losses_clip_subj_comp, losses_clip_cls_comp = losses_clip_comp.split(losses_clip_comp.shape[0] // 2, dim=0)
            loss_dict.update({f'{prefix}/loss_clip_subj_comp': losses_clip_subj_comp.mean()})
            loss_dict.update({f'{prefix}/loss_clip_cls_comp':  losses_clip_cls_comp.mean()})

            if self.use_noised_clip:
                clip_loss_thres = 0.33
            else:
                clip_loss_thres = 0.35

            are_output_qualified = (losses_clip <= clip_loss_thres)
            # clip loss is only applied to the subject composition instance. 
            are_output_qualified_firsthalf = are_output_qualified.clone()
            are_output_qualified_firsthalf[HALF_BS:] = False
            
            """             
            # DO NOT use CLIP loss by setting clip_loss_weight > 0. It hurts the performance.
            if self.clip_loss_weight > 0:
                # Even if no image is qualified, we still compute the loss with zero weight,
                # to release the computation graph and avoid memory leak.
                loss_clip = (losses_clip * are_output_qualified_firsthalf).sum() \
                                / (are_output_qualified_firsthalf.sum() + 0.001)
                loss += (self.clip_loss_weight * loss_clip) 
            """

            # Hard-code here. Suppose HALF_BS=1, i.e., two instances are in clip_images.
            # So the teacher instance is always indexed by 1.
            # is_teachable: The teacher instance is only teachable if it's qualified, and the 
            # compositional clip loss is smaller than the student.

            self.cls_subj_clip_margin = 0.006

            is_teachable = are_output_qualified[1] and losses_clip_comp[1] < losses_clip_comp[0] - self.cls_subj_clip_margin

            np.set_printoptions(precision=4, suppress=True)
            self.num_total_clip_iters += 1
            self.num_teachable_iters += int(is_teachable)
            teachable_frac = self.num_teachable_iters / self.num_total_clip_iters

            # clip_losses = torch.cat([losses_clip_comp, losses_clip_single], dim=0).data.cpu().numpy()
            #print("CLIP losses: {}, teachable frac: {:.1f}%".format( \
            #        clip_losses, teachable_frac*100))
            loss_dict.update({f'{prefix}/teachable_frac': teachable_frac})

            if not self.filter_with_clip_loss:
                # Still compute the loss metrics. 
                # But no real filtering, instead just teach on all instances.
                is_teachable = True
            # Try one more time, to see if the teacher instance is qualified.
            elif not is_teachable and self.do_comp_prompt_mix_reg and recur_depth == 0:
                if not is_comp_iter:
                    breakpoint()
                # Release computation graph of this iter.
                del model_output, x_recon, model_outputs, clip_images_code
                # init_ada_embedding_cache() will implicitly clear the cache.
                self.embedding_manager.init_ada_embedding_cache()
                return self.p_losses(x_start_, cond, t_, noise_, img_mask_, recur_depth=1)
        else:
            is_teachable = True
            
        if self.embedding_reg_weight > 0:
            self.embedding_manager.is_comp_iter = is_comp_iter
            loss_embedding_reg = self.embedding_manager.embedding_to_loss().mean()
            loss_dict.update({f'{prefix}/loss_emb_reg': loss_embedding_reg.detach()})
            loss += (self.embedding_reg_weight * loss_embedding_reg)

        if self.do_static_prompt_delta_reg:
            # do_ada_prompt_delta_reg controls whether to do ada comp delta reg here.
            static_delta_loss, ada_delta_loss = self.embedding_manager.calc_prompt_delta_loss( 
                                    self.do_ada_prompt_delta_reg, self.c_static_emb
                                    )
            loss_dict.update({f'{prefix}/static_delta_loss': static_delta_loss.mean().detach()})
            if ada_delta_loss != 0:
                loss_dict.update({f'{prefix}/ada_delta_loss': ada_delta_loss.mean().detach()})
            # The prompt delta loss for ada embeddings is only applied 
            # every self.composition_regs_iter_gap iterations. So the ada loss 
            # should be boosted proportionally to composition_regs_iter_gap. 
            # Divide it by 2 to reduce the proportion of ada emb loss relative to 
            # static emb loss in the total loss.                
            ada_comp_loss_boost_ratio = self.composition_regs_iter_gap / 2
            loss_comp_delta_reg = static_delta_loss + ada_comp_loss_boost_ratio * ada_delta_loss
            loss += (self.prompt_delta_reg_weight * loss_comp_delta_reg)
            # print(f'loss_comp_delta_reg: {loss_comp_delta_reg.mean():.6f}')

        #print(clip_prompts_comp)
        """ 
        if self.clip_loss_weight > 0:
            clip_images = self.differentiable_decode_first_stage(clip_images_code)
            clip_images_np = clip_images.detach().cpu().numpy()
            # clip text-image similarity usually < 0.4. So using 0.5 - similarity is sufficient 
            # to keep the loss positive.
            # txt_to_img_similarity() returns the pairwise similarities between the prompts and the images.
            # So if there are N prompts and N images, the returned tensor has shape (N, N).
            # We should only consider diagonal elements, assuming the prompts are 
            # matched with the images in order. Therefore, it's buggy to take the mean of the losses.
            # In our particular setting of batch size = 2 (HALF_BS = 1), clip_prompts_comp consists of 
            # two identical cls_comp_prompts, which leads to the correct results, but we better 
            # fix it for larger batch sizes. 

            # txt_to_img_similarity() uses a preprocesser that assumes the input images 
            # are a tensor with mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0].
            # (not really assume the images are with this mean and std; 
            # it's just as a way to specify the transformation parameters.)
            # In effect, it transforms clip_images by (clip_images + 1) / 2,
            # which is consistent with the output transformation in stable_txt2img.py.
            losses_clip_comp   = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_comp,   clip_images, 
                                                                                    reduction='diag')
            #losses_clip_single = 0.5 - self.clip_evaluator.txt_to_img_similarity(clip_prompts_single, clip_images, 
            #                                                                     reduction='diag')
        """
        
        if self.do_comp_prompt_mix_reg and is_teachable:
            # do_comp_prompt_mix_reg iterations. No ordinary image reconstruction loss.
            # Only regularize on intermediate features, i.e., intermediate features generated 
            # under subj_comp_prompts should satisfy the delta loss constraint:
            # F(subj_comp_prompts)  - F(mix(subj_comp_prompts, cls_comp_prompts)) \approx 
            # F(subj_single_prompts) - F(cls_single_prompts)
            loss_subj_attn_distill = 0
            loss_feat_distill      = 0

            # unet_feats is a dict as: layer_idx -> unet_feat. 
            # It contains all the intermediate 25 layers of UNet features.
            unet_feats = cond[2]['unet_feats']
            # unet_attns is a dict as: layer_idx -> attn_mat. 
            # It contains the 5 specified conditioned layers of UNet attentions, 
            # i.e., layers 7, 8, 12, 16, 17.
            unet_attns = cond[2]['unet_attns']
            # Set to 0 to disable distillation on attention weights of the subject.
            distill_subj_attn_weight = 0.1

            # Discard top layers and the first few bottom layers from distillation.
            # distill_layer_weights: relative weight of each distillation layer. 
            # distill_layer_weights are normalized using distill_overall_weight.
            # Conditioning layers are 7, 8, 12, 16, 17. All the 4 layers have 1280 channels.
            # But intermediate layers also contribute to distillation. They have small weights.
            # Layer 16 has strong face semantics, so it is given a small weight.
            feat_distill_layer_weights = { 7:  1., 8: 1.,   
                                          #9:  0.5, 10: 0.5, 11: 0.5, 
                                           12: 0.5, 
                                          # 16: 0.25, 17: 0.25,
                                         }

            attn_distill_layer_weights = { 7:  1., 8: 1.,
                                           #9:  0.5, 10: 0.5, 11: 0.5,
                                           12: 1.,
                                           16: 1., 17: 1.,
                                         }
            
            feat_distill_layer_weight_sum = np.sum(list(feat_distill_layer_weights.values()))
            feat_distill_layer_weights = { k: v / feat_distill_layer_weight_sum for k, v in feat_distill_layer_weights.items() }
            attn_distill_layer_weight_sum = np.sum(list(attn_distill_layer_weights.values()))
            attn_distill_layer_weights = { k: v / attn_distill_layer_weight_sum for k, v in attn_distill_layer_weights.items() }

            orig_placeholder_ind_B, orig_placeholder_ind_T = self.embedding_manager.placeholder_indices
            # cat_placeholder_ind_B: a list of batch indices that point to individual instances.
            # cat_placeholder_ind_T: a list of token indices that point to the placeholder token in each instance.
            cat_placeholder_ind_B  = orig_placeholder_ind_B.clone()
            cat_placeholder_ind_T  = orig_placeholder_ind_T.clone()
            # cond[0].shape[1]: 154, number of tokens in the mixed prompts.
            # Right shift the subject token indices by 77, 
            # to locate the subject token (placeholder) in the concatenated comp prompts.
            placeholder_indices_B = orig_placeholder_ind_B

            # The class prompts are at the latter half of the batch.
            # So we need to add the batch indices of the subject prompts, to locate
            # the corresponding class prompts.
            cls_placeholder_indices_B = placeholder_indices_B + HALF_BS * 2
            # Concatenate the placeholder indices of the subject prompts and class prompts.
            placeholder_indices_B = torch.cat([placeholder_indices_B, cls_placeholder_indices_B], dim=0)
            # stack then flatten() to interlace the two lists.
            placeholder_indices_T = torch.stack([orig_placeholder_ind_T, cat_placeholder_ind_T], dim=1).flatten()
            # placeholder_indices: 
            # ( tensor([0,  0,   1, 1,   2, 2,   3, 3]), 
            #   tensor([6,  83,  6, 83,  6, 83,  6, 83]) )
            placeholder_indices   = (placeholder_indices_B, placeholder_indices_T)

            for unet_layer_idx, unet_feat in unet_feats.items():
                if (unet_layer_idx not in feat_distill_layer_weights) and (unet_layer_idx not in attn_distill_layer_weights):
                    continue

                # each is [1, 1280, 16, 16]
                feat_subj_single, feat_subj_comps, feat_mix_single, feat_mix_comps \
                    = torch.split(unet_feat, unet_feat.shape[0] // 4, dim=0)
                
                # [4, 8, 256, 154] / [4, 8, 64, 154] =>
                # [4, 154, 8, 256] / [4, 154, 8, 64]
                # We don't need BP through attention into UNet.
                attn_mat = unet_attns[unet_layer_idx].permute(0, 3, 1, 2)
                # subj_attn: [4, 8, 256] / [4, 8, 64]
                subj_attn = attn_mat[placeholder_indices]
                # subj_attn_subj_single, ...: [2, 8, 256].
                # The first dim 2 is the two occurrences of the subject token 
                # in the two sets of prompts. Therefore, HALF_BS is still needed to 
                # determine its batch size in convert_attn_to_spatial_weight().
                subj_attn_subj_single, subj_attn_subj_comps, \
                subj_attn_mix_single,  subj_attn_mix_comps \
                    = torch.split(subj_attn, subj_attn.shape[0] // 4, dim=0)

                if (unet_layer_idx in attn_distill_layer_weights) and (distill_subj_attn_weight > 0):
                    attn_distill_layer_weight = attn_distill_layer_weights[unet_layer_idx]
                    attn_subj_delta = subj_attn_subj_comps - subj_attn_subj_single
                    attn_mix_delta  = subj_attn_mix_comps  - subj_attn_mix_single
                    loss_layer_subj_attn_distill = calc_delta_loss(attn_subj_delta, attn_mix_delta, 
                                                                   first_n_dims_to_flatten=2, 
                                                                   ref_grad_scale=0)
                    # loss_layer_subj_attn_distill = self.get_loss(attn_subj_delta, attn_mix_delta, mean=True)
                    # L2 loss tends to be smaller than delta loss. So we scale it up by 10.
                    loss_subj_attn_distill += loss_layer_subj_attn_distill * attn_distill_layer_weight #* 10
                    # print(f'{unet_layer_idx} loss_layer_subj_attn_distill: {loss_layer_subj_attn_distill:.3f}')

                use_subj_attn_as_spatial_weights = True
                if use_subj_attn_as_spatial_weights:
                    if unet_layer_idx not in feat_distill_layer_weights:
                        continue
                    feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]

                    feat_subj_single, feat_subj_comps, feat_mix_single, feat_mix_comps \
                        = torch.split(unet_feat, unet_feat.shape[0] // 4, dim=0)
                    
                    # convert_attn_to_spatial_weight() will detach attention weights to 
                    # avoid BP through attention.
                    spatial_weight_subj_single = convert_attn_to_spatial_weight(subj_attn_subj_single, HALF_BS, feat_subj_single.shape[2:])
                    spatial_weight_subj_comps  = convert_attn_to_spatial_weight(subj_attn_subj_comps,  HALF_BS, feat_subj_comps.shape[2:])
                    spatial_weight_mix_single  = convert_attn_to_spatial_weight(subj_attn_mix_single,  HALF_BS, feat_mix_single.shape[2:])
                    spatial_weight_mix_comps   = convert_attn_to_spatial_weight(subj_attn_mix_comps,   HALF_BS, feat_mix_comps.shape[2:])

                    # Use mix single/comps weights on both subject-only and mix features, 
                    # to reduce misalignment and facilitate distillation.
                    feat_subj_single = feat_subj_single * spatial_weight_mix_single
                    feat_subj_comps  = feat_subj_comps  * spatial_weight_mix_comps
                    feat_mix_single  = feat_mix_single  * spatial_weight_mix_single
                    feat_mix_comps   = feat_mix_comps   * spatial_weight_mix_comps

                pool_spatial_size = (2, 2) # (1, 1)

                pooler = nn.AdaptiveAvgPool2d(pool_spatial_size)
                # Pool the H, W dimensions to remove spatial information.
                # After pooling, feat_subj_single, feat_subj_comps, 
                # feat_mix_single, feat_mix_comps: [1, 1280] or [1, 640], ...
                feat_subj_single = pooler(feat_subj_single).reshape(feat_subj_single.shape[0], -1)
                feat_subj_comps  = pooler(feat_subj_comps).reshape(feat_subj_comps.shape[0], -1)
                feat_mix_single  = pooler(feat_mix_single).reshape(feat_mix_single.shape[0], -1)
                feat_mix_comps   = pooler(feat_mix_comps).reshape(feat_mix_comps.shape[0], -1)

                stop_feat_mix_grad  = False
                feat_mix_grad_scale = 0.1
                if stop_feat_mix_grad:
                    # feat_subj_single = feat_subj_single.detach()
                    feat_mix_single  = feat_mix_single.detach()
                    feat_mix_comps   = feat_mix_comps.detach()
                elif feat_mix_grad_scale != 1:
                    grad_scaler = GradientScaler(feat_mix_grad_scale)
                    # feat_subj_single = grad_scaler(feat_subj_single)
                    feat_mix_single  = grad_scaler(feat_mix_single)
                    feat_mix_comps   = grad_scaler(feat_mix_comps)

                distill_on_delta = True
                if distill_on_delta:
                    # ortho_subtract is in terms of the last dimension. So we pool the spatial dimensions first above.
                    feat_mix_delta  = ortho_subtract(feat_mix_comps,  feat_mix_single)
                    feat_subj_delta = ortho_subtract(feat_subj_comps, feat_subj_single)
                else:
                    feat_mix_delta  = feat_mix_comps
                    feat_subj_delta = feat_subj_comps
                    
                # feat_subj_delta, feat_mix_delta: [1, 1280], ...
                # Pool the spatial dimensions H, W to remove spatial information.
                # The gradient goes back to feat_subj_delta -> feat_subj_comps,
                # as well as feat_mix_delta -> feat_mix_comps.
                # If stop_single_grad, the gradients to feat_subj_single and feat_mix_single are stopped, 
                # as these two images should look good by themselves (since they only contain the subject).
                # Note the learning strategy to the single image features should be different from 
                # the single embeddings, as the former should be optimized to look good by itself,
                # while the latter should be optimized to cater for two objectives: 1) the conditioned images look good,
                # and 2) the embeddings are amendable to composition.
                loss_layer_feat_distill = self.get_loss(feat_subj_delta, feat_mix_delta, mean=True)
                
                # print(f'layer {unet_layer_idx} loss: {loss_layer_prompt_mix_reg:.4f}')
                loss_feat_distill += loss_layer_feat_distill * feat_distill_layer_weight

            loss_dict.update({f'{prefix}/loss_feat_distill': loss_feat_distill.detach()})
            loss_dict.update({f'{prefix}/loss_subj_attn_distill': loss_subj_attn_distill.detach()})
            loss_prompt_mix_reg = loss_feat_distill + loss_subj_attn_distill * distill_subj_attn_weight
                                    
            loss += self.composition_prompt_mix_reg_weight * loss_prompt_mix_reg * self.distill_loss_scale
            
            self.embedding_manager.placeholder_indices = None


        loss_dict.update({f'{prefix}/loss': loss.detach()})

        return loss, loss_dict

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

    def cache_and_log_generations(self, samples, max_cache_size=40):
        self.generation_cache.append(samples)
        self.num_cached_generations += len(samples)

        if self.num_cached_generations >= max_cache_size:
            grid_folder = self.logger._save_dir + f'/samples'
            os.makedirs(grid_folder, exist_ok=True)
            grid_filename = grid_folder + f'/{self.cache_start_iter}-{self.global_step}.png'
            save_grid(self.generation_cache, grid_filename, 10, do_normalize=True)
            print(f"{self.num_cached_generations} generations saved to {grid_filename}")
            
            self.generation_cache = []
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
