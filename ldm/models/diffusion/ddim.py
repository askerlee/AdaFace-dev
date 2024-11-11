"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from ldm.util import ortho_subtract

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        # model: LatentDiffusion (inherits from DDPM)
        # model is used by calling model.apply_model() -> 
        # DiffusionWrapper.forward() -> UNetModel.forward().
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        '''
        If ddim_num_steps = 50,
        ddim_timesteps = 
        [  1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,
         261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
         521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
         781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981]
        '''
       
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))                               # useless
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))                        # useless if not ddim_use_original_steps.
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))   # useless if not ddim_use_original_steps.

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))                # useless
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu()))) # useful
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))   # useless
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))     # useless
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1))) # useless

        # ddim sampling parameters
        # ddim_eta = 0, so ddim_sigmas are all 0s.
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        # ddim_sigmas_for_original_num_steps: all 0s, useless.
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,               # total number of steps
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                #if cbs != batch_size:
                #    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                # conditioning is actually (conditioning, c_in, extra_info).
                # So conditioning[0] is the actual conditioning.
                if isinstance(conditioning, tuple):
                    conditioning_ = conditioning[0]
                else:
                    conditioning_ = conditioning
                #if conditioning_.shape[0] != batch_size:
                #    print(f"Warning: Got {conditioning_.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    guidance_scale=guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond_context, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      guidance_scale=1., unconditional_conditioning=None,
                      **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        # time_range:
        # [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        #  721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481,
        #  461, 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221,
        #  201, 181, 161, 141, 121, 101,  81,  61,  41,  21,   1]
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # total_steps: 50, provided in the command line.
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # breakpoint()

        # Guidance annealing. First proposed in CLIP-Sculptor, CVPR 2023. Independently discovered here.
        if isinstance(guidance_scale, (list, tuple)):        
            max_guide_scale, min_guide_scale = guidance_scale
            print(f"Running DDIM Sampling with {total_steps} timesteps, scale {min_guide_scale}-{max_guide_scale}")
        else:
            # If max_guide_scale < 2, then guide_scale_step_delta = 0 and no annealing.
            min_guide_scale = max_guide_scale = max(2.0, guidance_scale)
            print(f"Running DDIM Sampling with {total_steps} timesteps, scale {max_guide_scale}")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # At least one guidance annealing step (i.e., two uncond guidance steps)
        max_guide_anneal_steps = total_steps - 1
        # guide_scale_step_delta: set to 0 to disable the guidance annealing.
        # Normally, after max_guide_anneal_steps annealing, the guidance scale becomes 1.
        guide_scale_step_delta = (max_guide_scale - min_guide_scale) / max_guide_anneal_steps
        guide_scale = max_guide_scale

        for i, step in enumerate(iterator):
            # step: 981, ..., 1.
            # index: 49, ..., 0.
            # index points to the correct elements in alphas, sigmas, sqrt_one_minus_alphas, etc.
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            # print(step)
            
            if mask is not None:
                assert x0 is not None
                # q_sample adds noise to x0, according to ts.
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                # img: random noise. img_orig: masked image after adding noise.
                img = img_orig * mask + (1. - mask) * img

            # use_original_steps=False, quantize_denoised=False, temperature=1.0,
            # noise_dropout=0.0, score_corrector=None, corrector_kwargs=None,
            # guidance_scale=10.0, 
            outs = self.p_sample_ddim(img, cond_context, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      guidance_scale=guide_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      **kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

            if i <= max_guide_anneal_steps:
                guide_scale = guide_scale - guide_scale_step_delta
            else:
                guide_scale = 1
                
        return img, intermediates

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            # Double the batch size for unconditional and conditional conditioning.
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)

            if isinstance(c, tuple):
                c_c, c_in_c, extra_info = c
                c_u, c_in_u, _ = unconditional_conditioning

                # Concatenated conditining embedding in the order of (conditional, unconditional).
                # NOTE: the original order is (unconditional, conditional). But if we use conv attn,
                # extra_info['subj_indices'] index the conditional prompts. If we prepend unconditonal prompts,
                # subj_indices need to be manually adjusted. 
                # (conditional, unconditional) don't need to adjust subj_indices.
                twin_c = torch.cat([c_c, c_u])
                # Concatenated input context (prompts) in the order of (conditional, unconditional).
                twin_in = sum([c_in_c, c_in_u], [])
                # Combined context tuple.
                c2 = (twin_c, twin_in, extra_info)
            else:
                c2 = torch.cat([c, unconditional_conditioning])

            # model.apply_model() -> DiffusionWrapper.forward() -> UNetModel.forward().
            e_t, e_t_uncond = self.model.apply_model(x_in, t_in, c2).chunk(2)
            # scale = 0: e_t = e_t_uncond. scale = 1: e_t = e_t.
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # score_corrector is None.
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # use_original_steps=False, so alphas = self.ddim_alphas.
        '''        
        (Pdb) alphas
        tensor([0.9985, 0.9805, 0.9609, 0.9399, 0.9170, 0.8931, 0.8672, 0.8403, 0.8120,
                0.7827, 0.7520, 0.7207, 0.6885, 0.6558, 0.6226, 0.5889, 0.5552, 0.5215,
                0.4883, 0.4553, 0.4229, 0.3914, 0.3606, 0.3308, 0.3022, 0.2749, 0.2490,
                0.2245, 0.2014, 0.1798, 0.1598, 0.1412, 0.1243, 0.1088, 0.0947, 0.0819,
                0.0705, 0.0604, 0.0514, 0.0435, 0.0366, 0.0305, 0.0254, 0.0210, 0.0172,
                0.0140, 0.0113, 0.0091, 0.0073, 0.0058], device='cuda:0')
        '''
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        # alphas_prev: [0.999] + alphas[:-1]
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # sigmas = self.ddim_sigmas are all 0s.
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        # sigma_t are all 0. so always noise = 0, no matter what unscaled_noise is.
        unscaled_noise = noise_like(x.shape, device, repeat_noise)
        noise = sigma_t * unscaled_noise * temperature
        # Otherwise, use the provided noise (for debugging purposes).
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # dir_xt is a scaled e_t.
        # Since noise is always 0, x_prev is a linear combination of pred_x0 and dir_xt,
        # i.e., a linear combination of x and e_t.
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond_context, t_start, guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)

        # Guidance annealing. First proposed in CLIP-Sculptor, CVPR 2023. Independently discovered here.
        max_guide_scale = guidance_scale
        # If max_guide_scale < 2, then guide_scale_step_delta = 0 and no annealing.
        min_guide_scale = min(2.0, max_guide_scale)
        # At least one guidance annealing step (i.e., two uncond guidance steps)
        max_guide_anneal_steps = total_steps - 1
        # guide_scale_step_delta: set to 0 to disable the guidance annealing.
        # Normally, after max_guide_anneal_steps annealing, the guidance scale becomes 1.
        guide_scale_step_delta = (max_guide_scale - min_guide_scale) / max_guide_anneal_steps
        guide_scale = max_guide_scale

        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond_context, ts, index=index, use_original_steps=use_original_steps,
                                          guidance_scale=guide_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if i <= max_guide_anneal_steps:
                guide_scale = guide_scale - guide_scale_step_delta
            else:
                guide_scale = 1
                
        return x_dec