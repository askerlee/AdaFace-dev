import torch
import numpy as np
import pytorch_lightning as pl
from diffusers import UNet2DConditionModel
from adaface.util import UNetEnsemble
from ConsistentID.lib.pipline_ConsistentID import ConsistentIDPipeline
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from adaface.arc2face_models import CLIPTextModelWrapper

class UNetTeacher(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = None
        # self.unet will be initialized in the child class.
        self.unet = None
        self.p_uses_cfg        = kwargs.get("p_uses_cfg", 0)
        # self.cfg_scale will be randomly sampled from cfg_scale_range.
        self.cfg_scale_range = kwargs.get("cfg_scale_range", [2, 4])
        self.cfg_scale       = 1
        if self.p_uses_cfg > 0:
            print(f"Using CFG with probability {self.p_uses_cfg} and scale range {self.cfg_scale_range}.")
        else:
            print(f"Never using CFG.")

    def forward(self, ddpm_model, x_start, noise, t, teacher_context, num_denoising_steps=1):
        assert num_denoising_steps <= 10
        if self.p_uses_cfg > 0:
            if teacher_context.shape[0] != x_start.shape[0] * 2:
                breakpoint()
            pos_context, neg_context = teacher_context.chunk(2, dim=0)
            self.uses_cfg = np.random.rand() < self.p_uses_cfg
            if self.uses_cfg:
                # Randomly sample a cfg_scale from cfg_scale_range.
                self.cfg_scale = np.random.uniform(*self.cfg_scale_range)
                print(f"Teacher samples CFG scale {self.cfg_scale:.1f}.")
            else:
                self.cfg_scale = 1
                print("Teacher does not use CFG.")
        else:
            if teacher_context.shape[0] != x_start.shape[0]:
                breakpoint()
            pos_context = teacher_context
            # Disable CFG. self.cfg_scale will be accessed by the student. 
            # So we need to make sure it is always set correctly, 
            # in case someday we want to switch from CFG to non-CFG during runtime.
            self.cfg_scale = 1

        # Initially, x_starts only contains the original x_start.
        x_starts    = [ x_start ]
        noises      = [ noise ]
        ts          = [ t ]
        noise_preds = []

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i in range(num_denoising_steps):
                x_start = x_starts[i]
                t       = ts[i]
                noise   = noises[i]
                # sqrt_alphas_cumprod[t] * x_start + sqrt_one_minus_alphas_cumprod[t] * noise
                x_noisy = ddpm_model.q_sample(x_start, t, noise)
                                
                # If do_arc2face_distill, then pos_context is [BS=6, 21, 768].
                noise_pred = self.unet(sample=x_noisy, timestep=t, encoder_hidden_states=pos_context,
                                       return_dict=False)[0]
                if self.uses_cfg:
                    # Usually we don't need to compute gradients w.r.t. the negative context.
                    with torch.no_grad():
                        neg_noise_pred = self.unet(sample=x_noisy, timestep=t, encoder_hidden_states=neg_context,
                                                return_dict=False)[0]
                    noise_pred = noise_pred * self.cfg_scale - neg_noise_pred * (self.cfg_scale - 1)

                noise_preds.append(noise_pred)
                
                # sqrt_recip_alphas_cumprod[t] * x_t - sqrt_recipm1_alphas_cumprod[t] * noise
                pred_x0 = ddpm_model.predict_start_from_noise(x_noisy, t, noise_pred)
                # The predicted x0 is used as the x_start for the next denoising step.
                x_starts.append(pred_x0)

                if i < num_denoising_steps - 1:
                    # NOTE: rand_like() samples from U(0, 1), not like randn_like().
                    relative_ts = torch.rand_like(t.float())
                    # Make sure at the middle step (i = sqrt(num_denoising_steps - 1), the timestep 
                    # is between 50% and 70% of the current timestep. So if num_denoising_steps = 5,
                    # we take timesteps within [0.5^0.66, 0.7^0.66] = [0.63, 0.79] of the current timestep.
                    # If num_denoising_steps = 4, we take timesteps within [0.5^0.72, 0.7^0.72] = [0.61, 0.77] 
                    # of the current timestep.
                    t_lb = t * np.power(0.5, np.power(num_denoising_steps - 1, -0.3))
                    t_ub = t * np.power(0.7, np.power(num_denoising_steps - 1, -0.3))
                    earlier_timesteps = (t_ub - t_lb) * relative_ts + t_lb
                    earlier_timesteps = earlier_timesteps.long()

                    # earlier_timesteps = ts[i+1] < ts[i].
                    ts.append(earlier_timesteps)

                    noise = torch.randn_like(pred_x0)
                    noises.append(noise)

        # Remove the original x_start from pred_x0s.
        pred_x0s = x_starts[1:]
        return noise_preds, pred_x0s, noises, ts
    
def create_arc2face_pipeline(base_model_path, dtype=torch.float16):
    text_encoder = CLIPTextModelWrapper.from_pretrained(
        'models/arc2face', subfolder="encoder", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        'models/arc2face', subfolder="arc2face", torch_dtype=dtype
    )
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )    
    pipeline = StableDiffusionPipeline.from_single_file(
            base_model_path,
            text_encoder=text_encoder,
            unet=unet,
            torch_dtype=dtype,
            safety_checker=None
        )
    pipeline.scheduler = noise_scheduler
    return pipeline

class Arc2FaceTeacher(UNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "arc2face"
        self.unet = UNet2DConditionModel.from_pretrained(
                        #"runwayml/stable-diffusion-v1-5", subfolder="unet"
                        'models/arc2face', subfolder="arc2face", torch_dtype=torch.float16
                    )

class UNetEnsembleTeacher(UNetTeacher):
    # unet_weights are not model weights, but scalar weights for individual unets.
    def __init__(self, unet, extra_unet_paths, unet_weights, device, **kwargs):
        super().__init__(**kwargs)
        self.name = "unet_ensemble"
        self.unet = UNetEnsemble(unet, extra_unet_paths, unet_weights, device)

class ConsistentIDTeacher(UNetTeacher):
    def __init__(self, base_model_path="models/ensemble/sd15-dste8-vae.safetensors", **kwargs):
        super().__init__(**kwargs)
        self.name = "consistentID"
        ### Load base model
        # In contrast to Arc2FaceTeacher or UNetEnsembleTeacher, ConsistentIDPipeline is not a torch.nn.Module.
        # We couldn't initialize the ConsistentIDPipeline to CPU first and wait it to be automatically moved to GPU.
        # Instead, we have to initialize it to GPU directly.
        pipe = create_consistentid_pipeline(base_model_path)
        # Release VAE to save memory. UNet and text_encoder is still needed for denoising 
        # (the unet is implemented in diffusers in fp16, so probably faster than the LDM unet).
        pipe.release_components(["vae"])
        self.pipe = pipe
        # Compatible with the UNetTeacher interface.
        self.unet = pipe.unet

    def to(self, device, torch_dtype):
        self.pipe.to(device, torch_dtype)
        return self
    
def create_consistentid_pipeline(base_model_path, dtype=torch.float16):
    pipe = ConsistentIDPipeline.from_single_file(
        base_model_path, 
        torch_dtype=dtype, 
    )
    # consistentID specific modules are still in fp32. Will be converted to fp16 
    # later with .to(device, torch_dtype) by the caller.
    pipe.load_ConsistentID_model(
        consistentID_weight_path="./models/ConsistentID/ConsistentID-v1.bin",
        bise_net_weight_path="./models/ConsistentID/BiSeNet_pretrained_for_ConsistentID.pth",
    )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )        
    pipe.scheduler = noise_scheduler

    return pipe

def create_unet_teacher(teacher_type, device, **kwargs):
    if teacher_type == "arc2face":
        return Arc2FaceTeacher(**kwargs)
    elif teacher_type == "unet_ensemble":
        # unet, extra_unet_paths and unet_weights are passed in kwargs.
        # Even if we distill from unet_ensemble, we still need to load arc2face for generating 
        # arc2face embeddings.
        # The first (optional) ctor param of UNetEnsembleTeacher is an instantiated unet, 
        # in our case, the ddpm unet. Ideally we should reuse it to save GPU RAM.
        # However, since the __call__ method of the ddpm unet takes different formats of params, 
        # for simplicity, we still use the diffusers unet.
        # unet_teacher is put on CPU first, then moved to GPU when DDPM is moved to GPU.
        return UNetEnsembleTeacher(device=device, **kwargs)
    elif teacher_type == "consistentID":
        return ConsistentIDTeacher(**kwargs)
    else:
        raise NotImplementedError(f"Teacher type {teacher_type} not implemented.")
    