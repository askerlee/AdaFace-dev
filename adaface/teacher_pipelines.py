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
        # self.unet will be initialized in the child class.
        self.unet = None

    # Only used for inference/distillation, so no_grad() is used.
    @torch.no_grad()
    def forward(self, ddpm_model, x_start, noise, t, teacher_context, num_denoising_steps=1):
        assert num_denoising_steps <= 10

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
                                
                # If do_arc2face_distill, then teacher_context is [BS=6, 21, 768].
                noise_pred = self.unet(sample=x_noisy, timestep=t, encoder_hidden_states=teacher_context,
                                       return_dict=False)[0]
                noise_preds.append(noise_pred)
                
                # sqrt_recip_alphas_cumprod[t] * x_t - sqrt_recipm1_alphas_cumprod[t] * noise
                pred_x0 = ddpm_model.predict_start_from_noise(x_noisy, t, noise_pred)
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
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
                        #"runwayml/stable-diffusion-v1-5", subfolder="unet"
                        'models/arc2face', subfolder="arc2face", torch_dtype=torch.float16
                    )

class UNetEnsembleTeacher(UNetTeacher):
    # unet_weights are not model weights, but scalar weights for individual unets.
    def __init__(self, unet, extra_unet_paths, unet_weights, device, **kwargs):
        super().__init__()
        self.unet = UNetEnsemble(unet, extra_unet_paths, unet_weights, device)

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

class ConsistentIDTeacher(UNetTeacher):
    def __init__(self, base_model_path, **kwargs):
        super().__init__()
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
    
    # Only used for inference/distillation, so no_grad() is used.
    @torch.no_grad()
    def forward(self, ddpm_model, x_start, noise, t, teacher_context, num_denoising_steps=1):
        # teacher_context: [BS, 81, 768]
        # teacher_context = torch.cat([student_prompt_embs, global_id_embeds], dim=1)     
        results = super().forward(ddpm_model, x_start, noise, t, teacher_context, num_denoising_steps)
        return results
    