import torch
import numpy as np
import pytorch_lightning as pl
from diffusers import UNet2DConditionModel
from adaface.util import UNetEnsemble, create_consistentid_pipeline
from diffusers import UNet2DConditionModel
from omegaconf.listconfig import ListConfig

def create_unet_teacher(teacher_type, device='cpu', **kwargs):
    # If teacher_type is a list with only one element, we dereference it.
    if isinstance(teacher_type, (tuple, list, ListConfig)) and len(teacher_type) == 1:
        teacher_type = teacher_type[0]

    if teacher_type == "arc2face":
        return Arc2FaceTeacher(**kwargs)
    elif teacher_type == "unet_ensemble":
        # unet, extra_unet_dirpaths and unet_weights are passed in kwargs.
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
    elif teacher_type == "simple_unet":
        return SimpleUNetTeacher(**kwargs)
    # Since we've dereferenced the list if it has only one element, 
    # this holding implies the list has more than one element. Therefore it's UNetEnsembleTeacher.
    elif isinstance(teacher_type, (tuple, list, ListConfig)):
        # teacher_type is a list of teacher types. So it's UNetEnsembleTeacher.
        return UNetEnsembleTeacher(unet_types=teacher_type, device=device, **kwargs)
    else:
        raise NotImplementedError(f"Teacher type {teacher_type} not implemented.")
    
class UNetTeacher(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = None
        # self.unet will be initialized in the child class.
        self.unet = None
        self.p_uses_cfg      = kwargs.get("p_uses_cfg", 0)
        # self.cfg_scale will be randomly sampled from cfg_scale_range.
        self.cfg_scale_range = kwargs.get("cfg_scale_range", [1.3, 2])
        # Initialize cfg_scale to 1. It will be randomly sampled during forward pass.
        self.cfg_scale       = 1
        if self.p_uses_cfg > 0:
            print(f"Using CFG with probability {self.p_uses_cfg} and scale range {self.cfg_scale_range}.")
        else:
            print(f"Never using CFG.")

    # Passing in ddpm_model to use its q_sample and predict_start_from_noise methods.
    # We don't implement the two functions here, because they involve a few tensors 
    # to be initialized, which will unnecessarily complicate the code.
    # noise: the initial noise for the first iteration.
    # t: the initial t. We will sample additional (num_denoising_steps - 1) smaller t.
    # uses_same_t: when sampling t, use the same t for all instances.
    def forward(self, ddpm_model, x_start, noise, t, teacher_context, 
                num_denoising_steps=1, uses_same_t=False):
        assert num_denoising_steps <= 10

        if self.p_uses_cfg > 0:
            self.uses_cfg = np.random.rand() < self.p_uses_cfg
            if self.uses_cfg:
                # Randomly sample a cfg_scale from cfg_scale_range.
                self.cfg_scale = np.random.uniform(*self.cfg_scale_range)
                if self.cfg_scale == 1:
                    self.uses_cfg = False

            if self.uses_cfg:
                print(f"Teacher samples CFG scale {self.cfg_scale:.1f}.")
            else:
                self.cfg_scale = 1
                print("Teacher does not use CFG.")

                # If p_uses_cfg > 0, we always pass both pos_context and neg_context to the teacher.
                # But the neg_context is only used when self.uses_cfg is True and cfg_scale > 1.
                # So we manually split the teacher_context into pos_context and neg_context, and only keep pos_context.
                if self.name == 'unet_ensemble':
                    teacher_pos_contexts = []
                    # teacher_context is a list of teacher contexts.
                    for teacher_context_i in teacher_context:
                        pos_context, neg_context = torch.chunk(teacher_context_i, 2, dim=0)
                        if pos_context.shape[0] != x_start.shape[0]:
                            breakpoint()             
                        teacher_pos_contexts.append(pos_context)
                    teacher_context = teacher_pos_contexts       
                else:
                    pos_context, neg_context = torch.chunk(teacher_context, 2, dim=0)
                    if pos_context.shape[0] != x_start.shape[0]:
                        breakpoint()
                    teacher_context = pos_context      
        else:
            # p_uses_cfg = 0. Never use CFG. 
            self.uses_cfg = False
            # In this case, the student only passes pos_context to the teacher,
            # so no need to split teacher_context into pos_context and neg_context.
            # self.cfg_scale will be accessed by the student,
            # so we need to make sure it is always set correctly, 
            # in case someday we want to switch from CFG to non-CFG during runtime.
            self.cfg_scale = 1

        if self.name == 'unet_ensemble':
            # teacher_context is a list of teacher contexts.
            for teacher_context_i in teacher_context:
                if teacher_context_i.shape[0] != x_start.shape[0] * (1 + self.uses_cfg):
                    breakpoint()
        else:
            if teacher_context.shape[0] != x_start.shape[0] * (1 + self.uses_cfg):
                breakpoint()
                
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
                
                if self.uses_cfg:
                    x_noisy2 = x_noisy.repeat(2, 1, 1, 1)
                    t2       = t.repeat(2)
                else:
                    x_noisy2 = x_noisy
                    t2 = t

                # If do_arc2face_distill, then pos_context is [BS=6, 21, 768].
                noise_pred = self.unet(sample=x_noisy2, timestep=t2, encoder_hidden_states=teacher_context,
                                       return_dict=False)[0]
                if self.uses_cfg and self.cfg_scale > 1:
                    pos_noise_pred, neg_noise_pred = torch.chunk(noise_pred, 2, dim=0)
                    noise_pred = pos_noise_pred * self.cfg_scale - neg_noise_pred * (self.cfg_scale - 1)

                # sqrt_recip_alphas_cumprod[t] * x_t - sqrt_recipm1_alphas_cumprod[t] * noise
                pred_x0 = ddpm_model.predict_start_from_noise(x_noisy, t, noise_pred)
                noise_preds.append(noise_pred)
                
                # The predicted x0 is used as the x_start for the next denoising step.
                x_starts.append(pred_x0)

                # Sample an earlier timestep for the next denoising step.
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

                    if uses_same_t:
                        # If uses_same_t, we use the same earlier_timesteps for all instances.
                        earlier_timesteps = earlier_timesteps[0].repeat(x_start.shape[0])

                    # earlier_timesteps = ts[i+1] < ts[i].
                    ts.append(earlier_timesteps)

                    noise = torch.randn_like(pred_x0)
                    noises.append(noise)

        return noise_preds, x_starts, noises, ts

class Arc2FaceTeacher(UNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "arc2face"
        self.unet = UNet2DConditionModel.from_pretrained(
                        #"runwayml/stable-diffusion-v1-5", subfolder="unet"
                        'models/arc2face', subfolder="arc2face", torch_dtype=torch.float16
                    )
        # Disable CFG. Even if p_uses_cfg > 0, the randomly drawn cfg_scale is still 1,
        # so the CFG is effectively disabled.       
        self.cfg_scale_range = [1, 1]

class UNetEnsembleTeacher(UNetTeacher):
    # unet_weights are not model weights, but scalar weights for individual unets.
    def __init__(self, unets, unet_types, extra_unet_dirpaths, unet_weights=None, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.name = "unet_ensemble"
        self.unet = UNetEnsemble(unets, unet_types, extra_unet_dirpaths, unet_weights, device)

class ConsistentIDTeacher(UNetTeacher):
    def __init__(self, base_model_path="models/sd15-dste8-vae.safetensors", **kwargs):
        super().__init__(**kwargs)
        self.name = "consistentID"
        ### Load base model
        # In contrast to Arc2FaceTeacher or UNetEnsembleTeacher, ConsistentIDPipeline is not a torch.nn.Module.
        # We couldn't initialize the ConsistentIDPipeline to CPU first and wait it to be automatically moved to GPU.
        # Instead, we have to initialize it to GPU directly.
        pipe = create_consistentid_pipeline(base_model_path)
        # Compatible with the UNetTeacher interface.
        self.unet = pipe.unet
        # Release VAE and text_encoder to save memory. UNet is still needed for denoising 
        # (the unet is implemented in diffusers in fp16, so probably faster than the LDM unet).
        pipe.release_components(["vae", "text_encoder"])

# We use the default cfg_scale_range=[1.3, 2] for SimpleUNetTeacher.
# Note p_uses_cfg=0.5 will also be passed in in kwargs.
class SimpleUNetTeacher(UNetTeacher):
    def __init__(self, unet_dirpath='models/ensemble/sd15-unet', 
                 torch_dtype=torch.float16, **kwargs):
        super().__init__(**kwargs)
        self.name = "simple_unet"
        self.unet = UNet2DConditionModel.from_pretrained(
                        unet_dirpath, torch_dtype=torch_dtype
                    )
