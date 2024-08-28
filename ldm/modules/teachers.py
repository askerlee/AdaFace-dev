import torch
import numpy as np
import pytorch_lightning as pl
from diffusers import UNet2DConditionModel
from adaface.util import UNetEnsemble
from adaface.face_id_to_img_prompt import ConsistentID_ID2ImgPrompt
from PIL import Image
from ConsistentID.lib.pipline_ConsistentID import ConsistentIDPipeline

class UNetTeacher(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # self.unet will be initialized in the child class.
        self.unet = None

    # Only used for inference/distillation, so no_grad() is used.
    @torch.no_grad()
    def forward(self, ddpm_model, x_start, noise, t, context, num_denoising_steps=1):
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
                                
                # If do_arc2face_distill, then context is [BS=6, 21, 768].
                noise_pred = self.unet(sample=x_noisy, timestep=t, encoder_hidden_states=context,
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

class ConsistentIDTeacher(UNetTeacher):
    def __init__(self, base_model_path, **kwargs):
        super().__init__()
        ### Load base model
        # In contrast to Arc2FaceTeacher or UNetEnsembleTeacher, ConsistentIDPipeline is not a torch.nn.Module.
        # We couldn't initialize the ConsistentIDPipeline to CPU first and wait it to be automatically moved to GPU.
        # Instead, we have to initialize it to GPU directly.
        pipe = ConsistentIDPipeline.from_single_file(
            base_model_path, 
            torch_dtype=torch.float16, 
        )
        ### Load consistentID_model checkpoints (these paths are fixed, so they are hardcoded).
        pipe.load_ConsistentID_model(
            consistentID_weight_path="./models/ConsistentID/ConsistentID-v1.bin",
            bise_net_weight_path="./models/ConsistentID/BiSeNet_pretrained_for_ConsistentID.pth",
        )
        # Release VAE to save memory. UNet and text_encoder is still needed for denoising 
        # (the unet is implemented in diffusers in fp16, so probably faster than the LDM unet).
        pipe.release_components(["vae"])
        self.pipe = pipe
        # Compatible with the UNetTeacher interface.
        self.unet = pipe.unet
        self.id2img_prompt_encoder = ConsistentID_ID2ImgPrompt(pipe)

    # Only used for inference/distillation, so no_grad() is used.
    @torch.no_grad()
    def forward(self, ddpm_model, x_start, noise, t, context, num_denoising_steps=1):
        batch_prompts, batch_images_unnorm = context
        batch_text_global_id_embeds = []

        # batch_images_unnorm: tensor of [BS, 512, 512, 3]
        for prompt, subj_image_ts in zip(batch_prompts, batch_images_unnorm):
            # subj_image_ts: [512, 512, 3]
            subj_image_obj = Image.fromarray(subj_image_ts.cpu().numpy().astype(np.uint8))
            # global_id_embeds: [1, 4, 768]
            global_id_embeds, _ = self.pipe.extract_global_id_embeds(subj_image_obj)

            text_embeds, _ = \
                self.pipe.encode_prompt(prompt, device=self.pipe.device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=False, negative_prompt=None)
            # text_global_id_embeds: [1, 81, 768]      
            text_global_id_embeds = torch.cat([text_embeds, global_id_embeds], dim=1)      
            batch_text_global_id_embeds.append(text_global_id_embeds)
        
        # batch_text_global_id_embeds: [BS, 81, 768]
        batch_text_global_id_embeds = torch.cat(batch_text_global_id_embeds, dim=0)
        results = super().forward(ddpm_model, x_start, noise, t, batch_text_global_id_embeds, num_denoising_steps)
        return results
    