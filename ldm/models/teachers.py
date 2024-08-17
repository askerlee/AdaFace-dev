import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from transformers import CLIPTokenizer
from adaface.util import get_arc2face_id_prompt_embs
from diffusers import UNet2DConditionModel
from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
from adaface.arc2face_models import CLIPTextModelWrapper
from adaface.util import UNetEnsemble

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
        self.text_encoder = CLIPTextModelWrapper.from_pretrained(
                            'models/arc2face', subfolder="encoder", torch_dtype=torch.float16
                            )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
    def gen_arc2face_prompt_embs(self, batch_size, pre_face_embs=None):
        # if pre_face_embs is None, generate random face embeddings [BS, 512].
        # Returns faceid_embeds, arc2face_prompt_emb.
        return get_arc2face_id_prompt_embs(None, self.tokenizer, self.text_encoder,
                                           extract_faceid_embeds=False, 
                                           pre_face_embs=pre_face_embs,
                                           image_folder=None, image_paths=None,
                                           images_np=None, 
                                           id_batch_size=batch_size,
                                           device=self.device,
                                           input_max_length=21, # Remove all paddings.
                                           noise_level=0, verbose=False)

class UNetEnsembleTeacher(UNetTeacher):
    def __init__(self, unet, extra_unet_paths, unet_weights, device, **kwargs):
        super().__init__()
        self.unet = UNetEnsemble(unet, extra_unet_paths, unet_weights, device)


'''
class ConsistentIDTeacher(UNetTeacher):
    def __init__(self, **kwargs):
        super().__init__()
        ### Load base model
        from ConsistentID.pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline
        pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )

        ### Load consistentID_model checkpoint
        pipe.load_ConsistentID_model(
            os.path.dirname(consistentID_path),
            subfolder="",
            weight_name=os.path.basename(consistentID_path),
            trigger_word="img",
        )     
'''
