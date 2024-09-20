import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from transformers import CLIPVisionModel
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput
import numpy as np
import argparse
from ConsistentID.lib.pipeline_ConsistentID import ConsistentIDPipeline
from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler,
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# perturb_tensor() adds a fixed amount of noise to the tensor.
def perturb_tensor(ts, perturb_std, perturb_std_is_relative=True, keep_norm=False,
                        std_dim=-1, norm_dim=-1, verbose=True):
    orig_ts = ts
    if perturb_std_is_relative:
        ts_std_mean = ts.std(dim=std_dim).mean().detach()

        perturb_std *= ts_std_mean
        # ts_std_mean: 50~80 for unnormalized images, perturb_std: 2.5-4 for 0.05 noise.
        if verbose:
            print(f"ts_std_mean: {ts_std_mean:.03f}, perturb_std: {perturb_std:.03f}")

    noise = torch.randn_like(ts) * perturb_std
    if keep_norm:
        orig_norm = ts.norm(dim=norm_dim, keepdim=True)
        ts = ts + noise
        new_norm  = ts.norm(dim=norm_dim, keepdim=True).detach()
        ts = ts * orig_norm / (new_norm + 1e-8)
    else:
        ts = ts + noise
    
    if verbose:
        print(f"Correlations between new and original tensors: {F.cosine_similarity(ts.flatten(), orig_ts.flatten(), dim=0).item():.03f}")
        
    return ts

def perturb_np_array(np_array, perturb_std, perturb_std_is_relative=True, std_dim=-1):
    ts = torch.from_numpy(np_array).to(dtype=torch.float32)
    ts = perturb_tensor(ts, perturb_std, perturb_std_is_relative, std_dim=std_dim)
    return ts.numpy().astype(np_array.dtype)

def calc_stats(emb_name, embeddings, mean_dim=0):
    print("%s:" %emb_name)
    repeat_count = [1] * embeddings.ndim
    repeat_count[mean_dim] = embeddings.shape[mean_dim]
    # Average across the mean_dim dim. 
    # Make emb_mean the same size as embeddings, as required by F.l1_loss.
    emb_mean = embeddings.mean(mean_dim, keepdim=True).repeat(repeat_count)
    l1_loss = F.l1_loss(embeddings, emb_mean)
    # F.l2_loss doesn't take sqrt. So the loss is very small. 
    # Compute it manually.
    l2_loss = ((embeddings - emb_mean) ** 2).mean().sqrt()
    norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
    print("L1: %.4f, L2: %.4f" %(l1_loss.item(), l2_loss.item()))
    print("Norms: min: %.4f, max: %.4f, mean: %.4f, std: %.4f" %(norms.min(), norms.max(), norms.mean(), norms.std()))


# Revised from RevGrad, by removing the grad negation.
class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_, debug=False):
        ctx.save_for_backward(alpha_, debug)
        output = input_
        if debug:
            print(f"input: {input_.abs().mean().item()}")
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        # saved_tensors returns a tuple of tensors.
        alpha_, debug = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_output2 = grad_output * alpha_
            if debug:
                print(f"grad_output2: {grad_output2.abs().mean().item()}")
        else:
            grad_output2 = None
        return grad_output2, None, None

class GradientScaler(nn.Module):
    def __init__(self, alpha=1., debug=False, *args, **kwargs):
        """
        A gradient scaling layer.
        This layer has no parameters, and simply scales the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)
        self._debug = torch.tensor(debug, requires_grad=False)

    def forward(self, input_):
        _debug = self._debug if hasattr(self, '_debug') else False
        return ScaleGrad.apply(input_, self._alpha.to(input_.device), _debug)

def gen_gradient_scaler(alpha, debug=False):
    if alpha == 1:
        return nn.Identity()
    if alpha > 0:
        return GradientScaler(alpha, debug=debug)
    else:
        assert alpha == 0
        # Don't use lambda function here, otherwise the object can't be pickled.
        return torch.detach

def pad_image_obj_to_square(image_obj, new_size=-1):
    # Remove alpha channel if it exists.
    if image_obj.mode == 'RGBA':
        image_obj = image_obj.convert('RGB')    

    # Pad input to be width == height
    width, height = orig_size = image_obj.size
    new_width, new_height = max(width, height), max(width, height)

    if width != height:    
        if width > height:
            pads = (0, (width - height) // 2)
        elif height > width:
            pads = ((height - width) // 2, 0)
        square_image_obj = Image.new("RGB", (new_width, new_height))
        # pads indicates the upper left corner to paste the input.
        square_image_obj.paste(image_obj, pads)
        #square_image_obj = square_image_obj.resize((512, 512))
        print(f"{width}x{height} -> {new_width}x{new_height} -> {square_image_obj.size}")
        long_short_ratio = max(width, height) / min(width, height)
    else:
        square_image_obj = image_obj
        pads = (0, 0)
        long_short_ratio = 1

    if new_size > 0:
        # Resize the shorter edge to 512.
        square_image_obj = square_image_obj.resize([int(new_size * long_short_ratio), int(new_size * long_short_ratio)])

    return square_image_obj, pads, orig_size

class UNetEnsemble(nn.Module):
    # The first unet is the unet already loaded in a pipeline.
    def __init__(self, unets, unet_types, extra_unet_paths, unet_weights=None, device='cuda', torch_dtype=torch.float16):
        super().__init__()

        self.unets = nn.ModuleList()
        if unets is not None:
            self.unets += unets

        if unet_types is not None:
            for unet_type in unet_types:
                if unet_type == "arc2face":
                    from adaface.arc2face_models import create_arc2face_pipeline
                    unet = create_arc2face_pipeline(unet_only=True)
                elif unet_type == "consistentID":
                    unet = create_consistentid_pipeline(unet_only=True)
                else:
                    breakpoint()
                self.unets.append(unet)

        if extra_unet_paths is not None:
            for unet_path in extra_unet_paths:
                unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch_dtype)
                self.unets.append(unet.to(device=device))

        if unet_weights is None:
            unet_weights = [1.] * len(self.unets)
        if len(self.unets) != len(unet_weights):
            breakpoint()
        unet_weights = torch.tensor(unet_weights, dtype=torch_dtype)
        unet_weights = unet_weights / unet_weights.sum()
        self.unet_weights = nn.Parameter(unet_weights, requires_grad=False)

        print(f"UNetEnsemble: {len(self.unets)} UNets loaded with weights: {self.unet_weights.data.cpu().numpy()}")
        self.dtype  = unet.dtype
        self.device = unet.device
        self.config = unet.config

    def forward(self, *args, **kwargs):
        return_dict = kwargs.get('return_dict', True)
        teacher_contexts = kwargs.pop('encoder_hidden_states', None)
        # Only one teacher_context is provided. That means all unets will use the same teacher_context.
        # We repeat the teacher_contexts to match the number of unets.
        if not isinstance(teacher_contexts, (list, tuple)):
            teacher_contexts = [teacher_contexts]
        if len(teacher_contexts) == 1 and len(self.unets) > 1:
            teacher_contexts = teacher_contexts * len(self.unets)

        samples = []

        for unet, teacher_context in zip(self.unets, teacher_contexts):
            sample = unet(encoder_hidden_states=teacher_context, *args, **kwargs)
            if not return_dict:
                sample = sample[0]
            else:
                sample = sample.sample

            samples.append(sample)

        samples = torch.stack(samples, dim=0)
        unet_weights = self.unet_weights.reshape(-1, *([1] * (samples.ndim - 1)))
        sample = (samples * unet_weights).sum(dim=0)

        if not return_dict:
            return (sample,)
        else:
            return UNet2DConditionOutput(sample=sample)

def create_consistentid_pipeline(base_model_path="models/ensemble/sd15-dste8-vae.safetensors", 
                                 dtype=torch.float16, unet_only=False):
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
    # We load the pipeline first, then use the unet in the pipeline.
    # Since the pipeline initialization will load LoRA into the unet, 
    # now we have the unet with LoRA loaded.
    if unet_only:
        # We release text_encoder and VAE to save memory.
        pipe.release_components(["text_encoder", "vae"])        
        return pipe.unet
    
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

@dataclass
class BaseModelOutputWithPooling2(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attn_mask: Optional[torch.FloatTensor] = None

# Revised from CLIPVisionTransformer to support attention mask. 
# self: a CLIPVisionTransformer instance.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L821
# pixel_values: preprocessed B*C*H*W images. [BS, 3, 224, 224]
# attn_mask: B*H*W attention mask.
def CLIPVisionTransformer_forward_with_mask(self, pixel_values = None, attn_mask=None, 
                                            output_attentions = None,
                                            output_hidden_states = None, return_dict = None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Visual tokens are flattended in embeddings().
        # self.embeddings: CLIPVisionEmbeddings.
        # hidden_states: [BS, 257, 1280]. 257: 16*16 (patch_embeds) + 1 (class_embeds).
        # 16*16 is output from Conv2d(3, 1280, kernel_size=(14, 14), stride=(14, 14), bias=False).
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        
        if attn_mask is not None:
            # feat_edge_size: 16.
            feat_edge_size = np.sqrt(hidden_states.shape[1] - 1).astype(int)
            # attn_mask: [BS, 512, 512] -> [BS, 1, 16, 16].
            attn_mask = F.interpolate(attn_mask.unsqueeze(1), size=(feat_edge_size, feat_edge_size), mode='nearest')
            # Flatten the mask: [BS, 1, 16, 16] => [BS, 1, 256].
            attn_mask = attn_mask.flatten(2)
            # Prepend 1 to the mask: [BS, 1, 256] => [BS, 1, 257]. 
            # This 1 corresponds to class_embeds, which is always attended to.
            attn_mask = torch.cat([torch.ones_like(attn_mask[:, :, :1]), attn_mask], dim=-1)
            attn_mask_pairs = torch.matmul(attn_mask.transpose(-1, -2), attn_mask).unsqueeze(1)
        else:
            attn_mask_pairs = None

        # encoder: CLIPEncoder.
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # New feature: (***The official documentation is wrong***)
            # attention_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length, sequence_length)`, *optional*):
            #                 Mask to avoid performing attention on pairs of token. Mask values selected in `[0, 1]`:
            #                 - 1 for pairs that are **not masked**,
            #                 - 0 for pairs that are **masked**.    
            # attention_mask is eventually used by CLIPEncoderLayer:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L370
            attention_mask=attn_mask_pairs,
            output_attentions=output_attentions,        # False
            output_hidden_states=output_hidden_states,  # True
            return_dict=return_dict,                    # True
        )

        # last_hidden_state: [BS, 257, 1280]
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # return_dict is True.
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling2(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            # Newly added: return resized flattened attention mask.
            # [BS, 1, 257] -> [BS, 257, 1]
            attn_mask=attn_mask.permute(0, 2, 1) if attn_mask is not None else None
        )

def CLIPVisionModel_forward_with_mask(self, pixel_values = None, attn_mask = None, output_attentions = None,
                                      output_hidden_states = None, return_dict = None):
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    return self.vision_model(
        pixel_values=pixel_values,
        attn_mask=attn_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

# patch_clip_image_encoder_with_mask() is applicable to both CLIPVisionModel and CLIPVisionModelWithProjection.
def patch_clip_image_encoder_with_mask(clip_image_encoder):
    clip_image_encoder.vision_model.forward = CLIPVisionTransformer_forward_with_mask.__get__(clip_image_encoder.vision_model)
    clip_image_encoder.forward = CLIPVisionModel_forward_with_mask.__get__(clip_image_encoder)
    return clip_image_encoder

class CLIPVisionModelWithMask(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace vision_model.forward() with the new one that supports mask.
        patch_clip_image_encoder_with_mask(self)
    
