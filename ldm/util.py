import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import abc
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, draw_bounding_boxes
import math, sys, re, cv2

from safetensors.torch import load_file as safetensors_load_file
import asyncio

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def pack_uint128s_to_tensor(device, *args):
    int64s = []

    for i in range(len(args)):
        # NOTE: LOSS of information. But NCCL doesn't support transfer of uint64 :(
        low_64  = args[i]         & 0x7FFFFFFFFFFFFFFF  # Lower 64 bits
        high_64 = (args[i] >> 64) & 0x7FFFFFFFFFFFFFFF  # Upper 64 bits
        int64s.append(torch.tensor([low_64, high_64], dtype=torch.int64, device=device))

    # int64s: [N, 2]
    int64s = torch.stack(int64s)
    return int64s

def unpack_tensor_to_uint128s(int64s):
    uint128s = []
    for i in range(int64s.shape[0]):
        int128 = int64s[i, 0].item() + (int64s[i, 1].item() << 64)
        uint128s.append(int128)
    return uint128s

# Count the number of trainable parameters per parameter group
def count_optimized_params(param_groups):
    num_total_params = 0
    for i, param_group in enumerate(param_groups):
        num_group_params = 0
        for param in param_group['params']:
            if param.requires_grad:
                num_group_params += param.numel()  # Count the number of parameters
        num_total_params += num_group_params
        print(f"Param group {i}: {num_group_params} trainable parameters")
    print(f"Total trainable parameters: {num_total_params}")

def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith(".ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
    elif ckpt.endswith(".safetensors"):
        sd = safetensors_load_file(ckpt, device="cpu")
        pl_sd = None
    else:
        print(f"Unknown checkpoint format: {ckpt}")
        sys.exit(1)

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res

# norm_dim=1: channel dim. 
# Calculate the norms along the channel dim at each spatial location.
# Then calculate the min, max, mean and std of the norms.
def calc_stats(emb_name, embeddings, mean_dim=0, norm_dim=1):
    repeat_count = [1] * embeddings.ndim
    repeat_count[mean_dim] = embeddings.shape[mean_dim]
    # Average across the mean_dim dim. 
    # Make emb_mean the same size as embeddings, as required by F.l1_loss.
    emb_mean = embeddings.mean(mean_dim, keepdim=True).repeat(repeat_count)
    l1_loss = F.l1_loss(embeddings, emb_mean)
    # F.l2_loss doesn't take sqrt. So the loss is very small. 
    # We compute it manually by taking sqrt.
    l2_loss = ((embeddings - emb_mean) ** 2).mean().sqrt()
    norms = torch.norm(embeddings, dim=norm_dim).detach().cpu().numpy()
    print(f"{emb_name}: L1 {l1_loss.item():.4f}, L2 {l2_loss.item():.4f}", end=", ")
    print(f"Norms: min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}, std: {norms.std():.4f}")


def split_string(input_string):
    pattern = r'"[^"]*"|\S+'
    substrings = re.findall(pattern, input_string)
    substrings = [ s.strip('"') for s in substrings ]
    return substrings

# The most important variables: "subjects", "subj_types", "data_folder"
def parse_subject_file(subject_file_path):
    subj_info = {}
    subj2attr = {}
    
    with open(subject_file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if re.search(r"^set -g [a-zA-Z_]+ ", line):
                # set -g subjects  alexachung    alita...
                # At least one character in the value (after the variable name).
                mat = re.search(r"^set -g ([a-zA-Z_]+)\s+(\S.*)", line)
                if mat is not None:
                    var_name = mat.group(1)
                    substrings = split_string(mat.group(2))
                    values = substrings

                    if len(values) == 1 and values[0].startswith("$"):
                        # e.g., set -g cls_strings    $cls_delta_strings
                        values = subj_info[values[0][1:]]

                    subj_info[var_name] = values
                else:
                    breakpoint()

    for var_name in [ "subjects", "subj_types", "data_folder" ]:
        if var_name not in subj_info:
            print("Variable %s is not defined in %s" %(var_name, subject_file_path))
            breakpoint()

    for var_name in [ "subj_types" ]:
        if var_name in subj_info:
            subj2attr[var_name] = {}
            if len(subj_info[var_name]) != len(subj_info['subjects']):
                print("Variable %s has %d elements, while there are %d subjects." 
                      %(var_name, len(subj_info[var_name]), len(subj_info['subjects'])))
                breakpoint()
            for i in range(len(subj_info['subjects'])):
                subj_name = subj_info['subjects'][i]
                subj2attr[var_name][subj_name] = subj_info[var_name][i]

    # The most important variables: "subjects", "cls_delta_strings", "data_folder", "class_names"
    return subj_info, subj2attr

# Orthogonal subtraction of b from a: the residual is orthogonal to b (on the last on_last_n_dims dims).
# NOTE: ortho_subtract(a, b) is scale-invariant w.r.t. (b * b_discount),
# but scales proportionally to the scale of a.
# a, b are n-dimensional tensors. Subtraction happens at the last on_last_n_dims dims.
# ortho_subtract(a, b) is not symmetric w.r.t. a and b, nor is ortho_l2loss(a, b).
# NOTE: always choose a to be something we care about, and b to be something as a reference.
def ortho_subtract(a, b, b_discount=1, on_last_n_dims=1, return_align_coeffs=False):
    assert a.ndim == b.ndim, "Tensors a and b must have the same number of dimensions"
    if on_last_n_dims > 1:
        for i in range(-on_last_n_dims, 0):
            assert a.shape[i] == b.shape[i] or a.shape[i] == 1 or b.shape[i] == 1, \
              "Tensors a and b must have the same shape on non-singleton dims"

        # There could still be exceptions, if a and b have singleton dims at non-matching dims.
        # Leave the check to torch.
        if a.numel() < b.numel():
            a = a.expand(b.shape)
        elif b.numel() < a.numel():
            b = b.expand(a.shape)
        
        orig_shape = a.shape
        a2 = a.reshape(*a.shape[:-on_last_n_dims], -1)
        b2 = b.reshape(*b.shape[:-on_last_n_dims], -1)
    else:
        a2 = a
        b2 = b

    dot_a_b = (a2 * b2).sum(dim=-1)
    dot_b_b = (b2 * b2).sum(dim=-1)

    w_optimal = dot_a_b / (dot_b_b + 1e-6)
    result = a2 - b2 * w_optimal.unsqueeze(-1) * b_discount

    if on_last_n_dims > 1:
        result = result.reshape(orig_shape)
        w_orig_shape = list(orig_shape)
        w_orig_shape[-on_last_n_dims:] = [1] * on_last_n_dims
        w_optimal = w_optimal.reshape(w_orig_shape)

    if return_align_coeffs:
        return result, w_optimal
    else:
        return result

def demean(x, demean_dims=[-1]):
    if demean_dims is not None:
        assert len(demean_dims) <= x.ndim, "demean_dims must be a subset of x's dims."
        # Usually len(demean_dims) < x.ndim.
        if len(demean_dims) == x.ndim:
            breakpoint()
    return x - x.mean(dim=demean_dims, keepdim=True)

# Eq.(2) in the StyleGAN-NADA paper.
# delta, ref_delta: [2, 16, 77, 768].
# emb_mask: [2, 77, 1]. Could be fractional, e.g., 0.5, to weight different tokens.
# do_demeans: a list of two bools, indicating whether to demean delta and ref_delta, respectively.
# ref_grad_scale = 0: no gradient will be BP-ed to the reference embedding.
# If reduction == 'none', return a 2D loss tensor of [Batch, Instances].
# If reduction == 'mean', return a scalar loss.
def calc_ref_cosine_loss(delta, ref_delta, emb_mask=None, 
                         exponent=2, do_demeans=[False, False],
                         first_n_dims_into_instances=2,
                         ref_grad_scale=0, aim_to_align=True, 
                         reduction='mean', debug=False):

    B = delta.shape[0]
    loss = 0
    losses = []

    # Calculate the loss for each sample in the batch, 
    # as the mask may be different for each sample.
    for i in range(B):
        # Keep the batch dimension when dealing with the i-th sample, so that
        # we don't need to mess with first_n_dims_into_instances.
        delta_i     = delta[[i]]
        ref_delta_i = ref_delta[[i]]
        emb_mask_i  = emb_mask[[i]] if emb_mask is not None else None

        # Remove useless tokens, e.g., the placeholder suffix token(s) and padded tokens.
        if emb_mask_i is not None:
            try:
                # delta_i_flattened_dims_shape: [1, 77, 768]
                delta_i_flattened_dims_shape = delta_i.shape[:first_n_dims_into_instances]
                # truncate_mask: [1, 77].
                truncate_mask = (emb_mask_i > 0).squeeze(-1).expand(delta_i_flattened_dims_shape)
                # delta_i: [1, 77, 768] => [58, 768].
                delta_i       = delta_i[truncate_mask]
                ref_delta_i   = ref_delta_i[truncate_mask]
                # Make emb_mask_i have the same shape as delta_i, 
                # except the last (embedding) dimension for computing the cosine loss.
                # delta_i: [1, 20, 768]. 
                # emb_mask_i: [1, 77, 1] => [1, 77] => [58]
                # Expanding to same shape is necessary, since the cosine of each embedding has an 
                # individual weight (no broadcasting happens).
                emb_mask_i    = emb_mask_i.squeeze(-1).expand(delta_i_flattened_dims_shape)[truncate_mask]
            except:
                breakpoint()

        else:
            # Flatten delta and ref_delta, by tucking the token dimensions into the batch dimension.
            # delta_i: [2464, 768], ref_delta_i: [2464, 768]
            delta_i     = delta_i.reshape(delta_i.shape[:first_n_dims_into_instances].numel(), -1)
            ref_delta_i = ref_delta_i.reshape(delta_i.shape)
            # emb_mask_i should have first_n_dims_into_instances dims before flattening.
            emb_mask_i  = emb_mask_i.flatten() if emb_mask_i is not None else None

        # A bias vector to a set of conditioning embeddings doesn't change the attention matrix 
        # (though changes the V tensor). So the bias is better removed.
        # Therefore, do demean() before cosine loss, 
        # to remove the effect of bias.
        # In addition, different ada layers have significantly different scales. 
        # But since cosine is scale invariant, de-scale is not necessary and won't have effects.
        # LN = demean & de-scale. So in theory, LN is equivalent to demean() here. But LN may introduce
        # numerical instability. So we use simple demean() here.

        if debug:
            breakpoint()

        if do_demeans[0]:
            delta_i      = demean(delta_i)
        if do_demeans[1]:
            ref_delta_i2 = demean(ref_delta_i)
        else:
            ref_delta_i2 = ref_delta_i

        # x * x.abs.pow(exponent - 1) will keep the sign of x after pow(exponent).
        grad_scaler = gen_gradient_scaler(ref_grad_scale)
        ref_delta_i2 = grad_scaler(ref_delta_i2)

        ref_delta_i_pow = ref_delta_i2 * ref_delta_i2.abs().pow(exponent - 1)

        # If not aim_to_align, then cosine_label = -1, i.e., the cosine loss will 
        # push delta_i to be orthogonal with ref_delta_i.
        cosine_label = 1 if aim_to_align else -1
        # losses_i: 1D tensor of losses for each embedding.
        # F.cosine_embedding_loss() computes the cosines of each pair of instance
        # in delta_i and ref_delta_i. 
        # Each cosine is invariant to the scale of the corresponding two instances.
        losses_i = F.cosine_embedding_loss(delta_i, ref_delta_i_pow, 
                                           torch.ones_like(delta_i[:, 0]) * cosine_label, 
                                           reduction='none')
        # emb_mask_i has been flatten to 1D. So it gives different embeddings 
        # different relative weights (after normalization).
        if emb_mask_i is not None:
            losses_i = losses_i * emb_mask_i

        if reduction == 'mean':
            if emb_mask_i is not None:
                loss_i = losses_i.sum() / (emb_mask_i.sum() + 1e-8)
            else:
                loss_i = losses_i.mean()

            loss += loss_i
        elif reduction == 'none':
            losses.append(losses_i)
        else:
            breakpoint()
            
    if reduction == 'mean':
        loss = loss / B
        return loss
    else:
        losses = torch.stack(losses, dim=0)
        return losses

# feat_base, feat_ex, ...: [2, 9, 1280].
# Last dim is the channel dim.
# feat_ex     is the extension (enriched features) of feat_base.
# ref_feat_ex is the extension (enriched features) of ref_feat_base.
# delta_types: feat_to_ref or ex_to_base or both.
def calc_delta_alignment_loss(feat_base, feat_ex, ref_feat_base, ref_feat_ex, 
                              ref_grad_scale=0.1, feat_base_grad_scale=0.05,
                              cosine_exponent=3, delta_types=['feat_to_ref', 'ex_to_base']):
        ref_grad_scaler = gen_gradient_scaler(ref_grad_scale)
        # Reduce the gradient to the reference features, 
        # as the reference features are supposed to be unchanged, as opposed to feat_*. 
        # (although it still has a learnable component from mixed subject prompt embeddings.)
        ref_feat_base_gs  = ref_grad_scaler(ref_feat_base)
        ref_feat_ex_gs    = ref_grad_scaler(ref_feat_ex)

        if feat_base_grad_scale == -1:
            # subj_attn_base/subj_attn_delta:   ref_grad_scale = 0.05 => feat_base_grad_scale = 0.025.
            # feat_base/feat_delta:             ref_grad_scale = 0.1  => feat_base_grad_scale = 0.05.
            feat_base_grad_scale = min(ref_grad_scale / 2, 1)

        feat_base_scaler  = gen_gradient_scaler(feat_base_grad_scale)
        # Reduce the gradient to feat_base features, to better reserve subject features.
        feat_base_gs      = feat_base_scaler(feat_base)

        # ortho_subtract() is done on the last dimension. 
        # Align tgt_delta to src_delta.
        losses_delta_align = {}
        for delta_choice in delta_types:
            if delta_choice == 'feat_to_ref':                
                src_delta = ortho_subtract(feat_base_gs, ref_feat_base_gs)
                tgt_delta = ortho_subtract(feat_ex,      ref_feat_ex_gs)
            elif delta_choice == 'ex_to_base':
                src_delta = ortho_subtract(ref_feat_ex_gs, ref_feat_base_gs)
                tgt_delta = ortho_subtract(feat_ex,        feat_base_gs)

            # ref_grad_scale=1: ref grad scaling is disabled within calc_ref_cosine_loss,
            # since we've done gs on ref_feat_base, ref_feat_ex, and feat_base.
            loss_delta_align = calc_ref_cosine_loss(tgt_delta, src_delta, 
                                                    exponent=cosine_exponent,
                                                    do_demeans=[False, False],
                                                    first_n_dims_into_instances=(feat_base.ndim - 1), 
                                                    ref_grad_scale=1,
                                                    aim_to_align=True)

            losses_delta_align[delta_choice] = loss_delta_align

        return losses_delta_align

def calc_and_print_stats(ts, ts_name=None):
    if ts_name is not None:
        print(f"{ts_name}: ", end='')
    print("max: %.4f, min: %.4f, mean: %.4f, std: %.4f" %(ts.max(), ts.min(), ts.abs().mean(), ts.std()))

def rand_like(x):
    # Collapse all dimensions except the last one (channel dimension).
    x_2d = x.reshape(-1, x.shape[-1])
    std = x_2d.std(dim=0, keepdim=True)
    mean = x_2d.mean(dim=0, keepdim=True)
    rand_2d = torch.randn_like(x_2d)
    rand_2d = rand_2d * std + mean
    return rand_2d.view(x.shape)

def calc_chan_locality(feat):
    feat_mean = feat.mean(dim=(0, 2, 3))
    feat_absmean = feat.abs().mean(dim=(0, 2, 3))
    # Max weight is capped at 5.
    # The closer feat_absmean are with feat_mean.abs(), 
    # the more spatially uniform (spanned across H, W) the feature values are.
    # Bigger  weights are given to locally  distributed channels. 
    # Smaller weights are given to globally distributed channels.
    # feat_absmean >= feat_mean.abs(). So always chan_weights >=1, and no need to clip from below.
    chan_weights = torch.clip(feat_absmean / (feat_mean.abs() + 0.001), max=5)
    chan_weights = chan_weights.detach() / chan_weights.mean()
    return chan_weights.detach()

# flat_attn: [2, 8, 256] => [1, 2, 8, 256] => max/mean => [1, 256] => spatial_attn: [1, 16, 16].
# spatial_attn [1, 16, 16] => spatial_weight [1, 16, 16].
# BS: usually 1 (actually HALF_BS).
def convert_attn_to_spatial_weight(flat_attn, BS, out_spatial_shape, reversed=True):
    # flat_attn: [2, 8, 256] => [1, 2, 8, 256].
    # The 1 in dim 0 is BS, the batch size of each group of prompts.
    # The 2 in dim 1 is the two occurrences of the subject tokens in the comp mix prompts 
    # (or repeated single prompts).
    # The 8 in dim 2 is the 8 transformer heads.
    # The 256 in dim 3 is the number of image tokens in the current layer.
    # We cannot simply unsqueeze(0) since BS=1 is just a special case for this function.
    flat_attn = flat_attn.detach().reshape(BS, -1, *flat_attn.shape[1:])
    # [1, 2, 8, 256] => L2 => [1, 256] => [1, 16, 16].
    # Un-flatten the attention map to the spatial dimensions, so as to
    # apply them as weights.
    # Mean among the 8 heads, then sum across the 2 occurrences of the subject tokens.

    spatial_scale = np.sqrt(flat_attn.shape[-1] / out_spatial_shape.numel())
    spatial_shape = (int(out_spatial_shape[0] * spatial_scale), int(out_spatial_shape[1] * spatial_scale))
    spatial_attn = flat_attn.mean(dim=2).sum(dim=1).reshape(BS, 1, *spatial_shape)
    spatial_attn = F.interpolate(spatial_attn, size=out_spatial_shape, mode='bilinear', align_corners=False)

    attn_mean, attn_std = spatial_attn.mean(dim=(2,3), keepdim=True), \
                           spatial_attn.std(dim=(2,3), keepdim=True)
    # Lower bound of denom is attn_mean / 2, in case attentions are too uniform and attn_std is too small.
    denom = torch.clamp(attn_std + 0.001, min = attn_mean / 2)
    M = -1 if reversed else 1
    # Normalize spatial_attn with mean and std, so that mean attn values are 0.
    # If reversed, then mean + x*std = exp(-x), i.e., the higher the attention value, the lower the weight.
    # The lower the attention value, the higher the weight, but no more than 1.
    spatial_weight = torch.exp(M * (spatial_attn - attn_mean) / denom).clamp(max=1)
    # Normalize spatial_weight so that the average weight across spatial dims of each instance is 1.
    spatial_weight = spatial_weight / spatial_weight.mean(dim=(2,3), keepdim=True)

    # spatial_attn is the subject attention on pixels. 
    # spatial_weight is for the background objects (other elements in the prompt), 
    # flat_attn has been detached before passing to this function. So no need to detach spatial_weight.
    return spatial_weight, spatial_attn

def gen_spatial_weight_using_loss_std(pixelwise_loss, out_spatial_shape=(64, 64)):
    #loss_inst_std = loss_recon.mean(dim=1, keepdim=True).std(dim=(0,1), keepdim=True).detach()
    # Don't take mean across dim 1 (4 channels), as the latent pixels may have different 
    # scales acorss the 4 channels.
    loss_inst_std = pixelwise_loss.std(dim=(0,1), keepdim=True).detach()
    # Smooth the loss_inst_std by average pooling. loss_inst_std: [1, 1, 64, 64] -> [1, 1, 31, 31].
    loss_inst_std = F.avg_pool2d(loss_inst_std, 4, 2)
    spatial_weight = loss_inst_std / (loss_inst_std.mean(dim=(2,3), keepdim=True) + 1e-8)
    # Resize spatial_weight to the original size. spatial_weight: [1, 1, 31, 31] -> [1, 1, 64, 64].
    spatial_weight = F.interpolate(spatial_weight, size=out_spatial_shape, mode='bilinear', align_corners=False)
    return spatial_weight

# Distribute an embedding to M positions, each with sqrt(M) fraction of the original embedding.
# text_embedding: [B, N, D]
def distribute_embedding_to_M_tokens(text_embedding, placeholder_indices_N, divide_scheme='sqrt_M'):
    if placeholder_indices_N is None:
        return text_embedding
    
    # In a do_teacher_filter iteration, placeholder_indices_N may consist of the indices of 2 instances.
    # So we need to deduplicate them first.
    placeholder_indices_N = torch.unique(placeholder_indices_N)
    M = len(placeholder_indices_N)
    # num_vectors_per_subj_token = 1. No patching is needed.
    if M == 1:
        return text_embedding
    
    # Location of the first embedding in a multi-embedding token.
    placeholder_indices_N0 = placeholder_indices_N[:1]
    # text_embedding, repl_text_embedding: [16, 77, 768].
    # Use (:, placeholder_indices_N) as index, so that we can index all 16 embeddings at the same time.
    repl_mask = torch.zeros_like(text_embedding)
    # Almost 0 everywhere, except those corresponding to the multi-embedding token.
    repl_mask[:, placeholder_indices_N] = 1
    repl_text_embedding = torch.zeros_like(text_embedding)
    # Almost 0 everywhere, except being the class embedding at locations of the multi-embedding token.
    # Repeat M times the class embedding (corresponding to the subject embedding 
    # "z" at placeholder_indices_N0); 
    # Divide them by D to avoid the cross-attention over-focusing on the class-level subject.
    if divide_scheme == 'sqrt_M':
        D = np.sqrt(M)
    elif divide_scheme == 'M':
        D = M
    elif divide_scheme == 'none' or divide_scheme is None:
        D = 1

    repl_text_embedding[:, placeholder_indices_N] = text_embedding[:, placeholder_indices_N0].repeat(1, M, 1) / D

    # Keep the embeddings at almost everywhere, but only replace the embeddings at placeholder_indices_N.
    # Directly replacing by slicing with placeholder_indices_N will cause errors.
    patched_text_embedding = text_embedding * (1 - repl_mask) + repl_text_embedding * repl_mask
    return patched_text_embedding

def distribute_embedding_to_M_tokens_by_dict(text_embedding, placeholder_indices_dict, divide_scheme='sqrt_M'):
    if placeholder_indices_dict is None:
        return text_embedding
    
    for k in placeholder_indices_dict:
        if placeholder_indices_dict[k] is None:
            continue

        ph_indices_N  = placeholder_indices_dict[k][1]
        if len(ph_indices_N) > 1:
            text_embedding = distribute_embedding_to_M_tokens(text_embedding, ph_indices_N)

    return text_embedding

def scan_cls_delta_strings(tokenized_text, placeholder_indices_1st,
                           subj_name_to_cls_delta_tokens, MAX_SEARCH_SPAN=5):

    # If initializer_string are not specified for this subject token, then in the class prompts,
    # the cls delta token is randomly drawn from a predefined set of single-word tokens. 
    # In this case, no need to combine cls delta string embs.
    if subj_name_to_cls_delta_tokens is None or len(subj_name_to_cls_delta_tokens) == 0:
        return []

    device = tokenized_text.device

    placeholder_indices_B = placeholder_indices_1st[0]
    # The batch indices of the first token of each instance should be unique.
    if len(placeholder_indices_B.unique()) != len(placeholder_indices_B):
        breakpoint()

    # All instances contain the subject token. No need to check and combine cls delta string embs.
    if len(placeholder_indices_B) == tokenized_text.shape[0]:
        return []

    BS      = tokenized_text.shape[0]
    HALF_BS = BS // 2
    # It should be a compositional distillation iteration or an inference iteration.
    # In both cases, the first half of the batch should contain subject embeddings, 
    # and the second half not.
    if len(placeholder_indices_B) != HALF_BS \
      or (placeholder_indices_B != torch.arange(0, HALF_BS, device=device)).any():
        breakpoint()

    cls_delta_string_indices = []

    # Enumerate the instances in the second half of the batch, and see if there are 
    # init_word_token sequences.
    for batch_i in range(HALF_BS, BS):
        tokenized_text_i = tokenized_text[batch_i]
        # The prompt index of the first class   token of the i-th             instance
        # should be aligned with 
        # the prompt index of the first subject token of the (i - HALF_BS)-th instance.
        start_index_N = placeholder_indices_1st[1][batch_i - HALF_BS]
        # tokenized_text: [B, N]. tokenized_text_i: [N].
        # embedded_text:  [B, N, 768]. 
        # Search within a span of MAX_SEARCH_SPAN tokens from start_index_N, 
        # where the subject token starts at.
        # MAX_SEARCH_SPAN should be the sum of all extra tokens
        # (all excluding the first of the init word tokens; the first corresponds to the subject token).
        # The default value 10 should be sufficient larger than this sum.
        found = False
        for j in range(MAX_SEARCH_SPAN+1):
            start_N = start_index_N + j

            for subj_name, cls_delta_tokens in subj_name_to_cls_delta_tokens.items():
                cls_delta_tokens  = cls_delta_tokens.to(device)
                M = len(cls_delta_tokens)

                if (tokenized_text_i[start_N:start_N+M] == cls_delta_tokens).all():
                    cls_delta_string_indices.append((batch_i, start_N.item(), M, subj_name))
                    found = True
                    break
            
            if found:
                break

    return cls_delta_string_indices

# Scan prompt_embedding to find the cls delta tokens, and combine them to 1 token.
# Once merged, that token can be distributed to M tokens by distribute_embedding_to_M_tokens(),
# to align with the subject tokens.
def merge_cls_token_embeddings(prompt_embedding, cls_delta_string_indices):
    # No cls delta strings found in the prompt.
    if cls_delta_string_indices is None or len(cls_delta_string_indices) == 0:
        return prompt_embedding

    # cls_delta_string_indices is a list of tuples, each tuple being
    # (batch_i, start_index_N, M=num_cls_delta_tokens, subj_name).
    # Sort first by batch index, then by start index. So that the index offsets within each instance will
    # add up correctly as we process all cls_delta_string_indices tuples within the current instance.
    # subj_name is used to look up the cls delta weights.
    cls_delta_string_indices = sorted(cls_delta_string_indices, key=lambda x: (x[0], x[1]))
    # batch_i2offset is used for multiple cls delta tokens in the same instance.
    # It records the offset of the embeddings of the next cls delta token in the current instance.
    batch_i2offset = {}

    prompt_embedding2 = prompt_embedding.clone()
    occurred_subj_names = {}

    # Scan prompt_embedding to find the cls delta tokens, and combine them to 1 token.
    for batch_i, start_index_N, M, subj_name in cls_delta_string_indices:
        i_off = batch_i2offset.get(batch_i, 0)
        # cls_delta_embeddings: [M, 768].
        cls_delta_embeddings = prompt_embedding[batch_i, start_index_N:start_index_N+M]
        # avg_cls_delta_embedding: [768].
        cls_delta_embedding_sum = cls_delta_embeddings.sum(dim=0)
        prompt_embedding2[batch_i, start_index_N-i_off] = cls_delta_embedding_sum
        # We combine all the cls delta tokens to 1 token cls_delta_embedding_sum, so that
        # their positions align with the subject tokens in the first half of the batch.
        # To do so, we move the embeddings (except the EOS) after the last cls delta token to the left,
        # overwriting the rest M-1 cls delta embeddings.
        # NOTE: if there are multiple subject tokens (e.g., 28 tokens), then only the first subject token
        # is aligned with the cls_delta_embedding_sum. 
        # The rest 27 tokens are aligned with the embeddings of ", ".
        # This misalignment will be patched by calling 
        # distribute_embedding_to_M_tokens_by_dict(cls_single_emb, placeholder2indices_1b) in LatentDiffusion::forward().
        prompt_embedding2[batch_i, start_index_N+1-i_off:-(M+i_off)] = prompt_embedding[batch_i, start_index_N+M:-1]
        batch_i2offset[batch_i] = i_off + M - 1
        occurred_subj_names[subj_name] = \
            occurred_subj_names.get(subj_name, 0) + 1

    #if len(occurred_subj_names) > 1:
    #    breakpoint()
        
    return prompt_embedding2

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

def get_clip_tokens_for_string(clip_tokenizer, string, force_single_token=False):
    '''
    # If string is a new token, add it to the tokenizer.
    if string not in clip_tokenizer.get_vocab():
        clip_tokenizer.add_tokens([string])
        # clip_tokenizer() returns [49406, 49408, 49407]. 
        # 49406: start of text, 49407: end of text, 49408: new token.
        new_token_id = clip_tokenizer(string)["input_ids"][1]
        print("Added new token to tokenizer: {} -> {}".format(string, new_token_id))
    '''
    batch_encoding = clip_tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # [49406, 11781,  4668, 49407, 49407...]. 49406: start of text, SOT, 49407: end of text, EOT
    # 11781,  4668: tokens of "stuffed animal".
    token_count = torch.count_nonzero(tokens - 49407) - 1
    assert token_count >= 1, f"No token found in string '{string}'"
    if force_single_token:
        assert token_count == 1, f"String '{string}' maps to more than a single token. Please use another string"

    # Remove the SOT and EOT tokens.
    return tokens[0, 1:1+token_count]

def get_bert_tokens_for_string(clip_tokenizer, string):
    token = clip_tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embeddings_for_clip_tokens(embedder, tokens):
    # embedder: CLIPTextModel.text_model.embeddings
    # embedder(tokens): [1, N, 768]. N: number of tokens. 
    # RETURN: [N, 768]
    return embedder(tokens)[0]

# new_token_embeddings: [new_num_tokens, 768].
def extend_nn_embedding(old_nn_embedding, new_token_embeddings):
    emb_dim         = old_nn_embedding.embedding_dim
    num_old_tokens  = old_nn_embedding.num_embeddings
    num_new_tokens  = new_token_embeddings.shape[0]
    num_tokens2     = num_old_tokens + num_new_tokens
    
    new_nn_embedding = nn.Embedding(num_tokens2, emb_dim, 
                                    device=old_nn_embedding.weight.device,
                                    dtype=old_nn_embedding.weight.dtype)

    old_num_tokens = old_nn_embedding.weight.shape[0]
    # Copy the first old_num_tokens embeddings from old_nn_embedding to new_nn_embedding.
    new_nn_embedding.weight.data[:old_num_tokens] = old_nn_embedding.weight.data
    # Copy the new embeddings to new_nn_embedding.
    new_nn_embedding.weight.data[old_num_tokens:] = new_token_embeddings

    print(f"Extended nn.Embedding from {num_old_tokens} to {num_tokens2} tokens.")
    return new_nn_embedding

# string2embedding: a dict of {string: embedding} to be added to the text encoder. 
# Each embedding: [1, 768].
# string_list: a list of strings to be added to the text encoder. 
# The corresponding embeddings are initialized to 0.
def extend_clip_text_embedder(text_embedder, string2embedding, string_list):
    if string_list is not None:
        for string in string_list:
            if string not in string2embedding:
                string2embedding[string] = torch.zeros(1, 768)

    if len(string2embedding) == 0:
        return None

    get_tokens_for_string = partial(get_clip_tokens_for_string, text_embedder.tokenizer)

    extended_token_embeddings = []
    num_new_tokens = 0

    for string, embedding in string2embedding.items():
        # sanity check.
        # Need to check before calling tokenizer.add_tokens([string]), otherwise,
        # even if string is already in the tokenizer, it will be given to a high priority
        # and cause parsing errors later.
        # For example if string='y', then later "toy" will be parsed as "to" and "y",
        # instead of a single word "toy".
        try:
            token = get_tokens_for_string(string, force_single_token=True)[0]
        except:
            num_added_tokens = text_embedder.tokenizer.add_tokens([string])
            if num_added_tokens == 0:
                print(f"Token '{string}' already exists in the tokenizer.")
                breakpoint()

            token = get_tokens_for_string(string, force_single_token=True)[0]
            # text_embedder.transformer.text_model.embeddings.token_embedding: 
            # torch.nn.modules.sparse.Embedding, [49408, 768]
            # cls_token: 49408, 49409...
            # So token_embedding needs to be expanded.
            print(f"Extended CLIP text encoder with string: {string} -> token {token}")
            extended_token_embeddings.append(embedding)
            num_new_tokens += 1

    if num_new_tokens == 0:
        return None
  
    # extended_token_embeddings: list of tensors, each [1, 768] => [num_new_tokens, 768].
    extended_token_embeddings = torch.cat(extended_token_embeddings, dim=0)

    return extended_token_embeddings

def list_np_images_to_4d_tensor(list_np, dtype=np.uint8):
    tensors = []
    for ar in list_np:
        ts = torch.tensor(ar, dtype=dtype)
        if ts.ndim == 3:
            ts = ts.unsqueeze(0)
        tensors.append(ts)
    # We have made sure that each tensor has a batch dimension. 
    # So we use cat instead of stack.
    ts = torch.cat(tensors, dim=0)
    return ts

# samples:   a (B, C, H, W) tensor.
# img_flags: a tensor of (B,) ints.
# samples should be between [0, 255] (uint8).
async def save_grid(samples, img_flags, grid_filepath, nrow, async_mode=False):
    # img_box indicates the whole image region.
    img_box = torch.tensor([0, 0, samples.shape[2], samples.shape[3]]).unsqueeze(0)

    colors = [ None, 'green', 'red', 'purple' ]
    if img_flags is not None:
        # Highlight the teachable samples.
        for i, img_flag in enumerate(img_flags):
            if img_flag > 0:
                # Draw a 4-pixel wide green bounding box around the image.
                samples[i] = draw_bounding_boxes(samples[i], img_box, colors=colors[img_flag], width=12)

    # grid_samples is a 3D np array: (C, H2, W2)
    grid_samples = make_grid(samples, nrow=nrow).cpu().numpy()
    # samples is transposed to: (H2, W2, C)
    grid_img = Image.fromarray(grid_samples.transpose([1, 2, 0]))
    if grid_filepath is not None:
        if async_mode:
            # Asynchronous saving
            await asyncio.to_thread(grid_img.save, grid_filepath)
        else:
            # Synchronous saving
            grid_img.save(grid_filepath)
            
    # return image to be shown on webui
    return grid_img

def save_grid_sync(*args, **kwargs):
    asyncio.run(save_grid(*args, **kwargs, async_mode=False))

def chunk_list(lst, num_chunks):
    chunk_size = int(np.ceil(len(lst) / num_chunks))
    # looping till length lst
    for i in range(0, len(lst), chunk_size): 
        yield lst[i:i + chunk_size]

def join_list_of_indices(*indices_list):
    list_of_indices_B, list_of_indices_N = [], []
    for indices_B, indices_N in indices_list:
        list_of_indices_B.append(indices_B)
        list_of_indices_N.append(indices_N)
        
    indices_B = torch.cat(list_of_indices_B, dim=0)
    indices_N = torch.cat(list_of_indices_N, dim=0)
    return (indices_B, indices_N)

def join_dict_of_indices_with_key_filter(indices_dict, key_filter_list):
    if indices_dict is None:
        return None
        
    sel_indices = [ indices_dict[k] for k in indices_dict if k in key_filter_list ]
    if len(sel_indices) == 0:
        return None
    
    sel_indices = join_list_of_indices(*sel_indices)
    return sel_indices

def halve_token_indices(token_indices):
    if isinstance(token_indices, dict):
        token_indices2 = {}
        for k, v in token_indices.items():
            token_indices2[k] = halve_token_indices(v)
        return token_indices2
    else:
        token_indices_half_B  = token_indices[0].chunk(2)[0]
        token_indices_half_N  = token_indices[1].chunk(2)[0]
        return (token_indices_half_B, token_indices_half_N)

def split_indices_by_instance(indices, as_dict=False):
    indices_B, indices_N = indices
    unique_indices_B = torch.unique(indices_B)
    if not as_dict:
        indices_by_instance = [ (indices_B[indices_B == uib], indices_N[indices_B == uib]) for uib in unique_indices_B ]
    else:
        indices_by_instance = { uib.item(): indices_N[indices_B == uib] for uib in unique_indices_B }
    return indices_by_instance

# n: repeated times.
def extend_indices_B_by_n_times(indices, n, block_offset):
    if indices is None:
        return None, None
    
    indices_B, indices_N = indices
    # The original indices_B corresponds to block_offset instances -> the block 0.
    # Extending with n blocks, each block is offseted by (block_offset * i),
    # so that block 0 is adjacent to block 1.
    # If block_offset = 2, n = 4, then indices_B_ext is like [0, 1] -> [0, 1, 2, 3, 4, 5, 6, 7].
    indices_B_ext = [ (indices_B + block_offset * i) for i in range(n) ]
    indices_N_ext = [ indices_N ] * n
    indices_ext_by_block = list(zip(indices_B_ext, indices_N_ext))

    indices_B_ext   = torch.cat(indices_B_ext, dim=0)
    indices_N_ext   = torch.cat(indices_N_ext, dim=0)
    indices_ext     = (indices_B_ext, indices_N_ext)
    return indices_ext, indices_ext_by_block

def double_token_indices(token_indices, bs_offset):
    if token_indices is None:
        return None
    
    token_indices_x2, token_indices_x2_by_block = \
        extend_indices_B_by_n_times(token_indices, 2, bs_offset)
    
    return token_indices_x2

def normalize_dict_values(d):
    value_sum = np.sum(list(d.values()))
    # If d is empty, do nothing.
    if value_sum == 0:
        return d
    
    d2 = { k: v / value_sum for k, v in d.items() }
    return d2

def add_dict_to_dict(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v
    return d1

def filter_dict_by_key(d, key_container):
    d2 = { k: v for k, v in d.items() if k in key_container }
    return d2

# each dict in dicts has the same keys.
# The values are either lists or tensors, or dicts of either lists or tensors.
def collate_dicts(dicts):
    d = {}

    for k, v in dicts[0].items():
        collection = [ d[k] for d in dicts ]
        if isinstance(v, list):
            d[k] = sum(collection, [])
        elif isinstance(v, torch.Tensor):
            d[k] = torch.cat(collection, dim=0)
        elif isinstance(v, dict):
            d[k] = collate_dicts(collection)
        else:
            breakpoint()

    return d

def extract_layerwise_value(v, layer_idx, v_is_layerwise_array, v_is_layerwise_dict):
    if v_is_layerwise_array:
        # Extract the layer-specific value from the layerwise array.
        return v[layer_idx]
    elif v_is_layerwise_dict:
        v_layer_dict = {}
        for k2, v2 in v.items():
            # Extract the layer-specific value from the layerwise array for each dict key.
            v_layer_dict[k2] = v2[layer_idx]
        return v_layer_dict
    else:
        return v
    
# mask could be binary or float.
def masked_mean(ts, mask, instance_weights=None, dim=None, keepdim=False):
    if instance_weights is None:
        instance_weights = 1
    if isinstance(instance_weights, torch.Tensor):
        instance_weights = instance_weights.view(list(instance_weights.shape) + [1] * (ts.ndim - instance_weights.ndim))
    
    if mask is not None:
        # Broadcast mask to the same shape as ts. 
        # Without this step, mask_sum will be wrong.
        mask = mask.expand(ts.shape)

    if mask is None:
        return (ts * instance_weights).mean()
    else:
        mask_sum = mask.sum(dim=dim, keepdim=keepdim)
        mask_sum = torch.maximum( mask_sum, torch.ones_like(mask_sum) * 1e-6 )
        return (ts * instance_weights * mask).sum(dim=dim, keepdim=keepdim) / mask_sum

# Masked L2 loss on a 3D or 4D tensor.
# If 3D: (B, N, C), N: number of image tokens. mask: (B, N, 1).
# If 4D: (B, C, H, W). mask: (B, 1, H, W).
def masked_l2_loss(predictions, targets, mask):
    # Ensure the mask has the correct dimension.
    if mask.ndim != predictions.ndim:
        breakpoint()

    # Calculate L2 loss without reduction (element-wise squared difference)
    l2_loss = (predictions - targets) ** 2
    
    # Apply mask: broadcast mask across the channel dimension
    masked_loss = l2_loss * mask

    non_batch_dims = tuple(range(1, mask.ndim))
    # Sum the loss over the spatial dimensions (H, W) or N, and the channel dimension (C).
    loss_per_batch = masked_loss.sum(dim=non_batch_dims)

    # Normalize by the number of unmasked elements in each batch.
    # NOTE: scale up mask_sum by the repeat factor due to broadcasting.
    mask_sum = mask.sum(dim=non_batch_dims) * predictions.shape[1:].numel() / mask.shape[1:].numel()
    loss_per_batch = loss_per_batch / (mask_sum + 1e-8)  # Adding a small epsilon to avoid division by zero

    # Take the mean across the batch
    loss = loss_per_batch.mean()

    return loss

def anneal_value(training_percent, final_percent, value_range):
    assert 0 - 1e-6 <= training_percent <= 1 + 1e-6
    v_init, v_final = value_range
    if training_percent < final_percent:
        v_annealed = v_init + (v_final - v_init) * training_percent
    else:
        # Stop at v_final.
        v_annealed = v_final

    return v_annealed

def anneal_array(training_percent, final_percent, begin_array, end_array):
    assert len(begin_array) == len(end_array)
    begin_array = np.array(begin_array)
    end_array   = np.array(end_array)
    annealed_array = anneal_value(training_percent, final_percent, (begin_array, end_array))
    return annealed_array

# fluct_range: range of fluctuation ratios.
def rand_annealed(training_percent, final_percent, mean_range, 
                  fluct_range=(0.8, 1.2), legal_range=(0, 1)):
    mean_annealed = anneal_value(training_percent, final_percent, value_range=mean_range)
    rand_lb = max(mean_annealed * fluct_range[0], legal_range[0])
    rand_ub = min(mean_annealed * fluct_range[1], legal_range[1])
    
    return torch.rand(1).item() * (rand_ub - rand_lb) + rand_lb

def torch_uniform(low, high, size=1, device=None):
    return torch.rand(size, device=device) * (high - low) + low

# true_prob_range = (p_init, p_final). 
# The prob of flipping true is gradually annealed from p_init to p_final.
def draw_annealed_bool(training_percent, final_percent, true_prob_range):
    true_p_annealed = anneal_value(training_percent, final_percent, value_range=true_prob_range)
    # Flip a coin, with prob of true being true_p_annealed.    
    return (torch.rand(1) < true_p_annealed).item()

# t: original t. t_annealed: randomly scaled t, scaled according to ratio_range.
# ratio_range: range of fluctuation ratios (could > 1 or < 1).
# keep_prob_range: range of annealed prob of keeping the original t. If (0, 0.5),
# then gradually increase the prob of keeping the original t from 0 to 0.5.
def probably_anneal_int_tensor(t, training_percent, num_timesteps, ratio_range, keep_prob_range=(0, 0.5)):
    t_annealed = t.clone()
    # Gradually increase the chance of keeping the original t from 0 to 0.5.
    do_keep = draw_annealed_bool(training_percent, final_percent=1., true_prob_range=keep_prob_range)
    if do_keep:
        return t_annealed
    
    ratio_lb, ratio_ub = ratio_range
    assert ratio_lb < ratio_ub

    if t.ndim > 0:
        for i, ti in enumerate(t):
            ti_lowerbound = min(max(int(ti * ratio_lb), 0), num_timesteps - 1)
            ti_upperbound = min(int(ti * ratio_ub) + 1, num_timesteps)
            # Draw t_annealeded from [t, t*1.3], if ratio_range = (1, 1.3).
            t_annealed[i] = torch.randint(ti_lowerbound, ti_upperbound, (1,)).item()
    else:
        t_lowerbound = min(max(int(t * ratio_lb), 0), num_timesteps - 1)
        t_upperbound = min(int(t * ratio_ub) + 1, num_timesteps)
        t_annealed   = torch.tensor(torch.randint.randint(t_lowerbound, t_upperbound, (1,)).item(), 
                                    dtype=t.dtype, device=t.device)

    return t_annealed

'''
    Example:
    # Gradually increase the chance of taking 5 or 7 denoising steps.
    p_num_denoising_steps    = [0.2, 0.2, 0.3, 0.3]
    # If p_num_denoising_steps is shorter, cand_num_denoising_steps will be truncated accordingly.
    # Since there are 4 elements in p_num_denoising_steps, 
    # the truncated cand_num_denoising_steps is [1, 2, 3, 5].
    cand_num_denoising_steps = [1, 2, 3, 5, 7]
'''
def sample_num_denoising_steps(max_num_unet_distill_denoising_steps, p_num_denoising_steps, cand_num_denoising_steps):
    # Filter out the candidate denoising steps that exceed the max_num_unet_distill_denoising_steps.
    # If max_num_unet_distill_denoising_steps = 5, then cand_num_denoising_steps = [1, 2, 3, 5].
    cand_num_denoising_steps = [ si for si in cand_num_denoising_steps \
                                    if si <= max_num_unet_distill_denoising_steps ]
    p_num_denoising_steps = p_num_denoising_steps[:len(cand_num_denoising_steps)]
    p_num_denoising_steps = p_num_denoising_steps / np.sum(p_num_denoising_steps)

    # num_denoising_steps: 1, 2, 3, 5, among which 3 and 5 are selected with bigger chances.
    sample_idx = torch.multinomial(p_num_denoising_steps, 1).item()
    num_denoising_steps = cand_num_denoising_steps[sample_idx]
    return num_denoising_steps

# target_spatial_area: Either (H, W) or flattened H*W. If it's based on an attention map, then
# its geometrical dimensions (H, W) have been flatten to H*W. In this case, we assume H = W.
# mask: always 4D.
# mode: either "nearest" or "nearest|bilinear". Other modes will be ignored.
def resize_mask_to_target_size(mask, mask_name, target_spatial_area, mode="nearest|bilinear", warn_on_all_zero=True):
    if isinstance(target_spatial_area, int):
        # Assume square feature maps, target_H = target_W.
        target_H = target_W = int(np.sqrt(target_spatial_area))
    elif len(target_spatial_area) == 2:
        target_H, target_W = target_spatial_area
    else:
        breakpoint()
    spatial_shape = (target_H, target_H)

    # NOTE: masks should avoid "bilinear" mode. If the object is too small in the mask, 
    # it may result in all-zero masks.
    # mask: [2, 1, 64, 64] => mask2: [2, 1, 8, 8].
    mask2_nearest  = F.interpolate(mask.float(), size=spatial_shape, mode='nearest')
    if mode == "nearest|bilinear":
        mask2_bilinear = F.interpolate(mask.float(), size=spatial_shape, mode='bilinear', align_corners=False)
        # Always keep larger mask sizes.
        # When the subject only occupies a small portion of the image,
        # 'nearest' mode usually keeps more non-zero pixels than 'bilinear' mode.
        # In the extreme case, 'bilinear' mode may result in all-zero masks.
        mask2 = torch.maximum(mask2_nearest, mask2_bilinear)
    else:
        mask2 = mask2_nearest

    if warn_on_all_zero and (mask2.sum(dim=(1,2,3)) == 0).any():
        # Very rare cases. Safe to skip.
        print(f"WARNING: {mask_name} has all-zero masks.")
    
    return mask2

# c1, c2: [32, 77, 768]. mix_indices: 1D index tensor.
# mix_scheme: 'add', 'concat', 'sdeltaconcat', 'adeltaconcat'.
# The masked tokens will have the same embeddings after mixing.
def mix_embeddings(mix_scheme, c1, c2, mix_indices=None, 
                   c1_mix_scale=1., c2_mix_weight=None,
                   use_ortho_subtract=True):

    assert c1 is not None
    if c2 is None:
        return c1
    assert c1.shape == c2.shape

    if c2_mix_weight is None:
        c2_mix_weight = 1.
        
    if mix_scheme == 'add':
        # c1_mix_scale is an all-one tensor. No need to mix.
        # c1_mix_scale = 1. No need to mix. 
        if isinstance(c1_mix_scale, torch.Tensor)       and (c1_mix_scale == 1).all() \
          or not isinstance(c1_mix_scale, torch.Tensor) and c1_mix_scale == 1:
            return c1

        if mix_indices is not None:
            scale_mask = torch.ones_like(c1)

            if type(c1_mix_scale) == torch.Tensor:
                # c1_mix_scale is only for the first one/few instances. 
                # Repeat it to cover all instances in the batch.
                if len(c1_mix_scale) < len(scale_mask):
                    assert len(scale_mask) % len(c1_mix_scale) == 0
                    BS = len(scale_mask) // len(c1_mix_scale)
                    c1_mix_scale = c1_mix_scale.repeat(BS)
                # c1_mix_scale should be a 1D or 2D tensor. Extend it to 3D.
                for _ in range(3 - c1_mix_scale.ndim):
                    c1_mix_scale = c1_mix_scale.unsqueeze(-1)

            scale_mask[:, mix_indices] = c1_mix_scale

            # 1 - scale_mask: almost 0 everywhere, except those corresponding to the placeholder tokens 
            # being 1 - c1_mix_scale.
            # c1, c2: [16, 77, 768].
            # Each is of a single instance. So only provides subj_indices_N 
            # (multiple token indices of the same instance).
            c_mix = c1 * scale_mask + c2 * (1 - scale_mask)
            #print("cls/subj/mix:", c1[:, mix_indices].norm().detach().item(), c2[:, mix_indices].norm().detach().item(), 
            #                       c_mix[:, mix_indices].norm().detach().item())
        else:
            # Mix the whole sequence.
            c_mix = c1 * c1_mix_scale + c2 * (1 - c1_mix_scale)
            #print("cls/subj/mix:", c1.norm().detach().item(), c2.norm().detach().item(), c_mix.norm().detach().item())

    elif mix_scheme == 'concat':
        c_mix = torch.cat([ c1, c2 * c2_mix_weight ], dim=1)
    elif mix_scheme == 'addconcat':
        c_mix = torch.cat([ c1, c1 * (1 - c2_mix_weight) + c2 * c2_mix_weight ], dim=1)

    # sdeltaconcat: subject-delta concat. Requires placeholder_indices.
    elif mix_scheme == 'sdeltaconcat':
        assert mix_indices is not None
        # delta_embedding is the difference between the subject embedding and the class embedding.
        if use_ortho_subtract:
            delta_embedding = ortho_subtract(c2, c1)
        else:
            delta_embedding = c2 - c1
            
        delta_embedding = delta_embedding[:, mix_indices]
        assert delta_embedding.shape[0] == c1.shape[0]

        c2_delta = c1.clone()
        # c2_mix_weight only boosts the delta embedding, and other tokens in c2 always have weight 1.
        c2_delta[:, mix_indices] = delta_embedding
        c_mix = torch.cat([ c1, c2_delta * c2_mix_weight ], dim=1)

    # adeltaconcat: all-delta concat.
    elif mix_scheme == 'adeltaconcat':
        # delta_embedding is the difference between all the subject tokens and the class tokens.
        if use_ortho_subtract:
            delta_embedding = ortho_subtract(c2, c1)
        else:
            delta_embedding = c2 - c1
            
        # c2_mix_weight scales all tokens in delta_embedding.
        c_mix = torch.cat([ c1, delta_embedding * c2_mix_weight ], dim=1)

    return c_mix

# mix_cls_subj_embeddings() is NO LONGER USED.
def mix_cls_subj_embeddings(prompt_emb, subj_indices_1b_N, cls_subj_mix_scale=0.8):
    subj_emb, cls_emb = prompt_emb.chunk(2)

    # First mix the prompt embeddings.
    # mix_embeddings('add', ...):  being subj_comp_emb almost everywhere, except those at subj_indices_1b_N,
    # where they are subj_comp_emb * cls_subj_mix_scale + cls_comp_emb * (1 - cls_subj_mix_scale).
    # subj_single_emb, cls_single_emb, subj_comp_emb, cls_comp_emb: [1, 77, 768].
    # Each is of a single instance. So only provides subj_indices_1b_N 
    # (multiple token indices of the same instance).
    mixed_emb = mix_embeddings('add', cls_emb, subj_emb, mix_indices=subj_indices_1b_N,
                                      c1_mix_scale=cls_subj_mix_scale)

    PROMPT_MIX_GRAD_SCALE = 0.05
    grad_scaler = gen_gradient_scaler(PROMPT_MIX_GRAD_SCALE)
    # mix_comp_emb receives smaller grad, since it only serves as the reference.
    # If we don't scale gradient on mix_comp_emb, chance is mix_comp_emb might be 
    # dominated by subj_comp_emb,
    # so that mix_comp_emb will produce images similar as subj_comp_emb does.
    # Scaling the gradient will improve compositionality but reduce face similarity.
    mixed_emb = grad_scaler(mixed_emb)

    # prompt_emb_mixed is the prompt embeddings of the prompts used in losses other than 
    # the prompt delta loss, e.g., used to estimate the ada embeddings.
    # prompt_emb_mixed: [4, 77, 768]
    # Unmixed embeddings and mixed embeddings will be merged in one batch for guiding
    # image generation and computing compositional mix loss.
    prompt_emb_mixed = torch.cat([ subj_emb, mixed_emb ], dim=0)

    return prompt_emb_mixed 

def select_and_repeat_instances(sel_indices, REPEAT, *args):
    rep_args = []
    for arg in args:
        if arg is not None:
            if isinstance(arg, (torch.Tensor, np.ndarray)):
                arg2 = arg[sel_indices].repeat([REPEAT] + [1] * (arg.ndim - 1))
            elif isinstance(arg, (list, tuple)):
                arg2 = arg[sel_indices] * REPEAT
            else:
                breakpoint()
        else:
            arg2 = None
        rep_args.append(arg2)

    return rep_args

# Extract the index to the first token in each instance.
# token_indices is a tuple of two 1D tensors: (token_indices_B, token_indices_T).
def extract_first_index_in_each_instance(token_indices):
    token_indices_by_instance = split_indices_by_instance(token_indices)
    token_indices_B_first_only = []
    token_indices_T_first_only = []
    for token_indices_B, token_indices_T in token_indices_by_instance:
        token_indices_B_first_only.append(token_indices_B[0])
        token_indices_T_first_only.append(token_indices_T[0])
    
    token_indices_B_first_only = torch.stack(token_indices_B_first_only, dim=0)
    token_indices_T_first_only = torch.stack(token_indices_T_first_only, dim=0)
    return (token_indices_B_first_only, token_indices_T_first_only)

# If do_sum, returned emb_attns is 3D. Otherwise 4D.
# indices are applied on the first 2 dims of attn_mat.
def sel_emb_attns_by_indices(attn_mat, indices, all_token_weights=None, do_sum=True, do_mean=False):
    indices_by_instance = split_indices_by_instance(indices)

    # emb_attns[0]: [1, 9, 8, 64]
    # 8: 8 attention heads. Last dim 64: number of image tokens.
    emb_attns   = [ attn_mat[inst_indices].unsqueeze(0) for inst_indices in indices_by_instance ]
    if all_token_weights is not None:
        # all_token_weights: [4, 77].
        # token_weights_by_instance[0]: [1, 9, 1, 1].
        token_weights = [ all_token_weights[inst_indices].reshape(1, -1, 1, 1) for inst_indices in indices_by_instance ]
    else:
        token_weights = [ 1 ] * len(indices_by_instance)

    # Apply token weights.
    emb_attns = [ emb_attns[i] * token_weights[i] for i in range(len(indices_by_instance)) ]

    # sum among K_subj_i subj embeddings -> [1, 8, 64]
    if do_sum:
        emb_attns   = [ emb_attns[i].sum(dim=1) for i in range(len(indices_by_instance)) ]
    elif do_mean:
        emb_attns   = [ emb_attns[i].mean(dim=1) for i in range(len(indices_by_instance)) ]

    emb_attns = torch.cat(emb_attns, dim=0)
    return emb_attns

# Textual inversion is supported, where prompt_embeddings is only one embedding.
# prompt_embeddings: size: [4, 16, 77, 768]. 4: batch_size. 16: number of UNet layers.
# embeddings of subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb. 
def calc_prompt_emb_delta_loss(prompt_embeddings, prompt_emb_mask, cls_delta_grad_scale=0.05):
    # prompt_embeddings contains 4 types of embeddings:
    # subj_single, subj_comp, cls_single, cls_comp.
    # prompt_embeddings: [4, 16, 77, 768].
    # cls_*: embeddings generated from prompts containing a class token (as opposed to the subject token).
    # Each is [1, 16, 77, 768]
    subj_single_emb, subj_comp_emb, cls_single_emb, cls_comp_emb = \
            prompt_embeddings.chunk(4)

    if prompt_emb_mask is not None:
        # Regularization on padding tokens.
        # prompt_emb_mask[prompt_emb_mask == 0] = 0.25
        # Exclude the start token.
        prompt_emb_mask[:, 0] = 0
        subj_single_mask, subj_comp_mask, cls_single_mask, cls_comp_mask = \
            prompt_emb_mask.chunk(4)
        
        # cls_single_mask == subj_single_mask, cls_comp_mask == subj_comp_mask
        # So only compute using subj_single_mask and subj_comp_mask.
        prompt_emb_mask_agg = subj_single_mask + subj_comp_mask
        # If a token appears both in single and comp prompts (all tokens in the single prompts), 
        # the aggregated mask value is 2. Convert to 1.
        # If a token appears only in the compositional part, 
        # the aggregated mask value is 1. Convert to 0.25.
        # If a token is padding, the aggregated mask value is 0.5. Convert to 0.0625.
        prompt_emb_mask_weighted = prompt_emb_mask_agg.pow(2) / 4
    else:
        prompt_emb_mask_weighted = None

    use_ortho_subtract = True
    # cls_emb_delta: [1, 16, 77, 768]. Should be a repeat of a tensor of size [1, 1, 77, 768]. 
    # Delta embedding between class single and comp embeddings.
    # by 16 times along dim=1, as cls_prompt_* doesn't contain placeholder_token.
    # subj_emb_delta: [1, 16, 77, 768]. Different values for each layer along dim=1.
    # Delta embedding between subject single and comp embeddings.
    # subj_emb_delta / ada_subj_delta should be aligned with cls_delta.
    if use_ortho_subtract:
        subj_emb_delta   = ortho_subtract(subj_comp_emb, subj_single_emb)
        cls_emb_delta    = ortho_subtract(cls_comp_emb,  cls_single_emb)
    else:
        subj_emb_delta   = subj_comp_emb  - subj_single_emb
        cls_emb_delta    = cls_comp_emb   - cls_single_emb

    loss_prompt_emb_delta   = \
        calc_ref_cosine_loss(subj_emb_delta, cls_emb_delta, 
                             emb_mask=prompt_emb_mask_weighted,
                             # Although the zero-shot subject embedding features are pretty balanced 
                             # and thus are already centered at each dimension,
                             # the class embeddings are not. So we still need to do demean on cls_emb_delta.
                             do_demeans=[False, True],
                             first_n_dims_into_instances=2,
                             ref_grad_scale=cls_delta_grad_scale,   # 0.05
                             aim_to_align=True)

    return loss_prompt_emb_delta

def calc_dyn_loss_scale(loss, base_loss_and_scale, ref_loss_and_scale, rel_scale_range=(0, 4)):
    """
    Return a loss scale as a function of loss.

    The scale is computed using linear interpolation based on the formula:
    scale_delta: the offset of the ref scale from the base scale, (ref_scale - base_scale).
    relative_scale: the offset of the output scale from the base scale, as a multiple of scale_delta.
    relative_scale = (loss - base_loss) / (ref_loss - base_loss)
    scale = relative_scale * scale_delta + base_scale

    Examples:
    Given base_loss_and_scale=(0.4, 0.01), ref_loss_and_scale=(0.6, 0.02), rel_scale_range=(-0.5, 10).
    Each loss_delta = ref_loss - base_loss = 0.6 - 0.4 = 0.2, corresponds to a scale_delta of 0.01.
    - When loss = 0.8:
        relative_scale = (0.8 - 0.4) / 0.2 = 2
        scale_delta = 0.01
        scale = 2 * 0.01 + 0.01 = 0.03

    - When loss = 0.1:
        relative_scale = (0.1 - 0.4) / 0.2 = -1.5
        relative_scale (clamped) = -0.5
        scale = -0.5 * 0.01 + 0.01 = 0.005
    """    
    base_loss, base_scale = base_loss_and_scale
    ref_loss, ref_scale   = ref_loss_and_scale
    rel_scale_lb, rel_scale_ub = rel_scale_range

    # Ensure the losses are not equal, avoiding division by zero
    assert ref_loss != base_loss, "ref_loss and base_loss cannot be the same."
    assert rel_scale_lb > -1,     "rel_scale_lb must be greater than -1, otherwise the scale can be negative."

    relative_scale = (loss.item() - base_loss) / (ref_loss - base_loss)
    relative_scale = np.clip(relative_scale, rel_scale_lb, rel_scale_ub)
    scale_delta = ref_scale - base_scale
    scale = relative_scale * scale_delta + base_scale
    return scale
    

def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x

# norm_pow: 0 (no normalization), 0.5 (sqrt), 1 (normalization by L2 norm).
def normalized_sum(losses_list, norm_pow=0):
    loss_sum = sum(losses_list)
    if norm_pow == 0 or len(losses_list) == 0:
        return loss_sum
    
    normalized_losses_list = [ loss / np.power(np.abs(to_float(loss)) + 1e-8, norm_pow) for loss in losses_list ]
    new_loss_sum = sum(normalized_losses_list)
    # Restore original loss_sum.
    normalized_loss_sum = new_loss_sum * to_float(loss_sum) / (to_float(new_loss_sum) + 1e-8)
    if torch.isnan(normalized_loss_sum):
        breakpoint()
    return normalized_loss_sum

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

# embeddings: [N, 768]. 
# noise_std_range: the noise std / embeddings std falls within this range.
# anneal_perturb_embedding() adds noise of the amount randomly selected from the noise_std_range.
def anneal_perturb_embedding(embeddings, training_percent, begin_noise_std_range, end_noise_std_range, 
                             perturb_prob, perturb_std_is_relative=True, keep_norm=False,
                             std_dim=-1, norm_dim=-1, verbose=True):
    if torch.rand(1) > perturb_prob:
        return embeddings
    
    if end_noise_std_range is not None:
        noise_std_lb = anneal_value(training_percent, 1, (begin_noise_std_range[0], end_noise_std_range[0]))
        noise_std_ub = anneal_value(training_percent, 1, (begin_noise_std_range[1], end_noise_std_range[1]))
    else:
        noise_std_lb, noise_std_ub = begin_noise_std_range
        
    perturb_std = torch.rand(1).item() * (noise_std_ub - noise_std_lb) + noise_std_lb

    noised_embeddings = perturb_tensor(embeddings, perturb_std, perturb_std_is_relative, 
                                       keep_norm, std_dim, norm_dim, verbose=verbose)
    return noised_embeddings

# At scaled background, fill new x_start with random values (100% noise). 
# At scaled foreground, fill new x_start with noised scaled x_start. 
def init_x_with_fg_from_training_image(x_start, fg_mask, filtered_fg_mask, 
                                       base_scale_range=(0.8, 1.0),
                                       fg_noise_amount=0.2):
    x_start_maskfilled = torch.where(filtered_fg_mask.bool(), x_start, torch.randn_like(x_start))
    fg_mask_percent = filtered_fg_mask.float().sum() / filtered_fg_mask.numel()
    # print(fg_mask_percent)
    base_scale_range_lb, base_scale_range_ub = base_scale_range

    # Since our input images are portrait photos, the foreground areas are usually quite large,
    # between 0.2~0.5.
    if fg_mask_percent > 0.2:
        # If fg areas are larger (>= 0.1 of the whole image), then scale down
        # more aggressively, to avoid it dominating the whole image.
        # But don't take extra_scale linearly to this ratio, which will make human
        # faces (usually around 20%-50%) too small.
        # fg_mask_percent = 0.3, extra_scale = (2/3) ** 0.35 = 0.87 => face ratio = 0.3 * 0.87 = 0.261.
        # fg_mask_percent = 0.5, extra_scale = (2/5) ** 0.35 = 0.73 => face ratio = 0.5 * 0.73 = 0.365.
        extra_scale = math.pow(0.2 / fg_mask_percent.item(), 0.35)
        scale_range_lb = base_scale_range_lb * extra_scale
        # scale_range_ub is at least 0.5.
        scale_range_ub = max(0.5, base_scale_range_ub * extra_scale)
        fg_rand_scale = torch.rand(1).item() * (scale_range_ub - scale_range_lb) + scale_range_lb
    else:
        # fg areas are small. Scale the fg area to 80%-100% of the original size.
        fg_rand_scale = torch.rand(1).item() * (base_scale_range_ub - base_scale_range_lb) + base_scale_range_lb

    # Resize x_start_maskfilled and filtered_fg_mask by rand_scale. They have different numbers of channels,
    # so we need to concatenate them at dim 1 before resizing.
    x_mask = torch.cat([x_start_maskfilled, fg_mask, filtered_fg_mask], dim=1)
    x_mask_scaled = F.interpolate(x_mask, scale_factor=fg_rand_scale, mode='bilinear', align_corners=False)

    # Pad filtered_fg_mask_scaled to the original size, with left/right padding roughly equal
    pad_w1 = int((x_start.shape[3] - x_mask_scaled.shape[3]) / 2)
    pad_h1 = int((x_start.shape[2] - x_mask_scaled.shape[2]) / 2)
    pad_w2 =      x_start.shape[3] - x_mask_scaled.shape[3] - pad_w1
    pad_h2 =      x_start.shape[2] - x_mask_scaled.shape[2] - pad_h1
    # Add random perturbation to pad_w1 and pad_h1 to reduce overfitting.
    # Max perturbation is 25% of the smaller padding.
    max_w_perturb = min(pad_w1 - 1, pad_w2 - 1, 4)
    max_h_perturb = min(pad_h1 - 1, pad_h2 - 1, 4)

    if max_w_perturb > 0:
        delta_w1 = torch.randint(-max_w_perturb, max_w_perturb, (1,)).item()
    else:
        delta_w1 = 0
    if max_h_perturb > 0:
        delta_h1 = torch.randint(-max_h_perturb, max_h_perturb, (1,)).item()
    else:
        delta_h1 = 0

    pad_w1 += delta_w1
    pad_w2 -= delta_w1
    pad_h1 += delta_h1
    pad_h2 -= delta_h1

    x_mask_scaled_padded = F.pad(x_mask_scaled, (pad_w1, pad_w2, pad_h1, pad_h2),
                                 mode='constant', value=0)

    C1, C2, C3 = x_start_maskfilled.shape[1], fg_mask.shape[1], filtered_fg_mask.shape[1]
    # Unpack x_mask_scaled_padded into x_start, fg_mask and filtered_fg_mask.
    # x_start_scaled_padded: [2, 4, 64, 64]. fg_mask/filtered_fg_mask: [2, 1, 64, 64].
    x_start_scaled_padded, fg_mask, filtered_fg_mask \
        = x_mask_scaled_padded.split([C1, C2, C3], dim=1)

    # In filtered_fg_mask, the padded areas are filled with 0. 
    # So these pixels always take values from the random tensor.
    # In fg area, x_start takes values from x_start_scaled_padded 
    # (the fg of x_start_scaled_padded is a scaled-down version of the fg of the original x_start).
    x_start = torch.where(filtered_fg_mask.bool(), x_start_scaled_padded, torch.randn_like(x_start))
    # Add noise to the fg area. Noise amount is fixed at 0.2.
    # At the fg area, keep 80% of the original x_start values and add 20% of noise. 
    # x_start: [2, 4, 64, 64]
    x_start = torch.randn_like(x_start) * fg_noise_amount + x_start * (1 - fg_noise_amount)
    return x_start, fg_mask, filtered_fg_mask

# pixel-wise recon loss, weighted by fg_pixel_weight and bg_pixel_weight separately.
# fg_pixel_weight, bg_pixel_weight: could be 1D tensors of batch size, or scalars.
# img_mask, fg_mask:    [BS, 1, 64, 64] or None.
# model_output, target: [BS, 4, 64, 64].
def calc_recon_loss(loss_func, model_output, target, img_mask, fg_mask, 
                    fg_pixel_weight=1, bg_pixel_weight=1):

    if img_mask is None:
        img_mask = torch.ones_like(model_output)
    if fg_mask is None:
        fg_mask = torch.ones_like(model_output)
    
    # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
    model_output = model_output * img_mask
    target       = target       * img_mask
    loss_recon_pixels = loss_func(model_output, target, reduction='none')

    # fg_mask,              weighted_fg_mask.sum(): 1747, 1747
    # bg_mask=(1-fg_mask),  weighted_fg_mask.sum(): 6445, 887
    weighted_fg_mask = fg_mask       * img_mask * fg_pixel_weight
    weighted_bg_mask = (1 - fg_mask) * img_mask * bg_pixel_weight
    weighted_fg_mask = weighted_fg_mask.expand_as(loss_recon_pixels)
    weighted_bg_mask = weighted_bg_mask.expand_as(loss_recon_pixels)

    loss_recon = (  (loss_recon_pixels * weighted_fg_mask).sum()     \
                    + (loss_recon_pixels * weighted_bg_mask).sum() )   \
                    / (weighted_fg_mask.sum() + weighted_bg_mask.sum() + 1e-6)

    return loss_recon, loss_recon_pixels

# Major losses for normal_recon iterations (loss_recon, loss_recon_subj_mb_suppress, etc.).
# (But there are still other losses used after calling this function.)
def calc_recon_and_complem_losses(model_output, target, ca_layers_activations,
                                  all_subj_indices, img_mask, fg_mask, instances_have_fg_mask, 
                                  bg_pixel_weight, BLOCK_SIZE):

    loss_subj_mb_suppress = calc_subj_masked_bg_suppress_loss(ca_layers_activations['attnscore'],
                                                                all_subj_indices, BLOCK_SIZE, fg_mask, 
                                                                instances_have_fg_mask)

    # Ordinary image reconstruction loss under the guidance of subj_single_prompts.
    loss_recon, _ = calc_recon_loss(F.mse_loss, model_output, target, img_mask, fg_mask, 
                                    fg_pixel_weight=1, bg_pixel_weight=bg_pixel_weight)

    # Calc the L2 norm of model_output.
    loss_pred_l2 = (model_output ** 2).mean()
    return loss_subj_mb_suppress, loss_recon, loss_pred_l2

# calc_feat_delta_and_attn_norm_loss() is used by calc_comp_prompt_distill_loss().
def calc_feat_delta_and_attn_norm_loss(ca_outfeats, ca_attns, subj_indices_2b, BLOCK_SIZE):
    # do_comp_feat_distill iterations. No ordinary image reconstruction loss.
    # Only regularize on intermediate features, i.e., intermediate features generated 
    # under subj_comp_prompts should satisfy the delta loss constraint:
    # F(subj_comp_prompts)  - F(mix(subj_comp_prompts, cls_comp_prompts)) \approx 
    # F(subj_single_prompts) - F(cls_single_prompts)

    # Avoid doing distillation on the first few bottom layers (little difference).
    # distill_layer_weights: relative weight of each distillation layer. 
    # distill_layer_weights are normalized using distill_overall_weight.
    # Most important conditioning layers are 7, 8, 12, 16, 17. All the 5 layers have 1280 channels.
    # But intermediate layers also contribute to distillation. They have small weights.

    # feature map distillation only uses delta loss on the features to reduce the 
    # class polluting the subject features.
    # Feature map spatial sizes are all 64*64.
    feat_distill_layer_weights = { 23: 1, 24: 1, 
                                 }

    # attn norm distillation is applied to almost all conditioning layers.
    attn_norm_distill_layer_weights = { 
                                        23: 1., 24: 1.,                                   
                                      }

    # Normalize the weights above so that each set sum to 1.
    feat_distill_layer_weights          = normalize_dict_values(feat_distill_layer_weights)
    attn_norm_distill_layer_weights     = normalize_dict_values(attn_norm_distill_layer_weights)

    # K_subj: 4, number of embeddings per subject token.
    K_subj = len(subj_indices_2b[0]) // len(torch.unique(subj_indices_2b[0]))
    subj_indices_4b = double_token_indices(subj_indices_2b, BLOCK_SIZE * 2)

    loss_layers_feat_delta_align        = []
    loss_layers_subj_attn_norm_distill  = []

    subj_single_feat_grad_scaler = gen_gradient_scaler(0.1)

    for unet_layer_idx, ca_outfeat in ca_outfeats.items():
        if (unet_layer_idx not in feat_distill_layer_weights) and (unet_layer_idx not in attn_norm_distill_layer_weights):
            continue

        # attn_mat: [4, 8, 256, 77] => [4, 77, 8, 256].
        # We don't need BP through attention into UNet.
        attn_mat = ca_attns[unet_layer_idx].permute(0, 3, 1, 2)
        # subj_attn_4b: [4, 8, 256]  (1 embedding  for 1 token)  => [4, 1, 8, 256] => [4, 8, 256]
        # or            [16, 8, 256] (4 embeddings for 1 token)  => [4, 4, 8, 256] => [4, 8, 256]
        # BLOCK_SIZE*4: this batch contains 4 blocks. Each block should have one instance.
        subj_attn_4b = attn_mat[subj_indices_4b].reshape(BLOCK_SIZE*4, K_subj, *attn_mat.shape[2:]).sum(dim=1)
        # subj_single_subj_attn, ...: [1, 8, 256] (1 embedding  for 1 token) 
        # or                          [1, 8, 256] (4 embeddings for 1 token)
        subj_single_subj_attn, subj_comp_subj_attn, cls_single_subj_attn, cls_comp_subj_attn \
            = subj_attn_4b.chunk(4)

        if unet_layer_idx in attn_norm_distill_layer_weights:
            attn_norm_distill_layer_weight     = attn_norm_distill_layer_weights[unet_layer_idx]

            cls_comp_subj_attn_gs       = cls_comp_subj_attn.detach()
            cls_single_subj_attn_gs     = cls_single_subj_attn.detach()

            # mean(dim=-1): average across the 64 feature channels.
            # Align the attention corresponding to each embedding individually.
            # Note cls_*subj_attn use *_gs versions.
            # The L1 loss of the average attention values of the subject tokens, at each head and each instance.
            loss_layer_subj_comp_attn_norm   = F.l1_loss(subj_comp_subj_attn.abs().mean(dim=-1), cls_comp_subj_attn_gs.abs().mean(dim=-1))
            loss_layer_subj_single_attn_norm = F.l1_loss(subj_single_subj_attn.abs().mean(dim=-1), cls_single_subj_attn_gs.abs().mean(dim=-1))
            # loss_subj_attn_norm_distill uses L1 loss, which tends to be in 
            # smaller magnitudes than the delta loss. So it will be scaled up later in p_losses().
            loss_layers_subj_attn_norm_distill.append(( loss_layer_subj_comp_attn_norm + loss_layer_subj_single_attn_norm ) \
                                                        * attn_norm_distill_layer_weight)

        if unet_layer_idx not in feat_distill_layer_weights:
            continue

        feat_distill_layer_weight = feat_distill_layer_weights[unet_layer_idx]

        # [4, 320, 64, 64] -> [4, 320, 31, 31]
        ca_outfeat = pool_feat_or_attn_mat(ca_outfeat)
        # ca_outfeat_3d: [4, 320, 31, 31] -> [4, 320, 961] -> [4, 961, 320]
        ca_outfeat_3d = ca_outfeat.reshape(*ca_outfeat.shape[:2], -1).permute(0, 2, 1)
        # subj_single_feat_3d, ...: [1, 961, 320]
        subj_single_feat_3d, subj_comp_feat_3d, cls_single_feat_3d, cls_comp_feat_3d \
            = ca_outfeat_3d.chunk(4)

        cls_single_feat_3d_gs  = cls_single_feat_3d.detach()
        cls_comp_feat_3d_gs    = cls_comp_feat_3d.detach()
        # subj_single_feat_grad_scaler reduces grad by 10x.
        subj_single_feat_3d_gs = subj_single_feat_grad_scaler(subj_single_feat_3d)

        # ortho_subtract() is done on each image token individually, potentially with different scales.
        comp_feat_delta   = ortho_subtract(subj_comp_feat_3d,      cls_comp_feat_3d_gs)
        # subj_single_feat is gs'ed by 10x to avoid it from degeneration.
        single_feat_delta = ortho_subtract(subj_single_feat_3d_gs, cls_single_feat_3d_gs)
            
        # single_feat_delta, comp_feat_delta: [1, 320], ...
        # Pool the spatial dimensions H, W to remove spatial information.
        # The gradient goes back to single_feat_delta -> subj_comp_feat,
        # as well as comp_feat_delta -> cls_comp_feat.
        # If stop_single_grad, the gradients to subj_single_feat and cls_single_feat are stopped, 
        # as these two images should look good by themselves (since they only contain the subject).
        # Note the learning strategy to the single image features should be different from 
        # the single embeddings, as the former should be optimized to look good by itself,
        # while the latter should be optimized to cater for two objectives: 1) the conditioned images look good,
        # and 2) the embeddings are amendable to composition.
        loss_layer_feat_delta_align = \
            calc_ref_cosine_loss(comp_feat_delta, single_feat_delta, 
                                 emb_mask=None,
                                 exponent=2, do_demeans=[False, False],
                                 first_n_dims_into_instances=2, 
                                 aim_to_align=True, 
                                 ref_grad_scale=1)
        loss_layers_feat_delta_align.append(loss_layer_feat_delta_align * feat_distill_layer_weight)

    loss_feat_delta_align       = sum(loss_layers_feat_delta_align)
    loss_subj_attn_norm_distill = sum(loss_layers_subj_attn_norm_distill)

    return loss_feat_delta_align, loss_subj_attn_norm_distill

# calc_subj_masked_bg_suppress_loss() is called only during normal recon iterations.
def calc_subj_masked_bg_suppress_loss(ca_attnscore, subj_indices, BLOCK_SIZE, 
                                      fg_mask, instance_has_fg_mask=None):
    if (subj_indices is None) or (len(subj_indices) == 0) or (fg_mask is None) \
        or (instance_has_fg_mask is not None and instance_has_fg_mask.sum() == 0):
        return 0

    # Discard the first few bottom layers from alignment.
    # attn_align_layer_weights: relative weight of each layer. 
    # Feature map spatial sizes are all 64*64.
    attn_align_layer_weights = { 23: 1, 24: 1, 
                                }
            
    # Normalize the weights above so that each set sum to 1.
    attn_align_layer_weights = normalize_dict_values(attn_align_layer_weights)
    # K_subj: 9, number of embeddings per subject token.
    K_subj = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
    subj_mb_suppress_scale      = 0.05
    mfmb_contrast_attn_margin   = 0.4

    # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
    #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
    subj_indices = (subj_indices[0][:BLOCK_SIZE*K_subj], subj_indices[1][:BLOCK_SIZE*K_subj])

    loss_layers_subj_mb_suppress    = []

    for unet_layer_idx, unet_attn in ca_attnscore.items():
        if (unet_layer_idx not in attn_align_layer_weights):
            continue

        attn_align_layer_weight = attn_align_layer_weights[unet_layer_idx]
        # [2, 8, 256, 77] / [2, 8, 64, 77] =>
        # [2, 77, 8, 256] / [2, 77, 8, 64]
        attn_mat = unet_attn.permute(0, 3, 1, 2)

        # subj_attn: [8, 8, 64] -> [2, 4, 8, 64] sum among K_subj embeddings -> [2, 8, 64]
        subj_attn = sel_emb_attns_by_indices(attn_mat, subj_indices, do_sum=True, do_mean=False)

        fg_mask2 = resize_mask_to_target_size(fg_mask, "fg_mask", subj_attn.shape[-1], 
                                                mode="nearest|bilinear")
        # Repeat 8 times to match the number of attention heads (for normalization).
        fg_mask2 = fg_mask2.reshape(BLOCK_SIZE, 1, -1).repeat(1, subj_attn.shape[1], 1)
        fg_mask3 = torch.zeros_like(fg_mask2)
        # Set fractional values (due to resizing) to 1.
        fg_mask3[fg_mask2 >  1e-6] = 1.

        bg_mask3 = (1 - fg_mask3)

        if (fg_mask3.sum(dim=(1, 2)) == 0).any():
            # Very rare cases. Safe to skip.
            print("WARNING: fg_mask3 has all-zero masks.")
            continue
        if (bg_mask3.sum(dim=(1, 2)) == 0).any():
            # Very rare cases. Safe to skip.
            print("WARNING: bg_mask3 has all-zero masks.")
            continue

        subj_attn_at_mf = subj_attn * fg_mask3
        # Protect subject emb activations on fg areas.
        subj_attn_at_mf = subj_attn_at_mf.detach()
        # subj_attn_at_mb: [BLOCK_SIZE, 8, 64].
        # mb: mask foreground locations, mask background locations.
        subj_attn_at_mb = subj_attn * bg_mask3

        # fg_mask3: [BLOCK_SIZE, 8, 64]
        # avg_subj_attn_at_mf: [BLOCK_SIZE, 1, 1]
        # keepdim=True, since attn probs at all locations will use them as references (subtract them).
        avg_subj_attn_at_mf = masked_mean(subj_attn_at_mf, fg_mask3, dim=(1,2), keepdim=True)

        '''
        avg_subj_attn_at_mb = masked_mean(subj_attn_at_mb, bg_mask3, dim=(1,2), keepdim=True)
        if 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
            print(f'layer {unet_layer_idx}')
            print(f'avg_subj_attn_at_mf: {avg_subj_attn_at_mf.mean():.4f}, avg_subj_attn_at_mb: {avg_subj_attn_at_mb.mean():.4f}')
        '''

        # Encourage avg_subj_attn_at_mf (subj_attn averaged at foreground locations) 
        # to be at least larger by mfmb_contrast_attn_margin = 0.4 than 
        # subj_attn_at_mb at any background locations.
        # If not, clamp() > 0, incurring a loss.
        # layer_subj_mb_excess: [BLOCK_SIZE, 8, 64].
        layer_subj_mb_excess = subj_attn_at_mb + mfmb_contrast_attn_margin - avg_subj_attn_at_mf
        # Compared to masked_mean(), mean() is like dynamically reducing the loss weight when more and more 
        # activations conform to the margin restrictions.
        loss_layer_subj_mb_suppress   = masked_mean(layer_subj_mb_excess, 
                                                    layer_subj_mb_excess > 0, 
                                                    instance_weights=instance_has_fg_mask)

        # loss_layer_subj_bg_contrast_at_mf is usually 0, 
        # so loss_subj_mb_suppress is much smaller than loss_bg_mf_suppress.
        # subj_mb_suppress_scale: 0.05.
        loss_layers_subj_mb_suppress.append(loss_layer_subj_mb_suppress \
                                            * attn_align_layer_weight * subj_mb_suppress_scale)
        
    loss_subj_mb_suppress = sum(loss_layers_subj_mb_suppress)

    return loss_subj_mb_suppress


def calc_comp_prompt_distill_loss(flow_model, ca_layers_activations, 
                                  is_comp_init_fg_from_training_image, fg_mask_avail_ratio,
                                  filtered_fg_mask, all_subj_indices_1b, BLOCK_SIZE, 
                                  loss_dict, session_prefix,
                                  recon_feat_objectives=['attn_out', 'outfeat'],
                                  recon_loss_discard_thres=0.4):
    # ca_outfeats is a dict as: layer_idx -> ca_outfeat. 
    # It contains the 3 specified cross-attention layers of UNet. i.e., layers 22, 23, 24.
    # Similar are ca_attns and ca_attns, each ca_outfeats in ca_outfeats is already 4D like [4, 8, 64, 64].
    ca_outfeats  = ca_layers_activations['outfeat']

    # NOTE: loss_comp_fg_bg_preserve is applied only when comp_init_fg_from_training_image, 
    # i.e., when the fg_mask is available.
    if is_comp_init_fg_from_training_image and fg_mask_avail_ratio > 0:
        # In fg_mask, if an instance has no mask, then its fg_mask is all 1, including the background. 
        # Therefore, using fg_mask for comp_init_fg_from_training_image will force the model remember 
        # the background in the training images, which is not desirable.
        # In filtered_fg_mask, if an instance has no mask, then its fg_mask is all 0, 
        # excluding the instance from the fg_bg_preserve_loss.

        comp_subj_bg_preserve_loss_dict = \
            calc_comp_subj_bg_preserve_loss(flow_model, ca_outfeats,
                                            ca_layers_activations['attn_out'],
                                            ca_layers_activations['q'],
                                            ca_layers_activations['attn'], 
                                            filtered_fg_mask, all_subj_indices_1b, BLOCK_SIZE,
                                            recon_feat_objectives=recon_feat_objectives,
                                            recon_loss_discard_thres=recon_loss_discard_thres,
                                            do_feat_attn_pooling=True)
        
        loss_names = [ 'loss_sc_recon_ssfg_attn_agg', 'loss_sc_recon_ssfg_flow', 'loss_sc_recon_ssfg_min', 
                       'loss_sc_recon_mc_attn_agg', 'loss_sc_recon_mc_flow', 'loss_sc_recon_mc_min',
                       'loss_sc_to_ssfg_flow_attns_distill', 'loss_sc_to_mc_flow_attns_distill',
                       'loss_comp_subj_bg_attn_suppress', 'sc_bg_percent' ]
        
        # loss_sc_recon_ssfg_attn_agg and loss_sc_recon_ssfg_flow, 
        # are returned to be monitored, not to be optimized.
        # Only their counterparts -- loss_sc_recon_ssfg_min, loss_comp_subj_bg_attn_suppress 
        # are optimized.
        loss_sc_recon_ssfg_attn_agg, loss_sc_recon_ssfg_flow, loss_sc_recon_ssfg_min, \
        loss_sc_recon_mc_attn_agg, loss_sc_recon_mc_flow, loss_sc_recon_mc_min, \
        loss_sc_to_ssfg_flow_attns_distill, loss_sc_to_mc_flow_attns_distill, \
        loss_comp_subj_bg_attn_suppress, sc_bg_percent \
            = [ comp_subj_bg_preserve_loss_dict.get(loss_name, 0) for loss_name in loss_names ] 

        for loss_name in loss_names:
            if loss_name in comp_subj_bg_preserve_loss_dict and comp_subj_bg_preserve_loss_dict[loss_name] > 0:
                loss_name2 = loss_name.replace('loss_', '')
                # Accumulate the loss values to loss_dict when there are multiple denoising steps.
                add_dict_to_dict(loss_dict, {f'{session_prefix}/{loss_name2}': comp_subj_bg_preserve_loss_dict[loss_name].mean().detach().item() })

        comp_subj_bg_attn_suppress_loss_scale = 0.02
        sc_recon_ssfg_loss_scale = 10
        # loss_sc_recon_mc is L2 loss, which is very small. So we scale it up by 300x.
        # loss_sc_recon_mc: 0.004, sc_recon_mc_loss_scale: 300 => 1.2
        sc_recon_mc_loss_scale   = 300

        # loss_sc_recon_ssfg_min: 0.1~0.12. -> 1~1.2.
        loss_sc_recon_ssfg = loss_sc_recon_ssfg_min * sc_recon_ssfg_loss_scale
        loss_sc_recon_mc   = loss_sc_recon_mc_min   * sc_recon_mc_loss_scale
        # loss_sc_recon_mc_min:             0.005~0.008 -> 0.25~0.4.
        # loss_comp_subj_bg_attn_suppress:  0.1~0.2     -> 0.002~0.004.
        # loss_sc_recon_mc_min has similar effects to suppress the subject attn values in the background tokens.
        # Therefore, loss_comp_subj_bg_attn_suppress is given a very small comp_subj_bg_attn_suppress_loss_scale = 0.02.
        loss_comp_fg_bg_preserve = loss_sc_recon_ssfg + loss_sc_recon_mc \
                                   + loss_comp_subj_bg_attn_suppress * comp_subj_bg_attn_suppress_loss_scale \
                                   + loss_sc_to_ssfg_flow_attns_distill + loss_sc_to_mc_flow_attns_distill
    else:
        loss_comp_fg_bg_preserve = torch.zeros(1, device=ca_outfeats[23].device)

    return loss_comp_fg_bg_preserve

# Intuition of comp_fg_bg_preserve_loss: 
# In distillation iterations, if comp_init_fg_from_training_image, then at fg_mask areas, x_start is initialized with 
# the noisy input images. (Otherwise in distillation iterations, x_start is initialized as pure noise.)
# Essentially, it's to mask the background out of the input images with noise.
# Therefore, intermediate features at the foreground with single prompts should be close to those of the original images.
# Features with comp prompts should be similar with the original images at the foreground.
# So features under comp prompts should be close to features under single prompts, at fg_mask areas.
# (The features at background areas under comp prompts are the compositional contents, which shouldn't be regularized.) 
# NOTE: subj_indices are used to compute loss_comp_subj_bg_attn_suppress.
# Only ss_fg_mask in (resized) filtered_fg_mask is used for calc_elastic_matching_loss().
def calc_comp_subj_bg_preserve_loss(flow_model, ca_outfeats, ca_attn_outs, ca_qs, ca_attns, 
                                    filtered_fg_mask, subj_indices, BLOCK_SIZE,
                                    recon_feat_objectives=['attn_out', 'outfeat'], 
                                    recon_loss_discard_thres=0.4, do_feat_attn_pooling=True):
    # No masks are available. loss_comp_subj_fg_feat_preserve, loss_comp_subj_bg_attn_suppress are both 0.
    if filtered_fg_mask is None or filtered_fg_mask.sum() == 0:
        return {}

    # Feature map spatial sizes are all 64*64.
    # Remove layer 22, as the losses at this layer are often too large 
    # and are discarded at a high percentage.
    elastic_matching_layer_weights = { 23: 1, 24: 1, 
                                     }
    
    # Normalize the weights above so that each set sum to 1.
    elastic_matching_layer_weights  = normalize_dict_values(elastic_matching_layer_weights)

    # K_subj: 4, number of embeddings per subject token.
    K_subj = len(subj_indices[0]) // len(torch.unique(subj_indices[0]))
    # subj_indices: ([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
    #                [5, 6, 7, 8, 6, 7, 8, 9, 5, 6, 7, 8, 6, 7, 8, 9]).
    # ind_subj_subj_B_1b, ind_subj_subj_N_1b: [0, 0, 0, 0], [5, 6, 7, 8].
    ind_subj_subj_B_1b, ind_subj_subj_N_1b = subj_indices[0][:BLOCK_SIZE*K_subj], subj_indices[1][:BLOCK_SIZE*K_subj]
    ind_subj_B = torch.cat([ind_subj_subj_B_1b,                     ind_subj_subj_B_1b + BLOCK_SIZE,
                            ind_subj_subj_B_1b + 2 * BLOCK_SIZE,    ind_subj_subj_B_1b + 3 * BLOCK_SIZE], dim=0)
    ind_subj_N = ind_subj_subj_N_1b.repeat(4)
    
    loss_dict = {}
    
    for unet_layer_idx, ca_outfeat in ca_outfeats.items():
        if unet_layer_idx not in elastic_matching_layer_weights:
            continue
        LAYER_W = elastic_matching_layer_weights[unet_layer_idx]

        # ca_outfeat: [4, 1280, 8, 8]
        ca_feat_h, ca_feat_w = ca_outfeat.shape[-2:]

        # ca_layer_q: [4, 1280, 64] -> [4, 1280, 8, 8]
        ca_layer_q  = ca_qs[unet_layer_idx]
        ca_attn_out = ca_attn_outs[unet_layer_idx]
        # This way of calculation ca_q_h is to consider the case when the height and width might not be the same.
        ca_q_h = int(np.sqrt(ca_layer_q.shape[2] * ca_outfeat.shape[2] // ca_outfeat.shape[3]))
        ca_q_w = ca_layer_q.shape[2] // ca_q_h
        ca_layer_q = ca_layer_q.reshape(ca_layer_q.shape[0], -1, ca_q_h, ca_q_w)

        # ca_attn_out: [B, D, N] -> [B, D, H, W].
        ca_attn_out = ca_attn_out.reshape(*ca_attn_out.shape[:2], ca_feat_h, ca_feat_w)

        # Some layers resize the input feature maps. So we need to resize ca_outfeat to match ca_layer_q.
        if ca_outfeat.shape[2:] != ca_layer_q.shape[2:]:
            ca_outfeat = F.interpolate(ca_outfeat, size=ca_layer_q.shape[2:], mode="bilinear", align_corners=False)
            
        ###### elastic matching loss ######
        # q of each layer is used to compute the correlation matrix between subject-single and subject-comp instances,
        # as well as class-single and class-comp instances.
        # ca_attn_out is used to compute the reconstruction loss between subject-single and subject-comp instances 
        # (using the correlation matrix), as well as class-single and class-comp instances.
        # Flatten the spatial dimensions of ca_attn_out.
        # ca_layer_q, ca_attn_out, ca_outfeat: [4, 1280, 8, 8] -> [4, 1280, 64].
        ca_layer_q  = ca_layer_q.reshape(*ca_layer_q.shape[:2], -1)
        ca_attn_out = ca_attn_out.reshape(*ca_attn_out.shape[:2], -1)
        ca_outfeat  = ca_outfeat.reshape(*ca_outfeat.shape[:2], -1)
        # fg_mask_4b: [4, 1, 64, 64] => [4, 1, 8, 8]
        fg_mask_4b \
            = resize_mask_to_target_size(filtered_fg_mask, "fg_mask_4b", (ca_feat_h, ca_feat_w), 
                                         mode="nearest|bilinear", warn_on_all_zero=False)
        # ss_fg_mask: [4, 1, 8, 8] -> [1, 1, 8, 8] 
        ss_fg_mask = fg_mask_4b.chunk(4)[0]
        # ss_fg_mask_3d: [1, 1, 8, 8] -> [1, 1, 64]. Spatial dims are collapsed.
        ss_fg_mask_3d = ss_fg_mask.reshape(*ss_fg_mask.shape[:2], -1)

        # sc_to_ss_fg_prob, mc_to_ms_fg_prob: [1, 1, 64]
        # removed loss_layer_ms_mc_fg_match to save computation.
        # loss_layer_subj_comp_map_single_align_with_cls: loss of alignment between two soft mappings: sc_to_ss_prob and mc_to_ms_prob.
        # sc_to_ss_fg_prob_below_mean is used as fg/bg soft masks of comp instances
        # to suppress the activations on background areas.
        
        losses_sc_recons, loss_flow_attns_distill, sc_to_ss_fg_prob, sc_to_whole_mc_prob = \
            calc_elastic_matching_loss(unet_layer_idx, flow_model, 
                                       ca_layer_q, ca_attn_out, ca_outfeat, 
                                       ss_fg_mask_3d, ca_feat_h, ca_feat_w, 
                                       recon_feat_objectives=recon_feat_objectives,
                                       recon_loss_discard_thres=recon_loss_discard_thres,
                                       num_flow_est_iters=12,
                                       do_feat_attn_pooling=do_feat_attn_pooling)

        if losses_sc_recons is None:
            continue

        loss_sc_recon_ssfg_attn_agg, loss_sc_recon_ssfg_flow, loss_sc_recon_ssfg_min = losses_sc_recons['ssfg']
        loss_sc_recon_mc_attn_agg, loss_sc_recon_mc_flow, loss_sc_recon_mc_min = losses_sc_recons['mc']
        loss_sc_to_ssfg_flow_attns_distill, loss_sc_to_mc_flow_attns_distill = \
            loss_flow_attns_distill['ssfg'], loss_flow_attns_distill['mc']
        
        add_dict_to_dict(loss_dict, 
                         { 'loss_sc_recon_ssfg_attn_agg':   loss_sc_recon_ssfg_attn_agg * LAYER_W,
                           'loss_sc_recon_ssfg_flow':       loss_sc_recon_ssfg_flow * LAYER_W,
                           'loss_sc_recon_ssfg_min':        loss_sc_recon_ssfg_min * LAYER_W,
                           'loss_sc_recon_mc_attn_agg':     loss_sc_recon_mc_attn_agg * LAYER_W,
                           'loss_sc_recon_mc_flow':         loss_sc_recon_mc_flow * LAYER_W,
                           'loss_sc_recon_mc_min':          loss_sc_recon_mc_min * LAYER_W,
                           'loss_sc_to_ssfg_flow_attns_distill': loss_sc_to_ssfg_flow_attns_distill * LAYER_W,
                           'loss_sc_to_mc_flow_attns_distill':   loss_sc_to_mc_flow_attns_distill * LAYER_W
                         })
        
        ##### unet_attn fg preservation loss & bg suppression loss #####
        unet_attn = ca_attns[unet_layer_idx]
        # attn_mat: [4, 8, 256, 77] => [4, 77, 8, 256] 
        attn_mat = unet_attn.permute(0, 3, 1, 2)
        # subj_subj_attn: [4, 77, 8, 256] -> [4 * K_subj, 8, 256] -> [4, K_subj, 8, 256]
        # attn_mat and subj_subj_attn are not pooled.
        subj_attn = attn_mat[ind_subj_B, ind_subj_N].reshape(BLOCK_SIZE * 4, K_subj, *attn_mat.shape[2:])
        # Sum over 9 subject embeddings. [4, K_subj, 8, 256] -> [4, 8, 256].
        # The scale of the summed attention won't be overly large, since we've done 
        # distribute_embedding_to_M_tokens() to them.
        subj_attn = subj_attn.sum(dim=1)
        H = int(np.sqrt(subj_attn.shape[-1]))
        # subj_attn_hw: [4, 8, 256] -> [4, 8, 8, 8].
        subj_attn_hw = subj_attn.reshape(*subj_attn.shape[:2], H, H)
        # At some layers, the output features are upsampled. So we need to 
        # upsample the attn map to match the output features.
        if subj_attn_hw.shape[2:] != (ca_feat_h, ca_feat_w):
            subj_attn_hw = F.interpolate(subj_attn_hw, size=(ca_feat_h, ca_feat_w), mode="bilinear", align_corners=False)

        # subj_attn_hw: [4, 8, 8, 8] -> [4, 8, 8, 8] -> [4, 8, 64].
        subj_attn_flat = subj_attn_hw.reshape(*subj_attn_hw.shape[:2], -1)

        subj_single_subj_attn, subj_comp_subj_attn, cls_single_subj_attn, cls_comp_subj_attn \
            = subj_attn_flat.chunk(4)

        cls_comp_subj_attn_gs = cls_comp_subj_attn.detach()

        subj_comp_subj_attn_pos   = subj_comp_subj_attn.clamp(min=0)
        cls_comp_subj_attn_gs_pos = cls_comp_subj_attn_gs.clamp(min=0)

        if do_feat_attn_pooling:
            subj_comp_subj_attn_pos   = pool_feat_or_attn_mat(subj_comp_subj_attn_pos,   (ca_feat_h, ca_feat_w))
            cls_comp_subj_attn_gs_pos = pool_feat_or_attn_mat(cls_comp_subj_attn_gs_pos, (ca_feat_h, ca_feat_w))

        # Suppress the subj attention probs on background areas in comp instances.
        # subj_comp_subj_attn: [1, 320, 961]. sc_to_whole_mc_prob: [1, 1, 961].
        loss_layer_comp_subj_bg_attn_suppress = masked_mean(subj_comp_subj_attn_pos, 
                                                            sc_to_whole_mc_prob.permute(0, 2, 1))

        # sc_to_whole_mc_prob.mean() = 1, but sc_to_ss_fg_prob.mean() = N_fg / 961 = 210 / 961 = 0.2185.
        # So we normalize sc_to_ss_fg_prob first, before comparing it with sc_to_whole_mc_prob.
        sc_bg_percent = (sc_to_whole_mc_prob > sc_to_ss_fg_prob / sc_to_ss_fg_prob.mean() + 0.01).float().mean()

        add_dict_to_dict(loss_dict,
                            { 'loss_comp_subj_bg_attn_suppress': loss_layer_comp_subj_bg_attn_suppress * LAYER_W,
                              'sc_bg_percent': sc_bg_percent * LAYER_W })
    
    return loss_dict

# features/attention pooling allows small perturbations of the locations of pixels.
# pool_feat_or_attn_mat() selects a proper pooling kernel size and stride size 
# according to the feature map size.
def pool_feat_or_attn_mat(feat_or_attn_mat, spatial_shape=None, retain_spatial=False, debug=False):
    # feature map size -> [kernel size, stride size] of the pooler.
    # 16 -> 4, 2 (output 7), 32 -> 4, 2 (output 15),  64 -> 8, 4 (output 15).
    feat_size2pooler_spec = { 8: [2, 1], 16: [2, 1], 32: [4, 2], 64: [4, 2] }
    # 3D should be attention maps. 4D should be feature maps.
    # For attention maps, the last 2 dims are flattened to 1D. So we need to unflatten them.
    feat_or_attn_mat0 = feat_or_attn_mat

    # Attention matrix without the head dim.
    if feat_or_attn_mat.ndim == 2:
        feat_or_attn_mat = feat_or_attn_mat.unsqueeze(1)

    # Attention matrix with the head dim.
    if feat_or_attn_mat.ndim == 3:
        do_unflatten = True
        if spatial_shape is not None:
            H, W = spatial_shape
        else:
            H = W = int(np.sqrt(feat_or_attn_mat.shape[-1]))

        feat_or_attn_mat = feat_or_attn_mat.reshape(*feat_or_attn_mat.shape[:2], H, W)
    else:
        do_unflatten = False

    if feat_or_attn_mat.shape[-1] not in feat_size2pooler_spec:
        if debug:
            print(f"{feat_or_attn_mat0.shape} not in feat_size2pooler_spec.")
        return feat_or_attn_mat0
    
    # 16 -> 4, 2 (output 7), 32 -> 4, 2 (output 15),  64 -> 8, 4 (output 15).
    pooler_kernel_size, pooler_stride = feat_size2pooler_spec[feat_or_attn_mat.shape[-1]]

    # feature pooling: allow small perturbations of the locations of pixels.
    # If subj_single_feat is 8x8, then after pooling, it becomes 3x3, too rough.
    # The smallest feat shape > 8x8 is 16x16 => 7x7 after pooling.
    pooler = nn.AvgPool2d(pooler_kernel_size, stride=pooler_stride)
    feat_or_attn_mat2 = pooler(feat_or_attn_mat)
    # If the spatial dims are unflattened and if retain_spatial=False,
    # we flatten the spatial dims.
    if feat_or_attn_mat2.ndim == 4 and do_unflatten and not retain_spatial:
        feat_or_attn_mat2 = feat_or_attn_mat2.reshape(*feat_or_attn_mat2.shape[:2], -1)
    # If the input is 2D, we should remove the extra dim unsqueezed above.
    if feat_or_attn_mat0.ndim == 2:
        feat_or_attn_mat2 = feat_or_attn_mat2.squeeze(1)

    if debug:
        print(f"{list(feat_or_attn_mat0.shape)} -> {list(feat_or_attn_mat.shape)} "
              f"(ks, stride)=({pooler_kernel_size}, {pooler_stride}) => {list(feat_or_attn_mat2.shape)}")

    return feat_or_attn_mat2

def resize_flow(flow, H, W):
    if flow.ndim == 4:
        flow[:, 0, :, :] *= W / flow.shape[3]
        flow[:, 1, :, :] *= H / flow.shape[2]
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    elif flow.ndim == 3:
        flow[0, :, :] *= W / flow.shape[2]
        flow[1, :, :] *= H / flow.shape[1]
        flow = F.interpolate(flow.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0]
    else:
        breakpoint()
    return flow

'''
def backward_warp_by_flow_np(image2, flow1to2):
    H, W, _ = image2.shape
    flow1to2 = flow1to2.copy()
    flow1to2[:, :, 0] += np.arange(W)  # Adjust x-coordinates
    flow1to2[:, :, 1] += np.arange(H)[:, None]  # Adjust y-coordinates
    image1_recovered = cv2.remap(image2, flow1to2, None, cv2.INTER_LINEAR)
    return image1_recovered
'''

#@torch.compiler.disable
@torch.compile
def backward_warp_by_flow(image2, flow1to2):
    # Assuming image2 is a PyTorch tensor of shape (B, C, H, W)
    B, C, H, W = image2.shape

    # Create meshgrid for coordinates.
    # NOTE: the default indexing is 'ij', so we need to use 'xy' to get the correct grid.
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    grid_x = grid_x.to(flow1to2.device)
    grid_y = grid_y.to(flow1to2.device)
    
    # Adjust the flow to compute absolute coordinates
    flow_x = flow1to2[:, 0, :, :] + grid_x  # Add x coordinates
    flow_y = flow1to2[:, 1, :, :] + grid_y  # Add y coordinates

    # Normalize flow values to the range [-1, 1] as required by grid_sample
    flow_x = 2.0 * flow_x / (W - 1) - 1.0
    flow_y = 2.0 * flow_y / (H - 1) - 1.0

    # Stack and reshape the flow to form the sampling grid for grid_sample
    # flow: [B, H, W, 2]
    flow = torch.stack((flow_x, flow_y), dim=-1)
    # Perform backward warping using grid_sample
    # align_corners: If set to True, the extrema (-1 and 1) are considered as referring 
    # to the center points of the input's corner pixels
    image1_recovered = F.grid_sample(image2, flow, mode='bilinear', padding_mode='zeros', align_corners=True)

    return image1_recovered

@torch.compiler.disable
def flow2attn(s2c_flow, H, W, mask_N=None):
    # Generate a diagonal attention matrix from comp tokens (feature dim) to comp tokens (spatial dims).
    c_diag_attn = torch.eye(H*W, device=s2c_flow.device, dtype=s2c_flow.dtype).reshape(1, H*W, H, W).repeat(s2c_flow.shape[0], 1, 1, 1)
    # Backwarp the diagonal attention matrix by the flow, so that the comp tokens at spatial dims
    # aligns with single tokens after warping. Therefore, c_flow_attn is from comp tokens 
    # (feature dim) to single tokens (spatial dims).
    c_flow_attn = backward_warp_by_flow(c_diag_attn, s2c_flow)
    # c_flow_attn: [BS, H*W, H, W] -> [BS, H*W, H*W]
    c_flow_attn = c_flow_attn.reshape(*c_flow_attn.shape[:2], -1)
    if mask_N is not None:
        # mask_N is applied at the single tokens dim.
        c_flow_attn = c_flow_attn[:, :, mask_N]
    return c_flow_attn

#@torch.compiler.disable
@torch.compile
def reconstruct_feat_with_attn_aggregation(sc_feat, sc_to_ssfg_mc_prob):
    # recon_sc_feat: [1, 1280, 961] * [1, 961, 961] => [1, 1280, 961]
    # ** We use the subj comp tokens to reconstruct the subj single tokens, not vice versa. **
    # Because we rely on ss_fg_mask to determine the fg area, which is only available for the subj single instance. 
    # Then we can compare the values of the recon'ed subj single tokens with the original values at the fg area.
    # sc_to_ss_mc_prob: [1, 961, N_fg + 961].
    # Weighted sum of the comp tokens (based on their matching probs) to reconstruct the single tokens.
    # sc_recon_ssfg_mc_feat: [1, 1280, 961] * [1, 961, N_fg] => [1, 1280, N_fg]
    sc_recon_ssfg_mc_feat = torch.matmul(sc_feat, sc_to_ssfg_mc_prob)
    # sc_recon_ssfg_mc_feat: [1, 1280, N_fg] => [1, N_fg, 1280]
    sc_recon_ssfg_mc_feat = sc_recon_ssfg_mc_feat.permute(0, 2, 1)
    # mc_recon_ms_feat = torch.matmul(mc_feat, mc_to_ms_prob)

    return sc_recon_ssfg_mc_feat

@torch.compiler.disable
def reconstruct_feat_with_matching_flow(flow_model, ss2sc_flow, ss_q, sc_q, sc_feat, ss_fg_mask_2d, 
                                        H, W, num_flow_est_iters=12):
    # Set background features to 0 tensors reduce noisy matching.
    # ss_q: [1, 1280, 961]. ss_fg_mask_2d: [1, 961] -> [1, 1, 961]
    if ss_fg_mask_2d is not None:
        ss_q = ss_q * ss_fg_mask_2d.unsqueeze(1)

    # If ss2sc_flow is not provided, estimate it using the flow model.
    # Otherwise use the provided flow.
    if ss2sc_flow is None:
        # Latent optical flow from subj single feature maps to subj comp feature maps.
        # Enabling grad seems to lead to quite bad results. Maybe updating q through flow is not a good idea.
        with torch.no_grad():
            #LINK gma/network.py#est_flow_from_feats
            # ss2sc_flow: [1, 2, H, W]
            ss2sc_flow = flow_model.est_flow_from_feats(ss_q, sc_q, H, W, num_iters=num_flow_est_iters, 
                                                        corr_normalized_by_sqrt_dim=False)

    # Resize sc_feat to [1, *, H, W] and warp it using ss2sc_flow, 
    # then collapse the spatial dimensions.
    sc_feat             = sc_feat.reshape(*sc_feat.shape[:2], H, W)
    sc_recon_ss_feat    = backward_warp_by_flow(sc_feat, ss2sc_flow)
    sc_recon_ss_feat    = sc_recon_ss_feat.reshape(*sc_recon_ss_feat.shape[:2], -1)

    # ss_fg_mask_2d's spatial dim is already collapsed. ss_fg_mask_2d: [1, 225]
    # nonzero() returns (B, N) indices of True values as ss_fg_mask_B, ss_fg_mask_N.
    # So we use ss_fg_mask_N to index the last dim of sc_recon_ss_feat.
    '''
    (Pdb) ss_fg_mask_N
    tensor([ 17,  18,  19,  20,  21,  22,  23,  24,  31,  32,  33,  34,  35,  36,
            37,  38,  39,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  61,
            62,  63,  64,  65,  66,  67,  68,  69,  70,  76,  77,  78,  79,  80,
            81,  82,  83,  84,  85,  91,  92,  93,  94,  95,  96,  97,  98,  99,
            100, 107, 108, 109, 110, 111, 112, 113, 114, 115, 122, 123, 124, 125,
            126, 127, 128, 129, 130, 137, 138, 139, 140, 141, 142, 143, 144, 153,
            154, 155, 156, 157, 158, 167, 168, 169, 170, 171, 172, 173, 182, 183,
            184, 185, 186, 187, 188], device='cuda:0')
    '''    

    sc_recon_ssfg_feat = sc_recon_ss_feat.permute(0, 2, 1)
    if ss_fg_mask_2d is not None:
        ss_fg_mask_B, ss_fg_mask_N = ss_fg_mask_2d.nonzero(as_tuple=True)    
        # sc_recon_ss_feat: [1, 1280, 961] -> [1, 961, 1280] -> [1, N_fg, 1280]
        sc_recon_ssfg_feat = sc_recon_ssfg_feat[:, ss_fg_mask_N]
    
    return sc_recon_ssfg_feat, ss2sc_flow

# We can not simply switch ss_feat/ss_q with sc_feat/sc_q, and also change sc_to_ss_prob to ss_map_sc_prob, 
# to get ss-recon-sc losses.
#@torch.compiler.disable
@torch.compile
def calc_sc_recon_ssfg_mc_losses(layer_idx, flow_model, target_feats, sc_feat, 
                                 ss2sc_flow, mc2sc_flow, sc_to_ss_mc_prob, 
                                 ss_fg_mask_2d, ss_q, sc_q, mc_q, H, W, 
                                 num_flow_est_iters, objective_name):
    sc_recon_feats_attn_agg     = {}
    sc_recon_feats_flow         = {}
    sc_recon_feats_flow_attn    = {}
    sc_attns = {}
    # sc_recon_ssfg_mc_feat_attn_agg: [1, 1280, N_fg + 961] 
    sc_recon_ssfg_mc_feat_attn_agg = \
        reconstruct_feat_with_attn_aggregation(sc_feat, sc_to_ss_mc_prob)
    N_fg = target_feats['ssfg'].shape[1]
    # Split sc_recon_ssfg_mc_feat_attn_agg into ssfg and mc parts.
    # sc_recon_feat*['ssfg']: [1, 1280, N_fg] => [1, N_fg, 1280]
    sc_recon_feats_attn_agg['ssfg'], sc_recon_feats_attn_agg['mc'] = \
        sc_recon_ssfg_mc_feat_attn_agg[:, :N_fg], sc_recon_ssfg_mc_feat_attn_agg[:, N_fg:]
    # sc_to_ss_mc_prob: [1, 961, N_fg + 961] -> sc_attns['ssfg']: [1, 961, N_fg], sc_attns['mc']: [1, 961, 961].
    sc_attns['ssfg'], sc_attns['mc'] = sc_to_ss_mc_prob[:, :, :N_fg], sc_to_ss_mc_prob[:, :, N_fg:]
    # ss_fg_mask_2d: [1, 961]. ss_fg_mask_N: [N_fg].
    ss_fg_mask_B, ss_fg_mask_N = ss_fg_mask_2d.nonzero(as_tuple=True)

    if flow_model is not None or ss2sc_flow is not None:
        # ss2sc_flow: [1, 2, H, W]
        # If ss2sc_flow is not provided, estimate it using the flow model.
        # Otherwise ss2sc_flow is passed to reconstruct_feat_with_matching_flow() to be used,
        # and return the newly estimated ss2sc_flow.     
        sc_recon_feats_flow['ssfg'], ss2sc_flow = \
            reconstruct_feat_with_matching_flow(flow_model, ss2sc_flow, ss_q, sc_q, sc_feat, 
                                                ss_fg_mask_2d, H, W, num_flow_est_iters=num_flow_est_iters)
        sc_recon_feats_flow_attn['ssfg'] = flow2attn(ss2sc_flow, H, W, mask_N=ss_fg_mask_N)
    else:
        ss2sc_flow                      = None
        sc_recon_feats_flow['ssfg']     = None
        sc_recon_feats_flow_attn['ssfg'] = None
        
    if flow_model is not None or mc2sc_flow is not None:
        # mc2sc_flow: [1, 2, H, W]
        # If mc2sc_flow is not provided, estimate it using the flow model.
        # Otherwise mc2sc_flow is passed to reconstruct_feat_with_matching_flow() to be used,
        # and return the newly estimated mc2sc_flow.
        sc_recon_feats_flow['mc'], mc2sc_flow = \
            reconstruct_feat_with_matching_flow(flow_model, mc2sc_flow, mc_q, sc_q, sc_feat, 
                                                None, H, W, num_flow_est_iters=num_flow_est_iters)
        sc_recon_feats_flow_attn['mc'] = flow2attn(mc2sc_flow, H, W, mask_N=None)
    else:
        mc2sc_flow                      = None
        sc_recon_feats_flow['mc']       = None
        sc_recon_feats_flow_attn['mc']  = None

    losses_sc_recons          = {}
    all_token_losses_sc_recon = {}
    loss_flow_attns_distill   = {}
    matching_type_names       = ['attn', 'flow']

    for feat_name in sc_recon_feats_attn_agg:
        print(f"{objective_name} sc->{feat_name}:", end=' ')
        target_feat = target_feats[feat_name]
        losses_sc_recons[feat_name] = [] 
        all_token_losses_sc_recon[feat_name] = []

        for i, sc_recon_feat in enumerate((sc_recon_feats_attn_agg[feat_name], sc_recon_feats_flow[feat_name])):
            if sc_recon_feat is None:
                loss_sc_recon = torch.tensor(0., device=sc_feat.device)
            else:
                # target_feat has no grad. So no need to cut off the gradients of target_feat.
                token_losses_sc_recon = F.mse_loss(sc_recon_feat, target_feat, reduction='none')
                # Here loss_sc_recon_ssfg (corresponding to loss_sc_recon_ssfg_attn_agg, loss_sc_recon_ssfg_flow) 
                # is only for debugging. 
                # The optimized loss_sc_recon_ssfg is loss_sc_recon_ssfg_min, computed by 
                # taking the tokenwise min of the two losses.
                loss_sc_recon = token_losses_sc_recon.mean()
                all_token_losses_sc_recon[feat_name].append(token_losses_sc_recon)

            losses_sc_recons[feat_name].append(loss_sc_recon)
            print(f"{matching_type_names[i]} {loss_sc_recon}", end=' ')

        # We have both attn and flow token losses.
        if len(all_token_losses_sc_recon[feat_name]) > 1:
            # all_token_losses_sc_recon_ssfg: [2, 1, 72, 320]. 72: number of fg tokens.
            all_token_losses_sc_recon_2types = torch.stack(all_token_losses_sc_recon[feat_name], dim=0)
            # Calculate the mean loss at each token by averaging across the feature dim.
            all_token_losses_sc_recon_2types = all_token_losses_sc_recon_2types.mean(dim=3)
            # Take the smaller loss tokenwise between attn and flow.
            min_token_losses_sc_recon = all_token_losses_sc_recon_2types.min(dim=0).values
            # flow at a token aligns better if the flow loss < attn loss with a margin of 0.0002.
            flow_is_better = all_token_losses_sc_recon_2types[1] < all_token_losses_sc_recon_2types[0] - 0.0002
            if flow_is_better.any():
                # flow_is_better: [1, 140] -> [1, 961, 140]. Added a dim to be broadcasted to the sc tokens dim.
                flow_is_better = flow_is_better.unsqueeze(1).expand(-1, sc_recon_feats_flow_attn[feat_name].shape[1], -1)
                # sc_recon_feats_flow_attn[feat_name]: [1, 961, 140].
                loss_flow_attn_distill = (sc_recon_feats_flow_attn[feat_name] - sc_attns[feat_name])[flow_is_better].abs().mean()
            else:
                loss_flow_attn_distill = torch.tensor(0., device=sc_feat.device)
            loss_sc_recon_min = min_token_losses_sc_recon.mean()
        else:
            loss_sc_recon_min = [ loss for loss in losses_sc_recons[feat_name] if loss != 0 ][0]
            loss_flow_attn_distill = torch.tensor(0., device=sc_feat.device)

        losses_sc_recons[feat_name].append(loss_sc_recon_min)
        loss_flow_attns_distill[feat_name] = loss_flow_attn_distill
        print(f"min {loss_sc_recon_min}, flow dist {loss_flow_attn_distill}")

    return losses_sc_recons, loss_flow_attns_distill, ss2sc_flow, mc2sc_flow

# recon_feat_objectives: ['attn_out', 'outfeat'], a list of feature names to be reconstructed,
# and the loss schemes for them.
# NOTE: in theory outfeat could precede attn_out when iterating recon_feat_objectives.keys(), but in the
# current implementation, attn_out is always iterated first. So we don't need to worry about the flow computation,
# which is based on attn_out, and reused for outfeat.
# We adopt L2 loss for attn_out and cosine for outfeat. After training, the loss on attn_out is usually much smaller than
# that on outfeat. Note attn_out is a feature map aggregated from the attention maps,
# so it's not probs, but similar to outfeat. So the small magnitude of the loss on attn_out indicates good matching,
# and isn't caused by inherently different scales of attn_out and outfeat.
# Using cosine instead of L2 for outfeat is because outfeat may have drastically different values for different tokens,
# and the loss may become too large sometimes and then discarded. 
# Cosine loss can limit the loss to [-1, 1], and is more robust to scale changes.
# Although in theory L2 and cosine losses may have different scales, for simplicity, 
# we still average them to get the total loss.
#@torch.compiler.disable
@torch.compile
def calc_elastic_matching_loss(layer_idx, flow_model, ca_q, ca_attn_out, ca_outfeat, ss_fg_mask_3d, H, W, 
                               sc_q_grad_scale=0.1, c_to_s_attn_norm_dims=(1,),
                               recon_feat_objectives=['attn_out', 'outfeat'], 
                               recon_loss_discard_thres=0.4, 
                               num_flow_est_iters=12, do_feat_attn_pooling=True):
    # ss_fg_mask_3d: [1, 1, 64*64]
    if ss_fg_mask_3d.sum() == 0:
        return None, None, None, None

    if do_feat_attn_pooling:
        # Pooling makes ca_outfeat spatially smoother, so that we'll get more continuous flow.
        # ca_attn_out, ca_outfeat, ca_q: [4, 1280, 64*64] -> [4, 1280, 31*31] = [4, 1280, 961].
        ca_attn_out  = pool_feat_or_attn_mat(ca_attn_out, (H, W))
        # retain_spatial=True to get the resized spatial dims H2, W2.
        ca_outfeat   = pool_feat_or_attn_mat(ca_outfeat,  (H, W), retain_spatial=True)
        ca_q         = pool_feat_or_attn_mat(ca_q,        (H, W))
        # ss_fg_mask_3d: [1, 1, 64*64] -> [1, 1, 31*31] = [1, 1, 961]
        ss_fg_mask_3d   = pool_feat_or_attn_mat(ss_fg_mask_3d,  (H, W))
        H2, W2       = ca_outfeat.shape[-2:]
        # Flatten the spatial dims of ca_outfeat for reconstruction.
        ca_outfeat   = ca_outfeat.reshape(*ca_outfeat.shape[:2], H2*W2)
    else:
        H2, W2       = H, W

    # ss_fg_mask_2d: [1, 961]
    ss_fg_mask_2d = ss_fg_mask_3d.bool().squeeze(1)
    ss_fg_mask_B, ss_fg_mask_N = ss_fg_mask_2d.nonzero(as_tuple=True)
    # ss_*: subj single, sc_*: subj comp, ms_*: class single, mc_*: class comp.
    # ss_q, sc_q, ms_q, mc_q: [4, 1280, 961] => [1, 1280, 961].
    ss_q, sc_q, ms_q, mc_q = ca_q.chunk(4)
    sc_q_grad_scaler = gen_gradient_scaler(sc_q_grad_scale)
    # Slowly update attn loras that influence sc_q.
    sc_q = sc_q_grad_scaler(sc_q)

    # fg_ss_q: [1, 961, 961] => [1, 961, N_fg]
    # filter with ss_fg_mask_N, so that we only care about 
    # the recon of the fg areas of the subj single instance.
    # NOTE: we allow BP to ca_qs, which mainly leads to update of attn loras.
    # What's interesting is after the initial several thousand iterations,
    # the composition becomes very bad (the face appears abruptly, incoherent with the surroundings),
    # probably due to the update of attn loras.
    # However, as training goes on, the composition seems to repair itself.
    # Maybe good compositions are a stable equilibrium for model params?
    fg_ss_q = ss_q[:, :, ss_fg_mask_N]
    # ss_mc_q: [1, 1280, N_fg + 961] = [1, 1280, N_fg + 961].
    ss_mc_q = torch.cat([fg_ss_q, mc_q], dim=2)

    # Similar to the scale of the attention scores.
    # Disable matching_score_scale, since when we were caching q, we've scaled it by 1/sqrt(sqrt(dim)).
    # Two q's multiplied together will be scaled by 1/sqrt(dim). Moreover, currently sc_to_ss_prob is quite uniform,
    # and disabling matching_score_scale can make sc_to_ss_prob more polarized.
    # num_heads = 8
    matching_score_scale = 1 #(ca_outfeat.shape[1] / num_heads) ** -0.5
    # Pairwise matching scores (961 subj comp image tokens) -> (961 subj single tokens + 961 cls comp tokens).
    # matmul() does multiplication on the last two dims.
    # sc_to_ss_prob: [1, 961, 1280] * [1, 1280, 1922] => [1, 961, 1922], (batch, sc, ss_mc).
    sc_to_ss_mc_score = torch.matmul(sc_q.transpose(1, 2).contiguous(), ss_mc_q) * matching_score_scale
    # NOTE: If sc_to_ss_prob is only normalized among the single tokens dim,
    # then each comp image token has a total contribution of 1 to all single tokens.
    # But some comp tokens are backgrounds which don't appear in the single instance, 
    # therefore this constraint is not reasonable.
    if c_to_s_attn_norm_dims == (1, 2):
        # Dims 0, 1, 2: (batch, sc, ss) -> (batch, sc * ss)
        # sc_to_ss_prob and mc_to_ms_prob are normalized along the joint (single, comp) 
        # tokens dims instead of the *single tokens* or the *comp tokens* dim.
        sc_to_ss_mc_score2 = sc_to_ss_mc_score.reshape(sc_to_ss_mc_score.shape[0], -1)
        # Normalize among joint (single, comp) tokens dims.
        sc_to_ss_mc_prob   = F.softmax(sc_to_ss_mc_score2, dim=1).reshape(sc_to_ss_mc_score.shape)
    elif c_to_s_attn_norm_dims == (1,):
        # Normalize among class comp tokens (sc dim).
        # Major Concern:
        # If we only normalize among the comp tokens dim, then some comp tokens have higer overall attentions
        # to the single instance (sum of attention across all single tokens) than others. This is reasonable.
        # However, some comp tokens, esp. high attention comp tokens, have 
        # almost uniform attention across single tokens, i.e., they are equally similar to all single tokens.
        # Such comp tokens may capture some common, low-freq features across single tokens, 
        # Eventually, a small fraction of the comp token is used to reconstruct all single tokens.
        # but such features are not interesting for facial reconstruction, 
        # as we want to make the facial comp tokens pay more attention to high-freq facial features.
        sc_to_ss_mc_prob  = F.softmax(sc_to_ss_mc_score, dim=1)
    else:
        breakpoint()

    # sc_to_ss_prob, sc_to_mc_prob: [1, 961, N_fg + 961] -> [1, 961, N_fg] and [1, 961, 961].
    # sc_to_ss_mc_prob.sum(dim=1) == [1, 1, ..., 1] (N_fg + 961 ones).
    # So after slicing, sc_to_ss_fg_tokens_prob, sc_to_mc_prob are also normalized along the subj comp tokens dim.
    sc_to_ss_fg_tokens_prob, sc_to_mc_prob = sc_to_ss_mc_prob[:, :, :fg_ss_q.shape[2]], sc_to_ss_mc_prob[:, :, fg_ss_q.shape[2]:]
    # View the whole fg area in the single instance as a single token.
    # Sum up the mapping probs from comp instances to the fg area of the single instances, so that 
    # ** each entry is the total prob of each image token in the comp instance
    # ** maps to the whole fg area in the single instance.
    # sc_to_ss_fg_prob: [1, 961, 961] * [1, 961, 1] => [1, 961, 1] => [1, 1, 961].
    # sums up over all ss fg tokens. So each entry in sc_to_ss_fg_prob 
    # is the total prob of each sc token maps to the whole fg area 
    # in the subj single instance, which could be >> 1.
    # sc_to_ss_fg_prob.max() == 2.3826, min() == 0.0007, sum() == 210 == N_fg.
    sc_to_ss_fg_prob = sc_to_ss_fg_tokens_prob.sum(dim=2, keepdim=True)
    # sc_to_whole_mc_prob: [1, 961, 961] => [1, 961, 1] => [1, 1, 961].
    # sc_to_whole_mc_prob.max() == 1.5185, min() == 0.5266, sum() == 961.
    sc_to_whole_mc_prob = sc_to_mc_prob.sum(dim=2, keepdim=True)

    ss2sc_flow = None
    mc2sc_flow = None
    losses_sc_recons = {}
    losses_flow_attns_distill = {}
    loss_flow_attns_distill = {}

    for feat_type in recon_feat_objectives:
        if feat_type == 'attn_out':
            # ca_attn_out: output of the CA layer, i.e., attention-aggregated v.
            feat_obj = ca_attn_out
            loss_scale = 1
        elif feat_type == 'outfeat':
            # outfeat: output of the transformer layer, i.e., attn_out transformed by a FFN.
            feat_obj = ca_outfeat
            # outfeat on L2 loss scheme incurs large loss values, so we scale it down.
            loss_scale = 0.5
        else:
            breakpoint()

        ss_feat, sc_feat, ms_feat, mc_feat = feat_obj.chunk(4)
        # Cut off the gradients into the subj single instance.
        # We don't need to cut off ms_feat, mc_feat, because they were generated with no_grad().
        ss_feat = ss_feat.detach()
        # ss_fg_mask_N:  [N_fg] indices on the flattened spatial dim.
        # Apply mask, permute features to the last dim. [1, 1280, 961] => [1, 961, 1280] => [1, N_fg, 1280]
        # TODO: this is buggy if block size > 1.
        ssfg_feat = ss_feat.permute(0, 2, 1)[:, ss_fg_mask_N]
        mc_feat   = mc_feat.permute(0, 2, 1)

        #### Compute fg reconstruction losses. ####
        objective_name = feat_type
        # Make the objective name have a fixed length by padding spaces.
        if len(objective_name) < 8:
            objective_name += ' ' * (8 - len(objective_name))

        # feat_type iterates ['attn_out', 'outfeat']. On attn_out, the input ss2sc_flow is None.
        # The estimated ss2sc_flow is returned and used in the next iteration.
        # On outfeat, the input ss2sc_flow is the ss2sc_flow estimated on attn_out.
        target_feats = { 'ssfg': ssfg_feat, 'mc': mc_feat }
        losses_sc_recons_obj, loss_flow_attns_distill_obj, ss2sc_flow, mc2sc_flow = \
            calc_sc_recon_ssfg_mc_losses(layer_idx, flow_model, 
                                         target_feats, sc_feat, 
                                         ss2sc_flow, mc2sc_flow, 
                                         sc_to_ss_mc_prob, ss_fg_mask_2d, 
                                         ss_q, sc_q, mc_q, H2, W2, num_flow_est_iters, 
                                         objective_name=objective_name)
        
        for feat_name, losses in losses_sc_recons_obj.items():
            if feat_name not in losses_sc_recons:
                losses_sc_recons[feat_name] = []

            losses = torch.tensor(losses) * loss_scale
            
            # If the recon loss is too large, it means there's probably spatial misalignment between the two features.
            # Optimizing w.r.t. this loss may lead to degenerate results.
            to_discard = losses[-1] > recon_loss_discard_thres
            if to_discard:
                print(f"Discard layer {layer_idx} {objective_name} {feat_name} loss: {losses[-1]}.")
                continue
            else:
                losses_sc_recons[feat_name].append(losses)

            if feat_name not in losses_flow_attns_distill:
                losses_flow_attns_distill[feat_name] = []

            losses_flow_attns_distill[feat_name].append(loss_flow_attns_distill_obj[feat_name])

    for feat_name, losses in losses_sc_recons.items():
        if len(losses) > 0:
            losses_sc_recons[feat_name] = torch.stack(losses, dim=0).mean(dim=0)
        else:
            # If all losses are discarded, return 3 x 0s.
            losses_sc_recons[feat_name] = torch.zeros(3, device=ss_feat.device)

        loss_flow_attns_distill[feat_name] = torch.stack(losses_flow_attns_distill[feat_name], dim=0).mean()

    return losses_sc_recons, loss_flow_attns_distill, sc_to_ss_fg_prob, sc_to_whole_mc_prob

