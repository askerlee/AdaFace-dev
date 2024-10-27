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
    # Release some RAM. Not sure if it really works.
    del sd, pl_sd

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
# NOTE: ortho_subtract(a, b) is scale-invariant w.r.t. b,
# but scales proportionally to the scale of a.
# a, b are n-dimensional tensors. Subtraction happens at the last on_last_n_dims dims.
# ortho_subtract(a, b) is not symmetric w.r.t. a and b, nor is ortho_l2loss(a, b).
# NOTE: always choose a to be something we care about, and b to be something as a reference.
def ortho_subtract(a, b, on_last_n_dims=1, return_align_coeffs=False):
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
    result = a2 - b2 * w_optimal.unsqueeze(-1)

    if on_last_n_dims > 1:
        result = result.reshape(orig_shape)
        w_orig_shape = list(orig_shape)
        w_orig_shape[-on_last_n_dims:] = [1] * on_last_n_dims
        w_optimal = w_optimal.reshape(w_orig_shape)

    if return_align_coeffs:
        return result, w_optimal
    else:
        return result

# Extract the components of a that aligns and orthos with b, respectively.
def decomp_align_ortho(a, b, return_align_coeffs=False):
    if return_align_coeffs:
        ortho, align_coeffs = ortho_subtract(a, b, return_align_coeffs=True)
        align = a - ortho
        return align, ortho, align_coeffs
    else:
        ortho = ortho_subtract(a, b)
        align = a - ortho
        return align, ortho

# Decompose a as ortho (w.r.t. b) and align (w.r.t. b) components.
# Scale down the align component by align_suppress_scale.
def directional_suppress(a, b, align_suppress_scale=1):
    if align_suppress_scale == 1 or b.abs().sum() < 1e-6:
        return a
    else:
        ortho = ortho_subtract(a, b)
        align = a - ortho
        return align * align_suppress_scale + ortho

def align_suppressed_add(a, b, align_suppress_scale=1):
    return a + directional_suppress(b, a, align_suppress_scale)

def calc_align_coeffs(a, b, on_last_n_dims=1):
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

        orig_shape = list(a.shape)
        a2 = a.reshape(*a.shape[:-on_last_n_dims], -1)
        b2 = b.reshape(*b.shape[:-on_last_n_dims], -1)
    else:
        a2 = a
        b2 = b

    dot_a_b = torch.einsum('...i,...i->...', a2, b2)
    dot_b_b = torch.einsum('...i,...i->...', b2, b2)
    w_optimal = dot_a_b / (dot_b_b + 1e-6)
    # Append N=on_last_n_dims empty dimensions to w_optimal.
    if on_last_n_dims > 1:
        orig_shape[-on_last_n_dims:] = [1] * on_last_n_dims
        w_optimal = w_optimal.reshape(orig_shape)

    return w_optimal

# ortho_subtract(a, b): the residual is orthogonal to b (on the last dimension).
# ortho_subtract(a, b) is not symmetric w.r.t. a and b, nor is ortho_l2loss(a, b).
# NOTE: always choose a to be something we care about, and b to be something as a reference.
def ortho_l2loss(a, b, mean=True, do_sqrt=False):
    residual = ortho_subtract(a, b)
    # F.mse_loss() is taking the square of all elements in the residual, then mean.
    # ortho_l2loss() keeps consistent with F.mse_loss().
    loss = residual * residual
    if mean:
        loss = loss.mean()
    if do_sqrt:
        loss = loss.sqrt()
    return loss

def normalized_l2loss(a, b, mean=True):
    a_norm = a.norm(dim=-1, keepdim=True) + 1e-6
    b_norm = b.norm(dim=-1, keepdim=True) + 1e-6
    a = a * (a_norm + b_norm) / (a_norm * 2)
    b = b * (a_norm + b_norm) / (b_norm * 2)
    diff = a - b
    # F.mse_loss() is taking the square of all elements in the residual, then mean.
    # normalized_l2loss() keeps consistent with F.mse_loss().
    loss = diff * diff
    if mean:
        loss = loss.mean()
    return loss

def power_loss(a, exponent=2, rev_pow=False):
    loss = a.abs().pow(exponent).abs().mean()
    if rev_pow:
        # pow(1/exponent): limit the scale of the loss. 
        # Recommended to be True if exponent > 2.
        loss = loss.pow(1/exponent)
    return loss

def clamp_prompt_embedding(clamp_value, *embs):
    if clamp_value <= 0:
        return embs[0] if len(embs) == 1 else embs
    
    clamp = lambda e: torch.clamp(e, min=-clamp_value, max=clamp_value) if e is not None else None
    return clamp(embs[0]) if len(embs) == 1 else [clamp(e) for e in embs]
    
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


def calc_init_word_embeddings(get_tokens_for_string, get_embeddings_for_tokens,
                              initializer_string, initializer_word_weights):
    if initializer_string is None:  
        # The background embedding is not initialized with any word embedding.
        # In this case,
        # init_word_embeddings = None,    init_word_weights    = None,
        # init_word_embeddings = None, avg_init_word_embedding = None.
        return None, None, None, None
    else:
        init_word_tokens = get_tokens_for_string(initializer_string)
        N = len(init_word_tokens)
        if initializer_word_weights is not None:
            init_word_weights = torch.tensor(initializer_word_weights, dtype=torch.float32)
            # Increase the weight of the main class word. 
            init_word_weights = init_word_weights ** 2
            init_word_weights = init_word_weights / init_word_weights.sum()
        else:
            # Equal weights for all words.
            init_word_weights = torch.ones(N, dtype=torch.float32) / N
        
        # init_word_embeddings: [2, 768]. avg_init_word_embedding: [1, 768].
        init_word_embeddings = get_embeddings_for_tokens(init_word_tokens.cpu())
        avg_init_word_embedding = (init_word_embeddings * init_word_weights.unsqueeze(1)).sum(dim=0, keepdim=True)

        return init_word_tokens, init_word_weights, init_word_embeddings, avg_init_word_embedding

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
def save_grid(samples, img_flags, grid_filepath, nrow):
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
        grid_img.save(grid_filepath)
    
    # return image to be shown on webui
    return grid_img

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

def split_indices_by_block(indices, block_size):
    indices_B, indices_N = indices
    max_block_idx = indices_B.max() // block_size
    for block_idx in range(max_block_idx + 1):
        block_indices_B = indices_B[indices_B // block_size == block_idx]
        block_indices_N = indices_N[indices_B // block_size == block_idx]
        yield (block_indices_B, block_indices_N)

# n: repeated times. If n = 0, no repeat. If n = 1, repeat once (double the length).
# If n = 1, indices = ([0, 0], [1, 2]) => ([0, 0, 0, 0], [1, 2, 3, 4]).
def extend_indices_N_by_n_times(indices, n):
    if indices is None:
        return None
    
    if n == 0:
        return indices
    
    device = indices[0].device
    indices_by_instance = split_indices_by_instance(indices)
    indices_by_instance_B_ext, indices_by_instance_N_ext = [], []
    for inst_indices_B, inst_indices_N in indices_by_instance:
        indices_by_instance_B_ext += [ inst_indices_B, torch.ones(n, dtype=int, device=device) * inst_indices_B[0] ]
        indices_by_instance_N_ext += [ inst_indices_N, torch.arange(1, n + 1, dtype=int, device=device) + inst_indices_N[-1] ]
    
    indices_B_ext = torch.cat(indices_by_instance_B_ext, dim=0)
    indices_N_ext = torch.cat(indices_by_instance_N_ext, dim=0)

    return (indices_B_ext, indices_N_ext)

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

def repeat_selected_instances(sel_indices, REPEAT, *args):
    rep_args = []
    for arg in args:
        if arg is not None:
            arg2 = arg[sel_indices].repeat([REPEAT] + [1] * (arg.ndim - 1))
        else:
            arg2 = None
        rep_args.append(arg2)

    return rep_args

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

def torch_uniform(low, high, size, device=None):
    return torch.rand(size, device=device) * (high - low) + low

# true_prob_range = (p_init, p_final). 
# The prob of flipping true is gradually annealed from p_init to p_final.
def draw_annealed_bool(training_percent, final_percent, true_prob_range):
    true_p_annealed = anneal_value(training_percent, final_percent, value_range=true_prob_range)
    # Flip a coin, with prob of true being true_p_annealed.    
    return (torch.rand(1) < true_p_annealed).item()

def probably_draw_float(rand_range, default_value, default_prob=0.5):
    lb, ub = rand_range
    assert lb < ub
    if torch.rand(1) < default_prob:
        return default_value
    else:
        return torch.rand(1).item() * (ub - lb) + lb

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

# init_ratio_range, final_ratio_range: ranges of fluctuation ratios (could > 1 or < 1).
# Gradually shift ratio_range from init_ratio_range to final_ratio_range.
def anneal_t_ratio(t, training_percent, num_timesteps, init_ratio_range, final_ratio_range):
    t_annealed = t.clone()    
    ratio_lb = anneal_value(training_percent, final_percent=1., value_range=(init_ratio_range[0], final_ratio_range[0]))
    ratio_ub = anneal_value(training_percent, final_percent=1., value_range=(init_ratio_range[1], final_ratio_range[1]))
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
        t_annealed = torch.tensor(torch.randint(t_lowerbound, t_upperbound, (1,)).item(), 
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

def select_piecewise_value(ranged_values, curr_pos, range_ub=1.0):
    for i, (range_lb, value) in enumerate(ranged_values):
        if i < len(ranged_values) - 1:
            range_ub = ranged_values[i + 1][0]
        else:
            range_ub = 1.0

        if range_lb <= curr_pos < range_ub:
            return value

    raise ValueError(f"curr_pos {curr_pos} is out of range.")

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

def repeat_selected_instances(sel_indices, REPEAT, *args):
    rep_args = []
    for arg in args:
        if arg is not None:
            if isinstance(arg, torch.Tensor):
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

    # sum among K_bg_i bg embeddings -> [1, 8, 64]
    if do_sum:
        emb_attns   = [ emb_attns[i].sum(dim=1) for i in range(len(indices_by_instance)) ]
    elif do_mean:
        emb_attns   = [ emb_attns[i].mean(dim=1) for i in range(len(indices_by_instance)) ]

    emb_attns = torch.cat(emb_attns, dim=0)
    return emb_attns
                
def gen_comp_extra_indices_by_block(prompt_emb_mask, list_indices_to_mask, block_size):
    # prompt_emb_mask: [4, 77, 1] => [4, 77]
    comp_extra_mask = prompt_emb_mask.squeeze(-1).clone()
    # Mask out the foreground and background embeddings.
    for indices_to_mask in list_indices_to_mask:
        if indices_to_mask is not None:
            comp_extra_mask[indices_to_mask] = 0

    comp_extra_indices = comp_extra_mask.nonzero(as_tuple=True)
    # split_indices_by_block() returns a generator. Convert to a list.
    comp_extra_indices_by_block = split_indices_by_block(comp_extra_indices, block_size)
    comp_extra_indices_by_block = list(comp_extra_indices_by_block)
    return comp_extra_indices_by_block

def replace_prompt_comp_extra(comp_prompts, single_prompts, new_comp_extras):
    new_comp_prompts = []
    for i in range(len(comp_prompts)):
        assert comp_prompts[i].startswith(single_prompts[i])
        if new_comp_extras[i] != '':
            # Replace the compositional prompt with the new compositional part.
            new_comp_prompts.append( single_prompts[i] + new_comp_extras[i] )
        else:
            # new_comp_extra is empty, i.e., the particular instance has no corresponding fg_mask.
            # Keep the original compositional prompt.
            new_comp_prompts.append( comp_prompts[i] )
    
    return new_comp_prompts

def extract_last_chunk_of_indices(token_indices, total_num_chunks=3):
    if token_indices is None:
        return None
    
    indiv_indices = split_indices_by_instance(token_indices)
    indiv_indices_half_B2, indiv_indices_half_N2 = [], []
    for indices_B, indices_N in indiv_indices:
        indices_B2_chunks = indices_B.chunk(total_num_chunks)
        indices_N2_chunks = indices_N.chunk(total_num_chunks)
        # Not enough chunks. Skip this instance.
        if len(indices_B2_chunks) < total_num_chunks:
            continue
        indices_B2 = indices_B2_chunks[-1]
        indices_N2 = indices_N2_chunks[-1]
        indiv_indices_half_B2.append(indices_B2)
        indiv_indices_half_N2.append(indices_N2)

    # If token_indices contain 9 indices for each instance, 
    # then the second chunk of 4 indices will be returned.
    token_indices_half_B2 = torch.cat(indiv_indices_half_B2, dim=0)
    token_indices_half_N2 = torch.cat(indiv_indices_half_N2, dim=0)
    return (token_indices_half_B2, token_indices_half_N2)

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

def calc_dyn_loss_scale(loss, base_loss_and_scale, ref_loss_and_scale, rel_scale_range=(-0.5, 10)):
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
                                       training_percent, base_scale_range=(0.8, 1.0),
                                       fg_noise_anneal_mean_range=(0.1, 0.5)):
    x_start_origsize = torch.where(filtered_fg_mask.bool(), x_start, torch.randn_like(x_start))
    fg_mask_percent = filtered_fg_mask.float().sum() / filtered_fg_mask.numel()
    # print(fg_mask_percent)
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
        # fg areas are small. Scale the fg area to 70%-100% of the original size.
        fg_rand_scale = torch.rand(1).item() * (base_scale_range_ub - base_scale_range_lb) + base_scale_range_lb

    # Resize x_start_origsize and filtered_fg_mask by rand_scale. They have different numbers of channels,
    # so we need to concatenate them at dim 1 before resizing.
    x_mask = torch.cat([x_start_origsize, fg_mask, filtered_fg_mask], dim=1)
    x_mask_scaled = F.interpolate(x_mask, scale_factor=fg_rand_scale, mode='bilinear', align_corners=False)

    # Pad filtered_fg_mask_scaled to the original size, with left/right padding roughly equal
    pad_w1 = int((x_start.shape[3] - x_mask_scaled.shape[3]) / 2)
    pad_w2 =      x_start.shape[3] - x_mask_scaled.shape[3] - pad_w1
    pad_h1 = int((x_start.shape[2] - x_mask_scaled.shape[2]) / 2)
    pad_h2 =      x_start.shape[2] - x_mask_scaled.shape[2] - pad_h1
    x_mask_scaled_padded = F.pad(x_mask_scaled, 
                                    (pad_w1, pad_w2, pad_h1, pad_h2),
                                    mode='constant', value=0)

    # Unpack x_start, fg_mask and filtered_fg_mask from x_mask_scaled_padded.
    # x_start_scaled_padded: [2, 4, 64, 64]. fg_mask/filtered_fg_mask: [2, 1, 64, 64].
    x_start_scaled_padded, fg_mask, filtered_fg_mask \
        = x_mask_scaled_padded[:, :4], x_mask_scaled_padded[:, [4]], \
            x_mask_scaled_padded[:, [5]]

    # In filtered_fg_mask, the padded areas are filled with 0. 
    # So these pixels always take values from the random tensor.
    # In fg area, x_start takes values from x_start_scaled_padded 
    # (the fg of x_start_scaled_padded is a scaled-down version of the fg of the original x_start).
    x_start = torch.where(filtered_fg_mask.bool(), x_start_scaled_padded, torch.randn_like(x_start))
    # Gradually increase the fg area's noise amount with mean increasing from 0.1 to 0.5.
    fg_noise_amount = rand_annealed(training_percent, final_percent=1, mean_range=fg_noise_anneal_mean_range)
    # At the fg area, keep 90% (beginning of training) ~ 50% (end of training) 
    # of the original x_start values and add 10% ~ 50% of noise. 
    # x_start: [2, 4, 64, 64]
    x_start = torch.randn_like(x_start) * fg_noise_amount + x_start * (1 - fg_noise_amount)
    return x_start, fg_mask, filtered_fg_mask

# prob_mat: [1, 64, 64].
def add_to_prob_mat_diagonal(prob_mat, p, renormalize_dim=None):
    # diagonal_delta: [64, 64].
    diagonal_delta = torch.diag(torch.ones(prob_mat.shape[-1], device=prob_mat.device)) * p
    if prob_mat.ndim > 2:
        # diagonal_delta: [1, 64, 64].
        ones_shape = [1] * (prob_mat.ndim - 2)
        diagonal_delta = diagonal_delta.reshape(ones_shape + list(diagonal_delta.shape))

    prob_mat = prob_mat + diagonal_delta
    if renormalize_dim is not None:
        # Re-normalize among the specified dimension after adding p to the diagonal.
        prob_mat = prob_mat / prob_mat.sum(dim=renormalize_dim, keepdim=True)
    return prob_mat

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

def backward_warp_by_flow_np(image2, flow1to2):
    H, W, _ = image2.shape
    flow1to2 = flow1to2.copy()
    flow1to2[:, :, 0] += np.arange(W)  # Adjust x-coordinates
    flow1to2[:, :, 1] += np.arange(H)[:, None]  # Adjust y-coordinates
    image1_recovered = cv2.remap(image2, flow1to2, None, cv2.INTER_LINEAR)
    return image1_recovered

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
    image1_recovered = F.grid_sample(image2, flow, mode='bilinear', padding_mode='zeros', align_corners=True)

    return image1_recovered

@torch.compile
def reconstruct_feat_with_attn_aggregation(sc_feat, sc_map_ss_prob, ss_fg_mask):
    # recon_sc_feat: [1, 1280, 64] * [1, 64, 64] => [1, 1280, 64]
    # ** We only use the subj comp tokens to reconstruct the subj single tokens, not vice versa. **
    # Because we rely on ss_fg_mask to determine the fg area, which is only available for the subj single instance. 
    # Then we can compare the values of the recon'ed subj single tokens with the original values at the fg area.
    ss_fg_mask_B, ss_fg_mask_N = ss_fg_mask.nonzero(as_tuple=True)
    # sc_map_ss_fg_prob: [1, 64, 64] => [1, 64, N_fg]
    # filter with ss_fg_mask_N, so that we only care about 
    # the recon of the fg areas of the subj single instance.
    sc_map_ss_fg_prob = sc_map_ss_prob[:, :, ss_fg_mask_N]

    # Weighted sum of the comp tokens (based on their matching probs) to reconstruct the single tokens.
    # sc_recon_ss_fg_feat: [1, 1280, 64] * [1, 64, N_fg] => [1, 1280, N_fg]
    sc_recon_ss_fg_feat = torch.matmul(sc_feat, sc_map_ss_fg_prob)
    # sc_recon_ss_fg_feat: [1, 1280, N_fg] => [1, N_fg, 1280]
    sc_recon_ss_fg_feat = sc_recon_ss_fg_feat.permute(0, 2, 1)
    # mc_recon_ms_feat = torch.matmul(mc_feat, mc_map_ms_prob)

    return sc_recon_ss_fg_feat
        
#@torch.compile
def reconstruct_feat_with_matching_flow(flow_model, s2c_flow, ss_q, sc_q, sc_feat, ss_fg_mask, 
                                        H, W, num_flow_est_iters=12):
    if H*W != sc_feat.shape[-1]:
        breakpoint()

    # Remove background features to reduce noisy matching.
    # ss_q: [1, 1280, 64]. ss_fg_mask: [1, 64] -> [1, 1, 64]
    ss_q = ss_q * ss_fg_mask.unsqueeze(1)

    # If s2c_flow is not provided, estimate it using the flow model.
    # Otherwise use the provided flow.
    if s2c_flow is None:
        # Latent optical flow from subj single feature maps to subj comp feature maps.
        # Enabling grad seems to lead to quite bad results. 
        # Maybe updating q through flow is not a good idea.
        with torch.no_grad():
            s2c_flow = flow_model.est_flow_from_feats(ss_q, sc_q, H, W, num_iters=num_flow_est_iters, 
                                                      corr_normalized_by_sqrt_dim=False)
        # s2c_flow = resize_flow(s2c_flow, H, W)

    sc_feat             = sc_feat.reshape(*sc_feat.shape[:2], H, W)
    sc_recon_ss_feat    = backward_warp_by_flow(sc_feat, s2c_flow)
    # Collapse the spatial dimensions again.
    sc_recon_ss_feat    = sc_recon_ss_feat.reshape(*sc_recon_ss_feat.shape[:2], -1)

    # ss_fg_mask's spatial dim is already collapsed. ss_fg_mask: [1, 225]
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
    ss_fg_mask_B, ss_fg_mask_N    = ss_fg_mask.nonzero(as_tuple=True)    
    # sc_recon_ss_feat: [1, 1280, 64] -> [1, 64, 1280] -> [1, N_fg, 1280]
    sc_recon_ss_fg_feat     = sc_recon_ss_feat.permute(0, 2, 1)[:, ss_fg_mask_N]

    return sc_recon_ss_fg_feat, s2c_flow

# We can not simply switch ss_feat/ss_q with sc_feat/sc_q, and also change sc_map_ss_prob to ss_map_sc_prob, 
# to get ss-recon-sc losses.
def calc_sc_recon_ss_fg_losses(layer_idx, flow_model, s2c_flow, ss_feat, sc_feat, sc_map_ss_prob, ss_fg_mask, 
                               ss_q, sc_q, H, W, num_flow_est_iters, do_recon_feat_delta=False):

    ss_fg_mask_B, ss_fg_mask_N = ss_fg_mask.nonzero(as_tuple=True)

    # sc_recon_ss_fg_feat*: [1, 1280, N_fg] => [1, N_fg, 1280]
    sc_recon_ss_fg_feat_attn_agg = \
        reconstruct_feat_with_attn_aggregation(sc_feat, sc_map_ss_prob, ss_fg_mask)
    if flow_model is not None or s2c_flow is not None:
        # If s2c_flow is not provided, estimate it using the flow model.
        # Otherwise use the provided flow, and return the estimated s2c_flow.     
        sc_recon_ss_fg_feat_flow, s2c_flow = \
            reconstruct_feat_with_matching_flow(flow_model, s2c_flow, ss_q, sc_q, sc_feat, 
                                                ss_fg_mask, H, W, num_flow_est_iters=num_flow_est_iters)
    else:
        s2c_flow = None
        sc_recon_ss_fg_feat_flow = None
        
    losses_sc_recon_ss_fg       = []
    all_token_losses_sc_recon_ss_fg = []

    # ss_fg_mask: bool of [1, 64] with N_fg True values.
    # Apply mask, permute features to the last dim. [1, 1280, 64] => [1, 64, 1280] => [1, N_fg, 1280]
    ss_fg_feat =  ss_feat.permute(0, 2, 1)[:, ss_fg_mask_N]
    # ms_fg_feat, mc_recon_ms_fg_feat = ... ms_feat, mc_recon_ms_feat

    loss_type_names = ['attn', 'flow']
    if not do_recon_feat_delta:
        objective_name = 'sc-recon-ss-fg' 
        cosine_exponent = 2
    else:
        # If do_recon_feat_delta, then ss_feat, sc_feat are actually ss_feat - ms_feat, sc_feat - mc_feat.
        objective_name = 'sc-ss-fg-delta'
        # If recon_feat_delta, the feat delta already second order, so we don't need to square the cosine.
        cosine_exponent = 2

    print(f"Layer {layer_idx}: {H}x{W} {objective_name} losses:", end=' ')

    for i, sc_recon_ss_fg_feat in enumerate((sc_recon_ss_fg_feat_attn_agg, sc_recon_ss_fg_feat_flow)):
        if sc_recon_ss_fg_feat is None:
            losses_sc_recon_ss_fg.append(0)
            continue

        # We use cosine loss, so that when the reconstructed features are of different scales,
        # the loss could still be small.
        # ref_grad_scale=0: don't BP to ss_fg_feat.
        # If reduction == 'none', return a 2D loss tensor of [Batch, Instances].
        # If reduction == 'mean', return a scalar loss.        
        token_losses_sc_recon_ss_fg = \
            calc_ref_cosine_loss(sc_recon_ss_fg_feat, ss_fg_feat, 
                                 exponent=cosine_exponent, do_demeans=[False, False],
                                 first_n_dims_into_instances=2, 
                                 ref_grad_scale=0, aim_to_align=True,
                                 reduction='none')

        # Here loss_sc_recon_ss_fg (corresponding to loss_sc_recon_ss_fg_attn_agg, loss_sc_recon_ss_fg_flow) 
        # is only for debugging. 
        # The optimized loss_sc_recon_ss_fg is loss_sc_recon_ss_fg_min, computed by 
        # taking the tokenwise min of the two losses.
        loss_sc_recon_ss_fg = token_losses_sc_recon_ss_fg.mean()
        losses_sc_recon_ss_fg.append(loss_sc_recon_ss_fg)
        all_token_losses_sc_recon_ss_fg.append(token_losses_sc_recon_ss_fg)

        print(f"{loss_type_names[i]}: {loss_sc_recon_ss_fg.item():.03f}", end=' ')

    # We have both attn and flow token losses.
    if len(all_token_losses_sc_recon_ss_fg) > 1:
        # all_token_losses_sc_recon_ss_fg: [2, 1, 1037]. 1037: number of fg tokens.
        all_token_losses_sc_recon_ss_fg = torch.stack(all_token_losses_sc_recon_ss_fg, dim=0)
        # Take the smaller loss tokenwise between attn and flow.
        token_losses_sc_recon_ss_fg = all_token_losses_sc_recon_ss_fg.min(dim=0).values
        loss_sc_recon_ss_fg_min = token_losses_sc_recon_ss_fg.mean()
        losses_sc_recon_ss_fg.append(loss_sc_recon_ss_fg_min)

        print(f"min : {loss_sc_recon_ss_fg_min.item():.03f}", end=' ')
    print()

    return losses_sc_recon_ss_fg, s2c_flow

def calc_ss_fg_recon_sc_losses(layer_idx, flow_model, c2s_flow, ss_feat, sc_feat, ss_map_sc_prob, ss_fg_mask, 
                               ss_q, sc_q, H, W, num_flow_est_iters):

    ss_fg_mask_B, ss_fg_mask_N = ss_fg_mask.nonzero(as_tuple=True)
    
    # ss_fg_recon_sc_feat*: [1, 1280, N_fg] => [1, N_fg, 1280]
    ss_fg_recon_sc_feat_attn_agg = \
        reconstruct_feat_with_attn_aggregation(sc_feat, ss_map_sc_prob, ss_fg_mask)
    if flow_model is not None or c2s_flow is not None:
        # If c2s_flow is not provided, estimate it using the flow model.
        # Otherwise use the provided flow, and return the estimated c2s_flow.     
        sc_recon_ss_fg_feat_flow, c2s_flow = \
            reconstruct_feat_with_matching_flow(flow_model, c2s_flow, ss_q, sc_q, sc_feat, 
                                                ss_fg_mask, H, W, num_flow_est_iters=num_flow_est_iters)
    else:
        ss_fg_recon_sc_feat_flow = None
        c2s_flow = None
        
    losses_ss_fg_recon_sc       = []
    all_token_losses_ss_fg_recon_sc = []

    # ss_fg_mask: bool of [1, 64] with N_fg True values.
    # Apply mask, permute features to the last dim. [1, 1280, 64] => [1, 64, 1280] => [1, N_fg, 1280]
    ss_fg_feat =  ss_feat.permute(0, 2, 1)[:, ss_fg_mask_N]
    # ms_fg_feat, mc_recon_ms_fg_feat = ... ms_feat, mc_recon_ms_feat

    loss_type_names = ['attn', 'flow']
    print(f"Layer {layer_idx}: {H}x{W} ss-fg-recon-sc losses:", end=' ')

    for i, ss_fg_recon_sc_feat in enumerate((ss_fg_recon_sc_feat_attn_agg, ss_fg_recon_sc_feat_flow)):
        if ss_fg_recon_sc_feat is None:
            losses_ss_fg_recon_sc.append(0)
            continue

        # We use cosine loss, so that when the reconstructed features are of different scales,
        # the loss could still be small.
        # ref_grad_scale=0: don't BP to ss_fg_feat.
        # If reduction == 'none', return a 2D loss tensor of [Batch, Instances].
        # If reduction == 'mean', return a scalar loss.        
        token_losses_ss_fg_recon_sc = \
            calc_ref_cosine_loss(ss_fg_recon_sc_feat, ss_fg_feat, 
                                 exponent=2, do_demeans=[False, False],
                                 first_n_dims_into_instances=2, 
                                 ref_grad_scale=0, aim_to_align=True,
                                 reduction='none')

        # Here loss_ss_fg_recon_sc (corresponding to loss_ss_fg_recon_sc_attn_agg, loss_ss_fg_recon_sc_flow) 
        # is only for debugging. 
        # The optimized loss_ss_fg_recon_sc is loss_ss_fg_recon_sc_min, computed by 
        # taking the tokenwise min of the two losses.
        loss_ss_fg_recon_sc = token_losses_ss_fg_recon_sc.mean()
        losses_ss_fg_recon_sc.append(loss_ss_fg_recon_sc)
        all_token_losses_ss_fg_recon_sc.append(token_losses_ss_fg_recon_sc)

        print(f"{loss_type_names[i]}: {loss_ss_fg_recon_sc.item():.03f}", end=' ')

    # We have both attn and flow token losses.
    if len(all_token_losses_ss_fg_recon_sc) > 1:
        # all_token_losses_ss_fg_recon_sc: [2, 1, 1037]. 1037: number of fg tokens.
        all_token_losses_ss_fg_recon_sc = torch.stack(all_token_losses_ss_fg_recon_sc, dim=0)
        # Take the smaller loss tokenwise between attn and flow.
        token_losses_ss_fg_recon_sc = all_token_losses_ss_fg_recon_sc.min(dim=0).values
        loss_ss_fg_recon_sc_min = token_losses_ss_fg_recon_sc.mean()
        losses_ss_fg_recon_sc.append(loss_ss_fg_recon_sc_min)

        print(f"min : {loss_ss_fg_recon_sc_min.item():.03f}", end=' ')
    print()

    return losses_ss_fg_recon_sc, c2s_flow

# recon_feat_objectives: a list of strings, each is either 'feat' or 'delta'. 
# Default reconstruct both.
#@torch.compile
def calc_elastic_matching_loss(layer_idx, flow_model, ca_outfeat, ss_fg_mask, H, W, fg_bg_cutoff_prob=0.25,
                               num_flow_est_iters=12, recon_feat_objectives=['feat', 'delta'], do_feat_attn_pooling=True):
    # ss_fg_mask: [1, 1, 64] => [1, 64]
    if ss_fg_mask.sum() == 0:
        return 0, 0, 0, None, None

    # ca_outfeat: [4, 1280, 64]
   
    if do_feat_attn_pooling:
        # Pooling makes ca_outfeat spatially smoother, so that we'll get more continuous flow.
        ca_outfeat  = pool_feat_or_attn_mat(ca_outfeat, (H, W))
        ss_fg_mask  = pool_feat_or_attn_mat(ss_fg_mask, (H, W))
        H2, W2      = ca_outfeat.shape[-2:]
        ca_outfeat  = ca_outfeat.reshape(*ca_outfeat.shape[:2], H2*W2)
    else:
        H2, W2      = H, W

    ss_fg_mask = ss_fg_mask.bool().squeeze(1)
    # ss_*: subj single, sc_*: subj comp, ms_*: class single, mc_*: class comp.
    # ss_feat, sc_feat, ms_feat, mc_feat: [4, 1280, 64] => [1, 1280, 64].
    ss_feat, sc_feat, ms_feat, mc_feat = ca_outfeat.chunk(4)
    ss_feat = ss_feat.detach()

    num_heads = 8
    # Similar to the scale of the attention scores.
    matching_score_scale = (ca_outfeat.shape[1] / num_heads) ** -0.5
    # sc_map_ss_score:        [1, 64, 64]. 
    # Pairwise matching scores (64 subj comp image tokens) -> (64 subj single image tokens).
    # We use ca_outfeat instead of ca_q to compute the correlation scores, so we scale it.
    # Moreover, sometimes sc_map_ss_score before scaling is 200~300, which is too large.
    sc_map_ss_score = torch.matmul(sc_feat.transpose(1, 2).contiguous(), ss_feat) * matching_score_scale
    # sc_map_ss_prob:   [1, 64, 64]. 
    # Pairwise matching probs (9 subj comp image tokens) -> (9 subj single image tokens).
    # Dims 0, 1, 2 are the batch, sc, ss dims, respectively.
    # NOTE: sc_map_ss_prob and mc_map_ms_prob are normalized among the **comp tokens** dim.
    # This can address scale changes (e.g. the subject is large in single tokens,
    # but becomes smaller in comp tokens). If they are normalized among the single tokens dim,
    # then each comp image token has a fixed total contribution to the reconstruction
    # of the single tokens, which can hardly handle scale changes.
    # Now they are normalized among the comp tokens dim, so that all comp tokens have a
    # total contribution of 1 to the reconstruction of each single token.
    sc_map_ss_prob  = F.softmax(sc_map_ss_score, dim=1)

    # matmul() does multiplication on the last two dims.
    mc_map_ms_score = torch.matmul(mc_feat.transpose(1, 2).contiguous(), ms_feat) * matching_score_scale
    # Normalize among class comp tokens (mc dim).
    mc_map_ms_prob  = F.softmax(mc_map_ms_score, dim=1)

    # Span ss_fg_mask to become a mask for the (fg, fg) pairwise matching scores,
    # i.e., only be 1 (considered in the masked_mean()) if both tokens are fg tokens.
    ss_fg_mask_pairwise = ss_fg_mask.unsqueeze(1) * ss_fg_mask.unsqueeze(2)
    
    loss_subj_comp_map_single_align_with_cls = masked_mean((sc_map_ss_prob - mc_map_ms_prob).abs(), ss_fg_mask_pairwise)

    s2c_flow = None
    losses_sc_recon_ss_fg_objs = []
    for recon_feat_objective in recon_feat_objectives:
        if recon_feat_objective == 'feat':
            do_recon_feat_delta = False
            ss_feat_obj = ss_feat
            sc_feat_obj = sc_feat
        elif recon_feat_objective == 'delta':
            do_recon_feat_delta = True
            ss_feat_obj = ortho_subtract(ss_feat, ms_feat)
            sc_feat_obj = ortho_subtract(sc_feat, mc_feat)
        else:
            breakpoint()
        losses_sc_recon_ss_fg_obj, s2c_flow = \
            calc_sc_recon_ss_fg_losses(layer_idx, flow_model, s2c_flow, ss_feat_obj, sc_feat_obj, 
                                       sc_map_ss_prob, ss_fg_mask, 
                                       ss_feat, sc_feat, H2, W2, num_flow_est_iters, 
                                       do_recon_feat_delta=do_recon_feat_delta)
        losses_sc_recon_ss_fg_objs.append(torch.tensor(losses_sc_recon_ss_fg_obj))
    
    losses_sc_recon_ss_fg = torch.stack(losses_sc_recon_ss_fg_objs, dim=0).mean(dim=0)

    '''
    ss_map_sc_score = sc_map_ss_score.transpose(1, 2)
    ss_map_sc_prob = F.softmax(ss_map_sc_score, dim=1)
    losses_ss_fg_recon_sc = calc_ss_fg_recon_sc_losses(layer_idx, flow_model, ss_feat, sc_feat, 
                                                       ss_map_sc_prob, ss_fg_mask, 
                                                       ss_q, sc_q, H, W, num_flow_est_iters, name='ss-recon-sc')
    '''
    losses_ss_fg_recon_sc = [0, 0, 0]

    # ss_fg_mask: [1, 64] => [1, 64, 1].
    ss_fg_mask = ss_fg_mask.float().unsqueeze(2)
    # Sum up the mapping probs from comp instances to the fg area of the single instances, so that 
    # ** each entry is the total prob of each image token in the comp instance
    # ** maps to the whole fg area in the single instance.
    # sc_map_ss_fg_prob: [1, 64, 64] * [1, 64, 1] => [1, 64, 1] => [1, 1, 64].
    # matmul() sums up over the ss tokens dim, filtered by ss_fg_mask.
    sc_map_ss_fg_prob = torch.matmul(sc_map_ss_prob, ss_fg_mask).permute(0, 2, 1)
    mc_map_ms_fg_prob = torch.matmul(mc_map_ms_prob, ss_fg_mask).permute(0, 2, 1)

    # sc_map_ss_fg_prob, mc_map_ms_fg_prob: [1, 1, 64].
    # The total prob of each image token in the subj comp instance maps to fg areas 
    # in the subj single instance. 
    # If this prob is low, i.e., the image token doesn't match to any tokens in the fg areas 
    # in the subj single instance, then this token is probably background.
    sc_map_ss_fg_prob_below_mean = fg_bg_cutoff_prob - sc_map_ss_fg_prob
    mc_map_ms_fg_prob_below_mean = fg_bg_cutoff_prob - mc_map_ms_fg_prob
    # Set large negative values to 0, which correspond to large positive probs in 
    # sc_map_ss_fg_prob/mc_map_ms_fg_prob at fg areas of the corresponding single instances,
    # likely being foreground areas. 
    # When sc_map_ss_fg_prob/mc_map_ms_fg_prob < fg_bg_cutoff_prob = 0.25, this token is likely to be bg.
    # The smaller sc_map_ss_fg_prob/mc_map_ms_fg_prob is, the more likely this token is bg.
    sc_map_ss_fg_prob_below_mean = torch.clamp(sc_map_ss_fg_prob_below_mean, min=0)
    mc_map_ms_fg_prob_below_mean = torch.clamp(mc_map_ms_fg_prob_below_mean, min=0)

    # Image tokens that don't map to any fg image tokens in subj single instance
    # (i.e., corresponding to small sc_map_ss_fg_prob elements)
    # are considered as bg tokens.
    # Note sc_to_ss_bg_prob is a soft mask, not a hard mask.
    # sc_to_ss_bg_prob: [1, 1, 64]. Can be viewed as a token-wise weight, used
    # to give CA layer output features different weights at different tokens.
    # sc_to_ss_bg_prob: [1, 1, 64] => [1, 64, 1]. values: 0 ~ fg_bg_cutoff_prob=0.25.
    sc_to_ss_bg_prob = sc_map_ss_fg_prob_below_mean.permute(0, 2, 1)

    # We compute cosine loss on the features dim. 
    # So we permute the features to the last dim.
    # sc_feat, mc_feat: [1, 1280, 64] => [1, 64, 1280].
    sc_feat = sc_feat.permute(0, 2, 1)
    mc_feat = mc_feat.permute(0, 2, 1)

    # ref_grad_scale=0: doesn't update through mc_feat (but since mc_feat is generated without
    # subj embeddings, and without grad, even if ref_grad_scale > 0, no gradients 
    # will be backpropagated to subj embeddings).
    loss_sc_mc_bg_match = calc_ref_cosine_loss(sc_feat, mc_feat, 
                                               emb_mask=sc_to_ss_bg_prob,
                                               exponent=2, do_demeans=[False, False],
                                               first_n_dims_into_instances=2, 
                                               aim_to_align=True, 
                                               ref_grad_scale=0)
    
    return loss_subj_comp_map_single_align_with_cls, losses_sc_recon_ss_fg, losses_ss_fg_recon_sc, \
           loss_sc_mc_bg_match, sc_map_ss_fg_prob_below_mean, mc_map_ms_fg_prob_below_mean
