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
import random, math, sys, re

from torch.optim.lr_scheduler import SequentialLR
from bisect import bisect_right
from safetensors.torch import load_file as safetensors_load_file

class SequentialLR2(SequentialLR):
    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            if scheduler.__dict__.get('start_from_epoch_0', True):
                scheduler.step(0)
            else:
                print(f"Skip setting epoch to 0 for the scheduler {type(scheduler)}.")
                scheduler.step()
                print(f"last_epoch = {self.last_epoch}, LR = {scheduler.get_last_lr()}")
        else:
            scheduler.step()

        self._last_lr = scheduler.get_last_lr()

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

# The most important variables: "subjects", "class_names", "broad_classes", "sel_set"
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
                    if re.match("broad_classes|are_faces", var_name):
                        values = [ int(s) for s in substrings ]
                    elif var_name == 'sel_set':
                        values = [ int(s) - 1 for s in substrings ]
                    else:
                        values = substrings

                    if len(values) == 1 and values[0].startswith("$"):
                        # e.g., set -g cls_strings    $cls_delta_strings
                        values = subj_info[values[0][1:]]

                    subj_info[var_name] = values
                else:
                    breakpoint()

    for var_name in [ "subjects", "class_names", "cls_delta_strings", "data_folder" ]:
        if var_name not in subj_info:
            print("Variable %s is not defined in %s" %(var_name, subject_file_path))
            breakpoint()

    if 'broad_classes' not in subj_info:
        # By default, all subjects are humans/animals, unless specified in the subject file.
        subj_info['broad_classes'] = [ 1 for _ in subj_info['subjects'] ]

    for var_name in [ "class_names", "cls_delta_strings", 
                      "bg_init_strings", "broad_classes", "are_faces" ]:
        if var_name in subj_info:
            subj2attr[var_name] = {}
            if len(subj_info[var_name]) != len(subj_info['subjects']):
                print("Variable %s has %d elements, while there are %d subjects." 
                      %(var_name, len(subj_info[var_name]), len(subj_info['subjects'])))
                breakpoint()
            for i in range(len(subj_info['subjects'])):
                subj_name = subj_info['subjects'][i]
                subj2attr[var_name][subj_name] = subj_info[var_name][i]
            
    if 'sel_set' not in subj_info:
        # 'sel_set' is not defined in the subject file. So select all subjects.
        subj_info['sel_set'] = list(range(len(subj_info['subjects'])))

    # The most important variables: "subjects", "cls_delta_strings", "data_folder", "class_names"
    return subj_info, subj2attr

# Orthogonal subtraction of b from a: the residual is orthogonal to b (on the last dimension).
# NOTE: ortho_subtract(a, b) is scale-invariant w.r.t. b.
# ortho_subtract(a, b) scales proportionally to the scale of a.
# a, b are n-dimensional tensors. Subtraction happens at the last dim.
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

# Normalize a, b to unit vectors, then do orthogonal subtraction.
# Only used in calc_layer_subj_comp_k_or_v_ortho_loss, to balance the scales of subj and comp embeddings.
def normalized_ortho_subtract(a, b):
    a_norm = a.norm(dim=-1, keepdim=True) + 1e-6
    b_norm = b.norm(dim=-1, keepdim=True) + 1e-6
    a = a * (a_norm + b_norm) / (a_norm * 2)
    b = b * (a_norm + b_norm) / (b_norm * 2)
    diff = ortho_subtract(a, b)
    return diff

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
# emb_mask: [2, 1, 77, 1]. Could be fractional, e.g., 0.5, to discount some tokens.
# ref_grad_scale = 0: no gradient will be BP-ed to the reference embedding.
def calc_ref_cosine_loss(delta, ref_delta, batch_mask=None, emb_mask=None, 
                         exponent=2, do_demean_first=False,
                         first_n_dims_to_flatten=3,
                         ref_grad_scale=0, aim_to_align=True, 
                         margin=0, debug=False):

    B = delta.shape[0]
    loss = 0
    if batch_mask is not None:
        assert batch_mask.shape == (B,)
        # All instances are not counted. So return 0.
        if batch_mask.sum() == 0:
            return 0
    else:
        batch_mask = torch.ones(B, device=delta.device)

    # Calculate the loss for each sample in the batch, 
    # as the mask may be different for each sample.
    for i in range(B):
        # Keep the batch dimension when dealing with the i-th sample.
        delta_i     = delta[[i]]
        ref_delta_i = ref_delta[[i]]
        emb_mask_i  = emb_mask[[i]] if emb_mask is not None else None

        # Remove useless tokens, e.g., the placeholder suffix token(s) and padded tokens.
        if emb_mask_i is not None:
            try:
                delta_i_flattened_dims_shape = delta_i.shape[:first_n_dims_to_flatten]
                truncate_mask = (emb_mask_i > 0).squeeze(-1).expand(delta_i_flattened_dims_shape)
                
                delta_i       = delta_i[truncate_mask]
                ref_delta_i   = ref_delta_i[truncate_mask]
                # Make emb_mask_i have the same shape as delta_i, 
                # except the last (embedding) dimension for computing the cosine loss.
                # delta_i: [1, 16, 20, 768]. 
                # emb_mask_i: [1, 1, 77, 1] => [1, 16, 77] => [1, 16, 20].
                # Expanding to same shape is necessary, since the cosine of each embedding has an 
                # individual weight (no broadcasting happens).
                emb_mask_i    = emb_mask_i.squeeze(-1).expand(delta_i_flattened_dims_shape)[truncate_mask]
            except:
                breakpoint()

        else:
            # Flatten delta and ref_delta, by tucking the layer and token dimensions into the batch dimension.
            # delta_i: [2464, 768], ref_delta_i: [2464, 768]
            delta_i     = delta_i.reshape(delta_i.shape[:first_n_dims_to_flatten].numel(), -1)
            ref_delta_i = ref_delta_i.reshape(delta_i.shape)
            # emb_mask_i should have first_n_dims_to_flatten dims before flattening.
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

        if do_demean_first:
            delta_i      = demean(delta_i)
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
        losses_i = F.cosine_embedding_loss(delta_i, ref_delta_i_pow, 
                                           torch.ones_like(delta_i[:, 0]) * cosine_label, 
                                           reduction='none')
        # emb_mask_i has been flatten to 1D. So it gives different embeddings 
        # different relative weights (after normalization).
        if emb_mask_i is not None:
            loss_i = (losses_i * emb_mask_i).sum() / (emb_mask_i.sum() + 1e-8)
        else:
            loss_i = losses_i.mean()

        loss_i = loss_i * batch_mask[i]

        if margin > 0:
            # Only incurs loss when loss_i is larger than margin.
            # If the loss is above the margin, subtracting the margin won't change the gradient,
            # as the margin is constant.            
            loss_i = torch.clamp(loss_i - margin, min=0)

        loss += loss_i

    loss /= batch_mask.sum()
    return loss

# feat_base, feat_ex, ...: [2, 9, 1280].
# Last dim is the channel dim.
# feat_ex     is the extension (enriched features) of feat_base.
# ref_feat_ex is the extension (enriched features) of ref_feat_base.
# delta_types: feat_to_ref or ex_to_base or both.
def calc_delta_alignment_loss(feat_base, feat_ex, ref_feat_base, ref_feat_ex, 
                              ref_grad_scale=0.1, feat_base_grad_scale=0.05,
                              use_cosine_loss=True, cosine_exponent=2,
                              delta_types=['feat_to_ref', 'ex_to_base']):
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
        # NOTE: use normalized_ortho_subtract() will reduce performance.
        # Align tgt_delta to src_delta.
        losses_delta_align = {}
        for delta_choice in delta_types:
            if delta_choice == 'feat_to_ref':                
                src_delta = ortho_subtract(feat_base_gs, ref_feat_base_gs)
                tgt_delta = ortho_subtract(feat_ex,      ref_feat_ex_gs)
            elif delta_choice == 'ex_to_base':
                src_delta = ortho_subtract(ref_feat_ex_gs, ref_feat_base_gs)
                tgt_delta = ortho_subtract(feat_ex,        feat_base_gs)

            if use_cosine_loss:
                # ref_grad_scale=1: ref grad scaling is disabled within calc_ref_cosine_loss,
                # since we've done gs on ref_feat_base, ref_feat_ex, and feat_base.
                loss_delta_align = calc_ref_cosine_loss(tgt_delta, src_delta, 
                                                        exponent=cosine_exponent,
                                                        do_demean_first=False,
                                                        first_n_dims_to_flatten=(feat_base.ndim - 1), 
                                                        ref_grad_scale=1,
                                                        aim_to_align=True)
            else:
                # ref_grad_scale=1: ref grad scaling is disabled within calc_ref_cosine_loss,
                # since we've done gs on ref_feat_base, ref_feat_ex, and feat_base.
                # do_sqr=True: square the loss, so that the loss is more sensitive to 
                # smaller (<< 1) align_coeffs.            
                loss_delta_align = calc_align_coeff_loss(tgt_delta, src_delta, 
                                                         margin=1., ref_grad_scale=1, do_sqr=True)
            
            losses_delta_align[delta_choice] = loss_delta_align

        return losses_delta_align

def calc_align_coeff_loss(f1, f2, margin=1., encourage_align=True, ref_grad_scale=1, do_sqr=True):
    ref_grad_scaler = gen_gradient_scaler(ref_grad_scale)
    # Reduce the gradient to the reference features, 
    # as the reference features are supposed to be unchanged, as opposed to feat_*. 
    # (although it still has a learnable component from mixed subject prompt embeddings.)
    f2_gs  = ref_grad_scaler(f2)    
    align_coeffs  = calc_align_coeffs(f1, f2_gs)
    if encourage_align:
        # We encourage f1 to express at least margin * f2, i.e.,
        # align_coeffs should be >= margin. So a loss is incurred if it's < margin.
        # do_sqr=True: square the loss, so that the loss is more sensitive to smaller (<< margin) align_coeffs.
        loss_align  = masked_mean(margin - align_coeffs,
                                  margin - align_coeffs > 0,
                                  do_sqr=do_sqr)
    else:
        # We discourage f1 to express more than margin * f2, i.e.,
        # align_coeffs should be <= margin. So a loss is incurred if it's > margin.
        loss_align  = masked_mean(align_coeffs - margin,
                                  align_coeffs - margin > 0,
                                  do_sqr=do_sqr)
    return loss_align

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
    spatial_shape2 = (int(out_spatial_shape[0] * spatial_scale), int(out_spatial_shape[1] * spatial_scale))
    spatial_attn = flat_attn.mean(dim=2).sum(dim=1).reshape(BS, 1, *spatial_shape2)
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
        # B is 16 * actual batch size (for layerwise embeddings).
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

def merge_cls_token_embeddings(prompt_embedding, cls_delta_string_indices, 
                               subj_name_to_cls_delta_token_weights=None):
    if cls_delta_string_indices is None or len(cls_delta_string_indices) == 0:
        return prompt_embedding

    device = prompt_embedding.device
    # cls_delta_string_indices is a list of tuples, each tuple is
    # (batch_i, start_N, num_cls_delta_tokens, subj_name).
    # Sort first by batch index, then by start index. So that the index offsets within each instance will
    # add up correctly as we process all cls_delta_string_indices tuples within the current instance.
    # subj_name is used to look up the cls delta weights.
    cls_delta_string_indices = sorted(cls_delta_string_indices, key=lambda x: (x[0], x[1]))
    batch_i2offset = {}

    prompt_embedding2 = prompt_embedding.clone()
    occurred_subj_names = {}

    for batch_i, start_index_N, M, subj_name in cls_delta_string_indices:
        i_off = batch_i2offset.get(batch_i, 0)
        # cls_delta_embeddings: [M, 768].
        cls_delta_embeddings = prompt_embedding[batch_i, start_index_N:start_index_N+M]
        if subj_name_to_cls_delta_token_weights is not None:
            # cls_delta_token_weights: [M] -> [M, 1].
            cls_delta_token_weights = subj_name_to_cls_delta_token_weights[subj_name].unsqueeze(1).to(device)
        else:
            cls_delta_token_weights = torch.ones(M, 1, device=device)
        # avg_cls_delta_embedding: [768].
        avg_cls_delta_embedding = (cls_delta_embeddings * cls_delta_token_weights).sum(dim=0)
        prompt_embedding2[batch_i, start_index_N-i_off] = avg_cls_delta_embedding
        # NOTE: Our purpose is to combine all the cls delta tokens to 1 token, so that
        # their positions align with the subject tokens in the first half of the batch.
        # To do so, we move the embeddings (except the EOS) after the last cls delta token to the left,
        # overwriting the rest M-1 cls delta embeddings.
        prompt_embedding2[batch_i, start_index_N+1-i_off:-(M+i_off)] = prompt_embedding[batch_i, start_index_N+M:-1]
        batch_i2offset[batch_i] = i_off + M - 1
        occurred_subj_names[subj_name] = \
            occurred_subj_names.get(subj_name, 0) + 1

    #if len(occurred_subj_names) > 1:
    #    breakpoint()
        
    return prompt_embedding2

# text_embedding: [B, N, D]
# If text_embedding is static embedding, then num_layers=16, 
# we need to reshape it to [B0, num_layers, N, D].
def fix_emb_scale(text_embedding, placeholder_indices, empty_context=None,
                  num_layers=1, scale_range=(1.0, 1.0), extra_scale=1):
    if placeholder_indices is None or scale_range == (1.0, 1.0):
        return text_embedding
    
    placeholder_indices_B, placeholder_indices_N = placeholder_indices
    M = len(torch.unique(placeholder_indices_N))
    B = text_embedding.shape[0]
    B0      = B // num_layers
    B_IND   = len(torch.unique(placeholder_indices_B))

    scale_range = (scale_range[0] * extra_scale, scale_range[1] * extra_scale)

    if B_IND > B0:
        breakpoint()
    
    text_embedding_shape = text_embedding.shape

    # It's possible B_IND < B, i.e., the processed token only appears in some of the prompts.
    # For example, the subject token only appears in the first half batch of a 
    # compositional distillation iteration.
    text_embedding = text_embedding.reshape(B0, num_layers, *text_embedding.shape[1:])
    scale_mask = torch.ones_like(text_embedding)
    
    SCALE_STEP = (scale_range[1] - scale_range[0]) / (num_layers - 1)
    # Linearly increase the scale of the subject embeddings from 0.5 to 1.5, in 16 steps
    # [0.5000, 0.5625, 0.6250, 0.6875, 0.7500, 0.8125, 0.8750, 0.9375, 
    #  1.0000, 1.0625, 1.1250, 1.1875, 1.2500, 1.3125, 1.3750, 1.4375]
    scales = scale_range[0] + torch.arange(0, num_layers, device=text_embedding.device).reshape(1, -1, 1) * SCALE_STEP

    scale_mask[placeholder_indices_B, :, placeholder_indices_N] = scales

    if empty_context is not None:
        scaled_text_embedding = text_embedding * scale_mask + empty_context * (1 - scale_mask)
    else:
        scaled_text_embedding = text_embedding * scale_mask

    # Change back to the original shape.
    scaled_text_embedding = scaled_text_embedding.reshape(text_embedding_shape)

    return scaled_text_embedding

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
          
# samples:   a list of (B, C, H, W) tensors.
# img_flags: a list of (B,) ints.
# If not do_normalize, samples should be between [0, 1] (float types) or [0, 255] (uint8).
# If do_normalize, samples should be between [-1, 1] (raw output from SD decode_first_stage()).
def save_grid(samples, img_flags, grid_filepath, nrow, do_normalize=False):
    if isinstance(samples[0], np.ndarray):
        samples = [ torch.from_numpy(e) for e in samples ]

    # grid is a 4D tensor: (B, C, H, W)
    if not isinstance(samples, torch.Tensor):
        grid = torch.cat(samples, 0)
    else:
        grid = samples
    # img_flags is a 1D tensor: (B,)
    if img_flags is not None and not isinstance(img_flags, torch.Tensor):
        img_flags = torch.cat(img_flags, 0)

    if grid.dtype != torch.uint8:
        if do_normalize:
            grid = torch.clamp((grid + 1.0) / 2.0, min=0.0, max=1.0)
        grid = (255. * grid).to(torch.uint8)

    # img_box indicates the whole image region.
    img_box = torch.tensor([0, 0, grid.shape[2], grid.shape[3]]).unsqueeze(0)

    colors = [ None, 'green', 'red', 'purple' ]
    if img_flags is not None:
        # Highlight the teachable samples.
        for i, img_flag in enumerate(img_flags):
            if img_flag > 0:
                # Draw a 4-pixel wide green bounding box around the image.
                grid[i] = draw_bounding_boxes(grid[i], img_box, colors=colors[img_flag], width=12)

    # grid is a 3D np array: (C, H2, W2)
    grid = make_grid(grid, nrow=nrow).cpu().numpy()
    # grid is transposed to: (H2, W2, C)
    grid_img = Image.fromarray(grid.transpose([1, 2, 0]))
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
    
    return np.random.uniform(rand_lb, rand_ub)

def torch_uniform(low, high, size, device=None):
    return torch.rand(size, device=device) * (high - low) + low

# true_prob_range = (p_init, p_final). 
# The prob of flipping true is gradually annealed from p_init to p_final.
def draw_annealed_bool(training_percent, final_percent, true_prob_range):
    true_p_annealed = anneal_value(training_percent, final_percent, value_range=true_prob_range)
    # Flip a coin, with prob of true being true_p_annealed.    
    return random.random() < true_p_annealed

# ratio_range: range of fluctuation ratios (could > 1 or < 1).
# keep_prob_range: range of annealed prob of keeping the original t. If (0, 0.5),
# then gradually increase the prob of keeping the original t from 0 to 0.5.
def probably_anneal_t(t, training_percent, num_timesteps, ratio_range, keep_prob_range=(0, 0.5)):
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
            t_annealed[i] = np.random.randint(ti_lowerbound, ti_upperbound)
    else:
        t_lowerbound = min(max(int(t * ratio_lb), 0), num_timesteps - 1)
        t_upperbound = min(int(t * ratio_ub) + 1, num_timesteps)
        t_annealed = torch.tensor(np.random.randint(t_lowerbound, t_upperbound), 
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
            t_annealed[i] = np.random.randint(ti_lowerbound, ti_upperbound)
    else:
        t_lowerbound = min(max(int(t * ratio_lb), 0), num_timesteps - 1)
        t_upperbound = min(int(t * ratio_ub) + 1, num_timesteps)
        t_annealed = torch.tensor(np.random.randint(t_lowerbound, t_upperbound), 
                                  dtype=t.dtype, device=t.device)

    return t_annealed

def select_piecewise_value(ranged_values, curr_pos, range_ub=1.0):
    for i, (range_lb, value) in enumerate(ranged_values):
        if i < len(ranged_values) - 1:
            range_ub = ranged_values[i + 1][0]
        else:
            range_ub = 1.0

        if range_lb <= curr_pos < range_ub:
            return value

    raise ValueError(f"curr_pos {curr_pos} is out of range.")

# feat_or_attn: 4D features or 3D attention. If it's attention, then
# its geometrical dimensions (H, W) have been flatten to 1D (last dim).
# mask:      always 4D.
# mode: either "nearest" or "nearest|bilinear". Other modes will be ignored.
def resize_mask_for_feat_or_attn(feat_or_attn, mask, mask_name, num_spatial_dims=1,
                                 mode="nearest|bilinear", warn_on_all_zero=True):
    # Assume square feature maps, target_H = target_W.
    target_H = int(np.sqrt(feat_or_attn.shape[-num_spatial_dims:].numel()))
    spatial_shape2 = (target_H, target_H)

    # NOTE: avoid "bilinear" mode. If the object is too small in the mask, 
    # it may result in all-zero masks.
    # mask: [2, 1, 64, 64] => mask2: [2, 1, 8, 8].
    mask2_nearest  = F.interpolate(mask.float(), size=spatial_shape2, mode='nearest')
    if mode == "nearest|bilinear":
        mask2_bilinear = F.interpolate(mask.float(), size=spatial_shape2, mode='bilinear', align_corners=False)
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

def gen_emb_mixer(BS, subj_indices_1b_N, CLS_SCALE_LAYERWISE_RANGE, device, use_layerwise_embedding=True,
                  N_CA_LAYERS=16, sync_layer_indices=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]):
    
    CLS_FIRST_LAYER_SCALE, CLS_FINAL_LAYER_SCALE = CLS_SCALE_LAYERWISE_RANGE

    if use_layerwise_embedding:
        SCALE_STEP = (CLS_FINAL_LAYER_SCALE - CLS_FIRST_LAYER_SCALE) / (len(sync_layer_indices) - 1)
        # Linearly decrease the scale of the class   embeddings from 1.0 to 0.7, 
        # [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9727, 0.9455, 0.9182, 
        #  0.8909, 0.8636, 0.8364, 0.8091, 0.7818, 0.7545, 0.7273, 0.7000]
        # i.e., 
        # Linearly increase the scale of the subject embeddings from 0.0 to 0.3.
        # [0.    , 0.    , 0.    , 0.    , 0.    , 0.0273, 0.0545, 0.0818,
        #  0.1091, 0.1364, 0.1636, 0.1909, 0.2182, 0.2455, 0.2727, 0.3   ]
        emb_k_or_v_layers_cls_mix_scales = torch.ones(BS, N_CA_LAYERS, device=device) 
        emb_k_or_v_layers_cls_mix_scales[:, sync_layer_indices] = \
            CLS_FIRST_LAYER_SCALE + torch.arange(0, len(sync_layer_indices), device=device).repeat(BS, 1) * SCALE_STEP
    else:
        # Same scale for all layers.
        # emb_k_or_v_layers_cls_mix_scales = [0.85, 0.85, ..., 0.85].
        # i.e., the subject embedding scales are [0.15, 0.15, ..., 0.15].
        AVG_SCALE = (CLS_FIRST_LAYER_SCALE + CLS_FINAL_LAYER_SCALE) / 2
        emb_k_or_v_layers_cls_mix_scales = AVG_SCALE * torch.ones(N_CA_LAYERS, device=device).repeat(BS, 1)

    # First mix the static embeddings.
    # mix_embeddings('add', ...):  being subj_comp_emb almost everywhere, except those at subj_indices_1b_N,
    # where they are subj_comp_emb * emb_k_or_v_layers_cls_mix_scales + cls_comp_emb * (1 - emb_k_or_v_layers_cls_mix_scales).
    # subj_comp_emb, cls_comp_emb, subj_single_emb, cls_single_emb: [16, 77, 768].
    # Each is of a single instance. So only provides subj_indices_1b_N 
    # (multiple token indices of the same instance).
    # emb_v_mixer will be reused later to mix ada embeddings, so make it a functional.
    emb_v_mixer = partial(mix_embeddings, 'add', mix_indices=subj_indices_1b_N)
    return emb_v_mixer, emb_k_or_v_layers_cls_mix_scales

# t_frac is a float scalar. 
def mix_static_vk_embeddings(c_static_emb, subj_indices_1b_N, 
                             training_percent,
                             t_frac=1.0, 
                             use_layerwise_embedding=True,
                             N_CA_LAYERS=16, 
                             K_CLS_SCALE_LAYERWISE_RANGE=[1.0, 1.0],
                             V_CLS_SCALE_LAYERWISE_RANGE=[1.0, 0.7],
                             # 7, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24
                             sync_layer_indices=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                             ):
    
    subj_emb, cls_emb = c_static_emb.chunk(2)
    BS = subj_emb.shape[0] // N_CA_LAYERS
    if not isinstance(t_frac, torch.Tensor):
        t_frac = torch.tensor(t_frac, dtype=c_static_emb.dtype, device=c_static_emb.device)
    if len(t_frac) == 1:
        t_frac = t_frac.repeat(BS)
    t_frac = t_frac.unsqueeze(1)

    if len(t_frac) != BS:
        breakpoint()
    assert 0 - 1e-6 <= training_percent <= 1 + 1e-6

    emb_v_mixer, emb_v_layers_cls_mix_scales = \
        gen_emb_mixer(BS, subj_indices_1b_N, V_CLS_SCALE_LAYERWISE_RANGE, c_static_emb.device,
                      use_layerwise_embedding, N_CA_LAYERS, sync_layer_indices)
    # Part of subject embedding is mixed into mix_emb_v. 
    # Proportions of cls_emb into mix_emb_v are specified by emb_v_layers_cls_mix_scales.
    mix_emb_v = emb_v_mixer(cls_emb, subj_emb, c1_mix_scale=emb_v_layers_cls_mix_scales.view(-1))

    emb_k_mixer, emb_k_layers_cls_mix_scales = \
        gen_emb_mixer(BS, subj_indices_1b_N, K_CLS_SCALE_LAYERWISE_RANGE, c_static_emb.device,
                        use_layerwise_embedding, N_CA_LAYERS, sync_layer_indices)
    # Part of subject embedding is mixed into mix k embedding.
    # Proportions of cls_emb into mix_emb_k are specified by emb_k_layers_cls_mix_scales.
    mix_emb_k = emb_k_mixer(cls_emb, subj_emb, c1_mix_scale=emb_k_layers_cls_mix_scales.view(-1))

    # The first  half of mix_emb_all_layers will be used as V in cross attention layers.
    # The second half of mix_emb_all_layers will be used as K in cross attention layers.
    mix_emb_all_layers = torch.cat([mix_emb_v, mix_emb_k], dim=1)

    PROMPT_MIX_GRAD_SCALE = 0.05
    grad_scaler = gen_gradient_scaler(PROMPT_MIX_GRAD_SCALE)
    # mix_comp_emb receives smaller grad, since it only serves as the reference.
    # If we don't scale gradient on mix_comp_emb, chance is mix_comp_emb might be 
    # dominated by subj_comp_emb,
    # so that mix_comp_emb will produce images similar as subj_comp_emb does.
    # Scaling the gradient will improve compositionality but reduce face similarity.
    mix_emb_all_layers   = grad_scaler(mix_emb_all_layers)

    # This copy of subj_emb will be simply 
    # repeated at the token dimension to match the token number of the mixed (concatenated) 
    # mix_emb embeddings.
    subj_emb2   = subj_emb.repeat(1, 2, 1)
    
    # Only mix sync_layer_indices layers.
    if use_layerwise_embedding:
        # sync_layer_indices = [4, 5, 6, 7, 8, 9, 10] #, 11, 12, 13]
        # 4, 5, 6, 7, 8, 9, 10 correspond to original layer indices 7, 8, 12, 16, 17, 18, 19.
        # (same as used in computing mixing loss)
        # layer_mask: [2, 16, 154, 768]
        layer_mask = torch.zeros_like(mix_emb_all_layers).reshape(-1, N_CA_LAYERS, *mix_emb_all_layers.shape[1:])
        # t_frac controls how much mix_emb_all_layers is mixed with subj_comp_emb2 into mix_comp_emb,
        # and how much mix_single_emb_all_layers is mixed with subj_single_emb2 into mix_single_emb.

        # layer_mask[:, sync_layer_indices]: [2, 7, 154, 768]
        # selected layers in layer_mask (used on subj_emb2) vary between [0, 1] 
        # according to t_frac and training_percent.
        # i.e., when training_percent=0,
        # when t=999, the proportions of subj_emb2 is 0.
        # when t=0,   the proportions of subj_emb2 is 1.
        #       when training_percent=1,
        # when t=999, the proportions of subj_emb2 is 0.3.
        # when t=0,   the proportions of subj_emb2 is 1.
        # Essentially, this is doing diffusion w.r.t. subj_emb2 proportions.
        layer_mask[:, sync_layer_indices] = 1 - t_frac.view(-1, 1, 1, 1) * (1 - training_percent * 0.3)
        layer_mask = layer_mask.reshape(-1, *mix_emb_all_layers.shape[1:])

        # Use most of the layers of embeddings in subj_comp_emb2, but 
        # replace sync_layer_indices layers with those from mix_emb_all_layers.
        # Do not assign with sync_layers as indices, which destroys the computation graph.
        mix_emb   =   subj_emb2          * layer_mask \
                    + mix_emb_all_layers * (1 - layer_mask)
        
    else:
        # There is only one layer of embeddings.
        mix_emb   = mix_emb_all_layers

    # c_static_emb_vk is the static embeddings of the prompts used in losses other than 
    # the static delta loss, e.g., used to estimate the ada embeddings.
    # If use_ada_embedding, then c_in2 will be fed again to CLIP text encoder to 
    # get the ada embeddings. Otherwise, c_in2 will be useless and ignored.
    # c_static_emb_vk: [64, 154, 768]
    # c_static_emb_vk will be added with the ada embeddings to form the 
    # conditioning embeddings in the U-Net.
    # Unmixed embeddings and mixed embeddings will be merged in one batch for guiding
    # image generation and computing compositional mix loss.
    c_static_emb_vk = torch.cat([ subj_emb2, mix_emb ], dim=0)

    # emb_v_mixer will be used later to mix ada embeddings in UNet.
    # extra_info['emb_v_mixer']                   = emb_v_mixer
    # extra_info['emb_v_layers_cls_mix_scales']  = emb_v_layers_cls_mix_scales
    
    return c_static_emb_vk 
    #, emb_v_mixer, emb_v_layers_cls_mix_scales, emb_k_mixer, emb_k_layers_cls_mix_scales

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

# cls_subj_indices, cls_comp_indices could be None. 
# In that case, subj_comp_emb_align is pushed towards 0.
# margin: 
def calc_layer_subj_comp_k_or_v_ortho_loss(seq_ks, subj_subj_indices, subj_comp_indices, 
                                           cls_subj_indices, cls_comp_indices,
                                           all_token_weights=None, 
                                           do_demean_first=False, cls_grad_scale=0.05,
                                           margin=0.6):

    # Put the 4 subject embeddings in the 2nd to last dimension for torch.mm().
    # The ortho losses on different "instances" are computed separately 
    # and there's no interaction among them.

    # seq_ks: [B, H, N, D] = [2, 8, 77, 160].
    # H = 8, number of attention heads. D: 160, number of image tokens.
    # seq_ks: [B, H, N, D] -> [B, N, H, D] = [2, 77, 8, 160].
    seq_ks = seq_ks.permute(0, 2, 1, 3)

    # subj_subj_ks, cls_subj_ks: [4,  8, 160] => [1, 4,  8, 160] => [1, 8, 160]
    # subj_comp_ks, cls_comp_ks: [15, 8, 160] => [1, 15, 8, 160] => [1, 8, 160]
    subj_subj_ks    = sel_emb_attns_by_indices(seq_ks, subj_subj_indices, 
                                               all_token_weights=all_token_weights,
                                               do_sum=False, do_mean=True, do_sqrt_norm=False)
    subj_comp_ks    = sel_emb_attns_by_indices(seq_ks, subj_comp_indices, 
                                               all_token_weights=all_token_weights,
                                               do_sum=False, do_mean=True, do_sqrt_norm=False)
    cls_subj_ks     = sel_emb_attns_by_indices(seq_ks, cls_subj_indices, 
                                               all_token_weights=all_token_weights,
                                               do_sum=False, do_mean=True, do_sqrt_norm=False)
    cls_comp_ks     = sel_emb_attns_by_indices(seq_ks, cls_comp_indices, 
                                               all_token_weights=all_token_weights,
                                               do_sum=False, do_mean=True, do_sqrt_norm=False)

    # The orthogonal projection of subj_subj_ks against subj_comp_ks_sum.
    # subj_comp_ks_sum will broadcast to the K_fg dimension of subj_subj_ks.
    subj_comp_emb_diff = normalized_ortho_subtract(subj_subj_ks, subj_comp_ks)
    # The orthogonal projection of cls_subj_ks against cls_comp_ks_sum.
    # cls_comp_ks_sum will broadcast to the K_fg dimension of cls_subj_ks_mean.
    cls_comp_emb_diff  = normalized_ortho_subtract(cls_subj_ks,  cls_comp_ks)
    # The two orthogonal projections should be aligned. That is, each embedding in subj_subj_ks 
    # is allowed to vary only along the direction of the orthogonal projections of class embeddings.
    # Don't compute the ortho loss on dress-type compositions, 
    # such as "z wearing a santa hat / z that is red", because the attended areas 
    # largely overlap with the subject, and making them orthogonal will 
    # hurt their expression in the image (e.g., push the attributes to the background).
    # Encourage subj_comp_emb_diff and cls_comp_emb_diff to be aligned (dot product -> 1).
    loss_layer_subj_comp_key_ortho = \
        calc_ref_cosine_loss(subj_comp_emb_diff, cls_comp_emb_diff, 
                             batch_mask=None, exponent=2,
                             do_demean_first=do_demean_first,  # default: False
                             first_n_dims_to_flatten=2,
                             ref_grad_scale=cls_grad_scale,
                             aim_to_align=True,
                             margin=margin)

    return loss_layer_subj_comp_key_ortho

# If do_sum, returned emb_attns is 3D. Otherwise 4D.
# indices are applied on the first 2 dims of attn_mat.
def sel_emb_attns_by_indices(attn_mat, indices, all_token_weights=None, 
                             do_sum=True, do_mean=False, do_sqrt_norm=False):

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

    if do_sqrt_norm:
        # Normalize each embedding by sqrt(K_bg_i).
        emb_attns   = [ emb_attns[i] / np.sqrt(len(inst_indices[0])) \
                        for i, inst_indices in enumerate(indices_by_instance) ]
    
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
            # new_comp_extra is empty, i.e., either wds is not enabled, or 
            # the particular instance has no corresponding fg_mask.
            # Keep the original compositional prompt.
            new_comp_prompts.append( comp_prompts[i] )
    
    return new_comp_prompts

def dampen_large_loss(loss, max_magnitude=1.):
    if loss > max_magnitude:
        loss = loss * max_magnitude / loss.item()

    return loss

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

# Textual inversion is supported, where static_embeddings is only one embedding.
# static_embeddings: size: [4, 16, 77, 768]. 4: batch_size. 16: number of UNet layers.
# embeddings of static_subj_single_emb, static_subj_comp_emb, static_cls_single_emb, static_cls_comp_emb. 
def calc_prompt_emb_delta_loss(static_embeddings, prompt_emb_mask, cls_delta_grad_scale=0.05):
    # static_embeddings contains 4 types of embeddings:
    # subj_single, subj_comp, cls_single, cls_comp.
    # static_embeddings: [4, 16, 77, 768].
    # cls_*: embeddings generated from prompts containing a class token (as opposed to the subject token).
    # Each is [1, 16, 77, 768]
    static_subj_single_emb, static_subj_comp_emb, static_cls_single_emb, static_cls_comp_emb = \
            static_embeddings.chunk(4)

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
        # prompt_emb_mask_weighted: [1, 77, 1] => [1, 1, 77, 1].
        prompt_emb_mask_weighted = prompt_emb_mask_weighted.unsqueeze(1)
    else:
        prompt_emb_mask_weighted = None

    use_ortho_subtract = True
    # static_cls_delta: [1, 16, 77, 768]. Should be a repeat of a tensor of size [1, 1, 77, 768]. 
    # Delta embedding between class single and comp embeddings.
    # by 16 times along dim=1, as cls_prompt_* doesn't contain placeholder_token.
    # static_subj_delta: [1, 16, 77, 768]. Different values for each layer along dim=1.
    # Delta embedding between subject single and comp embeddings.
    # static_subj_delta / ada_subj_delta should be aligned with cls_delta.
    if use_ortho_subtract:
        static_subj_delta   = ortho_subtract(static_subj_comp_emb, static_subj_single_emb)
        static_cls_delta    = ortho_subtract(static_cls_comp_emb,  static_cls_single_emb)
    else:
        static_subj_delta   = static_subj_comp_emb  - static_subj_single_emb
        static_cls_delta    = static_cls_comp_emb   - static_cls_single_emb

    loss_static_prompt_delta   = \
        calc_ref_cosine_loss(static_subj_delta, static_cls_delta, 
                             emb_mask=prompt_emb_mask_weighted,
                             do_demean_first=True,
                             first_n_dims_to_flatten=3,
                             ref_grad_scale=cls_delta_grad_scale,   # 0.05
                             aim_to_align=True)

    return loss_static_prompt_delta

def calc_dyn_loss_scale(loss, loss_base, loss_scale_base, min_scale_base_ratio=1, max_scale_base_ratio=2):
    # Setting loss_base to 0 will disable the loss.
    if loss_base == 0:
        return 0
    scale = loss.item() * loss_scale_base / loss_base
    min_scale = loss_scale_base * min_scale_base_ratio
    max_scale = loss_scale_base * max_scale_base_ratio
    scale = max(min(max_scale, scale), min_scale)
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

# add_noise_to_tensor() adds a fixed amount of noise to the tensor.
def add_noise_to_tensor(ts, noise_std, noise_std_is_relative=True, keep_norm=False,
                        std_dim=-1, norm_dim=-1):
    if noise_std_is_relative:
        ts_std_mean = ts.std(dim=std_dim).mean().detach()
        noise_std *= ts_std_mean

    noise = torch.randn_like(ts) * noise_std
    if keep_norm:
        orig_norm = ts.norm(dim=norm_dim, keepdim=True)
        ts = ts + noise
        new_norm  = ts.norm(dim=norm_dim, keepdim=True).detach()
        ts = ts * orig_norm / (new_norm + 1e-8)
    else:
        ts = ts + noise
        
    return ts

# embeddings: [N, 768]. 
# noise_std_range: the noise std / embeddings std falls within this range.
# anneal_add_noise_to_embedding() adds noise of the amount randomly selected from the noise_std_range.
def anneal_add_noise_to_embedding(embeddings, training_percent, begin_noise_std_range, end_noise_std_range, 
                                  add_noise_prob, noise_std_is_relative=True, keep_norm=False,
                                  std_dim=-1, norm_dim=-1):
    if random.random() > add_noise_prob:
        return embeddings
    
    if end_noise_std_range is not None:
        noise_std_lb = anneal_value(training_percent, 1, (begin_noise_std_range[0], end_noise_std_range[0]))
        noise_std_ub = anneal_value(training_percent, 1, (begin_noise_std_range[1], end_noise_std_range[1]))
    else:
        noise_std_lb, noise_std_ub = begin_noise_std_range
        
    noise_std = np.random.uniform(noise_std_lb, noise_std_ub)

    noised_embeddings = add_noise_to_tensor(embeddings, noise_std, noise_std_is_relative, keep_norm, std_dim, norm_dim)
    return noised_embeddings

# At scaled background, fill new x_start with random values (100% noise). 
# At scaled foreground, fill new x_start with noised scaled x_start. 
def init_x_with_fg_from_training_image(x_start, fg_mask, filtered_fg_mask, 
                                       training_percent, base_scale_range=(0.7, 1.0),
                                       fg_noise_anneal_mean_range=(0.1, 0.5)):
    x_start_origsize = torch.where(filtered_fg_mask.bool(), x_start, torch.randn_like(x_start))
    fg_mask_percent = filtered_fg_mask.float().sum() / filtered_fg_mask.numel()
    # print(fg_mask_percent)
    base_scale_range_lb, base_scale_range_ub = base_scale_range

    if fg_mask_percent > 0.1:
        # If fg areas are larger (>= 1/10 of the whole image), then scale down
        # more aggressively, to avoid it dominating the whole image.
        # Don't take extra_scale linearly to this ratio, which will make human
        # faces (usually taking around 20%-40%) too small.
        # Example: fg_mask_percent = 0.2, extra_scale = 0.5 ** 0.35 = 0.78.
        extra_scale = math.pow(0.1 / fg_mask_percent.item(), 0.35)
        scale_range_lb = base_scale_range_lb * extra_scale
        # scale_range_ub is at least 0.5.
        scale_range_ub = max(0.5, base_scale_range_ub * extra_scale)
        fg_rand_scale = np.random.uniform(scale_range_lb, scale_range_ub)
    else:
        # fg areas are small. Scale the fg area to 70%-100% of the original size.
        fg_rand_scale = np.random.uniform(base_scale_range_lb, base_scale_range_ub)

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

def gen_cfg_scales_for_stu_tea(tea_scale, stu_scale, num_teachers, device):
    cfg_scales_for_teacher   = torch.ones(num_teachers) * tea_scale
    cfg_scales_for_student   = torch.ones(num_teachers) * stu_scale
    cfg_scales_for_clip_loss = torch.cat([cfg_scales_for_student, cfg_scales_for_teacher], dim=0)
    cfg_scales_for_clip_loss = cfg_scales_for_clip_loss.to(device)
    return cfg_scales_for_clip_loss

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

def calc_elastic_matching_loss(ca_q, ca_outfeat, fg_mask, fg_bg_cutoff_prob=0.25,
                               single_q_grad_scale=0.1, single_feat_grad_scale=0.01,
                               mix_feat_grad_scale=0.05):
    # fg_mask: [1, 1, 64] => [1, 64]
    fg_mask = fg_mask.bool().squeeze(1)
    if fg_mask.sum() == 0:
        return 0, 0, 0, None, None

    single_q_grad_scaler    = gen_gradient_scaler(single_q_grad_scale)
    single_feat_grad_scaler = gen_gradient_scaler(single_feat_grad_scale)

    # ca_q, ca_outfeat: [4, 1280, 64]
    # ss_q, sc_q, ms_q, mc_q: [1, 1280, 64]. 
    # ss_*: subj single, sc_*: subj comp, ms_*: mix single, mc_*: mix comp.
    ss_q, sc_q, ms_q, mc_q = ca_q.chunk(4)
    ss_q_gs = single_q_grad_scaler(ss_q)
    ms_q_gs = single_q_grad_scaler(ms_q)

    #num_heads = 8
    # Similar to the scale of the attention scores.
    matching_score_scale = 1 #(ca_q.shape[1] / num_heads) ** -0.5
    #print('matching_score_scale:', matching_score_scale)
    # sc_map_ss_score:        [1, 64, 64]. 
    # Pairwise matching scores (9 subj comp image tokens) -> (9 subj single image tokens).
    sc_map_ss_score = torch.matmul(sc_q.transpose(1, 2).contiguous(), ss_q_gs) * matching_score_scale
    # sc_map_ss_prob:   [1, 64, 64]. 
    # Pairwise matching probs (9 subj comp image tokens) -> (9 subj single image tokens).
    # Normalize among subj comp tokens (sc dim).
    # NOTE: sc_map_ss_prob and mc_map_ms_prob are normalized among the comp instance dim,
    # this can address scale changes (e.g. the subject is large in single instances,
    # but becomes smaller in comp instances). If they are normalized among the single instance dim,
    # then each image token in the comp instance has a fixed total contribution to the reconstruction
    # of the single instance, which can hardly handle scale changes.
    sc_map_ss_prob  = F.softmax(sc_map_ss_score, dim=1)

    mc_map_ms_score = torch.matmul(mc_q.transpose(1, 2).contiguous(), ms_q_gs) * matching_score_scale
    # Normalize among mix comp tokens (mc dim).
    mc_map_ms_prob  = F.softmax(mc_map_ms_score, dim=1)
    # breakpoint()

    # ss_feat, sc_feat, ms_feat, mc_feat: [4, 1280, 64] => [1, 1280, 64].
    ss_feat, sc_feat, ms_feat, mc_feat = ca_outfeat.chunk(4)
    # recon_sc_feat: [1, 1280, 64] * [1, 64, 64] => [1, 1280, 64]
    # We can only use the subj comp tokens to reconstruct the subj single tokens, not vice versa. 
    # Because we need to apply fg_mask, which is only available for the subj single tokens. Then we
    # can compare the values of the recon subj single tokens with the original values at the fg area.
    # torch.einsum('b d i, b i j -> b d j', sc_feat, sc_map_ss_prob) is equivalent to
    # torch.matmul(sc_feat, sc_map_ss_prob). But maybe matmul is faster?
    # sc_map_ss_fg_prob: 
    fg_mask_B, fg_mask_N = fg_mask.nonzero(as_tuple=True)
    # sc_map_ss_fg_prob: [1, 64, 64] => [1, 64, N_fg]
    sc_map_ss_fg_prob = sc_map_ss_prob[:, :, fg_mask_N]
    # sc_recon_ss_fg_feat: [1, 1280, 64] * [1, 64, N_fg] => [1, 1280, N_fg]
    sc_recon_ss_fg_feat = torch.matmul(sc_feat, sc_map_ss_fg_prob)
    # sc_recon_ss_fg_feat: [1, 1280, N_fg] => [1, N_fg, 1280]
    sc_recon_ss_fg_feat = sc_recon_ss_fg_feat.permute(0, 2, 1)
    # mc_recon_ms_feat = torch.matmul(mc_feat, mc_map_ms_prob)

    # fg_mask: bool of [1, 64] with N_fg True values.
    # Apply mask, permute features to the last dim. [1, 1280, 64] => [1, 64, 1280] => [N_fg, 1280]
    ss_fg_feat =  ss_feat.permute(0, 2, 1)[:, fg_mask_N]
    # ms_fg_feat, mc_recon_ms_fg_feat = ... ms_feat, mc_recon_ms_feat
    
    ss_fg_feat_gs = single_feat_grad_scaler(ss_fg_feat)
    # ms_fg_feat_gs = single_feat_grad_scaler(ms_fg_feat)

    # Span the fg_mask to both H and W dimensions.
    fg_mask_HW = fg_mask.unsqueeze(1) * fg_mask.unsqueeze(2)

    loss_comp_single_map_align = masked_mean((sc_map_ss_prob - mc_map_ms_prob).abs(), fg_mask_HW)
    # single_grad_scale = 0.1: 0.1 gs on subj single / mix single features.
    # single features are still updated (although more slowly), to reduce the chance of 
    # generating single images without facial details.
    loss_sc_ss_fg_match = calc_ref_cosine_loss(sc_recon_ss_fg_feat, ss_fg_feat_gs, 
                                                exponent=2, do_demean_first=False,
                                                first_n_dims_to_flatten=2, ref_grad_scale=1)
    #loss_mc_ms_fg_match = calc_ref_cosine_loss(mc_recon_ms_fg_feat, ms_fg_feat_gs, 
    #                                            exponent=2, do_demean_first=False,
    #                                            first_n_dims_to_flatten=2, ref_grad_scale=1)
        
    # fg_mask: [1, 64] => [1, 64, 1].
    fg_mask = fg_mask.float().unsqueeze(2)
    # sc_map_ss_fg_prob: [1, 64, 64] * [1, 64, 1] => [1, 64, 1] => [1, 1, 64].
    sc_map_ss_fg_prob = torch.matmul(sc_map_ss_prob, fg_mask).permute(0, 2, 1)
    mc_map_ms_fg_prob = torch.matmul(mc_map_ms_prob, fg_mask).permute(0, 2, 1)

    # sc_map_ss_fg_prob, mc_map_ms_fg_prob: [1, 1, 64].
    # The total prob of each image token in the subj comp instance maps to fg areas 
    # in the subj single instance. 
    # If this prob is low, i.e., the image token doesn't match to any tokens in the fg areas 
    # in the subj single instance, then this token is probably background.
    # So sc_whole_ss_map_prob.mean(dim=2) is always 1.
    sc_map_ss_fg_prob_below_mean = fg_bg_cutoff_prob - sc_map_ss_fg_prob
    mc_map_ss_fg_prob_below_mean = fg_bg_cutoff_prob - mc_map_ms_fg_prob
    # Remove large negative values (corresponding to large positive probs in 
    # sc_ss_map_prob, mc_ms_map_prob at fg areas of the corresponding single instances),
    # which are likely to be foreground areas. 
    sc_map_ss_fg_prob_below_mean = torch.clamp(sc_map_ss_fg_prob_below_mean, min=0)
    mc_map_ss_fg_prob_below_mean = torch.clamp(mc_map_ss_fg_prob_below_mean, min=0)

    # Image tokens that don't map to any fg image tokens in subj single instance
    # (i.e., corresponding entries in sc_map_ss_fg_prob are small)
    # are considered as bg image tokens.
    # Since sc_map_ss_prob and mc_map_ms_prob are very close to each other (error is 1e-5),
    # we use sc_bg_prob as mc_bg_prob.
    # Note sc_bg_prob is a soft mask, not a hard mask.
    # sc_bg_prob: [1, 1, 64]. Can be viewed as a token-wise weight, used
    # to give CA layer output features different weights at different tokens.

    # sc_mc_bg_feat_diff: [1, 1280, 64] * [1, 1, 64] => [1, 1280, 64]
    # sc_mc_bg_feat_diff = (sc_feat - mc_feat_gs) * comp_bg_prob
    #loss_sc_mc_bg_match = power_loss(sc_mc_bg_feat_diff, exponent=2)
    
    # sc_feat, mc_feat: [1, 1280, 64] => [1, 64, 1280].
    sc_feat = sc_feat.permute(0, 2, 1)
    mc_feat = mc_feat.permute(0, 2, 1)
    # comp_bg_prob: [1, 1, 64] => [1, 64, 1].
    comp_bg_prob = mc_map_ss_fg_prob_below_mean.permute(0, 2, 1)

    loss_sc_mc_bg_match = calc_ref_cosine_loss(sc_feat, mc_feat, 
                                               emb_mask=comp_bg_prob,
                                               exponent=2, do_demean_first=False,
                                               first_n_dims_to_flatten=2, 
                                               ref_grad_scale=mix_feat_grad_scale)
    
    return loss_comp_single_map_align, loss_sc_ss_fg_match, \
           loss_sc_mc_bg_match, sc_map_ss_fg_prob_below_mean, mc_map_ss_fg_prob_below_mean
            # loss_mc_ms_fg_match, 
