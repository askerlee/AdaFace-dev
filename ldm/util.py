import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, draw_bounding_boxes
import random

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

# a, b are n-dimensional tensors.
# Orthogonal subtraction of b from a: the result of a-w*b is orthogonal to b (on the last dimension).
def ortho_subtract(a, b):
    assert a.shape == b.shape, "Tensors a and b must have the same shape"
    dot_a_b = torch.einsum('...i,...i->...', a, b)
    dot_b_b = torch.einsum('...i,...i->...', b, b)
    w_optimal = dot_a_b / dot_b_b
    return a - b * w_optimal.unsqueeze(-1)

def demean(x):
    return x - x.mean(dim=-1, keepdim=True)

# Eq.(2) in the StyleGAN-NADA paper.
# delta, ref_delta: [2, 16, 77, 768].
# emb_mask: [2, 77, 1]. Could be fractional, e.g., 0.5, to discount some tokens.
# ref_grad_scale = 0: no gradient will be BP-ed to the reference embedding.
def calc_delta_loss(delta, ref_delta, batch_mask=None, emb_mask=None, 
                    exponent=3, do_demean_first=True, repair_ref_bound_zeros=False,
                    first_n_dims_to_flatten=3,
                    ref_grad_scale=0, aim_to_align=True, debug=False):
    B = delta.shape[0]
    loss = 0
    if batch_mask is not None:
        assert batch_mask.shape == (B,)
        # All instances are not counted. So return 0.
        if batch_mask.sum() == 0:
            return 0
    else:
        batch_mask = torch.ones(B)

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
                # truncate_mask is squeezed to 1D, so that it can be used to index the
                # 4D tensor delta_i, ref_delta_i, emb_mask_i. 
                truncate_mask = (emb_mask_i > 0).squeeze()
                delta_i       = delta_i[:, :, truncate_mask]
                ref_delta_i   = ref_delta_i[:, :, truncate_mask]
                # Make emb_mask_i have the same shape as delta_i without the last (embedding) dimension.
                emb_mask_i    = emb_mask_i[:, :, truncate_mask, 0].expand(delta_i.shape[:-1])
            except:
                breakpoint()

        # Flatten delta and ref_delta, by tucking the layer and token dimensions into the batch dimension.
        # delta_i: [2464, 768], ref_delta_i: [2464, 768]
        delta_i     = delta_i.view(delta_i.shape[:first_n_dims_to_flatten].numel(), -1)
        ref_delta_i = ref_delta_i.view(delta_i.shape)
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
        # If do demean, then no need to repair_ref_bound_zeros. 
        # Since after demean, boundary zeros have been converted to negative values (for lower bound zeros) 
        # or positive values (for upper bound zeros).
        elif repair_ref_bound_zeros:
            min_value, max_value = ref_delta_i.min(), ref_delta_i.max()
            if min_value.abs() < 1e-6 or max_value.abs() < 1e-6:
                ref_delta_i2 = ref_delta_i.clone()
                # zero_to_nonzero_scale: convert zero to the scale of the average of non-zero values.
                zero_to_nonzero_scale = 0.03
                # 0 is either the lower bound or the upper bound, i.e., 
                # non-zero values are either all positive or all negative.
                if min_value.abs() < 1e-6 and (ref_delta_i > min_value).any():
                    # 0 is the lower bound. non-zero values are all positive.
                    # Convert to a small (relative to the magnitude of positive elements) negative value.
                    # We don't need a larger negative value, since the purpose is to let the cosine loss
                    # push the corresponding elements in delta_i to the negative direction. 
                    # If these values are too negative, the gradients will be too large and the delta loss
                    # pushes too aggressively to the negative direction, similar to what demean leads to. 
                    # Demean on masks has been verified to help composition but hurts subject authenticity too much.
                    # Probably because demean is too aggressive towards the negative direction.
                    # If ref_delta is a segmentation mask, then the mean is 1, and RHS is -0.03.
                    ref_delta_i2[ref_delta_i == min_value] = ref_delta_i[ref_delta_i > min_value].mean() \
                                                             * -zero_to_nonzero_scale
                # max_value.abs() < 1e-6. non-zero values are all negative.
                elif max_value.abs() < 1e-6 and (ref_delta_i < max_value).any():
                    # Convert to a small (relative to the magnitude of negative elements) positive value.
                    # mean() is negative, so RHS is positive.
                    ref_delta_i2[ref_delta_i == max_value] = ref_delta_i[ref_delta_i < max_value].mean() \
                                                             * -zero_to_nonzero_scale
            else:
                ref_delta_i2 = ref_delta_i
        else:
            ref_delta_i2 = ref_delta_i

        # x * x.abs.pow(exponent - 1) will keep the sign of x after pow(exponent).
        if ref_grad_scale == 0:
            ref_delta_i2 = ref_delta_i2.detach()
        else:
            grad_scaler = GradientScaler(ref_grad_scale)
            ref_delta_i2 = grad_scaler(ref_delta_i2)

        ref_delta_i_pow = ref_delta_i2 * ref_delta_i2.abs().pow(exponent - 1)

        # If not aim_to_align, then cosine_label = -1, i.e., the cosine loss will 
        # push delta_i to be orthogonal with ref_delta_i.
        cosine_label = 1 if aim_to_align else -1
        loss_i = F.cosine_embedding_loss(delta_i, ref_delta_i_pow, 
                                         torch.ones_like(delta_i[:, 0]) * cosine_label, 
                                         reduction='none')
        if emb_mask_i is not None:
            loss_i = (loss_i * emb_mask_i).sum() / emb_mask_i.sum()
        else:
            loss_i = loss_i.mean()

        loss_i = loss_i * batch_mask[i]

        loss += loss_i

    loss /= batch_mask.sum()
    return loss

def calc_stats(ts, ts_name=None):
    if ts_name is not None:
        print("%s: " %ts_name, end='')
    print("max: %.4f, min: %.4f, mean: %.4f, std: %.4f" %(ts.max(), ts.min(), ts.mean(), ts.std()))

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

# infeat_size: (h, w) of the input feature map (before flattening).
# H: number of heads.
def replace_rows_by_conv_attn(attn_mat, q, k, subj_indices, infeat_size, H, sim_scale):
    # input features x: [4, 4096, 320].
    # attn_mat: [32, 4096, 77]. 32: b * h. b = 4, h = 8.
    # q: [32, 4096, 40]. k: [32, 77, 40]. 32: b * h.
    # subj_indices: [0, 0, 0, 0, 1, 1, 1, 1], [6, 7, 8, 9, 6, 7, 8, 9].
    # prompts: 'a face portrait of a z, , ,  swimming in the ocean, with backlight', 
    #          'a face portrait of a z, , ,  swimming in the ocean, with backlight', 
    #          'a face portrait of a cat, , ,  swimming in the ocean, with backlight', 
    #          'a face portrait of a cat, , ,  swimming in the ocean, with backlight'
    # The first two prompts are identical, since this is a teacher filter iter.

    # attn_mat_shape: [32, 4096, 77]
    attn_mat_shape = attn_mat.shape

    indices_B, indices_N = subj_indices
    indices_B_uniq = torch.unique(indices_B)
    # BS: sub-batch size that contains the subject token. 
    # Probably BS < the full batch size.
    BS = len(indices_B_uniq)
    # M: number of embeddings for each subject token.
    M  = len(indices_N) // BS

    # attn_mat: [32, 4096, 77] => [4, 8, 4096, 77].
    attn_mat = attn_mat.reshape(-1, H, *attn_mat.shape[1:])
    # q: [32, 4096, 40] => [4, 8, 4096, 40].
    q = q.reshape(-1, H, *q.shape[1:])
    # k: [32, 77, 40] => [4, 8, 77, 40].
    k = k.reshape(-1, H, *k.shape[1:])
    # C: number of channels in the embeddings.
    C = q.shape[-1]
    # ks: conv kernel size.
    ks = int(np.sqrt(M))
    # if ks == 2, pad 1 pixel on the right and bottom.
    # pads: left, right, top, bottom.
    if ks == 2:
        pads = (0, 1, 0, 1)
    # if ks == 3, pad 1 pixel on each side.
    else:
        pads = (1, 1, 1, 1)
    padder = nn.ZeroPad2d(pads)

    # Traversed delta x: {0, 1} (ks=2) or {-1, 0, 1} (ks=3)
    # Add 1 to the upper bound to make the range inclusive.
    delta_x_bound = (-pads[0], pads[1] + 1)
    # Traversed delta y: {0, 1} (ks=2) or {-1, 0, 1} (ks=3).
    # Add 1 to the upper bound to make the range inclusive.
    delta_y_bound = (-pads[2], pads[3] + 1)

    # Clone to make attn_mat2 a non-leaf node. Otherwise, 
    # we can't do in-place assignment like attn_mat[indices_b, :, :, indices_n] = subj_attn_dxys.
    attn_mat2 = attn_mat.clone()

    for b in range(BS):
        subj_attn_dxys = []
        index_b = indices_B_uniq[b]
        # subj_q: [8, 4096, 40].
        subj_q = q[index_b]
        # subj_q_2d: [8, 4096, 40] => [8, 40, 4096] => [1, 320, 64, 64]
        subj_q_2d = subj_q.permute(0, 2, 1).reshape(1, H * C, *infeat_size)
        # subj_q_padded: [1, 320, 65, 65] (ks=2) or [1, 320, 66, 66] (ks=3).
        subj_q_padded = padder(subj_q_2d)

        # indices_b: [0, 0, 0, 0]. indices_n: [6, 7, 8, 9].
        indices_b = indices_B[b * M : b * M + M]
        indices_n = indices_N[b * M : b * M + M]
        if (indices_b != index_b).any():
            breakpoint()
        # subj_k: k[[0], :, [6,7,8,9]], shape [4, 8, 40] => [8, 40, 4], [H, C, M].
        # select the 4 subject embeddings (s1, s2, s3, s4) from k into subj_k.
        subj_k = k[indices_b, :, indices_n].permute(1, 2, 0)

        # subj_k -> conv weight: [8, 40, 2, 2]. First 2 is height (y), second 2 is width (x).
        # The input channel number is 40 * 8 = 320.
        # But due to grouping, the weight shape is [8, 40, 2, 2] instead of [8, 320, 2, 2].
        # Each output channel belongs to an individual group. 
        # The shape of the weight 40*8 should be viewed as 8 groups of 40*1.
        # subj_k: [8, 4, 40] => [8, 40, 4] => [8, 40, 2, 2].
        # So the 4 embeddings (s1, s2, s3, s4) in subj_k are arranged as 
        #                |  (s1 s2
        #              H |   s3 s4)
        #                    _____ W
        # subj_attn: [1, 8, 64, 64]
        # Note to scale attention scores by sim_scale, and divide by M.
        # sim_scale is to keep consistent to the original cross attention scores.
        # Divide by M so that the attention scores are evenly distributed to the M embeddings.
        subj_attn = F.conv2d(subj_q_padded, subj_k.reshape(H, C, ks, ks), groups=H) * sim_scale / M
        # Shift subj_attn (with 0 padding) to yield ks*ks slightly different attention maps 
        # for the M embeddings.
        # dx, dy: the relative position of a subject token to the center subject token.
        # dx: shift along rows    (width,  the last dim). 
        # dy: shift along columns (height, the second-last dim).
        # s1, s2, s3, s4 => s1 s2
        #                   s3 s4
        # Since the first  loop is over dy (vertical move,   or move within a column), 
        # and   the second loop is over dx (horizontal move, or move within a row), 
        # the traversed order is s1, s2, s3, s4.
        # NOTE: This order should be consistent with how subj_k is reshaped to the conv weight. 
        # Otherwise wrong attn maps will be assigend to some of the subject embeddings.
        for dy in range(*delta_y_bound):
            for dx in range(*delta_x_bound):
                if dx == 0 and dy == 0:
                    subj_attn_dxy = subj_attn
                elif dx <= 0 and dy <= 0:
                    subj_attn_dxy = F.pad(subj_attn[:, :, -dy:, -dx:], (0, -dx, 0, -dy))
                elif dx <= 0 and dy > 0:
                    subj_attn_dxy = F.pad(subj_attn[:, :, :-dy, -dx:], (0, -dx, dy, 0))
                elif dx > 0 and dy <= 0:
                    subj_attn_dxy = F.pad(subj_attn[:, :, -dy:, :-dx], (dx, 0,  0, -dy))
                else:
                    # dx > 0 and dy > 0
                    subj_attn_dxy = F.pad(subj_attn[:, :, :-dy, :-dx], (dx, 0,  dy, 0))

                # subj_attn_dxy: [1, 8, 64, 64].
                # Since first loop is over dy (each loop forms a column in the feature maps), 
                # and   second loop   over dx (each loop forms a row    in the feature maps), 
                # the order in subj_attn_dxys is s1, s2, s3, s4.
                subj_attn_dxys.append(subj_attn_dxy)

        # dx, dy: the relative position of a subject token to the center subject token.
        # s1, s2, s3, s4 => s1 s2
        #                   s3 s4
        # The traversed order is s1, s2, s3, s4. So we can flatten the list to get 
        # consecutive 4 columns of attention. 
        # subj_attn_dxys: [4, 8, 64, 64] => [4, 8, 4096].
        subj_attn_dxys = torch.cat(subj_attn_dxys, dim=0).reshape(M, H, -1)
        # attn_mat2: [4, 8, 4096, 77], [B, H, N, T]. 
        # B: the whole batch (>=BS). H: number of heads. 
        # N: number of visual tokens. T: number of text tokens.
        # attn_mat2[[0], :, :, [6,7,8,9]]: [4, 8, 4096]
        attn_mat2[indices_b, :, :, indices_n] = subj_attn_dxys

    # attn_mat2: [4, 8, 4096, 77] => [32, 4096, 77].
    return attn_mat2.reshape(attn_mat_shape)

def patch_multi_embeddings(text_embedding, placeholder_indices_N, divide_scheme='sqrt_M'):
    if placeholder_indices_N is None:
        return text_embedding
    
    # In a do_teacher_filter iteration, placeholder_indices_N may consist of the indices of 2 instances.
    # So we need to deduplicate them first.
    placeholder_indices_N = torch.unique(placeholder_indices_N)
    M = len(placeholder_indices_N)
    # num_vectors_per_token = 1. No patching is needed.
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

def scale_emb_in_embs(text_embedding, placeholder_indices_N, scale, scale_first_only=True):
    scale_mask = torch.ones_like(text_embedding)
    if scale_first_only:
        # Only scale the first embedding in a multi-embedding token.
        placeholder_indices_N = placeholder_indices_N[:1]
    scale_mask[:, placeholder_indices_N] = scale
    scaled_text_embedding = text_embedding * scale_mask
    return scaled_text_embedding

# Revised from RevGrad, by removing the grad negation.
class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * alpha_
        return grad_input, None

class GradientScaler(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return ScaleGrad.apply(input_, self._alpha.to(input_.device))

def gen_gradient_scaler(alpha):
    if alpha == 1:
        return lambda x: x
    if alpha > 0:
        return GradientScaler(alpha)
    else:
        return lambda x: x.detach()
    
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

def halve_token_indices(token_indices):
    token_indices_half_B  = token_indices[0].chunk(2)[0]
    token_indices_half_N  = token_indices[1].chunk(2)[0]
    return (token_indices_half_B, token_indices_half_N)

def normalize_dict_values(d):
    value_sum = np.sum(list(d.values()))
    # If d is empty, do nothing.
    if value_sum == 0:
        return d
    
    d2 = { k: v / value_sum for k, v in d.items() }
    return d2

def masked_mean(ts, mask, dim=None):
    if mask is None:
        return ts.mean()
    
    mask_sum = mask.sum(dim=dim)
    mask_sum = torch.maximum( mask_sum, torch.ones_like(mask_sum) * 1e-6 )
    return (ts * mask).sum(dim=dim) / mask_sum

# true_prob_range = (p_init, p_final). 
# The prob of flipping true is gradually annealed from p_init to p_final.
def flip_coin_annealed(training_percent, final_percent, true_prob_range):
    assert 0 - 1e-6 <= training_percent <= 1 + 1e-6
    p_init, p_final = true_prob_range
    # Gradually decrease the chance of flipping from the upperbound to lowerbound.
    if training_percent < final_percent:
        true_p_annealed = p_init + (p_final - p_init) * training_percent
    else:
        # Stop at p_final.
        true_p_annealed = p_final

    # Flip a coin, with prob of true being true_p_annealed.    
    return random.random() < true_p_annealed

def anneal_t(t, training_percent, num_timesteps, ratio_range, keep_prob_range=(0, 0.5)):
    t_anneal = t.clone()
    # Gradually increase the chance of keeping the original t from 0 to 0.5.
    do_keep = flip_coin_annealed(training_percent, final_percent=1., true_prob_range=keep_prob_range)
    if do_keep:
        return t_anneal
    
    ratio_lb, ratio_ub = ratio_range
    assert ratio_lb < ratio_ub

    for i, ti in enumerate(t):
        ti_lowerbound = max(int(ti * ratio_lb), 0)
        ti_upperbound = min(int(ti * ratio_ub) + 1, num_timesteps)
        t_anneal[i] = np.random.randint(ti_lowerbound, ti_upperbound)

# feat_or_attn: 4D features or 3D attention. If it's attention, then
# its geometrical dimensions (H, W) have been flatten to 1D (last dim).
# mask:      always 4D.
# mode: either "nearest" or "nearest|bilinear". Other modes will be ignored.
def scale_mask_for_feat_attn(feat_or_attn, mask, mask_name, mode="nearest|bilinear", warn_on_all_zero=True):
    if feat_or_attn.ndim == 3:
        spatial_scale = np.sqrt(feat_or_attn.shape[-1] / mask.shape[2:].numel())
    elif feat_or_attn.ndim == 4:
        spatial_scale = np.sqrt(feat_or_attn.shape[-2:].numel() / mask.shape[2:].numel())
    else:
        breakpoint()

    spatial_shape2 = (int(mask.shape[2] * spatial_scale), int(mask.shape[3] * spatial_scale))

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
        if isinstance(c1_mix_scale, torch.Tensor)       and (c1_mix_scale == 1).all():
            return c1
        # c1_mix_scale = 1. No need to mix.
        elif not isinstance(c1_mix_scale, torch.Tensor) and c1_mix_scale == 1:
            return c1

        if mix_indices is not None:
            scale_mask = torch.ones_like(c1)

            if type(c1_mix_scale) == torch.Tensor:
                # c1_mix_scale is only for one instance. Repeat it for all instances in the batch.
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
        else:
            # Mix the whole sequence.
            c_mix = c1 * c1_mix_scale + c2 * (1 - c1_mix_scale)

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

# t_frac is a float scalar. 
def mix_static_qv_embeddings(c_static_emb, subj_indices_half_N, 
                             t_frac=1.0,
                             use_layerwise_embedding=True,
                             N_LAYERS=16, 
                             LAYERS_CLS_E_SCALE_RANGE=(0.8, 0.2),
                             sync_layer_indices=[4, 5, 6, 7, 8, 9, 10]
                             ):
    
    subj_emb, cls_emb = c_static_emb.chunk(2)
    BS = subj_emb.shape[0] // N_LAYERS
    if not isinstance(t_frac, torch.Tensor):
        t_frac = torch.tensor(t_frac, dtype=c_static_emb.dtype, device=c_static_emb.device)
    if len(t_frac) == 1:
        t_frac = t_frac.repeat(BS)
    t_frac = t_frac.unsqueeze(1)

    if len(t_frac) != BS:
        breakpoint()
        
    FIRST_LAYER_CLS_E_SCALE, FINAL_LAYER_CLS_E_SCALE = LAYERS_CLS_E_SCALE_RANGE

    if use_layerwise_embedding:
        SCALE_STEP = (FINAL_LAYER_CLS_E_SCALE - FIRST_LAYER_CLS_E_SCALE) / (len(sync_layer_indices) - 1)
        # Linearly decrease the scale of the class   embeddings from 0.5 to 0.1, 
        # i.e., 
        # Linearly increase the scale of the subject embeddings from 0.5 to 0.9.
        # emb_v_layers_subj_mix_scales = [1.0, 1.0, 1.0, 1.0, 0.3, 0.4, 0.5, 0.6, 
        #                                 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0000]
        emb_v_layers_subj_mix_scales = torch.ones(BS, N_LAYERS, device=c_static_emb.device) 
        # Scale the class embeddings mix scale by t_frac.
        emb_v_layers_subj_mix_scales[:, sync_layer_indices] = \
            1 - torch.arange(FIRST_LAYER_CLS_E_SCALE, FINAL_LAYER_CLS_E_SCALE + SCALE_STEP, 
                             step=SCALE_STEP, device=c_static_emb.device).repeat(BS, 1) * t_frac
    else:
        # Same scale for all layers.
        # emb_v_layers_subj_mix_scales = [0.5, 0.5, ..., 0.5].
        AVG_SCALE = (FIRST_LAYER_CLS_E_SCALE + FINAL_LAYER_CLS_E_SCALE) / 2
        emb_v_layers_subj_mix_scales = 1 - torch.ones(N_LAYERS, device=c_static_emb.device).repeat(BS, 1) \
                                        * AVG_SCALE * t_frac

    # First mix the static embeddings.
    # mix_embeddings('add', ...):  being subj_comp_emb almost everywhere, except those at subj_indices_half_N,
    # where they are subj_comp_emb * emb_v_layers_subj_mix_scales + cls_comp_emb * (1 - emb_v_layers_subj_mix_scales).
    # subj_comp_emb, cls_comp_emb, subj_single_emb, cls_single_emb: [16, 77, 768].
    # Each is of a single instance. So only provides subj_indices_half_N 
    # (multiple token indices of the same instance).
    emb_v_mixer = partial(mix_embeddings, 'add', mix_indices=subj_indices_half_N)

    # Part of cls embedding is mixed into subject v embedding.
    mix_emb_v = emb_v_mixer(subj_emb, cls_emb, c1_mix_scale=emb_v_layers_subj_mix_scales.view(-1))
    # The first  half of mix_emb_all_layers will be used as V in cross attention layers.
    # The second half of mix_emb_all_layers will be used as K in cross attention layers.
    mix_emb_all_layers = torch.cat([mix_emb_v, cls_emb], dim=1)

    PROMPT_MIX_GRAD_SCALE = 0.05
    grad_scaler = gen_gradient_scaler(PROMPT_MIX_GRAD_SCALE)
    # mix_comp_emb receives smaller grad, since it only serves as the reference.
    # If we don't scale gradient on mix_comp_emb, chance is mix_comp_emb might be 
    # dominated by subj_comp_emb,
    # so that mix_comp_emb will produce images similar as subj_comp_emb does.
    # Scaling the gradient will improve compositionality but reduce face similarity.
    mix_emb_all_layers   = grad_scaler(mix_emb_all_layers)

    if use_layerwise_embedding:
        # sync_layer_indices = [4, 5, 6, 7, 8, 9, 10] #, 11, 12, 13]
        # 4, 5, 6, 7, 8, 9, 10 correspond to original layer indices 7, 8, 12, 16, 17, 18, 19.
        # (same as used in computing mixing loss)
        # layer_mask: [2, 16, 154, 768]
        layer_mask = torch.zeros_like(mix_emb_all_layers).reshape(-1, N_LAYERS, *mix_emb_all_layers.shape[1:])
        # t_frac controls how much mix_emb_all_layers is mixed with subj_comp_emb2 into mix_comp_emb,
        # and how much mix_single_emb_all_layers is mixed with subj_single_emb2 into mix_single_emb.

        # layer_mask[:, sync_layer_indices]: [2, 7, 154, 768]
        layer_mask[:, sync_layer_indices] = t_frac.view(-1, 1, 1, 1)
        layer_mask = layer_mask.reshape(-1, *mix_emb_all_layers.shape[1:])

        # This copy of subj_emb will be simply 
        # repeated at the token dimension to match the token number of the mixed (concatenated) 
        # mix_emb embeddings.
        subj_emb2   = subj_emb.repeat(1, 2, 1)

        # Use most of the layers of embeddings in subj_comp_emb2, but 
        # replace sync_layer_indices layers with those from mix_emb_all_layers.
        # Do not assign with sync_layers as indices, which destroys the computation graph.
        mix_emb   = subj_emb2 * (1 - layer_mask) \
                    + mix_emb_all_layers * layer_mask
        
    else:
        # There is only one layer of embeddings.
        mix_emb   = mix_emb_all_layers

    # c_static_emb2 is the static embeddings of the prompts used in losses other than 
    # the static delta loss, e.g., used to estimate the ada embeddings.
    # If use_ada_embedding, then c_in2 will be fed again to CLIP text encoder to 
    # get the ada embeddings. Otherwise, c_in2 will be useless and ignored.
    # c_static_emb2: [64, 154, 768]
    # c_static_emb2 will be added with the ada embeddings to form the 
    # conditioning embeddings in the U-Net.
    # Unmixed embeddings and mixed embeddings will be merged in one batch for guiding
    # image generation and computing compositional mix loss.
    c_static_emb2 = torch.cat([ subj_emb2, mix_emb ], dim=0)

    # emb_v_mixer will be used later to mix ada embeddings in UNet.
    # extra_info['emb_v_mixer']                   = emb_v_mixer
    # extra_info['emb_v_layers_subj_mix_scales']  = emb_v_layers_subj_mix_scales
    
    return c_static_emb2, emb_v_mixer, emb_v_layers_subj_mix_scales
