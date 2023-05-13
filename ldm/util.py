import importlib

import torch
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
# Orthogonal subtraction of b from a: the result of a-w*b is orthogonal to b.
def ortho_subtract(a, b):
    assert a.shape == b.shape, "Tensors a and b must have the same shape"
    dot_a_b = torch.einsum('...i,...i->...', a, b)
    dot_b_b = torch.einsum('...i,...i->...', b, b)
    w_optimal = dot_a_b / dot_b_b
    return a - b * w_optimal.unsqueeze(-1)

# c1, c2: [32, 77, 768].
# mix_scheme: 'add', 'concat', 'sdeltaconcat', 'adeltaconcat'.
def mix_embeddings(c1, c2, c2_mix_weight, mix_scheme='adeltaconcat', placeholder_indices=None,
                   use_ortho_subtract=True):
    assert c1 is not None
    if c2 is None:
        return c1

    c1_weight = 1

    if mix_scheme == 'add':
        c_mix = c1 * c1_weight + c2 * c2_mix_weight
    elif mix_scheme == 'concat':
        c_mix = torch.cat([ c1 * c1_weight, c2 * c2_mix_weight ], dim=1)
    # sdeltaconcat: subject-delta concat. Requires placeholder_indices.
    elif mix_scheme == 'sdeltaconcat':
        assert placeholder_indices is not None
        # delta_embedding is the difference between the subject embedding and the class embedding.
        if use_ortho_subtract:
            delta_embedding = ortho_subtract(c2, c1)
        else:
            delta_embedding = c2 - c1
            
        delta_embedding = delta_embedding[placeholder_indices]
        assert delta_embedding.shape[0] == c1.shape[0]

        c2_delta = c1.clone()
        # c2_mix_weight only boosts the delta embedding, and other tokens in c2 always have weight 1.
        c2_delta[placeholder_indices] = delta_embedding
        c_mix = torch.cat([ c1 * c1_weight, c2_delta * c2_mix_weight ], dim=1)

    # adeltaconcat: all-delta concat.
    elif mix_scheme == 'adeltaconcat':
        # delta_embedding is the difference between all the subject tokens and the class tokens.
        if use_ortho_subtract:
            delta_embedding = ortho_subtract(c2, c1)
        else:
            delta_embedding = c2 - c1
            
        # c2_mix_weight scales all tokens in delta_embedding.
        c_mix = torch.cat([ c1 * c1_weight, delta_embedding * c2_mix_weight ], dim=1)

    return c_mix

def demean(x):
    return x - x.mean(dim=-1, keepdim=True)

# Eq.(2) in the StyleGAN-NADA paper.
# delta, ref_delta: [2, 16, 77, 768].
# emb_mask: [2, 77, 1]
def calc_delta_loss(delta, ref_delta, emb_mask=None, exponent=3, 
                    do_demean_first=True, 
                    first_n_dims_to_flatten=3):
    B = delta.shape[0]
    loss = 0

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
                emb_mask_i    = emb_mask_i[:, :, truncate_mask]
                delta_i       = delta_i[:, :, truncate_mask]
                ref_delta_i   = ref_delta_i[:, :, truncate_mask]
            except:
                breakpoint()

        # Flatten delta and ref_delta, by tucking the layer and token dimensions into the batch dimension.
        # dela: [2464, 768], ref_delta: [2464, 768]
        delta_i     = delta_i.view(delta_i.shape[:first_n_dims_to_flatten].numel(), -1)
        ref_delta_i = ref_delta_i.view(ref_delta_i.shape[:first_n_dims_to_flatten].numel(), -1)

        # A bias vector to a set of conditioning embeddings doesn't change the attention matrix 
        # (though changes the V tensor). So the bias is better removed.
        # Therefore, do demean() before cosine loss, 
        # to remove the effect of bias.
        # In addition, different ada layers have significantly different scales. 
        # But since cosine is scale invariant, de-scale is not necessary and won't have effects.
        # LN = demean & de-scale. So in theory, LN is equivalent to demean() here. But LN may introduce
        # numerical instability. So we use simple demean() here.
        if do_demean_first:
            delta_i     = demean(delta_i)
            ref_delta_i = demean(ref_delta_i)

        # x * x.abs.pow(exponent - 1) will keep the sign of x after pow(exponent).
        ref_delta_i_pow = ref_delta_i * ref_delta_i.abs().pow(exponent - 1)
        loss_i = F.cosine_embedding_loss(delta_i, ref_delta_i_pow.detach(), 
                                         torch.ones_like(delta_i[:, 0]), 
                                         reduction='none')
        if emb_mask_i is not None:
            loss_i = (loss_i * emb_mask_i).sum() / emb_mask_i.sum()
        else:
            loss_i = loss_i.mean()

        loss += loss_i

    loss /= B
    return loss

def calc_stats(ts, ts_name=None):
    if ts_name is not None:
        print("%s: " %ts_name, end='')
    print("max: %.4f, min: %.4f, mean: %.4f, std: %.4f" %(ts.max(), ts.min(), ts.mean(), ts.std()))
