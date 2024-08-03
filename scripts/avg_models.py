#!/usr/bin/env python3
""" Checkpoint Averaging Script

This script averages all model weights for checkpoints in specified path that match
the specified filter wildcard. All checkpoints must be from the exact same model.

For any hope of decent results, the checkpoints should be from the same or child
(via resumes) training session. This can be viewed as similar to maintaining running
EMA (exponential moving average) of the model weights or performing SWA (stochastic
weight averaging), but post-training.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import torch
import torch.nn as nn
import argparse
import glob
import re
import copy
import math
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from scripts.ckpt_lib import load_ckpt

parser = argparse.ArgumentParser(description='PyTorch Checkpoint Averager')
parser.add_argument('--input', nargs="+", type=str, metavar='PATHS', required=True,
                    help='path(s) to base input folder containing checkpoints. The first is the base checkpoint.')
parser.add_argument('--output', type=str, default='avgemb.pt', metavar='PATH',
                    help='output file name of the averaged checkpoint')
parser.add_argument('--suffix', default='', type=str, metavar='WILDCARD',
                    help='checkpoint suffix')
parser.add_argument("--mod_pattern", type=str, default=None, 
                    help="Pattern in the modules to be implanted from the donor checkpoint")
parser.add_argument('--min', type=int, default=500, help='Minimal iteration of checkpoints to average')
parser.add_argument('--max', type=int, default=-1,  help='Maximum iteration of checkpoints to average')

def main():
    args = parser.parse_args()
    ckpt_patterns = args.input
    sel_checkpoint_filenames = []

    for ckpt_pattern in ckpt_patterns:
        if args.suffix is not None:
            if not args.suffix.startswith('*'):
                ckpt_pattern += '*'
            ckpt_pattern += args.suffix

        checkpoint_filenames = glob.glob(ckpt_pattern, recursive=True)
        if len(checkpoint_filenames) == 0:
            print("WARNING: No checkpoints matching '{}' and iteration >= {} in '{}'".format(
                  ckpt_pattern, args.min, args.input))
            continue

        sel_checkpoint_filenames.extend(checkpoint_filenames)

    avg_ckpt = {}
    avg_counts = {}
    for i, ckpt_filename in enumerate(sel_checkpoint_filenames):
        _, state_dict = load_ckpt(ckpt_filename)
        checkpoint = state_dict
        print(ckpt_filename)

        for k in checkpoint:
            # Skip ema weights
            if k.startswith("model_ema."):
                continue
            # First time seeing a module. Copy it from the base checkpoint 
            # (assuming modules names in all checkpoints are the consistent)
            if k not in avg_ckpt:
                avg_ckpt[k] = checkpoint[k]
                print(f"Copy {k}")
                avg_counts[k] = 1
            # Skip accumulating modules which do not match the pattern
            elif (args.mod_pattern is not None) and (not re.search(args.mod_pattern, k)):
                continue
            # A new occurrence of a previously seen nn.Module.
            elif isinstance(checkpoint[k], nn.Module):
                #print(f"nn.Module: {k}")
                avg_state_dict = avg_ckpt[k].state_dict()
                param_state_dict = checkpoint[k]
                for m_k, m_v in param_state_dict.state_dict().items():
                    if m_k not in avg_state_dict:
                        avg_state_dict[m_k] = copy.copy(m_v)
                        print(f"Copy {k}.{m_k}")
                    else:
                        avg_state_dict[m_k].data += m_v
                        print(f"Accumulate {k}.{m_k}")
                avg_ckpt[k].load_state_dict(avg_state_dict)
                avg_counts[k] += 1
            # Another occurrence of a previously seen nn.Parameter.
            elif isinstance(checkpoint[k], (nn.Parameter, torch.Tensor)):
                #print(f"nn.Parameter: {k}")
                avg_ckpt[k].data += checkpoint[k].data.to(avg_ckpt[k].dtype)
                avg_counts[k] += 1
                print(f"Accumulate {k}")
            else:
                print(f"NOT copying {type(checkpoint[k])}: {k}")
                pass
    
    for k in avg_ckpt:
        # safetensors use torch.Tensor instead of nn.Parameter.
        if isinstance(avg_ckpt[k], (nn.Parameter, torch.Tensor)) and avg_counts[k] > 1:
            print(f"Averaging nn.Parameter: {k}")
            try:
                avg_ckpt[k].data /= avg_counts[k]
            except:
                # num_batches_tracked in BatchNorm layers is long type.
                avg_ckpt[k].data //= math.floor(avg_counts[k])

        elif isinstance(avg_ckpt[k], nn.Module) and avg_counts[k] > 1:
            print(f"Averaging nn.Module: {k}")
            avg_state_dict = avg_ckpt[k].state_dict()
            for m_k, m_v in avg_state_dict.items():
                m_v.data = (m_v.data / avg_counts[k]).to(m_v.data.dtype)
            avg_ckpt[k].load_state_dict(avg_state_dict)
        else:
            print(f"NOT averaging {type(avg_ckpt[k])}: {k}")

    if args.output.endswith(".safetensors"):
        safetensors_save_file(avg_ckpt, args.output)
    else:
        torch.save(avg_ckpt, args.output)

    print("=> Saved state_dict to '{}'".format(args.output))


if __name__ == '__main__':
    main()
