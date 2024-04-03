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
import os
import glob
import hashlib
from collections import OrderedDict, defaultdict
import re
import copy

parser = argparse.ArgumentParser(description='PyTorch Checkpoint Averager')
parser.add_argument('--input', default='', type=str, metavar='PATH',
                    help='path to base input folder containing checkpoints')
parser.add_argument('--filter', default='*0.pt', type=str, metavar='WILDCARD',
                    help='checkpoint filter (path wildcard)')
parser.add_argument('--min', type=int, default=500, help='Minimal iteration of checkpoints to average')
parser.add_argument('--max', type=int, default=-1,  help='Maximum iteration of checkpoints to average')



def main():
    args = parser.parse_args()
    pattern = args.input
    if not args.input.endswith(os.path.sep) and not args.filter.startswith(os.path.sep):
        pattern += os.path.sep
    pattern += args.filter
    checkpoint_filenames = glob.glob(pattern, recursive=True)
    checkpoint_filenames = filter(lambda x: int(re.search(r"(\d+).pt", x).group(1)) >= args.min, checkpoint_filenames)
    if args.max > 0:
        checkpoint_filenames = filter(lambda x: int(re.search(r"(\d+).pt", x).group(1)) <= args.max, checkpoint_filenames)
    checkpoint_filenames = filter(lambda x: 'avg_' not in x, checkpoint_filenames)
    checkpoint_filenames = sorted(checkpoint_filenames, key=lambda x: int(re.search(r"(\d+).pt", x).group(1)))
    if len(checkpoint_filenames) == 0:
        print("Error: No checkpoints matching '{}' and iteration >= {} in '{}'".format(
                args.filter, args.min, args.input))
        return

    print("Selected checkpoints:")
    [print(c) for c in checkpoint_filenames]

    avg_ckpt = {}
    avg_counts = 0
    for c in checkpoint_filenames:
        checkpoint = torch.load(c, map_location='cpu')
        for k in checkpoint:
            if k not in avg_ckpt:
                avg_ckpt[k] = checkpoint[k]
            elif isinstance(checkpoint[k], nn.Module):
                #print(f"nn.Module: {k}")
                avg_state_dict = avg_ckpt[k].state_dict()
                param_state_dict = checkpoint[k]
                for m_k, m_v in param_state_dict.items():
                    if m_k not in avg_state_dict:
                        avg_state_dict[m_k] = copy.copy(m_v)
                    else:
                        avg_state_dict[m_k].data += m_v
            elif isinstance(checkpoint[k], nn.Parameter):
                #print(f"nn.Parameter: {k}")
                avg_ckpt[k].data += checkpoint[k].data

        avg_counts += 1
    
    for k in avg_ckpt:
        if isinstance(avg_ckpt[k], nn.Parameter):
            print(f"nn.Parameter: {k}")
            avg_ckpt[k].data /= avg_counts
        elif isinstance(avg_ckpt[k], nn.Module):
            print(f"nn.Module: {k}")
            avg_state_dict = avg_ckpt[k].state_dict()
            for m_k, m_v in avg_state_dict.items():
                m_v.data = (m_v.data / avg_counts).to(m_v.data.dtype)
        else:
            print(f"{type(avg_ckpt[k])}: {k}")

    output_filename = os.path.join(args.input, f"embeddings_gs-avg_{args.min}.pt")
    torch.save(avg_ckpt, output_filename)

    print("=> Saved state_dict to '{}'".format(output_filename))


if __name__ == '__main__':
    main()
