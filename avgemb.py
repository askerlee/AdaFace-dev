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
parser.add_argument('--output', default='averaged.pt', type=str, metavar='PATH',
                    help='output filename')
parser.add_argument('--min', type=int, help='Minimal iteration of checkpoints to average')


def main():
    args = parser.parse_args()
    pattern = args.input
    if not args.input.endswith(os.path.sep) and not args.filter.startswith(os.path.sep):
        pattern += os.path.sep
    pattern += args.filter
    checkpoint_filenames = glob.glob(pattern, recursive=True)
    checkpoint_filenames = filter(lambda x: int(re.search(r"(\d+).pt", x).group(1)) >= args.min, checkpoint_filenames)
    checkpoint_filenames = sorted(checkpoint_filenames, key=lambda x: int(re.search(r"(\d+).pt", x).group(1)))
    avg_checkpoint_filenames = checkpoint_filenames
    if len(avg_checkpoint_filenames) == 0:
        print("Error: No checkpoints matching '{}' and iteration >= {} in '{}'".format(
                args.filter, args.min, args.input))
        return

    print("Selected checkpoints:")
    [print(c) for c in checkpoint_filenames]

    avg_state_dict = {}
    avg_counts = {}
    for c in avg_checkpoint_filenames:
        checkpoint = torch.load(c, map_location='cpu')
        new_state_dict = checkpoint['string_to_param']
        if not new_state_dict:
            print("Error: Checkpoint ({}) doesn't exist".format(args.checkpoint))
            continue

        for k, v in new_state_dict.items():
            if k not in avg_state_dict:
                avg_state_dict[k] = copy.copy(v)
                avg_counts[k] = 1
            else:
                avg_state_dict[k].data = v
                avg_counts[k] += 1

    for k, v in avg_state_dict.items():
        v.data.div_(avg_counts[k])

    # float32 overflow seems unlikely based on weights seen to date, but who knows
    float32_info = torch.finfo(torch.float32)
    final_state_dict = {}
    for k, v in avg_state_dict.items():
        v = v.clamp(float32_info.min, float32_info.max)
        final_state_dict[k] = v.to(dtype=torch.float32)

    complete_state_dict = { 'string_to_token': checkpoint['string_to_token'],
                            'string_to_param': torch.nn.ParameterDict(final_state_dict) }

    output_filename = os.path.join(args.input, f"avg_{args.min}.pt")
    try:
        torch.save(complete_state_dict, output_filename, _use_new_zipfile_serialization=False)
    except:
        torch.save(complete_state_dict, output_filename)

    print("=> Saved state_dict to '{}'".format(output_filename))


if __name__ == '__main__':
    main()
