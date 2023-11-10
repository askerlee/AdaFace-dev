import torch
from evaluation.eval_utils import find_first_match
import sys
import os
import re

ckpt_sig = sys.argv[1]
if len(sys.argv) > 2:
    extra_sig = sys.argv[2]
else:
    extra_sig = ""

all_ckpt_names = os.listdir("logs")
# Sort all_ckpt_names by name (actually by timestamp in the name), so that most recent first.
all_ckpt_names.sort(reverse=True)
ckpt_name  = find_first_match(all_ckpt_names, ckpt_sig, extra_sig=extra_sig)
# embeddings_gs-{ckpt_iter}.pt
emb_folder    = f"logs/{ckpt_name}/checkpoints/"
iter2path = {}
print("emb_folder:", emb_folder)

for emb_path in os.listdir(emb_folder):
    if re.match(r"embeddings_gs-(\d+).pt", emb_path):
        ckpt_iter = re.match(r"embeddings_gs-(\d+).pt", emb_path).group(1)
        ckpt_iter = int(ckpt_iter)
        #if ckpt_iter % 500 == 0:
        iter2path[ckpt_iter] = emb_path

iterations = sorted(iter2path.keys())

for idx, iteration in enumerate(iterations):
    print(iteration)
    emb_path = os.path.join(emb_folder, iter2path[iteration])
    emb_ckpt = torch.load(emb_path)
    print('emb_global_scale_score: {:.4f}'.format(emb_ckpt['emb_global_scale_score'].sigmoid().item() + 0.5))
    for weight_name in ('layerwise_point_conv_attn_mix_weights', 'layerwise_conv_attn_weights'):
        if weight_name in emb_ckpt:
            print('layerwise_point_conv_attn_mix_weights:')
            print(emb_ckpt[weight_name].data)

tokens = emb_ckpt['string_to_emb_ema_dict'].keys()

for k in tokens:
    print(f"Token: {k}")

    for idx, iteration in enumerate(iterations):
        emb_path = os.path.join(emb_folder, iter2path[iteration])
        emb_ckpt = torch.load(emb_path)
        if idx == 0:
            prev_emb_ckpt = emb_ckpt
            continue

        prev_iteration = iterations[idx-1]

        emb_ema = emb_ckpt['string_to_emb_ema_dict'][k]
        prev_emb_ema = prev_emb_ckpt['string_to_emb_ema_dict'][k]
        curr_mean = emb_ema.embedding.abs().mean()
        prev_mean = prev_emb_ema.embedding.abs().mean()
        delta = (emb_ema.embedding - prev_emb_ema.embedding).abs().mean()
        print(f"{prev_iteration} -> {iteration}: {prev_mean:.4f}, {curr_mean:.4f}, {delta:.4f}")

        prev_emb_ckpt = emb_ckpt
