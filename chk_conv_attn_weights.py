import torch
from evaluation.eval_utils import find_first_match
import sys
import os
import re

ckpt_sig = sys.argv[1]
all_ckpt_names = os.listdir("logs")
# Sort all_ckpt_names by name (actually by timestamp in the name), so that most recent first.
all_ckpt_names.sort(reverse=True)
ckpt_name  = find_first_match(all_ckpt_names, ckpt_sig)
# embeddings_gs-{ckpt_iter}.pt
emb_folder    = f"logs/{ckpt_name}/checkpoints/"
iter2path = {}
print("emb_folder:", emb_folder)

for emb_path in os.listdir(emb_folder):
    if re.match(r"embeddings_gs-(\d+).pt", emb_path):
        ckpt_iter = re.match(r"embeddings_gs-(\d+).pt", emb_path).group(1)
        iter2path[int(ckpt_iter)] = emb_path

for iterations in sorted(iter2path.keys()):
    print(iterations)
    emb_path = os.path.join(emb_folder, iter2path[iterations])
    emb_ckpt = torch.load(emb_path)
    print(emb_ckpt['layerwise_conv_attn_weights'])
