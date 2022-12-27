import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import glob
import re

emb_ckpt_folder = sys.argv[1]
emb_ckpt_files = glob.glob(emb_ckpt_folder + "/embeddings_gs-*.pt")
emb_ckpt_files = sorted(emb_ckpt_files, key=lambda s:int(re.search(r"(\d+).pt", s).group(1)))

# enumerate files in emb_ckpt_folder
for emb_ckpt_filename in emb_ckpt_files:
    emb_ckpt = torch.load(emb_ckpt_filename, map_location='cpu')
    embeddings = emb_ckpt['string_to_param']['z']
    emb_mean = embeddings.mean(0, keepdim=True).repeat(embeddings.size(0), 1)
    l1_loss = F.l1_loss(embeddings, emb_mean)
    l2_loss = F.mse_loss(embeddings, emb_mean)
    norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
    print("%s: L1: %.3f, L2: %.3f" %(os.path.basename(emb_ckpt_filename), l1_loss.item(), l2_loss.item()))
    print("Norms: min: %.3f, max: %.3f, mean: %.3f, std: %.3f" %(norms.min(), norms.max(), norms.mean(), norms.std()))

