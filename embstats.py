import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.embedding_manager import LoraEmbedding
import sys
import os
import glob
import re
import numpy as np
np.set_printoptions(precision=3, suppress=True)

emb_ckpt_folder = sys.argv[1]
if len(sys.argv) > 2:
    # N: number of placeholder words. 3 is the default.
    N = int(sys.argv[2])
else:
    N = 3

# check if emb_ckpt_folder is a single file or a folder
if os.path.isfile(emb_ckpt_folder):
    emb_ckpt_files = [emb_ckpt_folder]
else:
    emb_ckpt_files = glob.glob(emb_ckpt_folder + "/embeddings_gs-*.pt")
    emb_ckpt_files = sorted(emb_ckpt_files, key=lambda s:int(re.search(r"(\d+).pt", s).group(1)))

def calc_stats(emb_name, embeddings):
    print("%s:" %emb_name)
    emb_mean = embeddings.mean(0, keepdim=True).repeat(embeddings.size(0), 1)
    l1_loss = F.l1_loss(embeddings, emb_mean)
    l2_loss = F.mse_loss(embeddings, emb_mean)
    norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
    print("L1: %.3f, L2: %.3f" %(l1_loss.item(), l2_loss.item()))
    print("Norms: min: %.3f, max: %.3f, mean: %.3f, std: %.3f" %(norms.min(), norms.max(), norms.mean(), norms.std()))


# enumerate files in emb_ckpt_folder
for emb_ckpt_filename in emb_ckpt_files:
    emb_ckpt = torch.load(emb_ckpt_filename, map_location='cpu')
    embeddings = emb_ckpt['string_to_param']['z']
    print("%s:" %os.path.basename(emb_ckpt_filename))
    if isinstance(embeddings, LoraEmbedding):
        print(embeddings.vec_weights.detach().cpu().numpy())
        #print(embeddings.lora_up.detach().cpu().numpy())
        lora_basis = embeddings.lora_basis.detach().cpu()
        calc_stats("lora_basis_placeholder", embeddings.lora_basis[:N])
        calc_stats("lora_basis_learned",     embeddings.lora_basis[N:])
        if not isinstance(embeddings.bias, int):
            calc_stats("bias", embeddings.bias)
        embeddings = embeddings(False)

    calc_stats("embeddings", embeddings)
    print()
