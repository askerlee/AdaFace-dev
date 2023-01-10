import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.embedding_manager import StaticLoraEmbedding, DynamicLoraEmbedding
import sys
import os
import glob
import re
import numpy as np
np.set_printoptions(precision=3, suppress=True)

emb_ckpt_folder = sys.argv[1]
if len(sys.argv) > 2:
    output_type = sys.argv[2]
else:
    output_type = 'all'
    
# output_type: static or all

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
    # F.l2_loss doesn't take sqrt. So the loss is very small. 
    # Compute it manually.
    l2_loss = ((embeddings - emb_mean) ** 2).mean().sqrt()
    norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
    print("L1: %.4f, L2: %.4f" %(l1_loss.item(), l2_loss.item()))
    print("Norms: min: %.4f, max: %.4f, mean: %.4f, std: %.4f" %(norms.min(), norms.max(), norms.mean(), norms.std()))


# enumerate files in emb_ckpt_folder
for emb_ckpt_filename in emb_ckpt_files:
    emb_ckpt = torch.load(emb_ckpt_filename, map_location='cpu')
    print("%s STATIC:" %emb_ckpt_filename)
    for key in emb_ckpt['string_to_param']:
        embeddings = emb_ckpt['string_to_param'][key]
        if isinstance(embeddings, StaticLoraEmbedding):
            print("basis_comm_weights:")
            print(embeddings.basis_comm_weights.detach().cpu().numpy())
            calc_stats("basis_rand_weights", embeddings.basis_rand_weights)
            print("bias_scales:")
            print(embeddings.bias_scales.squeeze().detach().cpu().numpy())
            basis_vecs = embeddings.basis_vecs.detach().cpu()
            N = embeddings.N
            NEG = embeddings.NEG
            calc_stats("basis_vecs_pos", embeddings.basis_vecs[:N])
            if NEG > 0:
                calc_stats("basis_vecs_neg",     embeddings.basis_vecs[N:N+NEG])

            calc_stats("basis_vecs_rand", embeddings.basis_vecs[N+NEG:])
            if not isinstance(embeddings.bias, int):
                calc_stats("bias", embeddings.bias)
            embeddings = embeddings(False)

        calc_stats("embeddings", embeddings)
        cosine_mat = F.cosine_similarity(embeddings[:,:,None], embeddings.t()[None,:,:])
        triu_indices = torch.triu_indices(cosine_mat.size(0), cosine_mat.size(1), offset=1)
        cosine_mat = cosine_mat[triu_indices[0], triu_indices[1]]
        print("Cosine: min: %.4f, max: %.4f, mean: %.4f, std: %.4f" %(cosine_mat.min(), cosine_mat.max(), cosine_mat.mean(), cosine_mat.std()))

        print()

    if output_type == 'all':
        print("%s DYNAMIC:" %emb_ckpt_filename)
        for key in emb_ckpt['string_to_dyn_embedder']:
            embeddings = emb_ckpt['string_to_dyn_embedder'][key]
            if isinstance(embeddings, DynamicLoraEmbedding):
                basis_vecs = embeddings.basis_vecs.detach().cpu()
                N = embeddings.N
                calc_stats("basis_vecs_pos", embeddings.basis_vecs[:N])
                calc_stats("basis_vecs_rand", embeddings.basis_vecs[N:])
                for map in embeddings.maps:
                    calc_stats("map", map.weight)
