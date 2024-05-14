import torch
from evaluation.eval_utils import find_first_match
import sys
import os
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sig", dest='ckpt_sig', type=str, required=True)
parser.add_argument("--sig2", type=str, default="")
parser.add_argument("--only100", action="store_true")
parser.add_argument("--skipnames", nargs="+", default=[])
parser.add_argument("--startiter", type=int, default=0)
args = parser.parse_args()

np.set_printoptions(precision=4, suppress=True)

all_ckpt_names = os.listdir("logs")
# Sort all_ckpt_names by name (actually by timestamp in the name), so that most recent first.
all_ckpt_names.sort(reverse=True)
ckpt_name  = find_first_match(all_ckpt_names, args.ckpt_sig, extra_sig=args.sig2)
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

emb_path = os.path.join(emb_folder, iter2path[iterations[0]])
emb_ckpt = torch.load(emb_path, map_location="cpu")
tokens = emb_ckpt['string_to_token'].keys()

if 'string_to_subj_basis_generator_dict' in emb_ckpt:
    for token in tokens:
        print(f"{token} => subj_basis_generator:")

        prev_subj_basis_generator = None

        for idx, iteration in enumerate(iterations):
            if (args.only100 and iteration % 100 != 0) or (iteration < args.startiter):
                continue
                    
            emb_path = os.path.join(emb_folder, iter2path[iteration])
            emb_ckpt = torch.load(emb_path, map_location="cpu")

            subj_basis_generator = emb_ckpt['string_to_subj_basis_generator_dict'][token]
            print(f"{iteration}:")

            if prev_subj_basis_generator is not None:
                param_total_norm  = 0
                param_total_delta = 0
                for param_name, param in subj_basis_generator.named_parameters():
                    if any([skipname in param_name for skipname in args.skipnames]):
                        continue
                    # Skip non-learnable parameters.
                    if param.requires_grad:
                        prev_param = prev_subj_basis_generator.state_dict()[param_name]
                        param_norm = torch.norm(param).item()
                        param_delta = torch.norm(param - prev_param).item()
                        param_total_norm  += param_norm
                        param_total_delta += param_delta
                        param_shape = list(param.shape)

                        print(f"{param_name}-{iteration} {param_shape} norm/diff: {param_norm:.4f}/{param_delta:.4f}")

                print(f"Total norm/diff: {param_total_norm:.4f}/{param_total_delta:.4f}")

            prev_subj_basis_generator = subj_basis_generator

for idx, iteration in enumerate(iterations):
    if args.only100 and iteration % 100 != 0:
        continue
            
    emb_path = os.path.join(emb_folder, iter2path[iteration])
    emb_ckpt = torch.load(emb_path, map_location="cpu")
    emb_global_scale_scores = emb_ckpt['emb_global_scale_scores'].sigmoid() + 0.5
    emb_global_scale_scores = emb_global_scale_scores.detach().cpu().numpy()
    print(f"{iteration} emb_global_scale_scores: {emb_global_scale_scores}")

    subj2conv_attn_layerwise_scales = emb_ckpt['subj2conv_attn_layerwise_scales']
    for subj_string, conv_attn_layerwise_scales in subj2conv_attn_layerwise_scales.items():
        conv_attn_layerwise_scales = conv_attn_layerwise_scales.detach().cpu().numpy()
        print(f"{iteration} {subj_string}: conv_attn_layerwise_scales = {conv_attn_layerwise_scales}")

print()
    
for k in tokens:
    print(f"Token: {k}")

    print("Attn Poolers:")

    for idx, iteration in enumerate(iterations):
        if args.only100 and iteration % 100 != 0:
            continue

        emb_path = os.path.join(emb_folder, iter2path[iteration])
        emb_ckpt = torch.load(emb_path, map_location="cpu")

        ada_embedder = emb_ckpt['string_to_ada_embedder'][k]
        attn_poolers = ada_embedder.poolers
        for i, attn_pooler in enumerate(attn_poolers):
            lora_to_fg_q_mean = attn_pooler.lora_to_fg_q.weight.abs().mean()
            lora_to_bg_q_mean = attn_pooler.lora_to_bg_q.weight.abs().mean()
            lora_to_k_mean = attn_pooler.lora_to_k.weight.abs().mean()

            print(f"{iteration}-{i}: lora_to_fg_q: {lora_to_fg_q_mean:.4f}, lora_to_bg_q: {lora_to_bg_q_mean:.4f}, lora_to_k: {lora_to_k_mean:.4f}")

    '''
    print("layer_coeff_maps bias:")

    for idx, iteration in enumerate(iterations):
        if args.only100 and iteration % 100 != 0:
            continue
                
        emb_path = os.path.join(emb_folder, iter2path[iteration])
        emb_ckpt = torch.load(emb_path, map_location="cpu")

        ada_embedder = emb_ckpt['string_to_ada_embedder'][k]
        attn_poolers = ada_embedder.poolers

        layer_coeff_map_means = [ layer_coeff_map.bias.abs().mean().item() for layer_coeff_map in ada_embedder.layer_coeff_maps ]
        layer_coeff_map_means = np.array(layer_coeff_map_means)

        print(f"{iteration}: {layer_coeff_map_means}")
    '''
    