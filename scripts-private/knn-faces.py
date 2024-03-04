import insightface
from insightface.app import FaceAnalysis

import torch
import torch.nn.functional as F
import cv2
import shutil
import os
import glob

gpu_id = 0
T = 0.65

base_folder  = '/data/shaohua/VGGface2_HQ_masks/'
face_encoder = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_encoder.prepare(ctx_id=gpu_id, det_size=(512, 512))

face_encoder.prepare(ctx_id=gpu_id)
trash_img_count = 0
trash_mask_count = 0
num_subjects = len(os.listdir(base_folder))
print(f'num_subjects={num_subjects}')
resumed_subj_folder = None #'n007809'
resumed = True

subj_embs = []
subj_folders = []

for subj_i, subj_folder in enumerate(os.listdir(base_folder)):
    #print(f"{subj_i+1}/{num_subjects}: {subj_folder}")
    subj_emb_path = os.path.join(base_folder, subj_folder, "mean_emb.pt")
    id_emb = torch.load(subj_emb_path)
    id_emb = F.normalize(id_emb, p=2, dim=1)
    subj_folders.append(subj_folder)
    subj_embs.append(id_emb)

print(f"Num subjects: {len(subj_embs)}")

# [N, 512]
subj_embs = torch.cat(subj_embs)
# [N, 512] * [512, N] => [N, N]
pairwise_sims = torch.matmul(subj_embs, subj_embs.t())
# Set the diagonal of the similarity matrix to zero
pairwise_sims.fill_diagonal_(0)
topk_sims, topk_indices = torch.topk(pairwise_sims, k=1, dim=1)
# Avg top-1 sim: 0.250
print(f"Avg top-1 sim: {topk_sims.mean().item():.3f}")
for i in range(len(subj_folders)):
    print(f"{subj_folders[i]}: {subj_folders[topk_indices[i].item()]}: {topk_sims[i].item():.3f}")
    subj1_image_path = glob.glob(os.path.join(base_folder, subj_folders[i], "*.jpg"))[0]
    subj2_image_path = glob.glob(os.path.join(base_folder, subj_folders[topk_indices[i].item()], "*.jpg"))[0]
    print(f"{subj1_image_path}\n{subj2_image_path}")
