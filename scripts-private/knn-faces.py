import insightface
from insightface.app import FaceAnalysis
from ldm.modules.subj_basis_generator import IP_MLPProjModel

import torch
import torch.nn.functional as F
import cv2
import shutil
import os
import glob

gpu_id = 0
T = 0.65

base_folder  = '/data/shaohua/VGGface2_HQ_masks/'
ip_model_ckpt_path = "models/ip-adapter/ip-adapter-faceid-portrait_sd15.bin"
face_encoder = FaceAnalysis(name="antelopev2", root='arc2face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_encoder.prepare(ctx_id=gpu_id, det_size=(512, 512))
face_proj_in = IP_MLPProjModel(cross_attention_dim=768, 
                               id_embeddings_dim=512, num_tokens=16)
face_proj_in.to("cuda")

ip_model_ckpt = torch.load(ip_model_ckpt_path, map_location="cuda")
face_proj_in.load_state_dict(ip_model_ckpt['image_proj'])
print(f"Subj face_proj_in is loaded from {ip_model_ckpt_path}")

trash_img_count = 0
trash_mask_count = 0
num_subjects = len(os.listdir(base_folder))
print(f'num_subjects={num_subjects}')
resumed_subj_folder = None 
resumed = True

for noise_std in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
    subj_embs = []
    face_proj_embs = []
    subj_folders = []

    for subj_i, subj_folder in enumerate(os.listdir(base_folder)):
        #print(f"{subj_i+1}/{num_subjects}: {subj_folder}")
        subj_emb_path = os.path.join(base_folder, subj_folder, "mean_emb.pt")
        id_emb = torch.load(subj_emb_path)
        id_emb = F.normalize(id_emb, p=2, dim=1)
        subj_folders.append(subj_folder)
        # Add noise to the embeddings
        noisy_emb = id_emb + noise_std * torch.randn_like(id_emb)
        subj_embs.append(noisy_emb)
        with torch.no_grad():
            face_proj_emb = face_proj_in(noisy_emb)
        face_proj_emb = F.normalize(face_proj_emb, p=2, dim=2)
        face_proj_embs.append(face_proj_emb)

    print(f"Num subjects: {len(subj_embs)}")

    # [N, 512]
    subj_embs = torch.cat(subj_embs, dim=0)
    print("subj_embs avg std: {:.4f}".format(subj_embs.std(dim=1).mean().item()))

    # [N, 16, 768]
    face_proj_embs = torch.cat(face_proj_embs, dim=0)
    mean_face_proj_emb = face_proj_embs.mean(dim=0, keepdim=True)

    '''
    if noise_std == 0:
        # Saved mean_face_proj_emb: [16, 768].
        torch.save(mean_face_proj_emb.squeeze(0), "models/ip-adapter/mean_face_proj_emb.pt")
        print("mean_face_proj_emb is saved to models/ip-adapter/mean_face_proj_emb.pt")
    '''
    
    face_proj_emb_mean_ratio = (face_proj_embs.abs() / (mean_face_proj_emb.abs() + 1e-6)).mean()
    print("face_proj_embs mean/ratio:")
    print(mean_face_proj_emb.abs().mean(), face_proj_emb_mean_ratio)
    #print(face_proj_embs.std(dim=(0,2)))

    # [N, 512] * [512, N] => [N, N]
    pairwise_sims = torch.matmul(subj_embs, subj_embs.t())
    # Set the diagonal of the similarity matrix to zero
    pairwise_sims.fill_diagonal_(0)

    topk_sims, topk_indices = torch.topk(pairwise_sims, k=1, dim=1)
    # Avg top-1 sim: 0.250
    print(f"Avg top-1 sim: {topk_sims.mean().item():.3f}")

    # [N, 16, 768] -> [N, 12288]
    orig_face_proj_embs     = F.normalize(face_proj_embs.flatten(start_dim=1), p=2, dim=1)
    demeaned_face_proj_embs = face_proj_embs - mean_face_proj_emb
    demeaned_face_proj_embs = F.normalize(demeaned_face_proj_embs.flatten(start_dim=1), p=2, dim=1)

    # [N, 512] * [512, N] => [N, N]
    pairwise_orig_face_proj_sims = torch.matmul(orig_face_proj_embs, orig_face_proj_embs.t())
    # Set the diagonal of the similarity matrix to zero
    pairwise_orig_face_proj_sims.fill_diagonal_(0)
          
    topk_orig_face_proj_sims, topk_orig_face_proj_indices = torch.topk(pairwise_orig_face_proj_sims, k=1, dim=1)
    # Avg top-1 sim: 0.250
    print(f"Avg top-1 face_proj sim: {topk_orig_face_proj_sims.mean().item():.3f}")
    print("Avg face_proj sim: {:.3f}".format(pairwise_orig_face_proj_sims.mean().item()))

    pairwise_dem_face_proj_sims = torch.matmul(demeaned_face_proj_embs, demeaned_face_proj_embs.t())
    # Set the diagonal of the similarity matrix to zero
    pairwise_dem_face_proj_sims.fill_diagonal_(0)
    topk_dem_face_proj_sims, topk_dem_face_proj_indices = torch.topk(pairwise_dem_face_proj_sims, k=1, dim=1)
    # Avg top-1 sim: 0.250
    print(f"Avg top-1 demeaned face_proj sim: {topk_dem_face_proj_sims.mean().item():.3f}")
    print("Avg demeaned face_proj sim: {:.3f}".format(pairwise_dem_face_proj_sims.mean().item()))

'''
for i in range(len(subj_folders)):
    print(f"{subj_folders[i]}: {subj_folders[topk_indices[i].item()]}: {topk_sims[i].item():.3f}")
    subj1_image_path = glob.glob(os.path.join(base_folder, subj_folders[i], "*.jpg"))[0]
    subj2_image_path = glob.glob(os.path.join(base_folder, subj_folders[topk_indices[i].item()], "*.jpg"))[0]
    print(f"{subj1_image_path}\n{subj2_image_path}")
'''
