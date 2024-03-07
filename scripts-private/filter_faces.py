import insightface
from insightface.app import FaceAnalysis

import torch
import torch.nn.functional as F
import cv2
import shutil
import os
import json

gpu_id = 0
T = 0.65

base_folder  = '/data/shaohua/VGGface2_HQ_masks/'
trash_folder = '/data/shaohua/VGGface2_HQ_masks_trash/'
#face_encoder = insightface.model_zoo.get_model('models/insightface/model.onnx', 
#                                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_encoder = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_encoder.prepare(ctx_id=gpu_id, det_size=(512, 512))

face_encoder.prepare(ctx_id=gpu_id)
trash_img_count = 0
trash_mask_count = 0
num_subjects = len(os.listdir(base_folder))
print(f'num_subjects={num_subjects}')
resumed_subj_folder = None #'n007809'
resumed = True

for subj_i, subj_folder in enumerate(os.listdir(base_folder)):
    image_fullpaths = []
    id_embs = []
    print(f"{subj_i+1}/{num_subjects}: {subj_folder}")
    subj_path = os.path.join(base_folder, subj_folder)
    subj_trash_path = os.path.join(trash_folder, subj_folder)
    if not resumed and resumed_subj_folder == subj_folder:
        resumed = True
    if not resumed:
        continue

    genders = []
    ages = []

    for image_i, image_path in enumerate(os.listdir(subj_path)):
        if "_mask.png" in image_path or ".pt" in image_path or ".json" in image_path:
            continue

        if image_i % 20 == 0:
            print(image_i)
        image_fullpath = os.path.join(subj_path, image_path)
        image_np = cv2.imread(image_fullpath)
        # id_emb_np: [1, 512].

        face_info = face_encoder.get(image_np)
        if len(face_info) == 0:
            if not os.path.exists(subj_trash_path):
                os.makedirs(subj_trash_path)
            image_trash_path = os.path.join(subj_trash_path, os.path.basename(image_fullpath))
            # Move to trash folder
            shutil.move(image_fullpath, image_trash_path)
            print('No face detected:', image_trash_path)
            trash_img_count += 1

            mask_fullpath = image_fullpath.replace('.jpg', '_mask.png')
            if os.path.exists(mask_fullpath):
                mask_trash_path = image_trash_path.replace('.jpg', '_mask.png')
                shutil.move(mask_fullpath, mask_trash_path)
                trash_mask_count += 1
            continue
        else:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
            # id_emb: [512,]
            id_emb = torch.from_numpy(face_info.normed_embedding).to(f"cuda:{gpu_id}")
            id_embs.append(id_emb)
            image_fullpaths.append(image_fullpath)
            gender  = face_info['gender']
            age     = face_info['age']
            genders.append(gender)
            ages.append(age)

    if len(id_embs) == 0:
        continue

    id_embs = torch.stack(id_embs)
    id_embs = F.normalize(id_embs, p=2, dim=1)
    # Compute pairwise similarities of the embeddings.
    mean_emb = id_embs.mean(dim=0, keepdim=True)
    mean_emb = F.normalize(mean_emb, p=2, dim=1)
    torch.save(mean_emb, os.path.join(subj_path, 'mean_emb.pt'))

    # [1, 512] * [512, BS] => [1, BS]
    sims_to_mean = torch.matmul(mean_emb, id_embs.t())[0]
    avg_sim = sims_to_mean.mean().item()
    print(f"Avg sim: {avg_sim:.3f}")

    # Majority vote for age and gender.
    avg_age     = sum(ages) / len(ages)
    avg_gender  = sum(genders) / len(genders)
    # Neutral face? 
    if avg_gender == 0.5:
        breakpoint()

    # Map gender and avg_age to one in ["man", "woman", "young man", "young woman", "boy", "girl"]:
    if avg_age <= 18 and avg_gender < 0.5:
        person_type = "girl"
    if avg_age <= 18 and avg_gender > 0.5:
        person_type = "boy"
    if avg_age > 18 and avg_age < 30 and avg_gender < 0.5:
        person_type = "young woman"
    if avg_age > 18 and avg_age < 30 and avg_gender > 0.5:
        person_type = "young man"
    if avg_age >= 30 and avg_gender < 0.5:
        person_type = "woman"
    if avg_age >= 30 and avg_gender > 0.5:
        person_type = "man"

    metainfo = { 'gender': round(avg_gender), 'age': round(avg_age), 'person_type': person_type,
                 'avg_sim': avg_sim }
    json.dump(metainfo, open(os.path.join(subj_path, 'metainfo.json'), 'w'))

    for i, sim in enumerate(sims_to_mean):
        if sim < T:
            if not os.path.exists(subj_trash_path):
                os.makedirs(subj_trash_path)
            image_trash_path = os.path.join(subj_trash_path, os.path.basename(image_fullpaths[i]))
            # Move to trash folder
            shutil.move(image_fullpaths[i], image_trash_path)
            print(f'sim={sim:.4f}, moved to trash:', image_trash_path)
            trash_img_count += 1
            mask_fullpath = image_fullpaths[i].replace('.jpg', '_mask.png')
            if os.path.exists(mask_fullpath):
                mask_trash_path = image_trash_path.replace('.jpg', '_mask.png')
                shutil.move(mask_fullpath, mask_trash_path)
                trash_mask_count += 1

print()
print(f'trash_img_count={trash_img_count}, trash_mask_count={trash_mask_count}')
