#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
# git clone https://github.com/zllrunning/face-parsing.PyTorch face_parsing
sys.path.append('face_parsing')
from face_parsing.model import BiSeNet

import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from pprint import pprint

def vis_parsing_maps(im, parsing_anno, stride, save_im, save_path):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    #vis_im = im.copy().astype(np.uint8)
    #vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # Convert to binary mask
        vis_parsing_anno[vis_parsing_anno!=0] = 255
        cv2.imwrite(save_path[:-4] +'_mask.png', vis_parsing_anno)
        #cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def gen_masks(ckpt_path, src_paths, result_path, exist_path=None, 
              trash_path_suffix='trash', inspect_path_suffix='inspect', max_imgs_per_person=-1):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if isinstance(src_paths, (list, tuple)):
        src_path = src_paths[0]
    else:
        src_path = src_paths

    if src_path.endswith('/') or src_path.endswith('\\'):
        src_path = src_path[:-1]
    
    # trash_path: a folder to save images with too few (<=9) parts.
    trash_path   = src_path + "_" + trash_path_suffix
    if not os.path.exists(trash_path):
        os.makedirs(trash_path)
    # inspect_path: a folder to save images with too many (>=18) parts.
    inspect_path = src_path + "_" + inspect_path_suffix
    if not os.path.exists(inspect_path):
        os.makedirs(inspect_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(ckpt_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    parts_number_stats = {}
    img_count = 0
    # If src_paths is a list/tuple, then we assume there's no subfolders under src_paths.
    # We use src_paths as subj_paths.
    if isinstance(src_paths, (list, tuple)):
        subj_paths = src_paths
        # Don't include subfolders in the path when saving files.
        # osp.join(result_path, "") will add a trailing "/" to result_path, which is harmless.
        subj_dirs = [ "" for _ in subj_paths ]
    # Otherwise, we assume there are subfolders under src_paths. We put them into subj_paths.
    else:
        subj_dirs = list(os.listdir(src_paths))
        subj_paths = []
        subj_dirs2 = []
        for subj_dir in subj_dirs:
            # Already exists in exist_path. Skip.
            if exist_path is not None and os.path.isdir(osp.join(exist_path, subj_dir)):
                print(f"{subj_path} has been processed before, skip")
                continue
            if not os.path.isdir(osp.join(src_paths, subj_dir)):
                continue
            subj_paths.append(osp.join(src_paths, subj_dir))
            subj_dirs2.append(subj_dir)
        subj_dirs = subj_dirs2

    for (subj_dir, subj_path) in zip(subj_dirs, subj_paths):
        print(f"Processing {subj_path} '{subj_dir}'")

        subj_img_count = 0
        for img_path in sorted(os.listdir(subj_path)):
            if img_path.endswith("_mask.png"):
                continue
            
            img_full_path = osp.join(subj_path, img_path)
            new_img_path  = osp.join(result_path, subj_dir, img_path)
            if os.path.exists(new_img_path):
                continue

            subj_img_count += 1
            if max_imgs_per_person > 0 and subj_img_count > max_imgs_per_person:
                break

            image_obj = Image.open(img_full_path)
            image_obj = image_obj.resize((512, 512), Image.BILINEAR).convert("RGB") 
            img = to_tensor(image_obj)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            with torch.no_grad():
                out = net(img)[0]

            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            unique_parts = np.unique(parsing)
            # print(img_path, unique_parts)                
            parts_number_stats[len(unique_parts)] = parts_number_stats.get(len(unique_parts), 0) + 1
            img_count += 1
            if img_count % 100 == 0:
                print(f'{img_count}: ', end="")
                pprint(parts_number_stats)

            # Bad images. Move to trash folder.
            if len(unique_parts) <= 9:
                img_trash_path = osp.join(trash_path, subj_dir, img_path)
                if not os.path.exists(osp.join(trash_path, subj_dir)):
                    os.makedirs(osp.join(trash_path, subj_dir))

                print(f"{img_full_path} -> {img_trash_path}")
                os.rename(img_full_path, img_trash_path)
                continue

            if len(unique_parts) >= 18:
                img_inspect_path = osp.join(inspect_path, subj_dir, img_path)
                if not os.path.exists(osp.join(inspect_path, subj_dir)):
                    os.makedirs(osp.join(inspect_path, subj_dir))
                os.rename(img_full_path, img_inspect_path)
                print(f'Image {img_inspect_path} has {len(unique_parts)} parts.')
                continue

            if not os.path.exists(osp.join(result_path, subj_dir)):
                os.makedirs(osp.join(result_path, subj_dir))

            print(f"{img_full_path} -> {new_img_path}")
            # Save the image, instead of copying it. So that the new image will be (512, 512).
            image_obj.save(new_img_path, compress_level=1)
            #shutil.copy(img_full_path, new_img_path)
            vis_parsing_maps(image_obj, parsing, stride=1, save_im=True, 
                                save_path=osp.join(result_path, subj_dir, img_path))

if __name__ == "__main__":
    ckpt_path = osp.join('models/BiSeNet', '79999_iter.pth')
    '''
    gen_masks(ckpt_path, src_paths='/path/to/VGGface2_HQ', 
             result_path='/path/to/VGGface2_HQ_masks2', 
             exist_path='/path/to/VGGface2_HQ_masks',
             max_imgs_per_person=-1)
    '''
    face_folder = sys.argv[1]
    face_folder_par_dir, face_folder_name = osp.split(face_folder)
    # If the face folder doesn't have subfolders, we need to put it in a list/tuple.
    gen_masks(ckpt_path, src_paths=[face_folder], 
              result_path=osp.join(face_folder_par_dir, f"{face_folder_name}_masks"))
            