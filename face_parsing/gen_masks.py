#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import shutil
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

def evaluate(ckpt_path, src_path, result_path, exist_path, 
             max_imgs_per_person=40,
             trash_path='trash', inspect_path='inspect'):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

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
    
    with torch.no_grad():
        for subj_path in os.listdir(src_path):
            if os.path.isdir(osp.join(exist_path, subj_path)):
                print(f"{subj_path} processed, skip")
                continue

            if not os.path.isdir(osp.join(src_path, subj_path)):
                continue

            subj_img_count = 0
            for img_path in os.listdir(osp.join(src_path, subj_path)):
                subj_img_count += 1
                if subj_img_count > max_imgs_per_person:
                    break

                img_full_path = osp.join(src_path,    subj_path, img_path)
                new_img_path  = osp.join(result_path, subj_path, img_path)
                if os.path.exists(new_img_path):
                    continue

                img = Image.open(img_full_path)
                image = img.resize((512, 512), Image.BILINEAR)
                img = to_tensor(image)
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
                    img_trash_path = osp.join(trash_path, subj_path, img_path)
                    if not os.path.exists(osp.join(trash_path, subj_path)):
                        os.makedirs(osp.join(trash_path, subj_path))

                    print(f"{img_full_path} -> {img_trash_path}")
                    os.rename(img_full_path, img_trash_path)
                    continue

                if len(unique_parts) >= 18:
                    img_inspect_path = osp.join(inspect_path, subj_path, img_path)
                    if not os.path.exists(osp.join(inspect_path, subj_path)):
                        os.makedirs(osp.join(inspect_path, subj_path))
                    os.rename(img_full_path, img_inspect_path)
                    print(f'Image {img_inspect_path} has {len(unique_parts)} parts.')
                    continue

                if not os.path.exists(osp.join(result_path, subj_path)):
                    os.makedirs(osp.join(result_path, subj_path))

                print(f"{img_full_path} -> {new_img_path}")
                os.rename(img_full_path, new_img_path)
                vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(result_path, subj_path, img_path))


if __name__ == "__main__":
    ckpt_pth = osp.join('res/cp', '79999_iter.pth')
    evaluate(ckpt_pth, src_path='/data/shaohua/VGGface2_HQ', 
             result_path='/data/shaohua/VGGface2_HQ_masks2', 
             exist_path='/data/shaohua/VGGface2_HQ_masks',
             trash_path='/data/shaohua/VGGface2_HQ_trash',
             inspect_path='/data/shaohua/VGGface2_HQ_inspect')
