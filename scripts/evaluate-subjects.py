import argparse, os, sys, glob

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import ImageDirEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_dir",
        type=str,
        help="Directory with subjects images used to train the model"
    )

    parser.add_argument(
        "--syn_dir",
        type=str,
        help="Directory with synthesized subjects images"
    )

    opt = parser.parse_args()
    device = 'cuda'
    evaluator = ImageDirEvaluator(device)

    gt_data_loader  = PersonalizedBase(opt.gt_dir,  set='evaluation', size=256, flip_p=0.0)
    syn_data_loader = PersonalizedBase(opt.syn_dir, set='evaluation', size=256, flip_p=0.0)

    gt_images = [torch.from_numpy(gt_data_loader[i]["image"]).permute(2, 0, 1) for i in range(gt_data_loader.num_images)]
    gt_images = torch.stack(gt_images, axis=0)
    syn_images = [torch.from_numpy(syn_data_loader[i]["image"]).permute(2, 0, 1) for i in range(syn_data_loader.num_images)]
    syn_images = torch.stack(syn_images, axis=0)

    sim_img, sim_text = evaluator.evaluate(gt_images, syn_images)

    print("Image similarity: ", sim_img)
    print("Text similarity: ", sim_text)
