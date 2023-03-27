import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import ImageDirEvaluator
from scripts.gen_subjects_for_eval import parse_subject_file, parse_range_str

# num_samples: only evaluate the last (latest) num_samples images. 
# If -1, then evaluate all images
def compare_folders(evaluator, gt_dir, samples_dir, prompt, num_samples=-1):
    gt_data_loader  = PersonalizedBase(gt_dir,      set='evaluation', size=256, flip_p=0.0)
    sample_data_loader = PersonalizedBase(samples_dir, set='evaluation', size=256, flip_p=0.0)

    sample_start_idx = 0 if num_samples == -1 else sample_data_loader.num_images - num_samples
    sample_range     = range(sample_start_idx, sample_data_loader.num_images)
    gt_images = [torch.from_numpy(gt_data_loader[i]["image"]).permute(2, 0, 1) for i in range(gt_data_loader.num_images)]
    gt_images = torch.stack(gt_images, axis=0)
    sample_images = [torch.from_numpy(sample_data_loader[i]["image"]).permute(2, 0, 1) for i in sample_range]
    sample_images = torch.stack(sample_images, axis=0)

    sim_img, sim_text = evaluator.evaluate(gt_images, sample_images, prompt)

    print(gt_dir, "vs", samples_dir)
    print(f"Image/text similarities: {sim_img:.3f} {sim_text:.3f}")
    return sim_img, sim_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", default='ada', choices=["ada", "ti", "db"], type=str, 
                        help="method to use for generating samples")
    
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="db-eval-dataset",
        help="Directory with images of subjects used to train the model"
    )

    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples-dbeval-ada",
        help="Directory with synthesized images of subjects"
    )
    parser.add_argument(
        "--subject_file",
        type=str,
        default="scripts/info-db-eval-subjects.sh",
        help="File with subjects to evaluate"
    )
    parser.add_argument(
        "--range", type=str, default=None, 
        help="Range of subjects to generate "
             "(Index starts from 1 and is inclusive, e.g., 1-30)"
    )
    # num_samples = 4
    parser.add_argument(
        "--num_samples", type=int, default=4,
        help="Number of samples to generate for each subject under each prompt"
    )

    opt = parser.parse_args()
    evaluator = ImageDirEvaluator('cuda')

    subjects, class_tokens, broad_classes = parse_subject_file(opt.subject_file, opt.method)
    low, high = parse_range_str(opt.range)
    subjects  = subjects[low:high]
    allsubj_sims_img, allsubj_sims_text = [], []

    for subject in subjects:
        print("Subject: ", subject)
        subject_gt_dir = os.path.join(opt.gt_dir, subject)
        subject_prompts_filepath = os.path.join(opt.samples_dir, subject + "-prompts.txt")
        print(f"Reading prompts from {subject_prompts_filepath}")
        subj_sims_img, subj_sims_text = [], []
        
        with open(subject_prompts_filepath, "r") as f:
            # splitlines() will remove the trailing newline. So no need to strip().
            lines = f.read().splitlines()
            indiv_subdirs_prompts = [ line.split("\t") for line in lines ]
            for indiv_subdir, prompt0, prompt_template in indiv_subdirs_prompts:
                # Remove subject placeholder from prompt
                prompt = prompt0.replace(" z ", " ")
                print(f"Prompt: {prompt}")
                subjprompt_samples_dir = os.path.join(opt.samples_dir, indiv_subdir)
                subjprompt_sim_img, subjprompt_sim_text = \
                    compare_folders(evaluator, subject_gt_dir, subjprompt_samples_dir, prompt, opt.num_samples)
                
                subj_sims_img.append(subjprompt_sim_img)
                subj_sims_text.append(subjprompt_sim_text)

        subj_sims_img_avg = np.mean(subj_sims_img)
        subj_sims_text_avg = np.mean(subj_sims_text)
        print(f"Mean image/text similarities: {subj_sims_img_avg:.3f} {subj_sims_text_avg:.3f}")
        allsubj_sims_img.append(subj_sims_img_avg)
        allsubj_sims_text.append(subj_sims_text_avg)
    
    allsubj_sims_img_avg = np.mean(allsubj_sims_img)
    allsubj_sims_text_avg = np.mean(allsubj_sims_text)
    print(f"Mean image/text similarities of all subjects: {allsubj_sims_img_avg:.3f} {allsubj_sims_text_avg:.3f}")

