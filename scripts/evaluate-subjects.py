import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import ImageDirEvaluator
from evaluation.vit_eval import ViTEvaluator
from scripts.gen_subjects_for_eval import parse_subject_file, parse_range_str

# num_samples: only evaluate the last (latest) num_samples images. 
# If -1, then evaluate all images
def compare_folders(clip_evator, dino_evator, gt_dir, samples_dir, prompt, num_samples=-1):
    gt_data_loader     = PersonalizedBase(gt_dir,      set='evaluation', size=256, flip_p=0.0)
    sample_data_loader = PersonalizedBase(samples_dir, set='evaluation', size=256, flip_p=0.0)

    sample_start_idx = 0 if num_samples == -1 else sample_data_loader.num_images - num_samples
    sample_range     = range(sample_start_idx, sample_data_loader.num_images)
    # sample_filenames = [sample_data_loader.image_paths[i] for i in sample_range]
    # print("Sample filenames:", sample_filenames)

    gt_images = [torch.from_numpy(gt_data_loader[i]["image"]).permute(2, 0, 1) for i in range(gt_data_loader.num_images)]
    gt_images = torch.stack(gt_images, axis=0)
    sample_images = [torch.from_numpy(sample_data_loader[i]["image"]).permute(2, 0, 1) for i in sample_range]
    sample_images = torch.stack(sample_images, axis=0)

    sim_img, sim_text = clip_evator.evaluate(sample_images, gt_images, prompt)

    # huggingface's ViT pipeline requires a list of PIL images as input.
    gt_pil_images     = [ gt_data_loader[i]["image_unnorm"] for i in range(gt_data_loader.num_images) ]
    sample_pil_images = [ sample_data_loader[i]["image_unnorm"] for i in sample_range ]

    sim_dino = dino_evator.img_to_img_similarity(gt_pil_images, sample_pil_images)

    print(os.path.basename(gt_dir), "vs", os.path.basename(samples_dir))
    print(f"Image/text/dino sim: {sim_img:.3f} {sim_text:.3f} {sim_dino:.3f}")
    return sim_img, sim_text, sim_dino

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
    parser.add_argument(
        "--class_name_format", type=str, default="long", choices=["short", "long"],
        help="Format of class name to use for prompt"
    )

    opt = parser.parse_args()
    clip_evator = ImageDirEvaluator('cuda')
    dino_evator = ViTEvaluator('cuda')

    # Always pass "db", no matter what the actual method is. 
    # So that class_tokens are the long class name, instead of the one-token short class name.
    # This helps better match the prompt with the image.
    subjects, class_tokens, broad_classes = parse_subject_file(opt.subject_file, "db")
    low, high = parse_range_str(opt.range)
    subjects  = subjects[low:high]
    class_tokens = class_tokens[low:high]
    allsubj_sims_img, allsubj_sims_text, allsubj_sims_dino = [], [], []
    subject_count = len(subjects)

    for i, subject in enumerate(subjects):
        print(f"{i+1}/{subject_count}  {subject}")
        class_token = class_tokens[i]

        subject_gt_dir = os.path.join(opt.gt_dir, subject)
        subject_prompts_filepath = os.path.join(opt.samples_dir, subject + "-prompts.txt")
        print(f"Reading prompts from {subject_prompts_filepath}")
        subj_sims_img, subj_sims_text, subj_sims_dino = [], [], []
        processed_prompts = {}

        with open(subject_prompts_filepath, "r") as f:
            # splitlines() will remove the trailing newline. So no need to strip().
            lines = f.read().splitlines()
            indiv_subdirs_prompts = [ line.split("\t") for line in lines ]
            for indiv_subdir, prompt0, prompt_template in indiv_subdirs_prompts:
                if opt.class_name_format == 'long':
                    # Prompt is different from prompt0 (prompt0 is used for ada generation). 
                    # It doesn't contain 'z', but contains the full class name (as opposed to the short one)
                    # Patch the class token of  "red cartoon". "a cartoon in the forest ..." doesn't make sense.
                    if class_token == 'cartoon':
                        class_token = "cartoon monster"

                    prompt = prompt_template.format("", class_token)
                else:
                    # Simply remove the subject placeholder from prompt0.
                    prompt = prompt0.replace(" z ", " ")

                if prompt in processed_prompts:
                    continue
                print(f"Prompt: {prompt}")
                subjprompt_samples_dir = os.path.join(opt.samples_dir, indiv_subdir)
                subjprompt_sim_img, subjprompt_sim_text, subjprompt_sim_dino = \
                    compare_folders(clip_evator, dino_evator, subject_gt_dir, subjprompt_samples_dir, prompt, opt.num_samples)
                
                subj_sims_img.append(subjprompt_sim_img.detach().cpu().numpy())
                subj_sims_text.append(subjprompt_sim_text.detach().cpu().numpy())
                subj_sims_dino.append(subjprompt_sim_dino.detach().cpu().numpy())

                processed_prompts[prompt] = True

        subj_sims_img_avg = np.mean(subj_sims_img)
        subj_sims_text_avg = np.mean(subj_sims_text)
        subj_sims_dino_avg = np.mean(subj_sims_dino)
        print(f"Mean image/text/dino sim: {subj_sims_img_avg:.3f} {subj_sims_text_avg:.3f} {subj_sims_dino_avg:.3f}")
        allsubj_sims_img.append(subj_sims_img_avg)
        allsubj_sims_text.append(subj_sims_text_avg)
        allsubj_sims_dino.append(subj_sims_dino_avg)

        allsubj_sims_img_avg = np.mean(allsubj_sims_img)
        allsubj_sims_text_avg = np.mean(allsubj_sims_text)
        allsubj_sims_dino_avg = np.mean(allsubj_sims_dino)
        print(f"All subjects mean image/text/dino sim: {allsubj_sims_img_avg:.3f} {allsubj_sims_text_avg:.3f} {allsubj_sims_dino_avg:.3f}")

        print()

    for i, subject in enumerate(subjects):
        print(f"{i+1} {subject}: {allsubj_sims_img[i]:.3f} {allsubj_sims_text[i]:.3f} {allsubj_sims_dino[i]:.3f}")

    print(f"All subjects: {allsubj_sims_img_avg:.3f} {allsubj_sims_text_avg:.3f} {allsubj_sims_dino_avg:.3f}")
