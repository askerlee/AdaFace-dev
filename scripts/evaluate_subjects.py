import argparse, os
import numpy as np

from scripts.eval_utils import init_evaluators, compare_folders, parse_subject_file, parse_range_str

def parse_args():
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
    parser.add_argument(
        "--selset", action="store_true",
        help="Whether to evaluate only the selected subset of subjects"
    )
    parser.add_argument("--skipselset", action="store_true",
                        help="Whether to generate all subjects except the selected subset")    
    parser.add_argument(
        "--num_samples", type=int, default=4,
        help="Number of samples to generate for each subject under each prompt"
    )
    parser.add_argument(
        "--class_name_format", type=str, default="long", choices=["short", "long"],
        help="Format of class name to use for prompt"
    )

    parser.add_argument(
        "--gpu",  type=int, default=0,
        help="GPU to use for evaluation"
    )
    # gt_self_compare
    #   If True,  compare the subject's gt images with themselves.
    #   If False, compare each subject's generated images with the subject's gt images.
    parser.add_argument(
        "--gt_self_compare", action="store_true",
        help="Whether to compare all images in the subject gt image folder"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    clip_evator, dino_evator = init_evaluators(args.gpu)

    # Always pass "db", no matter what the actual method is. 
    # So that class_tokens are the long class name, instead of the one-token short class name.
    # This helps better match the prompt with the image.
    subjects, class_tokens, broad_classes, sel_set = parse_subject_file(args.subject_file, "db")

    subject_indices = list(range(len(subjects)))
    if args.selset:
        subject_indices = sel_set

    range_indices   = parse_range_str(args.range)
    subject_indices = [ subject_indices[i] for i in range_indices ]
        
    allsubj_sims_img, allsubj_sims_text, allsubj_sims_dino = [], [], []
    subject_count = len(subject_indices)

    for i, subject_idx in enumerate(subject_indices):
        if args.skipselset and subject_idx in sel_set:
            continue
        subject      = subjects[subject_idx]
        class_token  = class_tokens[subject_idx]
        broad_class  = broad_classes[subject_idx]

        print(f"{i+1}/{subject_count}  {subject}")
        class_token = class_tokens[i]

        subject_gt_dir = os.path.join(args.gt_dir, subject)
        subject_prompts_filepath = os.path.join(args.samples_dir, subject + "-prompts.txt")
        print(f"Reading prompts from {subject_prompts_filepath}")
        subj_sims_img, subj_sims_text, subj_sims_dino = [], [], []
        processed_prompts = {}

        with open(subject_prompts_filepath, "r") as f:
            # splitlines() will remove the trailing newline. So no need to strip().
            lines = f.read().splitlines()
            indiv_subdirs_prompts = [ line.split("\t") for line in lines ]
            prompts_total_num = len(indiv_subdirs_prompts)
            prompt_idx = 0

            for _, indiv_subdir, prompt0, prompt_template in indiv_subdirs_prompts:
                prompt_idx += 1
                if args.class_name_format == 'long':
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
                print(f"{prompt_idx}/{prompts_total_num} Prompt: {prompt}")
                subjprompt_samples_dir = os.path.join(args.samples_dir, indiv_subdir)
                subjprompt_sim_img, subjprompt_sim_text, subjprompt_sim_dino = \
                    compare_folders(clip_evator, dino_evator, subject_gt_dir, subjprompt_samples_dir, prompt, args.num_samples)
                
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
