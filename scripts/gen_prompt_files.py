import argparse
import os
import re
import numpy as np
import csv
from evaluation.eval_utils import parse_range_str, format_prompt_list
from ldm.util import parse_subject_file

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_string", type=str, default="z", 
                        help="Subject placeholder string that represents the subject in prompts")
    parser.add_argument("--num_vectors_per_subj_token",
                        type=int, default=argparse.SUPPRESS,
                        help="Number of vectors per token. If > 1, use multiple embeddings to represent a subject.")
    parser.add_argument("--num_vectors_per_bg_token",
                        type=int, default=4,
                        help="Number of vectors for the background token. If > 1, use multiple embeddings to represent the background.")
 
    parser.add_argument("--ref_images", type=str, nargs='+', default=None,
                        help="Reference image for zero-shot learning. If not specified, use subject_gt_dir.")

    parser.add_argument("--prompt_set", dest='prompt_set_name', type=str, default='dreambench', 
                        choices=['dreambench', 'community', 'all'],
                        help="Subset of prompts to evaluate if --prompt is not specified")
    parser.add_argument("--gen_prompt_set_only", action="store_true",
                        help="Generate prompt set and exit")
    
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to use for generating samples, using {} for subject_string (default: None)")
    parser.add_argument("--neg_prompt", type=str, default=argparse.SUPPRESS,
                        help="Negative prompt to use for generating samples (default: None)")
    parser.add_argument("--use_pre_neg_prompt", type=str2bool, const=True, default=argparse.SUPPRESS, 
                        nargs="?", help="use predefined negative prompts")

    # Possible z_prefix_type: '' (none), 'class_name', or any user-specified string.
    parser.add_argument("--z_prefix_type", type=str, default='',
                        help="Prefix to prepend to z")
    # Possible z_suffix_type: '' (none), 'class_name', or any user-specified string.
    parser.add_argument("--z_suffix_type", type=str, default='', 
                        help="Append this string to the subject placeholder string during inference "
                             "(default: '' for humans/animals, 'class_name' for others)")
    parser.add_argument("--use_fp_trick", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use the 'face portrait' trick for the subject")
    parser.add_argument("--prompt_prefix", type=str, default="",
                        help="prefix to prepend to each prompt")
    # prompt_suffix: usually reduces the similarity.
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="suffix to append to the end of each prompt")
    parser.add_argument("--n_samples", type=int, default=-1, 
                        help="number of samples to generate for each test case")
    parser.add_argument("--bs", type=int, default=-1, 
                        help="batch size")
    parser.add_argument("--out_dir_tmpl", type=str, default="samples",
                        help="Template of parent directory to save generated samples")
    parser.add_argument("--scores_csv", type=str, default=None,
                        help="CSV file to save the evaluation scores")
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: batch_size)",
    )
    # File path containing composition case information
    parser.add_argument("--subjfile", type=str, default=None, 
                        help="subject info file for evaluation")
    # The range of indices of subjects to generate
    parser.add_argument("--range", type=str, default=None, 
                        help="Range of subjects to generate (Index starts from 1 and is inclusive, e.g., 1-30)")
    parser.add_argument("--compare_with_pardir", type=str, default=argparse.SUPPRESS,
                        help="Parent folder of subject images used for computing similarity with generated samples")
    parser.add_argument("--method", type=str, default="adaface",
                        choices=["adaface", "pulid"])    
    parser.add_argument("--dryrun", action="store_true",
                        help="Dry run: only print the commands without actual execution")

    args = parser.parse_args()
    return args, parser

if __name__ == "__main__":
    
    args, argparser = parse_args()

    args.orig_placeholder = args.subject_string
    outdir = args.out_dir_tmpl + "-" + args.method[:3]
    os.makedirs(outdir, exist_ok=True)

    for subject_type in ('man', 'woman', 'person'):
        if args.n_samples == -1:
            args.n_samples = 4
        if args.use_fp_trick:
            if args.method == 'adaface':
                fp_trick_string = "face portrait"
            elif args.method == 'pulid':
                fp_trick_string = "portrait"
        else:
            fp_trick_string = None


        print(f"Generating samples for {subject_type}: ")

        if args.z_prefix_type is None:
            z_prefix = ""
        elif args.z_prefix_type == 'class_name':
            z_prefix = subject_type
        else:
            z_prefix = args.z_prefix_type

        if args.z_suffix_type is None:
            z_suffix = ""
        elif args.z_suffix_type == 'class_name':
            z_suffix = subject_type
        else:
            z_suffix = args.z_suffix_type

        prompt_list, class_prompt_list = \
            format_prompt_list(args.subject_string, z_prefix, z_suffix, subject_type, 
                               broad_class=1, prompt_set_name=args.prompt_set_name, 
                               fp_trick_string=fp_trick_string)
        prompt_filepath = f"{outdir}/{subject_type}-prompts-{args.prompt_set_name}-{args.num_vectors_per_subj_token}.txt"
        PROMPTS = open(prompt_filepath, "w")

        for prompt, class_prompt in zip(prompt_list, class_prompt_list):
            if len(args.prompt_prefix) > 0:
                prompt = args.prompt_prefix + " " + prompt
                class_prompt = args.prompt_prefix + " " + class_prompt
            if len(args.prompt_suffix) > 0:
                prompt = prompt + ", " + args.prompt_suffix
                class_prompt = class_prompt + ", " + args.prompt_suffix

            # "{subj_name}" is a placeholder for the subject name. 
            # To be instantiated with the actual subject name by the stable_txt2img script.
            indiv_subdir = "{subj_name}" + "-" + prompt.replace(" ", "-")
            # In case some folder names are extremely long, truncate to 80 characters.
            indiv_subdir = indiv_subdir[:80]
            # class_prompt is saved in the prompt file as well 
            # and will be used for the CLIP text/image similarity evaluation.
            # class_prompt is not used.
            PROMPTS.write( "\t".join([str(args.n_samples), indiv_subdir, prompt, class_prompt]) + "\n" )

        PROMPTS.close()

        print(f"{len(prompt_list)} prompts saved to {prompt_filepath}")
        continue

    if args.gen_prompt_set_only:
        exit(0)

    if args.subjfile is not None:
        subj_info, subj2attr = parse_subject_file(args.subjfile)
        subjects, class_names, broad_classes = \
                subj_info['subjects'], subj_info['class_names'], subj_info['broad_classes']


    cls_delta_strings = subj_info['cls_delta_strings']

    subject_indices = list(range(len(subjects)))

    range_indices = parse_range_str(args.range)
    if range_indices is not None:
        subject_indices = [ subject_indices[i] for i in range_indices ]

    if args.scores_csv is None:
        args.scores_csv = f"{args.method}-{args.range}.csv"

    # Ty to create scores_csv file. If it already exists, make it empty.
    SCORES_CSV_FILE = open(args.scores_csv, "w")
    SCORES_CSV_FILE.close()

    for subject_idx in subject_indices:
        subject_name        = subjects[subject_idx]
        class_name          = class_names[subject_idx]
        broad_class         = broad_classes[subject_idx]
        cls_delta_string    = cls_delta_strings[subject_idx]

        print(subject_name, ":")

        command_line = f"python3 -m scripts.stable_txt2img --outdir {outdir}"
        
        if args.prompt is None:
            print(f"{len(prompt_list)} prompts saved to {prompt_filepath}")
            # Since we use a prompt file, we don't need to specify --n_samples.
            command_line += f" --from_file {prompt_filepath}"
        else:
            # Do not use a prompt file, but specify --n_samples, --prompt, and --indiv_subdir.
            # Note only the last prompt/class_prompt will be used. 
            command_line += f" --n_samples {args.n_samples} --indiv_subdir {indiv_subdir}"
            command_line += f" --prompt \"{prompt}\" --class_prompt \"{class_prompt}\""

        if args.scores_csv is not None:
            command_line += f" --scores_csv {args.scores_csv}"

        if hasattr(args, 'num_vectors_per_subj_token'):
            command_line += f" --subject_string {args.orig_placeholder} --num_vectors_per_subj_token {args.num_vectors_per_subj_token}"
        if hasattr(args, 'num_vectors_per_bg_token'):
            command_line += f" --background_string {args.background_string} --num_vectors_per_bg_token {args.num_vectors_per_bg_token}"

        if hasattr(args, 'neg_prompt'):
            command_line += f" --neg_prompt \"{args.neg_prompt}\""
        elif hasattr(args, 'use_pre_neg_prompt'):
            command_line += f" --use_pre_neg_prompt 1"

        if not hasattr(args, 'compare_with_pardir') and 'data_folder' in subj_info:
            args.compare_with_pardir = subj_info['data_folder'][0]

        if args.compare_with_pardir:
            # Do evaluation on authenticity/composition.
            subject_gt_dir = os.path.join(args.compare_with_pardir, subject_name)
            command_line += f" --compare_with {subject_gt_dir} --calc_face_sim"

            if isinstance(args.ref_images, (list, tuple)):
                args.ref_images = " ".join(args.ref_images)
            elif args.ref_images is None:
                command_line += f" --ref_images {subject_gt_dir}"
            else:
                command_line += f" --ref_images {args.ref_images}"
                        
        if args.n_rows > 0:
            command_line += f" --n_rows {args.n_rows}"
        
        if args.diffusers:
            command_line += f" --diffusers"     
        command_line += f" --method {args.method}"

        print(command_line)
        if not args.dryrun:
            os.system(command_line)

    if args.scores_csv is not None and not args.dryrun:
        if not os.path.exists(args.scores_csv):
            print(f"Error: {args.scores_csv} not found.")
            exit(0)

        print(f"Scores are saved to {args.scores_csv}")
        # Read the scores and print the average.
        csv_reader = csv.reader(open(args.scores_csv))
        scores = []
        for row in csv_reader:
            emb_sig, sims_face_avg, sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent = row
            print(f"{emb_sig}:\t{sims_face_avg}\t{sims_img_avg}\t{sims_text_avg}\t{sims_dino_avg}\t{except_img_percent}")

            sims_face_avg, sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent = \
                float(sims_face_avg), float(sims_img_avg), float(sims_text_avg), float(sims_dino_avg), float(except_img_percent)
            scores.append( [sims_face_avg, sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent] )

        if len(scores) == 0:
            print(f"Error: no scores found in {args.scores_csv}.")
        else:
            scores = np.array(scores)
            sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent = np.mean(scores[:, 1:], axis=0)
            if np.sum(scores[:, 0] > 0) > 0:
                # Skip 0 face similarity scores, as they are probably not on humans.
                sims_face_avg = np.mean(scores[:, 0][scores[:, 0] > 0])
            else:
                sims_face_avg = 0

            print(f"All subjects mean face/image/text/dino sim: {sims_face_avg:.3f} {sims_img_avg:.3f} {sims_text_avg:.3f} {sims_dino_avg:.3f}")
            print(f"Face exception: {except_img_percent*100:.1f}%")

