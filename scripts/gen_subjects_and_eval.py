import argparse
import os
import re
import numpy as np
import csv
from evaluation.eval_utils import parse_range_str, format_prompt_list, find_first_match, reformat_z_affix
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
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--tfgpu", type=int, default=argparse.SUPPRESS, 
                        help="ID of GPU to use for TensorFlow. Set to -1 to use CPU (slow).")

    parser.add_argument("--method", default='ada', choices=["ada", "static-layerwise", "ti", "db"], type=str, 
                        help="method for generating samples")
    parser.add_argument("--subject_string", type=str, default="z", 
                        help="Subject placeholder string that represents the subject in prompts")
    parser.add_argument("--num_vectors_per_subj_token",
                        type=int, default=argparse.SUPPRESS,
                        help="Number of vectors per token. If > 1, use multiple embeddings to represent a subject.")

    parser.add_argument("--include_bg_string", 
                        action="store_true", 
                        help="Whether to include the background string in the prompts. Only valid if --background_string is specified.")
    parser.add_argument("--background_string", 
                        type=str, default="y",
                        help="Background string which will be used in prompts to denote the background in training images.")
    parser.add_argument("--num_vectors_per_bg_token",
                        type=int, default=4,
                        help="Number of vectors for the background token. If > 1, use multiple embeddings to represent the background.")
 
    parser.add_argument("--zeroshot", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use zero-shot learning")  
    parser.add_argument("--no_id_emb", action="store_true",
                        help="Do not use face/DINO embeddings for zero-shot generation")
      
    parser.add_argument("--ref_images", type=str, nargs='+', default=None,
                        help="Reference image for zero-shot learning")

    parser.add_argument("--prompt_set", dest='prompt_set_name', type=str, default='all', 
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
    parser.add_argument("--z_prefix_type", type=str, default=argparse.SUPPRESS,
                        help="Prefix to prepend to z")
    parser.add_argument("--use_fp_trick", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use the 'face portrait' trick for the subject")
    # Possible z_suffix_type: '' (none), 'class_name', or any user-specified string.
    parser.add_argument("--z_suffix_type", type=str, default=argparse.SUPPRESS, 
                        help="Append this string to the subject placeholder string during inference "
                             "(default: '' for humans/animals, 'class_name' for others)")
    # extra_z_suffix: usually reduces the similarity of generated images to the real images.
    parser.add_argument("--extra_z_suffix", type=str, default="",
                        help="Extra suffix to append to the z suffix")
    parser.add_argument("--prompt_prefix", type=str, default="",
                        help="prefix to prepend to each prompt")
    # prompt_suffix: usually reduces the similarity.
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="suffix to append to the end of each prompt")
    parser.add_argument("--scale", type=float, nargs='+', default=[4, 1],
                        help="the guidance scale")
    parser.add_argument(
        "--use_first_gt_img_as_init",
        action='store_true',
        help="use the first image in the groundtruth folder as the initial image",
    )
    # Anything between 0 and 1 will cause blended images.
    parser.add_argument(
        "--init_img_weight",
        type=float,
        default=0.1,
        help="Weight of the initial image (if w, then w*img + (1-w)*noise)",
    )

    parser.add_argument("--n_samples", type=int, default=-1, 
                        help="number of samples to generate for each test case")
    parser.add_argument("--bs", type=int, default=-1, 
                        help="batch size")
    
    parser.add_argument("--steps", type=int, default=50, 
                        help="number of DDIM steps to generate samples")
    parser.add_argument("--ckpt_dir", type=str, default="logs",
                        help="parent directory containing checkpoints of all subjects")
    parser.add_argument("--ckpt_iter", type=str, default=None,
                        help="checkpoint iteration to use")
    parser.add_argument("--ckpt_sig", dest='ckpt_extra_sig', type=str, default="",
                        help="Extra signature that is part of the checkpoint directory name."
                             " Could be a regular expression.")
    parser.add_argument("--ckpt_name", type=str, default=argparse.SUPPRESS,
                        help="CKPT folder name.")
    
    parser.add_argument("--out_dir_tmpl", type=str, default="samples-dbeval",
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
    parser.add_argument("--subjfile", type=str, default="evaluation/info-subjects.sh", 
                        help="subject info script file")
    # The range of indices of subjects to generate
    parser.add_argument("--range", type=str, default=None, 
                        help="Range of subjects to generate (Index starts from 1 and is inclusive, e.g., 1-30)")
    parser.add_argument("--selset", action="store_true",
                        help="Whether to generate only the selected subset of subjects")
    parser.add_argument("--skipselset", action="store_true",
                        help="Whether to generate all subjects except the selected subset")
    
    parser.add_argument("--compare_with_pardir", type=str, default=argparse.SUPPRESS,
                        help="Parent folder of subject images used for computing similarity with generated samples")

    parser.add_argument("--bb_type", type=str, default="v15-dste8-vae", 
                        choices=["v14", "v15", "v15-ema", "v15-dste", "v15-dste8", "v15-dste8-vae", "v15-arte", "v15-rvte",
                                 "dreamshaper-v5", "dreamshaper-v6", "dreamshaper-v8", "ar-v16", "rv-v4"],
                        help="Type of checkpoints to use (default: v15-dste8-vae)")
    # [1, 1] will be normalized to [0.5, 0.5] internally.
    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[1, 1],
                        help="Weight of the skip connections of the last few layers of CLIP text embedder. " 
                             "NOTE: the last element is the weight of the last layer.")
    
    parser.add_argument("--is_face", type=str2bool, const=True, default=argparse.SUPPRESS, nargs="?",
                        help="Whether the generated samples are human faces",
    )    

    parser.add_argument("--prompt_mix_scheme", type=str, default=argparse.SUPPRESS, 
                        choices=['mix_hijk', 'mix_concat_cls'],
                        help="How to mix the subject prompt with the reference prompt")

    parser.add_argument("--prompt_mix_weight", type=float, default=0,
                        help="Weight of the reference prompt to be mixed with the subject prompt")  
    # --dryrun
    parser.add_argument("--dryrun", action="store_true",
                        help="Dry run: only print the commands without actual execution")
    # --debug
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")

    args = parser.parse_args()
    return args, parser

if __name__ == "__main__":
    
    args, argparser = parse_args()
    subj_info, subj2attr = parse_subject_file(args.subjfile)
    subjects, class_names, broad_classes, sel_set, ckpt_iters = \
            subj_info['subjects'], subj_info['class_names'], subj_info['broad_classes'], subj_info['sel_set'], subj_info['maxiters']

    args.orig_placeholder = args.subject_string
    # If num_vectors_per_subj_token == 3:
    # "z"    => "z, , "
    # Need to leave a space between multiple ",,", otherwise they are treated as one token.
    if hasattr(args, 'num_vectors_per_subj_token') and args.num_vectors_per_subj_token > 1:
        args.subject_string += ", " * (args.num_vectors_per_subj_token - 1)

    if hasattr(args, 'is_face'):
        are_faces = [args.is_face] * len(subjects)
    elif 'are_faces' in subj_info:
        are_faces = subj_info['are_faces']
    else:
        are_faces = [False] * len(subjects)

    cls_delta_strings = subj_info['cls_delta_strings']

    subject_indices = list(range(len(subjects)))
    if args.selset:
        subject_indices = sel_set

    range_indices   = parse_range_str(args.range)
    if range_indices is not None:
        subject_indices = [ subject_indices[i] for i in range_indices ]

    # Convert the list of weights to a string to be accepted by the command line.
    args.clip_last_layers_skip_weights = " ".join([ str(w) for w in args.clip_last_layers_skip_weights ])

    if args.method == 'db':
        # For DreamBooth, use_z_suffix is the default.
        args.z_suffix_type = 'cls_delta_string'

    all_ckpts = os.listdir(args.ckpt_dir)
    # Sort all_ckpts by name (actually by timestamp in the name), so that most recent first.
    all_ckpts.sort(reverse=True)

    if args.scores_csv is None:
        args.scores_csv = f"{args.method}-{args.range}.csv"

    # Ty to create scores_csv file. If it already exists, make it empty.
    SCORES_CSV_FILE = open(args.scores_csv, "w")
    SCORES_CSV_FILE.close()

    for subject_idx in subject_indices:
        if args.skipselset and subject_idx in sel_set:
            continue
        subject_name        = subjects[subject_idx]
        class_name          = class_names[subject_idx]
        broad_class         = broad_classes[subject_idx]
        is_face             = are_faces[subject_idx]
        cls_delta_string    = cls_delta_strings[subject_idx]

        print("Generating samples for subject: " + subject_name)
        z_prefix = reformat_z_affix(args, 'z_prefix_type', broad_class, class_name, cls_delta_string)
        z_suffix = reformat_z_affix(args, 'z_suffix_type', broad_class, class_name, cls_delta_string)

        if len(args.extra_z_suffix) > 0:
            z_suffix += " " + args.extra_z_suffix + ","

        if args.background_string and args.include_bg_string:
            background_string = "with background " + args.background_string + ", " * (args.num_vectors_per_bg_token - 1)
            bg_suffix = "-bg"
        else:
            background_string = ""
            bg_suffix = ""

        outdir = args.out_dir_tmpl + "-" + args.method
        os.makedirs(outdir, exist_ok=True)

        if args.prompt is None:
            if args.n_samples == -1:
                args.n_samples = 4
            if args.bs == -1:
                args.bs = 4
            # Usually cls_delta_string == class_name. But we can specify more fine-grained cls_delta_string.
            # E.g., format_prompt_list(placeholder="z", z_suffix="cat", cls_delta_string="tabby cat", broad_class=1)
            prompt_list, class_short_prompt_list, class_long_prompt_list = \
                format_prompt_list(args.subject_string, z_prefix, z_suffix, background_string, 
                                    class_name, cls_delta_string, 
                                    broad_class, args.prompt_set_name, args.use_fp_trick)
            prompt_filepath = f"{outdir}/{subject_name}-prompts-{args.prompt_set_name}{bg_suffix}-{args.num_vectors_per_subj_token}.txt"
            PROMPTS = open(prompt_filepath, "w")
        else:
            if args.n_samples == -1:
                args.n_samples = 8
            if args.bs == -1:
                args.bs = 8

            subject_string = args.subject_string
            if len(z_prefix) > 0:
                subject_string      = z_prefix + " " + subject_string
                class_name          = z_prefix + " " + class_name
                cls_delta_string    = z_prefix + " " + cls_delta_string

            prompt_tmpl = args.prompt if args.prompt != "" else "a {}"
            prompt = prompt_tmpl.format(subject_string + z_suffix)
            class_long_prompt  = prompt_tmpl.format(cls_delta_string + z_suffix)
            class_short_prompt = prompt_tmpl.format(class_name + z_suffix)
            # If --background_string is not specified, background_string is "".
            # Only add the background_string to prompt used for image generation,
            # not to class_long_prompt/class_short_prompt used for CLIP text/image similarity evaluation.
            prompt = prompt + background_string

            prompt_list             = [ prompt ]
            class_short_prompt_list = [ class_short_prompt ]
            class_long_prompt_list  = [ class_long_prompt ]

        print(subject_name, ":")

        for prompt, class_short_prompt, class_long_prompt in zip(prompt_list, class_short_prompt_list, class_long_prompt_list):
            if len(args.prompt_prefix) > 0:
                prompt = args.prompt_prefix + " " + prompt
                class_short_prompt = args.prompt_prefix + " " + class_short_prompt
                class_long_prompt  = args.prompt_prefix + " " + class_long_prompt
            if len(args.prompt_suffix) > 0:
                prompt = prompt + ", " + args.prompt_suffix
                class_short_prompt = class_short_prompt + ", " + args.prompt_suffix
                class_long_prompt  = class_long_prompt  + ", " + args.prompt_suffix

            print("  ", prompt)
            if args.prompt is None:
                indiv_subdir = subject_name + "-" + prompt.replace(" ", "-")
                # In case some folder names are extremely long, truncate to 80 characters.
                indiv_subdir = indiv_subdir[:80]
                # class_short_prompt, class_long_prompt are saved in the prompt file as well.
                # class_long_prompt is used for the CLIP text/image similarity evaluation.
                # class_short_prompt is not used.
                PROMPTS.write( "\t".join([str(args.n_samples), indiv_subdir, prompt, class_long_prompt, class_short_prompt ]) + "\n" )
            else:
                indiv_subdir = subject_name

        if args.gen_prompt_set_only:
            print(f"{len(prompt_list)} prompts saved to {prompt_filepath}")
            continue

        if hasattr(args, 'ckpt_name'):
            ckpt_name = args.ckpt_name
        else:
            ckpt_sig   = subject_name + "-" + args.method
            # Find the newest checkpoint that matches the subject name.
            ckpt_name  = find_first_match(all_ckpts, ckpt_sig, args.ckpt_extra_sig)

        if ckpt_name is None:
            print("ERROR: No checkpoint found for subject: " + subject_name)
            continue
            # breakpoint()

        if args.method == 'db':
            config_file = "v1-inference-db.yaml"
            ckpt_path   = f"{args.ckpt_dir}/{ckpt_name}/checkpoints/last.ckpt"
            bb_type = ""
        else:
            config_file = "v1-inference-" + args.method + ".yaml"
            if args.bb_type == 'v15-ema':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-pruned-emaonly.ckpt"
            elif args.bb_type == 'v15-dste':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-dste.ckpt"
            elif args.bb_type == 'v15-dste8':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-dste8.ckpt"
            elif args.bb_type == 'v15-dste8-vae':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt"
            elif args.bb_type == 'v15-arte':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-arte.ckpt"
            elif args.bb_type == 'v15-rvte':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-rvte.ckpt"
            elif args.bb_type == 'dreamshaper-v5':
                ckpt_path   = "models/dreamshaper/dreamshaper_5BakedVae.safetensors"
            elif args.bb_type == 'dreamshaper-v6':
                ckpt_path   = "models/dreamshaper/dreamshaper_631BakedVae.safetensors"
            elif args.bb_type == 'dreamshaper-v8':
                ckpt_path   = "models/dreamshaper/DreamShaper_8_pruned.safetensors"
            elif args.bb_type == 'ar-v16':
                ckpt_path   = "models/absolutereality/ar-v1-6.safetensors"
            elif args.bb_type == 'rv-v4':
                ckpt_path   = "models/realisticvision/realisticVisionV40_v40VAE.safetensors"
            elif args.bb_type == 'v14':
                ckpt_path   = "models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt"
            elif args.bb_type == 'v15':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-pruned.ckpt"
            else:
                print(f"ERROR: Unknown bb_type: {args.bb_type}")
                exit(0)

            # bb_type is used to tell stable_txt2img.py what suffix to put in the output image name.
            # If bb_type is the default value, then no suffix is appended. So specify as "" here.
            if args.bb_type == argparser.get_default('bb_type'):
                bb_type = ""
            else:
                bb_type = args.bb_type

            if args.ckpt_iter is None:
                ckpt_iter = ckpt_iters[broad_class]
            else:
                ckpt_iter = args.ckpt_iter

            emb_path    = f"{args.ckpt_dir}/{ckpt_name}/checkpoints/embeddings_gs-{ckpt_iter}.pt"
            if not os.path.exists(emb_path):
                emb_path2 = f"{args.ckpt_dir}/{ckpt_name}/embeddings_gs-{ckpt_iter}.pt"
                if not os.path.exists(emb_path2):
                    print(f"ERROR: Subject embedding not found: '{emb_path}' or '{emb_path2}'")
                    continue

                emb_path = emb_path2

        if isinstance(args.scale, (list, tuple)):
            args.scale = " ".join([ str(s) for s in args.scale ])

        command_line = f"python3 scripts/stable_txt2img.py --config configs/stable-diffusion/{config_file} --ckpt {ckpt_path} --bb_type '{bb_type}' --ddim_eta 0.0 --ddim_steps {args.steps} --gpu {args.gpu} --scale {args.scale} --broad_class {broad_class} --n_repeat 1 --bs {args.bs} --outdir {outdir}"
        if hasattr(args, 'tfgpu'):
            command_line += f" --tfgpu {args.tfgpu}"
            
        if args.prompt is None:
            PROMPTS.close()
            print(f"{len(prompt_list)} prompts saved to {prompt_filepath}")
            # Since we use a prompt file, we don't need to specify --n_samples.
            command_line += f" --from_file {prompt_filepath}"
        else:
            # Do not use a prompt file, but specify --n_samples, --prompt, and --indiv_subdir.
            # Note only the last prompt/class_prompt will be used. 
            command_line += f" --n_samples {args.n_samples} --indiv_subdir {indiv_subdir}"
            command_line += f" --prompt \"{prompt}\" --class_prompt \"{class_long_prompt}\""

            if hasattr(args, 'prompt_mix_scheme'):
                # Use the class_short_prompt as the reference prompt, 
                # as it's tokenwise aligned with the subject prompt.
                command_line += f" --ref_prompt \"{class_short_prompt}\""

        if args.scores_csv is not None:
            command_line += f" --scores_csv {args.scores_csv}"

        # prompt_mix_weight may < 0, in which case we enhance the expression of the subject.
        if hasattr(args, 'prompt_mix_scheme'):
            # Only specify the flags here. The actual reference prompt will be read from the prompt file.
            command_line += f" --prompt_mix_scheme {args.prompt_mix_scheme} --prompt_mix_weight {args.prompt_mix_weight}"

        if args.method != 'db':
            command_line += f" --embedding_paths {emb_path}"

        command_line += f" --clip_last_layers_skip_weights {args.clip_last_layers_skip_weights}"

        if args.zeroshot:            
            command_line += f" --zeroshot"
            if args.no_id_emb:
                command_line += f" --no_id_emb"

        if hasattr(args, 'num_vectors_per_subj_token'):
            command_line += f" --subject_string {args.orig_placeholder} --num_vectors_per_subj_token {args.num_vectors_per_subj_token}"
        if hasattr(args, 'num_vectors_per_bg_token'):
            command_line += f" --background_string {args.background_string} --num_vectors_per_bg_token {args.num_vectors_per_bg_token}"

        if hasattr(args, 'neg_prompt'):
            command_line += f" --neg_prompt \"{args.neg_prompt}\""
        elif hasattr(args, 'use_pre_neg_prompt'):
            command_line += f" --use_pre_neg_prompt 1"

        if args.use_first_gt_img_as_init:
            command_line += f" --use_first_gt_img_as_init --init_img_weight {args.init_img_weight}"

        if not hasattr(args, 'compare_with_pardir') and 'data_folder' in subj_info:
            args.compare_with_pardir = subj_info['data_folder'][0]

        if args.compare_with_pardir:
            # Do evaluation on authenticity/composition.
            subject_gt_dir = os.path.join(args.compare_with_pardir, subject_name)
            command_line += f" --compare_with {subject_gt_dir}"

            if is_face:
                # Tell stable_txt2img.py to do face-specific evaluation.
                command_line += f" --calc_face_sim"

            if args.zeroshot:
                if isinstance(args.ref_images, (list, tuple)):
                    args.ref_images = " ".join(args.ref_images)
                elif args.ref_images is None:
                    command_line += f" --ref_images {subject_gt_dir}"
                else:
                    command_line += f" --ref_images {args.ref_images}"
                            
        if args.n_rows > 0:
            command_line += f" --n_rows {args.n_rows}"
            
        if args.debug:
            command_line += f" --debug"
            
        print(command_line)
        if not args.dryrun:
            os.system(command_line)

    if args.gen_prompt_set_only:
        exit(0)

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

