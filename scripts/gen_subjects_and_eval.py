import argparse
import os
import re
import numpy as np
import csv
from evaluation.eval_utils import parse_subject_file, parse_range_str, get_prompt_list, find_first_match

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
    parser.add_argument("--method", default='ada', choices=["ada", "static-layerwise", "ti", "db"], type=str, 
                        help="method to use for generating samples")
    parser.add_argument("--placeholder", type=str, default="z", 
                        help="placeholder token for the subject")
    parser.add_argument("--num_vectors_per_token",
                        type=int, default=argparse.SUPPRESS,
                        help="Number of vectors per token. If > 1, use multiple embeddings to represent a subject.")
                        
    parser.add_argument("--prompt_set", type=str, default='all', choices=['all', 'hard'],
                        help="Subset of prompts to evaluate if --prompt is not specified")
    
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to use for generating samples, using {} for placeholder (default: None)")
    # Possible z_suffix_type: '' (none), 'db_prompt', 'class_token', or any user-specified string.
    parser.add_argument("--z_suffix_type", default=argparse.SUPPRESS, 
                        help="Append this string to the subject placeholder token during inference "
                             "(default: '' for humans/animals, 'class_token' for others)")
    # extra_z_suffix: usually reduces the similarity of generated images to the real images.
    parser.add_argument("--extra_z_suffix", type=str, default="",
                        help="Extra suffix to append to the z suffix")
    parser.add_argument("--z_prefix", type=str, default=argparse.SUPPRESS,
                        help="Prefix to prepend to z")
    parser.add_argument("--prompt_prefix", type=str, default="",
                        help="prefix to prepend to each prompt")
    # prompt_suffix: usually reduces the similarity.
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="suffix to append to the end of each prompt")
    parser.add_argument("--scale", type=float, default=5, 
                        help="the guidance scale")

    parser.add_argument("--n_samples", type=int, default=-1, 
                        help="number of samples to generate for each test case")
    parser.add_argument("--bs", type=int, default=-1, 
                        help="batch size")
    
    parser.add_argument("--steps", type=int, default=50, 
                        help="number of DDIM steps to generate samples")
    parser.add_argument("--ckpt_dir", type=str, default="logs",
                        help="parent directory containing checkpoints of all subjects")
    parser.add_argument("--ckpt_iter", type=int, default=-1,
                        help="checkpoint iteration to use")
    parser.add_argument("--ckpt_sig", dest='ckpt_extra_sig', type=str, default="",
                        help="Extra signature that is part of the checkpoint directory name."
                             " Could be a regular expression.")
    
    parser.add_argument("--out_dir_tmpl", type=str, default="samples-dbeval",
                        help="Template of parent directory to save generated samples")
    parser.add_argument("--scores_csv", type=str, default=None,
                        help="CSV file to save the evaluation scores")
    
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
    
    parser.add_argument("--compare_with_pardir", type=str, default=None,
                        help="Parent folder of subject images used for computing similarity with generated samples")

    parser.add_argument("--bb_type", type=str, default="v15-dste", 
                        choices=["v14", "v15", "v15-ema", "v15-dste", "v15-arte", "v15-rvte",
                                 "dreamshaper-v5", "dreamshaper-v6", "ar-v16", "rv-v4"],
                        help="Type of checkpoints to use (default: v15)")

    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[0.5],
                        help="Weight of the skip connection between the last layer and second last layer of CLIP text embedder")
    
    parser.add_argument("--is_face", type=str2bool, const=True, default=argparse.SUPPRESS, nargs="?",
                        help="Whether the generated samples are human faces",
    )    

    parser.add_argument("--prompt_mix_scheme", type=str, default=argparse.SUPPRESS, 
                        choices=['mix_hijk', 'mix_concat_cls'],
                        help="How to mix the subject prompt with the reference prompt")

    parser.add_argument("--prompt_mix_weight", type=float, default=0,
                        help="Weight of the reference prompt to be mixed with the subject prompt")  

    parser.add_argument("--ada_emb_weight",
                        type=float, default=-1,
                        help="Weight of ada embeddings (in contrast to static embeddings)")


    args = parser.parse_args()
    return args, parser

if __name__ == "__main__":
    
    args, argparser = parse_args()
    vars = parse_subject_file(args.subjfile, args.method)
    subjects, class_tokens, broad_classes, sel_set, ckpt_iters = \
            vars['subjects'], vars['class_tokens'], vars['broad_classes'], vars['sel_set'], vars['maxiters']

    # If num_vectors_per_token == 3:
    # "z"    => "z, , "
    # Need to leave a space between multiple ",,", otherwise they are treated as one token.
    if hasattr(args, 'num_vectors_per_token') and args.num_vectors_per_token > 1:
        args.placeholder += ", " * (args.num_vectors_per_token - 1)

    if hasattr(args, 'z_prefix'):
        # * 3 for 3 broad classes, i.e., all classes use the same args.z_prefix.
        z_prefixes = [args.z_prefix] * 3    
    elif 'z_prefixes' in vars and args.prompt is None:
        # Use z_prefixes from the subject info file if it exists, 
        # but only if it's not manual prompt generation
        z_prefixes = vars['z_prefixes']
        assert len(z_prefixes) == 3
    else:
        z_prefixes = [""] * 3

    if hasattr(args, 'is_face'):
        are_faces = [args.is_face] * len(subjects)
    elif 'are_faces' in vars:
        are_faces = vars['are_faces']
    else:
        are_faces = [False] * len(subjects)

    # db_prompts are phrases, and ada_prompts are multiple individual words.
    # So db_prompts better suit the CLIP text/image matching.
    class_long_tokens = vars['db_prompts']

    subject_indices = list(range(len(subjects)))
    if args.selset:
        subject_indices = sel_set

    range_indices   = parse_range_str(args.range)
    if range_indices is not None:
        subject_indices = [ subject_indices[i] for i in range_indices ]

    if args.method == 'db':
        # For DreamBooth, use_z_suffix is the default.
        args.z_suffix_type = 'db_prompt'

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
        class_token         = class_tokens[subject_idx]
        broad_class         = broad_classes[subject_idx]
        is_face             = are_faces[subject_idx]
        class_long_token    = class_long_tokens[subject_idx]
        z_prefix            = z_prefixes[broad_class]

        print("Generating samples for subject: " + subject_name)

        ckpt_sig   = subject_name + "-" + args.method
        # Find the newest checkpoint that matches the subject name.
        ckpt_name  = find_first_match(all_ckpts, ckpt_sig, args.ckpt_extra_sig)

        if ckpt_name is None:
            print("ERROR: No checkpoint found for subject: " + subject_name)
            continue
            # breakpoint()

        if not hasattr(args, 'z_suffix_type'):
            if broad_class == 1:
                # For ada/TI, if not human faces / animals, and z_suffix_type is not specified, 
                # then use class_token as the z suffix, to make sure the subject is always expressed.
                z_suffix_type = '' 
            else:
                z_suffix_type = 'class_token'
        else:
            z_suffix_type = args.z_suffix_type

        if z_suffix_type == 'db_prompt':
            # DreamBooth always uses db_prompt as z_suffix.
            z_suffix = " " + class_long_token
        elif z_suffix_type == 'class_token':
            # For Ada/TI, if we append class token to "z" -> "z dog", 
            # the chance of occasional under-expression of the subject may be reduced.
            # (This trick is not needed for human faces)
            # Prepend a space to class_token to avoid "a zcat" -> "a z cat"
            z_suffix = " " + class_token
        else:
            # z_suffix_type contains the actual z_suffix.
            if re.match(r"^[a-zA-Z0-9_]", z_suffix_type):
                # If z_suffix_type starts with a word, prepend a space to avoid "a zcat" -> "a z cat"
                z_suffix = " " + z_suffix_type
            else:
                # z_suffix_type starts with a punctuation, e.g., ",".
                z_suffix = z_suffix_type

        if len(args.extra_z_suffix) > 0:
            z_suffix += " " + args.extra_z_suffix + ","

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
            elif args.bb_type == 'v15-arte':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-arte.ckpt"
            elif args.bb_type == 'v15-rvte':
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-rvte.ckpt"
            elif args.bb_type == 'dreamshaper-v5':
                ckpt_path   = "models/dreamshaper/dreamshaper_5BakedVae.safetensors"
            elif args.bb_type == 'dreamshaper-v6':
                ckpt_path   = "models/dreamshaper/dreamshaper_631BakedVae.safetensors"
            elif args.bb_type == 'ar-v16':
                ckpt_path   = "models/absolutereality/ar-v1-6.safetensors"
            elif args.bb_type == 'rv-v4':
                ckpt_path   = "models/realisticvision/realisticVisionV40Fp16.cTYR.safetensors"
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

            if args.ckpt_iter == -1:
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

        outdir = args.out_dir_tmpl + "-" + args.method
        os.makedirs(outdir, exist_ok=True)

        if args.prompt is None:
            if args.n_samples == -1:
                args.n_samples = 4
            if args.bs == -1:
                args.bs = 4
            # E.g., get_prompt_list(placeholder="z", z_suffix="cat", class_long_token="tabby cat", broad_class=1)
            prompt_list, class_short_prompt_list, class_long_prompt_list = \
                get_prompt_list(args.placeholder, z_prefix, z_suffix, class_token, class_long_token, 
                               broad_class, args.prompt_set)
            prompt_filepath = f"{outdir}/{subject_name}-prompts-{args.prompt_set}.txt"
            PROMPTS = open(prompt_filepath, "w")
        else:
            if args.n_samples == -1:
                args.n_samples = 8
            if args.bs == -1:
                args.bs = 8

            placeholder = args.placeholder
            if len(z_prefix) > 0:
                placeholder      = z_prefix + " " + placeholder
                class_token      = z_prefix + " " + class_token
                class_long_token = z_prefix + " " + class_long_token

            prompt_tmpl = args.prompt if args.prompt != "" else "a {}"
            prompt = prompt_tmpl.format(placeholder + z_suffix)
            class_long_prompt = prompt_tmpl.format(class_long_token + z_suffix)
            class_short_prompt = prompt_tmpl.format(class_token + z_suffix)
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

        command_line = f"python3 scripts/stable_txt2img.py --config configs/stable-diffusion/{config_file} --ckpt {ckpt_path} --bb_type '{bb_type}' --ddim_eta 0.0 --ddim_steps {args.steps} --gpu {args.gpu} --scale {args.scale} --broad_class {broad_class} --n_repeat 1 --bs {args.bs} --outdir {outdir}"

        if args.prompt is None:
            PROMPTS.close()
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
        
        if args.ada_emb_weight != -1:
            command_line += f" --ada_emb_weight {args.ada_emb_weight}"
            
        if args.method != 'db':
            command_line += f" --embedding_paths {emb_path}"

        command_line += f" --clip_last_layers_skip_weights {args.clip_last_layers_skip_weights}"

        if hasattr(args, 'num_vectors_per_token'):
            command_line += f" --num_vectors_per_token {args.num_vectors_per_token}"
            
        if args.compare_with_pardir:
            # Do evaluation on authenticity/composition.
            subject_gt_dir = os.path.join(args.compare_with_pardir, subject_name)
            command_line += f" --compare_with {subject_gt_dir}"

            if is_face:
                # Tell stable_txt2img.py to do face-specific evaluation.
                command_line += f" --calc_face_sim"

        print(command_line)
        os.system(command_line)

    if args.scores_csv is not None:
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

