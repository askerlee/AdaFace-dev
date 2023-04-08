import argparse
import os
from scripts.eval_utils import parse_subject_file, parse_range_str, get_promt_list, find_first_match

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance_scale", type=float, default=10, help="guidance scale")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--method", default='ada', choices=["ada", "ti", "db"], type=str, 
                        help="method to use for generating samples")
    parser.add_argument("--placeholder", type=str, default="z", 
                        help="placeholder token for the subject")
    # Possible z_suffix_type: '' (none), 'db_prompt', 'class_token', or any user-specified string.
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to use for generating samples, using {} for placeholder (default: None)")
    parser.add_argument("--z_suffix_type", default="", 
                        help="Append this string to the subject placeholder token during inference (default: none)")
    # extra_z_suffix: usually reduces the similarity of generated images to the real images.
    parser.add_argument("--extra_z_suffix", type=str, default="",
                        help="Extra suffix to append to the z suffix")
    parser.add_argument("--z_prefix", type=str, default="",
                        help="Prefix to prepend to z")
    # prompt_suffix: usually reduces the similarity.
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="suffix to append to the end of each prompt")
    parser.add_argument("--plain", action="store_true",
                        help="Whether to generate plain samples without compositional prompts")
    parser.add_argument("--ref_prompt_mix_weight", type=float, default=0,
                        help="Weight of the reference prompt to be mixed with the subject prompt")    
    parser.add_argument("--scale", type=float, default=5, 
                        help="the guidance scale")
    # subj_scale: sometimes it improves the similarity, somtimes it reduces it.
    parser.add_argument("--subj_scale", type=float, default=1,
                        help="the subject embedding scale")
    
    parser.add_argument("--n_samples", type=int, default=-1, 
                        help="number of samples to generate for each test case")
    parser.add_argument("--bs", type=int, default=-1, 
                        help="batch size")
    
    parser.add_argument("--steps", type=int, default=50, 
                        help="number of DDIM steps to generate samples")
    parser.add_argument("--ckpt_dir", type=str, default="logs",
                        help="parent directory containing checkpoints of all subjects")
    parser.add_argument("--ckpt_iter", type=int, default=4000,
                        help="checkpoint iteration to use")
    parser.add_argument("--ckpt_sig", dest='ckpt_extra_sig', type=str, default="",
                        help="Extra signature that is part of the checkpoint directory name."
                             " Could be a regular expression.")
    
    parser.add_argument("--out_dir_tmpl", type=str, default="samples-dbeval",
                        help="Template of parent directory to save generated samples")

    # File path containing composition case information
    parser.add_argument("--subject_file", type=str, default="scripts/info-db-eval-subjects.sh", 
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
    
    parser.add_argument("--v14", dest='v15', action="store_false",
                        help="Whether to use v1.4 model (default: v1.5)")

    parser.add_argument("--clip_last_layer_skip_weight", type=float, default=-1,
                        help="Weight of the skip connection between the last layer and second last layer of CLIP text embedder")
    parser.add_argument("--clip_last_layer_skip_scheme", type=str, choices=["add", "concat"], 
                        default="add", 
                        help="Scheme for the skip connection between the last layer and second last layer of CLIP text embedder")
    parser.add_argument("--is_face", action="store_true",
                        help="Whether the generated samples are human faces")
                                                                    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    vars = parse_subject_file(args.subject_file, args.method)
    subjects, class_tokens, broad_classes, sel_set = vars['subjects'], vars['class_tokens'], \
                                                     vars['broad_classes'], vars['sel_set']
    # db_prompts are phrases, and ada_prompts are multiple individual words.
    # So db_prompts better suit the CLIP text/image matching.
    class_long_tokens = vars['db_prompts']

    if args.method == 'db':
        # For DreamBooth, use_z_suffix is the default.
        args.z_suffix_type = 'db_prompt'
    elif not args.is_face and args.z_suffix_type == '':
        # For ada/TI, if not human faces, and z_suffix_type is not specified, 
        # then use class_token as the z suffix.
        args.z_suffix_type = 'class_token'

    subject_indices = list(range(len(subjects)))
    if args.selset:
        subject_indices = sel_set

    range_indices   = parse_range_str(args.range)
    if range_indices is not None:
        subject_indices = [ subject_indices[i] for i in range_indices ]

    all_ckpts = os.listdir(args.ckpt_dir)
    # Sort all_ckpts by modification time, most recent first.
    all_ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(args.ckpt_dir, x)), reverse=True)

    for subject_idx in subject_indices:
        if args.skipselset and subject_idx in sel_set:
            continue
        subject_name = subjects[subject_idx]
        class_token  = class_tokens[subject_idx]
        broad_class  = broad_classes[subject_idx]
        class_long_token = class_long_tokens[subject_idx]
        print("Generating samples for subject: " + subject_name)

        ckpt_sig   = subject_name + "-" + args.method
        # Find the newest checkpoint that matches the subject name.
        ckpt_name  = find_first_match(all_ckpts, ckpt_sig, args.ckpt_extra_sig)

        if ckpt_name is None:
            print("ERROR: No checkpoint found for subject: " + subject_name)
            continue
            # breakpoint()

        if args.z_suffix_type == 'db_prompt':
            # DreamBooth always uses db_prompt as z_suffix.
            z_suffix = " " + class_long_token
        elif args.z_suffix_type == 'class_token':
            # For Ada/TI, if we append class token to "z" -> "z dog", 
            # the chance of occasional under-expression of the subject may be reduced.
            # (This trick is not needed for human faces)
            # Prepend a space to class_token to avoid "a zcat" -> "a z cat"
            z_suffix = " " + class_token
        else:
            z_suffix = " " + args.z_suffix_type

        if len(args.extra_z_suffix) > 0:
            z_suffix += " " + args.extra_z_suffix + ","
        
        if len(args.z_prefix) > 0:
            placeholder = args.z_prefix + " " + args.placeholder
        else:
            placeholder = args.placeholder

        if args.method == 'db':
            config_file = "v1-inference.yaml"
            ckpt_path   = f"logs/{ckpt_name}/checkpoints/last.ckpt"
        else:
            config_file = "v1-inference-" + args.method + ".yaml"
            if args.v15:
                ckpt_path   = "models/stable-diffusion-v-1-5/v1-5-pruned.ckpt"
            else:
                ckpt_path   = "models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt"
            emb_path    = f"logs/{ckpt_name}/checkpoints/embeddings_gs-{args.ckpt_iter}.pt"

        outdir = args.out_dir_tmpl + "-" + args.method
        os.makedirs(outdir, exist_ok=True)

        if not args.plain:
            if args.n_samples == -1:
                args.n_samples = 4
            if args.bs == -1:
                args.bs = 4
            # E.g., get_promt_list(placeholder="z", z_suffix="cat", class_long_token="tabby cat", broad_class=1)
            prompt_list, class_prompt_list = get_promt_list(placeholder, z_suffix, class_long_token, broad_class)
            prompt_filepath = f"{outdir}/{subject_name}-prompts.txt"
            PROMPTS = open(prompt_filepath, "w")
        else:
            if args.n_samples == -1:
                args.n_samples = 8
            if args.bs == -1:
                args.bs = 8

            prompt             = args.prompt.format("z") if args.prompt else "a z"
            class_short_prompt = args.prompt.format(class_token) if args.prompt else "a " + class_token
            class_long_prompt  = args.prompt.format(class_long_token) if args.prompt else "a " + class_long_token
            prompt_list             = [ prompt + z_suffix ]
            class_short_prompt_list = [ class_short_prompt + z_suffix ]
            class_long_prompt_list  = [ class_long_prompt  + z_suffix ]

        print(subject_name, ":")

        for prompt, class_short_prompt, class_long_prompt in zip(prompt_list, class_short_prompt_list, class_long_prompt_list):
            if len(args.prompt_suffix) > 0:
                prompt = prompt + ", " + args.prompt_suffix

            print("  ", prompt)
            if not args.plain:
                indiv_subdir = subject_name + "-" + prompt.replace(" ", "-")
                # class_short_prompt, class_long_prompt are saved in the prompt file as well, for evaluation later.
                PROMPTS.write( "\t".join([str(args.n_samples), indiv_subdir, prompt, class_long_prompt, class_short_prompt ]) + "\n" )
            else:
                indiv_subdir = subject_name

        command_line = f"python3 scripts/stable_txt2img.py --config configs/stable-diffusion/{config_file} --ckpt {ckpt_path} --ddim_eta 0.0 --ddim_steps {args.steps} --gpu {args.gpu} --scale {args.scale} --subj_scale {args.subj_scale} --broad_class {broad_class} --n_repeat 1 --bs {args.bs} --outdir {outdir}"

        if not args.plain:
            PROMPTS.close()
            # Since we use a prompt file, we don't need to specify --n_samples.
            command_line += f" --from_file {prompt_filepath}"
        else:
            # Do not use a prompt file, but specify --n_samples, --prompt, and --indiv_subdir.
            # Note only the last prompt/class_prompt/ref_prompt will be used. 
            command_line += f" --n_samples {args.n_samples} --indiv_subdir {indiv_subdir}"
            command_line += f" --prompt \"{prompt}\" --class_prompt \"{class_long_prompt}\""

            if args.ref_prompt_mix_weight != 0:
                # Use the class_short_prompt as the reference prompt, 
                # as it's tokenwise aligned with the subject prompt.
                command_line += f" --ref_prompt \"{class_short_prompt}\""

        # ref_prompt_mix_weight may < 0, in which case we enhance the expression of the subject.
        if args.ref_prompt_mix_weight != 0:
            # Only specify the flag here. The actual reference prompt will be read from the prompt file.
            command_line += f" --ref_prompt_mix_weight {args.ref_prompt_mix_weight}"
            
        if args.method != 'db':
            command_line += f" --embedding_paths {emb_path}"

        if args.clip_last_layer_skip_weight > 0:
            command_line += f" --clip_last_layer_skip_weight {args.clip_last_layer_skip_weight}"
            command_line += f" --clip_last_layer_skip_scheme {args.clip_last_layer_skip_scheme}"

        if args.compare_with_pardir:
            # Do evaluation on authenticity/composition.
            subject_gt_dir = os.path.join(args.compare_with_pardir, subject_name)
            command_line += f" --compare_with {subject_gt_dir}"

            if args.is_face:
                # Tell stable_txt2img.py to do face-specific evaluation.
                command_line += f" --is_face"

        print(command_line)
        os.system(command_line)
