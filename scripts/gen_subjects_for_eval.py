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
    parser.add_argument("--no_z_suffix", dest='use_z_suffix', action="store_false",
                        help="Do not append placeholder suffix to the subject placeholder (default: append)")
    # extra_z_suffix: usually reduces the similarity of generated images to the real images.
    parser.add_argument("--extra_z_suffix", type=str, default="",
                        help="Extra suffix to append to the z suffix")
    
    parser.add_argument("--scale", type=float, default=5, 
                        help="the guidance scale")
    # subj_scale: sometimes it improves the similarity, somtimes it reduces it.
    parser.add_argument("--subj_scale", type=float, default=1,
                        help="the subject embedding scale")
    
    parser.add_argument("--n_samples", type=int, default=4, 
                        help="number of samples to generate for each test case")
    parser.add_argument("--bs", type=int, default=4, 
                        help="batch size")
    # prompt_suffix: usually reduces the similarity.
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="suffix to append to the end of each prompt")
    
    parser.add_argument("--steps", type=int, default=50, 
                        help="number of DDIM steps to generate samples")
    parser.add_argument("--ckpt_dir", type=str, default="logs",
                        help="parent directory containing checkpoints of all subjects")
    parser.add_argument("--ckpt_iter", type=int, default=4000,
                        help="checkpoint iteration to use")
    parser.add_argument("--ckpt_extra_sig", type=str, default="",
                        help="Extra signature that is part of the checkpoint directory name."
                             " Could be a regular expression.")
    
    parser.add_argument("--out_dir_tmpl", type=str, default="samples-dbeval",
                        help="Template of parent directory to save generated samples")

    # composition case file path
    parser.add_argument("--subject_file", type=str, default="scripts/info-db-eval-subjects.sh", 
                        help="subject info script file")
    # range of subjects to generate
    parser.add_argument("--range", type=str, default=None, 
                        help="Range of subjects to generate (Index starts from 1 and is inclusive, e.g., 1-30)")
    parser.add_argument("--selset", action="store_true",
                        help="Whether to evaluate only the selected subset of subjects")
    parser.add_argument("--compare_with_pardir", type=str, default=None,
                        help="Parent folder of subject images used for computing similarity with generated samples")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    subjects, class_tokens, broad_classes, sel_set = parse_subject_file(args.subject_file, args.method)
    if args.method == 'db':
        # For DreamBooth, use_z_suffix is the default.
        args.use_z_suffix = True

    if args.selset:
        subjects      = [ subjects[i]       for i in sel_set ]
        class_tokens  = [ class_tokens[i]   for i in sel_set ]
        broad_classes = [ broad_classes[i]  for i in sel_set ]

    low, high     = parse_range_str(args.range)
    subjects      = subjects[low:high]
    class_tokens  = class_tokens[low:high]
    broad_classes = broad_classes[low:high]

    all_ckpts = os.listdir(args.ckpt_dir)
    # Sort all_ckpts by modification time, most recent first.
    all_ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(args.ckpt_dir, x)), reverse=True)

    for subject_name, class_token, broad_class in zip(subjects, class_tokens, broad_classes):
        ckpt_sig   = subject_name + "-" + args.method
        # Find the newest checkpoint that matches the subject name.
        ckpt_name  = find_first_match(all_ckpts, ckpt_sig, args.ckpt_extra_sig)

        if ckpt_name is None:
            print("ERROR: No checkpoint found for subject: " + subject_name)
            continue
            # breakpoint()

        if args.use_z_suffix:
            # For DreamBooth, use_z_suffix is the default.
            # For Ada/TI, if we append class token to "z" -> "z dog", 
            # the chance of occasional under-expression of the subject may be reduced.
            # (This trick is not needed for human faces)
            # Prepend a space to class_token to avoid "a zcat" -> "a z cat"
            class_token = " " + class_token
        else:
            class_token = ""

        if len(args.extra_z_suffix) > 0:
            class_token += " " + args.extra_z_suffix + ","
            
        if args.method == 'db':
            config_file = "v1-inference.yaml"
            ckpt_path   = f"logs/{ckpt_name}/checkpoints/last.ckpt"
        else:
            config_file = "v1-inference-" + args.method + ".yaml"
            ckpt_path   = "models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt"
            emb_path    = f"logs/{ckpt_name}/checkpoints/embeddings_gs-{args.ckpt_iter}.pt"

        outdir = args.out_dir_tmpl + "-" + args.method
        # E.g., get_promt_list(subject_name="cat2", placeholder="z", class_token="cat", broad_class=1)
        prompt_list, orig_prompt_list = get_promt_list(subject_name, args.placeholder, class_token, broad_class)
        prompt_filepath = f"{outdir}/{subject_name}-prompts.txt"
        os.makedirs(outdir, exist_ok=True)
        PROMPTS = open(prompt_filepath, "w")
        print(subject_name, ":")

        for prompt, orig_prompt in zip(prompt_list, orig_prompt_list):
            if len(args.prompt_suffix) > 0:
                prompt = prompt + ", " + args.prompt_suffix

            print("  ", prompt)
            indiv_subdir = subject_name + "-" + prompt.replace(" ", "-")
            # orig_prompt is saved in the prompt file as well, for evaluation later.
            PROMPTS.write( "\t".join([str(args.n_samples), indiv_subdir, prompt, orig_prompt]) + "\n" )

        PROMPTS.close()
        # Since we use a prompt file, we don't need to specify --n_samples.
        command_line = f"python3 scripts/stable_txt2img.py --config configs/stable-diffusion/{config_file} --ckpt {ckpt_path} --ddim_eta 0.0 --ddim_steps {args.steps} --gpu {args.gpu} --from_file {prompt_filepath} --scale {args.scale} --subj_scale {args.subj_scale} --broad_class {broad_class} --n_repeat 1 --bs {args.bs} --outdir {outdir}"
        if args.compare_with_pardir:
            subject_gt_dir = os.path.join(args.compare_with_pardir, subject_name)
            command_line += " --compare_with {args.subject_gt_dir}"
        if args.method != 'db':
            command_line += f" --embedding_paths {emb_path}"

        print(command_line)
        os.system(command_line)
