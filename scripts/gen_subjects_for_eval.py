import re
import argparse
import glob
import os

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
    # extra_z_suffix
    parser.add_argument("--extra_z_suffix", type=str, default="",
                        help="Extra suffix to append to the z suffix")
    
    parser.add_argument("--scale", type=float, default=5, 
                        help="the guidance scale")
    # subj_scale
    parser.add_argument("--subj_scale", type=float, default=1,
                        help="the subject embedding scale")
    
    parser.add_argument("--n_samples", type=int, default=4, 
                        help="number of samples to generate for each test case")
    parser.add_argument("--bs", type=int, default=4, 
                        help="batch size")
    # prompt_suffix
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
    parser.add_argument(
        "--selset", action="store_true",
        help="Whether to evaluate only the selected subset of subjects"
    )

    args = parser.parse_args()
    return args

def split_string(input_string):
    pattern = r'"[^"]*"|\S+'
    substrings = re.findall(pattern, input_string)
    substrings = [ s.strip('"') for s in substrings ]
    return substrings

def parse_subject_file(subject_file_path, method):
    subjects, class_tokens, broad_classes, sel_set = None, None, None, None

    with open(subject_file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if re.search(r"^set -[la] [a-zA-Z_]+ ", line):
                # set -l subjects  alexachung    alita...
                mat = re.search(r"^set -[la] ([a-zA-Z_]+)\s+(\S.+\S)", line)
                if mat is not None:
                    var_name = mat.group(1)
                    substrings = split_string(mat.group(2))
                    if var_name == "subjects":
                        subjects = substrings
                    elif var_name == "db_prompts" and method == "db":
                        class_tokens = substrings
                    elif var_name == "cls_tokens" and method != "db":
                        class_tokens = substrings
                    elif var_name == "broad_classes":
                        broad_classes = [ int(s) for s in substrings ]
                    elif var_name == 'sel_set':
                        sel_set = [ int(s) - 1 for s in substrings ]
                else:
                    breakpoint()

    if broad_classes is None:
        # By default, all subjects are humans/animals, unless specified in the subject file.
        broad_classes = [ 1 for _ in subjects ]

    if sel_set is None:
        sel_set = list(range(len(subjects)))

    if subjects is None or class_tokens is None:
        raise ValueError("subjects or db_prompts is None")
    
    return subjects, class_tokens, broad_classes, sel_set

def get_promt_list(subject_name, unique_token, class_token, broad_class):
    object_prompt_list = [
    # The space between "{0} {1}" is removed, so that prompts for ada/ti could be generated by
    # providing an empty class_token. To generate prompts for DreamBooth, 
    # provide a class_token starting with a space.
    'a {0}{1} in the jungle',
    'a {0}{1} in the snow',
    'a {0}{1} on the beach',
    'a {0}{1} on a cobblestone street',
    'a {0}{1} on top of pink fabric',
    'a {0}{1} on top of a wooden floor',
    'a {0}{1} with a city in the background',
    'a {0}{1} with a mountain in the background',
    'a {0}{1} with a blue house in the background',
    'a {0}{1} on top of a purple rug in a forest',
    'a {0}{1} with a wheat field in the background',
    'a {0}{1} with a tree and autumn leaves in the background',
    'a {0}{1} with the Eiffel Tower in the background',
    'a {0}{1} floating on top of water',
    'a {0}{1} floating in an ocean of milk',
    'a {0}{1} on top of green grass with sunflowers around it',
    'a {0}{1} on top of a mirror',
    'a {0}{1} on top of the sidewalk in a crowded street',
    'a {0}{1} on top of a dirt road',
    'a {0}{1} on top of a white rug',
    'a red {0}{1}',
    'a purple {0}{1}',
    'a shiny {0}{1}',
    'a wet {0}{1}',
    'a cube shaped {0}{1}'
    ]

    animal_prompt_list = [
    'a {0}{1} in the jungle',
    'a {0}{1} in the snow',
    'a {0}{1} on the beach',
    'a {0}{1} on a cobblestone street',
    'a {0}{1} on top of pink fabric',
    'a {0}{1} on top of a wooden floor',
    'a {0}{1} with a city in the background',
    'a {0}{1} with a mountain in the background',
    'a {0}{1} with a blue house in the background',
    'a {0}{1} on top of a purple rug in a forest',
    'a {0}{1} wearing a red hat',
    'a {0}{1} wearing a santa hat',
    'a {0}{1} wearing a rainbow scarf',
    'a {0}{1} wearing a black top hat and a monocle',
    'a {0}{1} in a chef outfit',
    'a {0}{1} in a firefighter outfit',
    'a {0}{1} in a police outfit',
    'a {0}{1} wearing pink glasses',
    'a {0}{1} wearing a yellow shirt',
    'a {0}{1} in a purple wizard outfit',
    'a red {0}{1}',
    'a purple {0}{1}',
    'a shiny {0}{1}',
    'a wet {0}{1}',
    'a cube shaped {0}{1}'
    ]

    # humans/animals and cartoon characters.
    if broad_class == 1 or broad_class == 2:
        orig_prompt_list = animal_prompt_list
    else:
        # object.
        orig_prompt_list = object_prompt_list
    
    prompt_list = [ prompt.format(unique_token, class_token) for prompt in orig_prompt_list ]
    return prompt_list, orig_prompt_list

# extra_sig could be a regular expression
def find_first_match(lst, search_term, extra_sig=""):
    for item in lst:
        if search_term in item and re.search(extra_sig, item):
            return item
    return None  # If no match is found

def parse_range_str(range_str):
    if range_str is not None:
        range_strs = range_str.split("-")
        # Only generate a particular subject.
        if len(range_strs) == 1:
            low, high = int(range_strs[0]) - 1, int(range_strs[0])
        else:
            # low is 1-indexed, converted to 0-indexed by -1.
            # high is inclusive, converted to exclusive without adding offset.
            low, high  = int(range_strs[0]) - 1, int(range_strs[1])
    else:
        low, high = 0, None

    return low, high

if __name__ == "__main__":
    
    args = parse_args()
    subjects, class_tokens, broad_classes, sel_set = parse_subject_file(args.subject_file, args.method)
    if args.method == 'db':
        # For DreamBooth, use_z_suffix is the default.
        args.use_z_suffix = True

    low, high     = parse_range_str(args.range)
    subjects      = subjects[low:high]
    class_tokens  = class_tokens[low:high]
    broad_classes = broad_classes[low:high]
    if args.selset:
        subjects      = [ subjects[i]       for i in sel_set ]
        class_tokens  = [ class_tokens[i]   for i in sel_set ]
        broad_classes = [ broad_classes[i]  for i in sel_set ]

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
            # Repeat each prompt for n_samples times in the prompt file. 
            # So that stable_txt2img.py generates n_samples images for each prompt.
            for i in range(args.n_samples):
                # orig_prompt is saved in the prompt file as well, for evaluation later.
                PROMPTS.write( "\t".join([indiv_subdir, prompt, orig_prompt]) + "\n" )

        PROMPTS.close()
        # Since we use a prompt file, we don't need to specify --n_samples.
        command_line = f"python3 scripts/stable_txt2img.py --config configs/stable-diffusion/{config_file} --ckpt {ckpt_path} --ddim_eta 0.0 --ddim_steps {args.steps} --gpu {args.gpu} --from-file {prompt_filepath} --scale {args.scale} --subj_scale {args.subj_scale} --broad_class {broad_class} --n_repeat 1 --bs {args.bs} --outdir {outdir}"
        if args.method != 'db':
            command_line += f" --embedding_paths {emb_path}"

        print(command_line)
        os.system(command_line)
