#!/usr/bin/fish
argparse 'gpu=' 'N=' 'subjfile=' -- $argv

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q _flag_N; and set N $_flag_N; or set N 128
set n_repeat (math ceil $N / 8)

set -q _flag_subjfile; and set subj_file $_flag_subjfile; or set subj_file evaluation/info-subjects.sh
if ! test -e $subj_file
    echo "Error: Subject file '$subj_file' does not exist."
    exit 1
end
source $subj_file

set -l indices0 (seq 1 (count $subjects))
set -q argv[1]; and set L $argv[1]; or set L 1
set -q argv[2]; and set H $argv[2]; or set H (count $subjects)
set -l indices $indices0[(seq $L $H)]

echo Generating reg images for $subjects[$indices]

for i in $indices
    set db_prompt0 "$db_prompts[$i]"
    set db_prompt  "$db_prompt0$db_suffix"
    set db_prompt0_nospace (string replace -a " " "" $db_prompt0)

    # if regularization_images/$db_prompt0_nospace exists, then skip
    if test -e regularization_images/$db_prompt0_nospace
        set reg_img_count (count (find regularization_images/$db_prompt0_nospace -maxdepth 1 -name '*jpg'))
        if test $reg_img_count -gt $N
            echo "'regularization_images/$db_prompt0_nospace' contains $reg_img_count images. Skip $i: $subjects[$i]"
            continue
        end
    end

    set fish_trace 1

    python3 scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_repeat $n_repeat --scale 10.0 --ddim_steps 50  --ckpt models/stable-diffusion-v-1-5/v1-5-pruned.ckpt --gpu $GPU --prompt $db_prompt --outdir regularization_images/$db_prompt0_nospace --indiv_subdir "" --no_preview

    set -e fish_trace
end

