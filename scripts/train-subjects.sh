#!/usr/bin/fish
# Trainign scripts for the 25 subjects, using AdaPrompt/TI/DreamBooth.
set self (status basename)
echo $self $argv

argparse --min-args 1 --max-args 3 'gpu=' 'extra=' 'maxiter=' 'subjfile=' 'selset' -- $argv
or begin
    echo "Usage: $self [--gpu ID] [--maxiter M] [--subjfile SUBJ] (ada|ti|db) [--selset|low high] [--extra EXTRA_ARGS]"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'db' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu ID] [--maxiter M] [--subjfile SUBJ] (ada|ti|db) [--selset|low high] [--extra EXTRA_ARGS]"
    exit 1
end

# set fish_trace 1
set -q _flag_subjfile; and set subj_file $_flag_subjfile; or set subj_file scripts/info-subjects.txt
fish $subj_file

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q argv[2]; and set L $argv[2]; or set L 1
set -q argv[3]; and set H $argv[3]; or set H (count $subjects)
set EXTRA_ARGS $_flag_extra

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ];
    set -q _flag_maxiter; and set max_iters $_flag_maxiter; or set max_iters 4000
else
    set -q _flag_maxiter; and set max_iters $_flag_maxiter; or set max_iters 800
end

# If --selset is specified, then only train the selected subjects, i.e., 4, 8, ..., 25.
set -q _flag_selset; and set -l indices 4 8 9 11 12 14 18 19 22 25; or set -l indices (seq $L $H)
# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in $indices
    set subject     $subjects[$i]
    set ada_prompt  $ada_prompts[$i]
    set ada_weight  (string split " " $ada_weights[$i])
    # initword is for TI only. It is the last word of the init words of ada
    set ti_initword (string split " " $ada_prompt)[-1]
    set db_prompt0 "$db_prompts[$i]"
    set db_prompt  "$db_prompt0$db_suffix"

    set fish_trace 1

    if [ $method = 'ti' ]
        echo $subject: $ti_initword
        python3 main.py --base configs/stable-diffusion/v1-finetune-ti.yaml  -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --gpus $GPU, --data_root $data_folder/$subject/ -n $subject-ti --no-test --max_training_steps $max_iters --placeholder_string "z" --init_word $ti_initword --init_word_weights 1 $EXTRA_ARGS
    else if [ $method = 'db' ]
        echo $subject: $db_prompt
        # $EXTRA_ARGS is not for DreamBooth. It is for AdaPrompt/TI only.
        python3 main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --gpus $GPU, --reg_data_root regularization_images/(string replace -a " " "" $db_prompt0) --data_root $data_folder/$subject -n $subject-dreambooth --no-test --max_training_steps $max_iters --token "z" --class_word $db_prompt
    else
        echo $subject: $ada_prompt $ada_weight
        python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --gpus $GPU, --data_root $data_folder/$subject/ -n $subject-ada --no-test --max_training_steps $max_iters --placeholder_string "z" --init_word $ada_prompt --init_word_weights $ada_weight $EXTRA_ARGS
    end

    set fish_trace 0
end
