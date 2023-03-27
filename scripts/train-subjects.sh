#!/usr/bin/fish
# Trainign scripts for the 25 subjects, using AdaPrompt/TI/DreamBooth.
set self (status basename)
echo $self $argv

argparse --min-args 1 --max-args 3 'gpu=' 'extra=' 'maxiter=' 'lr=' 'subjfile=' 'selset' 'use_cls_token' 'use_z_suffix' -- $argv
or begin
    echo "Usage: $self [--gpu ID] [--maxiter M] [--lr LR] [--subjfile SUBJ] [--use_cls_token] [--use_z_suffix] (ada|ti|db) [--selset|low high] [--extra EXTRA_ARGS]"
    echo "E.g.:  $self --gpu 0 --maxiter 4000 --subjfile scripts/info-db-eval-subjects.sh --use_cls_token ada 1 25"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'db' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu ID] [--maxiter M] [--subjfile SUBJ] [--use_cls_token] [--use_z_suffix] (ada|ti|db) [--selset|low high] [--extra EXTRA_ARGS]"
    echo "E.g.:  $self --gpu 0 --maxiter 4000 --subjfile scripts/info-db-eval-subjects.txt --use_cls_token ada 1 25"
    exit 1
end

set -q _flag_subjfile; and set subj_file $_flag_subjfile; or set subj_file scripts/info-subjects.txt
fish $subj_file; or exit 1

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q argv[2]; and set L $argv[2]; or set L 1
set -q argv[3]; and set H $argv[3]; or set H (count $subjects)
set EXTRA_ARGS0 $_flag_extra

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ];
    set -q _flag_maxiter; and set max_iters $_flag_maxiter; or set max_iters 4000
else
    set -q _flag_maxiter; and set max_iters $_flag_maxiter; or set max_iters 800
end

set -q _flag_lr; and set lr $_flag_lr; or set lr -1

#set fish_trace 1

# If --selset is specified, then only train the selected subjects, i.e., 4, 8, ..., 25.
set -q _flag_selset; and set -l indices 4 8 9 11 12 14 18 19 22 25; or set -l indices (seq $L $H)
# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in $indices
    set subject     $subjects[$i]
    set ada_prompt  $ada_prompts[$i]
    set ada_weight  (string split " " $ada_weights[$i])
    # If cls_tokens is specified in subjfile, cls_token = cls_tokens[$i]. 
    # Otherwise, cls_token is the last word of ada_prompt. 
    # For non-human cases "stuffed animal", the last word of ada_prompt is "animal", which is incorrecct. 
    # So we need an individual cls_tokens for non-human subjects. 
    # For "stuffed animal", the corresponding cls_token is "toy".
    # For humans, this is optional. If not specified, then cls_token = last word of ada_prompt.
    # Only use cls_token when --use_cls_token is specified.
    set -q cls_tokens; and set cls_token $cls_tokens[$i]; or set cls_token (string split " " $ada_prompt)[-1]
    set db_prompt0 "$db_prompts[$i]"
    set db_prompt  "$db_prompt0$db_suffix"

    if [ $method = 'ti' ]; or [ $method = 'ada' ]
        if [ $method = 'ada' ]
            set initword $ada_prompt
            set init_word_weights $ada_weight
        else
            set initword $cls_token
            set init_word_weights 1
        end

        # Reset EXTRA_ARGS1 to EXTRA_ARGS0 each time. 
        set EXTRA_ARGS1 $EXTRA_ARGS0

        # cls_token: the class token used in delta loss computation.
        # If --use_cls_token, and cls_tokens is provided in the subjfile, then use cls_token. 
        # Otherwise use the default cls_token "person".
        set -q _flag_use_cls_token; and set EXTRA_ARGS1 $EXTRA_ARGS1 --cls_delta_token $cls_token

        # z_suffix: append $cls_token as a suffix to "z" in the prompt. The prompt will be "a z <cls_token> <prompt>".
        # E.g., cls_token="toy", prompt="in a chair", then full prompt="a z toy in a chair".
        # If not specified, then no suffix is appended. The prompt will be "a z <prompt>". E.g. "a z in a chair".
        set -q _flag_use_z_suffix;  and set z_suffix $cls_token; or set -e z_suffix
        set -q z_suffix; and set EXTRA_ARGS1 $EXTRA_ARGS1 --placeholder_suffix $z_suffix

        # If $broad_classes are specified in subjfile, then use it. Otherwise, use the default value 1.
        set -q broad_classes; and set broad_class $broad_classes[$i]; or set broad_class 1

        echo $subject: --init_word $initword $EXTRA_ARGS1
        set fish_trace 1
        python3 main.py --base configs/stable-diffusion/v1-finetune-$method.yaml  -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --gpus $GPU, --data_root $data_folder/$subject/ -n $subject-$method --no-test --max_steps $max_iters --lr $lr --placeholder_string "z" --init_word $initword --init_word_weights $init_word_weights --broad_class $broad_class $EXTRA_ARGS1
    else
        echo $subject: $db_prompt
        set fish_trace 1
        # $EXTRA_ARGS is not for DreamBooth. It is for AdaPrompt/TI only.
        python3 main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --gpus $GPU, --reg_data_root regularization_images/(string replace -a " " "" $db_prompt0) --data_root $data_folder/$subject -n $subject-dreambooth --no-test --max_steps $max_iters --lr $lr --token "z" --class_word $db_prompt
    end

    set -e fish_trace
end
