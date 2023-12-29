#!/usr/bin/fish
# Trainign scripts for the 25 subjects, using AdaPrompt/TI/DreamBooth.
function parse_range_str
    set range_str $argv[1]

    if test -z $range_str
        return
    end

    set range_parts (string split ',' -- $range_str)

    for part in $range_parts
        if string match -qr '-' -- $part
            set a (echo $part | awk -F- '{print $1}')
            set b (echo $part | awk -F- '{print $2}')
            for i in (seq $a $b)
                echo $i
            end
        else
            set a $part
            echo $a
        end
    end
end

set self (status basename)
echo $self $argv

argparse --ignore-unknown --min-args 1 --max-args 20 'gpu=' 'maxiter=' 'lr=' 'subjfile=' 'bb_type=' 'num_vectors_per_token=' 'clip_last_layers_skip_weights=' 'cls_string_as_delta' 'use_conv_attn_kernel_size=' 'eval' -- $argv
or begin
    echo "Usage: $self [--gpu ID] [--maxiter M] [--lr LR] [--subjfile SUBJ] [--bb_type bb_type] [--num_vectors_per_token K] [--clip_last_layers_skip_weights w1,w2,...] [--cls_string_as_delta] [--eval] [--use_conv_attn_kernel_size K] (ada|ti|db) [low-high] [EXTRA_ARGS]"
    echo "E.g.:  $self --gpu 0 --maxiter 4000 --subjfile evaluation/info-dbeval-subjects.sh --cls_string_as_delta ada 1 25"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'static-layerwise' ]; or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'db' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu ID] [--maxiter M] [--lr LR] [--subjfile SUBJ] [--bb_type bb_type] [--num_vectors_per_token K] [--clip_last_layers_skip_weights w1,w2,...] [--cls_string_as_delta] [--eval] [--use_conv_attn_kernel_size K] (ada|ti|db) [|low-high] [EXTRA_ARGS]"
    echo "E.g.:  $self --gpu 0 --maxiter 4000 --subjfile evaluation/info-dbeval-subjects.sh --cls_string_as_delta ada 1 25"
    exit 1
end

set -q _flag_subjfile; and set subj_file $_flag_subjfile; or set subj_file evaluation/info-subjects.sh
if ! test -e $subj_file
    echo "Error: Subject file '$subj_file' does not exist."
    exit 1
end
source $subj_file

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
# BUGGY: if L, H are not specified, then $argv[2], $argv[3] may contain unrecognized arguments.
set -q argv[2]; and set range $argv[2]; or set range "1-"(count $subjects)
set EXTRA_TRAIN_ARGS0 $argv[3..-1]
set indices (parse_range_str $range)

set -q _flag_lr; and set lr $_flag_lr; or set -e lr
set -q _flag_min_rand_scaling; and set min_rand_scaling $_flag_min_rand_scaling; or set -e min_rand_scaling
#set fish_trace 1

# default bb_type is v15.
set -q _flag_bb_type; or set _flag_bb_type 'v15-dste8'

if [ "$_flag_bb_type" = 'v15' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-pruned.ckpt
else if [ "$_flag_bb_type" = 'v14' ]
    set sd_ckpt models/stable-diffusion-v-1-4-original/sd-v1-4.ckpt
else if [ "$_flag_bb_type" = 'v15-ema' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-pruned-emaonly.ckpt
else if [ "$_flag_bb_type" = 'v15-dste' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-dste.ckpt
else if [ "$_flag_bb_type" = 'v15-dste8' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-dste8.ckpt
else if [ "$_flag_bb_type" = 'v15-arte' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-arte.ckpt
else if [ "$_flag_bb_type" = 'v15-rvte' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-rvte.ckpt
else if [ "$_flag_bb_type" = 'dreamshaper-v5' ]
    set sd_ckpt models/dreamshaper/dreamshaper_5BakedVae.safetensors
    if not set -q _flag_maxiter
        set _flag_maxiter 2000
    end
else if [ "$_flag_bb_type" = 'dreamshaper-v6' ]
    set sd_ckpt models/dreamshaper/dreamshaper_631BakedVae.safetensors
    if not set -q _flag_maxiter
        set _flag_maxiter 2000
    end
else
    echo "Error: --bb_type must be one of 'v15', 'v14', 'v15-ema', 'v15-dste', 'v15-arte', 'v15-rvte', 'dreamshaper-v5', 'dreamshaper-v6'."
    exit 1
end

set EXTRA_EVAL_ARGS0  --bb_type $_flag_bb_type

if set -q _flag_num_vectors_per_token
    set EXTRA_TRAIN_ARGS0 $EXTRA_TRAIN_ARGS0 --num_vectors_per_token $_flag_num_vectors_per_token
    set EXTRA_EVAL_ARGS0 $EXTRA_EVAL_ARGS0   --num_vectors_per_token $_flag_num_vectors_per_token
end

if set -q _flag_use_conv_attn_kernel_size
    set EXTRA_TRAIN_ARGS0 $EXTRA_TRAIN_ARGS0 --use_conv_attn_kernel_size $_flag_use_conv_attn_kernel_size
    set EXTRA_EVAL_ARGS0 $EXTRA_EVAL_ARGS0   --use_conv_attn_kernel_size $_flag_use_conv_attn_kernel_size
end

set -q _flag_clip_last_layers_skip_weights; and set -l clip_last_layers_skip_weights (string split "," $_flag_clip_last_layers_skip_weights);
if set -q clip_last_layers_skip_weights
    set EXTRA_TRAIN_ARGS0 $EXTRA_TRAIN_ARGS0 --clip_last_layers_skip_weights $clip_last_layers_skip_weights
    # Only use clip_last_layers_skip_weights for training, and not for inference.
    # Seems that using [0.3, 0.4, 0.3] as clip_last_layers_skip_weights for inference leads to worse 
    # compositionality than using the default [0.5, 0.5].
    # set EXTRA_EVAL_ARGS0  $EXTRA_EVAL_ARGS0  --clip_last_layers_skip_weights $clip_last_layers_skip_weights
end

if set -q misc_train_opts
    set EXTRA_TRAIN_ARGS0 $EXTRA_TRAIN_ARGS0 $misc_train_opts
end

if set -q misc_infer_opts
    set EXTRA_EVAL_ARGS0 $EXTRA_EVAL_ARGS0 $misc_infer_opts
end

echo Training on $subjects[$indices]

# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in $indices
    set subject     $subjects[$i]
    set ada_prompt  $ada_prompts[$i]
    set ada_weight  (string split " " $ada_weights[$i])
    # If cls_strings is specified in subjfile, cls_string = cls_strings[$i]. 
    # Otherwise, cls_string is the last word of ada_prompt. 
    # For non-human cases "stuffed animal", the last word of ada_prompt is "animal", which is incorrecct. 
    # So we need an individual cls_strings for non-human subjects. 
    # For "stuffed animal", the corresponding cls_string is "toy".
    # For humans, this is optional. If not specified, then cls_string = last word of ada_prompt.
    # Only use cls_string as delta token when --cls_string_as_delta is specified.
    set -q cls_strings; and set cls_string $cls_strings[$i]; or set cls_string (string split " " $ada_prompt)[-1]
    set db_prompt0 "$db_prompts[$i]"
    set db_prompt  "$db_prompt0$db_suffix"
    set -q bg_init_words; and set bg_init_word $bg_init_words[$i]; or set bg_init_word ""

    if [ $method = 'ti' ]; or [ $method = 'ada' ]; or [ $method = 'static-layerwise' ]
        if [ $method = 'ada' ]; or [ $method = 'static-layerwise' ]
            set init_words $ada_prompt
            set init_word_weights $ada_weight
        else
            set init_words $cls_string
            set init_word_weights 1
        end

        # If $broad_classes are specified in subjfile, then use it. Otherwise, use the default value 1.
        set -q broad_classes; and set broad_class $broad_classes[$i]; or set broad_class 1
        
        if not set -q _flag_maxiter
            # -1: use the default max_iters.
            set -q maxiters; and set max_iters $maxiters[(math $broad_class+1)]; or set max_iters -1
        else
            # Use the specified max_iters.
            set max_iters $_flag_maxiter
        end

        # Reset EXTRA_TRAIN_ARGS1 to EXTRA_TRAIN_ARGS0 each time. 
        set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS0

        if set -q z_prefix_keys
            # Look up the map z_prefix_keys:z_prefix_values, and find the corresponding 
            # z_prefix for the current subject.
            if set -l z_prefix_index (contains -i -- $subject $z_prefix_keys)
                set z_prefix $z_prefix_values[$z_prefix_index]
                set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --common_placeholder_prefix $z_prefix
            end
        end

        # cls_string: the class token used in delta loss computation.
        # If --cls_string_as_delta, and cls_strings is provided in the subjfile, then use cls_string. 
        # Otherwise use the default cls_string "person".
        set -q _flag_cls_string_as_delta; and set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --cls_delta_string $cls_string
        # set -q use_fp_trick; and set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --use_fp_trick $use_fp_trick[$i]
        # If $prompt_mix_max[$i] is not -1 (default [0.1, 0.3]), then prompt_mix_range is 
        # ($prompt_mix_min = $prompt_mix_max / 3, $prompt_mix_max). Probably it will be [0.2, 0.6].
        #if set -q prompt_mix_max; and test $prompt_mix_max[$i] -ne -1
        #    set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --mix_range (math $prompt_mix_max[$i]/3) $prompt_mix_max[$i]
        #end
        
        if not set -q _flag_lr
            set -q lrs; and set lr $lrs[(math $broad_class+1)]
        end
        set -q lr; and set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --lr $lr
        set -q min_rand_scaling; and set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --min_rand_scaling $min_rand_scaling

        echo $subject: --init_words $init_words $EXTRA_TRAIN_ARGS1
        set fish_trace 1
        python3 main.py --base configs/stable-diffusion/v1-finetune-$method.yaml  -t --actual_resume $sd_ckpt --gpus $GPU, --data_root $data_folder/$subject/ -n $subject-$method --no-test --max_steps $max_iters --placeholder_string "z" --init_words $init_words --init_word_weights $init_word_weights --broad_class $broad_class --bg_init_words $bg_init_word $EXTRA_TRAIN_ARGS1

        if set -q _flag_eval
            if [ "$data_folder"  = 'dbeval-dataset' ]
                set out_dir_tmpl 'samples-dbeval'
            else if [ "$data_folder" = 'ti-dataset' ]
                set out_dir_tmpl 'samples-tieval'
            else
                set out_dir_tmpl 'samples'
            end
            # if $max_iters == -1, then gen_subjects_and_eval.py will use the default max_iters of the broad_class.
            set EXTRA_EVAL_ARGS $EXTRA_EVAL_ARGS0 --ckpt_iter $max_iters
            python3 scripts/gen_subjects_and_eval.py --method $method --scale 10 --gpu $GPU --subjfile $subj_file --out_dir_tmpl $out_dir_tmpl  --compare_with_pardir $data_folder --range $i $EXTRA_EVAL_ARGS
        end

    else
        echo $subject: $db_prompt

        # -1: use the default max_iters.
        set fish_trace 1
        # $EXTRA_TRAIN_ARGS is not for DreamBooth. It is for AdaPrompt/TI only.
        # --lr and --max_steps are absent in DreamBooth. 
        # It always uses the default lr and max_steps specified in the config file.
        python3 main_db.py --base configs/stable-diffusion/v1-finetune-db.yaml -t --actual_resume $sd_ckpt --gpus $GPU, --reg_data_root regularization_images/(string replace -a " " "" $db_prompt0) --data_root $data_folder/$subject -n $subject-db --no-test --token "z" --class_word $db_prompt
        
        if set -q _flag_eval
            if [ "$data_folder"  = 'dbeval-dataset' ]
                set out_dir_tmpl 'samples-dbeval'
            else if [ "$data_folder" = 'ti-dataset' ]
                set out_dir_tmpl 'samples-tieval'
            else
                set out_dir_tmpl 'samples'
            end
            python3 scripts/gen_subjects_and_eval.py --method $method --scale 10 --gpu $GPU --subjfile $subj_file --out_dir_tmpl $out_dir_tmpl  --compare_with_pardir $data_folder --range $i $EXTRA_EVAL_ARGS
        end
    end

    set -e fish_trace
end
