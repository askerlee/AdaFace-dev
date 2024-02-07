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

function contains_substring
    set -l needle $argv[1]
    set -l haystack $argv[2..-1]
    set index 1

    for chunk in $haystack
        set sublist (string split ',' -- $chunk)
        if contains $needle $sublist
            echo $index
            return 0
        end
        set -l index (math $index+1)
    end
    echo -1
end

set self (status basename)
echo $self $argv

argparse --ignore-unknown --min-args 1 --max-args 40 'gpu=' 'maxiter=' 'lr=' 'subjfile=' 'bb_type=' 'num_vectors_per_token=' 'clip_last_layers_skip_weights=' 'use_conv_attn_kernel_size=' 'eval' -- $argv
or begin
    echo "Usage: $self [--gpu ID] [--maxiter M] [--lr LR] [--subjfile SUBJ] [--bb_type bb_type] [--num_vectors_per_token K] [--clip_last_layers_skip_weights w1,w2,...] [--eval] [--use_conv_attn_kernel_size K] (ada|ti|db) [low-high] [EXTRA_ARGS]"
    echo "E.g.:  $self --gpu 0 --maxiter 4000 --subjfile evaluation/info-dbeval-subjects.sh ada 1 25"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'static-layerwise' ]; or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'db' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu ID] [--maxiter M] [--lr LR] [--subjfile SUBJ] [--bb_type bb_type] [--num_vectors_per_token K] [--clip_last_layers_skip_weights w1,w2,...] [--eval] [--use_conv_attn_kernel_size K] (ada|ti|db) [|low-high] [EXTRA_ARGS]"
    echo "E.g.:  $self --gpu 0 --maxiter 4000 --subjfile evaluation/info-dbeval-subjects.sh ada 1 25"
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
set -q _flag_bb_type; or set _flag_bb_type 'v15-dste8-vae'

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
else if [ "$_flag_bb_type" = 'v15-dste8-vae' ]
    set sd_ckpt models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt
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

for i in $indices
    set subject     $subjects[$i]
    set init_string  $init_strings[$i]
    set init_word_weights  (string split " " $all_init_word_weights[$i])
    set cls_string $init_string
    #set -q all_bg_init_words; and set bg_init_words $all_bg_init_words[$i]; or set bg_init_words ""
    set class_name $class_names[$i]

    if [ $method = 'ti' ]; or [ $method = 'ada' ]; or [ $method = 'static-layerwise' ]
        # If $broad_classes are specified in subjfile, then use it. Otherwise, use the default value 1.
        set -q broad_classes; and set broad_class $broad_classes[$i]; or set broad_class 1
        
        if not set -q _flag_maxiter
            # -1: use the default max_iters.
            set -q maxiters; and set max_iters $maxiters[(math $broad_class+1)]; or set max_iters -1
        else
            # Use the specified max_iters.
            set max_iters $_flag_maxiter
        end

        # Initialize EXTRA_TRAIN_ARGS1 to EXTRA_TRAIN_ARGS0 each time. 
        # cls_string: the class token used in delta loss computation.
        set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS0 --cls_delta_string $cls_string
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

        if set -q resume_from_ckpt; and test $resume_from_ckpt -eq 1
            set resumed_ckpt_idx (contains_substring $class_name $resumed_ckpt_keys)
            if test $resumed_ckpt_idx -eq -1
                echo "Error: No ckpt found for class $class_name in $resumed_ckpt_keys"
                exit 1
            end
            set emb_man_ckpt $resumed_ckpt_values[$resumed_ckpt_idx]
            set EXTRA_TRAIN_ARGS1 $EXTRA_TRAIN_ARGS1 --embedding_manager_ckpt $emb_man_ckpt --ckpt_params_perturb_ratio 0.2 --emb_reg_loss_scale 0.2
        end

        echo $subject: --init_string $init_string $EXTRA_TRAIN_ARGS1
        set fish_trace 1
        python3 main.py --base configs/stable-diffusion/v1-finetune-$method.yaml  -t --actual_resume $sd_ckpt --gpus $GPU, --data_roots $data_folder/$subject/ -n $subject-$method --no-test --max_steps $max_iters --subject_string "z" --init_string $init_string --init_word_weights $init_word_weights --broad_class $broad_class $EXTRA_TRAIN_ARGS1

        if set -q _flag_eval
            if [ "$data_folder"  = 'subjects-dreambench' ]
                set out_dir_tmpl 'samples-dbeval'
            else if [ "$data_folder" = 'subjects-ti' ]
                set out_dir_tmpl 'samples-tieval'
            else
                set out_dir_tmpl 'samples'
            end
            # if $max_iters == -1, then gen_subjects_and_eval.py will use the default max_iters of the broad_class.
            set EXTRA_EVAL_ARGS $EXTRA_EVAL_ARGS0 --ckpt_iter $max_iters
            python3 scripts/gen_subjects_and_eval.py --method $method --scale 10 4 --gpu $GPU --subjfile $subj_file --out_dir_tmpl $out_dir_tmpl  --compare_with_pardir $data_folder --range $i $EXTRA_EVAL_ARGS
        end

    else
        echo $subject: $init_string

        # -1: use the default max_iters.
        set fish_trace 1
        # $EXTRA_TRAIN_ARGS is not for DreamBooth. It is for AdaPrompt/TI only.
        # --lr and --max_steps are absent in DreamBooth. 
        # It always uses the default lr and max_steps specified in the config file.
        python3 main_db.py --base configs/stable-diffusion/v1-finetune-db.yaml -t --actual_resume $sd_ckpt --gpus $GPU, --reg_data_root regularization_images/(string replace -a " " "" $init_string) --data_roots $data_folder/$subject -n $subject-db --no-test --token "z" --class_word $init_string
        
        if set -q _flag_eval
            if [ "$data_folder"  = 'subjects-dreambench' ]
                set out_dir_tmpl 'samples-dbeval'
            else if [ "$data_folder" = 'subjects-ti' ]
                set out_dir_tmpl 'samples-tieval'
            else
                set out_dir_tmpl 'samples'
            end
            python3 scripts/gen_subjects_and_eval.py --method $method --scale 10 4 --gpu $GPU --subjfile $subj_file --out_dir_tmpl $out_dir_tmpl  --compare_with_pardir $data_folder --range $i $EXTRA_EVAL_ARGS
        end
    end

    set -e fish_trace
end
