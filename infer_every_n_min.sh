#!/bin/bash
N=$1    # Get the first argument passed to the script
RANGE=$2
SIG=$3
fish -c "cd ~/adaprompt; python3 -m scripts.gen_prompts_and_eval --out_dir_tmpl samples --subjfile ~/adaprompt/evaluation/info-subjects-ood.sh --adaface_ckpt_paths (ls -rt ~/adaprompt/logs/VGGface2_HQ_masks2025-*$SIG*_zero3-ada/checkpoints/embedding*|tail -1) --z_suffix_type class_name --method adaface --sep_log  --fp_trick_str portrait --gpus 0-1 --scale 5 --range $RANGE"
# Schedule the script to re-run after $N minutes
echo "~/adaprompt/infer_every_n_min.sh $N $RANGE $SIG >> ~/adaprompt/infer_every_n_min.log 2>&1" | at now + "$N" minutes
