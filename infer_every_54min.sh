#!/bin/bash
fish -c "cd ~/adaprompt; python3 -m scripts.gen_prompts_and_eval --out_dir_tmpl samples --subjfile ~/adaprompt/evaluation/info-subjects-ood.sh --adaface_ckpt_paths (ls -rt ~/adaprompt/logs/VGGface2_HQ_masks2024-11-07T06*_zero3-ada/checkpoints/embedding*|tail -1) --z_suffix_type class_name --method adaface --sep_log  --fp_trick_str portrait --gpus 0-1 --scale 6 --range 12-13"
echo "~/adaprompt/infer_every_54min.sh >> ~/adaprompt/infer54min.log 2>&1" | at now + 54 minutes

