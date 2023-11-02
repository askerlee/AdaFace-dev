#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <gpu_id>"
    exit
fi

set -x

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-5/v1-5-dste8.ckpt --gpus $1, --data_root private-data/ce/ -n ce-ada --no-test --max_steps 4500 --init_words 'asian man' --init_word_weights 1 3 --broad_class 1 --bg_init_words 'blank' --placeholder_string z --background_string y --num_vectors_per_token 9 --num_vectors_per_bg_token 4 --layerwise_lora_rank 5 --use_conv_attn --clip_last_layers_skip_weights 1 3 --randomize_clip_skip_weights --static_embedding_reg_weight 2e-5 --ada_embedding_reg_weight 1e-3 --cls_delta_token man --lr 8e-4 --rand_scale_range 0.5 1.0 --wds_comp_db_path ../img2dataset/mscoco/00000.tar --do_flip_v
