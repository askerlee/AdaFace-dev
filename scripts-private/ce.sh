#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <gpu_id>"
    exit
fi

set -x

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt --gpus $1, --data_roots subjects-private/ce/ -n ce-ada --no-test --max_steps 2000 --cls_delta_string 'asian man' --subj_init_word_weights 1 3 --broad_class 1 --subject_string z --background_string y --num_vectors_per_subj_token 9 --num_vectors_per_bg_token 4 --layerwise_lora_rank 5 --use_conv_attn_kernel_size 3 --clip_last_layers_skip_weights 1 1 --randomize_clip_skip_weights --cls_delta_string 'asian man' --lr 1e-5 --rand_scale_range 0.5 1.0 --embedding_manager_ckpt logs/donnieyen2024-01-12T21-29-01_donnieyen-ada/checkpoints/embeddings_gs-2000.pt
