#!/usr/bin/fish
set fish_trace 1
set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set MODEL_NAME "runwayml/stable-diffusion-v1-5"
set BS 2
set ACCUMU_STEPS 2
set LR 1e-4
set LR_TEXT 1e-5
set LR_TI 5e-4

# michelleyeoh .. zendaya
for i in (seq 14 25)
    set subject $subjects[$i]
    set INSTANCE_DIR "data/$subject"
    set OUTPUT_DIR   "exps$BS-$ACCUMU_STEPS/$subject"

    lora_pti \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_text_encoder \
    --resolution=512 \
    --train_batch_size=$BS \
    --gradient_accumulation_steps=$ACCUMU_STEPS \
    --scale_lr \
    --learning_rate_unet=$LR \
    --learning_rate_text=$LR_TEXT \
    --learning_rate_ti=$LR_TI \
    --color_jitter \
    --lr_scheduler="linear" \
    --lr_warmup_steps=0 \
    --placeholder_tokens="<s1>" \
    --use_template="object" \
    --save_steps=100 \
    --max_train_steps_ti=1000 \
    --max_train_steps_tuning=1000 \
    --perform_inversion=True \
    --clip_ti_decay \
    --weight_decay_ti=0.000 \
    --weight_decay_lora=0.001\
    --continue_inversion \
    --continue_inversion_lr=1e-4 \
    --device="cuda:1" \
    --lora_rank=1

end
