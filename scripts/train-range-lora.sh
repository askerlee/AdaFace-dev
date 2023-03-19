#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 2 --max-args 2 'gpu=' 'bs=' 'accu=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] low high"
    exit 1
end

# set fish_trace 1

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set L $argv[1]
set H $argv[2]

#                1                    2            3             4            5               6           7             8            9               10          11            12                    13             14          15                  16          17             18              19          20             21            22              23          24                   25    
set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set MODEL_NAME "runwayml/stable-diffusion-v1-5"
# Default BS=2, ACCUMU_STEPS=2. Could be overriden by --bs and --accu.
set -q _flag_bs; and set BS $_flag_bs; or set BS 2
set -q _flag_accu; and set ACCUMU_STEPS $_flag_accu; or set ACCUMU_STEPS 2
set LR 1e-4
set LR_TEXT 1e-5
set LR_TI 5e-4
echo "GPU=$GPU, BS=$BS, ACCUMU_STEPS=$ACCUMU_STEPS"

# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in (seq $L $H)
    set subject $subjects[$i]
    set INSTANCE_DIR "data/$subject"
    set OUTPUT_DIR   "exps$BS-$ACCUMU_STEPS/$subject"
    echo $subject

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
    --device="cuda:$GPU" \
    --lora_rank=1
    # --color_jitter \

end
