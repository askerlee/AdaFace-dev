#!/usr/bin/fish
set self (status basename)

if test (count $argv) -lt 3
    echo "Usage: $self GPU-ID L H"
    exit 1
end

# set fish_trace 1

set GPU $argv[1]
set L $argv[2]
set H $argv[3]
echo $self $GPU $L $H

set max_iters 800
set suffix ", instagram"
set -l subjects alexachung caradelevingne corgi donnieyen   gabrielleunion iainarmitage  jaychou     jenniferlawrence jiffpom    keanureeves lilbub lisa         masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez smritimandhana spikelee    stephenchow  taylorswift timotheechalamet  tomholland        zendaya
set -l prompts  girl       girl           corgi "asian man" "black woman"  "young boy"   "asian man" girl             "pom dog"  "white man" cat    "asian girl" "asian man" "asian woman" "black persian cat"  "white man" "asian woman" girl        "indian girl"  "black man" "asian man"  girl        "white young man" "white young man" girl

# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in (seq $L $H)
    set subject $subjects[$i]
    set prompt0 "$prompts[$i]"
    set prompt  "$prompt0$suffix"

    echo $subject: $prompt
    python3 main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --reg_data_root regularization_images/(string replace -a " " "" $prompt0) -n $subject-dreambooth --gpus $GPU, --data_root data/$subject --max_training_steps $max_iters --class_word $prompt --token z --no-test
end
