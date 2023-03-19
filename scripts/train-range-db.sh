#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 2 --max-args 2 'gpu=' 'maxiter=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] [--maxiter MAX_ITERS] low high"
    exit 1
end

# set fish_trace 1

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
# Default max_iters is 800, but can be overridden with --maxiter.
set -q _flag_maxiter; and set max_iters $_flag_maxiter; or set max_iters 800
set L $argv[1]
set H $argv[2]

set suffix ", instagram"
#                1                    2      3      4            5               6           7             8            9               10     11      12            13             14          15                  16          17             18              19        20           21          22              23           24               25
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
