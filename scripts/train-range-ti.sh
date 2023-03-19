#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 2 --max-args 2 'gpu=' 'extra=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] low high [--extra EXTRA_ARGS]"
    exit 1
end

# set fish_trace 1

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set L $argv[1]
set H $argv[2]
set EXTRA_ARGS $_flag_extra

#                1                    2            3             4            5               6           7             8            9               10          11            12                    13             14          15                  16          17             18              19          20             21            22              23          24                   25
set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet   tomholland            zendaya
set -l prompts  "young girl woman"  "young girl"  "corgi dog"  "asian man" "black woman"  "young boy"  "asian man" "young woman"    "pom dog"  "keanu cool man" "tabby cat"  "asian young girl"  "asian man" "asian woman" "black persian cat"  "white man" "asian woman" "young girl"   "indian girl"  "black man" "asian man"   "cute girl"  "french young man" "young handsome man" "young girl zendaya"

# $0 0 1 13:  train alexachung .. masatosakai, on GPU0
# $0 1 14 25: train michelleyeoh .. zendaya,   on GPU1
for i in (seq $L $H)
    set subject $subjects[$i]
    # Use the last word of the init words of ada as the init word of TI
    set initword (string split " " $prompts[$i])[-1]

    echo $subject: $initword
    python3 main.py --base configs/stable-diffusion/v1-finetune.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ti --gpus $GPU, --data_root data/$subject/  --placeholder_string "z" --no-test  --init_word $initword --init_word_weights 1 $EXTRA_ARGS
end
