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

set EXTRA_ARGS
set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set -l prompts  "young girl woman"  "young girl"  "corgi dog"  "asian man" "black woman"  "young boy"  "asian man" "young woman"    "pom dog"  "keanu cool man" "tabby cat"  "asian young girl"  "asian man" "asian woman" "black persian cat"  "white man" "asian woman" "young girl"   "indian girl"  "black man" "asian man"   "cute girl"  "french young man" "young handsome man" "young girl zendaya"
set -l weights  "1 2 2"             "1 2"         "2 1"        "1 2"       "1 2"          "1 2"        "1 2"       "1 2"            "1 1"      "2 1 2"          "1 2"        "1 1 2"             "1 2"        "1 2"        "1 1 3"              "1 2"        "1 2"        "1 2"          "1 2"          "1 2"        "1 2"        "1 2"        "1 1 2"            "1 1 2"              "1 2 2"

# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in (seq $L $H)
    set subject $subjects[$i]
    set prompt  $prompts[$i]
    set weight  (string split " " $weights[$i])

    echo $subject: $prompt $weight
    python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ada --gpus $GPU, --data_root data/$subject/  --placeholder_string "z" --no-test  --init_word $prompt --init_word_weights $weight $EXTRA_ARGS
end
