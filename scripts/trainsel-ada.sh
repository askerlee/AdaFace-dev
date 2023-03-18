#!/usr/bin/fish
if test -z "$GPU"
    set GPU 0
end

# set fish_trace 1
set EXTRA_ARGS

set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set -l prompts  "young girl woman"  "young girl"  "corgi dog"  "asian man" "black woman"  "young boy"  "asian man" "young woman"    "pom dog"  "keanu cool man" "tabby cat"  "asian young girl"  "asian man" "asian woman" "black persian cat"  "white man" "asian woman" "young girl"   "indian girl"  "black man" "asian man"   "cute girl"  "french young man" "young handsome man" "young girl zendaya"
set -l weights  "1 2 2"             "1 2"         "2 1"        "1 2"       "1 2"          "1 2"        "1 2"       "1 2"            "1 1"      "2 1 2"          "1 2"        "1 1 2"             "1 2"        "1 2"        "1 1 3"              "1 2"        "1 2"        "1 2"          "1 2"          "1 2"        "1 2"        "1 2"        "1 1 2"            "1 1 2"              "1 2 2"

# donnieyen jenniferlawrence jiffpom lilbub lisa michelleyeoh selenagomez smritimandhana taylorswift zendaya
for i in 4 8 9 11 12 14 18 19 22 25
    set subject $subjects[$i]
    set prompt  $prompts[$i]
    set weight  (string split " " $weights[$i])

    echo $subject: $prompt $weight
    python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ada --gpus $GPU, --data_root data/$subject/  --placeholder_string "z" --no-test  --init_word $prompt --init_word_weights $weight $EXTRA_ARGS
end
