#!/usr/bin/fish
set fish_trace 1
set -x GPU "0"
set -x suffix ", instagram"
set -l subjects alexachung caradelevingne corgi donnieyen   gabrielleunion iainarmitage  jaychou     jenniferlawrence jiffpom    keanureeves lilbub lisa         masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez smritimandhana spikelee    stephenchow  taylorswift timotheechalamet  tomholland        zendaya
set -l prompts  girl       girl           corgi "asian man" "black woman"  "young boy"   "asian man" girl             "pom dog"  "white man" cat    "asian girl" "asian man" "asian woman" "black persian cat"  "white man" "asian woman" girl        "indian girl"  "black man" "asian man"  girl        "white young man" "white young man" girl

# alexachung .. masatosakai
for i in (seq 13)
    set -x subject $subjects[$i]
    set -x prompt  $prompts[$i]
    python3 main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --reg_data_root regularization_images/(string replace -a " " "" $prompt) -n $subject-dreambooth --gpus $GPU, --data_root data/$subject --max_training_steps 800 --class_word "$prompt$suffix" --token z --no-test
end
