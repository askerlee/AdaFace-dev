#!/usr/bin/fish
set fish_trace 1
set GPU "1"
set EXTRA_ARGS #"--num_composition_samples_per_batch 0"

set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set -l prompts  "young girl woman"  "young girl"  "corgi dog"  "asian man" "black woman"  "young boy"  "asian man" "young woman"    "pom dog"  "keanu cool man" "tabby cat"  "asian young girl"  "asian man" "asian woman" "black persian cat"  "white man" "asian woman" "young girl"   "indian girl"  "black man" "asian man"   "cute girl"  "french young man" "young handsome man" "young girl zendaya"

# michelleyeoh .. zendaya
for i in (seq 14 25)
    set subject $subjects[$i]
    set initword (string split " " $prompts[$i])[-1]
    python3 main.py --base configs/stable-diffusion/v1-finetune.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ti --gpus $GPU, --data_root data/$subject/  --placeholder_string "z" --no-test  --init_word $initword --init_word_weights 1 $EXTRA_ARGS
end
