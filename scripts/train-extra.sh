set self (status basename)
echo $self $argv

argparse --min-args 0 --max-args 2 'gpu=' 'extra=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] [low high] [--extra EXTRA_ARGS]"
    exit 1
end

# set fish_trace 1

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set EXTRA_ARGS $_flag_extra
set -q argv[1]; and set L $argv[1]; or set L 1
set -q argv[2]; and set H $argv[2]; or set H 1

set -l subjects alita         
set -l prompts  "cyborg young girl"  
set -l weights  "1 1 2"             

# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in (seq $L $H)
    set subject $subjects[$i]
    set prompt  $prompts[$i]
    set weight  (string split " " $weights[$i])

    echo $subject: $prompt $weight
    python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ada --gpus $GPU, --data_root data-extra/$subject/  --placeholder_string "z" --no-test  --init_word $prompt --init_word_weights $weight $EXTRA_ARGS
end
