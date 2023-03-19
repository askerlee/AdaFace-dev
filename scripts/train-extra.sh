set self (status basename)
echo $self $argv

argparse --min-args 1 --max-args 3 'gpu=' 'extra=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] (ada|ti) [low high] [--extra EXTRA_ARGS]"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu GPU-ID] (ada|ti) [low high] [--extra EXTRA_ARGS]"
    exit 1
end

# set fish_trace 1
#                1
set -l subjects alita         
set -l prompts  "cyborg young girl"  
set -l weights  "1 1 2"             

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set EXTRA_ARGS $_flag_extra
set -q argv[2]; and set L $argv[2]; or set L 1
set -q argv[3]; and set H $argv[3]; or set H (count $subjects)


# $0 0 1 13: alexachung .. masatosakai, on GPU0
# $0 1 14 25: michelleyeoh .. zendaya,  on GPU1
for i in (seq $L $H)
    set subject $subjects[$i]
    set prompt  $prompts[$i]
    set weight  (string split " " $weights[$i])
    # initword is for TI only. It is the last word of the init words of ada
    set initword (string split " " $prompt)[-1]

    if [ $method = 'ti' ]
        echo $subject: $initword
        python3 main.py --base configs/stable-diffusion/v1-finetune-ti.yaml  -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ti  --gpus $GPU, --data_root data-extra/$subject/  --placeholder_string "z" --no-test  --init_word $initword --init_word_weights 1 $EXTRA_ARGS
    else
        echo $subject: $prompt $weight
        python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n $subject-ada --gpus $GPU, --data_root data-extra/$subject/  --placeholder_string "z" --no-test  --init_word $prompt   --init_word_weights $weight $EXTRA_ARGS
    end
end
