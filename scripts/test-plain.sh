#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 1 --max-args 3 'gpu=' 'steps=' 'scale=' 'niter=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] [--steps DDIM_STEPS] [--scale scale] [--niter n_iter] (ada|ti) [low high]"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu GPU-ID] [--steps DDIM_STEPS] [--scale scale] [--niter n_iter] (ada|ti) [low high]"
    exit 1
end

# set fish_trace 1
set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q _flag_steps; and set ddim_steps $_flag_steps; or set ddim_steps 100
set -q _flag_scale; and set scale $_flag_scale; or set scale 5
set -q _flag_niter; and set n_iter $_flag_niter; or set n_iter 1
set -q argv[2]; and set L $argv[2]; or set L 1
set -q argv[3]; and set H $argv[3]; or set H 25

#                1                    2            3             4            5               6           7             8            9               10          11            12                    13             14          15                  16          17             18              19          20             21            22              23          24                   25    
set -l subjects alexachung         caradelevingne corgi        donnieyen   gabrielleunion iainarmitage jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
#set fish_trace 1
set n_iter 1
set outdir samples-$method

for i in (seq $L $H)
    set subject $subjects[$i]
    set prompt "a z"
    set ckptname  (ls -1 -rt logs|grep $subject-$method|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end

    echo $subject: $ckptname 
    echo prompt=\"$prompt\", scale=$scale
    python3 scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-$method.yaml --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0 --n_samples 8 --ddim_steps $ddim_steps --embedding_paths logs/$ckptname/checkpoints/embeddings_gs-4000.pt --gpu $GPU --prompt $prompt --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $subject
end
