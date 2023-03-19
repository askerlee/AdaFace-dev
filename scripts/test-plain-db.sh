#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 0 --max-args 2 'gpu=' 'steps=' 'scale=' 'niter=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] [--steps DDIM_STEPS] [--scale scale] [--niter n_iter] [low high]"
    exit 1
end

# set fish_trace 1

set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q _flag_steps; and set ddim_steps $_flag_steps; or set ddim_steps 100
set -q _flag_scale; and set scale $_flag_scale; or set scale 5
set -q _flag_niter; and set n_iter $_flag_niter; or set n_iter 1
set -q argv[1]; and set L $argv[1]; or set L 1
set -q argv[2]; and set H $argv[2]; or set H 25

# set fish_trace 1
set suffix ", instagram"
#                1                    2      3      4            5               6           7             8            9               10     11      12            13             14          15                  16          17             18              19        20           21          22              23           24               25
set -l subjects alexachung caradelevingne corgi donnieyen   gabrielleunion ianarmitage  jaychou     jenniferlawrence jiffpom    keanureeves lilbub lisa         masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez smritimandhana spikelee    stephenchow  taylorswift timotheechalamet  tomholland        zendaya
set -l prompts  girl       girl           corgi "asian man" "black woman"  "young boy"  "asian man" girl             "pom dog"  "white man" cat    "asian girl" "asian man" "asian woman" "black persian cat"  "white man" "asian woman" girl        "indian girl"  "black man" "asian man"  girl        "white young man" "white young man" girl
set outdir samples-dreambooth

for i in (seq $L $H)
    set subject $subjects[$i]
    set prompt0 $prompts[$i]
    set prompt "a photo of a z $prompt0$suffix"
    set ckptname  (ls -1 -rt logs|grep $subject|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end

    echo $subject: $ckptname 
    echo Prompt: $prompt
    python3 scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --ddim_steps $ddim_steps --ckpt logs/$ckptname/checkpoints/last.ckpt --prompt $prompt --gpu $GPU --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $subject
end
