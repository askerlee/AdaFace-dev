#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 1 --max-args 1 'gpu=' 'steps=' 'scale=' 'niter=' 'ckptiter=' 'casefile=' -- $argv
or begin
    echo "Usage: $self [--gpu ID] [--steps T] [--scale S] [--niter N] [--ckptiter N2] [--casefile F] (ada|ti|db)"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]; or [ "$argv[1]" = 'db' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu ID] [--steps T] [--scale S] [--niter N] [--ckptiter N2] [--casefile F] (ada|ti|db)"
    exit 1
end

# set fish_trace 1
set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q _flag_steps; and set ddim_steps $_flag_steps; or set ddim_steps 100
set -q _flag_scale; and set scale $_flag_scale; or set scale 10
set -q _flag_niter; and set n_iter $_flag_niter; or set n_iter 1
# ckpt_iter is only for AdaPrompt and TI, not for DreamBooth.
set -q _flag_ckptiter; and set ckpt_iter $_flag_ckptiter; or set ckpt_iter 4000

set outdir samples-$method
if [ "$argv[1]" = 'db' ]
    # conformative to the original name in DreamBooth repo.
    set config v1-inference.yaml
else
    set config v1-inference-$method.yaml
end

set -q _flag_casefile; and set casefile $_flag_casefile; or set casefile scripts/info-compositions.txt
fish $casefile

for case in $cases
    echo \"$case\"
    set -l case2 (string split " | " $case)
    set subject         $case2[1]
    set prompt0         $case2[2]
    set folder          $case2[3]
    set db_class        $case2[4]
    if [ "$argv[1]" = 'db' ]
        set placeholder  "z $db_class"
    else
        set placeholder  "z"
    end

    if test -z (string match --entire "{}" $prompt0)
        set prompt "a $placeholder $prompt0"
    else
        set prompt (string replace "{}" $placeholder $prompt0)
    end

    set ckptname  (ls -1 -rt logs|grep $subject-$method|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end

    echo $subject: $ckptname 
    echo Prompt: $prompt
    if [ "$argv[1]" = 'db' ]
        python3 scripts/stable_txt2img.py --config configs/stable-diffusion/$config --ckpt logs/$ckptname/checkpoints/last.ckpt --ddim_eta 0.0 --n_samples 8 --ddim_steps $ddim_steps --gpu $GPU --prompt "$prompt" --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $folder
    else
        python3 scripts/stable_txt2img.py --config configs/stable-diffusion/$config --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0 --n_samples 8 --ddim_steps $ddim_steps --embedding_paths logs/$ckptname/checkpoints/embeddings_gs-$ckpt_iter.pt --gpu $GPU --prompt $prompt --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $folder
    end
end
