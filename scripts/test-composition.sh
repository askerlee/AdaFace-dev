#!/usr/bin/fish
set self (status basename)
echo $self $argv

argparse --min-args 1 --max-args 1 'gpu=' 'steps=' 'scale=' 'niter=' -- $argv
or begin
    echo "Usage: $self [--gpu GPU-ID] [--steps DDIM_STEPS] [--scale scale] [--niter n_iter] (ada|ti)"
    exit 1
end

if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]
    set method $argv[1]
else
    echo "Usage: $self [--gpu GPU-ID] [--steps DDIM_STEPS] [--scale scale] [--niter n_iter] (ada|ti)"
    exit 1
end

# set fish_trace 1
set -q _flag_gpu; and set GPU $_flag_gpu; or set GPU 0
set -q _flag_steps; and set ddim_steps $_flag_steps; or set ddim_steps 100
set -q _flag_scale; and set scale $_flag_scale; or set scale 10
set -q _flag_niter; and set n_iter $_flag_niter; or set n_iter 1

set outdir samples-$method
set config v1-inference-$method.yaml
fish scripts/composition-cases.sh

for case in $cases
    echo \"$case\"
    set -l case2 (string split " | " $case)
    set subject $case2[1]
    set prompt0 $case2[2]
    set folder  $case2[3]
    
    if test -z (string match --entire "{}" $prompt0)
        set prompt "a z $prompt0"
    else
        set prompt (string replace "{}" "z" $prompt0)
    end

    set ckptname  (ls -1 -rt logs|grep $subject-$method|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end

    echo $subject: $ckptname 
    echo Prompt: $prompt
    python3 scripts/stable_txt2img.py --config configs/stable-diffusion/$config --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0 --n_samples 8 --ddim_steps $ddim_steps --embedding_paths logs/$ckptname/checkpoints/embeddings_gs-4000.pt --gpu $GPU --prompt $prompt --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $folder
end
