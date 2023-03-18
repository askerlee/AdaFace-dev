#!/usr/bin/fish
#set fish_trace 1
if [ "$argv[1]" = 'ada' ];  or [ "$argv[1]" = 'ti' ]
    set M $argv[1]
else
    echo "Usage: $0 [ada|ti]"
    exit 1
end

if test -z "$GPU"
    set GPU "1"
end

set n_iter 1
set outdir samples-$M
set scale 10
set config v1-inference-$M.yaml
set ddim_steps 100
fish scripts/composition-cases.sh

for case in $cases
    echo \"$case\"
    set -l case2 (string split " | " $case)
    set subject $case2[1]
    set prompt  $case2[2]
    set folder  $case2[3]
    
    if test -z (string match --entire "{}" $prompt)
        set prompt2 "a z $prompt"
    else
        set prompt2 (string replace "{}" "z" $prompt)
    end

    echo Prompt: $prompt2

    set ckptname  (ls -1 -rt logs|grep $subject-$M|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end

    python3 scripts/stable_txt2img.py --config configs/stable-diffusion/$config --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0 --n_samples 8 --ddim_steps $ddim_steps --embedding_paths logs/$ckptname/checkpoints/embeddings_gs-4000.pt --gpu $GPU --prompt "$prompt2" --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $folder
end
