#!/usr/bin/fish
set fish_trace 1
set GPU "0"
set suffix ", instagram"
set -l subjects alexachung caradelevingne corgi donnieyen   gabrielleunion ianarmitage  jaychou     jenniferlawrence jiffpom    keanureeves lilbub lisa         masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez smritimandhana spikelee    stephenchow  taylorswift timotheechalamet  tomholland        zendaya
set -l prompts  girl       girl           corgi "asian man" "black woman"  "young boy"  "asian man" girl             "pom dog"  "white man" cat    "asian girl" "asian man" "asian woman" "black persian cat"  "white man" "asian woman" girl        "indian girl"  "black man" "asian man"  girl        "white young man" "white young man" girl
set n_iter 1
set scale 5
set outdir samples-dreambooth

for i in (seq 25)
    set subject $subjects[$i]
    set prompt0 $prompts[$i]
    set prompt "a photo of a z $prompt0$suffix"
    set ckptname  (ls -1 -rt logs|grep $subject|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end
    python3 scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --ddim_steps 100 --ckpt logs/$ckptname/checkpoints/last.ckpt --prompt $prompt --gpu $GPU --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $subject
end
