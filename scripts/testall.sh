#!/usr/bin/fish
set fish_trace 1
set GPU "0"
set -l subjects alexachung         caradelevingne corgi        donnieyen   iainarmitage gabrielleunion jaychou     jenniferlawrence jiffpom    keanureeves      lilbub       lisa                masatosakai michelleyeoh  princessmonstertruck ryangosling sandraoh      selenagomez    smritimandhana spikelee    stephenchow   taylorswift  timotheechalamet  tomholland            zendaya
set niter 1

for i in (seq 25)
    set subject $subjects[$i]
    set prompt  "a z"
    set ckptname  (ls -1 -rt logs|grep $subject|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end
    python3 scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-ada.yaml --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0 --n_samples 8 --ddim_steps 100 --embedding_paths logs/$CKPT/checkpoints/embeddings_gs-4000.pt --gpu $GPU --prompt "a z" --scale 5 --n_iter $niter --outdir samples --indiv_subdir $subject
end
