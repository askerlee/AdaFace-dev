#!/usr/bin/bash
if [ $# -lt 1 ]; then
    echo "Usage: $0 <embedding_path> <arg2> <arg3>..."
    exit 1
fi
python3 scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-layerwise.yaml --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 100 --prompt "a photo of a y" --embedding_path $1 $2 $3 $4 $5 $6 $7 $8 $9