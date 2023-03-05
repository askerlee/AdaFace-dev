GPU="1"
EXTRA_ARGS=""

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n selenagomez-ada --gpus $GPU, --data_root data/selenagomez/  --placeholder_string "z" --no-test  --init_word "young girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n smritimandhana-ada --gpus $GPU, --data_root data/smritimandhana/  --placeholder_string "z" --no-test  --init_word "indian girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n stevejobs-ada --gpus $GPU, --data_root data/stevejobs/  --placeholder_string "z" --no-test  --init_word "white man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n taylorswift-ada --gpus $GPU, --data_root data/taylorswift/  --placeholder_string "z" --no-test  --init_word "cute girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n zendaya-ada --gpus $GPU, --data_root data/zendaya/  --placeholder_string "z" --no-test  --init_word "young girl zendaya" --init_word_weights 1 2 2 $EXTRA_ARGS
