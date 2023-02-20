GPU="1"
EXTRA_ARGS="--num_composition_samples_per_batch 2"

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n michelleyeoh-lasr --gpus $GPU, --data_root data/michelleyeoh/  --placeholder_string "z" --no-test  --init_word "asian woman" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n monstertruck-lasr --gpus $GPU, --data_root data/princessmonstertruck/  --placeholder_string "y" --no-test  --init_word "black persian cat" --init_word_weights 1 1 3 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n sandraoh-lasr --gpus $GPU, --data_root data_unused/sandraoh/  --placeholder_string "z" --no-test  --init_word "asian woman" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n selenagomez-lasr --gpus $GPU, --data_root data/selenagomez/  --placeholder_string "z" --no-test  --init_word "young girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n smritimandhana-lasr --gpus $GPU, --data_root data/smritimandhana/  --placeholder_string "z" --no-test  --init_word "indian girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n spikelee-lasr --gpus $GPU, --data_root data/spikelee/  --placeholder_string "z" --no-test  --init_word "black man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n stephenchow-lasr --gpus $GPU, --data_root data/stephenchow/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n stevejobs-lasr --gpus $GPU, --data_root data/stevejobs/  --placeholder_string "z" --no-test  --init_word "white man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n taylorswift-lasr --gpus $GPU, --data_root data/taylorswift/  --placeholder_string "z" --no-test  --init_word "cute girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n timotheechalamet-lasr --gpus $GPU, --data_root data_unused/timotheechalamet/  --placeholder_string "z" --no-test  --init_word "french young man" --init_word_weights 1 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n tomholland-lasr --gpus $GPU, --data_root data_unused/tomholland/  --placeholder_string "z" --no-test  --init_word "young handsome man" --init_word_weights 1 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n xiuchao-lasr --gpus $GPU, --data_root data_unused/xiuchao  --placeholder_string "z" --no-test  --init_word "asian young girl" --init_word_weights 1 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n zendaya-lasr --gpus $GPU, --data_root data/zendaya/  --placeholder_string "z" --no-test  --init_word "young girl zendaya" --init_word_weights 1 2 2 $EXTRA_ARGS
