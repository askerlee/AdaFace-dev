4GPU="0"
EXTRA_ARGS="--num_composition_samples_per_batch 2"

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n alexachung-lasr --gpus $GPU, --data_root data/alexachung/  --placeholder_string "z" --no-test  --init_word "young girl woman" --init_word_weights 1 2 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n caradelevingne-lasr --gpus $GPU, --data_root data/caradelevingne/  --placeholder_string "z" --no-test  --init_word "young girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n corgi-lasr --gpus $GPU, --data_root data/corgi/ --placeholder_string "y" --no-test  --init_word "corgi dog" --init_word_weights 2 1 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n donnieyen-lasr --gpus $GPU, --data_root data/donnieyen/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n gabrielleunion-lasr --gpus $GPU, --data_root data/gabrielleunion  --placeholder_string "z" --no-test  --init_word "black woman" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jaychou-lasr --gpus $GPU, --data_root data/jaychou/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jenniferlawrence-lasr --gpus $GPU, --data_root data/jenniferlawrence/  --placeholder_string "z" --no-test  --init_word "young woman" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jiffpom-lasr --gpus $GPU, --data_root data/jiffpom/  --placeholder_string "y" --no-test  --init_word "pom dog" --init_word_weights 1 1 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jimchapman-lasr --gpus $GPU, --data_root data/jimchapman/  --placeholder_string "z" --no-test  --init_word "white man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n keanureeves-lasr --gpus $GPU, --data_root data/keanureeves/  --placeholder_string "z" --no-test  --init_word "keanu cool man" --init_word_weights 2 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n lilbub-lasr --gpus $GPU, --data_root data/lilbub/  --placeholder_string "y" --no-test  --init_word "tabby cat" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n lisa-lasr --gpus $GPU, --data_root data/lisa/  --placeholder_string "z" --no-test  --init_word "asian young girl" --init_word_weights 1 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-adaprompt.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n masatosakai-lasr --gpus $GPU, --data_root data/masatosakai/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS
