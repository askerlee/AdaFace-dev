GPU="0"
EXTRA_ARGS=""

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n alexachung-ada --gpus $GPU, --data_root data/alexachung/  --placeholder_string "z" --no-test  --init_word "young girl woman" --init_word_weights 1 2 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n caradelevingne-ada --gpus $GPU, --data_root data/caradelevingne/  --placeholder_string "z" --no-test  --init_word "young girl" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n corgi-ada --gpus $GPU, --data_root data/corgi/ --placeholder_string "y" --no-test  --init_word "corgi dog" --init_word_weights 2 1 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n donnieyen-ada --gpus $GPU, --data_root data/donnieyen/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n gabrielleunion-ada --gpus $GPU, --data_root data/gabrielleunion  --placeholder_string "z" --no-test  --init_word "black woman" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jaychou-ada --gpus $GPU, --data_root data/jaychou/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jenniferlawrence-ada --gpus $GPU, --data_root data/jenniferlawrence/  --placeholder_string "z" --no-test  --init_word "young woman" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jiffpom-ada --gpus $GPU, --data_root data/jiffpom/  --placeholder_string "y" --no-test  --init_word "pom dog" --init_word_weights 1 1 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jimchapman-ada --gpus $GPU, --data_root data/jimchapman/  --placeholder_string "z" --no-test  --init_word "white man" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n keanureeves-ada --gpus $GPU, --data_root data/keanureeves/  --placeholder_string "z" --no-test  --init_word "keanu cool man" --init_word_weights 2 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n lilbub-ada --gpus $GPU, --data_root data/lilbub/  --placeholder_string "y" --no-test  --init_word "tabby cat" --init_word_weights 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n lisa-ada --gpus $GPU, --data_root data/lisa/  --placeholder_string "z" --no-test  --init_word "asian young girl" --init_word_weights 1 1 2 $EXTRA_ARGS

python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n masatosakai-ada --gpus $GPU, --data_root data/masatosakai/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2 $EXTRA_ARGS
