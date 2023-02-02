GPU="0"
python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n alexachung-lasr --gpus $GPU, --data_root data/alexachung/  --placeholder_string "z" --no-test  --init_word "young girl woman" --init_word_weights 1 2 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n corgi-lasr --gpus $GPU, --data_root unused_data/corgi/ --placeholder_string "y" --no-test  --init_word "corgi dog" --init_word_weights 2 1

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n donnieyen-lasr --gpus $GPU, --data_root data/donnieyen/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n gabrielleunion-lasr --gpus $GPU, --data_root data/gabrielleunion  --placeholder_string "z" --no-test  --init_word "black woman" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jaychou-lasr --gpus $GPU, --data_root data/jaychou/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jenniferlawrence-lasr --gpus $GPU, --data_root data/jenniferlawrence/  --placeholder_string "z" --no-test  --init_word "young woman" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jiffpom-lasr --gpus $GPU, --data_root data/jiffpom/  --placeholder_string "y" --no-test  --init_word "pom dog" --init_word_weights 1 1

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n jimchapman-lasr --gpus $GPU, --data_root data/jimchapman/  --placeholder_string "z" --no-test  --init_word "white man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n keanureeves-lasr --gpus $GPU, --data_root data/keanureeves/  --placeholder_string "z" --no-test  --init_word "keanu cool man" --init_word_weights 2 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n lilbub-lasr --gpus $GPU, --data_root data/lilbub/  --placeholder_string "y" --no-test  --init_word "tabby cat" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n lisa-lasr --gpus $GPU, --data_root data/lisa/  --placeholder_string "z" --no-test  --init_word "asian young girl" --init_word_weights 1 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n masatosakai-lasr --gpus $GPU, --data_root data/masatosakai/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n michelleyeoh-lasr --gpus $GPU, --data_root data/michelleyeoh/  --placeholder_string "z" --no-test  --init_word "asian woman" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n monstertruck-lasr --gpus $GPU, --data_root data/princessmonstertruck/  --placeholder_string "y" --no-test  --init_word "black persian cat" --init_word_weights 1 1 3

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n sandraoh-lasr --gpus $GPU, --data_root unused_data/sandraoh/  --placeholder_string "z" --no-test  --init_word "asian woman" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n selenagomez-lasr --gpus $GPU, --data_root data/selenagomez/  --placeholder_string "z" --no-test  --init_word "young girl" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n smritimandhana-lasr --gpus $GPU, --data_root data/smritimandhana/  --placeholder_string "z" --no-test  --init_word "indian girl" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n spikelee-lasr --gpus $GPU, --data_root data/spikelee/  --placeholder_string "z" --no-test  --init_word "black man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n stephenchow-lasr --gpus $GPU, --data_root data/stephenchow/  --placeholder_string "z" --no-test  --init_word "asian man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n stevejobs-lasr --gpus $GPU, --data_root data/stevejobs/  --placeholder_string "z" --no-test  --init_word "white man" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n taylorswift-lasr --gpus $GPU, --data_root data/taylorswift/  --placeholder_string "z" --no-test  --init_word "cute girl" --init_word_weights 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n timotheechalamet-lasr --gpus $GPU, --data_root unused_data/timotheechalamet/  --placeholder_string "z" --no-test  --init_word "french young man" --init_word_weights 1 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n tomholland-lasr --gpus $GPU, --data_root unused_data/tomholland/  --placeholder_string "z" --no-test  --init_word "young handsome man" --init_word_weights 1 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n xiuchao-lasr --gpus $GPU, --data_root unused_data/xiuchao  --placeholder_string "z" --no-test  --init_word "asian young girl" --init_word_weights 1 1 2

python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n zendaya-lasr --gpus $GPU, --data_root data/zendaya/  --placeholder_string "z" --no-test  --init_word "young girl zendaya" --init_word_weights 1 2 2
