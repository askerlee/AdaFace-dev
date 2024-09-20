# AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization

> AdaFace: A Versatile Face Encoder for Zero-Shot Diffusion Model Personalization
>
> Shaohua Li, Xiuchao Sui, Hong Yang, Pin Nean Lai, Weide Liu, Xinxing Xu, Yong Liu, Rick Siow Mong Goh
>
> Abstract: Since the advent of diffusion models, personalizing these models -- conditioning them to render novel subjects -- has been widely studied. Recently, several methods propose training a dedicated image encoder on a large variety of subject images. This encoder maps the images to identity embeddings (ID embeddings). During inference, these ID embeddings, combined with conventional prompts, condition a diffusion model to generate new images of the subject. However, such methods often face challenges in achieving a good balance between authenticity and compositionality -- accurately capturing the subject's likeness while effectively integrating them into varied and complex scenes. A primary source for this issue is that the ID embeddings reside in the *image token space* (``image prompts"), which is not fully composable with the text prompt encoded by the CLIP text encoder. In this work, we present AdaFace, an image encoder that maps human faces into the *text prompt space*. After being trained only on 400K face images with 2 GPUs, it achieves high authenticity of the generated subjects and high compositionality with various text prompts. In addition, as the ID embeddings are integrated in a normal text prompt, it is highly compatible with existing pipelines and can be used without modification to generate authentic videos. We showcase the generated images and videos of celebrities under various compositional prompts. 
> 

## Online Demos
- AdaFace text-to-image: <a href="https://huggingface.co/spaces/adaface-neurips/adaface" style="display: inline-flex; align-items: center;">
  AdaFace 
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces" style="margin-left: 5px;">
  </a>
- AdaFace text-to-video: <a href="https://huggingface.co/spaces/adaface-neurips/adaface-animate" style="display: inline-flex; align-items: center;">
  AdaFace-Animate
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces" style="margin-left: 5px;">
  </a>

## Repo Description

This repo contains the official training and evaluation code, test data and sample generations for our AdaFace paper.

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up the environment using pip, please run:

```bash
pip install -r requirements0.txt
pip install -r requirements.txt
pip install -e .
```

## Downloading Pre-trained Models
1. Arc2Face
Arc2Face can be downloaded from https://github.com/foivospar/Arc2Face. The pretrained model weights should be placed in the `models/arc2face` directory.

2. SD-1.5
The stable diffusion model weight is the official SD 1.5 model from https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt. The VAE and text encoder are replaced with the [MSE-840000 finetuned VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/tree/main) and the [DreamShaper V8 text encoder](https://civitai.com/models/4384?modelVersionId=252914), respectively. The weights should be placed in the `models/stable-diffusion-v-1-5` directory:
```
python3 scripts/repl_textencoder.py --base_ckpt models/stable-diffusion-v-1-5/v1-5-pruned.ckpt --text_ckpt models/dreamshaper/DreamShaper_8_pruned.safetensors --out_ckpt models/stable-diffusion-v-1-5/v1-5-dste8.ckpt
python3 scripts/repl_vae.py --base_ckpt models/stable-diffusion-v-1-5/v1-5-dste8.ckpt --vae_ckpt models/stable-diffusion-v-1-5/vae-ft-mse-840000-ema-pruned.ckpt --out_ckpt models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt
```

\[Optional\] 
Some scripts are written for [fish shell](https://fishshell.com/). You can install it with `apt install fish`.

## Preparation of Training Data
We downloaded the VGGFace2 dataset and FFHQ dataset, and extracted the face masks using the [BiSeNet for face parsing](https://github.com/zllrunning/face-parsing.PyTorch) model. Bad images that do not contain clear full faces are discarded. The (face images, face masks) pairs are saved in the `VGGface2_HQ_masks` and `FFHQ_masks` directories, respectively. The naming convention is: if the face image is `FFHQ_masks/00000.png`, then the corresponding mask is named as `FFHQ_masks/00000_mask.png`.
Note that the face masks are only used for training and are not required for inference.

Our face segmentation and filtering code is [gen_face_masks.py](/scripts-private/gen_face_masks.py).

## Training
The training are divided into two stages: training the AdaFace inverse encoder by distillation on Arc2Face, and training the AdaFace with compositionality distillation.

In all training stages, `accumulate_grad_batches` is set to 2 in the `configs/stable-diffusion/v1-finetune-ada.yaml` file. This is to double the effective batch size from 4 to 8 (Stage 1) or 3 to 6 (Stage 2) for faster convergence. As a result, say after 120000 iterations with bs=4, the final checkpoint is equivalent to training with a batch size of 8 for 60000 iterations, and saved into a file `embeddings_gs-60000.pt`.

### Stage 1: Training AdaFace Inverse Encoder
In this stage, we only do image reconstruction by learning from an [Arc2Face](https://github.com/foivospar/Arc2Face) teacher model.

```bash
python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --base_model_path models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt --gpus 0,1 --data_roots /path/to/VGGface2_HQ_masks --mix_subj_data_roots /path/to/FFHQ_masks -n zero-ada --no-test --max_steps 120000 --subject_string z --background_string y --num_vectors_per_bg_token 4 --num_vectors_per_subj_token 16 --clip_last_layers_skip_weights 1 2 2 --randomize_clip_skip_weights --warmup_steps 600 --d_coef 1 --bs 4 --do_comp_teacher_filtering 1 --p_unet_distill_iter 1 --composition_regs_iter_gap 0
```

where `--composition_regs_iter_gap 0` disables the compositionality iterations. The `configs/stable-diffusion/v1-finetune-ada.yaml` file specifies the training configuration. The `--base_model_path` option specifies the SD-1.5 checkpoint to resume from. The `--data_roots` option specifies the directories containing the face images and masks of multiple subjects, where each subject is in an individual folder. The `--mix_subj_data_roots` option specifies the directories containing the faces of multiple subjects in the same folder.

The optimizer used is [Prodigy](https://github.com/konstmish/prodigy) with `d_coef` of 1. The learning rate is linearly decayed from 1 to 0.1 after the warmup steps. The learning rate and d_coef jointly control the actual learning rate. The `--clip_last_layers_skip_weights 1 2 2` option specifies that the output features from the CLIP text model are an weighted average of the last 3 layers, and the weights are randomly drawn from a Dirichlet prior $Dir(1, 2, 2)$. The `--p_unet_distill_iter 1` indicates that the AdaFace distillation on an Arc2Face teacher model is performed at every iteration. The `--subject_string z` and `--background_string y` options specify the subject and background tokens, respectively. The  `--num_vectors_per_subj_token 16` and `--num_vectors_per_bg_token 4` options specify the number of vectors per subject and background token, respectively.

The checkpoints will be saved in the log/<run_name> directory.

The training process above is expected to take 1-2 days on 2*A6000 GPUs. The same training process is repeated twice, totaling 240K steps. Suppose the first run is saved in `logs/VGGface2_HQ_masks2024-04-26T16-30-17_zero-ada`, then the second run can be resumed with `--adaface_ckpt_path logs/VGGface2_HQ_masks2024-04-26T16-30-17_zero-ada/checkpoints/embeddings_gs-60000.pt --extend_prompt2token_proj_attention_multiplier 4`. The two-round training is to allow Prodigy to reset the optimizer state to explore different local minima. The `--extend_prompt2token_proj_attention_multiplier 4` option increases the number of K and V in attention layers by 4x in the prompt2token projection CLIP model.

Optionally, you can specify `--save_meta_subj2person_type_cache_path /path/to/meta_subj2person_type.json` to cache the meta data for the VGGFace2 dataset, and load it later with `--load_meta_subj2person_type_cache_path /path/to/meta_subj2person_type.json` to speed up the data loading process.

### Stage 2: Training AdaFace with Compositionality Distillation
In this stage, we alternate between the compositionality distillation and the image reconstruction distillation. In compositionality distillation, the prompt is a compositional one randomly drawn with [/ldm/data/compositions.py](/ldm/data/compositions.py). The teacher is the denoised output from the SD-1.5 model, with the subject tokens replaced by a class token (e.g., `z0 ... z15 running in a park` -> `woman running in a park`). The teacher model is used to generate two images of the type of people, and the AdaFace model is used to generate two images of the subject. The compositional delta loss is computed between the two sets of images. The compositionality distillation is performed every 3 iterations.

```bash
python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --base_model_path models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt --gpus 0,1 --data_roots /path/to/VGGface2_HQ_masks --mix_subj_data_roots /path/to/FFHQ_masks -n zero3-ada --no-test --max_steps 60000 --subject_string z --background_string y --num_vectors_per_bg_token 4 --num_vectors_per_subj_token 16 --clip_last_layers_skip_weights 1 2 2 --randomize_clip_skip_weights --warmup_steps 1000 --d_coef 0.5 --bs 3 --p_unet_distill_iter 0.2 --composition_regs_iter_gap 3 --adaface_ckpt_path logs/VGGface2_HQ_masks2024-04-29T18-19-49_zero3-ada/checkpoints/embeddings_gs-30000.pt --extend_prompt2token_proj_attention_multiplier 2 --load_meta_subj2person_type_cache_path /path/to/meta_subj2person_type.json --max_num_denoising_steps 3
```

Note that the `--composition_regs_iter_gap 3` option specifies that the compositionality distillation is performed every 3 iterations. The `--max_num_denoising_steps 3` option specifies that the multi-step denoising is performed at most 3 times. The `--p_unet_distill_iter 0.2` option specifies that the AdaFace distillation on an Arc2Face teacher model is performed with a probability of `0.2` (in contrast to `1` in the first stage). The `--extend_prompt2token_proj_attention_multiplier 2` option increases the number of K and V in attention layers by 2x in the prompt2token projection CLIP model.

## Evaluation

To generate new images of the learned subject(s) and compute the CLIP scores and the face similarity with the reference images, run the following command:

```bash
python3 scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-ada.yaml --ckpt models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt --bb_type '' --ddim_eta 0.0 --ddim_steps 50 --gpu 1 --scale 4 1 --broad_class 1 --n_repeat 1 --bs 4 --outdir samples-ada --from_file samples-ada/iainarmitage-prompts-all-18.txt --clip_last_layers_skip_weights 1 1 --ref_images subjects-celebrity/iainarmitage/ --subject_string z --background_string y --num_vectors_per_bg_token 4 --use_pre_neg_prompt 1 --compare_with subjects-celebrity/iainarmitage/ --calc_face_sim --embedding_paths logs/VGGface2_HQ_masks05-10T22-42-17_zero3-ada/checkpoints/embeddings_gs-30000.pt
```

where `--from_file` specifies a file containing a list of text prompts, and `--ref_images` specifies the directory containing the reference images of the subject(s), which should be the same as `--compare_with` for face similarity computation. The `--embedding_paths` option specifies the path to the AdaFace encoder checkpoint. The `--broader_class` option specifies the broader class of the subject (always 1 for faces). The `--scale` option specifies two guidance scales for the classifier-free guidance; in this case, the scale linearly decreases from 4 to 1. The `--use_pre_neg_prompt 1` option specifies that the predefined negative prompt is used. The `--clip_last_layers_skip_weights 1 1` option specifies that the output features from the CLIP text model are an average of the last 2 layers.

The prompt list file, `samples-ada/iainarmitage-prompts-all-18.txt`, is generated by the following command:

```bash
python3 scripts/gen_subjects_and_eval.py --method ada --subjfile evaluation/info-subjects.sh --out_dir_tmpl
 samples --num_vectors_per_subj_token 16 --use_pre_neg_prompt --subject_string z --range 6 --gen_prompt_set_only
```
where `--subjfile` specifies the file containing the list of subjects, and `--range` specifies which subject to generate prompts for. In this case, the subject file is `evaluation/info-subjects.sh`, which corresponds to the subjects in the `subjects-celebrity` directory. The `--range 6` is the 6th subject in the directory, namely `iainarmitage`. If `--gen_prompt_set_only` is specified, only the prompt list file is generated without generating the images. Otherwise, it proceeds to generate the images as the previous command does.

## Integration with Other Applications
The AdaFace encoder is wrapped in the class [AdaFaceWrapper](/adaface/adaface_wrapper.py) for easy integration with other applications. 

`AdaFace-Animate` is an example of integrating AdaFace with AnimateDiff and ID-Animator for subject-driven video generation. The code is available at [AdaFace-Animate](https://huggingface.co/spaces/adaface-neurips/adaface-animate/tree/main). The online demo is available at [AdaFace-Animate Demo](https://huggingface.co/spaces/adaface-neurips/adaface-animate).

- TODO:
Integration with ControlNet.
