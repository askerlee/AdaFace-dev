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

\[Optional\] 
Some scripts are written for [fish shell](https://fishshell.com/). You can install it with `apt install fish`.

## Training

To invert an image set, run:

```bash
python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml
         -t --actual_resume models/stable-diffusion-v-1-5/v1-5-pruned.ckpt
         -n <run_name> --gpus 0, --no-test
         --data_root data/subject_images/
         --subject_string <subject_string>
         --cls_delta_string "word1 word2..." --subj_init_word_weights w1 w2...
```

where `subject_string` is a chosen uncommonly used single-token, such as ‘z’, ‘y’.

`cls_delta_string` is a few words that roughly describe the subject (e.g., 'man', ‘girl’, ‘dog’). It is used to initialize the low-rank semantic space of the subject embeddings.

`subj_init_word_weights` are the weights of each words in `cls_delta_string`. The number of weights are expected to be equal to the number of words. Intuitively, category words (”girl”, “man”, etc.) are given higher weights than modifier words (”young”, “chinese”, etc.).

The number of training iterations is specified in `configs/stable-diffusion/v1-finetune-ada.yaml`. We chose 4k training iterations.

To run on multiple GPUs, provide a comma-delimited list of GPU indices to the –gpus argument (e.g., `--gpus 0,1`)

Embeddings and output images will be saved in the log/<run_name> directory.

**Note**  All training set images should be upright, in square shapes, and clear (without obvious artifacts). Otherwise, the generated images will contain similar artifacts.

### Generation

To generate new images of the learned subject(s), run:

```bash
python scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-ada.yaml --ckpt models/stable-diffusion-v-1-5/v1-5-pruned.ckpt --ddim_eta 0.0
  --n_samples 8 --ddim_steps 50 --gpu 0 
  --embedding_paths logs/<run_name1>/checkpoints/embeddings_gs-4500.pt 
                    logs/<run_name2>/checkpoints/embeddings_gs-4500.pt
  --prompt "text containing the subject placeholder(s)" --scale 10 --n_iter 2
```

where multiple embedding paths can be specified for multi-subject composition. The prompt contains one or more placeholder tokens corresponding to the embedding checkpoints.

Example:

```bash
python3 scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-ada.yaml --ckpt models/stable-diffusion-v-1-5/v1-5-pruned.ckpt
--ddim_eta 0.0 --n_samples 8 --ddim_teps 50 --gpu 0
--embedding_paths 
   logs/donnieyen2023-02-20T23-52-39_donnieyen-ada/checkpoints/embeddings_gs-4500.pt 
   logs/lilbub2023-02-21T08-18-18_lilbub-ada/checkpoints/embeddings_gs-4500.pt 
--prompt "a z hugging a y" --scale 10 --n_iter 2
```
