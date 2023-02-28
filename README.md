# Layerwise Adaptive Subject Representations for Diffusion Models

> Layerwise Adaptive Subject Representations for Diffusion Models
Shaohua Li, Xiuchao Sui, Daquan Zhou, Menghan Zhou, Hong Yang, Weide Liu, Xinxing Xu, Yong Liu, Rick Goh
> 

> Abstract: A subject is a particular person, animal or object. This paper is about subject representation learning for diffusion models, to control diffusion models for subject-driven generation, i.e., rendering a specific subject or in a custom style in *novel contexts*. We propose three criteria for an ideal subject-driven generation method: 1) capture sufficient details of the subject with authenticity to retain its identity, 2) avoid forgetting existing concepts, and 3) be amenable to prompt compositions, such as modifying subject attributes or incorporating extra objects. The existing methods have yet to meet the three objectives satisfactorily. Here we propose Layerwise Adaptive Subject Representations (LASR), which for different layers of the diffusion model, generates a subject embedding specific to the dynamic layer features, with a layer-specific neural network. LASR is able to control the model precisely for generating more authentic images. Operating in the input embedding space, LASR naturally avoids forgetting of other concepts. The prompt composition ability is well maintained through a composition delta loss. Experiments on a set of few-shot subject images show that LASR is advantageous on all the three criteria than the representative subject-driven methods, Textual Inversion and DreamBooth.
> 

## Description

This repo contains the official code, data and sample generations for our LASR paper.

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run:

```bash
conda env create -f environment.yaml
conda activate ldm
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion).

Currently, the model can be downloaded by running:

```bash
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Usage

### Inversion

To invert an image set, run:

```bash
python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml
         -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt
         -n <run_name> --gpus 0, --no-test
         --data_root data/subject_images/
         --placeholder_string <placeholder_string>
         --init_word "word1 word2..." --init_word_weights w1 w2...
```

Example:

```jsx
python3 main.py --base configs/stable-diffusion/v1-finetune-lasr.yaml -t --actual_resume models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt -n alexachung-lasr --gpus 0, --data_root data/alexachung/  --placeholder_string "z" --no-test  --init_word "young girl woman" --init_word_weights 1 2 2
```

where `placeholder_string` is a chosen uncommonly used single-token, such as ‘z’, ‘y’.

`init_word` is a few words that roughly describe the subject (e.g., 'man', ‘girl’, ‘dog’). It is used to initialize the low-rank semantic space of the subject embeddings.

`init_word_weights` are the weights of each words in `init_word`. The number of weights are expected to be equal to the number of words. Intuitively, category words (”girl”, “man”, etc.) are given higher weights than modifier words (”young”, “chinese”, etc.).

The number of training iterations is specified in `configs/stable-diffusion/v1-finetune-lasr.yaml`. We chose 4k training iterations.

To run on multiple GPUs, provide a comma-delimited list of GPU indices to the –gpus argument (e.g., `--gpus 0,1`)

Embeddings and output images will be saved in the log/<run_name> directory.

**Note**  All training set images should be upright, in square shapes, and clear (without obvious artifacts). Otherwise, the generated images will contain similar artifacts.

### Generation

To generate new images of the learned concept, run:

```bash
python scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-lasr.yaml --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt --ddim_eta 0.0
--n_samples 8 --ddim_steps 100 --gpu 0 
--embedding_paths logs/<run_name1>/checkpoints/embeddings_gs-4000.pt 
                 logs/<run_name2>/checkpoints/embeddings_gs-4000.pt
--prompt "text containing the subject placeholder(s)" --scale 5 --n_iter 2
```

where multiple embedding paths can be specified for multi-subject composition. The prompt contains one or more placeholder tokens corresponding to the embedding checkpoints.

Example:

```bash
python3 scripts/stable_txt2img.py --config configs/stable-diffusion/v1-inference-lasr.yaml --ckpt models/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt 
--ddim_eta 0.0 --n_samples 8 --ddim_teps 100 --gpu 0
--embedding_paths 
logs/donnieyen2023-02-20T23-52-39_donnieyen-lasr/checkpoints/embeddings_gs-4000.pt 
logs/lilbub2023-02-21T08-18-18_lilbub-lasr/checkpoints/embeddings_gs-4000.pt 
--prompt "a z hugging a y" --scale 5 --n_iter 2
```

## Citation

If you make use of our work, please cite our paper:

```
@misc{gal2022textual,
      url = {https://arxiv.org/abs/xxxx.xxxx},
      author = {Li, Shaohua and Sui, Xiuchao and Zhou, Daquan and Zhou, Menghan and Yang, Hong and Liu, Weide and Xu, Xinxing and Liu, Yong and Goh, Rick},
      title = {Layerwise Adaptive Subject Representations for Diffusion Models},
      publisher = {arXiv},
      year = {2023},
      primaryClass={cs.CV}
}
```

## Results

Here are some sample results.

![https://www.notion.soimg/teaser.jpg](https://www.notion.soimg/teaser.jpg)

![https://www.notion.soimg/samples.jpg](https://www.notion.soimg/samples.jpg)

![https://www.notion.soimg/style.jpg](https://www.notion.soimg/style.jpg)