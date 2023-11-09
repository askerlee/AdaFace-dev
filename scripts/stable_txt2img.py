import argparse, os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
import re
import csv
import sys
from collections import namedtuple

from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config, mix_embeddings, save_grid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from evaluation.eval_utils import compare_folders, compare_face_folders_fast, \
                                  init_evaluators, set_tf_gpu
from ldm.data.personalized import PersonalizedBase

from safetensors.torch import load_file as safetensors_load_file


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )    
    # --use_pre_neg_prompt, use predefined negative prompts
    parser.add_argument(
        "--use_pre_neg_prompt",
        action='store_true',
        help="use predefined negative prompts",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--indiv_subdir",
        type=str,
        help="subdir to write individual images to",
        default="samples"
    )      
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=1,
        help="Go through all prompts this times",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="How many samples to produce for each given prompt. " 
             "Usually used if prompts are not loaded from a file (--from_file not specified)",
    )
    parser.add_argument("--bs", type=int, default=-1, 
                        help="batch size. If -1, use n_samples") 
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: batch_size)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference-ada.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/stable-diffusion-v-1-5/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--embedding_paths", 
        nargs="*", 
        type=str, default=None,
        help="One or more paths to pre-trained embedding manager checkpoints")

    parser.add_argument("--ada_emb_weight",
        type=float, default=-1,
        help="Weight of adaptive embeddings (in contrast to static embeddings)")

    parser.add_argument(
        "--init_img_paths",
        type=str,
        nargs='+', 
        default=None,
        help="path to the initial image(s) (multiple images will be averaged)",
    )  
    parser.add_argument(
        "--use_first_gt_img_as_init",
        action='store_true',
        help="use the first image in the groundtruth folder as the initial image",
    )
    # Anything between 0 and 1 will cause blended images.
    parser.add_argument(
        "--init_img_weight",
        type=float,
        default=0.1,
        help="Weight of the initial image (if w, then w*img + (1-w)*noise)",
    )
    parser.add_argument(
        "--init_mask_paths",
        type=str,
        nargs='+', 
        default=None,
        help="path to the initial mask(s)",
    )      
    # No preview
    parser.add_argument(
        "--no_preview",
        action='store_true',
        help="do not preview the image",
    )
    parser.add_argument("--broad_class", type=int, default=1,
                        help="Whether the subject is a human/animal, object or cartoon"
                             " (0: object, 1: human/animal, 2: cartoon)")
    parser.add_argument("--calc_face_sim", action="store_true",
                        help="If specified, assume the generated samples are human faces, "
                             "and compute face similarities with the groundtruth")
        
    parser.add_argument('--gpu', type=int,  default=0, help='ID of GPU to use. Set to -1 to use CPU (slow).')
    parser.add_argument("--compare_with", type=str, default=None,
                        help="Evaluate the similarity of generated samples with reference images in this folder")
    parser.add_argument("--class_prompt", type=str, default=None,
                        help="the original prompt used for text/image matching evaluation "
                             "(requires --compare_with to be specified)")
    parser.add_argument("--ref_prompt", type=str, default=None,
                        help="a class-level reference prompt to be mixed with the subject prompt "
                             "(if None, then don't mix with reference prompt)")
    parser.add_argument("--prompt_mix_scheme", type=str, default=argparse.SUPPRESS, 
                        choices=['mix_hijk', 'mix_concat_cls'],
                        help="How to mix the subject prompt with the reference prompt")
    
    parser.add_argument("--prompt_mix_weight", type=float, default=0,
                        help="Weight of the reference prompt to be mixed with the subject prompt (0 to disable)")
        
    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[0.5, 0.5],
                        help="Weight of the skip connections of the last few layers of CLIP text embedder. " 
                             "NOTE: the last element is the weight of the last layer.")

    parser.add_argument("--placeholder_string", 
                        type=str, default="z",
                        help="Placeholder string which will be used in prompts to denote the concept.")
    parser.add_argument("--background_string", 
                        type=str, default=None,
                        help="Background string which will be used in prompts to denote the background in training images.")
                    
    parser.add_argument("--num_vectors_per_token",
                        type=int, default=argparse.SUPPRESS,
                        help="Number of vectors per token. If > 1, use multiple embeddings to represent a subject.")
    parser.add_argument("--use_conv_attn",
                        action="store_true", 
                        help="Use convolutional attention at subject tokens")
    parser.add_argument("--default_point_conv_attn_mix_weight",
                        type=float, default=0.5,
                        help="Weight of convolutional attention (to be combined with pointwise attention)")
        
    parser.add_argument("--emb_ema_as_pooling_probe",
                        action="store_true", default=argparse.SUPPRESS,
                        help="Use EMA embedding as the pooling probe")
    # If normalize_subj_attn is not specified, then use the 'normalize_subj_attn' in the checkpoint.
    # If 'normalize_subj_attn' doesn't exist in the checkpoint, then normalize_subj_attn is False.
    parser.add_argument("--normalize_subj_attn", type=str2bool, nargs="?", const=True, default=argparse.SUPPRESS,
                        help="Whether to normalize the subject embedding attention scores")

    # bb_type: backbone checkpoint type. Just to append to the output image name for differentiation.
    # The backbone checkpoint is specified by --ckpt.
    parser.add_argument("--bb_type", type=str, default="")

    parser.add_argument("--scores_csv", type=str, default=None,
                        help="CSV file to save the evaluation scores")
    # --debug
    parser.add_argument("--debug", action="store_true",
                        help="debug mode")
    # --eval_blip
    parser.add_argument("--eval_blip", action="store_true",
                        help="Evaluate BLIP-diffusion models")
    # cls_token
    parser.add_argument("--cls_token", type=str, default=None,
                        help="Subject class name. Only requires for --eval_blip")
    
    args = parser.parse_args()
    return args

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith(".ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
    elif ckpt.endswith(".safetensors"):
        sd = safetensors_load_file(ckpt, device="cpu")
        pl_sd = None
    else:
        print(f"Unknown checkpoint format: {ckpt}")
        sys.exit(1)

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    # Release some RAM. Not sure if it really works.
    del sd, pl_sd

    return model

# copied from img2img.py
def load_img(path, h, w):
    image = Image.open(path).convert("RGB")
    w0, h0 = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # round w, h to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    # b h w c -> b c h w
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main(opt):

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    # No GPUs detected. Use CPU instead.
    if not torch.cuda.is_available():
        opt.gpu = -1
    else:
        torch.cuda.set_device(opt.gpu)
        torch.backends.cuda.matmul.allow_tf32 = True

    seed_everything(opt.seed)
    #predefined_negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    predefined_negative_prompt = "duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions"
    
    if not opt.eval_blip:
        config = OmegaConf.load(f"{opt.config}")

        if opt.use_conv_attn:
            assert opt.num_vectors_per_token in [4, 9, 16], \
                    f"Only support 4/9/16 embeddings per token but got {opt.num_vectors_per_token}. " \
                    "4 = 2*2 kernel, 9 = 3*3, 16 = 4*4."
            config.model.params.use_conv_attn = True
            config.model.params.default_point_conv_attn_mix_weight = opt.default_point_conv_attn_mix_weight
            config.model.params.personalization_config.params.default_point_conv_attn_mix_weight = opt.default_point_conv_attn_mix_weight
            kernel_desc_map = {4: "2x2", 9: "3x3", 16: "4x4"}
            kernel_desc = kernel_desc_map[opt.num_vectors_per_token]
            print(f"Use {kernel_desc} Conv Attention (default weight {opt.default_point_conv_attn_mix_weight}) by subject embeddings")

        model  = load_model_from_config(config, f"{opt.ckpt}")
        if opt.embedding_paths is not None:
            model.embedding_manager.load(opt.embedding_paths)
            model.embedding_manager.eval()
            opt.subj_model_path = opt.embedding_paths[0]
        else:
            opt.subj_model_path = opt.ckpt

        # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
        model.cond_stage_model.set_last_layers_skip_weights(opt.clip_last_layers_skip_weights)
        
        if hasattr(opt, 'num_vectors_per_token'):
            ckpt_num_vectors_per_token = model.embedding_manager.token2num_vectors[opt.placeholder_string]
            assert ckpt_num_vectors_per_token == opt.num_vectors_per_token, \
                   f"Number of vectors per token mismatch: command line {opt.num_vectors_per_token} != ckpt {ckpt_num_vectors_per_token}."

        if hasattr(opt, 'emb_ema_as_pooling_probe'):
            model.embedding_manager.set_emb_ema_as_pooling_probe(opt.emb_ema_as_pooling_probe)

        if opt.ada_emb_weight != -1 and model.embedding_manager is not None:
            model.embedding_manager.ada_emb_weight = opt.ada_emb_weight
        if opt.background_string is not None:
            model.embedding_manager.background_strings = [opt.background_string]

        # command line --normalize_subj_attn overrides the checkpoint.
        if hasattr(opt, 'normalize_subj_attn') and opt.normalize_subj_attn != model.embedding_manager.normalize_subj_attn:
            print(f"Override normalize_subj_attn in checkpoint ({model.embedding_manager.normalize_subj_attn} or absent) with {opt.normalize_subj_attn}")
            model.embedding_manager.normalize_subj_attn = opt.normalize_subj_attn

        device = torch.device(f"cuda:{opt.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        model  = model.to(device)
        model.cond_stage_model.device = device
                                
        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        if opt.neg_prompt == "" and opt.use_pre_neg_prompt:
            # negative prompt borrowed from BLIP-Diffusion.
            opt.neg_prompt = predefined_negative_prompt

    # eval_blip
    else:
        from lavis.models import load_model_and_preprocess
        blip_model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device=f"cuda:{opt.gpu}", is_eval=True)
        blip_model.load_checkpoint(opt.ckpt)
        cond_subject = opt.cls_token
        tgt_subject  = opt.cls_token
        cond_subjects = [txt_preprocess["eval"](cond_subject)]
        tgt_subjects  = [txt_preprocess["eval"](tgt_subject)]
        negative_prompt = predefined_negative_prompt
        opt.subj_model_path = ""
        
        class DummyScope(object):
            def __init__(self):
                pass
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass

        DummyModel = namedtuple('DummyModel', ['ema_scope'])
        # model.ema_scope() is a do-nothing object.
        model = DummyModel(DummyScope)

    os.makedirs(opt.outdir, exist_ok=True)

    if opt.scores_csv:
        SCORES_CSV_FILE = open(opt.scores_csv, "a")
        SCORES_CSV = csv.writer(SCORES_CSV_FILE)
    else:
        SCORES_CSV = None

    batch_size = opt.n_samples if opt.bs == -1 else opt.bs
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        assert opt.prompt is not None
        prompt = opt.prompt
        all_prompts = [prompt] * opt.n_samples
        # By default, batch_size = n_samples. In this case, chunking turns all_prompts into a list of length 1,
        # and the sole elment is a list containing "prompt" repeated n_samples times,
        # e.g. [ ['z', 'z', 'z', 'z'] ]. Then tqdm() will finish it in one iteration.
        batched_prompts = list(chunk(all_prompts, batch_size))
        batched_subdirs = [ opt.indiv_subdir ] * len(batched_prompts)
        # Append None to the end of batched_subdirs, for indiv_subdir change detection.
        batched_subdirs.append(None)

        if opt.compare_with:
            assert opt.class_prompt is not None, "Must specify --class_prompt when calculating CLIP similarities."

        batched_class_long_prompts = [ opt.class_prompt ] * len(batched_prompts)
        batched_ref_prompts        = [ opt.ref_prompt ]   * len(batched_prompts)

    else:
        print(f"Reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            # splitlines() will remove the trailing newline. So no need to strip().
            lines = f.read().splitlines()
            indiv_subdirs_prompts = [ line.split("\t") for line in lines ]
            n_repeats, indiv_subdirs, all_prompts, class_long_prompts, class_short_prompts \
                    = zip(*indiv_subdirs_prompts)
            # Repeat each prompt n_repeats times, and split into batches of size batch_size.
            # If there's remainder after chunks, the last chunk will be shorter than batch_size.

            batched_prompts = []
            batched_subdirs = []
            batched_class_long_prompts = []
            batched_ref_prompts = []

            # Repeat each prompt n_repeats times.
            for i, prompt in enumerate(all_prompts):
                n_repeat = int(n_repeats[i])
                # If in this line, n_repeat is larger than batch_size, we need to split it 
                # into n_batches > 1 batches. These batches share the same indiv_subdir,
                # class_long_prompt.
                # So no need to repeat them in advance. Just append n_batches copies of them 
                # to batched_subdirs and batched_class_long_prompts respectively below.
                indiv_subdir = indiv_subdirs[i]
                class_long_prompt = class_long_prompts[i]
                class_short_prompt = class_short_prompts[i]
                # The number of prompts in batched_prompts has to match the number of samples.
                # So we need to repeat the prompt by n_repeat times.
                prompts_repeated = [prompt] * n_repeat
                n_batches = n_repeat // batch_size
                if n_repeat % batch_size != 0:
                    n_batches += 1
                for bi in range(n_batches):
                    start_idx = batch_size * bi
                    end_idx   = batch_size * (bi + 1)
                    batched_prompts.append(prompts_repeated[start_idx:end_idx])

                batched_subdirs.extend([indiv_subdir] * n_batches)
                batched_class_long_prompts.extend([class_long_prompt] * n_batches)
                batched_ref_prompts.extend([class_short_prompt] * n_batches)

            # Append None to the end of batched_subdirs, for indiv_subdir change detection.
            batched_subdirs.append(None)

    if opt.compare_with:
        clip_evator, dino_evator = init_evaluators(-1)
        all_sims_img, all_sims_text, all_sims_dino = [], [], []
        if opt.calc_face_sim:
            all_sims_face = []
            all_normal_img_counts = []
            all_except_img_counts = []
            set_tf_gpu(opt.gpu)
    else:
        clip_evator, dino_evator = None, None

    if opt.fixed_code:
        # If init_img_paths or use_first_gt_img_as_init is specified, then start_code will be overwritten.
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    else:
        start_code = None

    if opt.use_first_gt_img_as_init:
        # Cannot specify init_img_paths and use_first_gt_img_as_init at the same time.
        assert opt.init_img_paths is None and opt.init_mask_paths is None, \
            "Cannot use 'init_img_paths'/'init_mask_paths' and 'use_first_gt_img_as_init' at the same time."
        assert opt.compare_with is not None, "Must specify --compare_with when using use_first_gt_img_as_init."
        gt_data_loader  = PersonalizedBase(opt.compare_with, set='evaluation', size=opt.H, flip_p=0.0)
        opt.init_img_paths  = gt_data_loader.image_paths
        opt.init_mask_paths = gt_data_loader.fg_mask_paths
        
    if opt.init_img_paths is not None:
        if opt.init_mask_paths is None:
            opt.init_mask_paths = [None] * len(opt.init_img_paths)

        avg_x_T = torch.zeros([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        for init_img_path, init_mask_path in zip(opt.init_img_paths, opt.init_mask_paths):
            init_img = load_img(init_img_path, opt.H, opt.W)
            print(f"Image {init_img_path}, as the init image, weight {opt.init_img_weight}")

            # init_img: [4, 3, 512, 512], [b c h w]
            init_img = init_img.repeat(batch_size, 1, 1, 1).to(device)

            if init_mask_path is not None:
                mask_obj = Image.open(init_mask_path).convert("L")
                print(f"Applied mask {init_mask_path}")
                mask     = torch.from_numpy(np.array(mask_obj)) / 255
                mask     = F.interpolate(mask[None, None], size=(opt.H // opt.f, opt.W // opt.f), mode='nearest')
                mask     = mask.repeat(batch_size, 1, 1, 1).to(device)
                mask_dict = { 'fg_mask': mask, 'aug_mask': None }
            else:
                mask = torch.ones_like(avg_x_T)
                mask_dict = None

            # move avg_init_img to latent space
            # x_T: [4, 4, 64, 64]
            x_T  = model.get_first_stage_encoding(model.encode_first_stage(init_img, mask_dict))  
            avg_x_T += torch.where(mask.bool(), x_T, torch.randn_like(x_T))

        # If approximating different images as indepent Gaussian variables, 
        # then their sum has a STD of sqrt(N). Normalization requires dividing by sqrt(N), instead of N.
        avg_x_T    /= np.sqrt(len(opt.init_img_paths))
        start_code  = avg_x_T * opt.init_img_weight + torch.randn_like(avg_x_T) * (1 - opt.init_img_weight)

    if not opt.eval_blip:
        # Normal evaluation.
        use_layerwise_embedding = config.model.params.use_layerwise_embedding
        if hasattr(opt, 'prompt_mix_scheme'):
            if opt.prompt_mix_scheme == 'mix_hijk':
                prompt_mix_weight = 1.0
            elif opt.prompt_mix_scheme == 'mix_concat_cls':
                prompt_mix_weight = opt.prompt_mix_weight
            else:
                raise NotImplementedError
        else:
            prompt_mix_weight = 0

        if opt.scale != 1.0:
            try:
                uc = model.get_learned_conditioning(batch_size * [opt.neg_prompt])
            except:
                breakpoint()
        else:
            uc = None
    else:
        # eval_blip
        use_layerwise_embedding = False
        prompt_mix_weight = 0
        uc = None

    if opt.neg_prompt != "":
        print("Negative prompt:", opt.neg_prompt)
        
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                sample_count = 0
                prompt_block_count = len(batched_prompts)
                for n in trange(opt.n_repeat, desc="Sampling"):
                    indiv_subdir = None

                    # prompts in a batch are just repetitions of the same prompt.
                    for p_i, prompts in enumerate(tqdm(batched_prompts, desc="prompts")):
                        print(f"\n{p_i+1}/{prompt_block_count}", prompts[0])

                        if prompt_mix_weight != 0:
                            # If ref_prompt is None (default), then ref_c is None, i.e., no mixing.
                            ref_prompt = batched_ref_prompts[p_i]
                            if ref_prompt is not None:
                                print("Mix with", ref_prompt)
                                ref_c = model.get_learned_conditioning(batch_size * [ref_prompt])
                            else:
                                ref_c = None
                        else:
                            ref_c = None

                        if opt.scale != 1.0:
                            # ref_prompt_mix doubles the number of channels of conditioning embeddings.
                            # So we need to repeat uc by 2.
                            if ref_c is not None:
                                uc_0 = uc[0].repeat(1, 2, 1)
                                uc = (uc_0, uc[1], uc[2])

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        if not opt.eval_blip:
                            c = model.get_learned_conditioning(prompts)
                            # ref_c is not None, implies (prompt_mix_weight != 0 and ref_prompt is not None).
                            if ref_c is not None:
                                # c / ref_c are tuples of (cond, prompts, extra_info).
                                c0_mix_all_layers = mix_embeddings(c[0], ref_c[0], prompt_mix_weight, 
                                                                mix_scheme='adeltaconcat')

                                if use_layerwise_embedding:
                                    # 4, 5, 6, 7, 8 correspond to original layer indices 7, 8, 12, 16, 17
                                    # (same as used in computing mixing loss)
                                    sync_layer_indices = [4, 5, 6, 7, 8]
                                    # [64, 154, 768] => [4, 16, 154, 768] => assign 1s => [64, 154, 768]
                                    layer_mask = torch.zeros_like(c0_mix_all_layers).reshape(-1, 16, *c0_mix_all_layers.shape[1:])
                                    layer_mask[:, sync_layer_indices] = 1
                                    layer_mask = layer_mask.reshape(c0_mix_all_layers.shape)
                                    # Use most of the layers of embeddings in subj_comps_emb, but 
                                    # replace sync_layer_indices layers with those from subj_comps_emb_mix_all_layers.
                                    # Do not assign with sync_layers as indices, which destroys the computation graph.
                                    c0_mix  = c[0].repeat(1, 2, 1) * (1 - layer_mask) \
                                            + c0_mix_all_layers * layer_mask

                                else:
                                    # There is only one layer of embeddings.
                                    c0_mix = c0_mix_all_layers

                                c[2]['iter_type'] = 'mix_hijk'
                                # c / ref_c are tuples of (cond, prompts, extra_info).
                                c = (c0_mix, c[1], c[2])

                            c[2]['normalize_subj_attn'] = model.embedding_manager.normalize_subj_attn
                            if opt.debug and ref_c is None:
                                c[2]['debug_attn'] = True

                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            # During inference, the batch size is *doubled*. 
                            # The first half contains negative samples, and the second half positive.
                            # When ada embedding is used, c is a tuple of (cond, ada_embedder).
                            # When both unconditional and conditional guidance are passed to ddim sampler, 
                            # ada_embedder of the conditional guidance is applied on both 
                            # unconditional and conditional embeddings within UNetModel.forward(). 
                            # But since unconditional prompt doesn't contain the placeholder token,
                            # ada_embedder won't change the unconditional embedding uc.
                            # scale = 0: e_t = e_t_uncond. scale = 1: e_t = e_t.
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x0=None,
                                                            mask=None,
                                                            x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            # x_samples_ddim: -1 ~ +1 -> 0 ~ 1.
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        else:
                            x_samples_ddim = []
                            blip_seed = 8888
                            for i_sample, prompt in enumerate(prompts):
                                stripped_prompt_parts = re.split("of z[, ]*", prompt)
                                if stripped_prompt_parts[1] == "":
                                    stripped_prompt = re.sub("^a ", "", stripped_prompt_parts[0])
                                    stripped_prompt = stripped_prompt.strip()
                                else:
                                    stripped_prompt = stripped_prompt_parts[1].strip()

                                if i_sample == 0:
                                    print("blip:", stripped_prompt)

                                stripped_prompts = [txt_preprocess["eval"](stripped_prompt)]
                                samples_info = {
                                    "cond_images":  None,
                                    "cond_subject": cond_subjects,
                                    "tgt_subject":  tgt_subjects,
                                    "prompt":       stripped_prompts,
                                }

                                samples = blip_model.generate(
                                    samples_info,
                                    seed=blip_seed + i_sample,
                                    guidance_scale=7.5,
                                    num_inference_steps=50,
                                    neg_prompt=negative_prompt,
                                    height=512,
                                    width=512,
                                )
                                x_samples_ddim.append(samples[0])

                        if not opt.skip_save:
                            indiv_subdir = batched_subdirs[p_i]
                            class_long_prompt = batched_class_long_prompts[p_i]
                            sample_dir = os.path.join(opt.outdir, indiv_subdir)
                            os.makedirs(sample_dir, exist_ok=True)                            

                            for i, x_sample in enumerate(x_samples_ddim):
                                base_count = len(os.listdir(sample_dir))
                                sample_file_path = os.path.join(sample_dir, f"{base_count:05}.jpg")

                                while os.path.exists(sample_file_path):
                                    base_count += 1
                                    sample_file_path = os.path.join(sample_dir, f"{base_count:05}.jpg")

                                if not opt.eval_blip:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(sample_file_path)
                                else:
                                    # eval_blip. x_sample is already a PIL image.
                                    x_sample.save(sample_file_path)
                                    # Convert x_sample to a torch tensor with a compatible shape.
                                    # H, W, C => C, H, W
                                    x_samples_ddim[i] = torch.from_numpy(np.array(x_sample)).permute(2, 0, 1).float() / 255.

                            if opt.eval_blip:
                                x_samples_ddim = torch.stack(x_samples_ddim, dim=0)

                            if opt.compare_with:
                                # There's an extra "None" after the last indiv_subdir in batched_subdirs 
                                # as boundary detection. So no need to worry (p_i + 1) may go out of bound.
                                next_indiv_subdir = batched_subdirs[p_i + 1]
                                # indiv_subdir will change in the next batch, or this is the end of the loop. 
                                # This means the current chunk (generated with the same prompt) is finished.
                                # So we evaluate the current chunk.
                                if next_indiv_subdir != indiv_subdir:
                                    print()
                                    sim_img, sim_text, sim_dino = \
                                        compare_folders(clip_evator, dino_evator, 
                                                        opt.compare_with, sample_dir, 
                                                        class_long_prompt, len(prompts))

                                    all_sims_img.append(sim_img.item())
                                    all_sims_text.append(sim_text.item())
                                    all_sims_dino.append(sim_dino.item())

                                    if opt.calc_face_sim:
                                        sim_face, normal_img_count, except_img_count = \
                                            compare_face_folders_fast(opt.compare_with, sample_dir, dst_num_samples=len(prompts))
                                        # sim_face is a float, so no need to detach().cpu().numpy().
                                        all_sims_face.append(sim_face)
                                        all_normal_img_counts.append(normal_img_count)
                                        all_except_img_counts.append(except_img_count)

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                        sample_count += batch_size

                # After opt.n_repeat passes of batched_prompts, save all sample images as an image grid
                if not opt.skip_grid:
                    # additionally, save as grid
                    # logs/gabrielleunion2023-05-24T18-33-34_gabrielleunion-ada/checkpoints/embeddings_gs-4500.pt
                    subjfolder_mat = re.search(r"([a-zA-Z0-9]+)(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_([^\/]+)", opt.subj_model_path)
                    if subjfolder_mat:
                        date_sig = subjfolder_mat.group(2)
                        # subjname_method: gabrielleunion-ada
                        subjname_method = subjfolder_mat.group(3)
                        iter_mat = re.search(r"(\d+).pt", opt.subj_model_path)
                        if iter_mat is not None:
                            iter_sig = iter_mat.group(1)
                        else:
                            iter_sig = "unknown"
                    else:
                        subjname_method = "unknown"
                        date_sig = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
                        iter_sig = "unknown"

                    experiment_sig = "-".join([date_sig, iter_sig, f"scale{opt.scale:.1f}"])
                    if opt.bb_type:
                        experiment_sig += "-" + opt.bb_type
                    if opt.neg_prompt != "":
                        experiment_sig += "-" + "neg"

                    # Use the first prompt of the current chunk from opt.from_file as the saved file name.
                    if opt.from_file:
                        prompt = prompts[0]
                    # Otherwise, use the prompt passed by the command line.
                    prompt_sig = prompt.replace(" ", "-")[:40]  # Cut too long prompt
                    grid_filepath = os.path.join(opt.outdir, f'{subjname_method}-{prompt_sig}-{experiment_sig}.jpg')
                    if os.path.exists(grid_filepath):
                        grid_count = 2
                        grid_filepath = os.path.join(opt.outdir, f'{subjname_method}-{prompt_sig}-{experiment_sig}-{grid_count}.jpg')
                        while os.path.exists(grid_filepath):
                            grid_count += 1
                            grid_filepath = os.path.join(opt.outdir, f'{subjname_method}-{prompt_sig}-{experiment_sig}-{grid_count}.jpg')

                    img = save_grid(all_samples, None, grid_filepath, nrow=n_rows)
                    
                toc = time.time()
            


    if not opt.skip_grid:
        print(f"Your samples are at: \n{grid_filepath}")
        if not (opt.no_preview or opt.from_file or opt.compare_with):
            os.spawnvp(os.P_NOWAIT, "gpicview", [ "gpicview", os.path.abspath(grid_filepath) ])
    else:
        print(f"Your samples are at: \n{opt.outdir}")

    if opt.compare_with:
        sims_img_avg, sims_text_avg, sims_dino_avg = np.mean(all_sims_img), np.mean(all_sims_text), np.mean(all_sims_dino)
        if opt.calc_face_sim:
            sims_face_avg = np.mean(all_sims_face)
        else:
            sims_face_avg = 0

        if opt.calc_face_sim:
            except_img_percent = np.sum(all_except_img_counts) / (np.sum(all_normal_img_counts) + np.sum(all_except_img_counts))
            print(f"Exception image percent: {except_img_percent*100:.1f}")
        else:
            except_img_percent = 0

        print(f"All samples mean face/image/text/dino sim: {sims_face_avg:.3f} {sims_img_avg:.3f} {sims_text_avg:.3f} {sims_dino_avg:.3f}")
        if SCORES_CSV is not None:
            subjfolder_mat = re.search(r"([a-zA-Z0-9_]+)(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_([^\/]+)", opt.subj_model_path)
            if subjfolder_mat:
                emb_sig  = subjfolder_mat.group(1) + subjfolder_mat.group(2)
            else:
                emb_sig  = "unknown"

            scores   = [sims_face_avg, sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent]
            scores   = [ f"{score:.4f}" for score in scores ]
            SCORES_CSV.writerow([emb_sig] + scores)

    if SCORES_CSV is not None:
        SCORES_CSV_FILE.close()

    return img #return img to be shown on webui

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
