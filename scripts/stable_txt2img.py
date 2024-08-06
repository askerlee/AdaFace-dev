import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
import re
import csv
from collections import namedtuple

from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import save_grid, extend_nn_embedding, load_model_from_config
from adaface.util import get_b_core_e_embeddings
from ldm.models.diffusion.ddim import DDIMSampler
from evaluation.eval_utils import compare_folders, compare_face_folders_fast, \
                                  init_evaluators, set_tf_gpu
from insightface.app import FaceAnalysis
from adaface.adaface_wrapper import AdaFaceWrapper

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
    parser.add_argument("--use_pre_neg_prompt", type=str2bool, const=True, default=True, nargs="?",
                        help="use predefined negative prompts")

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="samples-ada/"
    )
    parser.add_argument(
        "--indiv_subdir",
        type=str, default=None,
        help="subdir to write individual images to",
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
        nargs='+',
        default=[10, 4],
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
        default="models/stable-diffusion-v-1-5/v1-5-dste8-vae.ckpt",
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

    # No preview
    parser.add_argument(
        "--no_preview",
        action='store_true',
        help="do not preview the image",
    )
    parser.add_argument("--calc_face_sim", action="store_true",
                        help="If specified, assume the generated samples are human faces, "
                             "and compute face similarities with the groundtruth")
    parser.add_argument("--face_engine", type=str, default="deepface", choices=["deepface", "insightface"],
                        help="Face engine to use for face similarity calculation")
    
    parser.add_argument('--gpu', type=int,  default=0, help='ID of GPU to use.')
    #parser.add_argument("--tfgpu", type=int, default=argparse.SUPPRESS, help="ID of GPU to use for TensorFlow. Set to -1 to use CPU (slow).")

    parser.add_argument("--compare_with", type=str, default=None,
                        help="A folder of reference images, or a single reference image, used to evaluate the similarity of generated samples")
    parser.add_argument("--class_prompt", type=str, default=None,
                        help="the original prompt used for text/image matching evaluation "
                             "(requires --compare_with to be specified)")

    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[1, 1],
                        help="Weight of the skip connections of the last few layers of CLIP text embedder. " 
                             "NOTE: the last element is the weight of the last layer.")

    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")
    parser.add_argument("--background_string", 
                        type=str, default="y",
                        help="Background placeholder string used in prompts to denote the background in training images.")
                    
    parser.add_argument("--num_vectors_per_subj_token",
                        type=int, default=16,
                        help="Number of vectors per token. If > 1, use multiple embeddings to represent a subject.")
    parser.add_argument("--num_vectors_per_bg_token",
                        type=int, default=4,
                        help="Number of vectors per background token. If > 1, use multiple embeddings to represent a background.")
    parser.add_argument("--skip_loading_token2num_vectors", action="store_true",
                        help="Skip loading token2num_vectors from the checkpoint.")

    parser.add_argument("--zeroshot", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use zero-shot learning")                    
    parser.add_argument("--apply_arc2face_embs", action="store_true",
                        help="Evaluate Arc2Face forward embeddings")
    parser.add_argument("--apply_arc2face_inverse_embs", type=str2bool, nargs="?", 
                        const=True, default=False,
                        help="Evaluate Arc2Face inverse CLIP embeddings")
    parser.add_argument("--load_old_embman_ckpt", action="store_true", 
                        help="Load the old checkpoint for the embedding manager")       
    parser.add_argument("--ref_images", type=str, nargs='+', default=None,
                        help="Reference image(s) for zero-shot learning. Each item could be a path to an image or a directory.")
    
    # bb_type: backbone checkpoint type. Just to append to the output image name for differentiation.
    # The backbone checkpoint is specified by --ckpt.
    parser.add_argument("--bb_type", type=str, default="")
    parser.add_argument("--scores_csv", type=str, default=None,
                        help="CSV file to save the evaluation scores")
    parser.add_argument("--debug", action="store_true",
                        help="debug mode")
    parser.add_argument("--eval_blip", action="store_true",
                        help="Evaluate BLIP-diffusion models")
    parser.add_argument("--cls_string", type=str, default=None,
                        help="Subject class name. Only requires for --eval_blip")
    parser.add_argument("--diffusers", action="store_true", 
                        help="Zeroshot uses the diffusers implementation")
    # extra_unet_paths and unet_weights are only relevant for --diffusers
    parser.add_argument('--extra_unet_paths', type=str, nargs="+", 
                        default=['models/ensemble/rv4-unet', 'models/ensemble/ar18-unet'], 
                        help="Extra paths to the checkpoints of the UNet models")
    parser.add_argument('--unet_weights', type=float, nargs="+", default=[4, 2, 1], 
                        help="Weights for the UNet models")
        
    args = parser.parse_args()
    return args

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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


def filter_image(x):
    x = x.lower()
    inclusion_pats = [ ".jpg", ".jpeg", ".png", ".bmp" ]
    exclusion_pats = [ "_mask.png" ]
    return any([ pat in x for pat in inclusion_pats ]) and not any([ pat in x for pat in exclusion_pats ])

def main(opt):

    # No GPUs detected. Use CPU instead.
    if not torch.cuda.is_available():
        opt.gpu = -1
    else:
        torch.cuda.set_device(opt.gpu)
        torch.backends.cuda.matmul.allow_tf32 = True

    seed_everything(opt.seed)
    # More complex negative prompts may hurt the performance.
    # predefined_negative_prompt = "duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, disfigured, mutation"
    # This negative prompt is borrowed from PuLID.
    predefined_negative_prompt = "flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, " \
                                 "mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, " \
                                 "mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, " \
                                 "nude, naked, nsfw, topless, bare breasts"
    device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

    if opt.zeroshot:
        assert opt.ref_images is not None, "Must specify --ref_images for zero-shot learning"
        ref_image_paths = []
        for ref_image in opt.ref_images:
            if os.path.isdir(ref_image):
                ref_image_paths.extend([ os.path.join(ref_image, f) for f in os.listdir(ref_image) ])
            else:
                ref_image_paths.append(ref_image)
        ref_image_paths = list(filter(lambda x: filter_image(x), ref_image_paths))
        ref_images = [ np.array(Image.open(ref_image_path)) for ref_image_path in ref_image_paths ]
    else:
        ref_images = None

    if not opt.eval_blip and not opt.diffusers:
        config = OmegaConf.load(f"{opt.config}")
        config.model.params.do_zero_shot = opt.zeroshot
        config.model.params.personalization_config.params.do_zero_shot = opt.zeroshot
        config.model.params.personalization_config.params.token2num_vectors = {} 
        if opt.zeroshot:
            config.model.params.personalization_config.params.zs_cls_delta_string = 'person'

        if hasattr(opt, 'num_vectors_per_subj_token'):
            # Command line --num_vectors_per_subj_token overrides the checkpoint setting.
            config.model.params.personalization_config.params.token2num_vectors[opt.subject_string] = opt.num_vectors_per_subj_token
            opt.skip_loading_token2num_vectors = True
        if hasattr(opt, 'num_vectors_per_bg_token'):
            # Command line --num_vectors_per_bg_token doesn't override the checkpoint setting.
            config.model.params.personalization_config.params.token2num_vectors[opt.background_string] = opt.num_vectors_per_bg_token
        config.model.params.personalization_config.params.skip_loading_token2num_vectors = opt.skip_loading_token2num_vectors

        model = load_model_from_config(config, f"{opt.ckpt}")
        if opt.embedding_paths is not None:
            model.embedding_manager.load(opt.embedding_paths, load_old_embman_ckpt=opt.load_old_embman_ckpt)
            model.embedding_manager.eval()
            opt.subj_model_path = opt.embedding_paths[0]
        else:
            opt.subj_model_path = opt.ckpt

        # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
        model.cond_stage_model.set_last_layers_skip_weights(opt.clip_last_layers_skip_weights)

        # extend_placeholders() has to be called before extend_nn_embedding().
        model.embedding_manager.extend_placeholders([opt.subject_string], [opt.background_string],
                                                    opt.num_vectors_per_subj_token, opt.num_vectors_per_bg_token)

        if hasattr(opt, 'num_vectors_per_subj_token'):
            ckpt_num_vectors_per_subj_token = model.embedding_manager.token2num_vectors[opt.subject_string]
            if ckpt_num_vectors_per_subj_token != opt.num_vectors_per_subj_token:
                print(f"WARN: Number of vectors per token mismatch: command line {opt.num_vectors_per_subj_token} != ckpt {ckpt_num_vectors_per_subj_token}.")

        if model.embedding_manager.extended_token_embeddings is not None:
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding = \
                extend_nn_embedding(model.cond_stage_model.transformer.text_model.embeddings.token_embedding, 
                                    model.embedding_manager.extended_token_embeddings)
            model.embedding_manager.extended_token_embeddings = None

        model.embedding_manager.curr_subj_is_face = opt.calc_face_sim
        
        model  = model.to(device)
        model.cond_stage_model.device = device

        assert model.embedding_manager.do_zero_shot == opt.zeroshot, \
                f"Zero-shot learning mismatch: command line {opt.zeroshot} != ckpt {model.embedding_manager.do_zero_shot}."
        
        if opt.zeroshot:
            # zs_clip_features: [1, 514, 1280]. 
            # zs_id_embs: [1, 512] if is_face, or [2, 16, 512] if uses IP-adapter warm start; or [1, 384] if is object.
            zs_clip_features, zs_id_embs, _ = \
                model.encode_zero_shot_image_features(ref_images, fg_masks=None,
                                                      is_face=opt.calc_face_sim,
                                                      calc_avg=True, skip_non_faces=True,
                                                      image_paths=opt.ref_images, verbose=True)
        else:
            zs_clip_features, zs_id_embs = None, None

        sampler = DDIMSampler(model)

        if opt.neg_prompt == "" and opt.use_pre_neg_prompt:
            # negative prompt borrowed from BLIP-Diffusion.
            opt.neg_prompt = predefined_negative_prompt

    else:        
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

        if opt.diffusers:
            opt.subj_model_path = opt.embedding_paths[0]
            pipeline = AdaFaceWrapper("text2img", opt.ckpt, opt.subj_model_path, device, 
                                     opt.subject_string, opt.num_vectors_per_subj_token, 50,
                                     extra_unet_paths=opt.extra_unet_paths, unet_weights=opt.unet_weights)
            # adaface_subj_embs is not used. It is generated for the purpose of updating the text encoder (within this function call).
            adaface_subj_embs = pipeline.generate_adaface_embeddings(ref_image_paths, None, None, False, 
                                                                     out_id_embs_scale=1, noise_level=0, 
                                                                     update_text_encoder=True)
        # eval_blip
        else:
            from lavis.models import load_model_and_preprocess
            blip_model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device=f"cuda:{opt.gpu}", is_eval=True)
            blip_model.load_checkpoint(opt.ckpt)
            cond_subject = opt.cls_string
            tgt_subject  = opt.cls_string
            cond_subjects = [txt_preprocess["eval"](cond_subject)]
            tgt_subjects  = [txt_preprocess["eval"](tgt_subject)]
            negative_prompt = predefined_negative_prompt
            opt.subj_model_path = ""

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
        if opt.indiv_subdir is None:
            opt.indiv_subdir = prompt.replace(" ", "-")
            
        batched_subdirs = [ opt.indiv_subdir ] * len(batched_prompts)
        # Append None to the end of batched_subdirs, for indiv_subdir change detection.
        batched_subdirs.append(None)

        if opt.compare_with:
            assert opt.class_prompt is not None, "Must specify --class_prompt when calculating CLIP similarities."

        batched_class_long_prompts = [ opt.class_prompt ] * len(batched_prompts)

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

            # Append None to the end of batched_subdirs, for indiv_subdir change detection.
            batched_subdirs.append(None)

    if opt.compare_with:
        clip_evator, dino_evator = init_evaluators(-1)
        all_sims_img, all_sims_text, all_sims_dino = [], [], []
        if opt.calc_face_sim:
            all_sims_face = []
            all_normal_img_counts = []
            all_except_img_counts = []
            if opt.face_engine == "insightface":
                # FaceAnalysis will try to find the ckpt in: models/insightface/models/antelopev2. 
                # Note there's a second "model" in the path.
                insightface_app = FaceAnalysis(name='antelopev2', root='models/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                insightface_app.prepare(ctx_id=opt.gpu, det_size=(512, 512))
            else:
                insightface_app = None
                if hasattr(opt, 'tfgpu'):
                    set_tf_gpu(opt.tfgpu)
                else:
                    set_tf_gpu(opt.gpu)

    else:
        clip_evator, dino_evator = None, None

    if opt.fixed_code:
        # start_code is independent of the samples. Therefore it's shared by all samples.
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    else:
        start_code = None

    if not opt.eval_blip and not opt.diffusers:
        if opt.scale != 1.0:
            try:
                uc = model.get_learned_conditioning(batch_size * [opt.neg_prompt], embman_iter_type='empty')
            except:
                breakpoint()
        else:
            uc = None
    else:
        # eval_blip
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

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        if not opt.eval_blip and not opt.diffusers:
                            apply_arc2face_inverse_embs = opt.zeroshot and opt.apply_arc2face_inverse_embs
                            apply_arc2face_embs         = opt.zeroshot and opt.apply_arc2face_embs
                            # NOTE: model.embedding_manager.curr_subj_is_face is queried when generating zero-shot id embeddings. 
                            # We've assigned model.embedding_manager.curr_subj_is_face = opt.calc_face_sim above.
                            c = model.get_learned_conditioning(prompts, zs_clip_features=zs_clip_features,
                                                               zs_id_embs=zs_id_embs, 
                                                               apply_arc2face_inverse_embs=apply_arc2face_inverse_embs,
                                                               apply_arc2face_embs=apply_arc2face_embs)

                            if apply_arc2face_inverse_embs:
                                static_prompt_embedding = c[0].repeat(len(prompts), 1, 1)
                                c = (static_prompt_embedding, c[1], c[2])
                                # If the arc2face prompt is shorter than the uncond static prompt (77 tokens),
                                # Then we shorten the uncond static prompt to match the arc2face prompt.
                                if static_prompt_embedding.shape[1] < uc[0].shape[1]:
                                    uncond_prompt_embedding = get_b_core_e_embeddings(uc[0], length=static_prompt_embedding.shape[1] - 1)
                                    uc = (uncond_prompt_embedding, uc[1], uc[2])

                            if opt.debug:
                                c[2]['debug_attn'] = True

                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            # During inference, the batch size is *doubled*. 
                            # The first half contains negative samples, and the second half positive.
                            # scale = 0: e_t = e_t_uncond. scale = 1: e_t = e_t.
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=batch_size,
                                                             shape=shape,
                                                             verbose=False,
                                                             guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x0=None,
                                                             mask=None,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            # x_samples_ddim: -1 ~ +1 -> 0 ~ 1.
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if opt.diffusers:
                            noise = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                            x_samples_ddim = pipeline(noise, prompts[0], None, opt.scale[0], 
                                                      batch_size, verbose=True)
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

                                if not opt.eval_blip and not opt.diffusers:
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
                                            compare_face_folders_fast(opt.compare_with, sample_dir, dst_num_samples=len(prompts),
                                                                      face_engine=opt.face_engine, insightface_app=insightface_app)
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
                        if opt.zeroshot and opt.compare_with:
                            if opt.compare_with.endswith("/") or opt.compare_with.endswith("\\"):
                                opt.compare_with = opt.compare_with[:-1]
                            subj_gt_folder_name = os.path.basename(opt.compare_with)
                            # all-ada => all-ada-gabrielleunion
                            subjname_method += "-" + subj_gt_folder_name

                        iter_mat = re.search(r"(\d+).pt", opt.subj_model_path)
                        if iter_mat is not None:
                            iter_sig = iter_mat.group(1)
                        else:
                            iter_sig = "unknown"
                    else:
                        subjname_method = "unknown"
                        date_sig = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
                        iter_sig = "unknown"

                    if isinstance(opt.scale, (list, tuple)):
                        scale_sig = "scale" + "-".join([f"{scale:.1f}" for scale in opt.scale])
                    else:
                        scale_sig = f"scale{opt.scale:.1f}"
                    experiment_sig = "-".join([date_sig, iter_sig, scale_sig])

                    if opt.bb_type:
                        experiment_sig += "-" + opt.bb_type
                    if opt.neg_prompt != "":
                        experiment_sig += "-neg"

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

            if opt.zeroshot:
                emb_sig = subjname_method + "-" + emb_sig

            scores   = [sims_face_avg, sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent]
            scores   = [ f"{score:.4f}" for score in scores ]
            SCORES_CSV.writerow([emb_sig] + scores)

    if SCORES_CSV is not None:
        SCORES_CSV_FILE.close()

    return img #return img to be shown on webui

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
