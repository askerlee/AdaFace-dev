import numpy as np
import argparse, os, sys
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
import re
import csv

from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import save_grid, load_model_from_config, latent_to_numpy
from ldm.models.diffusion.ddim import DDIMSampler
from evaluation.eval_utils import compare_folders, compare_face_folders, \
                                  init_evaluators, set_tf_gpu
from insightface.app import FaceAnalysis

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
        "--same_start_code_for_prompts",
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
             "Usually used if prompts are not loaded from a file (--prompt_file not specified)",
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
        help="Conditional guidance scale",
    )
    parser.add_argument(
        "--prompt_file",
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
        default="models/sar/sar.safetensors",
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

    parser.add_argument("--calc_face_sim", action="store_true",
                        help="If specified, assume the generated samples are human faces, "
                             "and compute face similarities with the groundtruth")
    parser.add_argument("--face_engine", type=str, default="deepface", choices=["deepface", "insightface"],
                        help="Face engine to use for face similarity calculation")
    
    parser.add_argument('--gpu', type=int,  default=0, help='ID of GPU to use.')
    #parser.add_argument("--tfgpu", type=int, default=argparse.SUPPRESS, help="ID of GPU to use for TensorFlow. Set to -1 to use CPU (slow).")

    parser.add_argument("--subj_name", type=str, default="unknown", 
                        help="Subject name for the output images")
    parser.add_argument("--compare_with", type=str, nargs='+', default=None,
                        help="A list of reference images/folders, used to evaluate the similarity of generated samples")    
    parser.add_argument("--class_prompt", type=str, default=None,
                        help="the original prompt used for text/image matching evaluation "
                             "(requires --compare_with to be specified)")

    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[1],
                        help="Weight of the skip connections of the last few layers of CLIP text embedder. " 
                             "NOTE: the last element is the weight of the last layer.")

    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")

    parser.add_argument("--return_prompt_embs_type", type=str, choices=["text", "id", "text_id"],
                        default="text", help="The type of the returned prompt embeddings from get_text_conditioning()")     
    parser.add_argument("--ref_images", type=str, nargs='+', default=None,
                        help="Reference image(s) for zero-shot learning. Each item could be a path to an image or a directory.")
    
    # bb_type: backbone checkpoint type. Just to append to the output image name for differentiation.
    # The backbone checkpoint is specified by --ckpt.
    parser.add_argument("--bb_type", type=str, default="")
    parser.add_argument("--scores_csv", type=str, default=None,
                        help="CSV file to save the evaluation scores")
    parser.add_argument("--debug", action="store_true",
                        help="debug mode")
    parser.add_argument("--diffusers", type=str2bool, const=True, nargs="?", default=True,
                        help="Use the diffusers implementation which is faster than the original LDM")
    parser.add_argument("--method", type=str, default="adaface",
                        choices=["adaface", "pulid"])
    parser.add_argument("--adaface_encoder_types", type=str, nargs="+", default=["consistentID", "arc2face"],
                        choices=["arc2face", "consistentID"], help="Type(s) of the ID2Ada prompt encoders")
    parser.add_argument("--enabled_encoders", type=str, nargs="+", default=None,
                        choices=["arc2face", "consistentID"], 
                        help="List of enabled encoders (among the list of adaface_encoder_types)")
    parser.add_argument('--adaface_ckpt_paths', type=str, nargs="+", required=True)
    # If adaface_encoder_cfg_scales is not specified, the weights will be set to 
    # 6.0 (consistentID) and 1.0 (arc2face).
    parser.add_argument('--adaface_encoder_cfg_scales', type=float, nargs="+", default=None,    
                        help="CFG scales of output embeddings of the ID2Ada prompt encoders")
    # Options below are only relevant for --diffusers --method adaface.
    parser.add_argument("--main_unet_filepath", type=str, default=None,
                        help="Path to the checkpoint of the main UNet model, if you want to replace the default UNet within --ckpt")
    parser.add_argument('--extra_unet_dirpaths', type=str, nargs="*", 
                        default=[], 
                        help="Extra paths to the checkpoints of the UNet models")
    parser.add_argument('--unet_weights_in_ensemble', type=float, nargs="+", default=[1], 
                        help="Weights for the UNet models")
    parser.add_argument("--placeholder_tokens_pos", type=str, default="append",
                        choices=["prepend", "append"],
                        help="Position of the placeholder tokens in the prompt")
    # If specified, then should be a list, in which one value for each encoder. 
    # If enabled, the static image suffix embeddings of that encoder 
    # will be used during inference.
    parser.add_argument("--enable_static_img_suffix_embs", type=str2bool, nargs="+", default=None,
                        help="Enable the static image suffix embeddings during inference")
    parser.add_argument("--ablate_prompt_only_placeholders", type=str2bool,  
                        const=True, nargs="?", default=False,
                        help="Ablate the prompt to use only placeholders and no composition during inference")
    parser.add_argument("--ablate_prompt_no_placeholders", type=str2bool,
                        const=True, nargs="?", default=False,
                        help="Ablate the prompt to use no placeholders during inference")
    parser.add_argument("--repeat_prompt_for_each_encoder", type=str2bool,
                        const=True, nargs="?", default=True,
                        help="Repeat the prompt for each encoder during inference (default behavior)")
    parser.add_argument("--use_lcm", type=str2bool, const=True, nargs="?", default=False,
                        help="Use LCM to do quick inference")
    parser.add_argument("--use_ldm_pipeline", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use the LDM pipeline to do the inference")
    parser.add_argument("--diffusers_scheduler_name", type=str, default="ddim",
                        choices=["ddim", "pndm", "dpm++"], 
                        help="The scheduler name for the diffusers pipeline")
    parser.add_argument("--unet_uses_attn_lora", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use LoRA in the Diffusers UNet model")
    parser.add_argument("--shrink_cross_attn", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to suppress the subject attention in the subject-compositional instances")
        
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
        #torch.backends.cuda.matmul.allow_tf32 = True

    seed_everything(opt.seed)
    # More complex negative prompts may hurt the performance.
    # predefined_negative_prompt = "duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, disfigured, mutation"
    # This negative prompt is borrowed from PuLID.
    predefined_negative_prompt = "flaws in the eyes, flaws in the face, lowres, non-HDRi, low quality, worst quality, artifacts, noise, text, watermark, glitch, " \
                                 "mutated, ugly, disfigured, hands, partially rendered objects, partially rendered eyes, deformed eyeballs, cross-eyed, blurry, " \
                                 "mutation, duplicate, out of frame, cropped, mutilated, bad anatomy, deformed, bad proportions, " \
                                 "nude, naked, nsfw, topless, bare breasts"
    device = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

    if opt.neg_prompt == "" and opt.use_pre_neg_prompt:
        # negative prompt borrowed from BLIP-Diffusion.
        opt.neg_prompt = predefined_negative_prompt

    if opt.neg_prompt != "":
        print("Negative prompt:", opt.neg_prompt)

    assert opt.ref_images is not None, "Must specify --ref_images for zero-shot learning"
    ref_image_paths = []
    for ref_image in opt.ref_images:
        if os.path.isdir(ref_image):
            ref_image_paths.extend([ os.path.join(ref_image, f) for f in os.listdir(ref_image) ])
        else:
            ref_image_paths.append(ref_image)
    ref_image_paths = list(filter(lambda x: filter_image(x), ref_image_paths))
    ref_images = [ np.array(Image.open(ref_image_path)) for ref_image_path in ref_image_paths ]

    if opt.adaface_ckpt_paths is not None:
        first_subj_model_path = opt.adaface_ckpt_paths[0]
    else:
        first_subj_model_path = "uninitialized"

    # If not using the diffusers pipeline, then always use the LDM pipeline.
    if not opt.diffusers:
        opt.use_ldm_pipeline = True

    if opt.use_ldm_pipeline:
        config = OmegaConf.load(f"{opt.config}")
        config.model.params.personalization_config.params.cls_delta_string      = 'person'
        config.model.params.personalization_config.params.subject_strings       = [opt.subject_string]
        # Currently embedding manager only supports one type of prompt encoder.
        config.model.params.personalization_config.params.adaface_encoder_types = opt.adaface_encoder_types
        config.model.params.unet_uses_attn_lora = opt.unet_uses_attn_lora
        config.model.params.shrink_cross_attn = opt.shrink_cross_attn
        
        ldm_model = load_model_from_config(config, f"{opt.ckpt}")
        if opt.adaface_ckpt_paths is not None:
            ldm_model.embedding_manager.load(opt.adaface_ckpt_paths)
            ldm_model.embedding_manager.eval()

        # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
        ldm_model.cond_stage_model.set_last_layers_skip_weights(opt.clip_last_layers_skip_weights)
        ldm_model.embedding_manager.curr_subj_is_face = opt.calc_face_sim
        
        ldm_model  = ldm_model.to(device)
        ldm_model.cond_stage_model.device = device

        # subj_id2img_prompt_embs: [1, 4, 768] or [1, 16, 768] is in the image prompt space.
        face_image_count, faceid_embeds, subj_id2img_prompt_embs, neg_id_prompt_embs \
            = ldm_model.embedding_manager.id2ada_prompt_encoder.get_img_prompt_embs( \
                init_id_embs=None,
                pre_clip_features=None,
                image_paths=ref_image_paths,
                image_objs=ref_images,
                id_batch_size=1,
                perturb_at_stage=None,
                perturb_std=0,
                avg_at_stage='id_emb',
                verbose=True)

        sampler = DDIMSampler(ldm_model)

        if len(opt.adaface_encoder_types) == 1:
            num_vectors_per_subj_token = ldm_model.embedding_manager.token2num_vectors[opt.subject_string]
            placeholder_tokens_strs = [ opt.subject_string + "".join([", "] * (num_vectors_per_subj_token - 1)) ]
        else:
            # Joint_FaceID2AdaPrompt
            placeholder_tokens_strs = []
            for i in range(len(opt.adaface_encoder_types)):
                num_vectors_per_subj_token = ldm_model.embedding_manager.id2ada_prompt_encoder.encoders_num_id_vecs[i]
                if i == 0:
                    placeholder_tokens_str = opt.subject_string + "".join([", "] * (num_vectors_per_subj_token - 1))
                else:
                    placeholder_tokens_str = "".join([", "] * num_vectors_per_subj_token)
                placeholder_tokens_strs.append(placeholder_tokens_str)

        if opt.scale != 1.0:
            try:
                uc = ldm_model.get_text_conditioning([opt.neg_prompt], 
                                                 subj_id2img_prompt_embs = None,
                                                 clip_bg_features=None,
                                                 return_prompt_embs_type = 'text',
                                                 text_conditioning_iter_type = 'plain_text_iter')
            except:
                breakpoint()

    # Usually we don't use_ldm_pipeline and diffusers at the same time.
    # But if both are enabled, the LDM pipeline will be used to get the prompt embeddings,
    # and the diffusers pipeline will generate the images.
    if opt.diffusers:
        if opt.method == "adaface":
            from adaface.adaface_wrapper import AdaFaceWrapper

            pipeline = AdaFaceWrapper("text2img", opt.ckpt, opt.adaface_encoder_types, 
                                      opt.adaface_ckpt_paths, opt.adaface_encoder_cfg_scales,
                                      opt.enabled_encoders, opt.use_lcm,
                                      default_scheduler_name=opt.diffusers_scheduler_name, 
                                      subject_string=opt.subject_string, 
                                      num_inference_steps=opt.ddim_steps, 
                                      negative_prompt=opt.neg_prompt,
                                      unet_types=None,
                                      main_unet_filepath=opt.main_unet_filepath, extra_unet_dirpaths=opt.extra_unet_dirpaths, 
                                      unet_weights_in_ensemble=opt.unet_weights_in_ensemble, enable_static_img_suffix_embs=opt.enable_static_img_suffix_embs,
                                      unet_uses_attn_lora=opt.unet_uses_attn_lora,
                                      shrink_cross_attn=opt.shrink_cross_attn,
                                      device=device)
            # adaface_subj_embs is not used. It is generated for the purpose of updating the text encoder (within this function call).
            adaface_subj_embs = \
                pipeline.prepare_adaface_embeddings(ref_image_paths, None, 
                                                    perturb_at_stage='img_prompt_emb',
                                                    perturb_std=0, 
                                                    update_text_encoder=True)
            
        elif opt.method == "pulid":
            sys.path.append("pulid")
            from pulid.pipeline import PuLIDPipeline
            from pulid.utils import resize_numpy_image_long
            from pulid import attention_processor as attention

            first_subj_model_path = ""
            pipeline = PuLIDPipeline(device=device)
            
            attention.NUM_ZERO = 8
            attention.ORTHO = False
            attention.ORTHO_v2 = True

            id_embeddings = None
            for id_image_path in ref_image_paths:
                id_image = np.array(Image.open(id_image_path))
                id_image = resize_numpy_image_long(id_image, 1024)
                id_embedding = pipeline.get_id_embedding(id_image)
                # No face detected.
                if id_embedding is None:
                    continue
                if id_embeddings is None:
                    id_embeddings = id_embedding
                else:
                    id_embeddings = torch.cat(
                        (id_embeddings, id_embedding[:, :5]), dim=1
                    )

            print("id_embeddings:", id_embeddings.shape)

        # pulid requires full precision.
        opt.precision = "full"

    os.makedirs(opt.outdir, exist_ok=True)

    if opt.scores_csv:
        SCORES_CSV_FILE = open(opt.scores_csv, "a")
        SCORES_CSV = csv.writer(SCORES_CSV_FILE)
    else:
        SCORES_CSV = None

    if opt.compare_with is None:
        opt.compare_with = opt.ref_images

    batch_size = opt.n_samples if opt.bs == -1 else opt.bs
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.prompt_file:
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

        batched_class_prompts = [ opt.class_prompt ] * len(batched_prompts)

    else:
        print(f"Reading prompts from {opt.prompt_file}")
        with open(opt.prompt_file, "r") as f:
            # splitlines() will remove the trailing newline. So no need to strip().
            lines = f.read().splitlines()
            indiv_subdirs_prompts = [ line.split("\t") for line in lines ]
            n_repeats, indiv_subdirs, all_prompts, class_prompts \
                    = zip(*indiv_subdirs_prompts)
            # Repeat each prompt n_repeats times, and split into batches of size batch_size.
            # If there's remainder after chunks, the last chunk will be shorter than batch_size.

            batched_prompts = []
            batched_subdirs = []
            batched_class_prompts = []

            # Repeat each prompt n_repeats times.
            for i, prompt in enumerate(all_prompts):
                n_repeat = int(n_repeats[i])
                # If in this line, n_repeat is larger than batch_size, we need to split it 
                # into n_batches > 1 batches. These batches share the same indiv_subdir,
                # class_prompt.
                # So no need to repeat them in advance. Just append n_batches copies of them 
                # to batched_subdirs and batched_class_prompts respectively below.
                indiv_subdir = indiv_subdirs[i]
                class_prompt = class_prompts[i]
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
                batched_class_prompts.extend([class_prompt] * n_batches)

            # Append None to the end of batched_subdirs, for indiv_subdir change detection.
            batched_subdirs.append(None)

    if opt.compare_with:
        clip_evator, dino_evator = init_evaluators(device)
        all_sims_img, all_sims_text, all_sims_dino = [], [], []
        if opt.calc_face_sim:
            all_sims_face = []
            all_normal_img_counts = []
            all_except_img_counts = []
            if opt.face_engine == "insightface":
                # FaceAnalysis will try to find the ckpt in: models/insightface/models/antelopev2. 
                # Note there's a second "model" in the path.
                insightface_app = FaceAnalysis(name='antelopev2', root='models/insightface', providers=['CPUExecutionProvider'])
                insightface_app.prepare(ctx_id=0, det_size=(512, 512))
            else:
                # face_engine == "deepface". It cannot be pre-initialized.
                insightface_app = None
                if hasattr(opt, 'tfgpu'):
                    set_tf_gpu(opt.tfgpu)
                else:
                    set_tf_gpu(opt.gpu)

    else:
        clip_evator, dino_evator = None, None

    if opt.same_start_code_for_prompts:
        # start_code is independent of the samples. Therefore it's shared by all samples.
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    else:
        start_code = None

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            all_samples = []
            sample_count = 0
            prompt_block_count = len(batched_prompts)
            for n in trange(opt.n_repeat, desc="Sampling"):
                indiv_subdir = None

                # prompts in a batch are just repetitions of the same prompt.
                for p_i, prompts in enumerate(tqdm(batched_prompts, desc="prompts")):
                    print(f"\n{p_i+1}/{prompt_block_count}", prompts[0])

                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    if opt.use_ldm_pipeline:
                        prompts2 = []
                        for prompt in prompts:
                            # Remove the subject string 'z', then append the placeholder tokens to the prompt.
                            # If there is a word 'a' before the subject string or ',' after, then remove 'a z,'.
                            prompt2 = re.sub(r'\b(a|an|the)\s+' + opt.subject_string + r'\b,?', "", prompt)
                            prompt2 = re.sub(r'\b' + opt.subject_string + r'\b,?', "", prompt2)
                            prompt_segs = []
                            for placeholder_tokens_str in placeholder_tokens_strs:
                                prompt_segs.append(prompt2 + " " + placeholder_tokens_str)

                            prompt2 = "".join(prompt_segs)
                            prompts2.append(prompt2)

                        prompts = prompts2
                        print("Prompt:", prompts[0])

                        # NOTE: ldm_model.embedding_manager.curr_subj_is_face is queried when generating zero-shot id embeddings. 
                        # We've assigned ldm_model.embedding_manager.curr_subj_is_face = opt.calc_face_sim above.
                        c = ldm_model.get_text_conditioning(prompts, subj_id2img_prompt_embs = subj_id2img_prompt_embs,
                                                            clip_bg_features = None,
                                                            return_prompt_embs_type = opt.return_prompt_embs_type,
                                                            text_conditioning_iter_type = 'plain_text_iter')
                        
                        if opt.debug:
                            c[2]['debug_attn'] = True

                        uc2_emb = uc[0].repeat(batch_size, 1, 1)
                        uc2 = (uc2_emb, uc[1], uc[2])

                        if not opt.diffusers:
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
                                                            unconditional_conditioning=uc2,
                                                            eta=opt.ddim_eta,
                                                            x0=None,
                                                            mask=None,
                                                            x_T=start_code)

                            x_samples_ddim = ldm_model.decode_first_stage(samples_ddim)
                            # x_samples_ddim: -1 ~ +1 -> 0 ~ 1.
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    if opt.diffusers:
                        noise = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                        if opt.method == "adaface":
                            if opt.use_ldm_pipeline:
                                # The prompt embeddings have been generated above. So use them here.
                                prompt_embeds = (c[0][:1], uc[0][:1])
                            else:
                                prompt_embeds = None

                            x_samples_ddim = pipeline(noise, prompts[0], prompt_embeds, None, 
                                                      placeholder_tokens_pos=opt.placeholder_tokens_pos,
                                                      guidance_scale=opt.scale, out_image_count=batch_size, 
                                                      ablate_prompt_only_placeholders=opt.ablate_prompt_only_placeholders,
                                                      ablate_prompt_no_placeholders=opt.ablate_prompt_no_placeholders,
                                                      repeat_prompt_for_each_encoder=opt.repeat_prompt_for_each_encoder,
                                                      verbose=True)
                            
                        elif opt.method == "pulid":
                            x_samples_ddim = []
                            # Remove the subject string 'z' from the prompt.
                            prompt = re.sub(rf"a\s+{opt.subject_string},? ", "", prompts[0])
                            print("pulid:", prompt)
                            for bi in range(batch_size):
                                sample = pipeline.inference(prompt, (1, 768, 768), opt.neg_prompt,
                                                            id_embeddings, 0.8, 1.2, 4)[0]
                                sample = sample.resize((512, 512))
                                x_samples_ddim.append(sample)

                    # Otherwise, it's use_ldm_pipeline and not diffusers. Do nothing here, 
                    # since the images have been generated using LDM pipeline above.

                    if not opt.skip_save:
                        # "{subj_name}" is a placeholder in indiv_subdir for the subject name. 
                        # We instantiate it with the actual subject name by.
                        indiv_subdir = batched_subdirs[p_i].format(subj_name=opt.subj_name)
                        class_prompt = batched_class_prompts[p_i]
                        sample_dir = os.path.join(opt.outdir, indiv_subdir)
                        os.makedirs(sample_dir, exist_ok=True)                            

                        for i, x_sample in enumerate(x_samples_ddim):
                            base_count = len(os.listdir(sample_dir))
                            sample_file_path = os.path.join(sample_dir, f"{base_count:05}.jpg")

                            while os.path.exists(sample_file_path):
                                base_count += 1
                                sample_file_path = os.path.join(sample_dir, f"{base_count:05}.jpg")

                            if not opt.diffusers:
                                x_sample_np = latent_to_numpy(x_sample)
                                Image.fromarray(x_sample_np).save(sample_file_path)
                            else:
                                # diffusers. x_sample is already a PIL image.
                                x_sample.save(sample_file_path)
                                # Convert x_sample to a torch tensor with a compatible shape.
                                # H, W, C => C, H, W
                                x_samples_ddim[i] = torch.from_numpy(np.array(x_sample)).permute(2, 0, 1)

                        if opt.diffusers:
                            # x_samples_ddim: [batch_size, C, H, W]
                            x_samples_ddim = torch.stack(x_samples_ddim, dim=0)
                        else:
                            x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8)

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
                                                    class_prompt, len(prompts))

                                all_sims_img.append(sim_img.item())
                                all_sims_text.append(sim_text.item())
                                all_sims_dino.append(sim_dino.item())

                                if opt.calc_face_sim:
                                    sim_face, normal_img_count, except_img_count = \
                                        compare_face_folders(opt.compare_with, sample_dir, dst_num_samples=len(prompts),
                                                             face_engine=opt.face_engine, insightface_app=insightface_app)
                                    # sim_face is a float, so no need to detach().cpu().numpy().
                                    all_sims_face.append(sim_face)
                                    all_normal_img_counts.append(normal_img_count)
                                    all_except_img_counts.append(except_img_count)

                    if not opt.skip_grid:
                        all_samples.append(x_samples_ddim)
                
                    sample_count += batch_size

            # End of the loop over prompts.

            # After opt.n_repeat passes of batched_prompts, save all sample images as an image grid
            if not opt.skip_grid:
                # additionally, save as grid
                # logs/gabrielleunion2023-05-24T18-33-34_gabrielleunion-ada/checkpoints/embeddings_gs-4500.pt
                if opt.compare_with:
                    first_compared_folder = opt.compare_with[0]
                    if first_compared_folder.endswith("/") or first_compared_folder.endswith("\\"):
                        first_compared_folder = first_compared_folder[:-1]

                subj_name_method_sig = opt.subj_name + "-" + opt.method[:3]

                subjfolder_mat = re.search(r"([^\/]+)(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_([^\/]+).*-(\d+)\.pt", first_subj_model_path)
                if subjfolder_mat:
                    date_sig = subjfolder_mat.group(2)
                else:
                    date_sig = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

                iter_mat = re.search(r"(\d+[a-z]?).(pt|safetensors)", first_subj_model_path)
                if iter_mat is not None:
                    iter_sig = iter_mat.group(1)
                else:
                    iter_sig = "unknown"

                if opt.main_unet_filepath:
                    unet_pardir = os.path.basename(os.path.dirname(opt.main_unet_filepath))
                    subj_name_method_sig += "-" + unet_pardir

                scale_sig = f"scale{opt.scale:.1f}"
                experiment_sig = "-".join([date_sig, iter_sig, scale_sig])

                if opt.bb_type:
                    experiment_sig += "-" + opt.bb_type
                
                if opt.enabled_encoders:
                    experiment_sig += "-" + "-".join(opt.enabled_encoders)
                else:
                    experiment_sig += "-" + ",".join(opt.adaface_encoder_types)

                # Use the first prompt of the current chunk from opt.prompt_file as the saved file name.
                if opt.prompt_file:
                    prompt = prompts[0]
                # Otherwise, use the prompt passed by the command line.
                prompt_sig = prompt.replace(" ", "-")[:40]  # Cut too long prompt
                grid_filepath = os.path.join(opt.outdir, f'{subj_name_method_sig}-{prompt_sig}-{experiment_sig}.jpg')
                if os.path.exists(grid_filepath):
                    grid_count = 2
                    grid_filepath = os.path.join(opt.outdir, f'{subj_name_method_sig}-{prompt_sig}-{experiment_sig}-{grid_count}.jpg')
                    while os.path.exists(grid_filepath):
                        grid_count += 1
                        grid_filepath = os.path.join(opt.outdir, f'{subj_name_method_sig}-{prompt_sig}-{experiment_sig}-{grid_count}.jpg')

                # all_samples is a list of 4D tensors [batch_size, C, H, W]
                all_samples_cat = torch.cat(all_samples, 0)
                img = save_grid(all_samples_cat, None, grid_filepath, nrow=n_rows)
                
            toc = time.time()
        
    if not opt.skip_grid:
        print(f"Your samples are at: \n{grid_filepath}")
        #if not (opt.no_preview or opt.prompt_file or opt.compare_with):
        #    os.spawnvp(os.P_NOWAIT, "gpicview", [ "gpicview", os.path.abspath(grid_filepath) ])
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
            subjfolder_mat = re.search(r"([a-zA-Z0-9_]+)(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_([^\/]+)", first_subj_model_path)
            if subjfolder_mat:
                emb_sig  = subjfolder_mat.group(1) + subjfolder_mat.group(2)
            else:
                emb_sig  = opt.method

            emb_sig = subj_name_method_sig + "-" + emb_sig

            scores   = [sims_face_avg, sims_img_avg, sims_text_avg, sims_dino_avg, except_img_percent]
            scores   = [ f"{score:.4f}" for score in scores ]
            SCORES_CSV.writerow([emb_sig] + scores)

    if SCORES_CSV is not None:
        SCORES_CSV_FILE.close()

    return img #return img to be shown on webui

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
