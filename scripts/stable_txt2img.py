import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import re

from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.eval_utils import compare_folders, compare_face_folders, init_evaluators, set_tf_gpu

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
        help="if enabled, uses the same starting code across samples ",
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
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
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
    parser.add_argument(
        "--subj_scale",
        type=float,
        default=1.0,
        help="Scale of the subject embedding",
    )

    parser.add_argument("--ada_emb_weight",
        type=float, default=-1,
        help="Weight of adaptive embeddings (in contrast to static embeddings)")

    parser.add_argument(
        "--init_img",
        type=str,
        default=None,
        help="path to the initial image",
    )  
    # Anything between 0 and 1 will cause blended images.
    parser.add_argument(
        "--mask_weight",
        type=float,
        default=0.0,
        help="Weight of the initial image",
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
    parser.add_argument("--is_face", action="store_true",
                        help="Whether the generated samples are human faces")
        
    parser.add_argument('--gpu', type=int,  default=0, help='ID of GPU to use')
    parser.add_argument("--compare_with", type=str, default=None,
                        help="Evaluate the similarity of generated samples with reference images in this folder")
    parser.add_argument("--orig_prompt", type=str, default=None,
                        help="the original prompt used for text/image matching evaluation "
                             "(requires --compare_with to be specified)")

    parser.add_argument("--clip_last_layer_skip_weight", type=float, default=0.5,
                        help="Weight of the skip connection between the last layer and second last layer of CLIP text embedder")
    parser.add_argument("--clip_last_layer_skip_scheme", type=str, choices=["add", "concat"], 
                        default="add", 
                        help="Scheme for the skip connection between the last layer and second last layer of CLIP text embedder")
                                
    args = parser.parse_args()
    return args

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# copied from img2img.py
def load_img(path, h, w):
    image = Image.open(path).convert("RGB")
    w0, h0 = image.size
    print(f"loaded input image of size ({w0}, {h0}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main(opt):

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model  = load_model_from_config(config, f"{opt.ckpt}")
    if opt.embedding_paths is not None:
        model.embedding_manager.load(opt.embedding_paths)
    model.embedding_manager.subj_scale  = opt.subj_scale

    # cond_stage_model: ldm.modules.encoders.modules.FrozenCLIPEmbedder
    model.cond_stage_model.set_last_layer_skip(opt.clip_last_layer_skip_weight, 
                                               opt.clip_last_layer_skip_scheme)

    use_diff_ada_emb_weight = False
    if use_diff_ada_emb_weight and opt.ada_emb_weight == -1:
        # Smaller ada embedding weight for objects and cartoon characters, larger for humans.
        default_ada_emb_weights = [ 0.2, 0.5, 0.2 ]
        opt.ada_emb_weight = default_ada_emb_weights[opt.broad_class]

    if opt.ada_emb_weight != -1:
        model.embedding_manager.ada_emb_weight = opt.ada_emb_weight

    torch.cuda.set_device(opt.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model  = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)

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
        batched_orig_prompts = [ opt.orig_prompt ] * len(batched_prompts)

    else:
        print(f"Reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            # splitlines() will remove the trailing newline. So no need to strip().
            lines = f.read().splitlines()
            indiv_subdirs_prompts = [ line.split("\t") for line in lines ]
            n_repeats, indiv_subdirs, all_prompts, orig_prompts = zip(*indiv_subdirs_prompts)
            # Repeat each prompt n_repeats times, and split into batches of size batch_size.
            # If there's remainder after chunks, the last chunk will be shorter than batch_size.

            batched_prompts = []
            batched_subdirs = []
            batched_orig_prompts = []

            # Repeat each prompt n_repeats times.
            for i, prompt in enumerate(all_prompts):
                n_repeat = int(n_repeats[i])
                # If in this line, n_repeat is larger than batch_size, we need to split it 
                # into n_batches > 1 batches. These batches share the same indiv_subdir and orig_prompt.
                # So no need to repeat them in advance. Just append n_batches copies of them 
                # to batched_subdirs and batched_orig_prompts respectively below.
                indiv_subdir = indiv_subdirs[i]
                orig_prompt  = orig_prompts[i]
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
                batched_orig_prompts.extend([orig_prompt] * n_batches)

            # Append None to the end of batched_subdirs, for indiv_subdir change detection.
            batched_subdirs.append(None)

    if opt.compare_with:
        clip_evator, dino_evator = init_evaluators(opt.gpu)
        all_sims_img, all_sims_text, all_sims_dino = [], [], []
        if opt.is_face:
            all_sims_face = []
            set_tf_gpu(opt.gpu)
    else:
        clip_evator, dino_evator = None, None

    if opt.init_img is not None:
        assert opt.fixed_code is False
        init_img = load_img(opt.init_img, opt.H, opt.W)
        init_img = init_img.repeat([batch_size, 1, 1, 1]).to(device)
        # move init_img to latent space
        x0      = model.get_first_stage_encoding(model.encode_first_stage(init_img))  
        mask    = torch.ones_like(x0) * opt.mask_weight
    else:
        x0      = None
        mask    = None

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

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
                        print(f"\n{p_i+1}/{prompt_block_count}", prompts[0], "...", prompts[-1])
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        # When ada embedding is used, c is a tuple of (cond, ada_embedder).
                        # When both unconditional and conditional guidance are passed to ddim sampler, 
                        # ada_embedder of the conditional guidance is applied on both 
                        # unconditional and conditional embeddings within UNetModel.forward(). 
                        # But since unconditional prompt doesn't contain the placeholder token,
                        # ada_embedder won't change the unconditional embedding uc.
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x0=x0,
                                                         mask=mask,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            indiv_subdir = batched_subdirs[p_i]
                            orig_prompt  = batched_orig_prompts[p_i]
                            sample_dir = os.path.join(opt.outdir, indiv_subdir)
                            os.makedirs(sample_dir, exist_ok=True)                            

                            for i, x_sample in enumerate(x_samples_ddim):
                                base_count = len(os.listdir(sample_dir))
                                sample_file_path = os.path.join(sample_dir, f"{base_count:05}.jpg")

                                while os.path.exists(sample_file_path):
                                    base_count += 1
                                    sample_file_path = os.path.join(sample_dir, f"{base_count:05}.jpg")

                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(sample_file_path)

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
                                                        orig_prompt, len(prompts))

                                    all_sims_img.append(sim_img.item())
                                    all_sims_text.append(sim_text.item())
                                    all_sims_dino.append(sim_dino.item())

                                    if opt.is_face:
                                        sim_face = compare_face_folders(opt.compare_with, sample_dir, len(prompts))
                                        # sim_face is a float, so no need to detach().cpu().numpy().
                                        all_sims_face.append(sim_face)
                                        
                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                        sample_count += batch_size

                # After opt.n_repeat passes of batched_prompts, save all sample images as an image grid
                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    # logs/name2022-12-27T23-03-14_name/checkpoints/embeddings_gs-1200.pt
                    date_mat = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_([^\/]+)", opt.embedding_paths[0])
                    date_sig = date_mat.group(1)
                    ckpt_postfix = date_mat.group(2)
                    iter_mat = re.search(r"(\d+).pt", opt.embedding_paths[0])
                    iter_sig = iter_mat.group(1)

                    embedding_sig = "-".join([date_sig, iter_sig, f"scale{opt.scale:.1f}"])
                    # Use the first prompt of the current chunk from opt.from_file as the saved file name.
                    if opt.from_file:
                        prompt = prompts[0]
                    # Otherwise, use the prompt passed by the command line.
                    prompt_sig = prompt.replace(" ", "-")[:40]  # Cut too long prompt
                    grid_filepath = os.path.join(opt.outdir, f'{ckpt_postfix}-{prompt_sig}-{embedding_sig}.jpg')
                    if os.path.exists(grid_filepath):
                        grid_count = 2
                        grid_filepath = os.path.join(opt.outdir, f'{ckpt_postfix}-{prompt_sig}-{embedding_sig}-{grid_count}.jpg')
                        while os.path.exists(grid_filepath):
                            grid_count += 1
                            grid_filepath = os.path.join(opt.outdir, f'{ckpt_postfix}-{prompt_sig}-{embedding_sig}-{grid_count}.jpg')

                    Image.fromarray(grid.astype(np.uint8)).save(grid_filepath)

                toc = time.time()

    if not opt.skip_grid:
        print(f"Your samples are at: \n{grid_filepath}")
        if not (opt.no_preview or opt.from_file or opt.compare_with):
            os.spawnvp(os.P_NOWAIT, "gpicview", [ "gpicview", os.path.abspath(grid_filepath) ])
    else:
        print(f"Your samples are at: \n{opt.outdir}")

    if opt.compare_with:
        sims_img_avg, sims_text_avg, sims_dino_avg = np.mean(all_sims_img), np.mean(all_sims_text), np.mean(all_sims_dino)
        print(f"All samples mean image/text/dino sim: {sims_img_avg:.3f} {sims_text_avg:.3f} {sims_dino_avg:.3f}")
        if opt.is_face:
            sims_face_avg = np.mean(all_sims_face)
            print(f"All samples mean face sim: {sims_face_avg:.3f}")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
