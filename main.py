import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch

import pytorch_lightning as pl

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.personalized import SubjectSampler
from ldm.util import instantiate_from_config
import random

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    # learning rate
    parser.add_argument(
        "--lr",
        type=float, 
        default=argparse.SUPPRESS,
        help="learning rate",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--bs", type=int, 
        default=argparse.SUPPRESS,
        help="Batch size"
    )
    # num_nodes is inherent in Trainer class. No need to specify it here.
    '''
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="max steps",
    )
    '''

    parser.add_argument("--optimizer", dest='optimizer_type',
                        type=str, default=argparse.SUPPRESS, 
                        choices=['AdamW', 'AdamW8bit', 'Prodigy', 'CAdamW'],
                        help="Type of optimizer")
    parser.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS,
                        help="Number of warm up steps")
    
    parser.add_argument("--d_coef",
                        type=float,
                        default=argparse.SUPPRESS,
                        help="Coefficient for d_loss")
    
    parser.add_argument("--base_model_path", 
        type=str,
        required=True,
        default="models/stable-diffusion-v-1-5/v1-5-dste8-vae.safetensors",
        help="Path to model to actually resume from")
    parser.add_argument("--comp_unet_weight_path", 
        type=str,
        default="models/ensemble/sar-unet",
        help="Path to model on compositional distillation iterations. If None, then it's the same as base_model_path)")

    parser.add_argument("--data_roots", 
        type=str, 
        nargs='+', 
        help="Path(s) containing training images")
    parser.add_argument("--mix_subj_data_roots",
        type=str, nargs='+', default=None,
        help="Path(s) containing training images of mixed subjects")
    parser.add_argument("--load_meta_subj2person_type_cache_path",
        type=str, default=None,
        help="Path to load the cache of subject to person type mapping from")
    parser.add_argument("--save_meta_subj2person_type_cache_path",
        type=str, default=None,
        help="Path to save the cache of subject to person type mapping to")

    parser.add_argument("--adaface_encoder_types", type=str, nargs="+", default=argparse.SUPPRESS,
                        choices=["arc2face", "consistentID"], help="Type(s) of the ID2Ada prompt encoders")
    parser.add_argument("--enabled_encoders", type=str, nargs="+", default=argparse.SUPPRESS,
                        choices=["arc2face", "consistentID"], 
                        help="List of enabled encoders (among the list of adaface_encoder_types)")
    
    parser.add_argument('--adaface_ckpt_paths', type=str, nargs="+", 
                        default=[],
                        help="Initialize embedding manager from 1 or 2 checkpoints. "
                             "If 2 checkpoints, the second one provides the attn lora weights.")
    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")

    # default_cls_delta_string is also used as subj_init_string.
    parser.add_argument("--default_cls_delta_string",
        type=str, default='person',
        help="One or more words to be used in class-level prompts for delta loss")
    parser.add_argument("--num_vectors_per_subj_token",
        type=int, default=20,
        help="Number of vectors per subject token. If > 1, use multiple embeddings to represent a subject.")
    
    parser.add_argument("--prompt2token_proj_ext_attention_perturb_ratio", type=float, default=0.1,
                        help="Perturb ratio of the prompt2token projection extended attention")
    parser.add_argument("--p_gen_rand_id_for_id2img", type=float, default=argparse.SUPPRESS,
                        help="Probability of generating random faces during arc2face distillation")
    parser.add_argument("--max_num_unet_distill_denoising_steps", type=int, default=argparse.SUPPRESS,
                        help="Maximum number of denoising steps for UNet distillation (default 4). ")
    parser.add_argument("--max_num_comp_priming_denoising_steps", type=int, default=argparse.SUPPRESS,
                        help="Maximum number of denoising steps (default 4)")
    parser.add_argument("--comp_distill_denoising_steps_range", type=int, default=argparse.SUPPRESS, nargs=2,
                        help="Maximum number of denoising steps for composition distillation (default 3). ")
    parser.add_argument("--p_perturb_face_id_embs", type=float, default=argparse.SUPPRESS,
                        help="Probability of adding noise to real identity embeddings")
    parser.add_argument("--extend_prompt2token_proj_attention_multiplier", type=int, default=1,
                        help="Multiplier of the prompt2token projection attention")
    parser.add_argument("--unet_distill_iter_gap", type=int, default=argparse.SUPPRESS,
                        help="Do unet distillation every N steps in non-compositional iterations")
    parser.add_argument("--unet_teacher_types", type=str, nargs="*", default=argparse.SUPPRESS,
                        choices=["arc2face","consistentID"], 
                        help="Type of the UNet teacher. Multiple values imply unet_ensemble. ")
    parser.add_argument("--p_unet_teacher_uses_cfg", type=float, default=argparse.SUPPRESS,
                        help="The probability that the UNet teacher (as well as the student) uses the classifier-free guidance")    
    parser.add_argument("--unet_teacher_cfg_scale_range", type=float, nargs=2, default=argparse.SUPPRESS,
                        help="Range of the scale of the classifier-free guidance")
    parser.add_argument("--num_static_img_suffix_embs", type=int, default=4,
                        help="Number of extra static learnable image embeddings appended to input ID embeddings")    
    # UNet distillation always uses ffn LoRA, so there's no such an option.
    parser.add_argument("--unet_uses_attn_lora", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use attn LoRA in the cross-attn layers of the Diffusers UNet model")
    parser.add_argument("--recon_uses_ffn_lora", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use FFN LoRA in the reconstruction iterations")    
    parser.add_argument("--comp_uses_ffn_lora", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use FFN LoRA in the compositional distillation iterations")
    parser.add_argument("--unet_lora_rank", type=int, default=argparse.SUPPRESS,
                        help="Rank of the LoRA in the Diffusers UNet model")    
    parser.add_argument("--unet_lora_scale_down", type=float, default=8,
                        help="Scale down factor for the LoRA in the Diffusers UNet model")
    parser.add_argument("--load_unet_attn_lora_from_ckpt", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to load the attn LoRA modules from the checkpoint")
    parser.add_argument("--unet_ffn_adapters_to_load", type=str, nargs="*", 
                        default=['all'],
                        choices=['recon_loss', 'unet_distill', 'comp_distill', 'all', 'none'], 
                        help="Load these ffn adapters from the checkpoint")
    parser.add_argument("--p_shrink_cross_attn_in_comp_iters", type=float, default=argparse.SUPPRESS,
                        help="Whether to suppress the subject attention in the subject-compositional instances")
    parser.add_argument("--cross_attn_shrink_factor", type=float, default=argparse.SUPPRESS,
                        help="Shrink factor of the standard deviation of the subject attention")
    parser.add_argument("--attn_lora_layer_names", type=str, nargs="*", default=['q', 'k', 'v', 'out'],
                        choices=['q', 'k', 'v', 'out'], help="Names of the cross-attn components to apply LoRA on")
    parser.add_argument("--q_lora_updates_query", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether the q LoRA updates the query in the Diffusers UNet model. "
                             "If False, the q lora only updates query2.")
    parser.add_argument("--prompt_emb_delta_reg_weight", type=float, default=argparse.SUPPRESS,
                        help="Prompt delta regularization weight")

    parser.add_argument("--rand_scale_range", type=float, nargs=2, default=[0.4, 1.0],
                        help="Range of random scaling on training images (set to `1 1` to disable)")

    parser.add_argument("--comp_distill_iter_gap",
                        type=int, default=argparse.SUPPRESS,
                        help="Gaps between iterations for composition regularization. "
                             "Set to -1 to disable for ablation.")
    # nargs="?" and const=True: --use_fp_trick or --use_fp_trick True or --use_fp_trick 1 
    # are all equavalent.
    parser.add_argument("--use_fp_trick", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use the 'face portrait' trick for the subject")
    parser.add_argument("--use_face_flow_for_sc_matching_loss", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use face flow for the single-composition matching loss")
       
    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[1, 1],
                        help="Relative weights of the skip connections of the last few layers of CLIP text embedder. " 
                             "(The last element is the weight of the last layer, ...)")
    parser.add_argument("--randomize_clip_skip_weights", nargs="?", type=str2bool, const=True, default=False,
                        help="Whether to randomize the skip weights of CLIP text embedder. "
                             "If True, the weights are sampled from a Dirichlet distribution with clip_last_layers_skip_weights as the alpha.")
    # Since sometimes we repeat compositional part of the prompt for distillation, 
    # we extend clip prompt length to 97.
    parser.add_argument("--clip_prompt_max_length", type=int, default=97,
                        help="Maximum length of the prompt for CLIP text embedder")

    parser.add_argument("--no_wandb", dest='use_wandb', action="store_false", 
                        help="Disable wandb logging")    
    parser.add_argument("--log_attn_level", type=int, default=0,
                        help="Whether and at which level we log the attention weights for visualization")
    parser.add_argument("--ablate_img_embs", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to ablate the image embeddings")
    parser.add_argument("--apex", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use apex")
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

# Set placeholder strings and their corresponding initial words and weights.
# personalization_config_params = config.model.params.personalization_config.params.
# dataset: data.datasets['train'].
def set_placeholders_info(personalization_config_params, opt, dataset):
    # Only keep the first subject placeholder.
    personalization_config_params.subject_strings               = dataset.subject_strings[:1]
    personalization_config_params.subj_name_to_cls_delta_string = dict(zip(dataset.subject_names, dataset.cls_delta_strings))

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def make_worker_init_fn(global_seed):
    def worker_init_fn(_):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            dataset = worker_info.dataset
            dataset.worker_id = worker_id

            # Get DDP rank from torch or env
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = int(os.environ.get("RANK", 0))

            seed = rank * 10000 + global_seed * 1000 + worker_id
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            print(f"Rank {rank} worker {worker_id} seed set to {seed}")

    return worker_init_fn
        
# LightningDataModule: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html
# train: ldm.data.personalized.PersonalizedBase
class DataModuleFromConfig(pl.LightningDataModule):
    # train: the corresponding section in the config file,
    # used by instantiate_from_config(self.dataset_configs[k]).
    def __init__(self, batch_size, max_steps, train=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, 
                 use_worker_init_fn=False, seed=-1):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = max_steps
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn        # True
        self.seed = seed

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # _train_dataloader() is called within prepare_data().
    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = make_worker_init_fn(self.seed)
        else:
            init_fn = None
        
        shuffle = False
        # If there are multiple subjects, we use SubjectSampler to ensure that 
        # each batch contains data from one subject only.
        if self.datasets['train'].num_subjects > 1:
            shuffle = False
            sampler = SubjectSampler(self.datasets['train'].num_subjects, 
                                     self.datasets['train'].subject_names, 
                                     self.datasets['train'].subjects_are_faces, 
                                     self.datasets['train'].image_count_by_subj,
                                     self.num_batches, 
                                     self.batch_size, skip_non_faces=True)
        else:
            sampler = None

        # shuffle=True        
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          shuffle=shuffle, sampler=sampler,
                          num_workers=self.num_workers, 
                          worker_init_fn=init_fn, drop_last=True,
                          pin_memory=False)

    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = make_worker_init_fn(self.seed)
        else:
            init_fn = None

        shuffle = False

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = make_worker_init_fn(self.seed)
        else:
            init_fn = None

        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, timesig, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.timesig = timesig
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.timesig)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.timesig)))


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

class CustomCheckpointSaver(Callback):
    def __init__(self, save_dir, every_n_steps=500):
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)

        # Save a reference to itself, so that I can refer to it in ddpm.py:on_save_checkpoint().
        pl_module.custom_checkpoint_saver = self        
        # Get current global step
        global_step = trainer.global_step

        if global_step % self.every_n_steps == 0 and global_step > 0 and trainer.global_rank == 0:
            pl_module.on_save_checkpoint({})

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_lr: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    timesig = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""

        datadir_in_name = True
        if datadir_in_name:
            first_data_folder = opt.data_roots[0] if opt.data_roots else opt.mix_subj_data_roots[0]
            basename = os.path.basename(os.path.normpath(first_data_folder))
            # If we do multi-subject training, we need to replace the * with "all".
            basename = basename.replace("*", "all")
            timesig  = basename + timesig
            
        nowname = timesig + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir  = os.path.join(logdir, "configs")
    # If do zeroshot and setting seed, then the whole training sequence is deterministic, limiting the random space
    # it can explore. Therefore we don't set seed when doing zero-shot learning.
    if opt.seed > 0:
        seed_everything(opt.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = True

    '''    
    [rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. 
    GPU 1 has a total capacity of 47.41 GiB of which 270.69 MiB is free. Including non-PyTorch memory, 
    this process has 47.10 GiB memory in use. Of the allocated memory 33.18 GiB is allocated by PyTorch, 
    and 13.19 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try 
    setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  
    See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    '''

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["strategy"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            trainer_config["accelerator"] = "gpu"
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # Data config
        if hasattr(opt, "bs"):
            config.data.params.batch_size = opt.bs
        trainer_opt.num_nodes = opt.num_nodes

        # accumulate_grad_batches: Default is 2, specified in v1-finetune-ada.yaml.
        # If specified in command line, then override the default value.
        if opt.accumulate_grad_batches is not None:
            lightning_config.trainer.accumulate_grad_batches = opt.accumulate_grad_batches

        if opt.max_steps > 0:
            trainer_opt.max_steps = opt.max_steps
            # max_steps: Used to initialize DataModuleFromConfig.
            config.data.params.max_steps = opt.max_steps

        # If the global seed is specified in the command line, then we set 
        # seeds in the dataloaders as well for reproducible training data sequences.
        config.data.params.use_worker_init_fn   = (opt.seed > 0)
        config.data.params.seed                 = opt.seed

        config.data.params.train.params.subject_string = opt.subject_string
        if not hasattr(opt, 'adaface_encoder_types'):
            # Use the setting in the config file.
            opt.adaface_encoder_types = list(config.model.params.personalization_config.params.adaface_encoder_types)
        else:
            # Override the setting in the config file.
            config.model.params.personalization_config.params.adaface_encoder_types = opt.adaface_encoder_types

        if not hasattr(opt, 'enabled_encoders'):
            if hasattr(config.model.params.personalization_config.params, 'enabled_encoders'):
                # Use the setting in the config file.
                opt.enabled_encoders = list(config.model.params.personalization_config.params.enabled_encoders)
            else:
                # All the adaface_encoder_types are enabled by default.
                opt.enabled_encoders = opt.adaface_encoder_types
        else:
            # Override the setting in the config file.
            config.model.params.personalization_config.params.enabled_encoders = opt.enabled_encoders
        
        opt.num_adaface_encoder_types = len(opt.enabled_encoders)
        config.data.params.train.params.default_cls_delta_string    = opt.default_cls_delta_string
        config.data.params.train.params.num_vectors_per_subj_token  = \
            opt.num_vectors_per_subj_token + opt.num_static_img_suffix_embs * opt.num_adaface_encoder_types
        config.data.params.train.params.rand_scale_range = opt.rand_scale_range
        
        # config.data:
        # {'target': 'main.DataModuleFromConfig', 'params': {'batch_size': 2, 'num_workers': 2, 
        #  'wrap': False, 'train': {'target': 'ldm.data.personalized.PersonalizedBase', 
        #  'params': {'size': 512, 'set_name': 'train', 'repeats': 100, 
        #  'subject_string': 'z', 'data_roots': 'data/spikelee/'}}, 
        config.data.params.train.params.data_roots               = opt.data_roots
        config.data.params.train.params.mix_subj_data_roots      = opt.mix_subj_data_roots
        config.data.params.train.params.load_meta_subj2person_type_cache_path = opt.load_meta_subj2person_type_cache_path
        config.data.params.train.params.save_meta_subj2person_type_cache_path = opt.save_meta_subj2person_type_cache_path

        if hasattr(opt, 'p_gen_rand_id_for_id2img'):
            config.model.params.p_gen_rand_id_for_id2img    = opt.p_gen_rand_id_for_id2img
        
        if hasattr(opt, 'max_num_unet_distill_denoising_steps'):
            config.model.params.max_num_unet_distill_denoising_steps = opt.max_num_unet_distill_denoising_steps
        if hasattr(opt, 'max_num_comp_priming_denoising_steps'):
            config.model.params.max_num_comp_priming_denoising_steps = opt.max_num_comp_priming_denoising_steps
        if hasattr(opt, 'comp_distill_denoising_steps_range'):
            config.model.params.comp_distill_denoising_steps_range   = opt.comp_distill_denoising_steps_range
        if hasattr(opt, 'p_perturb_face_id_embs'):
            config.model.params.p_perturb_face_id_embs = opt.p_perturb_face_id_embs

        config.model.params.personalization_config.params.extend_prompt2token_proj_attention_multiplier   = opt.extend_prompt2token_proj_attention_multiplier
        config.model.params.personalization_config.params.num_static_img_suffix_embs = opt.num_static_img_suffix_embs
        gpus = opt.gpus.strip(",").split(',')
        device = f"cuda:{gpus[0]}" if len(gpus) > 0 else "cpu"

        config.model.params.personalization_config.params.prompt2token_proj_ext_attention_perturb_ratio = opt.prompt2token_proj_ext_attention_perturb_ratio
        config.model.params.personalization_config.params.load_unet_attn_lora_from_ckpt = opt.load_unet_attn_lora_from_ckpt
        if opt.unet_ffn_adapters_to_load == ['none']:
            config.model.params.personalization_config.params.unet_ffn_adapters_to_load = []
        else:
            config.model.params.personalization_config.params.unet_ffn_adapters_to_load = opt.unet_ffn_adapters_to_load
        
        if hasattr(opt, 'unet_distill_iter_gap'):
            config.model.params.unet_distill_iter_gap = opt.unet_distill_iter_gap
        if hasattr(opt, 'unet_teacher_types'):
            if "unet_ensemble" in opt.unet_teacher_types:
                assert len(opt.unet_teacher_types) == 1, \
                    "If 'unet_ensemble' is specified, this should be the only value for --unet_teacher_types."
            config.model.params.unet_teacher_types = opt.unet_teacher_types

        if hasattr(opt, 'p_unet_teacher_uses_cfg'):
            config.model.params.p_unet_teacher_uses_cfg     = opt.p_unet_teacher_uses_cfg
        if hasattr(opt, 'unet_teacher_cfg_scale_range'):
            config.model.params.unet_teacher_cfg_scale_range = opt.unet_teacher_cfg_scale_range

        if hasattr(opt, 'extra_unet_dirpaths'):
            config.model.params.extra_unet_dirpaths         = opt.extra_unet_dirpaths
        if hasattr(opt, 'unet_weights_in_ensemble'):
            # unet_weights_in_ensemble: not the model weights, but the scalar weights for the teacher UNet models.
            config.model.params.unet_weights_in_ensemble    = opt.unet_weights_in_ensemble

        if hasattr(opt, 'p_shrink_cross_attn_in_comp_iters'):
            config.model.params.p_shrink_cross_attn_in_comp_iters = opt.p_shrink_cross_attn_in_comp_iters
        if hasattr(opt, 'cross_attn_shrink_factor'):
            config.model.params.cross_attn_shrink_factor  = opt.cross_attn_shrink_factor
        config.model.params.attn_lora_layer_names = opt.attn_lora_layer_names
        config.model.params.q_lora_updates_query = opt.q_lora_updates_query

        # data: DataModuleFromConfig
        data = instantiate_from_config(config.data)
        # NOTE according to https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        # calling these ourselves should not be necessary. In trainer.fit(), lightning will calls data.setup().
        # However, some data structures in data['train'] are accessed before trainer.fit(), 
        # therefore we still call it here.
        # This step is SLOW. It takes 5 minutes to load the data.
        data.setup()
        # Suppose the meta_subj2person_type has been saved, we can load it directly and save another 5 minutes.
        if config.data.params.train.params.load_meta_subj2person_type_cache_path is None:
            config.data.params.train.params.load_meta_subj2person_type_cache_path = config.data.params.train.params.load_meta_subj2person_type_cache_path

        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # DDPM model config
        config.model.params.cond_stage_config.params.last_layers_skip_weights    = opt.clip_last_layers_skip_weights
        config.model.params.cond_stage_config.params.randomize_clip_skip_weights = opt.randomize_clip_skip_weights
        config.model.params.cond_stage_config.params.max_length                  = opt.clip_prompt_max_length
        config.model.params.use_fp_trick = opt.use_fp_trick
        
        config.model.params.use_face_flow_for_sc_matching_loss = opt.use_face_flow_for_sc_matching_loss
        config.model.params.unet_uses_attn_lora     = opt.unet_uses_attn_lora
        config.model.params.recon_uses_ffn_lora     = opt.recon_uses_ffn_lora
        config.model.params.comp_uses_ffn_lora      = opt.comp_uses_ffn_lora
        config.model.params.unet_lora_scale_down    = opt.unet_lora_scale_down
        if hasattr(opt, 'unet_lora_rank'):
            config.model.params.unet_lora_rank = opt.unet_lora_rank

        # Setting prompt_emb_delta_reg_weight to 0 will disable prompt delta regularization.
        if hasattr(opt, 'prompt_emb_delta_reg_weight'):
            config.model.params.prompt_emb_delta_reg_weight     = opt.prompt_emb_delta_reg_weight

        if hasattr(opt, 'comp_distill_iter_gap'):   
            config.model.params.comp_distill_iter_gap = opt.comp_distill_iter_gap

        if hasattr(opt, 'optimizer_type'):
            config.model.params.optimizer_type = opt.optimizer_type

        if hasattr(opt, 'warmup_steps'):
            if config.model.params.optimizer_type == 'Prodigy':
                config.model.params.prodigy_config.warm_up_steps                       = opt.warmup_steps
            else:
                config.model.params.adam_config.scheduler_config.params.warm_up_steps  = opt.warmup_steps

        if hasattr(opt, 'd_coef'):
            config.model.params.prodigy_config.d_coef = opt.d_coef

        if hasattr(opt, 'lr'):
            config.model.base_lr = opt.lr

        config.model.params.log_attn_level = opt.log_attn_level
        config.model.params.ablate_img_embs = opt.ablate_img_embs

        # Personalization config
        config.model.params.personalization_config.params.adaface_ckpt_paths    = opt.adaface_ckpt_paths    
        set_placeholders_info(config.model.params.personalization_config.params, opt, data.datasets['train'])

        if opt.base_model_path:
            config.model.params.base_model_path = opt.base_model_path
            config.model.params.comp_unet_weight_path = opt.comp_unet_weight_path

        # model will be loaded by ddpm.init_from_ckpt(). No need to load manually.
        model = instantiate_from_config(config.model)
        # model: ldm.models.diffusion.ddpm.LatentDiffusion, inherits from LightningModule.
        # model.cond_stage_model: FrozenCLIPEmbedder = text_embedder


        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        logger_name = "wandb" if opt.use_wandb else "testtube"
        default_logger_cfg = default_logger_cfgs[logger_name]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # Maintain the same frequency of saving checkpoints when accumulate_grad_batches > 1.
        # modelckpt_cfg.params.every_n_train_steps //= config.model.params.accumulate_grad_batches
        # print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        save_ckpt_every_n_steps = lightning_config.modelcheckpoint.params.every_n_train_steps

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "timesig": timesig,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "custom_checkpoint_saver": {
                "target": "main.CustomCheckpointSaver",
                "params": {
                    "every_n_steps": save_ckpt_every_n_steps,
                    "save_dir": ckptdir,
                }
            },
        }
        # default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["max_steps"] = trainer_opt.max_steps
        trainer_kwargs["log_every_n_steps"] = 10
        trainer_kwargs["profiler"] = opt.profiler

        if opt.apex:
            trainer_kwargs["amp_backend"]   = "apex"
            trainer_kwargs["amp_level"]     = "O3"
        else:
            trainer_kwargs["precision"]     = opt.precision
        
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # configure learning rate
        bs, base_lr, weight_decay = config.data.params.batch_size, config.model.base_lr, \
                                    config.model.weight_decay

        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        # scale_lr = True by default. So learning_rate is set to 2*base_lr.
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")
        
        model.weight_decay = weight_decay
        model.model.diffusion_model.debug_attn = opt.debug

        # model.create_clip_evaluator(f"cuda:{trainer.strategy.root_device.index}")

        '''
        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        '''

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        #signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                # trainer: pytorch_lightning.trainer.trainer.Trainer
                trainer.fit(model, data)
            except Exception as e:
                print(f"Exception caught: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()  # Print the full stack trace

        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
