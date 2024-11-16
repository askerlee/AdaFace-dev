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
        default=23,
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
                        choices=['AdamW', 'AdamW8bit', 'Adam8bit', 'NAdam', 'Prodigy',
                                 'AdEMAMix', 'AdEMAMixDistributedShampoo', 'OrthogonalNesterov'],
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
                        help="Initialize embedding manager from one or multiple checkpoints")
    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")

    # default_cls_delta_string is also used as subj_init_string.
    parser.add_argument("--default_cls_delta_string",
        type=str, default='person',
        help="One or more word tso be used in class-level prompts for delta loss")
    parser.add_argument("--num_vectors_per_subj_token",
        type=int, default=20,
        help="Number of vectors per subject token. If > 1, use multiple embeddings to represent a subject.")
    
    parser.add_argument("--prompt2token_proj_ext_attention_perturb_ratio", type=float, default=0.1,
                        help="Perturb ratio of the prompt2token projection extended attention")
    parser.add_argument("--p_gen_rand_id_for_id2img", type=float, default=argparse.SUPPRESS,
                        help="Probability of generating random faces during arc2face distillation")
    parser.add_argument("--max_num_unet_distill_denoising_steps", type=int, default=4,
                        help="Maximum number of denoising steps for UNet distillation (default 4). ")
    parser.add_argument("--max_num_comp_priming_denoising_steps", type=int, default=4,
                        help="Maximum number of denoising steps (default 4)")
    parser.add_argument("--range_comp_distill_denoising_steps", type=int, default=[2, 3], nargs=2,
                        help="Maximum number of denoising steps for composition distillation (default 3). ")
    parser.add_argument("--p_perturb_face_id_embs", type=float, default=argparse.SUPPRESS,
                        help="Probability of adding noise to real identity embeddings")
    parser.add_argument("--extend_prompt2token_proj_attention_multiplier", type=int, default=1,
                        help="Multiplier of the prompt2token projection attention")
    parser.add_argument("--unet_distill_iter_gap", type=int, default=argparse.SUPPRESS,
                        help="Do unet distillation every N steps in non-compositional iterations")
    parser.add_argument("--unet_teacher_types", type=str, nargs="*", default=argparse.SUPPRESS,
                        choices=["arc2face","consistentID", "unet_ensemble"], 
                        help="Type(s) of the UNet teacher. Multiple values imply unet_ensemble. "
                             "If 'unet_ensemble' is specified, this should be the only value, and "
                             "the types of unets are specified with --extra_unet_dirpaths.")
    parser.add_argument("--p_unet_teacher_uses_cfg", type=float, default=argparse.SUPPRESS,
                        help="The probability that the UNet teacher (as well as the student) uses the classifier-free guidance")    
    parser.add_argument("--unet_teacher_cfg_scale_range", type=float, nargs=2, default=argparse.SUPPRESS,
                        help="Range of the scale of the classifier-free guidance")
    parser.add_argument("--num_static_img_suffix_embs", type=int, default=4,
                        help="Number of extra static learnable image embeddings appended to input ID embeddings")    
    # --extra_unet_dirpaths and --unet_weights are only used when unet_teacher_types contains multiple values or is 'unet_ensemble'.
    parser.add_argument("--extra_unet_dirpaths", type=str, nargs="*", 
                        default=argparse.SUPPRESS, 
                        help="Extra paths to the checkpoints of the teacher UNet models (other than the default one)")
    parser.add_argument('--unet_weights', type=float, nargs="+", default=argparse.SUPPRESS,
                        help="Weights for the teacher UNet models")
    parser.add_argument("--use_ldm_unet", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use the LDM UNet implementation as the base UNet")
    parser.add_argument("--diffusers_unet_path", type=str, default='models/ensemble/sd15-unet',
                        help="Path to the Diffusers UNet model")
    parser.add_argument("--diffusers_unet_uses_lora", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use LoRA in the Diffusers UNet model")

    parser.add_argument("--prompt_emb_delta_reg_weight",
        type=float, default=argparse.SUPPRESS,
        help="Prompt delta regularization weight")

    parser.add_argument("--rand_scale_range",
                        type=float, nargs=2, 
                        default=[0.7, 1.0],
                        help="Range of random scaling on training images (set to 1 1 to disable)")

    parser.add_argument("--comp_distill_iter_gap",
                        type=int, default=argparse.SUPPRESS,
                        help="Gaps between iterations for composition regularization. "
                             "Set to -1 to disable for ablation.")
    # nargs="?" and const=True: --use_fp_trick or --use_fp_trick True or --use_fp_trick 1 
    # are all equavalent.
    parser.add_argument("--use_fp_trick", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use the 'face portrait' trick for the subject")
    parser.add_argument("--use_face_flow_for_sc_matching_loss", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use face flow for the single-composition matching loss")
    parser.add_argument("--gen_ss_from_frozen_subj_basis_generator", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to generate the subject-single ada embeddings from the frozen subject basis generator")
            
    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[1, 1],
                        help="Relative weights of the skip connections of the last few layers of CLIP text embedder. " 
                             "(The last element is the weight of the last layer, ...)")

    parser.add_argument("--randomize_clip_skip_weights", nargs="?", type=str2bool, const=True, default=False,
                        help="Whether to randomize the skip weights of CLIP text embedder. "
                             "If True, the weights are sampled from a Dirichlet distribution with clip_last_layers_skip_weights as the alpha.")

    parser.add_argument("--no_wandb", dest='use_wandb', action="store_false", 
                        help="Disable wandb logging")    
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
    # subjects_are_faces are always available in dataset.
    personalization_config_params.subj_name_to_being_faces = dict(zip(dataset.subject_names, dataset.subjects_are_faces))
    
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    return np.random.seed(np.random.get_state()[1][0] + worker_id)

# LightningDataModule: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html
# train: ldm.data.personalized.PersonalizedBase
class DataModuleFromConfig(pl.LightningDataModule):
    # train: the corresponding section in the config file,
    # used by instantiate_from_config(self.dataset_configs[k]).
    def __init__(self, batch_size, max_steps, train=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = max_steps
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn        # False
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
            init_fn = worker_init_fn
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
            init_fn = worker_init_fn
        else:
            init_fn = None

        shuffle = False

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
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

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

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

# ModeSwapCallback is never used in the code.
class ModeSwapCallback(Callback):

    def __init__(self, swap_step=2000):
        super().__init__()
        self.is_frozen = False
        self.swap_step = swap_step

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_step < self.swap_step and not self.is_frozen:
            self.is_frozen = True
            trainer.optimizers = [pl_module.configure_opt_embedding()]

        if trainer.global_step > self.swap_step and self.is_frozen:
            self.is_frozen = False
            trainer.optimizers = [pl_module.configure_opt_model()]

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
    # seed_everything(opt.seed, workers=True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    '''    
    [rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. 
    GPU 1 has a total capacity of 47.41 GiB of which 270.69 MiB is free. Including non-PyTorch memory, 
    this process has 47.10 GiB memory in use. Of the allocated memory 33.18 GiB is allocated by PyTorch, 
    and 13.19 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try 
    setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  
    See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    '''

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'

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

        config.model.params.personalization_config.params.gen_ss_from_frozen_subj_basis_generator = opt.gen_ss_from_frozen_subj_basis_generator
        
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
            
        config.model.params.max_num_unet_distill_denoising_steps = opt.max_num_unet_distill_denoising_steps
        config.model.params.max_num_comp_priming_denoising_steps = opt.max_num_comp_priming_denoising_steps
        config.model.params.range_comp_distill_denoising_steps   = opt.range_comp_distill_denoising_steps
        if hasattr(opt, 'p_perturb_face_id_embs'):
            config.model.params.p_perturb_face_id_embs = opt.p_perturb_face_id_embs

        config.model.params.personalization_config.params.extend_prompt2token_proj_attention_multiplier   = opt.extend_prompt2token_proj_attention_multiplier
        config.model.params.personalization_config.params.num_static_img_suffix_embs = opt.num_static_img_suffix_embs
        gpus = opt.gpus.strip(",").split(',')
        device = f"cuda:{gpus[0]}" if len(gpus) > 0 else "cpu"

        config.model.params.personalization_config.params.prompt2token_proj_ext_attention_perturb_ratio = opt.prompt2token_proj_ext_attention_perturb_ratio

        if hasattr(opt, 'unet_distill_iter_gap'):
            config.model.params.unet_distill_iter_gap = opt.unet_distill_iter_gap
        if hasattr(opt, 'unet_teacher_types'):
            if "unet_ensemble" in opt.unet_teacher_types:
                assert len(opt.unet_teacher_types) == 1, \
                    "If 'unet_ensemble' is specified, this should be the only value for --unet_teacher_types."
            config.model.params.unet_teacher_types = opt.unet_teacher_types

        if hasattr(opt, 'p_unet_teacher_uses_cfg'):
            config.model.params.p_unet_teacher_uses_cfg         = opt.p_unet_teacher_uses_cfg
        if hasattr(opt, 'unet_teacher_cfg_scale_range'):
            config.model.params.unet_teacher_cfg_scale_range    = opt.unet_teacher_cfg_scale_range

        if hasattr(opt, 'extra_unet_dirpaths'):
            config.model.params.extra_unet_dirpaths             = opt.extra_unet_dirpaths
        if hasattr(opt, 'unet_weights'):
            # unet_weights: not the model weights, but the scalar weights for the teacher UNet models.
            config.model.params.unet_weights                 = opt.unet_weights

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
        config.model.params.use_fp_trick = opt.use_fp_trick
        config.model.params.use_face_flow_for_sc_matching_loss = opt.use_face_flow_for_sc_matching_loss
        config.model.params.use_ldm_unet = opt.use_ldm_unet
        config.model.params.diffusers_unet_path = opt.diffusers_unet_path
        config.model.params.diffusers_unet_uses_lora = opt.diffusers_unet_uses_lora
        
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

        # Personalization config
        config.model.params.personalization_config.params.adaface_ckpt_paths    = opt.adaface_ckpt_paths    
        set_placeholders_info(config.model.params.personalization_config.params, opt, data.datasets['train'])

        if opt.base_model_path:
            config.model.params.base_model_path = opt.base_model_path
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

        # modelcheckpoint - use monitor to specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "save_top_k": 0,
            }
        }

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            modelckpt_cfg["params"]["monitor"] = model.monitor
            modelckpt_cfg["params"]["save_top_k"] = 0

        # Maintain the same frequency of saving checkpoints when accumulate_grad_batches > 1.
        # modelckpt_cfg.params.every_n_train_steps //= config.model.params.accumulate_grad_batches
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

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
        }
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

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
        
        if hasattr(trainer_opt, 'grad_clip'):
            trainer_kwargs["gradient_clip_val"] = trainer_opt.grad_clip
        
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

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                # trainer: pytorch_lightning.trainer.trainer.Trainer
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
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
