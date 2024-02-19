import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.data.personalized import SubjectSampler
from ldm.util import instantiate_from_config, extend_nn_embedding, init_zero_shot_image_encoders
import re
from safetensors.torch import load_file as safetensors_load_file

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith(".ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
    elif ckpt.endswith(".safetensors"):
        sd = safetensors_load_file(ckpt, device="cpu")
        pl_sd = None
    else:
        print(f"Unknown checkpoint format: {ckpt}")
        sys.exit(1)

    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # Release some RAM. Not sure if it really works.
    del sd, pl_sd
    return model

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
        default=-1,
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
    # max_steps is inherent in Trainer class. No need to specify it here.
    '''
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="max steps",
    )
    '''

    parser.add_argument("--optimizer", dest='optimizer_type',
                        type=str, default=argparse.SUPPRESS, choices=['AdamW', 'Prodigy'],
                        help="Type of optimizer")
    
    parser.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS,
                        help="Number of warm up steps")
    
    parser.add_argument("--d_coef",
                        type=float,
                        default=argparse.SUPPRESS,
                        help="Coefficient for d_loss")
    
    parser.add_argument("--actual_resume", 
        type=str,
        required=True,
        help="Path to model to actually resume from")

    parser.add_argument("--data_roots", 
        type=str, 
        nargs='+', 
        required=True, 
        help="Path(s) with training images")
    parser.add_argument("--subj_info_filepaths",
        type=str, nargs="*", default=argparse.SUPPRESS,
        help="Path to the subject info file (only necessary if multiple subjects are used)")

    parser.add_argument("--embedding_manager_ckpt", 
        type=str, 
        default="", 
        help="Initialize embedding manager from a checkpoint")
    parser.add_argument("--src_placeholders",
        type=str, nargs="?", const='1,1', default=None,
        help="Load the embedder components from these placeholders in the checkpoint")
    parser.add_argument("--loaded_embedder_components",
        type=str, default=None, 
        help="Embedder components to be loaded from the checkpoint (candidates: pooler,layer_coeff_maps)")
    parser.add_argument("--frozen_placeholder_set",
        type=str, default=None,
        help="Freeze the embedder components of this set of placeholders (candidates: subj,bg)")
    parser.add_argument("--frozen_embedder_components",
        type=str, default=None, 
        help="Embedder components to be frozen after loading from the checkpoint (candidates: pooler,layer_coeff_maps)")
        
    parser.add_argument("--ckpt_params_perturb_ratio",
        type=float, default=-1,
        help="Ratio of parameters in the loaded ckpt to be perturbed")
    parser.add_argument("--emb_reg_loss_scale",
        type=float, default=1,
        help="Scale of the pre-specified embedding regularization loss")
    
    parser.add_argument("--subject_string", 
                        type=str, default="z",
                        help="Subject placeholder string used in prompts to denote the concept.")
    parser.add_argument("--background_string", 
        type=str, default="y",
        help="Background placeholder string used in prompts to represent the background in training images.")
    parser.add_argument("--common_placeholder_prefix",
        type=str, default=None,
        help="Prefix of the placeholder string for all types of prompts. Default: None.")
    parser.add_argument("--compos_placeholder_prefix",
        type=str, default=None,
        help="Prefix of the placeholder string for compositional prompts. Default: None.")
    
    parser.add_argument("--init_string", 
        type=str, 
        help="Words used to initialize placeholder embedding")

    parser.add_argument("--bg_init_words", 
        type=str, default="unknown",    # 'unknown' should be a wild-card word to match various actual background patterns.
        help="Words used to initialize background embedding")

    parser.add_argument("--init_word_weights", nargs="*", 
        type=float, 
        help="Weights of each token in init_string")

    parser.add_argument("--cls_delta_string",
        type=str, default=None,
        help="One or more word tso be used in class-level prompts for delta loss")
    
    parser.add_argument("--num_vectors_per_subj_token",
        type=int, default=9,
        help="Number of vectors per subject token. If > 1, use multiple embeddings to represent a subject.")
    parser.add_argument("--num_vectors_per_bg_token",
        type=int, default=4,
        help="Number of vectors for the background token. If > 1, use multiple embeddings to represent the background.")
    
    parser.add_argument("--use_conv_attn_kernel_size",
                        type=int, default=None,
                        help="Use convolutional attention of subject tokens with this kernel size."
                             "Default: None, not specified.")

    parser.add_argument("--zeroshot", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use zero-shot learning")
                        
    parser.add_argument("--zs_clip_type", type=str, choices=['openai', 'laion'],
                        default='openai',
                        help="Type of zero-shot learning clip model")

    parser.add_argument("--layerwise_lora_rank", 
        type=int, default=5,
        help="Layerwise lora rank")
    parser.add_argument("--attn_pooler_feat_reduction_ratio",
        type=float, default=argparse.SUPPRESS,
        help="Feature reduction ratio adopted by the attention pooler")
    
    parser.add_argument("--static_embedding_reg_weight",
        type=float, default=-1,
        help="Static embedding regularization weight")
    parser.add_argument("--ada_embedding_reg_weight",
        type=float, default=-1,
        help="Ada embedding regularization weight")
        
    parser.add_argument("--ada_emb_weight",
        type=float, default=-1,
        help="Weight of ada embeddings (in contrast to static embeddings)")

    parser.add_argument("--prompt_emb_delta_reg_weight",
        type=float, default=-1,
        help="Prompt delta regularization weight")

    parser.add_argument("--padding_embs_align_loss_weight",
        type=float, default=-1,
        help="Weight of the padding embeddings alignment loss")
    parser.add_argument("--mix_prompt_distill_weight",
        type=float, default=-1,
        help="Weight of the mixed prompt distillation loss")
    
    parser.add_argument("--comp_fg_bg_preserve_loss_weight",
        type=float, default=-1,
        help="Weight of the composition foreground-background preservation loss")
    
    parser.add_argument("--rand_scale_range",
                        type=float, nargs=2, 
                        default=[0.7, 1.0],
                        help="Range of random scaling on training images (set to 1 1 to disable)")

    parser.add_argument("--composition_regs_iter_gaps",
                        type=int, default=argparse.SUPPRESS,
                        help="Gaps between iterations for composition regularization. "
                             "Set to -1 to disable for ablation.")
    
    # num_compositions_per_image: a value > 1 leads to better performance on prompt compositions
    parser.add_argument("--num_compositions_per_image",
                        type=int, default=1,
                        help="Number of composition samples for each image in a batch (default: 2)")
    parser.add_argument("--broad_class", type=int, default=1,
                        help="Whether the subject is a human/animal, object or cartoon (0: object, 1: human/animal, 2: cartoon)")
    # nargs="?" and const=True: --use_fp_trick or --use_fp_trick True or --use_fp_trick 1 
    # are all equavalent.
    parser.add_argument("--use_fp_trick", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to use the 'face portrait' trick for the subject")

    parser.add_argument("--wds_comp_db_path", type=str, default=None,
                        help="Path to the composition webdatabase .tar file")
    parser.add_argument("--wds_background_string", 
        type=str, default="w",
        help="Background string which will be used in wds prompts to represent the background in wds training images.")
    
    parser.add_argument("--clip_last_layers_skip_weights", type=float, nargs='+', default=[1, 1],
                        help="Relative weights of the skip connections of the last few layers of CLIP text embedder. " 
                             "(The last element is the weight of the last layer, ...)")

    parser.add_argument("--randomize_clip_skip_weights", nargs="?", type=str2bool, const=True, default=False,
                        help="Whether to randomize the skip weights of CLIP text embedder. "
                             "If True, the weights are sampled from a Dirichlet distribution with clip_last_layers_skip_weights as the alpha.")
    
    parser.add_argument("--no_wandb", dest='use_wandb', action="store_false", 
                        help="Disable wandb logging")    
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
    num_subjects = dataset.num_subjects
    if not hasattr(opt, 'subj_info_filepaths'):
        assert num_subjects == 1

    # Single subject. All params are specified in the opt arguments.
    if not hasattr(opt, 'subj_info_filepaths'):
        if opt.init_word_weights is not None and len(opt.init_word_weights) > 0:
            assert len(opt.init_word_weights) == len(re.split("\s+", opt.init_string))
        else:
            # None will be converted to a list of 1.0s in EmbeddingManager.
            opt.init_word_weights = None

        personalization_config_params.placeholder_strings       = [opt.subject_string]
        # opt.init_string has to be present. 
        # opt.init_word_weights could be absent.
        personalization_config_params.list_initializer_words[0]   = opt.init_string
        personalization_config_params.list_initializer_weights[0] = opt.init_word_weights
        personalization_config_params.list_cls_delta_strings[0]   = config.data.params.train.params.cls_delta_string
        personalization_config_params.token2num_vectors       = { opt.subject_string: opt.num_vectors_per_subj_token}

        if opt.background_string is not None:
            config.model.params.use_background_token = True
            personalization_config_params.placeholder_strings.append(opt.background_string)
            # Use a list to store both background_string and wds_background_string, 
            # if the latter is specified.
            personalization_config_params.list_initializer_words.append(opt.bg_init_words)
            personalization_config_params.list_initializer_weights.append(None)            
            personalization_config_params.list_cls_delta_strings.append(config.data.params.train.params.cls_bg_delta_string)
            # The background token can be represented with multiple embeddings.
            personalization_config_params.token2num_vectors[opt.background_string] = opt.num_vectors_per_bg_token
            personalization_config_params.background_strings = [opt.background_string]

            if opt.wds_comp_db_path is not None:
                personalization_config_params.placeholder_strings.append(opt.wds_background_string)
                # wds_background_strings share the same settings of the background string.
                personalization_config_params.list_initializer_words.append(opt.bg_init_words)
                personalization_config_params.list_initializer_weights.append(None)            
                personalization_config_params.list_cls_delta_strings.append(config.data.params.train.params.cls_bg_delta_string)
                personalization_config_params.token2num_vectors[opt.wds_background_string] = opt.num_vectors_per_bg_token
                personalization_config_params.background_strings.append(opt.wds_background_string)
    # Multiple subjects. The params are extracted from opt.subj_info_filepaths 
    # by the dataset object.
    else:
        if not opt.zeroshot:
            personalization_config_params.placeholder_strings       = dataset.subject_strings
            personalization_config_params.list_initializer_words    = dataset.cls_delta_strings
            personalization_config_params.list_initializer_weights  = dataset.list_initializer_weights
            personalization_config_params.list_cls_delta_strings    = dataset.cls_delta_strings
            personalization_config_params.token2num_vectors     = dict()
            for subject_string in dataset.subject_strings:
                personalization_config_params.token2num_vectors[subject_string] = opt.num_vectors_per_subj_token

            if opt.background_string is not None:
                config.model.params.use_background_token = True
                personalization_config_params.placeholder_strings       += dataset.background_strings
                personalization_config_params.list_initializer_words    += dataset.cls_bg_delta_strings
                personalization_config_params.list_initializer_weights  += dataset.list_bg_initializer_weights
                personalization_config_params.list_cls_delta_strings    += dataset.cls_bg_delta_strings
                personalization_config_params.background_strings        = dataset.background_strings

                for background_string in dataset.background_strings:
                    personalization_config_params.token2num_vectors[background_string] = opt.num_vectors_per_bg_token

            if opt.wds_comp_db_path is not None:
                # wds_background_strings share the same settings of the background string.
                personalization_config_params.placeholder_strings       += dataset.wds_background_strings
                personalization_config_params.list_initializer_words    += dataset.cls_bg_delta_strings
                personalization_config_params.list_initializer_weights  += dataset.list_bg_initializer_weights
                personalization_config_params.list_cls_delta_strings    += dataset.cls_bg_delta_strings
                personalization_config_params.background_strings        += dataset.wds_background_strings

                for wds_background_string in dataset.wds_background_strings:
                    personalization_config_params.token2num_vectors[wds_background_string] = opt.num_vectors_per_bg_token
        else:
            # Only keep the first subject and background placeholder.
            personalization_config_params.placeholder_strings       = dataset.subject_strings[:1]
            personalization_config_params.list_initializer_words    = dataset.cls_delta_strings[:1]
            personalization_config_params.list_initializer_weights  = dataset.list_initializer_weights[:1]
            personalization_config_params.list_cls_delta_strings    = dataset.cls_delta_strings[:1]
            personalization_config_params.token2num_vectors     = dict()
            for subject_string in dataset.subject_strings[:1]:
                personalization_config_params.token2num_vectors[subject_string] = opt.num_vectors_per_subj_token

            if opt.background_string is not None:
                config.model.params.use_background_token = True
                personalization_config_params.placeholder_strings       += dataset.background_strings[:1]
                personalization_config_params.list_initializer_words    += dataset.cls_bg_delta_strings[:1]
                personalization_config_params.list_initializer_weights  += dataset.list_bg_initializer_weights[:1]
                personalization_config_params.list_cls_delta_strings    += dataset.cls_bg_delta_strings[:1]
                personalization_config_params.background_strings        = dataset.background_strings[:1]

                for background_string in dataset.background_strings[:1]:
                    personalization_config_params.token2num_vectors[background_string] = opt.num_vectors_per_bg_token

            if opt.wds_comp_db_path is not None:
                # wds_background_strings share the same settings of the background string.
                personalization_config_params.placeholder_strings       += dataset.wds_background_strings[:1]
                personalization_config_params.list_initializer_words    += dataset.cls_bg_delta_strings[:1]
                personalization_config_params.list_initializer_weights  += dataset.list_bg_initializer_weights[:1]
                personalization_config_params.list_cls_delta_strings    += dataset.cls_bg_delta_strings[:1]
                personalization_config_params.background_strings        += dataset.wds_background_strings[:1]

                for wds_background_string in dataset.wds_background_strings[:1]:
                    personalization_config_params.token2num_vectors[wds_background_string] = opt.num_vectors_per_bg_token

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

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

# LightningDataModule: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html
# train: ldm.data.personalized.PersonalizedBase
# validation path is the same as train.
class DataModuleFromConfig(pl.LightningDataModule):
    # train, validation: the corresponding section in the config file,
    # used by instantiate_from_config(self.dataset_configs[k]).
    def __init__(self, batch_size, max_steps, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = max_steps
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn        # False
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # _train_dataloader() is called within prepare_data().
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        
        shuffle = False if is_iterable_dataset else True
        # If there are multiple subjects, we use SubjectSampler to ensure that 
        # each batch contains data from one subject only.
        if self.datasets['train'].num_subjects > 1:
            shuffle = False
            sampler = SubjectSampler(self.datasets['train'].num_subjects, self.num_batches, self.batch_size)
        else:
            sampler = None

        # shuffle=True        
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          shuffle=shuffle, sampler=sampler,
                          num_workers=self.num_workers, 
                          worker_init_fn=init_fn, drop_last=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
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

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
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

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.jpg".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
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
    #   base_learning_rate: float
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
    #      validation:
    #          target: path to validation dataset
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
            basename = os.path.basename(os.path.normpath(opt.data_roots[0]))
            # If we do multi-subject training, we need to replace the * with "all".
            basename = basename.replace("*", "all")
            timesig  = basename + timesig
            
        nowname = timesig + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = True
    
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
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # Data config
        config.data.params.train.params.subject_string       = opt.subject_string
        config.data.params.validation.params.subject_string  = opt.subject_string
        if hasattr(opt, 'subj_info_filepaths'):
            config.data.params.train.params.subj_info_filepaths      = opt.subj_info_filepaths
            config.data.params.validation.params.subj_info_filepaths = opt.subj_info_filepaths

        # common_placeholder_prefix, compos_placeholder_prefix
        config.data.params.train.params.common_placeholder_prefix       = opt.common_placeholder_prefix
        config.data.params.validation.params.common_placeholder_prefix  = opt.common_placeholder_prefix
        config.data.params.train.params.compos_placeholder_prefix       = opt.compos_placeholder_prefix
        config.data.params.validation.params.compos_placeholder_prefix  = opt.compos_placeholder_prefix        
        # broad_class
        config.data.params.train.params.broad_class             = opt.broad_class
        config.data.params.validation.params.broad_class        = opt.broad_class
        # If cls_delta_string is specified, use it for the cls_delta_string of the datasets. 
        # Otherwise, use init_string as the cls_delta_string of the datasets.
        config.data.params.train.params.cls_delta_string         = opt.cls_delta_string or opt.init_string
        config.data.params.validation.params.cls_delta_string    = opt.cls_delta_string or opt.init_string

        config.data.params.train.params.num_vectors_per_subj_token           = opt.num_vectors_per_subj_token
        config.data.params.validation.params.num_vectors_per_subj_token      = opt.num_vectors_per_subj_token
        config.data.params.train.params.num_vectors_per_bg_token        = opt.num_vectors_per_bg_token
        config.data.params.validation.params.num_vectors_per_bg_token   = opt.num_vectors_per_bg_token

        config.data.params.train.params.wds_comp_db_path                   = opt.wds_comp_db_path

        if opt.background_string is not None:
            config.data.params.train.params.background_string           = opt.background_string
            config.data.params.train.params.wds_background_string       = opt.wds_background_string
            config.data.params.validation.params.background_string      = opt.background_string
            config.data.params.validation.params.wds_background_string  = opt.wds_background_string
            config.data.params.train.params.cls_bg_delta_string         = opt.bg_init_words
            config.data.params.validation.params.cls_bg_delta_string    = opt.bg_init_words

        config.data.params.train.params.num_compositions_per_image = opt.num_compositions_per_image
        config.data.params.train.params.rand_scale_range = opt.rand_scale_range
        
        # config.data:
        # {'target': 'main.DataModuleFromConfig', 'params': {'batch_size': 2, 'num_workers': 2, 
        #  'wrap': False, 'train': {'target': 'ldm.data.personalized.PersonalizedBase', 
        #  'params': {'size': 512, 'set': 'train', 'repeats': 100, 
        #  'subject_string': 'z', 'data_roots': 'data/spikelee/'}}, 
        #  'validation': {'target': 'ldm.data.personalized.PersonalizedBase', 
        #  'params': {'size': 512, 'set': 'val', 'repeats': 10, 
        #  'subject_string': 'z', 'data_roots': 'data/spikelee/'}}}}
        config.data.params.train.params.data_roots       = opt.data_roots
        config.data.params.validation.params.data_roots  = opt.data_roots
        # max_steps: Used to initialize DataModuleFromConfig.
        config.data.params.max_steps = opt.max_steps

        # zero-shot settings.
        config.model.params.do_zero_shot = opt.zeroshot
        config.model.params.personalization_config.params.do_zero_shot = opt.zeroshot
        config.data.params.train.params.do_zero_shot        = opt.zeroshot
        config.data.params.validation.params.do_zero_shot   = opt.zeroshot

        if opt.zeroshot:
            gpus = opt.gpus.strip(",").split(',')
            # TODO: put clip image encoder on the same device as the model
            device = f"cuda:{gpus[0]}" if len(gpus) > 0 else "cpu"
            zs_image_emb_dim = init_zero_shot_image_encoders(opt.zs_clip_type, device)
            config.model.params.personalization_config.params.zs_image_emb_dim = zs_image_emb_dim
            config.model.params.personalization_config.params.emb_ema_as_pooling_probe_weight = 0
            
        # data: DataModuleFromConfig
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # DDPM model config
        config.model.params.cond_stage_config.params.last_layers_skip_weights    = opt.clip_last_layers_skip_weights
        config.model.params.cond_stage_config.params.randomize_clip_skip_weights = opt.randomize_clip_skip_weights
        config.model.params.use_fp_trick = opt.use_fp_trick

        if opt.static_embedding_reg_weight >= 0:
            config.model.params.static_embedding_reg_weight = opt.static_embedding_reg_weight
        if opt.ada_embedding_reg_weight >= 0:
            config.model.params.ada_embedding_reg_weight = opt.ada_embedding_reg_weight

        # Setting prompt_emb_delta_reg_weight to 0 will disable prompt delta regularization.
        if opt.prompt_emb_delta_reg_weight >= 0:
            config.model.params.prompt_emb_delta_reg_weight = opt.prompt_emb_delta_reg_weight

        if opt.padding_embs_align_loss_weight >= 0:
            config.model.params.padding_embs_align_loss_weight = opt.padding_embs_align_loss_weight
        if opt.comp_fg_bg_preserve_loss_weight >= 0:
            config.model.params.comp_fg_bg_preserve_loss_weight     = opt.comp_fg_bg_preserve_loss_weight
        if opt.mix_prompt_distill_weight >= 0:
            config.model.params.mix_prompt_distill_weight           = opt.mix_prompt_distill_weight

        if hasattr(opt, 'composition_regs_iter_gaps'):
            config.model.params.composition_regs_iter_gaps = opt.composition_regs_iter_gaps

        if hasattr(opt, 'optimizer_type'):
            config.model.params.optimizer_type = opt.optimizer_type

        if hasattr(opt, 'warmup_steps'):
            if config.model.params.optimizer_type == 'Prodigy':
                config.model.params.prodigy_config.warm_up_steps                       = opt.warmup_steps
            else:
                config.model.params.adam_config.scheduler_config.params.warm_up_steps  = opt.warmup_steps

        if hasattr(opt, 'd_coef'):
            config.model.params.prodigy_config.d_coef = opt.d_coef

        if opt.lr > 0:
            config.model.base_learning_rate = opt.lr

        if opt.max_steps > 0:
            trainer_opt.max_steps = opt.max_steps

        # Personalization config
        config.model.params.personalization_config.params.layerwise_lora_rank = opt.layerwise_lora_rank

        if hasattr(opt, 'attn_pooler_feat_reduction_ratio'):
            config.model.params.personalization_config.params.attn_pooler_feat_reduction_ratio \
                = opt.attn_pooler_feat_reduction_ratio
            
        if opt.ada_emb_weight != -1:
            config.model.params.personalization_config.params.ada_emb_weight = opt.ada_emb_weight

        if opt.use_conv_attn_kernel_size is not None and opt.use_conv_attn_kernel_size > 0:
            K = opt.use_conv_attn_kernel_size
            assert opt.num_vectors_per_subj_token >= K * K, \
                    f"--num_vectors_per_subj_token {opt.num_vectors_per_subj_token} should be at least {K*K}"
            config.model.params.personalization_config.params.use_conv_attn_kernel_size \
                = opt.use_conv_attn_kernel_size

        config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
        config.model.params.personalization_config.params.src_placeholders = opt.src_placeholders
        config.model.params.personalization_config.params.loaded_embedder_components      = opt.loaded_embedder_components
        config.model.params.personalization_config.params.frozen_placeholder_set = opt.frozen_placeholder_set
        config.model.params.personalization_config.params.frozen_embedder_components      = opt.frozen_embedder_components
        config.model.params.personalization_config.params.ckpt_params_perturb_ratio  = opt.ckpt_params_perturb_ratio
        config.model.params.personalization_config.params.emb_reg_loss_scale = opt.emb_reg_loss_scale

        set_placeholders_info(config.model.params.personalization_config.params, opt, data.datasets['train'])

        if opt.actual_resume:
            model = load_model_from_config(config, opt.actual_resume)
        else:
            model = instantiate_from_config(config.model)
        # model: ldm.models.diffusion.ddpm.LatentDiffusion, inherits from LightningModule.
        # model.cond_stage_model: FrozenCLIPEmbedder = text_embedder
        # Extend the token embeddings in CLIP text encoder for the new cls strings.
        # model is still on CPU. So no need to consider where extended_token_embeddings is located.
        if model.embedding_manager.extended_token_embeddings is not None:
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding = \
                extend_nn_embedding(model.cond_stage_model.transformer.text_model.embeddings.token_embedding, 
                                    model.embedding_manager.extended_token_embeddings)
            model.embedding_manager.extended_token_embeddings = None

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
                "target": "pytorch_lightning.loggers.TestTubeLogger",
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

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

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
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
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
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                    }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["max_steps"] = trainer_opt.max_steps
        if hasattr(trainer_opt, 'grad_clip'):
            trainer_kwargs["gradient_clip_val"] = trainer_opt.grad_clip
        
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # configure learning rate
        bs, base_lr, weight_decay = config.data.params.batch_size, config.model.base_learning_rate, \
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

        # model.create_clip_evaluator(f"cuda:{trainer.root_gpu}")

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
