import yaml 



class DictI(dict):
    def __init__(self):
        self.outdir='outputs'
        self.indiv_subdir='samples'
        self.skip_grid=False
        self.ddim_steps=20
        self.plms=False
        self.skip_save=False
        self.laion400m=False
        self.fixed_code=False
        self.ddim_eta=0.0
        self.n_repeat=1
        self.H=512
        self.W=512
        self.C=4
        self.f=8
        self.n_samples=4
        self.bs=8
        self.n_rows=0
        self.scale=10
        self.from_file=''
        self.config='configs/stable-diffusion/v1-inference-ada.yaml'
        self.ckpt ='models/stable-diffusion/v1-5-dste.ckpt'
        self.seed=42
        self.precision='autocast'
        self.embedding_paths=None
        self.subj_scale=1.0
        self.ada_emb_weight=-1
        self.init_img=None
        self.mask_weight=0.0
        self.no_preview=False
        self.broad_class=1
        self.calc_face_sim=False
        self.gpu=0
        self.compare_with=None
        self.class_prompt=None
        self.clip_last_layers_skip_weights=[0.5]
        self.debug = False

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


class DictT(dict):
    def __init__(self):
         self.name = ""
         self.resume = ""
         self.base = list()
         self.train = False
         self.no_test = False
         self.project = ""
         self.debug = False
         self.seed = 23
         self.postfix =""
         self.logdir ="logs"
         self.lr = "-1"
         self.scale_lr = True
         self.datadir_in_name = False
         self.data_root = ""
         self.actural_resume = ""
         self.embedding_manager_ckpt = ""
         self.placeholder_string = ""
         self.init_word = ""
         self.init_word_weights = 0.5
         self.init_neg_words = None
         self.cls_delta_token = None
         self.layerwise_lora_rank_token_ratio = -1
         self.embedding_reg_weight = -1
         self.ada_emb_weight = -1
         self.composition_delta_reg_weight = -1
         self.min_rand_scaling = 0.8
         self.max_rand_scaling = 1.05
         self.num_compositions_per_image = 1
         self.broad_class = 1
         self.clip_last_layers_skip_weights = [0.5]
         self.no_wandb = True

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'







