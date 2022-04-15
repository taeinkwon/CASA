from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  CASA Pipeline  ↓  ##############
_CN.CASA = CN()
_CN.CASA.BACKBONE_TYPE = 'FCL'
_CN.CASA.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.CASA.NUM_FRAMES = 20
# 1. CASA-backbone (local feature CNN) config
_CN.CASA.RESNETFPN = CN()
_CN.CASA.RESNETFPN.INITIAL_DIM = 128
_CN.CASA.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3
# FCL config
_CN.CASA.FCL = CN()
_CN.CASA.FCL.INITIAL_DIM = 75  # For pose of IKEA ASM => 51, PENN ACTION  =>75
# Projection Head config
_CN.CASA.PH = CN()
_CN.CASA.PH.TRUE = False
_CN.CASA.PH.BACKBONE_TYPE = 'PH'
# For pose of IKEA ASM => 51, PENN ACTION  =>75
_CN.CASA.PH.OUTPUT_DIM = _CN.CASA.PH.INPUT_DIM = 75
_CN.CASA.PH.HIDDEN_DIM = 75  # For pose of IKEA ASM => 51, PENN ACTION  =>75

# 2. CASA module config
_CN.CASA.MATCH = CN()
_CN.CASA.MATCH.PE = True  # should be enven number  #256
_CN.CASA.MATCH.D_MODEL = 75  # should be enven number  #256
_CN.CASA.MATCH.D_FFN = 256
_CN.CASA.MATCH.NHEAD = 5  # 8
_CN.CASA.MATCH.LAYER_NAMES = ['self', 'cross'] * 4
_CN.CASA.MATCH.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.CASA.MATCH.TEMP_BUG_FIX = True
_CN.CASA.MATCH.VIS_CONF_TRAIN = False
_CN.CASA.MATCH.VIS_CONF_VALIDATION = True
_CN.CASA.MATCH.USE_PRIOR = False
_CN.CASA.MATCH.THR = 0.1
_CN.CASA.MATCH.BORDER_RM = 2
# options: ['dual_softmax, 'bicross']
_CN.CASA.MATCH.MATCH_TYPE = 'dual_softmax'
_CN.CASA.MATCH.MATCH_ALGO = None
_CN.CASA.MATCH.DSMAX_TEMPERATURE = 0.1
_CN.CASA.MATCH.SIMILARITY = False

# 3. CASA Losses
# -- # coarse-level
_CN.CASA.LOSS = CN()
_CN.CASA.LOSS.TYPE = 'cross_entropy'  # ['focal', 'cross_entropy']
# ['classification', 'regression','regression_var]
_CN.CASA.LOSS.LOSS_TYPE = 'regression'
_CN.CASA.LOSS.WEIGHT = 1.0
# -- - -- # focal loss (coarse)
_CN.CASA.LOSS.FOCAL_ALPHA = 0.25
_CN.CASA.LOSS.FOCAL_GAMMA = 2.0
_CN.CASA.LOSS.POS_WEIGHT = 1.0
_CN.CASA.LOSS.NEG_WEIGHT = 1.0

_CN.CASA.EMBEDDER_TYPE = 'casa'  # casa, conv_embedder

_CN.CONSTRASTIVE = CN()
_CN.CONSTRASTIVE.TRAIN = False
_CN.CONSTRASTIVE.AUGMENTATION_STRATEGY = ['shuffle']

_CN.CLASSIFICATION = CN()
_CN.CLASSIFICATION.ACC_LIST = [0.1, 0.5, 1.0]

##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.NUM_FRAMES = 20
_CN.DATASET.NAME = "name"
_CN.DATASET.SAMPLING_STRATEGY = 'offset_uniform'
_CN.DATASET.FOLDER = "./"
_CN.DATASET.LOGDIR = './logs'

# Parameters from TCC,
_CN.DATASET.NUM_STEPS = 1  # number of frames that will be embedded jointly,
_CN.DATASET.FRAME_STRIDE = 15  # stride between context frames

_CN.DATASET.MAX_LENGTH = 250
_CN.DATASET.ATT_STYLE = False
_CN.DATASET.USE_NORM = True
_CN.DATASET.MANO = False
_CN.DATASET.SMPL = False
_CN.DATASET.TRAINVAL_DATA_SOURCE = None
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
# None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_LIST_PATH = None
_CN.DATASET.VAL_BATCH_SIZE = 1
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
# None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_LIST_PATH = None

# 2. dataset config
# general options
_CN.DATASET.AUGMENTATION_TYPE = None

_CN.EVAL = CN()
_CN.EVAL.EVENT_COMPLETION = False
_CN.EVAL.KENDALLS_TAU = False
_CN.EVAL.KENDALLS_TAU_STRIDE = 2  # 5 for Pouring, 2 for PennAction
_CN.EVAL.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean
##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 1e-3  # 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 100

# learning rate scheduler
# [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER = 'MultiStepLR'
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
# ELR: ExponentialLR, this value for 'step' interval
_CN.TRAINER.ELR_GAMMA = 0.999992

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5
_CN.TRAINER.SEED = 50


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
