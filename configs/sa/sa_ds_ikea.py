from pickle import TRUE
from src.config.default import _CN as cfg

# 'dual_softmax'  # 'dual_bicross'
cfg.CASA.MATCH.MATCH_TYPE = 'dual_softmax'
cfg.CASA.MATCH.MATCH_ALGO = None

cfg.DATASET.MAX_LENGTH = 250
cfg.DATASET.VAL_BATCH_SIZE = 256

cfg.EVAL.KENDALLS_TAU = True
cfg.EVAL.KENDALLS_TAU_STRIDE = 2  # 5 for Pouring, 2 for PennAction
cfg.EVAL.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean
cfg.EVAL.EVENT_COMPLETION = True

#cfg.DATASET.NUM_STEPS = 1

# cfg.TRAINER.SCALING = None  # this will be calculated automatically
cfg.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning
cfg.CASA.MATCH.NHEAD = 17  # ikea 17 or 3
cfg.CASA.MATCH.D_MODEL = 51  # dimension of the data, ikea_asm 51 pennaction 75
cfg.TRAINER.CANONICAL_LR = 3e-2  # 3e-3  # 1e-3 0.00005#
cfg.CONSTRASTIVE.TRAIN = True
cfg.CASA.MATCH.USE_PRIOR = True
cfg.DATASET.ATT_STYLE = True


cfg.CASA.FCL.INITIAL_DIM = 51  # For pose of IKEA ASM => 51, PENN ACTION  =>75
# For pose of IKEA ASM => 51, PENN ACTION  =>75
cfg.CASA.PH.OUTPUT_DIM = cfg.CASA.PH.INPUT_DIM = 51
cfg.CASA.PH.HIDDEN_DIM = 51

cfg.TRAINER.OPTIMIZER = "adam"  # adamw

cfg.DATASET.NUM_FRAMES = 20  # 20
cfg.CLASSIFICATION.ACC_LIST = [0.1, 0.5, 1.0]

cfg.EVAL.KENDALLS_TAU = False
cfg.EVAL.EVENT_COMPLETION = False

# Parameters from TCC,
# number of frames that will be embedded jointly, #2 for conv_embedder 1 for casa
cfg.DATASET.NUM_STEPS = 1
cfg.DATASET.FRAME_STRIDE = 15  # stride between context frames

cfg.DATASET.NAME = "kallax_shelf_drawer"  # 20
cfg.DATASET.SMPL = False
cfg.DATASET.USE_NORM = True
cfg.CASA.MATCH.VIS_CONF_TRAIN = False
cfg.CASA.MATCH.VIS_CONF_VALIDATION = False
cfg.TRAINER.WARMUP_STEP = 0
cfg.TRAINER.SCHEDULER_INTERVAL = 'epoch'
# [20, 40, 60, 80, 100, 120, 140]  # MSLR: MultiStepLR[70, 100, 120, 150]  #
cfg.TRAINER.MSLR_MILESTONES = [30, 40, 50, 60, 70, 80, 90, 100]
# cfg.CASA.MATCH.VIS_CONF_TRAIN
# ['fast','noise_vposer', 'noise_translation' 'noise_angle','flip']
cfg.CONSTRASTIVE.AUGMENTATION_STRATEGY = ['fast']
#'noise_vposer' 'translation', 'scale', 'shuffle', 'crop', 'rotation','flip' 'center' 'noise_translation' 'noise_angle' ,'fast'


cfg.CASA.PH.TRUE = True  # projection head
cfg.CASA.MATCH.PE = True  # positional encoding
cfg.CASA.MATCH.LAYER_NAMES = ['self', 'cross'] * 4
cfg.CASA.MATCH.SIMILARITY = True

#cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]


# TCC embbed network parameters
# List of conv layers defined as (channels, kernel_size, activate).
cfg.CASA.EMBEDDER_TYPE = 'casa'  # casa conv_embedder

cfg.CASA.NUM_FRAMES = cfg.DATASET.NUM_FRAMES

# classification regression regression_var
cfg.CASA.LOSS.LOSS_TYPE = 'regression'
