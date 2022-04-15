from configs.data.base import cfg


TRAIN_BASE_PATH = "npyrecords/"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "PennAction"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}/baseball_pitch_val.npy"
cfg.DATASET.TRAIN_NPZ_ROOT = ""

