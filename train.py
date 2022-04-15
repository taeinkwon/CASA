import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_casa import PL_CASA

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        '--main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained CASA')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--dataset_name', type=str, default=None,
        help='name of the dataset')
    parser.add_argument(
        '--data_folder', type=str, default=None,
        help='data folder path')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments

    
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()

    # Load main and data cfgs.
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)

    # Change the dataset folder and the name of the dataset 
    #config.DATASET.NAME = args.DATASET.NAME
    config.DATASET.LOGDIR = args.data_folder
    config.DATASET.NAME = args.dataset_name
    print("config.DATASET.TRAIN_DATA_ROOT",config.DATASET.TRAIN_DATA_ROOT)
    print("config.DATASET.NAME",config.DATASET.NAME)

    config.DATASET.TRAIN_DATA_ROOT = config.DATASET.TRAIN_DATA_ROOT + "/" + config.DATASET.NAME +"_train.npy"
    config.DATASET.VAL_DATA_ROOT = config.DATASET.TEST_DATA_ROOT = [config.DATASET.VAL_DATA_ROOT[0]  + "/" + config.DATASET.NAME +"_train.npy",
                                                                    config.DATASET.VAL_DATA_ROOT[1]  + "/" + config.DATASET.NAME +"_val.npy" ]    # To reproduce the results,
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / _scaling)

    config.DATASET.PATH = args.data_folder
    

    print("config.TRAINER.TRUE_BATCH_SIZE",config.TRAINER.TRUE_BATCH_SIZE)
    print("_scaling",_scaling)
    print("config.TRAINER.WARMUP_STEP",config.TRAINER.WARMUP_STEP)
    #config_par = configparser.RawConfigParser(allow_no_value=True)
    #config_par.readfp(config)

    # lightning module 
    profiler = build_profiler(args.profiler_name)
    model = PL_CASA(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info("CASA LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("CASA DataModule initialized!")

    # TensorBoard Logger
    save_dir = os.path.join(config.DATASET.LOGDIR,'logs')
    
    logger = TensorBoardLogger(
        save_dir=save_dir, name=os.path.join(config.DATASET.NAME, args.exp_name), default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dataset_log_path = os.path.join(save_dir,config.DATASET.NAME)
    if not os.path.exists(dataset_log_path):
        os.mkdir(dataset_log_path)
    experiment_path = os.path.join(save_dir,config.DATASET.NAME ,args.exp_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    print("experiment_path",experiment_path)


    if not os.path.exists(logger.log_dir):
        os.mkdir(logger.log_dir)

    with open("{}/config.yaml".format(logger.log_dir), "w") as f:
        f.write(config.dump())
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(verbose=True, mode='max',
                                    save_top_k =-1,
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}',
                                    every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer #fast_dev_run=True should be here
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler,
        fast_dev_run=False)
    loguru_logger.info("Trainer initialized!")
    loguru_logger.info("Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
