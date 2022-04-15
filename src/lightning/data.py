import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    ChainDataset,
    RandomSampler,
    dataloader
)

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split, collate_fixed_len
from src.utils.misc import tqdm_joblib
from src.datasets.pennaction import PennActionDataset


class MultiSceneDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()
        # 1. data config
        # Train and Val should from the same data source
        self.config = config
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating

        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_batch_size = config.DATASET.VAL_BATCH_SIZE
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH

        self.sampling_strategy = config.DATASET.SAMPLING_STRATEGY
        self.num_frames = config.DATASET.NUM_FRAMES
        # 2. dataset config
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)
        self.use_norm = config.DATASET.USE_NORM
        
        # 0.125. for training casa.
        self.coarse_scale = 1 / config.CASA.RESOLUTION[0]

        self.contrastive = config.CONSTRASTIVE.TRAIN
        self.augmentation_strategy = config.CONSTRASTIVE.AUGMENTATION_STRATEGY

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': self.val_batch_size,  # 1
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': self.val_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }



        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test',
                         'predict'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit' or stage == 'predict':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                mode='train',
                pose_dir=self.train_pose_root)

            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [
                        self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        mode='val',
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    mode='val',
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')

        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                mode='test',
                pose_dir=self.test_pose_root)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       mode='train',
                       pose_dir=None):
        """ Setup train / val / test set"""

        dataset_builder = self._build_concat_dataset
        local_npz_names = data_root
        return dataset_builder(data_root, local_npz_names, split_npz_root, 
                               mode=mode, pose_dir=pose_dir)

    def _build_concat_dataset(
        self,
        data_root,
        npz_names,
        npz_dir,
        mode,
        pose_dir=None
    ):

        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in [
            'train'] else self.test_data_source

        npz_path = data_root  # osp.join(npz_dir, npz_name)
        # print("npz_path",npz_path)
        if mode in ['train']:
            datasets = []
            if data_source == 'PennAction':
                datasets.append(
                    PennActionDataset(npz_path,
                                      num_frames=self.num_frames,
                                      sampling_strategy=self.sampling_strategy,
                                      mode=mode,
                                      augment_fn=augment_fn,
                                      coarse_scale=self.coarse_scale,
                                      contrastive=self.contrastive,
                                      augmentation_strategy=self.augmentation_strategy,
                                      use_norm=self.use_norm,
                                      config=self.config))
            else:
                raise NotImplementedError()
        else:
            datasets = []

            if data_source == 'PennAction':
                print("npz_path", npz_path)


                
                for npz_path_elem in npz_path:
                    # Append train and validation sets together.
                    datasets.append(PennActionDataset(npz_path_elem,
                                                        num_frames=self.num_frames,
                                                        sampling_strategy=self.sampling_strategy,
                                                        mode=mode,
                                                        augment_fn=augment_fn,
                                                        coarse_scale=self.coarse_scale,
                                                        contrastive=self.contrastive,
                                                        val=True,
                                                        augmentation_strategy=self.augmentation_strategy,
                                                        use_norm=self.use_norm,
                                                        config=self.config))

            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)

    def train_dataloader(self):
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
       
        dataloader = DataLoader(
            self.train_dataset, shuffle=True, **self.train_loader_params)
        print("self.train_dataset", len(self.train_dataset))
        print("dataloader.__len__", len(dataloader.dataset))
        return dataloader

    def val_dataloader(self):
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = DataLoader(
            self.val_dataset, **self.val_loader_params)

        print("self.val_taset", len(self.val_dataset))
        print("dataloader.__len__", len(dataloader.dataset))
        return dataloader

    def predict_dataloader(self):
        """ Build validation dataloader for H2O/Penn Action/IKEA ASM. """
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = DataLoader(
            self.val_dataset, **self.val_loader_params)

        print("predict datset length", len(self.val_dataset))
        print("dataloader.__len__", len(dataloader.dataset))
        return dataloader

    def test_dataloader(self, *args, **kwargs):
        logger.info(
            f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
