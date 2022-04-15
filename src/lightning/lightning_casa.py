
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import copy
import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from src.evaluation.event_completion import EventCompletion
from src.evaluation.classification import Classification
from src.evaluation.kendalls_tau import KendallsTau
# from src.utils.classification import Classification
import os

from src.casa.utils.matching import dual_softmax, dual_bicross

from src.casa import CASA
from src.casa.utils.supervision import compute_supervision_coarse
from src.losses.casa_loss import CASALoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.plotting import vis_conf_matrix
from src.utils.misc import lower_config
from src.utils.profiler import PassThroughProfiler


class PL_CASA(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()

        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.casa_cfg = lower_config(_config['casa'])
        # CLASSIFICATION.ACC_LIST
        self.acc_list = _config['classification']['acc_list']
        self.profiler = profiler or PassThroughProfiler()
        # print("_config",_config)
        self.vis_conf_train = _config['casa']['match']['vis_conf_train']
        self.vis_conf_val = _config['casa']['match']['vis_conf_validation']
        # Matcher: CASA
        self.matcher = CASA(config=_config['casa'])
        self.loss = CASALoss(_config)
        self.temperature = _config['casa']['match']['dsmax_temperature']
        self.thr = _config['casa']['match']['thr']
        self.match_type = _config['casa']['match']['match_type']
        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')[
                'state_dict']
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir
        self.classification = Classification(self.config)
        self.eventcompletion = EventCompletion(self.config)
        self.kendallstau = KendallsTau(self.config)

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(
                    f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        # print("batch.shape",batch)
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("CASA"):
            self.matcher(batch)


        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _valtest_inference(self, batch):
        with self.profiler.profile("CASA"):
            self.matcher(batch, False)

    def _test_inference(self, batch):

        with self.profiler.profile("CASA"):
            self.matcher(batch, True) 


    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            # compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            # compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['keypoints0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)]}
            # 'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def on_train_epoch_start(self):
        pass


    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
        if self.vis_conf_train:
            target_value = 0
            for ii in range(len(outputs)):
                index_id = (outputs[ii]['pair_id'] ==
                            target_value).nonzero(as_tuple=True)
                # print("index_id[0]",index_id[0])
                if len(index_id[0]):
                    batch_num = ii
                    index_in_batch = index_id
            conf_matrix = outputs[batch_num]['conf_matrix'][index_in_batch].cpu().detach().numpy()[
                0]
            # print("conf_matrix",np.shape(conf_matrix))
            if not os.path.exists("{}/vis".format(self.logger.log_dir)):
                os.mkdir("{}/vis".format(self.logger.log_dir))
            if not os.path.exists("{}/vis/conf_matrix_train".format(self.logger.log_dir)):
                os.mkdir("{}/vis/conf_matrix_train".format(self.logger.log_dir))
            save_path = "{0}/vis/conf_matrix_train/{1:06d}.png".format(
                self.logger.log_dir, self.current_epoch)
            vis_conf_matrix(conf_matrix, save_path)

    def validation_step(self, batch, batch_idx):
        self._valtest_inference(batch)


        return {'emb0': batch['emb0'], 'emb1': batch['emb0'], 'len0': batch['len0'], 'len1': batch['len0'], 'label0': batch['label0'],
                'label1': batch['label0'], 'mode': batch['mode'], 'pair_names': batch['pair_names']}


    def predict_step(self, batch, batch_idx):
        self._valtest_inference(batch)
        return {'emb0': batch['emb0'], 'label0': batch['label0'], 'steps0': batch['steps0'], 'len0': batch['len0'],
                'mode': batch['mode'], 'keypoints0': batch['keypoints0'], 'pair_names': batch['pair_names']}

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets

        MEAN_EMB = True
        if MEAN_EMB:
            train_embs = {}
            val_embs = {}
            train_labels = {}
            val_labels = {}
        else:
            train_embs = []
            val_embs = []
            train_labels = []
            val_labels = []
        # print("outputs",outputs)

        for output in outputs:
            emb0 = output['emb0'].cpu().detach().numpy()
            emb1 = output['emb1'].cpu().detach().numpy()
            label0 = output['label0'].cpu().detach().numpy()
            label1 = output['label1'].cpu().detach().numpy()
            len0 = output['len0'].cpu().detach().numpy()
            len1 = output['len1'].cpu().detach().numpy()

            len_data = len(output['pair_names'])
            # print("len_data",len_data)

            for ii in range(len_data):

                if self.config.DATASET.NAME == "kallax_shelf_drawer":
                    key1 = output['pair_names'][ii]
                    key2 = -1
                else:
                    # key1 = int(output['pair_names'][ii])
                    key1 = output['pair_names'][ii]
                    key2 = -1
                if MEAN_EMB:

                    if output['mode'][ii] == 'train':
                        if key1 in train_embs.keys():
                            train_embs[key1].append(emb0[ii][:len0[ii]])
                            train_labels[key1].append(
                                label0[ii][:len0[ii]])
                        else:
                            train_embs[key1] = [emb0[ii][:len0[ii]]]
                            train_labels[key1] = [label0[ii][:len0[ii]]]

                        if key2 in train_embs.keys():
                            train_embs[key2].append(emb1[ii][:len1[ii]])
                            train_labels[key2].append(
                                label1[ii][:len1[ii]])
                        elif key2 is not -1:
                            train_embs[key2] = [emb1[ii][:len1[ii]]]
                            train_labels[key2] = [label1[ii][:len1[ii]]]

                    elif output['mode'][ii] == 'val':
                        if key1 in val_embs.keys():
                            val_embs[key1].append(emb0[ii][:len0[ii]])
                            val_labels[key1].append(label0[ii][:len0[ii]])
                        else:
                            val_embs[key1] = [emb0[ii][:len0[ii]]]
                            val_labels[key1] = [label0[ii][:len0[ii]]]

                        if key2 in val_embs.keys():
                            val_embs[key2].append(emb1[ii][:len1[ii]])
                            val_labels[key2].append(label1[ii][:len1[ii]])
                        elif key2 is not -1:
                            val_embs[key2] = [emb1[ii][:len1[ii]]]
                            val_labels[key2] = [label1[ii][:len1[ii]]]

                else:
                    if output['mode'][ii] == 'train':
                        # print("train_hi")
                        train_embs.append(emb0[ii])
                        train_labels.append(label0[ii])
                        train_embs.append(emb1[ii])
                        train_labels.append(label1[ii])
                    elif output['mode'][ii] == 'val':
                        val_embs.append(emb0[ii])
                        val_labels.append(label0[ii])
                        val_embs.append(emb1[ii])
                        val_labels.append(label1[ii])
        # print()
        datasets = {'train_dataset': {'embs': train_embs, 'labels': train_labels},
                    'val_dataset': {'embs': val_embs, 'labels': val_labels}}
        print(self.profiler.summary())
        # datasets_event = copy.deepcopy(datasets)
        (train_accs, val_accs) = self.classification.evaluate_embeddings(
            datasets, emb_mean=False, DICT=True, acc_list=self.acc_list)
        if self.config.EVAL.EVENT_COMPLETION:
            train_completion_score, val_completion_score = self.eventcompletion.evaluate_embeddings(
                datasets, emb_mean=False, DICT=True)

        if self.config.EVAL.KENDALLS_TAU:
            datasets_pair = {}
            train_dataset = []
            val_dataset = []
            datasets_pair['train_dataset'] = {}
            datasets_pair['val_dataset'] = {}
            for output in outputs:
                # print("output['mode']", output['mode'][0])
                # print("output['pair_names'][",output['pair_names'])
                emb0 = output['emb0'].cpu().detach().numpy()
                emb1 = output['emb1'].cpu().detach().numpy()
                label0 = output['label0'].cpu().detach().numpy()
                label1 = output['label1'].cpu().detach().numpy()
                len0 = output['len0'].cpu().detach().numpy()
                len1 = output['len1'].cpu().detach().numpy()
                len_data = len(output['pair_names'])
                for ii in range(len_data):
                    if output['mode'][ii] == 'train':
                        train_dataset.append(
                            [emb0[ii][:len0[ii]], emb1[ii][:len1[ii]]])
                    else:
                        val_dataset.append(
                            [emb0[ii][:len0[ii]], emb1[ii][:len1[ii]]])
            datasets_pair['train_dataset']['embs'] = train_dataset
            datasets_pair['val_dataset']['embs'] = val_dataset


            (train_tau, val_tau) = self.kendallstau.evaluate_embeddings(
                datasets)
            # print("(train_tau, val_tau)", (train_tau, val_tau))

        if val_accs != 0:
            acc_list = self.acc_list
            for ii, train_acc in enumerate(train_accs):
                self.logger.experiment.add_scalar('classification/train_{}_accuracy'.format(acc_list[ii]),
                                                  train_acc, global_step=self.current_epoch)
            for ii, val_acc in enumerate(val_accs):
                self.logger.experiment.add_scalar('classification/val_{}_accuracy'.format(acc_list[ii]),
                                                  val_acc, global_step=self.current_epoch)
            if self.config.EVAL.KENDALLS_TAU:
                self.logger.experiment.add_scalar(
                    'kendalls_tau/train', train_tau, global_step=self.current_epoch)
                self.logger.experiment.add_scalar(
                    'kendalls_tau/val', val_tau, global_step=self.current_epoch)
            if self.config.EVAL.EVENT_COMPLETION:
                self.logger.experiment.add_scalar(
                    'event_progress/train', train_completion_score, global_step=self.current_epoch)
                self.logger.experiment.add_scalar(
                    'event_progress/val', val_completion_score, global_step=self.current_epoch)

        # Visualize matrix
        if self.vis_conf_val:
            # Calculate conf_matrix for visualization
            if self.match_type == "dual_softmax":
                conf_matrix, _, _ = dual_softmax(torch.tensor(np.expand_dims(datasets['train_dataset']['embs'][0], axis=0)),
                                                 torch.tensor(np.expand_dims(
                                                     datasets['train_dataset']['embs'][1], axis=0)), self.temperature)
            elif self.match_type == "dual_bicross":
                conf_matrix = dual_bicross(torch.tensor(np.expand_dims(datasets['train_dataset']['embs'][0], axis=0)),
                                           torch.tensor(np.expand_dims(
                                               datasets['train_dataset']['embs'][1], axis=0)))
            # make folders
            if not os.path.exists("{}/vis".format(self.logger.log_dir)):
                os.mkdir("{}/vis".format(self.logger.log_dir))
            if not os.path.exists("{}/vis/conf_matrix_val".format(self.logger.log_dir)):
                os.mkdir("{}/vis/conf_matrix_val".format(self.logger.log_dir))

            save_path = "{0}/vis/conf_matrix_val/{1:06d}.png".format(
                self.logger.log_dir, self.current_epoch)

            vis_conf_matrix(conf_matrix.cpu().detach().numpy()[0], save_path)

            mask = conf_matrix > 0.0

            if self.match_type == "dual_softmax":
                i_mask = conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]
                j_mask = conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]
                mask = mask * i_mask * j_mask
            elif self.match_type == "dual_bicross":
                thres = 0.5
                mask = conf_matrix > thres

            if not os.path.exists("{}/vis/mask_train".format(self.logger.log_dir)):
                os.mkdir("{}/vis/mask_train".format(self.logger.log_dir))
            save_path_mask = "{0}/vis/mask_train/{1:06d}.png".format(self.logger.log_dir,
                                                                     self.current_epoch)

            vis_conf_matrix(mask.cpu().detach().numpy()[
                            0].astype(float), save_path_mask)

            # Todo: create matching from both videos and visualize it.

    def test_step(self, batch, batch_idx):
        self._test_inference(batch)
        return {'emb0': batch['emb0'], 'emb1': batch['emb1'], 'len0': batch['len0'], 'len1': batch['len1'], 'label0': batch['label0'],
                'label1': batch['label1'], 'mode': batch['mode'], 'pair_names': batch['pair_names']}

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        MEAN_EMB = True
        if MEAN_EMB:
            train_embs = {}
            val_embs = {}
            train_labels = {}
            val_labels = {}
        else:
            train_embs = []
            val_embs = []
            train_labels = []
            val_labels = []

        for output in outputs:
            emb0 = output['emb0'].cpu().detach().numpy()
            emb1 = output['emb1'].cpu().detach().numpy()
            label0 = output['label0'].cpu().detach().numpy()
            label1 = output['label1'].cpu().detach().numpy()
            len0 = output['len0'].cpu().detach().numpy()
            len1 = output['len1'].cpu().detach().numpy()

            len_data = len(output['pair_names'])
            # print("len_data",len_data)

            for ii in range(len_data):

                # key1 = int(output['pair_names'][ii])
                key1 = output['pair_names'][ii]
                key2 = -1
                if MEAN_EMB:
                    if output['mode'][ii] == 'train':
                        if key1 in train_embs.keys():
                            train_embs[key1].append(emb0[ii][:len0[ii]])
                            train_labels[key1].append(
                                label0[ii][:len0[ii]])
                        else:
                            train_embs[key1] = [emb0[ii][:len0[ii]]]
                            train_labels[key1] = [label0[ii][:len0[ii]]]

                        if key2 in train_embs.keys():
                            train_embs[key2].append(emb1[ii][:len1[ii]])
                            train_labels[key2].append(
                                label1[ii][:len1[ii]])
                        elif key2 is not -1:
                            train_embs[key2] = [emb1[ii][:len1[ii]]]
                            train_labels[key2] = [label1[ii][:len1[ii]]]

                    elif output['mode'][ii] == 'val':
                        if key1 in val_embs.keys():
                            val_embs[key1].append(emb0[ii][:len0[ii]])
                            val_labels[key1].append(label0[ii][:len0[ii]])
                        else:
                            val_embs[key1] = [emb0[ii][:len0[ii]]]
                            val_labels[key1] = [label0[ii][:len0[ii]]]

                        if key2 in val_embs.keys():
                            val_embs[key2].append(emb1[ii][:len1[ii]])
                            val_labels[key2].append(label1[ii][:len1[ii]])
                        elif key2 is not -1:
                            val_embs[key2] = [emb1[ii][:len1[ii]]]
                            val_labels[key2] = [label1[ii][:len1[ii]]]
            datasets = {'train_dataset': {'embs': train_embs, 'labels': train_labels},
                        'val_dataset': {'embs': val_embs, 'labels': val_labels}}
        datasets_pair = {}
        train_dataset = []
        val_dataset = []
        datasets_pair['train_dataset'] = {}
        datasets_pair['val_dataset'] = {}

        (train_accs, val_accs) = self.classification.evaluate_embeddings(
            datasets, acc_list=self.acc_list, DICT=True)
        train_completion_score, val_completion_score = self.eventcompletion.evaluate_embeddings(
            datasets, emb_mean=False, DICT=True)
        (train_tau, val_tau) = self.kendallstau.evaluate_embeddings(
            datasets)
