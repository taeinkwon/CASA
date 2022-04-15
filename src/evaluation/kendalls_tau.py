r"""Evaluation train and val loss using the algo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from loguru import logger as loguru_logger
import copy
#FLAGS = flags.FLAGS


def _get_kendalls_tau(embs_list, stride, tau_dist):
    """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
    num_seqs = len(embs_list)

    taus = np.zeros((num_seqs * (num_seqs - 1)))
    idx = 0
    for i in range(num_seqs):
        query_feats = embs_list[i][::stride]
        for j in range(num_seqs):
            if i == j:
                continue
            candidate_feats = embs_list[j][::stride]
            dists = cdist(query_feats, candidate_feats,
                            tau_dist)
            nns = np.argmin(dists, axis=1)
            taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
            idx += 1

    # Remove NaNs.
    taus = taus[~np.isnan(taus)]
    tau = np.mean(taus)

    # logging.info('Iter[{}/{}] {} set alignment tau: {:.4f}'.format(
    #    global_step.numpy(), CONFIG.TRAIN.MAX_ITERS, split, tau))

    #tf.summary.scalar('kendalls_tau/%s_align_tau' % split, tau, step=global_step)
    return tau


class KendallsTau():
    """Calculate Kendall's Tau."""

    def __init__(self, conf):
        super(KendallsTau, self).__init__()
        self.conf = conf

    def evaluate_embeddings(self, datasets_ori):
        """Labeled evaluation."""

        datasets = copy.deepcopy(datasets_ori)

 
        train_emb = []
        train_label = []
        val_emb = []
        val_label = []

        for key, emb in datasets['train_dataset']['embs'].items():
            train_emb.append(np.average(np.array(emb), axis=0))
            train_label.append(
                datasets['train_dataset']['labels'][key][0])

        for key, emb in datasets['val_dataset']['embs'].items():
            val_emb.append(np.average(np.array(emb), axis=0))
            val_label.append(datasets['val_dataset']['labels'][key][0])

        datasets['train_dataset']['embs'] = train_emb
        datasets['train_dataset']['labels'] = train_label
        datasets['val_dataset']['embs'] = val_emb
        datasets['val_dataset']['labels'] = val_label

        train_embs = datasets['train_dataset']['embs']

        train_tau = _get_kendalls_tau(
            train_embs,
            self.conf.EVAL.KENDALLS_TAU_STRIDE, self.conf.EVAL.KENDALLS_TAU_DISTANCE)

        val_embs = datasets['val_dataset']['embs']

        val_tau = _get_kendalls_tau(
            val_embs, self.conf.EVAL.KENDALLS_TAU_STRIDE, self.conf.EVAL.KENDALLS_TAU_DISTANCE)

        loguru_logger.info('train set alignment tau: {:.5f}'.format(train_tau))
        loguru_logger.info('val set alignment tau: {:.5f}'.format(val_tau))
        return train_tau, val_tau
