r"""Evaluation on per-frame labels for classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from loguru import logger as loguru_logger
import pytorch_lightning as pl

import concurrent.futures as cf

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import copy

FLAGS = flags.FLAGS


def fit_linear_model(train_embs, train_labels,
                     val_embs, val_labels):
    """Fit a linear classifier."""
    lin_model = LogisticRegression(max_iter=100000, solver='lbfgs',
                                   multi_class='multinomial', verbose=0)
    lin_model.fit(train_embs, train_labels)
    train_acc = lin_model.score(train_embs, train_labels)
    val_acc = lin_model.score(val_embs, val_labels)
    return lin_model, train_acc, val_acc


def fit_svm_model(train_embs, train_labels,
                  val_embs, val_labels):
    """Fit a SVM classifier."""
    # svm_model = LinearSVC(verbose=0)
    svm_model = SVC(decision_function_shape='ovo', verbose=0)
    svm_model.fit(train_embs, train_labels)
    train_acc = svm_model.score(train_embs, train_labels)
    val_acc = svm_model.score(val_embs, val_labels)
    return svm_model, train_acc, val_acc


def fit_linear_models(train_embs, train_labels, val_embs, val_labels,
                      model_type='linear'):
    """Fit Log Regression and SVM Models."""
    if model_type == 'linear':
        _, train_acc, val_acc = fit_linear_model(train_embs, train_labels,
                                                 val_embs, val_labels)
    elif model_type == 'svm':
        _, train_acc, val_acc = fit_svm_model(train_embs, train_labels,
                                              val_embs, val_labels)
    else:
        raise ValueError('%s model type not supported' % model_type)
    return train_acc, val_acc


class Classification():
    """Classification using small linear models."""

    def __init__(self, config):
        self.config = config

    def evaluate_embeddings(self, datasets_ori, emb_mean=False, DICT=False, acc_list=[0.1, 0.5, 1.0]):
        """Labeled evaluation."""
        fractions = acc_list    # CONFIG.EVAL.CLASSIFICATION_FRACTIONS # [0.1, 0.5, 1.0]
        datasets = copy.deepcopy(datasets_ori)

        if self.config.DATASET.NAME == 'kallax_shelf_drawer':
            BACKGROUND_LABEL = True
        else:
            BACKGROUND_LABEL = False

        if datasets['train_dataset']['embs'] == [] or datasets['val_dataset']['embs'] == []:
            loguru_logger.info(
                'Empty embeddings')
            return (0.0, 0.0)

        if DICT:
            if emb_mean:
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

            else:
                train_emb = []
                train_label = []
                val_emb = []
                val_label = []
                for key, emb in datasets['train_dataset']['embs'].items():
                    train_emb.append(datasets['train_dataset']['embs'][key][0])
                    train_label.append(
                        datasets['train_dataset']['labels'][key][0])

                for key, emb in datasets['val_dataset']['embs'].items():
                    val_emb.append(datasets['val_dataset']['embs'][key][0])
                    val_label.append(datasets['val_dataset']['labels'][key][0])
                datasets['train_dataset']['embs'] = train_emb
                datasets['train_dataset']['labels'] = train_label
                datasets['val_dataset']['embs'] = val_emb
                datasets['val_dataset']['labels'] = val_label

        val_embs = np.concatenate(datasets['val_dataset']['embs'])
        val_labels = np.concatenate(datasets['val_dataset']['labels'])

        if BACKGROUND_LABEL:
            val_embs = val_embs[val_labels.astype(bool)]
            val_labels = val_labels[val_labels.astype(bool)]-1

        val_accs = []
        train_accs = []
        train_dataset = datasets['train_dataset']
        num_samples = len(train_dataset['embs'])

        def worker(fraction_used):
            num_samples_used = max(1, int(fraction_used * num_samples))
            train_embs = np.concatenate(
                train_dataset['embs'][:num_samples_used])
            train_labels = np.concatenate(
                train_dataset['labels'][:num_samples_used])

            if BACKGROUND_LABEL:
                train_embs = train_embs[train_labels.astype(bool)]
                train_labels = train_labels[train_labels.astype(bool)]-1
            return fit_linear_models(train_embs, train_labels, val_embs, val_labels)

        with cf.ThreadPoolExecutor(max_workers=len(fractions)) as executor:
            results = executor.map(worker, fractions)
            for (fraction, (train_acc, val_acc)) in zip(fractions, results):
                loguru_logger.info('[Global step:  Classification {} Fraction'
                                   'Train Accuracy: {:.5f},'.format(fraction, train_acc))
                loguru_logger.info('[Global step: ] Classification {} Fraction'
                                   'Val Accuracy: {:.5f},'.format(fraction,
                                                                  val_acc))
                train_accs.append(train_acc)
                val_accs.append(val_acc)

        return (train_accs, val_accs)
