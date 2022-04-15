r"""Evaluation on detecting key events using a RNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import concurrent.futures as cf
from loguru import logger as loguru_logger
import numpy as np
import sklearn
import copy
from dataset_splits import DATASET_TO_NUM_CLASSES

from src.evaluation.task_utils import get_targets_from_labels, unnormalize
#from config import ENVCONFIG
FLAGS = flags.FLAGS


class VectorRegression(sklearn.base.BaseEstimator):
    """Class to perform regression on multiple outputs."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        _, m = y.shape
        # Fit a separate regressor for each column of y
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(x, y[:, i])
                            for i in range(m)]
        return self

    def predict(self, x):
        # Join regressors' predictions
        res = [est.predict(x)[:, np.newaxis] for est in self.estimators_]
        return np.hstack(res)

    def score(self, x, y):
        # Join regressors' scores
        res = [est.score(x, y[:, i]) for i, est in enumerate(self.estimators_)]
        return np.mean(res)


def get_error(predictions, labels, seq_lens, num_classes, prefix):
    """Get error based on predictions."""
    errs = []
    for i in range(num_classes - 1):
        abs_err = 0
        for j in range(len(predictions)):
            # Choose last seq_len steps as our preprocessing pads sequences in
            # front with zeros.
            unnorm_preds = unnormalize(predictions[j][:, i])
            unnorm_labels = unnormalize(labels[j][:, i])

            abs_err += abs(unnorm_labels - unnorm_preds) / seq_lens[j]

        err = abs_err / len(predictions)
        logging.info('{} {} Fraction Error: '
                     '{:.3f},'.format(prefix, i, err))
        # tf.summary.scalar('event_completion/%s_%d_error' % (prefix, i),
        #                  err, step=global_step)
        errs.append(err)

    avg_err = np.mean(errs)

    logging.info(' {} Fraction Error: '
                 '{:.3f},'.format(prefix, avg_err))
    # tf.summary.scalar('event_completion/avg_error_%s' % prefix,
    #                  avg_err, step=global_step)

    return avg_err


def fit_model(train_embs, train_labels, val_embs, val_labels,
              num_classes, prefix, report_error=False):
    """Linear Regression to regress to fraction completed."""

    train_seq_lens = [len(x) for x in train_labels]
    val_seq_lens = [len(x) for x in val_labels]

    train_embs = np.concatenate(train_embs, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_embs = np.concatenate(val_embs, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    lin_model = VectorRegression(sklearn.linear_model.LinearRegression())
    lin_model.fit(train_embs, train_labels)

    train_score = lin_model.score(train_embs, train_labels)
    val_score = lin_model.score(val_embs, val_labels)

    # To debug linear regression
    val_predictions = lin_model.predict(val_embs)
    train_predictions = lin_model.predict(train_embs)

    # print("train_predictions",train_predictions)
    # print("train_labels",train_labels)
    # print("val_predictions",val_predictions)
    # print("val_labels",val_labels)

    # Not used for evaluation right now.
    if report_error:
        val_predictions = lin_model.predict(val_embs)
        train_predictions = lin_model.predict(train_embs)

        train_labels = np.array_split(train_labels,
                                      np.cumsum(train_seq_lens))[:-1]
        train_predictions = np.array_split(train_predictions,
                                           np.cumsum(train_seq_lens))[:-1]
        val_labels = np.array_split(val_labels,
                                    np.cumsum(val_seq_lens))[:-1]
        val_predictions = np.array_split(val_predictions,
                                         np.cumsum(val_seq_lens))[:-1]

        get_error(train_predictions, train_labels, train_seq_lens,
                  num_classes, 'train_' + prefix)
        get_error(val_predictions, val_labels, val_seq_lens,
                  num_classes, 'val_' + prefix)

    return train_score, val_score


class EventCompletion():
    """Predict event completion using linear regression."""

    def __init__(self, config):
        super(EventCompletion, self).__init__()
        self.config = config

    def evaluate_embeddings(self, datasets_ori, DICT=True, emb_mean=True):
        """Labeled evaluation."""

        datasets = copy.deepcopy(datasets_ori)

        num_classes = DATASET_TO_NUM_CLASSES[self.config.DATASET.NAME]  # 4
        # print("num_class",num_classes)

        #DICT = True
        #emb_mean = True

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
                    #    for ii in range(len(emb)):
                    train_emb.append(datasets['train_dataset']['embs'][key][0])
                    train_label.append(
                        datasets['train_dataset']['labels'][key][0])

                for key, emb in datasets['val_dataset']['embs'].items():
                    #    for ii in range(len(emb)):
                    val_emb.append(datasets['val_dataset']['embs'][key][0])
                    val_label.append(datasets['val_dataset']['labels'][key][0])
                datasets['train_dataset']['embs'] = train_emb
                datasets['train_dataset']['labels'] = train_label
                datasets['val_dataset']['embs'] = val_emb
                datasets['val_dataset']['labels'] = val_label

        train_embs = datasets['train_dataset']['embs']
        val_embs = datasets['val_dataset']['embs']

        # print("train_embs",np.size(train_embs))

        if not train_embs or not val_embs:
            logging.warn(
                'All embeddings are NAN. Something is wrong with model.')
            return 1.0

        val_labels = get_targets_from_labels(
            datasets['val_dataset']['labels'],  num_classes)
        train_labels = get_targets_from_labels(
            datasets['train_dataset']['labels'], num_classes)

        #print("train_labels", np.shape(train_labels))
        #print("train_embs", np.shape(train_embs))
        #print("val_labels", val_labels)

        results = fit_model(train_embs, train_labels, val_embs, val_labels,
                            num_classes, '%s_%s' % ('Penn', str(1)))
        train_score, val_score = results

        prefix = '%s_%s' % ('Penn', str(1))
        loguru_logger.info('Event Completion {} Fraction Train '
                           'Score: {:.5f},'.format(prefix,
                                                   train_score))
        loguru_logger.info('Event Completion {} Fraction Val '
                           'Score: {:.5f},'.format(prefix,
                                                   val_score))

        return train_score, val_score
