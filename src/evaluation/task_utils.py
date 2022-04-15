from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def regression_labels_for_class(labels, class_idx,LAST=False):
      # Assumes labels are ordered. Find the last occurrence of particular class.
  #print("labels",labels)
  #print("class_idx",class_idx)
  #print("np.argwhere(labels == class_idx)",np.argwhere(labels == class_idx))
  if LAST:
    transition_frame = len(labels)
  else:
    transition_frame = np.argwhere(labels == class_idx)[-1, 0]
  return (np.arange(float(len(labels))) - transition_frame) / len(labels)



def get_regression_labels(class_labels, num_classes):
  regression_labels = []
  for i in range(num_classes - 1):
    if i in class_labels:
      regression_labels.append(regression_labels_for_class(class_labels, i))
    else:
      if i == num_classes - 2:
        regression_labels.append(regression_labels_for_class(class_labels, i,LAST=True))
        print("last",regression_labels_for_class(class_labels, i,LAST=True))
      else: 
        regression_labels.append(regression_labels[i-1])
  return np.stack(regression_labels, axis=1)


def get_targets_from_labels(all_class_labels, num_classes):
  all_regression_labels = []
  for class_labels in all_class_labels:
    all_regression_labels.append(get_regression_labels(class_labels,
                                                       num_classes))
  return all_regression_labels


def unnormalize(preds):
  seq_len = len(preds)
  return np.mean([i - pred * seq_len for i, pred in enumerate(preds)])
