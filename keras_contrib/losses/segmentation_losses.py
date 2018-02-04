""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
import sys
from keras import losses
import keras.backend as K
from ..metrics.segmentation_metrics import _end_mean
from ..metrics.segmentation_metrics import _metric_2d_adaptor
from .jaccard import jaccard_distance
from .jaccard import binary_jaccard_distance


def binary_crossentropy(y_true, y_pred, from_logits=False):
    """ Same as keras.losses.binary_crossentropy for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, metric=K.binary_crossentropy,
                              summary=_end_mean, from_logits=from_logits)


def mean_squared_error(y_true, y_pred):
    """ Same as keras.losses.mean_squared_error for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, metric=losses.mean_squared_error, summary=_end_mean)


def mean_absolute_error(y_true, y_pred):
    """ Same as keras.losses.mean_absolute_error for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, metric=losses.mean_absolute_error, summary=_end_mean)


def mean_intersection_over_union(y_true, y_pred):
    """ Same as keras_contrib.losses.jaccard_distance for 2d label data with one-hot channels.
    """
    return _metric_2d_adaptor(y_true, y_pred, metric=jaccard_distance, summary=_end_mean)


def binary_mean_intersection_over_union(y_true, y_pred):
    """ Same as keras_contrib.losses.jaccard_distance for 2d label data with one-hot channels.
    """
    return _metric_2d_adaptor(y_true, y_pred, metric=binary_jaccard_distance, summary=_end_mean)