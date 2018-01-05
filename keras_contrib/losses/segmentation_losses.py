""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
from keras import losses
import keras.backend as K
import tensorflow as tf
import sys


def _end_mean(x, axis=-1):
    """ Same as K.mean, but defaults to the final axis.
    """
    return K.mean(x, axis=axis)


def _loss_2d_adaptor(y_true, y_pred, loss=None, summary=_end_mean, **kwargs):
    """ Adapt a one dimensional loss function to work with 2d segmentation data.
    """
    if loss is None:
        raise ValueError("You must provide a loss function such as binary_crossentropy")
    pred_shape = K.int_shape(y_pred)
    true_shape = K.int_shape(y_true)
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    result = loss(y_pred_reshaped, y_true_reshaped, **kwargs)

    if summary is not None:
        result = summary(result)

    if len(true_shape) >= 3:
        return K.reshape(result, true_shape[:-1])
    else:
        return result


def binary_crossentropy(y_true, y_pred, from_logits=False):
    """ Same as keras.losses.binary_crossentropy for 2d label data.
    """
    return _loss_2d_adaptor(y_true, y_pred, loss=K.binary_crossentropy,
                            summary=_end_mean, from_logits=from_logits)


def mean_squared_error(y_true, y_pred):
    """ Same as keras.losses.mean_squared_error for 2d label data.
    """
    return _loss_2d_adaptor(y_true, y_pred, loss=losses.mean_squared_error, summary=_end_mean)


def mean_absolute_error(y_true, y_pred):
    """ Same as keras.losses.mean_absolute_error for 2d label data.
    """
    return _loss_2d_adaptor(y_true, y_pred, loss=losses.mean_absolute_error, summary=_end_mean)
