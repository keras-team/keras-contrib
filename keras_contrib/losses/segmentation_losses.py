""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
import keras.backend as K
import tensorflow as tf
import sys


def binary_crossentropy(y_true, y_pred):
    pred_shape = K.int_shape(y_pred)
    true_shape = K.int_shape(y_true)
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))
    print('pred_shape: ' + str(pred_shape), 'pred_shape: ' + str(pred_shape), )
    result = K.binary_crossentropy(y_pred_reshaped, y_true_reshaped)
    result = K.mean(result, axis=-1)

    if len(true_shape) >= 3:
        return K.reshape(result, true_shape[:-1])
    else:
        return result