""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
import keras.backend as K
import tensorflow as tf
import sys


def binary_crossentropy(y_true, y_pred):
    y_true_reshaped = K.flatten(y_true)
    y_pred_reshaped = K.flatten(y_pred)

    return K.binary_crossentropy(y_pred_reshaped, y_true_reshaped)


def binary_crossentropy_excluding_first_label(y_true, y_pred):
    y_true = y_true[:, :, :, 1:]
    y_pred = y_pred[:, :, :, 1:]

    y_true_reshaped = K.flatten(y_true)
    y_pred_reshaped = K.flatten(y_pred)

    return K.binary_crossentropy(y_pred_reshaped, y_true_reshaped)
