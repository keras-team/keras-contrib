""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
import keras.backend as K
import sys


def pixel_accuracy(y_true, y_pred):
    pred_shape = K.int_shape(y_pred)
    true_shape = K.int_shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    correct_pixels_per_class = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32')

    return K.sum(correct_pixels_per_class) / K.cast(K.prod(true_shape), dtype='float32')


def mean_accuracy(y_true, y_pred):
    pred_shape = K.int_shape(y_pred)
    true_shape = K.int_shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    equal_entries = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    correct_pixels_per_class = K.sum(equal_entries, axis=1)
    n_pixels_per_class = K.sum(y_true_reshaped, axis=1)

    # epsilon added to avoid dividing by zero
    acc = correct_pixels_per_class / (n_pixels_per_class + K.epsilon())

    return K.mean(acc)


def mean_intersection_over_union(y_true, y_pred):
    pred_shape = K.int_shape(y_pred)
    true_shape = K.int_shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    equal_entries = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped, axis=1) + K.sum(y_pred_reshaped, axis=1)

    # epsilon added to avoid dividing by zero
    iou = intersection / ((union_per_class - intersection) + K.epsilon())

    return K.mean(iou)
