""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
import keras.backend as K
import tensorflow as tf
import sys


def pixel_accuracy(y_true, y_pred):
    s = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape(y_true, tf.stack([-1, s[1]*s[2], s[-1]]))
    y_pred_reshaped = K.reshape(y_pred, tf.stack([-1, s[1]*s[2], s[-1]]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), nb_classes=s[-1])
    correct_pixels_per_class = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32')

    return K.sum(correct_pixels_per_class) / K.cast(K.prod(s), dtype='float32')


def mean_accuracy(y_true, y_pred):
    s = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape(y_true, tf.stack([-1, s[1]*s[2], s[-1]]))
    y_pred_reshaped = K.reshape(y_pred, tf.stack([-1, s[1]*s[2], s[-1]]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), nb_classes=s[-1])
    equal_entries = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    correct_pixels_per_class = K.sum(equal_entries, axis=1)
    n_pixels_per_class = K.sum(y_true_reshaped, axis=1)

    acc = correct_pixels_per_class / n_pixels_per_class
    acc_mask = tf.is_finite(acc)
    acc_masked = tf.boolean_mask(acc, acc_mask)

    return K.mean(acc_masked)


def mean_intersection_over_union(y_true, y_pred):
    s = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape(y_true, tf.stack([-1, s[1]*s[2], s[-1]]))
    y_pred_reshaped = K.reshape(y_pred, tf.stack([-1, s[1]*s[2], s[-1]]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), nb_classes=s[-1])
    equal_entries = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped, axis=1) + K.sum(y_pred_reshaped, axis=1)

    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou, iou_mask)

    return K.mean(iou_masked)
