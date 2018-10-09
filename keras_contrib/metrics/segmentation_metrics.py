""" Loss functions for semantic segmentation algorithms.

    adapted from: https://github.com/theduynguyen/Keras-FCN
"""
import sys
from keras import metrics
import keras.backend as K


def _end_mean(x, axis=-1):
    """ Same as K.mean, but defaults to the final axis.
    """
    return K.mean(x, axis=axis)


def _metric_2d_adaptor(y_true, y_pred, metric=None, summary=_end_mean, **kwargs):
    """ Adapt a one dimensional loss function to work with 2d segmentation data.
    """
    if metric is None:
        raise ValueError("You must provide a metric function such as binary_crossentropy")
    pred_shape = K.shape(y_pred)
    true_shape = K.shape(y_true)
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    result = metric(y_pred_reshaped, y_true_reshaped, **kwargs)

    if summary is not None:
        result = summary(result)

    if len(true_shape) >= 3:
        return K.reshape(result, true_shape[:-1])
    else:
        return result


def categorical_pixel_accuracy(y_true, y_pred):
    pred_shape = K.shape(y_pred)
    true_shape = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    correct_pixels_per_class = K.cast(
        K.equal(clf_pred, y_true_reshaped), dtype='float32')

    return K.mean(correct_pixels_per_class)


def mean_accuracy(y_true, y_pred):
    pred_shape = K.shape(y_pred)
    true_shape = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    equal_entries = K.cast(
        K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    correct_pixels_per_class = K.sum(equal_entries, axis=1)
    n_pixels_per_class = K.sum(y_true_reshaped, axis=1)

    # epsilon added to avoid dividing by zero
    acc = (correct_pixels_per_class + K.epsilon()) / (n_pixels_per_class + K.epsilon())

    return K.mean(acc)


def binary_accuracy(y_true, y_pred):
    """ Same as keras.metrics.binary_accuracy for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, loss=metrics.binary_accuracy, summary=_end_mean)


def categorical_accuracy(y_true, y_pred):
    """ Same as keras.metrics.categorical_accuracy for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, loss=metrics.categorical_accuracy, summary=_end_mean)


def top_k_categorical_accuracy(y_true, y_pred):
    """ Same as keras.metrics.top_k_categorical_accuracy for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, loss=metrics.top_k_categorical_accuracy, summary=_end_mean)


def sparse_top_k_categorical_accuracy(y_true, y_pred):
    """ Same as keras.metrics.categorical_accuracy for 2d label data.
    """
    return _metric_2d_adaptor(y_true, y_pred, loss=metrics.sparse_top_k_categorical_accuracy, summary=_end_mean)


def mean_intersection_over_union(y_true, y_pred, smooth=None, axis=-1):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.

    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.

    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?

    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.

    Also see jaccard which takes a slighty different approach.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # References

    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.

    https://en.wikipedia.org/wiki/Jaccard_index

    """
    if smooth is None:
        smooth = K.epsilon()
    pred_shape = K.shape(y_pred)
    true_shape = K.shape(y_true)

    # reshape such that w and h dim are multiplied together
    y_pred_reshaped = K.reshape(y_pred, (-1, pred_shape[-1]))
    y_true_reshaped = K.reshape(y_true, (-1, true_shape[-1]))

    # correctly classified
    clf_pred = K.one_hot(K.argmax(y_pred_reshaped), num_classes=true_shape[-1])
    equal_entries = K.cast(
        K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(
        y_true_reshaped, axis=1) + K.sum(
            y_pred_reshaped, axis=1)

    # smooth added to avoid dividing by zero
    iou = (intersection + smooth) / (
        (union_per_class - intersection) + smooth)

    return K.mean(iou)