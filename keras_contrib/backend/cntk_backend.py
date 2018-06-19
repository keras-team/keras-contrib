from keras.backend import cntk_backend as KCN
from keras.backend.cntk_backend import logsumexp
import cntk as C
import numpy as np


def clip(x, min_value, max_value):
    """Element-wise value clipping.

    If min_value > max_value, clipping range is [min_value,min_value].

    # Arguments
        x: Tensor or variable.
        min_value: Tensor, float, int, or None.
            If min_value is None, defaults to -infinity.
        max_value: Tensor, float, int, or None.
            If max_value is None, defaults to infinity.

    # Returns
        A tensor.
    """
    if max_value is None:
        max_value = np.inf
    if min_value is None:
        min_value = -np.inf
    max_value = C.maximum(min_value, max_value)
    return C.clip(x, min_value, max_value)


def moments(x, axes, shift=None, keep_dims=False):
    ''' Calculates and returns the mean and variance of the input '''
    mean, variant = KCN._moments(x, axes=axes, shift=shift, keep_dims=keep_dims)
    return mean, variant
