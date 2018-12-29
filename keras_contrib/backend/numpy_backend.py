import numpy as np
from keras import backend as K


def extract_image_patches(X, ksizes, strides,
                          padding='valid',
                          data_format='channels_first'):
    raise NotImplementedError


def depth_to_space(input, scale, data_format=None):
    raise NotImplementedError


def moments(x, axes, shift=None, keep_dims=False):
    mean_batch = np.mean(x, axis=tuple(axes), keepdims=keep_dims)
    var_batch = np.var(x, axis=tuple(axes), keepdims=keep_dims)
    return mean_batch, var_batch
