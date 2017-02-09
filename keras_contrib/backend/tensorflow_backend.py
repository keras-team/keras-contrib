import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc
from keras import backend as K
import numpy as np
import os
import warnings
from keras.backend.common import floatx, _EPSILON, image_dim_ordering, reset_uids
py_all = all

def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        padding = 'SAME'
    elif border_mode == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid border mode:', border_mode)
    return padding


def extract_image_patches(X, ksizes, ssizes, border_mode="same", dim_ordering="tf"):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    ssizes : 2-d tuple with the strides size
    border_mode : 'same' or 'valid'
    dim_ordering : 'tf' or 'th'
    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_border_mode(border_mode)
    if dim_ordering == "th":
        X = K.permute_dimensions(X, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = K.int_shape(X)
    patches = tf.extract_image_patches(X, kernel, strides, [1, 1, 1, 1], padding)
    # Reshaping to fit Theano
    bs, w, h, ch = K.int_shape(patches)
    patches = tf.reshape(tf.transpose(tf.reshape(patches, [bs, w, h, -1, ch_i]), [0, 1, 2, 4, 3]),
                         [bs, w, h, ch_i, ksizes[0], ksizes[1]])
    if dim_ordering == "tf":
        patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches
