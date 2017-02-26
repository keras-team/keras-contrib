import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
try:
    from tensorflow.python.ops import ctc_ops as ctc
except ImportError:
    import tensorflow.contrib.ctc as ctc
from keras import backend as K
from keras.backend import tensorflow_backend as KTF
import numpy as np
import os
import warnings
from keras.backend.common import floatx, _EPSILON, image_dim_ordering, reset_uids
from keras.backend.tensorflow_backend import _preprocess_conv3d_input
from keras.backend.tensorflow_backend import _preprocess_conv3d_kernel
from keras.backend.tensorflow_backend import _preprocess_border_mode
from keras.backend.tensorflow_backend import _postprocess_conv3d_output
from keras.backend.tensorflow_backend import _preprocess_border_mode
py_all = all


def _preprocess_deconv_output_shape(x, shape, dim_ordering):
    if dim_ordering == 'th':
        shape = (shape[0],) + tuple(shape[2:]) + (shape[1],)

    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape


def deconv3d(x, kernel, output_shape, strides=(1, 1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    '''3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of transposed 3D convolution.

    # Raises
        ValueError: if `dim_ordering` is neither `tf` or `th`.
    '''
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv3d_input(x, dim_ordering)
    output_shape = _preprocess_deconv_output_shape(x, output_shape,
                                                   dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    kernel = tf.transpose(kernel, (0, 1, 2, 4, 3))
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv3d_output(x, dim_ordering)


def extract_image_patches(x, ksizes, ssizes, border_mode="same",
                          dim_ordering="tf"):
    '''
    Extract the patches from an image
    # Parameters

        x : The input image
        ksizes : 2-d tuple with the kernel size
        ssizes : 2-d tuple with the strides size
        border_mode : 'same' or 'valid'
        dim_ordering : 'tf' or 'th'

    # Returns
        The (k_w,k_h) patches extracted
        TF ==> (batch_size,w,h,k_w,k_h,c)
        TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_border_mode(border_mode)
    if dim_ordering == "th":
        x = KTF.permute_dimensions(x, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = KTF.int_shape(x)
    patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                       padding)
    # Reshaping to fit Theano
    bs, w, h, ch = KTF.int_shape(patches)
    patches = tf.reshape(tf.transpose(
        tf.reshape(patches, [bs, w, h, -1, ch_i]), [0, 1, 2, 4, 3]),
                         [bs, w, h, ch_i, ksizes[0], ksizes[1]])
    if dim_ordering == "tf":
        patches = KTF.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''

    return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)
