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
from keras.backend.tensorflow_backend import _preprocess_conv3d_input
from keras.backend.tensorflow_backend import _preprocess_conv3d_kernel
from keras.backend.tensorflow_backend import _preprocess_border_mode
from keras.backend.tensorflow_backend import _postprocess_conv3d_output
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
    output_shape = _preprocess_deconv_output_shape(x, output_shape, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    kernel = tf.transpose(kernel, (0, 1, 3, 4, 2))
    padding = _preprocess_border_mode(border_mode)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv3d_output(x, dim_ordering)
