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
from keras.backend.common import floatx, _EPSILON, image_data_format, reset_uids
from keras.backend.tensorflow_backend import _preprocess_conv3d_input
from keras.backend.tensorflow_backend import _preprocess_conv3d_kernel
from keras.backend.tensorflow_backend import _postprocess_conv3d_output
from keras.backend.tensorflow_backend import _preprocess_padding
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _postprocess_conv2d_output

py_all = all


def _preprocess_deconv_output_shape(x, shape, data_format):
    if data_format == 'channels_first':
        shape = (shape[0],) + tuple(shape[2:]) + (shape[1],)

    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape


def deconv3d(x, kernel, output_shape, strides=(1, 1, 1),
             padding='valid',
             data_format='default',
             image_shape=None, filter_shape=None):
    '''3D deconvolution (i.e. transposed convolution).

    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.

    # Returns
        A tensor, result of transposed 3D convolution.

    # Raises
        ValueError: if `image_data_format` is neither 
                    `channels_last` or `channels_first`.
    '''
    if data_format == 'default':
        data_format = image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Unknown image data format ' + str(data_format))

    x = _preprocess_conv3d_input(x, data_format)
    output_shape = _preprocess_deconv_output_shape(x, output_shape,
                                                   data_format)
    kernel = _preprocess_conv3d_kernel(kernel, data_format)
    kernel = tf.transpose(kernel, (0, 1, 2, 4, 3))
    padding = _preprocess_padding(padding)
    strides = (1,) + strides + (1,)

    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides,
                               padding=padding)
    return _postprocess_conv3d_output(x, data_format)


def extract_image_patches(x, ksizes, ssizes, padding="same",
                          data_format="channels_last"):
    '''
    Extract the patches from an image
    # Parameters

        x : The input image
        ksizes : 2-d tuple with the kernel size
        ssizes : 2-d tuple with the strides size
        padding : 'same' or 'valid'
        data_format : 'channels_last' or 'channels_first'

    # Returns
        The (k_w,k_h) patches extracted
        TF ==> (batch_size,w,h,k_w,k_h,c)
        TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    kernel = [1, ksizes[0], ksizes[1], 1]
    strides = [1, ssizes[0], ssizes[1], 1]
    padding = _preprocess_padding(padding)
    if data_format == "channels_first":
        x = KTF.permute_dimensions(x, (0, 2, 3, 1))
    bs_i, w_i, h_i, ch_i = KTF.int_shape(x)
    patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                       padding)
    # Reshaping to fit Theano
    bs, w, h, ch = KTF.int_shape(patches)
    patches = tf.reshape(patches, [bs, w, h, -1, ch_i])
    patches = tf.reshape(tf.transpose(patches, [0, 1, 2, 4, 3]),
                         [bs, w, h, ch_i, ksizes[0], ksizes[1]])
    if data_format == "channels_last":
        patches = KTF.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def depth_to_space(input, scale):
    ''' Uses phase shift algorithm to convert channels/depth for spatial resolution '''

    input = _preprocess_conv2d_input(input, image_data_format())
    out = tf.depth_to_space(input, scale)
    out = _postprocess_conv2d_output(out, image_data_format())
    return out


def moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''

    return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)
