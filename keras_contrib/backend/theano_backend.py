import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.neighbours import images2neibs
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
from theano.printing import Print

try:
    import theano.sparse as th_sparse_module
except ImportError:
    th_sparse_module = None
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign
from keras import backend as K
from keras.backend import theano_backend as KTH
import inspect
import numpy as np
from keras.backend.common import _FLOATX, floatx, _EPSILON, image_data_format
from keras.backend.theano_backend import _preprocess_conv3d_input
from keras.backend.theano_backend import _preprocess_conv3d_kernel
from keras.backend.theano_backend import _preprocess_conv3d_filter_shape
from keras.backend.theano_backend import _preprocess_padding
from keras.backend.theano_backend import _postprocess_conv3d_output
from keras.backend.theano_backend import _preprocess_conv2d_input
from keras.backend.theano_backend import _postprocess_conv2d_output

import itertools

py_all = all


def conv2d(x, kernel, strides=(1, 1), padding='valid', data_format='channels_first',
           image_shape=None, filter_shape=None):
    '''
    padding: string, "same" or "valid".
    '''
    if data_format not in {'channels_first', 'channels_last'}:
        raise Exception('Unknown data_format ' + str(data_format))

    if data_format == 'channels_last':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.dimshuffle((0, 3, 1, 2))
        kernel = kernel.dimshuffle((3, 2, 0, 1))
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

    if padding == 'same':
        th_padding = 'half'
        np_kernel = kernel.eval()
    elif padding == 'valid':
        th_padding = 'valid'
    else:
        raise Exception('Border mode not supported: ' + str(padding))

    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None

    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)

    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)

    conv_out = T.nnet.conv2d(x, kernel,
                             border_mode=th_padding,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=filter_shape)

    if padding == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]

    if data_format == 'channels_last':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def deconv3d(x, kernel, output_shape, strides=(1, 1, 1),
             padding='valid',
             data_format=None, filter_shape=None):
    '''3D deconvolution (transposed convolution).

    # Arguments
        kernel: kernel tensor.
        output_shape: desired dimensions of output.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "channels_last" or "channels_first".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    '''
    flip_filters = False
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))

    if data_format == 'channels_last':
        output_shape = (output_shape[0], output_shape[4], output_shape[1],
                        output_shape[2], output_shape[3])

    x = _preprocess_conv3d_input(x, data_format)
    kernel = _preprocess_conv3d_kernel(kernel, data_format)
    kernel = kernel.dimshuffle((1, 0, 2, 3, 4))
    th_padding = _preprocess_padding(padding)

    if hasattr(kernel, '_keras_shape'):
        kernel_shape = kernel._keras_shape
    else:
        # Will only work if `kernel` is a shared variable.
        kernel_shape = kernel.eval().shape

    filter_shape = _preprocess_conv3d_filter_shape(filter_shape, data_format)
    filter_shape = tuple(filter_shape[i] for i in (1, 0, 2, 3, 4))

    conv_out = T.nnet.abstract_conv.conv3d_grad_wrt_inputs(
        x, kernel, output_shape,
        filter_shape=filter_shape,
        border_mode=th_padding,
        subsample=strides,
        filter_flip=not flip_filters)

    conv_out = _postprocess_conv3d_output(conv_out, x, padding,
                                          kernel_shape, strides, data_format)
    return conv_out


def extract_image_patches(X, ksizes, strides, padding="valid", data_format="channels_first"):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    strides : 2-d tuple with the strides size
    padding : 'same' or 'valid'
    data_format : 'channels_last' or 'channels_first'
    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    patch_size = ksizes[1]
    if padding == "same":
        padding = "ignore_borders"
    if data_format == "channels_last":
        X = KTH.permute_dimensions(X, [0, 3, 1, 2])
    # Thanks to https://github.com/awentzonline for the help!
    batch, c, w, h = KTH.shape(X)
    xs = KTH.shape(X)
    num_rows = 1 + (xs[-2] - patch_size) // strides[1]
    num_cols = 1 + (xs[-1] - patch_size) // strides[1]
    num_channels = xs[-3]
    patches = images2neibs(X, ksizes, strides, padding)
    # Theano is sorting by channel
    patches = KTH.reshape(patches, (batch, num_channels, KTH.shape(patches)[0] // num_channels, patch_size, patch_size))
    patches = KTH.permute_dimensions(patches, (0, 2, 1, 3, 4))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = KTH.reshape(patches, (batch, num_rows, num_cols, num_channels, patch_size, patch_size))
    if data_format == "channels_last":
        patches = KTH.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def depth_to_space(input, scale, data_format=None):
    ''' Uses phase shift algorithm to convert channels/depth for spatial resolution '''
    if data_format is None:
        data_format = image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)

    b, k, row, col = input.shape
    output_shape = (b, k // (scale ** 2), row * scale, col * scale)

    out = T.zeros(output_shape)
    r = scale

    for y, x in itertools.product(range(scale), repeat=2):
        out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x:: r * r, :, :])

    out = _postprocess_conv2d_output(out, input, None, None, None, data_format)
    return out


def moments(x, axes, shift=None, keep_dims=False):
    ''' Calculates and returns the mean and variance of the input '''

    mean_batch = KTH.mean(x, axis=axes, keepdims=keep_dims)
    var_batch = KTH.var(x, axis=axes, keepdims=keep_dims)

    return mean_batch, var_batch
