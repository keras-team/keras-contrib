import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
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
import inspect
import numpy as np
from keras.backend.common import _FLOATX, floatx, _EPSILON, image_dim_ordering
from keras.backend.theano_backend import _preprocess_conv3d_input
from keras.backend.theano_backend import _preprocess_conv3d_kernel
from keras.backend.theano_backend import _preprocess_conv3d_filter_shape
from keras.backend.theano_backend import _preprocess_border_mode
from keras.backend.theano_backend import _postprocess_conv3d_output
py_all = all


def deconv3d(x, kernel, output_shape, strides=(1, 1, 1),
             border_mode='valid',
             dim_ordering='default',
             image_shape=None, filter_shape=None):
    '''3D deconvolution (transposed convolution).

    # Arguments
        kernel: kernel tensor.
        output_shape: desired dimensions of output.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    '''
    flip_filters = False
    if dim_ordering == 'default':
        dim_ordering = image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        output_shape = (output_shape[0], output_shape[4], output_shape[1],
                        output_shape[2], output_shape[3])

    x = _preprocess_conv3d_input(x, dim_ordering)
    kernel = _preprocess_conv3d_kernel(kernel, dim_ordering)
    kernel = kernel.dimshuffle((1, 0, 2, 3, 4))
    th_border_mode = _preprocess_border_mode(border_mode)

    if hasattr(kernel, '_keras_shape'):
        kernel_shape = kernel._keras_shape
    else:
        # Will only work if `kernel` is a shared variable.
        kernel_shape = kernel.eval().shape

    filter_shape = _preprocess_conv3d_filter_shape(dim_ordering, filter_shape)
    filter_shape = tuple(filter_shape[i] for i in (1, 0, 2, 3, 4))

    conv_out = T.nnet.abstract_conv.conv3d_grad_wrt_inputs(
        x, kernel, output_shape,
        filter_shape=filter_shape,
        border_mode=th_border_mode,
        subsample=strides,
        filter_flip=not flip_filters)

    conv_out = _postprocess_conv3d_output(conv_out, x, border_mode,
                                          kernel_shape, strides, dim_ordering)
    return conv_out
