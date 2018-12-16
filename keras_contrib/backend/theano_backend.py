from theano import tensor as T
from theano.sandbox.neighbours import images2neibs
import numpy as np

try:
    import theano.sparse as th_sparse_module
except ImportError:
    th_sparse_module = None
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign
from keras.backend import theano_backend as KTH
from keras.backend.common import image_data_format
from keras.backend.theano_backend import _preprocess_conv3d_input
from keras.backend.theano_backend import _preprocess_conv3d_kernel
from keras.backend.theano_backend import _preprocess_conv3d_filter_shape
from keras.backend.theano_backend import _preprocess_padding
from keras.backend.theano_backend import _postprocess_conv3d_output
from keras.backend.theano_backend import _preprocess_conv2d_input
from keras.backend.theano_backend import _postprocess_conv2d_output
from keras.backend.theano_backend import logsumexp

py_all = all

def extract_image_patches(X, ksizes, strides, padding='valid', data_format='channels_first'):
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
    if padding == 'same':
        padding = 'ignore_borders'
    if data_format == 'channels_last':
        X = KTH.permute_dimensions(X, [0, 3, 1, 2])
    # Thanks to https://github.com/awentzonline for the help!
    batch, c, w, h = KTH.shape(X)
    xs = KTH.shape(X)
    num_rows = 1 + (xs[-2] - patch_size) // strides[1]
    num_cols = 1 + (xs[-1] - patch_size) // strides[1]
    num_channels = xs[-3]
    patches = images2neibs(X, ksizes, strides, padding)
    # Theano is sorting by channel
    patches = KTH.reshape(patches, (batch, num_channels, num_rows * num_cols, patch_size, patch_size))
    patches = KTH.permute_dimensions(patches, (0, 2, 1, 3, 4))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = KTH.reshape(patches, (batch, num_rows, num_cols, num_channels, patch_size, patch_size))
    if data_format == 'channels_last':
        patches = KTH.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches


def depth_to_space(input, scale, data_format=None):
    ''' Uses phase shift algorithm to convert channels/depth for spatial resolution '''
    if data_format is None:
        data_format = image_data_format()
    data_format = data_format.lower()
    input = _preprocess_conv2d_input(input, data_format)

    b, k, row, col = input.shape
    out_channels = k // (scale ** 2)
    x = T.reshape(input, (b, scale, scale, out_channels, row, col))
    x = T.transpose(x, (0, 3, 4, 1, 5, 2))
    out = T.reshape(x, (b, out_channels, row * scale, col * scale))

    out = _postprocess_conv2d_output(out, input, None, None, None, data_format)
    return out


def moments(x, axes, shift=None, keep_dims=False):
    ''' Calculates and returns the mean and variance of the input '''

    mean_batch = KTH.mean(x, axis=axes, keepdims=keep_dims)
    var_batch = KTH.var(x, axis=axes, keepdims=keep_dims)

    return mean_batch, var_batch
