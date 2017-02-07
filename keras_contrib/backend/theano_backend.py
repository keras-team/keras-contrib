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
import inspect
import numpy as np
from keras.backend.common import _FLOATX, floatx, _EPSILON, image_dim_ordering
py_all = all


def extract_image_patches(X, ksizes, strides, border_mode="valid", dim_ordering="th"):
    '''
    Extract the patches from an image
    Parameters
    ----------
    X : The input image
    ksizes : 2-d tuple with the kernel size
    strides : 2-d tuple with the strides size
    border_mode : 'same' or 'valid'
    dim_ordering : 'tf' or 'th'
    Returns
    -------
    The (k_w,k_h) patches extracted
    TF ==> (batch_size,w,h,k_w,k_h,c)
    TH ==> (batch_size,w,h,c,k_w,k_h)
    '''
    patch_size = ksizes[1]
    if border_mode == "same":
        border_mode = "ignore_border"
    if dim_ordering == "tf":
        X = K.permute_dimensions(X, [0, 2, 3, 1])
    # Thanks to https://github.com/awentzonline for the help!
    batch, c, w, h = K.shape(X)
    xs = K.shape(X)
    num_rows = 1 + (xs[-2] - patch_size) // strides[1]
    num_cols = 1 + (xs[-1] - patch_size) // strides[1]
    num_channels = xs[-3]
    patches = images2neibs(X, ksizes, strides, border_mode)
    # Theano is sorting by channel
    patches = K.reshape(patches, (batch, num_channels, K.shape(patches)[0] // num_channels, patch_size, patch_size))
    patches = K.permute_dimensions(patches, (0, 2, 1, 3, 4))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = K.reshape(patches, (batch, num_rows, num_cols, num_channels, patch_size, patch_size))

    if dim_ordering == "tf":
        patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
    return patches
