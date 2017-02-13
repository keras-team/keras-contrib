import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras.utils.np_utils import conv_input_length
from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib.layers import convolutional, pooling


# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_border_modes = ['valid', 'same', 'full']
else:
    _convolution_border_modes = ['valid', 'same']


@keras_test
def test_deconvolution_3d():
    nb_samples = 6
    nb_filter = 4
    stack_size = 2
    kernel_dim1 = 12
    kernel_dim2 = 10
    kernel_dim3 = 8

    for batch_size in [None, nb_samples]:
        for border_mode in _convolution_border_modes:
            for subsample in [(1, 1, 1), (2, 2, 2)]:
                if border_mode == 'same' and subsample != (1, 1, 1):
                    continue

                dim1 = conv_input_length(kernel_dim1, 7, border_mode, subsample[0])
                dim2 = conv_input_length(kernel_dim2, 5, border_mode, subsample[1])
                dim3 = conv_input_length(kernel_dim3, 3, border_mode, subsample[2])
                layer_test(convolutional.Deconvolution3D,
                           kwargs={'nb_filter': nb_filter,
                                   'kernel_dim1': 7,
                                   'kernel_dim2': 5,
                                   'kernel_dim3': 3,
                                   'output_shape': (batch_size, nb_filter, dim1, dim2, dim3),
                                   'border_mode': border_mode,
                                   'subsample': subsample,
                                   'dim_ordering': 'th'},
                           input_shape=(nb_samples, stack_size, kernel_dim1, kernel_dim2, kernel_dim3),
                           fixed_batch_size=True)

                layer_test(convolutional.Deconvolution3D,
                           kwargs={'nb_filter': nb_filter,
                                   'kernel_dim1': 7,
                                   'kernel_dim2': 5,
                                   'kernel_dim3': 3,
                                   'output_shape': (batch_size, nb_filter, dim1, dim2, dim3),
                                   'border_mode': border_mode,
                                   'dim_ordering': 'th',
                                   'W_regularizer': 'l2',
                                   'b_regularizer': 'l2',
                                   'activity_regularizer': 'activity_l2',
                                   'subsample': subsample},
                           input_shape=(nb_samples, stack_size, kernel_dim1, kernel_dim2, kernel_dim3),
                           fixed_batch_size=True)

                layer_test(convolutional.Deconvolution3D,
                           kwargs={'nb_filter': nb_filter,
                                   'kernel_dim1': 7,
                                   'kernel_dim2': 5,
                                   'kernel_dim3': 3,
                                   'output_shape': (nb_filter, dim1, dim2, dim3),
                                   'border_mode': border_mode,
                                   'dim_ordering': 'th',
                                   'W_regularizer': 'l2',
                                   'b_regularizer': 'l2',
                                   'activity_regularizer': 'activity_l2',
                                   'subsample': subsample},
                           input_shape=(nb_samples, stack_size, kernel_dim1, kernel_dim2, kernel_dim3))


if __name__ == '__main__':
    pytest.main([__file__])
