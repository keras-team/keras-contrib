import pytest
import numpy as np
import itertools
from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import layer_test
from keras.utils.conv_utils import conv_input_length
from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib.layers import convolutional, pooling
from keras.models import Sequential

# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_border_modes = ['valid', 'same']
else:
    _convolution_border_modes = ['valid', 'same']


def test_cosineconvolution_2d():
    num_samples = 2
    num_filter = 2
    stack_size = 3
    num_row = 10
    num_col = 6

    if K.backend() == 'theano':
        data_format = 'channels_first'
    elif K.backend() == 'tensorflow':
        data_format = 'channels_last'

    for border_mode in _convolution_border_modes:
        for subsample in [(1, 1), (2, 2)]:
            for use_bias_mode in [True, False]:
                if border_mode == 'same' and subsample != (1, 1):
                    continue

                layer_test(convolutional.CosineConvolution2D,
                           kwargs={'filters': num_filter,
                                   'kernel_size': (3, 3),
                                   'padding': border_mode,
                                   'strides': subsample,
                                   'use_bias': use_bias_mode,
                                   'data_format': data_format},
                           input_shape=(num_samples, num_row, num_col, stack_size))

                layer_test(convolutional.CosineConvolution2D,
                           kwargs={'filters': num_filter,
                                   'kernel_size': (3, 3),
                                   'padding': border_mode,
                                   'strides': subsample,
                                   'use_bias': use_bias_mode,
                                   'data_format': data_format,
                                   'kernel_regularizer': 'l2',
                                   'bias_regularizer': 'l2',
                                   'activity_regularizer': 'l2'},
                           input_shape=(num_samples, num_row, num_col, stack_size))

    if data_format == 'channels_first':
        X = np.random.randn(1, 3, 5, 5)
        input_dim = (3, 5, 5)
        W0 = X[:, :, ::-1, ::-1]
    elif data_format == 'channels_last':
        X = np.random.randn(1, 5, 5, 3)
        input_dim = (5, 5, 3)
        W0 = X[0, :, :, :, None]

    model = Sequential()
    model.add(convolutional.CosineConvolution2D(1, (5, 5), use_bias=True,
                                                input_shape=input_dim,
                                                data_format=data_format))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = W0
    W[1] = np.asarray([1.])
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, np.ones((1, 1, 1, 1), dtype=K.floatx()), atol=1e-5)

    model = Sequential()
    model.add(convolutional.CosineConvolution2D(1, (5, 5), use_bias=False,
                                                input_shape=input_dim,
                                                data_format=data_format))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = -2 * W0
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, -np.ones((1, 1, 1, 1), dtype=K.floatx()), atol=1e-5)


def test_sub_pixel_upscaling():
    num_samples = 2
    num_row = 16
    num_col = 16
    input_dtype = K.floatx()

    for scale_factor in [2, 3, 4]:
        input_data = np.random.random((num_samples, 4 * (scale_factor ** 2), num_row, num_col))
        input_data = input_data.astype(input_dtype)

        if K.image_data_format() == 'channels_last':
            input_data = input_data.transpose((0, 2, 3, 1))

        input_tensor = K.variable(input_data)
        expected_output = K.eval(KC.depth_to_space(input_tensor,
                                                   scale=scale_factor))

        layer_test(convolutional.SubPixelUpscaling,
                   kwargs={'scale_factor': scale_factor},
                   input_data=input_data,
                   expected_output=expected_output,
                   expected_output_dtype=K.floatx())


if __name__ == '__main__':
    pytest.main([__file__])
