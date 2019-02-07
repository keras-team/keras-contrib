import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import layer_test
from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib.layers import convolutional
from keras.models import Sequential

# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_border_modes = ['valid', 'same']
    data_format = 'channels_first'
else:
    _convolution_border_modes = ['valid', 'same']
    data_format = 'channels_last'


@pytest.mark.parametrize('border_mode', _convolution_border_modes)
@pytest.mark.parametrize('subsample', [(1, 1), (2, 2)])
@pytest.mark.parametrize('use_bias_mode', [True, False])
@pytest.mark.parametrize('use_regularizer', [True, False])
def test_cosineconvolution_2d(border_mode,
                              subsample,
                              use_bias_mode,
                              use_regularizer):
    num_samples = 2
    num_filter = 2
    stack_size = 3
    num_row = 10
    num_col = 6

    if border_mode == 'same' and subsample != (1, 1):
        return

    kwargs = {'filters': num_filter,
              'kernel_size': (3, 3),
              'padding': border_mode,
              'strides': subsample,
              'use_bias': use_bias_mode,
              'data_format': data_format}
    if use_regularizer:
        kwargs.update({'kernel_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2'})

    layer_test(convolutional.CosineConvolution2D,
               kwargs=kwargs,
               input_shape=(num_samples, num_row, num_col, stack_size))


def test_cosineconvolution_2d_correctness():
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


@pytest.mark.parametrize('scale_factor', [2, 3, 4])
def test_sub_pixel_upscaling(scale_factor):
    num_samples = 2
    num_row = 16
    num_col = 16
    input_dtype = K.floatx()

    nb_channels = 4 * (scale_factor ** 2)
    input_data = np.random.random((num_samples, nb_channels, num_row, num_col))
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

def test_drop_connect_dense():
    layer_test(convolutional.DropConnectDense,
               prob=0.1,
               kwargs={'units': 3},
               input_shape=(3, 2))

    layer_test(convolutional.DropConnectDense,
               prob=0.2
               kwargs={'units': 3},
               input_shape=(3, 4, 2))

    layer_test(convolutional.DropConnectDense,
               prob=0.4
               kwargs={'units': 3},
               input_shape=(None, None, 2))

    layer_test(convolutional.DropConnectDense,
               prob=0.05
               kwargs={'units': 3},
               input_shape=(3, 4, 5, 2))

    layer_test(convolutional.DropConnectDense,
               prob=0.075
               kwargs={'units': 3,
                       'kernel_regularizer': regularizers.l2(0.01),
                       'bias_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.L1L2(l1=0.01, l2=0.01),
                       'kernel_constraint': constraints.MaxNorm(1),
                       'bias_constraint': constraints.max_norm(1)},
               input_shape=(3, 2))

    layer = layers.Dense(3,
                         prob=0.15
                         kernel_regularizer=regularizers.l1(0.01),
                         bias_regularizer='l1')
    layer.build((None, 4))
    assert len(layer.losses) == 2

if __name__ == '__main__':
    pytest.main([__file__])
