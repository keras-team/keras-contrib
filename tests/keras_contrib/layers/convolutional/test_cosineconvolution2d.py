import numpy as np
import pytest
from keras import backend as K
from keras.models import Sequential
from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import CosineConvolution2D

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

    layer_test(CosineConvolution2D,
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
    model.add(CosineConvolution2D(1, (5, 5), use_bias=True,
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
    model.add(CosineConvolution2D(1, (5, 5),
                                  use_bias=False,
                                  input_shape=input_dim,
                                  data_format=data_format))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = -2 * W0
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, -np.ones((1, 1, 1, 1), dtype=K.floatx()), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
