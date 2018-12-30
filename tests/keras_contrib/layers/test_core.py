import pytest
import numpy as np

from keras import regularizers
from keras import constraints
from keras.models import Sequential
from keras import backend as K
from keras_contrib.layers import core
from keras_contrib.utils.test_utils import layer_test
from numpy.testing import assert_allclose


@pytest.mark.parametrize('input_shape', [(3, 2),
                                         (3, 4, 2),
                                         (None, None, 2),
                                         (3, 4, 5, 2)])
def test_cosinedense(input_shape):

    layer_test(core.CosineDense,
               kwargs={'units': 3},
               input_shape=input_shape)


def test_cosinedense_reg_constraint():
    layer_test(core.CosineDense,
               kwargs={'units': 3,
                       'kernel_regularizer': regularizers.l2(0.01),
                       'bias_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.l2(0.01),
                       'kernel_constraint': constraints.MaxNorm(1),
                       'bias_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


def test_cosinedense_correctness():
    X = np.random.randn(1, 20)
    model = Sequential()
    model.add(core.CosineDense(1, use_bias=True, input_shape=(20,)))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = X.T
    W[1] = np.asarray([1.])
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, np.ones((1, 1), dtype=K.floatx()), atol=1e-5)

    X = np.random.randn(1, 20)
    model = Sequential()
    model.add(core.CosineDense(1, use_bias=False, input_shape=(20,)))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = -2 * X.T
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, -np.ones((1, 1), dtype=K.floatx()), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
