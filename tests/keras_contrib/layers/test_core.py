import pytest
import numpy as np

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib.layers import core
from keras.utils.test_utils import layer_test, keras_test
from numpy.testing import assert_allclose


@keras_test
def test_cosinedense():
    from keras import regularizers
    from keras import constraints
    from keras.models import Sequential

    layer_test(core.CosineDense,
               kwargs={'units': 3},
               input_shape=(3, 2))

    layer_test(core.CosineDense,
               kwargs={'units': 3},
               input_shape=(3, 4, 2))

    layer_test(core.CosineDense,
               kwargs={'units': 3},
               input_shape=(None, None, 2))

    layer_test(core.CosineDense,
               kwargs={'units': 3},
               input_shape=(3, 4, 5, 2))

    layer_test(core.CosineDense,
               kwargs={'units': 3,
                       'kernel_regularizer': regularizers.l2(0.01),
                       'bias_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.l2(0.01),
                       'kernel_constraint': constraints.MaxNorm(1),
                       'bias_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))

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

@keras_test
def test_mcdropout():
    from keras.models import Sequential
    from keras.layers import Dense

    model1 = Sequential()
    model1.add(Dense(32, input_shape=(100,)))
    model1.add(core.Dropout(0.5, mc_dropout=True))
    model1.add(Dense(1))
    model1.compile(optimizer='sgd', loss='mse')

    model2 = Sequential()
    model2.add(Dense(32, input_shape=(100,)))
    model2.add(core.Dropout(0.5, mc_dropout=False))
    model2.add(Dense(1))
    model2.compile(optimizer='sgd', loss='mse')

    X = np.random.randn(1, 100)

    y = []
    for i in range(100):
        y.append(model1.predict(X))
    y = np.asarray(y)
    assert(y.std() > 1e-5)

    y = []
    for i in range(100):
        y.append(model2.predict(X))
    y = np.asarray(y)
    assert(y.std() < 1e-5)

if __name__ == '__main__':
    pytest.main([__file__])
