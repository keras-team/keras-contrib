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
               kwargs={'output_dim': 3},
               input_shape=(3, 2))

    layer_test(core.CosineDense,
               kwargs={'output_dim': 3},
               input_shape=(3, 4, 2))

    layer_test(core.CosineDense,
               kwargs={'output_dim': 3},
               input_shape=(None, None, 2))

    layer_test(core.CosineDense,
               kwargs={'output_dim': 3},
               input_shape=(3, 4, 5, 2))

    layer_test(core.CosineDense,
               kwargs={'output_dim': 3,
                       'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.activity_l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))

    X = np.random.randn(1, 20)
    model = Sequential()
    model.add(core.CosineDense(1, bias=True, input_shape=(20,)))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = X.T
    W[1] = np.asarray([1.])
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, np.ones((1, 1), dtype=K.floatx()))

    X = np.random.randn(1, 20)
    model = Sequential()
    model.add(core.CosineDense(1, bias=False, input_shape=(20,)))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = -X.T
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, -np.ones((1, 1), dtype=K.floatx()))


if __name__ == '__main__':
    pytest.main([__file__])
