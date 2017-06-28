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
def test_separablefc():
    from keras import regularizers
    from keras import constraints
    from keras import layers
    from keras.models import Sequential

    # Expected usage is after a stack of convolutional layers
    # and before densely connected layers
    # Reference: https://doi.org/10.1101/146431

    # Input
    np.random.seed(123)
    X = np.random.randn(1, 10, 4)
    
    # Model Test 1
    model1 = Sequential()
    model1.add(layers.convolutional.Conv1D(input_shape=(10,4),
                                           nb_filter=3,
                                           filter_length=2))
    model1.add(core.SeparableFC(output_dim=2, symmetric=True))
    model1.add(layers.core.Dense(output_dim=1))
    model1.compile(loss='mse', optimizer='rmsprop')
    out1 = model1.predict(X)
    assert_allclose(out1, np.ones((1, 1), dtype=K.floatx()), atol=1e-5)
    
    # Model Test 2
    model2 = Sequential()
    model2.add(layers.convolutional.Conv1D(input_shape=(10,4),
                                           nb_filter=3,
                                           filter_length=2))
    model2.add(core.SeparableFC(output_dim=2,
                                symmetric=True,
                                smoothness_penalty=10.0,
                                smoothness_l1=True,
                                smoothness_second_diff=True))
    model2.add(layers.core.Dense(output_dim=1))
    model2.compile(loss='binary_crossentropy', optimizer='Adam')
    out2 = model2.predict(X)
    assert_allclose(out2, np.ones((1, 1), dtype=K.floatx()), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
