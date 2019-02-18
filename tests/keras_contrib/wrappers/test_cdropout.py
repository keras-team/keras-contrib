import pytest
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal
from numpy.testing import assert_equal

from keras_contrib.wrappers import ConcreteDropout


def test_cdropout():
    #  Data
    in_dim = 20
    init_prop = .1
    np.random.seed(1)
    X = np.random.randn(1, in_dim)

    #  Model
    inputs = Input(shape=(in_dim,))
    dense = Dense(1, use_bias=True, input_shape=(in_dim,))
    #  Model, normal
    cd = ConcreteDropout(dense, in_dim, prob_init=(init_prop, init_prop))
    x = cd(inputs)
    model = Model(inputs, x)
    model.compile(loss=None, optimizer='rmsprop')
    #  Model, reference w/o Dropout
    x_ref = dense(inputs)
    model_ref = Model(inputs, x_ref)
    model_ref.compile(loss='mse', optimizer='rmsprop')

    #  Check about correct 3rd weight (equal to initial value)
    W = model.get_weights()
    assert_array_almost_equal(W[2], [np.log(init_prop)])

    #  Check if ConcreteDropout in prediction phase is the same as no dropout
    out = model.predict(X)
    out_ref = model_ref.predict(X)
    assert_allclose(out, out_ref, atol=1e-5)

    #  Check if ConcreteDropout has the right amount of losses deposited
    assert_equal(len(model.losses), 1)

    #  Check if the loss correspons the the desired value
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))
    p = np.squeeze(sigmoid(W[2]))
    kernel_regularizer = cd.weight_regularizer * np.sum(np.square(W[0])) / (1. - p)
    dropout_regularizer = (p * np.log(p) + (1. - p) * np.log(1. - p))
    dropout_regularizer *= cd.dropout_regularizer * in_dim
    loss = np.sum(kernel_regularizer + dropout_regularizer)
    eval_loss = model.evaluate(X)
    assert_approx_equal(eval_loss, loss)


if __name__ == '__main__':
    pytest.main([__file__])
