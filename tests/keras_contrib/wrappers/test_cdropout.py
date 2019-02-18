import pytest
import numpy as np

from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal
from numpy.testing import assert_equal
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv3D
from keras.models import Model
from keras_contrib.wrappers import ConcreteDropout


def test_cdropout():
    #  DATA
    in_dim = 20
    init_prop = .1
    np.random.seed(1)
    X = np.random.randn(1, in_dim)

    #  MODEL
    inputs = Input(shape=(in_dim,))
    dense = Dense(1, use_bias=True)
    #  Model, normal
    cd = ConcreteDropout(dense, in_dim, prob_init=(init_prop, init_prop))
    x = cd(inputs)
    model = Model(inputs, x)
    model.compile(loss=None, optimizer='rmsprop')
    #  Model, reference w/o Dropout
    x_ref = dense(inputs)
    model_ref = Model(inputs, x_ref)
    model_ref.compile(loss='mse', optimizer='rmsprop')

    #  CHECKS
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


def test_cdropout_conv():
    #  DATA
    in_dim = 20
    init_prop = .1
    np.random.seed(1)
    X = np.random.randn(1, in_dim, in_dim, 1)

    #  MODEL
    inputs = Input(shape=(in_dim, in_dim, 1,))
    conv2d = Conv2D(1, (3, 3))
    #  Model, normal
    cd = ConcreteDropout(conv2d, in_dim, prob_init=(init_prop, init_prop))
    x = cd(inputs)
    model = Model(inputs, x)
    model.compile(loss=None, optimizer='rmsprop')
    #  Model, reference w/o Dropout
    x_ref = conv2d(inputs)
    model_ref = Model(inputs, x_ref)
    model_ref.compile(loss=None, optimizer='rmsprop')

    #  CHECKS
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
    dropout_regularizer *= cd.dropout_regularizer * 1  # only channels are dropped
    loss = np.sum(kernel_regularizer + dropout_regularizer)
    eval_loss = model.evaluate(X)
    assert_approx_equal(eval_loss, loss)


def test_cdropout_1d_layer():
    """To be replaced with a real function test, if implemented.
    """
    in_dim = 20
    init_prop = .1

    with pytest.raises(ValueError):
        inputs = Input(shape=(in_dim, 1,))
        ConcreteDropout(Conv1D(1, 3),
                        in_dim,
                        prob_init=(init_prop, init_prop))(inputs)


def test_cdropout_3d_layer():
    """To be replaced with a real function test, if implemented.
    """
    in_dim = 20
    init_prop = .1

    with pytest.raises(ValueError):
        inputs = Input(shape=(in_dim, in_dim, in_dim, 1,))
        ConcreteDropout(Conv3D(1, 3),
                        in_dim,
                        prob_init=(init_prop, init_prop))(inputs)


if __name__ == '__main__':
    pytest.main([__file__])
