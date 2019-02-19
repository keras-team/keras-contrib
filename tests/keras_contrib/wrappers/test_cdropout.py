import pytest
import numpy as np

from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal
from numpy.testing import assert_equal
from keras import backend as K
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv3D
from keras.models import Model
from keras_contrib.wrappers import ConcreteDropout


@pytest.fixture
def clear_session_after_test():
    """Overridden: make session cleanup manually.
    """
    pass


@pytest.fixture(scope='module')
def dense_model():
    """Initialize to be tested dense model. Executed once.
    """
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
    model_ref.compile(loss=None, optimizer='rmsprop')

    yield {'model': model,
           'model_ref': model_ref,
           'concrete_dropout': cd,
           'init_prop': init_prop,
           'in_dim': in_dim,
           'X': X}
    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()


def test_cdropout_dense_3rdweight(dense_model):
    """Check about correct 3rd weight (equal to initial value)
    """
    model = dense_model['model']
    init_prop = dense_model['init_prop']

    W = model.get_weights()
    assert_array_almost_equal(W[2], [np.log(init_prop)])


def test_cdropout_dense_identity(dense_model):
    """Check if ConcreteDropout in prediction phase is the same as no dropout
    """
    model = dense_model['model']
    model_ref = dense_model['model_ref']
    X = dense_model['X']

    out = model.predict(X)
    out_ref = model_ref.predict(X)
    assert_allclose(out, out_ref, atol=1e-5)


def test_cdropout_dense_loss(dense_model):
    """Check if ConcreteDropout has the right amount of losses deposited
    """
    model = dense_model['model']

    assert_equal(len(model.losses), 1)


def test_cdropout_dense_loss_value(dense_model):
    """Check if the loss corresponds the the desired value
    """
    model = dense_model['model']
    X = dense_model['X']
    cd = dense_model['concrete_dropout']
    in_dim = dense_model['in_dim']

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    W = model.get_weights()
    p = np.squeeze(sigmoid(W[2]))
    kernel_regularizer = cd.weight_regularizer * np.sum(np.square(W[0])) / (1. - p)
    dropout_regularizer = (p * np.log(p) + (1. - p) * np.log(1. - p))
    dropout_regularizer *= cd.dropout_regularizer * in_dim
    loss = np.sum(kernel_regularizer + dropout_regularizer)

    eval_loss = model.evaluate(X)
    assert_approx_equal(eval_loss, loss)


@pytest.fixture(scope='module')
def conv2d_model():
    """Initialize to be tested conv model. Executed once.
    """
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

    yield {'model': model,
           'model_ref': model_ref,
           'concrete_dropout': cd,
           'init_prop': init_prop,
           'in_dim': in_dim,
           'X': X}
    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        K.clear_session()


def test_cdropout_conv2d_3rdweight(conv2d_model):
    """Check about correct 3rd weight (equal to initial value)
    """
    model = conv2d_model['model']
    init_prop = conv2d_model['init_prop']

    W = model.get_weights()
    assert_array_almost_equal(W[2], [np.log(init_prop)])


def test_cdropout_conv2d_identity(conv2d_model):
    """Check if ConcreteDropout in prediction phase is the same as no dropout
    """
    model = conv2d_model['model']
    model_ref = conv2d_model['model_ref']
    X = conv2d_model['X']

    out = model.predict(X)
    out_ref = model_ref.predict(X)
    assert_allclose(out, out_ref, atol=1e-5)


def test_cdropout_conv2d_loss(conv2d_model):
    """Check if ConcreteDropout has the right amount of losses deposited
    """
    model = conv2d_model['model']

    assert_equal(len(model.losses), 1)


def test_cdropout_conv2d_loss_value(conv2d_model):
    """Check if the loss corresponds the the desired value
    """
    model = conv2d_model['model']
    X = conv2d_model['X']
    cd = conv2d_model['concrete_dropout']

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    W = model.get_weights()
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
