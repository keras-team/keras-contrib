import pytest
import numpy as np
from keras_contrib.utils.test_utils import is_tf_keras
from numpy.testing import assert_allclose
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam

from keras.losses import sparse_categorical_crossentropy
from keras import backend as K
from keras_contrib.losses import DSSIMObjective

allobj = []


def test_objective_shapes_3d():
    y_a = K.variable(np.random.random((5, 6, 7)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)


def test_objective_shapes_2d():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (6,)


def test_cce_one_hot():
    y_a = K.variable(np.random.randint(0, 7, (5, 6)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    objective_output = sparse_categorical_crossentropy(y_a, y_b)
    assert K.eval(objective_output).shape == (5, 6)

    y_a = K.variable(np.random.randint(0, 7, (6,)))
    y_b = K.variable(np.random.random((6, 7)))
    assert K.eval(sparse_categorical_crossentropy(y_a, y_b)).shape == (6,)


def test_DSSIM_channels_last():
    prev_data = K.image_data_format()
    K.set_image_data_format('channels_last')
    for input_dim, kernel_size in zip([32, 33], [2, 3]):
        input_shape = [input_dim, input_dim, 3]
        X = np.random.random_sample(4 * input_dim * input_dim * 3)
        X = X.reshape([4] + input_shape)
        y = np.random.random_sample(4 * input_dim * input_dim * 3)
        y = y.reshape([4] + input_shape)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape,
                         activation='relu'))
        model.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape,
                         activation='relu'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=DSSIMObjective(kernel_size=kernel_size),
                      metrics=['mse'],
                      optimizer=adam)
        model.fit(X, y, batch_size=2, epochs=1, shuffle='batch')

        # Test same
        x1 = K.constant(X, 'float32')
        x2 = K.constant(X, 'float32')
        dssim = DSSIMObjective(kernel_size=kernel_size)
        assert_allclose(0.0, K.eval(dssim(x1, x2)), atol=1e-4)

        # Test opposite
        x1 = K.zeros([4] + input_shape)
        x2 = K.ones([4] + input_shape)
        dssim = DSSIMObjective(kernel_size=kernel_size)
        assert_allclose(0.5, K.eval(dssim(x1, x2)), atol=1e-4)

    K.set_image_data_format(prev_data)


@pytest.mark.xfail(is_tf_keras,
                   reason='TODO fix this.',
                   strict=True)
def test_DSSIM_channels_first():
    prev_data = K.image_data_format()
    K.set_image_data_format('channels_first')
    for input_dim, kernel_size in zip([32, 33], [2, 3]):
        input_shape = [3, input_dim, input_dim]
        X = np.random.random_sample(4 * input_dim * input_dim * 3)
        X = X.reshape([4] + input_shape)
        y = np.random.random_sample(4 * input_dim * input_dim * 3)
        y = y.reshape([4] + input_shape)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape,
                         activation='relu'))
        model.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape,
                         activation='relu'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=DSSIMObjective(kernel_size=kernel_size), metrics=['mse'],
                      optimizer=adam)
        model.fit(X, y, batch_size=2, epochs=1, shuffle='batch')

        # Test same
        x1 = K.constant(X, 'float32')
        x2 = K.constant(X, 'float32')
        dssim = DSSIMObjective(kernel_size=kernel_size)
        assert_allclose(0.0, K.eval(dssim(x1, x2)), atol=1e-4)

        # Test opposite
        x1 = K.zeros([4] + input_shape)
        x2 = K.ones([4] + input_shape)
        dssim = DSSIMObjective(kernel_size=kernel_size)
        assert_allclose(0.5, K.eval(dssim(x1, x2)), atol=1e-4)

    K.set_image_data_format(prev_data)


if __name__ == '__main__':
    pytest.main([__file__])
