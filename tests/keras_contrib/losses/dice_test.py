import numpy as np
import pytest
from keras import backend as K
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from numpy.testing import assert_allclose

from keras_contrib.losses import dice_score, dice_loss

allobj = [dice_score, dice_loss]


def test_objective_value_2d():
    y_a = K.variable(np.ones((1, 3, 3)))
    y_b = K.variable(np.expand_dims(np.eye(3), 0))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        print(K.eval(objective_output))
        assert_allclose(0.5, K.eval(objective_output), atol=1e-4)


def test_DICE_channels_last():
    prev_data = K.image_data_format()
    K.set_image_data_format('channels_last')
    for input_dim, kernel_size in zip([32, 33], [2, 3]):
        input_shape = [input_dim, input_dim, 3]
        X = np.random.random_sample(4 * input_dim * input_dim * 3).reshape([4] + input_shape)
        y = np.random.random_sample(4 * input_dim * input_dim * 3).reshape([4] + input_shape)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=dice_loss, metrics=['mse'], optimizer=adam)
        model.fit(X, y, batch_size=2, epochs=1, shuffle='batch')

        # Test same
        x1 = K.constant(X > 0.5, 'float32')
        x2 = K.constant(X > 0.5, 'float32')
        assert_allclose(1.0, K.eval(dice_score(x1, x2)), atol=1e-4)

        # Test opposite
        x1 = K.zeros([4] + input_shape)
        x2 = K.ones([4] + input_shape)
        assert_allclose(0.0, K.eval(dice_score(x1, x2)), atol=1e-4)

    K.set_image_data_format(prev_data)


def test_DICE_channels_first():
    prev_data = K.image_data_format()
    K.set_image_data_format('channels_first')
    for input_dim, kernel_size in zip([32, 33], [2, 3]):
        input_shape = [3, input_dim, input_dim]
        X = np.random.random_sample(4 * input_dim * input_dim * 3).reshape([4] + input_shape)
        y = np.random.random_sample(4 * input_dim * input_dim * 3).reshape([4] + input_shape)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=dice_loss, metrics=['mse'], optimizer=adam)
        model.fit(X, y, batch_size=2, epochs=1, shuffle='batch')

        # Test same
        x1 = K.constant(X > 0.5, 'float32')
        x2 = K.constant(X > 0.5, 'float32')
        assert_allclose(1.0, K.eval(dice_score(x1, x2)), atol=1e-4)

        # Test opposite
        x1 = K.zeros([4] + input_shape)
        x2 = K.ones([4] + input_shape)
        assert_allclose(0, K.eval(dice_score(x1, x2)), atol=1e-4)

    K.set_image_data_format(prev_data)


if __name__ == '__main__':
    pytest.main([__file__])
