import pytest
import numpy as np

from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import keras_test
from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib.layers import SpatialTransformer
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Activation
np.random.seed(1)


@keras_test
def test_spatial_transformer_network():
    x_train = np.ones((1, 60, 60, 1))

    b = np.zeros((2, 3))
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()

    W = np.zeros((50, 6))
    weights = [W, b]

    localisation_network = Sequential()

    localisation_network.add(MaxPooling2D(pool_size=(2, 2), input_shape=x_train.shape[1:]))
    localisation_network.add(Conv2D(20, (5, 5)))
    localisation_network.add(Flatten())
    localisation_network.add(Dense(50))
    localisation_network.add(Activation('relu'))
    localisation_network.add(Dense(6, weights=weights))

    model = Sequential()

    model.add(SpatialTransformer(localisation_network=localisation_network,
                                 output_dim=(30, 30), input_shape=x_train.shape[1:]))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    out = model.predict(np.ones_like(x_train))

    assert_allclose(out, np.array([[0.11505639, 0.11439, 0.10204504, 0.09712951, 0.09869344,
                                    0.06929098, 0.07713472, 0.08842108, 0.11671967, 0.12111916]]))


if __name__ == '__main__':
    pytest.main([__file__])
