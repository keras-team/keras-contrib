from __future__ import print_function
import pytest
import numpy as np

from keras.utils import test_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical


def get_test_data():
    np.random.seed(1337)
    (x_train, y_train), _ = test_utils.get_test_data(num_train=1000,
                                                     num_test=200,
                                                     input_shape=(10,),
                                                     classification=True,
                                                     num_classes=2)
    y_train = to_categorical(y_train)
    return x_train, y_train


def get_model(input_dim, num_hidden, output_dim):
    model = Sequential()
    model.add(Dense(num_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


def _test_optimizer(optimizer, target=0.75):
    x_train, y_train = get_test_data()
    model = get_model(x_train.shape[1], 10, y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    assert history.history['acc'][-1] >= target
    config = optimizers.serialize(optimizer)
    optim = optimizers.deserialize(config)
    new_config = optimizers.serialize(optim)
    assert config == new_config
