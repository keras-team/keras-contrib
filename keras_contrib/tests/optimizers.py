from __future__ import print_function

import numpy as np
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.utils.test_utils import get_test_data

np.random.seed(1337)

(X_train, y_train), (X_test, y_test) = get_test_data(num_train=1000,
                                                     num_test=200,
                                                     input_shape=(10,),
                                                     classification=True,
                                                     num_classes=2)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def get_model(input_dim, nb_hidden, output_dim):
    model = Sequential()
    model.add(Dense(nb_hidden, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


def validate_optimizer(optimizer, target=0.89):
    model = get_model(X_train.shape[1], 10, y_train.shape[1])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=12, batch_size=16,
                        validation_data=(X_test, y_test), verbose=2)
    config = optimizer.get_config()
    assert type(config) == dict
    assert history.history['val_acc'][-1] >= target
