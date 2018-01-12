from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import get_test_data
from keras import backend as K
from keras.utils import np_utils
from keras_contrib import callbacks

import numpy as np
import pytest

input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20


def test_LearningRateWarmRestarter():
    np.random.seed(1337)
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=train_samples,
                                                         num_test=test_samples,
                                                         input_shape=(input_dim,),
                                                         classification=True,
                                                         num_classes=num_classes)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    model = Sequential()
    model.add(Dense(num_hidden, input_dim=input_dim))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    max_lr = 0.1
    cbks = [callbacks.LearningRateWarmRestarter(min_lr=0., max_lr=max_lr, num_restart_epochs=5, factor=2)]
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cbks, epochs=6)
    assert (float(K.get_value(model.optimizer.lr)) - max_lr) < K.epsilon()

if __name__ == '__main__':
    pytest.main([__file__])
