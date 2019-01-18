import pytest
import numpy as np
import os
import shutil
from keras.utils import to_categorical
from keras.layers import Layer, Input, Dense, Dropout, BatchNormalization
from keras_contrib.utils.test_utils import to_list, unpack_singleton
from keras_contrib.utils.test_utils import get_test_data
from keras import Model
from keras import backend as K
from keras_contrib.callbacks import TensorBoardGrouped

input_dim = 2
num_hidden = 4
num_classes = 2
batch_size = 5
train_samples = 20
test_samples = 20


def data_generator(x, y, batch_size):
    x = to_list(x)
    y = to_list(y)
    max_batch_index = len(x[0]) // batch_size
    i = 0
    while 1:
        x_batch = [array[i * batch_size: (i + 1) * batch_size] for array in x]
        x_batch = unpack_singleton(x_batch)

        y_batch = [array[i * batch_size: (i + 1) * batch_size] for array in y]
        y_batch = unpack_singleton(y_batch)
        yield x_batch, y_batch
        i += 1
        i = i % max_batch_index


# Changing the default arguments of get_test_data.
def get_data_callbacks(num_train=train_samples,
                       num_test=test_samples,
                       input_shape=(input_dim,),
                       classification=True,
                       num_classes=num_classes):
    return get_test_data(num_train=num_train,
                         num_test=num_test,
                         input_shape=input_shape,
                         classification=classification,
                         num_classes=num_classes)


def test_TensorBoard(tmpdir):
    np.random.seed(np.random.randint(1, 1e7))
    filepath = str(tmpdir / 'logs')

    (X_train, y_train), (X_test, y_test) = get_data_callbacks()
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)

    class DummyStatefulMetric(Layer):

        def __init__(self, name='dummy_stateful_metric', **kwargs):
            super(DummyStatefulMetric, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.state = K.variable(value=0, dtype='int32')

        def reset_states(self):
            pass

        def __call__(self, y_true, y_pred):
            return self.state

    inp = Input((input_dim,))
    hidden = Dense(num_hidden, activation='relu')(inp)
    hidden = Dropout(0.1)(hidden)
    hidden = BatchNormalization()(hidden)
    output = Dense(num_classes, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', DummyStatefulMetric()])

    # we must generate new callbacks for each test, as they aren't stateless
    def callbacks_factory(histogram_freq):
        return [TensorBoardGrouped(log_dir=filepath,
                                   histogram_freq=histogram_freq,
                                   write_images=True, write_grads=True,
                                   batch_size=5)]

    # fit without validation data
    model.fit(X_train, y_train, batch_size=batch_size,
              callbacks=callbacks_factory(histogram_freq=0),
              epochs=3)

    # fit with validation data and accuracy
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=callbacks_factory(histogram_freq=0), epochs=2)

    # fit generator without validation data
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        callbacks=callbacks_factory(histogram_freq=0))

    # fit generator with validation data and accuracy
    train_generator = data_generator(X_train, y_train, batch_size)
    model.fit_generator(train_generator, len(X_train), epochs=2,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks_factory(histogram_freq=1))

    assert os.path.isdir(filepath)
    shutil.rmtree(filepath)
    assert not tmpdir.listdir()


if __name__ == '__main__':
    pytest.main([__file__])
