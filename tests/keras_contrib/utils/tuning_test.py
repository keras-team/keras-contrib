import pytest
from scipy.stats.distributions import uniform

import keras
from keras.utils.test_utils import get_test_data
from keras.utils.test_utils import keras_test
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, InputLayer, Dropout, Input, concatenate
from keras.utils.np_utils import to_categorical

from keras_contrib.utils.tuning import create_grid_configurations, create_random_configurations, SearchWrapper


# Simple callback to demonstrate that callback state is preserved even when model is dumped to disk.
class DummyCallback(keras.callbacks.Callback):
    def __init__(self):
        super(DummyCallback, self).__init__()
        self.total_batches = 0
        self.total_epochs = 0

    def on_batch_end(self, batch, logs=None):
        self.total_batches += 1

    def on_epoch_end(self, epoch, logs=None):
        self.total_epochs += 1


@keras_test
def test_sequential_model_grid():
    """
    Test grid search for Sequential() models. Both generator and non-generator case are tested.
    Only one model is stored in memory at a time.
    """
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=100,
                                                         num_test=100,
                                                         input_shape=(28, 28, 3),
                                                         classification=True,
                                                         num_classes=10)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    gen = ImageDataGenerator().flow(X_train, y_train)
    gen_test = ImageDataGenerator().flow(X_test, y_test)

    def example_network(a=.1, b=1):
        nn = Sequential()
        nn.add(InputLayer(input_shape=(28, 28, 3)))
        nn.add(Flatten())
        nn.add(Dropout(a))
        nn.add(Dense(b))
        nn.add(Dense(10, activation='softmax'))
        nn.compile(optimizer='sgd', loss='categorical_crossentropy')
        return nn

    search_grid = {
        'a': [.1, .2],
        'b': [1]
    }

    configs = create_grid_configurations(search_grid)
    callbacks = [DummyCallback() for _ in configs]
    models = [SearchWrapper(example_network, callbacks=[cbk], **hyperparameters)
              for hyperparameters, cbk in zip(configs, callbacks)]

    for m in models:
        m.fit(X_train, y_train, epochs=2, verbose=0)  # 4 steps per epoch (barely), two epochs
        assert m.model is None  # Ensure model is not in memory
        m.evaluate(X_test, y_test, verbose=0)

    for m in models:
        m.fit_generator(gen, steps_per_epoch=3, epochs=2, verbose=0)  # 3 steps per epoch is close enough, two epochs
        assert m.model is None  # Ensure model is not in memory
        m.evaluate_generator(gen_test, steps=3)

    # Make sure callbacks persist even when model is dumped to disk
    assert callbacks[0].total_epochs == 4
    assert callbacks[0].total_batches == 14


@keras_test
def test_sequential_model_random():
    """
    Test random search for Sequential() models. Non-generator and partial_fit methods are tested.
    Only one model is stored in memory at a time.
    """
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=100,
                                                         num_test=100,
                                                         input_shape=(28, 28, 3),
                                                         classification=True,
                                                         num_classes=10)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    gen = ImageDataGenerator().flow(X_train, y_train)
    gen_test = ImageDataGenerator().flow(X_test, y_test)

    def example_network(a=.1, b=1):
        nn = Sequential()
        nn.add(InputLayer(input_shape=(28, 28, 3)))
        nn.add(Flatten())
        nn.add(Dropout(a))
        nn.add(Dense(b))
        nn.add(Dense(10, activation='softmax'))
        nn.compile(optimizer='sgd', loss='categorical_crossentropy')
        return nn

    search_grid = {
        'a': uniform(0, .5),
        'b': [1, 2]
    }

    configs = create_random_configurations(search_grid, 3)
    callbacks = [DummyCallback() for _ in configs]
    models = [SearchWrapper(example_network, callbacks=[cbk], **hyperparameters)
              for hyperparameters, cbk in zip(configs, callbacks)]

    assert len(models) == 3

    for m in models:
        m.fit(X_train, y_train, epochs=2, verbose=0)  # 4 steps per epoch (barely), two epochs
        assert m.model is None  # Ensure model is not in memory
        m.evaluate(X_test, y_test, verbose=0)

    for m in models:
        m.start_train_partial(steps_per_epoch=3)
        for i in range(6):
            m.partial_fit(*next(gen))
        assert m.model is not None  # Model should be in memory in case we want to keep calling fit_partial
        m.save_and_clear_model()
        assert m.model is None  # Ensure model is not in memory now
        m.evaluate_generator(gen_test, steps=3)

    # Make sure callbacks persist even when model is dumped to disk
    assert callbacks[0].total_epochs == 4
    assert callbacks[0].total_batches == 14


@keras_test
def test_graphical_model():
    """
    Test random search for Model() models. Both generator and partial fit case are tested.
    Only one model is stored in memory at a time.
    """
    (X_train, y_train), (X_test, y_test) = get_test_data(num_train=100,
                                                         num_test=100,
                                                         input_shape=(28, 28, 3),
                                                         classification=True,
                                                         num_classes=10)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    gen = ImageDataGenerator().flow(X_train, y_train)
    # gen_test = ImageDataGenerator().flow(X_test, y_test)

    def example_network(a=.1, b=1):
        inp1 = Input(shape=(28, 28, 3))
        inp2 = Input(shape=(28, 28, 3))
        nn = concatenate([inp1, inp2])
        nn = Flatten()(nn)
        nn = Dropout(a)(nn)
        nn = Dense(b)(nn)
        outp = Dense(10, activation='softmax')(nn)

        model = Model(inputs=[inp1, inp2], outputs=[outp])
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        return model

    search_grid = {
        'a': uniform(0, .5),
        'b': [1, 2]
    }

    configs = create_random_configurations(search_grid, 3)
    callbacks = [DummyCallback() for _ in configs]
    models = [SearchWrapper(example_network, callbacks=[cbk], **hyperparameters)
              for hyperparameters, cbk in zip(configs, callbacks)]

    assert len(models) == 3

    for m in models:
        m.fit([X_train, X_train], y_train, epochs=2, verbose=0)  # 4 steps per epoch (barely), two epochs
        assert m.model is None  # Ensure model is not in memory
        m.evaluate([X_test, X_test], y_test, verbose=0)

    for m in models:
        m.start_train_partial(steps_per_epoch=3)
        for i in range(6):
            X, y = next(gen)
            m.partial_fit([X, X], [y])
        assert m.model is not None  # Model should be in memory in case we want to keep calling fit_partial
        m.save_and_clear_model()
        assert m.model is None  # Ensure model is not in memory now

        # m.evaluate_generator(map(lambda g: ([g[0], g[0]], g[1]), gen_test), steps=3) # Python 3
        m.evaluate([X_test, X_test], y_test, verbose=0)

    # Make sure callbacks persist even when model is dumped to disk
    assert callbacks[0].total_epochs == 4
    assert callbacks[0].total_batches == 14


def test_configuration_creators():
    """
    Test to make sure configurations are properly being generated.
    """
    search_grid = {
        'side': ['salad', 'rice', 'potatoes'],
        'entree': ['steak', 'salmon', 'curry'],
        'meal': ['dinner']
    }

    configs = create_grid_configurations(search_grid)
    assert len(configs) == 3 ** 2
    assert {'side': 'rice', 'entree': 'salmon',
            'meal': 'dinner'} in configs  # Grid will always contain every config combo

    # Now make sure scipy sampling works, as well as random configs:
    search_random = {
        'seconds': uniform(-100, 200),
        'task': ['eat', 'sleep', 'write unit tests']
    }

    configs = create_random_configurations(search_random, 42)
    assert len(configs) == 42


if __name__ == '__main__':
    pytest.main([__file__])
