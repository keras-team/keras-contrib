import pytest
import warnings
import numpy as np

from keras_contrib import callbacks
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K


def test_DeadDeadReluDetector():
    n_samples = 9

    input_shape = (n_samples, 3, 4)  # 4 input features
    shape_out = (n_samples, 3, 10)  # 10 output features
    shape_weights = (4, 10)

    # ignore batch size
    input_shape_dense = tuple(input_shape[1:])

    def do_test(weights, expected_warnings, verbose):
        with warnings.catch_warnings(record=True) as w:
            dataset = np.ones(input_shape)    # data to be fed as training
            model = Sequential()
            model.add(Dense(10, activation='relu', input_shape=input_shape_dense,
                            use_bias=False, weights=[weights], name='dense'))
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones(shape_out),
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )
            assert len(w) == expected_warnings
            for warn_item in w:
                assert issubclass(warn_item.category, RuntimeWarning)
                assert "dead neurons" in str(warn_item.message)

    weights_1_dead = np.ones(shape_weights)  # weights that correspond to NN with 1/10 neurons dead
    weights_2_dead = np.ones(shape_weights)  # weights that correspond to NN with 2/10 neurons dead

    weights_1_dead[:, 0] = 0
    weights_2_dead[:, 0:2] = 0

    do_test(weights_1_dead, verbose=True, expected_warnings=1)
    do_test(weights_1_dead, verbose=False, expected_warnings=0)
    do_test(weights_2_dead, verbose=True, expected_warnings=1)


def test_DeadDeadReluDetector_bias():
    n_samples = 9

    input_shape = (n_samples, 4)  # 4 input features
    shape_weights = (4, 10)
    shape_out = (n_samples, 10)  # 10 output features
    shape_bias = (10, )

    # ignore batch size
    input_shape_dense = tuple(input_shape[1:])

    def do_test(weights, bias, expected_warnings, verbose):
        with warnings.catch_warnings(record=True) as w:
            dataset = np.ones(input_shape)  # data to be fed as training
            model = Sequential()
            model.add(Dense(10, activation='relu', input_shape=input_shape_dense,
                            use_bias=True, weights=[weights, bias], name='dense'))
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            # model.compile(optimizer=None, loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones(shape_out),
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )
            assert len(w) == expected_warnings
            for warn_item in w:
                assert issubclass(warn_item.category, RuntimeWarning)
                assert "dead neurons" in str(warn_item.message)

    weights_1_dead = np.ones(shape_weights)  # weights that correspond to NN with 1/10 neurons dead
    weights_2_dead = np.ones(shape_weights)  # weights that correspond to NN with 2/10 neurons dead

    weights_1_dead[:, 0] = 0
    weights_2_dead[:, 0:2] = 0

    bias = np.zeros(shape_bias)

    do_test(weights_1_dead, bias, verbose=True, expected_warnings=1)
    do_test(weights_1_dead, bias, verbose=False, expected_warnings=0)
    do_test(weights_2_dead, bias, verbose=True, expected_warnings=1)


def test_DeadDeadReluDetector_conv():
    n_samples = 9

    # (5, 5) kernel, 4 input featuremaps and 10 output featuremaps
    if K.image_data_format() == 'channels_last':
        input_shape = (n_samples, 5, 5, 4)
    else:
        input_shape = (n_samples, 4, 5, 5)

    # ignore batch size
    input_shape_conv = tuple(input_shape[1:])
    shape_weights = (5, 5, 4, 10)
    shape_out = (n_samples, 10)

    def do_test(weights_bias, expected_warnings, verbose):
        with warnings.catch_warnings(record=True) as w:

            dataset = np.ones(input_shape)    # data to be fed as training
            model = Sequential()
            model.add(Conv2D(10, (5, 5), activation='relu', input_shape=input_shape_conv,
                             use_bias=True, weights=weights_bias, name='conv'))
            model.add(Flatten())    # to handle Theano's categorical crossentropy
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones(shape_out),
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )
            assert len(w) == expected_warnings
            for warn_item in w:
                assert issubclass(warn_item.category, RuntimeWarning)
                assert "dead neurons" in str(warn_item.message)

    weights_1_dead = np.ones(shape_weights)      # weights that correspond to NN with 1/10 neurons dead
    weights_1_dead[..., 0] = 0
    weights_2_dead = np.ones(shape_weights)    # weights that correspond to NN with 2/10 neurons dead
    weights_2_dead[..., 0:2] = 0

    bias = np.zeros((10, ))

    weights_bias_1_dead = [weights_1_dead, bias]
    weights_bias_2_dead = [weights_2_dead, bias]

    do_test(weights_bias_1_dead, verbose=True, expected_warnings=1)
    do_test(weights_bias_1_dead, verbose=False, expected_warnings=0)
    do_test(weights_bias_2_dead, verbose=True, expected_warnings=1)


if __name__ == '__main__':
    pytest.main([__file__])
