import pytest
import warnings
import numpy as np

from keras_contrib import callbacks
from keras.models import Sequential
from keras.layers import Dense


def test_DeadDeadReluDetector():
    def do_test(weights, expected_warnings, verbose):
        with warnings.catch_warnings(record=True) as w:
            dataset = np.ones((1, 1, 1))    # data to be fed as training
            model = Sequential()
            model.add(Dense(10, activation='relu', input_shape=(1, 1), use_bias=False, weights=[weights]))
            model.compile(optimizer='sgd', loss='categorical_crossentropy')
            model.fit(
                dataset,
                np.ones((1, 1, 10)),
                epochs=1,
                callbacks=[callbacks.DeadReluDetector(dataset, verbose=verbose)],
                verbose=False
            )
            assert len(w) == expected_warnings
            for warn_item in w:
                assert issubclass(warn_item.category, RuntimeWarning)
                assert "dead neurons" in str(warn_item.message)

    weights_1_dead = np.ones((1, 10))      # weights that correspond to NN with 1/10 neurons dead
    weights_1_dead[:, 0] = 0
    weights_2_dead = np.ones((1, 10))      # weights that correspond to NN with 2/10 neurons dead
    weights_2_dead[:, 0] = 0
    weights_2_dead[:, 1] = 0

    do_test(weights_1_dead, verbose=True, expected_warnings=1)
    do_test(weights_1_dead, verbose=False, expected_warnings=0)
    do_test(weights_2_dead, verbose=True, expected_warnings=1)


if __name__ == '__main__':
    pytest.main([__file__])
