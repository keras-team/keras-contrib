"""Tests for functions in keras_contrib/layers/functional.py."""

import pytest

from keras.layers import Dense, Input
from keras.models import Model
from keras_contrib.layers import repeat, sequence


def test_sequence():
    input_layer = Input(shape=(16,))
    output = sequence(
        Dense(8),
        Dense(1),
    )(input_layer)
    model = Model(input_layer, output)
    assert len(model.layers) == 3
    assert model.layers[1].__class__.__name__ == 'Dense'
    assert model.layers[2].__class__.__name__ == 'Dense'
    assert model.layers[1].get_output_shape_at(0) == (None, 8)
    assert model.layers[2].get_output_shape_at(0) == (None, 1)


def test_repeat():
    input_layer = Input(shape=(16,))
    output = repeat(2, lambda: Dense(8))(input_layer)
    model = Model(input_layer, output)
    assert len(model.layers) == 3
    assert model.layers[1].__class__.__name__ == 'Dense'
    assert model.layers[2].__class__.__name__ == 'Dense'
    assert id(model.layers[1]) != id(model.layers[2])


if __name__ == '__main__':
    pytest.main([__file__])
