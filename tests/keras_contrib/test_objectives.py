import numpy as np
import pytest
from keras import backend as K
from numpy.testing import assert_allclose

from keras_contrib import backend as KC
from keras_contrib import objectives

allobj = []


def test_objective_shapes_3d():
    y_a = K.variable(np.random.random((5, 6, 7)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)


def test_objective_shapes_2d():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (6,)


def test_dssim_same():
    x = np.random.random_sample(30 * 30 * 3).reshape([1, 30, 30, 3])
    x1 = KC.variable(x)
    loss = objectives.DSSIMObjective()
    assert_allclose([0.0], KC.eval(loss(x1, x1)), atol=1.0e-4)


def test_dssim_opposite():
    x = np.zeros([1, 30, 30, 3])
    x1 = KC.variable(x)
    y = np.ones([1, 30, 30, 3])
    y1 = KC.variable(y)
    loss = objectives.DSSIMObjective()
    assert_allclose([0.5], KC.eval(loss(x1, y1)), atol=1.0e-4)


def test_dssim_compile():
    from keras.models import Sequential
    from keras.layers import Convolution2D
    x = np.zeros([1, 30, 30, 3])
    loss = objectives.DSSIMObjective()
    model = Sequential()
    model.add(Convolution2D(3, 3, 3, border_mode="same", input_shape=(30, 30, 3)))
    model.compile("rmsprop", loss)
    model.fit([x], [x], 1, 1)


if __name__ == "__main__":
    pytest.main([__file__])
