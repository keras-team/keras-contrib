import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib import constraints


test_values = [0.1, 0.5, 3, 8, 1e-7]
np.random.seed(3537)
example_array = np.random.random((100, 100)) * 100. - 50.
example_array[0, 0] = 0.  # 0 could possibly cause trouble


def test_clip():
    clip_instance = constraints.clip()
    clipped = clip_instance(K.variable(example_array))
    assert(np.max(np.abs(K.eval(clipped))) <= K.cast_to_floatx(0.01))
    clip_instance = constraints.clip(0.1)
    clipped = clip_instance(K.variable(example_array))
    assert(np.max(np.abs(K.eval(clipped))) <= K.cast_to_floatx(0.1))


test_signal = np.array([4, 1, 0, 1, 4])
example_matrix = np.array([test_signal, -test_signal, 2 * test_signal,
                           2 * test_signal, -test_signal, test_signal])


def test_curvature():
    # Signal curvature within acceptable range
    constraint_instance = constraints.curvature(m=20.0)
    W = K.variable(value=example_matrix)
    new_signal = constraint_instance.__call__(W)
    assert(np.array_equal(K.eval(new_signal), K.eval(W)))
    # Signal curvature exceeds acceptable range
    constraint_instance2 = constraints.curvature(m=1.0)
    new_signal2 = constraint_instance2.__call__(W)
    x = K.eval(new_signal2)
    diff1 = x[:, 1:] - x[:, :-1]
    diff2 = diff1[:, 1:] - diff1[:, :-1]
    assert(np.max(np.abs(diff2)) <= 1.0)

if __name__ == '__main__':
    pytest.main([__file__])
