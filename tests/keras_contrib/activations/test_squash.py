from keras_contrib import activations
import keras.backend as K
import numpy as np
from numpy.testing import assert_allclose


def get_standard_values():
    """A set of floats used for testing squash.
    """
    return np.array([[0, 0.1, 0.5, 0.9, 1.0]], dtype=K.floatx())


def test_squash_valid():
    """Test using a reference implementation of squash.
    """
    def squash(x, axis=-1):
        s_squared_norm = np.sum(np.square(x), axis) + 1e-7
        scale = np.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
        return scale * x

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.squash(x)])
    test_values = get_standard_values()

    result = f([test_values])[0]
    expected = squash(test_values)
    assert_allclose(result, expected, rtol=1e-05)


test_squash_valid()
