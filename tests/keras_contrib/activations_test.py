import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib import activations


def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.
    '''
    return np.array([[0, 0.1, 0.5, 0.9, 1.0]], dtype=K.floatx())


if __name__ == '__main__':
    pytest.main([__file__])
