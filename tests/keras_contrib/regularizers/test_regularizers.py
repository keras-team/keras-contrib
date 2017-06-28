import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib import regularizers


test_signal = np.array([4, 1, 0, 1, 4])
example_matrix = np.array([test_signal, -test_signal, 2*test_signal,
                           2*test_signal, -test_signal, test_signal])

def test_curvature_regularizer():
    W = K.variable(value=example_matrix)
    # Zero regularization
    regularizer_instance0 = regularizers.smoothness(0.0, l1=True, second_diff=False)
    regularization0 = regularizer_instance0.__call__(W)
    assert(K.eval(regularization0) == 0.0)
    # L1 Norm on first differences 
    regularizer_instance1 = regularizers.smoothness(1.0, l1=True, second_diff=False)
    regularization1 = regularizer_instance1.__call__(W)
    assert(K.eval(regularization1) >= 2.6)
    assert(K.eval(regularization1) <= 2.7)
    # L1 Norm on second differences
    regularizer_instance2 = regularizers.smoothness(1.0, l1=True, second_diff=True)
    regularization2 = regularizer_instance2.__call__(W)
    assert(K.eval(regularization2) >= 2.6)
    assert(K.eval(regularization2) <= 2.7)
    # L2 Norm on first differences
    regularizer_instance3 = regularizers.smoothness(1.0, l1=False, second_diff=False)
    regularization3 = regularizer_instance3.__call__(W)
    assert(K.eval(regularization3) == 10.0)
    # L2 Norm on second differences 
    regularizer_instance4 = regularizers.smoothness(1.0, l1=False, second_diff=True)
    regularization4 = regularizer_instance4.__call__(W)
    assert(K.eval(regularization4) == 8.0)

if __name__ == '__main__':
    pytest.main([__file__])
