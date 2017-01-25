import pytest
import numpy as np

from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib import initializations


# 2D tensor test fixture
FC_SHAPE = (100, 100)

# 4D convolution in th order. This shape has the same effective shape as FC_SHAPE
CONV_SHAPE = (25, 25, 2, 2)

# The equivalent shape of both test fixtures
SHAPE = (100, 100)


def _runner(init, shape, target_mean=None, target_std=None,
            target_max=None, target_min=None):
    variable = init(shape)
    output = K.get_value(variable)
    lim = 1e-2
    if target_std is not None:
        assert abs(output.std() - target_std) < lim
    if target_mean is not None:
        assert abs(output.mean() - target_mean) < lim
    if target_max is not None:
        assert abs(output.max() - target_max) < lim
    if target_min is not None:
        assert abs(output.min() - target_min) < lim


'''
# Example :

@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_uniform(tensor_shape):
    _runner(initializations.uniform, tensor_shape, target_mean=0.,
            target_max=0.05, target_min=-0.05)

'''


if __name__ == '__main__':
    pytest.main([__file__])
