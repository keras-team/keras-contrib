import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import SReLU


@pytest.mark.parametrize('kwargs', [{}, {'shared_axes': 1}])
def test_srelu(kwargs):
    layer_test(SReLU, kwargs=kwargs, input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
