import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import SineReLU


@pytest.mark.parametrize('epsilon', [0.0025, 0.0035, 0.0045])
def test_sine_relu(epsilon):
    layer_test(SineReLU, kwargs={'epsilon': epsilon}, input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
