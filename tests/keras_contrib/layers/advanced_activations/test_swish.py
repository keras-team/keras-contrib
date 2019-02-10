import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import Swish


@pytest.mark.parametrize('trainable', [False, True])
def test_swish(trainable):
    layer_test(Swish, kwargs={'beta': 1.0, 'trainable': trainable},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
