import pytest
from keras.utils.test_utils import layer_test, keras_test
from keras_contrib.layers import advanced_activations


@keras_test
def test_pelu():
    layer_test(advanced_activations.PELU, kwargs={},
               input_shape=(2, 3, 4))


@keras_test
def test_pelu_share():
    layer_test(advanced_activations.PELU, kwargs={'shared_axes': 1},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
