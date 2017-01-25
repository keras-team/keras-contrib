import pytest
from keras.utils.test_utils import layer_test, keras_test
from keras_contrib import layers


@keras_test
def test_pelu():
    from keras_contrib.layers.advanced_activations import PELU
    layer_test(PELU, kwargs={},
               input_shape=(2, 3, 4))


@keras_test
def test_pelu_share():
    from keras_contrib.layers.advanced_activations import PELU
    layer_test(PELU, kwargs={'shared_axes': 1},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
