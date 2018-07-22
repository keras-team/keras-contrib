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


@keras_test
def test_srelu():
    layer_test(advanced_activations.SReLU, kwargs={},
               input_shape=(2, 3, 4))


@keras_test
def test_srelu_share():
    layer_test(advanced_activations.SReLU, kwargs={'shared_axes': 1},
               input_shape=(2, 3, 4))


@keras_test
def test_swish_constant():
    layer_test(advanced_activations.Swish, kwargs={'beta': 1.0, 'trainable': False},
               input_shape=(2, 3, 4))


@keras_test
def test_swish_trainable():
    layer_test(advanced_activations.Swish, kwargs={'beta': 1.0, 'trainable': True},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
