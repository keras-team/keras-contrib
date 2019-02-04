import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import advanced_activations


@pytest.mark.parametrize('kwargs', [
    {},
    {'shared_axes': 1}
])
def test_pelu(kwargs):
    layer_test(advanced_activations.PELU, kwargs=kwargs,
               input_shape=(2, 3, 4))


def test_sine_relu():
    for epsilon in [0.0025, 0.0035, 0.0045]:
        layer_test(advanced_activations.SineReLU, kwargs={'epsilon': epsilon},
                   input_shape=(2, 3, 4))


@pytest.mark.parametrize('kwargs', [
    {},
    {'shared_axes': 1}
])
def test_srelu(kwargs):
    layer_test(advanced_activations.SReLU, kwargs=kwargs,
               input_shape=(2, 3, 4))


@pytest.mark.parametrize('trainable', [False, True])
def test_swish(trainable):
    layer_test(advanced_activations.Swish, kwargs={'beta': 1.0, 'trainable': trainable},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
