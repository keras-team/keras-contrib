import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import advanced_activations


@pytest.mark.parametrize('kwargs', [{}, {'shared_axes': 1}])
def test_pseu(kwargs):
    layer_test(advanced_activations.PSEU,
               kwargs={'alpha_init': 0.1},
               input_shape=(2, 3, 4))
