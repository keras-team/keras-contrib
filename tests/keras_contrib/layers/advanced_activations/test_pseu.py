import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import PSEU


@pytest.mark.parametrize('trainable', [True, False])
@pytest.mark.parametrize('alpha_init', [-0.1, 0, 0.1])
def test_pseu(trainable,
              alpha_init):
    layer_test(PSEU,
               kwargs={'alpha_init': alpha_init,
                       'trainable': trainable},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
