# -*- coding: utf-8 -*-
import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import PSEU


@pytest.mark.parametrize('trainable', [True, False])
@pytest.mark.parametrize('alpha_init', [-0.1, 0., 0.1])
@pytest.mark.parametrize('initializer', ['glorot_uniform', None])
def test_pseu(alpha_init,
              trainable,
              initializer):
    layer_test(PSEU,
               kwargs={'trainable': trainable,
                       'alpha_init': alpha_init,
                       'initializer': initializer},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
