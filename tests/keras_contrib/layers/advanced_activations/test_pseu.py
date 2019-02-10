# -*- coding: utf-8 -*-
import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import PSEU


@pytest.mark.parametrize('trainable', [True, False])
def test_pseu(trainable,
              alpha_init):
    layer_test(PSEU,
               kwargs={'trainable': trainable},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
