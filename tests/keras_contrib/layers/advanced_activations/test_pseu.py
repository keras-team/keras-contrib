# -*- coding: utf-8 -*-
import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import PSEU


@pytest.mark.parametrize('alpha_initial', [-0.1, 0., 0.1])
def test_pseu(alpha_initial):
    layer_test(PSEU,
               kwargs={'alpha_initial': alpha_initial},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
