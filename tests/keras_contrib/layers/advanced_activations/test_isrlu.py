# -*- coding: utf-8 -*-
import pytest
from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers import ISRLU


@pytest.mark.parametrize('alpha', [0.2, 0.3, 0.01])
def test_isrlu(alpha):
    layer_test(ISRLU,
               kwargs={'alpha': alpha},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
