import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras_contrib import backend as K
from keras.models import Sequential
from keras_contrib.layers import convolutional_recurrent
from keras.utils.test_utils import layer_test


if __name__ == '__main__':
    pytest.main([__file__])
