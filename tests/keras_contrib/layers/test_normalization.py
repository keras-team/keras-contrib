import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Dense, Activation, Input
from keras.utils.test_utils import layer_test, keras_test
from keras_contrib.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K
from keras_contrib import backend as KC



if __name__ == '__main__':
    pytest.main([__file__])
