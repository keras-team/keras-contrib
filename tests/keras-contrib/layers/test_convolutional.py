import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras.utils.np_utils import conv_input_length
from keras import backend as K
from keras_contrib import backend as KC
from keras_contrib.layers import convolutional, pooling


# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_border_modes = ['valid', 'same', 'full']
else:
    _convolution_border_modes = ['valid', 'same']


if __name__ == '__main__':
    pytest.main([__file__])
