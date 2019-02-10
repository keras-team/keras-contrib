import numpy as np
import pytest
from keras import backend as K

from keras_contrib import backend as KC
from keras_contrib.layers import SubPixelUpscaling
from keras_contrib.utils.test_utils import layer_test

# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_border_modes = ['valid', 'same']
    data_format = 'channels_first'
else:
    _convolution_border_modes = ['valid', 'same']
    data_format = 'channels_last'


@pytest.mark.parametrize('scale_factor', [2, 3, 4])
def test_sub_pixel_upscaling(scale_factor):
    num_samples = 2
    num_row = 16
    num_col = 16
    input_dtype = K.floatx()

    nb_channels = 4 * (scale_factor ** 2)
    input_data = np.random.random((num_samples, nb_channels, num_row, num_col))
    input_data = input_data.astype(input_dtype)

    if K.image_data_format() == 'channels_last':
        input_data = input_data.transpose((0, 2, 3, 1))

    input_tensor = K.variable(input_data)
    expected_output = K.eval(KC.depth_to_space(input_tensor,
                                               scale=scale_factor))

    layer_test(SubPixelUpscaling,
               kwargs={'scale_factor': scale_factor},
               input_data=input_data,
               expected_output=expected_output,
               expected_output_dtype=K.floatx())


if __name__ == '__main__':
    pytest.main([__file__])
