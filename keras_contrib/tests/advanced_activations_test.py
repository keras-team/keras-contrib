import pytest
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras import layers


@keras_test
def test_sine_relu():
    for epsilon in [0.0025, 0.0035, 0.0045]:
        layer_test(layers.SineReLU, kwargs={'epsilon': epsilon},
                   input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
