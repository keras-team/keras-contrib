import pytest
import numpy as np

import keras
from keras import backend as K
from keras_contrib.losses import segmentation_losses


shaped_obj = [segmentation_losses.binary_crossentropy]


def test_objective_shapes():
    shapes = [(5, 6, 7), (4, 5, 6, 7)]
    for shape in shapes:
        y_a = K.variable(np.random.random(shape))
        y_b = K.variable(np.random.random(shape))
        for obj in shaped_obj:
            print(obj.__name__)
            objective_output = obj(y_a, y_b)
            assert K.eval(objective_output).shape == shape[:-1]


if __name__ == '__main__':
    pytest.main([__file__])
