import pytest
import numpy as np

import keras
from keras import backend as K
from keras_contrib.metrics import segmentation_metrics


shaped_obj = [segmentation_metrics.pixel_accuracy,
              segmentation_metrics.mean_accuracy,
              segmentation_metrics.mean_intersection_over_union]


def test_objective_shapes():
    shapes = [(5, 6, 7), (4, 5, 6, 7)]
    for shape in shapes:
        y_a = K.variable(np.random.random(shape))
        y_b = K.variable(np.random.random(shape))
        for obj in shaped_obj:
            print(obj.__name__)
            objective_output = obj(y_a, y_b)
            assert K.eval(objective_output).shape == tuple()


if __name__ == '__main__':
    pytest.main([__file__])
