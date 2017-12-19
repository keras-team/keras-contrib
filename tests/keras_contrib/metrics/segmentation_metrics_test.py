import pytest
import numpy as np

import keras
from keras import backend as K
from keras_contrib.losses import segmentation_metrics


allobj = [segmentation_metrics.pixel_accuracy,
          segmentation_metrics.mean_accuracy,
          segmentation_metrics.mean_intersection_over_union]


def test_objective_shapes_3d():
    y_a = K.variable(np.random.random((5, 6, 7)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)


def test_objective_shapes_4d():
    y_a = K.variable(np.random.random((4, 5, 6, 7)))
    y_b = K.variable(np.random.random((4, 5, 6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)


def test_objective_shapes_2d():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (6,)


if __name__ == '__main__':
    pytest.main([__file__])
