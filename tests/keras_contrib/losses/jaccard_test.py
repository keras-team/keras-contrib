import pytest

from keras_contrib.losses import jaccard_distance
from keras_contrib.utils.test_utils import is_tf_keras
from keras import backend as K
import numpy as np


@pytest.mark.xfail(is_tf_keras,
                   reason='TODO fix this.',
                   strict=True)
def test_jaccard_distance():
    # all_right, almost_right, half_right, all_wrong
    y_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                       [0, 0, 1., 0.]])
    y_pred = np.array([[0, 0, 1, 0], [0, 0, 0.9, 0], [0, 0, 0.1, 0],
                       [1, 1, 0.1, 1.]])

    r = jaccard_distance(
        K.variable(y_true),
        K.variable(y_pred), )
    if K.is_keras_tensor(r):
        assert K.int_shape(r) == (4, )

    all_right, almost_right, half_right, all_wrong = K.eval(r)
    assert all_right == 0, 'should converge on zero'
    assert all_right < almost_right
    assert almost_right < half_right
    assert half_right < all_wrong


def test_jaccard_distance_shapes_3d():
    y_a = K.variable(np.random.random((5, 6, 7)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    objective_output = jaccard_distance(y_a, y_b)
    assert K.eval(objective_output).shape == (5, 6)


def test_jaccard_distance_shapes_2d():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    objective_output = jaccard_distance(y_a, y_b)
    assert K.eval(objective_output).shape == (6, )
