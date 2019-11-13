import pytest

from keras_contrib.losses import dice_loss
from keras_contrib.utils.test_utils import is_tf_keras
from keras import backend as K
import numpy as np


def test_dice_loss_shapes_scalar():
    y_true = np.random.randn(3, 4)
    y_pred = np.random.randn(3, 4)

    L = dice_loss(
        K.variable(y_true),
        K.variable(y_pred), )
    assert K.is_tensor(L), 'should be a Tensor'
    assert L.shape == ()
    assert K.eval(L).shape == ()


def test_dice_loss_for_same_array():
    y_true = np.random.randn(3, 4)
    y_pred = y_true.copy()

    L = dice_loss(
        K.variable(y_true),
        K.variable(y_pred), )
    assert K.eval(L) == 0, 'loss should be zero'


def test_dice_loss_for_zero_array():
    y_true = np.array([1])
    y_pred = np.array([0])

    L = dice_loss(
        K.variable(y_true),
        K.variable(y_pred), )
    assert K.eval(L) == 0.5, 'loss should equal 0.5'

