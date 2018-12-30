from __future__ import print_function
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import Padam


def test_padam():
    optimizers._test_optimizer(Padam())
    optimizers._test_optimizer(Padam(decay=1e-3))
