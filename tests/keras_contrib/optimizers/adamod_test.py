from __future__ import print_function
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import Adamod


def test_adamod():
    optimiers._test_optimizer(Adamod())
    optimizers._test_optimizer(Adamod(beta_3=0.9999))
