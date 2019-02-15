from __future__ import print_function
import pytest
from keras_contrib.utils.test_utils import is_tf_keras
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import ftml


@pytest.mark.xfail(is_tf_keras,
                   reason='TODO fix this.',
                   strict=True)
def test_ftml():
    optimizers._test_optimizer(ftml())
    optimizers._test_optimizer(ftml(lr=0.003, beta_1=0.8,
                                    beta_2=0.9, epsilon=1e-5,
                                    decay=1e-3))
