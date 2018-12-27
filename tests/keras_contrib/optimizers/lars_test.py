from __future__ import print_function
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import lars

optimizers._test_optimizer(LARS(0.01))
optimizers._test_optimizer(LARS(0.01, use_nesterov = True))
