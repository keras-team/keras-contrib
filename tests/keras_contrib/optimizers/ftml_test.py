from __future__ import print_function
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import ftml

optimizers._test_optimizer(ftml())
optimizers._test_optimizer(ftml(lr=0.003, beta_1=0.8, beta_2=0.9, epsilon=1e-5, decay=1e-3))
