from __future__ import print_function
from keras_contrib.tests import optimizers
from keras_contrib.optimizers import Yogi

optimizers._test_optimizer(Yogi())
optimizers._test_optimizer(Yogi(beta_1=0.9, beta_2=0.9))
optimizers._test_optimizer(Yogi(beta_1=0.9, beta_2=0.99))
optimizers._test_optimizer(Yogi(beta_1=0.9, beta_2=0.999))
optimizers._test_optimizer(Yogi(beta_1=0.9, beta_2=0.999, lr=0.001))
