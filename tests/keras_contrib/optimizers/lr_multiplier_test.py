from keras_contrib.tests import optimizers
from keras_contrib.optimizers import LearningRateMultiplier
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler


def test_lr_multiplier():
    opt1 = LearningRateMultiplier(SGD, lr=0.01, momentum=0.9, nesterov=True)
    optimizers._test_optimizer(opt1, target=0.95)

    mult = {'dense': 10}
    opt2 = LearningRateMultiplier(SGD, lr_multipliers=mult,
                                  lr=0.001, momentum=0.9, nesterov=True)
    optimizers._test_optimizer(opt2, target=0.95)
