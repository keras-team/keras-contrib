from keras_contrib.tests import optimizers
from keras_contrib.optimizers import LearningRateMultiplier
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler

def test_lr_multiplier():
    mult={'dense':10}
    optimizers._test_optimizer(LearningRateMultiplier(SGD, lr=0.01, momentum=0.9, nesterov=True), target=0.95)
    optimizers._test_optimizer(LearningRateMultiplier(SGD, lr_multipliers=mult, lr=0.001,
                                                      momentum=0.9, nesterov=True), target=0.95)

