import numpy as np
from keras.callbacks import Callback
import keras.backend as K


class LearningRateWarmRestarter(Callback):
    """Warm Restart Learning rate update rule.
    # Arguments
        min_lr: lower bound on the learning rate.
        max_lr: upper bound on the learning rate.
        num_restart_epochs:  restart learning rate from `max_lr` at every `num_restart_epochs`.
        factor: factor by which the number of restart epochs will be increased. new_num_restart_epochs = num_restart_epochs*factor.

    # Reference
        [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf)
    """

    def __init__(self, min_lr=0., max_lr=0.1, num_restart_epochs=5, factor=1):
        super(LearningRateWarmRestarter, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_restart_epochs = num_restart_epochs
        self.factor = factor
        self.cumsum_end_num_restart_epochs = 0

        if factor < 1:
            raise ValueError('"factor" must be larger than 0')

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        t = epoch - self.cumsum_end_num_restart_epochs
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1. + np.cos(t / self.num_restart_epochs * np.pi))

        if t == self.num_restart_epochs:
            self.cumsum_end_num_restart_epochs += self.num_restart_epochs
            self.num_restart_epochs *= self.factor

        K.set_value(self.model.optimizer.lr, lr)
