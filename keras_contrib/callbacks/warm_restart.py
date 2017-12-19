import numpy as np
from keras.callbacks import Callback

class WarmRestart(Callback):
    """Warm Restart Learning rate update rule.
    # Arguments
        eta_min: float. min value of learning rate
        eta_max: float. max value of learning rate
        T_0: initial the number of epochs to restart
        T_mult: increase scale factor of T_0.

    # Reference
        [SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS](https://arxiv.org/pdf/1608.03983.pdf)
    """

    def __init__(self, eta_min=0., eta_max=0.1, T_0=5, T_mult=1):
        super(WarmRestart, self).__init__()
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = 0
        self.cum_previous_Ti = 0

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        T_cur = epoch - self.cum_previous_Ti
        print(epoch, T_cur, self.cum_previous_Ti)

        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1. + np.cos(T_cur / self.T_i * np.pi))
        
        if T_cur == self.T_i:
            self.cum_previous_Ti += self.T_i
            self.T_i *= self.T_mult

        K.set_value(self.model.optimizer.lr, lr)
