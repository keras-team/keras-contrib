from keras.optimizers import Optimizer
from keras.utils import get_custom_objects


class LearningRateMultiplier(Optimizer):
    """Optimizer wrapper for per layer learning rate.

    This wrapper is used to add per layer learning rates by
    providing per layer factors which are multiplied with the
    learning rate of the optimizer.

    Note: This is a wrapper and does not implement any
    optimization algorithm.

    # Arguments
        optimizer: An optimizer class to be wrapped.
        lr_multipliers: Dictionary of the per layer factors. For
            example `optimizer={'conv_1/kernel':0.5, 'conv_1/bias':0.1}`.
            If for kernel and bias the same learning rate is used, the
            user can specify `optimizer={'conv_1':0.5}`.
        **kwargs: The arguments for instantiating the wrapped optimizer
            class.
    """
    def __init__(self, optimizer, lr_multipliers=None, **kwargs):
        self._class = optimizer
        self._optimizer = optimizer(**kwargs)
        self._lr_multipliers = lr_multipliers or {}

    def _get_multiplier(self, param):
        for k in self._lr_multipliers.keys():
            if k in param.name:
                return self._lr_multipliers[k]

    def get_updates(self, loss, params):
        mult_lr_params = {p: self._get_multiplier(p) for p in params
                          if self._get_multiplier(p)}
        base_lr_params = [p for p in params if self._get_multiplier(p) is None]

        updates = []
        base_lr = self._optimizer.lr
        for param, multiplier in mult_lr_params.items():
            self._optimizer.lr = base_lr * multiplier
            updates.extend(self._optimizer.get_updates(loss, [param]))

        self._optimizer.lr = base_lr
        updates.extend(self._optimizer.get_updates(loss, base_lr_params))

        return updates

    def get_config(self):
        config = {'optimizer': self._class,
                  'lr_multipliers': self._lr_multipliers}
        base_config = self._optimizer.get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def __getattr__(self, name):
        return getattr(self._optimizer, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(LearningRateMultiplier, self).__setattr__(name, value)
        else:
            self._optimizer.__setattr__(name, value)


get_custom_objects().update({'LearningRateMultiplier': LearningRateMultiplier})
