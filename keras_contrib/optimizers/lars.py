from keras import backend as K
from keras.optimizers import Optimizer
from keras.utils.generic_utils import get_custom_objects


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
        - Gradual learning rate warm-up
        - Linear learning rate scaling
        - Poly rule learning rate decay
    Note, LARS scaling is currently only enabled for dense tensors.

    Args:
        lr: A `Tensor` or floating point value. The base learning rate.
        momentum: A floating point value. Momentum hyperparameter.
        weight_decay: A floating point value. Weight decay hyperparameter.
        eeta: LARS coefficient as used in the paper. Dfault set to LARS
            coefficient from the paper. (eeta / weight_decay) determines the
            highest scaling factor in LARS.
        epsilon: Optional epsilon parameter to be set in models that have very
            small gradients. Default set to 0.0.
        skip_list: List of strings to enable skipping variables from LARS scaling.
            If any of the strings in skip_list is a subset of var.name, variable
            'var' is skipped from LARS scaling. For a typical classification model
            with batch normalization, the skip_list is ['batch_normalization',
            'bias']
        nesterov: when set to True, nesterov momentum will be enabled
    """

    def __init__(self,
                 lr,
                 momentum=0.9,
                 weight_decay=0.0001,
                 eeta=0.001,
                 epsilon=0.0,
                 skip_list=None,
                 nesterov=False,
                 **kwargs):

        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay is not positive: %s" % weight_decay)
        print(kwargs)
        super(LARS, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.eeta = K.variable(eeta, name='eeta')
        self.epsilon = epsilon
        self.skip_list = skip_list
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        weights = self.get_weights()
        self.updates = [K.update_add(self.iterations, 1)]
        scaled_lr = self.lr
        if self.skip_list is None or not any(p in params.name
                                             for p in self.skip_list):
            w_norm = K.sqrt(K.sum([K.sum(K.square(w)) for w in weights]))
            g_norm = K.sqrt(K.sum([K.sum(K.square(g)) for g in grads]))
            scaled_lr = K.switch(K.greater(w_norm * g_norm, K.zeros([1])),
                                 K.expand_dims((self.eeta * w_norm /
                                                (g_norm + self.weight_decay * w_norm +
                                                 self.epsilon)) * self.lr),
                                 K.ones([1]) * self.lr)
        if K.backend() == 'theano':
            scaled_lr = scaled_lr[0]  # otherwise theano raise broadcasting error
        # momentum
        moments = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v0 = (m * self.momentum)
            v1 = scaled_lr * g  # velocity
            v = v0 - v1
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + (v * self.momentum) - v1
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'epsilon': self.epsilon,
                  'eeta': float(K.get_value(self.eeta)),
                  'nesterov': self.nesterov,
                  'skip_list': self.skip_list}
        base_config = super(LARS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))