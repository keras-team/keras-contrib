from keras import backend as K
from keras.optimizers import Optimizer


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
        nesterov: when set to True, nesterov momentum will be enabled
    """

    def __init__(self,
                 lr,
                 momentum=0.9,
                 weight_decay=0.0001,
                 eeta=0.001,
                 epsilon=0.0,
                 nesterov=False,
                 **kwargs):

        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay is not positive: %s" % weight_decay)
        super(LARS, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.eeta = K.variable(eeta, name='eeta')
        self.epsilon = epsilon
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        weights = self.get_weights()
        self.updates = [K.update_add(self.iterations, 1)]
        scaled_lr = self.lr
        w_norm = K.sqrt(K.sum([K.sum(K.square(weight))
                               for weight in weights]))
        g_norm = K.sqrt(K.sum([K.sum(K.square(grad))
                               for grad in grads]))
        scaled_lr = K.switch(K.greater(w_norm * g_norm, K.zeros([1])),
                             K.expand_dims((self.eeta * w_norm /
                                            (g_norm + self.weight_decay * w_norm +
                                             self.epsilon)) * self.lr),
                             K.ones([1]) * self.lr)
        if K.backend() == 'theano':
            scaled_lr = scaled_lr[0]  # otherwise theano raise broadcasting error
        # momentum
        moments = [K.zeros(K.int_shape(param), dtype=K.dtype(param))
                   for param in params]
        self.weights = [self.iterations] + moments
        for param, grad, moment in zip(params, grads, moments):
            v0 = (moment * self.momentum)
            v1 = scaled_lr * grad  # velocity
            veloc = v0 - v1
            self.updates.append(K.update(moment, veloc))

            if self.nesterov:
                new_param = param + (veloc * self.momentum) - v1
            else:
                new_param = param + veloc

            # Apply constraints.
            if getattr(param, 'constraint', None) is not None:
                new_param = param.constraint(new_param)

            self.updates.append(K.update(param, new_param))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'epsilon': self.epsilon,
                  'eeta': float(K.get_value(self.eeta)),
                  'nesterov': self.nesterov}
        base_config = super(LARS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
