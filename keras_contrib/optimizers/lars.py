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
        learning_rate: A `Tensor` or floating point value. The base learning rate.
        momentum: A floating point value. Momentum hyperparameter.
        weight_decay: A floating point value. Weight decay hyperparameter.
        eeta: LARS coefficient as used in the paper. Dfault set to LARS
            coefficient from the paper. (eeta / weight_decay) determines the highest
            scaling factor in LARS.
        epsilon: Optional epsilon parameter to be set in models that have very
            small gradients. Default set to 0.0.
        skip_list: List of strings to enable skipping variables from LARS scaling.
            If any of the strings in skip_list is a subset of var.name, variable
            'var' is skipped from LARS scaling. For a typical classification model
            with batch normalization, the skip_list is ['batch_normalization',
            'bias']
        use_nesterov: when set to True, nesterov momentum will be enabled
    """
    def __init__(
      self,
      learning_rate,
      momentum=0.9,
      weight_decay=0.0001,
      # The LARS coefficient is a hyperparameter
      eeta=0.001,
      epsilon=0.0,
      # Enable skipping variables from LARS scaling.
      # TODO(sameerkm): Enable a direct mechanism to pass a
      # subset of variables to the optimizer.
      skip_list=None,
      use_nesterov=False):

        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay should be positive: %s" % weight_decay)
        super(LARSOptimizer, self).__init__(use_locking=False, name=name)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(learning_rate, name='lr')
            self._momentum = K.variable(momentum, name='momentum')
            self._weight_decay = K.variable(weight_decay, name='weight_decay')
            self._eeta = K.variable(eeta, name='eeta')
        self._epsilon = epsilon
        self._skip_list = skip_list
        self._use_nesterov = use_nesterov

    def get_config(self):
        config = {'lr' : float(K.get_value(self.lr)),
                  'momentum' : float(K.get_value(self.beta_1)),
                  'weight_decay' : float(K.get_value(self._weight_decay)),
                  'epsilon' : self.epsilon,
                  'etaa' : float(K.get_value(self._eeta)),
                  'nesterov' : self._use_nesterov,
                  'skip_list' : self._skip_list}
        base_config = super(LARS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grad = self.get_gradients(loss, params)
        weights = self.get_weights()
        self.updates = [K.update_add(self.iterations, 1)]
        scaled_lr = self.lr
        if self._skip_list is None or not any(p in params.name
                                              for p in self._skip_list):
            w_norm = K.sqrt(sum([K.sum(K.square(w)) for w in weights]))
            g_norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            if w_norm > 0:
                if g_norm > 0:
                    trust_ratio = (self._eeta * w_norm / 
                                  (g_norm + self._weight_decay * w_norm + self._epsilon))
                else:
                    trust_ration = 1.0
            else:
                trust_ratio = 1.0
            scaled_lr = self.lr*trust_ratio

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - scaled_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self._use_nesterov:
                new_p = p + self.momentum * v - scaled_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

get_custom_objects().update({'LARS': LARS})
