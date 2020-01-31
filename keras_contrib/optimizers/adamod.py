from keras import backend as K
from keras.optimizers import Optimizer


class Adamod(Optimizer):
    """Adamod optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        beta_3: float, 0 < beta < 1. Generally close to 1.

    # References
        - [An Adaptive and Momental Bound Method for Stochastic Learning](
           https://arxiv.org/abs/1910.12249)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, beta_3=0.999,
                 **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Adamod, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.beta_3 = K.variable(beta_3, name='beta_3')
            self.decay = K.variable(self.initial_decay, name='decay')

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t))
        m_bias_correction = 1.0 / (1. - K.pow(self.beta_1, t))

        ms = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='v_' + str(i))
              for (i, p) in enumerate(params)]
        ss = [K.zeros(K.int_shape(p),
              dtype=K.dtype(p),
              name='s_' + str(i))
              for (i, p) in enumerate(params)]

        self.weights = [self.iterations] + ms + vs + ss

        for p, g, m, v, s in zip(params, grads, ms, vs, ss):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            n_t = lr_t / (K.sqrt(v_t) + self.epsilon)
            s_t = (self.beta_3 * s) + (1. - self.beta_3) * n_t
            nhat_t = K.minimum(n_t, s_t)
            p_t = p - nhat_t * m_t * m_bias_correction

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(s, s_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'beta_3': float(K.get_value(self.beta_3)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adamod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
