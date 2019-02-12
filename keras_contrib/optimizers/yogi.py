from keras import backend as K
from keras.optimizers import Optimizer


class Yogi(Optimizer):
    """Yogi optimizer.
    Yogi is a variation of Adam that controls the increase in effective
    learning rate, which (according to the paper) leads to even better
    performance than Adam with similar theoretical guarantees on convergence.
    Default parameters follow those provided in the original paper, Tab.1
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adaptive Methods for Nonconvex Optimization](
           https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization)

    If you open an issue or a pull request about the Yogi optimizer,
    please add 'cc @MarcoAndreaBuchmann' to notify him.
    """

    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-3, decay=0., **kwargs):
        super(Yogi, self).__init__(**kwargs)
        if beta_1 <= 0 or beta_1 >= 1:
            raise ValueError("beta_1 has to be in ]0, 1[")
        if beta_2 <= 0 or beta_2 >= 1:
            raise ValueError("beta_2 has to be in ]0, 1[")

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        if epsilon <= 0:
            raise ValueError("epsilon has to be larger than 0")
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            g2 = K.square(g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = v - (1. - self.beta_2) * K.sign(v - g2) * g2
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Yogi, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
