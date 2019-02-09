from __future__ import absolute_import
from keras.optimizers import Optimizer
from keras import backend as K


class FTML(Optimizer):
    """FTML optimizer.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 0.5.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [FTML - Follow the Moving Leader in Deep Learning](
        http://www.cse.ust.hk/~szhengac/papers/icml17.pdf)
    """

    def __init__(self, lr=0.0025, beta_1=0.6, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(FTML, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.decay = K.variable(decay)
        self.epsilon = epsilon
        self.inital_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        lr_t = lr / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params]
        zs = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        ds = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + zs + vs + ds

        for p, g, z, v, d in zip(params, grads, zs, vs, ds):
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            d_t = (K.sqrt(v_t / (1. - K.pow(self.beta_2, t)))
                   + self.epsilon) / lr_t
            sigma_t = d_t - self.beta_1 * d
            z_t = self.beta_1 * z + (1. - self.beta_1) * g - sigma_t * p

            p_t = - z_t / d_t

            self.updates.append(K.update(z, z_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(d, d_t))

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
        base_config = super(FTML, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
