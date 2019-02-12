# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers import Layer


class PSEU(Layer):
    """Parametric Soft Exponential Unit
    See: https://arxiv.org/pdf/1602.01321.pdf by Godfrey and Gashler
    Reference: https://github.com/keras-team/keras/issues/3842 (@hobson)
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha_initial: Initial value of the alpha weights (float)

    # Example
        model = Sequential()
        model.add(Dense(10))
        model.add(PSEU(alpha_initial=0.2))

    Soft Exponential f(α, x):
        α == 0:  x
        α  > 0:  (exp(αx)-1) / α + α
        α  < 0:  -ln(1-α(x + α)) / α
    """
    def __init__(self,
                 alpha_initial=0.1,
                 **kwargs):

        super(PSEU, self).__init__(**kwargs)
        self.alpha_initial = alpha_initial

    def alpha_initializer(self, input_shape):
        return self.alpha_initial * K.ones(input_shape)

    def build(self, input_shape):
        new_input_shape = input_shape[1:]
        self.alphas = self.add_weight(shape=new_input_shape,
                                      name='{}_alphas'.format(self.name),
                                      initializer=self.alpha_initializer)
        self.build = True

    def call(self, x):
        if self.alpha_initial < 0:
            return - K.log(1 - self.alphas * (x + self.alphas)) / self.alphas
        elif self.alpha_initial > 0:
            return self.alphas + (K.exp(self.alphas * x) - 1.) / self.alphas
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha_initial': self.alpha_initial}
        base_config = super(PSEU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
