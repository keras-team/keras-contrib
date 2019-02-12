# -*- coding: utf-8 -*-
from keras import backend as K
from keras import constraints
from keras import initializers
from keras.layers import Layer
from keras import regularizers


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
        alpha_init: Initial value of the alpha weights (float)
        regularizer: Regularizer for alpha weights.
        constraint: Constraint for alpha weights.

    # Example
        model = Sequential()
        model.add(Dense(10))
        model.add(PSEU())

    Soft Exponential f(α, x):
        α == 0:  x
        α  > 0:  (exp(αx)-1) / α + α
        α  < 0:  -ln(1-α(x + α)) / α
    """
    def __init__(self,
                 alpha_init=0.1,
                 regularizer=None,
                 constraint=None,
                 **kwargs):

        super(PSEU, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        # Add random initializer
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

    def alpha_initializer(self, input_shape):
        return self.alpha_init * K.ones(input_shape)

    def build(self, input_shape):
        new_input_shape = input_shape[1:]
        self.alphas = self.add_weight(shape=new_input_shape,
                                      name='{}_alphas'.format(self.name),
                                      initializer=self.alpha_initializer,
                                      regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.build = True

    def call(self, x):
        if self.alpha_init < 0:
            return - K.log(1 - self.alphas * (x + self.alphas)) / self.alphas
        elif self.alpha_init > 0:
            return self.alphas + (K.exp(self.alphas * x) - 1.) / self.alphas
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha_init': float(self.alpha_init),
                  'regularizer': regularizers.serialize(self.regularizer),
                  'constraint': constraints.serialize(self.constraint)}

        base_config = super(PSEU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
