# -*- coding: utf-8 -*-
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras import backend as K
import numpy as np


class PSEU(Layer):
    """Parametric Soft Exponential Unit with trainable alpha
    See: https://arxiv.org/pdf/1602.01321.pdf by Godfrey and Gashler
    Reference: https://github.com/keras-team/keras/issues/3842 (@hobson)
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        initializer: Initializer for alpha weights.
        alpha_init: Initial value of the alpha weights (float)
                    This value overrides any specified initializer
                    by default, but, one can use their initializer
                    of choice by specifying alpha_init=None.
        regularizer: Regularizer for alpha weights.
        constraint: Constraint for alpha weights.
        trainable: Whether the alpha weights are trainable or not

    # Example
        model = Sequential()
        model.add(Dense(10))
        model.add(PSEU())

    Note : Specify alpha_init=None to use other intializers

    Soft Exponential f(α, x):
        α == 0:  x
        α  > 0:  (exp(αx)-1) / α + α
        α  < 0:  -ln(1-α(x + α)) / α
    """
    def __init__(self,
                 alpha_init=0.1,
                 initializer='glorot_uniform',
                 regularizer=None,
                 constraint=None,
                 trainable=True,
                 **kwargs):

        super(PSEU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_init = alpha_init
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)
        self.trainable = trainable

    def build(self, input_shape):
        new_input_shape = input_shape[1:]
        self.alphas = self.add_weight(shape=new_input_shape,
                                      name='{}_alphas'.format(self.name),
                                      initializer=self.initializer,
                                      regularizer=self.regularizer,
                                      constraint=self.constraint)

        if self.trainable:
            self.trainable_weights = [self.alphas]

        if self.alpha_init is not None:
            self.set_weights([self.alpha_init * np.ones(new_input_shape)])

        self.build = True

    def call(self, x, mask=None):
        if self.alpha_init is not None and self.alpha_init < 0:
            return - K.log(1 - self.alphas * (x + self.alphas)) / self.alphas
        elif self.alpha_init is not None and self.alpha_init > 0:
            return self.alphas + (K.exp(self.alphas * x) - 1.) / self.alphas
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        if self.alpha_init is None:
            config = {'alpha_init': initializers.serialize(self.initializer),
                      'regularizer': regularizers.serialize(self.regularizer),
                      'constraint': constraints.serialize(self.constraint),
                      'trainable': self.trainable}
        else:
            config = {'alpha_init': float(self.alpha_init),
                      'regularizer': regularizers.serialize(self.regularizer),
                      'constraint': constraints.serialize(self.constraint),
                      'trainable': self.trainable}

        base_config = super(PSEU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
