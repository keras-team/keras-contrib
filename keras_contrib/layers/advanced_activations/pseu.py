# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers import Layer
from keras_contrib.utils.test_utils import to_tuple
from keras_contrib.utils.test_utils import is_tf_keras


class PSEU(Layer):
    """Parametric Soft Exponential Unit
    See: https://arxiv.org/pdf/1602.01321.pdf by Godfrey and Gashler
    Reference: https://github.com/keras-team/keras/issues/3842 (@hobson)
    Soft Exponential f(α, x):
        α == 0:  x
        α  > 0:  (exp(αx)-1) / α + α
        α  < 0:  -ln(1-α(x + α)) / α
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: Value of the alpha weights (float)
        NOTE : This function can become unstable for
               negative values of α. In particular, the
               function returns NaNs when α < 0 and x <= 1/α
               (where x is the input).
               If the function starts returning NaNs for α < 0,
               try decreasing the magnitude of α.
               Alternatively, you can normalize the data into fixed
               ranges before passing it to PSEU.
               Adjust α based on your specific dataset
               and use-case.
    # Example
        model = Sequential()
        model.add(Dense(10, input_shape=(5,))
        model.add(PSEU(alpha=0.2))
    """
    def __init__(self,
                 alpha=0.1,
                 **kwargs):

        super(PSEU, self).__init__(**kwargs)
        self.alpha = alpha
        self.trainable = False

    def alpha_initializer(self, input_shape, dtype='float32', **kwargs):
        return self.alpha * K.ones(input_shape,
                                   dtype=dtype)

    def build(self, input_shape):
        input_shape = to_tuple(input_shape)
        new_input_shape = input_shape[1:]
        self.alphas = self.add_weight(shape=new_input_shape,
                                      name='{}_alphas'.format(self.name),
                                      initializer=self.alpha_initializer,
                                      trainable=self.trainable)
        self.build = True

    def call(self, x):
        if self.alpha < 0:
            return - K.log(1 - self.alphas * (x + self.alphas)) / self.alphas
        elif self.alpha > 0:
            return self.alphas + (K.exp(self.alphas * x) - 1.) / self.alphas
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha': self.alpha,
                  'trainable': self.trainable}
        base_config = super(PSEU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
