# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers import Layer


class ISRLU(Layer):
    """Inverse Square Root Linear Unit
    See: https://arxiv.org/pdf/1710.09967.pdf by AI Perf
    Reference: https://en.wikipedia.org/wiki/Activation_function
    Inverse Square Root Linear activation f(α, x):
        α >= 0: x
        α < 0:  x / sqrt(1 + α * x^2)

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha: Value of the alpha weights (float)
        NOTE : This function can become unstable for
               negative values of α (it may return
               NaNs).
               If this happens, try limiting the magnitude
               of α below a certain threshold, such that
               1 + α * x^2 is always positive.
               Alternatively, you can normalize the inputs
               into fixed ranges before passing them to ISRLU.
               Adjust the value of α based on your specific
               dataset and use-case.

    # Example
        model = Sequential()
        model.add(Dense(5, input_shape=(15,))
        model.add(ISRLU(alpha=0.3))
    """
    def __init__(self,
                 alpha=0.1,
                 **kwargs):

        super(ISRLU, self).__init__(**kwargs)
        self.alpha = alpha

    def alpha_initializer(self, input_shape):
        return self.alpha * K.ones(input_shape)

    def build(self, input_shape):
        new_input_shape = input_shape[1:]
        self.alphas = self.add_weight(shape=new_input_shape,
                                      name='{}_alphas'.format(self.name),
                                      initializer=self.alpha_initializer,
                                      trainable=False)
        self.build = True

    def call(self, x):
        if self.alpha < 0:
            return x / K.sqrt(1 + α * K.square(x))
        elif self.alpha >= 0:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(ISRLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
