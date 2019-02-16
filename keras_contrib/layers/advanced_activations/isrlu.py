# -*- coding: utf-8 -*-
from keras import backend as K
from keras.layers import Layer
from keras_contrib.utils.test_utils import to_tuple
from keras_contrib.utils.test_utils import is_tf_keras


class ISRLU(Layer):
    """Inverse Square Root Linear Unit
    See: https://arxiv.org/pdf/1710.09967.pdf by AI Perf
    Reference: https://en.wikipedia.org/wiki/Activation_function
    Inverse Square Root Linear activation f(α, x):
        x >= 0: x
        x < 0:  x / sqrt(1 + α * x^2)
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
                NaNs). In particular, this happens when
                α < 0 and x < -1/sqrt(α).
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
        model.add(ISRLU(alpha=-0.3))
    """
    def __init__(self,
                 alpha=0.1,
                 **kwargs):

        super(ISRLU, self).__init__(**kwargs)
        self.alpha = alpha
        self.trainable = False

    if is_tf_keras:
        def alpha_initializer(self, input_shape, dtype='float32', partition_info=None):
            return self.alpha * K.ones(input_shape,
                                       dtype=dtype)

    else:
        def alpha_initializer(self, input_shape, dtype='float32'):
            return self.alpha * K.ones(input_shape,
                                       dtype=dtype)

    def build(self, input_shape):
        input_shape = to_tuple(input_shape)
        new_input_shape = input_shape[1:]
        self.alphas = self.add_weight(shape=new_input_shape,
                                      name='{}_alphas'.format(self.name),
                                      initializer=self.alpha_initializer,
                                      trainable=False)
        self.build = True

    def call(self, x):
        def inverse_quadratic_square_root(x):
            return x / K.sqrt(1 + self.alphas * K.square(x))

        return K.switch(K.less(x, K.zeros_like(x)), inverse_quadratic_square_root(x), x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha': self.alpha,
                  'trainable': self.trainable}
        base_config = super(ISRLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
