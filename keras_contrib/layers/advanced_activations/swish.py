from keras import backend as K
from keras.layers import Layer


class Swish(Layer):
    """ Swish (Ramachandranet al., 2017)

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        beta: float >= 0. Scaling factor
            if set to 1 and trainable set to False (default),
            Swish equals the SiLU activation (Elfwing et al., 2017)
        trainable: whether to learn the scaling factor during training or not

    # References
        - [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        - [Sigmoid-weighted linear units for neural network function
           approximation in reinforcement learning](https://arxiv.org/abs/1702.03118)
    """

    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.scaling_factor = K.variable(self.beta,
                                         dtype=K.floatx(),
                                         name='scaling_factor')
        if self.trainable:
            self._trainable_weights.append(self.scaling_factor)
        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return inputs * K.sigmoid(self.scaling_factor * inputs)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
