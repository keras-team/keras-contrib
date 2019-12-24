from keras.layers import Layer
from keras import backend as K
from keras.layers import InputSpec
from keras.initializers import Constant


class Swish(Layer):
    """ Swish (Ramachandranet al., 2017)

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        initial_beta: float >= 0. Scaling factor
            if set to 1 and trainable set to False (default),
            Swish equals the SiLU activation (Elfwing et al., 2017)
        trainable: whether to learn the scaling factor during training or not

    # References
        - [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        - [Sigmoid-weighted linear units for neural network function
           approximation in reinforcement learning](https://arxiv.org/abs/1702.03118)
    """
    """
    Swish activation function with a trainable parameter referred to as 'beta' in https://arxiv.org/abs/1710.05941"""
    def __init__(self, trainable = True, initial_beta = 1., **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable
        self.initial_beta = initial_beta
        self.beta_initializer = Constant(value=self.initial_beta)
        self.__name__ = 'swish'

    def build(self, input_shape):
        self.beta = self.add_weight(shape=[1], name='beta',
                                    initializer=self.beta_initializer,
                                     trainable=trainable)
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True

    def call(self, inputs):
        return inputs * K.sigmoid(self.beta * inputs)

    def get_config(self):
        config = {'trainable': self.trainable,
                  'initial_beta': self.initial_beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
