from .. import initializations
from keras.engine import Layer
from keras.utils.generic_utils import get_custom_objects
from .. import backend as K
import numpy as np


class PELU(Layer):
    """Parametric Exponential Linear Unit.
    It follows:
    `f(x) = alphas * (exp(x / betas) - 1) for x < 0`,
    `f(x) = (alphas / betas) * x for x >= 0`,
    where `alphas` & `betas` are learned arrays with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alphas_init: initialization function for the alpha variable weights.
        betas_init: initialization function for the beta variable weights.
        weights: initial weights, as a list of a single Numpy array.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # References
        - [PARAMETRIC EXPONENTIAL LINEAR UNIT FOR DEEP CONVOLUTIONAL NEURAL NETWORKS](https://arxiv.org/abs/1605.09332v3)
    """

    def __init__(self, alphas_init='one', betas_init='one', weights=None, shared_axes=None, **kwargs):
        self.supports_masking = True
        self.alphas_init = initializations.get(alphas_init)
        self.betas_init = initializations.get(betas_init)
        self.initial_weights = weights
        if not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        super(PELU, self).__init__(**kwargs)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes[0] is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        # Initialised as ones to emulate the default ELU
        self.alphas = self.alphas_init(param_shape,
                                       name='{}_alphas'.format(self.name))
        self.betas = self.betas_init(param_shape,
                                     name='{}_betas'.format(self.name))

        self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            pos = K.relu(x) * (K.pattern_broadcast(self.alphas, self.param_broadcast) /
                               K.pattern_broadcast(self.betas, self.param_broadcast))
            neg = (K.pattern_broadcast(self.alphas, self.param_broadcast) *
                   (K.exp(((x - K.abs(x)) * 0.5) / K.pattern_broadcast(self.betas, self.param_broadcast)) - 1))
        else:
            pos = K.relu(x) * self.alphas / self.betas
            neg = self.alphas * (K.exp(((x - K.abs(x)) * 0.5) / self.betas) - 1)
        return neg + pos

    def get_config(self):
        config = {'alphas_init': self.alphas_init.__name__,
                  'betas_init': self.betas_init.__name__}
        base_config = super(PELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({"PELU": PELU})
