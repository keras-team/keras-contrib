from __future__ import absolute_import
from . import backend as K
from keras.activations import *


def selu(x, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)

    # Arguments
        x: A tensor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.
        scale: A scalar, to ensure the slope larger than 1.0 for positive inputs.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    return scale * K.elu(x, alpha)
