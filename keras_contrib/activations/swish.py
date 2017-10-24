from __future__ import absolute_import
from keras.activations import *
from keras import backend as K


def swish(x):
    """Swish: a Self-Gated Activation Function (Prajith Ramachandran et al. 2017)

    # Arguments
        x: A tensor or variable to compute the activation function for.

    # References
        - [Swish: a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941)
    """
    return K.sigmoid(x) * x
