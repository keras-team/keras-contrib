from __future__ import absolute_import

from six.moves import zip

from . import backend as K
from keras.utils.generic_utils import get_from_module, get_custom_objects

if K.backend() == 'tensorflow':
    import tensorflow as tf


def get(identifier, kwargs=None):
    if K.backend() == 'tensorflow':
        # Wrap TF optimizer instances
        if isinstance(identifier, tf.train.Optimizer):
            return TFOptimizer(identifier)
    # Instantiate a Keras optimizer
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs)
