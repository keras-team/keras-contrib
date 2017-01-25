from __future__ import absolute_import
from . import backend as K
from keras.utils.generic_utils import get_from_module

from keras.regularizers import *


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
