from __future__ import absolute_import
import numpy as np
from . import backend as K
from keras.utils.generic_utils import get_from_module

from keras.initializers import *


def get(identifier, **kwargs):
    return get_from_module(identifier, globals(),
                           'initialization', kwargs=kwargs)
