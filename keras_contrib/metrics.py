from . import backend as K
from keras.utils.generic_utils import get_from_module

from keras.metrics import *


def get(identifier):
    return get_from_module(identifier, globals(), 'metric')
