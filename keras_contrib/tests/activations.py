import numpy as np

from keras import backend as K


def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.
    '''
    return np.array([[0, 0.1, 0.5, 0.9, 1.0]], dtype=K.floatx())


def validate_activation(activation):
    activation(get_standard_values())
