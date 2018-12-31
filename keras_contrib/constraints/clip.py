from __future__ import absolute_import
from keras import backend as K
from keras.constraints import Constraint


class Clip(Constraint):
    """Clips weights to [-c, c].

    # Arguments
        c: Clipping parameter.
    """

    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}
