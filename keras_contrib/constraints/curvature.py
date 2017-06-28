from __future__ import absolute_import
from keras_contrib import backend as K
from keras.constraints import *

class CurvatureConstraint(Constraint):
    """ Specific to SeparableFC
        Constrains the second differences of weights in W_pos.
        
    # Arguments
        m: the maximum allowed curvature which constrains
            second differences of adjacent weights in the length dimension
            to be within the specified range
    """

    def __init__(self, m=1.0):
        self.m = m

    def __call__(self, p):
        import numpy as np
        mean_p = K.mean(p, axis=1)
        (num_output, length) = K.int_shape(p)
        diff1 = p[:,1:] - p[:,:-1]
        mean_diff1 = K.mean(diff1, axis=1)
        diff2 = diff1[:,1:] - diff1[:,:-1]
        desired_diff2 = K.clip(diff2, -1.0 * float(self.m), float(self.m))

        il1 = np.triu_indices(length-2)
        mask1 = np.ones((num_output, length-1, length-2))
        mask1[:, il1[0], il1[1]] = 0.0
        kmask1 = K.variable(value=mask1)
        mat1 = kmask1 * K.repeat_elements(K.expand_dims(desired_diff2, 1), length-1, 1)
        desired_diff1 = K.squeeze(K.squeeze(
                K.dot(mat1, K.ones((1, length-2, num_output)))[:,:,:1,:1], axis=2), axis=2)
        desired_diff1 += K.repeat_elements(K.expand_dims(
                mean_diff1 - K.mean(desired_diff1, axis=1), -1), length-1, axis=1)

        il2 = np.triu_indices(length-1)
        mask2 = np.ones((num_output, length, length-1))
        mask2[:, il2[0], il2[1]] = 0.0
        kmask2 = K.variable(value=mask2)
        mat2 = kmask2 * K.repeat_elements(K.expand_dims(desired_diff1, 1), length, 1)
        desired_p = K.squeeze(K.squeeze(
                K.dot(mat2, K.ones((1, length-1, num_output)))[:,:,:1,:1], axis=2), axis=2)
        desired_p += K.repeat_elements(K.expand_dims(
                mean_p - K.mean(desired_p, axis=1), -1), length, axis=1)

        return desired_p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m}  