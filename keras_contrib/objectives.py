from __future__ import absolute_import

import keras_contrib.backend as KC
from keras.utils.generic_utils import get_from_module

from keras.objectives import *


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')


class DSSIMObjective():
    def __init__(self, batch_size, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You shoul add a regularization term like a l2 loss in addition to this one.
        :param batch_size: Batch size used in the model
        :param k1: Parameter of the SSIM (default 0.01)
        :param k2: Parameter of the SSIM (default 0.03)
        :param kernel_size: Size of the sliding window (default 3)
        :param max_value: Max value of the output (default 1.0)
        """
        self.__name__ = "DSSIMObjective"
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.dim_ordering = K.image_dim_ordering()
        self.backend = KC.backend()

    def __int_shape(self, x):
        return KC.int_shape(x) if self.backend == "tensorflow" else KC.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = KC.reshape(y_true, [self.batch_size] + list(self.__int_shape(y_pred)[1:]))
        y_pred = KC.reshape(y_pred, [self.batch_size] + list(self.__int_shape(y_pred)[1:]))
        patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, "valid", self.dim_ordering)
        patches_true = KC.extract_image_patches(y_true, kernel, kernel, "valid", self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = KC.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = KC.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        u_true = KC.mean(patches_true, axis=-1)
        u_pred = KC.mean(patches_pred, axis=-1)
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        std_true = K.sqrt(var_true + KC.epsilon())
        std_pred = K.sqrt(var_pred + KC.epsilon())
        c1 = (self.k1 * self.max_value) ** 2
        c2 = (self.k2 * self.max_value) ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)
