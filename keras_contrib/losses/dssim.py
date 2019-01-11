from __future__ import absolute_import
import keras_contrib.backend as KC
from keras import backend as K


class DSSIMObjective:
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5

    Note : You should add a regularization term like a l2 loss in addition to this one.
    Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
           not be the `kernel_size` for an output of 32.

    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    def __int_shape(self, x):
        return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a
        # gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid',
                                                self.dim_ordering)
        patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid',
                                                self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = ((K.square(u_true)
                  + K.square(u_pred)
                  + self.c1) * (var_pred + var_true + self.c2))
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)
