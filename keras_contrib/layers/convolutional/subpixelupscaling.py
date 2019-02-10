# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras.layers import Layer

from keras_contrib import backend as KC
from keras_contrib.utils.conv_utils import normalize_data_format


class SubPixelUpscaling(Layer):
    """ Sub-pixel convolutional upscaling layer.

    This layer requires a Convolution2D prior to it,
    having output filters computed according to
    the formula :

        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)

    This layer performs the depth to space operation on
    the convolution filters, and returns a
    tensor with the size as defined below.

    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)

        # Optional
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```

    In practice, it is useful to have a second convolution layer after the
    SubPixelUpscaling layer to speed up the learning process.

    However, if you are stacking multiple
    SubPixelUpscaling blocks, it may increase
    the number of parameters greatly, so the
    Convolution layer after SubPixelUpscaling
    layer can be removed.

    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.

    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)`
        if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)`
        if data_format='channels_last'.

    # References
        - [Real-Time Single Image and Video Super-Resolution Using an
           Efficient Sub-Pixel Convolutional Neural Network](
           https://arxiv.org/abs/1609.05158)
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = KC.depth_to_space(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            new_k = k // (self.scale_factor ** 2)
            new_r = r * self.scale_factor
            new_c = c * self.scale_factor
            return b, new_k, new_r, new_c
        else:
            b, r, c, k = input_shape
            new_r = r * self.scale_factor
            new_c = c * self.scale_factor
            new_k = k // (self.scale_factor ** 2)
            return b, new_r, new_c, new_k

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
