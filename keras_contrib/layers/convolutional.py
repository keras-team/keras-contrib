# -*- coding: utf-8 -*-
from __future__ import absolute_import
import functools

from .. import backend as K
from .. import activations
from .. import initializations
from .. import regularizers
from .. import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers.convolutional import Convolution3D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.np_utils import conv_output_length
from keras.utils.np_utils import conv_input_length


class Deconvolution3D(Convolution3D):
    """Transposed convolution operator for filtering windows of 3-D inputs.

    The need for transposed convolutions generally arises from the desire to
    use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape
    of the output of some convolution to something that has the shape
    of its input while maintaining a connectivity pattern
    that is compatible with said convolution.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128, 128)` for a 128x128x128 volume with
    three channels.

    To pass the correct `output_shape` to this layer,
    one could use a test model to predict and observe the actual output shape.

    # Examples

    ```python
        # TH dim ordering.
        # apply a 3x3x3 transposed convolution
        # with stride 1x1x1 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 3, 14, 14, 14),
                                  border_mode='valid',
                                  input_shape=(3, 12, 12, 12)))

        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 3, 12, 12, 12))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 3, 14, 14, 14)

        # apply a 3x3x3 transposed convolution
        # with stride 2x2x2 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 3, 25, 25, 25),
                                  subsample=(2, 2, 2),
                                  border_mode='valid',
                                  input_shape=(3, 12, 12, 12)))
        model.summary()

        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 3, 12, 12, 12))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 3, 25, 25, 25)
    ```

    ```python
        # TF dim ordering.
        # apply a 3x3x3 transposed convolution
        # with stride 1x1x1 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 14, 14, 14, 3),
                                  border_mode='valid',
                                  input_shape=(12, 12, 12, 3)))

        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 12, 12, 12, 3))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 14, 14, 14, 3)

        # apply a 3x3x3 transposed convolution
        # with stride 2x2x2 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 25, 25, 25, 3),
                                  subsample=(2, 2, 2),
                                  border_mode='valid',
                                  input_shape=(12, 12, 12, 3)))
        model.summary()

        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 12, 12, 12, 3))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 25, 25, 25, 3)
    ```

    # Arguments
        nb_filter: Number of transposed convolution filters to use.
        kernel_dim1: Length of the first dimension in the transposed convolution kernel.
        kernel_dim2: Length of the second dimension in the transposed convolution kernel.
        kernel_dim3: Length of the third dimension in the transposed convolution kernel.
        output_shape: Output shape of the transposed convolution operation.
            tuple of integers
            `(nb_samples, nb_filter, conv_dim1, conv_dim2, conv_dim3)`.
             It is better to use
             a dummy input and observe the actual output shape of
             a layer, as specified in the examples.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano/TensorFlow function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        subsample: tuple of length 3. Factor by which to oversample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if dim_ordering='tf'.

    # Output shape
        5D tensor with shape:
        `(samples, nb_filter, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, nb_filter)` if dim_ordering='tf'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.

    # References
        - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
        - [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
        - [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """
    def __init__(self, nb_filter, kernel_dim1, kernel_dim2, kernel_dim3,
                 output_shape, init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1, 1),
                 dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for Deconvolution3D:', border_mode)
        if len(output_shape) == 4:
            # missing the batch size
            output_shape = (None,) + tuple(output_shape)

        self.output_shape_ = output_shape

        super(Deconvolution3D, self).__init__(nb_filter,
                                              kernel_dim1, kernel_dim2, kernel_dim3,
                                              init=init,
                                              activation=activation,
                                              weights=weights,
                                              border_mode=border_mode,
                                              subsample=subsample,
                                              dim_ordering=dim_ordering,
                                              W_regularizer=W_regularizer,
                                              b_regularizer=b_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              W_constraint=W_constraint,
                                              b_constraint=b_constraint,
                                              bias=bias,
                                              **kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            conv_dim1 = self.output_shape_[2]
            conv_dim2 = self.output_shape_[3]
            conv_dim3 = self.output_shape_[4]
            return (input_shape[0], self.nb_filter, conv_dim1, conv_dim2, conv_dim3)
        elif self.dim_ordering == 'tf':
            conv_dim1 = self.output_shape_[1]
            conv_dim2 = self.output_shape_[2]
            conv_dim3 = self.output_shape_[3]
            return (input_shape[0], conv_dim1, conv_dim2, conv_dim3, self.nb_filter)
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def call(self, x, mask=None):
        output = K.deconv3d(x, self.W, self.output_shape_,
                            strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_shape': self.output_shape_}
        base_config = super(Deconvolution3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

Deconv3D = Deconvolution3D
get_custom_objects().update({"Deconvolution3D": Deconvolution3D})
get_custom_objects().update({"Deconv3D": Deconv3D})


class SubPixelUpscaling(Layer):

    def __init__(self, scale_factor=2, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = K.depth_to_space(x, self.scale_factor)
        return y

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))

    def get_config(self):
        config = {'scale_factor': self.scale_factor}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
