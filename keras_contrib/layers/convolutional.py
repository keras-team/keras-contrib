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
import numpy as np


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


class CosineConvolution2D(Layer):
    """Cosine Normalized Convolution operator for filtering windows of two-dimensional inputs.
    Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks
    https://arxiv.org/pdf/1702.05870.pdf

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(CosineConvolution2D(64, 3, 3,
                                border_mode='same',
                                input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(CosineConvolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        subsample: tuple of length 2. Factor by which to subsample output.
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
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='default',
                 W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for CosineConvolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(CosineConvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
            self.W_norm_shape = (1, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
            self.W_norm_shape = (self.nb_row, self.nb_col, stack_size, 1)
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        self.W = self.add_weight(self.W_shape,
                                 initializer=functools.partial(self.init,
                                                               dim_ordering=self.dim_ordering),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.W_norm = K.variable(np.ones(self.W_norm_shape), name='{}_W_norm'.format(self.name))

        if self.bias:
            self.b = self.add_weight((self.nb_filter,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)

    def call(self, x, mask=None):
        b, xb = 0., 0.
        if self.dim_ordering == 'th':
            W_sum_axes = [1, 2, 3]
            if self.bias:
                b = K.reshape(self.b, (self.nb_filter, 1, 1, 1))
                xb = 1.
        elif self.dim_ordering == 'tf':
            W_sum_axes = [0, 1, 2]
            if self.bias:
                b = K.reshape(self.b, (1, 1, 1, self.nb_filter))
                xb = 1.

        Wnorm = K.sqrt(K.sum(K.square(self.W), axis=W_sum_axes, keepdims=True) + K.square(b) + K.epsilon())
        xnorm = K.sqrt(K.conv2d(K.square(x), self.W_norm, strides=self.subsample,
                       border_mode=self.border_mode,
                       dim_ordering=self.dim_ordering,
                       filter_shape=self.W_norm_shape) + xb + K.epsilon())

        W = self.W / Wnorm

        output = K.conv2d(x, W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)

        if K.backend() == 'theano':
            xnorm = K.pattern_broadcast(xnorm, [False, True, False, False])

        output /= xnorm

        if self.bias:
            b /= Wnorm
            if self.dim_ordering == 'th':
                b = K.reshape(b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                b = K.reshape(b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
            b /= xnorm
            output += b
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(CosineConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

CosineConv2D = CosineConvolution2D
get_custom_objects().update({"CosineConvolution2D": CosineConvolution2D})
get_custom_objects().update({"CosineConv2D": CosineConv2D})


class SubPixelUpscaling(Layer):

    def __init__(self, scale_factor=2, dim_ordering='default', **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.dim_ordering = dim_ordering

        if self.dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = K.depth_to_space(x, self.scale_factor)
        return y

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'dim_ordering': self.dim_ordering}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({'SubPixelUpscaling': SubPixelUpscaling})
