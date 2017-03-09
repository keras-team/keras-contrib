# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

import copy
import inspect
import types as python_types
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces
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


class _CosineConv(Layer):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(_CosineConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def bias_div(bias, k, data_format=None):
        """Divides bias vector by a tensor.

        # Arguments
            bias: Bias tensor.
            k: Tensor or variable.
            data_format: Data format for 3D, 4D or 5D tensors:
                one of "channels_first", "channels_last".

        # Returns
            Output tensor.

        # Raises
            ValueError: In case of invalid `data_format` argument.
        """
        if data_format is None:
            data_format = image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))
        if K.ndim(x) == 5:
            if data_format == 'channels_first':
                x = K.reshape(bias, (1, K.shape(bias)[0], 1, 1, 1)) / k
            elif data_format == 'channels_last':
                x = K.reshape(bias, (1, 1, 1, 1, K.shape(bias)[0])) / k
        elif K.ndim(x) == 4:
            if data_format == 'channels_first':
                x = K.reshape(bias, (1, K.shape(bias)[0], 1, 1)) / k
            elif data_format == 'channels_last':
                x = K.reshape(bias, (1, 1, 1, K.shape(bias)[0])) / k
        elif K.ndim(x) == 3:
            if data_format == 'channels_first':
                x = K.reshape(bias, (1, K.shape(bias)[0], 1)) / k
            elif data_format == 'channels_last':
                x = K.reshape(bias, (1, 1, K.shape(bias)[0])) / k
        else:
            x = bias / k
        return x

    def broadcast_mul(x, k, data_format=None):
        """Multiplies a tensor by a vector.

        # Arguments
            x: Tensor.
            k: Vector.
            data_format: Data format for 3D, 4D or 5D tensors:
                one of "channels_first", "channels_last".

        # Returns
            Output tensor.

        # Raises
            ValueError: In case of invalid `data_format` argument.
        """
        if data_format is None:
            data_format = image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))
        if K.ndim(x) == 5:
            if data_format == 'channels_first':
                x *= K.reshape(k, (1, K.shape(k)[0], 1, 1, 1))
            elif data_format == 'channels_last':
                x *= K.reshape(k, (1, 1, 1, 1, K.shape(k)[0]))
        elif K.ndim(x) == 4:
            if data_format == 'channels_first':
                x *= K.reshape(k, (1, K.shape(k)[0], 1, 1))
            elif data_format == 'channels_last':
                x *= K.reshape(k, (1, 1, 1, K.shape(k)[0]))
        elif K.ndim(x) == 3:
            if data_format == 'channels_first':
                x *= K.reshape(k, (1, K.shape(k)[0], 1))
            elif data_format == 'channels_last':
                x *= K.reshape(k, (1, 1, K.shape(k)[0]))
        else:
            x *= k
        return x

    def build(self, input_shape):
        self.sum_axes = range(len(kernel_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
            del self.sum_axes[0]
        else:
            channel_axis = -1
            del self.sum_axes[-1]
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        kernel_norm_shape = self.kernel_size + (input_dim, 1)
        self.broadcast_pattern = [False]*len(kernel_shape)
        self.broadcast_pattern[channel_axis] = True

        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_norm = K.variable(np.ones(kernel_norm_shape), name='kernel_norm')

        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        knorm = K.sum(K.square(self.W), axis=self.sum_axes) + K.epsilon()
        if self.use_bias:
            b, xb = self.bias, 1.
            knorm += K.square(self.bias)
        else:
            xb = 0.

        knorm = K.sqrt(knorm)

        if self.rank == 1:
            xnorm = K.sqrt(K.conv1d(
                K.square(inputs),
                self.kernel_norm,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0]) +
                xb +
                K.epsilon())
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            xnorm = K.sqrt(K.conv2d(
                K.square(inputs),
                self.kernel_norm,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate) +
                xb +
                K.epsilon())
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            xnorm = K.sqrt(K.conv3d(
                K.square(inputs),
                self.kernel_norm,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate) +
                xb +
                K.epsilon())
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if K.backend() == 'theano':
            xnorm = K.pattern_broadcast(xnorm, self.broadcast_pattern)

        kxnorm = self.broadcast_mul(xnorm, knorm)
        outputs /= kxnorm

        if self.use_bias:
            b = self.bias_div(b, kxnorm)
            outputs += b

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_CosineConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# class CosineConvolution2D(Layer):
#     """Cosine Normalized Convolution operator for filtering windows of two-dimensional inputs.
#     Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks
#     https://arxiv.org/pdf/1702.05870.pdf

#     When using this layer as the first layer in a model,
#     provide the keyword argument `input_shape`
#     (tuple of integers, does not include the sample axis),
#     e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

#     # Examples

#     ```python
#         # apply a 3x3 convolution with 64 output filters on a 256x256 image:
#         model = Sequential()
#         model.add(CosineConvolution2D(64, 3, 3,
#                                 border_mode='same',
#                                 input_shape=(3, 256, 256)))
#         # now model.output_shape == (None, 64, 256, 256)

#         # add a 3x3 convolution on top, with 32 output filters:
#         model.add(CosineConvolution2D(32, 3, 3, border_mode='same'))
#         # now model.output_shape == (None, 32, 256, 256)
#     ```

#     # Arguments
#         nb_filter: Number of convolution filters to use.
#         nb_row: Number of rows in the convolution kernel.
#         nb_col: Number of columns in the convolution kernel.
#         init: name of initialization function for the weights of the layer
#             (see [initializations](../initializations.md)), or alternatively,
#             Theano function to use for weights initialization.
#             This parameter is only relevant if you don't pass
#             a `weights` argument.
#         activation: name of activation function to use
#             (see [activations](../activations.md)),
#             or alternatively, elementwise Theano function.
#             If you don't specify anything, no activation is applied
#             (ie. "linear" activation: a(x) = x).
#         weights: list of numpy arrays to set as initial weights.
#         border_mode: 'valid', 'same' or 'full'
#             ('full' requires the Theano backend).
#         subsample: tuple of length 2. Factor by which to subsample output.
#             Also called strides elsewhere.
#         W_regularizer: instance of [WeightRegularizer](../regularizers.md)
#             (eg. L1 or L2 regularization), applied to the main weights matrix.
#         b_regularizer: instance of [WeightRegularizer](../regularizers.md),
#             applied to the bias.
#         activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
#             applied to the network output.
#         W_constraint: instance of the [constraints](../constraints.md) module
#             (eg. maxnorm, nonneg), applied to the main weights matrix.
#         b_constraint: instance of the [constraints](../constraints.md) module,
#             applied to the bias.
#         dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
#             (the depth) is at index 1, in 'tf' mode is it at index 3.
#             It defaults to the `image_dim_ordering` value found in your
#             Keras config file at `~/.keras/keras.json`.
#             If you never set it, then it will be "tf".
#         bias: whether to include a bias
#             (i.e. make the layer affine rather than linear).

#     # Input shape
#         4D tensor with shape:
#         `(samples, channels, rows, cols)` if dim_ordering='th'
#         or 4D tensor with shape:
#         `(samples, rows, cols, channels)` if dim_ordering='tf'.

#     # Output shape
#         4D tensor with shape:
#         `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
#         or 4D tensor with shape:
#         `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
#         `rows` and `cols` values might have changed due to padding.
#     """

#     def __init__(self, nb_filter, nb_row, nb_col,
#                  init='glorot_uniform', activation=None, weights=None,
#                  border_mode='valid', subsample=(1, 1), dim_ordering='default',
#                  W_regularizer=None, b_regularizer=None,
#                  activity_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#         if dim_ordering == 'default':
#             dim_ordering = K.image_dim_ordering()
#         if border_mode not in {'valid', 'same', 'full'}:
#             raise ValueError('Invalid border mode for CosineConvolution2D:', border_mode)
#         self.nb_filter = nb_filter
#         self.nb_row = nb_row
#         self.nb_col = nb_col
#         self.init = initializations.get(init)
#         self.activation = activations.get(activation)
#         self.border_mode = border_mode
#         self.subsample = tuple(subsample)
#         if dim_ordering not in {'tf', 'th'}:
#             raise ValueError('dim_ordering must be in {tf, th}.')
#         self.dim_ordering = dim_ordering

#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)

#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)

#         self.bias = bias
#         self.input_spec = [InputSpec(ndim=4)]
#         self.initial_weights = weights
#         super(CosineConvolution2D, self).__init__(**kwargs)

#     def build(self, input_shape):
#         if self.dim_ordering == 'th':
#             stack_size = input_shape[1]
#             self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
#             self.W_norm_shape = (1, stack_size, self.nb_row, self.nb_col)
#         elif self.dim_ordering == 'tf':
#             stack_size = input_shape[3]
#             self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
#             self.W_norm_shape = (self.nb_row, self.nb_col, stack_size, 1)
#         else:
#             raise ValueError('Invalid dim_ordering:', self.dim_ordering)
#         self.W = self.add_weight(self.W_shape,
#                                  initializer=functools.partial(self.init,
#                                                                dim_ordering=self.dim_ordering),
#                                  name='{}_W'.format(self.name),
#                                  regularizer=self.W_regularizer,
#                                  constraint=self.W_constraint)

#         self.W_norm = K.variable(np.ones(self.W_norm_shape), name='{}_W_norm'.format(self.name))

#         if self.bias:
#             self.b = self.add_weight((self.nb_filter,),
#                                      initializer='zero',
#                                      name='{}_b'.format(self.name),
#                                      regularizer=self.b_regularizer,
#                                      constraint=self.b_constraint)
#         else:
#             self.b = None

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#         self.built = True

#     def get_output_shape_for(self, input_shape):
#         if self.dim_ordering == 'th':
#             rows = input_shape[2]
#             cols = input_shape[3]
#         elif self.dim_ordering == 'tf':
#             rows = input_shape[1]
#             cols = input_shape[2]
#         else:
#             raise ValueError('Invalid dim_ordering:', self.dim_ordering)

#         rows = conv_output_length(rows, self.nb_row,
#                                   self.border_mode, self.subsample[0])
#         cols = conv_output_length(cols, self.nb_col,
#                                   self.border_mode, self.subsample[1])

#         if self.dim_ordering == 'th':
#             return (input_shape[0], self.nb_filter, rows, cols)
#         elif self.dim_ordering == 'tf':
#             return (input_shape[0], rows, cols, self.nb_filter)

#     def call(self, x, mask=None):
#         b, xb = 0., 0.
#         if self.dim_ordering == 'th':
#             W_sum_axes = [1, 2, 3]
#             if self.bias:
#                 b = K.reshape(self.b, (self.nb_filter, 1, 1, 1))
#                 xb = 1.
#         elif self.dim_ordering == 'tf':
#             W_sum_axes = [0, 1, 2]
#             if self.bias:
#                 b = K.reshape(self.b, (1, 1, 1, self.nb_filter))
#                 xb = 1.

#         Wnorm = K.sqrt(K.sum(K.square(self.W), axis=W_sum_axes, keepdims=True) + K.square(b) + K.epsilon())
#         xnorm = K.sqrt(K.conv2d(K.square(x), self.W_norm, strides=self.subsample,
#                                 border_mode=self.border_mode,
#                                 dim_ordering=self.dim_ordering,
#                                 filter_shape=self.W_norm_shape) + xb + K.epsilon())

#         W = self.W / Wnorm

#         output = K.conv2d(x, W, strides=self.subsample,
#                           border_mode=self.border_mode,
#                           dim_ordering=self.dim_ordering,
#                           filter_shape=self.W_shape)

#         if K.backend() == 'theano':
#             xnorm = K.pattern_broadcast(xnorm, [False, True, False, False])

#         output /= xnorm

#         if self.bias:
#             b /= Wnorm
#             if self.dim_ordering == 'th':
#                 b = K.reshape(b, (1, self.nb_filter, 1, 1))
#             elif self.dim_ordering == 'tf':
#                 b = K.reshape(b, (1, 1, 1, self.nb_filter))
#             else:
#                 raise ValueError('Invalid dim_ordering:', self.dim_ordering)
#             b /= xnorm
#             output += b
#         output = self.activation(output)
#         return output

#     def get_config(self):
#         config = {'nb_filter': self.nb_filter,
#                   'nb_row': self.nb_row,
#                   'nb_col': self.nb_col,
#                   'init': self.init.__name__,
#                   'activation': self.activation.__name__,
#                   'border_mode': self.border_mode,
#                   'subsample': self.subsample,
#                   'dim_ordering': self.dim_ordering,
#                   'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
#                   'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
#                   'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
#                   'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
#                   'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
#                   'bias': self.bias}
#         base_config = super(CosineConvolution2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

class CosineConv1D(_CosineConv):
    """1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"causal"` results in causal (dilated) convolutions, e.g. output[t]
            depends solely on input[:t-1]. Useful when modeling temporal data
            where the model should not violate the temporal order.
            See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CosineConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = super(CosineConv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

CosineConvolution1D = CosineConv1D
get_custom_objects().update({"CosineConvolution1D": CosineConvolution1D})
get_custom_objects().update({"CosineConv1D": CosineConv1D})

class CosineConv2D(_CosineConv):
    """2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, width, height, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, width, height)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CosineConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(CosineConv2D, self).get_config()
        config.pop('rank')
        return config

CosineConvolution2D = CosineConv2D
get_custom_objects().update({"CosineConvolution2D": CosineConvolution2D})
get_custom_objects().update({"CosineConv2D": CosineConv2D})


class CosineConv3D(_CosineConv):
    """3D convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 3)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.

    # Output shape
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CosineConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def get_config(self):
        config = super(CosineConv3D, self).get_config()
        config.pop('rank')
        return config

CosineConvolution3D = CosineConv3D
get_custom_objects().update({"CosineConvolution3D": CosineConvolution3D})
get_custom_objects().update({"CosineConv3D": CosineConv3D})


class SubPixelUpscaling(Layer):
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).

    This layer requires a Convolution2D prior to it, having output nb_filter computed according to
    the formula :

        nb_filter = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)

    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.

    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)

        [Optional]
        x = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(u)
    ```

        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.

        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.

    # Arguments
        scale_factor: Upscaling factor.
        dim_ordering: Can be 'default', 'th' or 'tf'.

    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if dim_ordering='tf'.

    """

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
