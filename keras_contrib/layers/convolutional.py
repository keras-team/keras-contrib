# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial

from keras import backend as K
from keras_contrib import backend as KC
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras.layers import InputSpec
from keras.utils import get_custom_objects
from keras_contrib.utils.conv_utils import conv_output_length
from keras_contrib.utils.conv_utils import normalize_data_format
import numpy as np


class CosineConvolution2D(Layer):
    """Cosine Normalized Convolution operator for filtering
    windows of two-dimensional inputs.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(CosineConvolution2D(64, 3, 3,
                                padding='same',
                                input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(CosineConvolution2D(32, 3, 3, padding='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```

    # Arguments
        filters: Number of convolution filters to use.
        kernel_size: kernel_size: An integer or tuple/list of
            2 integers, specifying the
            dimensions of the convolution window.
        init: name of initialization function for the weights of the layer
            (see [initializers](https://keras.io/initializers)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](https://keras.io/activations)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        padding: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        strides: tuple of length 2. Factor by which to strides output.
            Also called strides elsewhere.
        kernel_regularizer: instance of [WeightRegularizer](
            https://keras.io/regularizers)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        bias_regularizer: instance of [WeightRegularizer](
            https://keras.io/regularizers), applied to the use_bias.
        activity_regularizer: instance of [ActivityRegularizer](
            https://keras.io/regularizers), applied to the network output.
        kernel_constraint: instance of the [constraints](
            https://keras.io/constraints) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        bias_constraint: instance of the [constraints](
            https://keras.io/constraints) module, applied to the use_bias.
        data_format: 'channels_first' or 'channels_last'.
            In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode is it at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `'channels_last'`.
        use_bias: whether to include a use_bias
            (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, nekernel_rows, nekernel_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, nekernel_rows, nekernel_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.


    # References
        - [Cosine Normalization: Using Cosine Similarity Instead
           of Dot Product in Neural Networks](https://arxiv.org/pdf/1702.05870.pdf)
    """

    def __init__(self, filters, kernel_size,
                 kernel_initializer='glorot_uniform', activation=None, weights=None,
                 padding='valid', strides=(1, 1), data_format=None,
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        if padding not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for CosineConvolution2D:', padding)
        self.filters = filters
        self.kernel_size = kernel_size
        self.nb_row, self.nb_col = self.kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.padding = padding
        self.strides = tuple(strides)
        self.data_format = normalize_data_format(data_format)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(CosineConvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            stack_size = input_shape[1]
            self.kernel_shape = (self.filters, stack_size, self.nb_row, self.nb_col)
            self.kernel_norm_shape = (1, stack_size, self.nb_row, self.nb_col)
        elif self.data_format == 'channels_last':
            stack_size = input_shape[3]
            self.kernel_shape = (self.nb_row, self.nb_col, stack_size, self.filters)
            self.kernel_norm_shape = (self.nb_row, self.nb_col, stack_size, 1)
        else:
            raise ValueError('Invalid data_format:', self.data_format)
        self.W = self.add_weight(shape=self.kernel_shape,
                                 initializer=partial(self.kernel_initializer),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        kernel_norm_name = '{}_kernel_norm'.format(self.name)
        self.kernel_norm = K.variable(np.ones(self.kernel_norm_shape),
                                      name=kernel_norm_name)

        if self.use_bias:
            self.b = self.add_weight(shape=(self.filters,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid data_format:', self.data_format)

        rows = conv_output_length(rows, self.nb_row,
                                  self.padding, self.strides[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.padding, self.strides[1])

        if self.data_format == 'channels_first':
            return input_shape[0], self.filters, rows, cols
        elif self.data_format == 'channels_last':
            return input_shape[0], rows, cols, self.filters

    def call(self, x, mask=None):
        b, xb = 0., 0.
        if self.data_format == 'channels_first':
            kernel_sum_axes = [1, 2, 3]
            if self.use_bias:
                b = K.reshape(self.b, (self.filters, 1, 1, 1))
                xb = 1.
        elif self.data_format == 'channels_last':
            kernel_sum_axes = [0, 1, 2]
            if self.use_bias:
                b = K.reshape(self.b, (1, 1, 1, self.filters))
                xb = 1.

        tmp = K.sum(K.square(self.W), axis=kernel_sum_axes, keepdims=True)
        Wnorm = K.sqrt(tmp + K.square(b) + K.epsilon())

        tmp = KC.conv2d(K.square(x), self.kernel_norm, strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        filter_shape=self.kernel_norm_shape)
        xnorm = K.sqrt(tmp + xb + K.epsilon())

        W = self.W / Wnorm

        output = KC.conv2d(x, W, strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           filter_shape=self.kernel_shape)

        if K.backend() == 'theano':
            xnorm = K.pattern_broadcast(xnorm, [False, True, False, False])

        output /= xnorm

        if self.use_bias:
            b /= Wnorm
            if self.data_format == 'channels_first':
                b = K.reshape(b, (1, self.filters, 1, 1))
            elif self.data_format == 'channels_last':
                b = K.reshape(b, (1, 1, 1, self.filters))
            else:
                raise ValueError('Invalid data_format:', self.data_format)
            b /= xnorm
            output += b
        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'use_bias': self.use_bias}
        base_config = super(CosineConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


CosineConv2D = CosineConvolution2D
get_custom_objects().update({'CosineConvolution2D': CosineConvolution2D})
get_custom_objects().update({'CosineConv2D': CosineConv2D})


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


get_custom_objects().update({'SubPixelUpscaling': SubPixelUpscaling})

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
     """Capsule Layer implementation in Keras
       
       The Capsule Layer is a Neural Network Layer which helps modeling relationships in image and sequential data better.
       It achieves this by understanding the spatial relationships between objects (in images) or words (in text) by encoding
       additional information about the image or text, such as angle of rotation, thickness and brightness, relative proportions etc.
       This layer can be used instead of pooling layers to lower dimensions and still capture important information about the 
       relationships and structures within the data. A normal pooling layer would lose a lot of this information.
       
       This layer can be used on the output of any layer which has a 3-D output (including batch_size). For example, in 
       image classification, it can be used on the output of a Conv2D layer for Computer Vision applications. Also, it can be 
       used on the output of a GRU or LSTM Layer (Bidirectional or Unidirectional) for NLP applications
       
       # Example usage :
       
           1). Computer Vision
           input_image = Input(shape=(None, None, 3))
           conv_2d = Conv2D(64, (3, 3), activation='relu')(input_image)
           capsule = Capsule(num_capsule=10, dim_capsule=16, routings=3, share_weights=True)(conv_2d)
       
           2). NLP
           maxlen = 72
           max_features = 120000
           input_text = Input(shape=(maxlen,))
           embedding = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(input_text)
           bi_gru = Bidirectional(GRU(64, return_seqeunces=True))(embedding)
           capsule = Capsule(num_capsule=5, dim_capsule=5, routings=4, share_weights=True)(bi_gru)
           
       # Arguments
           num_capsule : Number of Capsules (int)
           dim_capsules : Dimensions of the vector output of each Capsule (int)
           routings : Number of dynamic routings in the Capsule Layer (int)
           share_weights : Whether to share weights between Capsules or not (boolean)
           
       # Input shape
            3D tensor with shape:
            (batch_size, input_num_capsule, input_dim_capsule) [any 3-D Tensor with the first dimension as batch_size]
        
       # Output shape
            3D tensor with shape:
            (batch_size, num_capsule, dim_capsule)
       # References
        - [Dynamic-Routing-Between-Capsules](https://arxiv.org/pdf/1710.09829.pdf)
        - [Keras-Examples-CIFAR10-CNN-Capsule](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn_capsule.py)"""
        
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, axis=1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    
    def get_config(self):
        config = {'num_capsule': self.num_capsule,
                  'dim_capsule': self.dim_capsule,
                  'routings': self.routings,
                  'share_weights': self.share_weights, 
                  'activation': activations.serialize(self.activation)}
        
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
 
get_custom_objects().update({'Capsule': Capsule})
