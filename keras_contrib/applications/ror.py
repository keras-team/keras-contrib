# -*- coding: utf-8 -*-
"""Residual of Residual Network models for Keras.

# Reference

- [Residual Networks of Residual Networks: Multilevel Residual Networks](https://arxiv.org/abs/1608.02908)

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

TH_WEIGHTS_PATH = 'https://github.com/titu1994/Residual-of-Residual-Networks/releases/download/v0.2/ror_wrn_40_2_th_kernels_th_dim_ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/titu1994/Residual-of-Residual-Networks/releases/download/v0.2/ror_wrn_40_2_tf_kernels_tf_dim_ordering.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/Residual-of-Residual-Networks/releases/download/v0.2/ror_wrn_40_2_th_kernels_th_dim_ordering_no_top.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/Residual-of-Residual-Networks/releases/download/v0.2/ror_wrn_40_2_tf_kernels_tf_dim_ordering_no_top.h5'


def ResidualOfResidual(depth=40, width=2, dropout_rate=0.0,
                       include_top=True, weights='cifar10',
                       input_tensor=None, input_shape=None,
                       classes=10):
    """Instantiate the Residual of Residual Network architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_dim_ordering="tf"` in your Keras config
        at ~/.keras/keras.json.

        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.

        # Arguments
            depth: number or layers in the DenseNet
            width: width of the network
            dropout_rate: dropout rate
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                "cifar10" (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `tf` dim ordering)
                or `(3, 32, 32)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.
    """

    if weights not in {'cifar10', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'cifar10' and include_top and classes != 10:
        raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
                         ' as true, `classes` should be 10')

    if (depth - 4) % 6 != 0:
        raise ValueError('Depth of the network must be such that (depth - 4)'
                         'should be divisible by 6.')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_pre_residual_of_residual(classes, img_input, include_top,
                                          depth, width, dropout_rate)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='residual_of_residual')

    # load weights
    if weights == 'cifar10':
        if (depth == 40) and (width == 2) and (dropout_rate == 0.0):
            # Default parameters match. Weights for this model exist:

            if K.image_dim_ordering() == 'th':
                if include_top:
                    weights_path = get_file('ror_wrn_40_2_th_kernels_th_dim_ordering.h5',
                                            TH_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('ror_wrn_40_2_th_kernels_th_dim_ordering_no_top.h5',
                                            TH_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)

                if K.backend() == 'tensorflow':
                    warnings.warn('You are using the TensorFlow backend, yet you '
                                  'are using the Theano '
                                  'image dimension ordering convention '
                                  '(`image_dim_ordering="th"`). '
                                  'For best performance, set '
                                  '`image_dim_ordering="tf"` in '
                                  'your Keras config '
                                  'at ~/.keras/keras.json.')
                    convert_all_kernels_in_model(model)
            else:
                if include_top:
                    weights_path = get_file('ror_wrn_40_2_tf_kernels_tf_dim_ordering_no_top.h5',
                                            TF_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('ror_wrn_40_2_tf_kernels_tf_dim_ordering_no_top.h5',
                                            TF_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)

                if K.backend() == 'theano':
                    convert_all_kernels_in_model(model)

    return model


def __initial_conv_block(input, k=1, dropout=0.0, initial=False):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if initial:
        if K.image_dim_ordering() == "th":
            init = Convolution2D(16 * k, 1, 1, init='he_normal', border_mode='same')(init)
        else:
            init = Convolution2D(16 * k, 1, 1, init='he_normal', border_mode='same')(init)

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, 3, 3, border_mode='same', init='he_normal')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, 3, 3, border_mode='same', init='he_normal')(x)

    m = merge([init, x], mode='sum')
    return m


def __conv_block(input, nb_filters=32, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 32 * k, else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != nb_filters * k:
            init = Convolution2D(nb_filters * k, 1, 1, init='he_normal', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != nb_filters * k:
            init = Convolution2D(nb_filters * k, 1, 1, init='he_normal', border_mode='same')(init)

    x = BatchNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters * k, 3, 3, border_mode='same', init='he_normal')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filters * k, 3, 3, border_mode='same', init='he_normal')(x)

    m = merge([init, x], mode='sum')
    return m


def __create_pre_residual_of_residual(nb_classes, img_input, include_top, depth=28, width=1, dropout=0.0):
    '''
    Creates a Residual Network of Residual Network with specified parameters

    Example : To create a Pre-RoR model, use k = 1
              model = ResidualOfResidual(depth=28, width=1) # Pre-RoR-3

              To create a RoR-WRN model, use k > 1
              model = ResidualOfResidual(depth=28, width=10) # Pre-RoR-3,  RoR-3-WRN-28-10

    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: depth of the network
        width: width of the network
        dropout: Adds dropout if value is greater than 0.0.
                 Note : Generally not used in RoR

    Returns: a Keras Model
    '''

    N = (depth - 4) // 6

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Initial convolution layer
    x = Convolution2D(16, 3, 3, border_mode='same', init='he_normal')(img_input)
    nb_conv = 4  # Dont count 4 long residual connections in WRN models

    conv0_level1_shortcut = Convolution2D(64 * width, 1, 1, border_mode='same', subsample=(4, 4),
                                          name='conv0_level1_shortcut')(x)

    conv1_level2_shortcut = Convolution2D(16 * width, 1, 1, border_mode='same',
                                          name='conv1_level2_shortcut')(x)
    for i in range(N):
        initial = (i == 0)
        x = __initial_conv_block(x, k=width, dropout=dropout, initial=initial)
        nb_conv += 2

    # Add Level 2 shortcut
    x = merge([x, conv1_level2_shortcut], mode='sum')

    x = MaxPooling2D((2, 2))(x)

    conv2_level2_shortcut = Convolution2D(32 * width, 1, 1, border_mode='same',
                                          name='conv2_level2_shortcut')(x)
    for i in range(N):
        x = __conv_block(x, k=width, dropout=dropout)
        nb_conv += 2

    # Add Level 2 shortcut
    x = merge([x, conv2_level2_shortcut], mode='sum')

    x = MaxPooling2D((2, 2))(x)

    conv3_level2_shortcut = Convolution2D(64 * width, 1, 1, border_mode='same',
                                          name='conv3_level2_shortcut')(x)
    for i in range(N):
        x = __conv_block(x, nb_filters=64, k=width, dropout=dropout)
        nb_conv += 2

    # Add Level 2 shortcut
    x = merge([x, conv3_level2_shortcut], mode='sum')

    # Add Level 1 shortcut
    x = merge([x, conv0_level1_shortcut], mode='sum')
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)

    return x
