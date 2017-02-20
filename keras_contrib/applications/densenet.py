# -*- coding: utf-8 -*-
"""DenseNet models for Keras.

# Reference

- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) for image labeling
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
   for Fully Convolutional Network (FCN) based image segmentation

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

TH_WEIGHTS_PATH = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Theano-Backend-TH-dim-ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Tensorflow-Backend-TF-dim-ordering.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Theano-Backend-TH-dim-ordering-no-top.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/titu1994/DenseNet/releases/download/v2.0/DenseNet-40-12-Tensorflow-Backend-TF-dim-ordering-no-top.h5'


def DenseNet(depth=40, nb_dense_block=3, growth_rate=12, nb_dense_block_layers=-1, nb_filter=16,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1E-4,
             include_top=True, weights='cifar10',
             input_tensor=None, input_shape=None,
             classes=10, segmentation=False):
    """Instantiate the DenseNet architecture,
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
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_dense_block_layers: number of layers in each dense block. 
                   Can be an -1, a positive integer or a list
                   If -1, it computes the nb_layer from depth.
                   If positive integer, a set number of layers per dense block.
                   If list, nb_layer is used as provided.
                   Note that list size must be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
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
            segmentation: False default produces single label per image with 
                DenseNet structure, True produces segmentation with single 
                label per pixel with FCN Densenet structure.

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

    x = __create_dense_net(classes, img_input, include_top=include_top, depth=depth, 
                           nb_dense_block=nb_dense_block, growth_rate=growth_rate, 
                        nb_filter=nb_filter, nb_dense_block_layers=nb_dense_block_layers,
                        bottleneck=bottleneck, reduction=reduction, 
                        dropout_rate=dropout_rate, weight_decay=weight_decay, 
                        segmentation=segmentation, input_shape=input_shape)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    # load weights
    if weights == 'cifar10':
        if (depth == 40) and (nb_dense_block == 3) and (growth_rate == 12) and (nb_filter == 16) and \
                (bottleneck is False) and (reduction == 0.0) and (dropout_rate == 0.0) and (weight_decay == 1E-4):
            # Default parameters match. Weights for this model exist:

            if K.image_dim_ordering() == 'th':
                if include_top:
                    weights_path = get_file('densenet_40_12_th_dim_ordering_th_kernels.h5',
                                            TH_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('densenet_40_12_th_dim_ordering_th_kernels_no_top.h5',
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
                    weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels.h5',
                                            TF_WEIGHTS_PATH,
                                            cache_subdir='models')
                else:
                    weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels_no_top.h5',
                                            TF_WEIGHTS_PATH_NO_TOP,
                                            cache_subdir='models')

                model.load_weights(weights_path)

                if K.backend() == 'theano':
                    convert_all_kernels_in_model(model)

    return model


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Convolution2D(inter_channel, 1, 1, init='he_uniform', border_mode='same', bias=False,
                          W_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __transition_down_block(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D

    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2), border_mode='same')(x)

    return x


def __transition_up_block(ip, nb_filters, output_shape=None, weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)

    Args:
        ip: keras tensor
        nb_filters: number of layers
        output_shape: required if type = 'deconv'. Output shape of tensor
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''
    x = Deconvolution2D(nb_filters, 3, 3, output_shape, activation='relu', border_mode='same',
                        subsample=(2, 2))(ip)

    return x


def __dense_block(x, nb_dense_block_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_dense_block_layers: the number of layers of __conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_dense_block_layers of __conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x_list = [x]

    for i in range(nb_dense_block_layers):
        x = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(x)
        x = merge(x_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def __create_dense_net(nb_classes, img_input, include_top=True, depth=40, nb_dense_block=5, growth_rate=12, 
                        nb_filter=16, nb_dense_block_layers=4, bottleneck=False, reduction=0.0, 
                        dropout_rate=None, weight_decay=1E-4, segmentation=False, input_shape=None):
    ''' Build the DenseNet model

    Args:
        nb_classes: number of classes
        img_input: tensor of images with shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers for image labeling 
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_dense_block_layers: number of layers in each dense block. Can be an -1, a positive integer or a list

                   If -1, it computes the nb_layer from depth

                   If positive integer, a set number of layers per dense block

                   If list, nb_layer is used as provided.
                   Note that list size must be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        input_shape: input tensor shape (channels, rows, columns) or (rows, columns, channels)
                     needed when segmentation=True to define the output shape

    Returns: keras tensor with nb_dense_block_layers of __conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    if segmentation == False:
        assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
    else:
        if concat_axis == 1: # th dim ordering
            _, rows, cols = input_shape
        else:
            rows, cols, _ = input_shape

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # layers in each dense block
    if type(nb_dense_block_layers) is int and nb_dense_block_layers == -1:
        if segmentation == False:
            nb_dense_block_layers = int((depth - 4) / 3)
        else:
            nb_dense_block_layers = 4

    # layers in each dense block
    if type(nb_dense_block_layers) is list or type(nb_dense_block_layers) is tuple:
        nb_dense_block_layers = list(nb_dense_block_layers) # Convert tuple to list

        assert len(nb_dense_block_layers) == (nb_dense_block + 1), "If list, nb_layer is used as provided. " \
                                                        "Note that list size must be (nb_dense_block + 1)"

        final_nb_layer = nb_dense_block_layers[-1]
        nb_dense_block_layers = nb_dense_block_layers[:-1]

    else:
        final_nb_layer = nb_dense_block_layers
        nb_dense_block_layers = [nb_dense_block_layers] * nb_dense_block

    if bottleneck:
        nb_dense_block_layers = [int(layer // 2) for layer in nb_dense_block_layers]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))(img_input)

    if segmentation == False:
         # Add dense blocks for image labeling
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block(x, nb_dense_block_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                        dropout_rate=dropout_rate, weight_decay=weight_decay)
            # add __transition_block
            x = __transition_down_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        # The last __dense_block does not have a __transition_block
        x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        # image labeling version of final activation layer
        if include_top:
            x = Dense(nb_classes, activation='softmax', W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(x)
        return x

    else:
        # perform image segmentation using FCN densenet
        skip_connection = x
        skip_list = []
        # Add dense blocks and transition down block for image segmentation
        for block_idx in range(nb_dense_block):
            x, nb_filter = __dense_block(x, nb_dense_block_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                        dropout_rate=dropout_rate, weight_decay=weight_decay)
            # Skip connection
            x = merge([x, skip_connection], mode='concat', concat_axis=concat_axis)
            skip_list.append(x)

            # add transition_block
            x = transition_down_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression) # this is calculated inside transition_down_block

            # Preserve transition for next skip connection after dense
            skip_connection = x

        # The last dense_block does not have a transition_down_block
        x, nb_filter = dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

        if K.image_dim_ordering() == 'th':
            out_shape = [batch_size, nb_filter, rows // 16, cols // 16]
        else:
            out_shape = [batch_size, rows // 16, cols // 16, nb_filter]

        # Add dense blocks and transition up block
        for block_idx in range(nb_dense_block):
            if K.image_dim_ordering() != 'th':
                out_shape[3] = nb_filter

            x = __transition_up_block(x, nb_filters=nb_filter, type=upscaling_type, output_shape=out_shape)

            if K.image_dim_ordering() == 'th':
                out_shape[2] *= 2
                out_shape[3] *= 2
            else:
                out_shape[1] *= 2
                out_shape[2] *= 2

            x = merge([x, skip_list.pop()], mode='concat', concat_axis=concat_axis)

            x, nb_filter = dense_block(x, nb_dense_block_layers[-block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)

        x = Convolution2D(nb_classes, 1, 1, activation='linear', border_mode='same', W_regularizer=l2(weight_decay),
                        bias=False)(x)
        # Image Segmentation version of final Activation layer
        if include_top:
            if K.image_dim_ordering() == 'th':
                channel, row, col = input_shape
            else:
                row, col, channel = input_shape

            x = Reshape((row * col, nb_classes))(x)
            x = Activation('softmax')(x)
            x = Reshape((row,col,nb_classes))(x)

    return x
