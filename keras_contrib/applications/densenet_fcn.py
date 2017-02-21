# -*- coding: utf-8 -*-
"""FCN DenseNet Models for Keras.

# Reference

- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D, AtrousConvolution2D, UpSampling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

from keras_contrib.layers.convolutional import SubPixelUpscaling


def DenseNetFCN(nb_dense_block=5, growth_rate=12, nb_filter=16, nb_layers_per_block=4,
                bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1E-4,
                include_top=True, weights=None,
                input_tensor=None, input_shape=None, classes=10,
                upsampling_conv=128, upsampling_type='upscaling', batchsize=None):
    """Instantiate the DenseNet FCN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_dim_ordering="tf"` in your Keras config
        at ~/.keras/keras.json.

        # Arguments
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a positive integer or a list.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
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
            upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
            upsampling_type: Can be one of 'upsampling', 'deconv', 'atrous' and
                'subpixel'. Defines type of upsampling algorithm used.
            batchsize: Fixed batch size. This is a temporary requirement for
                computation of output shape in the case of Deconvolution2D layers.
                Parameter will be removed in next iteration of Keras, which infers
                output shape of deconvolution layers automatically.

        # Returns
            A Keras model instance.
    """

    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv', 'atrous', 'subpixel']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv", "atrous" or "subpixel".')

    if upsampling_type == 'deconv' and batchsize is None:
        raise ValueError('If "upsampling_type" is deconvoloution, then a fixed '
                         'batch size must be provided in batchsize parameter.')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. Argument '
                         'value was %d.' % (nb_layers_per_block))

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_fcn_dense_net(classes, img_input, include_top, nb_dense_block,
                               growth_rate, nb_filter, bottleneck, reduction,
                               dropout_rate, weight_decay, nb_layers_per_block,
                               upsampling_conv, upsampling_type, batchsize, input_shape)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='fcn-densenet')

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
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_layers: the number of layers of __conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_layers of __conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x_list = [x]

    for i in range(nb_layers):
        x = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(x)
        x = merge(x_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def __transition_up_block(ip, nb_filters, type='upsampling', output_shape=None, weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)

    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv', or 'atrous'. Determines type of upsampling performed
        output_shape: required if type = 'deconv'. Output shape of tensor
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    if type == 'upsampling':
        x = UpSampling2D()(ip)
    elif type == 'subpixel':
        x = Convolution2D(nb_filters, 3, 3, activation="relu", border_mode='same', W_regularizer=l2(weight_decay),
                          bias=False)(ip)
        x = SubPixelUpscaling(r=2, channels=int(nb_filters // 4))(x)
        x = Convolution2D(nb_filters, 3, 3, activation="relu", border_mode='same', W_regularizer=l2(weight_decay),
                          bias=False)(x)
    elif type == 'atrous':
        # waiting on https://github.com/fchollet/keras/issues/4018
        x = AtrousConvolution2D(nb_filters, 3, 3, activation="relu", W_regularizer=l2(weight_decay),
                                bias=False, atrous_rate=(2, 2))(ip)
    else:
        x = Deconvolution2D(nb_filters, 3, 3, output_shape, activation='relu', border_mode='same',
                            subsample=(2, 2))(ip)

    return x


def __create_fcn_dense_net(nb_classes, img_input, include_top, nb_dense_block=5, growth_rate=12,
                           nb_filter=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4,
                           nb_layers_per_block=4, upsampling_conv=128, upsampling_type='upsampling', batchsize=None,
                           input_shape=None):
    ''' Build the DenseNet model

    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        upsampling_type: Can be one of 'upsampling', 'deconv', 'atrous' and
            'subpixel'. Defines type of upsampling algorithm used.
        batchsize: Fixed batch size. This is a temporary requirement for
            computation of output shape in the case of Deconvolution2D layers.
            Parameter will be removed in next iteration of Keras, which infers
            output shape of deconvolution layers automatically.
        input_shape: Only used for shape inference in fully convolutional networks.

    Returns: keras tensor with nb_layers of __conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    if concat_axis == 1:  # th dim ordering
        _, rows, cols = input_shape
    else:
        rows, cols, _ = input_shape

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # check if upsampling_conv has minimum number of filters
    # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
    assert upsampling_conv > 12 and upsampling_conv % 4 == 0, "Parameter `upsampling_conv` number of channels must " \
                                                              "be a positive number divisible by 4 and greater " \
                                                              "than 12"

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), "If list, nb_layer is used as provided. " \
                                                       "Note that list size must be (nb_dense_block + 1)"

        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]

    else:
        final_nb_layer = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * nb_dense_block

    if bottleneck:
        nb_layers = [int(layer // 2) for layer in nb_layers]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Convolution2D(48, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))(img_input)

    skip_connection = x
    skip_list = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Skip connection
        x = merge([x, skip_connection], mode='concat', concat_axis=concat_axis)
        skip_list.append(x)

        # add transition_block
        x = __transition_down_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                                    weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

        # Preserve transition for next skip connection after dense
        skip_connection = x

    # The last dense_block does not have a transition_down_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    if K.image_dim_ordering() == 'th':
        out_shape = [batchsize, nb_filter, rows // 16, cols // 16]
    else:
        out_shape = [batchsize, rows // 16, cols // 16, nb_filter]

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        if K.image_dim_ordering() != 'th':
            out_shape[3] = nb_filter

        x = __transition_up_block(x, nb_filters=nb_filter, type=upsampling_type, output_shape=out_shape)

        if K.image_dim_ordering() == 'th':
            out_shape[2] *= 2
            out_shape[3] *= 2
        else:
            out_shape[1] *= 2
            out_shape[2] *= 2

        x = merge([x, skip_list.pop()], mode='concat', concat_axis=concat_axis)

        x, nb_filter = __dense_block(x, nb_layers[-block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = Convolution2D(nb_classes, 1, 1, activation='linear', border_mode='same', W_regularizer=l2(weight_decay),
                      bias=False)(x)

    if K.image_dim_ordering() == 'th':
        channel, row, col = input_shape
    else:
        row, col, channel = input_shape

    x = Reshape((row * col, nb_classes))(x)

    x = Activation('softmax')(x)

    x = Reshape((row, col, nb_classes))(x)

    return x
