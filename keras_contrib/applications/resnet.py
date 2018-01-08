"""ResNet v1, v2, and segmentation models for Keras.

# Reference

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

Reference material for extended functionality:

- [ResNeXt](https://arxiv.org/abs/1611.05431) for Tiny ImageNet support.
- [Dilated Residual Networks](https://arxiv.org/pdf/1705.09914) for segmentation support.
- [Deep Residual Learning for Instrument Segmentation in Robotic Surgery](https://arxiv.org/abs/1703.08580)
  for segmentation support.

Implementation adapted from: github.com/raghakot/keras-resnet
"""
from __future__ import division

import six
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


def _bn_relu(x, bn_name=None, relu_name=None, verbose=False):
    """Helper to build a BN -> relu block
    """
    if verbose:
        print("    _bn_relu")
    norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    verbose = conv_params.setdefault("verbose", False)

    def f(x):
        if verbose:
            print("    _conv_bn_relu")

        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)

        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name, verbose=verbose)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation function.
    This is the ResNet v2 scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    verbose = conv_params.setdefault("verbose", False)

    def f(x):
        if verbose:
            print("    _bn_relu_conv")
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name, verbose=verbose)

        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name)(activation)

    return f


def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None,
              verbose=False):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input_feature)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if verbose:
            print("    SHORTCUT not right: input -", shortcut.shape, "residual -", residual.shape)
            print("    reshaping via a convolution...")

        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'

        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name=conv_name_base)(input_feature)

        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'

        shortcut = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base)(shortcut)

    if verbose:
        print("    SHORTCUT : input -", shortcut.shape, "residual -", residual.shape)

    return add([shortcut, residual])


def _residual_block(block_function, filters, blocks, stage,
                    transition_strides=None, transition_dilation_rates=None,
                    dilation_rates=None, is_first_layer=False, dropout=None,
                    residual_unit=_bn_relu_conv, verbose=False):
    """Builds a residual block with repeating bottleneck blocks.

       stage: integer, current stage label, used for generating layer names
       blocks: number of blocks 'a','b'..., current block label, used for generating layer names
       transition_strides: a list of tuples for the strides of each transition
       transition_dilation_rates: a list of tuples for the dilation rate of each transition
    """
    if transition_dilation_rates is None:
        transition_dilation_rates = [(1, 1)] * blocks
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [(1, 1)] * blocks

    def f(x):
        for i in range(blocks):
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=(is_first_layer and i == 0),
                               dropout=dropout,
                               residual_unit=residual_unit,
                               verbose=verbose)(x)
        return x

    return f


def _block_name_base(stage, block):
    """Get the convolution name base and batch normalization name base defined by stage and block.

    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the paper and keras
    and beyond 26 blocks they will simply be numbered.
    """
    if block >= 0 and block < 26:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    return conv_name_base, bn_name_base


def basic_block(filters, stage, block, transition_strides=(1, 1),
                dilation_rate=(1, 1), is_first_block_of_first_layer=False,
                dropout=None, residual_unit=_bn_relu_conv, verbose=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input_features):
        if verbose:
            print("    basic block")

        conv_name_base, bn_name_base = _block_name_base(stage, block)

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name=conv_name_base + '2a',
                              bn_name=bn_name_base + '2a',
                              verbose=verbose)(input_features)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b',
                          verbose=verbose)(x)

        return _shortcut(input_features, x,
                         conv_name_base=conv_name_base, bn_name_base=bn_name_base,
                         verbose=verbose)

    return f


def bottleneck(filters, stage, block, transition_strides=(1, 1),
               dilation_rate=(1, 1), is_first_block_of_first_layer=False,
               dropout=None, residual_unit=_bn_relu_conv,
               verbose=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input_feature):
        if verbose:
            print("    bottleneck block")

        conv_name_base, bn_name_base = _block_name_base(stage, block)

        if is_first_block_of_first_layer and residual_unit == _bn_relu_conv:
            # ResNetv2: don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv2D(filters=filters, kernel_size=(1, 1),
                       strides=transition_strides,
                       dilation_rate=dilation_rate,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4),
                       name=conv_name_base + '2a')(input_feature)
        else:
            # ResNetv1: *DOES* call here on the first block and other blocks.
            # ResNetv2: *DOES NOT* call here on the first block, but does otherwise.
            x = residual_unit(filters=filters, kernel_size=(1, 1),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name=conv_name_base + '2a',
                              bn_name=bn_name_base + '2a',
                              verbose=verbose)(input_feature)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name=conv_name_base + '2b',
                          bn_name=bn_name_base + '2b',
                          verbose=verbose)(x)

        if dropout is not None:
            x = Dropout(dropout)(x)

        x = residual_unit(filters=filters * 4, kernel_size=(1, 1),
                          conv_name_base=conv_name_base + '2c',
                          bn_name_base=bn_name_base + '2c',
                          verbose=verbose)(x)

        return _shortcut(input_feature, x,
                         conv_name_base=conv_name_base, bn_name_base=bn_name_base,
                         verbose=verbose)

    return f


def _handle_dim_ordering(verbose=False):
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if verbose:
        print("_handle_dim_ordering")
    if K.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _string_to_function(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def obtain_input_shape(input_shape, require_flatten=True):
    # Determine proper input shape
    return _obtain_input_shape(input_shape,
                               default_size=224,
                               min_size=8,
                               data_format=K.image_data_format(),
                               require_flatten=require_flatten)


def ResNet(input_shape=None, classes=10, block='bottleneck', residual_unit='v2', repetitions=None,
           initial_filters=64, activation='softmax', include_top=True, input_tensor=None,
           dropout=None, transition_dilation_rate=(1, 1), initial_strides=(2, 2),
           initial_kernel_size=(7, 7), initial_pooling='max', initial_padding='same', final_pooling=None,
           top='classification', weights=None, verbose=False):
    """Builds a custom ResNet like architecture. Defaults to ResNet50 v2.

    Args:
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` dim ordering)
            or `(3, 224, 224)` (with `channels_first` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 8.
            E.g. `(224, 224, 3)` would be one valid value.
        classes: The number of outputs at final softmax layer
        block: The block function to use. This is either `'basic'` or `'bottleneck'`.
            The original paper used `basic` for layers < 50.
        residual_unit: the basic residual unit, 'v1' for conv bn relu, 'v2' for bn relu conv.
            See [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
            for details.
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are doubled and the input size is halved.
            Default of None implies the ResNet50v2 values of [3, 4, 6, 3].
        initial_filters: Number of features in the first layer of ResNet.
        activation: Actication of classifier (last) layer of ResNet.
            Must be one of "softmax", "sigmoid", or None'.
        include_top: Boolean indicating whether to include the classification (last)
            layer in the built model or not.
        input_tensor: (Optional) A sample tensor to set the input_shape, dtype, etc.
        dropout: None for no dropout, otherwise rate of dropout from 0 to 1.
            Based on [Wide Residual Networks.(https://arxiv.org/pdf/1605.07146) paper.
        transition_dilation_rate: Dilation rate for transition layers. Used for
            pixel-wise prediction tasks such as image segmentation. For semantic
            segmentation of images, use a dilation rate of (2, 2).
        initial_strides: Stride of the very first residual unit and MaxPooling2D call,
            with default (2, 2), set to (1, 1) for small images like cifar.
        initial_kernel_size: kernel size of the very first convolution, (7, 7) for imagenet
            and (3, 3) for small image datasets like tiny imagenet and cifar.
            See [ResNeXt](https://arxiv.org/abs/1611.05431) paper for details.
        initial_pooling: Determine if there will be an initial pooling layer,
            'max' for imagenet and None for small image datasets.
            See [ResNeXt](https://arxiv.org/abs/1611.05431) paper for details.
        initial_padding: One of 'valid' or 'same' applied to the initial MaxPooling2D
            so it is possible to match the Keras ResNet2D weight dimensions exactly (case-insensitive).
        final_pooling: Optional pooling mode for feature extraction at the final model layer
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        top: Defines final layers to evaluate based on a specific problem type. Options are
            'classification' for ImageNet style problems, 'segmentation' for problems like
            the Pascal VOC dataset, and None to exclude these layers entirely.
        weights: Path to a pre-trained keras weights file which will be loaded from disk.
        verbose: True prints layer names as they are built, False disables the printouts.

    Returns:
        The keras `Model`.
    """
    if activation not in ['softmax', 'sigmoid', None]:
        raise ValueError('activation must be one of "softmax", "sigmoid", or None')
    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')
    if repetitions is None:
        repetitions = [3, 4, 6, 3]

    if len(input_shape) != 3:
        if K.image_data_format() == 'channels_last':
            raise Exception("channels_last: Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
        else:
            raise Exception("channels_first: Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    # Determine proper input shape
    input_shape = obtain_input_shape(input_shape=input_shape, require_flatten=include_top)

    _handle_dim_ordering(verbose=verbose)

    if block == 'basic':
        block_fn = basic_block
    elif block == 'bottleneck':
        block_fn = bottleneck
    elif isinstance(block, six.string_types):
        block_fn = _string_to_function(block)
    else:
        block_fn = block

    if residual_unit == 'v2':
        residual_unit = _bn_relu_conv
    elif residual_unit == 'v1':
        residual_unit = _conv_bn_relu
    elif isinstance(residual_unit, six.string_types):
        residual_unit = _string_to_function(residual_unit)
    else:
        residual_unit = residual_unit

    # Initial
    img_input = Input(shape=input_shape, tensor=input_tensor)
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size, strides=initial_strides,
                      verbose=verbose, conv_name='conv1', bn_name='bn_conv1')(img_input)
    if initial_pooling == 'max':
        if verbose:
            print("    MaxPooling2D")
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding=initial_padding)(x)

    block = x
    filters = initial_filters
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        stage = i + 2  # First Conv2D is stage 1, i == 0 is stage 2
        block = _residual_block(block_fn, filters=filters,
                                stage=stage, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit,
                                verbose=verbose)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block, verbose=verbose)

    # Classifier block
    if include_top and top is 'classification':
        if verbose:
            print("    classification")
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=classes, activation=activation, kernel_initializer="he_normal", name='fc' + str(classes))(x)

    elif include_top and top is 'segmentation':
        if verbose:
            print("    segmentation")
        x = Conv2D(classes, (1, 1), activation='linear', padding='same')(x)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        x = Reshape((row * col, classes))(x)
        x = Activation(activation)(x)
        x = Reshape((row, col, classes))(x)

    elif final_pooling == 'avg':
        if verbose:
            print("    GlobalAveragePooling2D")
        x = GlobalAveragePooling2D()(x)

    elif final_pooling == 'max':
        if verbose:
            print("    GlobalMaxPooling2D")
        x = GlobalMaxPooling2D()(x)

    model = Model(inputs=img_input, outputs=x)

    if weights is not None and weights != 'imagenet':
        model.load_weights(weights)

    return model


def ResNet18(input_shape=None, classes=10, **kwargs):
    """ResNet with 18 layers and v2 residual units
    """
    return ResNet(input_shape, classes, basic_block, repetitions=[2, 2, 2, 2], **kwargs)


def ResNet34(input_shape=None, classes=10, **kwargs):
    """ResNet with 34 layers and v2 residual units
    """
    return ResNet(input_shape, classes, basic_block, repetitions=[3, 4, 6, 3], **kwargs)


def ResNet50(input_shape=None, classes=1000, **kwargs):
    """ResNet with 50 layers and v2 residual units
    """
    return ResNet(input_shape, classes, bottleneck, repetitions=[3, 4, 6, 3], **kwargs)


def ResNet101(input_shape=None, classes=1000, **kwargs):
    """ResNet with 101 layers and v2 residual units
    """
    return ResNet(input_shape, classes, bottleneck, repetitions=[3, 4, 23, 3], **kwargs)


def ResNet152(input_shape=None, classes=1000, **kwargs):
    """ResNet with 152 layers and v2 residual units
    """
    return ResNet(input_shape, classes, bottleneck, repetitions=[3, 8, 36, 3], **kwargs)
