<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/convolutional.py#L19)</span>
### CosineConvolution2D

```python
keras_contrib.layers.CosineConvolution2D(filters, kernel_size, kernel_initializer='glorot_uniform', activation=None, weights=None, padding='valid', strides=(1, 1), data_format=None, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True)
```

Cosine Normalized Convolution operator for filtering windows of two-dimensional inputs.
Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks
https://arxiv.org/pdf/1702.05870.pdf

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the sample axis),
e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

__Examples__


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

__Arguments__

- __filters__: Number of convolution filters to use.
- __kernel_size__: kernel_size: An integer or tuple/list of 2 integers, specifying the
    dimensions of the convolution window.
- __init__: name of initialization function for the weights of the layer
    (see [initializers](../initializers.md)), or alternatively,
    Theano function to use for weights initialization.
    This parameter is only relevant if you don't pass
    a `weights` argument.
- __activation__: name of activation function to use
    (see [activations](../activations.md)),
    or alternatively, elementwise Theano function.
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: a(x) = x).
- __weights__: list of numpy arrays to set as initial weights.
- __padding__: 'valid', 'same' or 'full'
    ('full' requires the Theano backend).
- __strides__: tuple of length 2. Factor by which to strides output.
    Also called strides elsewhere.
- __kernel_regularizer__: instance of [WeightRegularizer](../regularizers.md)
    (eg. L1 or L2 regularization), applied to the main weights matrix.
- __bias_regularizer__: instance of [WeightRegularizer](../regularizers.md),
    applied to the use_bias.
- __activity_regularizer__: instance of [ActivityRegularizer](../regularizers.md),
    applied to the network output.
- __kernel_constraint__: instance of the [constraints](../constraints.md) module
    (eg. maxnorm, nonneg), applied to the main weights matrix.
- __bias_constraint__: instance of the [constraints](../constraints.md) module,
    applied to the use_bias.
- __data_format__: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
    (the depth) is at index 1, in 'channels_last' mode is it at index 3.
    It defaults to the `image_data_format` value found in your
    Keras config file at `~/.keras/keras.json`.
    If you never set it, then it will be "tf".
- __use_bias__: whether to include a use_bias
    (i.e. make the layer affine rather than linear).

__Input shape__

4D tensor with shape:
`(samples, channels, rows, cols)` if data_format='channels_first'
or 4D tensor with shape:
`(samples, rows, cols, channels)` if data_format='channels_last'.

__Output shape__

4D tensor with shape:
`(samples, filters, nekernel_rows, nekernel_cols)` if data_format='channels_first'
or 4D tensor with shape:
`(samples, nekernel_rows, nekernel_cols, filters)` if data_format='channels_last'.
`rows` and `cols` values might have changed due to padding.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/convolutional.py#L246)</span>
### SubPixelUpscaling

```python
keras_contrib.layers.SubPixelUpscaling(scale_factor=2, data_format=None)
```

Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
(https://arxiv.org/abs/1609.05158).

This layer requires a Convolution2D prior to it, having output filters computed according to
the formula :

filters = k * (scale_factor * scale_factor)
where k = a user defined number of filters (generally larger than 32)
scale_factor = the upscaling factor (generally 2)

This layer performs the depth to space operation on the convolution filters, and returns a
tensor with the size as defined below.

__Example :__

```python
# A standard subpixel upscaling block
x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
u = SubPixelUpscaling(scale_factor=2)(x)

[Optional]
x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
```

In practice, it is useful to have a second convolution layer after the
SubPixelUpscaling layer to speed up the learning process.

However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
layer can be removed.

__Arguments__

- __scale_factor__: Upscaling factor.
- __data_format__: Can be None, 'channels_first' or 'channels_last'.

__Input shape__

4D tensor with shape:
`(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
or 4D tensor with shape:
`(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.

__Output shape__

4D tensor with shape:
`(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
or 4D tensor with shape:
`(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.

